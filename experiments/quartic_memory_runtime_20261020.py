"""Exact node q^4 closure with one nonnegative projected coefficient."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from closure_training_runtime import TrainingClosureROM, objective as _original_objective
from symbolic_closure_rom import SymbolicClosureROM, _BASE_TERMS, _compiled_term_kind, term_key
from train_local_memory_0098 import graph

QUARTIC_TERM = {'channel':'node','expression':{'op':'pow4','arg':{'op':'var','name':'q'}}}
TERMS = {'N':[_BASE_TERMS[i] for i in (0,1,3,4,5,7,8)]}
TERMS['Q'] = TERMS['N']+[QUARTIC_TERM]
KEYS = {k:tuple(term_key(t) for t in v) for k,v in TERMS.items()}


def quartic_index(model):
    matches = [i for i,t in enumerate(model.terms) if term_key(t)==term_key(QUARTIC_TERM)]
    if len(matches)>1: raise ValueError('duplicate quartic coefficient')
    return matches[0] if matches else None


def _check(model):
    keys = tuple(term_key(t) for t in model.terms)
    if model.arm!='L10-S' or model.dim!=40 or model.n!=10 or model.structured.storage_specs:
        raise ValueError('only fixed ten-machine40-state closure is supported')
    if len(set(keys))!=len(keys) or set(keys) not in (set(KEYS['N']),set(KEYS['Q'])):
        raise ValueError('exact N/Q term sets required, including every retained expression')
    if len(keys)!=model.coefficients.numel() or tuple(model._term_kinds)!=tuple(_compiled_term_kind(t) for t in model.terms):
        raise ValueError('expression/coefficient/compiled-kind identities differ')


def _tensors(value):
    if torch.is_tensor(value): yield value
    elif isinstance(value,dict):
        for v in value.values(): yield from _tensors(v)
    elif isinstance(value,(tuple,list)):
        for v in value: yield from _tensors(v)


def _finite(value):
    return all(bool(torch.isfinite(t).all()) for t in _tensors(value))


def require_projected(model):
    _check(model)
    i=quartic_index(model)
    if not _finite(model.state_dict()) or (i is not None and not bool(model.coefficients[i]>=0)):
        raise FloatingPointError('save/export requires finite projected nonnegative quartic coefficient')


class QuarticMemoryRuntime(TrainingClosureROM):
    def _clear_runtime(self):
        super()._clear_runtime(); self._quartic_cache=None; self._quartic_validated=False

    def _products(self, runtime_constants=None):
        coefficients = dict(zip(self._term_kinds, self.coefficients.unbind()))
        constants = runtime_constants if runtime_constants is not None else self._constants()
        _, _, storage, quadratic, _, _, _, weights = constants
        products = {
            "node": coefficients["node_pq"] / self.p_scale,
            "storage_q": storage[10:] + quadratic,
            "edge_sine": (coefficients["edge_sine"] * weights
                          if "edge_sine" in coefficients else None),
            "edge_cosine": (coefficients["edge_cosine"] * weights
                            if "edge_cosine" in coefficients else None),
            "port_w": coefficients["port_w"] / self.input_scale_omega,
            "port_p": coefficients["port_p"] / self.p_scale,
            "port_ec": coefficients["port_ec"] * self.port_c_weight,
        }
        identity = torch.eye(self.n, dtype=self.mass.dtype, device=self.mass.device)
        products["port_Q"] = (coefficients["port_q"] * identity +
            coefficients["port_nq"] * (self.neighbor_A.T - identity))
        products['quartic_index'] = quartic_index(self)
        index = products['quartic_index']
        products['quartic_coefficient'] = None if index is None else self.coefficients[index]
        return products


    def begin_graph_cache(self):
        _check(self)
        super().begin_graph_cache()
        try:
            self._quartic_cache=self._products(self._runtime_cache)
            self._quartic_validated=True
        except BaseException:
            self._clear_runtime(); raise

    def end_graph_cache(self): self._clear_runtime()

    def with_terms(self,terms):
        if getattr(self,'_runtime_cache',None) is not None: raise RuntimeError('change terms after graph closes')
        result=super().with_terms(terms); result._clear_runtime(); _check(result)
        return result

    def field(self, z, u=None):
        if getattr(self, '_runtime_cache', None) is None or not getattr(self, '_quartic_validated', False):
            _check(self)
        z = torch.as_tensor(z, dtype=self.mass.dtype, device=self.mass.device)
        runtime = getattr(self, "_runtime_cache", None)
        if runtime is None:
            runtime = self._constants()
        A, coupling, storage, quadratic, sin0, cos0, absD, weights = (
            runtime)
        products = getattr(self, "_quartic_cache", None)
        if products is None:
            products = self._products(runtime)
        if z.shape[-1] != 40:
            raise ValueError("linear cache expects a 40-wide interface state")
        d, w, p, q = z.split(10, dim=-1)
        angle = self.phi + d @ self.base_D.T
        sin, cos = angle.sin(), angle.cos()
        s, c = sin - sin0, cos - cos0
        gd = (s * coupling) @ self.base_D
        gp = storage[:10] * p
        gq = products["storage_q"] * q

        node_scale = products["node"]
        gp = gp + q * node_scale
        gq = gq + p * node_scale
        edge_sine = products["edge_sine"]
        if edge_sine is not None:
            gd = gd + ((q @ self.base_D.T) * cos * edge_sine) @ self.base_D
            gq = gq + (s * edge_sine) @ self.base_D
        edge_cosine = products["edge_cosine"]
        if edge_cosine is not None:
            gd = gd + (-(q @ absD.T) * sin * edge_cosine) @ self.base_D
            gq = gq + (c * edge_cosine) @ absD

        coefficient = products['quartic_coefficient']
        if coefficient is not None:
            # Never branch on a==0: its derivative must remain connected.
            gq = gq + 4 * coefficient * q.pow(3)

        pv = (w * products["port_w"] + p * products["port_p"] +
              q @ products["port_Q"] + c @ products["port_ec"])
        if u is not None:
            pv = torch.as_tensor(u, dtype=z.dtype, device=z.device)
        if pv.shape[-1] != 10:
            raise ValueError("external port input must have width 10")
        gradient = torch.cat((gd, w, gp, gq), -1)
        f = gradient @ A
        return torch.cat((f[..., :10], (f[..., 10:20] + pv) / self.mass,
                          f[..., 20:]), -1)

    def export(self,path):
        if getattr(self,'_runtime_cache',None) is not None:
            raise RuntimeError('export only after graph closes')
        require_projected(self)
        if self.mass.device.type!='cpu': return copy.deepcopy(self).cpu().export(path)
        return super().export(path)

    @classmethod
    def from_export(cls,path):
        payload=json.loads(Path(path).read_text(encoding='utf-8'))
        if payload['coefficients']!=payload['state_dict']['coefficients']:
            raise ValueError('export coefficient copies differ')
        model=super().from_export(path)
        require_projected(model)
        return model


def enable_runtime(model):
    """Enable the narrow runtime in place, retaining all Parameter identities."""
    if not isinstance(model,SymbolicClosureROM): raise TypeError('symbolic closure model required')
    _check(model)
    require_projected(model)
    if any(getattr(model,n,None) is not None for n in ('_runtime_cache','_graph_cache','_linear_cache','_quartic_cache')):
        raise RuntimeError('enable runtime only after all graphs close')
    identities=tuple(id(p) for p in model.parameters())
    model.__class__=QuarticMemoryRuntime; model._clear_runtime()
    if identities!=tuple(id(p) for p in model.parameters()): raise RuntimeError('runtime activation changed parameters')
    return model


def objective(model,store,samples,horizon,force):
    return _original_objective(model,store,samples,horizon,force)


def build_variant(source,variant):
    if not isinstance(source,SymbolicClosureROM): raise TypeError('symbolic same-source model required')
    _check(source)
    if variant not in TERMS or set(term_key(t) for t in source.terms)!=set(KEYS['N']):
        raise ValueError('N/Q require exactly the common delete7 source')
    if any(getattr(source,n,None) is not None for n in ('_runtime_cache','_graph_cache','_linear_cache','_quartic_cache')):
        raise RuntimeError('clone source only with graphs clear')
    clone=copy.deepcopy(source); clone.__class__=TrainingClosureROM; clone._clear_runtime()
    terms=copy.deepcopy(source.terms)+([copy.deepcopy(QUARTIC_TERM)] if variant=='Q' else [])
    model=clone.with_terms(terms); enable_runtime(model)
    if variant=='Q':
        with torch.no_grad(): model.coefficients[quartic_index(model)].zero_()
    return model


def project_quartic(model,opt):
    """Check before projection; clamp only a, retaining Adam first/second moments."""
    if getattr(model,'_runtime_cache',None) is not None:
        raise RuntimeError('project only after graph closes')
    _check(model)
    if not _finite([model.state_dict(),[p.grad for p in model.parameters()],opt.state_dict()]):
        raise FloatingPointError('nonfinite parameters/gradients/Adam before projection')
    i=quartic_index(model)
    if i is None: return {'present':False,'index':None,'before':None,'after':None,'gradient':None,'projected':False}
    before=float(model.coefficients[i].detach())
    grad=None if model.coefficients.grad is None else float(model.coefficients.grad[i].detach())
    with torch.no_grad(): model.coefficients[i].clamp_(min=0.)
    return {'present':True,'index':i,'before':before,'after':float(model.coefficients[i].detach()),
            'gradient':grad,'projected':before<0.}


def one_update(model,opt,store,samples,horizon,force):
    opt.zero_grad(set_to_none=True)
    try:
        with graph(model):
            value,components=objective(model,store,samples,horizon,force)
            if not _finite([value,components]): raise FloatingPointError('nonfinite original objective')
            value.backward()
        params=[p for p in model.parameters() if p.requires_grad]
        if not _finite([p.grad for p in params]): raise FloatingPointError('nonfinite raw gradients')
        i=quartic_index(model)
        before=None if i is None else float(model.coefficients[i].detach())
        raw_grad=None if i is None or model.coefficients.grad is None else float(model.coefficients.grad[i].detach())
        norm=torch.nn.utils.clip_grad_norm_(params,100.)
        if not _finite(norm): raise FloatingPointError('nonfinite clipped gradient norm')
        opt.step(); model.end_graph_cache()
        projection=project_quartic(model,opt)
        projection.update(raw_gradient=raw_grad,before_update=before,
            adam_delta=None if i is None else projection['before']-before,
            projected_update=None if i is None else projection['after']-before)
        return {'loss':float(value.detach()),'gradient_norm':float(norm.detach()),
                'components':{k:float(v.detach()) for k,v in components.items()},'quartic':projection}
    finally:
        model.end_graph_cache()

