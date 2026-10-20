"""Exact D9 closure: nonnegative quartic energy plus signed cubic momentum port."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from closure_training_runtime import TrainingClosureROM, objective as _original_objective
from symbolic_closure_rom import SymbolicClosureROM, _BASE_TERMS, _compiled_term_kind, term_key
from train_local_memory_0098 import graph

QUARTIC_TERM = {'channel':'node','expression':{'op':'pow4','arg':{'op':'var','name':'q'}}}
D_TERM = {'channel':'port','expression':{'op':'pow3','arg':{'op':'var','name':'q'}}}
N_TERMS = [_BASE_TERMS[i] for i in (0,1,3,4,5,7,8)]
TERMS = N_TERMS+[QUARTIC_TERM,D_TERM]
TERMS_KEYS = tuple(term_key(t) for t in TERMS)
A_KEY,B_KEY = term_key(QUARTIC_TERM),term_key(D_TERM)


def coefficient_indices_bykey(model):
    keys=[term_key(t) for t in model.terms]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate coefficient AST identities')
    return {key:i for i,key in enumerate(keys)}


def coefficient_indices(model):
    bykey=coefficient_indices_bykey(model)
    return {'a':bykey[A_KEY],'b':bykey[B_KEY]}


def quartic_index(model): return coefficient_indices(model)['a']


def _check(model):
    keys = tuple(term_key(t) for t in model.terms)
    if model.arm!='L10-S' or model.dim!=40 or model.n!=10 or model.structured.storage_specs:
        raise ValueError('only fixed ten-machine40-state closure is supported')
    if len(set(keys))!=len(keys) or set(keys)!=set(TERMS_KEYS):
        raise ValueError('exact D9 AST term set required')
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


def require_projection(model):
    _check(model)
    i=quartic_index(model)
    if not _finite(model.state_dict()) or (i is not None and not bool(model.coefficients[i]>=0)):
        raise FloatingPointError('save/export requires finite projected nonnegative quartic coefficient')


class DRuntime(TrainingClosureROM):
    def _clear_runtime(self):
        super()._clear_runtime(); self._d_cache=None; self._d_validated=False

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
        products['quartic_coefficient'] = self.coefficients[index]
        products['cubic_coefficient'] = self.coefficients[coefficient_indices(self)['b']]
        return products


    def begin_graph_cache(self):
        _check(self)
        super().begin_graph_cache()
        try:
            self._d_cache=self._products(self._runtime_cache)
            self._d_validated=True
        except BaseException:
            self._clear_runtime(); raise

    def end_graph_cache(self): self._clear_runtime()

    def with_terms(self,terms):
        if getattr(self,'_runtime_cache',None) is not None: raise RuntimeError('change terms after graph closes')
        result=super().with_terms(terms); result._clear_runtime(); _check(result)
        return result

    def field(self, z, u=None):
        if getattr(self, '_runtime_cache', None) is None or not getattr(self, '_d_validated', False):
            _check(self)
        z = torch.as_tensor(z, dtype=self.mass.dtype, device=self.mass.device)
        runtime = getattr(self, "_runtime_cache", None)
        if runtime is None:
            runtime = self._constants()
        A, coupling, storage, quadratic, sin0, cos0, absD, weights = (
            runtime)
        products = getattr(self, "_d_cache", None)
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
        pv = pv + products['cubic_coefficient'] * q.pow(3)
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
        require_projection(self)
        if self.mass.device.type!='cpu': return copy.deepcopy(self).cpu().export(path)
        return super().export(path)

    @classmethod
    def from_export(cls,path):
        payload=json.loads(Path(path).read_text(encoding='utf-8'))
        if payload['coefficients']!=payload['state_dict']['coefficients']:
            raise ValueError('export coefficient copies differ')
        model=super().from_export(path)
        require_projection(model)
        return model


def enable_runtime(model):
    """Enable exact D9 in place; no coefficient replacement or old-Q guard."""
    if not isinstance(model,SymbolicClosureROM): raise TypeError('symbolic model required')
    _check(model);require_projection(model)
    if any(getattr(model,n,None) is not None for n in ('_runtime_cache','_graph_cache','_linear_cache','_quartic_cache','_d_cache')):
        raise RuntimeError('enable D only with every graph cache clear')
    identities=tuple(id(p) for p in model.parameters())
    model.__class__=DRuntime;model._clear_runtime()
    if tuple(id(p) for p in model.parameters())!=identities: raise RuntimeError('D activation changed parameters')
    return model


def build_D(source):
    if not isinstance(source,SymbolicClosureROM): raise TypeError('symbolic delete7 source required')
    keys=tuple(term_key(t) for t in source.terms)
    if len(keys)!=7 or set(keys)!=set(term_key(t) for t in N_TERMS) or source.coefficients.numel()!=7:
        raise ValueError('D must start from exact common seven-term O512 source')
    if tuple(source._term_kinds)!=tuple(_compiled_term_kind(t) for t in source.terms) or not _finite(source.state_dict()):
        raise ValueError('source compiled kinds/state invalid')
    if any(getattr(source,n,None) is not None for n in ('_runtime_cache','_graph_cache','_linear_cache','_quartic_cache','_d_cache')):
        raise RuntimeError('clone source after graphs close')
    clone=copy.deepcopy(source);clone.__class__=TrainingClosureROM;clone._clear_runtime()
    model=clone.with_terms(copy.deepcopy(source.terms)+[copy.deepcopy(QUARTIC_TERM),copy.deepcopy(D_TERM)])
    enable_runtime(model)
    with torch.no_grad():
        for i in coefficient_indices(model).values(): model.coefficients[i].zero_()
    return model


def extra_port_power(model,z):
    """Signed added H-rate sum(omega*b*q^3), in H-unit/s, without extra1/M."""
    b=model.coefficients[coefficient_indices(model)['b']]
    return (z[...,10:20]*b*z[...,30:40].pow(3)).sum(-1)


def objective(model,store,samples,horizon,force):
    return _original_objective(model,store,samples,horizon,force)


def project_a(model,opt):
    """Validate parameters/gradients/moments before clamping a alone; retain b."""
    if getattr(model,'_runtime_cache',None) is not None: raise RuntimeError('project after graph closes')
    _check(model)
    if not _finite([model.state_dict(),[p.grad for p in model.parameters()],opt.state_dict()]):
        raise FloatingPointError('nonfinite state/gradient/Adam before a projection')
    indices=coefficient_indices(model);result={}
    for name,i in indices.items():
        value=float(model.coefficients[i].detach())
        grad=None if model.coefficients.grad is None else float(model.coefficients.grad[i].detach())
        result[name]={'index':i,'before':value,'after':value,'gradient':grad}
    a=indices['a']
    with torch.no_grad():model.coefficients[a].clamp_(min=0.)
    result['a'].update(after=float(model.coefficients[a].detach()),projected=result['a']['before']<0.)
    return result


def one_update(model,opt,store,samples,horizon,force):
    opt.zero_grad(set_to_none=True)
    try:
        with graph(model):
            value,components=objective(model,store,samples,horizon,force)
            if not _finite([value,components]): raise FloatingPointError('nonfinite original objective')
            value.backward()
        params=[p for p in model.parameters() if p.requires_grad]
        if not _finite([p.grad for p in params]): raise FloatingPointError('nonfinite raw gradients')
        indices=coefficient_indices(model)
        before={k:float(model.coefficients[i].detach()) for k,i in indices.items()}
        raw={k:None if model.coefficients.grad is None else float(model.coefficients.grad[i].detach()) for k,i in indices.items()}
        norm=torch.nn.utils.clip_grad_norm_(params,100.)
        if not _finite(norm): raise FloatingPointError('nonfinite clipped gradient norm')
        opt.step();model.end_graph_cache();traces=project_a(model,opt)
        for name,t in traces.items():
            t.update(raw_gradient=raw[name],before_update=before[name],adam_delta=t['before']-before[name])
        traces['a']['projected_update']=traces['a']['after']-before['a']
        return {'loss':float(value.detach()),'gradient_norm':float(norm.detach()),
                'components':{k:float(v.detach()) for k,v in components.items()},'quartic':traces['a'],'cubic_port':traces['b']}
    finally:model.end_graph_cache()


