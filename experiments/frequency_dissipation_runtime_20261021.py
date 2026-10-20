"""Exact W8: unchanged N7 energy and a nonpositive cubic frequency port.

Output mappings are rebuilt for each optimizer graph. Only the added coefficient is projected after Adam; moments are retained.
There is no new energy term, scale, parameter block or export schema.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from closure_training_runtime import TrainingClosureROM, objective
from symbolic_closure_rom import SymbolicClosureROM, _BASE_TERMS, _compiled_term_kind, term_key
from train_local_memory_0098 import graph

N_TERMS = [_BASE_TERMS[i] for i in (0, 1, 3, 4, 5, 7, 8)]
PORT_TERM = {'channel': 'port', 'expression': {'op': 'pow3', 'arg': {'op': 'var', 'name': 'w'}}}
TERMS = N_TERMS + [PORT_TERM]
TERMS_KEYS = tuple(term_key(t) for t in TERMS)
B_KEY = term_key(PORT_TERM)
_CACHES = ('_graph_cache', '_runtime_cache', '_linear_cache', '_quartic_cache', '_d_cache', '_fused_cache', '_w_cache')


def _tensors(value):
    if torch.is_tensor(value): yield value
    elif isinstance(value, dict):
        for child in value.values(): yield from _tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value: yield from _tensors(child)


def _finite(value): return all(bool(torch.isfinite(t).all()) for t in _tensors(value))


def _check(model):
    keys = [term_key(t) for t in model.terms]
    if (not isinstance(model, TrainingClosureROM) or model.arm != 'L10-S'
            or model.n != 10 or model.dim != 40 or model.structured.storage_specs
            or len(keys) != 8 or len(set(keys)) != 8 or set(keys) != set(TERMS_KEYS)
            or model.coefficients.numel() != 8
            or tuple(model._term_kinds) != tuple(_compiled_term_kind(t) for t in model.terms)):
        raise ValueError('exact W8 AST identities and ten-machine 40-state model required')
    if not bool(torch.isfinite(model.input_scale_omega).all()) or not bool((model.input_scale_omega > 0).all()):
        raise ValueError('fixed frequency scales must be finite and strictly positive')


def coefficient_indices(model):
    _check(model)
    return {'b_W': [term_key(t) for t in model.terms].index(B_KEY)}


def b_index(model): return coefficient_indices(model)['b_W']


def require_finite(model, projected=True):
    _check(model)
    if not _finite(model.state_dict()): raise FloatingPointError('nonfinite W8 model')
    if projected and not bool(model.coefficients[b_index(model)] <= 0):
        raise FloatingPointError('W8 coefficient must be projected nonpositive')


class WRuntime(TrainingClosureROM):
    def _clear_runtime(self):
        super()._clear_runtime()
        for name in _CACHES: setattr(self, name, None)

    def _maps(self, constants):
        _check(self)
        A, coupling, storage, quadratic, sin0, cos0, _, weights = constants
        ed, ew, ep, eq = torch.eye(40, device=A.device, dtype=A.dtype).split(10, dim=0)
        bykind = dict(zip(self._term_kinds, self.coefficients.unbind()))
        node = bykind['node_pq'] / self.p_scale
        G = (ew.T @ ew + ep.T @ (storage[:10, None] * ep)
             + eq.T @ ((storage[10:] + quadratic)[:, None] * eq)
             + eq.T @ (node[..., None] * ep) + ep.T @ (node[..., None] * eq))
        scale = torch.cat((torch.ones_like(self.mass), self.mass.reciprocal(),
                           torch.ones_like(self.mass), torch.ones_like(self.mass)))
        AS = A * scale
        C = ew / self.mass[:, None]
        eye = torch.eye(10, device=A.device, dtype=A.dtype)
        P = (ew.T * (bykind['port_w'] / self.input_scale_omega)
             + ep.T * (bykind['port_p'] / self.p_scale)
             + eq.T @ (bykind['port_q'] * eye
                        + bykind['port_nq'] * (self.neighbor_A.T - eye)))
        edge = bykind['edge_sine'] * weights
        D = self.base_D
        storage_linear = G @ AS
        # Only the normalized frequency port has a cubic term. Energy stays N7.
        return dict(storage=storage_linear, linear=storage_linear + P @ C,
                    K=(coupling[:, None] * D @ ed + edge[:, None] * D @ eq) @ AS,
                    T=(edge[:, None] * D @ ed) @ AS,
                    U=(bykind['port_ec'] * self.port_c_weight) @ C,
                    V=self.coefficients[b_index(self)] * C / self.input_scale_omega[:, None].pow(3),
                    C=C, sin0=sin0, cos0=cos0)

    def begin_graph_cache(self):
        _check(self)
        super().begin_graph_cache()
        try: self._w_cache = self._maps(self._runtime_cache)
        except BaseException:
            self._clear_runtime()
            raise

    def end_graph_cache(self): self._clear_runtime()

    def field(self, z, u=None):
        try:
            z = torch.as_tensor(z, device=self.mass.device, dtype=self.mass.dtype)
            if z.ndim == 0 or z.shape[-1] != 40: raise ValueError('40-wide state required')
            m = getattr(self, '_w_cache', None)
            if m is None: m = self._maps(self._constants())
            d, w, _, q = z.split(10, -1)
            angle = self.phi + d @ self.base_D.T
            cos = angle.cos()
            s, c = angle.sin() - m['sin0'], cos - m['cos0']
            f = z @ (m['linear'] if u is None else m['storage'])
            f = f + s @ m['K'] + ((q @ self.base_D.T) * cos) @ m['T']
            if u is None:
                f = f + c @ m['U'] + w.pow(3) @ m['V']
            else:
                u = torch.as_tensor(u, dtype=z.dtype, device=z.device)
                if u.ndim == 0 or u.shape[-1] != 10: raise ValueError('10-wide port required')
                f = f + u @ m['C']
            return f
        except BaseException:
            self._clear_runtime()
            raise

    def with_terms(self, terms):
        if getattr(self, '_runtime_cache', None) is not None: raise RuntimeError('active graph')
        result = super().with_terms(terms)
        _check(result)
        result._clear_runtime()
        return result

    def export(self, path):
        if any(getattr(self, n, None) is not None for n in _CACHES): raise RuntimeError('active graph')
        require_finite(self)
        if self.mass.device.type != 'cpu': return copy.deepcopy(self).cpu().export(path)
        return super().export(path)

    @classmethod
    def from_export(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload['coefficients'] != payload['state_dict']['coefficients']:
            raise ValueError('coefficient copies differ')
        model = super().from_export(path)
        require_finite(model)
        model._clear_runtime()
        return model


def enable_runtime(model):
    require_finite(model)
    if any(getattr(model, n, None) is not None for n in _CACHES):
        raise RuntimeError('enable W8 only outside every graph')
    identities = tuple(id(p) for p in model.parameters())
    model.__class__ = WRuntime
    model._clear_runtime()
    if identities != tuple(id(p) for p in model.parameters()): raise RuntimeError('parameter replacement')
    return model


def build_W(source):
    if not isinstance(source, SymbolicClosureROM): raise TypeError('symbolic N7 source required')
    keys = [term_key(t) for t in source.terms]
    if (len(keys) != 7 or set(keys) != set(term_key(t) for t in N_TERMS)
            or source.coefficients.numel() != 7
            or tuple(source._term_kinds) != tuple(_compiled_term_kind(t) for t in source.terms)):
        raise ValueError('W8 must start from exact common N7 source')
    if not _finite(source.state_dict()): raise FloatingPointError('nonfinite source')
    if any(getattr(source, n, None) is not None for n in _CACHES): raise RuntimeError('active source graph')
    clone = copy.deepcopy(source)
    clone.__class__ = TrainingClosureROM
    clone._clear_runtime()
    model = clone.with_terms(copy.deepcopy(source.terms) + [copy.deepcopy(PORT_TERM)])
    enable_runtime(model)
    with torch.no_grad(): model.coefficients[b_index(model)].zero_()
    return model


def extra_port_field(model, z):
    """Ten added interface frequency derivatives, in rad/s²."""
    z = torch.as_tensor(z, dtype=model.mass.dtype, device=model.mass.device)
    if z.ndim == 0 or z.shape[-1] != 40: raise ValueError('40-wide state required')
    return model.coefficients[b_index(model)] * (z[..., 10:20] / model.input_scale_omega).pow(3) / model.mass


def extra_port_power(model, z):
    """Added H-rate b*sum(omega**4/s**3), H-unit/s, not MW."""
    z = torch.as_tensor(z, dtype=model.mass.dtype, device=model.mass.device)
    if z.ndim == 0 or z.shape[-1] != 40: raise ValueError('40-wide state required')
    return model.coefficients[b_index(model)] * (z[..., 10:20].pow(4) / model.input_scale_omega.pow(3)).sum(-1)


def project_b(model, opt):
    """Clamp b only after finite checks; preserve the complete Adam state."""
    if any(getattr(model,n,None) is not None for n in _CACHES): raise RuntimeError('project after graph closes')
    require_finite(model, projected=False)
    if not _finite([[p.grad for p in model.parameters()],opt.state_dict()]):
        raise FloatingPointError('nonfinite gradients or Adam before projection')
    index=b_index(model)
    before=float(model.coefficients[index].detach())
    gradient=None if model.coefficients.grad is None else float(model.coefficients.grad[index].detach())
    with torch.no_grad():model.coefficients[index].clamp_(max=0.)
    after=float(model.coefficients[index].detach())
    return dict(index=index,before=before,after=after,pre_projection=before,post_projection=after,
                post_adam=before,gradient=gradient,projected=before>0.,projection_hit=before>0.)


def one_update(model, opt, store, samples, horizon, force):
    opt.zero_grad(set_to_none=True)
    try:
        with graph(model):
            value, components = objective(model, store, samples, horizon, force)
            if not _finite([value, components]): raise FloatingPointError('nonfinite original objective')
            value.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        if not _finite([model.state_dict(), [p.grad for p in params], opt.state_dict()]):
            raise FloatingPointError('nonfinite pre-update state')
        index = b_index(model)
        before = float(model.coefficients[index].detach())
        raw = None if model.coefficients.grad is None else float(model.coefficients.grad[index].detach())
        norm = torch.nn.utils.clip_grad_norm_(params, 100.)
        if not _finite(norm): raise FloatingPointError('nonfinite norm')
        clipped = None if model.coefficients.grad is None else float(model.coefficients.grad[index].detach())
        opt.step()
        if not _finite([model.state_dict(), opt.state_dict()]): raise FloatingPointError('nonfinite updated state')
        trace=project_b(model,opt)
        trace.update(before_update=before,pre_adam=before,raw_gradient=raw,gradient=clipped,
                     adam_delta=trace['before']-before,projected_update=trace['after']-before,
                     clipped=bool(norm>100.))
        return dict(loss=float(value.detach()), gradient_norm=float(norm.detach()),
                    components={k: float(v.detach()) for k, v in components.items()},
                    frequency_dissipation=trace)
    finally: model.end_graph_cache()
