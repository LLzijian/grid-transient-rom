"""Exact E8: signed memory-amplitude modulation of edge energy.

Output mappings are rebuilt for each optimizer graph. There is no coefficient
projection, additional port, extra scale or export schema.
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
EDGE_TERM = {'channel':'edge','expression':{'op':'mul','args':[{'op':'pow2','arg':{'op':'add','args':[{'op':'var','name':'qi'},{'op':'var','name':'qj'}]}},{'op':'var','name':'c'}]}}
TERMS = N_TERMS + [EDGE_TERM]
TERMS_KEYS = tuple(term_key(t) for t in TERMS)
A_E_KEY = term_key(EDGE_TERM)
_CACHES = ('_graph_cache', '_runtime_cache', '_linear_cache', '_quartic_cache', '_d_cache', '_fused_cache', '_e_cache')


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
        raise ValueError('exact E8 AST identities and ten-machine 40-state model required')


def coefficient_indices(model):
    _check(model)
    return {'a_E': [term_key(t) for t in model.terms].index(A_E_KEY)}


def a_E_index(model): return coefficient_indices(model)['a_E']


def require_finite(model):
    _check(model)
    if not _finite(model.state_dict()): raise FloatingPointError('nonfinite E8 model')


class ERuntime(TrainingClosureROM):
    def _clear_runtime(self):
        super()._clear_runtime()
        for name in _CACHES: setattr(self, name, None)

    def _maps(self, constants):
        _check(self)
        A, coupling, storage, quadratic, sin0, cos0, absD, weights = constants
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
        amplitude = self.coefficients[a_E_index(self)]
        return dict(storage=storage_linear, linear=storage_linear + P @ C,
                    K=(coupling[:, None] * D @ ed + edge[:, None] * D @ eq) @ AS,
                    T=(edge[:, None] * D @ ed) @ AS,
                    U=(bykind['port_ec'] * self.port_c_weight) @ C,
                    E_delta=(-amplitude * weights[:, None] * D @ ed) @ AS,
                    E_q=(2 * amplitude * weights[:, None] * absD @ eq) @ AS,
                    absD=absD,
                    C=C, sin0=sin0, cos0=cos0)

    def begin_graph_cache(self):
        _check(self)
        super().begin_graph_cache()
        try: self._e_cache = self._maps(self._runtime_cache)
        except BaseException:
            self._clear_runtime()
            raise

    def end_graph_cache(self): self._clear_runtime()

    def field(self, z, u=None):
        try:
            z = torch.as_tensor(z, device=self.mass.device, dtype=self.mass.dtype)
            if z.ndim == 0 or z.shape[-1] != 40: raise ValueError('40-wide state required')
            m = getattr(self, '_e_cache', None)
            if m is None: m = self._maps(self._constants())
            d, _, _, q = z.split(10, -1)
            angle = self.phi + d @ self.base_D.T
            cos = angle.cos()
            s, c = angle.sin() - m['sin0'], cos - m['cos0']
            f = z @ (m['linear'] if u is None else m['storage'])
            f = f + s @ m['K'] + ((q @ self.base_D.T) * cos) @ m['T']
            r = q @ m['absD'].T
            f = f + (r.square() * angle.sin()) @ m['E_delta'] + (r * c) @ m['E_q']
            if u is None:
                f = f + c @ m['U']
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
        raise RuntimeError('enable E8 only outside every graph')
    identities = tuple(id(p) for p in model.parameters())
    model.__class__ = ERuntime
    model._clear_runtime()
    if identities != tuple(id(p) for p in model.parameters()): raise RuntimeError('parameter replacement')
    return model


def build_E(source):
    if not isinstance(source, SymbolicClosureROM): raise TypeError('symbolic N7 source required')
    keys = [term_key(t) for t in source.terms]
    if (len(keys) != 7 or set(keys) != set(term_key(t) for t in N_TERMS)
            or source.coefficients.numel() != 7
            or tuple(source._term_kinds) != tuple(_compiled_term_kind(t) for t in source.terms)):
        raise ValueError('E8 must start from exact common N7 source')
    if not _finite(source.state_dict()): raise FloatingPointError('nonfinite source')
    if any(getattr(source, n, None) is not None for n in _CACHES): raise RuntimeError('active source graph')
    clone = copy.deepcopy(source)
    clone.__class__ = TrainingClosureROM
    clone._clear_runtime()
    model = clone.with_terms(copy.deepcopy(source.terms) + [copy.deepcopy(EDGE_TERM)])
    enable_runtime(model)
    with torch.no_grad(): model.coefficients[a_E_index(model)].zero_()
    return model


def extra_energy_components(model, z):
    """Added energy, analytic gradients and separate 40-wide interface fields.

    Field norms mix the established interface coordinates and carry no common
    force unit. This evaluates a state, never an initializer or rollout.
    """
    _check(model)
    z = torch.as_tensor(z, dtype=model.mass.dtype, device=model.mass.device)
    if z.ndim == 0 or z.shape[-1] != 40: raise ValueError('40-wide state required')
    d, _, _, q = z.split(10, -1)
    D = model.base_D; S = D.abs()
    k = model.base_coupling / model.base_coupling.mean().clamp_min(1e-15)
    r = q @ S.T; angle = model.phi + d @ D.T
    c = angle.cos() - model.phi.cos()
    a = model.coefficients[a_E_index(model)]
    gd = -a * ((k * r.square() * angle.sin()) @ D)
    gq = 2 * a * ((k * r * c) @ S)
    J, R = model.matrices()
    scale = torch.cat((torch.ones_like(model.mass), model.mass.reciprocal(),
                       torch.ones_like(model.mass), torch.ones_like(model.mass)))
    AS = (J-R).T * scale
    zero = torch.zeros_like(gd)
    return dict(energy=a*(k*r.square()*c).sum(-1), g_delta=gd, g_q=gq,
                field_delta=torch.cat((gd,zero,zero,zero),-1) @ AS,
                field_q=torch.cat((zero,zero,zero,gq),-1) @ AS)


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
        index = a_E_index(model)
        before = float(model.coefficients[index].detach())
        raw = None if model.coefficients.grad is None else float(model.coefficients.grad[index].detach())
        norm = torch.nn.utils.clip_grad_norm_(params, 100.)
        if not _finite(norm): raise FloatingPointError('nonfinite norm')
        clipped = None if model.coefficients.grad is None else float(model.coefficients.grad[index].detach())
        opt.step()
        if not _finite([model.state_dict(), opt.state_dict()]): raise FloatingPointError('nonfinite updated state')
        after = float(model.coefficients[index].detach())
        return dict(loss=float(value.detach()), gradient_norm=float(norm.detach()),
                    components={k: float(v.detach()) for k, v in components.items()},
                    edge_memory=dict(index=index, before_update=before, before=after, after=after,
                                    raw_gradient=raw, gradient=clipped, adam_delta=after-before))
    finally: model.end_graph_cache()
