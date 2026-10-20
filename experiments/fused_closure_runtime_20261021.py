"""Exact optional output-map fusion for the registered N7 and D9 equations.

Mappings belong to one autograd graph, never to an integration state. No new
parameters, buffers, coordinate conventions, objective or export format.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import torch
from closure_training_runtime import TrainingClosureROM, objective
from symbolic_closure_rom import SymbolicClosureROM, term_key, _compiled_term_kind
from cubic_memory_port_runtime_20261021 import (
    N_TERMS, TERMS_KEYS, A_KEY, B_KEY, _finite, project_a,
    one_update as _d_update)
from train_local_memory_0098 import graph

N_KEYS = frozenset(term_key(t) for t in N_TERMS)
D_KEYS = frozenset(TERMS_KEYS)


def variant(model):
    keys = [term_key(t) for t in model.terms]
    if (not isinstance(model, TrainingClosureROM) or model.arm != 'L10-S'
            or model.n != 10 or model.dim != 40 or model.structured.storage_specs
            or len(keys) != len(set(keys)) or len(keys) != model.coefficients.numel()
            or tuple(model._term_kinds) != tuple(_compiled_term_kind(t) for t in model.terms)):
        raise ValueError('exact N7 or D9 training model required')
    if set(keys) == N_KEYS: return 'N'
    if set(keys) == D_KEYS: return 'D'
    raise ValueError('unsupported AST identities')


def require_projection(model):
    kind = variant(model)
    if not _finite(model.state_dict()): raise FloatingPointError('nonfinite model')
    if kind == 'D':
        i = [term_key(t) for t in model.terms].index(A_KEY)
        if not bool(model.coefficients[i] >= 0):
            raise FloatingPointError('quartic coefficient must be projected')


class FusedClosureRuntime(TrainingClosureROM):
    def _clear_runtime(self):
        super()._clear_runtime()
        for name in ('_fused_cache', '_linear_cache', '_quartic_cache', '_d_cache'):
            setattr(self, name, None)

    def _maps(self, constants):
        kind = variant(self)
        A, coupling, storage, quadratic, sin0, cos0, _, weights = constants
        # Selectors map the row-vector interface into each ten-state block.
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
        port_linear = P @ C
        result = dict(storage=storage_linear, linear=storage_linear + port_linear,
                      K=(coupling[:, None] * D @ ed + edge[:, None] * D @ eq) @ AS,
                      T=(edge[:, None] * D @ ed) @ AS,
                      U=(bykind['port_ec'] * self.port_c_weight) @ C,
                      C=C, sin0=sin0, cos0=cos0, Vs=None, V=None)
        if kind == 'D':
            bykey = dict(zip((term_key(t) for t in self.terms), self.coefficients.unbind()))
            vs = 4 * bykey[A_KEY] * (eq @ AS)
            result.update(Vs=vs, V=vs + bykey[B_KEY] * C)
        return result

    def begin_graph_cache(self):
        variant(self)
        super().begin_graph_cache()
        try: self._fused_cache = self._maps(self._runtime_cache)
        except BaseException:
            self._clear_runtime()
            raise

    def end_graph_cache(self): self._clear_runtime()

    def field(self, z, u=None):
        try:
            z = torch.as_tensor(z, device=self.mass.device, dtype=self.mass.dtype)
            if z.ndim == 0 or z.shape[-1] != 40: raise ValueError('40-wide state required')
            m = getattr(self, '_fused_cache', None)
            if m is None: m = self._maps(self._constants())
            d, _, _, q = z.split(10, -1)
            angle = self.phi + d @ self.base_D.T
            cos = angle.cos()
            s, c = angle.sin() - m['sin0'], cos - m['cos0']
            f = z @ (m['linear'] if u is None else m['storage'])
            f = f + s @ m['K'] + ((q @ self.base_D.T) * cos) @ m['T']
            if u is None:
                f = f + c @ m['U']
                if m['V'] is not None: f = f + q.pow(3) @ m['V']
            else:
                u = torch.as_tensor(u, dtype=z.dtype, device=z.device)
                if u.ndim == 0 or u.shape[-1] != 10: raise ValueError('10-wide port required')
                if m['Vs'] is not None: f = f + q.pow(3) @ m['Vs']
                f = f + u @ m['C']
            return f
        except BaseException:
            self._clear_runtime()
            raise

    def with_terms(self, terms):
        if getattr(self, '_runtime_cache', None) is not None: raise RuntimeError('active graph')
        result = super().with_terms(terms)
        variant(result)
        result._clear_runtime()
        return result

    def export(self, path):
        if getattr(self, '_runtime_cache', None) is not None: raise RuntimeError('active graph')
        self._clear_runtime()
        require_projection(self)
        if self.mass.device.type != 'cpu': return copy.deepcopy(self).cpu().export(path)
        return super().export(path)

    @classmethod
    def from_export(cls, path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        if payload['coefficients'] != payload['state_dict']['coefficients']:
            raise ValueError('coefficient copies differ')
        model = super().from_export(path)
        require_projection(model)
        model._clear_runtime()
        return model


def enable_fused_runtime(model):
    require_projection(model)
    if any(getattr(model, n, None) is not None for n in
           ('_graph_cache', '_runtime_cache', '_linear_cache', '_quartic_cache', '_d_cache', '_fused_cache')):
        raise RuntimeError('enable only outside a graph')
    ids = tuple(id(p) for p in model.parameters())
    model.__class__ = FusedClosureRuntime
    model._clear_runtime()
    assert ids == tuple(id(p) for p in model.parameters())
    return model


def one_update(model, opt, store, samples, horizon, force):
    if variant(model) == 'D': return _d_update(model, opt, store, samples, horizon, force)
    opt.zero_grad(set_to_none=True)
    try:
        with graph(model):
            value, components = objective(model, store, samples, horizon, force)
            if not _finite([value, components]): raise FloatingPointError('nonfinite objective')
            value.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        if not _finite([model.state_dict(), [p.grad for p in params], opt.state_dict()]):
            raise FloatingPointError('nonfinite pre-update state')
        norm = torch.nn.utils.clip_grad_norm_(params, 100.)
        if not _finite(norm): raise FloatingPointError('nonfinite norm')
        opt.step()
        if not _finite([model.state_dict(), opt.state_dict()]): raise FloatingPointError('nonfinite updated state')
        return dict(loss=float(value.detach()), gradient_norm=float(norm.detach()),
                    components={k:float(v.detach()) for k,v in components.items()})
    finally: model.end_graph_cache()
