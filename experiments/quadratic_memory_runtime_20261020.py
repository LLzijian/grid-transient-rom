"""Exact float64 A/B/C edge-memory runtimes for the registered fixed study.

C uses H_extra=b sum_e weight_e (q_i+q_j)^2 c_e. The existing generic
compiler remains authoritative for energy, export, and independent comparisons.
"""
from __future__ import annotations

import copy
import torch

from closure_training_runtime import TrainingClosureROM
from linear_cache_runtime_20261019 import LinearCacheRuntime
from symbolic_closure_rom import _BASE_TERMS, term_key, _compiled_term_kind

QUADRATIC_TERM = {'channel': 'edge', 'expression': {'op': 'mul', 'args': [
    {'op': 'pow2', 'arg': {'op': 'add', 'args': [
        {'op': 'var', 'name': 'qi'}, {'op': 'var', 'name': 'qj'}]}},
    {'op': 'var', 'name': 'c'}]}}
MAIN8 = [_BASE_TERMS[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)]
TERMS = {'A': MAIN8,
         'B': [t for t in MAIN8 if term_key(t) != term_key(_BASE_TERMS[2])],
         'C': [QUADRATIC_TERM if term_key(t) == term_key(_BASE_TERMS[2]) else t for t in MAIN8]}
KEYS = {v: tuple(term_key(t) for t in ts) for v, ts in TERMS.items()}


def _check(model):
    if model.arm != 'L10-S' or model.n != 10 or model.dim != 40:
        raise ValueError('quadratic runtime requires ten-machine 40-state L10-S')
    if model.structured.storage_specs:
        raise ValueError('quadratic runtime does not support other structured storage')
    keys = tuple(term_key(t) for t in model.terms)
    if keys not in KEYS.values():
        raise ValueError('quadratic runtime requires exact ordered A, B or C terms')
    if tuple(model._term_kinds) != tuple(_compiled_term_kind(t) for t in model.terms):
        raise ValueError('compiled kinds do not match the registered expressions')
    if model.coefficients.numel() != len(keys):
        raise ValueError('coefficient count differs from expression count')


class QuadraticMemoryRuntime(TrainingClosureROM):
    def _clear_runtime(self):
        super()._clear_runtime()
        self._quadratic_cache = None
        self._quadratic_validated = False

    def _products(self, constants):
        products = LinearCacheRuntime._linear_products(self, constants)
        coefficients = dict(zip((term_key(t) for t in self.terms), self.coefficients.unbind()))
        b = coefficients.get(term_key(QUADRATIC_TERM))
        products['quadratic_edge'] = None if b is None else b * constants[-1]
        return products

    def begin_graph_cache(self):
        _check(self)
        if getattr(self, '_quadratic_cache', None) is not None:
            raise RuntimeError('nested quadratic graph cache')
        super().begin_graph_cache()
        try:
            self._quadratic_cache = self._products(self._runtime_cache)
            self._quadratic_validated = True
        except BaseException:
            self._clear_runtime()
            raise

    def end_graph_cache(self):
        self._clear_runtime()

    def with_terms(self, terms):
        if getattr(self, '_runtime_cache', None) is not None:
            raise RuntimeError('change terms only after closing the graph cache')
        result = super().with_terms(terms)
        result._clear_runtime()
        _check(result)
        return result

    def field(self, z, u=None):
        if not getattr(self, '_quadratic_validated', False):
            _check(self)
            self._quadratic_validated = True
        z = torch.as_tensor(z, dtype=self.mass.dtype, device=self.mass.device)
        if z.shape[-1] != 40:
            raise ValueError('quadratic runtime requires width-40 interface state')
        runtime = getattr(self, '_runtime_cache', None)
        runtime = self._constants() if runtime is None else runtime
        A, coupling, storage, quadratic, sin0, cos0, absD, weights = runtime
        products = getattr(self, '_quadratic_cache', None)
        products = self._products(runtime) if products is None else products
        d, w, p, q = z.split(10, dim=-1)
        angle = self.phi + d @ self.base_D.T
        sin, cos = angle.sin(), angle.cos()
        s, c = sin - sin0, cos - cos0
        gd = (s * coupling) @ self.base_D
        gp = storage[:10] * p + q * products['node']
        gq = products['storage_q'] * q + p * products['node']
        sine = products['edge_sine']
        gd = gd + ((q @ self.base_D.T) * cos * sine) @ self.base_D
        gq = gq + (s * sine) @ self.base_D
        cosine = products['edge_cosine']
        if cosine is not None:
            gd = gd + (-(q @ absD.T) * sin * cosine) @ self.base_D
            gq = gq + (c * cosine) @ absD
        b = products['quadratic_edge']
        if b is not None:
            Q = q @ absD.T
            gd = gd + (-b * Q.square() * sin) @ self.base_D
            gq = gq + (2 * b * Q * c) @ absD
        pv = (w * products['port_w'] + p * products['port_p'] +
              q @ products['port_Q'] + c @ products['port_ec'])
        if u is not None:
            pv = torch.as_tensor(u, dtype=z.dtype, device=z.device)
        if pv.shape[-1] != 10:
            raise ValueError('external port input must have width 10')
        f = torch.cat((gd, w, gp, gq), -1) @ A
        return torch.cat((f[..., :10], (f[..., 10:20] + pv) / self.mass, f[..., 20:]), -1)

    def export(self, path):
        if getattr(self, '_runtime_cache', None) is not None or getattr(self, '_quadratic_cache', None) is not None:
            raise RuntimeError('export only after closing the graph cache')
        if self.mass.device.type != 'cpu':
            return copy.deepcopy(self).cpu().export(path)
        return super().export(path)


def build_variant(source, variant):
    """Clone a common MAIN8 endpoint; preserve seven terms exactly and set b=0.

    Optimizer construction and random-state restoration belong to the runner.
    The source instance and Parameter objects are never modified.
    """
    if type(source) not in (TrainingClosureROM, LinearCacheRuntime):
        raise TypeError('source must be an unwrapped TrainingClosureROM or LinearCacheRuntime')
    if variant not in TERMS:
        raise ValueError('variant must be A, B or C')
    if any(getattr(source, name, None) is not None for name in ('_runtime_cache', '_graph_cache', '_linear_cache')):
        raise RuntimeError('build variants only when source graph caches are clear')
    _check(source)
    if tuple(term_key(t) for t in source.terms) != KEYS['A']:
        raise ValueError('all variants require the identical MAIN8 source')
    clone = copy.deepcopy(source)
    # Unwrap only the private clone, bypassing LinearCacheRuntime's A/B-only
    # with_terms guard. The new runtime independently checks all A/B/C syntax.
    clone.__class__ = TrainingClosureROM
    clone._clear_runtime()
    clone = clone.with_terms(copy.deepcopy(TERMS[variant]))
    clone.__class__ = QuadraticMemoryRuntime
    clone._clear_runtime()
    _check(clone)
    if variant == 'C':
        with torch.no_grad():
            i = next(i for i, t in enumerate(clone.terms) if term_key(t) == term_key(QUADRATIC_TERM))
            clone.coefficients[i].zero_()
    return clone
