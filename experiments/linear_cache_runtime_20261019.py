"""Per-training-graph linear-combination cache for the registered L10-S closures.

The cache retains autograd links to every parameter-derived value and is cleared
with the existing TrainingClosureROM graph cache. This module adds no parameters,
buffers, export fields, or generic expression handling.
"""
from __future__ import annotations

import torch

from closure_training_runtime import TrainingClosureROM
from symbolic_closure_rom import _BASE_TERMS, term_key


_MAIN8 = [_BASE_TERMS[index] for index in (0, 1, 2, 3, 4, 5, 7, 8)]
_DELETE7 = [term for term in _MAIN8 if term_key(term) != term_key(_BASE_TERMS[2])]
_MAIN8_KEYS = tuple(term_key(term) for term in _MAIN8)
_DELETE7_KEYS = tuple(term_key(term) for term in _DELETE7)
_ALLOWED_KINDS = {
    _MAIN8_KEYS: ("node_pq", "edge_sine", "edge_cosine", "port_w", "port_p", "port_q", "port_ec", "port_nq"),
    _DELETE7_KEYS: ("node_pq", "edge_sine", "port_w", "port_p", "port_q", "port_ec", "port_nq"),
}


def _check_model(model):
    if not isinstance(model, TrainingClosureROM):
        raise TypeError("linear cache requires a TrainingClosureROM instance")
    if model.arm != "L10-S" or model.n != 10 or model.dim != 40:
        raise ValueError("linear cache supports only the 40-dimensional L10-S model")
    if model.structured.storage_specs:
        raise ValueError("linear cache does not support extra structured storage specs")
    keys = tuple(term_key(term) for term in model.terms)
    if keys not in _ALLOWED_KINDS:
        raise ValueError("linear cache requires the complete ordered MAIN8 or edge-cosine-deleted terms")
    if tuple(model._term_kinds) != _ALLOWED_KINDS[keys]:
        raise ValueError("compiled term kinds do not match the registered term keys and order")
    return keys


class LinearCacheRuntime(TrainingClosureROM):
    """TrainingClosureROM with parameter products shared within one graph."""

    def _clear_runtime(self):
        super()._clear_runtime()
        self._linear_cache = None
        self._linear_validated = False

    def _linear_products(self, runtime_constants=None):
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
        return products

    def begin_graph_cache(self):
        _check_model(self)
        self._linear_validated = True
        super().begin_graph_cache()
        try:
            self._linear_cache = self._linear_products(self._runtime_cache)
        except BaseException:
            self._clear_runtime()
            raise

    def end_graph_cache(self):
        self._clear_runtime()

    def with_terms(self, terms):
        if getattr(self, "_runtime_cache", None) is not None or getattr(self, "_linear_cache", None) is not None:
            raise RuntimeError("change terms only when no graph cache is active")
        result = super().with_terms(terms)
        result._clear_runtime()
        _check_model(result)
        result._linear_validated = True
        return result

    def field(self, z, u=None):
        if not getattr(self, "_linear_validated", False):
            _check_model(self)
            self._linear_validated = True
        z = torch.as_tensor(z, dtype=self.mass.dtype, device=self.mass.device)
        runtime = getattr(self, "_runtime_cache", None)
        if runtime is None:
            runtime = self._constants()
        A, coupling, storage, quadratic, sin0, cos0, absD, weights = (
            runtime)
        products = getattr(self, "_linear_cache", None)
        if products is None:
            products = self._linear_products(runtime)
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

    def export(self, path):
        if getattr(self, "_runtime_cache", None) is not None or getattr(self, "_linear_cache", None) is not None:
            raise RuntimeError("export requires a cleared training graph cache")
        return super().export(path)


def enable_linear_cache(model):
    """Switch a supported runtime instance in place, retaining Parameter objects."""
    _check_model(model)
    if type(model) is LinearCacheRuntime:
        return model
    if type(model) is not TrainingClosureROM:
        raise TypeError("only an unwrapped TrainingClosureROM can enable the linear cache")
    if getattr(model, "_runtime_cache", None) is not None or getattr(model, "_graph_cache", None) is not None:
        raise RuntimeError("enable the linear cache only when no graph is active")
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    model.__class__ = LinearCacheRuntime
    model._linear_cache = None
    model._linear_validated = True
    if tuple(id(parameter) for parameter in model.parameters()) != parameter_ids:
        raise RuntimeError("linear cache activation changed Parameter objects")
    _check_model(model)
    return model
