"""Bounded symbolic energy and port closures for the L10-S memory ROM.

The grammar is deliberately small and self contained.  Expressions operate on
fixed node, edge, or port feature atoms; their values and feature derivatives
are evaluated together, without differentiating through an input state.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from local_memory_rom_0098 import LocalMemoryROM, _tensor
from structured_rom import StructuredROM


_VARIABLES = {
    "node": ("p", "q"),
    "edge": ("pi", "pj", "qi", "qj", "s", "c"),
    "port": ("w", "p", "q", "es", "ec", "nq"),
}
_MAX_TERMS = 12
_MAX_DEPTH = 3
_MAX_OP_NODES = 12
_MAX_TANH = 1
_DEGREE_LIMIT = {"node": 4, "edge": 4, "port": 3}
_BASE_TERMS = [
    {"channel": "node", "expression": {"op": "mul", "args": [
        {"op": "var", "name": "p"}, {"op": "var", "name": "q"}]}},
    {"channel": "edge", "expression": {"op": "mul", "args": [
        {"op": "add", "args": [
            {"op": "var", "name": "qi"},
            {"op": "mul", "args": [{"op": "const", "value": -1}, {"op": "var", "name": "qj"}]}]},
        {"op": "var", "name": "s"}]}},
    {"channel": "edge", "expression": {"op": "mul", "args": [
        {"op": "add", "args": [{"op": "var", "name": "qi"}, {"op": "var", "name": "qj"}]},
        {"op": "var", "name": "c"}]}},
]
_BASE_TERMS.extend({"channel": "port", "expression": {"op": "var", "name": name}}
                   for name in ("w", "p", "q", "es", "ec", "nq"))


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonicalize_expression(expression: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable, recursively canonical JSON AST."""
    if not isinstance(expression, Mapping) or "op" not in expression:
        raise ValueError("expression nodes must be JSON objects with an op")
    op = expression["op"]
    if op == "var":
        if set(expression) != {"op", "name"} or not isinstance(expression["name"], str):
            raise ValueError("var nodes require exactly a string name")
        return {"op": "var", "name": expression["name"]}
    if op == "const":
        if set(expression) != {"op", "value"} or expression["value"] not in (-1, 1) or isinstance(expression["value"], bool):
            raise ValueError("const nodes support exactly -1 or 1")
        return {"op": "const", "value": int(expression["value"])}
    if op in ("add", "mul"):
        if set(expression) != {"op", "args"} or not isinstance(expression["args"], (list, tuple)) or len(expression["args"]) < 2:
            raise ValueError(f"{op} nodes require at least two arguments")
        args: list[dict[str, Any]] = []
        for child in expression["args"]:
            c = canonicalize_expression(child)
            args.extend(c["args"] if c["op"] == op else [c])
        args.sort(key=_json_key)
        if op == "mul":
            sign = 1
            rest = []
            for child in args:
                if child.get("op") == "const":
                    sign *= child["value"]
                else:
                    rest.append(child)
            if not rest:
                return {"op": "const", "value": sign}
            if sign < 0:
                rest.append({"op": "const", "value": -1})
            args = sorted(rest, key=_json_key)
        return args[0] if len(args) == 1 else {"op": op, "args": args}
    if op in ("pow2", "pow3", "pow4"):
        if set(expression) != {"op", "arg"}:
            raise ValueError(f"{op} nodes require exactly one arg")
        return {"op": op, "arg": canonicalize_expression(expression["arg"])}
    if op == "tanh":
        if set(expression) != {"op", "arg", "scale"} or expression["scale"] not in (.5, 1, 2):
            raise ValueError("tanh nodes require scale .5, 1, or 2 and exactly one arg")
        return {"op": "tanh", "arg": canonicalize_expression(expression["arg"]),
                "scale": float(expression["scale"])}
    raise ValueError(f"unsupported AST op: {op!r}")


def _expression_stats(expression: Mapping[str, Any], variables: tuple[str, ...]) -> tuple[int, int, int, int]:
    """Return depth, op-node count, tanh count, and conservative degree."""
    op = expression["op"]
    if op in ("var", "const"):
        return 0, 0, 0, 1 if op == "var" else 0
    children = expression["args"] if op in ("add", "mul") else [expression["arg"]]
    stats = [_expression_stats(x, variables) for x in children]
    depth = 1 + max(s[0] for s in stats)
    nodes = 1 + sum(s[1] for s in stats)
    tanh_count = (1 if op == "tanh" else 0) + sum(s[2] for s in stats)
    if op == "add":
        degree = max(s[3] for s in stats)
    elif op == "mul":
        degree = sum(s[3] for s in stats)
    elif op.startswith("pow"):
        degree = stats[0][3] * int(op[-1])
    else:  # Bound the argument too, so tanh cannot hide a high-degree subtree.
        degree = stats[0][3]
    return depth, nodes, tanh_count, degree


def validate_expression(expression: Mapping[str, Any], channel: str) -> dict[str, Any]:
    """Canonicalize and validate one expression for its evaluation channel."""
    if channel not in _VARIABLES:
        raise ValueError("channel must be node, edge, or port")
    ast = canonicalize_expression(expression)

    def check_vars(node: Mapping[str, Any]) -> None:
        if node["op"] == "var":
            if node["name"] not in _VARIABLES[channel]:
                raise ValueError(f"variable {node['name']!r} is not allowed in {channel}")
        elif node["op"] in ("add", "mul"):
            for child in node["args"]:
                check_vars(child)
        elif node["op"] not in ("const",):
            check_vars(node["arg"])

    check_vars(ast)
    depth, op_nodes, tanh_count, degree = _expression_stats(ast, _VARIABLES[channel])
    if depth > _MAX_DEPTH:
        raise ValueError(f"expression depth exceeds {_MAX_DEPTH}")
    if op_nodes > _MAX_OP_NODES:
        raise ValueError(f"expression has more than {_MAX_OP_NODES} operator nodes")
    if tanh_count > _MAX_TANH or (tanh_count and channel != "port"):
        raise ValueError("at most one tanh is allowed, and only in port expressions")
    if degree > _DEGREE_LIMIT[channel]:
        raise ValueError(f"expression degree exceeds {_DEGREE_LIMIT[channel]} for {channel}")
    return ast


def expression_key(expression: Mapping[str, Any]) -> str:
    """Stable identity for an AST, independent of add/mul argument order."""
    return _json_key(canonicalize_expression(expression))


def _canonical_terms(terms: Any) -> list[dict[str, Any]]:
    if not isinstance(terms, (list, tuple)):
        raise ValueError("terms must be a list")
    if len(terms) > _MAX_TERMS:
        raise ValueError(f"at most {_MAX_TERMS} symbolic terms are allowed")
    out = []
    keys = set()
    tanh_total = 0
    for term in terms:
        if not isinstance(term, Mapping) or set(term) != {"channel", "expression"}:
            raise ValueError("each term requires channel and expression")
        channel = term["channel"]
        ast = validate_expression(term["expression"], channel)
        tanh_total += _expression_stats(ast, _VARIABLES[channel])[2]
        item = {"channel": channel, "expression": ast}
        key = term_key(item)
        if key in keys:
            raise ValueError("duplicate symbolic terms are not allowed")
        keys.add(key)
        out.append(item)
    if tanh_total > _MAX_TANH:
        raise ValueError("at most one tanh is allowed across the model")
    return out


def term_key(term: Mapping[str, Any]) -> str:
    """Stable identity for a channel/expression pair."""
    if not isinstance(term, Mapping) or set(term) != {"channel", "expression"}:
        raise ValueError("term requires channel and expression")
    channel = term["channel"]
    ast = validate_expression(term["expression"], channel)
    return _json_key({"channel": channel, "expression": ast})


def _torch_eval(ast: Mapping[str, Any], features: torch.Tensor,
                variables: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate an AST and its exact feature gradient using torch operations."""
    op = ast["op"]
    if op == "var":
        i = variables.index(ast["name"])
        value = features[..., i]
        grad = torch.zeros_like(features)
        grad[..., i] = 1.0
        return value, grad
    if op == "const":
        return features[..., 0] * 0 + ast["value"], torch.zeros_like(features)
    if op in ("add", "mul"):
        args = [_torch_eval(child, features, variables) for child in ast["args"]]
        value, grad = args[0]
        for other, other_grad in args[1:]:
            if op == "add":
                value, grad = value + other, grad + other_grad
            else:
                grad, value = grad * other.unsqueeze(-1) + other_grad * value.unsqueeze(-1), value * other
        return value, grad
    value, grad = _torch_eval(ast["arg"], features, variables)
    if op.startswith("pow"):
        power = int(op[-1])
        return value.pow(power), grad * (power * value.pow(power - 1)).unsqueeze(-1)
    scale = ast["scale"]
    result = torch.tanh(scale * value)
    return result, grad * (scale * (1 - result.square())).unsqueeze(-1)


def _numpy_eval(ast: Mapping[str, Any], features: np.ndarray,
                variables: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy AST evaluator and analytic feature gradient."""
    op = ast["op"]
    if op == "var":
        i = variables.index(ast["name"])
        grad = np.zeros_like(features)
        grad[..., i] = 1.0
        return features[..., i], grad
    if op == "const":
        return features[..., 0] * 0 + ast["value"], np.zeros_like(features)
    if op in ("add", "mul"):
        args = [_numpy_eval(child, features, variables) for child in ast["args"]]
        value, grad = args[0]
        for other, other_grad in args[1:]:
            if op == "add":
                value, grad = value + other, grad + other_grad
            else:
                grad, value = grad * other[..., None] + other_grad * value[..., None], value * other
        return value, grad
    value, grad = _numpy_eval(ast["arg"], features, variables)
    if op.startswith("pow"):
        power = int(op[-1])
        return value**power, grad * (power * value**(power - 1))[..., None]
    result = np.tanh(ast["scale"] * value)
    return result, grad * (ast["scale"] * (1 - result**2))[..., None]


def _torch_value(ast: Mapping[str, Any], features: torch.Tensor,
                 variables: tuple[str, ...]) -> torch.Tensor:
    """Value-only evaluator used by port closures on the hot path."""
    op = ast["op"]
    if op == "var":
        return features[..., variables.index(ast["name"])]
    if op == "const":
        return features[..., 0] * 0 + ast["value"]
    if op == "add":
        return torch.stack([_torch_value(child, features, variables)
                            for child in ast["args"]], dim=0).sum(0)
    if op == "mul":
        values = [_torch_value(child, features, variables) for child in ast["args"]]
        result = values[0]
        for value in values[1:]:
            result = result * value
        return result
    value = _torch_value(ast["arg"], features, variables)
    if op.startswith("pow"):
        return value.pow(int(op[-1]))
    return torch.tanh(ast["scale"] * value)


def _compiled_term_kind(term: Mapping[str, Any]) -> str:
    """Select exact algebraic fast paths for the seeded expressions."""
    key = term_key(term)
    for index, baseline in enumerate(_BASE_TERMS[:3]):
        if key == term_key(baseline):
            return ("node_pq", "edge_sine", "edge_cosine")[index]
    if term["channel"] == "port" and term["expression"]["op"] == "var":
        return "port_" + term["expression"]["name"]
    return "generic_" + term["channel"]


def _ast_at_zero(ast: Mapping[str, Any]) -> float:
    """Evaluate an expression at the physical feature origin without tensors."""
    op = ast["op"]
    if op == "var":
        return 0.0
    if op == "const":
        return float(ast["value"])
    if op == "add":
        return sum(_ast_at_zero(child) for child in ast["args"])
    if op == "mul":
        value = 1.0
        for child in ast["args"]:
            value *= _ast_at_zero(child)
        return value
    value = _ast_at_zero(ast["arg"])
    if op.startswith("pow"):
        return value ** int(op[-1])
    return float(np.tanh(ast["scale"] * value))


def _torch_feature_eval(ast: Mapping[str, Any], features: torch.Tensor,
                        variables: tuple[str, ...], channel: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate an expression with physical-origin centering."""
    value, grad = _torch_eval(ast, features, variables)
    if channel == "edge":
        reverse_features = torch.stack((features[..., 1], features[..., 0], features[..., 3],
                                        features[..., 2], -features[..., 4], features[..., 5]), -1)
        reverse_value, reverse_grad = _torch_eval(ast, reverse_features, variables)
        reverse_grad = torch.stack((reverse_grad[..., 1], reverse_grad[..., 0],
                                    reverse_grad[..., 3], reverse_grad[..., 2],
                                    -reverse_grad[..., 4], reverse_grad[..., 5]), -1)
        value = .5 * (value + reverse_value)
        grad = .5 * (grad + reverse_grad)
    zero_features = torch.zeros_like(features)
    zero_value, zero_grad = _torch_eval(ast, zero_features, variables)
    if channel == "edge":
        reverse_zero = torch.stack((zero_features[..., 1], zero_features[..., 0],
                                    zero_features[..., 3], zero_features[..., 2],
                                    -zero_features[..., 4], zero_features[..., 5]), -1)
        reverse_zero_value, reverse_zero_grad = _torch_eval(ast, reverse_zero, variables)
        reverse_zero_grad = torch.stack((reverse_zero_grad[..., 1], reverse_zero_grad[..., 0],
                                         reverse_zero_grad[..., 3], reverse_zero_grad[..., 2],
                                         -reverse_zero_grad[..., 4], reverse_zero_grad[..., 5]), -1)
        zero_value = .5 * (zero_value + reverse_zero_value)
        zero_grad = .5 * (zero_grad + reverse_zero_grad)
    if channel == "port":
        return value - zero_value, grad
    return value - zero_value - (zero_grad * features).sum(-1), grad - zero_grad


def _numpy_feature_eval(ast: Mapping[str, Any], features: np.ndarray,
                        variables: tuple[str, ...], channel: str) -> tuple[np.ndarray, np.ndarray]:
    """NumPy counterpart of the centered torch feature evaluator."""
    value, grad = _numpy_eval(ast, features, variables)
    if channel == "edge":
        reverse_features = np.stack((features[..., 1], features[..., 0], features[..., 3],
                                     features[..., 2], -features[..., 4], features[..., 5]), axis=-1)
        reverse_value, reverse_grad = _numpy_eval(ast, reverse_features, variables)
        reverse_grad = np.stack((reverse_grad[..., 1], reverse_grad[..., 0],
                                 reverse_grad[..., 3], reverse_grad[..., 2],
                                 -reverse_grad[..., 4], reverse_grad[..., 5]), axis=-1)
        value = .5 * (value + reverse_value)
        grad = .5 * (grad + reverse_grad)
    zero_features = np.zeros_like(features)
    zero_value, zero_grad = _numpy_eval(ast, zero_features, variables)
    if channel == "edge":
        reverse_zero = np.stack((zero_features[..., 1], zero_features[..., 0],
                                 zero_features[..., 3], zero_features[..., 2],
                                 -zero_features[..., 4], zero_features[..., 5]), axis=-1)
        reverse_zero_value, reverse_zero_grad = _numpy_eval(ast, reverse_zero, variables)
        reverse_zero_grad = np.stack((reverse_zero_grad[..., 1], reverse_zero_grad[..., 0],
                                      reverse_zero_grad[..., 3], reverse_zero_grad[..., 2],
                                      -reverse_zero_grad[..., 4], reverse_zero_grad[..., 5]), axis=-1)
        zero_value = .5 * (zero_value + reverse_zero_value)
        zero_grad = .5 * (zero_grad + reverse_zero_grad)
    if channel == "port":
        return value - zero_value, grad
    return value - zero_value - (zero_grad * features).sum(-1), grad - zero_grad


def evaluate_expression_torch(expression: Mapping[str, Any], features: torch.Tensor,
                              channel: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered expression values and analytic feature gradients."""
    ast = validate_expression(expression, channel)
    f = features if torch.is_tensor(features) else _tensor(features)
    if f.shape[-1] != len(_VARIABLES[channel]):
        raise ValueError(f"{channel} features must end in {len(_VARIABLES[channel])} values")
    return _torch_feature_eval(ast, f, _VARIABLES[channel], channel)


def evaluate_expression_numpy(expression: Mapping[str, Any], features: Any,
                              channel: str) -> tuple[np.ndarray, np.ndarray]:
    """Independent NumPy version of the centered AST evaluator."""
    ast = validate_expression(expression, channel)
    f = np.asarray(features, dtype=np.float64)
    if f.shape[-1] != len(_VARIABLES[channel]):
        raise ValueError(f"{channel} features must end in {len(_VARIABLES[channel])} values")
    return _numpy_feature_eval(ast, f, _VARIABLES[channel], channel)


def _torch_features(model: "SymbolicClosureROM", x: torch.Tensor):
    d, pi, p, q = model._split(x)
    pnorm = p / model.p_scale
    eta = d @ model.base_D.T
    angle = model.phi + eta
    sine = torch.sin(angle) - torch.sin(model.phi)
    cosine = torch.cos(angle) - torch.cos(model.phi)
    ds = torch.cos(angle)[..., :, None] * model.base_D
    dc = -torch.sin(angle)[..., :, None] * model.base_D
    positive = model.base_D.clamp_min(0)
    negative = (-model.base_D).clamp_min(0)
    ep = torch.einsum("...n,en->...e", pnorm, positive)
    ej = torch.einsum("...n,en->...e", pnorm, negative)
    eqi = torch.einsum("...n,en->...e", q, positive)
    eqj = torch.einsum("...n,en->...e", q, negative)
    edge = torch.stack((ep, ej, eqi, eqj, sine, cosine), -1)
    node = torch.stack((pnorm, q), -1)

    w = (pi / model.mass) / model.input_scale_omega
    es = sine @ model.port_s_weight
    ec = cosine @ model.port_c_weight
    nq = q @ model.neighbor_A.T - q
    port = torch.stack((w, pnorm, q, es, ec, nq), -1)
    return node, edge, port, (ds, dc, positive, negative)


def _torch_term_value_gradient(model: "SymbolicClosureROM", term: Mapping[str, Any],
                               features, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channel = term["channel"]
    vals = {"node": _VARIABLES["node"], "edge": _VARIABLES["edge"], "port": _VARIABLES["port"]}
    fdict = dict(zip(("node", "edge", "port"), features[:3]))
    f = fdict[channel]
    value, grad = _torch_feature_eval(term["expression"], f, vals[channel], channel)
    d, pi, p, q = model._split(x)
    gd = torch.zeros_like(d)
    gp = torch.zeros_like(p)
    gq = torch.zeros_like(q)
    if channel == "node":
        contribution = value.sum(-1)
        gp = grad[..., :, 0] / model.p_scale
        gq = grad[..., :, 1]
    elif channel == "edge":
        ds, dc, positive, negative = features[3]
        norm_k = model.base_coupling / model.base_coupling.mean().clamp_min(1e-15)
        weight = norm_k
        contribution = (weight * value).sum(-1)
        weighted_grad = weight[..., None] * grad
        gp = (torch.einsum("...e,en->...n", weighted_grad[..., 0], positive)
              + torch.einsum("...e,en->...n", weighted_grad[..., 1], negative)) / model.p_scale
        gq = (torch.einsum("...e,en->...n", weighted_grad[..., 2], positive)
              + torch.einsum("...e,en->...n", weighted_grad[..., 3], negative))
        gd = (torch.einsum("...e,...en->...n", weighted_grad[..., 4], ds)
              + torch.einsum("...e,...en->...n", weighted_grad[..., 5], dc))
    else:
        contribution = value.sum(-1)
        # Port terms are one scalar contribution per machine.  Their energy
        # gradient is unused; port_values consumes the value only.
    full_grad = torch.cat((gd, torch.zeros_like(pi), gp, gq), -1)
    return contribution, full_grad


class SymbolicClosureROM(LocalMemoryROM):
    """L10-S memory ROM with a bounded symbolic H and port expression list."""

    model_kind = "symbolic_closure_v1"

    def __init__(self, base: Any, input_scale: Any = None, p_scale: Any = None,
                 arm: str = "L10-S", seed: int = 98,
                 terms: list[dict[str, Any]] | None = None,
                 coefficients: Any = None):
        if arm != "L10-S":
            raise ValueError("symbolic_closure_v1 requires the L10-S arm")
        super().__init__(base, input_scale=input_scale, p_scale=p_scale, arm=arm, seed=seed)
        self.terms = _canonical_terms(_BASE_TERMS if terms is None else terms)
        self._term_kinds = [_compiled_term_kind(term) for term in self.terms]
        self._port_zero_values = [_ast_at_zero(term["expression"]) if term["channel"] == "port" else 0.0
                                  for term in self.terms]
        initial = torch.zeros(len(self.terms), dtype=self.mass.dtype, device=self.mass.device)
        if coefficients is not None:
            supplied = torch.as_tensor(coefficients, dtype=self.mass.dtype, device=self.mass.device).reshape(-1)
            if supplied.numel() != len(self.terms) or not torch.isfinite(supplied).all():
                raise ValueError("coefficients must be a finite value for every term")
            initial.copy_(supplied)
        self.coefficients = nn.Parameter(initial)
        with torch.no_grad():
            self.template_mask[:3] = 0
            self.template_mask[7] = 0
        for parameter in list(self._template_scalars[:3]) + [self.port_raw]:
            parameter.requires_grad_(False)

    @classmethod
    def _base_from_model(cls, source: LocalMemoryROM) -> StructuredROM:
        structured = source.structured
        base = StructuredROM(
            mass=source.mass, D=source.base_D, coupling=source.base_coupling,
            delta_star=source.delta_star,
            damping=structured._pos("damping").detach(), gain=source.gain,
            jpi_p=structured._pos("jpi_p").detach(),
            rpp=structured._pos("rpp").detach(),
            storage_base=structured._pos("storage_base").detach()[:source.n],
            n_q=0, storage=(), trainable=True,
        ).double()
        base.rate = source.rate.detach().clone()
        return base

    @classmethod
    def from_source(cls, source: LocalMemoryROM,
                    terms: list[dict[str, Any]] | None = None) -> "SymbolicClosureROM":
        """Copy a source S23/LocalMemoryROM exactly and swap H/port closures."""
        if not isinstance(source, LocalMemoryROM) or source.arm != "L10-S":
            raise ValueError("source must be an L10-S LocalMemoryROM or S23")
        wanted = _canonical_terms(_BASE_TERMS if terms is None else terms)
        model = cls(cls._base_from_model(source), input_scale=source.input_scale.detach(),
                    p_scale=source.p_scale.detach(), arm="L10-S", seed=source.seed,
                    terms=wanted)
        # The wrapper's active-edge buffer records the original graph mask and
        # can be longer than the already-filtered graph matrices.
        model.active_edge = source.active_edge.detach().clone()
        copied, unexpected = model.load_state_dict(source.state_dict(), strict=False)
        if unexpected or set(copied) != {"coefficients"}:
            raise ValueError(f"source state is incompatible; missing={copied}, unexpected={unexpected}")
        src_params = dict(source.named_parameters())
        for name, parameter in model.named_parameters():
            if name in src_params:
                parameter.requires_grad_(src_params[name].requires_grad)

        defaults = {term_key(t): float(source.template_coefficients[i].detach()) for i, t in enumerate(_BASE_TERMS[:3])}
        port_scale = float(source.template_mask[7].detach())
        defaults.update({term_key(t): float((source.port_raw[i] * port_scale).detach())
                         for i, t in enumerate(_BASE_TERMS[3:])})
        with torch.no_grad():
            model.coefficients.copy_(torch.tensor(
                [defaults.get(term_key(t), 0.0) for t in model.terms],
                dtype=model.coefficients.dtype, device=model.coefficients.device))
            model.template_mask[:3] = 0
            model.template_mask[7] = 0
        for parameter in model._template_scalars[:3]:
            parameter.requires_grad_(False)
        model.port_raw.requires_grad_(False)
        model._graph_cache = None
        return model

    def with_terms(self, terms: list[dict[str, Any]]) -> "SymbolicClosureROM":
        """Return a copy with matching coefficients retained and new ones zero."""
        canonical = _canonical_terms(terms)
        old = {term_key(t): self.coefficients[i].detach().clone()
               for i, t in enumerate(self.terms)}
        result = copy.deepcopy(self)
        result.terms = canonical
        result._term_kinds = [_compiled_term_kind(term) for term in canonical]
        result._port_zero_values = [_ast_at_zero(term["expression"]) if term["channel"] == "port" else 0.0
                                    for term in canonical]
        values = [old.get(term_key(t), self.coefficients.new_zeros(())).clone() for t in canonical]
        result.coefficients = nn.Parameter(
            torch.stack(values) if values else self.coefficients.new_empty((0,)),
            requires_grad=self.coefficients.requires_grad)
        return result

    def energy_and_gradient(self, internal: torch.Tensor):
        x = internal if torch.is_tensor(internal) else _tensor(internal)
        x = x.to(dtype=self.mass.dtype, device=self.mass.device)
        # The three legacy H templates are masked and frozen in this model.
        # Evaluate only the inherited M3 skeleton and retained q quadratic,
        # avoiding construction of the unused legacy H feature graph.
        base_energy, base_gradient = self.structured.energy_and_gradient(x)
        d, pi, p, q = self._split(x)
        q_quadratic = self.q_quadratic
        q_energy = .5 * q_quadratic * q.square().sum(-1)
        q_gradient = torch.cat((torch.zeros_like(x[..., :30]), q_quadratic * q), -1)
        base_energy = base_energy + q_energy
        base_gradient = base_gradient + q_gradient
        extra_energy = x.new_zeros(x.shape[:-1])
        gd = torch.zeros_like(d)
        gp = torch.zeros_like(p)
        gq = torch.zeros_like(q)
        generic_features = None
        eta = None
        sine = None
        cosine = None
        norm_k = None
        for coefficient, term, kind in zip(self.coefficients, self.terms, self._term_kinds):
            if term["channel"] == "port":
                continue
            if kind == "node_pq":
                pnorm = p / self.p_scale
                extra_energy = extra_energy + coefficient * (pnorm * q).sum(-1)
                gp = gp + coefficient * q / self.p_scale
                gq = gq + coefficient * pnorm
            elif kind in ("edge_sine", "edge_cosine"):
                if eta is None:
                    eta = d @ self.base_D.T
                    angle = self.phi + eta
                    sine = torch.sin(angle) - torch.sin(self.phi)
                    cosine = torch.cos(angle) - torch.cos(self.phi)
                    norm_k = self.base_coupling / self.base_coupling.mean().clamp_min(1e-15)
                if kind == "edge_sine":
                    edge_difference = q @ self.base_D.T
                    extra_energy = extra_energy + coefficient * (norm_k * edge_difference * sine).sum(-1)
                    gd = gd + coefficient * ((norm_k * edge_difference * torch.cos(self.phi + eta)) @ self.base_D)
                    gq = gq + coefficient * ((norm_k * sine) @ self.base_D)
                else:
                    endpoint_sum = q @ torch.abs(self.base_D).T
                    extra_energy = extra_energy + coefficient * (norm_k * endpoint_sum * cosine).sum(-1)
                    gd = gd + coefficient * ((-norm_k * endpoint_sum * torch.sin(self.phi + eta)) @ self.base_D)
                    gq = gq + coefficient * ((norm_k * cosine) @ torch.abs(self.base_D))
            else:
                if generic_features is None:
                    generic_features = _torch_features(self, x)
                value, gradient = _torch_term_value_gradient(self, term, generic_features, x)
                extra_energy = extra_energy + coefficient * value
                gd = gd + coefficient * gradient[..., :10]
                gp = gp + coefficient * gradient[..., 20:30]
                gq = gq + coefficient * gradient[..., 30:]
        extra_gradient = torch.cat((gd, torch.zeros_like(pi), gp, gq), -1)
        return base_energy + extra_energy, base_gradient + extra_gradient

    def port_values(self, internal: torch.Tensor) -> torch.Tensor:
        x = internal if torch.is_tensor(internal) else _tensor(internal)
        x = x.to(dtype=self.mass.dtype, device=self.mass.device)
        if x.shape[-1] != self.dim:
            x = self.interface_to_state(x)
        out = x.new_zeros(x.shape[:-1] + (self.n,))
        port_features = None
        pnorm = None
        sine = None
        cosine = None
        w = None
        nq = None
        for index, (coefficient, term, kind) in enumerate(zip(self.coefficients, self.terms, self._term_kinds)):
            if term["channel"] != "port":
                continue
            if kind.startswith("port_"):
                name = kind[5:]
                d, pi, p, q = self._split(x)
                if name == "w":
                    if w is None:
                        w = (pi / self.mass) / self.input_scale_omega
                    value = w
                elif name == "p":
                    if pnorm is None:
                        pnorm = p / self.p_scale
                    value = pnorm
                elif name == "q":
                    value = q
                elif name in ("es", "ec"):
                    if sine is None:
                        eta = d @ self.base_D.T
                        angle = self.phi + eta
                        sine = torch.sin(angle) - torch.sin(self.phi)
                        cosine = torch.cos(angle) - torch.cos(self.phi)
                    value = sine @ self.port_s_weight if name == "es" else cosine @ self.port_c_weight
                else:  # nq
                    if nq is None:
                        nq = q @ self.neighbor_A.T - q
                    value = nq
            else:
                if port_features is None:
                    port_features = self._port_feature_values(x)
                value = _torch_value(term["expression"], port_features, _VARIABLES["port"])
                value = value - self._port_zero_values[index]
            out = out + coefficient * value
        return out

    def _port_feature_values(self, x: torch.Tensor) -> torch.Tensor:
        """Build the six port atoms only for a non-seeded port expression."""
        d, pi, p, q = self._split(x)
        eta = d @ self.base_D.T
        angle = self.phi + eta
        sine = torch.sin(angle) - torch.sin(self.phi)
        cosine = torch.cos(angle) - torch.cos(self.phi)
        return torch.stack(((pi / self.mass) / self.input_scale_omega,
                            p / self.p_scale,
                            q,
                            sine @ self.port_s_weight,
                            cosine @ self.port_c_weight,
                            q @ self.neighbor_A.T - q), dim=-1)

    def complexity(self, state: torch.Tensor | None = None) -> torch.Tensor:
        del state
        return self.coefficients.abs().sum() + self.template_coefficients[3:7].abs().sum()

    def complexity_counts(self) -> dict[str, Any]:
        inherited = super().complexity_counts()
        per_channel = {key: 0 for key in _VARIABLES}
        op_nodes = 0
        active_terms = 0
        for coefficient, term in zip(self.coefficients.detach(), self.terms):
            per_channel[term["channel"]] += 1
            _, nodes, _, _ = _expression_stats(term["expression"], _VARIABLES[term["channel"]])
            op_nodes += nodes
            active_terms += int(coefficient != 0)
        inherited["symbolic_closure"] = {
            "expressions": len(self.terms), "active_expressions": active_terms,
            "expressions_by_channel": per_channel, "operator_nodes": op_nodes,
            "max_expressions": _MAX_TERMS, "max_depth": _MAX_DEPTH,
            "expanded_symbolic_rhs_contributions": "feature derivatives are evaluated analytically; inherited count excludes symbolic terms",
        }
        return inherited

    def config(self) -> dict[str, Any]:
        config = super().config()
        config["storage_base"] = self.structured._pos("storage_base").detach().tolist()
        return config

    def _export_payload(self) -> dict[str, Any]:
        J, R = self.matrices()
        physical = {
            "mass": self.mass.detach().cpu().tolist(),
            "D": self.base_D.detach().cpu().tolist(),
            "delta_star": self.delta_star.detach().cpu().tolist(),
            "phi": self.phi.detach().cpu().tolist(),
            "graph_coupling": self.base_coupling.detach().cpu().tolist(),
            "m3_coupling": self.structured._pos("coupling").detach().cpu().tolist(),
            "storage_base": self.structured._pos("storage_base").detach().cpu().tolist(),
            "input_scale_omega": self.input_scale_omega.detach().cpu().tolist(),
            "p_scale": self.p_scale.detach().cpu().tolist(),
            "neighbor_A": self.neighbor_A.detach().cpu().tolist(),
            "port_s_weight": self.port_s_weight.detach().cpu().tolist(),
            "port_c_weight": self.port_c_weight.detach().cpu().tolist(),
            "J": J.detach().cpu().tolist(), "R": R.detach().cpu().tolist(),
            "B": self.port_matrix().detach().cpu().tolist(),
        }
        return {
            "format": "symbolic_closure_v1", "model_kind": self.model_kind,
            "config": self.config(), "terms": copy.deepcopy(self.terms),
            "coefficients": self.coefficients.detach().cpu().tolist(),
            "state_dict": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
            "physical": physical, "counts": self.complexity_counts(),
        }

    def export(self, path: str | Path) -> dict[str, Any]:
        payload = self._export_payload()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any],
                    terms: list[dict[str, Any]] | None = None) -> "SymbolicClosureROM":
        base = StructuredROM(
            mass=config["mass"], D=config["D"], coupling=config["coupling"],
            delta_star=config.get("delta_star"), damping=config["damping"],
            gain=config["gain"], jpi_p=config.get("jpi_p"), rpp=config.get("rpp"),
            storage_base=np.asarray(config.get("storage_base", 1 / np.asarray(config["gain"])))[:10],
            n_q=0, storage=(), trainable=True,
        ).double()
        base.rate = _tensor(config.get("rate", np.ones(10)))
        model = cls(base, input_scale=config.get("input_scale"), p_scale=config.get("p_scale"),
                    arm=config.get("arm", "L10-S"), seed=int(config.get("seed", 98)), terms=terms)
        if "active_edge" in config:
            model.active_edge = torch.as_tensor(config["active_edge"], dtype=torch.bool)
        return model

    @classmethod
    def from_export(cls, path: str | Path) -> "SymbolicClosureROM":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "symbolic_closure_v1":
            raise ValueError("unsupported symbolic closure export")
        model = cls.from_config(payload["config"], terms=payload["terms"])
        state = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in payload["state_dict"].items()}
        model.load_state_dict(state, strict=True)
        return model


def _numpy_term_value_gradient(term: Mapping[str, Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channel = term["channel"]
    return _numpy_feature_eval(term["expression"], features, _VARIABLES[channel], channel)


def _numpy_symbolic_field(payload: Mapping[str, Any], state: np.ndarray,
                          external_input: np.ndarray | None = None) -> np.ndarray:
    physical = payload["physical"]
    mass = np.asarray(physical["mass"], dtype=np.float64)
    D = np.asarray(physical["D"], dtype=np.float64)
    phi = np.asarray(physical["phi"], dtype=np.float64)
    n = len(mass)
    z = np.asarray(state, dtype=np.float64)
    scalar = z.ndim == 1
    z = np.atleast_2d(z)
    if z.shape[-1] != 4 * n:
        raise ValueError(f"expected interface state width {4*n}")
    d, omega, p, q = z[:, :n], z[:, n:2*n], z[:, 2*n:3*n], z[:, 3*n:]
    pi = omega * mass
    eta = d @ D.T
    angle = phi + eta
    sine = np.sin(angle) - np.sin(phi)
    cosine = np.cos(angle) - np.cos(phi)
    ds = np.cos(angle)[..., :, None] * D
    dc = -np.sin(angle)[..., :, None] * D
    positive = np.maximum(D, 0.)
    negative = np.maximum(-D, 0.)
    pscale = np.asarray(physical["p_scale"], dtype=np.float64)
    pnorm = p / pscale
    node = np.stack((pnorm, q), axis=-1)
    edge = np.stack((pnorm @ positive.T, pnorm @ negative.T,
                     q @ positive.T, q @ negative.T, sine, cosine), axis=-1)
    features = {"node": node, "edge": edge}
    gd = np.zeros_like(d)
    gp = np.zeros_like(p)
    gq = np.zeros_like(q)
    coupling = np.asarray(physical["m3_coupling"], dtype=np.float64)
    storage = np.asarray(physical["storage_base"], dtype=np.float64)
    gd += ((np.sin(angle) - np.sin(phi)) * coupling) @ D
    gpi = pi / mass
    gp += storage[:n] * p
    gq += storage[n:] * q

    # LocalMemoryROM retains its shared q quadratic outside the replaced H terms.
    st = payload["state_dict"]
    q_raw = np.asarray(st["_q_quadratic_raw"], dtype=np.float64).reshape(()).item()
    qquadratic = np.logaddexp(0., q_raw)
    gq += qquadratic * q

    norm_k = np.asarray(physical["graph_coupling"], dtype=np.float64)
    norm_k = norm_k / max(norm_k.mean(), 1e-15)
    coeffs = np.asarray(payload["coefficients"], dtype=np.float64)
    for coefficient, term in zip(coeffs, payload["terms"]):
        channel = term["channel"]
        if channel == "node":
            value, grad = _numpy_term_value_gradient(term, features[channel])
            gp += coefficient * (grad[..., :, 0] / pscale)
            gq += coefficient * grad[..., :, 1]
        elif channel == "edge":
            value, grad = _numpy_term_value_gradient(term, features[channel])
            weight = norm_k
            wg = coefficient * weight[..., None] * grad
            gp += (np.einsum("be,en->bn", wg[..., 0], positive)
                   + np.einsum("be,en->bn", wg[..., 1], negative)) / pscale
            gq += (np.einsum("be,en->bn", wg[..., 2], positive)
                   + np.einsum("be,en->bn", wg[..., 3], negative))
            gd += np.einsum("be,ben->bn", wg[..., 4], ds) + np.einsum("be,ben->bn", wg[..., 5], dc)

    e = np.concatenate((gd, gpi, gp, gq), axis=-1)
    if external_input is None:
        neighbor = np.asarray(physical.get("neighbor_A", []), dtype=np.float64)
        # These fixed graph maps are required to evaluate symbolic port terms.
        if neighbor.size == 0:
            raise ValueError("symbolic export is missing fixed port geometry")
        sw = np.asarray(physical["port_s_weight"], dtype=np.float64)
        cw = np.asarray(physical["port_c_weight"], dtype=np.float64)
        w = omega / np.asarray(physical["input_scale_omega"], dtype=np.float64)
        es = sine @ sw
        ec = cosine @ cw
        nq = q @ neighbor.T - q
        port_features = np.stack((w, pnorm, q, es, ec, nq), axis=-1)
        port = np.zeros((len(z), n), dtype=np.float64)
        for coefficient, term in zip(coeffs, payload["terms"]):
            if term["channel"] == "port":
                value, _ = _numpy_term_value_gradient(term, port_features)
                port += coefficient * value
    else:
        port = np.asarray(external_input, dtype=np.float64)
        if port.shape[-1] != n:
            raise ValueError(f"external input must have width {n}")
        port = np.atleast_2d(port)
    J = np.asarray(physical["J"], dtype=np.float64)
    R = np.asarray(physical["R"], dtype=np.float64)
    B = np.asarray(physical["B"], dtype=np.float64)
    f = e @ (J - R).T + port @ B.T
    out = np.concatenate((f[:, :n], f[:, n:2*n] / mass, f[:, 2*n:3*n], f[:, 3*n:]), axis=-1)
    return out[0] if scalar else out


def numpy_field(model_or_export: SymbolicClosureROM | Mapping[str, Any], state: Any,
                u: Any = None) -> np.ndarray:
    """Independent NumPy interface-field evaluation for a model or v1 export."""
    payload = model_or_export._export_payload() if isinstance(model_or_export, SymbolicClosureROM) else model_or_export
    if payload.get("format") != "symbolic_closure_v1":
        raise ValueError("expected symbolic_closure_v1 model or export")
    # Keep fixed graph summaries in the export used by this evaluator.
    return _numpy_symbolic_field(payload, np.asarray(state), None if u is None else np.asarray(u))


__all__ = [
    "SymbolicClosureROM", "validate_expression", "canonicalize_expression",
    "expression_key", "term_key", "evaluate_expression_torch",
    "evaluate_expression_numpy", "numpy_field",
]
