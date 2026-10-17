"""Self-contained NumPy runtime exported from a symbolic closure ROM."""
from __future__ import annotations

import functools
import json
from typing import Any, Mapping

import numpy as np

_VARIABLES = {'node': ('p', 'q'), 'edge': ('pi', 'pj', 'qi', 'qj', 's', 'c'), 'port': ('w', 'p', 'q', 'es', 'ec', 'nq')}

def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

@functools.lru_cache(maxsize=1024)
def _origin_expression(serialized_ast: str, channel: str):
    """Parameter-independent origin of the shared, symmetrized feature function."""
    ast=json.loads(serialized_ast)
    value,gradient=_numpy_eval(ast,np.zeros(len(_VARIABLES[channel])),_VARIABLES[channel])
    if channel=='edge':
        gradient=.5*(gradient+np.asarray([gradient[1],gradient[0],gradient[3],gradient[2],-gradient[4],gradient[5]]))
    return float(value),tuple(float(v) for v in gradient)

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
    zero_value,origin_gradient=_origin_expression(_json_key(ast),channel)
    zero_grad=np.asarray(origin_gradient,dtype=np.float64)
    if channel == "port":
        return value - zero_value, grad
    return value - zero_value - (zero_grad * features).sum(-1), grad - zero_grad

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
            _, origin_gradient = _numpy_eval(term['expression'],np.zeros(6),_VARIABLES['edge'])
            if origin_gradient[5]:
                gd += (coefficient*weight*origin_gradient[5]*(-np.sin(phi+eta)+np.sin(phi))) @ D

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

def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _encoder_features(history, physical):
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (11, 30):
        raise ValueError("history must have shape [batch, 11, 30]")
    mass = np.asarray(physical["mass"], dtype=np.float64)
    rate = np.asarray(physical.get("rate", np.ones_like(mass)), dtype=np.float64)
    gain = np.asarray(physical["gain"], dtype=np.float64)
    input_scale_delta = np.asarray(physical["input_scale_delta"], dtype=np.float64)
    input_scale_omega = np.asarray(physical["input_scale_omega"], dtype=np.float64)
    neighbor_A = np.asarray(physical["neighbor_A"], dtype=np.float64)
    d = history[..., :10]
    w = history[..., 10:20]
    coi = d - (d * mass).sum(axis=-1, keepdims=True) / mass.sum()
    dnorm = coi / input_scale_delta
    wnorm = w / input_scale_omega
    nei_d = (coi @ neighbor_A.T) - coi
    nei_w = w @ neighbor_A.T
    nei_d = nei_d / input_scale_delta
    nei_w = nei_w / input_scale_omega
    m = mass / max(float(mass.mean()), 1e-15)
    r = rate / max(float(rate.mean()), 1e-15)
    g = gain / max(float(gain.mean()), 1e-15)
    local = np.stack((dnorm, wnorm, nei_d, nei_w,
                      np.broadcast_to(m, dnorm.shape),
                      np.broadcast_to(r, dnorm.shape),
                      np.broadcast_to(g, dnorm.shape)), axis=-1)
    return local


def _local_gru_last(sequence, state_dict):
    """One-layer, batch-first PyTorch GRU in reset/update/new gate order."""
    weight_ih = np.asarray(state_dict["local_gru.weight_ih_l0"], dtype=np.float64)
    weight_hh = np.asarray(state_dict["local_gru.weight_hh_l0"], dtype=np.float64)
    bias_ih = np.asarray(state_dict["local_gru.bias_ih_l0"], dtype=np.float64)
    bias_hh = np.asarray(state_dict["local_gru.bias_hh_l0"], dtype=np.float64)
    hidden_size = weight_hh.shape[1]
    hidden = np.zeros((sequence.shape[0], hidden_size), dtype=np.float64)
    for t in range(sequence.shape[1]):
        input_gates = sequence[:, t, :] @ weight_ih.T + bias_ih
        hidden_gates = hidden @ weight_hh.T + bias_hh
        ir, iz, inn = np.split(input_gates, 3, axis=-1)
        hr, hz, hn = np.split(hidden_gates, 3, axis=-1)
        reset = _sigmoid(ir + hr)
        update = _sigmoid(iz + hz)
        new = np.tanh(inn + reset * hn)
        hidden = (1.0 - update) * new + update * hidden
    return hidden


def encode(payload, history):
    """Reproduce L10-S local 7-feature GRU encoding with NumPy only."""
    history = np.asarray(history, dtype=np.float64)
    local = _encoder_features(history, payload["physical"])
    batch, time, machines, width = local.shape
    sequence = local.transpose(0, 2, 1, 3).reshape(batch * machines, time, width)
    state_dict = payload["state_dict"]
    hidden = _local_gru_last(sequence, state_dict)
    head = np.asarray(state_dict["local_head.weight"], dtype=np.float64)
    raw = (hidden @ head.T).reshape(batch, machines)

    zero = np.zeros_like(history)
    local_zero = _encoder_features(zero, payload["physical"])
    sequence_zero = local_zero.transpose(0, 2, 1, 3).reshape(batch * machines, time, width)
    hidden_zero = _local_gru_last(sequence_zero, state_dict)
    raw_zero = (hidden_zero @ head.T).reshape(batch, machines)
    return 0.1 * (raw - raw_zero)


def field(payload, z):
    """Evaluate the exported 40-coordinate interface field."""
    return _numpy_symbolic_field(payload, np.asarray(z, dtype=np.float64))


def rollout(payload, history, steps, dt):
    """Run fixed-step RK4 without future inputs; return initial and advanced states."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 3 or history.shape[1:] != (11, 30):
        raise ValueError("history must have shape [batch, 11, 30]")
    if int(steps) != steps or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    dt = float(dt)
    state = np.concatenate((history[:, -1, :], encode(payload, history)), axis=-1)
    trajectory = [state.copy()]
    for _ in range(int(steps)):
        k1 = field(payload, state)
        k2 = field(payload, state + 0.5 * dt * k1)
        k3 = field(payload, state + 0.5 * dt * k2)
        k4 = field(payload, state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory.append(state.copy())
    return np.stack(trajectory, axis=1)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="NumPy-only symbolic closure ROM rollout")
    parser.add_argument("model", help="path to model.json")
    parser.add_argument("history", help="path to history.npy with shape [batch, 11, 30]")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--output", default="trajectory.npy")
    args = parser.parse_args(argv)
    with open(args.model, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    history = np.load(args.history)
    result = rollout(payload, history, args.steps, args.dt)
    np.save(args.output, result)
    print(f"wrote {args.output} with shape {result.shape}")


if __name__ == "__main__":
    main()
