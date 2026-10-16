"""Export a self-contained NumPy simulator for a symbolic closure ROM."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import symbolic_closure_rom as _symbolic
from symbolic_closure_rom import SymbolicClosureROM


_NUMPY_RUNTIME = r'''
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
'''


def _simulate_source() -> str:
    """Assemble the runtime from current evaluator sources at export time."""
    helper_names = ["_json_key"]
    if hasattr(_symbolic, "_origin_expression"):
        helper_names.append("_origin_expression")
    helper_names.extend((
        "_numpy_eval", "_numpy_feature_eval", "_numpy_term_value_gradient",
        "_numpy_symbolic_field",
    ))
    variables = repr(_symbolic._VARIABLES)
    helpers = "\n\n".join(inspect.getsource(getattr(_symbolic, name)).strip()
                            for name in helper_names)
    prelude = '''"""Self-contained NumPy runtime exported from a symbolic closure ROM."""
from __future__ import annotations

import functools
import json
from typing import Any, Mapping

import numpy as np

_VARIABLES = ''' + variables + "\n\n"
    return prelude + helpers + "\n\n" + _NUMPY_RUNTIME.lstrip()


def _equations_markdown(payload: dict[str, Any]) -> str:
    physical = payload["physical"]
    terms = payload["terms"]
    coefficients = payload["coefficients"]
    lines = [
        "# Symbolic closure ROM equations",
        "",
        "This package evaluates the serialized L10-S model in its formal model-storage units; these values are not SI units. The serialized scales are used as-is, without conversion.",
        "",
        "## State and field",
        "",
        "The interface state is `z = (delta, omega, p, q)` with 10 coordinates per block. The energy-gradient coordinates use `x = (delta, pi, p, q)` where `pi = mass * omega`. The exported NumPy field evaluates the serialized `J`, `R`, and `B` matrices as `f = (J - R) grad(H) + B u`, with the model's port closure supplying `u`. It converts the second block back by dividing by mass; the displacement kinematics are `delta_dot = omega`.",
        "",
        "The physical phase is `phi = D delta_star`; edge phase changes are `eta = D delta`, with `s = sin(phi + eta) - sin(phi)` and `c = cos(phi + eta) - cos(phi)`. The fixed M3 energy is represented by the serialized coupling and storage arrays. The retained shared quadratic contributes `0.5 * q_quadratic * sum(q**2)`.",
        "",
        "## Feature scaling and AST terms",
        "",
        "Node features are `(p / p_scale, q)`; `q` remains in the serialized raw state scale. Edge features are `(p_i, p_j, q_i, q_j, s, c)`, where the endpoint powers are normalized by `p_scale` and use the positive and negative parts of `D`. The AST names `pi` and `pj` denote these endpoint powers; momentum is the separate state block `pi = mass * omega`. With `eta = delta @ D.T`, the angle features are `s = sin(phi + eta) - sin(phi)` and `c = cos(phi + eta) - cos(phi)`. Edge expressions are averaged with their endpoint-reversed form (`i <-> j`, `s -> -s`, `c` unchanged), then centered in physical coordinates `(p_i, p_j, q_i, q_j, eta)` by subtracting the symmetrized equilibrium value and first derivative. After feature-origin centering, the compiler restores `g_c0 * (c + sin(phi) * eta)`, where `g_c0` is the symmetrized expression's derivative with respect to `c` at the feature origin. This retains the angle curvature of a standalone cosine expression. Node energy expressions subtract their feature-origin value and tangent. Port features are `(omega / input_scale_omega, p / p_scale, q, es, ec, nq)`; port expressions subtract their value at the feature origin. `es` and `ec` use the serialized fixed port weights, and `nq = q @ neighbor_A.T - q`.",
        "",
        "For each node AST `F`, its energy contribution is `c * sum_i [F(x_i) - F(0) - grad(F)(0) dot x_i]`. For each edge AST, let `z_e = (p_i / p_scale, p_j / p_scale, q_i, q_j, eta)` and let `G_e(z_e)` be the endpoint-symmetrized expression composed with the angle feature map above. Its contribution is `c * sum_e [k_e / mean(k) * (G_e(z_e) - G_e(0) - grad_z(G_e)(0) dot z_e)]`. A port AST contributes `u_i = c * [F(y_i) - F(0)]`. The JSON terms below are the exact ASTs, and each listed coefficient is the serialized numeric value used by the simulator.",
        "",
        "The encoder uses the original seven local features per machine: `coi = delta - sum(mass * delta) / sum(mass)`, `dnorm = coi / input_scale_delta`, `wnorm = w / input_scale_omega`, `nei_d = (coi @ neighbor_A.T - coi) / input_scale_delta`, `nei_w = (w @ neighbor_A.T) / input_scale_omega`, `mass / mean(mass)`, `rate / mean(rate)`, and `gain / mean(gain)`. These seven features feed the serialized one-layer GRU and linear head; `p` does not enter the encoder. Its latent state is `0.1 * (raw(history) - raw(zero_history))`. The history endpoint's 30 physical coordinates, including `p`, are concatenated with this latent state to initialize rollout.",
        "",
        "### Serialized symbolic terms",
        "",
    ]
    if terms:
        for index, (term, coefficient) in enumerate(zip(terms, coefficients)):
            lines.extend((
                f"{index + 1}. Channel `{term['channel']}`, coefficient `{float(coefficient):.17g}`",
                "",
                "```json",
                json.dumps(term["expression"], sort_keys=True, indent=2),
                "```",
                "",
            ))
    else:
        lines.extend(("No symbolic terms are present.", ""))
    lines.extend((
        "The symbolic complexity metadata records operator counts and limits. The inherited expanded-RHS cost field is intentionally retained as descriptive text; this package does not infer or invent an expanded contribution count.",
        "",
        "## Numerical arrays",
        "",
        f"The model has {len(physical['mass'])} machines and {len(physical['D'])} serialized edge rows. `mass`, `D`, `delta_star`, `phi`, graph coupling, M3 coupling, storage, scales, fixed port maps, and `J`/`R`/`B` are stored in `model.json`.",
        "",
    ))
    return "\n".join(lines)


def _readme() -> str:
    return """# NumPy symbolic closure package

This directory contains `model.json`, a standalone `simulate.py`, and `equations.md`.
The simulator imports only NumPy and Python standard-library modules; it does not
depend on the project source tree or PyTorch.

Run a fixed-step rollout from an 11-sample history shaped `[batch, 11, 30]`:

```text
python simulate.py model.json history.npy --steps 20 --dt 0.001 --output trajectory.npy
```

The output includes the initialized state followed by each RK4 state, with shape
`[batch, steps + 1, 40]`. See `equations.md` for feature scaling, AST terms, and
the serialized energy/field conventions.
"""


def export_numpy_package(model: SymbolicClosureROM, output_dir: str | Path) -> dict[str, Path]:
    """Write a self-contained NumPy model package and return its file paths."""
    if not isinstance(model, SymbolicClosureROM):
        raise TypeError("model must be a SymbolicClosureROM")
    payload = model._export_payload()
    scale = payload["config"]["input_scale"]
    payload["physical"]["input_scale_delta"] = scale[:10]
    payload["physical"]["gain"] = payload["config"]["gain"]
    payload["physical"]["rate"] = payload["config"]["rate"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output / "model.json",
        "simulator": output / "simulate.py",
        "equations": output / "equations.md",
        "readme": output / "README.md",
    }
    paths["model"].write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    paths["simulator"].write_text(_simulate_source(), encoding="utf-8")
    paths["equations"].write_text(_equations_markdown(payload), encoding="utf-8")
    paths["readme"].write_text(_readme(), encoding="utf-8")
    return paths


__all__ = ["export_numpy_package"]
