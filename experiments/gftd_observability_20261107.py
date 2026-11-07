"""Registered, data-free local observability probe; no fitting or model selection."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates

SCALES = np.array([0.1] * 6 + [0.5] + [0.1] * 3)
THRESHOLDS = (1e-8, 1e-10, 1e-12)


def functional(modulation):
    return GeneratingFunctionalKuramoto(
        mass=np.array([1.0, 1.5, 2.0]), damping=np.full(3, 0.2),
        k_edge=np.array([1.0, 0.8, 1.2]), eta_edge=np.zeros(3),
        gamma_amp=np.full(3, 0.5), lambda_amp=np.ones(3),
        lambda_mem=np.full(3, 1 / 0.7), tau_amp=np.full(3, 5.0),
        tau_mem_sync=np.full(3, 1 / (0.7 * 0.3)), tau_mem_coi=1 / (0.7 * 0.3),
        j_omega_mem=np.full(3, 0.7), storage_p=np.ones(3), gain_p=np.ones(3),
        rate_p=np.full(3, 0.3),
        gates=StructuralGates(np.ones(3), np.full(3, modulation), np.ones(3), s_coi=0.0),
    )


def rk4(f, state, step):
    k1 = f.rhs(state)
    k2 = f.rhs(state + step * k1 / 2)
    k3 = f.rhs(state + step * k2 / 2)
    k4 = f.rhs(state + step * k3)
    result = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Nonfinite probe trajectory")
    return result


def sampled(f, initial, duration, step):
    subdivisions = int(round(0.01 / step))
    assert subdivisions > 0 and np.isclose(subdivisions * step, 0.01)
    state = initial.copy()
    states = [state.copy()]
    for _ in range(int(round(duration / 0.01))):
        for _ in range(subdivisions):
            state = rk4(f, state, step)
        states.append(state.copy())
    return np.asarray(states)


def observations(states, mass):
    theta = states[:, :3]
    relative = theta - (theta @ mass / mass.sum())[:, None]
    return np.concatenate((relative, states[:, 3:6]), axis=1).ravel() / 0.1


def displaced(initial, modulation, direction):
    scaled = np.asarray(direction) * SCALES
    state = initial.copy()
    state[9:12] += scaled[:3]
    state[12:15] += scaled[3:6]
    state[6:9] += scaled[7:]
    return state, modulation + scaled[6]


def history(initial, modulation, step):
    f = functional(modulation)
    states = sampled(f, initial, 0.1, step)
    return observations(states, f.mass), states[-1, 6:9] / 0.1, states[-1]


def jacobians(initial, modulation, step, epsilon):
    base_h, base_p, _ = history(initial, modulation, step)
    columns_h, columns_p = [], []
    for column in range(10):
        shift = np.eye(10)[column] * epsilon
        plus_h, plus_p, _ = history(*displaced(initial, modulation, shift), step)
        if column == 6 and modulation == 0.0:
            twice_h, twice_p, _ = history(*displaced(initial, modulation, 2 * shift), step)
            columns_h.append((-3 * base_h + 4 * plus_h - twice_h) / (2 * epsilon))
            columns_p.append((-3 * base_p + 4 * plus_p - twice_p) / (2 * epsilon))
        else:
            minus_h, minus_p, _ = history(*displaced(initial, modulation, -shift), step)
            columns_h.append((plus_h - minus_h) / (2 * epsilon))
            columns_p.append((plus_p - minus_p) / (2 * epsilon))
    return np.column_stack(columns_h), np.column_stack(columns_p)


def condition_on_endpoint(history_jacobian, endpoint_jacobian):
    auxiliary = endpoint_jacobian[:, 7:]
    spectrum = np.linalg.svd(auxiliary, compute_uv=False)
    if spectrum[-1] <= spectrum[0] * 1e-10:
        raise ValueError("Endpoint p constraint is numerically unresolved")
    compensation = -np.linalg.solve(auxiliary, endpoint_jacobian[:, :7])
    conditional = history_jacobian[:, :7] + history_jacobian[:, 7:] @ compensation
    residual = endpoint_jacobian[:, :7] + auxiliary @ compensation
    return conditional, compensation, spectrum, float(np.max(np.abs(residual)))


def directional_check(initial, modulation, direction, compensation, history_step, future_step):
    displacement = np.concatenate((direction, compensation @ direction)) * 1e-3
    base_h, base_p, endpoint = history(initial, modulation, history_step)
    f = functional(modulation)
    base_future = observations(sampled(f, endpoint, 1.0, future_step), f.mass)
    outcomes = []
    for sign in (-1, 1):
        candidate, candidate_modulation = displaced(initial, modulation, sign * displacement)
        if not 0 <= candidate_modulation <= 1:
            continue
        h, p, end = history(candidate, candidate_modulation, history_step)
        candidate_f = functional(candidate_modulation)
        future = observations(sampled(candidate_f, end, 1.0, future_step), candidate_f.mass)
        outcomes.append({
            "sign": sign, "modulation": float(candidate_modulation),
            "history_scaled_rms": float(np.sqrt(np.mean((h - base_h) ** 2))),
            "endpoint_p_scaled_norm": float(np.linalg.norm(p - base_p)),
            "future_scaled_rms": float(np.sqrt(np.mean((future - base_future) ** 2))),
        })
    return outcomes


def cases():
    zero = np.zeros(15)
    angle = zero.copy(); angle[:3] = [0.1, -0.05, 0]
    excited = angle.copy()
    excited[3:6] = [0.05, -0.02, 0.01]
    excited[9:12] = [0.05, 0, -0.05]
    excited[12:15] = [0.02, -0.01, 0]
    return (zero, angle, excited, -excited)


def run_probe():
    records = []
    for modulation in (0.0, 0.5):
        for index, initial in enumerate(cases()):
            diagnostics = []
            for step in (0.0025, 0.00125):
                for epsilon in (1e-4, 5e-5, 2.5e-5):
                    jh, jp = jacobians(initial, modulation, step, epsilon)
                    conditional, compensation, p_spectrum, residual = condition_on_endpoint(jh, jp)
                    _, singular, vt = np.linalg.svd(conditional, full_matrices=False)
                    zero_columns = [0, 1, 2, 6] if index == 0 else ([0, 1, 2] if modulation == 0 else [])
                    negative_calibration = float(np.max(np.abs(conditional[:, zero_columns]))) if zero_columns else None
                    if negative_calibration is not None and negative_calibration > 1e-9:
                        raise AssertionError("Equilibrium negative calibration failed")
                    subspaces = {}
                    for threshold in THRESHOLDS:
                        basis = vt[singular <= singular[0] * threshold]
                        subspaces[str(threshold)] = (basis.T @ basis).tolist()
                    diagnostics.append({
                        "history_step": step, "difference_step": epsilon,
                        "singular_values": singular.tolist(),
                        "relative_singular_values": (singular / singular[0]).tolist(),
                        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
                        "ranks": {str(t): int(np.sum(singular > singular[0] * t)) for t in THRESHOLDS},
                        "near_null_projectors": subspaces,
                        "endpoint_auxiliary_singular_values": p_spectrum.tolist(),
                        "endpoint_auxiliary_ranks": {str(t): int(np.sum(p_spectrum > p_spectrum[0] * t)) for t in THRESHOLDS},
                        "linear_endpoint_constraint_max_error": residual,
                        "negative_calibration_columns": zero_columns,
                        "negative_calibration_max_error": negative_calibration,
                        "selected_direction": vt[-1].tolist(),
                        "directions": directional_check(initial, modulation, vt[-1], compensation, step, 4 * step),
                    })
            reference = diagnostics[-1]
            for diagnostic in diagnostics:
                diagnostic["spectrum_distance_to_fine"] = float(np.linalg.norm(
                    np.asarray(diagnostic["singular_values"]) - reference["singular_values"]))
                diagnostic["near_null_projector_distances_to_fine"] = {
                    str(t): float(np.linalg.norm(np.asarray(diagnostic["near_null_projectors"][str(t)]) -
                                               reference["near_null_projectors"][str(t)]))
                    for t in THRESHOLDS
                }
            records.append({"case": index, "modulation": modulation, "initial_state": initial.tolist(), "diagnostics": diagnostics})
    return {"scope": "Local conditional sensitivity on eight fixed synthetic states; no recovery, fitting or general identifiability claim.",
            "parameter_scales": SCALES.tolist(), "output_scale": 0.1, "records": records}


def main():
    output = ROOT / "results/gftd_observability_20261107"
    release = json.loads((output / "execution_release.json").read_text(encoding="utf-8-sig"))
    if release["released"] is not True:
        raise PermissionError("Probe has not been released")
    for relative, expected in release["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    started = output / "started.json"
    with started.open("x", encoding="utf-8") as stream:
        json.dump({"action_id": "gftd-observability-20261107-01", "status": "RUNNING"}, stream)
    begin = time.perf_counter()
    receipt = {"action_id": "gftd-observability-20261107-01", "status": "FAILED"}
    try:
        summary = run_probe()
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        receipt["status"] = "COMPLETED"
        receipt["cases"] = len(summary["records"])
        receipt["summary_sha256"] = hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()
    except Exception as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - begin
        receipt["training"] = False
        receipt["real_data_reads"] = False
        receipt["all_source_unchanged"] = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == s for p, s in release["source_sha256"].items())
        (output / "execution_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
