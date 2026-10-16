"""Development-only E1 evaluation and dimension-independent cached modal audit.

Imports are data-free. The CLI explicitly reads the inherited development
cohort; modal targets use the frozen 0078 cache and never recompute FOM poles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import block_diag, null_space

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from compare_physical_modes import (DAMP_GATE, FREQ_GATE, MAC_GATE, OMEGA_S,
                                    assignment, ordinary_modes, physical_output)
from evaluate_structured_rom import _safe, evaluate

CACHE = ROOT / "private_data/task2_control_modes_0078"
TARGET_IDS = [62, 64, 66, 68, 71, 73, 77, 79, 84, 91, 98, 100, 104]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_angle_embedding(dim):
    if dim < 30:
        raise ValueError("expected full delta, omega, p and optional memory state")
    return block_diag(null_space(np.ones((1, 10))), np.eye(dim - 10))


def model_dimension(model):
    # A zero-step rollout is also the public-interface fallback for wrappers.
    for name in ("interface_dim", "dim"):
        if hasattr(model, name):
            return int(getattr(model, name))
    with torch.no_grad():
        path = model.rollout(torch.zeros((1, 11, 30), dtype=torch.float64), 0, .01)
    return int(path.shape[-1])


def jacobian_qa(model):
    dim = model_dimension(model)
    zero = torch.zeros(dim, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        f0 = model.field(zero.detach())
        shifted = zero.detach().clone()
        shifted[:10] += .37
        gauge = model.field(shifted) - f0
    with torch.enable_grad():
        matrix = torch.autograd.functional.jacobian(model.field, zero).detach().cpu().numpy()
    if matrix.shape != (dim, dim):
        raise ValueError("field must preserve full interface state dimension")
    g = np.r_[np.ones(10), np.zeros(dim - 10)]
    qa = {"f0_max_abs": float(f0.abs().max()), "gauge_max_abs": float(gauge.abs().max()),
          "jacobian_gauge_max_abs": float(np.max(np.abs(matrix @ g))),
          "finite": bool(np.isfinite(matrix).all())}
    for h in (1e-5, 1e-6):
        fd = np.empty_like(matrix)
        with torch.no_grad():
            for j in range(dim):
                plus, minus = zero.detach().clone(), zero.detach().clone()
                plus[j] += h
                minus[j] -= h
                fd[:, j] = ((model.field(plus) - model.field(minus)) / (2 * h)).cpu().numpy()
        qa[f"fd_{h}_relative_fro"] = float(np.linalg.norm(fd - matrix) / max(1e-300, np.linalg.norm(matrix)))
        qa[f"fd_{h}_max_abs"] = float(np.max(np.abs(fd - matrix)))
    qa["f0_gate"] = qa["f0_max_abs"] <= 1e-10
    qa["gauge_gate"] = max(qa["gauge_max_abs"], qa["jacobian_gauge_max_abs"]) <= 1e-10
    qa["fd_gate"] = all(qa[f"fd_{h}_relative_fro"] <= 1e-7 for h in (1e-5, 1e-6))
    qa["pass"] = all(qa[k] for k in ("finite", "f0_gate", "gauge_gate", "fd_gate"))
    return matrix, qa


def cached_targets(cache, mass):
    cache = Path(cache)
    evidence = json.loads((cache / "evidence.json").read_text(encoding="utf-8"))
    targets = evidence["targets"]
    if [int(t["index"]) for t in targets] != TARGET_IDS:
        raise AssertionError("frozen thirteen target identities changed")
    if not evidence.get("integrity_pass") or not evidence.get("hashes_unchanged"):
        raise AssertionError("cached FOM evidence does not have valid source integrity")
    if evidence["thresholds"] != {"frequency_relative": FREQ_GATE, "damping_absolute": DAMP_GATE, "mac": MAC_GATE}:
        raise AssertionError("historical matching rules changed")
    with np.load(cache / "modal_arrays.npz", allow_pickle=False) as data:
        indices = np.asarray(data["target_recomputed_indices"], dtype=int)
        values = np.asarray(data["descriptor_eigenvalues"])
        vectors = np.asarray(data["descriptor_right_vectors"])
        if not np.array_equal(data["target_ids"], TARGET_IDS) or indices.shape != (13,) or len(set(indices)) != 13:
            raise AssertionError("cached target mapping malformed")
        if not np.allclose(data["mass"], mass, rtol=1e-10, atol=1e-12):
            raise AssertionError("candidate and FOM physical masses differ")
        if np.any(indices < 0) or np.any(indices >= len(values)) or vectors.shape[1] != len(values):
            raise AssertionError("cached descriptor index outside pole/vector arrays")
        outputs = [physical_output(vectors[:, i], mass, omega_scale=OMEGA_S) for i in indices]
        if not np.allclose(np.column_stack(outputs), data["target_outputs"], rtol=1e-10, atol=1e-12):
            raise AssertionError("cached physical output alignment changed")
    for target, value, output in zip(targets, values[indices], outputs, strict=True):
        old = complex(float(target["real"]), float(target["imag"]))
        if abs(old - value) / max(1., abs(old)) > 1e-6:
            raise AssertionError("target pole and recomputed index do not align")
        if not np.isfinite(output).all() or np.linalg.norm(output) <= 1e-12:
            raise AssertionError("invalid physical target output")
    return targets, outputs


def modal(model, *, cache=CACHE, output=None):
    cache = Path(cache)
    hashes = {name: digest(cache / name) for name in ("modal_arrays.npz", "evidence.json")}
    mass = np.asarray(model.mass.detach().cpu() if torch.is_tensor(model.mass) else model.mass)
    targets, target_outputs = cached_targets(cache, mass)
    matrix, qa = jacobian_qa(model)
    Q = common_angle_embedding(len(matrix))
    reduced = Q.T @ matrix @ Q
    arrays = {"jacobian": matrix, "embedding": Q, "reduced_jacobian": reduced,
              "target_outputs": np.column_stack(target_outputs), "target_ids": np.asarray(TARGET_IDS)}
    result = {"test_read": False, "fom_recomputed": False, "full_dimension": len(matrix),
              "reduced_dimension": Q.shape[1], "qa": qa, "targets": targets,
              "thresholds": {"frequency_relative": FREQ_GATE, "damping_absolute": DAMP_GATE, "mac": MAC_GATE},
              "scope": "development L3 screen; spectral alpha is local only; no L2 region or certificate",
              "source_hashes_before": hashes, "matched_count": None, "spectral_alpha": None}
    if qa["pass"]:
        values, vectors, residual = ordinary_modes(reduced)
        pairs = assignment(targets, target_outputs, values, vectors, mass, Q)
        result.update(pairs)
        result.update(matched_count=len(pairs["matched"]), spectral_alpha=float(np.max(values.real)),
                      max_eigen_residual=float(residual),
                      poles=[{"index": i, "real": float(v.real), "imag": float(v.imag)} for i, v in enumerate(values)])
        arrays.update(eigenvalues=values, right_vectors=vectors,
                      output_vectors=np.column_stack([physical_output(vectors[:, i], mass, Q) for i in range(len(values))]))
    result["source_hashes_after"] = {name: digest(cache / name) for name in hashes}
    if hashes != result["source_hashes_after"]:
        raise RuntimeError("modal source changed during audit")
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "modal_arrays.npz", **arrays)
        (output / "evidence.json").write_text(json.dumps(_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    return result


def energy_power_qa(model):
    """Deterministic construction QA in full internal coordinates, not L2."""
    required = ("interface_to_state", "energy_and_gradient", "energy", "matrices",
                "port_matrix", "port_values", "internal_field")
    if not all(hasattr(model, name) for name in required):
        return {"status": "not_evaluated", "reason": "explicit model power interface unavailable"}
    dim = model_dimension(model)
    grid = torch.arange(1, 1 + 8 * dim, dtype=torch.float64).reshape(8, dim)
    interface = .03 * torch.sin(grid)
    with torch.enable_grad():
        internal = model.interface_to_state(interface).detach().requires_grad_(True)
        H, e = model.energy_and_gradient(internal)
        exact_e = torch.autograd.grad(model.energy(internal).sum(), internal)[0]
        J, R = model.matrices()
        B, u = model.port_matrix(), model.port_values(internal)
        actual = model.internal_field(internal)
        hdot = (exact_e * actual).sum(-1)
        loss = torch.einsum("bi,ij,bj->b", e, R, e)
        supply = ((e @ B) * u).sum(-1)
        defect = hdot + loss - supply
    array = lambda x: x.detach().cpu().numpy()
    values = {"energy": array(H), "hdot": array(hdot), "dissipation": array(loss),
              "actual_port_supply": array(supply), "power_defect": array(defect)}
    finite = all(np.isfinite(value).all() for value in values.values())
    skew = float(np.max(np.abs(array(J + J.T))))
    symmetry = float(np.max(np.abs(array(R - R.T))))
    minimum = float(np.linalg.eigvalsh(array((R + R.T) / 2)).min())
    gradient_error = float(np.max(np.abs(array(e - exact_e))))
    power_error = float(np.max(np.abs(array(defect)) / np.maximum(1., np.abs(array(hdot)) + np.abs(array(loss)) + np.abs(array(supply)))))
    angle_port = float(np.max(np.abs(array(B[:10])))) if B.numel() else 0.
    ok = finite and max(skew, symmetry, gradient_error, power_error, angle_port) <= 1e-10 and minimum >= -1e-10
    return {"status": "pass" if ok else "failed", "pass": bool(ok), "finite": finite,
            "probe_count": 8, "J_skew_max_abs": skew, "R_symmetry_max_abs": symmetry,
            "R_min_eigenvalue": minimum, "gradient_max_abs_error": gradient_error,
            "power_scaled_max_abs_error": power_error, "angle_port_max_abs": angle_port,
            "probes": _safe(values), "scope": "construction QA including actual ports; no L2 certificate"}


def run(checkpoint, output):
    from train_local_memory_0098 import load_model
    from train_joint_latent import BASE, NETWORK, ROOTS, read_development
    from evaluate_l1_latent import _read_validation_starts
    from kiki.models.private_pm import MechanicalROM

    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    started = time.perf_counter()
    paths = {"model": checkpoint / "model.pt", "checkpoint": checkpoint / "checkpoint.json",
             "base": BASE, "network": NETWORK, "evaluation_code": Path(__file__),
             "inherited_evaluation_code": ROOT / "experiments/evaluate_structured_rom.py",
             "l1_code": ROOT / "experiments/evaluate_l1_latent.py",
             "modal_code": ROOT / "experiments/compare_physical_modes.py",
             "training_code": ROOT / "experiments/train_local_memory_0098.py",
             "model_code": ROOT / "experiments/local_memory_rom_0098.py",
             "core_code": ROOT / "experiments/structured_rom.py",
             "grammar_code": ROOT / "experiments/structure_grammar.py",
             "data_reader_code": ROOT / "experiments/train_joint_latent.py"}
    for filename in ("normalization.npz",):
        if (checkpoint / filename).exists():
            paths[filename] = checkpoint / filename
    for path in ROOTS:
        for name in ("dataset_index.csv", "splits.json"):
            paths[f"{path.name}_{name}"] = path / name
    before = {key: digest(path) for key, path in paths.items()}
    model = load_model(checkpoint)
    model.eval()
    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False))
    network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]])
    network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    development, excluded, identities = read_development(base, network)
    starts, latency = _read_validation_starts(base.pm_star)
    if len(starts) != 107 or len(latency) != 10 or sum(e["label"] == "unstable" for e in development["validation"]) != 12:
        raise AssertionError("historical validation denominators changed")
    output.mkdir(parents=True)
    metadata = {"loop": "0098", "checkpoint": str(checkpoint), "source_hashes_before": before,
                "identities": identities, "excluded": excluded,
                "latency_rows": [{k: v for k, v in row.items() if k not in ("inertia", "equilibrium_delta")} for row in latency]}
    trajectories = evaluate(model, development["validation"], starts, output=output / "trajectories", metadata=metadata)
    modes = modal(model, output=output / "modal")
    power = energy_power_qa(model)
    after = {key: digest(path) for key, path in paths.items()}
    if before != after:
        raise RuntimeError("evaluation source changed during execution")
    result = {"loop": "0098", "test_read": False, "checkpoint": str(checkpoint),
              "source_hashes_before": before, "source_hashes_after": after,
              "seconds": time.perf_counter() - started,
              "trajectory_evidence": "trajectories/evidence.json", "modal_evidence": "modal/evidence.json",
              "numerical_gate_pass": trajectories["model_numerical_gate_pass"],
              "modal_qa_pass": modes["qa"]["pass"], "modal_match_count": modes["matched_count"],
              "energy_power_qa": power,
              "L2": "not_established"}
    (output / "evidence.json").write_text(json.dumps(_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(_safe(run(args.checkpoint, args.output)), allow_nan=False))


if __name__ == "__main__":
    main()
