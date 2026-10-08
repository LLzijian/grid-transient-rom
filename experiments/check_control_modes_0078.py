"""Read-only physical-output modal audit for retained C-free3000."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch
from scipy.linalg import block_diag, null_space

from compare_physical_modes import (  # noqa: E402
    DEFAULT_COUPLED, DEFAULT_FULL, DEFAULT_M3, DEFAULT_M3_LINEARIZATION,
    DEFAULT_VERIFIED, FREQ_GATE, DAMP_GATE, MAC_GATE, OMEGA_S,
    assignment, descriptor_modes, ordinary_modes, physical_output, pole_match,
    relmax, sha,
)

CHECKPOINT = Path("private_data/task2_structure_0078_extend_C-free/step3000")
NETWORK = Path("private_data/task2_electrical_ports_0048/physical_network.npz")
MATRICES = Path("private_data/task2_long_sparse_audit_0061/matrices.npz")


def finite_difference(field, x: np.ndarray, h: float) -> np.ndarray:
    x = np.asarray(x, float); out = np.empty((x.size, x.size), float)
    for j in range(x.size):
        xp, xm = x.copy(), x.copy(); xp[j] += h; xm[j] -= h
        out[:, j] = (field(xp) - field(xm)) / (2 * h)
    return out


def jacobian_qa(model) -> tuple[np.ndarray, dict[str, float]]:
    zero = torch.zeros(33, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        f0 = model.field(zero.detach()); shifted = zero.detach().clone(); shifted[:10] += .37
        gauge = model.field(shifted) - f0
    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(lambda x: model.field(x), zero, create_graph=False).detach().numpy()
    x = np.zeros(33)
    def field(v):
        with torch.no_grad():
            return model.field(torch.as_tensor(v, dtype=torch.float64)).numpy()
    fd = {str(h): finite_difference(field, x, h) for h in (1e-5, 1e-6)}
    g = np.r_[np.ones(10), np.zeros(23)]
    qa = {"f0_max_abs": float(f0.abs().max()), "gauge_max_abs": float(gauge.abs().max()),
          "jacobian_gauge_max_abs": float(np.max(np.abs(J @ g)))}
    for h, value in fd.items():
        qa[f"fd_{h}_relative_fro"] = float(np.linalg.norm(J - value) / max(1e-300, np.linalg.norm(J)))
        qa[f"fd_{h}_max_abs"] = float(np.max(np.abs(J - value)))
    return J, qa


def _source_paths(full, verified, coupled, m3, checkpoint):
    return {
        "model": checkpoint / "model.pt", "checkpoint": checkpoint / "checkpoint.json", "normalization": checkpoint / "normalization.npz",
        "m3": m3, "network": NETWORK, "matrices": MATRICES,
        "descriptor": full / "descriptor.npz", "descriptor_evidence": full / "evidence.json", "verified": verified,
        "coupled": coupled, "coupled_evidence": coupled.parent / "evidence.json",
        "m3_linearization": DEFAULT_M3_LINEARIZATION,
        "code": Path(__file__), "compare_code": Path(__file__).with_name("compare_physical_modes.py"),
        "trainer": Path(__file__).with_name("train_structured_rom.py"),
        "structured_core": Path(__file__).with_name("structured_rom.py"),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    full, verified, coupled, m3 = map(Path, (args.fullorder, args.verified, args.coupled, args.m3)); checkpoint = Path(args.checkpoint)
    paths = _source_paths(full, verified, coupled, m3, checkpoint)
    if any(not p.exists() for p in paths.values()): raise FileNotFoundError([str(p) for p in paths.values() if not p.exists()])
    before = {k: sha(v) for k, v in paths.items()}
    meta = json.loads((checkpoint / "checkpoint.json").read_text())
    if meta.get("arm") != "C-free" or meta.get("step") != 3000:
        raise AssertionError("expected the retained C-free3000 checkpoint")
    from train_structured_rom import load_model
    model = load_model(checkpoint); model.eval()
    J33, qa = jacobian_qa(model)
    qa["f0_gate"] = qa["f0_max_abs"] <= 1e-10; qa["gauge_gate"] = max(qa["gauge_max_abs"], qa["jacobian_gauge_max_abs"]) <= 1e-10
    qa["fd_gate"] = all(qa[f"fd_{h}_relative_fro"] <= 1e-7 for h in ("1e-05", "1e-06"))
    if not np.isfinite(J33).all() or not all(qa[k] for k in ("f0_gate", "gauge_gate", "fd_gate")):
        raise AssertionError(f"candidate Jacobian QA failed: {qa}")
    N = null_space(np.ones((1, 10))); Q = block_diag(N, np.eye(23)); J32 = Q.T @ J33 @ Q
    fvalues, fvectors, reliable, fd = descriptor_modes(full)
    with np.load(coupled, allow_pickle=False) as z, np.load(m3, allow_pickle=False) as m, np.load(DEFAULT_M3_LINEARIZATION, allow_pickle=False) as ml:
        j29, mass = z["M3_verified"], z["param_mass"]; m3_mass = m["mass"]
        if relmax(j29, ml["matrix"]) > 1e-10: raise AssertionError("cached M3 verified matrix mismatch")
    if relmax(mass, m3_mass) > 1e-10 or relmax(model.mass.detach().numpy(), m3_mass) > 1e-10:
        raise AssertionError("candidate physical mass mapping mismatch")
    if relmax(fd["tf"][:10], np.ones(10)) > 1e-10 or relmax(fd["tf"][10:20], m3_mass * OMEGA_S) > 1e-10 or relmax(fd["A"][:10, 10:20], OMEGA_S * np.eye(10)) > 1e-10: raise AssertionError("descriptor unit/inertia mapping mismatch")
    ce = json.loads((coupled.parent / "evidence.json").read_text()); ve = json.loads(verified.read_text()); fe = json.loads((full / "evidence.json").read_text())
    expected_sources = {"base": m3, "network": NETWORK, "matrices": MATRICES, "verified_evidence": verified}
    source_integrity = all(ce.get("source_sha256", {}).get(k) == sha(v) for k, v in expected_sources.items())
    identity_ok = all(fe[k] == ve["fullorder_identity"][k] for k in ("case_sha256", "physical_signature", "equilibrium_state_error", "equilibrium_algebraic_error"))
    if not source_integrity or not identity_ok:
        raise AssertionError("verified source or FOM equilibrium identity mismatch")
    targets = ve["fullorder_electromechanical_modes"]; target_ids = [int(t["index"]) for t in targets]
    expected_ids = [62, 64, 66, 68, 71, 73, 77, 79, 84, 91, 98, 100, 104]
    if target_ids != expected_ids: raise AssertionError("verified targets changed")
    pole_map, pole_error = pole_match(fvalues, ve["fullorder_all_modes"])
    if any(float(t["imag"]) <= 1e-6 or not t["participation_identifiable"] or float(t["electromechanical_participation"]) < .1 or not reliable[pole_map[int(t["index"])]] for t in targets):
        raise AssertionError("fixed target eligibility/eigenvector reliability mismatch")
    target_vectors = [physical_output(fvectors[:, pole_map[i]], mass, omega_scale=OMEGA_S) for i in target_ids]
    if not all(np.isfinite(v).all() and np.linalg.norm(v) > 1e-12 for v in target_vectors):
        raise AssertionError("invalid FOM physical-output target")
    sv, sr, s_residual = ordinary_modes(J32)
    s = assignment(targets, target_vectors, sv, sr, mass, Q)
    before_after = {"finite_modes": int(len(fvalues)), "infinite_modes": int(fd["infinite_modes"]), "indeterminate_pairs": int(fd["indeterminate_pairs"]), "max_descriptor_residual": float(fd["max_residual"]), "pole_match_max_relative": pole_error, "source_integrity": source_integrity, "equilibrium_identity": identity_ok}
    after = {k: sha(v) for k, v in paths.items()}; hashes_unchanged = before == after
    if not hashes_unchanged: raise AssertionError("modal source changed during execution")
    result = {"test_read": False, "qa": qa, "descriptor": before_after, "mapping": {"omega_s": OMEGA_S, "tf_delta_rel": relmax(fd["tf"][:10], np.ones(10)), "tf_omega_rel": relmax(fd["tf"][10:20], m3_mass * OMEGA_S), "delta_dot_omega_rel": relmax(fd["A"][:10, 10:20], OMEGA_S * np.eye(10))}, "C": {"matched_count": len(s["matched"]), "all_pair_metrics": s["all_pair_metrics"], "matched": s["matched"], "unmatched_fom": s["unmatched_fom"], "unmatched_rom": s["unmatched_rom"], "excluded_null_output_rom": s["excluded_null_output_rom"], "oscillatory_rom_count": s["oscillatory_rom_count"], "max_eigen_residual": s_residual}, "thresholds": {"frequency_relative": FREQ_GATE, "damping_absolute": DAMP_GATE, "mac": MAC_GATE}, "hashes_before": before, "hashes_after": after, "hashes_unchanged": hashes_unchanged}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    result["checkpoint"] = str(checkpoint)
    result["targets"] = targets
    result["integrity_pass"] = True
    np.savez_compressed(out / "modal_arrays.npz", J33=J33, Q=Q, J32=J32, descriptor_eigenvalues=fvalues, descriptor_right_vectors=fvectors, C_eigenvalues=sv, C_right_vectors=sr, target_recomputed_indices=np.asarray([pole_map[i] for i in target_ids], int), C_output_vectors=np.column_stack([physical_output(sr[:, i], mass, Q) for i in range(sr.shape[1])]), target_outputs=np.column_stack(target_vectors), target_ids=np.asarray(target_ids), mass=mass)
    (out / "evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, default=CHECKPOINT); p.add_argument("--fullorder", type=Path, default=DEFAULT_FULL); p.add_argument("--verified", type=Path, default=DEFAULT_VERIFIED); p.add_argument("--coupled", type=Path, default=DEFAULT_COUPLED); p.add_argument("--m3", type=Path, default=DEFAULT_M3); p.add_argument("--output", type=Path, required=True); a = p.parse_args(); run(a)


if __name__ == "__main__": main()
