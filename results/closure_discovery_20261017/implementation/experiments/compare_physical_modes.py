"""Loop0071 physical-output modal comparison (descriptive, no simulation)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Keep the small generalized-eigenvalue audit single-threaded.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

OMEGA_S = 2 * np.pi * 60.0
S_DELTA = 0.47675314588683515
S_OMEGA = 1.0033667249602924
FREQ_GATE, DAMP_GATE, MAC_GATE = 0.15, 0.05, 0.8
DEFAULT_FULL = Path("private_data/task2_fullorder_descriptor_0035")
DEFAULT_VERIFIED = Path("private_data/task2_verified_modes_0035b/evidence.json")
DEFAULT_COUPLED = Path("private_data/task2_coupled_equilibrium_0065/coupled_equilibrium.npz")
DEFAULT_COUPLED_EVIDENCE = Path("private_data/task2_coupled_equilibrium_0065/evidence.json")
DEFAULT_M3 = Path("private_data/task2_m3_0020/model.npz")
DEFAULT_M3_LINEARIZATION = Path("private_data/task2_verified_modes_0035b/M3_linearization.npz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relmax(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


def physical_output(vector: np.ndarray, mass: np.ndarray, embed: np.ndarray | None = None,
                    omega_scale: float = 1.0) -> np.ndarray:
    """Return [COI delta/s_delta, omega_rad_s/s_omega]."""
    full = np.asarray(vector) if embed is None else np.asarray(embed) @ np.asarray(vector)
    delta, omega = full[:10], full[10:20] * omega_scale
    coi = delta - np.sum(mass * delta) / np.sum(mass)
    return np.r_[coi / S_DELTA, omega / S_OMEGA]


def mac(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.vdot(a, a).real * np.vdot(b, b).real)
    if den <= 0:
        return 0.0
    return float(np.clip(abs(np.vdot(a, b)) ** 2 / den, 0.0, 1.0))


def ordinary_modes(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    matrix = np.asarray(matrix, float)
    if not np.isfinite(matrix).all():
        raise AssertionError("nonfinite ROM matrix")
    values, vectors = np.linalg.eig(matrix)
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise AssertionError("nonfinite ROM poles/eigenvectors")
    residual = max(float(np.linalg.norm(matrix @ vectors[:, i] - values[i] * vectors[:, i]) /
                        max(1.0, np.linalg.norm(matrix) * np.linalg.norm(vectors[:, i])))
                   for i in range(len(values)))
    if residual > 1e-8:
        raise AssertionError("ROM eigen residual exceeds preregistered bound")
    return values, vectors, residual


def descriptor_modes(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(directory / "descriptor.npz", allow_pickle=False) as data:
        fx, fy, gx, gy = (np.asarray(data[k], float) for k in ("fx", "fy", "gx", "gy"))
        tf = np.asarray(data["time_constants"], float)
        state_names = [str(x) for x in data["state_names"]]
        algebraic_names = [str(x) for x in data["algebraic_names"]]
    a = np.block([[fx, fy], [gx, gy]])
    e_diag = np.r_[tf, np.zeros(gy.shape[0])]
    if not np.isfinite(a).all() or not np.isfinite(e_diag).all():
        raise AssertionError("nonfinite descriptor matrices")
    e = np.diag(e_diag)
    homogeneous, left, right = eig(a, e, left=True, homogeneous_eigvals=True)
    finite = np.abs(homogeneous[1]) > 1e-10 * np.sum(np.abs(homogeneous), axis=0)
    indeterminate = int(np.sum(np.sum(np.abs(homogeneous), axis=0) < 1e-10))
    values = homogeneous[0, finite] / homogeneous[1, finite]
    left, right = left[:, finite], right[:, finite]
    if not np.isfinite(values).all() or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError("nonfinite descriptor poles/eigenvectors")
    residuals = np.linalg.norm(a @ right - (e @ right) * values, axis=0) / (
        (np.linalg.norm(a) + np.abs(values) * np.linalg.norm(e)) * np.linalg.norm(right, axis=0)
    )
    overlap = np.sum((left.conj() * e_diag[:, None]) * right, axis=0)
    reliable = np.abs(overlap) >= 1e-12
    rows = [{"index": i, "real": float(v.real), "imag": float(v.imag),
             "frequency_hz": float(abs(v.imag) / (2 * np.pi)),
             "damping_ratio": None if abs(v) < 1e-10 else float(-v.real / abs(v)),
             "reliable": bool(reliable[i]), "residual": float(residuals[i])}
            for i, v in enumerate(values)]
    expected = [f"delta GENROU {i}" for i in range(1, 11)] + [f"omega GENROU {i}" for i in range(1, 11)]
    if state_names[:20] != expected:
        raise AssertionError("GENROU delta/omega order mismatch")
    if len(values) != 150 or int(np.sum(~finite)) != 549 or indeterminate:
        raise AssertionError("original descriptor finite/infinite count mismatch")
    if float(np.max(residuals)) > 1e-8:
        raise AssertionError("original descriptor eigen residual exceeds preregistered bound")
    return values, right, reliable, {"rows": rows, "e_diag": e_diag, "tf": tf, "state_names": state_names,
                                     "algebraic_names": algebraic_names, "max_residual": float(np.max(residuals)),
                                     "infinite_modes": int(np.sum(~finite)), "indeterminate_pairs": indeterminate,
                                     "A": a}


def pole_match(recomputed: np.ndarray, stored: list[dict[str, object]]) -> tuple[dict[int, int], float]:
    reference = np.array([complex(float(r["real"]), float(r["imag"])) for r in stored])
    distance = np.abs(reference[:, None] - recomputed[None, :]) / np.maximum(1.0, np.abs(reference[:, None]))
    ri, ci = linear_sum_assignment(distance)
    maximum = float(np.max(distance[ri, ci]))
    if len(reference) != 150 or len(recomputed) != 150 or maximum > 1e-6:
        raise AssertionError("verified 150-pole matching failed")
    return {int(r): int(c) for r, c in zip(ri, ci, strict=True)}, maximum


def assignment(targets: list[dict[str, object]], target_vectors: list[np.ndarray], values: np.ndarray,
               vectors: np.ndarray, mass: np.ndarray, embed: np.ndarray | None) -> dict[str, object]:
    outputs = [physical_output(vectors[:, i], mass, embed) for i in range(len(values))]
    if any(not np.isfinite(v).all() for v in outputs):
        raise AssertionError("nonfinite ROM physical output vector")
    candidates = [i for i, v in enumerate(values) if v.imag > 1e-6 and np.linalg.norm(outputs[i]) > 1e-12]
    metrics: list[dict[str, object]] = []
    cost = np.full((len(targets), len(candidates) + len(targets)), 1e6, float)
    for ti, (target, fv) in enumerate(zip(targets, target_vectors, strict=True)):
        f = float(target["frequency_hz"]); z = float(target["damping_ratio"])
        for cj, ri in enumerate(candidates):
            rv, rz = values[ri], float(-values[ri].real / abs(values[ri]))
            rf = float(abs(values[ri].imag) / (2 * np.pi)); m = mac(fv, outputs[ri])
            df_raw, dz, dm = abs(rf - f) / f, abs(rz - z), (1 - m) / .2
            df, dz_norm = df_raw / FREQ_GATE, dz / DAMP_GATE
            ok = df_raw <= FREQ_GATE and dz <= DAMP_GATE and m >= MAC_GATE
            cost[ti, cj] = (df * df + dz_norm * dz_norm + dm * dm) / 1000 if ok else 1e6
            metrics.append({"fom_index": int(target["index"]), "rom_index": int(ri), "frequency_hz": rf,
                            "damping_ratio": rz, "frequency_relative_difference": df_raw,
                            "damping_absolute_difference": dz, "d_f": df, "d_z": dz_norm, "d_m": dm,
                            "mac": m, "eligible": bool(ok),
                            "cost": float(cost[ti, cj])})
    cost[:, len(candidates):] = 1.0
    ri, ci = linear_sum_assignment(cost)
    matched, used = [], set()
    for r, c in zip(ri, ci, strict=True):
        if c < len(candidates) and cost[r, c] < 1.0:
            matched.append(metrics[r * len(candidates) + c]); used.add(candidates[c])
    target_ids = {int(x["index"]) for x in targets}
    matched_ids = {int(x["fom_index"]) for x in matched}
    return {"all_pair_metrics": metrics, "matched": matched,
            "unmatched_fom": sorted(target_ids - matched_ids),
            "unmatched_rom": [int(i) for i in candidates if i not in used],
            "excluded_null_output_rom": [int(i) for i, v in enumerate(values) if v.imag > 1e-6 and np.linalg.norm(outputs[i]) <= 1e-12],
            "oscillatory_rom_count": len(candidates)}


def run(args: argparse.Namespace) -> dict[str, object]:
    full, verified, coupled, m3_path = map(Path, (args.fullorder, args.verified, args.coupled, args.m3))
    s_dir = Path(args.s_model_dir)
    network = Path(args.network); matrices = Path(args.matrices); m3_linearization = Path(args.m3_linearization)
    paths = {"descriptor": full / "descriptor.npz", "descriptor_evidence": full / "evidence.json",
             "verified_evidence": verified, "coupled_equilibrium": coupled, "coupled_evidence": coupled.parent / "evidence.json",
             "m3_model": m3_path, "m3_linearization": m3_linearization,
             "s_model": s_dir / "model.pt", "s_normalization": s_dir / "normalization.npz", "s_evidence": s_dir / "evidence.json",
             "network": network, "matrices": matrices}
    before = {name: sha(path) for name, path in paths.items()}
    fvalues, fvectors, reliable, fd = descriptor_modes(full)
    with np.load(coupled, allow_pickle=False) as z, np.load(m3_path, allow_pickle=False) as m, np.load(m3_linearization, allow_pickle=False) as ml:
        j32, embed_s, j29, mass = z["J32"], z["E"], z["M3_verified"], z["param_mass"]
        m3_mass = m["mass"]
        if relmax(j29, ml["matrix"]) > 1e-10:
            raise AssertionError("0065 cached M3_verified differs from verified M3 matrix")
    if j32.shape != (32, 32) or embed_s.shape != (33, 32) or j29.shape != (29, 29):
        raise AssertionError("cached S/M3 matrix or gauge shape mismatch")
    if relmax(fd["tf"][:10], np.ones(10)) > 1e-10 or relmax(fd["tf"][10:20], m3_mass * OMEGA_S) > 1e-10 or relmax(fd["A"][:10, 10:20], OMEGA_S * np.eye(10)) > 1e-10:
        raise AssertionError("physical omega conversion/inertia mapping mismatch")
    if relmax(mass, m3_mass) > 1e-10:
        raise AssertionError("S/M3 mass mapping mismatch")
    coupled_evidence = json.loads((coupled.parent / "evidence.json").read_text())
    recorded = coupled_evidence["source_sha256"]
    expected_sources = {"model": paths["s_model"], "normalization": paths["s_normalization"],
                        "base": m3_path, "network": network, "matrices": matrices, "verified_evidence": verified}
    if any(recorded[k] != sha(v) for k, v in expected_sources.items()):
        raise AssertionError("0065 recorded source provenance mismatch")
    evidence = json.loads(verified.read_text())
    full_evidence = json.loads((full / "evidence.json").read_text())
    for key in ("case_sha256", "physical_signature", "equilibrium_state_error", "equilibrium_algebraic_error"):
        if full_evidence[key] != evidence["fullorder_identity"][key]:
            raise AssertionError("full-order provenance identity mismatch")
    stored_all = evidence["fullorder_all_modes"]
    pole_map, pole_error = pole_match(fvalues, stored_all)
    targets = evidence["fullorder_electromechanical_modes"]
    if len(targets) != 13 or [int(t["index"]) for t in targets] != [62, 64, 66, 68, 71, 73, 77, 79, 84, 91, 98, 100, 104]:
        raise AssertionError("verified target selection changed")
    if any(float(t["imag"]) <= 1e-6 or not t["participation_identifiable"] or float(t["electromechanical_participation"]) < .1 for t in targets):
        raise AssertionError("verified target eligibility changed")
    if any(not reliable[pole_map[int(t["index"])] ] for t in targets):
        raise AssertionError("selected target has unreliable generalized eigenvectors")
    target_vectors = [physical_output(fvectors[:, pole_map[int(t["index"])]], mass, omega_scale=OMEGA_S) for t in targets]
    if any(not np.isfinite(v).all() for v in target_vectors):
        raise AssertionError("nonfinite FOM physical output vector")
    sv, sr, s_residual = ordinary_modes(j32); mv, mr, m3_residual = ordinary_modes(j29)
    s = assignment(targets, target_vectors, sv, sr, mass, embed_s)
    m3_embed = np.zeros((30, 29)); m3_embed[:9, :9] = np.eye(9); m3_embed[10:20, 9:19] = np.eye(10); m3_embed[20:, 19:] = np.eye(10)
    m3 = assignment(targets, target_vectors, mv, mr, m3_mass, m3_embed)
    after = {name: sha(path) for name, path in paths.items()}
    if before != after:
        raise AssertionError("source changed during modal audit")
    result = {"test_read": False, "source_sha256": {**before, "code": sha(Path(__file__))},
              "descriptor_checks": {"finite_modes": 150, "infinite_modes": fd["infinite_modes"],
                                    "indeterminate_pairs": fd["indeterminate_pairs"], "max_eigen_residual": fd["max_residual"],
                                    "verified_pole_match_max_relative": pole_error},
              "mapping_checks": {"omega_s": OMEGA_S, "tf_delta": relmax(fd["tf"][:10], np.ones(10)),
                                  "tf_omega_vs_m3_mass_omega_s": relmax(fd["tf"][10:20], m3_mass * OMEGA_S),
                                  "fx_delta_omega_vs_omega_s": relmax(fd["A"][:10, 10:20], OMEGA_S * np.eye(10))},
              "targets": targets, "S": {**s, "max_eigen_residual": s_residual}, "M3": {**m3, "max_eigen_residual": m3_residual},
              "thresholds": {"frequency_relative": FREQ_GATE, "damping_absolute": DAMP_GATE, "mac": MAC_GATE}}
    def pair_array(rows: list[dict[str, object]]) -> np.ndarray:
        fields = ("fom_index", "rom_index", "frequency_hz", "damping_ratio", "frequency_relative_difference",
                  "damping_absolute_difference", "d_f", "d_z", "d_m", "mac", "eligible", "cost")
        return np.asarray([[float(r[k]) for k in fields] for r in rows], dtype=float).reshape((-1, len(fields)))
    fom_outputs = np.column_stack([physical_output(fvectors[:, i], mass, omega_scale=OMEGA_S) for i in range(fvectors.shape[1])])
    s_outputs = np.column_stack([physical_output(sr[:, i], mass, embed_s) for i in range(sr.shape[1])])
    m3_outputs = np.column_stack([physical_output(mr[:, i], m3_mass, m3_embed) for i in range(mr.shape[1])])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "modal_arrays.npz", fom_eigenvalues=fvalues, fom_output_vectors=fom_outputs,
                        S_eigenvalues=sv, S_output_vectors=s_outputs, M3_eigenvalues=mv, M3_output_vectors=m3_outputs,
                        S_pair_metrics=pair_array(s["all_pair_metrics"]), M3_pair_metrics=pair_array(m3["all_pair_metrics"]),
                        target_recomputed_indices=np.asarray([pole_map[int(t["index"])] for t in targets], dtype=int))
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fullorder", type=Path, default=DEFAULT_FULL)
    p.add_argument("--verified", type=Path, default=DEFAULT_VERIFIED)
    p.add_argument("--coupled", type=Path, default=DEFAULT_COUPLED)
    p.add_argument("--m3", type=Path, default=DEFAULT_M3)
    p.add_argument("--s-model-dir", type=Path, default=Path("private_data/task2_long_0060_5s"))
    p.add_argument("--network", type=Path, default=Path("private_data/task2_electrical_ports_0048/physical_network.npz"))
    p.add_argument("--matrices", type=Path, default=Path("private_data/task2_long_sparse_audit_0061/matrices.npz"))
    p.add_argument("--m3-linearization", type=Path, default=DEFAULT_M3_LINEARIZATION)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
