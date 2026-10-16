"""Local quotient-space energy and linear-field checks for symbolic ROM exports.

This module reads exported model parameters only.  It does not load trajectories,
dataset splits, checkpoints, or event records.  The quotient removes the common
angle direction from the first ten internal coordinates and leaves all other
internal coordinates unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DIM = 40
EPS64 = np.finfo(np.float64).eps
WITNESS_SCOPE = "linearized quadratic witness only; nonlinear radius not certified"
FIXED_POINT_TOL = 1.0e-10
GAUGE_INVARIANCE_TOL = 1.0e-10


def quotient_basis() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic Q (40x39) and the normalized common-angle vector."""
    gauge = np.zeros(DIM, dtype=np.float64)
    gauge[:10] = 1.0 / np.sqrt(10.0)
    e0 = np.zeros(10, dtype=np.float64)
    e0[0] = 1.0
    v = e0 - np.ones(10, dtype=np.float64) / np.sqrt(10.0)
    householder = np.eye(10, dtype=np.float64) - 2.0 * np.outer(v, v) / np.dot(v, v)
    U = householder[:, 1:]
    Q = np.zeros((DIM, DIM - 1), dtype=np.float64)
    Q[:10, :9] = U
    Q[10:, 9:] = np.eye(30, dtype=np.float64)
    return Q, gauge


def _matrix_tolerance(matrix: np.ndarray, multiplier: float = 100.0) -> float | None:
    if not np.isfinite(matrix).all():
        return None
    norm = float(np.linalg.norm(matrix, ord=2))
    return multiplier * EPS64 * max(1.0, norm)


def _complex_pairs(values: np.ndarray | None) -> list[list[float | None]] | None:
    if values is None:
        return None
    result = []
    for value in np.asarray(values).reshape(-1):
        z = complex(value)
        result.append([
            float(z.real) if np.isfinite(z.real) else None,
            float(z.imag) if np.isfinite(z.imag) else None,
        ])
    return result


def _finite_float(value: float | np.floating[Any] | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _extract_local_derivatives(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if getattr(model, "dim", None) != DIM:
        raise ValueError(f"expected a 40-dimensional internal model, got {getattr(model, 'dim', None)!r}")
    try:
        parameter = next(model.parameters())
        device = parameter.device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    origin = torch.zeros(DIM, dtype=torch.float64, device=device, requires_grad=True)
    with torch.enable_grad():
        energy = model.energy(origin)
        if energy.numel() != 1:
            raise ValueError("energy at a single internal state must be scalar")
        energy_gradient = torch.autograd.grad(energy, origin, create_graph=False, retain_graph=True)[0]
        hessian = torch.autograd.functional.hessian(
            lambda x: model.energy(x), origin, create_graph=False, strict=False, vectorize=False
        )
        field_value = model.internal_field(origin)
        jacobian = torch.autograd.functional.jacobian(
            lambda x: model.internal_field(x), origin, create_graph=False, strict=False, vectorize=False
        )
    H = hessian.detach().to(dtype=torch.float64, device="cpu").numpy().copy()
    A = jacobian.detach().to(dtype=torch.float64, device="cpu").numpy().copy()
    grad_H0 = energy_gradient.detach().to(dtype=torch.float64, device="cpu").numpy().copy()
    f0 = field_value.detach().to(dtype=torch.float64, device="cpu").numpy().copy()
    H0 = float(energy.detach().to(dtype=torch.float64, device="cpu").item())
    if H.shape != (DIM, DIM) or A.shape != (DIM, DIM):
        raise ValueError(f"energy Hessian and field Jacobian must both be 40x40; got {H.shape} and {A.shape}")
    if grad_H0.shape != (DIM,) or f0.shape != (DIM,):
        raise ValueError(f"origin energy gradient and field must both have width 40; got {grad_H0.shape} and {f0.shape}")
    return H, A, grad_H0, f0, H0


def analyze_model(model: Any, *, source_sha256: str | None = None,
                  source_path: str | None = None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute local energy/field derivatives and their common-angle quotient."""
    H, A, grad_H0, f0, H0 = _extract_local_derivatives(model)
    Q, gauge = quotient_basis()
    projected_H = Q.T @ H @ Q
    projected_A = Q.T @ A @ Q

    h_finite = bool(np.isfinite(H).all())
    a_finite = bool(np.isfinite(A).all())
    kh_finite = bool(np.isfinite(projected_H).all())
    ka_finite = bool(np.isfinite(projected_A).all())
    h_tol = _matrix_tolerance(H)
    a_tol = _matrix_tolerance(A)
    kh_tol = _matrix_tolerance(projected_H)
    ka_tol = _matrix_tolerance(projected_A)
    h_sym = float(np.linalg.norm(H - H.T, ord=2)) if h_finite else None
    kh_sym = float(np.linalg.norm(projected_H - projected_H.T, ord=2)) if kh_finite else None

    raw_H_eigenvalues = np.linalg.eigvals(H) if h_finite else None
    raw_A_eigenvalues = np.linalg.eigvals(A) if a_finite else None
    quotient_K_eigenvalues = (
        np.linalg.eigvalsh((projected_H + projected_H.T) * 0.5) if kh_finite else None
    )
    quotient_A_eigenvalues = np.linalg.eigvals(projected_A) if ka_finite else None

    h_positive_count = (
        int(np.count_nonzero(np.linalg.eigvalsh((H + H.T) * 0.5) > h_tol))
        if h_finite and h_tol is not None else None
    )
    k_positive_count = (
        int(np.count_nonzero(quotient_K_eigenvalues > kh_tol))
        if quotient_K_eigenvalues is not None and kh_tol is not None else None
    )
    h_strict_pd = bool(h_finite and h_tol is not None and h_sym is not None
                       and h_sym <= h_tol and h_positive_count == DIM)
    k_strict_pd = bool(kh_finite and kh_tol is not None and kh_sym is not None
                       and kh_sym <= kh_tol and k_positive_count == DIM - 1)

    stable_mode_count = None
    unstable_mode_count = None
    neutral_mode_count = None
    strict_hurwitz = False
    if quotient_A_eigenvalues is not None and ka_tol is not None:
        real_parts = quotient_A_eigenvalues.real
        stable_mode_count = int(np.count_nonzero(real_parts < -ka_tol))
        unstable_mode_count = int(np.count_nonzero(real_parts > ka_tol))
        neutral_mode_count = int(len(real_parts) - stable_mode_count - unstable_mode_count)
        strict_hurwitz = stable_mode_count == DIM - 1

    gram = Q.T @ Q
    projector = Q @ Q.T
    expected_projector = np.eye(DIM) - np.outer(gauge, gauge)
    gauge_energy_residual = float(np.linalg.norm(H @ gauge)) if h_finite else None
    gauge_field_column_residual = float(np.linalg.norm(A @ gauge)) if a_finite else None
    f0_finite = bool(np.isfinite(f0).all())
    grad_H0_finite = bool(np.isfinite(grad_H0).all())
    H0_finite = bool(np.isfinite(H0))
    f0_norm_inf = float(np.linalg.norm(f0, ord=np.inf)) if f0_finite else None
    grad_H0_norm_inf = float(np.linalg.norm(grad_H0, ord=np.inf)) if grad_H0_finite else None
    Q_orthogonality_error = float(np.linalg.norm(gram - np.eye(DIM - 1), ord=2))
    quotient_projector_error = float(np.linalg.norm(projector - expected_projector, ord=2))
    equilibrium_pass = bool(f0_finite and f0_norm_inf is not None and f0_norm_inf <= FIXED_POINT_TOL)
    energy_stationary_pass = bool(grad_H0_finite and grad_H0_norm_inf is not None
                                  and grad_H0_norm_inf <= FIXED_POINT_TOL)
    gauge_invariance_pass = bool(
        h_finite and a_finite
        and gauge_energy_residual is not None and gauge_energy_residual <= GAUGE_INVARIANCE_TOL
        and gauge_field_column_residual is not None and gauge_field_column_residual <= GAUGE_INVARIANCE_TOL
        and Q_orthogonality_error <= GAUGE_INVARIANCE_TOL
        and quotient_projector_error <= GAUGE_INVARIANCE_TOL
    )
    quotient_eligible = bool(equilibrium_pass and gauge_invariance_pass and grad_H0_finite and H0_finite)
    lyapunov_eligible = bool(strict_hurwitz and quotient_eligible)
    lyapunov_P = np.empty((0, 0), dtype=np.float64)
    lyapunov_residual = np.empty((0, 0), dtype=np.float64)
    lyapunov = {
        "attempted": lyapunov_eligible,
        "positive_definite": None,
        "residual_norm": None,
        "residual_tolerance": None,
        "verified": False,
        "interpretation": WITNESS_SCOPE if lyapunov_eligible else None,
    }
    if strict_hurwitz and not quotient_eligible:
        lyapunov["reason_skipped"] = "projected matrix is strictly Hurwitz algebraically, but a verified equilibrium and gauge-invariant quotient are required"
    if lyapunov_eligible:
        try:
            from scipy.linalg import solve_continuous_lyapunov

            lyapunov_P = np.asarray(
                solve_continuous_lyapunov(projected_A.T, -np.eye(DIM - 1)), dtype=np.float64
            )
            lyapunov_residual = projected_A.T @ lyapunov_P + lyapunov_P @ projected_A + np.eye(DIM - 1)
            p_tol = _matrix_tolerance(lyapunov_P)
            residual_scale = max(
                1.0,
                float(np.linalg.norm(projected_A.T @ lyapunov_P, ord=2)),
                float(np.linalg.norm(lyapunov_P @ projected_A, ord=2)),
            )
            residual_tol = 1000.0 * EPS64 * residual_scale
            residual_norm = float(np.linalg.norm(lyapunov_residual, ord=2))
            p_sym = float(np.linalg.norm(lyapunov_P - lyapunov_P.T, ord=2))
            p_min = float(np.linalg.eigvalsh((lyapunov_P + lyapunov_P.T) * 0.5).min())
            p_positive = bool(p_tol is not None and p_sym <= p_tol and p_min > p_tol)
            lyapunov.update({
                "positive_definite": p_positive,
                "residual_norm": residual_norm,
                "residual_tolerance": residual_tol,
                "verified": bool(p_positive and residual_norm <= residual_tol),
                "P_symmetry_error": p_sym,
                "P_min_eigenvalue": p_min,
                "P_tolerance": p_tol,
            })
        except Exception as exc:  # Retain the local derivative result if SciPy fails.
            lyapunov["error"] = f"{type(exc).__name__}: {exc}"
            lyapunov["interpretation"] = "linearized Lyapunov computation unavailable; nonlinear radius not certified"
    linear_field_interpretation = (
        "eligible local quotient linearization at a verified equilibrium"
        if quotient_eligible else
        "algebraic spectrum only; quotient_eligible is required for local linearized-field interpretation"
    )
    quotient = {
        "dimension_full": DIM,
        "dimension_reduced": DIM - 1,
        "internal_coordinates": "(delta[10], pi[10], p[10], q[10]); no physical rescaling applied",
        "Q_transpose_Q_error": Q_orthogonality_error,
        "QQ_transpose_expected_projector_error": quotient_projector_error,
        "energy_gauge_residual_norm": gauge_energy_residual,
        "field_gauge_column_residual_norm": gauge_field_column_residual,
        "gauge_invariance_pass": gauge_invariance_pass,
        "gauge_invariance_tolerance_absolute": GAUGE_INVARIANCE_TOL,
    }
    summary = {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "coordinates": "internal (delta, pi, p, q); derivatives are at the zero state",
        "finite": {"H": h_finite, "A": a_finite, "K_projected": kh_finite, "A_projected": ka_finite},
        "tolerances": {
            "rule": "100 * float64_eps * max(1, spectral_norm(matrix)); Lyapunov residual uses 1000 * eps * max(1, ||Aq.T P||2, ||P Aq||2)",
            "H": h_tol, "A": a_tol, "K_projected": kh_tol, "A_projected": ka_tol,
        },
        "symmetry_error_norm2": {"H": h_sym, "K_projected": kh_sym},
        "origin": {
            "H0": H0 if H0_finite else None,
            "gradH0": grad_H0.tolist() if grad_H0_finite else None,
            "f0": f0.tolist() if f0_finite else None,
            "H0_finite": H0_finite,
            "gradH0_finite": grad_H0_finite,
            "f0_finite": f0_finite,
            "gradH0_norm_inf": grad_H0_norm_inf,
            "f0_norm_inf": f0_norm_inf,
            "fixed_point_tolerance_absolute": FIXED_POINT_TOL,
            "equilibrium_pass": equilibrium_pass,
            "energy_stationary_pass": energy_stationary_pass,
            "quotient_eligible": quotient_eligible,
            "eligibility_rule": "equilibrium_pass and gauge_invariance_pass and finite H0 and finite gradH0; gradH0 need not vanish for a field equilibrium",
        },
        "quotient": quotient,
        "raw_H_eigenvalues_real_imag": _complex_pairs(raw_H_eigenvalues),
        "raw_A_eigenvalues_real_imag": _complex_pairs(raw_A_eigenvalues),
        "quotient_K_eigenvalues": [float(x) for x in quotient_K_eigenvalues] if quotient_K_eigenvalues is not None else None,
        "quotient_A_eigenvalues_real_imag": _complex_pairs(quotient_A_eigenvalues),
        "energy_curvature": {
            "raw_positive_eigenvalue_count": h_positive_count,
            "raw_strict_positive_definite": h_strict_pd,
            "quotient_positive_eigenvalue_count": k_positive_count,
            "quotient_strict_positive_definite": k_strict_pd,
        },
        "linear_field": {
            "strict_hurwitz": strict_hurwitz,
            "strictly_stable_mode_count": stable_mode_count,
            "strictly_unstable_mode_count": unstable_mode_count,
            "near_neutral_mode_count": neutral_mode_count,
            "interpretation": linear_field_interpretation,
        },
        "lyapunov": lyapunov,
        "scope": "local derivatives and quotient-space spectra only; no nonlinear radius, attraction domain, or global stability claim",
    }
    arrays = {
        "Q": Q,
        "gauge": gauge,
        "H": H,
        "A": A,
        "H0": np.asarray(H0, dtype=np.float64),
        "gradH0": grad_H0,
        "f0": f0,
        "K_projected_raw": projected_H,
        "A_projected": projected_A,
        "H_raw_eigenvalues": np.asarray(raw_H_eigenvalues) if raw_H_eigenvalues is not None else np.empty((0,), dtype=np.complex128),
        "A_raw_eigenvalues": np.asarray(raw_A_eigenvalues) if raw_A_eigenvalues is not None else np.empty((0,), dtype=np.complex128),
        "K_projected_eigenvalues": np.asarray(quotient_K_eigenvalues) if quotient_K_eigenvalues is not None else np.empty((0,), dtype=np.float64),
        "A_projected_eigenvalues": np.asarray(quotient_A_eigenvalues) if quotient_A_eigenvalues is not None else np.empty((0,), dtype=np.complex128),
        "lyapunov_P": lyapunov_P,
        "lyapunov_residual": lyapunov_residual,
    }
    return summary, arrays


def _load_model(path: Path) -> Any:
    experiments = Path(__file__).resolve().parent
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    from symbolic_closure_rom import SymbolicClosureROM

    return SymbolicClosureROM.from_export(path)


def analyze_export(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    source = Path(path)
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    model = _load_model(source)
    summary, arrays = analyze_model(model, source_sha256=source_sha, source_path=str(source.resolve()))
    return summary, arrays, source_sha


def run_cli(model_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = []
    for raw_path in model_paths:
        path = Path(raw_path)
        record, arrays, sha = analyze_export(path)
        base = f"{path.stem}_{sha[:10]}"
        matrix_path = out / f"{base}.npz"
        np.savez_compressed(matrix_path, **arrays)
        record["matrix_file"] = matrix_path.name
        records.append(record)
    result = {
        "format": "symbolic_closure_local_math_check_v1",
        "analyzer_source_sha256": source_code_sha,
        "model_count": len(records),
        "models": records,
        "data_access": "none; only model JSON exports were read",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="one or more symbolic_closure_v1 model.json exports")
    parser.add_argument("--output-dir", "-o", required=True, help="directory for summary.json and matrix NPZ files")
    args = parser.parse_args(argv)
    result = run_cli(args.models, args.output_dir)
    print(json.dumps({"summary": str((Path(args.output_dir) / 'summary.json').resolve()),
                      "model_count": result["model_count"],
                      "source_shas": [m["source_sha256"] for m in result["models"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
