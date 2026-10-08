"""Numerical linear-solve contract for Loop0090 harmonic assembly.

This file deliberately contains no descriptor/model loading or physical
assembly.  The caller supplies each already-assembled complex hidden block.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.linalg
from scipy.linalg import LinAlgWarning


BACKWARD_LIMIT = 1e-10
RHO_LIMIT = 1e-6


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _record(start: float, H: np.ndarray, B: np.ndarray, *, error: str | None,
            X: np.ndarray | None = None, Y: np.ndarray | None = None,
            row_scale: np.ndarray | None = None, column_scale: np.ndarray | None = None,
            equilibrated_H: np.ndarray | None = None, equilibrated_B: np.ndarray | None = None,
            singular_values: np.ndarray | None = None, metrics: dict | None = None,
            solve_attempts: int = 0, svd_attempts: int = 0, qa_pass: bool = False) -> dict:
    base_metrics = {"backward_original": None, "backward_equilibrated": None,
                    "kappa": None, "gamma_n": None, "rho": None,
                    "seconds": float(time.perf_counter() - start)}
    if metrics is not None:
        base_metrics.update(metrics)
    return {"qa_pass": bool(qa_pass), "error": error, "H": H, "B": B, "X": X, "Y": Y,
            "row_scale": row_scale, "column_scale": column_scale,
            "equilibrated_H": equilibrated_H, "equilibrated_B": equilibrated_B,
            "singular_values": singular_values, "metrics": base_metrics,
            "solve_attempts": solve_attempts, "svd_attempts": svd_attempts}


def _column_backward(H: np.ndarray, X: np.ndarray, B: np.ndarray) -> np.ndarray:
    residual = H @ X - B
    hnorm = np.linalg.norm(H, "fro")
    numer = np.linalg.norm(residual, axis=0)
    denom = hnorm * np.linalg.norm(X, axis=0) + np.linalg.norm(B, axis=0)
    answer = np.zeros_like(numer)
    np.divide(numer, denom, out=answer, where=denom>0)
    answer[(denom==0)&(numer!=0)] = np.inf
    return answer


def solve_scaled(H: np.ndarray, B: np.ndarray, deadline: float) -> dict:
    """Solve one complex multi-RHS block with the preregistered scaling.

    A returned record is always sufficient for a failure artifact.  Numeric
    QA failures retain the raw/equilibrated matrices and computed solution;
    solver and deadline failures retain the original inputs without retrying.
    """
    started = time.perf_counter()
    raw_h = np.asarray(H, dtype=np.complex128).copy()
    raw_b = np.asarray(B, dtype=np.complex128).copy()
    if time.perf_counter() > deadline:
        return _record(started, raw_h, raw_b, error="deadline_before_scaling")
    if raw_h.ndim != 2 or raw_h.shape[0] != raw_h.shape[1] or raw_b.ndim != 2 or raw_b.shape[0] != raw_h.shape[0]:
        return _record(started, raw_h, raw_b, error="shape")
    if not (np.isfinite(raw_h).all() and np.isfinite(raw_b).all()):
        return _record(started, raw_h, raw_b, error="nonfinite_input")

    rowmax = np.max(np.abs(raw_h), axis=1)
    if not np.isfinite(rowmax).all() or np.any(rowmax == 0):
        return _record(started, raw_h, raw_b, error="zero_or_nonfinite_row_scale")
    dr = 1.0 / rowmax
    left = dr[:, None] * raw_h
    colmax = np.max(np.abs(left), axis=0)
    if not np.isfinite(colmax).all() or np.any(colmax == 0):
        return _record(started, raw_h, raw_b, error="zero_or_nonfinite_column_scale", row_scale=dr)
    dc = 1.0 / colmax
    heq = left * dc[None, :]
    beq = dr[:, None] * raw_b
    if not all(np.isfinite(value).all() for value in (dr, dc, heq, beq)):
        return _record(started, raw_h, raw_b, error='nonfinite_scaling', row_scale=dr,
                       column_scale=dc, equilibrated_H=heq, equilibrated_B=beq)
    if time.perf_counter() > deadline:
        return _record(started, raw_h, raw_b, error="deadline_before_svd", row_scale=dr, column_scale=dc,
                       equilibrated_H=heq, equilibrated_B=beq)
    try:
        singular_values = scipy.linalg.svdvals(heq, check_finite=False)
    except Exception as exc:
        return _record(started, raw_h, raw_b, error=f"svd_{type(exc).__name__}", row_scale=dr, column_scale=dc,
                       equilibrated_H=heq, equilibrated_B=beq, svd_attempts=1)
    if time.perf_counter() > deadline:
        return _record(started, raw_h, raw_b, error="deadline_after_svd", row_scale=dr, column_scale=dc,
                       equilibrated_H=heq, equilibrated_B=beq, singular_values=singular_values, svd_attempts=1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            y = scipy.linalg.solve(heq, beq, assume_a="gen", check_finite=False)
    except Exception as exc:
        return _record(started, raw_h, raw_b, error=f"solve_{type(exc).__name__}", row_scale=dr, column_scale=dc,
                       equilibrated_H=heq, equilibrated_B=beq, singular_values=singular_values,
                       solve_attempts=1, svd_attempts=1)
    x = dc[:, None] * y
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        return _record(started, raw_h, raw_b, error="nonfinite_solution", X=x, Y=y, row_scale=dr, column_scale=dc,
                       equilibrated_H=heq, equilibrated_B=beq, singular_values=singular_values,
                       solve_attempts=1, svd_attempts=1)
    if time.perf_counter() > deadline:
        return _record(started, raw_h, raw_b, error="deadline_after_solve", X=x, Y=y, row_scale=dr, column_scale=dc,
                       equilibrated_H=heq, equilibrated_B=beq, singular_values=singular_values,
                       solve_attempts=1, svd_attempts=1)

    original = _column_backward(raw_h, x, raw_b)
    equilibrated = _column_backward(heq, y, beq)
    smax, smin = float(singular_values[0]), float(singular_values[-1])
    kappa = np.inf if smin <= 0 else smax / smin
    n = raw_h.shape[0]
    eps = np.finfo(np.float64).eps
    gamma = n * eps / (1.0 - n * eps)
    rho = kappa * max(float(np.max(equilibrated)), gamma)
    metrics = {"backward_original": _finite_or_none(np.max(original)),
               "backward_equilibrated": _finite_or_none(np.max(equilibrated)),
               "kappa": _finite_or_none(kappa), "gamma_n": _finite_or_none(gamma), "rho": _finite_or_none(rho)}
    timed_out = time.perf_counter() > deadline
    passed = bool(not timed_out and all(value is not None for value in metrics.values()) and metrics["backward_original"] <= BACKWARD_LIMIT and
                  metrics["backward_equilibrated"] <= BACKWARD_LIMIT and metrics["rho"] <= RHO_LIMIT)
    return _record(started, raw_h, raw_b, error=None if passed else ('deadline_after_checks' if timed_out else "numeric_gate"), X=x, Y=y, row_scale=dr,
                   column_scale=dc, equilibrated_H=heq, equilibrated_B=beq, singular_values=singular_values,
                   metrics=metrics, solve_attempts=1, svd_attempts=1, qa_pass=passed)
