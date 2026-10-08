"""Bounded helpers for the Loop0073 differentiable joint-energy screen.

The module contains no data reader and never launches a training loop.  It
implements the fixed 224-term origin storage algebra, exact full-coupled
Jacobian quotienting, the single Clarabel soft local SDP, and calibration
diagnostics used by the root-owned campaign.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import fit_sparse_storage as sparse  # noqa: E402


N = 32
LIBRARY_SIZE = 224
PQ_COUNT = 6
FIT_MARGIN = 1e-5
RESIDUAL_TOL = 1e-6
SOLVER_MAX_ITER = 200
JOINT_NAMES = (
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "head.weight", "head.bias", "c", "amplitude", "alpha", "xi",
    "decoder.weight", "decoder.bias",
)


def storage_scales(Hks: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Fixed ``s_k=max(||V H_k V||_F,1e-8)`` scales."""
    h = np.asarray(Hks, dtype=float)
    v = np.asarray(V, dtype=float)
    if h.shape != (LIBRARY_SIZE, N, N) or v.shape != (N, N):
        raise ValueError("Hks/V shape")
    return np.maximum(np.linalg.norm(np.einsum("ij,kjl,lm->kim", v, h, v), axis=(1, 2)), 1e-8)


def select_energy_support(coefficients: np.ndarray, scales: np.ndarray, limit: int = 18) -> np.ndarray:
    """Retain all six Pq slots and the fixed top-18 standardized terms."""
    c = np.asarray(coefficients, dtype=float); s = np.asarray(scales, dtype=float)
    if c.shape != (LIBRARY_SIZE,) or s.shape != (LIBRARY_SIZE,): raise ValueError("coefficient/scale shape")
    return sparse.select_support(c, s, limit=limit)


def _torch_maps(dtype, device):
    import torch
    return (torch.as_tensor(sparse.P32, dtype=dtype, device=device),
            torch.as_tensor(sparse.E33, dtype=dtype, device=device))


def exact_full_jacobian(model, *, create_graph: bool = True):
    """Differentiate the live 33-state full coupled field at the origin."""
    import torch
    try:
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
    except StopIteration:
        dtype, device = torch.float64, torch.device("cpu")
    zero = torch.zeros(33, dtype=dtype, device=device, requires_grad=True)
    def field(x):
        value = model.field(x[None])
        return value[0] if value.ndim == 2 else value
    return torch.autograd.functional.jacobian(field, zero, create_graph=create_graph, strict=False)


def exact_j32(model, *, create_graph: bool = True):
    """Return ``P J33 E`` while retaining gradients to ODE-dependent tensors."""
    j33 = exact_full_jacobian(model, create_graph=create_graph)
    p, e = _torch_maps(j33.dtype, j33.device)
    return p @ j33 @ e


def analytic_j32(params: dict[str, np.ndarray]) -> np.ndarray:
    """Use the registered 0065 analytic full Jacobian and quotient it."""
    from audit_coupled_equilibrium import analytic_jacobian
    j33 = analytic_jacobian(params)
    return sparse.P32 @ j33 @ sparse.E33


def compare_j32(model, params: dict[str, np.ndarray]) -> float:
    """Relative max difference between analytic and live full-autograd J32."""
    actual = exact_j32(model, create_graph=False).detach().cpu().numpy()
    reference = analytic_j32(params)
    return float(np.max(np.abs(actual-reference) / np.maximum(1., np.abs(reference))))


def _torch_matrix(value, *, dtype, device):
    import torch
    return torch.as_tensor(np.asarray(value, dtype=float), dtype=dtype, device=device)


def energy_matrices(gamma, H0: np.ndarray, Hks: np.ndarray):
    """Construct differentiable H from fixed origin Hessian basis matrices."""
    import torch
    if not torch.is_tensor(gamma):
        gamma = torch.as_tensor(gamma, dtype=torch.float64)
    h0 = _torch_matrix(H0, dtype=gamma.dtype, device=gamma.device)
    hs = _torch_matrix(Hks, dtype=gamma.dtype, device=gamma.device)
    return h0 + torch.tensordot(gamma, hs, dims=([0], [0]))


def energy_lh(model, gamma, H0: np.ndarray, Hks: np.ndarray, V: np.ndarray):
    """Differentiable ``mean(relu(1e-5-eig(VDV))^2)``."""
    import torch
    J = exact_j32(model, create_graph=True)
    H = energy_matrices(gamma, H0, Hks)
    v = _torch_matrix(V, dtype=J.dtype, device=J.device)
    D = -(J.transpose(-1, -2) @ H + H @ J)
    eigenvalues = torch.linalg.eigvalsh(v @ D @ v)
    return torch.relu(FIT_MARGIN - eigenvalues).square().mean(), eigenvalues, D


def energy_lh_from_j32(J, gamma, H0: np.ndarray, Hks: np.ndarray, V: np.ndarray):
    """Same LH helper for a supplied differentiable J32 (useful in tests)."""
    import torch
    if not torch.is_tensor(J): J = torch.as_tensor(J, dtype=torch.float64)
    H = energy_matrices(gamma, H0, Hks)
    v = _torch_matrix(V, dtype=J.dtype, device=J.device)
    D = -(J.transpose(-1, -2) @ H + H @ J)
    eig = torch.linalg.eigvalsh(v @ D @ v)
    return torch.relu(FIT_MARGIN-eig).square().mean(), eig, D


def first_gradient_norm(loss, model, names: Sequence[str] = JOINT_NAMES):
    """Euclidean norm over the exact twelve tensors; unused gradients are zero."""
    import torch
    by_name = dict(model.named_parameters())
    missing = [name for name in names if name not in by_name]
    if missing: raise ValueError(f"missing allowed tensors: {missing}")
    tensors = [by_name[name] for name in names]
    if not loss.requires_grad:
        gradients = [None] * len(tensors)
    else:
        gradients = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    total = torch.zeros((), dtype=loss.dtype, device=loss.device)
    for tensor, gradient in zip(tensors, gradients):
        if gradient is not None: total = total + gradient.square().sum()
    return torch.sqrt(total), gradients


def calibrate_lambda(loss_data, loss_energy, model, names: Sequence[str] = JOINT_NAMES) -> tuple[float, float, float]:
    """Set the sole registered ``lambdaH=.1*gL/gH`` or fail closed."""
    g_l, _ = first_gradient_norm(loss_data, model, names)
    g_h, _ = first_gradient_norm(loss_energy, model, names)
    values = [float(g_l.detach()), float(g_h.detach())]
    if not all(np.isfinite(values)) or values[0] <= 1e-12 or values[1] <= 1e-12:
        raise FloatingPointError(f"unusable energy bias gradients: gL={values[0]}, gH={values[1]}")
    lam = .1 * values[0] / values[1]
    if not np.isfinite(lam): raise FloatingPointError("nonfinite lambdaH")
    return lam, values[0], values[1]


def _normalized_matrices(H: np.ndarray, J: np.ndarray, V: np.ndarray):
    D = -(J.T @ H + H @ J)
    return V @ H @ V, V @ D @ V, D


def validate_soft_solution(H0: np.ndarray, Hks: np.ndarray, V: np.ndarray, J: np.ndarray,
                           gamma: np.ndarray, t: float, scales: np.ndarray,
                           support: np.ndarray | None = None) -> dict[str, object]:
    """Independently validate a candidate SDP solution and fail closed."""
    h0 = np.asarray(H0, float); hs = np.asarray(Hks, float); v = np.asarray(V, float)
    j = np.asarray(J, float); c = np.asarray(gamma, float); s = np.asarray(scales, float)
    if support is None: support = np.ones(LIBRARY_SIZE, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if c.shape != (LIBRARY_SIZE,) or support.shape != (LIBRARY_SIZE,):
        raise ValueError("gamma/support shape")
    exact = c.copy(); exact[~support] = 0.
    tv = float(t)
    # Build the candidate first so an invalid numerical record retains the
    # actual matrices whenever they can be formed; never call eigvalsh on
    # non-finite/overflowed matrices.
    try:
        Hraw = h0 + np.tensordot(exact, hs, axes=(0, 0))
        with np.errstate(over="ignore", invalid="ignore"):
            Hn, Dn, Draw = _normalized_matrices(Hraw, j, v)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        Hraw = np.full((N, N), np.nan); Hn = np.full((N, N), np.nan)
        Dn = Draw = np.full((N, N), np.nan)
    inputs_finite = bool(np.isfinite(h0).all() and np.isfinite(hs).all() and
                         np.isfinite(v).all() and np.isfinite(j).all() and
                         np.isfinite(c).all() and np.isfinite(s).all() and np.isfinite(tv))
    matrices_finite = bool(np.isfinite(Hraw).all() and np.isfinite(Hn).all() and np.isfinite(Dn).all())
    if (not inputs_finite or not matrices_finite or tv < 0):
        return {"valid": False, "finite": False, "inactive_max": np.inf,
                "residual_h": np.inf, "residual_d": np.inf,
                "H_normalized": Hn, "D_normalized": Dn,
                "H_raw": Hraw, "D_raw": Draw,
                "gamma": exact, "t": tv, "beta": s*exact}
    residual_h = max(0., FIT_MARGIN-float(np.linalg.eigvalsh(Hn).min()))
    residual_d = max(0., FIT_MARGIN-float(np.linalg.eigvalsh(Dn + tv*np.eye(N)).min()))
    inactive = float(np.max(np.abs(c[~support]))) if np.any(~support) else 0.
    finite = bool(np.isfinite(exact).all() and np.isfinite(tv) and np.isfinite(Hn).all() and np.isfinite(Dn).all())
    return {"valid": bool(finite and inactive <= RESIDUAL_TOL and residual_h <= RESIDUAL_TOL and residual_d <= RESIDUAL_TOL),
            "finite": finite, "inactive_max": inactive, "residual_h": residual_h,
            "residual_d": residual_d, "H_normalized": Hn, "D_normalized": Dn,
            "H_raw": Hraw, "D_raw": Draw, "gamma": exact, "t": tv, "beta": s*exact}


def solve_soft_storage(H0: np.ndarray, Hks: np.ndarray, V: np.ndarray, J: np.ndarray,
                       scales: np.ndarray | None = None, support: np.ndarray | None = None) -> dict[str, object]:
    """One fixed Clarabel solve, with independent residual/finiteness checks."""
    import cvxpy as cp
    h0 = np.asarray(H0, float); hs = np.asarray(Hks, float); v = np.asarray(V, float); j = np.asarray(J, float)
    if h0.shape != (N, N) or hs.shape != (LIBRARY_SIZE, N, N) or v.shape != (N, N) or j.shape != (N, N):
        raise ValueError("origin matrix shapes")
    if scales is None: scales = storage_scales(hs, v)
    scales = np.asarray(scales, float)
    if scales.shape != (LIBRARY_SIZE,) or np.any(~np.isfinite(scales)) or np.any(scales < 1e-8):
        raise ValueError("invalid fixed scales")
    if support is None: support = np.ones(LIBRARY_SIZE, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if support.shape != (LIBRARY_SIZE,): raise ValueError("support shape")
    gamma = cp.Variable(LIBRARY_SIZE); t = cp.Variable(nonneg=True)
    H = h0 + sum(gamma[k] * hs[k] for k in range(LIBRARY_SIZE))
    D = -(j.T @ H + H @ j)
    constraints = [v @ H @ v - FIT_MARGIN*np.eye(N) >> 0,
                   v @ D @ v + t*np.eye(N) - FIT_MARGIN*np.eye(N) >> 0]
    if np.any(~support):
        constraints.append(gamma[~support] == 0)
    beta = cp.multiply(scales, gamma)
    problem = cp.Problem(cp.Minimize(t + 1e-4*cp.sum(cp.abs(beta))), constraints)
    try:
        problem.solve(solver="CLARABEL", max_iter=SOLVER_MAX_ITER,
                      tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_feas=1e-8, verbose=False)
    except Exception as exc:
        return {"status": "solver_error", "feasible": False, "error": str(exc)}
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or gamma.value is None or t.value is None:
        return {"status": str(problem.status), "feasible": False,
                "iterations": getattr(problem.solver_stats, "num_iters", None)}
    raw_gamma = np.asarray(gamma.value, float); tv = float(t.value)
    validation = validate_soft_solution(h0, hs, v, j, raw_gamma, tv, scales, support)
    result_gamma = validation["gamma"]; Hn = validation["H_normalized"]; Dn = validation["D_normalized"]
    return {"status": str(problem.status), "feasible": bool(validation["valid"]), "solver_feasible": True,
            "iterations": getattr(problem.solver_stats, "num_iters", None), "gamma": result_gamma,
            "t": tv, "beta": scales*result_gamma, "scales": scales,
            "H_normalized": Hn, "D_normalized": Dn, "H_raw": validation["H_raw"],
            "D_raw": validation["D_raw"], "gamma_raw": raw_gamma,
            "residual_h": validation["residual_h"],
            "residual_d": validation["residual_d"], "inactive_max": validation["inactive_max"],
            "objective": float(problem.value),
            "dual_H": None if constraints[0].dual_value is None else np.asarray(constraints[0].dual_value),
            "dual_D": None if constraints[1].dual_value is None else np.asarray(constraints[1].dual_value)}


def assert_zero_field(model, tolerance: float = 1e-10) -> float:
    import torch
    rank = int(getattr(model, "rank", 3)); zero = torch.zeros((1, 30+rank), dtype=torch.float64)
    value = float(model.field(zero).abs().max().detach())
    if not np.isfinite(value) or value > tolerance: raise AssertionError(f"f(0)={value} exceeds {tolerance}")
    return value


def synthetic_matrices(seed: int = 73):
    rng = np.random.default_rng(seed); Q = rng.normal(size=(N, N)); H0 = Q.T@Q + np.eye(N)
    hs = rng.normal(size=(LIBRARY_SIZE, N, N)); hs = .5*(hs + hs.transpose(0, 2, 1))
    V = np.diag(np.linspace(.5, 1.5, N)); J = -np.eye(N)
    return H0, hs, V, J
