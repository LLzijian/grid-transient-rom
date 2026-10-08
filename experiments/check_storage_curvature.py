"""Loop0069 origin feasibility checker for the single physical curvature term.

This module adds exactly one coefficient to the immutable Loop0068 library.
The synthetic entry point is data-free; the normal entry point loads only the
registered 0065/S/M3/network artifacts and writes an origin-matrix proof
artifact after the caller explicitly requests a run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DEPS = ROOT / ".loop-work/research_deps_0067"
if RESEARCH_DEPS.is_dir() and str(RESEARCH_DEPS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DEPS))
sys.path.insert(0, str(ROOT / "experiments"))

import fit_sparse_storage as sparse  # noqa: E402


N = 32
PHYSICAL = 29
LIBRARY_SIZE = 225
CHI_INDEX = 224


def _sym(a: np.ndarray) -> np.ndarray:
    return .5 * (np.asarray(a, dtype=float) + np.asarray(a, dtype=float).T)


def mass_coi_projector(mass: np.ndarray) -> np.ndarray:
    """The registered mass-weighted COI projector R."""
    m = np.asarray(mass, dtype=float)
    if m.shape != (10,) or not np.isfinite(m).all() or np.any(m <= 0):
        raise ValueError("mass must be positive 10-vector")
    return np.eye(10) - np.ones((10, 1)) @ (m[None, :] / m.sum())


def angle_feature_derivative(params: dict[str, np.ndarray]) -> np.ndarray:
    """Exact 52-by-10 centered normalized angle-feature derivative at zero.

    This is the same ``gd`` construction used by the registered 0065
    coupled-equilibrium audit: sine block first, centered cosine block second.
    """
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    h = p["angle_incidence"]
    nedge = h.shape[1]
    scale = p["feature_scale"]
    phase = p["delta_star"] @ h
    return np.vstack((np.diag(np.cos(phase) / scale[20:20+nedge]) @ h.T,
                      np.diag(-np.sin(phase) / scale[20+nedge:20+2*nedge]) @ h.T))


def instantaneous_storage_curvature(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return R, gd, ad and the fixed gauge-preserving Ks."""
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    R = mass_coi_projector(p["mass"])
    gd = angle_feature_derivative(p)
    alpha = p["alpha"] * p["alpha_support"]
    ad = alpha.T @ gd
    raw = p["Cq"] @ ad
    Ks = R.T @ _sym(raw) @ R
    return {"R": R, "gd": gd, "ad": ad, "raw": raw, "Ks": _sym(Ks)}


def chi_value(z: np.ndarray, params: dict[str, np.ndarray], Ks: np.ndarray | None = None) -> np.ndarray:
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    if Ks is None: Ks = instantaneous_storage_curvature(p)["Ks"]
    x = sparse.reduced_to_full(np.asarray(z, dtype=float))
    d = x[..., :10]
    return .5 * np.einsum("...i,ij,...j->...", d, Ks, d)


def chi_gradient(z: np.ndarray, params: dict[str, np.ndarray], Ks: np.ndarray | None = None) -> np.ndarray:
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    if Ks is None: Ks = instantaneous_storage_curvature(p)["Ks"]
    x = sparse.reduced_to_full(np.asarray(z, dtype=float))
    g = np.zeros_like(x)
    g[..., :10] = np.einsum("ij,...j->...i", Ks, x[..., :10])
    return g @ sparse.E33


def chi_hessian_origin(params: dict[str, np.ndarray], Ks: np.ndarray | None = None) -> np.ndarray:
    if Ks is None: Ks = instantaneous_storage_curvature(params)["Ks"]
    ed = sparse.E33[:10, :9]
    h = np.zeros((N, N), dtype=float)
    h[:9, :9] = ed.T @ Ks @ ed
    return _sym(h)


def augmented_library(z: np.ndarray, params: dict[str, np.ndarray]):
    """Append chi to the exact 224-column Loop0068 library."""
    values = sparse.library_values(z, params)
    gradients = sparse.library_gradients(z, params)
    hessians = sparse.library_hessians_origin(params)
    Ks = instantaneous_storage_curvature(params)["Ks"]
    chi = chi_value(z, params, Ks)[..., None]
    chig = chi_gradient(z, params, Ks)[..., None, :]
    chih = chi_hessian_origin(params, Ks)[None, ...]
    return (np.concatenate((values, chi), axis=-1),
            np.concatenate((gradients, chig), axis=-2),
            np.concatenate((hessians, chih), axis=0))


def augmented_origin_hessian(params: dict[str, np.ndarray], coefficients: np.ndarray | None = None) -> np.ndarray:
    h3 = sparse.h3_hessian_origin(params)
    if coefficients is None: return h3
    c = np.asarray(coefficients, dtype=float)
    if c.shape != (LIBRARY_SIZE,): raise ValueError("expected 225 coefficients")
    _, _, hs = augmented_library(np.zeros((1, N)), params)
    return _sym(h3 + np.tensordot(c, hs, axes=(0, 0)))


def normalized_origin_matrices(params: dict[str, np.ndarray], coefficients: np.ndarray | None = None):
    h3 = sparse.h3_hessian_origin(params)
    H = augmented_origin_hessian(params, coefficients)
    W = sparse._w_matrix(h3)
    J = np.asarray(params["J32"], dtype=float)
    invw = np.diag(1 / np.sqrt(np.diag(W)))
    L = -(J.T @ H + H @ J)
    return H, L, W, invw @ H @ invw, invw @ L @ invw


def solve_origin_feasibility(params: dict[str, np.ndarray]) -> dict[str, object]:
    """Solve the registered zero-objective 225-variable local screen."""
    import cvxpy as cp
    h3 = sparse.h3_hessian_origin(params)
    _, _, W, _, _ = normalized_origin_matrices(params)
    hs = augmented_library(np.zeros((1, N)), params)[2]
    J = np.asarray(params["J32"], dtype=float)
    invw = np.diag(1 / np.sqrt(np.diag(W)))
    c = cp.Variable(LIBRARY_SIZE)
    H = h3 + sum(c[k] * hs[k] for k in range(LIBRARY_SIZE))
    Hn = invw @ H @ invw
    Ln = invw @ (-(J.T @ H + H @ J)) @ invw
    con_h = Hn - 1e-6*np.eye(N) >> 0
    con_l = Ln - 1e-6*np.eye(N) >> 0
    problem = cp.Problem(cp.Minimize(0), [con_h, con_l])
    try:
        problem.solve(solver="CLARABEL", max_iter=200, tol_gap_abs=1e-8,
                      tol_gap_rel=1e-8, tol_feas=1e-8, verbose=False)
    except Exception as exc:
        return {"status": "solver_error", "feasible": False, "error": str(exc)}
    stats = problem.solver_stats
    feasible = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and c.value is not None
    coeff = np.asarray(c.value, dtype=float) if feasible else np.empty(0, dtype=float)
    Hn = Ln = None
    if feasible:
        _, _, _, Hn, Ln = normalized_origin_matrices(params, coeff)
    result: dict[str, object] = {
        "status": str(problem.status), "feasible": bool(feasible),
        "iterations": None if stats is None else getattr(stats, "num_iters", None),
        "coefficients": coeff, "H_normalized": Hn, "Lyapunov_normalized": Ln,
        "postsolve_hessian_min_W": float(np.linalg.eigvalsh(Hn).min()) if feasible else None,
        "postsolve_lyapunov_min_W": float(np.linalg.eigvalsh(Ln).min()) if feasible else None,
        "dual_hessian": None if con_h.dual_value is None else np.asarray(con_h.dual_value),
        "dual_lyapunov": None if con_l.dual_value is None else np.asarray(con_l.dual_value),
    }
    return result


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registered_sources(root: Path | str = ROOT) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    root = Path(root)
    eq = root / "private_data/task2_coupled_equilibrium_0065/coupled_equilibrium.npz"
    evidence = eq.parent / "evidence.json"
    m3 = root / "private_data/task2_m3_0020/model.npz"
    network = root / "private_data/task2_electrical_ports_0048/physical_network.npz"
    S = root / "private_data/task2_long_0060_5s"
    paths = {"model": S/"model.pt", "normalization": S/"normalization.npz", "base": m3, "network": network}
    if not eq.exists() or not evidence.exists() or any(not p.exists() for p in paths.values()):
        raise FileNotFoundError("registered 0065/S/M3/network source missing")
    expected = json.loads(evidence.read_text(encoding="utf-8")).get("source_sha256", {})
    hashes = {k: source_sha256(v) for k, v in paths.items()}
    for key, value in hashes.items():
        if expected.get(key) != value:
            raise AssertionError(f"frozen source hash mismatch: {key}")
    with np.load(eq, allow_pickle=False) as s:
        params = {"mass": s["param_mass"], "incidence": s["param_incidence"],
                  "angle_incidence": s["param_angle_incidence"], "delta_star": s["param_delta_star"],
                  "pm_star": s["param_pm_star"], "coupling": s["param_coupling"],
                  "damping": s["param_damping"], "rate": s["param_rate"], "gain": s["param_gain"],
                  "feature_scale": s["param_feature_scale"], "J32": s["J32"],
                  "alpha": s["param_alpha"], "alpha_support": s["param_alpha_support"],
                  "Cq": s["param_Cq"]}
    return params, hashes


def synthetic_params() -> dict[str, np.ndarray]:
    p = sparse.synthetic_params()
    rng = np.random.default_rng(69)
    p["alpha"] = rng.normal(size=(52, 3))*.03
    p["alpha_support"] = np.ones((52, 3))
    return p


def synthetic_checks() -> dict[str, float | bool | list[int]]:
    p = synthetic_params(); curv = instantaneous_storage_curvature(p); Ks = curv["Ks"]
    z = np.random.default_rng(6901).normal(size=(4, N))*.2
    eps = 1e-6; max_grad = 0.
    for i in range(4):
        for j in (0, 4, 19, 31):
            up, dn = z.copy(), z.copy(); up[i, j] += eps; dn[i, j] -= eps
            fd = (chi_value(up, p)[i] - chi_value(dn, p)[i])/(2*eps)
            max_grad = max(max_grad, abs(float(fd-chi_gradient(z, p)[i, j])))
    h = chi_hessian_origin(p, Ks)
    gp = chi_gradient((eps*np.eye(N)[5])[None], p)[0]
    gm = chi_gradient((-eps*np.eye(N)[5])[None], p)[0]
    max_hessian = float(np.max(np.abs((gp-gm)/(2*eps)-h[:, 5])))
    gauge = np.ones(10)
    auto = autograd_checks(p, z[0])
    return {"library_shape": list(augmented_library(z, p)[0].shape),
            "Ks_symmetric": bool(np.allclose(Ks, Ks.T)),
            "Ks_gauge_max": float(np.max(np.abs(Ks @ gauge))),
            "chi_gradient_fd_max": float(max_grad),
            "chi_hessian_fd_max": max_hessian, **auto}


def autograd_checks(params: dict[str, np.ndarray], z: np.ndarray) -> dict[str, float]:
    """Compare chi's analytic derivatives with torch autograd on one point."""
    import torch
    Ks = torch.as_tensor(instantaneous_storage_curvature(params)["Ks"], dtype=torch.float64)
    E = torch.as_tensor(sparse.E33, dtype=torch.float64)
    point = torch.tensor(np.asarray(z, dtype=float), dtype=torch.float64, requires_grad=True)
    def fun(y):
        d = (y @ E.T)[:10]
        return .5 * (d @ Ks @ d)
    auto_g = torch.autograd.functional.jacobian(fun, point).detach().numpy()
    auto_h = torch.autograd.functional.hessian(fun, point).detach().numpy()
    return {"chi_autograd_gradient_max": float(np.max(np.abs(auto_g - chi_gradient(np.asarray(z)[None], params)[0]))),
            "chi_autograd_hessian_max": float(np.max(np.abs(auto_h - chi_hessian_origin(params))))}


def _jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, dict): return {str(k): _jsonable(v) for k, v in x.items()}
    return x


def run(output: Path) -> dict[str, object]:
    if output.exists(): raise FileExistsError(output)
    params, hashes = load_registered_sources()
    result = solve_origin_feasibility(params)
    output.mkdir(parents=True)
    coeff = np.asarray(result.pop("coefficients", np.empty(0)), float)
    H, L, W, Hn, Ln = normalized_origin_matrices(params, coeff if coeff.size else None)
    np.savez_compressed(output / "storage_curvature_matrices.npz", coefficients=coeff,
                        H=H, Lyapunov=L, W=W, H_normalized=Hn,
                        Lyapunov_normalized=Ln, Ks=instantaneous_storage_curvature(params)["Ks"],
                        R=instantaneous_storage_curvature(params)["R"],
                        gd=instantaneous_storage_curvature(params)["gd"],
                        ad=instantaneous_storage_curvature(params)["ad"])
    report = _jsonable({**result, "source_sha256": hashes, "test_read": False,
                        "solver": "CLARABEL", "max_iter": 200,
                        "tolerances": {"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8},
                        "chi_index": CHI_INDEX, "library_size": LIBRARY_SIZE})
    report['matrix_role'] = 'candidate' if coeff.size else 'fixed_H3_reference_no_solution'
    report['code_sha256'] = source_sha256(Path(__file__))
    (output / "evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.synthetic:
        print(json.dumps(synthetic_checks(), indent=2, allow_nan=False)); return
    if args.output is None: raise SystemExit("--output is required for the explicit frozen-source run")
    print(json.dumps(run(args.output), indent=2, allow_nan=False))


if __name__ == "__main__": main()
