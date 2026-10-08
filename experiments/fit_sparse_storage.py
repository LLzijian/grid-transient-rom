"""Loop0068 sparse-storage candidate fitter.

The implementation in this file is deliberately small and functional.  It
contains the immutable 32-coordinate library, its exact origin derivatives,
the two Clarabel fits, and domain/probe helpers.  The normal command line
entry point only runs a synthetic contract smoke unless an explicit
development input is supplied; in particular this module never enumerates or
opens a TEST split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Clarabel/cvxpy is intentionally isolated to the registered Loop0067 target;
# importing this module otherwise remains a NumPy-only synthetic/unit helper.
_RESEARCH_DEPS = Path(__file__).resolve().parents[1] / ".loop-work/research_deps_0067"
if _RESEARCH_DEPS.is_dir() and str(_RESEARCH_DEPS) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_DEPS))

import numpy as np


N = 32
NPHYS = 29
NQ = 3
NEDGE = 26
LIBRARY_SIZE = 224
PQ_COUNT = 6
DEFAULT_SEED = 68
BOUNDARY_SEED = 6801


def _as_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize array-valued physics parameters without changing semantics."""
    p = {k: np.asarray(v, dtype=float) for k, v in params.items()}
    if "angle_incidence" not in p:
        inc = p["incidence"]
        coupling = p["coupling"]
        p["angle_incidence"] = inc[:, coupling > 0]
    if p["angle_incidence"].shape[1] != NEDGE:
        raise ValueError("Loop0068 requires the immutable 26 active angle edges")
    if "feature_scale" not in p:
        p["feature_scale"] = np.ones(20 + 2 * NEDGE)
    return p


def gauge_maps() -> tuple[np.ndarray, np.ndarray]:
    """Return the registered 32-by-33 quotient and its 33-by-32 embedding."""
    project = np.zeros((N, 33), dtype=float)
    embed = np.zeros((33, N), dtype=float)
    project[:9, :9] = np.eye(9); project[:9, 9] = -1
    project[9:19, 10:20] = np.eye(10)
    project[19:29, 20:30] = np.eye(10); project[29:, 30:] = np.eye(3)
    embed[:9, :9] = np.eye(9); embed[10:20, 9:19] = np.eye(10)
    embed[20:30, 19:29] = np.eye(10); embed[30:, 29:] = np.eye(3)
    return project, embed


P32, E33 = gauge_maps()


def reduced_to_full(z: np.ndarray) -> np.ndarray:
    """Embed ``[delta_1-delta_10, omega, p, q]`` into the 33-state gauge."""
    a = np.asarray(z, dtype=float)
    if a.shape[-1] != N:
        raise ValueError("expected 32 reduced coordinates")
    return a @ E33.T


def full_to_reduced(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.shape[-1] != 33:
        raise ValueError("expected 33 full coordinates")
    return a @ P32.T


def _physical(params: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = _as_params(params)
    inc = p["angle_incidence"]
    coupling = p["coupling"]
    if coupling.size == inc.shape[1]:
        active = coupling
    else:
        active = coupling[coupling > 0]
    phase = p["delta_star"] @ inc
    k0 = (inc * (active * np.cos(phase))) @ inc.T
    return inc, phase, (k0 + k0.T) / 2


def h3_value(z: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate immutable H3; p is restored to ``p+pm_star`` at this API."""
    p = _as_params(params); x = reduced_to_full(z)
    d, w, dp = x[..., :10], x[..., 10:20], x[..., 20:30]
    inc, phase, _ = _physical(p)
    shift = d @ inc
    active = p["coupling"] if p["coupling"].size == inc.shape[1] else p["coupling"][p["coupling"] > 0]
    potential = np.sum(active *
                       (np.cos(phase) - np.cos(phase + shift) - np.sin(phase) * shift), axis=-1)
    # MechanicalROM.energy receives physical Pm, not the incremental p state.
    pm = dp + p.get("pm_star", np.zeros(10))
    increment = pm - p.get("pm_star", np.zeros(10))
    return .5 * np.sum(p["mass"] * w*w + increment*increment/p["gain"], axis=-1) + potential


def h3_gradient(z: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    p = _as_params(params); x = reduced_to_full(z)
    d, w, dp = x[..., :10], x[..., 10:20], x[..., 20:30]
    inc, phase, _ = _physical(p)
    active = p["coupling"] if p["coupling"].size == inc.shape[1] else p["coupling"][p["coupling"] > 0]
    force = ((np.sin(d @ inc + phase) - np.sin(phase)) * active) @ inc.T
    g = np.concatenate((force, p["mass"] * w, dp / p["gain"], np.zeros_like(x[..., 30:])), axis=-1)
    return g @ E33


def h3_hessian_origin(params: dict[str, np.ndarray]) -> np.ndarray:
    p = _as_params(params); _, _, k0 = _physical(p)
    full = np.zeros((33, 33), dtype=float)
    full[:10, :10] = k0
    full[10:20, 10:20] = np.diag(p["mass"])
    full[20:30, 20:30] = np.diag(1.0 / p["gain"])
    return E33.T @ full @ E33


def library_names() -> list[str]:
    out = [f"0.5*q{i}^2" for i in range(3)]
    out += [f"q{i}*q{j}" for i, j in ((0, 1), (0, 2), (1, 2))]
    out += [f"q{i}*omega{j}" for i in range(3) for j in range(10)]
    out += [f"q{i}*p{j}" for i in range(3) for j in range(10)]
    out += [f"q{i}*sin_edge{e}" for i in range(3) for e in range(NEDGE)]
    out += [f"q{i}*cos_edge{e}" for i in range(3) for e in range(NEDGE)]
    out += ["delta_K0_omega", "omega_p"]
    if len(out) != LIBRARY_SIZE:
        raise AssertionError(len(out))
    return out


def _parts(z: np.ndarray, params: dict[str, np.ndarray]):
    p = _as_params(params); a = np.asarray(z, dtype=float)
    if a.shape[-1] != N:
        raise ValueError("expected (...,32) coordinates")
    x = reduced_to_full(a)
    inc, phase, k0 = _physical(p)
    eta = x[..., :10] @ inc + phase
    return p, x, inc, phase, k0, eta, np.sin(eta)-np.sin(phase), np.cos(eta)-np.cos(phase)


def library_values(z: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Raw 224-column correction library in the preregistered order."""
    p, x, inc, phase, k0, eta, sine, cosine = _parts(z, params)
    q, w, dp = x[..., 30:], x[..., 10:20], x[..., 20:30]
    cols = []
    for i in range(3): cols.append(.5*q[..., i]**2)
    for i, j in ((0, 1), (0, 2), (1, 2)): cols.append(q[..., i]*q[..., j])
    for i in range(3):
        for j in range(10): cols.append(q[..., i]*w[..., j])
    for i in range(3):
        for j in range(10): cols.append(q[..., i]*dp[..., j])
    for i in range(3):
        for e in range(NEDGE): cols.append(q[..., i]*sine[..., e])
    for i in range(3):
        for e in range(NEDGE): cols.append(q[..., i]*cosine[..., e])
    cols.append(np.einsum("...i,ij,...j->...", x[..., :10], k0, w))
    cols.append(np.sum(w * dp, axis=-1))
    return np.stack(cols, axis=-1)


def library_gradients(z: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Exact gradients, shape ``(...,224,32)``."""
    p, x, inc, phase, k0, eta, sine, cosine = _parts(z, params)
    q, w, dp = x[..., 30:], x[..., 10:20], x[..., 20:30]
    shape = x.shape[:-1] + (LIBRARY_SIZE, N)
    g = np.zeros(shape, dtype=float); k = 0
    for i in range(3): g[..., k, 29+i] += q[..., i]; k += 1
    for i, j in ((0, 1), (0, 2), (1, 2)):
        g[..., k, 29+i] += q[..., j]; g[..., k, 29+j] += q[..., i]; k += 1
    for i in range(3):
        for j in range(10):
            g[..., k, 29+i] += w[..., j]; g[..., k, 9+j] += q[..., i]; k += 1
    for i in range(3):
        for j in range(10):
            g[..., k, 29+i] += dp[..., j]; g[..., k, 19+j] += q[..., i]; k += 1
    for i in range(3):
        for e in range(NEDGE):
            g[..., k, 29+i] += sine[..., e]
            g[..., k, :9] += q[..., i, None] * np.cos(eta[..., e])[..., None] * inc[:9, e]
            k += 1
    for i in range(3):
        for e in range(NEDGE):
            g[..., k, 29+i] += cosine[..., e]
            g[..., k, :9] += q[..., i, None] * -np.sin(eta[..., e])[..., None] * inc[:9, e]
            k += 1
    d, ww = x[..., :10], x[..., 10:20]
    g[..., k, :9] = np.einsum("ij,...j->...i", k0, ww)[..., :9]
    g[..., k, 9:19] = np.einsum("ij,...i->...j", k0, d)
    k += 1
    g[..., k, 9:19] = dp; g[..., k, 19:29] = w
    return g


def library_hessians_origin(params: dict[str, np.ndarray]) -> np.ndarray:
    """Exact Hessian of each raw library column at the origin."""
    p, _, inc, phase, k0, *_ = _parts(np.zeros((1, N)), params)
    hs = np.zeros((LIBRARY_SIZE, N, N), dtype=float); k = 0
    for i in range(3): hs[k, 29+i, 29+i] = 1.; k += 1
    for i, j in ((0, 1), (0, 2), (1, 2)):
        hs[k, 29+i, 29+j] = hs[k, 29+j, 29+i] = 1.; k += 1
    for i in range(3):
        for j in range(10):
            hs[k, 29+i, 9+j] = hs[k, 9+j, 29+i] = 1.; k += 1
    for i in range(3):
        for j in range(10):
            hs[k, 29+i, 19+j] = hs[k, 19+j, 29+i] = 1.; k += 1
    for trig, sign in (("sin", 1.), ("cos", -1.)):
        for i in range(3):
            for e in range(NEDGE):
                v = np.zeros(N); v[:9] = sign * (np.cos(phase[e]) if trig == "sin" else np.sin(phase[e])) * inc[:9, e]
                hs[k, 29+i] += v; hs[k, :, 29+i] += v
                k += 1
    hs[k, :9, 9:19] = E33[:10, :9].T @ k0 @ E33[10:20, 9:19]
    hs[k, 9:19, :9] = hs[k, :9, 9:19].T; k += 1
    hs[k, 9:19, 19:29] = np.eye(10); hs[k, 19:29, 9:19] = np.eye(10)
    return hs


def library_with_origin(z: np.ndarray, params: dict[str, np.ndarray]):
    return library_values(z, params), library_gradients(z, params), library_hessians_origin(params)


def normalized_library(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(phi, dtype=float)
    if a.ndim != 2 or a.shape[1] != LIBRARY_SIZE: raise ValueError("library matrix shape")
    scale = np.maximum(np.sqrt(np.mean(a*a, axis=0)), 1e-8)
    return a / scale, scale


def _w_matrix(h3h: np.ndarray) -> np.ndarray:
    return np.diag(np.r_[np.diag(h3h[:NPHYS, :NPHYS]), np.ones(NQ)])


def local_matrices(params: dict[str, np.ndarray], coefficients: np.ndarray | None = None):
    h3h = h3_hessian_origin(params); hs = library_hessians_origin(params)
    h = h3h if coefficients is None else h3h + np.tensordot(np.asarray(coefficients), hs, axes=(0, 0))
    w = _w_matrix(h3h); j = np.asarray(params["J32"], dtype=float)
    lyap = -(j.T @ h + h @ j)
    return h, lyap, w


def local_feasibility(params: dict[str, np.ndarray], coefficients: np.ndarray | None = None,
                      *, margin: float = 1e-6) -> dict[str, object]:
    """Check fixed local PSD/Lyapunov constraints without fitting or data."""
    h, lyap, w = local_matrices(params, coefficients)
    hn = np.linalg.eigvalsh(h)
    # W is diagonal positive in the intended physical contract; generalized
    # eigenvalues are used so reporting is invariant to coordinate scaling.
    inv = np.diag(1 / np.sqrt(np.diag(w)))
    hmin = float(np.linalg.eigvalsh(inv @ h @ inv).min())
    lmin = float(np.linalg.eigvalsh(inv @ lyap @ inv).min())
    return {"hessian_min_W": hmin, "lyapunov_min_W": lmin,
            "hessian_pass": bool(hmin >= margin), "lyapunov_pass": bool(lmin >= margin),
            "pass": bool(hmin >= margin and lmin >= margin),
            "raw_hessian_min": float(hn.min()), "raw_lyapunov_min": float(np.linalg.eigvalsh(lyap).min())}


def _dot_terms(z: np.ndarray, params: dict[str, np.ndarray], field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad = library_gradients(z, params)
    # (...,224,32) dot (...,32) -> (...,224)
    return np.einsum("...ki,...i->...k", grad, field), np.einsum("...i,...i->...", h3_gradient(z, params), field)


def fit_l1(z: np.ndarray, params: dict[str, np.ndarray], field: np.ndarray,
           href: np.ndarray, hscale: float, scales: np.ndarray,
           *, support: np.ndarray | None = None, l1_weight: float = .001):
    """Run exactly one Clarabel fit; no solver fallback or parameter sweep."""
    import cvxpy as cp
    z = np.asarray(z, float); field = np.asarray(field, float)
    phi = library_values(z, params); hg, h3dot = _dot_terms(z, params, field)
    h3h = h3_hessian_origin(params); hs = library_hessians_origin(params); w = _w_matrix(h3h)
    j = np.asarray(params["J32"], float); c = cp.Variable(LIBRARY_SIZE)
    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != (LIBRARY_SIZE,): raise ValueError("support shape")
        constraints = [c[~support] == 0]
    else: constraints = []
    H = h3h + sum(c[k] * hs[k] for k in range(LIBRARY_SIZE))
    invw = np.diag(1 / np.sqrt(np.diag(w)))
    Hn = invw @ H @ invw
    Ln = invw @ (-(j.T@H + H@j)) @ invw
    constraints += [Hn - 1e-6*np.eye(N) >> 0, Ln - 1e-6*np.eye(N) >> 0]
    values = h3_value(z, params)
    pred = values + phi @ c
    dot = h3dot + hg @ c
    wz = np.einsum("ij,nj,ni->n", w, z, z)
    standardized = cp.multiply(scales, c)
    objective = cp.sum_squares((pred - href) / hscale) / len(z)
    objective += l1_weight * cp.sum(cp.abs(standardized)) / hscale
    objective += 10 * cp.sum_squares(cp.pos((1e-6*wz - pred) / hscale)) / len(z)
    objective += 10 * cp.sum_squares(cp.pos((dot + 1e-6*wz) / hscale)) / len(z)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver="CLARABEL", max_iter=200, tol_gap_abs=1e-8,
                      tol_gap_rel=1e-8, tol_feas=1e-8, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"Clarabel sparse-storage solve failed: {exc}") from exc
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or c.value is None:
        raise RuntimeError(f"inadequate numerical solution: status={problem.status}")
    out = np.asarray(c.value, float)
    if not np.isfinite(out).all(): raise FloatingPointError("nonfinite Clarabel coefficients")
    return out, {"status": problem.status, "objective": float(problem.value),
                 "support_size": int(np.count_nonzero(np.abs(out) > 1e-12))}


def select_support(first: np.ndarray, scales: np.ndarray, limit: int = 18) -> np.ndarray:
    first = np.asarray(first, float); scales = np.asarray(scales, float)
    if first.shape != (LIBRARY_SIZE,) or scales.shape != (LIBRARY_SIZE,): raise ValueError("coefficient shape")
    score = np.abs(first * scales); order = np.lexsort((np.arange(LIBRARY_SIZE), -score))
    keep = np.zeros(LIBRARY_SIZE, dtype=bool); keep[:PQ_COUNT] = True
    chosen = [i for i in order if i >= PQ_COUNT][:limit]
    keep[chosen] = True
    return keep


def fit_two_stage(z: np.ndarray, params: dict[str, np.ndarray], field: np.ndarray,
                  p65: np.ndarray) -> dict[str, object]:
    phi = library_values(z, params); _, scales = normalized_library(phi)
    h3h = h3_hessian_origin(params)
    den = 2 * np.trace(np.asarray(p65)[:NPHYS, :NPHYS]);
    if abs(den) < 1e-14: raise ValueError("P65 physical trace is zero")
    a = np.trace(h3h[:NPHYS, :NPHYS]) / den
    href = a * np.einsum("ni,ij,nj->n", z, p65, z)
    hscale = max(float(np.median(href[href > 0])) if np.any(href > 0) else 0., 1e-6)
    first, first_meta = fit_l1(z, params, field, href, hscale, scales, l1_weight=.001)
    support = select_support(first, scales)
    second, second_meta = fit_l1(z, params, field, href, hscale, scales,
                                  support=support, l1_weight=0.)
    return {"first": first, "second": second, "support": support, "scales": scales,
            "href": href, "hscale": hscale, "href_a": a,
            "first_meta": first_meta, "second_meta": second_meta,
            "names": library_names()}


def solve_local_feasibility(params: dict[str, np.ndarray], *, return_matrices: bool = False) -> dict[str, object]:
    """Solve only the registered origin PSD/Lyapunov feasibility system."""
    import cvxpy as cp
    h3h = h3_hessian_origin(params); hs = library_hessians_origin(params)
    j = np.asarray(params["J32"], float); w = _w_matrix(h3h)
    invw = np.diag(1 / np.sqrt(np.diag(w))); c = cp.Variable(LIBRARY_SIZE)
    H = h3h + sum(c[k] * hs[k] for k in range(LIBRARY_SIZE))
    Hn = invw @ H @ invw; Ln = invw @ (-(j.T@H + H@j)) @ invw
    constraints = [Hn - 1e-6*np.eye(N) >> 0, Ln - 1e-6*np.eye(N) >> 0]
    problem = cp.Problem(cp.Minimize(0), constraints)
    try:
        problem.solve(solver="CLARABEL", max_iter=200, tol_gap_abs=1e-8,
                      tol_gap_rel=1e-8, tol_feas=1e-8, verbose=False)
    except Exception as exc:
        return {"status": "solver_error", "error": str(exc), "feasible": False}
    feasible = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and c.value is not None
    stats = problem.solver_stats
    result = {"status": str(problem.status), "feasible": bool(feasible),
              "iterations": None if stats is None else getattr(stats, "num_iters", None)}
    if feasible:
        coeff = np.asarray(c.value, float); result.update(local_feasibility(params, coeff, margin=.5e-6))
        result["support_size"] = int(np.count_nonzero(np.abs(coeff) > 1e-10))
        if return_matrices: result["coefficients"] = coeff
    elif return_matrices:
        result["coefficients"] = np.empty(0, dtype=float)
    if return_matrices and all(v.dual_value is not None for v in constraints):
        result["dual_matrices"] = np.stack([v.dual_value for v in constraints])
    return result


def load_local_sources(root: Path | str = Path(".")) -> dict[str, np.ndarray]:
    """Load only the registered S/J/P65, M3 and topology artifacts.

    This function intentionally names concrete development artifacts and does
    not inspect split manifests or enumerate any TEST payload.
    """
    root = Path(root)
    eq = root / "private_data/task2_coupled_equilibrium_0065/coupled_equilibrium.npz"
    m3 = root / "private_data/task2_m3_0020/model.npz"
    if not eq.exists() or not m3.exists(): raise FileNotFoundError("registered local source missing")
    network = root / "private_data/task2_electrical_ports_0048/physical_network.npz"
    S = root / "private_data/task2_long_0060_5s"
    evidence = eq.parent / "evidence.json"
    if not network.exists() or not (S / "model.pt").exists() or not (S / "normalization.npz").exists():
        raise FileNotFoundError("registered S/M3/network source missing")
    if evidence.exists():
        expected = json.loads(evidence.read_text(encoding="utf-8")).get("source_sha256", {})
        paths = {"model": S / "model.pt", "normalization": S / "normalization.npz",
                 "base": m3, "network": network}
        actual = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()}
        if any(expected.get(k) != v for k, v in actual.items()):
            raise AssertionError("registered S/M3/network source hash mismatch")
    else:
        raise FileNotFoundError("registered0065 provenance missing")
    with np.load(eq, allow_pickle=False) as s, np.load(m3, allow_pickle=False) as b:
        p = {"mass": s["param_mass"], "incidence": s["param_incidence"],
             "angle_incidence": s["param_angle_incidence"], "delta_star": s["param_delta_star"],
             "pm_star": s["param_pm_star"], "coupling": s["param_coupling"],
             "damping": s["param_damping"], "rate": s["param_rate"], "gain": s["param_gain"],
             "feature_scale": s["param_feature_scale"], "J32": s["J32"],
             "P65": s["lyapunov_P"], "Cq": s["param_Cq"], "xi": s["param_xi"],
             "support": s["param_support"], "alpha": s["param_alpha"],
             "alpha_support": s["param_alpha_support"]}
        for key in ("mass", "incidence", "delta_star", "pm_star", "coupling", "damping", "rate", "gain"):
            if key in b and not np.allclose(p[key], b[key], atol=1e-12, rtol=1e-12):
                raise AssertionError(f"M3/source mismatch: {key}")
    return p


def ray_intersection(direction: np.ndarray, scales: np.ndarray, angle_matrix: np.ndarray,
                     angle_offset: np.ndarray | None = None) -> float:
    d = np.asarray(direction, float); s = np.asarray(scales, float); A = np.asarray(angle_matrix, float)
    if d.shape != (N,) or s.shape != (N,) or A.shape[1] != N: raise ValueError("ray shapes")
    bounds = [float(v / abs(u)) for v, u in zip(s, d) if abs(u) > 0]
    # For |offset + t*A@direction| <= pi, intersect the positive ray with
    # both affine half-spaces.  This retains the restored generator-angle
    # offset rather than incorrectly treating the equilibrium as zero.
    off = np.zeros(A.shape[0]) if angle_offset is None else np.asarray(angle_offset, float)
    if off.shape != (A.shape[0],): raise ValueError("angle offset shape")
    ad = A @ d
    for b, u in zip(off, ad):
        if abs(u) < 1e-15:
            if abs(b) > np.pi + 1e-12: return 0.
        elif u > 0: bounds.append(float((np.pi - b) / u))
        else: bounds.append(float((-np.pi - b) / u))
    return min(bounds, default=0.)


def domain_probes(scales: np.ndarray, angle_matrix: np.ndarray, *, n_interior=2048,
                  n_boundary=1024, seed=DEFAULT_SEED, angle_offset=None, boundary_seed=None):
    """Generate registered interior and exact boundary ray/polytope probes."""
    s = np.asarray(scales, float)
    def draw(rng, count, radii):
        rows = []
        while len(rows) < count:
            direction = s * rng.uniform(-1., 1., size=N)
            t = ray_intersection(direction, s, angle_matrix, angle_offset)
            if t <= 0: continue
            rows.append(t * direction * (rng.uniform(0., .95) if radii else 1.))
        return np.asarray(rows)
    rng = np.random.default_rng(seed)
    # Boundary directions are fresh draws from the same registered seed
    # stream; the separate 6801 seed is reserved for independent checks.
    return draw(rng, n_interior, True), draw(rng, n_boundary, False)


def coi(x: np.ndarray, mass: np.ndarray) -> np.ndarray:
    a = np.asarray(x, float); m = np.asarray(mass, float)
    return a - np.sum(a*m, axis=-1, keepdims=True)/np.sum(m)


def cumulative_prefix_ok(coarse: np.ndarray, fine: np.ndarray, mass: np.ndarray,
                         *, tolerance=1e-3, los_coarse=np.inf, los_fine=np.inf,
                         substep_finite: np.ndarray | None = None) -> np.ndarray:
    """Return the strict cumulative .01-prefix mask, retaining every failure."""
    c, f = np.asarray(coarse, float), np.asarray(fine, float)
    if c.shape != f.shape or c.ndim not in (2, 3) or c.shape[-1] < 20: raise ValueError("rollout shape")
    single = c.ndim == 2
    if single: c, f = c[None], f[None]
    ne, nt = c.shape[:2]; out = np.zeros((ne, nt), dtype=bool)
    m = np.asarray(mass, float)
    sf = None if substep_finite is None else np.asarray(substep_finite, dtype=bool)
    if sf is not None:
        if sf.ndim == 2: sf = sf[None]
        if sf.shape[:2] != (ne, nt): raise ValueError("substep finite shape")
    # Registered physical normalization used by the frozen rollout audit.
    ds, ws = .47675314588683515, 1.0033667249602924
    lc = np.broadcast_to(np.asarray(los_coarse, float), (ne,))
    lf = np.broadcast_to(np.asarray(los_fine, float), (ne,))
    for eidx in range(ne):
        good = True
        for k in range(nt):
            if not np.isfinite(c[eidx, k]).all() or not np.isfinite(f[eidx, k]).all(): good = False
            if sf is not None and not bool(sf[eidx, k].all()): good = False
            if k*.01 >= min(lc[eidx], lf[eidx]): good = False
            err = c[eidx, :k+1, :20] - f[eidx, :k+1, :20]
            de, we = coi(err[..., :10], m), coi(err[..., 10:20], m)
            rmse = np.sqrt(.5*np.mean((de/ds)**2 + (we/ws)**2))
            out[eidx, k] = good and bool(rmse <= tolerance)
    return out[0] if single else out


def synthetic_params(seed: int = 68) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed); nedge = NEDGE
    inc = np.zeros((10, 45));
    for e in range(45):
        i, j = e % 10, (e + 1) % 10; inc[i, e] = 1.; inc[j, e] = -1.
    coupling = np.r_[np.linspace(.2, .8, nedge), np.zeros(45-nedge)]
    h = inc[:, :nedge]
    p = {"mass": np.linspace(1., 2., 10), "incidence": inc, "angle_incidence": h,
         "delta_star": rng.normal(size=10)*.02, "pm_star": np.ones(10),
         "coupling": coupling, "damping": np.ones(10)*.3, "rate": np.ones(10)*.4,
         "gain": np.ones(10)*.8, "feature_scale": np.ones(20+2*nedge),
         "J32": -np.eye(32), "P65": np.eye(32),
         "Cq": rng.normal(size=(10, 3))*.02, "xi": np.zeros((76, 3)),
         "support": np.ones((76, 3))}
    return p


def synthetic_smoke() -> dict[str, object]:
    p = synthetic_params(); rng = np.random.default_rng(68); z = rng.normal(size=(32, N))*.03
    phi, grad, hs = library_with_origin(z, p)
    # Derivative consistency against centered differences for both H3 and a
    # nontrivial correction vector; no fit and no private data are touched.
    c = rng.normal(size=LIBRARY_SIZE)*.01; h = h3_value(z, p) + phi@c
    g = h3_gradient(z, p) + np.einsum("nki,k->ni", grad, c)
    eps = 2e-6; fd = (h3_value(z+eps*np.eye(N)[None,0],p)+library_values(z+eps*np.eye(N)[None,0],p)@c -
                       h3_value(z-eps*np.eye(N)[None,0],p)-library_values(z-eps*np.eye(N)[None,0],p)@c)/(2*eps)
    assert np.max(np.abs(fd-g[:,0])) < 1e-6
    local = local_feasibility(p, np.zeros(LIBRARY_SIZE), margin=-np.inf)
    return {"library_shape": list(phi.shape), "hessian_shape": list(hs.shape), "derivative_max": float(np.max(np.abs(fd-g[:,0]))), "local": local}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--local-feasibility", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if not (args.synthetic or args.local_feasibility):
        raise SystemExit("refusing implicit real-data fit; use --synthetic or an explicit development runner")
    if args.local_feasibility:
        params = load_local_sources(); result = solve_local_feasibility(params, return_matrices=True)
        coeff = np.asarray(result.pop("coefficients", np.empty(0)), float)
        dual = np.asarray(result.pop("dual_matrices", np.empty(0)), float)
        if args.output is not None:
            if args.output.exists(): raise FileExistsError(args.output)
            args.output.mkdir(parents=True)
            h3h = h3_hessian_origin(params); hs = library_hessians_origin(params)
            if coeff.size == LIBRARY_SIZE:
                h = h3h + np.tensordot(coeff, hs, axes=(0, 0))
            else:
                h = h3h
            j = np.asarray(params["J32"], float); w = _w_matrix(h3h); lyap = -(j.T@h+h@j)
            np.savez_compressed(args.output / "local_matrices.npz", coefficients=coeff, H=h,
                                lyapunov=lyap, W=w, h3_hessian=h3h, library_hessians=hs,
                                J32=j, dual_matrices=dual)
            evidence = {**result, "source_sha256": {"model": hashlib.sha256((Path("private_data/task2_long_0060_5s")/"model.pt").read_bytes()).hexdigest(),
                       "normalization": hashlib.sha256((Path("private_data/task2_long_0060_5s")/"normalization.npz").read_bytes()).hexdigest(),
                       "base": hashlib.sha256(Path("private_data/task2_m3_0020/model.npz").read_bytes()).hexdigest(),
                       "network": hashlib.sha256(Path("private_data/task2_electrical_ports_0048/physical_network.npz").read_bytes()).hexdigest()},
                        "matrix_role": "candidate" if coeff.size else "fixed_H3_reference_no_solution",
                        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "local_source_sha256": hashlib.sha256(Path("private_data/task2_coupled_equilibrium_0065/coupled_equilibrium.npz").read_bytes()).hexdigest(),
                        "test_read": False, "solver": "CLARABEL", "max_iter": 200,
                        "tolerances": {"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8, "tol_feas": 1e-8}}
            (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    else:
        result = synthetic_smoke()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__": main()
