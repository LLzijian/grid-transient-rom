"""Pure L2/L3 analysis of the final GFTD functional."""
from __future__ import annotations

from fractions import Fraction as F
from math import factorial
import hashlib
import json

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from kiki.evaluation.modal import DmdMode, match_dmd_modes, modal_assurance_criterion
from kiki.physics.kuramoto_functional import BOUNDED_MODULATION_KERNEL, EXP_MODULATION_KERNEL


def fraction(value):
    if isinstance(value, (F, str, int)):
        return F(value)
    return F.from_float(float(value))


def _qarray(values):
    return [fraction(v) for v in values]


def _fail(reason, **details):
    return dict(status="unknown", reason=reason, **details)


def _l2_fail(reason, **details):
    return dict(status="not_certified", reason=reason,
                true_initialization_set_coverage="unknown", epsilon=None, **details)


def _functional_hash(f):
    from dataclasses import fields
    payload = {}
    for field in fields(f):
        value = getattr(f, field.name)
        if field.name == "gates":
            value = {k: np.asarray(getattr(value, k)).tolist() for k in ("s_edge", "s_mod", "s_mem", "s_coi")}
        elif value is not None:
            value = np.asarray(value).tolist()
        payload[field.name] = value
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _parameters(f, delta_star):
    n = f.n_nodes
    theta = _qarray(delta_star)
    if len(theta) != n or n < 2:
        raise ValueError("delta_star dimension or node count")
    mean = sum(theta) / n
    theta = [t - mean for t in theta]
    K = [fraction(k) * fraction(s) for k, s in zip(f.k_edge, f.gates.s_edge, strict=True)]
    beta = [k * fraction(s) for k, s in zip(K, f.gates.s_mod, strict=True)]
    gamma = _qarray(f.gamma_amp)
    eta = _qarray(f.eta_edge)
    phi = [theta[i]-theta[j] for i, j in f.edges]
    return dict(n=n, theta=theta, K=K, beta=beta, gamma=gamma, eta=eta,
                delta=[abs(a-b) for a, b in zip(phi, eta, strict=True)],
                phi=phi, Gamma=[abs(gamma[i])+abs(gamma[j]) for i, j in f.edges])


def _bounded_kernel(functional):
    kernel = getattr(functional, "modulation_kernel", None)
    if kernel not in (EXP_MODULATION_KERNEL, BOUNDED_MODULATION_KERNEL):
        raise ValueError("unknown amplitude modulation kernel")
    return kernel == BOUNDED_MODULATION_KERNEL


def l2_certificate(functional, delta_star):
    """Exactly 13 rational candidates; no data-dependent radius selection."""
    return dict(_l2_certificate(functional, delta_star), taylor_calls=0)


def _l2_certificate(functional, delta_star):
    f = functional
    try:
        bounded = _bounded_kernel(f)
    except ValueError as exc:
        return _l2_fail("unknown_kernel", detail=str(exc))
    try:
        p = _parameters(f, delta_star)
        n, K, beta = p["n"], p["K"], p["beta"]
        if any(not 0 <= b <= k for k, b in zip(K, beta, strict=True)):
            return _l2_fail("edge_parameter_domain")
        if not bounded and any(beta):
            return _l2_fail("static_exp_requires_zero_beta")
        positive = ("mass", "storage_p", "lambda_amp", "lambda_mem", "damping", "rate_p", "tau_amp", "tau_mem_sync")
        arrays = {name: _qarray(getattr(f, name)) for name in positive}
        if any(len(v) != n or min(v) <= 0 for v in arrays.values()):
            return _l2_fail("strict_storage_or_dissipation_missing")
        active = [e for e, k in enumerate(K) if k > 0]
        modulated = [e for e, b in enumerate(beta) if b > 0]
        connected = {0}
        for _ in range(n):
            for e in active:
                i, j = f.edges[e]
                if i in connected or j in connected:
                    connected.update((i, j))
        if len(connected) != n:
            return _l2_fail("active_graph_disconnected")
        # Clipped sigmoid >= 1/(1+exp(40)) > 1/(1+3**40), since e<3.
        gate_lower = F(1, 1 + 3**40)
        dissipation = {}
        for port, tau in (("amp", "tau_amp"), ("mem", "tau_mem_sync")):
            lower = gate_lower if getattr(f, "gate_weight_"+port) is not None else F(1)
            dissipation[port] = [str(lower / t) for t in arrays[tau]]
        A0 = max(max(abs(p["phi"][e]), abs(p["eta"][e])) for e in active)
        Gmax = max(p["Gamma"][e] for e in active)
        Gbeta = max((p["Gamma"][e] for e in modulated), default=F(0))
        cap = min(F(1), (F(7, 5)-A0)/3)
        if Gmax:
            cap = min(cap, 1/(2*Gmax))
        if cap <= 0:
            return _l2_fail("no_positive_radius_cap")
        ell = 4*min(K[e] for e in active)/(n*(n-1))
        g = F(3, 2)*sum(K[e]*p["delta"][e] for e in active)
        g += F(1, 2)*sum(beta[e]*p["Gamma"][e]*p["delta"][e]**2 for e in modulated)
        ma = min(arrays["lambda_amp"])
        candidates = []
        for j in range(13):
            R = cap / 2**j
            alpha, h = A0+F(3, 2)*R, R*(Gbeta if bounded else Gmax)
            row = dict(index=j, R=str(R), status="not_certified")
            if bounded and h > F(1, 4):
                row.update(reason="bounded_modulation_domain", b=str(h))
            elif not (alpha < F(7, 5) and h < 1):
                row["reason"] = "angle_or_exponential_domain"
            else:
                cos_lower = 1-alpha**2/2
                mt = cos_lower*(1-h)*ell
                C = sum(beta[e]*p["Gamma"][e]*(2*R+F(3, 2)*p["delta"][e]) for e in modulated)/(1-h)
                det = mt*ma-C*C
                if cos_lower <= 0 or det <= 0:
                    row["reason"] = "curvature_lower_bound_nonpositive"
                else:
                    mu = min(det/(mt+ma), min(arrays["mass"]), min(arrays["storage_p"]), min(arrays["lambda_mem"]))
                    if not g < mu*R/4:
                        row["reason"] = "center_gradient_gate"
                    else:
                        rho = mu*R*R/4
                        # Row sums bound the positive 2x2 Hessian comparison matrix.
                        A = 2*sum(K)/(1-h)
                        amp_curvature = sum(
                            beta[e]*p["Gamma"][e]**2*(p["delta"][e]+F(3, 2)*R)**2
                            for e in modulated
                        )
                        D = max(arrays["lambda_amp"]) + (
                            (1+2*h if bounded else 1)*amp_curvature/(2*(1-h))
                        )
                        L = max(A+C, D+C, max(arrays["mass"]), max(arrays["storage_p"]), max(arrays["lambda_mem"]))
                        row.update(status="certified", mu=str(mu), rho=str(rho), g=str(g),
                                   mixed_upper=str(C), hessian_upper=str(L))
                        row["b" if bounded else "h"] = str(h)
            candidates.append(row)
        eligible = [r for r in candidates if r["status"] == "certified"]
        if not eligible:
            return _l2_fail("no_candidate_passed", candidates=candidates)
        selected = max(eligible, key=lambda r: (F(r["rho"]), -F(r["R"])))
        return dict(status="certified", selected=selected, candidates=candidates,
                    functional_sha256=_functional_hash(f), modulation_kernel=f.modulation_kernel,
                    center_theta=[str(t) for t in p["theta"]],
                    exact_K=[str(k) for k in K], exact_beta=[str(b) for b in beta],
                    dissipation_lower=dissipation,
                    exact_operator_products={
                        "J_Kr": [str(fraction(s)*fraction(j)*fraction(kr)) for s, j, kr in zip(f.gates.s_mem, f.j_omega_mem, f.lambda_mem, strict=True)],
                        "Kr_over_tau_mem": [str(fraction(kr)/fraction(t)) for kr, t in zip(f.lambda_mem, f.tau_mem_sync, strict=True)],
                        "gain_Sp": [str(fraction(gain)*fraction(sp)) for gain, sp in zip(f.gain_p, f.storage_p, strict=True)]},
                    actual_point_count=0,
                    dimension=5*n-1, scope="ROM attraction to equilibrium phase orbit",
                    true_initialization_set_coverage="unknown", epsilon=None)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return _l2_fail("invalid_parameters", detail=str(exc))


def _add(a, b):
    return (a[0]+b[0], a[1]+b[1])


def _scale(a, s):
    return (a[0]*s, a[1]*s) if s >= 0 else (a[1]*s, a[0]*s)


def _mul(a, b):
    products = [x*y for x in a for y in b]
    return min(products), max(products)


def taylor_interval(x, kind):
    """Fixed protocol degrees and Lagrange/geometric remainders, all rational."""
    x = fraction(x)
    if kind == "exp":
        if abs(x) >= 1:
            raise ValueError("exp interval requires |x|<1")
        value = sum(x**k/factorial(k) for k in range(19))
        error = abs(x)**19/(factorial(19)*(1-abs(x)))
    elif kind in ("sin", "cos"):
        offset = int(kind == "sin")
        value = sum((-1)**k*x**(2*k+offset)/factorial(2*k+offset) for k in range(19))
        degree = 38 if offset else 37
        error = abs(x)**degree/factorial(degree)
    else:
        raise ValueError("unknown Taylor function")
    return value-error, value+error


class TaylorBudget:
    def __init__(self, limit):
        self.limit = limit
        self.calls = 0

    def interval(self, x, kind):
        if self.calls >= self.limit:
            raise ValueError("Taylor call budget exhausted")
        self.calls += 1
        return taylor_interval(x, kind)


def outward_interval(interval):
    lo, hi = map(fraction, interval)
    if lo > hi:
        raise ValueError("interval endpoints out of order")
    grid = 2**128
    return F(lo*grid//1, grid), F(-((-hi*grid)//1), grid)


def _tanh_interval(exp_interval):
    lo, hi = outward_interval(exp_interval)
    if lo <= 0:
        raise ValueError("exp(2h) interval must be strictly positive")
    lo, hi = outward_interval(((lo-1)/(lo+1), (hi-1)/(hi+1)))
    if not -1 < lo <= hi < 1:
        raise ValueError("tanh interval outside outer exponential domain")
    return lo, hi


def _bounded_kernel_interval(h, interval):
    h = fraction(h)
    if abs(h) > F(1, 4):
        raise ValueError("bounded kernel interval requires |h|<=1/4")
    lo, hi = _tanh_interval(interval(2*h, "exp"))
    left = outward_interval(interval(lo, "exp"))
    right = outward_interval(interval(hi, "exp"))
    return outward_interval((left[0], right[1]))


def exact_quotient_norm_squared(state, center_theta):
    values, theta = _qarray(state), _qarray(center_theta)
    n = len(theta)
    if len(values) != 5*n:
        raise ValueError("state dimension")
    d = [values[i]-theta[i] for i in range(n)]
    return sum(x*x for x in d)-sum(d)**2/n+sum(x*x for x in values[n:])


def l2_point(certificate, functional, state, *, taylor_budget=None):
    budget = TaylorBudget(270) if taylor_budget is None else taylor_budget
    before = budget.calls

    def interval(x, kind):
        if budget.calls-before >= 270:
            raise ValueError("point Taylor call budget exhausted")
        return budget.interval(x, kind)

    result = _l2_point(certificate, functional, state, interval)
    return dict(result, taylor_calls=budget.calls-before)


def _l2_point(certificate, functional, state, interval):
    """One exact envelope for one qualified returned point; epsilon stays unknown."""
    f = functional
    try:
        bounded = _bounded_kernel(f)
    except ValueError as exc:
        return _l2_fail("unknown_kernel", detail=str(exc))
    if certificate.get("status") != "certified":
        return _l2_fail("model_not_certified")
    try:
        if certificate.get("modulation_kernel") != f.modulation_kernel:
            return _l2_fail("certificate_kernel_mismatch")
        if certificate.get("functional_sha256") != _functional_hash(f):
            return _l2_fail("functional_changed_after_certificate")
        n = f.n_nodes
        z = _qarray(state)
        row = certificate["selected"]
        R, mu, rho, g = (F(row[k]) for k in ("R", "mu", "rho", "g"))
        norm2 = exact_quotient_norm_squared(z, certificate["center_theta"])
        if norm2 >= R*R:
            return _l2_fail("point_outside_strict_ball", norm_squared=str(norm2))
        K, beta = _qarray(certificate["exact_K"]), _qarray(certificate["exact_beta"])
        theta, omega, p, a, r = [z[i*n:(i+1)*n] for i in range(5)]
        energy = F(0)
        for block, name in ((omega, "mass"), (p, "storage_p"), (a, "lambda_amp"), (r, "lambda_mem")):
            energy += sum(fraction(c)*v*v/2 for c, v in zip(getattr(f, name), block, strict=True))
        H = (energy, energy)
        quantize = outward_interval if bounded else lambda bounds: bounds
        gamma, eta = _qarray(f.gamma_amp), _qarray(f.eta_edge)
        for e, (i, j) in enumerate(f.edges):
            if K[e] == 0:
                continue
            phi = theta[i]-theta[j]
            psi = _add(quantize(interval(eta[e], "cos")), _scale(quantize(interval(phi, "cos")), -1))
            psi = _add(psi, _scale(quantize(interval(eta[e], "sin")), -(phi-eta[e])))
            if bounded and beta[e] == 0:
                coefficient = (K[e], K[e])
            else:
                h = gamma[i]*a[i]+gamma[j]*a[j]
                kernel = _bounded_kernel_interval(h, interval) if bounded else interval(h, "exp")
                coefficient = _add((K[e]-beta[e], K[e]-beta[e]), _scale(kernel, beta[e]))
            H = quantize(_add(H, _mul(coefficient, psi)))
        upper = H[1]+g*g/(2*mu)
        return dict(status="certified" if upper < rho else "not_certified",
                    reason=None if upper < rho else "energy_margin_unresolved",
                    energy_interval=[str(v) for v in H], V_upper=str(upper),
                    margin_lower=str(1-upper/rho), norm_squared=str(norm2),
                    true_initialization_set_coverage="unknown", epsilon=None)
    except (KeyError, ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        return _l2_fail("point_envelope_failed", detail=str(exc))


def graph_correspondence(kron, functional, delta_star):
    if kron.get("status") != "qualified":
        return _fail("Kron_unqualified")
    try:
        _bounded_kernel(functional)
    except ValueError as exc:
        return _fail("unknown_kernel", detail=str(exc))
    Y = np.array(kron["Yred_real"])+1j*np.array(kron["Yred_imag"])
    k = functional.effective_edge_couplings(np.zeros(functional.n_nodes))
    y = np.array([abs(Y[i, j]) for i, j in functional.edges])
    correlation = None if np.ptp(k) == 0 or np.ptp(y) == 0 else float(spearmanr(k, y).statistic)
    p = _parameters(functional, delta_star)
    edge_rows = [dict(nodes=[i, j], exact_K=str(p["K"][e]), exact_beta=str(p["beta"][e]),
                      raw_supported=bool(p["K"][e] > 0),
                      reference_angle_stiffness=float(k[e]*np.cos(float(p["phi"][e]))),
                      admittance_magnitude=float(y[e]))
                 for e, (i, j) in enumerate(functional.edges)]
    return dict(status="reported", spearman=correlation, edges=edge_rows,
                null_graph=bool(all(k == 0 for k in p["K"])),
                admittance_weights=(y/y.sum()).tolist() if y.sum() else None,
                effective_weights=(k/k.sum()).tolist() if k.sum() else None,
                scope="terminal admittance versus effective angle coupling; report-only")


def _pole_distance(a, b):
    return abs(a[:, None]-b[None, :])/np.maximum(1., abs(b[None, :]))


def _pair_error(a, b):
    if len(a) != len(b):
        return float("inf")
    if not len(a):
        return 0.
    distance = _pole_distance(a, b)
    rows, cols = linear_sum_assignment(distance)
    return float(np.max(distance[rows, cols]))


def _mode_record(pole, shape, index):
    norm = float(np.linalg.norm(shape))
    if norm <= np.finfo(float).eps or not np.isfinite(shape).all():
        return None
    vector = shape/norm
    return dict(index=int(index), real=float(pole.real), imag=float(pole.imag),
                frequency_hz=float(abs(pole.imag)/(2*np.pi)),
                damping_ratio=float(-pole.real/abs(pole)), decay_rate=float(pole.real),
                strength=norm, shape_real=vector.real.tolist(), shape_imag=vector.imag.tolist())


def quotient_basis(n):
    """Fixed Helmert basis; floating basis is only for numerical L3, never L2."""
    Q = np.zeros((n, n-1))
    for j in range(n-1):
        Q[:j+1, j] = 1/np.sqrt((j+1)*(j+2))
        Q[j+1, j] = -(j+1)/np.sqrt((j+1)*(j+2))
    if np.max(abs(Q.T@Q-np.eye(n-1))) > 1e-12 or np.max(abs(Q.sum(axis=0))) > 1e-12:
        raise ValueError("quotient basis qualification")
    return Q


def _central_jacobian(function, x, factor):
    columns = []
    for j in range(len(x)):
        h = factor*max(1., abs(x[j]))
        d = np.zeros_like(x)
        d[j] = h
        columns.append((function(x+d)-function(x-d))/(2*h))
    J = np.column_stack(columns)
    if not np.isfinite(J).all():
        raise ValueError("nonfinite finite-difference Jacobian")
    return J


def _rom_candidates(J, observation):
    values, vectors = np.linalg.eig(J)
    residual = np.linalg.norm(J@vectors-vectors*values, axis=0)/(
        (np.linalg.norm(J)+abs(values))*np.linalg.norm(vectors, axis=0))
    modes, excluded = [], []
    for j, pole in enumerate(values):
        frequency = abs(pole.imag)/(2*np.pi)
        reason = None
        if pole.imag <= 1e-10:
            reason = "nonpositive_imaginary"
        elif frequency < .01 or frequency > 10:
            reason = "outside_fixed_frequency_band"
        elif abs(pole) <= np.finfo(float).eps:
            reason = "zero_pole"
        record = None if reason else _mode_record(pole, observation@vectors[:, j], j)
        if reason is None and record is None:
            reason = "unobservable_or_nonfinite_omega"
        if reason:
            excluded.append(dict(index=j, real=float(pole.real), imag=float(pole.imag), reason=reason))
        else:
            modes.append(record)
    return modes, excluded, float(np.max(residual)) if np.isfinite(residual).all() else float("inf")


def rom_modes(functional, delta_star, certificate=None):
    """Bounded equilibrium search and two 49D grids; only common theta removed."""
    f, n = functional, functional.n_nodes
    counters = dict(newton_iterations=0, gradient_calls=0, rhs_calls=0, eig_count=0)
    try:
        _bounded_kernel(f)
        Q = quotient_basis(n)
        T = block_diag(Q, np.eye(n), np.eye(n), np.eye(n), np.eye(n))
        theta = np.asarray(delta_star, float)
        if theta.shape != (n,) or not np.isfinite(theta).all():
            return _fail("invalid_equilibrium_center", counters=counters)
        theta = theta-theta.mean()
        center = np.r_[theta, np.zeros(4*n)]
        y = np.zeros(2*n-1)
        R = float(F(certificate["selected"]["R"])) if certificate and certificate.get("status") == "certified" else 1.
        def state(v):
            z = center.copy()
            z[:n] += Q@v[:n-1]
            z[3*n:4*n] = v[n-1:]
            return z
        def gradient(v):
            counters["gradient_calls"] += 1
            gt, _, _, ga, _ = f.gradients(state(v))
            return np.r_[Q.T@gt, ga]
        def rhs(z):
            counters["rhs_calls"] += 1
            return f.rhs(z)
        for _ in range(20):
            grad = gradient(y)
            if np.max(abs(rhs(state(y)))) <= 1e-10:
                break
            counters["newton_iterations"] += 1
            J = _central_jacobian(gradient, y, 1e-6)
            direction = np.linalg.solve(J, -grad)
            accepted = False
            for j in range(10):
                proposal = y+direction/2**j
                if np.linalg.norm(proposal) > R:
                    continue
                if certificate and certificate.get("status") == "certified" and exact_quotient_norm_squared(state(proposal), certificate["center_theta"]) >= F(certificate["selected"]["R"])**2:
                    continue
                if np.linalg.norm(gradient(proposal)) < np.linalg.norm(grad):
                    y, accepted = proposal, True
                    break
            if not accepted:
                return _fail("Newton_no_descent", counters=counters)
        equilibrium = state(y)
        residual = float(np.max(abs(rhs(equilibrium))))
        if not np.isfinite(residual) or residual > 1e-10:
            return _fail("ROM_equilibrium_unqualified", equilibrium_residual=residual, counters=counters)
        observation = np.zeros((n, 5*n))
        observation[:, n:2*n] = np.diag(np.sqrt(f.mass))
        reduced_rhs = lambda v: T.T@rhs(equilibrium+T@v)
        # Steps are in fixed reduced state coordinates at the equilibrium.
        coordinate = T.T@equilibrium
        def grid(factor):
            columns = []
            for j in range(5*n-1):
                h = factor*max(1., abs(coordinate[j]))
                d = np.zeros(5*n-1)
                d[j] = h
                columns.append((reduced_rhs(d)-reduced_rhs(-d))/(2*h))
            return np.column_stack(columns)
        main, fine = grid(1e-6), grid(5e-7)
        difference = float(np.linalg.norm(main-fine)/max(1., np.linalg.norm(fine)))
        if not np.isfinite(main).all() or not np.isfinite(fine).all() or difference >= 1e-6:
            return _fail("ROM_Jacobian_grid_unresolved", jacobian_difference=difference, counters=counters)
        counters["eig_count"] += 1
        modes_main, excluded_main, error_main = _rom_candidates(main, observation@T)
        counters["eig_count"] += 1
        modes_fine, excluded_fine, error_fine = _rom_candidates(fine, observation@T)
        poles_main = np.array([complex(r["real"], r["imag"]) for r in modes_main])
        poles_fine = np.array([complex(r["real"], r["imag"]) for r in modes_fine])
        pole_error = _pair_error(poles_main, poles_fine)
        all_main = modes_main+excluded_main
        all_fine = modes_fine+excluded_fine
        dm = np.array([complex(r["real"], r["imag"]) for r in all_main])
        df = np.array([complex(r["real"], r["imag"]) for r in all_fine])
        rr, cc = linear_sum_assignment(_pole_distance(dm, df))
        boundary_changed = any((i < len(modes_main)) != (j < len(modes_fine)) for i, j in zip(rr, cc, strict=True))
        if boundary_changed:
            return _fail("ROM_candidate_boundary_membership_changed", counters=counters)
        if max(error_main, error_fine) >= 1e-8 or pole_error >= 1e-6:
            return _fail("ROM_modes_grid_unresolved", pole_difference=pole_error, counters=counters)
        return dict(status="qualified", modes=modes_fine, excluded=excluded_fine,
                    main_excluded=excluded_main, equilibrium=equilibrium.tolist(),
                    equilibrium_a=equilibrium[3*n:4*n].tolist(),
                    equilibrium_residual=residual, jacobian_difference=difference,
                    pole_difference=pole_error, eigen_residuals=[error_main, error_fine],
                    dimension=5*n-1, retained_omega=n, counters=counters)
    except (ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return _fail("ROM_modal_analysis_failed", detail=str(exc), counters=counters)


def match_modes(fom, rom):
    if fom.get("status") != "qualified" or rom.get("status") != "qualified":
        return _fail("unqualified_modal_input")
    keys = ("frequency_hz", "damping_ratio", "decay_rate", "strength", "shape_real", "shape_imag")
    ref = [DmdMode(**{k: row[k] for k in keys}) for row in fom["modes"]]
    cand = [DmdMode(**{k: row[k] for k in keys}) for row in rom["modes"]]
    matches = match_dmd_modes(ref, cand, max_relative_frequency_error=float("inf"))
    # Retain identities from the same full-cost assignment used by the shared helper.
    if ref and cand:
        cost = np.array([[abs(c.frequency_hz-r.frequency_hz)/max(r.frequency_hz, np.finfo(float).eps)
                          +1-modal_assurance_criterion(np.array(r.shape_real)+1j*np.array(r.shape_imag),
                                                       np.array(c.shape_real)+1j*np.array(c.shape_imag))
                          for c in cand] for r in ref])
        rows, cols = linear_sum_assignment(cost)
    else:
        rows, cols = [], []
    for match, i, j in zip(matches, rows, cols, strict=True):
        match.update(reference_index=int(i), candidate_index=int(j))
    return dict(status="reported", matches=matches,
                unmatched_reference=[i for i in range(len(ref)) if i not in rows],
                unmatched_candidate=[i for i in range(len(cand)) if i not in cols],
                scope="all assigned pairs, report-only; no MAC or frequency-error selection")
