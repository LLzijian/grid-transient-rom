"""N-node numerical primitives for conditional GFTD discovery and initialization.

No structure selection, parameter search, optimization budget or truth access lives here.
The weak equation is A_static*u + A_exp*v + Z*(rho/.1) = b,
with k=u+v, beta=v, and rho the physical memory at the history start.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

HISTORY_STEPS = (.0025, .00125)
RANK_THRESHOLDS = (1e-8, 1e-10, 1e-12)
STATE_SCALE = .1


def _array(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _nodes(n_nodes):
    if int(n_nodes) != n_nodes or n_nodes < 2:
        raise ValueError("at least two nodes required")
    return int(n_nodes)


def validate_observation(packet, n_nodes):
    """Copy only legal observations; time is relative to the final endpoint."""
    n = _nodes(n_nodes)
    allowed = {"theta_history", "omega_history", "history_times", "p_end", "sigma"}
    if set(packet) != allowed:
        raise ValueError("only theta/omega history, times, endpoint p and known sigma are allowed")
    result = {name: _array(value, name).copy() for name, value in packet.items()}
    if (result["theta_history"].shape != (11, n) or result["omega_history"].shape != (11, n)
            or result["p_end"].shape != (n,)):
        raise ValueError("exactly eleven node-aligned observations required")
    times = result["history_times"]
    if times.shape != (11,) or not np.allclose(times, np.linspace(-.1, 0., 11), atol=1e-12, rtol=0):
        raise ValueError("history_times must cover -.1 through 0 seconds at .01 seconds")
    if result["sigma"].shape != () or result["sigma"] < 0:
        raise ValueError("known sigma must be a nonnegative scalar")
    return result


def boole_weights(t):
    t = _array(t, "quadrature times")
    if t.ndim != 1 or len(t) < 5 or (len(t)-1) % 4:
        raise ValueError("Boole requires a uniform grid with 4k intervals")
    steps = np.diff(t)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], atol=1e-14, rtol=1e-12):
        raise ValueError("Boole grid must be strictly increasing and uniform")
    weights = np.zeros(len(t))
    for start in range(0, len(t)-1, 4):
        weights[start:start+5] += np.array([7., 32., 12., 32., 7.])*2*steps[0]/45
    return weights


def weak_test_functions(t):
    """Four Legendre bubbles and physical-time derivatives on a 100 ms window."""
    t = _array(t, "test times")
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0) or not np.isclose(t[-1]-t[0], .1, atol=1e-12, rtol=0):
        raise ValueError("test functions require an increasing 100 ms window")
    tau = (t-t[0])/.1
    x = 2*tau-1
    polynomials = (np.ones_like(x), x, (3*x*x-1)/2, (5*x**3-3*x)/2)
    derivatives = (np.zeros_like(x), np.ones_like(x), 3*x, (15*x*x-3)/2)
    v = np.stack([tau*(1-tau)*p for p in polynomials])
    dv = np.stack([((1-2*tau)*p+2*tau*(1-tau)*dp)/.1
                   for p, dp in zip(polynomials, derivatives)])
    return v, dv


def _port_constants(functional):
    """The exponential memory basis requires constant diagonal dissipation."""
    f = functional
    n = _nodes(f.n_nodes)
    if f.gate_weight_mem is not None or f.mem_laplacian_weights is not None or f.gates.s_coi != 0:
        raise ValueError("port reconstruction supports constant diagonal memory dissipation only")
    names = ("gain_p", "storage_p", "rate_p", "j_omega_mem", "lambda_mem", "tau_mem_sync")
    values = {name: _array(getattr(f, name), name) for name in names}
    values["s_mem"] = _array(f.gates.s_mem, "s_mem")
    if any(value.shape != (n,) for value in values.values()):
        raise ValueError("port constants must be node-aligned")
    if (np.any(values["storage_p"] <= 0) or np.any(values["rate_p"] < 0)
            or np.any(values["lambda_mem"] <= 0) or np.any(values["tau_mem_sync"] <= 0)):
        raise ValueError("invalid port storage or decay constants")
    j = values["s_mem"]*values["j_omega_mem"]
    return values, j, values["lambda_mem"]/values["tau_mem_sync"]


def _time_rk4(rhs, t, state, step):
    a = rhs(t, state)
    b = rhs(t+step/2, state+step*a/2)
    c = rhs(t+step/2, state+step*b/2)
    d = rhs(t+step, state+step*c)
    result = state+step*(a+2*b+2*c+d)/6
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite port reconstruction")
    return result


def reconstruct_ports(t, omega, p_end, functional):
    """Reconstruct p backward from p_end and forced r forward from r(t0)=0."""
    t = _array(t, "port times")
    omega, p_end = _array(omega, "omega"), _array(p_end, "p_end")
    n = functional.n_nodes
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0) or omega.shape != (len(t), n) or p_end.shape != (n,):
        raise ValueError("increasing times and node-aligned port observations required")
    constants, j, decay = _port_constants(functional)
    def forcing(time):
        return np.array([np.interp(time, t, omega[:, node]) for node in range(n)])
    p, rf = np.zeros_like(omega), np.zeros_like(omega)
    p[-1] = p_end
    for index in range(len(t)-1, 0, -1):
        p[index-1] = _time_rk4(
            lambda time, value: -constants["rate_p"]*constants["storage_p"]*value-constants["gain_p"]*forcing(time),
            t[index], p[index], t[index-1]-t[index])
    for index in range(len(t)-1):
        rf[index+1] = _time_rk4(lambda time, value: j*forcing(time)-decay*value,
                              t[index], rf[index], t[index+1]-t[index])
    return p, rf


def prepare_weak_history(packet, functional, step=.0025):
    """Cache terms independent of unknown edge coefficients and candidate a."""
    f = functional
    packet = validate_observation(packet, f.n_nodes)
    if step not in HISTORY_STEPS:
        raise ValueError("unregistered reconstruction step")
    constants, j, decay = _port_constants(f)
    t = np.linspace(-.1, 0., int(round(.1/step))+1)
    dense = {name: np.column_stack([np.interp(t, packet["history_times"], packet[name][:, node])
                                    for node in range(f.n_nodes)])
             for name in ("theta_history", "omega_history")}
    theta, omega = dense["theta_history"], dense["omega_history"]
    p, rf = reconstruct_ports(t, omega, packet["p_end"], f)
    v, dv = weak_test_functions(t)
    weights = boole_weights(t)
    weighted = v*weights
    sine = np.sin(theta@f.incidence)-np.sin(f.eta_edge)
    edge_integrals = weighted@sine
    A_static = np.vstack([f.incidence*row[None, :] for row in edge_integrals])/STATE_SCALE
    inertial = ((dv*weights)@(omega*f.mass)).reshape(-1)/STATE_SCALE
    damping = (weighted@(omega*f.damping)).reshape(-1)/STATE_SCALE
    p_term = (weighted@(p*f.gain_p*f.storage_p)).reshape(-1)/STATE_SCALE
    rf_term = (weighted@(rf*j*f.lambda_mem)).reshape(-1)/STATE_SCALE
    integrated = weighted@(np.exp(-(t-t[0])[:, None]*decay)*j*f.lambda_mem)
    Z = np.vstack([np.diag(row) for row in integrated])
    fixed = {name: value.copy() for name, value in constants.items()}
    fixed.update(mass=f.mass.copy(), damping=f.damping.copy(), eta_edge=f.eta_edge.copy(), gamma_amp=f.gamma_amp.copy())
    return dict(t=t, theta=theta, omega=omega, p=p, rf=rf, A_static=A_static, Z=Z,
                b=inertial-damping+p_term-rf_term, inertial=inertial, damping=damping,
                p_term=p_term, rf_term=rf_term, fixed_parameters=fixed)


def conditional_dictionary(cache, functional, a_history, *, a_times):
    """Hold candidate a independent when forming the partial derivative U_theta.

    All candidate edges are present; the functional's unknown k and edge gates
    deliberately do not enter this conditional dictionary.
    """
    f = functional
    constants, _, _ = _port_constants(f)
    current = dict(constants, mass=f.mass, damping=f.damping, eta_edge=f.eta_edge, gamma_amp=f.gamma_amp)
    if any(not np.array_equal(cache["fixed_parameters"][name], value) for name, value in current.items()):
        raise ValueError("fixed physics changed; rebuild the weak-history cache")
    a, a_times = _array(a_history, "candidate a"), _array(a_times, "candidate times")
    t = cache["t"]
    if a.shape != (len(t), f.n_nodes) or not np.array_equal(a_times, t):
        raise ValueError("candidate a must match the full quadrature times and nodes")
    indices = np.asarray(f.edges, dtype=int)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        exponent = a[:, indices[:, 0]]*f.gamma_amp[indices[:, 0]]+a[:, indices[:, 1]]*f.gamma_amp[indices[:, 1]]
        exponential = np.exp(exponent)
    v, _ = weak_test_functions(t)
    sine = np.sin(cache["theta"]@f.incidence)-np.sin(f.eta_edge)
    edge_integrals = (v*boole_weights(t))@(sine*exponential)
    A_exp = np.vstack([f.incidence*row[None, :] for row in edge_integrals])/STATE_SCALE
    A_static = cache["A_static"].copy()
    return dict(A_static=A_static, A_exp=A_exp, A=np.hstack((A_static, A_exp)),
                explanation=np.hstack((A_static, A_exp-A_static)), Z=cache["Z"].copy(), b=cache["b"].copy())


def combine_systems(systems):
    if not systems:
        raise ValueError("at least one weak system required")
    A = np.vstack([system["A"] for system in systems])
    b = np.concatenate([system["b"] for system in systems])
    Z = np.zeros((len(b), sum(system["Z"].shape[1] for system in systems)))
    row = column = 0
    for system in systems:
        rows, columns = system["Z"].shape
        Z[row:row+rows, column:column+columns] = system["Z"]
        row += rows
        column += columns
    return A, Z, b


def ranks(singular):
    return {str(t): int(np.sum(singular > t*singular[0])) if singular.size and singular[0] > 0 else 0
            for t in RANK_THRESHOLDS}


def _linear_system(A, b):
    A, b = _array(A, "matrix"), _array(b, "target")
    if A.ndim != 2 or b.shape != (A.shape[0],) or not len(b):
        raise ValueError("nonempty row-aligned linear system required")
    return A, b


def project_memory(A, Z, b):
    A, b = _linear_system(A, b)
    Z = _array(Z, "memory matrix")
    if Z.ndim != 2 or Z.shape[0] != len(b):
        raise ValueError("memory matrix must have matching rows")
    U, singular, Vt = np.linalg.svd(Z, full_matrices=False)
    rank = ranks(singular)[str(1e-10)]
    basis = U[:, :rank]
    residual_A, residual_b = A-basis@(basis.T@A), b-basis@(basis.T@b)
    return dict(X=residual_A/np.sqrt(len(b)), y=residual_b/np.sqrt(len(b)), U=U, s=singular, Vt=Vt,
                memory_rank=rank, memory_ranks=ranks(singular), orthogonality_A=basis.T@residual_A,
                orthogonality_b=basis.T@residual_b)


def recover_memory(A, b, coefficients, projection):
    """Return physical rho at the history start, not r at its endpoint."""
    A, b = _linear_system(A, b)
    coefficients = _array(coefficients, "coefficients")
    if coefficients.shape != (A.shape[1],):
        raise ValueError("coefficients must match dictionary columns")
    rank = projection["memory_rank"]
    return STATE_SCALE*(projection["Vt"][:rank].T@(
        (projection["U"][:, :rank].T@(b-A@coefficients))/projection["s"][:rank]))


def nnls_kkt(X, y, coefficients):
    X, y = _linear_system(X, y)
    coefficients = _array(coefficients, "coefficients")
    if coefficients.shape != (X.shape[1],):
        raise ValueError("coefficients must match the actual NNLS columns")
    with np.errstate(over="raise", invalid="raise"):
        gradient = X.T@(X@coefficients-y)
        violation = np.where(coefficients > 1e-10, np.abs(gradient), np.maximum(0., -gradient))
        v = float(np.max(violation)) if len(coefficients) else 0.
        norm = float(np.linalg.norm(X, 2)) if X.size else 0.
        scale = norm*(norm*float(np.linalg.norm(coefficients))+float(np.linalg.norm(y)))
    if not np.isfinite(gradient).all() or not np.isfinite(scale):
        raise FloatingPointError("nonfinite NNLS optimality audit")
    feasible = bool(np.all(coefficients >= -1e-12))
    relative = v/scale if scale > 0 else None
    qualified = feasible and (relative <= 1e-9 if relative is not None else bool(np.all(gradient == 0)))
    return dict(gradient=gradient, violation=v, data_scale=scale, relative_violation=relative,
                feasible=feasible, qualified=bool(qualified), trivial_optimum=bool(qualified and scale == 0))


def solve_nnls(X, y):
    """One actual NNLS call (maxiter=200), or an exact empty-column algebra term."""
    X, y = _linear_system(X, y)
    record = dict(actual_algebra_terms=1, actual_nnls_calls=int(X.shape[1] > 0), nnls_completed=0, nnls_qualified=0)
    try:
        coefficients = nnls(X, y, maxiter=200)[0] if X.shape[1] else np.zeros(0)
        record["nnls_completed"] = record["actual_nnls_calls"]
        audit = nnls_kkt(X, y, coefficients)
        record.update(coefficients=coefficients, loss=float(np.sum((X@coefficients-y)**2)), audit=audit,
                      status="qualified" if audit["qualified"] else "solver_unresolved",
                      nnls_qualified=int(audit["qualified"])*record["actual_nnls_calls"])
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        record.update(status="solver_unresolved", coefficients=None, loss=None, audit=None, error=repr(exc))
    return record


def angle_basis(n_nodes):
    """Deterministic Euclidean orthonormal complement of the arithmetic common angle."""
    n = _nodes(n_nodes)
    columns = []
    for index in range(n-1):
        value = np.eye(n)[:, index]-np.full(n, 1./n)
        for _ in range(2):
            for previous in columns:
                value -= previous*(previous@value)
        value /= np.linalg.norm(value)
        if value[np.flatnonzero(np.abs(value) > 1e-14)[0]] < 0:
            value = -value
        columns.append(value)
    return np.column_stack(columns)


def rk4(functional, state, step):
    state = _array(state, "state")
    functional.split_state(state)
    if not np.isfinite(step) or step == 0:
        raise ValueError("finite nonzero RK4 step required")
    a = functional.rhs(state)
    b = functional.rhs(state+step*a/2)
    c = functional.rhs(state+step*b/2)
    d = functional.rhs(state+step*c)
    result = state+step*(a+2*b+2*c+d)/6
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite trajectory")
    return result


def backward_history(functional, endpoint, step=.0025, *, dense=False):
    if step not in HISTORY_STEPS:
        raise ValueError("unregistered history step")
    state = _array(endpoint, "endpoint").copy()
    functional.split_state(state)
    states = [state.copy()]
    for _ in range(int(round(.1/step))):
        state = rk4(functional, state, -step)
        states.append(state.copy())
    states = np.asarray(states[::-1])
    return states if dense else states[::int(round(.01/step))]


def observations(states, mass):
    mass, states = _array(mass, "mass"), _array(states, "states")
    if mass.ndim != 1 or mass.size < 2 or np.any(mass <= 0) or states.ndim != 2 or states.shape[1] != 5*len(mass):
        raise ValueError("positive mass and node-aligned 5N states required")
    n = len(mass)
    theta = states[:, :n]
    relative = theta-(theta@mass/mass.sum())[:, None]
    return np.concatenate((relative, states[:, n:2*n]), axis=1).ravel()/STATE_SCALE


def history_residual(functional, predicted, packet):
    packet = validate_observation(packet, functional.n_nodes)
    if np.shape(predicted) != (11, 5*functional.n_nodes):
        raise ValueError("predicted history must have eleven 5N states")
    theta = packet["theta_history"]
    relative = theta-(theta@functional.mass/functional.mass.sum())[:, None]
    measured = np.concatenate((relative, packet["omega_history"]), axis=1).ravel()/STATE_SCALE
    return observations(predicted, functional.mass)-measured
