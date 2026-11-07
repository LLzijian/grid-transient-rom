"""Finite Run2 GFTD parameter search and causal two-stage initialization."""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import fields
import hashlib
import itertools
import json
import time

import numpy as np
from scipy.optimize import least_squares

from . import gftd_discovery as core
from ..physics.kuramoto_functional import StructuralGates

PROTOCOL_SHA256 = "54622ad9da5fea1afb326140c6d4288a51bddca2aae896ef10926f951b09dcb0"
Q_VERSION = "mgs2_p0_ordered_positive_v1"
_BATCH_CACHE = OrderedDict()


class FitFailure(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class InitializationFailure(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _physics(f, *, include_edges=True):
    values = {field.name: getattr(f, field.name) for field in fields(f) if field.name != "gates"}
    values["gates"] = {field.name: getattr(f.gates, field.name) for field in fields(f.gates)}
    if not include_edges:
        values.pop("k_edge")
        values["gates"].pop("s_edge")
        values["gates"].pop("s_mod")
    return values


def _config(f, mode, sigma):
    return dict(format="gftd_causal_two_stage_v1", protocol_sha256=PROTOCOL_SHA256, mode=mode,
                sigma=float(sigma), q_version=Q_VERSION, Q=core.angle_basis(f.n_nodes).tolist(),
                model_sha256=_hash(_physics(f)), history_step=.0025, fine_history_step=.00125,
                future_step=.01, fine_future_step=.005, proxy_start_calls=200,
                simple_calls=600, smooth_calls=1200, max_nfev=80, jacobian_step=2.5e-5,
                ftol=1e-10, xtol=1e-10, gtol=1e-10)


def validate_config(functional, config):
    try:
        if not isinstance(config, dict) or config.get("mode") not in ("static", "modulated"):
            raise ValueError("a complete frozen causal initialization config is required")
        sigma = float(config["sigma"])
        if not np.isfinite(sigma) or sigma < 0 or _hash(config) != _hash(_config(functional, config["mode"], sigma)):
            raise ValueError("initialization config or model fingerprint changed")
        core._port_constants(functional)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InitializationFailure(str(exc), dict(status="invalid_configuration", error=repr(exc))) from exc
    return copy.deepcopy(config)


def _counts():
    return dict(algebra_terms_planned=0, algebra_terms_actual=0, algebra_terms_skipped=0,
                nnls_attempts=0, nnls_completed=0, nnls_qualified=0, reconstruction_attempts=0, reconstructions=0,
                optimization_tasks=0, actual_residual_calls=0, integrations_attempted=0,
                integrations_completed=0, rk4_steps_attempted=0, rk4_steps_completed=0)


def _add(target, source):
    for key in target:
        target[key] += source.get(key, 0)


def _parameters(f, k, beta):
    result = copy.deepcopy(f)
    k, beta = np.asarray(k, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    if (k.shape != (f.n_edges,) or beta.shape != k.shape or not np.isfinite(k).all()
            or not np.isfinite(beta).all() or np.any(k < 0) or np.any(beta < 0) or np.any(beta > k)):
        raise ValueError("edge parameters require finite k>=beta>=0")
    result.k_edge = k.copy()
    result.gates = StructuralGates(s_edge=(k > 0).astype(float),
        s_mod=np.divide(beta, k, out=np.zeros_like(k), where=k > 0), s_mem=f.gates.s_mem.copy(), s_coi=0.)
    result.__post_init__()
    return result


def _integrate(f, endpoint, step, duration, counts, *, backward=False, dense=False):
    counts["integrations_attempted"] += 1
    state = np.asarray(endpoint, dtype=np.float64).copy()
    states = [state.copy()]
    for _ in range(int(round(duration/step))):
        counts["rk4_steps_attempted"] += 1
        state = core.rk4(f, state, -step if backward else step)
        counts["rk4_steps_completed"] += 1
        states.append(state.copy())
    counts["integrations_completed"] += 1
    states = np.asarray(states[::-1] if backward else states)
    return states if dense else states[::int(round(.01/step))]


def _layout(n, sigma, mode, role):
    noisy = sigma > 0 and role != "simple"
    amplitude = mode == "modulated" and role != "simple"
    sizes = [n-1 if noisy else 0, n if noisy else 0, n if amplitude else 0, n]
    offsets = np.cumsum([0]+sizes)
    slices = tuple(slice(offsets[index], offsets[index+1]) for index in range(4))
    upper = np.concatenate([np.full(size, 5. if index < 2 else 2.) for index, size in enumerate(sizes)])
    return slices, -upper, upper


def _endpoint(packet, f, u, mode, role):
    slices, low, high = _layout(f.n_nodes, float(packet["sigma"]), mode, role)
    u = np.asarray(u, dtype=np.float64)
    if u.shape != low.shape or not np.isfinite(u).all() or np.any(u < low) or np.any(u > high):
        raise ValueError("invalid bounded endpoint variables")
    theta, omega = packet["theta_history"][-1].copy(), packet["omega_history"][-1].copy()
    if slices[0].stop:
        theta += .1*float(packet["sigma"])*(core.angle_basis(f.n_nodes)@u[slices[0]])
        omega += .1*float(packet["sigma"])*u[slices[1]]
    a = .1*u[slices[2]] if slices[2].stop > slices[2].start else np.zeros(f.n_nodes)
    return f.pack_state(theta, omega, packet["p_end"], a, .1*u[slices[3]])


def _jacobian(residual, u, low, high):
    h, center, columns = 2.5e-5, None, []
    for index in range(len(u)):
        shift = np.eye(len(u))[index]*h
        if u[index]-h >= low[index] and u[index]+h <= high[index]:
            column = (residual(u+shift)-residual(u-shift))/(2*h)
        else:
            if center is None:
                center = residual(u)
            sign = 1 if u[index]-h < low[index] else -1
            column = sign*(-3*center+4*residual(u+sign*shift)-residual(u+2*sign*shift))/(2*h)
        columns.append(column)
    return np.column_stack(columns)


class _BudgetReached(RuntimeError):
    pass


def infer_endpoint(f, packet, *, mode, role, call_limit, initial=None, feasible=None):
    """A finite local task; every residual evaluation, including Jac, is counted."""
    packet = core.validate_observation(packet, f.n_nodes)
    _, low, high = _layout(f.n_nodes, float(packet["sigma"]), mode, role)
    initial = np.zeros_like(low) if initial is None else np.asarray(initial, dtype=np.float64).copy()
    _endpoint(packet, f, initial, mode, role)
    counts, trace, jacobian = _counts(), [], None
    counts["optimization_tasks"] = 1
    best, baseline, result = None, None, None
    divisor = max(float(packet["sigma"]), 1e-8)
    begin = time.perf_counter()
    def residual(u):
        nonlocal best
        if counts["actual_residual_calls"] >= call_limit:
            raise _BudgetReached("actual historical residual call budget reached")
        counts["actual_residual_calls"] += 1
        state = _endpoint(packet, f, u, mode, role)
        history = _integrate(f, state, .0025, .1, counts, backward=True, dense=True)
        raw = core.history_residual(f, history[::4], packet)
        sse = float(raw@raw)
        if not np.isfinite(sse):
            raise FloatingPointError("nonfinite historical loss")
        if best is None or sse < best["observation_sse"]:
            best = dict(u=np.array(u).copy(), state=state, observation_sse=sse,
                        history_mse=sse/raw.size, history_rms=float(np.sqrt(sse/raw.size)),
                        residual=raw, dense_history=history)
        return raw/divisor
    def derivative(u):
        nonlocal jacobian
        matrix = _jacobian(residual, u, low, high)
        jacobian = dict(u=np.array(u).copy(), optimizer_column_norms=np.linalg.norm(matrix, axis=0),
                        observation_column_norms=np.linalg.norm(matrix, axis=0)*divisor)
        return matrix
    def callback(intermediate_result):
        previous = trace[-1]["u"] if trace else initial
        trace.append(dict(u=np.array(intermediate_result.x).copy(), cost=float(intermediate_result.cost),
                          nfev=int(intermediate_result.nfev), step_norm=float(np.linalg.norm(intermediate_result.x-previous))))
    try:
        if feasible is not None:
            residual(np.asarray(feasible, dtype=np.float64))
            # The direct residual at this point is retained in best; never compare a reconstructed frozen loss.
            baseline = float(best["observation_sse"])
        residual(initial)
        result = least_squares(residual, initial, jac=derivative, method="trf", bounds=(low, high),
            x_scale=1., ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=80, callback=callback)
        status = "converged" if result.success else "budget_limited" if result.status == 0 else "failed"
        reason = str(result.message)
    except _BudgetReached as exc:
        status, reason = "budget_limited", str(exc)
    except (ValueError, FloatingPointError, OverflowError, RuntimeError) as exc:
        status, reason = "failed", repr(exc)
    gradient = None if result is None else np.asarray(result.grad)
    if gradient is not None and not np.isfinite(gradient).all():
        gradient = None
        status, reason = "failed", "nonfinite optimizer gradient"
    kkt = None
    if gradient is not None and best is not None and np.array_equal(result.x, best["u"]):
        projected = gradient.copy()
        projected[(result.x <= low+1e-7) & (gradient >= 0)] = 0.
        projected[(result.x >= high-1e-7) & (gradient <= 0)] = 0.
        kkt = float(np.max(np.abs(projected)))
    return dict(status=status, reason=reason, best=best, counts=counts, actual_residual_calls=counts["actual_residual_calls"],
                initial=initial, feasible=feasible, same_step_feasible_sse=baseline, optimizer_residual_divisor=divisor,
                scipy_nfev=None if result is None else int(result.nfev), scipy_njev=None if result is None else int(result.njev),
                scipy_status=None if result is None else int(result.status),
                scipy_success=None if result is None else bool(result.success),
                optimizer_gradient=gradient, gradient_evaluated_at=None if result is None else result.x,
                observation_gradient=None if gradient is None else gradient*divisor**2,
                projected_gradient_kkt=kkt, observation_projected_gradient_kkt=None if kkt is None else kkt*divisor**2,
                jacobian=jacobian, lower_bounds=low, upper_bounds=high, step_trace=trace,
                final_accepted_step_norm=None if not trace else trace[-1]["step_norm"],
                active_bounds=None if best is None else np.flatnonzero((best["u"] <= low+1e-7) | (best["u"] >= high-1e-7)),
                optimization_sufficiency="unknown", wall_seconds=time.perf_counter()-begin)


def numerical_check(f, state, packet):
    counts, begin = _counts(), time.perf_counter()
    record = dict(passed=False, counts=counts)
    try:
        histories, roundtrips, futures = [], [], []
        for step in (.0025, .00125):
            history = _integrate(f, state, step, .1, counts, backward=True)
            histories.append(core.observations(history, f.mass))
            returned = _integrate(f, history[0], step, .1, counts)
            roundtrips.append(float(np.max(np.abs((returned[-1]-state)/.1))))
        for step in (.01, .005):
            futures.append(core.observations(_integrate(f, state, step, 1., counts), f.mass))
        history_delta, future_delta = histories[0]-histories[1], futures[0]-futures[1]
        p_error = float(np.max(np.abs(state[2*f.n_nodes:3*f.n_nodes]-packet["p_end"])))
        record.update(roundtrip_scaled_max=roundtrips, history_main_fine_rms=float(np.sqrt(np.mean(history_delta**2))),
                      future_main_fine_rms=float(np.sqrt(np.mean(future_delta**2))),
                      future_main_fine_peak=float(np.max(np.abs(future_delta))), p_endpoint_error=p_error)
        values = np.r_[roundtrips, record["history_main_fine_rms"], record["future_main_fine_rms"], record["future_main_fine_peak"], p_error]
        record["passed"] = bool(np.isfinite(values).all() and np.all(values <= [1e-7, 1e-7, 1e-7, 1e-5, 1e-4, 0.]))
    except (ValueError, FloatingPointError, OverflowError) as exc:
        record["error"] = repr(exc)
    record["wall_seconds"] = time.perf_counter()-begin
    return record


def initialize_complete(functional, p_end_packet, config):
    config = validate_config(functional, config)
    try:
        packet = core.validate_observation(p_end_packet, functional.n_nodes)
    except (ValueError, TypeError) as exc:
        raise InitializationFailure(str(exc), dict(status="invalid_observation", error=repr(exc))) from exc
    if float(packet["sigma"]) != config["sigma"]:
        raise InitializationFailure("noise configuration changed", dict(status="invalid_configuration"))
    mode, counts = config["mode"], _counts()
    diagnostics = dict(status="blocked_by_stage1", counts=counts, stage1=None, stage2=None,
                       stage1_numerical=None, final_numerical=None, returned_point=None)
    first = infer_endpoint(functional, packet, mode=mode, role="simple", call_limit=600)
    diagnostics["stage1"] = first
    _add(counts, first["counts"])
    if first["best"] is None or first["status"] not in ("converged", "budget_limited"):
        raise InitializationFailure("blocked_by_stage1", _plain(diagnostics))
    check = numerical_check(functional, first["best"]["state"], packet)
    diagnostics["stage1_numerical"] = check
    _add(counts, check["counts"])
    if not check["passed"]:
        raise InitializationFailure("blocked_by_stage1", _plain(diagnostics))
    _, low, _ = _layout(functional.n_nodes, config["sigma"], mode, "smooth")
    feasible = np.zeros_like(low)
    feasible[-functional.n_nodes:] = first["best"]["u"]
    second = infer_endpoint(functional, packet, mode=mode, role="smooth", call_limit=1200,
                            initial=feasible, feasible=feasible)
    diagnostics["stage2"] = second
    _add(counts, second["counts"])
    if second["best"] is None:
        diagnostics["status"] = "no_output"
        raise InitializationFailure("no_output", _plain(diagnostics))
    old_loss, new_loss = second["same_step_feasible_sse"], second["best"]["observation_sse"]
    improved = old_loss is not None and _improves(new_loss, old_loss)
    state = second["best"]["state"] if improved else first["best"]["state"]
    diagnostics.update(returned_point="stage2" if improved else "stage1", no_improvement=not improved,
                       selected_projected_gradient_kkt=second["projected_gradient_kkt"] if improved else None)
    if np.array_equal(state, first["best"]["state"]):
        final_check = copy.deepcopy(check)
        diagnostics["final_check_reused_stage1"] = True
    else:
        final_check = numerical_check(functional, state, packet)
        _add(counts, final_check["counts"])
        diagnostics["final_check_reused_stage1"] = False
    diagnostics["final_numerical"] = final_check
    if not final_check["passed"]:
        diagnostics["status"] = "numerically_unresolved"
        raise InitializationFailure("numerically_unresolved", _plain(diagnostics))
    diagnostics["status"] = "completed"
    return np.array(state).copy(), _plain(diagnostics)


def _improves(new, old):
    return new < old-1e-14*max(1., abs(old))


def project_systems(systems):
    """Block SVD with the global memory cutoff and total-row RMS scale."""
    if not systems:
        raise ValueError("at least one event system required")
    decompositions = [np.linalg.svd(system["Z"], full_matrices=False) for system in systems]
    singular = np.concatenate([item[1] for item in decompositions])
    cutoff = 1e-10*float(np.max(singular)) if singular.size else 0.
    rows = sum(len(system["b"]) for system in systems)
    blocks, matrices, vectors = [], [], []
    for system, (U, s, Vt) in zip(systems, decompositions):
        rank = int(np.sum(s > cutoff))
        basis = U[:, :rank]
        matrices.append((system["A"]-basis@(basis.T@system["A"]))/np.sqrt(rows))
        vectors.append((system["b"]-basis@(basis.T@system["b"]))/np.sqrt(rows))
        blocks.append(dict(U=U, s=s, Vt=Vt, memory_rank=rank, memory_ranks=core.ranks(s)))
    return dict(X=np.vstack(matrices), y=np.concatenate(vectors), blocks=blocks,
                memory_ranks=core.ranks(np.sort(singular)[::-1]), total_rows=rows)


def information_diagnostics(X, memory_ranks):
    norms = np.linalg.norm(X, axis=0)
    singular = np.linalg.svd(X, compute_uv=False)
    rank_info = core.ranks(singular)
    correlation = np.divide(X.T@X, norms[:, None]*norms[None, :],
                            out=np.zeros((X.shape[1], X.shape[1])), where=norms[:, None]*norms[None, :] > 0)
    conditional = []
    for column in range(X.shape[1]):
        U, s, _ = np.linalg.svd(np.delete(X, column, axis=1), full_matrices=False)
        other_ranks = core.ranks(s)
        rank = other_ranks[str(1e-10)]
        norm = float(np.linalg.norm(X[:, column]-U[:, :rank]@(U[:, :rank].T@X[:, column])))
        ratio = norm/norms[column] if norms[column] > 0 else 0.
        stable = all(len(set(values.values())) <= 1 for values in (rank_info, other_ranks, memory_ranks))
        other_correlations = np.delete(np.abs(correlation[column]), column)
        qualified = norms[column] > 1e-6 and ratio > 1e-8 and stable and bool(np.all(other_correlations < 1-1e-8))
        conditional.append(dict(norm=norm, ratio=float(ratio), other_ranks=other_ranks,
                                rank_threshold_sensitive=not stable, qualified=bool(qualified)))
    return dict(column_norms=norms, frobenius_norm=float(np.linalg.norm(X)), singular_values=singular,
                ranks=rank_info, memory_ranks=memory_ranks, correlation=correlation,
                conditional_columns=conditional, columns_qualified=[item["qualified"] for item in conditional])


def _rank_signature(info):
    groups = [info["ranks"], info["memory_ranks"]]+[item["other_ranks"] for item in info["conditional_columns"]]
    return tuple(tuple(group[str(threshold)] for threshold in core.RANK_THRESHOLDS) for group in groups)


def _penalty(sigma, edge_mask, mod_mask):
    return (1e-6+.1*sigma)**2*(int(np.sum(edge_mask))+int(np.sum(mod_mask)))


def _solve_mask(X, y, edge_mask, mod_mask, counts):
    e = len(edge_mask)
    mask = np.r_[edge_mask, mod_mask] if X.shape[1] == 2*e else np.asarray(edge_mask)
    result = core.solve_nnls(X[:, mask], y)
    counts["algebra_terms_actual"] += 1
    counts["nnls_attempts"] += result["actual_nnls_calls"]
    counts["nnls_completed"] += result["nnls_completed"]
    counts["nnls_qualified"] += result["nnls_qualified"]
    coefficients = None
    if result["coefficients"] is not None:
        coefficients = np.zeros(X.shape[1])
        coefficients[mask] = result["coefficients"]
    result.update(coefficients=coefficients, edge_mask=np.array(edge_mask).copy(), mod_mask=np.array(mod_mask).copy(),
                  open_columns=np.flatnonzero(mask).tolist())
    return result


def _edge_values(coefficients, e):
    if len(coefficients) == e:
        return coefficients.copy(), np.zeros(e)
    return coefficients[:e]+coefficients[e:], coefficients[e:].copy()


def _threshold_and_debias(X, y, selected, e, counts):
    k, beta = _edge_values(selected["coefficients"], e)
    edge_mask, mod_mask = k >= .05, (k >= .05) & (beta >= .025)
    debias = _solve_mask(X, y, edge_mask, mod_mask, counts)
    if debias["coefficients"] is None:
        return dict(status="solver_unresolved", debias=debias, raw_k=None, raw_beta=None,
                    edge_mask=edge_mask, mod_mask=mod_mask, threshold_inconsistent=True,
                    edge_threshold_inconsistent=True, mod_threshold_inconsistent=True)
    k, beta = _edge_values(debias["coefficients"], e)
    edge_bad, mod_bad = bool(np.any(k[edge_mask] < .05)), bool(np.any(beta[mod_mask] < .025))
    return dict(status=debias["status"], debias=debias, raw_k=k, raw_beta=beta,
                edge_mask=edge_mask, mod_mask=mod_mask, threshold_inconsistent=edge_bad or mod_bad,
                edge_threshold_inconsistent=edge_bad, mod_threshold_inconsistent=mod_bad)


def backward_select(X, y, sigma, edge_mask, mod_mask, counts):
    """Finite deterministic deletion; unresolved comparisons invalidate this proposal."""
    e = len(edge_mask)
    capacity, start = 2*e*(2*e+1)+1, counts["algebra_terms_actual"]
    counts["algebra_terms_planned"] += capacity
    record = dict(status="solver_unresolved", comparisons=[], selected=None, raw_k=None, raw_beta=None,
                  projected_X=X.copy(), projected_y=y.copy(),
                  edge_mask=np.array(edge_mask).copy(), mod_mask=np.array(mod_mask).copy())
    try:
        if np.linalg.norm(X) <= 1e-6:
            record["status"] = "insufficient_information"
            return record
        current = _solve_mask(X, y, edge_mask, mod_mask, counts)
        current["J"] = None if current["loss"] is None else current["loss"]+_penalty(sigma, edge_mask, mod_mask)
        record["comparisons"].append(dict(role="initial", solution=current))
        if current["status"] != "qualified":
            return record
        for iteration in range(2*e):
            candidates = []
            for edge in range(e):
                for action in ("delete_edge", "close_modulation"):
                    if not current["edge_mask"][edge] or (action == "close_modulation" and not current["mod_mask"][edge]):
                        continue
                    edges, mods = current["edge_mask"].copy(), current["mod_mask"].copy()
                    mods[edge] = False
                    if action == "delete_edge":
                        edges[edge] = False
                    candidate = _solve_mask(X, y, edges, mods, counts)
                    candidate["J"] = None if candidate["loss"] is None else candidate["loss"]+_penalty(sigma, edges, mods)
                    record["comparisons"].append(dict(iteration=iteration, edge=edge, action=action, solution=candidate))
                    if candidate["status"] != "qualified":
                        return record
                    candidates.append(candidate)
            best = current
            for candidate in candidates:
                if _improves(candidate["J"], best["J"]):
                    best = candidate
            if best is current:
                break
            current = best
        record.update(selected=current, **_threshold_and_debias(X, y, current, e, counts))
        return record
    finally:
        skipped = capacity-(counts["algebra_terms_actual"]-start)
        if skipped < 0:
            raise AssertionError("backward selection finite algebra budget exceeded")
        counts["algebra_terms_skipped"] += skipped
        record["unused_algebra_items"] = dict(count=skipped, reason="early_stop_cached_current_or_unresolved")


def _enumerate_static(X, y, sigma, counts):
    e = X.shape[1]
    if e != 6:
        return dict(status="not_run_non_k4", counts=_counts())
    start = counts["algebra_terms_actual"]
    counts["algebra_terms_planned"] += 65
    record = dict(status="insufficient_information", enumeration=[], projected_X=X.copy(), projected_y=y.copy())
    try:
        if np.linalg.norm(X) <= 1e-6:
            return record
        for mask in range(64):
            edges = np.array([(mask >> edge) & 1 for edge in range(e)], dtype=bool)
            result = _solve_mask(X, y, edges, np.zeros(e, dtype=bool), counts)
            result["J"] = None if result["loss"] is None else result["loss"]+_penalty(sigma, edges, np.zeros(e, dtype=bool))
            record["enumeration"].append(result)
        available = [item for item in record["enumeration"] if item["J"] is not None]
        minimum = min(item["J"] for item in available)
        ties = [item for item in available if item["J"]-minimum <= 1e-14*max(1., abs(minimum))]
        selected = min(ties, key=lambda item: (int(np.sum(item["edge_mask"])), np.flatnonzero(item["edge_mask"]).tolist()))
        record.update(J_min=minimum, tied_supports=[np.flatnonzero(item["edge_mask"]).tolist() for item in ties],
                      selected=selected, **_threshold_and_debias(X, y, selected, e, counts))
        if any(item["status"] != "qualified" for item in record["enumeration"]):
            record["status"] = "solver_unresolved"
        return record
    finally:
        skipped = 65-(counts["algebra_terms_actual"]-start)
        counts["algebra_terms_skipped"] += skipped
        record["unused_algebra_items"] = dict(count=skipped, reason="insufficient_information")


def _prepare_batch(f, packets):
    key = _hash(dict(physics=_physics(f, include_edges=False), packets=packets, steps=core.HISTORY_STEPS))
    if key in _BATCH_CACHE:
        _BATCH_CACHE.move_to_end(key)
        return copy.deepcopy(_BATCH_CACHE[key]), True, key
    counts = _counts()
    try:
        result = _build_preparation(f, packets, counts)
    except (ValueError, FloatingPointError, OverflowError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise FitFailure("shared preparation failed", _plain(dict(status="failed", preparation_key=key,
                         counts=counts, error=repr(exc)))) from exc
    _BATCH_CACHE[key] = copy.deepcopy(result)
    while len(_BATCH_CACHE) > 8:
        _BATCH_CACHE.popitem(last=False)
    return result, False, key


def _build_preparation(f, packets, counts):
    grids = {}
    for step in core.HISTORY_STEPS:
        grids[step] = []
        for packet in packets:
            counts["reconstruction_attempts"] += 1
            grids[step].append(core.prepare_weak_history(packet, f, step))
            counts["reconstructions"] += 1
    static_systems = {step: [dict(A=item["A_static"], Z=item["Z"], b=item["b"]) for item in cache]
                      for step, cache in grids.items()}
    main = project_systems(static_systems[.0025])
    info = information_diagnostics(main["X"], main["memory_ranks"])
    selected = backward_select(main["X"], main["y"], float(packets[0]["sigma"]),
                               np.ones(f.n_edges, dtype=bool), np.zeros(f.n_edges, dtype=bool), counts)
    references = {}
    for step, systems in static_systems.items():
        projected = main if step == .0025 else project_systems(systems)
        reference = _enumerate_static(projected["X"], projected["y"], float(packets[0]["sigma"]), counts)
        if selected["raw_k"] is not None:
            value = float(np.sum((projected["X"]@selected["raw_k"]-projected["y"])**2))
            value += _penalty(float(packets[0]["sigma"]), selected["edge_mask"], selected["mod_mask"])
            reference["frozen_main_backward_J"] = value
            reference["frozen_main_raw_support"] = np.flatnonzero(selected["edge_mask"]).tolist()
            reference["frozen_main_objective_gap"] = None if reference.get("J_min") is None else value-reference["J_min"]
        references[str(step)] = reference
    return dict(grids=grids, initial_static=selected, initial_information=info, references=references, counts=counts)


def _proxy_candidate(f, packets, mode, edge_mask, mod_mask):
    counts, events = _counts(), []
    for packet in packets:
        slices, low, _ = _layout(f.n_nodes, float(packet["sigma"]), mode, "proxy")
        starts = [np.zeros_like(low), np.zeros_like(low)]
        alternate = (-1.)**np.arange(f.n_nodes)
        starts[1][slices[2] if mode == "modulated" else slices[3]] = (.2 if mode == "modulated" else .1)*alternate
        tasks = [infer_endpoint(f, packet, mode=mode, role="proxy", call_limit=200, initial=start) for start in starts]
        for task in tasks:
            _add(counts, task["counts"])
        eligible = [index for index, task in enumerate(tasks) if task["best"] is not None and task["status"] in ("converged", "budget_limited")]
        if not eligible:
            return dict(status="no_output", events=events+[dict(tasks=tasks, selected=None)], counts=counts, objective=None)
        selected = min(eligible, key=lambda index: (tasks[index]["best"]["history_mse"], index))
        first, second = (task["best"] for task in tasks)
        differences = None if first is None or second is None else dict(
            amplitude=(first["state"][3*f.n_nodes:4*f.n_nodes]-second["state"][3*f.n_nodes:4*f.n_nodes])/.1,
            memory=(first["state"][4*f.n_nodes:]-second["state"][4*f.n_nodes:])/.1,
            history_mse=first["history_mse"]-second["history_mse"])
        events.append(dict(tasks=tasks, selected=selected, best=tasks[selected]["best"], start_differences=differences))
    objective = float(np.mean([event["best"]["history_mse"] for event in events]))
    objective += _penalty(float(packets[0]["sigma"]), edge_mask, mod_mask)
    return dict(status="completed", events=events, counts=counts, objective=objective)


def _systems_for_candidate(prepared, f, candidate, step, counts):
    systems = []
    for cache, event in zip(prepared["grids"][step], candidate["events"]):
        dense = event["best"]["dense_history"] if step == .0025 else _integrate(
            f, event["best"]["state"], step, .1, counts, backward=True, dense=True)
        systems.append(core.conditional_dictionary(cache, f, dense[:, 3*f.n_nodes:4*f.n_nodes], a_times=cache["t"]))
    return systems


def mechanism_information(prepared, f, candidate, mode, counts):
    by_grid = {}
    for step in core.HISTORY_STEPS:
        systems = _systems_for_candidate(prepared, f, candidate, step, counts)
        explanation = [dict(system, A=system["explanation"] if mode == "modulated" else system["A_static"]) for system in systems]
        projection = project_systems(explanation)
        info = information_diagnostics(projection["X"], projection["memory_ranks"])
        event_info = []
        for system in explanation:
            local = project_systems([system])
            event_info.append(information_diagnostics(local["X"], local["memory_ranks"]))
        by_grid[str(step)] = dict(shared=info, events=event_info,
                                 explanation_projected_X=projection["X"], explanation_projected_y=projection["y"])
    coarse, fine = (by_grid[str(step)]["shared"] for step in core.HISTORY_STEPS)
    same_ranks = _rank_signature(coarse) == _rank_signature(fine)
    e = f.n_edges
    k_qualified = same_ranks and all(coarse["columns_qualified"][:e]) and all(fine["columns_qualified"][:e])
    beta_qualified = mode == "modulated" and same_ranks and all(coarse["columns_qualified"][e:]) and all(fine["columns_qualified"][e:])
    return dict(grids=by_grid, rank_signatures_agree=same_ranks, k_qualified=bool(k_qualified),
                beta_qualified=bool(beta_qualified), beta_status="fixed_by_model" if mode == "static" else "conditional_only",
                coordinate_transform="u=k-beta; v=beta; solver=[A_static,A_exp]; explanation=[A_static,A_exp-A_static]",
                row_scale="divide force rows by .1, then projected rows by sqrt(total_rows)",
                column_scale="none", joint_identifiability="unknown")


def fit_shared(functional, validated_p_end_packets, *, mode):
    begin, counts = time.perf_counter(), _counts()
    fit_state = dict(status="failed", counts=counts, trace=[], joint_identifiability="unknown")
    try:
        if mode not in ("static", "modulated") or not validated_p_end_packets:
            raise ValueError("a nonempty legal batch and static/modulated mode are required")
        f = copy.deepcopy(functional)
        packets = [core.validate_observation(packet, f.n_nodes) for packet in validated_p_end_packets]
        fit_state["event_count"] = len(packets)
        sigma = float(packets[0]["sigma"])
        if any(float(packet["sigma"]) != sigma for packet in packets):
            raise ValueError("all batch observations require the same known sigma")
        prepared, hit, key = _prepare_batch(f, packets)
        fit_state.update(preparation_key=key, preparation_cache_hit=hit, shared_preparation_counts=prepared["counts"],
                         initial_static=prepared["initial_static"], initial_information=prepared["initial_information"],
                         static_enumeration_references=prepared["references"])
        if not hit:
            _add(counts, prepared["counts"])
        selected = prepared["initial_static"]
        initial_identified = (selected["status"] == "qualified" and not selected["threshold_inconsistent"]
                              and all(prepared["initial_information"]["columns_qualified"]))
        unresolved_preparation = selected["status"] == "solver_unresolved"
        if mode == "static" and initial_identified:
            k, beta = selected["raw_k"], np.zeros(f.n_edges)
            edge_mask, mod_mask = selected["edge_mask"].copy(), np.zeros(f.n_edges, dtype=bool)
        else:
            k = np.full(f.n_edges, .2)
            beta = np.full(f.n_edges, .05 if mode == "modulated" else 0.)
            edge_mask, mod_mask = np.ones(f.n_edges, dtype=bool), np.full(f.n_edges, mode == "modulated")
        fit_state["initial_prior_not_identified"] = not initial_identified
        current_f = _parameters(f, k, beta)
        incumbent = _proxy_candidate(current_f, packets, mode, edge_mask, mod_mask)
        _add(counts, incumbent["counts"])
        fit_state["initial_candidate"] = incumbent
        if incumbent["status"] != "completed":
            raise FitFailure("no legal initial shared candidate", _plain(fit_state))
        edge_bad, mod_bad = False, False
        prior_only = not (mode == "static" and initial_identified)
        for iteration in range(3):
            record = dict(iteration=iteration, incumbent_objective=incumbent["objective"], accepted=False)
            fit_state["trace"].append(record)
            planned_before = counts["algebra_terms_planned"]
            try:
                systems = _systems_for_candidate(prepared, current_f, incumbent, .0025, counts)
                actual_systems = systems if mode == "modulated" else [dict(system, A=system["A_static"]) for system in systems]
                projected = project_systems(actual_systems)
                proposal = backward_select(projected["X"], projected["y"], sigma, edge_mask, mod_mask, counts)
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                skipped = 0
                if counts["algebra_terms_planned"] == planned_before:
                    skipped = 2*f.n_edges*(2*f.n_edges+1)+1
                    counts["algebra_terms_planned"] += skipped
                    counts["algebra_terms_skipped"] += skipped
                record.update(reason="parameter_nonfinite", error=repr(exc), coefficients=dict(
                    status="parameter_nonfinite", comparisons=[], unused_algebra_items=dict(count=skipped, reason="parameter_nonfinite")),
                    proxy_skipped=dict(tasks=2*len(packets), reason="parameter_nonfinite"))
                continue
            record["coefficients"] = proposal
            if proposal["status"] != "qualified":
                record["reason"] = proposal["status"]
                record["proxy_skipped"] = dict(tasks=2*len(packets), reason="coefficient_proposal_unresolved")
                continue
            try:
                proposed_f = _parameters(f, proposal["raw_k"], proposal["raw_beta"])
            except (ValueError, FloatingPointError, OverflowError) as exc:
                record.update(reason="parameter_infeasible", error=repr(exc),
                              proxy_skipped=dict(tasks=2*len(packets), reason="parameter_infeasible"))
                continue
            candidate = _proxy_candidate(proposed_f, packets, mode, proposal["edge_mask"], proposal["mod_mask"])
            _add(counts, candidate["counts"])
            record["candidate"] = candidate
            if candidate["status"] != "completed":
                record["reason"] = "candidate_no_output"
                continue
            coefficients = proposal["debias"]["coefficients"]
            record["conditional_rho_at_history_start"] = [core.recover_memory(
                system["A"], system["b"], coefficients, block)
                for system, block in zip(actual_systems, projected["blocks"])]
            record["conditional_before_reinference_weak_loss"] = float(np.sum((projected["X"]@coefficients-projected["y"])**2))
            try:
                updated_systems = _systems_for_candidate(prepared, proposed_f, candidate, .0025, counts)
                if mode == "static":
                    updated_systems = [dict(system, A=system["A_static"]) for system in updated_systems]
                updated = project_systems(updated_systems)
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                record.update(reason="parameter_nonfinite_after_reinference", error=repr(exc))
                continue
            record["conditional_after_reinference_weak_loss"] = float(np.sum((updated["X"]@coefficients-updated["y"])**2))
            if _improves(candidate["objective"], incumbent["objective"]):
                incumbent, current_f = candidate, proposed_f
                edge_mask, mod_mask = proposal["edge_mask"].copy(), proposal["mod_mask"].copy()
                edge_bad, mod_bad = proposal["edge_threshold_inconsistent"], proposal["mod_threshold_inconsistent"]
                prior_only = False
                record.update(accepted=True, reason="strict_history_objective_improvement")
            else:
                record["reason"] = "no_improvement"
        try:
            info = mechanism_information(prepared, current_f, incumbent, mode, counts)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            info = dict(k_qualified=False, beta_qualified=False, status="numerically_unresolved", error=repr(exc),
                        beta_status="fixed_by_model" if mode == "static" else "unknown", joint_identifiability="unknown")
        k_support = np.flatnonzero(edge_mask).tolist() if info["k_qualified"] and not edge_bad and not unresolved_preparation and not prior_only else None
        mod_support = np.flatnonzero(mod_mask).tolist() if info["beta_qualified"] and not edge_bad and not mod_bad and not unresolved_preparation and not prior_only else None
        null_reasons = dict(k=[], beta=["fixed_by_model"] if mode == "static" else [])
        for reason, failed in (("prior_only", prior_only), ("preparation_numerically_unresolved", unresolved_preparation),
                               ("parent_edge_threshold_inconsistent", edge_bad)):
            if failed:
                null_reasons["k"].append(reason)
                if mode == "modulated":
                    null_reasons["beta"].append(reason)
        if not info["k_qualified"]:
            null_reasons["k"].append("conditional_information_not_qualified")
        if mode == "modulated":
            if not info["beta_qualified"]:
                null_reasons["beta"].append("conditional_information_not_qualified")
            if mod_bad:
                null_reasons["beta"].append("modulation_threshold_inconsistent")
        fit_state.update(status="completed" if k_support is not None and (mode == "static" or mod_support is not None) else "not_identified",
            raw_edge_mask=edge_mask, raw_mod_mask=mod_mask, support=k_support, modulation_support=mod_support,
            k=current_f.k_edge, beta=current_f.k_edge*current_f.gates.s_mod, information=info,
            threshold_inconsistent=edge_bad or mod_bad, edge_threshold_inconsistent=edge_bad,
            mod_threshold_inconsistent=mod_bad, prior_only=prior_only, preparation_numerically_unresolved=unresolved_preparation,
            from_data=not prior_only, null_reasons=null_reasons,
            accepted_proposal_indices=[row["iteration"] for row in fit_state["trace"] if row["accepted"]],
            selected_proxy=incumbent, selected_objective=incumbent["objective"], event_count=len(packets),
            wall_seconds=time.perf_counter()-begin)
        config = _config(current_f, mode, sigma)
        plain = _plain(fit_state)
        json.dumps(plain, allow_nan=False)
        return dict(functional=current_f, config=config, fit_state=plain, status=fit_state["status"])
    except FitFailure:
        raise
    except (ValueError, FloatingPointError, OverflowError, RuntimeError, np.linalg.LinAlgError) as exc:
        fit_state.update(error=repr(exc), wall_seconds=time.perf_counter()-begin)
        raise FitFailure(str(exc), _plain(fit_state)) from exc
