"""Registered fit297 version: full static preparation and bounded subset adaptation."""
from __future__ import annotations

import copy
import time
import numpy as np

from . import gftd_discovery as core
from . import gftd_fitting as old

PROTOCOL_SHA256 = "ca22743c53fe4fe398960e111838a373f8c4686eab0974c5daf6d9c28435062f"
FIT_FORMAT = "gftd_fit297_v1"
FORMAT = "gftd_causal_two_stage_fit297_v1"
FitFailure = old.FitFailure
InitializationFailure = old.InitializationFailure
_CACHE = {}
_PREPARATION_FAILURES = {}
_LEDGERS = {}
_LIMITS = dict(information_svd=1878, conditional_projections=585, memory_svd=4000)


class _AnalysisLimit(RuntimeError):
    pass


def validate_fit_config(f, config, n):
    allowed = {"format", "protocol_sha256", "adaptation_indices", "input_scale_delta", "input_scale_omega", "rstar", "k_ref", "source_sha256"}
    if not isinstance(config, dict) or set(config) != allowed or config["format"] != FIT_FORMAT or config["protocol_sha256"].lower() != PROTOCOL_SHA256:
        raise ValueError("complete registered fit297 configuration required")
    if n < 1:
        raise ValueError("nonempty legal batch required")
    m = min(24, n)
    expected = [j*n//m for j in range(m)]
    indices = config["adaptation_indices"]
    if not isinstance(indices, list) or any(type(i) is not int for i in indices) or indices != expected:
        raise ValueError("adaptation indices must be the registered ordered uniform subset")
    for key in ("input_scale_delta", "input_scale_omega", "rstar"):
        scale = np.asarray(config[key], dtype=np.float64)
        if scale.shape != (f.n_nodes,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("finite positive node scales required")
    if not np.isfinite(config["k_ref"]) or config["k_ref"] <= 0:
        raise ValueError("positive finite k_ref required")
    sources = config["source_sha256"]
    if not isinstance(sources, dict) or not sources or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdefABCDEF" for c in v) for k, v in sources.items()):
        raise ValueError("frozen source hashes required")
    core._port_constants(f)
    return old._plain(copy.deepcopy(config))


def make_config(f, mode, request, training_count, preparation_key):
    request = validate_fit_config(f, request, training_count)
    if mode not in ("static", "modulated"):
        raise ValueError("unknown fit mode")
    result = old._config(f, mode, 0.)
    result.update(format=FORMAT, protocol_sha256=PROTOCOL_SHA256, fit_request=request,
        training_count=training_count, preparation_key=preparation_key,
        history_theta_scale=request["input_scale_delta"], history_omega_scale=request["input_scale_omega"],
        r_scale=request["rstar"], a_scale=.1, endpoint_box=[-2., 2.],
        proxy_alternate_amplitude=.2, proxy_alternate_memory=.1, proposals=3,
        edge_threshold=.05*request["k_ref"], modulation_threshold=.025*request["k_ref"],
        history_objective="event_equal_mean_scaled_residual_squared_plus_1e-12_mask_count",
        weak_force_scale=.1, local_residual_divisor=1e-8, fit_request_sha256=old._hash(request))
    return old._plain(result)


def validate_config(f, config):
    try:
        if not isinstance(config, dict) or config.get("format") != FORMAT:
            raise ValueError("complete fit297 initialization config required")
        expected = make_config(f, config["mode"], config["fit_request"], config["training_count"], config["preparation_key"])
        if old._hash(config) != old._hash(expected):
            raise ValueError("fit297 configuration or model fingerprint changed")
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise old.InitializationFailure(str(exc), dict(status="invalid_configuration", error=repr(exc))) from exc
    return copy.deepcopy(config)


def endpoint(packet, f, u, mode, role, config):
    state = old._endpoint(packet, f, u, mode, role)
    state[4*f.n_nodes:] = np.asarray(config["r_scale"])*np.asarray(u)[-f.n_nodes:]
    return state


def history_residual(f, predicted, packet, config):
    if np.shape(predicted) != (11, 5*f.n_nodes) or not np.isfinite(predicted).all():
        raise ValueError("finite eleven-point 5N history required")
    theta = predicted[:, :f.n_nodes]-packet["theta_history"]
    theta -= (theta@f.mass/f.mass.sum())[:, None]
    omega = predicted[:, f.n_nodes:2*f.n_nodes]-packet["omega_history"]
    return np.concatenate((theta/np.asarray(config["history_theta_scale"]),
                           omega/np.asarray(config["history_omega_scale"])), axis=1).ravel()


def initialize_complete(f, packet, config):
    config = validate_config(f, config)
    if float(packet.get("sigma", -1.)) != 0.:
        raise old.InitializationFailure("fit297 requires sigma zero", dict(status="invalid_observation"))
    return old._initialize_validated(f, packet, config)


def _counts():
    return dict(old._counts(), information_svd=0, conditional_projections=0, memory_svd=0,
                information_history_steps=0)


def _consume(key, ledger, counts):
    if ledger[key] >= _LIMITS[key]:
        raise _AnalysisLimit(key+" finite analysis limit reached")
    ledger[key] += 1
    counts[key] += 1


def analysis_budget_snapshot(preparation_key):
    ledger = _LEDGERS.get(preparation_key)
    if ledger is None:
        return dict(status="unknown", reason="preparation cache unavailable")
    return {key: dict(planned=limit, actual=ledger[key], skipped=limit-ledger[key]) for key, limit in _LIMITS.items()}


def _factors(system, ledger, counts):
    _consume("memory_svd", ledger, counts)
    U, s, Vt = np.linalg.svd(system["Z"], full_matrices=False)
    return dict(U=U, s=s, Vt=Vt)


def project_cached(systems, factors):
    """Use the same global cutoff as a block-diagonal SVD; normalize this scope."""
    if not systems or len(systems) != len(factors):
        raise ValueError("aligned nonempty cached event systems required")
    singular = np.concatenate([factor["s"] for factor in factors])
    cutoff = float(np.max(singular))*1e-10 if singular.size else 0.
    rows = sum(len(system["b"]) for system in systems)
    matrices, vectors, blocks = [], [], []
    for system, factor in zip(systems, factors):
        rank = int(np.sum(factor["s"] > cutoff))
        basis = factor["U"][:, :rank]
        matrices.append((system["A"]-basis@(basis.T@system["A"]))/np.sqrt(rows))
        vectors.append((system["b"]-basis@(basis.T@system["b"]))/np.sqrt(rows))
        blocks.append(dict(factor, memory_rank=rank))
    return dict(X=np.vstack(matrices), y=np.concatenate(vectors), blocks=blocks, total_rows=rows,
                memory_ranks=core.ranks(np.sort(singular)[::-1]))


def _information(X, memory_ranks, ledger, counts, *, event=False):
    _consume("information_svd", ledger, counts)
    singular = np.linalg.svd(X, compute_uv=False)
    norms = np.linalg.norm(X, axis=0)
    rank_info = core.ranks(singular)
    result = dict(column_norms=norms, singular_values=singular, ranks=rank_info,
        memory_ranks=memory_ranks, rows=X.shape[0], columns=X.shape[1],
        projected_dimension_bound=X.shape[0]-memory_ranks[str(1e-10)],
        dimension_insufficient=X.shape[0]-memory_ranks[str(1e-10)] < X.shape[1], event_is_qualification_gate=False)
    if event:
        return result
    correlation = np.divide(X.T@X, norms[:, None]*norms[None, :], out=np.zeros((X.shape[1], X.shape[1])), where=norms[:, None]*norms[None, :] > 0)
    conditional = []
    for column in range(X.shape[1]):
        _consume("conditional_projections", ledger, counts)
        _consume("information_svd", ledger, counts)
        U, s, _ = np.linalg.svd(np.delete(X, column, axis=1), full_matrices=False)
        other_ranks = core.ranks(s)
        rank = other_ranks[str(1e-10)]
        norm = float(np.linalg.norm(X[:, column]-U[:, :rank]@(U[:, :rank].T@X[:, column])))
        ratio = norm/norms[column] if norms[column] > 0 else 0.
        stable = all(len(set(group.values())) <= 1 for group in (rank_info, other_ranks, memory_ranks))
        qualified = norms[column] > 1e-6 and ratio > 1e-8 and stable and np.all(np.delete(np.abs(correlation[column]), column) < 1-1e-8)
        conditional.append(dict(norm=norm, ratio=float(ratio), other_ranks=other_ranks, qualified=bool(qualified)))
    result.update(conditional_columns=conditional, correlation=correlation,
                  columns_qualified=[row["qualified"] for row in conditional])
    return result


def _key(f, packets, request):
    return old._hash(dict(physics=old._physics(f, include_edges=False), packets=packets,
                         scales=request, grids=core.HISTORY_STEPS))


def _prepare(f, packets, request):
    key = _key(f, packets, request)
    if key in _PREPARATION_FAILURES:
        original = copy.deepcopy(_PREPARATION_FAILURES[key])
        raise old.FitFailure("same preparation already failed; replay prohibited", dict(
            status="preparation_failed_cached", counts=_counts(), preparation_key=key,
            preparation_cache_hit=True, original_preparation_failure=original,
            original_preparation_counts=original["counts"], preparation_cost_charged=False))
    if key in _CACHE:
        return copy.deepcopy(_CACHE[key]), True, key
    counts, ledger = _counts(), {key: 0 for key in _LIMITS}
    _LEDGERS[key] = ledger
    prepared = dict(grids={}, factors={}, counts=counts)
    try:
        for step in core.HISTORY_STEPS:
            grids, factors = [], []
            for packet in packets:
                counts["reconstruction_attempts"] += 1
                cache = core.prepare_weak_history(packet, f, step)
                counts["reconstructions"] += 1
                grids.append(cache)
                factors.append(_factors(cache, ledger, counts))
            prepared["grids"][step], prepared["factors"][step] = grids, factors
        static = [dict(A=cache["A_static"], Z=cache["Z"], b=cache["b"]) for cache in prepared["grids"][.0025]]
        projected = project_cached(static, prepared["factors"][.0025])
        info = _information(projected["X"], projected["memory_ranks"], ledger, counts)
        selected = old.backward_select(projected["X"], projected["y"], 0., np.ones(f.n_edges, dtype=bool),
            np.zeros(f.n_edges, dtype=bool), counts, k_ref=request["k_ref"])
        prepared.update(initial_information=info, initial_static=selected)
    except Exception as exc:
        failure = old._plain(dict(status="preparation_failed", counts=counts, error=repr(exc), preparation_key=key,
                                 preparation_cost_charged=True))
        _PREPARATION_FAILURES[key] = copy.deepcopy(failure)
        raise old.FitFailure("full preparation failed", failure) from exc
    _CACHE.clear()
    _CACHE[key] = copy.deepcopy(prepared)
    return prepared, False, key


def _candidate_systems(f, prepared, candidate, indices, step, counts):
    systems = []
    for index, event in zip(indices, candidate["events"]):
        cache = prepared["grids"][step][index]
        dense = event["best"]["dense_history"]
        if step != .0025:
            before = counts["rk4_steps_attempted"]
            try:
                dense = old._integrate(f, event["best"]["state"], step, .1, counts, backward=True, dense=True)
            finally:
                counts["information_history_steps"] += counts["rk4_steps_attempted"]-before
        systems.append(core.conditional_dictionary(cache, f, dense[:, 3*f.n_nodes:4*f.n_nodes], a_times=cache["t"]))
    return systems


def _project_candidate(systems, prepared, indices, step, mode):
    actual = systems if mode == "modulated" else [dict(system, A=system["A_static"]) for system in systems]
    return project_cached(actual, [prepared["factors"][step][index] for index in indices])


def _audit(f, prepared, candidate, indices, mode, key, counts):
    ledger, grids = _LEDGERS[key], {}
    try:
        for step in core.HISTORY_STEPS:
            systems = _candidate_systems(f, prepared, candidate, indices, step, counts)
            explanation = [dict(system, A=system["explanation"] if mode == "modulated" else system["A_static"]) for system in systems]
            factors = [prepared["factors"][step][index] for index in indices]
            projected = project_cached(explanation, factors)
            shared = _information(projected["X"], projected["memory_ranks"], ledger, counts)
            events = []
            for system, factor in zip(explanation, factors):
                local = project_cached([system], [factor])
                events.append(_information(local["X"], local["memory_ranks"], ledger, counts, event=True))
            grids[str(step)] = dict(shared=shared, events=events, total_rows=projected["total_rows"])
        main, fine = (grids[str(step)]["shared"] for step in core.HISTORY_STEPS)
        same = old._rank_signature(main) == old._rank_signature(fine)
        return dict(status="completed", grids=grids, indices=list(indices), rank_signatures_agree=same,
            k_qualified=bool(same and all(main["columns_qualified"][:f.n_edges]) and all(fine["columns_qualified"][:f.n_edges])),
            beta_qualified=bool(mode == "modulated" and same and all(main["columns_qualified"][f.n_edges:]) and all(fine["columns_qualified"][f.n_edges:])),
            beta_status="fixed_by_model" if mode == "static" else "conditional_only", joint_identifiability="unknown")
    except (_AnalysisLimit, ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return dict(status="unknown", reason=repr(exc), grids=grids, indices=list(indices),
                    k_qualified=False, beta_qualified=False, joint_identifiability="unknown")


def _compact(value):
    if isinstance(value, dict):
        return {key: _compact(item) for key, item in value.items() if key not in (
            "dense_history", "residual", "projected_X", "projected_y", "explanation_projected_X", "explanation_projected_y")}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def fit_shared(functional, validated_p_end_packets, *, mode, config):
    counts, begin = _counts(), time.perf_counter()
    details = dict(counts=counts, trace=[])
    try:
        f = copy.deepcopy(functional)
        packets = [core.validate_observation(packet, f.n_nodes) for packet in validated_p_end_packets]
        request = validate_fit_config(f, config, len(packets))
        if mode not in ("static", "modulated") or any(float(packet["sigma"]) != 0. for packet in packets):
            raise ValueError("registered mode and sigma zero required")
        prepared, hit, key = _prepare(f, packets, request)
        if not hit:
            old._add(counts, prepared["counts"])
        indices = request["adaptation_indices"]
        subset = [packets[index] for index in indices]
        selected = prepared["initial_static"]
        initial_identified = selected["status"] == "qualified" and not selected["threshold_inconsistent"] and all(prepared["initial_information"]["columns_qualified"])
        unresolved = selected["status"] == "solver_unresolved"
        if mode == "static" and initial_identified:
            k, beta, edges = selected["raw_k"], np.zeros(f.n_edges), selected["edge_mask"].copy()
        else:
            k = np.full(f.n_edges, request["k_ref"])
            beta = np.full(f.n_edges, .25*request["k_ref"] if mode == "modulated" else 0.)
            edges = np.ones(f.n_edges, dtype=bool)
        mods = np.full(f.n_edges, mode == "modulated")
        current = old._parameters(f, k, beta)
        init_config = make_config(current, mode, request, len(packets), key)
        incumbent = old._proxy_candidate(current, subset, mode, edges, mods, config=init_config)
        old._add(counts, incumbent["counts"])
        details.update(initial_candidate=incumbent, initial_static=selected)
        if incumbent["status"] != "completed":
            raise old.FitFailure("no legal initial subset candidate", old._plain(details))
        prior_only = not (mode == "static" and initial_identified)
        edge_bad, mod_bad = False, False
        for iteration in range(3):
            record = dict(iteration=iteration, accepted=False, incumbent_objective=incumbent["objective"])
            details["trace"].append(record)
            planned = counts["algebra_terms_planned"]
            try:
                systems = _candidate_systems(current, prepared, incumbent, indices, .0025, counts)
                projected = _project_candidate(systems, prepared, indices, .0025, mode)
                proposal = old.backward_select(projected["X"], projected["y"], 0., edges, mods, counts, k_ref=request["k_ref"])
                record["coefficients"] = proposal
                if proposal["status"] != "qualified":
                    record["reason"] = proposal["status"]
                    continue
                try:
                    proposed = old._parameters(f, proposal["raw_k"], proposal["raw_beta"])
                except ValueError as exc:
                    record.update(reason="parameter_infeasible", error=repr(exc),
                                  proxy_skipped=dict(tasks=2*len(indices), reason="parameter_infeasible"))
                    continue
                candidate = old._proxy_candidate(proposed, subset, mode, proposal["edge_mask"], proposal["mod_mask"],
                    config=make_config(proposed, mode, request, len(packets), key))
                old._add(counts, candidate["counts"])
                record["candidate"] = candidate
                if candidate["status"] != "completed":
                    record["reason"] = "candidate_no_output"
                    continue
                coef = proposal["debias"]["coefficients"]
                record["weak_before_reinference"] = float(np.sum((projected["X"]@coef-projected["y"])**2))
                updated_systems = _candidate_systems(proposed, prepared, candidate, indices, .0025, counts)
                updated = _project_candidate(updated_systems, prepared, indices, .0025, mode)
                record["weak_after_reinference"] = float(np.sum((updated["X"]@coef-updated["y"])**2))
                if old._improves(candidate["objective"], incumbent["objective"]):
                    incumbent, current = candidate, proposed
                    edges, mods = proposal["edge_mask"].copy(), proposal["mod_mask"].copy()
                    edge_bad, mod_bad = proposal["edge_threshold_inconsistent"], proposal["mod_threshold_inconsistent"]
                    prior_only = False
                    record.update(accepted=True, reason="strict_subset_history_improvement")
                else:
                    record["reason"] = "no_improvement"
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                skipped = 0
                if counts["algebra_terms_planned"] == planned:
                    skipped = 2*f.n_edges*(2*f.n_edges+1)+1
                    counts["algebra_terms_planned"] += skipped
                    counts["algebra_terms_skipped"] += skipped
                record.update(reason="parameter_nonfinite", error=repr(exc), skipped_algebra=skipped)
        info = _audit(current, prepared, incumbent, indices, mode, key, counts)
        support = np.flatnonzero(edges).tolist() if info["k_qualified"] and not edge_bad and not prior_only and not unresolved else None
        mod_support = np.flatnonzero(mods).tolist() if info["beta_qualified"] and not edge_bad and not mod_bad and not prior_only and not unresolved else None
        details["selected_proxy"] = incumbent
        private = old._plain(details)
        state = dict(status="completed" if support is not None and (mode == "static" or mod_support is not None) else "not_identified",
            counts=counts, preparation_key=key, preparation_cache_hit=hit, shared_preparation_counts=prepared["counts"],
            adaptation_indices=indices, training_count=len(packets), adaptation_count=len(indices),
            initial_information=prepared["initial_information"], information=info, support=support, modulation_support=mod_support,
            raw_edge_mask=edges, raw_mod_mask=mods, k=current.k_edge, beta=current.k_edge*current.gates.s_mod,
            prior_only=prior_only, from_data=not prior_only, preparation_numerically_unresolved=unresolved,
            edge_threshold_inconsistent=edge_bad, mod_threshold_inconsistent=mod_bad,
            accepted_proposal_indices=[row["iteration"] for row in details["trace"] if row["accepted"]],
            null_reasons=dict(k=[] if support is not None else ["prior_threshold_solver_or_conditional_information"],
                             beta=["fixed_by_model"] if mode == "static" else [] if mod_support is not None else ["prior_threshold_solver_or_conditional_information"]),
            trace=_compact(private["trace"]), private_details_sha256=old._hash(private),
            selected_objective=incumbent["objective"], analysis_budget=analysis_budget_snapshot(key),
            joint_identifiability="unknown", wall_seconds=time.perf_counter()-begin)
        return dict(functional=current, config=make_config(current, mode, request, len(packets), key),
                    fit_state=old._plain(state), private_details=private, status=state["status"])
    except old.FitFailure:
        raise
    except (ValueError, TypeError, FloatingPointError, OverflowError, RuntimeError) as exc:
        raise old.FitFailure(str(exc), old._plain(dict(details, error=repr(exc), wall_seconds=time.perf_counter()-begin))) from exc


def coverage_information(functional, validated_p_end_packets, states, config):
    config = validate_config(functional, config)
    counts, begin = _counts(), time.perf_counter()
    packets = [core.validate_observation(packet, functional.n_nodes) for packet in validated_p_end_packets]
    key = _key(functional, packets, config["fit_request"])
    if key != config["preparation_key"] or key not in _CACHE:
        return dict(status="unknown", reason="matching preparation cache unavailable", counts=counts)
    if len(states) != len(packets):
        raise ValueError("one state or None per original legal input required")
    indices, events = [], []
    try:
        for index, state in enumerate(states):
            if state is None:
                continue
            functional.split_state(state)
            if not np.array_equal(np.asarray(state)[2*functional.n_nodes:3*functional.n_nodes], packets[index]["p_end"]):
                raise ValueError("coverage endpoint p must be exact")
            before = counts["rk4_steps_attempted"]
            try:
                dense = old._integrate(functional, state, .0025, .1, counts, backward=True, dense=True)
            finally:
                counts["information_history_steps"] += counts["rk4_steps_attempted"]-before
            indices.append(index)
            events.append(dict(best=dict(state=np.asarray(state), dense_history=dense)))
    except (ValueError, FloatingPointError, OverflowError, RuntimeError, np.linalg.LinAlgError) as exc:
        return old._plain(dict(status="unknown", reason=repr(exc), counts=counts, indices=indices,
            failure_index=index, preparation_key=key, scope="deployment_information_only",
            analysis_budget=analysis_budget_snapshot(key), wall_seconds=time.perf_counter()-begin))
    if not indices:
        return dict(status="unknown", reason="no successful initialized points", indices=[], counts=counts)
    info = _audit(functional, _CACHE[key], dict(events=events), indices, config["mode"], key, counts)
    return old._plain(dict(info, scope="deployment_successful_only_not_parameter_selection", counts=counts,
                          preparation_key=key, excluded_indices=[i for i, state in enumerate(states) if state is None],
                          analysis_budget=analysis_budget_snapshot(key), wall_seconds=time.perf_counter()-begin))
