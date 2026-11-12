"""Registered continuation from the two fixed parents on the 24 train slots."""
from __future__ import annotations

import copy
import math
import re
import time

import numpy as np

from . import gftd_discovery as core
from . import gftd_fit297 as fit297
from . import gftd_fitting as old
from ..physics.kuramoto_functional import BOUNDED_MODULATION_KERNEL, EXP_MODULATION_KERNEL

FORMAT = "gftd_matched_prediction_v1"
RUN_ID = "gftd_matched_prediction_20261109"
PROTOCOL_SHA256 = "0816203b8bcf37f506f0bbbb8e0fac531a747707c2d0f3612340b9ff69efbb43"
INPUT_SHA256 = "b2607fc66c9bb96c5d6f85a6e074598e186ffab377c8aac3009699e1ea035817"
EVENT_ORDER_SHA256 = "9a76d08d0423302a2b07374e9f2a92646db3be089cadeebaadace5b985699de4"
PARENTS = {
    "static": (EXP_MODULATION_KERNEL, "ed79e2cdea1f6830c052b9dae8fb8c2b748170e8e15ec0a5a8e3034d72b787cc"),
    "modulated": (BOUNDED_MODULATION_KERNEL, "f8eabacfeac00074f135c7af826d8d89d8f23c445c788485d0fb79a0f54b0637"),
}
TRAINING_COUNT = 297
ADAPTATION_INDICES = tuple(j * TRAINING_COUNT // 24 for j in range(24))
PROPOSALS = 3
START_CALLS = 1200
NUMERICAL_ERRORS = (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError)


def validate_request(f, request, mode, count):
    keys = {"format", "run_id", "protocol_sha256", "parent_model_sha256", "kernel", "mode", "training_count",
            "adaptation_slots", "parent_fit297_config", "source_sha256", "input_sha256", "event_order_sha256"}
    if not isinstance(request, dict) or set(request) != keys:
        raise ValueError("complete matched prediction request required")
    if (request["format"], request["run_id"], request["protocol_sha256"],
        request["input_sha256"], request["event_order_sha256"]) != (
        FORMAT, RUN_ID, PROTOCOL_SHA256, INPUT_SHA256, EVENT_ORDER_SHA256
    ):
        raise ValueError("matched prediction run or input identity changed")
    if (mode not in PARENTS or request["mode"] != mode
            or (request["kernel"], request["parent_model_sha256"]) != PARENTS[mode]
            or f.modulation_kernel != request["kernel"]):
        raise ValueError("matched prediction parent, kernel or mode changed")
    if (type(request["training_count"]) is not int or request["training_count"] != TRAINING_COUNT
            or count != TRAINING_COUNT or f.n_nodes != 10 or f.n_edges != 45):
        raise ValueError("matched prediction requires the original 297 K10 packets")
    slots = request["adaptation_slots"]
    if not isinstance(slots, list) or len(slots) != len(ADAPTATION_INDICES):
        raise ValueError("24 ordered adaptation slots required")
    for local_slot, (global_index, slot) in enumerate(zip(ADAPTATION_INDICES, slots)):
        if (not isinstance(slot, dict) or set(slot) != {"global_index", "local_slot", "event_id"}
                or type(slot["global_index"]) is not int or slot["global_index"] != global_index
                or type(slot["local_slot"]) is not int or slot["local_slot"] != local_slot
                or not isinstance(slot["event_id"], str) or not slot["event_id"].strip()):
            raise ValueError("invalid registered global-to-local mapping")
    if len({slot["event_id"] for slot in slots}) != len(slots):
        raise ValueError("adaptation event IDs must be unique")
    sources = request["source_sha256"]
    if not isinstance(sources, dict) or not sources or any(
        not isinstance(path, str) or not path.strip() or not isinstance(digest, str)
        or re.fullmatch("[0-9a-f]{64}", digest) is None for path, digest in sources.items()
    ):
        raise ValueError("frozen source closure hashes required")
    parent = fit297.validate_config(f, request["parent_fit297_config"])
    if parent["mode"] != mode or parent["training_count"] != count:
        raise ValueError("parent initialization mode or scope changed")
    if mode == "static" and np.any(f.k_edge * f.gates.s_edge * f.gates.s_mod != 0):
        raise ValueError("static parent must have zero effective modulation")
    return copy.deepcopy(request)


def _counts():
    return dict(old._counts(old.dop853_step), proxy_rhs_calls=0, qualification_suites=0,
                qualification_rhs_calls=0, diagnostic_residual_calls=0, diagnostic_rhs_calls=0,
                information_rhs_calls=0, port_rhs_calls=0, memory_svd=0, information_svd=0,
                conditional_projections=0)


def _prepare(f, packets, counts, ledger):
    grids = {}

    def count_port_rhs():
        counts["port_rhs_calls"] += 1

    for step in core.HISTORY_STEPS:
        caches, factors = [], []
        for packet in packets:
            counts["reconstruction_attempts"] += 1
            cache = core.prepare_weak_history(packet, f, step, on_port_rhs=count_port_rhs)
            counts["reconstructions"] += 1
            caches.append(cache)
            factors.append(fit297._factors(cache, ledger, counts))
        grids[step] = caches, factors
    return grids


def _eligible(task):
    best = task["best"]
    return (task["status"] in ("converged", "budget_limited") and best is not None
            and all(np.isfinite(best[key]).all() for key in ("u", "state", "history_mse", "residual", "dense_history")))


def _candidate(f, packets, slots, mode, config, counts, record, *, edge_mask, mod_mask):
    events = record["events"] = []
    record.update(status="in_progress", objective=None, model_sha256=config["model_sha256"],
                  edge_mask=edge_mask.copy(), mod_mask=mod_mask.copy())
    for packet, slot in zip(packets, slots):
        event = dict(slot, tasks=[], selected=None, best=None, qualification=None, qualification_status="not_run")
        events.append(event)
        slices, low, _ = old._layout(f.n_nodes, 0., mode, "proxy")
        zero, alternate = np.zeros_like(low), np.zeros_like(low)
        alternate[slices[2] if mode == "modulated" else slices[3]] = (
            (.2 if mode == "modulated" else .1) * (-1.) ** np.arange(f.n_nodes)
        )
        for start in (zero, alternate):
            task = old.infer_endpoint(f, packet, mode=mode, role="proxy", call_limit=START_CALLS,
                                      initial=start, config=config, stepper=old.dop853_step)
            old._add(counts, task["counts"])
            counts["proxy_rhs_calls"] += task["counts"]["actual_rhs_calls"]
            event["tasks"].append(task)
        eligible = [i for i, task in enumerate(event["tasks"]) if _eligible(task)]
        if eligible:
            selected = min(eligible, key=lambda i: event["tasks"][i]["best"]["history_mse"])
            best = event["tasks"][selected]["best"]
            event.update(selected=selected, best=best, endpoint_sha256=old._hash(best["state"]))
    if any(event["best"] is None for event in events):
        record["status"] = "no_output"
        return
    record["objective"] = (float(np.mean([event["best"]["history_mse"] for event in events]))
                           + old._penalty(0., edge_mask, mod_mask))
    if not math.isfinite(record["objective"]):
        record.update(status="no_output", objective=None)
        return
    for packet, event in zip(packets, events):
        capture = event["qualification_arrays"] = {}
        counts["qualification_suites"] += 1
        check = old.numerical_check(f, event["best"]["state"], packet, stepper=old.dop853_step, capture=capture)
        event["qualification"] = check
        old._add(counts, check["counts"])
        counts["qualification_rhs_calls"] += check["counts"]["actual_rhs_calls"]
        event["qualification_status"] = "passed" if check["passed"] else "failed"
        event["qualification_array_sha256"] = {leg: old._hash(array) for leg, array in capture.items()}
    record["status"] = ("completed" if all(event["qualification"]["passed"] for event in events)
                        else "numerical_failure")


def _system(f, cache, dense, mode, *, explanation=False):
    if mode == "static":
        return dict(A=cache["A_static"], b=cache["b"], Z=cache["Z"])
    system = core.conditional_dictionary(cache, f, dense[:, 3*f.n_nodes:4*f.n_nodes], a_times=cache["t"])
    return dict(system, A=system["explanation"]) if explanation else system


def _audit(f, prepared, candidate, mode, ledger, counts):
    grids = {}
    for step, (caches, factors) in prepared.items():
        events, systems = [], []
        for cache, factor, event in zip(caches, factors, candidate["events"]):
            row = dict(local_slot=event["local_slot"], status="unknown")
            events.append(row)
            try:
                dense = event["best"]["dense_history"]
                if step != .0025:
                    before = counts["actual_rhs_calls"]
                    try:
                        dense = old._integrate(f, event["best"]["state"], step, .1, counts,
                                               backward=True, dense=True, stepper=old.dop853_step)
                    finally:
                        counts["information_rhs_calls"] += counts["actual_rhs_calls"] - before
                system = _system(f, cache, dense, mode, explanation=True)
                systems.append(system)
                local = fit297.project_cached([system], [factor])
                row.update(status="completed", information=fit297._information(
                    local["X"], local["memory_ranks"], ledger, counts, event=True))
            except NUMERICAL_ERRORS as exc:
                row.update(status="unknown", error=repr(exc))
        grid = grids[str(step)] = dict(shared=None, events=events)
        if len(systems) == len(caches):
            try:
                shared = fit297.project_cached(systems, factors)
                grid["shared"] = fit297._information(shared["X"], shared["memory_ranks"], ledger, counts)
            except NUMERICAL_ERRORS as exc:
                grid["error"] = repr(exc)
    main, fine = (grids[str(step)]["shared"] for step in core.HISTORY_STEPS)
    same = None if main is None or fine is None else old._rank_signature(main) == old._rank_signature(fine)
    qualified = None if same is None else [
        same and a and b for a, b in zip(main["columns_qualified"], fine["columns_qualified"])
    ]
    return dict(status="unknown" if same is None else "completed", grids=grids, rank_signatures_agree=same,
                columns_qualified=qualified, joint_identifiability="unknown")


def _adequacy(residual, u, raw, low, high):
    """Evaluate the registered observation-scale diagnostics at the retained point."""
    records, projected = [], []
    for h in (2.5e-5, 1.25e-5):
        row = dict(h=h, status="unknown")
        records.append(row)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                jac = old._jacobian(residual, u, low, high, h=h)
                g = jac.T @ raw
                norms = np.linalg.norm(jac, axis=0)
                denominator = np.linalg.norm(jac) * np.linalg.norm(raw)
                if not all(np.isfinite(x).all() for x in (jac, g, norms, denominator)):
                    raise FloatingPointError("nonfinite diagnostic column, gradient or norm")
                g[(u <= low + 1e-7) & (g >= 0)] = 0.
                g[(u >= high - 1e-7) & (g <= 0)] = 0.
                projected.append(g)
                row.update(status="completed", column_norms=norms, projected_gradient=g,
                           eta=float(np.max(np.abs(g)) / max(denominator, 1e-16)))
        except NUMERICAL_ERRORS as exc:
            row["error"] = repr(exc)
            projected.append(None)
    a, b = projected
    trials = [dict(step=step, status="not_run", relative_reduction=None, improved=None) for step in (.01, .05)]
    out = dict(steps=records, trials=trials, d_g=None, locally_consistent=None, small_relative_gradient=None)
    if b is None:
        return out
    b_norm = np.max(np.abs(b))
    if a is not None:
        a_norm = np.max(np.abs(a))
        disagreement = float(np.max(np.abs(a - b)) / max(a_norm, b_norm, 1e-16))
        out.update(d_g=disagreement, locally_consistent=disagreement <= .1)
    out["small_relative_gradient"] = records[1]["eta"] <= 1e-3
    sse = float(raw @ raw)
    for trial in trials:
        if b_norm == 0:
            trial["status"] = "zero_direction"
            continue
        point = np.clip(u - trial["step"] * (b / b_norm), low, high)
        if np.array_equal(point, u):
            trial["status"] = "unchanged_point"
            continue
        try:
            value = residual(point)
            trial_sse = float(value @ value)
            if not math.isfinite(trial_sse):
                raise FloatingPointError("nonfinite diagnostic SSE")
            difference = sse - trial_sse
            reduction = difference / max(sse, 1e-16)
            trial.update(status="completed", sse=trial_sse, sse_change=-difference,
                         relative_reduction=reduction, improved=reduction > 1e-6)
        except NUMERICAL_ERRORS as exc:
            trial.update(status="unknown", error=repr(exc))
    return out


def _diagnose(f, packets, candidate, mode, config, counts):
    results = []
    for packet, event in zip(packets, candidate["events"]):
        task = event["tasks"][event["selected"]]
        best = event["best"]
        start_calls, start_rhs = counts["diagnostic_residual_calls"], counts["actual_rhs_calls"]

        def residual(u):
            counts["diagnostic_residual_calls"] += 1
            state = fit297.endpoint(packet, f, u, mode, "proxy", config)
            dense = old._integrate(f, state, .0025, .1, counts, backward=True, dense=True, stepper=old.dop853_step)
            return fit297.history_residual(f, dense[::4], packet, config)

        try:
            result = _adequacy(residual, best["u"], best["residual"], task["lower_bounds"], task["upper_bounds"])
        finally:
            counts["diagnostic_rhs_calls"] += counts["actual_rhs_calls"] - start_rhs
        results.append(dict(result, local_slot=event["local_slot"], optimizer_status=task["status"],
                            optimizer_same_point_gradient=task["projected_gradient_kkt"] is not None,
                            active_bounds=task["active_bounds"], residual_calls=counts["diagnostic_residual_calls"]-start_calls,
                            observation_sse=best["observation_sse"]))
    return results


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def fit_shared(functional, validated_packets, *, mode, config):
    """Run only the registered shared adaptation and final historical diagnostics."""
    wall, cpu = time.perf_counter(), time.process_time()
    counts, candidates, trace = _counts(), [], []
    details = dict(status="invalid_request", counts=counts, cost_unknown=False, candidates=candidates, trace=trace)
    try:
        request = validate_request(functional, config, mode, len(validated_packets))
        f = copy.deepcopy(functional)
        if any(float(packet["sigma"]) != 0. for packet in validated_packets):
            raise ValueError("matched prediction requires sigma zero")
        slots = request["adaptation_slots"]
        packets = [core.validate_observation(validated_packets[slot["global_index"]], f.n_nodes) for slot in slots]
        parent = request["parent_fit297_config"]
        fit_request, old_key = parent["fit_request"], parent["preparation_key"]
        key = old._hash(dict(request=request, packets=packets, steps=core.HISTORY_STEPS,
                             coefficients_sha256=old.FIXED_DP8_COEFFICIENTS_SHA256))
        details.update(request=request, preparation_key=key, status="preparing")
        old.make_fixed_dp8_config(f, parent)
        ledger = dict.fromkeys(("memory_svd", "information_svd", "conditional_projections"), 0)
        prepared = _prepare(f, packets, counts, ledger)
        incumbent = {}
        candidates.append(incumbent)
        details["status"] = "fitting"
        c = f.k_edge * f.gates.s_edge
        _candidate(f, packets, slots, mode, parent, counts, incumbent,
                   edge_mask=c > 0, mod_mask=c * f.gates.s_mod > 0)
        if incumbent["status"] != "completed":
            details["status"] = "fit_failed"
            raise ValueError("initial parent proxy has no numerically qualified candidate")
        selected_index, threshold_bad = 0, True
        current, init_config = f, parent
        for iteration in range(PROPOSALS):
            row = dict(iteration=iteration, accepted=False, candidate_index=None)
            trace.append(row)
            planned = counts["algebra_terms_planned"]
            try:
                caches, factors = prepared[.0025]
                systems = [_system(current, cache, event["best"]["dense_history"], mode)
                           for cache, event in zip(caches, incumbent["events"])]
                projected = fit297.project_cached(systems, factors)
                proposal = old.backward_select(
                    projected["X"], projected["y"], 0., np.ones(f.n_edges, dtype=bool),
                    np.full(f.n_edges, mode == "modulated"), counts, k_ref=fit_request["k_ref"])
                row["proposal"] = proposal
                if proposal["status"] != "qualified":
                    row["reason"] = proposal["status"]
                    continue
                proposed = old._parameters(f, proposal["raw_k"], proposal["raw_beta"])
                proposal_config = fit297.make_config(proposed, mode, fit_request, TRAINING_COUNT, old_key)
            except NUMERICAL_ERRORS as exc:
                if counts["algebra_terms_planned"] == planned:
                    skipped = 2*f.n_edges*(2*f.n_edges+1)+1
                    counts["algebra_terms_planned"] += skipped
                    counts["algebra_terms_skipped"] += skipped
                row.update(reason="proposal_unresolved", error=repr(exc))
                continue
            candidate = {}
            row["candidate_index"] = len(candidates)
            candidates.append(candidate)
            _candidate(proposed, packets, slots, mode, proposal_config, counts, candidate,
                       edge_mask=proposal["edge_mask"], mod_mask=proposal["mod_mask"])
            if candidate["status"] != "completed":
                row["reason"] = candidate["status"]
            elif old._improves(candidate["objective"], incumbent["objective"]):
                selected_index, threshold_bad = row["candidate_index"], proposal["threshold_inconsistent"]
                current, init_config, incumbent = proposed, proposal_config, candidate
                row.update(accepted=True, reason="strict_subset_history_improvement")
            else:
                row["reason"] = "no_improvement"
        adequacy = _diagnose(current, packets, incumbent, mode, init_config, counts)
        information = _audit(current, prepared, incumbent, mode, ledger, counts)
        qualified = information["columns_qualified"]
        c = current.k_edge * current.gates.s_edge
        beta = c * current.gates.s_mod
        support = (np.flatnonzero(c > 0) if not threshold_bad and qualified is not None
                   and all(qualified[:f.n_edges]) else None)
        modulation_support = (np.flatnonzero(beta > 0) if support is not None and mode == "modulated"
                              and all(qualified[f.n_edges:]) else None)
        status = "completed" if support is not None and (mode == "static" or modulation_support is not None) else "not_identified"
        details.update(status=status, selected_candidate_index=selected_index)
        private = _json_value(details)
        fit_state = _json_value(dict(
            format=FORMAT, status=status, run_id=RUN_ID, protocol_sha256=PROTOCOL_SHA256, request_sha256=old._hash(request),
            kernel=request["kernel"], mode=mode, parent_model_sha256=request["parent_model_sha256"],
            preparation_key=key, parent_preparation_key=old_key, adaptation_slots=slots, training_count=TRAINING_COUNT,
            strategy=dict(proxy_start_calls=START_CALLS, proxy_starts=2, proposals=PROPOSALS,
                          full_library=True, history_stepper="fixed_dp8", qualification_suites_per_candidate_packet=1),
            counts=counts, selected_candidate_index=selected_index, selected_objective=incumbent["objective"],
            candidate_statuses=[record["status"] for record in candidates],
            accepted_proposal_indices=[row["iteration"] for row in trace if row["accepted"]],
            k=c, beta=beta, support=support, modulation_support=modulation_support,
            adequacy=adequacy, information=information, joint_identifiability="unknown",
            private_details_sha256=old._hash(private),
            wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu))
        return dict(functional=current, config=old.make_fixed_dp8_config(current, init_config),
                    fit_state=fit_state, private_details=private, status=status)
    except Exception as exc:
        if details["status"] == "preparing":
            details["status"] = "preparation_failed"
            details["cost_unknown"] = not isinstance(exc, NUMERICAL_ERRORS)
        elif details["status"] not in ("invalid_request", "fit_failed"):
            details["status"] = "execution_unresolved"
            details["cost_unknown"] = True
        raise old.FitFailure(str(exc), _json_value(dict(details, error=repr(exc),
                             wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu))) from exc
