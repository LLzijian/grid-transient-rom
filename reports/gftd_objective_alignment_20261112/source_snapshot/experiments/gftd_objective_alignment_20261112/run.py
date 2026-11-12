"""Isolated, released three-point history comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

if __name__ in ("__main__", "__mp_main__"):
    for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[thread_variable] = "1"

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from experiments.gftd_matched_prediction_20261109 import run as evidence
from kiki.models import gftd_discovery as core, gftd_fit297 as fit297
from kiki.models import gftd_fitting as fitting, gftd_matched_prediction as matched
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates

RUN = "gftd_objective_alignment_20261112"
RESULTS = Path("results") / RUN
PRIVATE = Path("artifacts") / RUN
PROTOCOL = Path("campaigns") / RUN / "protocol.md"
PROTOCOL_SHA = "18fe52a0d766c3bbe8e9e8d4941d4ef15a633c549d47d58f9e3d303841944c4c"
ADOPTION_SHA = "71ac3598e9dabcdf006da13980b007bcc35dedec69dcf0d2348ab5a63effce59"
ENDPOINT_SHA = "ebc92ecdfbff3a182948569fc8ae3f60ed6dad458757bcafe5835824e23c07e6"
MODES = {"static": "exp_v1", "modulated": "bounded_exp_tanh_v1"}
POINTS = {"half": .5, "quarter": .25, "eighth": .125}
SLOTS = tuple(j * 297 // 24 for j in range(24))
KEYS = {
    "static": "663a1882ed49bc5680b6fe11ca9ceef2383f43ab02a5363226a2bfa480a49107",
    "modulated": "2f20dddccf309fec0eb45ca3b4ca22fc4c6c0ecce5a6e623c87912864be3660a",
}
NEW_SOURCES = {f"experiments/{RUN}/{name}.py" for name in ("run", "qualify")}
LIMITS = dict(optimization_tasks=288, actual_residual_calls=345600, proxy_rhs_calls=165888000,
              qualification_suites=144, qualification_rhs_calls=933120,
              witness_residual_calls=144, witness_rhs_calls=69120, actual_rhs_calls=166890240)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_json(ref, path):
    return json.loads(evidence.read_ref(ref, path))


def release_identity(phase, release, sources, runtime):
    require(release.get("run") == RUN and release.get("phase") == phase
            and release.get("released") is True and isinstance(release.get("action_id"), str)
            and bool(release["action_id"]), "explicit stage release required")
    require(release.get("protocol_sha256") == PROTOCOL_SHA
            and release.get("design_adoption_sha256") == ADOPTION_SHA
            and release.get("endpoint_digest_sha256") == ENDPOINT_SHA, "release identity changed")
    require(release.get("source_sha256") == sources and release.get("runtime") == runtime,
            "frozen source or runtime differs")


def frozen_context(phase, release):
    adoption = json.loads(evidence.read_bound(RESULTS / "design_adoption.json", ADOPTION_SHA))
    require(adoption["status"] == "design_accepted_for_minimal_implementation", "design adoption missing")
    for key in ("design", "protocol", "review", "endpoint_digest"):
        ref = adoption["references"][key]
        evidence.read_ref(ref, Path(ref["path"]))
    require(adoption["references"]["protocol"]["sha256"] == PROTOCOL_SHA
            and adoption["references"]["endpoint_digest"]["sha256"] == ENDPOINT_SHA,
            "design adoption bindings changed")
    endpoint = json.loads(evidence.read_bound(RESULTS / "endpoint_identity_digest.json", ENDPOINT_SHA))
    old = bound_json(endpoint["references"]["prepare"], evidence.RESULTS / "prepare_1.json")
    old_sources = old["source_sha256"]
    require(len(old_sources) == 43 and endpoint["frozen_source_count"] == 43
            and all(old_sources.get(path) == digest for path, digest in endpoint["core_source_sha256"].items()),
            "old core closure differs")
    sources = dict(old_sources)
    for path in NEW_SOURCES:
        sources[path] = evidence.sha(evidence.local_path(path).read_bytes())
    evidence.verify_sources({"source_sha256": sources})
    runtime = evidence.runtime()
    require(runtime == old["runtime"] == endpoint["runtime"], "old numerical runtime differs")
    release_identity(phase, release, sources, runtime)
    return endpoint, old


def claim(path, identity):
    return evidence.save_json(path, identity, exclusive=True)


def restore_physics(physics):
    return GeneratingFunctionalKuramoto(**{k: v for k, v in physics.items() if k != "gates"},
                                        gates=StructuralGates(**physics["gates"]))


def effective(functional, mode):
    gates = functional.gates
    require(functional.modulation_kernel == MODES[mode] and gates.s_coi == 0
            and np.all((0 <= gates.s_mod) & (gates.s_mod <= gates.s_edge) & (gates.s_edge <= 1))
            and np.all(np.isfinite(gates.s_mem)) and np.all((0 <= gates.s_mem) & (gates.s_mem <= 1)),
            "kernel or gate identity changed")
    c = functional.k_edge * gates.s_edge
    beta = c * gates.s_mod
    require(np.all(np.isfinite(c)) and np.all(np.isfinite(beta))
            and np.all((0 <= beta) & (beta <= c))
            and (mode != "static" or not np.any(beta)), "invalid effective coefficients")
    return c, beta


def check_masks(effective_values, masks):
    require(all(np.array_equal(v > 0, mask) for v, mask in zip(effective_values, masks, strict=True)),
            "saved mask differs from actual support")


def same_config(actual, original, fingerprint):
    require(actual == {**original, "model_sha256": fingerprint}, "non-model configuration changed")


def verify_slots(slots, actual):
    require(len(slots) == len(actual) == 24
            and tuple(s["global_index"] for s in slots) == SLOTS
            and tuple(s["local_slot"] for s in slots) == tuple(range(24))
            and len({s["event_id"] for s in slots}) == 24
            and all(all(item.get(k) == v for k, v in slot.items())
                    for slot, item in zip(slots, actual, strict=True)), "ordered 24 slots differ")


def checked_mode_cache(endpoint, slots, packets, mode):
    item = endpoint["models"][mode]
    references = item["references"]
    job = bound_json(references["job"], evidence.RESULTS / "fit/jobs" / f"{mode}.json")
    base = evidence.ARTIFACTS / "fit" / mode
    model = bound_json(references["final_model"], base / "model.json")
    details = json.loads(gzip.decompress(evidence.read_ref(references["details"], base / "fit_details.json.gz")))
    request = details["request"]
    require(job["status"] == "completed" and job["cost_unknown"] is False
            and job["final_model"] == references["final_model"] and job["details"] == references["details"]
            and job["selected_candidate_index"] == details["selected_candidate_index"] == 0
            and job["accepted_proposal_indices"] == [] and model["fit_mode"] == mode
            and job["prepare_sha256"] == endpoint["references"]["prepare"]["sha256"]
            and fitting._hash(details) == model["fit_state"]["private_details_sha256"], "old fit evidence differs")
    verify_slots(slots, request["adaptation_slots"])
    aggregate = fitting._hash(dict(request=request, packets=packets, steps=core.HISTORY_STEPS,
                                   coefficients_sha256=fitting.FIXED_DP8_COEFFICIENTS_SHA256))
    require(aggregate == KEYS[mode] == details["preparation_key"] == job["preparation_key"]
            == item["preparation_key"], "old matched packet aggregation differs")
    physics = {**model["functional"], "gates": model["gates"]}
    parent = restore_physics(physics)
    parent_config = request["parent_fit297_config"]
    require(model["inference_config"]["fit297_config"] == parent_config
            and model["inference_config"]["method"] == "fixed_dp8"
            and model["inference_config"]["coefficients_sha256"] == fitting.FIXED_DP8_COEFFICIENTS_SHA256
            and fitting._hash(parent_config) == item["parent_config_sha256"], "old inner configuration differs")
    fit297.validate_config(parent, parent_config)
    raw = details["trace"][0]["proposal"]
    proposal = fitting._parameters(parent, raw["raw_k"], raw["raw_beta"])
    require(fitting._hash(fitting._physics(parent, include_edges=False))
            == fitting._hash(fitting._physics(proposal, include_edges=False))
            == item["non_edge_canonical_json_sha256"], "old non-edge fields differ")
    proposal_fp = fitting._hash(fitting._physics(proposal))
    proposal_config = fit297.make_config(proposal, mode, parent_config["fit_request"], 297,
                                         parent_config["preparation_key"])
    same_config(proposal_config, parent_config, proposal_fp)
    require(fitting._hash(proposal_config) == item["proposal_config_sha256"], "old proposal config differs")
    objectives, coefficients = [], []
    eps = np.finfo(np.float64).eps
    for index, functional in enumerate((parent, proposal)):
        candidate = details["candidates"][index]
        frozen = item["endpoints"][f"t{index}"]
        fp = fitting._hash(fitting._physics(functional))
        require(candidate["status"] == "completed" and fp == candidate["model_sha256"]
                == frozen["functional_fingerprint_sha256"], "old endpoint fingerprint differs")
        c, beta = effective(functional, mode)
        check_masks((c, beta), (candidate["edge_mask"], candidate["mod_mask"]))
        for name, values in (("c", c), ("beta", beta), ("x", c-beta)):
            require(evidence.sha(values.astype("<f8").tobytes()) == frozen["effective_coordinate_sha256_le_float64"][name],
                    "old endpoint coefficients differ")
        verify_slots(slots, candidate["events"])
        opened = int(np.count_nonzero(c) + np.count_nonzero(beta))
        mean_mse = float(np.mean([event["best"]["history_mse"] for event in candidate["events"]]))
        require(opened == frozen["open_terms"] and candidate["objective"] == frozen["evaluated_objective"]
                and abs(mean_mse + 1e-12 * opened - candidate["objective"]) <= 8*eps*max(1., abs(candidate["objective"])),
                "old endpoint objective differs")
        objectives.append(dict(objective=candidate["objective"], mean_history_mse=mean_mse,
                               open_terms=opened, fingerprint=fp))
        coefficients.append(dict(c=c.tolist(), beta=beta.tolist()))
    require(all(row["candidate_index"] == index and row["reason"] == "no_improvement"
                and not row["accepted"] and details["candidates"][index]["model_sha256"] == proposal_fp
                for index, row in enumerate(details["trace"], 1)), "old proposal was not unique")
    parent_points = []
    for event, expected in zip(details["candidates"][0]["events"], item["endpoints"]["t0"]["endpoint_cache"], strict=True):
        best = event["best"]
        u, residual = np.asarray(best["u"], dtype=np.float64), np.asarray(best["residual"], dtype=np.float64)
        require(event["selected"] == expected["selected_start"] and u.shape == (item["endpoint_dimension"],)
                and residual.shape == (220,) and np.all(np.isfinite(residual)) and np.all((-2 <= u) & (u <= 2))
                and fitting._hash(best["u"]) == expected["u"]["canonical_json_sha256"]
                and fitting._hash(best["residual"]) == expected["raw_residual"]["canonical_json_sha256"]
                and fitting._hash(best["state"]) == event["endpoint_sha256"] == expected["state"]["canonical_json_sha256"],
                "old parent coordinate cache differs")
        mse = float(residual @ residual) / residual.size
        require(abs(mse-best["history_mse"]) <= 8*eps*max(1., abs(best["history_mse"])), "old parent MSE differs")
        parent_points.append(dict(u=best["u"], mse=mse))
    return dict(mode=mode, slots=slots, physics=physics, delta_star=model["delta_star"], config=parent_config,
                preparation_key=aggregate, endpoints=coefficients, objectives=objectives, parent_points=parent_points)


def legal_packets(old, slots):
    raw = json.loads(evidence.read_bound(evidence.INPUT, evidence.INPUT_SHA))
    receipt = raw["receipt"]
    order = receipt["ordered_all_ids"]
    require(len(order) == len(raw["packets"]) == 297 and len(set(order)) == 297
            and order == receipt["legal_ids"] and not receipt["invalid"]
            and all(receipt["denominators"].get(k) == v for k, v in old["denominators"].items())
            and old["assets"]["input"]["sha256"] == evidence.INPUT_SHA
            and evidence.sha(evidence.json_bytes(order)) == old["event_order_sha256"],
            "original legal input identity differs")
    packets, hashes = [], []
    for slot in slots:
        index = slot["global_index"]
        original = raw["packets"][index]
        require(order[index] == slot["event_id"] and set(original) ==
                {"theta_history", "omega_history", "history_times", "p0", "sigma"}, "illegal packet identity or fields")
        renamed = {("p_end" if k == "p0" else k): v for k, v in original.items()}
        validated = core.validate_observation(renamed, 10)
        require(validated["sigma"] == 0., "nonzero observation noise")
        packets.append(validated)
        hashes.append(dict(**slot, packet_sha256=evidence.sha(evidence.json_bytes(original))))
    return packets, hashes


def prepare(identity, endpoint, old):
    path = RESULTS / "prepare.json"
    evidence_ref = claim(RESULTS / "prepare_claim.json", identity)
    receipt = dict(**identity, claim=evidence_ref, status="running", raw_history_opened=False, caches={})
    try:
        slots = old["adaptation_slots"]
        require(slots == endpoint["input_binding"]["ordered_adaptation_slots"], "frozen slot list differs")
        verify_slots(slots, slots)
        common = bound_json(endpoint["references"]["common_coordinates"],
                            evidence.ARTIFACTS / "fit/common_coordinates.json")
        receipt["raw_history_opened"] = True
        packets, hashes = legal_packets(old, slots)
        for mode in MODES:
            cache = checked_mode_cache(endpoint, slots, packets, mode)
            require(cache["delta_star"] == common["delta_star"] and cache["physics"]["mass"] == common["mass"],
                    "common coordinates differ")
            receipt["caches"][mode] = evidence.save_json(PRIVATE / "prepare" / f"{mode}.json", cache, exclusive=True)
        receipt["packets"] = evidence.save_json(PRIVATE / "prepare/packets.json",
                                                fitting._plain([dict(slot=slot, packet=packet)
                                                                for slot, packet in zip(slots, packets, strict=True)]),
                                                exclusive=True)
        receipt.update(status="completed", slots=slots, forward_packet_hashes=hashes,
                       old_individual_packet_hashes=None, matched_preparation_keys=KEYS)
    except Exception as error:
        receipt.update(status="interrupted", error_type=type(error).__name__)
        raise
    finally:
        evidence.save_json(path, receipt, exclusive=True)


def mapped_support(target, actual, union):
    for wanted, got, support in zip(target, actual, union, strict=True):
        tolerance = 8*np.finfo(np.float64).eps*np.maximum(np.abs(wanted), np.finfo(np.float64).tiny)
        require(np.all(np.isfinite(wanted)) and np.all(np.isfinite(got))
                and np.all(np.abs(got-wanted) <= tolerance)
                and np.array_equal(wanted > 0, support) and np.array_equal(got > 0, support)
                and np.all(got[wanted == 0] == 0), "mapping error or support changed")


def interior(parent, mode, ends, t):
    require(t in POINTS.values(), "endpoint reevaluation is outside this protocol")
    (c0, b0), (c1, b1) = [tuple(np.asarray(end[key], dtype=np.float64) for key in ("c", "beta")) for end in ends]
    x = (1-t)*(c0-b0) + t*(c1-b1)
    y = (1-t)*b0 + t*b1
    target = x+y, y
    union = (c0 > 0) | (c1 > 0), (b0 > 0) | (b1 > 0)
    require(np.all((0 <= y) & (y <= target[0])), "nonnegative coordinate construction failed")
    candidate = fitting._parameters(parent, *target)
    actual = effective(candidate, mode)
    mapped_support(target, actual, union)
    require(fitting._hash(fitting._physics(candidate, include_edges=False))
            == fitting._hash(fitting._physics(parent, include_edges=False)), "non-edge fields changed")
    return candidate, target, actual


def witness(functional, packet, mode, config, parent, candidate_mse):
    counts = fitting._counts(fitting.dop853_step)
    mse = None
    try:
        state = fit297.endpoint(packet, functional, parent["u"], mode, "proxy", config)
        dense = fitting._integrate(functional, state, .0025, .1, counts, backward=True,
                                  dense=True, stepper=fitting.dop853_step)
        raw = fit297.history_residual(functional, dense[::4], packet, config)
        value = float(raw @ raw) / raw.size
        if math.isfinite(value):
            mse = value
    except (FloatingPointError, OverflowError, ValueError):
        pass
    require(counts["actual_rhs_calls"] <= 480 and (mse is None or counts["actual_rhs_calls"] == 480),
            "witness RHS limit or count differs")
    return dict(mse=mse, minus_parent=None if mse is None else mse-parent["mse"],
                minus_candidate=None if mse is None or candidate_mse is None else mse-candidate_mse,
                rhs_calls=counts["actual_rhs_calls"])


def summarize_witness(rows):
    require(len(rows) == 24, "incomplete witness denominator")
    result = {}
    for key in ("minus_parent", "minus_candidate"):
        values = [row[key] for row in rows if row[key] is not None and math.isfinite(row[key])]
        result[key] = dict(finite=len(values), mean=math.fsum(values)/24 if len(values) == 24 else None)
    return result


def check_costs(counts):
    for key, limit in LIMITS.items():
        value = counts[key]
        require(type(value) is int and 0 <= value <= limit // 6, f"candidate cost violation: {key}")
    require(counts["actual_rhs_calls"] == counts["proxy_rhs_calls"] + counts["qualification_rhs_calls"],
            "shared helper RHS ledger differs")


def choose(parent, records):
    best = dict(point="parent", objective=parent)
    for record in records:
        if record["status"] == "completed" and fitting._improves(record["objective"], best["objective"]):
            best = dict(point=record["point"], objective=record["objective"])
    return best


def candidate_job(identity, cache, packets, parent, name, t):
    mode, slots, config = cache["mode"], cache["slots"], cache["config"]
    key = f"{mode}_{name}"
    receipt_path = RESULTS / "science/jobs" / f"{key}.json"
    private_path = PRIVATE / "science" / key
    job_identity = dict(**identity, job=key, mode=mode, point=name, t=t)
    job_claim = claim(RESULTS / "science/claims" / f"{key}.json", job_identity)
    wall, cpu = time.perf_counter(), time.process_time()
    counts = matched._counts()
    counts.update(witness_rhs_calls=0, witness_residual_calls=0)
    receipt = dict(**job_identity, claim=job_claim, status="interrupted", objective=None,
                   cost_unknown=False, known_counts=counts)
    record, witnesses = {}, []
    pending_witness = False
    try:
        verify_slots(slots, slots)
        require(len(packets) == 24 and len(cache["parent_points"]) == 24, "missing packets or parent coordinates")
        functional, target, actual = interior(parent, mode, cache["endpoints"], t)
        physics = fitting._plain(fitting._physics(functional))
        fp = fitting._hash(physics)
        inner = fit297.make_config(functional, mode, config["fit_request"], 297, config["preparation_key"])
        same_config(inner, config, fp)
        arrays = dict(target=dict(c=target[0], beta=target[1]), actual=dict(c=actual[0], beta=actual[1]))
        payload = fitting._plain(dict(physics=physics, delta_star=cache["delta_star"], inner_config=inner,
                                      endpoints=cache["endpoints"], **arrays))
        coefficients_ref = evidence.save_json(private_path / "coefficients.json", payload, exclusive=True)
        restored = bound_json(coefficients_ref, private_path / "coefficients.json")
        require(restored == payload and fitting._hash(restored) == fitting._hash(payload),
                "candidate serialization changed")
        functional = restore_physics(restored["physics"])
        require(fitting._hash(fitting._physics(functional)) == fp, "round-trip functional differs")
        recovered = effective(functional, mode)
        require(all(np.array_equal(x, y) for x, y in zip(recovered, actual, strict=True)),
                "round-trip coefficients differ")
        inner = restored["inner_config"]
        same_config(inner, config, fp)
        edge_mask, mod_mask = (v > 0 for v in recovered)
        opened = int(edge_mask.sum() + mod_mask.sum())
        receipt.update(coefficients=coefficients_ref, functional_sha256=fp, config_sha256=fitting._hash(inner),
                       edge_mask=edge_mask.tolist(), modulation_mask=mod_mask.tolist(), open_terms=opened,
                       mapping_max_abs_error={k: float(np.max(np.abs(a-b)))
                                              for k, a, b in zip(("c", "beta"), recovered, target, strict=True)},
                       coefficient_hashes={k: fitting._hash(payload[k]) for k in ("target", "actual", "endpoints")})
        receipt["cost_unknown"] = True
        matched._candidate(functional, packets, slots, mode, inner, counts, record,
                           edge_mask=edge_mask, mod_mask=mod_mask)
        check_costs(counts)
        require(counts["optimization_tasks"] == 48 and counts["actual_residual_calls"] <= 57600
                and counts["qualification_suites"] <= 24, "candidate task ledger differs")
        verify_slots(slots, record["events"])
        require(all(len(event["tasks"]) == 2 and all(0 <= task["actual_residual_calls"] <= 1200
                    for task in event["tasks"]) for event in record["events"]), "candidate start ledger differs")
        require(record["status"] in ("completed", "no_output", "numerical_failure")
                and record["model_sha256"] == fp, "unexpected candidate return")
        tasks = [task for event in record["events"] for task in event["tasks"]]
        require(sum(task["actual_residual_calls"] for task in tasks) == counts["actual_residual_calls"]
                and sum(task["counts"]["actual_rhs_calls"] for task in tasks) == counts["proxy_rhs_calls"],
                "candidate per-start counts differ")
        gates = [event["qualification"] for event in record["events"] if event["qualification"] is not None]
        require(len(gates) == counts["qualification_suites"] == (0 if record["status"] == "no_output" else 24)
                and sum(gate["counts"]["actual_rhs_calls"] for gate in gates) == counts["qualification_rhs_calls"],
                "candidate per-event qualification ledger differs")
        receipt["cost_unknown"] = False
        for packet, slot, parent_point, event in zip(packets, slots, cache["parent_points"], record["events"], strict=True):
            pending_witness = True
            current = event["best"]
            result = witness(functional, packet, mode, inner, parent_point,
                             None if current is None else current["history_mse"])
            counts["witness_residual_calls"] += 1
            counts["witness_rhs_calls"] += result["rhs_calls"]
            witnesses.append(dict(**slot, **result))
            pending_witness = False
        check_costs(counts)
        require(counts["witness_residual_calls"] == 24, "incomplete witness work")
        obj = record["objective"]
        parent_obj = cache["objectives"][0]["objective"]
        mse = None if obj is None else float(np.mean([x["best"]["history_mse"] for x in record["events"]]))
        require(obj is None or abs(obj - mse - opened*1e-12) <= 8*np.finfo(float).eps*max(1., abs(obj)),
                "candidate selection objective differs")
        receipt.update(status=record["status"], objective=obj, mean_history_mse=mse, penalty=opened*1e-12,
                       delta_from_parent=None if obj is None else obj-parent_obj,
                       improves_parent=record["status"] == "completed" and fitting._improves(obj, parent_obj),
                       witness=summarize_witness(witnesses), denominator=24,
                       finite_selected=count_finite(record), qualification_passed=sum(
                           x["qualification_status"] == "passed" for x in record["events"]))
    except Exception as error:
        receipt.update(status="interrupted", error_type=type(error).__name__,
                       cost_unknown=receipt["cost_unknown"] or pending_witness)
    finally:
        receipt["known_rhs_including_witness"] = counts["actual_rhs_calls"] + counts["witness_rhs_calls"]
        receipt["worker_wall_seconds"], receipt["worker_cpu_seconds"] = time.perf_counter()-wall, time.process_time()-cpu
        try:
            receipt["details"] = evidence.save(
                private_path / "details.json.gz",
                gzip.compress(evidence.json_bytes(matched._json_value(dict(candidate=record, witnesses=witnesses))), mtime=0),
                exclusive=True)
        finally:
            receipt_ref = evidence.save_json(receipt_path, receipt, exclusive=True)
    return receipt_ref, receipt


def count_finite(record):
    return sum(event["best"] is not None for event in record["events"])


def run_mode(identity, prepared, mode):
    cache = bound_json(prepared["caches"][mode], PRIVATE / "prepare" / f"{mode}.json")
    serialized = bound_json(prepared["packets"], PRIVATE / "prepare/packets.json")
    verify_slots(prepared["slots"], [row["slot"] for row in serialized])
    packets = [core.validate_observation(row["packet"], 10) for row in serialized]
    require(all(packet["sigma"] == 0. for packet in packets), "private packet noise differs")
    require(len(packets) == 24 and cache["mode"] == mode and cache["slots"] == prepared["slots"]
            and cache["preparation_key"] == KEYS[mode], "private prepared cache identity differs")
    parent = restore_physics(cache["physics"])
    require(fitting._hash(fitting._physics(parent)) == cache["objectives"][0]["fingerprint"],
            "cached parent fingerprint differs")
    records, refs = [], []
    for name, t in POINTS.items():
        ref, record = candidate_job(identity, cache, packets, parent, name, t)
        refs.append(ref)
        records.append(record)
        if record["status"] == "interrupted":
            break
    complete = len(records) == 3 and all(x["status"] != "interrupted" for x in records)
    return dict(status="completed" if complete else "interrupted", jobs=refs, records=records,
                endpoints=cache["objectives"],
                selection=choose(cache["objectives"][0]["objective"], records) if complete else None)


def scientific(identity, prepared):
    stage_claim = claim(RESULTS / "science/claim.json", identity)
    report = dict(**identity, claim=stage_claim, status="running", modes={}, truth_cache_opened=False)
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = {mode: pool.submit(run_mode, identity, prepared, mode) for mode in MODES}
            for mode, future in futures.items():
                try:
                    report["modes"][mode] = future.result()
                except Exception as error:
                    report["modes"][mode] = dict(status="unknown", error_type=type(error).__name__)
        complete = all(item["status"] == "completed" for item in report["modes"].values())
        reports = [row for item in report["modes"].values() for row in item.get("records", [])]
        costs = {key: sum(row["known_counts"][key] for row in reports) for key in LIMITS}
        costs["actual_rhs_calls"] += costs["witness_rhs_calls"]
        require(all(costs[key] <= limit for key, limit in LIMITS.items()), "total ledger exceeds protocol")
        cost_unknown = any(row["cost_unknown"] for row in reports) or any(
            item["status"] == "unknown" for item in report["modes"].values())
        report.update(status="completed" if complete else "interrupted", paired_complete=complete,
                      known_costs=costs, actual_costs=None if cost_unknown else costs, cost_unknown=cost_unknown)
    except Exception:
        report.update(status="interrupted", cost_unknown=True, actual_costs=None)
        raise
    finally:
        report["host_wall_seconds"], report["host_cpu_seconds"] = time.perf_counter()-wall, time.process_time()-cpu
        evidence.save_json(RESULTS / "science/execution.json", report, exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "science"))
    parser.add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    raw = evidence.local_path(args.release).read_bytes()
    release = json.loads(raw)
    endpoint, old = frozen_context(args.phase, release)
    identity = dict(run=RUN, phase=args.phase, action_id=release["action_id"],
                    release_sha256=evidence.sha(raw), protocol_sha256=PROTOCOL_SHA,
                    design_adoption_sha256=ADOPTION_SHA, endpoint_digest_sha256=ENDPOINT_SHA,
                    source_sha256=release["source_sha256"], runtime=release["runtime"])
    qualification = release["qualification"]
    attempt = qualification["attempt"]
    require(type(attempt) is int and 1 <= attempt <= 3, "qualification attempt outside protocol")
    record = bound_json(qualification["receipt"], RESULTS / f"qualification_{attempt}.json")
    require(record["status"] == "passed" and record["attempt"] == attempt
            and record["source_sha256"] == release["source_sha256"]
            and record["runtime"] == release["runtime"] and record["protocol_sha256"] == PROTOCOL_SHA
            and record["run"] == RUN and len(record["groups"]) == 6 and not record["forbidden_attempts"]
            and 0 <= record["mock_callbacks"] <= 1000
            and record["cumulative_mock_callbacks"] <= 3000 and record["real_science_calls"] == 0,
            "frozen qualification missing or inconsistent")
    identity["qualification"] = qualification
    if args.phase == "prepare":
        prepare(identity, endpoint, old)
        return
    prepare_ref = release["prepare"]
    prepared = bound_json(prepare_ref, RESULTS / "prepare.json")
    require(prepared["status"] == "completed" and prepared["run"] == RUN and prepared["phase"] == "prepare"
            and prepared["source_sha256"] == release["source_sha256"] and prepared["runtime"] == release["runtime"]
            and prepared["protocol_sha256"] == PROTOCOL_SHA and prepared["endpoint_digest_sha256"] == ENDPOINT_SHA
            and prepared["qualification"] == qualification and prepared["matched_preparation_keys"] == KEYS
            and prepared["slots"] == old["adaptation_slots"], "released prepare identity differs")
    scientific(dict(**identity, prepare=prepare_ref), prepared)


if __name__ == "__main__":
    main()
