"""Pure frozen-checkpoint decision; never loads events, models, or an optimizer.

Run only after all eleven evaluation receipts are complete. The decision is a
suggested source for a future adaptation; it does not adopt or train a model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import checkpoint_selection_20261030 as audit

LABELS = audit.LABELS
CANDIDATES = LABELS[2:]
SCHEMA = "checkpoint_selection_decision_20261030_v1"
SCIENTIST = ROOT / "reports/checkpoint_selection_20261030_scientist.md"
PROTOCOL = ROOT / "reports/checkpoint_selection_20261030_protocol.md"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def le(a, b):
    return finite(a) and finite(b) and a <= b


def multiple(value, factor):
    return value * factor if finite(value) else None


def ready(row):
    return bool(row.get("numerically_qualified")) and all(finite(row.get(k, {}).get("time_s")) for k in ("truth", "fine", "coarse"))


def errors(profile, ids):
    if not ids or any(not ready(profile["timing_actual15"].get(eid, {})) for eid in ids):
        return {"E": None, "u": None}
    rows = [profile["timing_actual15"][eid] for eid in ids]
    return {"E": sum(abs(r["fine"]["time_s"] - r["truth"]["time_s"]) for r in rows) / len(rows),
            "u": max(abs(r["fine"]["time_s"] - r["coarse"]["time_s"]) for r in rows)}


def load_completed():
    """Read only completed evaluator artifacts; reject missing or changed evidence."""
    manifest_path = audit.PRIVATE / "manifest.json"
    manifest = audit.read(manifest_path)
    manifest_sha = audit.sha(manifest_path)
    if manifest["protocol"] != audit.identity(PROTOCOL):
        raise ValueError("technical_invalid: protocol differs from evaluation manifest")
    for relative, expected in manifest["source"].items():
        if audit.sha(ROOT / relative) != expected:
            raise ValueError("technical_invalid: evaluation helper source changed")
    for item in [manifest["data"]["fit_cache"], manifest["data"]["prepare"], *manifest["models"].values()]:
        if audit.sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("technical_invalid: source/data/model hash changed")
    missing = [label for label in LABELS if not (audit.PRIVATE / label / "complete.json").is_file()]
    if missing:
        raise ValueError("technical_invalid: missing completed checkpoints: " + ",".join(missing))
    summaries, details, bindings = {}, {}, {}
    expected_labels = manifest["data"]["event_labels"]
    if len(expected_labels) != 297 or list(expected_labels.values()).count("stable") != 267 or list(expected_labels.values()).count("unstable") != 30:
        raise ValueError("technical_invalid: fit297 denominator changed")
    for label in LABELS:
        folder = audit.PRIVATE / label
        receipt = audit.read(folder / "complete.json")
        if receipt.get("status") != "COMPLETED":
            raise ValueError("technical_invalid: incomplete receipt")
        summary = audit.verify_complete(folder, manifest_sha)
        if summary.get("status") != "COMPLETED" or summary.get("checkpoint") != label or summary.get("manifest_sha256") != manifest_sha or summary["model"] != {k: manifest["models"][label][k] for k in ("path", "sha256")}:
            raise ValueError("technical_invalid: summary identity differs")
        item = summary["per_event"]
        if item not in receipt["artifacts"] or audit.sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("technical_invalid: per-event receipt differs")
        detail = audit.read(ROOT / item["path"])
        if detail.get("checkpoint") != label:
            raise ValueError("technical_invalid: per-event checkpoint differs")
        for scope in ("8s", "actual15"):
            rows = detail["classification"][scope]["rows"]
            if len(rows) != 297 or len({r["event_id"] for r in rows}) != 297 or {r["event_id"]: r["label"] for r in rows} != expected_labels:
                raise ValueError("technical_invalid: classification fit_ids/labels differ")
            counts = detail["classification"][scope]["original_counts"]
            pk = "classification_8s" if scope == "8s" else "classification_actual15s"
            if counts != {k: v for k, v in summary["decision_profile"][pk].items() if k != "rows"}:
                raise ValueError("technical_invalid: classification profile differs")
        if detail["timing_actual15"] != summary["decision_profile"]["timing_actual15"] or set(detail["timing_actual15"]) != set(expected_labels):
            raise ValueError("technical_invalid: timing fit_ids/profile differs")
        summaries[label], details[label] = summary, detail
        bindings[label] = {"summary": audit.identity(folder / "summary.json"), "complete": audit.identity(folder / "complete.json"), "per_event": item}
    anchor_path = audit.PRIVATE / "anchors.json"
    anchors = audit.read(anchor_path)
    expected_refs = {r: bindings[r]["summary"]["sha256"] for r in ("R_N", "A96")}
    if anchors.get("manifest_sha256") != manifest_sha or anchors.get("reference_summary_hashes") != expected_refs:
        raise ValueError("technical_invalid: frozen anchor manifest/reference hashes differ")
    return summaries, details, {"manifest": audit.identity(manifest_path), "checkpoints": bindings, "anchors": audit.identity(anchor_path)}, anchors


def detections(detail, scope):
    return {r["event_id"] for r in detail["classification"][scope]["rows"] if r["label"] == "unstable" and r["numerically_admissible"] and r["predicted_los"]}


def identity_delta(candidate, reference, scope):
    c, r = detections(candidate, scope), detections(reference, scope)
    def false(d):
        return {x["event_id"] for x in d["classification"][scope]["rows"] if x["label"] == "stable" and x["numerically_admissible"] and x["predicted_los"]}
    return {"new_missed_ids": sorted(r - c), "repaired_missed_ids": sorted(c - r), "new_false_los_ids": sorted(false(candidate) - false(reference)),
            "unresolved_ids": sorted(x["event_id"] for x in candidate["classification"][scope]["rows"] if not x["numerically_admissible"])}


def pareto_earliest(vectors):
    """Minimize every original component; exact ties remain non-dominated."""
    front = []
    for name, values in vectors.items():
        if not all(finite(v) for v in values):
            raise ValueError("nonfinite Pareto component")
        dominated = any(other != name and all(a <= b for a, b in zip(vals, values)) and any(a < b for a, b in zip(vals, values)) for other, vals in vectors.items())
        if not dominated:
            front.append(name)
    front.sort(key=lambda name: int(name[4:]))
    return front, front[0] if front else None


def decide(summaries, details, *, anchors=None, target_fn=None, valid_fn=None):
    """No I/O; injections exist only for synthetic tests of the frozen rule."""
    if set(summaries) != set(LABELS) or set(details) != set(LABELS):
        raise ValueError("all eleven completed checkpoints are required before selection")
    if target_fn is None:
        from memory_transport_study_20261029 import efficiency_target
        target_fn = efficiency_target
    if valid_fn is None:
        from state_gated_memory_study_20261028 import valid
        valid_fn = valid
    profiles = {label: summaries[label]["decision_profile"] for label in LABELS}
    refs = ("A96", "R_N")
    p96, rn = profiles["A96"], profiles["R_N"]
    protected = {scope: sorted(set.union(*(detections(details[r], scope) for r in refs))) for scope in ("8s", "actual15")}
    ref_ids = set(p96["timing_actual15"]) | set(rn["timing_actual15"])
    ref_truth_mismatch = sorted(eid for eid in ref_ids if eid not in p96["timing_actual15"] or eid not in rn["timing_actual15"] or p96["timing_actual15"][eid]["truth"]["time_s"] != rn["timing_actual15"][eid]["truth"]["time_s"])
    anchor = sorted(eid for eid in ref_ids if all(ready(profiles[r]["timing_actual15"].get(eid, {})) for r in refs))
    if anchors is not None:
        expected_truth = {eid: p96["timing_actual15"][eid]["truth"]["time_s"] for eid in anchor}
        if anchors.get("detection_union") != protected or anchors.get("T_anchor") != anchor or anchors.get("truth_times") != expected_truth or ref_truth_mismatch:
            raise ValueError("technical_invalid: frozen anchors differ from original reference reconstruction")
        # Use the frozen values only after exact reconstruction checks.
        protected = anchors["detection_union"]
        anchor = anchors["T_anchor"]
    ref_errors = {r: errors(profiles[r], anchor) for r in refs}
    anchor_valid = bool(anchor) and not ref_truth_mismatch and all(valid_fn(profiles[r]) for r in refs)
    rows, vectors = {}, {}
    for label in CANDIDATES:
        p = profiles[label]
        # This is the original frozen target, including its original timing check.
        original = target_fn(p, p96, rn)
        risk, a96risk = p.get("unstable_actual15_risk"), p96.get("unstable_actual15_risk")
        perturb = p.get("unstable_actual15_numerical_score_perturbation")
        a96perturb = p96.get("unstable_actual15_numerical_score_perturbation")
        actual_misses = p["classification_actual15s"]["missed_resolved_unstable"]
        misses8 = p["classification_8s"]["missed_resolved_unstable"]
        gain = le(risk, multiple(a96risk, .95)) and all(finite(v) for v in (perturb, a96perturb)) and a96risk - risk > a96perturb + perturb
        gates = {"numerical_qualification": all(valid_fn(profiles[r]) for r in (*refs, label)),
            "stable_risk_5s": le(p["horizon_risks"]["5"]["value"], multiple(rn["horizon_risks"]["5"]["value"], 1.05)),
            "stable_risk_8s": le(p["horizon_risks"]["8"]["value"], multiple(rn["horizon_risks"]["8"]["value"], 1.05)),
            "common_omega_rmse_8s": le(p["common_frequency_rmse_8s"], multiple(rn["common_frequency_rmse_8s"], 1.05)),
            "normalized_p_rmse_8s": le(p["normalized_p_rmse_8s"], multiple(rn["normalized_p_rmse_8s"], 1.05)),
            "unstable_actual15_risk": le(risk, rn.get("unstable_actual15_risk")), "actual15_misses": le(actual_misses, 21), "misses_8s": le(misses8, 3),
            "false_los_8s": p["classification_8s"]["false_los_resolved_stable"] == 0,
            "false_los_actual15": p["classification_actual15s"]["false_los_resolved_stable"] == 0,
            "actual_improvement": le(actual_misses, 20) or gain, "original_efficiency_target": original.get("passed") is True}
        changes = {scope: {ref: identity_delta(details[label], details[ref], scope) for ref in refs} for scope in protected}
        for scope, ids in protected.items():
            gates[f"protected_detection_{scope}"] = set(ids) <= detections(details[label], scope)
        time = p["timing_actual15"]
        missing = [eid for eid in anchor if not ready(time.get(eid, {}))]
        truth_mismatch = [eid for eid in anchor if eid in time and time[eid]["truth"]["time_s"] != p96["timing_actual15"][eid]["truth"]["time_s"]]
        gates.update(T_anchor_nonempty_and_refs_valid=anchor_valid, T_anchor_coverage=not missing, T_truth_equal=not truth_mismatch and not ref_truth_mismatch)
        ec = errors(p, anchor)
        common = sorted(eid for eid in anchor if ready(time.get(eid, {})))
        for ref in refs:
            er = ref_errors[ref]
            gates[f"T_E_vs_{ref}"] = bool(anchor_valid and not missing and not truth_mismatch and le(ec["E"], er["E"] + .01 + ec["u"] + er["u"]))
        failures = [key for key, value in gates.items() if not value]
        vector = [actual_misses, misses8, risk, p["horizon_risks"]["5"]["value"], p["horizon_risks"]["8"]["value"], p["common_frequency_rmse_8s"], p["normalized_p_rmse_8s"], ec["E"]]
        if not failures:
            vectors[label] = vector
        extra_timing = {eid: {**row, "signed_error_s": row["fine"]["time_s"] - row["truth"]["time_s"]} for eid, row in time.items() if eid not in anchor and ready(row)}
        rows[label] = {"step": int(label[4:]), "eligible": not failures, "gates": gates, "failed_gates": failures, "first_failure": failures[0] if failures else None,
            "original_efficiency_target": original, "identity_changes": changes, "pareto_components": vector,
            "timing": {"fixed_anchor_metrics": ec, "common_event_ids": common, "missing_anchor_ids": missing, "truth_mismatch_ids": truth_mismatch,
                       "extra_confirmed_events": extra_timing, "per_anchor": {eid: time.get(eid) for eid in anchor}},
            "source_optimization_health": {"status": "insufficient_history" if int(label[4:]) < 64 else "unknown", "audited": False, "gate_applied": False},
            "current_optimizer_health": "not_applicable", "current_optimizer_calls": 0}
    front, selected = pareto_earliest(vectors)
    return {"schema": SCHEMA, "status": "COMPLETED", "scope": "consumed train fit297; 267 stable/30 unstable; nine exposed checkpoints; no independent confirmation",
        "protected_detection_sets": {scope: {"event_ids": ids, "sha256": digest(ids), "count": len(ids)} for scope, ids in protected.items()},
        "T_anchor": {"event_ids": anchor, "sha256": digest(anchor), "count": len(anchor), "valid": anchor_valid, "reference_truth_mismatch_ids": ref_truth_mismatch, "reference_metrics": ref_errors},
        "candidates": rows, "eligible_checkpoints": list(vectors), "pareto_front": front, "suggested_adaptation_source": selected,
        "selection_opportunities": 9, "model_adopted": False, "model_decision": "retain_evidence", "scientific_status": "work_start_found" if selected else "unresolved",
        "future_training_executed": False, "optimizer_calls": 0, "val_read": False, "test_read": False, "new_fom": 0}


def public_copy(decision):
    """Only counts, set digests, metrics and checkpoint names leave private storage."""
    result = {k: v for k, v in decision.items() if k not in ("protected_detection_sets", "T_anchor", "candidates", "inputs")}
    result["protected_detection_sets"] = {s: {k: v for k, v in cell.items() if k != "event_ids"} for s, cell in decision["protected_detection_sets"].items()}
    result["T_anchor"] = {k: v for k, v in decision["T_anchor"].items() if k not in ("event_ids", "reference_truth_mismatch_ids")}
    result["T_anchor"]["reference_truth_mismatch_count"] = len(decision["T_anchor"]["reference_truth_mismatch_ids"])
    result["candidates"] = {label: {k: row[k] for k in ("step", "eligible", "gates", "failed_gates", "first_failure", "pareto_components", "source_optimization_health", "current_optimizer_health", "current_optimizer_calls")} for label, row in decision["candidates"].items()}
    for label, row in decision["candidates"].items():
        result["candidates"][label]["timing_metrics"] = row["timing"]["fixed_anchor_metrics"]
        result["candidates"][label]["identity_changes"] = {scope: {ref: {key.replace("_ids", "_count"): len(ids) for key, ids in cells.items()} for ref, cells in refs.items()} for scope, refs in row["identity_changes"].items()}
    result["private_decision_sha256"] = digest(decision)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    summaries, details, binding, anchors = load_completed()
    decision = decide(summaries, details, anchors=anchors)
    decision["inputs"] = binding
    decision["source"] = {"decider": audit.identity(Path(__file__)), "scientist": audit.identity(SCIENTIST), "protocol": audit.identity(PROTOCOL),
                          "original_efficiency_target": audit.identity(ROOT / "experiments/memory_transport_study_20261029.py")}
    public = public_copy(decision)
    for path, payload in ((audit.PRIVATE / "decision.json", decision), (audit.PUBLIC / "decision_public.json", public)):
        if path.exists():
            if audit.read(path) != payload:
                raise ValueError("immutable decision differs; never overwrite")
        else:
            audit.write_once(path, payload)
    print(json.dumps({"status": decision["status"], "suggested_adaptation_source": decision["suggested_adaptation_source"], "model_adopted": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
