"""Read-only bounded audit of the nine original A_30311 checkpoints.

No model, optimizer, restore, RNG restoration, rollout, or event payload loading.
Only the explicitly named source records are opened; outputs are immutable.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
BASE = ROOT / "private_data/memory_transport_20261029/study01_prelaunch_retry1"
RUN = BASE / "study/A_30311"
A96 = ROOT / "private_data/state_gated_memory_20261028/study01/study/A_28311/checkpoints/step000096/model.json"
PRIVATE = ROOT / "private_data/checkpoint_selection_20261030/source_health.json"
PUBLIC = ROOT / "results/checkpoint_selection_20261030/source_health_public.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def asset(path):
    return {"path": Path(path).relative_to(ROOT).as_posix(), "sha256": sha(path)}


def stats(values):
    values = list(values)
    return {"count": len(values), "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None}


def finite_tree(value):
    import torch
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(x) for x in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def journal_audit():
    path = BASE / "optimizer_calls.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    attempts, completed, errors = {}, {}, []
    for row in records:
        cid = row["call_id"]
        if row["event"] == "optimizer_step_attempt":
            if cid in attempts:
                errors.append("duplicate_attempt:" + cid)
            attempts[cid] = row
        elif row["event"] == "optimizer_step_completed":
            if cid in completed or cid not in attempts or any(attempts[cid][key] != row[key] for key in ("category", "run_id", "step")):
                errors.append("unmatched_completion:" + cid)
            completed[cid] = row
        else:
            errors.append("unknown_journal_event")
    pending = sorted(set(attempts) - set(completed))
    formal = [r for r in completed.values() if r["category"] == "formal" and r["run_id"] == "A_30311"]
    pending_formal = [cid for cid in pending if attempts[cid]["category"] == "formal" and attempts[cid]["run_id"] == "A_30311"]
    by_category = {name: sum(r["category"] == name for r in completed.values()) for name in sorted({r["category"] for r in records})}
    result = {"source": asset(path), "project_scope": "original memory_transport study including QA and A/S/P formal jobs",
        "project_completed": len(completed), "project_attempted": len(attempts), "project_pending_ids": pending,
        "completed_by_category": by_category, "expected_original_project_completed": 492,
        "A_30311_formal_completed": len(formal), "A_30311_formal_steps": sorted(r["step"] for r in formal),
        "A_30311_formal_pending_ids": pending_formal, "errors": errors}
    result["A_30311_accounted"] = not errors and not pending_formal and sorted(r["step"] for r in formal) == list(range(1, 129))
    result["project_accounted"] = not errors and not pending and len(completed) == 492
    return result


def checkpoint_audit(step, endpoint_trace, journal):
    import torch
    from memory_transport_study_20261029 import optimization
    folder = RUN / "checkpoints" / f"step{step:06d}"
    row = {"step": step, "status": "unknown", "checks": {}, "errors": []}
    try:
        meta = read(folder / "checkpoint.json")
        row["assets"] = {name: asset(folder / name) for name in ("checkpoint.json", "model.json", "training.pt")}
        checks = row["checks"]
        checks["model_sha_matches_metadata"] = row["assets"]["model.json"]["sha256"] == meta["model_sha256"]
        checks["training_sha_matches_metadata"] = row["assets"]["training.pt"]["sha256"] == meta["training_sha256"]
        if not all(checks.values()):
            row["errors"].append("hash_mismatch; checkpoint tensors not opened")
            return row
        model = read(folder / "model.json")
        state = torch.load(folder / "training.pt", map_location="cpu", weights_only=False)
        checks["model_all_numeric_finite"] = finite_tree(model)
        checks["training_tensors_finite"] = finite_tree(state)
        checks["step_counters_match"] = meta["step"] == state["step"] == step
        trace = meta["trace"]
        checks["trace_steps_exact"] = [x["step"] for x in trace] == list(range(1, step + 1))
        checks["trace_matches_endpoint_prefix"] = trace == endpoint_trace[:step]
        checks["trace_finite"] = finite_tree(trace)
        row["trace_length"] = len(trace)
        row["gradient_norm"] = stats(x["gradient_norm"] for x in trace)
        row["clip_threshold"] = 100.
        row["clipped_steps"] = [x["step"] for x in trace if x["gradient_norm"] > 100]
        row["clip_count"] = len(row["clipped_steps"])
        names = sorted({name for x in trace for name in x["block_updates"]})
        row["block_updates"] = {name: {**stats(x["block_updates"][name]["relative_l2"] for x in trace if name in x["block_updates"]),
            "large_change_count_above_0_25": sum(x["block_updates"].get(name, {}).get("relative_l2", 0) > .25 for x in trace)} for name in names}
        opt = state["optimizer"]
        params = [pid for group in opt["param_groups"] for pid in group["params"]]
        checks["Adam_state_complete"] = len(opt["state"]) == len(params) if step else len(opt["state"]) == 0
        counters = {str(pid): float(s["step"]) for pid, s in opt["state"].items()}
        checks["Adam_counters_match"] = all(value == step for value in counters.values())
        checks["Adam_finite"] = finite_tree(opt)
        checks["Adam_layout_matches"] = len(opt["param_groups"]) == len(meta["parameter_layout"]) and all(
            group["block"] == layout["block"] and group["lr"] == layout["lr"] and len(group["params"]) == len(layout["names"])
            for group, layout in zip(opt["param_groups"], meta["parameter_layout"]))
        checks["Adam_moment_shapes_match_model"] = all(
            tuple(opt["state"][pid][key].shape) == tuple(torch.as_tensor(model["state_dict"][name]).shape)
            for group, layout in zip(opt["param_groups"], meta["parameter_layout"])
            for pid, name in zip(group["params"], layout["names"]) for key in ("exp_avg", "exp_avg_sq") if pid in opt["state"])
        checks["source_code_hashes_match"] = all(sha(ROOT / path) == expected for path, expected in meta["identity"]["code"].items())
        checks["source_A96_hashes_match"] = all(sha(A96.parent / name) == expected for name, expected in meta["identity"]["source"]["files"].items())
        checks["formal_journal_accounted"] = journal.get("A_30311_accounted") is True
        checks["project_journal_accounted"] = journal.get("project_accounted") is True
        row["Adam_step_counters"] = counters
        row["recorded_source_training_seconds"] = sum(x["seconds"] for x in trace)
        row["original_last64_rule"] = optimization(trace) if step >= 64 else None
        row["status"] = ("insufficient_history" if step < 64 else "optimization_unresolved" if row["original_last64_rule"]["optimization_unresolved"] else "healthy_on_recorded_checks") if all(checks.values()) else "unknown"
        row["errors"] = [name for name, value in checks.items() if not value]
    except Exception as exc:
        row["errors"].append(repr(exc))
    return row


def compatibility_audit():
    row = {"status": "unknown", "no_new_rollout": True}
    try:
        summary_path = BASE / "qualification/summary.json"
        summary = read(summary_path)
        binding = summary["per_variant"]["A"]["A_A96_compatibility"]
        path = Path(binding["path"])
        path.resolve().relative_to((BASE / "qualification").resolve())
        record = read(path)
        prepare = read(BASE / "prepare.json")
        a0_path = RUN / "checkpoints/step000000/model.json"
        a0, a96 = read(a0_path), read(A96)
        common = a96["state_dict"]
        extra = sorted(set(a0["state_dict"]) - set(common))
        zero = BASE / "qualification/A_zero_fit297"
        zero_receipt = read(zero / "complete.json")
        a96_evidence = ROOT / "private_data/state_gated_memory_20261028/study01/study/A_28311/fit000096/evidence.json"
        checks = {"compatibility_hash_matches_qualification": sha(path) == binding["sha256"],
            "qualified": record.get("qualified") is True and summary["per_variant"]["A"].get("A_A96_compatibility_qualified") is True,
            "bidirectional": record.get("bidirectional") is True,
            "scope_original_8s_actual15": record.get("scope") == "A96_A0_fit297_bidirectional_RK4_fine005_base01_8s_actual15",
            "fit_ids_exact": len(record["event_ids"]) == 297 and len(set(record["event_ids"])) == 297 and set(record["event_ids"]) == set(prepare["fit_ids"]),
            "all_297_rows_qualified": len(record["rows"]) == 297 and all(x["qualified"] for x in record["rows"]),
            "A96_evidence_hash_matches": sha(a96_evidence) == record["source_evidence_sha256"],
            "zero_evidence_receipt_matches": zero_receipt["status"] == "COMPLETED" and zero_receipt["event_count"] == 297 and sha(zero / "evidence.json") == zero_receipt["evidence_sha256"],
            "common_exported_state_exact": set(common) <= set(a0["state_dict"]) and all(a0["state_dict"][name] == value for name, value in common.items()),
            "only_fixed_plane_extra_state": extra == ["u", "v"], "A0_variant_A": a0.get("variant") == "A", "history_dt_equal": a0.get("history_dt") == a96.get("history_dt") == .0025}
        row.update(status="qualified_existing_bidirectional_evidence" if all(checks.values()) else "unknown", checks=checks,
            assets={"qualification": asset(summary_path), "compatibility": asset(path), "prepare": asset(BASE / "prepare.json"), "A0_model": asset(a0_path), "A96_model": asset(A96), "A96_evidence": asset(a96_evidence), "zero_evidence": asset(zero / "evidence.json"), "zero_complete": asset(zero / "complete.json")},
            record_count=record.get("record_count"), scope=record.get("scope"), event_ids_sha256=hashlib.sha256(json.dumps(sorted(record["event_ids"])).encode()).hexdigest(),
            note="Original numerical compatibility plus equal shared exported state; no claim that different class byte exports are identical.")
    except Exception as exc:
        row["error"] = repr(exc)
    return row


def main():
    started = time.perf_counter()
    if PRIVATE.exists() or PUBLIC.exists():
        raise FileExistsError("immutable source-health outputs already exist")
    import torch
    torch.set_num_threads(1)
    try:
        journal = journal_audit()
    except Exception as exc:
        journal = {"status": "unknown", "error": repr(exc)}
    try:
        endpoint = read(RUN / "checkpoints/step000128/checkpoint.json")["trace"]
    except Exception:
        endpoint = None
    rows = {f"step{step:06d}": checkpoint_audit(step, endpoint, journal) for step in range(0, 129, 16)} if endpoint is not None else {f"step{step:06d}": {"step": step, "status": "unknown", "errors": ["endpoint trace unreadable"]} for step in range(0, 129, 16)}
    result = {"schema": "checkpoint_source_health_20261030_v1", "source": asset(Path(__file__)), "original_health_rule": asset(ROOT / "experiments/memory_transport_study_20261029.py"),
        "journal": journal, "checkpoints": rows, "A0_A96_compatibility": compatibility_audit(), "current_optimizer_calls": 0,
        "current_training_seconds": 0., "new_rollouts": 0, "event_payloads_read": False, "val_read": False, "test_read": False,
        "optimization_sufficiency": "not_established_by_health_checks", "wall_seconds": time.perf_counter() - started}
    public = {key: result[key] for key in ("schema", "current_optimizer_calls", "current_training_seconds", "new_rollouts", "event_payloads_read", "val_read", "test_read", "optimization_sufficiency", "wall_seconds")}
    public["source_sha256"] = result["source"]["sha256"]
    public["original_health_rule_sha256"] = result["original_health_rule"]["sha256"]
    public["checkpoints"] = {label: {key: value for key, value in row.items() if key not in ("assets", "errors", "clipped_steps")} for label, row in rows.items()}
    public["journal"] = {key: journal.get(key) for key in ("project_completed", "project_attempted", "completed_by_category", "expected_original_project_completed", "A_30311_formal_completed", "A_30311_accounted", "project_accounted")}
    public["journal"]["project_pending_count"] = len(journal["project_pending_ids"]) if "project_pending_ids" in journal else None
    public["journal"]["A_30311_pending_count"] = len(journal["A_30311_formal_pending_ids"]) if "A_30311_formal_pending_ids" in journal else None
    comp = result["A0_A96_compatibility"]
    public["A0_A96_compatibility"] = {key: comp.get(key) for key in ("status", "checks", "record_count", "scope", "event_ids_sha256", "no_new_rollout")}
    public["private_source_health_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    for path, payload in ((PRIVATE, result), (PUBLIC, public)):
        with path.open("x", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
    print(json.dumps({"status": "COMPLETED", "checkpoint_statuses": {k: v["status"] for k, v in rows.items()}, "compatibility": comp["status"], "wall_seconds": result["wall_seconds"], "optimizer_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
