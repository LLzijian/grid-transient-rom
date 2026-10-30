"""Frozen fit297 checkpoint audit. Import is inert; there is no training entry point.

``evaluate --protocol reports/<frozen protocol>.md [--checkpoint step000016]``
registers all source/data/model hashes before inference. Outputs have fixed roots;
completed receipts are verified and never recomputed. Terminal failures need
--retry-failed; an attempt without a receipt is deliberately not replayed.
The selection rule is owned by the separately frozen scientific protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]
PRIVATE = ROOT / "private_data/checkpoint_selection_20261030"
PUBLIC = ROOT / "results/checkpoint_selection_20261030"
SGM = ROOT / "private_data/state_gated_memory_20261028/study01"
TRANSPORT = ROOT / "private_data/memory_transport_20261029/study01_prelaunch_retry1/study/A_30311"
RN = ROOT / "private_data/force_graph_20261028/study01/study/N6101_M_28102/checkpoints/step000192/model.json"
STEPS = tuple(range(0, 129, 16))
LABELS = ("R_N", "A96") + tuple(f"step{x:06d}" for x in STEPS)
SCHEMA = "checkpoint_selection_20261030_v1"
# Original completed paired evaluations, not the fine-only probe curves.
REUSE = {
    "R_N": (SGM / "qualification/R_N_fit297", "140af6d7d30d663f66a95f31b92348903283d973e0091f9121d6e1548cfbeb19", "937f00b740875913923487b8d772f40d2a8145a88eba2cdaf61f2d889715bfc9"),
    "A96": (SGM / "study/A_28311/fit000096", "22e2b30d9c4e6dc4d6162d11e3eeb2707a9a6b5d461fef956a3aff8e4e15c5fb", "ec8c78cbff0b6dd65bd3ba6dfc5a0ed65e7fa418227c90adc3a703311aebadf4"),
    "step000128": (TRANSPORT / "fit000128", "84f476426d2e24b8c461313f383db37f76239becead3690d33016a5059e998f4", "a74ea3093dbc4d357baa24af5d84072caa0840a2bc76cf2ca5040c964ab2ef03"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    """Exclusive publication: never overwrite an artifact, even after failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def identity(path):
    path = Path(path).resolve()
    path.relative_to(ROOT.resolve())
    return {"path": path.relative_to(ROOT.resolve()).as_posix(), "sha256": sha(path)}


def model_spec(label):
    if label == "R_N":
        return {"export": RN, "loader": "force_graph"}
    if label == "A96":
        return {"export": SGM / "study/A_28311/checkpoints/step000096/model.json", "loader": "state_gated"}
    if label not in LABELS:
        raise ValueError("unregistered checkpoint")
    return {"export": TRANSPORT / "checkpoints" / label / "model.json", "loader": "transport"}


def register(protocol, prepared, events):
    """Freeze input identities before any rollout or cached-metric computation."""
    import state_gated_memory_study_20261028 as sgm
    import force_graph_study_20261028 as fg
    ids = [e["id"] for e in events]
    if len(ids) != 297 or len(set(ids)) != 297 or set(ids) != set(prepared["fit_ids"]):
        raise ValueError("fit297 identity changed")
    if sum(e["label"] == "stable" for e in events) != 267 or sum(e["label"] == "unstable" for e in events) != 30:
        raise ValueError("fit297 267/30 denominator changed")
    cache = fg.OLD_BASE / "fit_events.npz"
    if sha(cache) != prepared["fit_cache_sha256"]:
        raise ValueError("fit cache digest changed")
    code = sgm.code_identity()
    for name in ("checkpoint_selection_20261030.py", "closure_probe_20261030.py", "force_exchange_study_20261028.py", "memory_transport_model_20261029.py", "evaluate_l1_latent.py", "evaluate_long_latent.py"):
        code[f"experiments/{name}"] = sha(ROOT / "experiments" / name)
    reused = {}
    for label, (folder, expected, model_hash) in REUSE.items():
        if sha(folder / "evidence.json") != expected or sha(model_spec(label)["export"]) != model_hash:
            raise ValueError(f"{label}: frozen reuse evidence/model identity changed")
        evidence = read(folder / "evidence.json")
        reused[label] = {"evidence": identity(folder / "evidence.json"), "complete": identity(folder / "complete.json"),
                         "profile": identity(folder / "state_gated_profile.json"), "predictions": evidence["prediction_files"]}
    manifest = {"schema": SCHEMA, "protocol": identity(protocol), "source": code,
        "data": {"fit_cache": identity(cache), "prepare": identity(SGM / "prepare.json"), "fit_ids": prepared["fit_ids"],
                 "event_labels": {e["id"]: e["label"] for e in events}, "event_count": 297, "stable": 267, "unstable": 30},
        "models": {name: {**identity(model_spec(name)["export"]), "loader": model_spec(name)["loader"]} for name in LABELS},
        "reuse": reused, "method": {"future_dt": [.01, .005], "history_dt": [.0025, .00125], "dtype": "float64", "device": "cpu", "threads": 1},
        "runtime": {"python": sys.version, "torch": sgm.torch.__version__, "numpy": sgm.np.__version__, "executable": str(Path(sys.executable).resolve())},
        "optimizer_calls": 0, "val_read": False, "test_read": False, "new_fom": 0,
        "model_adopted": False, "selection_rule": None}
    path = PRIVATE / "manifest.json"
    if path.exists():
        if read(path) != manifest:
            raise ValueError("immutable manifest differs; do not reuse or overwrite this run")
    else:
        write_once(path, manifest)
    return manifest, sha(path)


def validate_predictions(evidence, events):
    """Check original immutable arrays against the exact fit cohort and histories."""
    import numpy as np
    byid = {e["id"]: e for e in events}
    seen = []
    for entry in evidence["prediction_files"]:
        path = Path(entry["path"]).resolve()
        path.relative_to((ROOT / "private_data").resolve())
        if sha(path) != entry["sha256"]:
            raise ValueError("prediction digest changed")
        with np.load(path, allow_pickle=False) as data:
            ids = [str(x) for x in data["event_ids"]]
            if data["fine"].shape != (len(ids), 3001, 40) or data["coarse"].shape != (len(ids), 1501, 40):
                raise ValueError("paired prediction shape changed")
            if data["histories"].shape != (len(ids), 11, 30):
                raise ValueError("history shape changed")
            for eid, history in zip(ids, data["histories"]):
                if eid not in byid or not np.array_equal(history, byid[eid]["x"][:11]):
                    raise ValueError("prediction event/history differs from frozen fit")
            seen.extend(ids)
    if len(seen) != len(events) or len(set(seen)) != len(events) or set(seen) != set(byid):
        raise ValueError("prediction fit_ids incomplete or duplicated")
    if evidence.get("event_count") != len(events) or evidence.get("val_read") is not False or evidence.get("test_read") is not False:
        raise ValueError("evidence scope invalid")
    if evidence.get("history_dt_main") != .0025 or evidence.get("history_dt_reference") != .00125:
        raise ValueError("paired causal history refinement missing")
    for rows in evidence["rows"].values():
        if any(r["event_id"] not in byid for r in rows):
            raise ValueError("foreign event in metric rows")
    extras = evidence["extras"]
    if len(extras) != len(events) or {e["event_id"] for e in extras} != set(byid):
        raise ValueError("extras fit_ids changed")


def reuse_evidence(label, model, events):
    """No rollout: recompute missing actual/T summaries from original paired arrays."""
    import numpy as np
    import state_gated_memory_study_20261028 as sgm
    folder, expected, model_hash = REUSE[label]
    if sha(folder / "evidence.json") != expected or sha(model_spec(label)["export"]) != model_hash:
        raise ValueError("reuse binding differs")
    evidence = read(folder / "evidence.json")
    complete = read(folder / "complete.json")
    if complete != {"status": "COMPLETED", "event_count": 297, "evidence_sha256": expected}:
        raise ValueError("original completion receipt differs")
    validate_predictions(evidence, events)
    extras = {e["event_id"]: e for e in evidence["extras"]}
    byid = {e["id"]: e for e in events}
    for entry in evidence["prediction_files"]:
        with np.load(entry["path"], allow_pickle=False) as data:
            for eid, h, f, c in zip(data["event_ids"], data["histories"], data["fine"], data["coarse"]):
                eid = str(eid)
                extras[eid].update(sgm.actual_extra(model, byid[eid], h, f, c))
                extras[eid]["coarse_point_errors"] = sgm._coarse_points(model, byid[eid], c)
    profile = sgm.profile(evidence["rows"], list(extras.values()), events, evidence["seconds"], model.complexity_counts()["raw_trainable_scalar_count"])
    original_profile = read(folder / "state_gated_profile.json")
    if profile != original_profile:
        raise ValueError("reconstructed original decision profile differs")
    evidence["decision_profile"] = profile
    return evidence


def classification_detail(extras, profile, key):
    """Preserve original missed/false counts; expose category and error identities."""
    from frequency_dissipation_20261021 import actual15_qualified
    actual = key == "classification_15s"
    original = profile["classification_actual15s" if actual else "classification_8s"]
    rows = []
    for extra in extras:
        pair = extra[key]
        qualified = bool(actual15_qualified(extra["actual_interval15"], pair) if actual else extra["scientific_numeric_8s"])
        c = pair["fine_classification"]
        rows.append({"event_id": extra["event_id"], "label": extra["label"], "numerically_admissible": qualified,
            "category": c["category"], "predicted_los": c["predicted_los"], "trigger_index": c.get("trigger_index"),
            "missed_resolved_unstable": qualified and extra["label"] == "unstable" and not c["predicted_los"],
            "false_los_resolved_stable": qualified and extra["label"] == "stable" and c["predicted_los"]})
    ids = lambda predicate: sorted(r["event_id"] for r in rows if predicate(r))
    if sum(r["missed_resolved_unstable"] for r in rows) != original["missed_resolved_unstable"] or sum(r["false_los_resolved_stable"] for r in rows) != original["false_los_resolved_stable"]:
        raise ValueError("classification identity projection differs from original counts")
    return {"original_counts": {k: v for k, v in original.items() if k != "rows"}, "denominator": len(rows),
        "missed_ids": ids(lambda r: r["missed_resolved_unstable"]), "false_los_ids": ids(lambda r: r["false_los_resolved_stable"]),
        "numerical_unresolved_ids": ids(lambda r: not r["numerically_admissible"]),
        "latency_uncovered_ids": ids(lambda r: r["category"] == "latency_uncovered"),
        "window_no_los_unsettled_ids": ids(lambda r: r["category"] == "window_no_los_unsettled"), "rows": rows,
        "note": "Original numerical and classification definitions; category sets can overlap numerical_unresolved."}


def timing_common(profiles):
    """Original sgm.timing_comparison common-T/E/u arithmetic, without a winner rule."""
    if not profiles:
        return {"status": "UNRESOLVED", "common_event_ids": [], "metrics": {}}
    ids = set.intersection(*(set(p["timing_actual15"]) for p in profiles.values()))
    common = [eid for eid in sorted(ids) if all(p["timing_actual15"][eid]["numerically_qualified"] and
        all(p["timing_actual15"][eid][k]["time_s"] is not None for k in ("truth", "fine", "coarse")) for p in profiles.values())]
    metrics = {}
    if common:
        for label, profile in profiles.items():
            rows = [profile["timing_actual15"][eid] for eid in common]
            metrics[label] = {"E": sum(abs(r["fine"]["time_s"] - r["truth"]["time_s"]) for r in rows) / len(rows),
                              "u": max(abs(r["fine"]["time_s"] - r["coarse"]["time_s"]) for r in rows)}
    return {"status": "COMPLETED" if common else "UNRESOLVED", "common_event_ids": common, "event_count": len(common), "metrics": metrics}


def progress(label, status, **kwargs):
    print(json.dumps({"checkpoint": label, "status": status, **kwargs}, ensure_ascii=False), flush=True)


def verify_complete(folder, manifest_sha):
    receipt = read(folder / "complete.json")
    if receipt["manifest_sha256"] != manifest_sha or receipt["summary_sha256"] != sha(folder / "summary.json"):
        raise ValueError("completed checkpoint receipt differs")
    for item in receipt["artifacts"]:
        path = (ROOT / item["path"]).resolve()
        path.relative_to(ROOT.resolve())
        if sha(path) != item["sha256"]:
            raise ValueError("completed artifact changed")
    return read(folder / "summary.json")


def evaluate_one(label, events, prepared, manifest_sha, *, retry_failed=False):
    import closure_probe_20261030 as probe
    import state_gated_memory_study_20261028 as sgm
    folder = PRIVATE / label
    if (folder / "complete.json").exists():
        result = verify_complete(folder, manifest_sha)
        progress(label, "SKIPPED_VERIFIED_COMPLETED")
        return result
    attempts = sorted(folder.glob("attempt*"))
    if attempts:
        if not (attempts[-1] / "failure.json").is_file():
            raise RuntimeError(f"{label}: attempt has no terminal receipt; reconcile its real process before any replay")
        if not retry_failed:
            raise RuntimeError(f"{label}: preserved terminal failure; explicit --retry-failed required")
    attempt = folder / f"attempt{len(attempts) + 1:03d}"
    write_once(attempt / "started.json", {"manifest_sha256": manifest_sha, "checkpoint": label, "pid": os.getpid(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "optimizer_calls": 0})
    started = time.perf_counter()
    progress(label, "STARTED", mode="reuse_original_paired_evidence" if label in REUSE else "fresh_paired_evaluation")
    try:
        model = probe.load_model(model_spec(label))
        if hasattr(model, "history_dt") and model.history_dt != .0025:
            raise ValueError("registered model history_dt differs")
        with sgm.torch.no_grad():
            evidence = reuse_evidence(label, model, events) if label in REUSE else sgm.evaluate(model, events, attempt / "paired", prepared)
        validate_predictions(evidence, events)
        profile = evidence["decision_profile"]
        details = {"8s": classification_detail(evidence["extras"], profile, "classification_8s"),
                   "actual15": classification_detail(evidence["extras"], profile, "classification_15s")}
        event_rows = {"schema": SCHEMA, "checkpoint": label, "classification": details, "objective_point_error": evidence["rows"]["objective_point_error"],
            "timing_actual15": profile["timing_actual15"], "actual_interval15": {e["event_id"]: e["actual_interval15"] for e in evidence["extras"]},
            "numerical": evidence["rows"]["numerical"], "scientific_numeric_8s": {e["event_id"]: e["scientific_numeric_8s"] for e in evidence["extras"]}}
        write_once(attempt / "per_event.json", sgm.safe(event_rows))
        result = {"schema": SCHEMA, "checkpoint": label, "status": "COMPLETED", "manifest_sha256": manifest_sha,
            "model": identity(model_spec(label)["export"]), "cohort": {"train": "fit297", "event_count": 297, "stable": 267, "unstable": 30},
            "decision_profile": profile, "classification": {scope: {k: v for k, v in d.items() if k != "rows"} for scope, d in details.items()},
            "original_inference_cost": {"seconds": evidence["seconds"], "timing": evidence.get("timing"), "scope": evidence.get("timing_scope"), "reused": label in REUSE},
            "current_audit_seconds": time.perf_counter() - started, "fresh_rollouts": label not in REUSE,
            "optimizer_calls": 0, "val_read": False, "test_read": False, "new_fom": 0, "model_adopted": False,
            "per_event": identity(attempt / "per_event.json"), "prediction_files": evidence["prediction_files"]}
        # Check every source/data/model binding again before publishing a completion.
        manifest = read(PRIVATE / "manifest.json")
        for path, digest in manifest["source"].items():
            if sha(ROOT / path) != digest:
                raise ValueError("source changed during checkpoint evaluation")
        for item in [manifest["protocol"], manifest["data"]["fit_cache"], manifest["data"]["prepare"], *manifest["models"].values()]:
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("protocol/data/model changed during checkpoint evaluation")
        write_once(folder / "summary.json", sgm.safe(result))
        artifacts = [identity(attempt / "per_event.json")]
        artifacts += [{"path": Path(e["path"]).resolve().relative_to(ROOT.resolve()).as_posix(), "sha256": e["sha256"]} for e in evidence["prediction_files"]]
        # Aggregate-only public output: no event identity or prediction paths.
        public = {k: result[k] for k in ("schema", "checkpoint", "status", "manifest_sha256", "cohort", "current_audit_seconds", "fresh_rollouts", "optimizer_calls", "val_read", "test_read", "new_fom", "model_adopted")}
        public["metrics"] = {k: profile[k] for k in ("horizon_risks", "common_frequency_rmse_8s", "normalized_p_rmse_8s", "unstable_interval_risk", "unstable_actual15_risk", "all_required_rows_qualified")}
        public["metrics"]["horizon_risks"] = {h: {k: v for k, v in cell.items() if k != "unresolved_ids"} for h, cell in profile["horizon_risks"].items()}
        public["classification"] = {scope: {"original_counts": d["original_counts"], "denominator": d["denominator"], **{k.replace("_ids", "_count"): len(v) for k, v in d.items() if k.endswith("_ids")}} for scope, d in details.items()}
        write_once(PUBLIC / f"{label}.json", sgm.safe(public))
        artifacts.append(identity(PUBLIC / f"{label}.json"))
        write_once(folder / "complete.json", {"status": "COMPLETED", "manifest_sha256": manifest_sha, "summary_sha256": sha(folder / "summary.json"), "artifacts": artifacts})
        progress(label, "COMPLETED", seconds=result["current_audit_seconds"], qualified=profile["all_required_rows_qualified"])
        return result
    except BaseException as exc:
        write_once(attempt / "failure.json", {"status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started, "failure_is_physical_instability": False})
        progress(label, "FAILED", error=repr(exc))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["evaluate"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=LABELS, action="append")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    import closure_probe_20261030 as probe
    import state_gated_memory_study_20261028 as sgm
    sgm.torch.set_num_threads(1)
    prepared, events = probe.load_frozen_events()
    _, digest = register(args.protocol, prepared, events)
    labels = args.checkpoint or list(LABELS)
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate requested checkpoint")
    for label in labels:
        evaluate_one(label, events, prepared, digest, retry_failed=args.retry_failed)
    profiles = {}
    for label in LABELS:
        folder = PRIVATE / label
        if (folder / "complete.json").exists():
            profiles[label] = verify_complete(folder, digest)["decision_profile"]
    # Name includes the set, so extending the completed set never overwrites a summary.
    name = hashlib.sha256("|".join(profiles).encode()).hexdigest()[:16]
    comparison = {"schema": SCHEMA, "manifest_sha256": digest, "checkpoints": list(profiles), "all_eleven_completed": len(profiles) == 11,
        "timing_all_common": timing_common(profiles), "timing_against_references": {label: timing_common({k: profiles[k] for k in ("R_N", "A96", label)}) for label in profiles if label not in ("R_N", "A96") and "R_N" in profiles and "A96" in profiles},
        "selection_rule": None, "selected_checkpoint": None, "model_adopted": False}
    path = PRIVATE / f"comparison_{name}.json"
    if path.exists():
        if read(path) != comparison:
            raise ValueError("immutable comparison differs")
    else:
        write_once(path, comparison)
    progress("all", "COMPLETED", completed=list(profiles), comparison=str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
