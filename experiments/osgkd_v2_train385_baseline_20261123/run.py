"""Released read-only parent baseline; self-check never opens scientific payloads."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
from experiments.osgkd_v2_early_closure_20261123.run import identity, read, require, repo_path, verified
from kiki.evaluation.osgkd_validation import evaluate_train, evaluate_val
from kiki.models.osgkd import OSGKD

RUN = "osgkd_v2_train385_baseline_20261123"
PROTOCOL = "campaigns/osgkd_v2/train385_baseline_protocol.md"
SOURCES = {f"experiments/{RUN}/run.py", PROTOCOL, "src/kiki/data/osgkd_train_val.py",
           "src/kiki/data/osgkd.py", "src/kiki/evaluation/osgkd_validation.py",
           "src/kiki/evaluation/osgkd.py", "src/kiki/models/osgkd.py",
           "src/kiki/models/osgkd_fitting.py", "experiments/evaluate_l1_latent.py",
           "experiments/osgkd_v2_early_closure_20261123/run.py"}
INPUTS = {"clean_summary": "results/osgkd_v2_p1a_clean_parent_20261123/pair01/summary.json",
          "clean_analysis": "results/osgkd_v2_p1a_clean_parent_20261123/analysis.json",
          "p1b_analysis": "results/osgkd_v2_p1b_electrical_unlock_20261123/analysis.json"}
CHECKPOINT_SHA = "291853229dd6738a9132fcacce53d104f83d1a6045b1d3823733549759def49f"
PUBLIC_LIMIT = 512 * 1024
typed_ref = verified


def atomic_write(path, value, *, public=False):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    require(not public or len(raw) < PUBLIC_LIMIT, "public record exceeds 512 KiB; no detail fallback permitted")
    require(not path.exists(), "output already exists")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # Exclusive claim directory and exclusive temporary file prevent concurrent replacement.
    os.rename(temporary, path)
    return identity(path)


def public_summary(value):
    """Replace all private identity lists with counts, recursively."""
    if isinstance(value, dict):
        return {key[:-4] + "_count" if key.endswith("_ids") else key:
                len(item) if key.endswith("_ids") else public_summary(item)
                for key, item in value.items()}
    if isinstance(value, list):
        raise ValueError("public summary cannot contain per-event arrays")
    return value


def data_identity(events):
    """Bind canonical packets, including dtype/shape, without publishing their IDs."""
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, np.ndarray):
            require(not value.dtype.hasobject, "object array forbidden")
            visit({"dtype": value.dtype.str, "shape": list(value.shape)})
            raw = np.ascontiguousarray(value).tobytes()
            digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
        elif isinstance(value, dict):
            digest.update(b"{")
            for key in sorted(value):
                visit(key); visit(value[key])
            digest.update(b"}")
        elif isinstance(value, (tuple, list)):
            digest.update(b"[")
            for item in value:
                visit(item)
            digest.update(b"]")
        else:
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
    ordered = sorted(events, key=lambda e: e["id"])
    require(len({e["id"] for e in ordered}) == len(ordered), "duplicate packet IDs")
    visit([{key: event[key] for key in ("id", "group", "label", "history", "t", "targets", "masks",
                                      "metadata", "missing_reason", "original_points") if key in event}
           for event in ordered])
    return {"algorithm": "sha256-canonical-packets-v1", "sha256": digest.hexdigest(), "events": len(events)}


def distribution(values):
    return {"count": len(values), "mean_s": float(np.mean(values)) if values else None,
            "minimum_s": min(values) if values else None, "maximum_s": max(values) if values else None,
            "median_s": float(np.median(values)) if values else None}


def timing_summary(rows, events):
    unstable = [r for r in rows if r["label"] == "unstable"]
    counts = Counter({key: 0 for key in ("within_record", "tail", "no_trigger", "uncovered", "numerical_unresolved", "latency_uncovered")})
    lags, tails = [], []
    event_map = {e["id"]: e for e in events}
    actual_hits = 0
    for row in unstable:
        actual = row["classification_actual"]
        actual_hits += bool(actual and actual["los_pair_pass"] and actual["truth_supported"] and
                            actual["classification"]["predicted_los"])
        item = row["classification_8s"]
        if item is None:
            counts["uncovered"] += 1; continue
        category = item["classification"]["category"]
        if category in ("latency_uncovered", "numerical_unresolved"):
            counts[category] += 1; continue
        require(item["los_pair_pass"], "unqualified classifier category")
        trigger = item["predicted_time_record_s"]
        if trigger is None:
            counts["no_trigger"] += 1; continue
        event = event_map[row["event_id"]]
        truth_end = .1 + float(event["t"][-1])
        true = event["metadata"]["true_event_time_record_s"]
        category = "within_record" if trigger <= truth_end + 1e-12 else "tail"
        counts[category] += 1
        if true is not None and category == "tail":
            tails.append(trigger - true)
        elif true is not None and item["truth_supported"] and item["timing_comparable"]:
            lags.append(trigger - true)
    stable = [r for r in rows if r["label"] == "stable"]
    false = sum(bool(r["classification_8s"] and r["classification_8s"]["los_pair_pass"] and
                     r["classification_8s"]["truth_supported"] and
                     r["classification_8s"]["classification"]["predicted_los"]) for r in stable)
    return {"unstable_raw_denominator": len(unstable),
            "unstable_input_eligible_denominator": sum(event_map[r["event_id"]]["history"] is not None for r in unstable),
            "actual_interval_hits": actual_hits, "autonomous_8s": dict(counts),
            "qualified_within_record_signed_lag": distribution(lags),
            "qualified_within_record_absolute_lag": distribution([abs(x) for x in lags]), "qualified_tail_signed_lag": distribution(tails),
            "stable_8s_false_alarms": false, "stable_denominator": len(stable)}


def validate(release):
    require(tuple(release.get(k) for k in ("run_id", "job_id", "released", "device", "threads", "recompute")) ==
            (RUN, "eval01", True, "cpu", 1, False), "fixed eval01 CPU release required")
    require(SOURCES <= release.get("source_hashes", {}).keys(), "source hashes incomplete")
    for name, sha in release["source_hashes"].items():
        require(identity(repo_path(name))["sha256"] == sha, "source identity differs: " + name)
    require(set(release["input_refs"]) == set(INPUTS), "exact input_refs required")
    inputs = {}
    for key, name in INPUTS.items():
        require(release["input_refs"][key]["path"] == name, "noncanonical input reference")
        inputs[key] = read(release["input_refs"][key])
    summary, clean, p1b = (inputs[k] for k in INPUTS)
    choice = clean["choice"]
    require(summary["status"] == "completed" and summary["result"]["fixed_endpoint"] == 1860 and
            not summary["result"]["stopped_early"] and clean["job"] == release["input_refs"]["clean_summary"] and
            choice in ("control", "candidate") and clean["gates"][choice]["passed"], "qualified clean parent required")
    require(clean.get("run_id") == "osgkd_v2_p1a_clean_parent_20261123" and
            p1b.get("run_id") == "osgkd_v2_p1b_electrical_unlock_20261123", "analysis identity differs")
    ref = release["checkpoint_ref"]
    require(ref == summary["result"]["treatments"][choice]["checkpoint"] == clean["inputs"][choice]["checkpoint"] ==
            p1b["inputs"]["parent"]["checkpoint"] and ref["sha256"] == CHECKPOINT_SHA, "parent checkpoint binding differs")
    require(ref["path"] == f"artifacts/osgkd_v2_p1a_clean_parent_20261123/pair01/{choice}/attempt_1860.pt",
            "canonical parent checkpoint required")
    verified(ref)
    return summary


def score_split(model, events, constants, split, directory):
    require(split in ("train", "val"), "train or val required")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    packet_identity = data_identity(events)
    atomic_write(directory / "started.json", {"status":"started", "data_identity":packet_identity})
    result = (evaluate_train if split == "train" else evaluate_val)(model, events, constants)
    eligible = [e for e in events if e["history"] is not None]
    expected_rhs = sum(24*(max(800,min(1500,len(e["t"])-1)) if e["label"] == "unstable" else 800) for e in eligible)
    work = result["summary"]["work"]
    require(work == {"rhs_forward":expected_rhs,"rhs_recompute":0,"encoder_events":len(eligible)},
            "evaluation work differs from registered numerical contract")
    rows_ref = atomic_write(directory / "rows.json", result["rows"])
    metrics_ref = atomic_write(directory / "metrics.json", result["summary"])
    timing = timing_summary(result["rows"], events)
    classes = result["summary"]["classification"][split]
    require(timing["actual_interval_hits"] == len(classes["actual"]["detected_unstable_ids"]) and
            timing["stable_8s_false_alarms"] == len(classes["8s"]["false_los_ids"]) and
            sum(timing["autonomous_8s"].values()) == timing["unstable_raw_denominator"], "classifier tally differs")
    unstable_rows = [r for r in result["rows"] if r["label"] == "unstable"]
    autonomous = timing["autonomous_8s"]
    qualified_los = sum(bool(r["classification_8s"] and r["classification_8s"]["los_pair_pass"] and
                             r["classification_8s"]["classification"]["predicted_los"]) for r in unstable_rows)
    require(autonomous["within_record"] + autonomous["tail"] == qualified_los, "autonomous LOS tally differs")
    for category in ("latency_uncovered", "numerical_unresolved"):
        require(autonomous[category] == sum(bool(r["classification_8s"] and
                r["classification_8s"]["classification"]["category"] == category) for r in unstable_rows),
                "classifier category tally differs")
    completion = atomic_write(directory / "completed.json", {"status":"completed", "rows_ref":rows_ref,"metrics_ref":metrics_ref})
    print(json.dumps({"split":split,"status":"completed","work":work}),flush=True)
    return {"data_identity":packet_identity,"summary":public_summary(result["summary"]),"timing":timing,
            "rows_ref":rows_ref,"metrics_ref":metrics_ref,"completed_ref":completion}


def execute(path):
    start, cpu = perf_counter(), process_time()
    path = path.resolve(); require(path.is_relative_to(ROOT), "repository release required")
    release = json.loads(path.read_bytes()); parent = validate(release)
    local, public = (ROOT / side / RUN / "eval01" for side in ("artifacts", "results"))
    require(not local.exists() and not public.exists(), "eval01 already claimed; no rerun permitted")
    local.mkdir(parents=True, exist_ok=False)
    receipt = {"run_id": RUN, "job_id": "eval01", "release_ref": identity(path), "release": release,
               "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
               "command": [sys.executable, *sys.argv], "environment": {"python": sys.version.split()[0],
               "torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu", "dtype": "float64", "threads": 1}}
    atomic_write(local / "started.json", receipt)
    public.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(1)
        from kiki.data.osgkd_train_val import load_train, load_val
        train, constants = load_train(); val = load_val(constants)
        require(not {e["id"] for e in train} & {e["id"] for e in val}, "split ID overlap")
        require(not {e["group"] for e in train} & {e["group"] for e in val}, "split group overlap")
        require(Counter(e["label"] for e in train) == {"stable":342,"unstable":43} and
                Counter(e["label"] for e in val) == {"stable":95,"unstable":22} and
                all(e["history"] is not None for e in train) and
                Counter(e["label"] for e in val if e["history"] is None) == {"unstable":10}, "split population differs")
        saved = torch.load(verified(release["checkpoint_ref"]), map_location="cpu", weights_only=True)
        require(saved["attempt"] == 1860 and saved["model"]["config"]["variant"] == "C1" and
                saved["provenance"]["audit_ref"] == parent["release"]["input_refs"]["audit"], "audited parent_1860 C1 required")
        model = OSGKD.from_checkpoint(saved["model"]).cpu().double().eval().requires_grad_(False)
        require(model.n == 10, "ten-machine model required")
        for key in ("mass", "damping", "F", "Sp", "Se", "omega_scale", "theta_ref"):
            expected = np.asarray(constants[key])
            if key in ("mass", "damping"):
                expected = np.diag(expected)
            require(np.array_equal(getattr(model,key).numpy(), expected), "model frozen constant differs: " + key)
        edges = (np.stack(np.triu_indices(10, k=1)) if
                 constants.get("candidate_graph") == "undirected_complete_K10" else np.asarray(constants["edge_index"]))
        require(np.array_equal(model.edge_index.numpy(), edges), "frozen candidate graph differs")
        outputs = {}
        for split, events in (("train", train), ("val", val)):
            outputs[split] = score_split(model, events, constants, split, local / split)
        encoders = sum(v["summary"]["work"]["encoder_events"] for v in outputs.values())
        require(encoders == 492, "exact 492 initialized events required")
        evaluated_rhs = sum(v["summary"]["work"]["rhs_forward"] for v in outputs.values())
        require(evaluated_rhs <= 10_370_400, "RHS cap exceeded")
        phases = {"training":{"updates":0,"event_paths":0,"rhs_forward":0},
                  "evaluation":{"updates":0,"event_paths":2*encoders,"rhs_forward":evaluated_rhs}, "new_fom":0}
        receipt.update(status="completed", result=outputs, phases=phases, cost={"wall_s":perf_counter()-start,"cpu_s":process_time()-cpu},
                       completed_utc=datetime.now(timezone.utc).isoformat())
        ref = atomic_write(public / "summary.json", receipt, public=True)
        atomic_write(local / "completed.json", {"status":"completed", "summary_ref":ref})
    except Exception as error:
        atomic_write(local / "failed.json", {"status":"failed","error":repr(error)})
        raise


def self_check():
    event = {"id":"synthetic", "group":"synthetic", "label":"stable", "history":{"theta":np.zeros((11,10))}}
    original = data_identity([event]); event["history"]["theta"][0,0] = 1.
    assert data_identity([event]) != original
    assert public_summary({"missing_history_ids":["private"],"metric":{"official":None}}) == {
        "missing_history_count":1,"metric":{"official":None}}
    try:
        public_summary(["private"])
    except ValueError:
        pass
    else:
        raise AssertionError("public arrays accepted")
    assert distribution([])["mean_s"] is None
    item = {"los_pair_pass":True,"truth_supported":False,"predicted_time_record_s":.4,
            "classification":{"predicted_los":True,"category":"los"}}
    row = {"event_id":"u", "label":"unstable", "classification_actual":None,"classification_8s":item}
    packet = {"id":"u","history":{},"t":np.arange(11)*.01,
              "metadata":{"true_event_time_record_s":.2}}
    timing = timing_summary([row],[packet])
    assert timing["autonomous_8s"]["tail"] == 1 and timing["actual_interval_hits"] == 0
    assert timing["qualified_within_record_signed_lag"]["count"] == 0
    assert np.isclose(timing["qualified_tail_signed_lag"]["mean_s"],.2)
    missing = {**row,"classification_8s":None}
    assert timing_summary([missing],[{**packet,"history":None}])["autonomous_8s"]["uncovered"] == 1
    print(json.dumps({"self_check":"passed","scientific_payloads_opened":False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    sub.add_parser("evaluate").add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    self_check() if args.command == "self-check" else execute(args.release)
