"""Apply the frozen clean-parent gates to the two completed endpoint evaluations."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
from experiments.osgkd_v2_early_closure_20261123.run import identity, read, require, verified, write
from kiki.data.osgkd import load_fit297
from kiki.evaluation.osgkd import motion_metrics, risk, summarize

RUN = "osgkd_v2_p1a_clean_parent_20261123"
LOCAL = ROOT / "artifacts" / RUN / "pair01"
PUBLIC = ROOT / "results" / RUN
ARMS = ("control", "candidate")
HITS = {"expansion_0106", "pilot_0207", "pilot_0254"}
SCOPES = ("stable_1s", "stable_2s", "stable_5s", "stable_8s", "unstable_actual")


def compare_gates(scores, rows, parent, parent_rows):
    stable = scores["splits"]["train_all"]
    reference = parent["splits"]["train_all"]
    classification = scores["classification"]["train_all"]
    triggered = set(classification["actual"]["detected_unstable_ids"])
    times = {}
    for event in sorted(HITS):
        before = parent_rows[event]["classification_actual"]
        after = rows[event]["classification_actual"]
        times[event] = [x["predicted_time_record_s"] if x["timing_comparable"] else None for x in (before, after)]
    covered = (
        len(rows) == 297 and all(r["classification_8s"]["los_pair_pass"] for r in rows.values())
        and all(stable[key]["covered"] == stable[key]["denominator"] for key in SCOPES)
        and all(not classification["8s"][k] for k in ("numerical_unresolved_ids", "latency_uncovered_ids"))
        and all(not classification["actual"][k] for k in ("numerical_unresolved_ids", "latency_uncovered_ids", "uncovered_ids"))
    )
    values = {key: stable[key]["official"]["motion_risk_v1"] if stable[key]["official"] else None for key in SCOPES}
    checks = {"original_coverage": covered,
              "stable_8s_risk": values["stable_8s"] is not None and values["stable_8s"] < 1.,
              "stable_8s_false_alarms": not classification["8s"]["false_los_ids"],
              "parent_hits_preserved": HITS <= triggered,
              "parent_trigger_timing": all(a is not None and b is not None and b <= a+.01+1e-12
                                           for a, b in times.values())}
    for key in ("stable_1s", "stable_2s"):
        checks[key] = values[key] is not None and values[key] <= 1.05 * reference[key]["official"]["motion_risk_v1"]
    return {"passed": all(checks.values()), "checks": checks, "fine_risks": values, "parent_hit_times_record_s": times}


def main_risk(ref, events, constants):
    mass = np.asarray(constants["mass"])
    weights = mass / mass.sum()
    scale = np.asarray(constants["omega_scale"])
    scores = []
    with np.load(verified(ref), allow_pickle=False) as paths:
        for index, event in enumerate(events):
            if event["label"] == "stable":
                target = event["targets"]
                truth = np.column_stack((target["theta"][1:801], target["omega"][1:801]))
                prediction = paths[f"event_{index:04d}_main"][1:801, :20]
                require(truth.shape == prediction.shape == (800, 20), "variable stable horizon")
                scores.append(motion_metrics(prediction, truth, weights, scale, constants["s_coi"])["mean_motion_error"])
    require(len(scores) == 267, "stable score count differs")
    return risk(scores)


def choose(gates, main):
    usable = [name for name in ARMS if gates[name]["passed"]]
    if len(usable) < 2:
        return usable[0] if usable else "parent"
    fine = {name: gates[name]["fine_risks"]["stable_8s"] for name in ARMS}
    return "candidate" if fine["candidate"] < fine["control"] and main["candidate"] < main["control"] else "control"


def main():
    started, cpu = perf_counter(), process_time()
    marker = json.loads((LOCAL / "completed.json").read_bytes())
    receipt = read(marker["summary_ref"])
    result = receipt["result"]
    require(receipt["run_id"] == RUN and receipt["status"] == "completed" and result["fixed_endpoint"] == 1860,
            "completed fixed paired endpoints required")
    release = read(receipt["release_ref"])
    require(release == receipt["release"], "release content differs")
    for name, digest in release["source_hashes"].items():
        require(identity(ROOT / name)["sha256"] == digest, "source differs: " + name)
    parent_analysis = read(release["parent_analysis_ref"])
    parent_input = parent_analysis["inputs"]["control"]
    require(parent_input["checkpoint_ref"] == release["checkpoint_ref"], "wrong analysis parent")
    parent = parent_analysis["official_scores"]["control"]
    require(set(parent["classification"]["train_all"]["actual"]["detected_unstable_ids"]) == HITS,
            "original hit identities differ")
    parent_rows = {r["event_id"]: r for r in read(parent_input["rows_ref"])}
    events, constants = load_fit297(release["input_refs"]["audit"])
    order = read(result["paired_batch_sequence_ref"])
    require(len(order) == 128 and result["training_rhs_forward"] <= 13_107_200, "paired limits differ")
    official, gates, inputs, main_scores = {}, {}, {}, {}
    for name in ARMS:
        treatment = result["treatments"][name]
        history = read(treatment["attempts_ref"])
        require([r["event_ids"] for r in history] == order and [r["attempt"] for r in history] == list(range(1733, 1861)),
                "not the frozen paired sequence")
        root = LOCAL / name / "evaluation"
        summary = read(treatment["evaluation_summary_ref"])
        refs = {"rows": identity(root / "rows.json"), "predictions": identity(root / "predictions.npz")}
        rows = read(refs["rows"])
        require([r["event_id"] for r in rows] == [e["id"] for e in events], "evaluation order differs")
        scores = summarize(rows, constants)
        require(scores == {k: summary[k] for k in scores}, "cached scores differ")
        official[name] = scores
        gates[name] = compare_gates(scores, {r["event_id"]: r for r in rows}, parent, parent_rows)
        inputs[name] = {**refs, "checkpoint": treatment["checkpoint"], "summary": treatment["evaluation_summary_ref"]}
    if all(g["passed"] for g in gates.values()):
        main_scores = {name: main_risk(inputs[name]["predictions"], events, constants) for name in ARMS}
    write(PUBLIC / "analysis.json",
          {"run_id": RUN, "job": marker["summary_ref"], "parent_analysis_ref": release["parent_analysis_ref"],
           "inputs": inputs, "gates": gates, "official_scores": {"parent": parent, **official},
           "choice": choose(gates, main_scores), "main_stable_8s_risks_for_tie_check": main_scores,
           "cost": {"wall_s": perf_counter()-started, "cpu_s": process_time()-cpu}})


if __name__ == "__main__":
    main()
