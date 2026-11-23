"""Evaluate the preregistered short-window and fixed-endpoint P1b contrast."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
from experiments.osgkd_v2_early_closure_20261123.run import decomposition, identity, read, require, verified, write
from kiki.data.osgkd import load_fit297
from kiki.evaluation.osgkd import motion_metrics, port_metrics, summarize
from kiki.models.osgkd import OSGKD

RUN = "osgkd_v2_early_electrical_20261123"
LOCAL = ROOT / "artifacts" / RUN / "pair01"
PUBLIC = ROOT / "results" / RUN


@torch.no_grad()
def predict_ports(model, state):
    ports = model.ports(torch.as_tensor(state, dtype=torch.float64))
    return {"omega": state[:, 10:20], "tau_m": ports.tau_m.numpy(), "tau_e": ports.tau_e.numpy()}


def early_score(event, state, ports, mask, steps, constants):
    motion = None
    speed = None
    if np.isfinite(state[:, :20]).all():
        mass = np.asarray(constants["mass"])
        weights = mass / mass.sum()
        truth = np.column_stack((event["targets"]["theta"], event["targets"]["omega"]))[:steps+1]
        motion = motion_metrics(
            state[1:, :20], truth[1:], weights, np.asarray(constants["omega_scale"]), constants["s_coi"],
        )["mean_motion_error"]
        speed = float((state[-1, 10:20] - truth[-1, 10:20]) @ weights)**2
    result = {"motion_mse": motion, "common_speed_endpoint_squared_error": speed}
    for key, scale in (("tau_m", "S_m"), ("tau_e", "Se")):
        observed = event["targets"][key][1:steps+1]
        original_mask = event["masks"][key][1:steps+1]
        prediction = ports[key][1:]
        score = (port_metrics(prediction, observed, original_mask, np.asarray(constants[scale]))
                 if np.isfinite(prediction[original_mask]).all() else None)
        result[f"{key}_normalized_mse"] = score["normalized_sse"] / score["count"] if score and score["count"] else None
    obs = {key: event["targets"][key][:steps+1] for key in ports}
    result["net_impulse_mse"] = decomposition(obs, ports, constants, mask, steps)["squared_means"]["E"]
    return result


def evaluate_prefixes(events, constants, models, archives):
    rows, evaluated_points = [], 0
    for i, event in enumerate(events):
        stop = min(25, len(event["t"])-1)
        require(stop >= 3, "expected at least the frozen 0.03 s window")
        states = {name: file[f"event_{i:04d}_fine"][:stop+1] for name, file in archives.items()}
        require(all(value.shape == (stop+1, 50) for value in states.values()), "saved trajectory shape differs")
        ports = {name: predict_ports(models[name], state) for name, state in states.items()}
        evaluated_points += len(states) * (stop+1)
        record = {"event_id": event["id"], "label": event["label"], "windows": {}}
        for label, steps in (("0.03", 3), ("capped_0.25", stop)):
            window = slice(0, steps+1)
            mask = np.ones(10, dtype=bool)
            for key in ("theta", "omega", "tau_m", "tau_e"):
                mask &= event["masks"][key][window].all(axis=0)
            for prediction in ports.values():
                for value in prediction.values():
                    mask &= np.isfinite(value[window]).all(axis=0)
            record["windows"][label] = {
                "h_s": .01 * steps,
                "arms": {name: early_score(event, state[window], {k: v[window] for k, v in ports[name].items()},
                                           mask, steps, constants) for name, state in states.items()},
            }
        rows.append(record)
    return rows, evaluated_points


def early_aggregate(rows, cohorts):
    groups = {}
    for group, ids in cohorts.items():
        groups[group] = {}
        for window in ("0.03", "capped_0.25"):
            groups[group][window] = {}
            for arm in ("parent", "control", "candidate"):
                values = [row["windows"][window]["arms"][arm] for row in rows if row["event_id"] in ids]
                metrics = {}
                for key in values[0]:
                    observed = [x[key] for x in values if x[key] is not None]
                    metrics[key] = {"covered": len(observed),
                                    "mean": float(np.mean(observed)) if len(observed) == len(ids) else None}
                groups[group][window][arm] = {"denominator": len(ids), "metrics": metrics}
    return groups


def practical_gates(scores, early, full_rows):
    stable, classes = ({name: score[key]["train_all"] for name, score in scores.items()} for key in ("splits", "classification"))
    hits = set(classes["parent"]["actual"]["detected_unstable_ids"])
    times = {}
    for event_id in sorted(hits):
        times[event_id] = {}
        for arm, rows in full_rows.items():
            entry = rows[event_id]["classification_actual"]
            times[event_id][arm] = (entry["predicted_time_record_s"]
                                    if entry and entry["timing_comparable"] else None)
    early_misses = early["parent_misses"]["capped_0.25"]

    def within(after, reference, factor):
        return after is not None and reference is not None and after <= factor*reference

    checks = {
        "numerical_and_coverage": all(classes[name]["8s"][key] == [] for name in ("control", "candidate")
                                      for key in ("numerical_unresolved_ids", "latency_uncovered_ids"))
        and all(not classes[name]["actual"][key] for name in ("control", "candidate")
                for key in ("numerical_unresolved_ids", "latency_uncovered_ids", "uncovered_ids"))
        and all(len(full_rows[name]) == 297 and all(row["classification_8s"]["los_pair_pass"] for row in full_rows[name].values())
                and all(stable[name][key]["covered"] == stable[name][key]["denominator"]
                        for key in ("stable_1s", "stable_2s", "stable_5s", "stable_8s", "unstable_actual"))
                for name in ("control", "candidate")),
        "early_electrical": within(early_misses["candidate"]["metrics"]["tau_e_normalized_mse"]["mean"],
                                   early_misses["control"]["metrics"]["tau_e_normalized_mse"]["mean"], .90),
        "early_motion": within(early_misses["candidate"]["metrics"]["motion_mse"]["mean"],
                               early_misses["control"]["metrics"]["motion_mse"]["mean"], .95),
        "stable_8s_false_alarms": not classes["candidate"]["8s"]["false_los_ids"],
        "parent_hits_preserved": hits <= set(classes["candidate"]["actual"]["detected_unstable_ids"]),
        "parent_trigger_timing": all(x["candidate"] is not None and x["parent"] is not None and
                                     all(x["candidate"] <= x[name] + .01 + 1e-12
                                         for name in ("parent", "control") if x[name] is not None)
                                     for x in times.values()),
    }
    for h in (1, 2, 8):
        scope = f"stable_{h}s"
        risks = {name: group[scope]["official"]["motion_risk_v1"] if group[scope]["official"] else None
                 for name, group in stable.items()}
        checks[f"stable_{h}s_risk"] = all(within(risks["candidate"], risks[name], 1.05)
                                        for name in ("control", "parent"))
    return {"checks": checks, "passed": all(checks.values()), "parent_hit_times_record_s": times}


def main():
    start, cpu = perf_counter(), process_time()
    torch.set_num_threads(1)
    completion = json.loads((LOCAL / "completed.json").read_bytes())
    receipt = read(completion["summary_ref"])
    result = receipt["result"]
    require(receipt["run_id"] == RUN and receipt["status"] == "completed" and result["qualification"]["passed"]
            and not result["stopped_early"] and result["actual_endpoint"] == 1732,
            "two fixed evaluated endpoints required")
    release = read(receipt["release_ref"])
    require(release == receipt["release"], "release content changed")
    for name, expected in release["source_hashes"].items():
        require(identity(ROOT / name)["sha256"] == expected, "frozen source differs: " + name)
    force_diagnostic = read(release["early_closure_ref"])
    events, constants = load_fit297(release["audit_ref"])
    inputs = {"parent": force_diagnostic["release"]["models"]["long"]}
    for arm in ("control", "candidate"):
        directory = f"artifacts/{RUN}/pair01/{arm}/evaluation"
        treatment = result["treatments"][arm]
        require(treatment["evaluation_summary_ref"]["path"] == directory + "/summary.json", "unexpected evaluation path")
        inputs[arm] = {"checkpoint_ref": treatment["checkpoint"], "rows_ref": identity(ROOT / directory / "rows.json"),
                       "predictions_ref": identity(ROOT / directory / "predictions.npz")}
    with ExitStack() as stack:
        models, archives, all_rows, scores = {}, {}, {}, {}
        parent_pair = read(release["parent_pair_ref"])
        for arm, refs in inputs.items():
            saved = torch.load(verified(refs["checkpoint_ref"]), map_location="cpu", weights_only=True)
            require(saved["attempt"] == (1700 if arm == "parent" else 1732), "checkpoint endpoint differs")
            models[arm] = OSGKD.from_checkpoint(saved["model"]).eval().requires_grad_(False)
            archives[arm] = stack.enter_context(np.load(verified(refs["predictions_ref"]), allow_pickle=False))
            rows = read(refs["rows_ref"])
            require([x["event_id"] for x in rows] == [e["id"] for e in events], "unexpected evaluation event order")
            all_rows[arm] = {row["event_id"]: row for row in rows}
            scores[arm] = summarize(rows, constants)
            stored = read(parent_pair["result"]["treatments"]["8s"]["evaluation_summary_ref"] if arm == "parent"
                          else result["treatments"][arm]["evaluation_summary_ref"])
            require(scores[arm] == {key: stored[key] for key in scores[arm]}, "saved evaluator summary differs")
        prefixes, work = evaluate_prefixes(events, constants, models, archives)
    misses = set(force_diagnostic["result"]["group_event_ids"]["long_misses"])
    require(len(misses) == 27 and misses == set(scores["parent"]["classification"]["train_all"]["actual"]["missed_unstable_ids"]),
            "parent miss group differs")
    cohorts = {
        "train_all": {e["id"] for e in events}, "stable": {e["id"] for e in events if e["label"] == "stable"},
        "unstable": {e["id"] for e in events if e["label"] == "unstable"}, "parent_misses": misses,
        "train_fit": set(constants["fit_ids"]), "train_check": set(constants["check_ids"]),
    }
    early = early_aggregate(prefixes, cohorts)
    gates = practical_gates(scores, early, all_rows)
    rows_ref = write(LOCAL / "early_metric_rows.json", prefixes)
    analysis = {"run_id": RUN, "source": identity(Path(__file__).resolve()), "job": completion["summary_ref"],
                "inputs": inputs, "early_rows_ref": rows_ref, "early": early, "official_scores": scores,
                "practical_gates": gates, "port_state_points": work,
                "cost": {"wall_s": perf_counter()-start, "cpu_s": process_time()-cpu}}
    write(PUBLIC / "analysis.json", analysis)


if __name__ == "__main__":
    main()
