"""Registered motion-only metrics on complete fit297 denominators."""
from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import torch

from experiments.evaluate_l1_latent import apply_numerical_gate, classify_prediction
from kiki.models.osgkd_fitting import IntegrationWork, integrate


def risk(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return None
    return float(.8 * values.mean() + .2 * np.sort(values)[-math.ceil(.1*len(values)):].mean())


def numerical_gate(main, fine, omega_scale):
    finite = bool(np.isfinite(main).all() and np.isfinite(fine).all())
    rms = maximum = None
    if finite:
        scale = np.r_[np.ones(10), omega_scale, np.full(10, 4.), np.ones(10), np.full(10, 4.)]
        with np.errstate(over="ignore", invalid="ignore"):
            difference = (main - fine) / scale
            rms = float(np.sqrt(np.mean(difference*difference)))
            maximum = float(np.max(np.abs(difference)))
        if not (math.isfinite(rms) and math.isfinite(maximum)):
            rms = maximum = None
    return {"finite": finite, "rms": rms, "maximum": maximum,
            "passed": finite and rms is not None and rms <= 1e-4 and maximum <= 1e-3}


def motion_metrics(prediction, truth, weights, omega_scale, s_coi):
    error = prediction - truth
    common_angle = error[:, :10] @ weights
    common_speed = error[:, 10:20] @ weights
    angle = error[:, :10] - common_angle[:, None]
    speed = error[:, 10:20] - common_speed[:, None]
    point = (.5*np.mean(angle*angle, axis=1)
             + .5*np.mean((speed/omega_scale)**2, axis=1) + (common_speed/s_coi)**2)
    return {"mean_motion_error": float(point.mean()), "points": len(point),
            "common_speed_sse": float(common_speed @ common_speed),
            "common_speed_rmse_rad_s": float(np.sqrt(np.mean(common_speed**2)))}


def port_metrics(prediction, truth, mask, scale):
    count = int(mask.sum())
    if not count:
        return {"count": 0, "sse": None, "normalized_sse": None,
                "rmse_system_pu_torque": None, "normalized_rmse": None}
    error = prediction - np.where(mask, truth, 0.)
    sse = float(np.sum(error[mask]**2))
    normalized_sse = float(np.sum((error/scale)[mask]**2))
    return {"count": count, "sse": sse, "normalized_sse": normalized_sse,
            "rmse_system_pu_torque": math.sqrt(sse/count),
            "normalized_rmse": math.sqrt(normalized_sse/count)}


def classification(event, main, fine, constants, steps):
    weights = np.asarray(constants["mass"])
    history = np.column_stack((event["history"]["theta"], event["history"]["omega"]))
    meta = event["metadata"]
    classes = []
    for prediction in (main, fine):
        raw = np.concatenate((history, prediction[1:steps+1, :20])).copy()
        raw[:, 10:20] = 1 + raw[:, 10:20] / constants["omega_s"]
        classes.append(classify_prediction(raw, weights,
                       true_event_time_s=meta["true_event_time_record_s"]))
    coarse, reference = classes
    same_los = coarse["predicted_los"] == reference["predicted_los"]
    trigger_gap = (abs(coarse["trigger_index"]-reference["trigger_index"])*.01
                   if coarse["predicted_los"] and reference["predicted_los"] else None)
    numerical = numerical_gate(main[:steps+1], fine[:steps+1], np.asarray(constants["omega_scale"]))
    passed = numerical["passed"] and same_los and (trigger_gap is None or trigger_gap <= .02+1e-12)
    selected = apply_numerical_gate(reference, passed)
    predicted = None if selected["trigger_index"] is None else selected["trigger_index"]*.01
    actual_end = .1 + float(event["t"][-1])
    true = meta["true_event_time_record_s"]
    comparable = (predicted is not None and true is not None
                  and max(predicted, true) <= min(actual_end, .1+steps*.01)+1e-12)
    truth_supported = ((predicted if predicted is not None else .1+steps*.01)
                       <= actual_end+1e-12)
    delay = meta.get("record_start_delay_s")
    return {"classification": selected, "numerical": numerical, "los_pair_pass": bool(passed),
            "coarse_classification": coarse, "fine_classification": reference,
            "settled_disagreement": coarse["window_settled"] != reference["window_settled"],
            "main_fine_trigger_gap_s": trigger_gap, "predicted_time_record_s": predicted,
            "predicted_time_since_clearing_s": None if predicted is None or delay is None else predicted+delay,
            "true_time_record_s": true, "timing_comparable": bool(comparable),
            "truth_supported": bool(truth_supported),
            "trigger_time_error_s": predicted-true if comparable else None}


def score_event(event, main, fine, fine_ports, constants):
    mass = np.asarray(constants["mass"])
    weights = mass / mass.sum()
    omega_scale = np.asarray(constants["omega_scale"])
    available = len(event["t"]) - 1
    scopes = {}
    for scope in ("1", "2", "5", "8", "actual"):
        requested = min(available, 1500) if scope == "actual" else int(scope)*100
        steps = min(available, requested)
        complete_truth = steps > 0
        covered_prediction = steps < len(main) and steps < len(fine)
        numerical = (numerical_gate(main[:steps+1], fine[:steps+1], omega_scale)
                     if covered_prediction else {"passed": False, "finite": False, "rms": None, "maximum": None})
        row = {"truth_available": bool(complete_truth), "prediction_available": bool(covered_prediction),
               "requested_points": requested, "full_horizon_truth": available >= requested,
               "points": steps, "numerical": numerical, "metrics": None}
        if complete_truth and covered_prediction and numerical["passed"]:
            target = event["targets"]
            truth = np.column_stack((target["theta"][1:steps+1], target["omega"][1:steps+1]))
            metrics = motion_metrics(fine[1:steps+1, :20], truth, weights,
                                     omega_scale, float(constants["s_coi"]))
            for key, scale in (("tau_m", "S_m"), ("tau_e", "Se")):
                metrics[key] = port_metrics(fine_ports[key][1:steps+1], target[key][1:steps+1],
                                            event["masks"][key][1:steps+1], np.asarray(constants[scale]))
            row["metrics"] = metrics
        scopes[scope] = row
    real_steps = min(available, 1500)
    return {"event_id": event["id"], "group": event["group"], "label": event["label"],
            "actual_future_points": available, "scopes": scopes,
            "classification_8s": classification(event, main, fine, constants, 800),
            "classification_actual": classification(event, main, fine, constants, real_steps)
                if 0 < real_steps < len(main) else None}


def aggregate_metrics(rows, scope):
    good = [row["scopes"][scope]["metrics"] for row in rows if row["scopes"][scope]["metrics"] is not None]
    values = None
    if good:
        values = {"motion_risk_v1": risk([x["mean_motion_error"] for x in good]),
                  "common_speed_rmse_rad_s": math.sqrt(sum(x["common_speed_sse"] for x in good)
                                                         / sum(x["points"] for x in good))}
        for key in ("tau_m", "tau_e"):
            valid = [x[key] for x in good if x[key]["count"]]
            count = sum(x["count"] for x in valid)
            values[key] = {"observed_components": count,
                           "rmse_system_pu_torque": math.sqrt(sum(x["sse"] for x in valid)/count) if count else None,
                           "normalized_rmse": math.sqrt(sum(x["normalized_sse"] for x in valid)/count) if count else None}
    return {"denominator": len(rows), "covered": len(good),
            "short_record_ids": [x["event_id"] for x in rows if not x["scopes"][scope]["full_horizon_truth"]],
            "unresolved_ids": [x["event_id"] for x in rows if x["scopes"][scope]["metrics"] is None],
            "official": values if len(good) == len(rows) and rows else None,
            "covered_only_diagnostic": values}


def summarize(rows, constants):
    output = {"metric": "motion_risk_v1; excludes mechanical term of legacy risk",
              "splits": {}, "classification": {}}
    for split, ids in (("train_all", {r["event_id"] for r in rows}),
                       ("train_fit", set(constants["fit_ids"])), ("train_check", set(constants["check_ids"]))):
        cohort = [r for r in rows if r["event_id"] in ids]
        stable = [r for r in cohort if r["label"] == "stable"]
        unstable = [r for r in cohort if r["label"] == "unstable"]
        output["splits"][split] = {**{f"all_{h}s": aggregate_metrics(cohort, str(h)) for h in (1, 2, 5, 8)},
                                   **{f"stable_{h}s": aggregate_metrics(stable, str(h)) for h in (1, 2, 5, 8)},
                                   "unstable_actual": aggregate_metrics(unstable, "actual")}
        output["classification"][split] = {}
        for scope in ("8s", "actual"):
            population = cohort if scope == "8s" else unstable
            counts = {key: [] for key in ("detected_unstable_ids", "missed_unstable_ids", "false_los_ids",
                                          "numerical_unresolved_ids", "latency_uncovered_ids",
                                          "unsettled_ids", "settled_ids", "uncovered_ids",
                                          "autonomous_los_ids", "autonomous_settled_ids", "autonomous_unsettled_ids")}
            for row in population:
                item = row[f"classification_{scope}"]
                if item is None:
                    counts["uncovered_ids"].append(row["event_id"])
                    continue
                c = item["classification"]
                if c["category"] in ("numerical_unresolved", "latency_uncovered"):
                    counts[c["category"]+"_ids"].append(row["event_id"])
                    continue
                autonomous = ("los" if c["predicted_los"] else
                              "settled" if c["window_settled"] else "unsettled")
                counts[f"autonomous_{autonomous}_ids"].append(row["event_id"])
                if not item["truth_supported"]:
                    counts["uncovered_ids"].append(row["event_id"])
                    continue
                if c["predicted_los"]:
                    key = "detected_unstable_ids" if row["label"] == "unstable" else "false_los_ids"
                    counts[key].append(row["event_id"])
                else:
                    if row["label"] == "unstable":
                        counts["missed_unstable_ids"].append(row["event_id"])
                    counts["settled_ids" if c["window_settled"] else "unsettled_ids"].append(row["event_id"])
            output["classification"][split][scope] = {"denominator": len(population),
                "stable_denominator": len(stable) if scope == "8s" else 0,
                "unstable_denominator": len(unstable), **counts}
    return output


@torch.no_grad()
def evaluate_fit297(model, events, constants, *, output_directory: Path):
    if len(events) != 297 or len({e["id"] for e in events}) != 297:
        raise ValueError("exact fit297 denominator required")
    if sum(e["label"] == "stable" for e in events) != 267:
        raise ValueError("original stability denominators changed")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if any((output_directory / name).exists() for name in ("predictions.npz", "rows.json", "summary.json")):
        raise FileExistsError("evaluation output already exists")
    start, cpu_start = perf_counter(), process_time()
    work = IntegrationWork()
    model.eval()
    rows, arrays = [], {}
    groups = {}
    for index, event in enumerate(events):
        steps = max(800, min(1500, len(event["t"])-1)) if event["label"] == "unstable" else 800
        groups.setdefault(steps, []).append(index)
    device = model.mass.device
    for steps, indices in groups.items():
        for begin in range(0, len(indices), 16):
            batch_indices = indices[begin:begin+16]
            batch = [events[index] for index in batch_indices]
            theta = torch.as_tensor(np.stack([e["history"]["theta"] for e in batch]), dtype=torch.float64, device=device)
            omega = torch.as_tensor(np.stack([e["history"]["omega"] for e in batch]), dtype=torch.float64, device=device)
            work.encoder_events += len(batch)
            initial = model.initialize(theta, omega)
            main = integrate(model, initial, steps, work=work)
            fine = integrate(model, initial, steps, internal_step=.0025, work=work)
            ports = model.ports(fine)
            main, fine = main.cpu().numpy(), fine.cpu().numpy()
            mechanical, electrical = ports.tau_m.cpu().numpy(), ports.tau_e.cpu().numpy()
            for local, index in enumerate(batch_indices):
                arrays[f"event_{index:04d}_main"] = main[local]
                arrays[f"event_{index:04d}_fine"] = fine[local]
                row = score_event(events[index], main[local], fine[local],
                                  {"tau_m": mechanical[local], "tau_e": electrical[local]}, constants)
                rows.append(row)
    by_id = {row["event_id"]: row for row in rows}
    rows = [by_id[e["id"]] for e in events]
    summary = summarize(rows, constants)
    summary["work"] = work.snapshot()
    summary["cost"] = {"wall_s": perf_counter()-start, "cpu_s": process_time()-cpu_start,
                       "scope": "initialization, both integrals and scoring; serialization excluded"}
    np.savez_compressed(output_directory / "predictions.npz", **arrays)
    for name, value in (("rows.json", rows), ("summary.json", summary)):
        (output_directory / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    return summary
