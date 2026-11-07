"""Frozen simple-only confirmation on independent artificial noise directions."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.stats import beta

from gftd_initialization_20261107 import (
    ROOT, backward_history, cases, digest, functional, history_residual, infer_endpoint,
    numerical_check, observations, prediction_metrics, sampled, validate_observation, write_json,
)

OUTPUT = ROOT / "results/gftd_noise_confirmation_20261107"
PROTOCOL_SHA256 = "64a36873e3704fa96fce5ad440d2b7857696805e5b3578ba1a6d21ec2beb6b0c"
FINAL_CLASSES = ("pass", "prediction_fail", "numerically_unresolved", "optimization_no_output")


def noise_direction(case, rep):
    if case not in range(4) or rep not in range(16):
        raise ValueError("unregistered noise direction")
    return np.random.Generator(np.random.PCG64(120000+100*case+rep)).standard_normal((11, 6))


def generate_packages(output):
    f = functional(.5)
    observed, truth, plan = {}, {}, []
    for case, initial in enumerate(cases()):
        history = sampled(f, initial, .1, .00125)
        coarse = sampled(f, initial, .1, .0025)
        future = sampled(f, history[-1], 1., .005)
        coarse_future = sampled(f, history[-1], 1., .01)
        hist_delta = observations(history, f.mass)-observations(coarse, f.mass)
        future_delta = observations(future, f.mass)-observations(coarse_future, f.mass)
        reference = dict(history_rms=float(np.sqrt(np.mean(hist_delta**2))),
                         future_rms=float(np.sqrt(np.mean(future_delta**2))),
                         future_peak=float(np.max(np.abs(future_delta))))
        reference["passed"] = (reference["history_rms"] <= 1e-7 and reference["future_rms"] <= 1e-5
                               and reference["future_peak"] <= 1e-4)
        qualification = numerical_check(f, history[-1])
        if case == 0 and (np.any(history) or np.any(future) or np.any(coarse) or np.any(coarse_future)):
            raise ValueError("zero-state generation calibration failed")
        directions = [(None, 0., np.zeros((11, 6)))]
        for rep in range(16):
            noise = noise_direction(case, rep)
            directions.extend((rep, sigma, noise) for sigma in (1e-6, 1e-4))
        for rep, sigma, noise in directions:
            key = f"input_{len(observed):03d}"
            measurement = history[:, :6] + .1*sigma*noise
            observed[key] = dict(theta_history=measurement[:, :3], omega_history=measurement[:, 3:6],
                                 p_end=history[-1, 6:9], history_times=np.linspace(-.1, 0., 11), sigma=sigma)
            plan.append(dict(input_id=key, case=case, rep=rep, sigma=sigma,
                             seed=None if rep is None else 120000+100*case+rep))
            truth[key] = dict(case=case, rep=rep, sigma=sigma, noise=noise,
                              history=history, future=future, initial=initial,
                              coarse_history=coarse, coarse_future=coarse_future,
                              reference_numerical=reference, endpoint_qualification=qualification)
    write_json(Path(output)/"observations.json", observed)
    write_json(Path(output)/"isolated_truth.json", truth)
    write_json(Path(output)/"input_plan.json", plan)
    return Path(output)/"observations.json", Path(output)/"isolated_truth.json", Path(output)/"input_plan.json"


def evaluate_numerical(f, task):
    if task["best"] is None:
        return None, None
    begin = time.perf_counter()
    try:
        result = numerical_check(f, np.asarray(task["best"]["state"]))
        result["integration_cost"] = dict(wall_seconds=time.perf_counter()-begin, rk4_steps=540)
        return result, None
    except (FloatingPointError, ValueError, OverflowError) as exc:
        return None, dict(error=repr(exc), wall_seconds=time.perf_counter()-begin, rk4_steps=None)


def refit_eligible(check):
    if check is None:
        return False
    roundtrip = np.asarray(check.get("roundtrip_scaled_max", []), dtype=np.float64)
    future = np.asarray([check.get("future_main_fine_rms", np.inf),
                         check.get("future_main_fine_peak", np.inf)], dtype=np.float64)
    history = check.get("history_main_fine_rms", np.inf)
    return bool(roundtrip.shape == (2,) and np.isfinite(roundtrip).all()
                and np.all(roundtrip <= 1e-7) and np.isfinite(future).all()
                and future[0] <= 1e-5 and future[1] <= 1e-4
                and np.isfinite(history) and history > 1e-7)


def fit_packages(observed, output=None):
    f = functional(.5)
    records = {key: dict(status="planned", main=None, numerical=None, fine=None, fine_numerical=None)
               for key in sorted(observed)}
    def checkpoint(stage):
        if output is not None:
            tmp = Path(output)/"fits.partial.tmp.json"
            write_json(tmp, dict(stage=stage, records=records))
            tmp.replace(Path(output)/"fits.partial.json")
    checkpoint("planned")
    for key in records:
        record = records[key]
        record["main"] = infer_endpoint(observed[key], f, amplitude_free=False, initial=np.zeros(3))
        record["numerical"], record["numerical_error"] = evaluate_numerical(f, record["main"])
        record["status"] = "completed"
        checkpoint("main_fits")
        print(json.dumps(dict(input_id=key, status=record["main"]["status"],
                              calls=record["main"]["actual_residual_calls"])), flush=True)
    requests = [(row["numerical"]["history_main_fine_rms"], key) for key, row in records.items()
                if row["main"]["status"] != "failed" and refit_eligible(row["numerical"])]
    requests.sort(key=lambda item: (-item[0], item[1]))
    for _, key in requests[:4]:
        row = records[key]
        row["fine"] = infer_endpoint(observed[key], f, amplitude_free=False,
                                     initial=np.asarray(row["main"]["best"]["u"]), step=.00125)
        row["fine_numerical"], row["fine_numerical_error"] = evaluate_numerical(f, row["fine"])
        checkpoint("fine_refits")
    return dict(records=records, requested_fine_refits=len(requests), performed_fine_refits=min(4, len(requests)))


def statistical_summary(rows):
    if any(row["classification"] not in FINAL_CLASSES+("planned", "not_run") for row in rows):
        raise ValueError("unknown result classification")
    layers = []
    for case in range(4):
        for sigma in (1e-6, 1e-4):
            layer = [row for row in rows if row["case"] == case and row["sigma"] == sigma]
            reps = [row["rep"] for row in layer]
            if len(reps) != len(set(reps)):
                raise ValueError("duplicate direction in a statistical layer")
            if any(rep not in range(16) for rep in reps):
                raise ValueError("unknown direction in a statistical layer")
            complete = set(reps) == set(range(16)) and all(row["classification"] in FINAL_CLASSES for row in layer)
            counts = {label: sum(row["classification"] == label for row in layer)
                      for label in FINAL_CLASSES + ("planned", "not_run")}
            k = counts["pass"]
            interval = None
            if complete:
                interval = [0. if k == 0 else float(beta.ppf(.025, k, 17-k)),
                            1. if k == 16 else float(beta.ppf(.975, k+1, 16-k))]
            layers.append(dict(case=case, sigma=sigma, denominator=16, represented_inputs=len(layer),
                               actual_results=sum(counts[label] for label in FINAL_CLASSES),
                               counts=counts, complete=complete, successes=k, cp95=interval,
                               simultaneous_failure_upper95=None))
    all_complete = all(layer["complete"] for layer in layers)
    if all_complete:
        for layer in layers:
            failures = 16-layer["successes"]
            layer["simultaneous_failure_upper95"] = (
                1. if failures == 16 else float(beta.ppf(1-.05/8, failures+1, 16-failures)))
    return dict(layers=layers, simultaneous_bound_available=all_complete,
                independent_unit="one new Gaussian direction within a fixed case; noise levels are paired",
                noiseless_rows=[row for row in rows if row["sigma"] == 0.])


def score_task(f, task, observation, truth):
    if task is None or task["best"] is None:
        return None
    state = np.asarray(task["best"]["state"])
    begin = time.perf_counter()
    try:
        history = backward_history(f, state, task["history_step"])
        main = sampled(f, state, 1., .01)
        fine = sampled(f, state, 1., .005)
        main_metrics = prediction_metrics(f, main, truth["future"])
        fine_metrics = prediction_metrics(f, fine, truth["future"])
        end_error = state[6:9]-observation["p_end"]
        return dict(history_observed_rms=task["best"]["history_rms"],
                    scoring_integration_cost=dict(wall_seconds=time.perf_counter()-begin,
                                                  rk4_steps=300+int(round(.1/task["history_step"]))),
                    history_clean=prediction_metrics(f, history, truth["history"]),
                    future_main=main_metrics, future_fine=fine_metrics,
                    future_main_pass=main_metrics["joint_rms"] <= 1e-3 and main_metrics["peak"] <= 1e-2,
                    future_fine_pass=fine_metrics["joint_rms"] <= 1e-3 and fine_metrics["peak"] <= 1e-2,
                    p_constraint_error=end_error,
                    amplitude_scaled_error=(state[9:12]-np.asarray(truth["history"])[-1, 9:12])/.1,
                    memory_scaled_error=(state[12:15]-np.asarray(truth["history"])[-1, 12:15])/.1,
                    endpoint_observation_scaled_noise=(state[:6]-np.asarray(truth["history"])[-1, :6])/.1,
                    future_start_scaled_rms=float(np.sqrt(np.mean((observations(main[:1], f.mass)-
                                                                    observations(np.asarray(truth["future"])[:1], f.mass))**2))))
    except (FloatingPointError, ValueError, OverflowError) as exc:
        return dict(error=repr(exc), scoring_integration_cost=dict(wall_seconds=time.perf_counter()-begin, rk4_steps=None))


def score_frozen(fits_path, fits_hash, obs_path, truth_path, plan_path, parent_summary_path):
    if digest(fits_path) != fits_hash:
        raise ValueError("fits hash changed before truth loading")
    fits = json.loads(Path(fits_path).read_text(encoding="utf-8"))
    if fits.get("observations_sha256") != digest(obs_path):
        raise ValueError("observations changed after fitting")
    observed = json.loads(Path(obs_path).read_text(encoding="utf-8"))
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    parent = json.loads(Path(parent_summary_path).read_text(encoding="utf-8"))
    previous = {row["case"]: row for row in parent["scores"] if row["sigma"] == 0.}
    f = functional(.5)
    rows = []
    for entry in plan:
        key = entry["input_id"]
        row = dict(entry, classification="not_run")
        record = fits["records"].get(key)
        if record is None or record["status"] != "completed":
            row["classification"] = "planned" if record and record["status"] == "planned" else "not_run"
            rows.append(row)
            continue
        observation = validate_observation(observed[key])
        t = truth[key]
        main = score_task(f, record["main"], observation, t)
        fine = score_task(f, record["fine"], observation, t)
        row.update(main=main, fine_refit=fine, optimization_status=record["main"]["status"],
                   numerical=record["numerical"], fine_numerical=record["fine_numerical"])
        if main is None:
            row["classification"] = "optimization_no_output"
        elif "error" in main:
            row["classification"] = "numerically_unresolved"
        else:
            qualified = (record["main"]["status"] != "failed"
                         and record["numerical"] is not None and record["numerical"]["passed"]
                         and t["endpoint_qualification"]["passed"] and t["reference_numerical"]["passed"]
                         and np.all(np.asarray(main["p_constraint_error"]) == 0))
            consistent = main["future_main_pass"] == main["future_fine_pass"]
            if record["fine"] is not None:
                qualified = (record["main"]["status"] != "failed" and record["fine"]["status"] != "failed"
                             and refit_eligible(record["numerical"])
                             and record["fine_numerical"] is not None
                             and record["fine_numerical"]["passed"] and fine is not None and "error" not in fine
                             and t["endpoint_qualification"]["passed"] and t["reference_numerical"]["passed"]
                             and np.all(np.asarray(main["p_constraint_error"]) == 0))
                if qualified:
                    qualified = np.all(np.asarray(fine["p_constraint_error"]) == 0)
                    consistent = consistent and fine["future_main_pass"] == fine["future_fine_pass"] == main["future_main_pass"]
            row["classification"] = ("numerically_unresolved" if not qualified or not consistent
                                      else "pass" if main["future_main_pass"] else "prediction_fail")
            if entry["sigma"] == 0.:
                old = previous[entry["case"]]
                old_group = old["groups"]["simple"]
                old_metric = old_group["candidates"][old_group["selected_index"]]["future_fine"]
                row["parent_noiseless_regression"] = dict(
                    future_metric_differences={name: main["future_fine"][name]-old_metric[name] for name in old_metric},
                    previous_future_pass=old_group["future_pass"], current_future_pass=main["future_fine_pass"])
            if entry["case"] == 0 and entry["sigma"] == 0.:
                row["equilibrium_negative_calibration"] = (
                    main["future_main"]["joint_rms"] <= 1e-8 and main["future_fine"]["joint_rms"] <= 1e-8)
        rows.append(row)
    error_summaries = []
    for sigma in (1e-6, 1e-4):
        layer = [row for row in rows if row["sigma"] == sigma]
        available = [row["main"]["future_fine"] for row in layer
                     if row.get("main") is not None and "error" not in row["main"]]
        error_summaries.append(dict(sigma=sigma, denominator=64, scored_outputs=len(available),
            numerically_qualified=sum(row["classification"] in ("pass", "prediction_fail") for row in layer),
            mean_joint_rms=None if not available else float(np.mean([m["joint_rms"] for m in available])),
            worst_joint_rms=None if not available else max(m["joint_rms"] for m in available),
            worst_peak=None if not available else max(m["peak"] for m in available)))
    return dict(fits_sha256=fits_hash, rows=rows, statistics=statistical_summary(rows), error_summaries=error_summaries)


def main():
    if digest(ROOT/"reports/gftd_noise_confirmation_20261107_protocol.md") != PROTOCOL_SHA256:
        raise ValueError("confirmation protocol changed")
    if np.__version__ != "2.5.2" or scipy.__version__ != "1.18.0":
        raise ValueError("registered numerical library versions are required")
    release = json.loads((OUTPUT/"execution_release.json").read_text(encoding="utf-8-sig"))
    if release.get("released") is not True:
        raise PermissionError("confirmation execution release is required")
    for path, expected in release["source_sha256"].items():
        if digest(ROOT/path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    with (OUTPUT/"started.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(action_id=release["action_id"], status="RUNNING"), stream)
    begin = time.perf_counter()
    receipt = dict(action_id=release["action_id"], status="FAILED", numpy=np.__version__, scipy=scipy.__version__)
    try:
        obs_path, truth_path, plan_path = generate_packages(OUTPUT)
        observations_hash = digest(obs_path)
        observed = json.loads(obs_path.read_text(encoding="utf-8"))
        fits = fit_packages(observed, output=OUTPUT)
        if digest(obs_path) != observations_hash:
            raise ValueError("observations changed during fitting")
        fits["observations_sha256"] = observations_hash
        fits_path = OUTPUT/"fits.json"
        write_json(fits_path, fits)
        fits_hash = digest(fits_path)
        write_json(OUTPUT/"fits_frozen.json", dict(fits_sha256=fits_hash, observations_sha256=digest(obs_path)))
        summary = score_frozen(fits_path, fits_hash, obs_path, truth_path, plan_path,
                               ROOT/"results/gftd_initialization_20261107/summary.json")
        write_json(OUTPUT/"summary.json", summary)
        tasks = [task for row in fits["records"].values() for task in (row["main"], row["fine"]) if task is not None]
        if len(tasks) > 136 or sum(task["actual_residual_calls"] for task in tasks) > 81600:
            raise AssertionError("registered optimization budget exceeded")
        calibration = [row.get("equilibrium_negative_calibration") for row in summary["rows"]
                       if row["case"] == 0 and row["sigma"] == 0.]
        if calibration != [True]:
            raise AssertionError("noiseless equilibrium calibration failed")
        receipt.update(status="COMPLETED", fits_sha256=fits_hash, summary_sha256=digest(OUTPUT/"summary.json"),
                       optimization_tasks=len(tasks), actual_residual_calls=sum(task["actual_residual_calls"] for task in tasks))
    except Exception as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin, real_data_reads=False, shared_parameter_fitting=False,
                       source_unchanged=all(digest(ROOT/p) == h for p,h in release["source_sha256"].items()))
        write_json(OUTPUT/"execution_receipt.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
