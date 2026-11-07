"""Paired endpoint smoothing and continued hard-anchor optimization diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import least_squares

from gftd_initialization_20261107 import (
    ROOT, backward_history, digest, functional, history_residual, validate_observation, write_json,
)
from gftd_noise_confirmation_20261107 import evaluate_numerical, refit_eligible, score_task

OUTPUT = ROOT / "results/gftd_endpoint_denoising_20261107"
PARENT = ROOT / "results/gftd_noise_confirmation_20261107"
PROTOCOL_SHA256 = "f0fe91401d43bb84d6b7c52ca8a47ecac32a474a215a42be4157df9f46af41f7"
GROUPS = ("anchor", "smooth")
CALL_LIMIT = 1200


def angle_basis(mass):
    b = np.asarray(mass, dtype=np.float64)/np.sum(mass)
    projection = np.eye(3)-np.outer(np.ones(3), b)
    columns = []
    for index in range(2):
        value = projection[:, index].copy()
        for previous in columns:
            value -= previous * (previous @ value)
        value /= np.linalg.norm(value)
        first = np.flatnonzero(np.abs(value) > 1e-14)[0]
        if value[first] < 0:
            value = -value
        columns.append(value)
    return np.column_stack(columns)


def bounds(smooth):
    upper = np.r_[np.full(5, 5.), np.full(3, 2.)] if smooth else np.full(3, 2.)
    return -upper, upper


def endpoint(observation, f, u, smooth):
    low, high = bounds(smooth)
    u = np.asarray(u, dtype=np.float64)
    if u.shape != low.shape or not np.isfinite(u).all() or np.any(u < low) or np.any(u > high):
        raise ValueError("invalid bounded endpoint variables")
    theta = np.asarray(observation["theta_history"])[-1].copy()
    omega = np.asarray(observation["omega_history"])[-1].copy()
    if smooth:
        if observation["sigma"] <= 0:
            raise ValueError("noiseless inputs must reuse the parent")
        scale = .1 * float(observation["sigma"])
        theta += scale * (angle_basis(f.mass) @ u[:2])
        omega += scale * u[2:5]
    return f.pack_state(theta, omega, observation["p_end"], np.zeros(3), .1*u[-3:])


def jacobian(residual, u, smooth):
    low, high = bounds(smooth)
    h = 2.5e-5
    center = None
    columns = []
    for index in range(len(u)):
        shift = np.eye(len(u))[index]*h
        if u[index]-h >= low[index] and u[index]+h <= high[index]:
            column = (residual(u+shift)-residual(u-shift))/(2*h)
        else:
            if center is None:
                center = residual(u)
            sign = 1 if u[index]-h < low[index] else -1
            column = sign * (-3*center+4*residual(u+sign*shift)-residual(u+2*sign*shift))/(2*h)
        columns.append(column)
    return np.column_stack(columns)


class BudgetReached(RuntimeError):
    pass


def infer_endpoint(observation, f, parent_u, *, smooth, initial=None, step=.0025):
    """Local optimizer with fixed endpoint variables; no truth or case arguments."""
    observation = validate_observation(observation)
    if float(observation["sigma"]) == 0:
        raise ValueError("noiseless inputs must reuse the parent without optimization")
    parent_u = np.asarray(parent_u, dtype=np.float64)
    if parent_u.shape != (3,) or not np.isfinite(parent_u).all() or np.any(np.abs(parent_u) > 2):
        raise ValueError("parent r must be a finite legal three-vector")
    feasible = np.r_[np.zeros(5), parent_u] if smooth else parent_u.copy()
    initial = feasible.copy() if initial is None else np.asarray(initial, dtype=np.float64)
    endpoint(observation, f, initial, smooth)
    scale = max(float(observation["sigma"]), 1e-8)
    low, high = bounds(smooth)
    calls, nonfinite = 0, 0
    best = None
    last_loss = None
    baseline_loss, initial_loss = None, None
    last_jacobian = None
    trace = []
    begin = time.perf_counter()

    def residual(u):
        nonlocal calls, nonfinite, best, last_loss
        if calls >= CALL_LIMIT:
            raise BudgetReached("1200 actual residual integrations exhausted")
        calls += 1
        try:
            state = endpoint(observation, f, u, smooth)
            raw = history_residual(f, backward_history(f, state, step), observation)
            if not np.isfinite(raw).all():
                raise FloatingPointError("nonfinite historical residual")
        except (FloatingPointError, OverflowError):
            nonfinite += 1
            raise
        loss = float(raw @ raw/2)
        last_loss = loss
        if best is None or loss < best["observation_loss"]:
            best = dict(u=np.array(u).copy(), state=state, observation_loss=loss,
                        history_rms=float(np.sqrt(np.mean(raw**2))))
        return raw/scale

    def derivative(u):
        nonlocal last_jacobian
        matrix = jacobian(residual, u, smooth)
        last_jacobian = dict(u=np.array(u).copy(), optimizer_column_norms=np.linalg.norm(matrix, axis=0),
                             observation_column_norms=np.linalg.norm(matrix, axis=0)*scale)
        return matrix

    def callback(intermediate_result):
        previous = trace[-1]["u"] if trace else initial
        trace.append(dict(u=np.array(intermediate_result.x).copy(), cost=float(intermediate_result.cost),
                          nfev=int(intermediate_result.nfev), step_norm=float(np.linalg.norm(intermediate_result.x-previous))))

    result = None
    try:
        residual(feasible)
        baseline_loss = last_loss
        residual(initial)
        initial_loss = last_loss
        result = least_squares(residual, initial, jac=derivative, method="trf", bounds=(low, high),
                               x_scale=1., ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=80, callback=callback)
        status = "converged" if result.success else "budget_limited" if result.status == 0 else "failed"
        reason = str(result.message)
    except BudgetReached as exc:
        status, reason = "budget_limited", str(exc)
    except (FloatingPointError, ValueError, OverflowError) as exc:
        status, reason = "failed", repr(exc)
    gradient = None if result is None else np.asarray(result.grad)
    kkt = None
    if gradient is not None and best is not None and np.array_equal(result.x, best["u"]):
        projected = gradient.copy()
        projected[(result.x <= low+1e-7) & (gradient >= 0)] = 0
        projected[(result.x >= high-1e-7) & (gradient <= 0)] = 0
        kkt = float(np.max(np.abs(projected)))
    guard = best is not None and baseline_loss is not None and initial_loss is not None and (
        best["observation_loss"] <= min(baseline_loss, initial_loss))
    return dict(status=status, reason=reason, best=best, parent_same_step_loss=baseline_loss,
                initial_same_step_loss=initial_loss, objective_guard_passed=bool(guard),
                actual_residual_calls=calls, history_step=step, smooth=smooth, initial=initial, feasible_parent=feasible,
                optimizer_residual_divisor=scale, scipy_nfev=None if result is None else int(result.nfev),
                scipy_njev=None if result is None else int(result.njev),
                scipy_optimality=None if result is None else float(result.optimality),
                optimizer_gradient=gradient, observation_gradient=None if gradient is None else gradient*scale**2,
                gradient_evaluated_at=None if result is None else result.x,
                projected_gradient_kkt=kkt, observation_projected_gradient_kkt=None if kkt is None else kkt*scale**2,
                jacobian=last_jacobian, lower_bounds=low, upper_bounds=high,
                active_bounds=None if best is None else np.flatnonzero((best["u"] <= low+1e-7) | (best["u"] >= high-1e-7)),
                step_trace=trace, final_accepted_step_norm=None if not trace else trace[-1]["step_norm"],
                nonfinite_evaluations=nonfinite, wall_seconds=time.perf_counter()-begin)


def fit_packages(observed, parent_fits, output=None):
    f = functional(.5)
    records = {key: {group: dict(status="planned", main=None, fine=None, numerical=None, fine_numerical=None)
                     for group in GROUPS} for key in sorted(observed)}
    def checkpoint(stage):
        if output is not None:
            temporary = Path(output)/"fits.partial.tmp.json"
            write_json(temporary, dict(stage=stage, records=records))
            temporary.replace(Path(output)/"fits.partial.json")
    checkpoint("planned")
    for key, groups in records.items():
        obs = validate_observation(observed[key])
        parent_best = parent_fits["records"][key]["main"]["best"]
        for group, data in groups.items():
            if float(obs["sigma"]) == 0:
                data.update(status="reused_parent", parent_input_id=key)
                continue
            task = infer_endpoint(obs, f, parent_best["u"], smooth=group == "smooth")
            task["parent_frozen_main_loss"] = parent_best["observation_loss"]
            task["frozen_parent_objective_guard"] = task["best"] is not None and (
                task["best"]["observation_loss"] <= parent_best["observation_loss"])
            data.update(status="completed", main=task)
            data["numerical"], data["numerical_error"] = evaluate_numerical(f, task)
        checkpoint("main_fits")
        print(json.dumps(dict(input_id=key, groups={name: dict(status=data["status"],
            calls=None if data["main"] is None else data["main"]["actual_residual_calls"]) for name,data in groups.items()})), flush=True)
    requests = [(data["numerical"]["history_main_fine_rms"], key, group)
                for key, groups in records.items() for group, data in groups.items()
                if data["main"] is not None and data["main"]["status"] != "failed"
                and refit_eligible(data["numerical"])]
    requests.sort(key=lambda row: (-row[0], row[1], row[2]))
    for _, key, group in requests[:4]:
        data = records[key][group]
        parent_best = parent_fits["records"][key]["main"]["best"]
        task = infer_endpoint(observed[key], f, parent_best["u"], smooth=group == "smooth",
                              initial=data["main"]["best"]["u"], step=.00125)
        task["parent_frozen_main_loss"] = parent_best["observation_loss"]
        task["same_step_minus_frozen_parent_loss"] = (
            None if task["parent_same_step_loss"] is None else task["parent_same_step_loss"]-parent_best["observation_loss"])
        data["fine"] = task
        data["fine_numerical"], data["fine_numerical_error"] = evaluate_numerical(f, task)
        checkpoint("fine_refits")
    return dict(records=records, Q=angle_basis(f.mass), requested_fine_refits=len(requests),
                performed_fine_refits=min(4, len(requests)))


def classified_prediction(record, scored, truth):
    task = record["main"]
    if task is None or task["best"] is None:
        return "optimization_no_output"
    valid = (task["status"] != "failed" and task["objective_guard_passed"] and task["frozen_parent_objective_guard"]
             and truth["reference_numerical"]["passed"] and truth["endpoint_qualification"]["passed"]
             and scored["main"] is not None and "error" not in scored["main"])
    check = record["numerical"]
    valid = valid and check is not None and check["passed"]
    if record["fine"] is not None:
        fine = record["fine"]
        valid = (task["status"] != "failed" and task["objective_guard_passed"] and task["frozen_parent_objective_guard"]
                 and refit_eligible(check) and fine["status"] != "failed" and fine["objective_guard_passed"]
                 and record["fine_numerical"] is not None and record["fine_numerical"]["passed"]
                 and truth["reference_numerical"]["passed"] and truth["endpoint_qualification"]["passed"]
                 and scored["main"] is not None and "error" not in scored["main"]
                 and scored["fine"] is not None and "error" not in scored["fine"])
    if not valid:
        return "numerically_unresolved"
    outcomes = []
    for candidate in scored.values():
        if candidate is not None:
            if np.any(np.asarray(candidate["p_constraint_error"]) != 0):
                return "numerically_unresolved"
            outcomes.extend((candidate["future_main_pass"], candidate["future_fine_pass"]))
    if len(set(outcomes)) != 1:
        return "numerically_unresolved"
    return "pass" if outcomes[0] else "prediction_fail"


def adoption_conditions(rows, step, use_refit=False):
    def metrics(row, group):
        result = row[group]
        return (result.get("refit_metrics") or result["metrics"]) if use_refit else result["metrics"]
    def passed(row, group):
        result = row[group]
        values = metrics(row,group)
        if values is None:
            return False
        metric = values[step]
        return result["classification"] == "pass" and metric["joint_rms"] <= 1e-3 and metric["peak"] <= 1e-2
    complete = len(rows) == 132 and len({row["input_id"] for row in rows}) == 132
    target = next((row for row in rows if row["input_id"] == "input_072"), None)
    high = [row for row in rows if row["sigma"] == 1e-4]
    available = all(metrics(row,group) is not None for row in high for group in ("parent",)+GROUPS)
    means = {group: None for group in ("parent",)+GROUPS}
    if len(high) == 64 and available:
        means = {group: float(np.mean([metrics(row,group)[step]["joint_rms"] for row in high]))
                 for group in means}
    numerical = all(row[group]["classification"] in ("pass", "prediction_fail")
                    for row in rows for group in GROUPS)
    conditions = dict(complete=complete, target_repaired=target is not None and passed(target,"smooth"),
        no_parent_regressions=all(not passed(row,"parent") or passed(row,"smooth") for row in rows),
        no_anchor_regressions=all(not passed(row,"anchor") or passed(row,"smooth") for row in rows),
        high_noise_mean_not_above_parent=means["smooth"] is not None and means["smooth"] <= means["parent"],
        high_noise_mean_not_above_anchor=means["smooth"] is not None and means["smooth"] <= means["anchor"],
        all_numerically_valid=numerical)
    return dict(conditions=conditions, meets_all=all(conditions.values()), high_noise_means=means)


def score_frozen(fits_path, fits_hash, observed_path, parent_fits_path, parent_summary_path, truth_path, plan_path):
    if digest(fits_path) != fits_hash:
        raise ValueError("fits changed before truth scoring")
    fits = json.loads(Path(fits_path).read_text(encoding="utf-8"))
    for name,path in (("observations",observed_path),("parent_fits",parent_fits_path)):
        if digest(path) != fits["input_sha256"][name]:
            raise ValueError(f"{name} changed during fitting")
    observed = json.loads(Path(observed_path).read_text(encoding="utf-8"))
    parent = json.loads(Path(parent_summary_path).read_text(encoding="utf-8"))
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    parent_rows = {row["input_id"]: row for row in parent["rows"]}
    f = functional(.5)
    rows = []
    for entry in plan:
        key = entry["input_id"]
        old = parent_rows[key]
        row = dict(entry, parent=dict(classification=old["classification"],
            metrics=None if old["main"] is None or "error" in old["main"] else dict(
                future_main=old["main"]["future_main"], future_fine=old["main"]["future_fine"])))
        obs = validate_observation(observed[key])
        for group in GROUPS:
            record = fits["records"][key][group]
            if record["status"] == "reused_parent":
                row[group] = dict(classification=old["classification"],metrics=row["parent"]["metrics"],
                                  reused_parent_score=old, reused_parent_exact=True)
                continue
            if record["status"] != "completed":
                row[group] = dict(classification=record["status"],metrics=None)
                continue
            scored = {name: score_task(f, record[name], obs, truth[key]) for name in ("main","fine")}
            for candidate in scored.values():
                if candidate is not None and "error" not in candidate:
                    candidate["endpoint_estimation_scaled_error"] = candidate.pop("endpoint_observation_scaled_noise")
            classification = classified_prediction(record, scored, truth[key])
            metrics = None
            if scored["main"] is not None and "error" not in scored["main"]:
                metrics = {name: scored["main"][name] for name in ("future_main","future_fine")}
            row[group] = dict(classification=classification,metrics=metrics,scored=scored,
                              optimization_status=record["main"]["status"])
            row[group]["refit_metrics"] = None
            if scored["fine"] is not None and "error" not in scored["fine"]:
                row[group]["refit_metrics"] = {name:scored["fine"][name] for name in ("future_main","future_fine")}
            if record["main"]["best"] is not None:
                state = np.asarray(record["main"]["best"]["state"])
                row[group]["endpoint_correction"] = (state[:6]-np.r_[obs["theta_history"][-1],obs["omega_history"][-1]])
                row[group]["common_angle_correction"] = float(f.coi_weights @ row[group]["endpoint_correction"][:3])
        rows.append(row)
    main = adoption_conditions(rows,"future_main")
    fine = adoption_conditions(rows,"future_fine")
    refined_main = adoption_conditions(rows,"future_main",use_refit=True)
    refined_fine = adoption_conditions(rows,"future_fine",use_refit=True)
    consistent = all(result["conditions"] == main["conditions"] for result in (fine,refined_main,refined_fine))
    comparisons = {}
    for baseline in ("parent","anchor"):
        records = []
        for row in rows:
            left, right = row["smooth"]["metrics"],row[baseline]["metrics"]
            delta = None if left is None or right is None else left["future_fine"]["joint_rms"]-right["future_fine"]["joint_rms"]
            label = "unavailable" if delta is None else "same" if abs(delta) <= 1e-10 else "improved" if delta < 0 else "worsened"
            records.append(dict(input_id=row["input_id"],difference=delta,classification=label))
        comparisons[baseline] = dict(records=records,counts={label:sum(r["classification"]==label for r in records)
            for label in ("same","improved","worsened","unavailable")})
    negative = [row for row in rows if row["case"] == 0 and row["sigma"] == 0.]
    calibration = len(negative)==1 and all(negative[0][group]["metrics"][step]["joint_rms"] <= 1e-8
        for group in GROUPS for step in ("future_main","future_fine"))
    worst = {}
    for group in ("parent",)+GROUPS:
        available = [row[group]["metrics"] for row in rows if row[group]["metrics"] is not None]
        worst[group] = dict(scored_outputs=len(available),denominator=132,
            main_joint_rms=None if not available else max(value["future_main"]["joint_rms"] for value in available),
            fine_joint_rms=None if not available else max(value["future_fine"]["joint_rms"] for value in available),
            main_peak=None if not available else max(value["future_main"]["peak"] for value in available),
            fine_peak=None if not available else max(value["future_fine"]["peak"] for value in available))
    return dict(rows=rows,adoption_main=main,adoption_fine=fine,comparisons=comparisons,
                adoption_refined_main=refined_main,adoption_refined_fine=refined_fine,worst_errors=worst,
                adoption_consistent=consistent,
                worth_independent_confirmation=all(value["meets_all"] for value in (main,fine,refined_main,refined_fine))
                    and consistent and calibration,
                noiseless_equilibrium_calibration=calibration)


def main():
    if digest(ROOT/"reports/gftd_endpoint_denoising_20261107_protocol.md") != PROTOCOL_SHA256:
        raise ValueError("endpoint protocol changed")
    if np.__version__ != "2.5.2" or scipy.__version__ != "1.18.0":
        raise ValueError("registered numerical library versions required")
    release = json.loads((OUTPUT/"execution_release.json").read_text(encoding="utf-8-sig"))
    if release.get("released") is not True:
        raise PermissionError("root execution release required")
    for path,expected in release["source_sha256"].items():
        if digest(ROOT/path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    with (OUTPUT/"started.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(action_id=release["action_id"],status="RUNNING"),stream)
    begin = time.perf_counter()
    receipt = dict(action_id=release["action_id"],status="FAILED",numpy=np.__version__,scipy=scipy.__version__)
    try:
        hashes = {name:digest(PARENT/filename) for name,filename in (
            ("observations","observations.json"),("parent_fits","fits.json"))}
        observed = json.loads((PARENT/"observations.json").read_text(encoding="utf-8"))
        parent_fits = json.loads((PARENT/"fits.json").read_text(encoding="utf-8"))
        if len(observed) != 132 or sum(float(obs["sigma"])>0 for obs in observed.values()) != 128:
            raise ValueError("the frozen 132-input scope changed")
        fits = fit_packages(observed,parent_fits,OUTPUT)
        fits["input_sha256"] = hashes
        fits["source_sha256"] = release["source_sha256"]
        fits["parent_reference_paths"] = {name:str(PARENT/name) for name in (
            "observations.json","fits.json","summary.json","isolated_truth.json","input_plan.json")}
        fits_path = OUTPUT/"fits.json"
        write_json(fits_path,fits)
        fits_hash = digest(fits_path)
        write_json(OUTPUT/"fits_frozen.json",dict(fits_sha256=fits_hash,input_sha256=hashes))
        result = score_frozen(fits_path,fits_hash,PARENT/"observations.json",PARENT/"fits.json",
                              PARENT/"summary.json",PARENT/"isolated_truth.json",PARENT/"input_plan.json")
        write_json(OUTPUT/"summary.json",result)
        tasks = [task for groups in fits["records"].values() for data in groups.values()
                 for task in (data["main"],data["fine"]) if task is not None]
        if len(tasks)>260 or sum(task["actual_residual_calls"] for task in tasks)>312000:
            raise AssertionError("finite optimization budget exceeded")
        if not result["noiseless_equilibrium_calibration"]:
            raise AssertionError("reused noiseless calibration failed")
        receipt.update(status="COMPLETED",fits_sha256=fits_hash,summary_sha256=digest(OUTPUT/"summary.json"),
                       optimization_tasks=len(tasks),actual_residual_calls=sum(t["actual_residual_calls"] for t in tasks))
    except Exception as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin,real_data_reads=False,new_noise_directions=0,
                       source_unchanged=all(digest(ROOT/p)==h for p,h in release["source_sha256"].items()))
        write_json(OUTPUT/"execution_receipt.json",receipt)
        print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    main()
