"""Bounded endpoint initialization under a frozen, known artificial ODE."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import least_squares

from gftd_observability_20261107 import ROOT, cases, functional, observations, rk4, sampled

OUTPUT = ROOT / "results/gftd_initialization_20261107"
CALL_LIMIT = 600
PROTOCOL_SHA256 = "8399a3e0bcea8114e1d62db3ce852b16d39716d9cc7c9ea68da201d8909c5ce4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def serial(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, default=serial, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_observation(observation):
    required = {"theta_history", "omega_history", "history_times", "p_end", "sigma"}
    if set(observation) != required:
        raise ValueError("only theta/omega history, times, endpoint p and known sigma are allowed")
    result = {key: np.asarray(value, dtype=np.float64).copy() for key, value in observation.items()}
    for key in ("theta_history", "omega_history"):
        if result[key].shape != (11, 3):
            raise ValueError("history must have shape (11,3)")
    if result["p_end"].shape != (3,):
        raise ValueError("endpoint p must have shape (3,)")
    if result["history_times"].shape != (11,) or not np.allclose(
            result["history_times"], np.linspace(-.1, 0, 11), atol=1e-12, rtol=0):
        raise ValueError("history times must be eleven points over the final 100 ms")
    if result["sigma"].shape != () or float(result["sigma"]) not in (0., 1e-6, 1e-4):
        raise ValueError("unknown noise scale")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("all observation values must be finite")
    return result


def endpoint(observation, u, amplitude_free):
    u = np.asarray(u, dtype=np.float64)
    if u.shape != ((6,) if amplitude_free else (3,)) or not np.isfinite(u).all() or np.any(np.abs(u) > 2):
        raise ValueError("invalid bounded endpoint variables")
    amp = .1 * u[:3] if amplitude_free else np.zeros(3)
    return np.concatenate((observation["theta_history"][-1], observation["omega_history"][-1],
                           observation["p_end"], amp, .1 * u[-3:]))


def backward_history(f, end, step):
    if step not in (.0025, .00125):
        raise ValueError("unregistered history step")
    state = np.asarray(end, dtype=np.float64).copy()
    states = [state.copy()]
    for _ in range(10):
        for _ in range(int(round(.01 / step))):
            state = rk4(f, state, -step)
        states.append(state.copy())
    return np.asarray(states[::-1])


def history_residual(f, predicted, observation):
    measured = np.column_stack((observation["theta_history"], observation["omega_history"], np.zeros((11, 9))))
    return observations(predicted, f.mass) - observations(measured, f.mass)


class CallBudgetReached(RuntimeError):
    pass


def bounded_jacobian(residual, u):
    """Fixed absolute scaled-coordinate step, with feasible second-order stencils."""
    u = np.asarray(u, dtype=np.float64)
    step = 2.5e-5
    columns = []
    center = None
    for index in range(u.size):
        shift = np.eye(u.size)[index] * step
        if u[index] - step >= -2 and u[index] + step <= 2:
            column = (residual(u+shift)-residual(u-shift))/(2*step)
        else:
            if center is None:
                center = residual(u)
            sign = 1 if u[index] - step < -2 else -1
            column = sign * (-3*center + 4*residual(u+sign*shift)-residual(u+2*sign*shift))/(2*step)
        columns.append(column)
    return np.column_stack(columns)


def infer_endpoint(observation, f, *, amplitude_free, initial, step=.0025):
    """Fit only endpoint a/r. The argument whitelist excludes truth and case IDs."""
    observation = validate_observation(observation)
    initial = np.asarray(initial, dtype=np.float64)
    endpoint(observation, initial, amplitude_free)
    scale = max(float(observation["sigma"]), 1e-8)
    calls, nonfinite = 0, 0
    best = None
    trace = []
    begin = time.perf_counter()

    def residual(u):
        nonlocal calls, nonfinite, best
        if calls >= CALL_LIMIT:
            raise CallBudgetReached("600 actual residual integrations exhausted")
        calls += 1
        try:
            state = endpoint(observation, u, amplitude_free)
            raw = history_residual(f, backward_history(f, state, step), observation)
            if not np.isfinite(raw).all():
                raise FloatingPointError("nonfinite historical residual")
        except (FloatingPointError, OverflowError):
            nonfinite += 1
            raise
        loss = float(raw @ raw / 2)
        if best is None or loss < best["observation_loss"]:
            best = dict(u=np.array(u).copy(), state=state.copy(), observation_loss=loss,
                        history_rms=float(np.sqrt(np.mean(raw**2))))
        return raw / scale

    def callback(intermediate_result):
        previous = trace[-1]["u"] if trace else initial
        trace.append(dict(u=np.array(intermediate_result.x).copy(),
                          step_norm=float(np.linalg.norm(intermediate_result.x-previous)),
                          optimizer_cost=float(intermediate_result.cost), nfev=int(intermediate_result.nfev)))

    result = None
    reason = None
    try:
        result = least_squares(residual, initial, method="trf", jac=lambda u: bounded_jacobian(residual, u), x_scale=1.,
                               bounds=(-2., 2.), ftol=1e-10, xtol=1e-10, gtol=1e-10,
                               max_nfev=80, callback=callback)
        status = "converged" if result.success else "budget_limited" if result.status == 0 else "failed"
        reason = str(result.message)
        raw = np.asarray(result.fun) * scale
        if result.success:
            best = dict(u=result.x.copy(), state=endpoint(observation, result.x, amplitude_free),
                        observation_loss=float(raw @ raw / 2), history_rms=float(np.sqrt(np.mean(raw**2))))
    except CallBudgetReached as exc:
        status, reason = "budget_limited", str(exc)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        status, reason = "failed", repr(exc)
    gradient = None if result is None else np.asarray(result.grad)
    kkt = None
    if gradient is not None:
        projected = gradient.copy()
        projected[(result.x <= -2 + 1e-7) & (gradient >= 0)] = 0.
        projected[(result.x >= 2 - 1e-7) & (gradient <= 0)] = 0.
        kkt = float(np.max(np.abs(projected)))
    selected_u = None if best is None else np.asarray(best["u"])
    if result is not None and selected_u is not None and not np.array_equal(selected_u, result.x):
        kkt = None
    return dict(status=status, reason=reason, best=best, actual_residual_calls=calls,
                scipy_nfev=None if result is None else int(result.nfev),
                scipy_njev=None if result is None else int(result.njev),
                scipy_optimality=None if result is None else float(result.optimality),
                optimizer_gradient=gradient, observation_gradient=None if gradient is None else gradient * scale**2,
                projected_gradient_kkt=kkt,
                observation_projected_gradient_kkt=None if kkt is None else kkt * scale**2,
                gradient_evaluated_at=None if result is None else result.x,
                final_accepted_step_norm=None if not trace else trace[-1]["step_norm"],
                optimizer_residual_divisor=scale,
                active_bounds=None if selected_u is None else np.flatnonzero(np.abs(selected_u) >= 2-1e-7),
                nonfinite_evaluations=nonfinite, step_trace=trace, initial=initial,
                history_step=step, amplitude_free=amplitude_free, wall_seconds=time.perf_counter()-begin)


def numerical_check(f, end):
    histories = [backward_history(f, end, step) for step in (.0025, .00125)]
    roundtrips = [float(np.max(np.abs(sampled(f, history[0], .1, step)[-1]-end))/.1)
                  for history, step in zip(histories, (.0025, .00125))]
    discrepancy = float(np.sqrt(np.mean((observations(histories[0], f.mass)-
                                          observations(histories[1], f.mass))**2)))
    futures = [sampled(f, end, 1., step) for step in (.01, .005)]
    difference = observations(futures[0], f.mass)-observations(futures[1], f.mass)
    future_rms, future_peak = float(np.sqrt(np.mean(difference**2))), float(np.max(np.abs(difference)))
    return dict(roundtrip_scaled_max=roundtrips, history_main_fine_rms=discrepancy,
                future_main_fine_rms=future_rms, future_main_fine_peak=future_peak,
                passed=bool(max(roundtrips) <= 1e-7 and discrepancy <= 1e-7
                            and future_rms <= 1e-5 and future_peak <= 1e-4))


def generate_packages(output):
    """Generate and persist truth in a separate phase; return only file paths."""
    f = functional(.5)
    observations_package, truth_package = {}, {}
    for case_index, initial in enumerate(cases()):
        fine = sampled(f, initial, .1, .00125)
        coarse = sampled(f, initial, .1, .0025)
        fine_future = sampled(f, fine[-1], 1., .005)
        coarse_future = sampled(f, coarse[-1], 1., .01)
        reference_history_rms = float(np.sqrt(np.mean((observations(fine, f.mass)-observations(coarse, f.mass))**2)))
        reference_future_difference = observations(fine_future, f.mass)-observations(coarse_future, f.mass)
        reference_future_rms = float(np.sqrt(np.mean(reference_future_difference**2)))
        reference_future_peak = float(np.max(np.abs(reference_future_difference)))
        endpoint_qualification = numerical_check(f, fine[-1])
        noise = np.random.Generator(np.random.PCG64(110700+case_index)).standard_normal((11, 6))
        for sigma in (0., 1e-6, 1e-4):
            key = f"input_{len(observations_package):03d}"
            noisy = fine[:, :6] + .1*sigma*noise
            observations_package[key] = dict(theta_history=noisy[:, :3], omega_history=noisy[:, 3:6],
                history_times=np.linspace(-.1, 0., 11), p_end=fine[-1, 6:9], sigma=sigma)
            truth_package[key] = dict(case=case_index, sigma=sigma, initial=initial, noise_standard_normal=noise,
                history=fine, coarse_history=coarse, future=fine_future, coarse_future=coarse_future,
                endpoint_qualification=endpoint_qualification,
                reference_numerical=dict(history_rms=reference_history_rms, future_rms=reference_future_rms,
                    future_peak=reference_future_peak, passed=reference_history_rms <= 1e-7
                    and reference_future_rms <= 1e-5 and reference_future_peak <= 1e-4))
    obs_path, truth_path = output / "observations.json", output / "isolated_truth.json"
    write_json(obs_path, observations_package)
    write_json(truth_path, truth_package)
    return obs_path, truth_path


def fit_observations(observations_package, output=None):
    f = functional(.5)
    records = {}
    def checkpoint(stage):
        if output is not None:
            temporary = Path(output) / "fits.partial.tmp.json"
            write_json(temporary, dict(stage=stage, records=records))
            temporary.replace(Path(output) / "fits.partial.json")
    for key, observation in observations_package.items():
        groups = {}
        for group, starts in (("simple", [np.zeros(3)]),
                              ("joint", [np.zeros(6), np.array([.5, -.5, .5, 0., 0., 0.])])):
            candidates = [infer_endpoint(observation, f, amplitude_free=group == "joint", initial=start)
                          for start in starts]
            usable = [i for i, task in enumerate(candidates) if task["status"] != "failed" and task["best"] is not None]
            chosen = min(usable, key=lambda i: candidates[i]["best"]["observation_loss"]) if usable else None
            check = None
            error = None
            if chosen is not None:
                try:
                    check = numerical_check(f, candidates[chosen]["best"]["state"])
                except (FloatingPointError, ValueError, OverflowError) as exc:
                    error = repr(exc)
            groups[group] = dict(candidates=candidates, selected_index=chosen, numerical=check,
                                 numerical_error=error, fine_refit=None)
        simple = groups["simple"]
        if simple["selected_index"] is not None:
            simple_best = simple["candidates"][simple["selected_index"]]["best"]
            # a_end=0 is feasible in the joint problem; this is evaluation, not optimization.
            guard_u = np.concatenate((np.zeros(3), simple_best["u"]))
            guard_state = endpoint(validate_observation(observation), guard_u, True)
            if not np.array_equal(guard_state, simple_best["state"]):
                raise AssertionError("joint feasible guard changed the simple endpoint")
            guard_loss = simple_best["observation_loss"]
            groups["joint"]["simple_feasible_observation_loss"] = guard_loss
            groups["joint"]["both_starts_worse_than_simple_feasible"] = all(
                task["best"] is None or task["best"]["observation_loss"] > guard_loss
                for task in groups["joint"]["candidates"])
        else:
            groups["joint"]["simple_feasible_observation_loss"] = None
            groups["joint"]["both_starts_worse_than_simple_feasible"] = None
        records[key] = groups
        checkpoint("main_fits")
        progress = [dict(group=group, status=task["status"], calls=task.get("actual_residual_calls"))
                    for group, data in groups.items() for task in data["candidates"]]
        print(json.dumps(dict(input_id=key, tasks=progress)), flush=True)
    requests = [(data["numerical"]["history_main_fine_rms"], key, group)
                for key, groups in records.items() for group, data in groups.items()
                if data["numerical"] is not None and data["numerical"]["history_main_fine_rms"] > 1e-7]
    requests.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, key, group in requests[:4]:
        data = records[key][group]
        best = data["candidates"][data["selected_index"]]["best"]
        task = infer_endpoint(observations_package[key], f, amplitude_free=group == "joint",
                              initial=best["u"], step=.00125)
        data["fine_refit"] = task
        if task["status"] != "failed" and task["best"] is not None:
            try:
                data["fine_numerical"] = numerical_check(f, task["best"]["state"])
            except (FloatingPointError, ValueError, OverflowError) as exc:
                data["fine_numerical_error"] = repr(exc)
        checkpoint("fine_refits")
    return dict(records=records, requested_fine_refits=len(requests), performed_fine_refits=min(4, len(requests)))


def prediction_metrics(f, estimate, truth):
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    difference = (observations(estimate, f.mass)-observations(truth, f.mass)).reshape(-1, 6)
    return dict(joint_rms=float(np.sqrt(np.mean(difference**2))),
                theta_rms=float(np.sqrt(np.mean(difference[:, :3]**2))),
                omega_rms=float(np.sqrt(np.mean(difference[:, 3:]**2))),
                common_omega_rms=float(np.sqrt(np.mean((difference[:, 3:] @ f.coi_weights)**2))),
                peak=float(np.max(np.abs(difference))),
                p_rms=float(np.sqrt(np.mean(((estimate[:, 6:9]-truth[:, 6:9])/.1)**2))))


def score_frozen_fits(fits_path, fits_sha256, observations_path, truth_path):
    if digest(fits_path) != fits_sha256:
        raise ValueError("fit artifact changed before truth scoring")
    fits = json.loads(Path(fits_path).read_text(encoding="utf-8"))
    observation_package = json.loads(Path(observations_path).read_text(encoding="utf-8"))
    truth_package = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    f = functional(.5)
    scores = []
    for key, groups in fits["records"].items():
        truth = truth_package[key]
        observation = validate_observation(observation_package[key])
        row = dict(input_id=key, case=truth["case"], sigma=truth["sigma"], groups={})
        for group, data in groups.items():
            result = dict(candidates=[], selected_index=data["selected_index"], numerical_unresolved=True)
            for task in data["candidates"] + ([data["fine_refit"]] if data["fine_refit"] else []):
                if task["best"] is None:
                    result["candidates"].append(None)
                    continue
                state = np.asarray(task["best"]["state"])
                try:
                    history = backward_history(f, state, task["history_step"])
                    future = [sampled(f, state, 1., step) for step in (.01, .005)]
                    metrics = [prediction_metrics(f, prediction, np.asarray(truth["future"])) for prediction in future]
                    result["candidates"].append(dict(
                        history_observed_rms=task["best"]["history_rms"],
                        clean_history=prediction_metrics(f, history, np.asarray(truth["history"])),
                        future_main=metrics[0], future_fine=metrics[1],
                        future_states_main=future[0], future_states_fine=future[1],
                        amplitude_scaled_error=(state[9:12]-np.asarray(truth["history"])[-1, 9:12])/.1,
                        memory_scaled_error=(state[12:15]-np.asarray(truth["history"])[-1, 12:15])/.1,
                        endpoint_p_error=state[6:9]-observation["p_end"]))
                except (FloatingPointError, ValueError, OverflowError) as exc:
                    result["candidates"].append(dict(error=repr(exc)))
            chosen = data["selected_index"]
            if chosen is not None and result["candidates"][chosen] and "error" not in result["candidates"][chosen]:
                candidate = result["candidates"][chosen]
                main_pass = candidate["future_main"]["joint_rms"] <= 1e-3 and candidate["future_main"]["peak"] <= 1e-2
                fine_pass = candidate["future_fine"]["joint_rms"] <= 1e-3 and candidate["future_fine"]["peak"] <= 1e-2
                valid = bool(data["numerical"] and data["numerical"]["passed"] and truth["endpoint_qualification"]["passed"])
                valid = valid and truth["reference_numerical"]["passed"]
                if data["fine_refit"] is not None:
                    refined = result["candidates"][-1]
                    fine_check = data.get("fine_numerical")
                    valid = bool(fine_check and fine_check["passed"]
                                 and truth["endpoint_qualification"]["passed"]
                                 and truth["reference_numerical"]["passed"]
                                 and refined and "error" not in refined)
                    if valid:
                        refined_main = refined["future_main"]["joint_rms"] <= 1e-3 and refined["future_main"]["peak"] <= 1e-2
                        refined_fine = refined["future_fine"]["joint_rms"] <= 1e-3 and refined["future_fine"]["peak"] <= 1e-2
                        valid = refined_main == main_pass == fine_pass == refined_fine
                result.update(numerical_unresolved=not(valid and main_pass == fine_pass), future_pass=main_pass)
            row["groups"][group] = result
            if group == "joint":
                result["simple_feasible_observation_loss"] = data["simple_feasible_observation_loss"]
                result["optimization_adequacy"] = (
                    "insufficient_against_simple_feasible" if data["both_starts_worse_than_simple_feasible"]
                    else "unknown")
                first, second = result["candidates"][:2]
                if first is not None and second is not None and "error" not in first and "error" not in second:
                    result["joint_start_endpoint_state_difference"] = (
                        np.asarray(data["candidates"][0]["best"]["state"])
                        - np.asarray(data["candidates"][1]["best"]["state"]))
                    state_difference = result["joint_start_endpoint_state_difference"]
                    result["joint_start_amplitude_scaled_difference"] = state_difference[9:12]/.1
                    result["joint_start_memory_scaled_difference"] = state_difference[12:15]/.1
                    result["joint_start_future_main_difference"] = prediction_metrics(
                        f, first["future_states_main"], second["future_states_main"])
                    result["joint_start_future_difference"] = prediction_metrics(
                        f, first["future_states_fine"], second["future_states_fine"])
        if any(value["numerical_unresolved"] for value in row["groups"].values()):
            row["classification"] = "numerically_unresolved"
        else:
            a, b = (row["groups"][name]["future_pass"] for name in ("simple", "joint"))
            row["classification"] = ("both_pass" if a and b else "joint_only" if b else "simple_only" if a else "both_fail")
        if truth["case"] == 0 and truth["sigma"] == 0.:
            row["equilibrium_negative_calibration"] = all(
                data["selected_index"] is not None
                and data["candidates"][data["selected_index"]] is not None
                and "error" not in data["candidates"][data["selected_index"]]
                and data["candidates"][data["selected_index"]]["future_fine"]["joint_rms"] <= 1e-8
                for data in row["groups"].values())
        scores.append(row)
    return dict(fits_sha256=fits_sha256, scores=scores)


def main():
    if digest(ROOT / "reports/gftd_initialization_20261107_protocol.md") != PROTOCOL_SHA256:
        raise ValueError("initialization protocol differs from the reviewed design")
    if np.__version__ != "2.5.2" or scipy.__version__ != "1.18.0":
        raise ValueError("the registered NumPy/SciPy versions are required")
    release = json.loads((OUTPUT / "execution_release.json").read_text(encoding="utf-8-sig"))
    if release.get("released") is not True:
        raise PermissionError("initialization experiment requires a frozen execution release")
    for relative, expected in release["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"frozen source changed: {relative}")
    with (OUTPUT / "started.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(action_id=release["action_id"], status="RUNNING"), stream)
    begin = time.perf_counter()
    receipt = dict(action_id=release["action_id"], status="FAILED", numpy=np.__version__, scipy=scipy.__version__)
    try:
        obs_path, truth_path = generate_packages(OUTPUT)
        observation_package = json.loads(obs_path.read_text(encoding="utf-8"))
        fits = fit_observations(observation_package, output=OUTPUT)
        fits_path = OUTPUT / "fits.json"
        write_json(fits_path, fits)
        fits_sha = digest(fits_path)
        write_json(OUTPUT / "fits_frozen.json", dict(fits_sha256=fits_sha, observations_sha256=digest(obs_path)))
        summary = score_frozen_fits(fits_path, fits_sha, obs_path, truth_path)
        write_json(OUTPUT / "summary.json", summary)
        tasks = [task for groups in fits["records"].values() for data in groups.values()
                 for task in data["candidates"] + ([data["fine_refit"]] if data["fine_refit"] else [])]
        receipt.update(status="COMPLETED", fits_sha256=fits_sha, summary_sha256=digest(OUTPUT / "summary.json"),
                       optimization_tasks=len(tasks), actual_residual_calls=sum(task["actual_residual_calls"] for task in tasks))
    except Exception as exc:
        receipt["error"] = repr(exc)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin, real_data_reads=False,
                       shared_parameter_fitting=False, source_unchanged=all(
                           digest(ROOT / p) == h for p, h in release["source_sha256"].items()))
        write_json(OUTPUT / "execution_receipt.json", receipt)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
