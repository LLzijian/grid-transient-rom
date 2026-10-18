"""Train-only genuine short FOM acquisition for the 2026-10-19 protocol.

This separate fork stops ANDES integration early; it never truncates a full run.
The strict default ends at clearing+100ms. Historical observations start at the
first converged state after clearing, so that default can be ineligible for the
11-point encoder. An explicit alternate rule ends 100ms after the *observed*
first record. Neither rule supplies a stable label. No continuation state is
claimed; selected conditions must be rerun from the original equilibrium.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import multiprocessing
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

SOURCE_SIMULATOR_SHA256 = "239b574f7b17b26c2c315f5be6b7aebaf05e82e1c91c61b32d48135caf452ab8"
STOP_RULES = ("clearing_plus_100ms", "first_record_plus_100ms")


def build_short_config(baseline: Mapping[str, Any], *, step: float = .005,
                       stop_rule: str = "clearing_plus_100ms") -> dict[str, Any]:
    """Only horizon, registered dual step and explicit stop rule may change."""
    if baseline.get("postfault_horizon_s") != 15.0 or baseline.get("disable_pss") is not False:
        raise ValueError("expected original 15s restored-system config with PSS")
    if baseline.get("sample_interval_s") != .01 or step not in (.005, .0025):
        raise ValueError("expected original sampling and registered integration steps")
    if stop_rule not in STOP_RULES:
        raise ValueError("unknown short-history stop rule")
    config = copy.deepcopy(dict(baseline))
    config.update(postfault_horizon_s=.1, integration_step_s=step,
                  short_history_stop_rule=stop_rule)
    return config


def validate_train_condition(event: Mapping[str, Any],
                             allowed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Limit to documented train bus/line endpoints and registered parameter domain."""
    value = {key: event[key] for key in ("bus", "line", "location", "duration_s", "xf")}
    endpoint = (str(value["bus"]), str(value["line"]), float(value["location"]))
    legal = {(str(row["bus"]), str(row["line"]), float(row["location"])) for row in allowed}
    if endpoint not in legal:
        raise ValueError("condition is outside documented train endpoints")
    duration, xf = float(value["duration_s"]), float(value["xf"])
    if not (math.isfinite(duration) and .05 <= duration <= .6 and
            math.isfinite(xf) and .001 <= xf <= .01):
        raise ValueError("condition is outside registered duration/reactance domain")
    return value


def short_history_metadata(relative_times, start: float, clearing: float,
                           solver_end: float, config: Mapping[str, Any], label: str):
    times = np.asarray(relative_times, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError("invalid short-history time axis")
    if label not in ("unknown", "unstable"):
        raise ValueError("short simulation cannot establish a stable outcome")
    expected = clearing + .1 if config["short_history_stop_rule"] == STOP_RULES[0] else start + .1
    if solver_end > expected + 1e-8:
        raise ValueError("solver integrated beyond registered short endpoint")
    if label == "unknown" and abs(solver_end - expected) > 1e-8:
        raise ValueError("short solver did not reach registered endpoint")
    if (abs(times[0]) > 1e-10 or np.any(np.diff(times) <= 0)
            or not np.allclose(np.diff(times), .01, rtol=0, atol=1e-9)
            or start + times[-1] > solver_end + 1e-8):
        raise ValueError("invalid or extrapolated short-history samples")
    eligible = bool(label == "unknown" and len(times) >= 11 and times[10] >= .1 - 1e-8)
    return {"acquisition_role": "short_history_only", "label": label,
            "stop_rule": config["short_history_stop_rule"],
            "requested_solver_end_absolute_s": expected,
            "solver_end_absolute_s": solver_end,
            "record_start_absolute_s": start,
            "sample_absolute_times_s": (start + times).tolist(),
            "sample_since_clearing_s": (start + times - clearing).tolist(),
            "history_eligible": eligible,
            "ineligible_reason": None if eligible else "physical_stop_or_less_than_100ms_record_span",
            "forecast_origin_since_clearing_s": float(start + times[10] - clearing) if eligible else None,
            "full_horizon_outcome": "unknown" if label == "unknown" else "unstable",
            "continuation_supported": False,
            "source_simulator_sha256": SOURCE_SIMULATOR_SHA256,
            "config_exception": {"postfault_horizon_s": {"original": 15., "effective": .1},
                                 "integration_step_s": config["integration_step_s"]}}


def _converged_endpoint_target(requested: float, now: float) -> float:
    """Close only floating-point residue after a genuinely converged step.

    ANDES 2.0.0 calc_h otherwise attempts a ~1e-15s integration step and
    reports a convergence failure. Preserve the measured state/time; move only
    the solver's target to the already stored endpoint. The bound is far below
    the existing 1e-8s endpoint qualification and cannot hide an actual step.
    """
    remainder = float(requested) - float(now)
    roundoff = min(1e-8, 32 * np.finfo(float).eps * max(1., abs(requested)))
    return float(now) if 0 < remainder <= roundoff else float(requested)


def _simulate_short_impl(event, config, solver_workdir=None, diagnostics=None):
    """Short-run fork of the immutable generator, with no future integration."""
    from experiments.generate_private_dataset import (andes, physical_parameters, state_mapping,
                                          _solver_load_kwargs, check_arrays, los_index)
    capture_diagnostics = diagnostics is not None
    if diagnostics is None:
        diagnostics = {}
    if andes.__version__ != config["andes_version"]:
        raise ValueError("ANDES version differs from frozen config")
    load_kwargs, codegen_workdir = _solver_load_kwargs(solver_workdir)
    ss = andes.load(andes.get_case(config["case"]), **load_kwargs)
    for model in ss.models.values():
        if model.group == "TimedEvent" and model.n:
            model.u.v[:] = [0] * model.n
    if config["disable_pss"]:
        ss.IEEEST.u.v[:] = [0] * ss.IEEEST.n
    uid = ss.Line.idx.v.index(event["line"])
    endpoint = ss.Line.bus1.v[uid] if event["location"] == 0.0 else ss.Line.bus2.v[uid]
    if endpoint != event["bus"] or event["location"] not in (0.0, 1.0):
        raise ValueError("fault_endpoint_mismatch")
    clearing = config["fault_start_s"] + event["duration_s"]
    ss.add("Fault", {"idx": "private_fault", "bus": event["bus"],
                     "tf": config["fault_start_s"], "tc": clearing, "rf": 0., "xf": event["xf"]})
    if codegen_workdir is not None:
        if (codegen_workdir / "andes_pycode" / "__init__.py").is_file():
            ss.codegen._load_calls()
        if not ss.with_calls:
            ss.prepare(quick=True, nomp=True)
    ss.setup()
    if not ss.PFlow.run():
        raise ValueError("powerflow_failure")
    tds = ss.TDS
    tds.config.tstep = config["integration_step_s"]
    tds.config.tol = config["newton_tolerance"]
    tds.config.max_iter = config["newton_max_iterations"]
    tds.config.fixt = 1
    tds.config.shrinkt = 0
    tds.config.criteria = 0
    tds.config.no_tqdm = 1
    tds.config.tf = clearing + config["postfault_horizon_s"]
    tds.init()
    def _residual_info(value):
        values = np.asarray(value, dtype=float)
        return {"finite": bool(np.isfinite(values).all()),
                "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
                "l2": float(np.linalg.norm(values)) if values.size else 0.0,
                "size": int(values.size)}

    if capture_diagnostics:
        diagnostics.update({
        "initial_f_convention": "stored after TDS.test_init masks no_check_init rows; not an unmasked raw derivative",
        "initial_f": _residual_info(getattr(ss.dae, "f", np.array([]))),
        "initial_g": _residual_info(getattr(ss.dae, "g", np.array([]))),
        "initial_xy_finite": bool(np.isfinite(ss.dae.xy).all()),
        "attempted_steps": 0, "converged_steps": 0, "failed_steps": 0,
        "step_end_times_s": [], "step_sizes_s": [], "scheduled_event_times_s": {
            "fault_start": float(config["fault_start_s"]), "clearing": float(clearing)},
        })
    if not tds.test_ok or not np.isfinite(ss.dae.xy).all():
        raise ValueError("dynamic_initialization_failure")
    inertia = ss.GENROU.M.v.copy()  # ANDES setup converts M to system MVA base.
    equilibrium = {key: getattr(ss.GENROU, variable).v.copy()
                   for key, variable in (("delta", "delta"), ("omega", "omega"), ("Pm", "tm"), ("Pe", "Pe"))}
    equilibrium["full_state"] = ss.dae.x.copy()
    equilibrium["algebraic_state"] = ss.dae.y.copy()
    signature = physical_parameters(ss)
    mapping = state_mapping(ss)
    status = {"failed_step": False, "physical_stop": False, "post_checked": False,
              "pre_checked": False, "post_sample_start": None, "sample_delta": [], "sample_omega": [],
              "previous_time": None, "previous_delta": None, "previous_omega": None,
              "fault_applied": False}
    step = tds.itm_step
    original_do_switch = tds.do_switch

    def checked_step():
        target = float(ss.dae.t)
        if capture_diagnostics:
            diagnostics["attempted_steps"] += 1
        ok = step()
        if capture_diagnostics:
            diagnostics["step_end_times_s"].append(target)
            diagnostics["step_sizes_s"].append(float(getattr(tds, "h", config["integration_step_s"])))
            if config["fault_start_s"] - .02 <= target <= config["fault_start_s"] + 1e-8:
                diagnostics.setdefault("pre_fault_residuals", []).append({
                    "time_s": target, "f": _residual_info(getattr(ss.dae, "f", np.array([]))),
                    "g": _residual_info(getattr(ss.dae, "g", np.array([])))})
            if ok and min(abs(target-config["fault_start_s"]), abs(target-clearing)) <= 2*config["integration_step_s"]+1e-8:
                diagnostics.setdefault("event_neighborhood", []).append({
                    "time_s": target, "fault_active_before_switch": bool(np.any(ss.Fault.uf.v != 0)),
                    "stored_f": _residual_info(ss.dae.f), "stored_g": _residual_info(ss.dae.g)})
        if not ok:
            status["failed_step"] = True
            if capture_diagnostics:
                diagnostics["failed_steps"] += 1
            tds.busted = True
            tds.err_msg = "private dataset rejects any nonconverged integration step"
        else:
            if capture_diagnostics:
                diagnostics["converged_steps"] += 1
        return ok

    def physical_stop():
        now = float(ss.dae.t)
        if not np.isfinite(ss.dae.xy).all():
            status["failed_step"] = True
            tds.busted = True
            return True
        if config["fault_start_s"] + 1e-8 < now < clearing - 1e-8:
            if not np.any(ss.Fault.uf.v != 0):
                raise ValueError("fault_was_not_applied")
            status["fault_applied"] = True
        if config["fault_start_s"] - .02 <= now < config["fault_start_s"] - 1e-8:
            if (np.max(abs(ss.GENROU.omega.v - 1)) > config["pre_fault_speed_tolerance_pu"]
                    or np.max(abs(ss.GENROU.delta.v - equilibrium["delta"])) > config["pre_fault_angle_drift_rad"]):
                raise ValueError("pre_fault_not_steady")
            status["pre_checked"] = True
        if now <= clearing + 1e-8:
            return True
        if not status["post_checked"]:
            if physical_parameters(ss) != signature or np.any(ss.Fault.uf.v != 0):
                raise ValueError("postfault_system_not_restored")
            status["post_checked"] = True
            status["post_sample_start"] = now
            if config["short_history_stop_rule"] == "first_record_plus_100ms":
                # Observed first post-clearing converged state, not a guessed delay.
                tds.config.tf = now + config["postfault_horizon_s"]
        next_sample = status["post_sample_start"] + len(status["sample_delta"]) * config["sample_interval_s"]
        if now >= next_sample - 1e-8:
            previous = status["previous_time"]
            fraction = 1.0 if previous is None else (next_sample-previous)/(now-previous)
            if not -1e-8 <= fraction <= 1+1e-8:
                raise ValueError("LOS_sampling_would_extrapolate")
            for key in ("delta", "omega"):
                current = getattr(ss.GENROU, key).v
                prior = current if previous is None else status[f"previous_{key}"]
                status[f"sample_{key}"].append(prior + fraction*(current-prior))
            hold = config["los_hold_samples"]
            if los_index(status["sample_delta"][-hold:], status["sample_omega"][-hold:], inertia, config) is not None:
                status["physical_stop"] = True
                # Called after a converged, stored step. End normally at this
                # time; never relabel an ANDES numerical failure as instability.
                tds.config.tf = now
        status["previous_time"] = now
        status["previous_delta"] = ss.GENROU.delta.v.copy()
        status["previous_omega"] = ss.GENROU.omega.v.copy()
        target = float(tds.config.tf)
        closed_target = _converged_endpoint_target(target, now)
        if closed_target != target:
            tds.config.tf = closed_target
            diagnostics["endpoint_roundoff_closure"] = {
                "requested_s": target, "converged_s": now,
                "difference_s": target - now, "state_or_time_changed": False}
        return True

    def observed_do_switch():
        if not capture_diagnostics:
            return original_do_switch()
        before = np.asarray(getattr(ss.Fault.uf, "v", np.array([]))).copy()
        before_time = float(ss.dae.t)
        result = original_do_switch()
        after = np.asarray(getattr(ss.Fault.uf, "v", np.array([]))).copy()
        if not np.array_equal(before, after):
            diagnostics.setdefault("switch_callbacks", []).append({
                "time_s": before_time,
                "fault_active_before": bool(np.any(before != 0)),
                "fault_active_after": bool(np.any(after != 0)),
            })
        return result

    tds.itm_step = checked_step
    tds.check_criteria = physical_stop
    if capture_diagnostics:
        tds.do_switch = observed_do_switch
    success = tds.run(no_summary=True)
    endpoints = np.asarray(diagnostics["step_end_times_s"], dtype=float) if capture_diagnostics else np.array([])
    if capture_diagnostics:
      for name, target in diagnostics["scheduled_event_times_s"].items():
        if endpoints.size:
            before = endpoints[endpoints <= target + 1e-8]
            after = endpoints[endpoints >= target - 1e-8]
            diagnostics.setdefault("event_brackets", {})[name] = {
                "target_s": target,
                "present_within_1e-8_s": bool(np.any(np.abs(endpoints - target) <= 1e-8)),
                "before_s": None if before.size == 0 else float(before[-1]),
                "after_s": None if after.size == 0 else float(after[0]),
            }
        else:
            diagnostics.setdefault("event_brackets", {})[name] = {
                "target_s": target, "present_within_1e-8_s": False,
                "before_s": None, "after_s": None,
            }
    if capture_diagnostics and diagnostics["step_sizes_s"]:
        diagnostics["step_size_min_s"] = float(np.min(diagnostics["step_sizes_s"]))
        diagnostics["step_size_max_s"] = float(np.max(diagnostics["step_sizes_s"]))
    if capture_diagnostics:
        diagnostics["solver_final_time_s"] = float(ss.dae.t)
        diagnostics["solver_success"] = bool(success and not tds.busted and not status["failed_step"])
    if not success or tds.busted or status["failed_step"]:
        raise ValueError("numerical_TDS_failure")
    if (not status["pre_checked"] or not status["post_checked"] or not status["fault_applied"]
            or physical_parameters(ss) != signature):
        raise ValueError("missing_baseline_or_restoration_check")
    raw_time = np.asarray(ss.dae.ts.t)
    x, y = np.asarray(ss.dae.ts.x), np.asarray(ss.dae.ts.y)
    if np.any(np.diff(raw_time) <= 0) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid_raw_solver_output")
    start = status["post_sample_start"]
    dt = config["sample_interval_s"]
    count = min(int(np.floor((raw_time[-1]-start)/dt+1e-7))+1,
                round(config["postfault_horizon_s"]/dt)+1)
    grid = start + np.arange(count)*dt
    if grid[0] < raw_time[0] or grid[-1] > raw_time[-1]+1e-8:
        raise ValueError("resampling_would_extrapolate")

    def interpolate(array):
        return np.stack([np.interp(grid, raw_time, array[:, j]) for j in range(array.shape[1])], axis=1)

    full_state, algebraic = interpolate(x), interpolate(y)
    arrays = {"t": np.arange(count)*dt, "full_state": full_state, "algebraic_state": algebraic}
    for key, variable in (("delta", "delta"), ("omega", "omega"), ("Pm", "tm"),
                          ("Pe", "Pe"), ("Qe", "Qe"), ("te", "te")):
        var = getattr(ss.GENROU, variable)
        arrays[key] = full_state[:, var.a] if var.v_code == "x" else algebraic[:, var.a]
    arrays["bus_v"] = algebraic[:, ss.Bus.v.a]
    arrays["bus_a"] = algebraic[:, ss.Bus.a.a]
    arrays["terminal_v"] = np.hypot(algebraic[:, ss.GENROU.vd.a], algebraic[:, ss.GENROU.vq.a])
    check_arrays(arrays, full_state, config)
    trigger = los_index(arrays["delta"], arrays["omega"], inertia, config)
    if status["physical_stop"]:
        if trigger != len(grid)-1:
            raise ValueError("export_LOS_does_not_match_solver_stop")
        label, reason = "unstable", "physical_loss_of_synchronism"
    else:
        if trigger is not None:
            raise ValueError("short_history_LOS_inconsistent_with_solver_stop")
        label, reason = "unknown", "short_history_horizon_reached"
    metadata = {"fault_line": event["line"], "fault_location": event["location"], "fault_bus": event["bus"],
                "fault_type": config["fault_type"], "fault_duration_s": event["duration_s"], "fault_xf": event["xf"],
                "clearing_time_s": clearing, "record_start_delay_s": start-clearing, "label": label,
                "termination_reason": reason, "event_time_s": float(arrays["t"][-1]) if label == "unstable" else "",
                "trajectory_end_s": float(arrays["t"][-1]), "valid_length": len(grid),
                "postfault_signature": signature, "solver_success": True, "failed_steps": 0}
    if capture_diagnostics:
        record_time = float(arrays["t"][-1]) if status["physical_stop"] else None
        diagnostics.update({"label": label, "physical_stop_record_time_s": record_time,
                            "physical_stop_since_clearing_s": None if record_time is None else record_time + float(start-clearing),
                            "physical_stop_absolute_time_s": None if record_time is None else record_time + float(start),
                            "record_start_delay_s": float(start-clearing), "physical_stop": bool(status["physical_stop"])})
    metadata.update(short_history_metadata(arrays["t"], start, clearing,
                                           float(ss.dae.t), config, label))
    return arrays, metadata, equilibrium, inertia, mapping, ss.dae.y_name


def _short_worker(event, baseline, config, output, connection):
    from experiments import closure_discovery_data as data
    diagnostics = {}
    try:
        reference = data._load_reference_fingerprint()
        data._assert_config_compatible(baseline, reference)
        attestation = data._runtime_system_attestation(baseline, reference)
        source = Path(__file__).with_name("generate_private_dataset.py")
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SIMULATOR_SHA256:
            raise ValueError("short-history fork source simulator identity changed")
        started = time.perf_counter()
        result = _simulate_short_impl(event, config, diagnostics=diagnostics)
        data._assert_result_system_compatible(result, reference)
        attestation = dict(attestation,
                           short_effective_config_sha256=data._config_sha256(config),
                           config_exception=result[1]["config_exception"])
        raw_file = data._persist_raw(Path(output), "short_history", config["integration_step_s"],
                                     result, attestation)
        message = {"status": "complete", "raw_file": raw_file,
                   "metadata": result[1], "diagnostics": diagnostics,
                   "simulation_wall_seconds": time.perf_counter() - started}
    except BaseException as error:
        message = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                   "diagnostics": diagnostics}
    try:
        data._write_json(Path(output) / "worker_result.json", message)
        connection.send(message)
    finally:
        connection.close()


def run_short_history(event: Mapping[str, Any], output: str | Path, *, max_seconds: float,
                      step: float = .005, stop_rule: str = "clearing_plus_100ms"):
    """One bounded train-only attempt. Never retries or replaces failed conditions.

    ``max_seconds`` includes process startup and simulation. Process cleanup has
    an additional bounded 2s allowance and actual elapsed time is always saved.
    The separate raw format accepts unknown outcomes and must not be passed to
    the historical full-horizon qualification/label adapters.
    """
    from experiments import closure_discovery_data as data
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    event = validate_train_condition(event, data.allowed_train_conditions())
    baseline = data._safe_config()
    config = build_short_config(baseline, step=step, stop_rule=stop_rule)
    output = Path(output).resolve()
    local_root = (data.ROOT / "private_data").resolve()
    if output == local_root or not output.is_relative_to(local_root):
        raise ValueError("short-history output must be a new directory under private_data")
    output.mkdir(parents=True, exist_ok=False)
    data._write_json(output / "request.json", {"condition": event, "baseline_config": baseline,
                     "effective_config": config, "max_seconds": max_seconds,
                     "baseline_config_sha256": data._config_sha256(baseline),
                     "source_simulator_sha256": SOURCE_SIMULATOR_SHA256,
                     "scope": "train_short_history_only"})
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_short_worker,
                              args=(event, baseline, config, str(output), sender))
    started = time.perf_counter()
    message = {"status": "failed", "error": "worker not started"}
    try:
        process.start()
        sender.close()
        if receiver.poll(max(0., max_seconds - (time.perf_counter() - started))):
            try:
                message = receiver.recv()
            except EOFError:
                message = {"status": "failed", "error": "worker exited without result"}
        else:
            message = {"status": "timeout", "error": "short FOM exceeded process cap"}
    except BaseException as error:
        message = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        raise
    finally:
        if process.pid is not None:
            process.join(timeout=min(1., max(0., max_seconds - (time.perf_counter() - started))))
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.)
            message.update(worker_pid=process.pid, worker_exitcode=process.exitcode)
        sender.close()
        receiver.close()
        message["process_wall_seconds"] = time.perf_counter() - started
        data._write_json(output / "attempt.json", message)
    return message


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--step", type=float, choices=(.005, .0025), default=.005)
    parser.add_argument("--stop-rule", choices=STOP_RULES, default=STOP_RULES[0])
    args = parser.parse_args()
    result = run_short_history(json.loads(Path(args.condition_json).read_text(encoding="utf-8")),
                               args.output, max_seconds=args.max_seconds,
                               step=args.step, stop_rule=args.stop_rule)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
