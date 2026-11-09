"""Prepare and execute the registered bounded-kernel initialization run."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys
import time
import uuid

THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
if __name__ in ("__main__", "__mp_main__"):
    for variable in THREADS:
        os.environ[variable] = "1"

import numpy as np
import scipy
from scipy.integrate import Radau

from kiki.models import GFTDKuramotoModel, gftd_discovery as core, gftd_fit297, gftd_fitting as fitting
from kiki.physics.kuramoto_functional import BOUNDED_MODULATION_KERNEL as KERNEL

ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_bounded_initialization_20261109"
RESULTS = Path("results") / RUN
ARTIFACTS = Path("artifacts") / RUN
MODEL = ARTIFACTS / "candidate_model.json"
INPUTS = Path("private_data/gftd_fit297_20261107/inputs.json")
PARENT = "gftd_initialization_precision_20261109"
PARENT_MODEL = Path("artifacts") / PARENT / "candidate/pilot/modulated_model.json"
PARENT_ROWS = Path("results") / PARENT / "candidate/full/rows.json"
INPUT_MANIFEST = Path("results/gftd_fit297_20261107/raw_artifacts_manifest.json")
FROZEN = {
    Path("campaigns") / RUN / "protocol.md": "019a2dc59c4c47d761baa6a3856b675fa5efafaad62fec5830c6286daad3353c",
    Path("designs") / f"{RUN}.md": "c6089e0a133b4d696801734ae6cc830c60db13cbfe125d3ecbf64ee829912a1b",
    PARENT_MODEL: "59f73171eae057ec87b6d5618dbbe224687721d1fc117cd73a50aafdb8c78dac",
    PARENT_ROWS: "67547a04914f141eaf54617d361d3e88427fae921de858d5fa6771d0784ba099",
    INPUT_MANIFEST: "adbf068cc14b9c361af531171a2e44f13cf7c22e93eea82f4b111c071d0859b1",
}
PILOT = (0, 8, 10, 27, 57, 72, 89, 99, 117, 124, 127, 129, 133, 138, 152, 181,
         204, 221, 236, 251, 265, 267, 268, 282, 296)
DENOMINATORS = {"all": 297, "stable": 267, "unstable": 30}
GRID = np.linspace(0., 1., 101)
RHS_LIMIT = 24000
FUTURE_STEPS = {.01: "main", .005: "fine"}
MODEL_IN_WORKER = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value, *, allow_nan=False):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=allow_nan) + "\n").encode()


def reference(path, data):
    return dict(path=path.as_posix(), sha256=sha(data), bytes=len(data))


def read_bound(path, digest, size=None):
    data = (ROOT / path).read_bytes()
    if sha(data) != digest or (size is not None and len(data) != size):
        raise ValueError(f"frozen bytes changed: {path}")
    return data


def read_ref(ref, path):
    if ref["path"] != path.as_posix():
        raise ValueError(f"unexpected evidence path: {path}")
    return read_bound(path, ref["sha256"], ref["bytes"])


def save(path, data, *, exclusive=False):
    dest = ROOT / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    pending = dest if exclusive else dest.with_name(f"{dest.name}.pending-{uuid.uuid4().hex}")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        # Windows readers can briefly prevent an atomic replacement.
        for attempt in range(30):
            try:
                os.replace(pending, dest)
                break
            except PermissionError:
                if attempt == 29:
                    raise
                time.sleep(.05)
    return reference(path, data)


def save_json(path, value, *, exclusive=False):
    return save(path, json_bytes(value), exclusive=exclusive)


def runtime():
    threads = {name: os.environ.get(name) for name in THREADS}
    if set(threads.values()) != {"1"}:
        raise ValueError("run with one BLAS thread per worker")
    return dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, threads=threads)


def source_sha256():
    paths = {Path(__file__).resolve().relative_to(ROOT).as_posix(),
             f"experiments/{RUN}/qualify.py",
             "tests/test_gftd_bounded_initialization_runner.py", "tests/test_gftd_bounded_kernel.py"}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and (name == "kiki" or name.startswith("kiki.")):
            paths.add(Path(path).resolve().relative_to(ROOT).as_posix())
    return {path: sha((ROOT / path).read_bytes()) for path in sorted(paths)}


def old_rows(data):
    cached = json.loads(data)
    modes = {mode: [row for row in cached if row["mode"] == mode] for mode in ("static", "modulated")}
    if len(cached) != 594 or any([row["index"] for row in rows] != list(range(297)) for rows in modes.values()):
        raise ValueError("old row denominator/order changed")
    rows, pilot = [], {8, 124, 138, 204, 268, 296}
    for static, exp in zip(modes["static"], modes["modulated"], strict=True):
        row = {key: exp[key] for key in ("index", "event_id", "label")}
        if any(static[key] != value for key, value in row.items()):
            raise ValueError("old static and exp event identities differ")
        for name, old in (("old_static", static), ("old_exp", exp)):
            candidate = old["candidate"]
            row[name] = dict(passed=candidate["passed"], status=candidate["status"],
                             history_mse=candidate["history_mse"]["candidate"])
        if not row["old_exp"]["passed"]:
            pilot.add(row["index"])
        rows.append(row)
    if (len({row["event_id"] for row in rows}) != 297
            or Counter(row["label"] for row in rows) != {k: DENOMINATORS[k] for k in ("stable", "unstable")}
            or sum(row["old_static"]["passed"] for row in rows) != 297
            or sum(row["old_exp"]["passed"] for row in rows) != 282):
        raise ValueError("old event identities or fixed-DP8 cache changed")
    for label in ("stable", "unstable"):
        successes = [row["index"] for row in rows if row["label"] == label and row["old_exp"]["passed"]]
        pilot.update(successes[j * len(successes) // 4] for j in range(4))
    if tuple(sorted(pilot)) != PILOT:
        raise ValueError("old metadata no longer produces the registered 25-packet pilot")
    return rows


def prepare():
    frozen = {path: read_bound(path, digest) for path, digest in FROZEN.items()}
    rows = old_rows(frozen[PARENT_ROWS])
    [asset] = [entry for entry in json.loads(frozen[INPUT_MANIFEST])["files"] if entry["path"] == INPUTS.as_posix()]
    asset = {key: asset[key] for key in ("path", "bytes", "sha256")}
    sources = source_sha256()
    claim = save_json(RESULTS / "prepare_claim.json", dict(run=RUN, pid=os.getpid(), source_sha256=sources),
                      exclusive=True)
    model = GFTDKuramotoModel.load(ROOT / PARENT_MODEL)
    old_outer = model.inference_config
    old_inner = old_outer["fit297_config"]
    if old_outer["format"] != fitting.FIXED_DP8_FORMAT or old_inner["training_count"] != 297:
        raise ValueError("source checkpoint lacks the registered DP8/fit297 config")
    bounded = replace(model.functional, modulation_kernel=KERNEL)
    lineage = sha(json_bytes(dict(parent_sha256=FROZEN[PARENT_MODEL], kernel=KERNEL, refit=False)))
    inner = gftd_fit297.make_config(bounded, old_inner["mode"], old_inner["fit_request"], 297, lineage)
    if {k: v for k, v in old_inner.items() if k not in {"model_sha256", "preparation_key"}} != {
            k: v for k, v in inner.items() if k not in {"model_sha256", "preparation_key"}}:
        raise ValueError("bounded conversion changed initializer constants")
    model.functional, model.gates = bounded, bounded.gates
    model.inference_config = fitting.make_fixed_dp8_config(bounded, inner)
    model.fit_state = model.last_initialization = None
    pending = ROOT / MODEL.with_name("candidate_model.preparing.json")
    model.save(pending)
    data = pending.read_bytes()
    payload, original = json.loads(data), json.loads(frozen[PARENT_MODEL])
    if payload["functional"].pop("modulation_kernel") != KERNEL or any(
            payload[key] != original[key] for key in ("functional", "gates", "encoder", "delta_star")):
        raise ValueError("bounded conversion changed shared parameters")
    model_ref = save(MODEL, data)
    pending.unlink()
    prepared = dict(run=RUN, kernel=KERNEL, claim=claim, refit=False, runtime=runtime(), source_sha256=sources,
                    frozen_sha256={path.as_posix(): digest for path, digest in FROZEN.items()},
                    model=model_ref, config_sha256=sha(json_bytes(model.inference_config)),
                    parent_preparation_key=old_inner["preparation_key"], input=asset, pilot=list(PILOT),
                    event_order_sha256=sha(json_bytes([row["event_id"] for row in rows])),
                    denominators=DENOMINATORS)
    return save_json(RESULTS / "prepare.json", prepared)


def load_packets(prepared, rows):
    raw = json.loads(read_ref(prepared["input"], INPUTS))
    receipt = raw["receipt"]
    ids = [row["event_id"] for row in rows]
    if (receipt["legal_ids"] != ids or receipt["ordered_all_ids"] != ids or receipt["invalid"]
            or any(receipt["denominators"][key] != value for key, value in DENOMINATORS.items())
            or sha(json_bytes(ids)) != prepared["event_order_sha256"]):
        raise ValueError("frozen inputs do not match the original events and denominator")
    result = []
    for index, original in zip(range(297), raw["packets"], strict=True):
        if set(original) != {"theta_history", "omega_history", "history_times", "p0", "sigma"}:
            raise ValueError(f"unexpected fields in registered packet {index}")
        packet = core.validate_observation(
            {("p_end" if key == "p0" else key): value for key, value in original.items()}, 10)
        if packet.pop("sigma") != 0.:
            raise ValueError(f"nonzero sigma in registered packet {index}")
        packet["p0"] = packet.pop("p_end")
        result.append((packet, sha(json_bytes(original))))
    return result


def stage_summary(stage):
    if stage is None:
        return None
    best = stage["best"]
    keys = ("status", "reason", "actual_residual_calls", "same_step_feasible_sse",
            "projected_gradient_kkt", "observation_projected_gradient_kkt", "active_bounds")
    return dict({key: stage[key] for key in keys}, history_mse=None if best is None else best["history_mse"],
                counts=stage["counts"], optimization_sufficiency="unknown")


class CaptureIdentityError(Exception):
    """A runner contract failure must escape the production optimizer's numerical handlers."""


def initialize(model, packet, record, arrays, diagnostics):
    """Collect the original gate arrays without another model evaluation."""
    original = fitting._integrate
    captured, live, gate_slots = {}, {}, {}

    def capture(f, endpoint, step, duration, counts, *, backward=False, dense=False, stepper=None):
        live[id(counts)] = counts
        capture_future = duration == 1. and not backward and not dense and step in FUTURE_STEPS
        if f is not model.functional or stepper is not fitting.dop853_step:
            raise CaptureIdentityError("unexpected functional or integration method")
        if capture_future:
            gate = gate_slots.setdefault(id(counts), len(gate_slots) + 1)
            key = f"dp8_gate{gate}_{FUTURE_STEPS[step]}"
            if gate > 2 or key in captured:
                raise CaptureIdentityError("unexpected extra DP8 gate integration")
        states = original(f, endpoint, step, duration, counts, backward=backward, dense=dense, stepper=stepper)
        if capture_future:
            endpoint_sha = sha(endpoint.tobytes())
            if (states.shape != (101, 50) or states.dtype != np.float64
                    or sha(states[0].tobytes()) != endpoint_sha):
                raise CaptureIdentityError("DP8 gate array did not preserve its endpoint")
            captured[key] = endpoint_sha
            arrays[key] = states.copy()
        return states

    fitting._integrate = capture
    state = None
    try:
        try:
            state = model.initialize(packet)
            diagnostics.update(model.last_initialization)
        except fitting.InitializationFailure as error:
            diagnostics.update(error.diagnostics)
    finally:
        fitting._integrate = original
        record["known_initialization_rhs_calls"] = sum(c["actual_rhs_calls"] for c in live.values())
        record["known_initialization_residual_calls"] = sum(c["actual_residual_calls"] for c in live.values())

    counts = diagnostics.get("counts")
    record["initialization_counts"] = counts
    if diagnostics.get("method") != "fixed_dp8" or counts is None:
        raise ValueError("initializer did not return the frozen DP8 diagnostics")
    stages = [diagnostics[stage] for stage in ("stage1", "stage2")]
    residuals = [0 if stage is None else stage["actual_residual_calls"] for stage in stages]
    if (not 0 <= residuals[0] <= 600 or not 0 <= residuals[1] <= 1200
            or sum(residuals) != counts["actual_residual_calls"]
            or counts["actual_rhs_calls"] != record["known_initialization_rhs_calls"]
            or not record["known_initialization_residual_calls"] <= counts["actual_residual_calls"]
            or not 12 * counts["fixed_dp8_steps_completed"] <= counts["actual_rhs_calls"]
            <= 12 * counts["fixed_dp8_steps_attempted"] <= 480 * sum(residuals) + 2 * 6480):
        raise ValueError("initializer work accounting differs from the registered budget")
    stage1_gates, final_gates = diagnostics["stage1_numerical"], diagnostics["final_numerical"]
    residual_rhs = sum(stage["counts"]["actual_rhs_calls"] for stage in stages if stage is not None)
    checks = (stage1_gates,) if diagnostics.get("final_check_reused_stage1") else (stage1_gates, final_gates)
    gate_rhs = sum(check["counts"]["actual_rhs_calls"] for check in checks if check is not None)
    if (residual_rhs > 480 * sum(residuals) or gate_rhs > 6480 * sum(check is not None for check in checks)
            or residual_rhs + gate_rhs != counts["actual_rhs_calls"]):
        raise ValueError("historical residual or six-gate RHS work differs from its allowance")
    record["initialization_rhs_by_operation"] = dict(historical_residual=residual_rhs, six_gates=gate_rhs)
    selected = diagnostics["returned_point"]
    comparison_stage = selected or next(
        (stage for stage in ("stage2", "stage1")
         if diagnostics[stage] is not None and diagnostics[stage]["best"] is not None), None)
    record.update(initialization_status=diagnostics["status"], six_gates_passed=state is not None,
                  selected_stage=selected, formal_point=None, stage1=stage_summary(stages[0]),
                  stage2=stage_summary(stages[1]), stage1_gates=stage1_gates, final_gates=final_gates,
                  comparison_stage=comparison_stage, no_improvement=diagnostics.get("no_improvement"),
                  final_check_reused_stage1=diagnostics.get("final_check_reused_stage1"),
                  dp8_capture_endpoints=captured)
    if state is None:
        if diagnostics["status"] not in ("blocked_by_stage1", "no_output", "numerically_unresolved"):
            raise ValueError("unexpected initializer failure status")
        return None, None
    if (state.dtype != np.float64 or state.shape != (50,) or not np.isfinite(state).all()
            or state[20:30].tobytes() != packet["p0"].tobytes() or diagnostics["status"] != "completed"
            or not final_gates["passed"] or final_gates["p_endpoint_error"] != 0.
            or selected not in ("stage1", "stage2")
            or np.asarray(diagnostics[selected]["best"]["state"], dtype=np.float64).tobytes() != state.tobytes()):
        raise ValueError("formal output violates its frozen point, p0 anchor or six gates")
    endpoint_sha = sha(state.tobytes())
    gate = 1 if diagnostics["final_check_reused_stage1"] else 2
    paths = {}
    for kind in FUTURE_STEPS.values():
        key = f"dp8_gate{gate}_{kind}"
        if captured.get(key) != endpoint_sha or not np.isfinite(arrays[key]).all():
            raise ValueError("no matching original DP8 path for the formal output")
        paths[kind] = arrays[key]
    arrays["formal_state"] = state
    record["formal_point"] = dict(stage=selected, state_sha256=endpoint_sha, p0_exact=True)
    return state, paths


class RhsLimit(RuntimeError):
    pass


def radau_reference(functional, state, record, *, rhs_limit=RHS_LIMIT):
    """Count callbacks across setup, numerical Jacobians, steps and dense output."""
    samples = np.full((len(GRID), state.size), np.nan)
    samples[0] = state
    calls = dict(solver_setup=0, solver_step=0, dense_output=0)
    record.update(status="running", actual_rhs_calls=0, rhs_calls_by_operation=calls, rhs_limit=rhs_limit,
                  rtol=1e-11, atol=1e-13, max_step=.005, output_rows=1, last_accepted_time=0.,
                  state_sha256=sha(state.tobytes()))
    operation, count, solver = "solver_setup", 1, None
    started, cpu = time.perf_counter(), time.process_time()

    def rhs(t, value):
        if record["actual_rhs_calls"] == rhs_limit:
            raise RhsLimit("registered Radau RHS limit reached")
        calls[operation] += 1
        record["actual_rhs_calls"] += 1
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return functional.rhs(value)

    try:
        solver = Radau(rhs, 0., state.copy(), 1., rtol=1e-11, atol=1e-13, max_step=.005)
        while solver.status == "running":
            operation = "solver_step"
            message = solver.step()
            if solver.status == "failed":
                raise FloatingPointError(message)
            if not np.isfinite(solver.y).all():
                raise FloatingPointError("nonfinite accepted Radau state")
            record["last_accepted_time"] = float(solver.t)
            if count < len(GRID) and GRID[count] <= solver.t:
                operation = "dense_output"
                dense = solver.dense_output()
                while count < len(GRID) and GRID[count] <= solver.t:
                    value = dense(GRID[count])
                    if value.shape != state.shape or not np.isfinite(value).all():
                        raise FloatingPointError("nonfinite or misaligned Radau dense output")
                    samples[count] = value
                    count += 1
        record["status"] = "completed" if count == len(GRID) and solver.t == 1. else "incomplete"
    except (RhsLimit, FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as error:
        record.update(status="rhs_limit" if isinstance(error, RhsLimit) else "solver_failure",
                      failure_operation=operation, failure_type=type(error).__name__)
    finally:
        record.update(output_rows=count, last_output_time=float(GRID[count - 1]),
                      wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu,
                      scipy_nfev=getattr(solver, "nfev", None), scipy_njev=getattr(solver, "njev", None),
                      scipy_nlu=getattr(solver, "nlu", None))
    return samples


def reference_comparison(mass, dp8, radau, completed):
    finite = (np.isfinite(dp8) & np.isfinite(radau)).all(axis=1)
    first_missing = np.flatnonzero(~finite)
    common = int(first_missing[0]) if first_missing.size else len(GRID)
    result = dict(confirmed=False, common_rows=common, common_through_s=None, observation_rms=None,
                  observation_peak=None, full_state_max_relative=None)
    if common:
        x, y = dp8[:common], radau[:common]
        with np.errstate(over="ignore", invalid="ignore"):
            delta = core.observations(x, mass) - core.observations(y, mass)
            rms, peak = float(np.sqrt(np.mean(delta**2))), float(np.max(np.abs(delta)))
            scale = 1. + np.maximum(np.abs(x), np.abs(y))
            difference = np.abs(x - y) / scale
            overflow = ~np.isfinite(difference)
            difference[overflow] = np.abs(x[overflow] / scale[overflow] - y[overflow] / scale[overflow])
        relative = float(np.max(difference))
        result.update(common_through_s=float(GRID[common - 1]),
                      observation_rms=rms if np.isfinite(rms) else None,
                      observation_peak=peak if np.isfinite(peak) else None,
                      full_state_max_relative=relative if np.isfinite(relative) else None,
                      confirmed=bool(completed and common == len(GRID) and rms <= 1e-6
                                     and peak <= 1e-5 and relative <= 1e-5))
    return result


def job_paths(phase, index):
    return RESULTS / phase / "jobs" / f"{index:03d}.json", ARTIFACTS / phase / f"{index:03d}"


def write_arrays(path, arrays):
    data = io.BytesIO()
    np.savez_compressed(data, **arrays)
    return save(path, data.getvalue())


def setup_worker(prepared):
    global MODEL_IN_WORKER
    if source_sha256() != prepared["source_sha256"] or runtime() != prepared["runtime"]:
        raise ValueError("worker source or runtime differs from the frozen preparation")
    read_ref(prepared["model"], MODEL)
    model = GFTDKuramotoModel.load(ROOT / MODEL)
    if (model.functional.modulation_kernel != KERNEL or model.fit_mode != "modulated"
            or model.fit_state is not None or model.inference_config["method"] != "fixed_dp8"
            or sha(json_bytes(model.inference_config)) != prepared["config_sha256"]):
        raise ValueError("worker loaded the wrong kernel, parameter lineage or initializer")
    MODEL_IN_WORKER = model


def run_job(row, packet, packet_sha256, action, prepared):
    path, private = job_paths(action["phase"], row["index"])
    meta = dict(action, **{k: row[k] for k in ("index", "event_id", "label")},
                input_sha256=packet_sha256, model_sha256=prepared["model"]["sha256"],
                config_sha256=prepared["config_sha256"])
    claim = save_json(path.parent.parent / "claims" / path.name, dict(meta, pid=os.getpid()), exclusive=True)
    start, cpu = time.perf_counter(), time.process_time()
    record = dict(meta, status="running", claim=claim, initialization_counts=None, reference=dict(status="not_run"),
                  diagnostics=None, initialization_arrays=None, reference_arrays=None)
    arrays, diagnostics = {}, {}

    def persist():
        record.update(worker_wall_seconds=time.perf_counter() - start, worker_cpu_seconds=time.process_time() - cpu)
        save_json(path, record)

    try:
        state, dp8 = initialize(MODEL_IN_WORKER, packet, record, arrays, diagnostics)
        persist()
        raw = gzip.compress(json_bytes(diagnostics, allow_nan=True), mtime=0)
        record["diagnostics"] = save(private / "diagnostics.json.gz", raw)
        if arrays:
            record["initialization_arrays"] = write_arrays(private / "initialization.npz", dict(arrays, future_times=GRID))
        if state is not None:
            record["reference"]["status"] = "pending"
            persist()
            samples = radau_reference(MODEL_IN_WORKER.functional, state, record["reference"])
            reference_record = record["reference"]
            reference_record.update({kind: reference_comparison(
                MODEL_IN_WORKER.functional.mass, dp8[kind], samples, reference_record["status"] == "completed")
                for kind in ("main", "fine")})
            persist()
            record["reference_arrays"] = write_arrays(private / "radau.npz", dict(times=GRID, states=samples))
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="interrupted", error_type=type(error).__name__)
        raise
    finally:
        try:
            persist()
        except OSError:
            print(json.dumps({k: record.get(k) for k in (
                "action_id", "index", "status", "initialization_counts", "known_initialization_rhs_calls",
                "known_initialization_residual_calls", "worker_wall_seconds", "worker_cpu_seconds")}
                | {"known_reference_rhs_calls": record["reference"].get("actual_rhs_calls")}), file=sys.stderr, flush=True)
            raise


def phase_jobs(action, rows, packets):
    completed, partial, unknown = {}, {}, []
    phase = action["phase"]
    for row in rows:
        index = row["index"]
        path, _ = job_paths(phase, index)
        claim_path = path.parent.parent / "claims" / path.name
        if not (ROOT / claim_path).exists():
            continue
        try:
            claim_raw = (ROOT / claim_path).read_bytes()
            claim = json.loads(claim_raw)
            raw = (ROOT / path).read_bytes()
            result = json.loads(raw)
            if (any(result[k] != v or claim[k] != v for k, v in action.items())
                    or result["index"] != index or result["event_id"] != row["event_id"]
                    or result["claim"] != reference(claim_path, claim_raw)
                    or result["input_sha256"] != packets[index][1]):
                raise ValueError(f"result/claim identity changed for packet {index}")
        except (OSError, ValueError, KeyError, TypeError):
            unknown.append(index)
            continue
        pair = (reference(path, raw), result)
        if result["status"] == "completed":
            completed[index] = pair
        else:
            partial[index] = pair
    return completed, partial, unknown


def eligible_pilot(release, bindings):
    pilot = json.loads(read_bound(RESULTS / "pilot/execution.json", release["pilot_execution_sha256"]))
    action = pilot["action"]
    if (action["phase"] != "pilot" or action["action_id"] == release["action_id"]
            or pilot["bindings"] != bindings or pilot["status"] != "completed"
            or not pilot["eligible_for_full"] or pilot["unknown_jobs"] or pilot["partial_jobs"]):
        raise ValueError("full phase requires the eligible pilot under the same frozen bindings")
    return pilot


def reuse_pilot(pilot, rows, packets):
    action = pilot["action"]
    selected = [rows[index] for index in PILOT]
    results, partial, unknown = phase_jobs(action, selected, packets)
    if (set(results) != set(PILOT) or partial or unknown
            or {str(index): ref for index, (ref, _) in results.items()} != pilot["jobs"]):
        raise ValueError("original pilot results changed or are incomplete")
    if not any(result["formal_point"] and result["formal_point"]["p0_exact"] for _, result in results.values()):
        raise ValueError("pilot has no formal p0-exact six-gate output")
    for index, (_, result) in results.items():
        _, private = job_paths("pilot", index)
        read_ref(result["diagnostics"], private / "diagnostics.json.gz")
        if result["initialization_arrays"] is not None:
            read_ref(result["initialization_arrays"], private / "initialization.npz")
        if result["formal_point"] is not None:
            read_ref(result["reference_arrays"], private / "radau.npz")
    return results


def row_table(old, jobs):
    table = []
    for row in old:
        job = jobs.get(row["index"])
        result = None if job is None else job[1]
        row = dict(row, evidence=None if job is None else job[0], formal_point=None,
                   six_gates_passed=None, fine_reference_confirmed=None, main_reference_confirmed=None,
                   history_mse=None, paired_history_mse_difference=None, tags=[])
        tags = row["tags"]
        if result is None:
            tags.append("not_run")
            table.append(row)
            continue
        row.update(six_gates_passed=result["six_gates_passed"], formal_point=result["formal_point"])
        if result["initialization_status"] == "blocked_by_stage1":
            tags.append("stage1_blocked")
        if result["initialization_status"] == "numerically_unresolved" and result["selected_stage"] == "stage2":
            tags.append("stage2_selected_unqualified")
        if result["initialization_status"] == "no_output":
            tags.append("stage2_no_output")
        for name in ("stage1", "stage2"):
            stage = result[name]
            if stage is not None:
                if stage["status"] == "budget_limited":
                    tags.append(f"{name}_optimization_limit")
                if stage["status"] == "failed":
                    tags.append(f"{name}_optimization_failed")
                if stage["counts"]["integrations_completed"] < stage["counts"]["integrations_attempted"]:
                    tags.append(f"{name}_historical_evaluation_failed")
        for name in ("stage1_gates", "final_gates"):
            gates = result[name]
            if gates is not None and "error" in gates:
                tags.append(f"{name}_evaluation_failed")
        comparison = result["comparison_stage"]
        if comparison is not None:
            candidate = result[comparison]["history_mse"]
            previous = row["old_exp"]["history_mse"]
            row.update(history_mse=candidate, history_mse_stage=comparison,
                       history_mse_at_formal_point=bool(result["formal_point"]),
                       paired_history_mse_difference=None if previous is None else candidate - previous)
        reference_record = result["reference"]
        if reference_record["status"] == "completed":
            for method in ("fine", "main"):
                if not reference_record[method]["confirmed"]:
                    tags.append(f"{method}_reference_inconsistent")
        else:
            tags.append(f"reference_{reference_record['status']}")
        for method in ("fine", "main"):
            row[f"{method}_reference_confirmed"] = reference_record.get(method, {}).get("confirmed")
        table.append(row)
    return table


def summarize(table):
    summary = {}
    for label, denominator in DENOMINATORS.items():
        rows = [row for row in table if label == "all" or row["label"] == label]
        done = [row for row in rows if row["six_gates_passed"] is not None]
        summary[label] = dict(denominator=denominator, completed=len(done),
            tags=dict(Counter(tag for row in rows for tag in row["tags"])),
            old_static_passed=sum(row["old_static"]["passed"] for row in rows),
            old_exp_passed=sum(row["old_exp"]["passed"] for row in rows),
            **{name: [row["event_id"] for row in rows if row[name] is True]
               for name in ("six_gates_passed", "fine_reference_confirmed", "main_reference_confirmed")},
            newly_passed=[row["event_id"] for row in done if row["six_gates_passed"] and not row["old_exp"]["passed"]],
            lost_pass=[row["event_id"] for row in done if not row["six_gates_passed"] and row["old_exp"]["passed"]],
            paired_history_ids=[row["event_id"] for row in done if row["paired_history_mse_difference"] is not None])
    return summary


def costs(completed, partial):
    total = dict(residual_calls=0, initialization_rhs_calls=0, reference_rhs_calls=0,
                 worker_wall_seconds=0., worker_cpu_seconds=0.)
    for _, result in (*completed.values(), *partial.values()):
        initialization = result["initialization_counts"]
        if initialization is None:
            total["residual_calls"] += result.get("known_initialization_residual_calls", 0)
            total["initialization_rhs_calls"] += result.get("known_initialization_rhs_calls", 0)
        else:
            total["residual_calls"] += initialization["actual_residual_calls"]
            total["initialization_rhs_calls"] += initialization["actual_rhs_calls"]
        total["reference_rhs_calls"] += result["reference"].get("actual_rhs_calls", 0)
        for key in ("worker_wall_seconds", "worker_cpu_seconds"):
            total[key] += result.get(key, 0.)
    return total


def check_release(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("release must be in the workspace")
    raw = path.read_bytes()
    release = json.loads(raw)
    phase = release.get("phase")
    if (release.get("released") is not True or phase not in ("pilot", "full")
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", release.get("action_id", ""))
            or type(release.get("workers")) is not int or release["workers"] < 1):
        raise PermissionError("root phase release with action ID and worker count required")
    bindings = {k: release[k] for k in ("prepare_sha256", "qualification_sha256", "design_adoption",
                                        "implementation_review")}
    for key in ("design_adoption", "implementation_review"):
        ref = bindings[key]
        asset = Path(ref["path"])
        if (asset.parent not in (RESULTS, Path("reports") / RUN)
                or not (ROOT / asset).resolve().is_relative_to(ROOT)):
            raise ValueError(f"release requires a bound local {key}")
        read_ref(ref, asset)
    prepared = json.loads(read_bound(RESULTS / "prepare.json", bindings["prepare_sha256"]))
    qualification = json.loads(read_bound(RESULTS / "qualification.json", bindings["qualification_sha256"]))
    if (qualification.get("passed") is not True or qualification["prepare_sha256"] != bindings["prepare_sha256"]
            or qualification["source_sha256"] != prepared["source_sha256"]
            or not 0 < qualification["counts"]["rhs"] <= 128
            or not 0 < qualification["counts"]["hamiltonian"] <= 2000):
        raise PermissionError("qualification failed or does not bind this preparation")
    if (prepared["run"] != RUN or prepared["kernel"] != KERNEL or prepared["refit"] is not False
            or prepared["source_sha256"] != source_sha256() or prepared["runtime"] != runtime()
            or prepared["frozen_sha256"] != {path.as_posix(): digest for path, digest in FROZEN.items()}):
        raise ValueError("execution differs from the frozen sources, assets or runtime")
    for asset, digest in FROZEN.items():
        read_bound(asset, digest)
    read_ref(prepared["model"], MODEL)
    action = dict(run=RUN, phase=phase, action_id=release["action_id"], release_sha256=sha(raw),
                  prepare_sha256=bindings["prepare_sha256"])
    return release, bindings, action, prepared


def run(path):
    release, bindings, action, prepared = check_release(path)
    phase, workers = action["phase"], release["workers"]
    pilot = eligible_pilot(release, bindings) if phase == "full" else None
    output = RESULTS / phase
    claim = save_json(output / "started.json", dict(action, pid=os.getpid()), exclusive=True)
    started, cpu = time.perf_counter(), time.process_time()
    rows = selected = packets = []
    reused = {}
    receipt = dict(action=action, claim=claim, bindings=bindings, status="running", workers=workers,
                   eligible_for_full=False, rows=None)
    try:
        rows = old_rows(read_bound(PARENT_ROWS, FROZEN[PARENT_ROWS]))
        packets = load_packets(prepared, rows)
        if pilot is not None:
            reused = reuse_pilot(pilot, rows, packets)
        selected = [row for row in rows if (row["index"] in PILOT) == (phase == "pilot")]
        receipt["planned_jobs"] = [row["index"] for row in selected]
        save_json(output / "execution.json", receipt)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=setup_worker, initargs=(prepared,)) as pool:
            remaining = iter(selected)

            def submit(row):
                return pool.submit(run_job, row, *packets[row["index"]], action, prepared)

            pending = {submit(row) for _, row in zip(range(workers), remaining)}
            try:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    pending.update(submit(row) for _, row in zip(done, remaining))
            finally:
                for future in pending:
                    future.cancel()
        completed, partial, unknown = phase_jobs(action, selected, packets)
        if len(completed) != len(selected) or partial or unknown or source_sha256() != prepared["source_sha256"]:
            raise ValueError("missing job receipts or execution source changed")
        table = row_table(rows, reused | completed)
        receipt.update(status="completed", rows=save_json(output / "rows.json", table), summary=summarize(table),
                       eligible_for_full=phase == "pilot" and any(
                           result["formal_point"] is not None and result["formal_point"]["p0_exact"]
                           for _, result in completed.values()))
    except BaseException as error:
        receipt.update(status="interrupted", error_type=type(error).__name__)
        raise
    finally:
        completed, partial, unknown = phase_jobs(action, selected, packets)
        known = costs(completed, partial)
        receipt.update(jobs={str(index): ref for index, (ref, _) in completed.items()},
                       partial_jobs={str(index): ref for index, (ref, _) in partial.items()}, unknown_jobs=unknown,
                       unclaimed_jobs=[row["index"] for row in selected
                                       if row["index"] not in completed and row["index"] not in partial
                                       and row["index"] not in unknown],
                       reused_pilot={str(index): ref for index, (ref, _) in reused.items()},
                       known_new_costs=known, actual_new_costs=None if partial or unknown else known,
                       host_wall_seconds=time.perf_counter() - started, host_cpu_seconds=time.process_time() - cpu)
        try:
            save_json(output / "execution.json", receipt)
        except OSError:
            print(json.dumps({key: receipt[key] for key in ("action", "status", "known_new_costs", "unknown_jobs")}),
                  file=sys.stderr, flush=True)
            raise
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare()))
    else:
        result = run(args.release)
        print(json.dumps({key: result[key] for key in ("action", "status", "eligible_for_full", "known_new_costs")}))
