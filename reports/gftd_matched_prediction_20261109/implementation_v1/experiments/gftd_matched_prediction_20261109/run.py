"""Run the frozen matched fit, deployment and train297 scoring stages."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import hashlib
from importlib.metadata import version
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
from scipy.integrate import Radau

from kiki.evaluation import gftd_analysis as analysis
from kiki.models import GFTDKuramotoModel, RunContext, gftd_discovery as core, gftd_fitting as fitting
from kiki.models import gftd_matched_prediction as matched
from kiki.numerics import dop853_step

ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_matched_prediction_20261109"
CURRENT = "gftd_bounded_initialization_20261109"
RESULTS, ARTIFACTS = Path("results") / RUN, Path("artifacts") / RUN
CURRENT_RESULTS = Path("results") / CURRENT
CURRENT_PREPARE_SHA = "22dce2d0c342a63fa2c79819cffa1c66f870f37eb9cca76e54a8ebe90f703447"
PROTOCOL_SHA = "0816203b8bcf37f506f0bbbb8e0fac531a747707c2d0f3612340b9ff69efbb43"
FIT_FORMAT = "gftd_matched_prediction_v1"
INPUT = Path("private_data/gftd_fit297_20261107/inputs.json")
INPUT_SHA = "b2607fc66c9bb96c5d6f85a6e074598e186ffab377c8aac3009699e1ea035817"
EVENT_ORDER_SHA = "9a76d08d0423302a2b07374e9f2a92646db3be089cadeebaadace5b985699de4"
OLD_ROWS = Path("results/gftd_initialization_precision_20261109/candidate/full/rows.json")
PARENTS = {
    "static": (Path("artifacts/gftd_initialization_precision_20261109/candidate/pilot/static_model.json"),
               "ed79e2cdea1f6830c052b9dae8fb8c2b748170e8e15ec0a5a8e3034d72b787cc", "exp_v1"),
    "modulated": (Path("artifacts/gftd_bounded_initialization_20261109/candidate_model.json"),
                  "f8eabacfeac00074f135c7af826d8d89d8f23c445c788485d0fb79a0f54b0637", "bounded_exp_tanh_v1"),
}
CONSTANTS = (Path("private_data/gftd_fit297_20261107/constants.json"),
             "030c09f3feae91a3f8c73106027429ad902a1cc190f2d665186eaec2edb24f49")
FOM = (Path("private_data/gftd_fit297_20261107/resume01/fom_modes.json"),
       "dfc799649ee8a28adac80daeb61bdd0e50cb79a60e25584c52a8d250b4db76b9")
KRON = (Path("private_data/gftd_fit297_20261107/resume01/kron_reference.json"),
        "f26831fd40eb3ff442ec4bdde773351dead41c477e875f96e6c5be9086161956")
FUTURE = (Path("private_data/r_memory_20261025/run01_repair01/fit_events.npz"),
          "3058c06fe23479a2c494203bd379ca495c54953b6d8b993a6591c26905cc4252")
FROZEN = {
    Path("campaigns") / RUN / "protocol.md": PROTOCOL_SHA,
    Path("designs") / f"{RUN}.md": "6bbb63daa3f02ed155f88e82930b3434ac1b48e4bf823606f53d50768326fbd3",
    Path("reports") / RUN / "design_review.md": "8322c6791a192b9bcd0e881d7b53078527d3f17ccad61f1b14fdcd8562972b5b",
    RESULTS / "design_adoption.json": "e4204021f95e49e782147d66e56624599661d7e58716a77500bd6e34031008bc",
    OLD_ROWS: "67547a04914f141eaf54617d361d3e88427fae921de858d5fa6771d0784ba099",
    Path("reports/gftd_fit297_20261107/originals/gftd_fit297_20261107_l1_adapter_contract.md"):
        "2d09f180266875a6100c15c61cc915fc5a2e7752d1de641dce6be603d6c74252",
    Path("experiments/gftd_fit297_analysis_20261107.py"):
        "2faa8cdd83a98f71c044e20983b2b08d0f394f76b64c15b24f2c63617dab929e",
}
DENOMINATORS = {"all": 297, "stable": 267, "unstable": 30}
SLOTS = tuple(j * 297 // 24 for j in range(24))
GRID = np.linspace(0., 1., 101)
RHS_LIMIT = 24000
FUTURE_STEPS = {.01: "main", .005: "fine"}
TESTS = ("tests/test_gftd_bounded_kernel.py", "tests/test_gftd_matched_fit.py",
         "tests/test_gftd_matched_analysis.py", "tests/test_gftd_matched_runner.py")
SCORE_SOURCES = (
    "experiments/gftd_fit297_l1_20261107.py", "experiments/evaluate_structured_rom.py",
    "experiments/evaluate_long_latent.py", "experiments/evaluate_l1_latent.py",
    "experiments/frequency_dissipation_20261021.py", "experiments/train_joint_latent.py",
    "experiments/run_private_m2.py", "src/kiki/models/private_pm.py", "src/kiki/models/private_rom.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value, *, allow_nan=False):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=allow_nan) + "\n").encode()


def reference(path, data):
    return dict(path=path.as_posix(), sha256=sha(data), bytes=len(data))


def read_bound(path, digest, size=None):
    data = local_path(path).read_bytes()
    if sha(data) != digest or (size is not None and len(data) != size):
        raise ValueError(f"frozen bytes changed: {path}")
    return data


def read_ref(ref, path):
    if ref["path"] != path.as_posix():
        raise ValueError(f"unexpected evidence path: {path}")
    return read_bound(path, ref["sha256"], ref["bytes"])


def save(path, data, *, exclusive=False):
    dest = local_path(path)
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


def local_path(path):
    path = Path(path)
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT) or resolved == ROOT:
        raise ValueError("path must resolve within the workspace")
    return resolved


def imported_sources():
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT):
                rel = path.relative_to(ROOT)
                if rel.parts[0] in ("src", "experiments", "tests") and rel.suffix == ".py":
                    paths.add(rel.as_posix())
    return paths


def sources():
    run_dir = f"experiments/{RUN}"
    paths = imported_sources() | set(SCORE_SOURCES) | set(TESTS) | {
        f"{run_dir}/run.py", f"{run_dir}/qualify.py", "tests/conftest.py",
    }
    return {path: sha(local_path(path).read_bytes()) for path in sorted(paths)}


def verify_sources(prepared):
    frozen = prepared["source_sha256"]
    if not imported_sources() <= frozen.keys():
        raise ValueError("an imported repository source is outside the frozen closure")
    for path, digest in frozen.items():
        read_bound(path, digest)


def runtime():
    threads = {name: os.environ.get(name) for name in THREADS}
    if set(threads.values()) != {"1"}:
        raise ValueError("one BLAS thread per worker required")
    return dict(python=sys.version.split()[0], threads=threads,
                **{name: version(name) for name in ("numpy", "scipy", "h5py", "torch", "pytest")})


def release_record(path, phase=None):
    raw = local_path(path).read_bytes()
    release = json.loads(raw)
    if (release.get("run") != RUN or release.get("released") is not True
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", release.get("action_id", ""))
            or (phase is not None and release.get("phase") != phase)):
        raise PermissionError("matching root phase release required")
    return release, sha(raw)


def attempt_path(stem, attempt):
    if type(attempt) is not int or not 0 <= attempt <= 2:
        raise ValueError("unregistered qualification attempt")
    suffix = "" if attempt == 0 else f"_{attempt}"
    return RESULTS / f"{stem}{suffix}.json"


def frozen_refs():
    return {
        "parents": {mode: dict(path=path.as_posix(), sha256=digest, kernel=kernel)
                    for mode, (path, digest, kernel) in PARENTS.items()},
        **{name: dict(path=path.as_posix(), sha256=digest) for name, (path, digest) in {
            "input": (INPUT, INPUT_SHA), "constants": CONSTANTS, "fom": FOM, "kron": KRON, "future": FUTURE,
        }.items()},
    }


def public_rows():
    cached = json.loads(read_bound(OLD_ROWS, FROZEN[OLD_ROWS]))
    rows = [{key: row[key] for key in ("index", "event_id", "label")} for row in cached if row["mode"] == "static"]
    validate_order(rows)
    return rows


def validate_order(rows):
    ids = [row["event_id"] for row in rows]
    if (len(rows) != 297 or [row["index"] for row in rows] != list(range(297)) or len(set(ids)) != 297
            or Counter(row["label"] for row in rows) != {"stable": 267, "unstable": 30}
            or sha(json_bytes(ids)) != EVENT_ORDER_SHA):
        raise ValueError("original train297 order or denominator changed")
    return ids


def prepare(path):
    release, release_sha = release_record(path, "prepare")
    attempt = release["qualification_attempt"]
    output = attempt_path("prepare", attempt)
    if (ROOT / RESULTS / "fit/started.json").exists():
        raise PermissionError("new preparation cannot follow scientific execution")
    frozen = {p.as_posix(): reference(p, read_bound(p, digest)) for p, digest in FROZEN.items()}
    required = {"current_closeout", "current_review", "current_source_archive", "current_execution",
                "implementation_review", "qualification_clarification"}
    if attempt:
        required.add("technical_fix")
        previous = json.loads(local_path(attempt_path("qualification", attempt - 1)).read_bytes())
        if previous.get("passed") is not False or previous.get("cost_unknown") is not False:
            raise PermissionError("an accounted failed qualification must precede a technical correction")
    prereqs = release["prerequisites"]
    if set(prereqs) != required or len({ref["path"] for ref in prereqs.values()}) != len(required):
        raise ValueError("independent review, closure and freeze references required")
    actual = {key: read_ref(ref, Path(ref["path"])) for key, ref in prereqs.items()}
    if prereqs["current_execution"]["path"] != (CURRENT_RESULTS / "full/execution.json").as_posix():
        raise ValueError("wrong bounded initialization execution")
    prior = json.loads(actual["current_execution"])
    if (prior["status"] != "completed" or prior["partial_jobs"] or prior["unknown_jobs"]
            or len(prior["jobs"]) != 272 or len(prior["reused_pilot"]) != 25
            or prior["bindings"]["prepare_sha256"] != CURRENT_PREPARE_SHA):
        raise PermissionError("bounded initialization execution is not closed")
    original = json.loads(read_bound(CURRENT_RESULTS / "prepare.json", CURRENT_PREPARE_SHA))
    archived = json.loads(actual["current_source_archive"])
    archive_rows = archived["files"]
    if (archived["run"] != CURRENT or archived["status"] != "archived_and_verified"
            or archived["prepare_sha256"] != CURRENT_PREPARE_SHA or len(archive_rows) != 26
            or {row["source_path"]: row["sha256"] for row in archive_rows} != original["source_sha256"]):
        raise ValueError("the original 26-source archive must match the bounded preparation")
    for row in archive_rows:
        read_ref(row, Path("reports") / CURRENT / "source_snapshot" / row["source_path"])
    rows = public_rows()
    slots = [dict(global_index=index, local_slot=slot, event_id=rows[index]["event_id"])
             for slot, index in enumerate(SLOTS)]
    if not hasattr(RunContext, "gftd_matched_prediction_v1"):
        raise RuntimeError("registered public fit dispatch must be installed before preparation")
    source = sources()
    claim = save_json(attempt_path("prepare_claim", attempt),
                      dict(run=RUN, action_id=release["action_id"], release_sha256=release_sha, pid=os.getpid()),
                      exclusive=True)
    return save_json(output, dict(run=RUN, claim=claim, qualification_attempt=attempt, prerequisites=prereqs,
                                  source_sha256=source, runtime=runtime(), frozen=frozen, assets=frozen_refs(),
                                  event_order_sha256=EVENT_ORDER_SHA, adaptation_slots=slots,
                                  denominators=DENOMINATORS, original_payload_reads=0))


def bound_preparation(release):
    prepared = json.loads(read_bound(attempt_path("prepare", release["qualification_attempt"]),
                                     release["prepare_sha256"]))
    if (prepared["run"] != RUN or prepared["assets"] != frozen_refs() or prepared["runtime"] != runtime()
            or prepared["qualification_attempt"] != release["qualification_attempt"]):
        raise ValueError("prepared run, asset identities or runtime changed")
    for refs in (prepared["frozen"], prepared["prerequisites"]):
        for ref in refs.values():
            read_ref(ref, Path(ref["path"]))
    verify_sources(prepared)
    return prepared


def load_packets(rows):
    raw = json.loads(read_bound(INPUT, INPUT_SHA))
    ids = validate_order(rows)
    receipt = raw["receipt"]
    if (receipt["legal_ids"] != ids or receipt["ordered_all_ids"] != ids or receipt["invalid"]
            or any(receipt["denominators"][key] != n for key, n in DENOMINATORS.items())):
        raise ValueError("frozen legal packet cohort changed")
    result = []
    for row, original in zip(rows, raw["packets"], strict=True):
        if set(original) != {"theta_history", "omega_history", "history_times", "p0", "sigma"}:
            raise ValueError("unexpected legal packet fields")
        packet = core.validate_observation(
            {("p_end" if key == "p0" else key): value for key, value in original.items()}, 10)
        if packet["sigma"] != 0.:
            raise ValueError(f"nonzero sigma in packet {row['index']}")
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
        if f is not model.functional or stepper is not fitting.dop853_step:
            raise CaptureIdentityError("unexpected functional or integration method")
        leg = None
        if duration == 1. and not backward and not dense and step in FUTURE_STEPS:
            leg = FUTURE_STEPS[step]
        elif duration == .1 and not dense and step in (.0025, .00125):
            leg = ("history_reverse_" if backward else "history_return_") + ("main" if step == .0025 else "fine")
        if leg is not None:
            gate = gate_slots.setdefault(id(counts), len(gate_slots) + 1)
            key = f"dp8_gate{gate}_{leg}"
            if gate > 2 or key in captured:
                raise CaptureIdentityError("unexpected extra DP8 gate integration")
        states = original(f, endpoint, step, duration, counts, backward=backward, dense=dense, stepper=stepper)
        if leg is not None:
            endpoint_sha = sha(endpoint.tobytes())
            shape = (101 if duration == 1. else 11, 50)
            if (states.shape != shape or states.dtype != np.float64
                    or sha(states[-1 if backward else 0].tobytes()) != endpoint_sha):
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
    anchor = np.r_[packet["theta_history"][-1], packet["omega_history"][-1], packet["p0"]]
    if (state.dtype != np.float64 or state.shape != (50,) or not np.isfinite(state).all()
            or state[:30].tobytes() != anchor.tobytes() or diagnostics["status"] != "completed"
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


def write_arrays(path, arrays):
    data = io.BytesIO()
    np.savez_compressed(data, **arrays)
    return save(path, data.getvalue())


def prediction(functional, state, step, record):
    steps = round(15. / step)
    path = np.full((steps + 1, 50), np.nan)
    path[0] = state
    started, cpu = time.perf_counter(), time.process_time()
    record.update(status="running", step=step, actual_rhs_calls=0, steps_attempted=0, steps_completed=0,
                  state_sha256=sha(state.tobytes()))

    def rhs(value):
        record["actual_rhs_calls"] += 1
        return functional.rhs(value)

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for index in range(steps):
                record["steps_attempted"] += 1
                path[index + 1] = dop853_step(rhs, path[index], step)
                record["steps_completed"] += 1
        record["status"] = "completed"
    except (FloatingPointError, OverflowError, ValueError) as error:
        record.update(status="numerical_failure", failure_type=type(error).__name__,
                      first_failed_step=record["steps_attempted"])
    finally:
        record.update(wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
    if not 12 * record["steps_completed"] <= record["actual_rhs_calls"] <= 12 * record["steps_attempted"]:
        raise ValueError("forecast work accounting violated the DP8 step allowance")
    return path


def load_parent(mode):
    path, digest, kernel = PARENTS[mode]
    read_bound(path, digest)
    model = GFTDKuramotoModel.load(local_path(path))
    config = fitting.validate_config(model.functional, model.inference_config)
    if (model.fit_mode != mode or model.functional.modulation_kernel != kernel
            or config["format"] != fitting.FIXED_DP8_FORMAT or config["fit297_config"]["training_count"] != 297
            or model.parent_initializer is not None):
        raise ValueError("registered parent lost its kernel, mode or fixed DP8 contract")
    if mode == "static" and np.any(model.functional.gates.s_mod != 0.):
        raise ValueError("static parent has modulation")
    return model


def fixed_identity(model):
    values = fitting._physics(model.functional, include_edges=False)
    values.pop("modulation_kernel", None)
    scales = model.inference_config["fit297_config"]["fit_request"]
    return fitting._hash(dict(physics=values, delta_star=model.delta_star,
                              scales={k: scales[k] for k in ("input_scale_delta", "input_scale_omega", "rstar", "k_ref")}))


def common_parents():
    first, second = (load_parent(mode) for mode in PARENTS)
    if fixed_identity(first) != fixed_identity(second):
        raise ValueError("parent non-edge constants, coordinates or registered scales differ")
    return fitting._plain(dict(delta_star=first.delta_star, mass=first.functional.mass))


def job_paths(action, key):
    stage = action["phase"]
    return RESULTS / stage / "jobs" / f"{key}.json", ARTIFACTS / stage / key


def job_record(action, key, **identity):
    path, private = job_paths(action, key)
    meta = dict(action, job=key, **identity)
    claim = save_json(path.parent.parent / "claims" / path.name, dict(meta, pid=os.getpid()), exclusive=True)
    return path, private, dict(meta, claim=claim, status="running")


def persist_job(path, record, clock):
    start, cpu = clock
    record.update(worker_wall_seconds=time.perf_counter() - start, worker_cpu_seconds=time.process_time() - cpu)
    try:
        return save_json(path, record)
    except OSError:
        print(json.dumps({key: record.get(key) for key in ("action_id", "job", "status")}
                         | {"known_costs": job_costs(record)}), file=sys.stderr, flush=True)
        raise


def check_fit_counts(mode, counts):
    caps = dict(optimization_tasks=192, actual_residual_calls=230400, proxy_rhs_calls=110592000,
                qualification_suites=96, qualification_rhs_calls=622080, information_rhs_calls=23040,
                port_rhs_calls=23040, memory_svd=48, nnls_attempts=24573, algebra_terms_actual=24573)
    diagnostic = 1056 if mode == "static" else 2016
    caps.update(diagnostic_residual_calls=diagnostic, diagnostic_rhs_calls=480 * diagnostic,
                information_svd=140 if mode == "static" else 230,
                conditional_projections=90 if mode == "static" else 180)
    for key, cap in caps.items():
        if type(counts.get(key)) is not int or not 0 <= counts[key] <= cap:
            raise ValueError(f"shared fit accounting outside registered allowance: {key}")
    components = ("proxy_rhs_calls", "qualification_rhs_calls", "diagnostic_rhs_calls", "information_rhs_calls")
    if (counts["actual_rhs_calls"] != sum(counts[key] for key in components)
            or counts["proxy_rhs_calls"] > 480 * counts["actual_residual_calls"]
            or counts["qualification_rhs_calls"] > 6480 * counts["qualification_suites"]
            or counts["diagnostic_rhs_calls"] > 480 * counts["diagnostic_residual_calls"]):
        raise ValueError("shared fit RHS breakdown does not reconcile")


def fit_job(action, mode, prepared, packets):
    clock = time.perf_counter(), time.process_time()
    path, private, result = job_record(action, mode, mode=mode, parent_sha256=PARENTS[mode][1], input_sha256=INPUT_SHA)
    result.update(known_costs=None, final_model=None, details=None)
    try:
        model = load_parent(mode)
        fixed = fixed_identity(model)
        request = dict(format=FIT_FORMAT, run_id=RUN, protocol_sha256=PROTOCOL_SHA, mode=mode,
                       parent_model_sha256=PARENTS[mode][1], kernel=PARENTS[mode][2], training_count=297,
                       adaptation_slots=prepared["adaptation_slots"], input_sha256=INPUT_SHA,
                       event_order_sha256=EVENT_ORDER_SHA, source_sha256=prepared["source_sha256"],
                       parent_fit297_config=model.inference_config["fit297_config"])
        result["request_sha256"] = sha(json_bytes(request))
        try:
            fit_result = model.fit(packets, [], RunContext(gftd_matched_prediction_v1=request))
        except fitting.FitFailure as error:
            details = error.diagnostics
            result["known_costs"] = details.get("counts")
            persist_job(path, result, clock)
            result["details"] = save(private / "fit_failure.json.gz",
                                     gzip.compress(json_bytes(details, allow_nan=True), mtime=0))
            fit_status = details.get("status")
            numerical = isinstance(error.__cause__, (FloatingPointError, OverflowError, np.linalg.LinAlgError))
            if fit_status != "fit_failed" and not (
                    numerical and fit_status in ("preparation_failed", "execution_unresolved")):
                raise
            check_fit_counts(mode, result["known_costs"])
            result.update(status="fit_failed", fit_status=fit_status, failure_type=type(error).__name__)
        else:
            state, details = model.fit_state, model.fit_private_details
            result["known_costs"] = state["counts"]
            persist_job(path, result, clock)
            check_fit_counts(mode, state["counts"])
            if (fit_result.status not in {"completed", "not_identified"} or model.fit_mode != mode
                    or model.functional.modulation_kernel != PARENTS[mode][2] or not model._uses_fixed_dp8
                    or fixed_identity(model) != fixed or state["adaptation_slots"] != request["adaptation_slots"]
                    or fitting._hash(details) != state["private_details_sha256"]):
                raise ValueError("public fit returned incompatible configuration or lineage")
            result["details"] = save(private / "fit_details.json.gz",
                                     gzip.compress(json_bytes(details, allow_nan=True), mtime=0))
            pending = local_path(private / "model.preparing.json")
            model.save(pending)
            result["final_model"] = save(private / "model.json", pending.read_bytes())
            pending.unlink()
            result.update(status="completed", fit_status=fit_result.status,
                          config_sha256=sha(json_bytes(model.inference_config)),
                          preparation_key=state["preparation_key"],
                          selected_candidate_index=state["selected_candidate_index"],
                          accepted_proposal_indices=state["accepted_proposal_indices"])
    except BaseException as error:
        result.update(status="interrupted", error_type=type(error).__name__)
        raise
    finally:
        persist_job(path, result, clock)


MODELS_IN_WORKER = {}


def final_model(mode, job):
    path = ARTIFACTS / "fit" / mode / "model.json"
    read_ref(job["final_model"], path)
    model = GFTDKuramotoModel.load(local_path(path))
    if (not model._uses_fixed_dp8 or model.functional.modulation_kernel != PARENTS[mode][2]
            or model.fit_mode != mode or sha(json_bytes(model.inference_config)) != job["config_sha256"]):
        raise ValueError("final checkpoint no longer matches the frozen fit")
    return model


def setup_deploy(prepared, fits):
    ensure_worker(prepared)
    for mode, job in fits.items():
        if job["status"] == "completed":
            MODELS_IN_WORKER[mode] = final_model(mode, job)


def deploy_job(action, mode, row, packet, packet_sha, model_sha):
    clock = time.perf_counter(), time.process_time()
    key = f"{mode}_{row['index']:03d}"
    path, private, result = job_record(action, key, mode=mode, **row, packet_sha256=packet_sha,
                                       model_sha256=model_sha)
    result.update(initialization_counts=None, formal_point=None, reference=None, prediction=None,
                  diagnostics=None, initialization_arrays=None, reference_arrays=None, prediction_arrays=None)
    arrays, diagnostics = {}, {}
    try:
        model = MODELS_IN_WORKER[mode]
        state, paths = initialize(model, {k: v for k, v in packet.items() if k != "sigma"}, result, arrays, diagnostics)
        persist_job(path, result, clock)
        result["diagnostics"] = save(private / "diagnostics.json.gz",
                                     gzip.compress(json_bytes(diagnostics, allow_nan=True), mtime=0))
        if arrays:
            result["initialization_arrays"] = write_arrays(private / "initialization.npz", arrays)
        if state is not None:
            result["reference"] = {}
            persist_job(path, result, clock)
            radau = radau_reference(model.functional, state, result["reference"])
            for kind in FUTURE_STEPS.values():
                result["reference"][kind] = reference_comparison(
                    model.functional.mass, paths[kind], radau, result["reference"]["status"] == "completed")
            persist_job(path, result, clock)
            result["reference_arrays"] = write_arrays(private / "radau.npz", dict(times=GRID, states=radau))
            result["prediction"], forecast = {}, {}
            for step, kind in FUTURE_STEPS.items():
                result["prediction"][kind] = {}
                persist_job(path, result, clock)
                forecast[kind] = prediction(model.functional, state, step, result["prediction"][kind])
            persist_job(path, result, clock)
            result["prediction_arrays"] = write_arrays(private / "prediction.npz", forecast)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="interrupted", error_type=type(error).__name__)
        raise
    finally:
        persist_job(path, result, clock)


def process_jobs(worker, arguments, workers, *, initializer, initargs):
    iterator = iter(arguments)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initializer, initargs=initargs) as pool:
        pending = {pool.submit(worker, *args) for _, args in zip(range(workers), iterator)}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                pending.update(pool.submit(worker, *args) for _, args in zip(done, iterator))
        finally:
            for future in pending:
                future.cancel()


def job_costs(job):
    clock = {key: job.get(key, 0.) for key in ("worker_wall_seconds", "worker_cpu_seconds")}
    if "known_costs" in job:
        return clock | (job["known_costs"] or {})
    init = job["initialization_counts"]
    rhs = job.get("known_initialization_rhs_calls", 0) if init is None else init["actual_rhs_calls"]
    residual = job.get("known_initialization_residual_calls", 0) if init is None else init["actual_residual_calls"]
    return clock | dict(initialization_rhs_calls=rhs, initialization_residual_calls=residual,
                        optimization_tasks=0 if init is None else init["optimization_tasks"],
                        reference_rhs_calls=(job["reference"] or {}).get("actual_rhs_calls", 0),
                        prediction_rhs_calls=sum(item.get("actual_rhs_calls", 0)
                                                 for item in (job["prediction"] or {}).values()))


def reconcile(action, keys):
    records, final, partial, unknown, unclaimed = {}, {}, {}, [], []
    costs = Counter()
    for key in keys:
        path, _ = job_paths(action, key)
        claim_path = path.parent.parent / "claims" / path.name
        if not local_path(claim_path).exists():
            unclaimed.append(key)
            continue
        try:
            raw = local_path(path).read_bytes()
            job = json.loads(raw)
            claim = json.loads(read_ref(job["claim"], claim_path))
            if (job["job"] != key or claim["job"] != key
                    or any(job[k] != value or claim[k] != value for k, value in action.items())
                    or any(k != "pid" and job.get(k) != value for k, value in claim.items())):
                raise ValueError("receipt no longer matches the job claim")
        except (OSError, KeyError, TypeError, ValueError):
            unknown.append(key)
            continue
        ref = reference(path, raw)
        records[key] = ref, job
        target = final if job["status"] in ("completed", "fit_failed") else partial
        target[key] = ref
        costs.update(job_costs(job))
    return records, dict(jobs=final, partial_jobs=partial, unknown_jobs=unknown, unclaimed_jobs=unclaimed,
                         known_costs=dict(costs), actual_costs=None if unknown or partial else dict(costs))


def completed_jobs(action, keys):
    records, summary = reconcile(action, keys)
    if len(summary["jobs"]) != len(keys) or summary["unknown_jobs"] or summary["partial_jobs"]:
        raise ValueError("incomplete job receipts require reconciliation")
    return records


def bound_stage(release, phase, binding):
    path = RESULTS / phase / "execution.json"
    record = json.loads(read_bound(path, release[f"{phase}_execution_sha256"]))
    action = record["action"]
    if (action["run"] != RUN or action["phase"] != phase or record["status"] != "completed"
            or action["action_id"] == release["action_id"] or record["bindings"] != binding
            or record["unknown_jobs"] or record["partial_jobs"] or record["unclaimed_jobs"]):
        raise PermissionError(f"same-preparation {phase} completion must be frozen first")
    return record


def frozen_fits(release, binding):
    phase = bound_stage(release, "fit", binding)
    path = RESULTS / "fit/model_freeze.json"
    if phase["freeze"]["sha256"] != release["model_freeze_sha256"]:
        raise ValueError("downstream release must explicitly bind the model freeze")
    frozen = json.loads(read_ref(phase["freeze"], path))
    if (frozen["action"] != phase["action"] or frozen["jobs"] != phase["jobs"]
            or set(frozen["jobs"]) != set(PARENTS)):
        raise ValueError("model freeze is incomplete")
    fits = {}
    for mode, ref in frozen["jobs"].items():
        job_path, private = job_paths(phase["action"], mode)
        job = json.loads(read_ref(ref, job_path))
        if job["status"] not in ("completed", "fit_failed") or job["mode"] != mode:
            raise ValueError("frozen fit outcome changed")
        details = "fit_details.json.gz" if job["status"] == "completed" else "fit_failure.json.gz"
        read_ref(job["details"], private / details)
        if job["status"] == "completed":
            read_ref(job["final_model"], private / "model.json")
        elif job["final_model"] is not None:
            raise ValueError("failed fit cannot supply a frozen model")
        fits[mode] = job
    return frozen, fits


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return finite_json(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def analysis_job(action, mode, fit, rows, jobs, taylor, fom, kron):
    clock = time.perf_counter(), time.process_time()
    key = f"analysis_{mode}"
    path, private, result = job_record(action, key, mode=mode, model_sha256=fit["final_model"]["sha256"])
    before = taylor.calls
    counts = dict(l2_radii=0, l2_points=0, l2_taylor_calls=0, l3_rhs_calls=0, l3_gradient_calls=0, l3_eig=0)
    result.update(known_costs=counts, analysis=None)
    try:
        model = final_model(mode, fit)
        f = model.functional
        cert = analysis.l2_certificate(f, model.delta_star)
        counts["l2_radii"] = len(cert.get("candidates", []))
        points = []
        for row in rows:
            job_key = f"{mode}_{row['index']:03d}"
            job = jobs[job_key][1]
            point = job["formal_point"]
            if point is None:
                points.append(None)
                continue
            _, data_path = job_paths(action, job_key)
            with np.load(io.BytesIO(read_ref(job["initialization_arrays"], data_path / "initialization.npz")),
                         allow_pickle=False) as archive:
                state = archive["formal_state"]
            if sha(state.tobytes()) != point["state_sha256"]:
                raise ValueError("analysis endpoint differs from the frozen selected point")
            counts["l2_points"] += 1
            points.append(analysis.l2_point(cert, f, state, taylor_budget=taylor))
        rom = analysis.rom_modes(f, model.delta_star, cert)
        counters = rom.get("counters", {})
        counts.update(l2_taylor_calls=taylor.calls - before, l3_rhs_calls=counters.get("rhs_calls", 0),
                      l3_gradient_calls=counters.get("gradient_calls", 0), l3_eig=counters.get("eig_count", 0))
        if any(counts[k] > cap for k, cap in dict(l2_radii=13, l2_points=297, l3_rhs_calls=217,
                                                 l3_gradient_calls=980, l3_eig=2).items()):
            raise ValueError("L2/L3 work exceeds the registered bounds")
        persist_job(path, result, clock)
        record = dict(certificate=cert, points=points, rom_modes=rom, modal_match=analysis.match_modes(fom, rom),
                      graph_correspondence=analysis.graph_correspondence(kron, f, model.delta_star))
        result["analysis"] = save(private / "analysis.json.gz", gzip.compress(json_bytes(finite_json(record)), mtime=0))
        result.update(status="completed", certificate_status=cert["status"], modal_status=rom["status"])
    except BaseException as error:
        result.update(status="interrupted", error_type=type(error).__name__)
        raise
    finally:
        counts["l2_taylor_calls"] = taylor.calls - before
        persist_job(path, result, clock)


def deployment_table(rows, fits, jobs):
    table = {}
    for mode, fit in fits.items():
        result = []
        for row in rows:
            item = dict(row, evidence=None, initialization_status="not_run", formal_point=None,
                        six_gates_passed=None, fine_reference_confirmed=None, main_reference_confirmed=None,
                        prediction_arrays=None, prediction=None)
            key = f"{mode}_{row['index']:03d}"
            if fit["status"] == "completed":
                evidence, job = jobs[key]
                for field in ("initialization_status", "formal_point", "six_gates_passed",
                              "prediction_arrays", "prediction"):
                    item[field] = job[field]
                item["evidence"] = evidence
                for kind in ("fine", "main"):
                    item[f"{kind}_reference_confirmed"] = ((job["reference"] or {}).get(kind) or {}).get("confirmed")
            result.append(item)
        table[mode] = result
    return table


def deployment_summary(table):
    summary = {}
    for mode, rows in table.items():
        summary[mode] = {}
        for label, denominator in DENOMINATORS.items():
            cohort = [row for row in rows if label == "all" or row["label"] == label]
            summary[mode][label] = dict(
                denominator=denominator, statuses=dict(Counter(row["initialization_status"] for row in cohort)),
                **{field: [row["event_id"] for row in cohort if row[field] is True]
                   for field in ("six_gates_passed", "fine_reference_confirmed", "main_reference_confirmed")})
    return summary


def deploy_stage(action, prepared, rows, packets, fits, workers, keys):
    modes = [mode for mode in PARENTS if fits[mode]["status"] == "completed"]
    fom, kron = None, None
    if modes:
        fom, kron = (json.loads(read_bound(*spec)) for spec in (FOM, KRON))
        if fom["status"] != "qualified" or len(fom["modes"]) != 13:
            raise ValueError("cached FOM reference is not the original qualified 13 modes")
    args = [(action, mode, row, *packets[row["index"]], fits[mode]["final_model"]["sha256"])
            for row in rows for mode in modes]
    if args:
        process_jobs(deploy_job, args, workers, initializer=setup_deploy, initargs=(prepared, fits))
    deploy_keys = [f"{mode}_{row['index']:03d}" for row in rows for mode in modes]
    jobs = completed_jobs(action, deploy_keys)
    table = deployment_table(rows, fits, jobs)
    freeze = save_json(RESULTS / "deploy/initialization_freeze.json",
                       dict(action=action, rows=table, event_order_sha256=EVENT_ORDER_SHA,
                            models={mode: fits[mode]["final_model"] for mode in PARENTS},
                            denominators=DENOMINATORS, summary=deployment_summary(table)))
    if modes:
        taylor = analysis.TaylorBudget(160380)
        for mode in modes:
            analysis_job(action, mode, fits[mode], rows, jobs, taylor, fom, kron)
    completed_jobs(action, keys)
    return freeze


def frozen_deployment(release, binding, fits, rows):
    stage = bound_stage(release, "deploy", binding)
    ref = stage["freeze"]
    if ref["sha256"] != release["initialization_freeze_sha256"]:
        raise ValueError("score must explicitly bind the initialization freeze")
    frozen = json.loads(read_ref(ref, RESULTS / "deploy/initialization_freeze.json"))
    if (frozen["action"] != stage["action"] or frozen["event_order_sha256"] != EVENT_ORDER_SHA
            or frozen["denominators"] != DENOMINATORS or set(frozen["rows"]) != set(PARENTS)
            or frozen["models"] != {mode: job["final_model"] for mode, job in fits.items()}):
        raise ValueError("initialization freeze does not match the final model pair")
    for mode, table in frozen["rows"].items():
        for row, frozen_row in zip(rows, table, strict=True):
            if any(frozen_row[key] != value for key, value in row.items()):
                raise ValueError("initialization freeze changed event order")
            ref = frozen_row["evidence"]
            key = f"{mode}_{row['index']:03d}"
            if fits[mode]["status"] == "fit_failed":
                if ref is not None or frozen_row["formal_point"] is not None:
                    raise ValueError("failed model was used for deployment")
                continue
            if ref != stage["jobs"].get(key) or ref is None:
                raise ValueError("a deployment result lacks frozen evidence")
            path, _ = job_paths(frozen["action"], key)
            record = json.loads(read_ref(ref, path))
            if (record["status"] != "completed" or record["model_sha256"] != fits[mode]["final_model"]["sha256"]
                    or any(record[field] != frozen_row[field] for field in
                           ("formal_point", "six_gates_passed", "prediction_arrays", "prediction"))):
                raise ValueError("deployment result differs from its frozen selection")
    return frozen


def scoring_module(prepared):
    sys.path.insert(0, str(ROOT / "experiments"))
    import gftd_fit297_l1_20261107 as l1
    if prepared is not None:
        verify_sources(prepared)
    return l1


def load_prediction(frozen, mode, row):
    key = f"{mode}_{row['index']:03d}"
    _, private = job_paths(frozen["action"], key)
    with np.load(io.BytesIO(read_ref(row["prediction_arrays"], private / "prediction.npz")),
                 allow_pickle=False) as archive:
        main, fine = archive["main"], archive["fine"]
    if (main.shape != (1501, 50) or fine.shape != (3001, 50)
            or main.dtype != np.float64 or fine.dtype != np.float64
            or sha(main[0].tobytes()) != row["formal_point"]["state_sha256"]
            or main[0].tobytes() != fine[0].tobytes()):
        raise ValueError("forecast grids do not share their exact frozen endpoint")
    return main, fine


def verify_scoring_event(event, row, packet, delta_star):
    if (str(event["id"]) != row["event_id"] or event["label"] != row["label"]
            or "group" not in event or "classification_meta" not in event):
        raise ValueError("truth cache event identity or classification metadata changed")
    x = event["x"]
    if (x.dtype != np.float64 or x.ndim != 2 or x.shape[1] != 30 or len(x) < 11
            or not np.array_equal(x[:11, :10] + delta_star, packet["theta_history"])
            or not np.array_equal(x[:11, 10:20], packet["omega_history"])
            or not np.array_equal(x[10, 20:30], packet["p0"])):
        raise ValueError("truth cache and legal packet do not correspond")


def score_stage(prepared, frozen, model_freeze, fits, rows, packets, receipt):
    l1 = scoring_module(prepared)
    coordinates = json.loads(read_ref(model_freeze["common_coordinates"], ARTIFACTS / "fit/common_coordinates.json"))
    delta_star = np.asarray(coordinates["delta_star"], dtype=np.float64)
    mass = np.asarray(coordinates["mass"], dtype=np.float64)
    constants = json.loads(read_bound(*CONSTANTS))
    p_scale = np.asarray(constants["scales"]["p_scale"], dtype=np.float64)
    if (delta_star.shape != (10,) or not np.isfinite(delta_star).all()
            or any(v.shape != (10,) or not np.isfinite(v).all() or (v <= 0).any() for v in (mass, p_scale))):
        raise ValueError("scoring coordinates or frozen positive scales changed")
    records = {mode: [] for mode in PARENTS}
    future, expected = FUTURE
    receipt["truth_cache_opened"] = True
    save_json(RESULTS / "score/execution.json", receipt)
    with local_path(future).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError("frozen future cache changed")
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            ids = [str(event["id"]) for event in metadata]
            if (ids != validate_order(rows) or [event["label"] for event in metadata] != [row["label"] for row in rows]
                    or len(metadata) != 297):
                raise ValueError("future cache order, labels or denominator changed")
            for row, meta, (packet, _) in zip(rows, metadata, packets, strict=True):
                index = row["index"]
                event = dict(meta, x=archive[f"event_{index:04d}_x"])
                verify_scoring_event(event, row, packet, delta_star)
                for mode in PARENTS:
                    selected = frozen["rows"][mode][index]
                    if selected["formal_point"] is None:
                        stage = "fit" if fits[mode]["status"] == "fit_failed" else "initialization"
                        record = l1.failure_event(event, stage, RuntimeError(selected["initialization_status"]))
                    else:
                        main, fine = load_prediction(frozen, mode, selected)
                        record = l1.score_event(event, fine, main, delta_star=delta_star, mass=mass, p_scale=p_scale)
                        record["integration_errors"] = [kind for kind, status in selected["prediction"].items()
                                                        if status["status"] != "completed"]
                    ref = save_json(RESULTS / "score/events" / f"{mode}_{index:03d}.json", finite_json(record))
                    receipt["scored_rows"].append(ref)
                    records[mode].append(record)
    return save_json(RESULTS / "score/l1_summary.json",
                     finite_json(l1.paired_summary(records["static"], records["modulated"])))


def qualification_binding(release, prepared):
    attempt = release["qualification_attempt"]
    binding = {key: release[key] for key in ("qualification_attempt", "prepare_sha256", "qualification_sha256")}
    qualification = json.loads(read_bound(attempt_path("qualification", attempt), binding["qualification_sha256"]))
    if (qualification["run"] != RUN or qualification["qualification_attempt"] != attempt
            or qualification["prepare_sha256"] != binding["prepare_sha256"]
            or qualification["source_sha256"] != prepared["source_sha256"] or qualification["passed"] is not True
            or qualification["cost_unknown"] or qualification["violations"]
            or not 0 < qualification["cumulative_counts"]["rhs"] <= 128
            or not 0 < qualification["cumulative_counts"]["hamiltonian"] <= 2000):
        raise PermissionError("the fixed metered implementation qualification must pass first")
    read_ref(qualification["claim"], attempt_path("qualification_claim", attempt))
    return binding


def ensure_worker(prepared):
    verify_sources(prepared)
    if runtime() != prepared["runtime"]:
        raise ValueError("worker runtime changed")


def execute(path):
    release, release_sha = release_record(path)
    phase = release["phase"]
    if phase not in ("fit", "deploy", "score"):
        raise ValueError("unknown scientific stage")
    prepared = bound_preparation(release)
    binding = qualification_binding(release, prepared)
    workers = release.get("workers")
    if (type(workers) is not int or workers < 1 or (phase == "fit" and workers > len(PARENTS))
            or (phase == "score" and workers != 1)):
        raise ValueError("invalid worker count for the registered stage")
    rows = public_rows()
    model_freeze, fits, frozen = None, None, None
    if phase != "fit":
        model_freeze, fits = frozen_fits(release, binding)
    if phase == "deploy" and (release["fom_sha256"] != FOM[1] or release["kron_sha256"] != KRON[1]):
        raise ValueError("deployment release must explicitly bind original FOM and Kron caches")
    if phase == "score":
        if release["future_sha256"] != FUTURE[1]:
            raise ValueError("score release must explicitly bind original train cache")
        frozen = frozen_deployment(release, binding, fits, rows)
    output = RESULTS / phase
    action = dict(run=RUN, phase=phase, action_id=release["action_id"], release_sha256=release_sha,
                  prepare_sha256=binding["prepare_sha256"], qualification_sha256=binding["qualification_sha256"])
    claim = save_json(output / "started.json", dict(action, pid=os.getpid()), exclusive=True)
    clock = time.perf_counter(), time.process_time()
    keys = list(PARENTS) if phase == "fit" else []
    if phase == "deploy":
        modes = [mode for mode in PARENTS if fits[mode]["status"] == "completed"]
        keys = [f"{mode}_{row['index']:03d}" for row in rows for mode in modes]
        keys.extend(f"analysis_{mode}" for mode in modes)
    receipt = dict(action=action, claim=claim, bindings=binding, status="running", planned_jobs=keys,
                   workers=workers, denominators=DENOMINATORS, freeze=None, truth_cache_opened=False, scored_rows=[])
    try:
        save_json(output / "execution.json", receipt)
        packets = load_packets(rows)
        if phase == "fit":
            coords = save_json(ARTIFACTS / "fit/common_coordinates.json", common_parents())
            arguments = [(action, mode, prepared, [packet for packet, _ in packets]) for mode in PARENTS]
            process_jobs(fit_job, arguments, workers, initializer=ensure_worker, initargs=(prepared,))
            jobs = completed_jobs(action, keys)
            receipt["freeze"] = save_json(output / "model_freeze.json",
                                          dict(action=action, jobs={key: ref for key, (ref, _) in jobs.items()},
                                               common_coordinates=coords))
        elif phase == "deploy":
            receipt["freeze"] = deploy_stage(action, prepared, rows, packets, fits, workers, keys)
        else:
            receipt["freeze"] = score_stage(prepared, frozen, model_freeze, fits, rows, packets, receipt)
        verify_sources(prepared)
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="interrupted", error_type=type(error).__name__)
        raise
    finally:
        _, reconciliation = reconcile(action, keys)
        receipt.update(reconciliation, host_wall_seconds=time.perf_counter() - clock[0],
                       host_cpu_seconds=time.process_time() - clock[1])
        try:
            save_json(output / "execution.json", receipt)
        except OSError:
            print(json.dumps({key: receipt[key] for key in ("action", "status", "known_costs", "unknown_jobs")}),
                  file=sys.stderr, flush=True)
            raise
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--prepare", type=Path)
    command.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.prepare is not None:
        print(json.dumps(prepare(args.prepare)))
    else:
        result = execute(args.release)
        print(json.dumps({key: result[key] for key in ("action", "status", "freeze", "known_costs")}))

