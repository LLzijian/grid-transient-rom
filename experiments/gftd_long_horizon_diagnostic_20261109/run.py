"""Long-horizon diagnostics from the closed initialization Run's frozen states."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import io
from itertools import combinations
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np
import scipy
from scipy.integrate import DOP853, Radau

from kiki.models import GFTDKuramotoModel
from kiki.models.gftd_discovery import observations
from kiki.models.gftd_fitting import FIXED_DP8_COEFFICIENTS_SHA256
from kiki.numerics import dop853_step
from kiki.physics.kuramoto_functional import bregman_kuramoto_potential


ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_long_horizon_diagnostic_20261109"
PARENT = "gftd_initialization_precision_20261109"
OUTPUT = Path("results") / RUN
PRIVATE = Path("artifacts") / RUN
PROTOCOL = Path("campaigns") / RUN / "protocol.md"
PROTOCOL_SHA256 = "9d8287dd7cb717dbb703585034977bfddc8bde72a970a5f301d8adadce58e74e"
SOURCE_PATHS = (
    f"experiments/{RUN}/run.py", "tests/test_gftd_long_horizon_diagnostic.py",
    "src/kiki/data/__init__.py", "src/kiki/data/adapter.py", "src/kiki/data/contracts.py",
    "src/kiki/data/schema.py", "src/kiki/data/splits.py", "src/kiki/data/store.py",
    "src/kiki/__init__.py", "src/kiki/models/__init__.py", "src/kiki/models/base.py",
    "src/kiki/models/gftd_kuramoto.py", "src/kiki/models/gftd_fit297.py", "src/kiki/models/gftd_fitting.py",
    "src/kiki/models/gftd_discovery.py", "src/kiki/numerics.py", "src/kiki/physics/__init__.py",
    "src/kiki/physics/aggregation.py", "src/kiki/physics/constants.py", "src/kiki/physics/coordinates.py",
    "src/kiki/physics/graph.py", "src/kiki/physics/kuramoto_benchmarks.py",
    "src/kiki/physics/kuramoto_functional.py", "src/kiki/physics/units.py",
)
PARENT_OUTPUT = Path("results") / PARENT
PARENT_RAW = Path("artifacts") / PARENT / "candidate"
PARENT_MANIFEST = PARENT_OUTPUT / "raw_artifacts_manifest.json"
PARENT_EXECUTION = PARENT_OUTPUT / "candidate/full/execution.json"
PARENT_ROWS = PARENT_OUTPUT / "candidate/full/rows.json"
PARENT_DIGEST = PARENT_OUTPUT / "candidate_future_scientific_digest.json"
PARENT_HASHES = {
    PARENT_MANIFEST: "28f8d6dff0ed2d04b48540fcd4fa37f3ec3e996b3f67314006ba391dd55f86eb",
    PARENT_EXECUTION: "70ab37bb76a48030eceac7cbf272e1bbd28761d7c1905c3b2a65b08283741436",
    PARENT_ROWS: "67547a04914f141eaf54617d361d3e88427fae921de858d5fa6771d0784ba099",
    PARENT_DIGEST: "94a2aed5cf9d7b48a218a922f076cca7a7714cfa37cc560c3f431f13f497f733",
}
CASES = (
    ("modulated_008", None), ("modulated_124", None), ("modulated_138", None), ("modulated_204", None),
    ("static_027", None), ("static_000", None), ("modulated_296", None), ("modulated_048", None),
    ("modulated_268", "stage2"), ("modulated_268", "stage1"),
)
STAGE_STATE_HASHES = {
    "stage1": "055b46d107410d868c10eb69a1b2c7490c933938a1ffe65f3066cbcba81f5a46",
    "stage2": "37382722d19bd318c616b8c59f43390d004c874f3c2c509ee7d01188dc36c469",
}
STAGE_P0_HASH = "3cafb5e809889c5664c11dcb686bd5462308a3824c7f0029ebb164d396ea225f"
TARGET_SAMPLES = (241, 129, 188, 454, 1500, 1500, 1500, 110, 100, 100)
REFERENCES = {
    "radau": (Radau, 1e-9, 1e-11), "radau_strict": (Radau, 1e-11, 1e-13),
    "dop853_adaptive": (DOP853, 1e-11, 1e-13),
}
FIXED_STEPS = {"main": .01, "fine": .005, "finer": .0025, "finest": .00125}
NUMERICAL_ERRORS = (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError)
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
RHS_VERSION = "original_strict"


@dataclass
class FrozenCase:
    state: np.ndarray
    cache: dict[str, np.ndarray] | None
    integrals: dict | None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def workspace_path(relative):
    path = ROOT / relative
    resolved = path.resolve()
    if Path(relative).is_absolute() or resolved != path or not resolved.is_relative_to(ROOT):
        raise ValueError(f"unexpected workspace path: {relative}")
    return path


def checked_bytes(relative, expected_sha, expected_size=None):
    raw = workspace_path(relative).read_bytes()
    if sha(raw) != expected_sha or (expected_size is not None and len(raw) != expected_size):
        raise ValueError(f"frozen input changed: {relative}")
    return raw


def read_ref(ref, relative, manifest=None):
    name = relative.as_posix()
    if ref["path"] != name or type(ref["bytes"]) is not int or ref["bytes"] < 1:
        raise ValueError(f"unexpected artifact reference: {relative}")
    if manifest is not None:
        record = manifest[name]
        if (record["sha256"] != ref["sha256"] or record["bytes"] != ref["bytes"]
                or record["local_only"] is not True or record["non_upload"] is not True):
            raise ValueError(f"raw manifest differs: {relative}")
    return checked_bytes(relative, ref["sha256"], ref["bytes"])


def claim(relative, payload):
    path = workspace_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(payload)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(data)


def save_asset(relative, data):
    path = workspace_path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f"{path.name}.pending-{uuid.uuid4().hex}")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return dict(path=relative.as_posix(), sha256=sha(data), bytes=len(data))


def checked_state(state, expected_sha=None):
    if state.dtype != np.float64 or state.shape != (50,) or not np.isfinite(state).all():
        raise ValueError("expected a finite 50D float64 frozen state")
    if expected_sha is not None and sha(state.tobytes()) != expected_sha:
        raise ValueError("frozen state bytes changed")
    return state


def stage_states(raw):
    diagnostic = json.loads(gzip.decompress(raw))
    if (diagnostic["status"] != "numerically_unresolved" or diagnostic["returned_point"] != "stage2"
            or diagnostic["method"] != "fixed_dp8" or diagnostic["stage1_numerical"]["passed"] is not True
            or diagnostic["final_numerical"]["passed"] is not False
            or "underflow encountered in exp" not in diagnostic["final_numerical"]["error"]):
        raise ValueError("the saved stage1/stage2 diagnostic identity changed")
    states = {stage: checked_state(np.array(diagnostic[stage]["best"]["state"], dtype=np.float64), digest)
              for stage, digest in STAGE_STATE_HASHES.items()}
    if any(sha(state[20:30].tobytes()) != STAGE_P0_HASH for state in states.values()):
        raise ValueError("the saved stage1/stage2 endpoint anchor changed")
    if states["stage1"][:30].tobytes() != states["stage2"][:30].tobytes():
        raise ValueError("the saved stage1/stage2 observations changed")
    return states


def load_cached(raw, state, integrals):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        keys = [key for key in archive.files if key.startswith(("main_", "fine_"))]
        arrays = {key: archive[key] for key in ("times", "initial_state", *keys)}
    if arrays["initial_state"].dtype != np.float64 or arrays["initial_state"].tobytes() != state.tobytes():
        raise ValueError("cached trajectory has a different initial state")
    if not np.array_equal(arrays["times"], np.linspace(0., 15., 1501)):
        raise ValueError("cached trajectory uses a different comparison grid")
    for side, step, stride in (("main", .01, 1), ("fine", .005, 2)):
        leg = integrals[side]
        path = arrays[f"{side}_path"]
        last = checked_state(arrays[f"{side}_last_state"])
        accepted = leg["accepted_steps"]
        finite_rows = accepted // stride + 1
        if (leg["step"] != step or leg["target_time"] != 15. or not 0 <= accepted <= 1500 * stride
                or leg["last_accepted_time"] != accepted * step
                or leg["completed"] != (accepted == 1500 * stride)
                or leg["actual_rhs_calls"] != 12 * accepted + leg["interrupted_step_rhs_calls"]
                or path.dtype != np.float64 or path.shape != (1501, 50)
                or path[0].tobytes() != state.tobytes() or not np.isfinite(path[:finite_rows]).all()
                or not np.isnan(path[finite_rows:]).all()
                or (accepted % stride == 0 and last.tobytes() != path[finite_rows - 1].tobytes())):
            raise ValueError(f"cached {side} trajectory and receipt disagree")
        if not leg["completed"]:
            checked_state(arrays[f"{side}_failure_state"])
    return arrays


def load_inputs():
    execution = json.loads(checked_bytes(PARENT_EXECUTION, PARENT_HASHES[PARENT_EXECUTION]))
    manifest = json.loads(checked_bytes(PARENT_MANIFEST, PARENT_HASHES[PARENT_MANIFEST]))
    rows = json.loads(checked_bytes(PARENT_ROWS, PARENT_HASHES[PARENT_ROWS]))
    checked_bytes(PARENT_DIGEST, PARENT_HASHES[PARENT_DIGEST])
    row_index = {f"{row['mode']}_{row['index']:03d}": row for row in rows}
    if (execution["status"] != "COMPLETED" or execution["phase"] != "full" or execution["method"] != "fixed_dp8"
            or manifest["integrity_status"] != "verified"
            or manifest["full_execution_sha256"] != PARENT_HASHES[PARENT_EXECUTION]):
        raise ValueError("closed parent evidence is not verified")
    raw_assets = {item["path"]: item for item in manifest["files"]}
    coefficients = b"".join(np.asarray(value, dtype="<f8").tobytes() for value in (DOP853.A, DOP853.B))
    if sha(coefficients) != FIXED_DP8_COEFFICIENTS_SHA256:
        raise ValueError("fixed DP8 coefficients changed")

    model_refs = execution["models"]
    models = {}
    for mode in ("static", "modulated"):
        ref = model_refs[mode]
        path = PARENT_RAW / "pilot" / f"{mode}_model.json"
        read_ref(ref, path, raw_assets)
        model = GFTDKuramotoModel.load(workspace_path(path))
        read_ref(ref, path, raw_assets)
        if (model.fit_mode != mode or model.n_nodes != 10 or model.inference_config["method"] != "fixed_dp8"
                or sha(json_bytes(model.inference_config)) != ref["config_sha256"]):
            raise ValueError(f"frozen model configuration changed: {mode}")
        models[mode] = model.functional

    cases, records = {}, {}
    for (name, stage), target in zip(CASES, TARGET_SAMPLES, strict=True):
        mode, index = name.split("_")
        phase = "pilot" if name == "static_000" else "full"
        init_refs = execution["reused_pilot"] if phase == "pilot" else execution["tasks"]["initialize"]
        init_ref = init_refs[name]
        row = row_index[name]
        if row["candidate"]["evidence"] != init_ref:
            raise ValueError(f"parent row and initialization receipt disagree: {name}")
        init_path = PARENT_OUTPUT / "candidate" / phase / "initialize" / f"{name}.json"
        init = json.loads(read_ref(init_ref, init_path))
        if (init["mode"], init["index"], init["phase"], init["event_id"], init["label"]) != (
                mode, int(index), phase, row["event_id"], row["label"]):
            raise ValueError(f"frozen initialization identity changed: {name}")
        case_id = f"{name}_{stage}" if stage else name
        record = dict(mode=mode, event_id=init["event_id"], label=init["label"], stage=stage or init["stage"],
                      target_samples=target, target_time=target / 100,
                      initialization=init_ref, state=None, diagnostics=None, future=None, arrays=None)
        arrays, integrals = None, None
        if stage is None:
            if init["status"] != "completed" or init["passed"] is not True or init["p0_exact"] is not True:
                raise ValueError(f"frozen initialization did not qualify: {name}")
            record["state"] = init["state"]
            state_path = PARENT_RAW / phase / "states" / f"{name}.npz"
            with np.load(io.BytesIO(read_ref(init["state"], state_path, raw_assets)), allow_pickle=False) as archive:
                if archive.files != ["state"]:
                    raise ValueError(f"unexpected persisted state keys: {name}")
                state = checked_state(archive["state"], init["state_sha256"])
            record["future"] = execution["tasks"]["future"][name]
            if row["continuation"]["evidence"] != record["future"]:
                raise ValueError(f"parent row and continuation receipt disagree: {name}")
            future_path = PARENT_OUTPUT / "candidate/full/future" / f"{name}.json"
            future = json.loads(read_ref(record["future"], future_path))
            if (future["phase"], future["mode"], future["index"], future["event_id"]) != (
                    "full", mode, int(index), init["event_id"]):
                raise ValueError(f"frozen continuation identity changed: {name}")
            record["arrays"], integrals = future["arrays"], future["integrals"]
            cache_path = PARENT_RAW / "full/trajectories" / f"{name}.npz"
            arrays = load_cached(read_ref(record["arrays"], cache_path, raw_assets), state, integrals)
        else:
            if (init["status"] != "numerically_unresolved" or init["stage"] != "stage2" or init["passed"] is not False
                    or init["state"] is not None or name in execution["tasks"]["future"]):
                raise ValueError("stage diagnostic cannot stand in for a returned endpoint")
            record["diagnostics"] = init["diagnostics"]
            diag_path = PARENT_RAW / phase / "diagnostics" / f"{name}.json.gz"
            state = stage_states(read_ref(record["diagnostics"], diag_path, raw_assets))[stage]
        cases[case_id] = FrozenCase(state, arrays, integrals)
        records[case_id] = dict(record, state_sha256=sha(state.tobytes()), p0_sha256=sha(state[20:30].tobytes()))
    inventory = dict(schema=f"{RUN}_inputs_v1", parents={p.as_posix(): h for p, h in PARENT_HASHES.items()},
                     fixed_dp8_coefficients_sha256=FIXED_DP8_COEFFICIENTS_SHA256, models=model_refs, cases=records)
    return models, cases, inventory


class RhsLimit(RuntimeError):
    pass


class AuditedRhs:
    def __init__(self, functional, limit):
        self.functional = functional
        self.limit = limit
        self.calls = 0
        self.operation = None
        self.step = None
        self.substage = None
        self.by_operation = {}
        self.trial = []
        self.failure = None

    def __call__(self, t, state):
        evaluated = self.calls < self.limit
        self.trial.append((float(t), np.asarray(state).copy(), self.operation, evaluated))
        try:
            if not evaluated:
                raise RhsLimit("registered RHS call limit reached")
            self.calls += 1
            self.by_operation[self.operation] = self.by_operation.get(self.operation, 0) + 1
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return self.functional.rhs(state)
        except (RhsLimit, *NUMERICAL_ERRORS) as error:
            self.failure = dict(time=float(t), step=self.step, substage=self.substage, operation=self.operation,
                                error=repr(error), evaluated=evaluated)
            raise


def integrate(functional, state, samples, method, *, rhs_limit=None):
    if type(samples) is not int or samples < 1:
        raise ValueError("positive integer count of 0.01 s output intervals required")
    duration = samples / 100
    grid = np.linspace(0., duration, samples + 1)
    path = np.full((samples + 1, state.size), np.nan)
    path[0] = state
    accepted_times, accepted_states = [0.], [state.copy()]
    next_grid, attempted, dense_count = 1, 0, 0
    fixed = FIXED_STEPS.get(method)
    n_steps = samples * round(.01 / fixed) if fixed is not None else None
    limit = rhs_limit if rhs_limit is not None else (12 * n_steps if fixed is not None else 240 * samples)
    adapter = AuditedRhs(functional, limit)
    instance = None
    error, status = None, "completed"
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        if fixed is not None:
            stride = round(.01 / fixed)
            adapter.operation = "fixed_dp8"

            def field(value):
                adapter.substage += 1
                return adapter((attempted - 1) * fixed + fixed * DOP853.C[adapter.substage - 1], value)

            for attempted in range(1, n_steps + 1):
                adapter.trial.clear()
                adapter.step, adapter.substage = attempted, 0
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    value = dop853_step(field, accepted_states[-1], fixed)
                if adapter.substage != 12:
                    raise RuntimeError("fixed DP8 step must use twelve RHS calls")
                accepted_times.append(attempted * fixed)
                accepted_states.append(value)
                if attempted % stride == 0:
                    path[next_grid] = value
                    next_grid += 1
        else:
            solver, rtol, atol = REFERENCES[method]
            adapter.operation = "solver_setup"
            instance = solver(adapter, 0., state, duration, rtol=rtol, atol=atol, max_step=.005)
            while instance.status == "running":
                attempted += 1
                adapter.trial.clear()
                adapter.step, adapter.operation = attempted, "solver_step"
                message = instance.step()
                if instance.status == "failed":
                    raise FloatingPointError(message)
                functional.split_state(instance.y)
                t = float(instance.t)
                accepted_times.append(t)
                accepted_states.append(instance.y.copy())
                if next_grid < len(grid) and grid[next_grid] <= t:
                    adapter.operation = "dense_output"
                    dense_count += 1
                    dense = instance.dense_output()
                    while next_grid < len(grid) and grid[next_grid] <= t:
                        value = dense(grid[next_grid])
                        functional.split_state(value)
                        path[next_grid] = value
                        next_grid += 1
    except (RhsLimit, *NUMERICAL_ERRORS) as exc:
        error = repr(exc)
        status = "rhs_limit" if isinstance(exc, RhsLimit) else (
            "rhs_failure" if adapter.failure is not None else "solver_failure" if fixed is None else "step_assembly_failure")
    if error is None and (next_grid != len(grid) or abs(accepted_times[-1] - duration) > 1e-14):
        status, error = "incomplete", "finite common output grid did not reach the target"
    failure = adapter.failure
    if failure is None and error is not None:
        failure = dict(time=None, step=adapter.step, substage=None, operation=adapter.operation, error=error,
                       evaluated=None)
    record = dict(method=method, rhs_version=RHS_VERSION, status=status, completed=status == "completed",
                  target_time=duration, step=fixed, last_accepted_time=accepted_times[-1],
                  last_output_time=float(grid[next_grid - 1]), output_rows=next_grid,
                  external_step_attempts=attempted, accepted_steps=len(accepted_times) - 1,
                  internal_rejected_steps=None, dense_output_attempts=dense_count, actual_rhs_calls=adapter.calls,
                  rhs_calls_by_operation=adapter.by_operation, rhs_limit=limit, error=error, failure=failure,
                  wall_seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu_started)
    arrays = dict(grid_times=grid, grid_states=path, accepted_times=np.asarray(accepted_times),
                  accepted_states=np.asarray(accepted_states))
    if error is not None and adapter.trial:
        times, states, operations, evaluated = zip(*adapter.trial, strict=True)
        arrays.update(terminal_trial_times=np.array(times), terminal_trial_states=np.array(states),
                      terminal_trial_operations=np.array(operations), terminal_trial_rhs_evaluated=np.array(evaluated))
        if adapter.failure is not None:
            arrays["failure_state"] = arrays["terminal_trial_states"][-1]
    if fixed is not None:
        record["interrupted_step_rhs_calls"] = adapter.calls - 12 * record["accepted_steps"]
    else:
        record.update(solver=solver.__name__, rtol=rtol, atol=atol, max_step=.005,
                      **{f"solver_{key}": None if instance is None else getattr(instance, key)
                         for key in ("nfev", "njev", "nlu")})
    return record, arrays


PAIR_LIMITS = (1e-6, 1e-5, 1e-5)
FLOOR = 1e-12
COMPONENTS = ("theta", "omega", "p", "a", "r")


def metrics(values):
    peak = float(np.max(np.abs(values)))
    if not np.isfinite(peak):
        return None
    return dict(rms=peak * float(np.sqrt(np.mean((values / peak)**2))) if peak else 0., peak=peak)


def differences(mass, left, right):
    with np.errstate(over="ignore", invalid="ignore"):
        obs = metrics(observations(left, mass) - observations(right, mass))
        delta = left - right
        scale = 1. + np.maximum(np.abs(left), np.abs(right))
        relative = np.abs(delta) / scale
        overflow = ~np.isfinite(relative)
        relative[overflow] = np.abs(left[overflow] / scale[overflow] - right[overflow] / scale[overflow])
    if obs is None or not np.isfinite(relative).all():
        return None
    return dict(observed=obs, state_relative_max=float(np.max(relative)),
                components={name: metrics(part) for name, part in
                            zip(COMPONENTS, np.split(delta, 5, axis=1), strict=True)})


def agreed(delta):
    if delta is None:
        return False
    values = (delta["observed"]["rms"], delta["observed"]["peak"], delta["state_relative_max"])
    return all(value <= limit for value, limit in zip(values, PAIR_LIMITS, strict=True))


def available(path):
    bad = np.flatnonzero(~np.isfinite(path).all(axis=1))
    return int(bad[0]) if len(bad) else len(path)


def reference_prefix(mass, paths):
    common = min(available(path) for path in paths.values())
    keep = common
    for left, right in combinations(paths.values(), 2):
        x, y = left[:common], right[:common]
        with np.errstate(over="ignore", invalid="ignore"):
            delta = np.abs(observations(x, mass) - observations(y, mass)).reshape(common, -1)
            scale = 1. + np.maximum(np.abs(x), np.abs(y))
            relative = np.abs(x - y) / scale
            overflow = ~np.isfinite(relative)
            relative[overflow] = np.abs(x[overflow] / scale[overflow] - y[overflow] / scale[overflow])
        bad = np.flatnonzero(~((np.max(delta, axis=1) <= PAIR_LIMITS[1])
                              & (np.max(relative, axis=1) <= PAIR_LIMITS[2])))
        if len(bad):
            keep = min(keep, int(bad[0]))
        # Peak-bounded values can be squared safely even if the later trajectory diverged.
        rms = np.sqrt(np.cumsum(np.mean(delta[:keep]**2, axis=1)) / np.arange(1, keep + 1))
        bad = np.flatnonzero(rms > PAIR_LIMITS[0])
        if len(bad):
            keep = int(bad[0])
    return common, keep


def pairs_at(mass, paths, n):
    return {f"{left}:{right}": differences(mass, paths[left][:n], paths[right][:n])
            if min(available(paths[left]), available(paths[right])) >= n else None
            for left, right in combinations(paths, 2)}


def fixed_precision(mass, paths, reference, reference_rows):
    if set(FIXED_STEPS) - paths.keys():
        return None
    n = min(reference_rows, *(available(paths[key]) for key in FIXED_STEPS))
    deltas = {name: differences(mass, paths[name][:n], reference[:n]) for name in FIXED_STEPS}
    errors = [None if delta is None else delta["observed"]["rms"] for delta in deltas.values()]
    ratios = [None if x is None or y is None or min(x, y) <= FLOOR else x / y
              for x, y in zip(errors[:-1], errors[1:], strict=True)]
    decreasing = (None not in errors and all(x > y for x, y in zip(errors[:-1], errors[1:], strict=True))
                  and errors[-1] <= errors[1] / 4.)
    matching_fine = all(delta is not None and delta["observed"]["rms"] <= PAIR_LIMITS[0]
                        and delta["observed"]["peak"] <= PAIR_LIMITS[1]
                        for delta in (deltas["finer"], deltas["finest"]))
    return dict(end_time=(n - 1) / 100, deltas=deltas, adjacent_rms_ratios=ratios,
                floor_limited=[name for name, err in zip(FIXED_STEPS, errors, strict=True)
                               if err is not None and err <= FLOOR],
                decreasing=decreasing, finest_two_match_observations=matching_fine,
                evidence=n >= 11 and (decreasing or matching_fine))


def modulation_beta(functional):
    return functional.gates.s_edge * functional.k_edge * functional.gates.s_mod


def continuous_growth(functional, paths, n):
    active = modulation_beta(functional) > 0.
    if n < 11 or not active.any():
        return dict(window=None, applicable=bool(active.any()), evaluated=False, matching_windows=0)
    endpoints = np.array(functional.edges)[active]
    amplitudes = np.array([path[:n, 3 * functional.n_nodes:4 * functional.n_nodes] for path in paths.values()])
    A = np.max(np.abs(amplitudes), axis=2)
    node_terms = amplitudes * functional.gamma_amp
    q = np.max(node_terms[:, :, endpoints[:, 0]] + node_terms[:, :, endpoints[:, 1]], axis=2)
    increments = [np.diff(values, axis=1) >= -1e-8 for values in (A, q)]
    count, tested, first = 0, 0, None
    for begin in range(n - 5):
        for length in (5, 10, 20):
            end = begin + length
            if end >= n:
                break
            tested += 1
            if (not np.all(A[:, end] >= np.maximum(1., 2 * A[:, begin]))
                    or not np.all(q[:, end] - q[:, begin] >= 2.)):
                continue
            fractions = [values[:, begin:end].mean(axis=1) for values in increments]
            if not all(np.all(values >= .8) for values in fractions):
                continue
            agreement = pairs_at(functional.mass, {name: path[begin:end + 1] for name, path in paths.items()}, length + 1)
            if not all(agreed(delta) for delta in agreement.values()):
                continue
            count += 1
            if first is None:
                first = dict(start=begin / 100, end=end / 100, agreement=agreement,
                             methods={name: dict(A_start=float(A[i, begin]), A_end=float(A[i, end]),
                                                 q_start=float(q[i, begin]), q_end=float(q[i, end]),
                                                 A_increment_fraction=float(fractions[0][i]),
                                                 q_increment_fraction=float(fractions[1][i]))
                                      for i, name in enumerate(paths)})
    return dict(window=first, applicable=True, evaluated=True, windows_checked=tested, matching_windows=count)


def inspect_state(functional, state, counts, *, sample_budget=None):
    info, arrays = {}, {}

    def evaluate(name, operation):
        counts[name] = counts.get(name, 0) + 1
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                value = operation()
            if not np.isfinite(value).all():
                raise FloatingPointError("nonfinite diagnostic output")
            return value
        except NUMERICAL_ERRORS as error:
            info.setdefault("errors", {})[name] = repr(error)
            return None

    try:
        theta, omega, _, amplitude, _ = functional.split_state(state)
    except ValueError as error:
        return dict(state_error=repr(error), exponents={}), arrays
    endpoints = np.array(functional.edges)
    beta = modulation_beta(functional)
    active = beta > 0.
    psi = evaluate("psi", lambda: bregman_kuramoto_potential(functional.incidence.T @ theta, functional.eta_edge))
    arrays.update(state=state, beta=beta, active=active)
    info["amplitude_range"] = [float(amplitude.min()), float(amplitude.max())]
    if psi is not None:
        arrays["psi"] = psi
        product = evaluate("beta_psi", lambda: beta * psi)
        if product is not None:
            arrays["beta_psi"] = product
    exponents = {
        "endpoint_sum": evaluate("endpoint_sum", lambda: (
            (functional.gamma_amp * amplitude)[endpoints[:, 0]] + (functional.gamma_amp * amplitude)[endpoints[:, 1]])),
        "incidence_sum": evaluate("incidence_sum", lambda: np.abs(functional.incidence).T @ (functional.gamma_amp * amplitude)),
    }
    info["exponents"] = {}
    for name, values in exponents.items():
        if values is None:
            info["exponents"][name] = None
            continue
        arrays[name] = values
        violations = {"overflow": [], "underflow": []}
        log_tiny = np.log(np.finfo(np.float64).tiny)
        log_max = np.log(np.finfo(np.float64).max)
        for edge in np.flatnonzero((values <= log_tiny) | (values >= log_max)):
            counts["exp_range_check"] = counts.get("exp_range_check", 0) + 1
            try:
                with np.errstate(over="raise", under="raise", invalid="raise"):
                    np.exp(values[edge])
            except FloatingPointError as error:
                kind = "underflow" if values[edge] < 0. else "overflow"
                violations[kind].append(int(edge))
        info["exponents"][name] = {
            group: dict(min=float(values[mask].min()), max=float(values[mask].max()),
                        **{kind: [edge for edge in edges if mask[edge]] for kind, edges in violations.items()})
            if mask.any() else None for group, mask in (("active", active), ("inactive", ~active))}
    if sample_budget is not None:
        if sample_budget[0] >= 500:
            raise RuntimeError("registered single-point RHS limit reached")
        hamiltonian = evaluate("hamiltonian", lambda: functional.hamiltonian(state))
        gradients = evaluate("gradients", lambda: np.asarray(functional.gradients(state)))
        sample_budget[0] += 1
        rhs = evaluate("rhs", lambda: functional.rhs(state))
        info.update(hamiltonian=None if hamiltonian is None else float(hamiltonian), power=None,
                    rhs_peak=None if rhs is None else float(np.max(np.abs(rhs))))
        if rhs is not None:
            arrays["rhs"] = rhs
        if gradients is not None:
            arrays["amplitude_gradient"] = gradients[3]
        if gradients is not None and rhs is not None:
            def power():
                _, _, gp, ga, gr = gradients
                ra, rm = functional.dissipative_operators(state)
                dissipation = ((functional.damping * omega**2).sum() + (functional.rate_p * gp**2).sum()
                               + (ra * ga**2).sum() + gr @ rm @ gr)
                chain = gradients.ravel() @ rhs
                return np.array([chain, -dissipation, abs(chain + dissipation)])

            balance = evaluate("power_balance", power)
            if balance is not None:
                info["power"] = dict(zip(("chain", "expected", "absolute_residual"), balance.tolist(), strict=True))
    return info, arrays


def exponent_labels(exponents):
    labels = set()
    for forms in exponents.values():
        if forms is None:
            continue
        active, inactive = forms["active"], forms["inactive"]
        for kind in ("overflow", "underflow"):
            if active and active[kind]:
                labels.add(f"active_exp_{kind}")
            if inactive and inactive[kind]:
                labels.add("inactive_zero_exp_out_of_range")
    return sorted(labels)


def phase_geometry(functional, path):
    theta = path[:, :functional.n_nodes]
    relative = theta - (theta @ functional.mass / functional.mass.sum())[:, None]
    edges = theta @ functional.incidence
    turns = np.trunc((edges - edges[0]) / (2. * np.pi))
    crossings = [int(count) for count in np.abs(np.diff(turns, axis=0)).sum(axis=0)]
    return dict(max_coi_angle_span=float(np.max(np.ptp(relative, axis=1))), crossings=crossings), edges


def summarize_case(functional, case, samples, jobs, sample_budget):
    references, fixed, saved, diagnoses, counts = {}, {}, {}, {}, {}
    records = {method: record for method, record in jobs.items()}
    if case.cache is not None:
        for method in ("main", "fine"):
            fixed[method] = case.cache[f"{method}_path"][:samples + 1]
            records[method] = case.integrals[method]
    for method, record in jobs.items():
        if record["state_sha256"] != sha(case.state.tobytes()):
            raise ValueError("job initial state bytes differ from the frozen case")
        ref = record["arrays"]
        path = PRIVATE / "phase1/jobs" / record["case_id"] / f"{method}.npz"
        with np.load(io.BytesIO(read_ref(ref, path)), allow_pickle=False) as archive:
            data = {key: archive[key] for key in ("grid_states", "accepted_states")}
            if "failure_state" in archive:
                data["failure_state"] = archive["failure_state"]
        if any(data[key][0].tobytes() != case.state.tobytes() for key in ("grid_states", "accepted_states")):
            raise ValueError("job trajectory does not start at its frozen initial state")
        paths = references if method in REFERENCES else fixed
        paths[method] = data["grid_states"]
        if not record["completed"]:
            diagnoses[method] = (data["accepted_states"][-1], data.get("failure_state"),
                                 bool(record["failure"]["evaluated"] and record["status"] == "rhs_failure"))
    if case.cache is not None:
        for method in ("main", "fine"):
            if not records[method]["completed"]:
                diagnoses[method] = (case.cache[f"{method}_last_state"], case.cache[f"{method}_failure_state"], True)
    common, consistent = reference_prefix(functional.mass, references)
    horizons = sorted({100, 200, 500, 800, 1500, samples})
    paths = references | fixed
    pairs = lambda n: pairs_at(functional.mass, paths, n)
    reference_end = (consistent - 1) / 100
    end_by_method = {name: (available(path) - 1) / 100 for name, path in paths.items()}
    complete = consistent == samples + 1 and all(records[name]["completed"] for name in REFERENCES)
    summary = dict(reference=dict(common_end=(common - 1) / 100, agreement_end=reference_end, completed=complete,
                                  agreement_pairs=pairs_at(functional.mass, references, consistent)),
                   horizons={str(index / 100): pairs(index + 1) for index in horizons},
                   common_grid_end_by_method=end_by_method,
                   last_accepted_by_method={name: record["last_accepted_time"] for name, record in records.items()},
                   common_grid_end_by_pair={f"{left}:{right}": min(end_by_method[left], end_by_method[right])
                                            for left, right in combinations(paths, 2)},
                   fixed_precision=fixed_precision(functional.mass, fixed, references["radau_strict"], consistent),
                   growth=continuous_growth(functional, references, consistent), samples={}, terminal={})
    labels = set()
    if not complete:
        labels.add("reference_unresolved")
    if common > consistent:
        labels.add("reference_inconsistent")
    if summary["growth"]["window"] is not None:
        labels.add("continuous_growth_window")
    inactive_candidate = False
    for method, (last, failure, evaluated_failure) in diagnoses.items():
        record = records[method]
        terminal = {"status": record.get("status", "cached_rhs_failure")}
        labels.add(terminal["status"])
        for which, state in (("last_accepted", last), ("failure_state", failure)):
            if state is None:
                terminal[which] = None
                continue
            info, arrays = inspect_state(functional, state, counts)
            terminal[which] = info
            saved.update({f"terminal_{method}_{which}_{key}": value for key, value in arrays.items()})
            if which == "failure_state" and evaluated_failure:
                point_labels = exponent_labels(info["exponents"])
                terminal["failure_exponent_labels"] = point_labels
                labels.update(point_labels)
                inactive_candidate |= "inactive_zero_exp_out_of_range" in point_labels
        summary["terminal"][method] = terminal
    for method, path in references.items():
        summary["samples"][method] = {}
        indices = sorted({0, 10, 50, 100, samples // 2, samples, consistent - 1})
        for index in indices:
            if index >= available(path):
                continue
            info, arrays = inspect_state(functional, path[index], counts, sample_budget=sample_budget)
            info["inside_reference_agreement"] = index < consistent
            summary["samples"][method][str(index / 100)] = info
            saved.update({f"sample_{method}_{index}_{key}": value for key, value in arrays.items()})
    if not np.any(modulation_beta(functional) > 0.):
        summary["phase_geometry"] = {}
        for method, path in paths.items():
            n = min(available(path), consistent)
            summary["phase_geometry"][method], saved[f"geometry_{method}_unwrapped_edges"] = phase_geometry(
                functional, path[:n])
            summary["phase_geometry"][method]["end_time"] = (n - 1) / 100
    summary.update(labels=sorted(labels), algebra_and_sample_counts=counts,
                   inactive_zero_skip_candidate=inactive_candidate)
    return summary, saved


def save_arrays(relative, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return dict(save_asset(relative, buffer.getvalue()), local_only=True, non_upload=True)


def run_case_jobs(action, case_id, functional, state, samples, methods):
    runtime()
    output = OUTPUT / "phase1/jobs" / case_id
    private = PRIVATE / "phase1/jobs" / case_id
    results = {}
    for method in methods:
        started, cpu_started = time.perf_counter(), time.process_time()
        identity = dict(action_id=f"{action}:{case_id}:{method}", phase_action_id=action, case_id=case_id,
                        method=method, rhs_version=RHS_VERSION, state_sha256=sha(state.tobytes()))
        path = output / f"{method}.json"
        if workspace_path(path).exists():
            raise FileExistsError(path)
        claim(output / f"{method}_started.json", identity)
        try:
            record, arrays = integrate(functional, state, samples, method)
            record.update(identity)
            record["arrays"] = save_arrays(private / f"{method}.npz", arrays)
            record.update(worker_wall_seconds=time.perf_counter() - started,
                          worker_cpu_seconds=time.process_time() - cpu_started)
            ref = save_asset(path, json_bytes(record))
        except BaseException as error:
            save_asset(path, json_bytes(dict(identity, status="interrupted", error=repr(error),
                                             actual_rhs_calls=None, worker_wall_seconds=time.perf_counter() - started,
                                             worker_cpu_seconds=time.process_time() - cpu_started)))
            raise
        results[method] = (record, ref)
    return case_id, results


def phase_plan(inventory):
    four_grids = {"modulated_008", "modulated_124", "modulated_138", "modulated_204", "static_027",
                  "modulated_268_stage2"}
    targets = {name: row["target_samples"] for name, row in inventory["cases"].items()}
    references = {name: tuple(REFERENCES) for name in targets}
    fixed = {name: tuple(FIXED_STEPS) if name == "modulated_268_stage2" else ("main", "fine")
             if name == "modulated_268_stage1" else ("finer", "finest")
             for name in targets if name in four_grids or name == "modulated_268_stage1"}
    ref_legs, fixed_legs = len(references) * len(REFERENCES), sum(map(len, fixed.values()))
    ref_calls = sum(targets.values()) * 240 * len(REFERENCES)
    fixed_calls = sum(12 * targets[name] * round(.01 / FIXED_STEPS[method])
                      for name, methods in fixed.items() for method in methods)
    if (ref_legs, fixed_legs, ref_calls, fixed_calls) != (30, 16, 4191840, 383328):
        raise ValueError("phase1 work plan differs from the frozen protocol")
    return references, fixed


def runtime():
    threads = {name: os.environ.get(name) for name in THREADS}
    if set(threads.values()) != {"1"}:
        raise ValueError("require single-thread BLAS, OMP and MKL in each worker")
    return dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, threads=threads)


def execute(release_path):
    expected = OUTPUT / "phase1_release.json"
    if release_path != expected:
        raise ValueError(f"phase1 release must be {expected.as_posix()}")
    raw = workspace_path(expected).read_bytes()
    release = json.loads(raw)
    action = release["action_id"]
    if (release.get("released") is not True or release.get("phase") != "phase1"
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", action)
            or release["protocol_sha256"] != PROTOCOL_SHA256 or release["runtime"] != runtime()
            or type(release["workers"]) is not int or release["workers"] < 1):
        raise PermissionError("frozen root phase1 release is required")
    checked_bytes(PROTOCOL, PROTOCOL_SHA256)
    adoption = json.loads(checked_bytes(OUTPUT / "design_adoption.json", release["design_adoption_sha256"]))
    review_path = Path("reports") / RUN / "design_review.md"
    if (adoption["run_id"] != RUN or adoption["protocol_sha256"] != PROTOCOL_SHA256
            or adoption["review_reference"] != review_path.as_posix()
            or adoption["review_sha256"] != release["design_review_sha256"]):
        raise ValueError("release is not bound to the adopted protocol and independent design review")
    checked_bytes(review_path, release["design_review_sha256"])
    if set(release["source_sha256"]) != set(SOURCE_PATHS):
        raise ValueError("phase1 source identity is incomplete")
    for path, digest in release["source_sha256"].items():
        checked_bytes(path, digest)
    prepared = checked_bytes(OUTPUT / "prepare.json", release["prepare_sha256"])
    models, cases, inventory = load_inputs()
    if prepared != json_bytes(inventory):
        raise ValueError("verified parent inputs differ from the prepared manifest")
    plans = phase_plan(inventory)
    output = OUTPUT / "phase1"
    identity = dict(action_id=action, release_sha256=sha(raw), pid=os.getpid())
    receipt = dict(identity, status="RUNNING", runtime=release["runtime"], workers=release["workers"],
                   tasks={}, summaries={}, prepared_sha256=release["prepare_sha256"],
                   protocol_sha256=PROTOCOL_SHA256, design_adoption_sha256=release["design_adoption_sha256"],
                   design_review_sha256=release["design_review_sha256"], source_sha256=release["source_sha256"])
    claim(output / "started.json", identity)
    started = time.perf_counter()
    persist = lambda: save_asset(output / "execution.json", json_bytes(receipt))
    records = {name: {} for name in cases}
    try:
        persist()
        with ProcessPoolExecutor(max_workers=release["workers"]) as pool:
            for plan in plans:
                pending = []
                for name, methods in plan.items():
                    row = inventory["cases"][name]
                    pending.append(pool.submit(run_case_jobs, action, name, models[row["mode"]], cases[name].state,
                                               row["target_samples"], methods))
                for task in pending:
                    name, results = task.result()
                    for method, (record, ref) in results.items():
                        records[name][method] = record
                        receipt["tasks"][record["action_id"]] = ref
                    persist()
        sample_budget = [0]
        for name, case in cases.items():
            row = inventory["cases"][name]
            summary, arrays = summarize_case(models[row["mode"]], case, row["target_samples"], records[name],
                                             sample_budget)
            summary["arrays"] = save_arrays(PRIVATE / "phase1/summaries" / f"{name}.npz", arrays)
            receipt["summaries"][name] = save_asset(output / "summaries" / f"{name}.json", json_bytes(summary))
        if sample_budget[0] > 210 or len(receipt["tasks"]) != 46:
            raise RuntimeError("phase1 sampling or integration count differs from plan")
        all_legs = [record for case in records.values() for record in case.values()]
        receipt.update(status="COMPLETED", sample_rhs_calls=sample_budget[0],
                       actual_integration_rhs_calls=sum(record["actual_rhs_calls"] for record in all_legs),
                       worker_cpu_seconds=sum(record["worker_cpu_seconds"] for record in all_legs),
                       worker_wall_seconds=sum(record["worker_wall_seconds"] for record in all_legs))
    except BaseException as error:
        receipt.update(status="INTERRUPTED", error=repr(error))
        raise
    finally:
        receipt["host_wall_seconds"] = time.perf_counter() - started
        persist()
    print(f"Completed phase1 with {len(receipt['tasks'])} integrations. {(output / 'execution.json').as_posix()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.release is not None:
        execute(args.release)
        return
    checked_bytes(PROTOCOL, PROTOCOL_SHA256)
    runtime()
    _, _, inventory = load_inputs()
    phase_plan(inventory)
    path = OUTPUT / "prepare.json"
    digest = claim(path, inventory)
    print(f"Verified {len(inventory['cases'])} frozen states. {path.as_posix()} {digest}")


if __name__ == "__main__":
    main()
