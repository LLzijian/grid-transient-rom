"""Loop0067 matched-interval timing benchmark.

The benchmark is deliberately separate from dataset generation and training.
It selects two fixed TRAIN events, replays the original ANDES case, and
compares the existing S and 0058 JointROM checkpoints over the same
post-record-start interval.  ``--smoke`` invokes one replay and ``--run``
invokes the preregistered protocol; no replay is performed on import.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# These must be set before importing NumPy/Torch/ANDES in the benchmark
# process.  They are intentionally process-local and do not touch the shared
# environment or the project's original virtual environment.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
VENDORED_DEPS = ROOT / ".loop-work/research_deps_0067"
if str(VENDORED_DEPS) not in sys.path:
    sys.path.insert(0, str(VENDORED_DEPS))
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import andes  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluate_long_latent import _construct  # noqa: E402
from train_joint_latent import BASE, NETWORK, JointROM, tensor  # noqa: E402
from kiki.data.private_dataset import canonical_arrays, los_index  # noqa: E402
from kiki.models.private_pm import MechanicalROM  # noqa: E402
from kiki.physics import coi_project  # noqa: E402

DT = 0.01
FOM_STEP = 0.005
MAX_INTERVAL_S = 8.0
SCALES = np.array([0.47675314588683515, 1.0033667249602924])
CASE_RELATIVE = "ieee39/ieee39_full.xlsx"
SOURCE_DIRS = (ROOT / "private_data/ieee39_postfault", ROOT / "private_data/ieee39_expansion")
MODEL_DIRS = {
    "S": ROOT / "private_data/task2_long_0060_5s",
    "0058": ROOT / "private_data/task2_joint_0058",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_value(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _row_is_eligible(row: dict[str, str]) -> bool:
    return (row.get("label", "") in {"stable", "unstable"}
            and float(row.get("trajectory_end_s", "0")) > 0.1
            and row.get("solver_success", "False").lower() == "true"
            and int(row.get("valid_length", "0")) > 11)


def select_events(roots: Iterable[Path] = SOURCE_DIRS) -> dict[str, dict[str, Any]]:
    """Select lexicographically first eligible stable and unstable TRAIN IDs."""
    candidates: dict[str, list[dict[str, Any]]] = {"stable": [], "unstable": []}
    for root in roots:
        splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
        train_ids = set(splits["train"])
        with (root / "dataset_index.csv").open(encoding="utf-8", newline="") as stream:
            rows = {row["event_id"]: row for row in csv.DictReader(stream)}
        for event_id in train_ids:
            row = rows[event_id]
            if _row_is_eligible(row):
                candidates[row["label"]].append({"root": root, "event_id": event_id, "row": row})
    selected = {}
    for label, values in candidates.items():
        if not values:
            raise ValueError(f"no eligible TRAIN event for {label}")
        selected[label] = min(values, key=lambda item: (item["event_id"], str(item["root"])))
    return selected


def matched_interval(event: dict[str, Any], max_seconds: float = MAX_INTERVAL_S) -> dict[str, Any]:
    """Return the exact record-relative interval used by both engines."""
    x = np.asarray(event["x"])
    available_steps = max(0, len(x) - 11)
    requested_steps = min(round(max_seconds / DT), available_steps)
    return {
        "record_start_s": 0.0,
        "post100_start_s": 0.1,
        "steps": int(requested_steps),
        "duration_s": float(requested_steps * DT),
        "end_s": float(0.1 + requested_steps * DT),
        "true_remaining_steps": int(available_steps),
    }


def load_train_event(candidate: dict[str, Any], base: MechanicalROM) -> dict[str, Any]:
    """Read one explicit TRAIN event and its equilibrium metadata only."""
    root = Path(candidate["root"])
    event_id = candidate["event_id"]
    with h5py.File(root / "dataset.h5", "r") as file:
        eq = {key: file["equilibrium"][key][:] for key in ("delta", "omega", "Pm", "Pe")}
        raw_group = file["events"][event_id]
        raw = {key: raw_group[key][:] for key in ("t", "delta", "omega", "Pm", "Pe")}
        inertia = file["inertia"][:]
        omega_s = float(file.attrs["omega_s"])
        stored_signature = str(file.attrs["postfault_signature"])
        stored_case_sha = str(file.attrs["case_sha256"])
    converted = canonical_arrays(raw, eq, inertia, omega_s)
    x = np.column_stack((converted["delta_sync"], converted["omega_sync"], converted["Pm_raw"] - base.pm_star))
    return {
        "id": event_id, "root": root, "row": candidate["row"], "x": x,
        "raw": raw, "equilibrium": eq, "inertia": inertia,
        "omega_s": omega_s, "postfault_signature": stored_signature,
        "case_sha256": stored_case_sha,
    }


def verify_case_and_event(event: dict[str, Any], case: Path) -> None:
    """Verify frozen case identity and the TRAIN event's stored provenance."""
    case_sha = sha256(case)
    if event["case_sha256"] != case_sha:
        raise AssertionError(f"case SHA mismatch for {event['id']}")
    if event["row"].get("postfault_signature") != event["postfault_signature"]:
        raise AssertionError(f"postfault signature mismatch for {event['id']}")
    if not np.isfinite(event["x"]).all() or len(event["x"]) <= 11:
        raise ValueError(f"invalid TRAIN event {event['id']}")


def normalized_coi_rmse(pred: np.ndarray, reference: np.ndarray, mass: np.ndarray) -> dict[str, float]:
    """Registered normalized COI delta/omega discretization errors."""
    pred = np.asarray(pred, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(pred) != len(reference):
        raise ValueError("matched-interval trajectory lengths differ")
    count = len(pred)
    if count == 0:
        raise ValueError("empty trajectory")
    error = pred[:count, :20] - reference[:count, :20]
    delta = coi_project(error[:, :10], mass) / SCALES[0]
    omega = coi_project(error[:, 10:20], mass) / SCALES[1]
    return {
        "delta": float(np.sqrt(np.mean(delta * delta))),
        "omega": float(np.sqrt(np.mean(omega * omega))),
        "combined": float(np.sqrt(0.5 * np.mean(delta * delta + omega * omega))),
    }


def common_frequency_rmse(pred: np.ndarray, reference: np.ndarray, mass: np.ndarray) -> float:
    """Common synchronous frequency error in rad/s."""
    count = min(len(pred), len(reference))
    err = np.asarray(pred[:count, 10:20]) - np.asarray(reference[:count, 10:20])
    common = (err * mass).sum(axis=1) / mass.sum()
    return float(np.sqrt(np.mean(common * common)))


def interpolate_solver(times: np.ndarray, values: np.ndarray, grid: np.ndarray, step: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpolate actual solver samples to the record-relative grid."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if len(times) == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("solver times must be strictly increasing")
    if grid[0] < times[0] - 1e-8 or grid[-1] > times[-1] + 1e-8:
        raise ValueError("grid outside solver coverage")
    if times[-1] > grid[-1] + step + 1e-8:
        raise ValueError("solver endpoint overshoots by more than one step")
    clipped = np.clip(grid, times[0], times[-1])
    result = np.column_stack([np.interp(clipped, times, values[:, column]) for column in range(values.shape[1])])
    boundary = {
        "grid_start_s": float(grid[0]), "grid_end_s": float(grid[-1]),
        "solver_start_s": float(times[0]), "solver_end_s": float(times[-1]),
        "start_gap_s": float(grid[0] - times[0]), "end_overshoot_s": float(times[-1] - grid[-1]),
        "within_one_step": bool(times[-1] >= grid[-1] - 1e-8 and times[-1] <= grid[-1] + step + 1e-8),
    }
    return result, boundary


def rom_rollout_timed(model: JointROM, history: np.ndarray, steps: int, dt: float = FOM_STEP) -> tuple[np.ndarray, dict[str, float]]:
    """One causal 11-state encoding followed by autonomous RK4."""
    history_tensor = tensor(np.asarray(history)[None, ...])
    torch.set_num_threads(1)
    with torch.no_grad():
        encode_start = time.perf_counter()
        q = model.encode(history_tensor)
        encode_seconds = time.perf_counter() - encode_start
        state = torch.cat((history_tensor[:, -1], q), dim=-1)
        result = [state]
        ode_start = time.perf_counter()
        for _ in range(int(steps)):
            state = model.rk4(state, dt)
            result.append(state)
        ode_seconds = time.perf_counter() - ode_start
    values = torch.stack(result, dim=1)[0].cpu().numpy()
    return values, {"gru_seconds": encode_seconds, "integration_seconds": ode_seconds,
                    "end_to_end_seconds": encode_seconds + ode_seconds}


def median_repetitions(records: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, float]:
    """Summarize exactly the measured repetitions by median, never minimum."""
    return {field: float(np.median([float(row[field]) for row in records])) for field in fields}


def _build_system(event: dict[str, Any], config: dict[str, Any], workdir: Path):
    case_start = time.perf_counter()
    case = Path(andes.get_case(CASE_RELATIVE))
    system = andes.load(str(case), setup=False, no_output=True, default_config=True,
                        pycode_path=str(workdir / "andes_pycode"),
                        output_path=str(workdir / "andes_output"),
                        no_undill=True, autogen_stale=False)
    for model in system.models.values():
        if model.group == "TimedEvent" and model.n:
            model.u.v[:] = [0] * model.n
    if config["disable_pss"]:
        system.IEEEST.u.v[:] = [0] * system.IEEEST.n
    row = event["row"]
    clearing = float(row["clearing_time_s"])
    uid = system.Line.idx.v.index(row["fault_line"])
    endpoint = system.Line.bus1.v[uid] if float(row["fault_location"]) == 0.0 else system.Line.bus2.v[uid]
    if int(endpoint) != int(row["fault_bus"]):
        raise ValueError("fault_endpoint_mismatch")
    system.add("Fault", {"idx": "timing_fault", "bus": int(row["fault_bus"]),
                          "tf": float(config["fault_start_s"]), "tc": clearing,
                          "rf": 0., "xf": float(row["fault_xf"])})
    if (workdir / 'andes_pycode' / '__init__.py').is_file():
        system.codegen._load_calls()
    if not system.with_calls:
        system.prepare(quick=True, nomp=True)
    system.setup()
    case_code_seconds = time.perf_counter() - case_start
    pflow_start = time.perf_counter()
    if not system.PFlow.run():
        raise ValueError("powerflow_failure")
    pflow_seconds = time.perf_counter() - pflow_start
    tds = system.TDS
    tds.config.tstep = float(config["integration_step_s"])
    tds.config.tol = float(config["newton_tolerance"])
    tds.config.max_iter = int(config["newton_max_iterations"])
    tds.config.fixt = 1
    tds.config.shrinkt = 0
    tds.config.criteria = 0
    tds.config.no_tqdm = 1
    return system, tds, case, {"case_code_seconds": case_code_seconds,
                               "pflow_seconds": pflow_seconds}


def replay_fom(event: dict[str, Any], config: dict[str, Any], workdir: Path,
               solver_step: float = FOM_STEP) -> dict[str, Any]:
    """Replay one event over the matched interval with step-level timing.

    This function is intentionally not called by tests or import.  It keeps
    the generator's restoration/fault checks but does not claim a 15-second
    stable outcome for this timing-only interval.
    """
    system, tds, case, init_timing = _build_system(event, config, workdir)
    row = event["row"]
    clearing = float(row["clearing_time_s"])
    record_delay = float(row["record_start_delay_s"])
    interval = matched_interval(event)
    record_start_absolute = clearing + record_delay
    post100_absolute = record_start_absolute + interval["post100_start_s"]
    end_absolute = post100_absolute + interval["duration_s"]
    tds.config.tstep = float(solver_step)
    tds.config.tf = end_absolute + solver_step
    setup_done = time.perf_counter()
    tds.init()
    pflow_tds_seconds = time.perf_counter() - setup_done
    if not tds.test_ok or not np.isfinite(system.dae.xy).all():
        raise ValueError("dynamic_initialization_failure")
    equilibrium = event["equilibrium"]
    for name, actual in (("delta", system.GENROU.delta.v), ("omega", system.GENROU.omega.v),
                         ("Pm", system.GENROU.tm.v), ("Pe", system.GENROU.Pe.v)):
        if not np.allclose(actual, equilibrium[name], rtol=0., atol=1e-8):
            raise ValueError(f"equilibrium_mismatch_{name}")
    stored_signature = event["postfault_signature"]
    inertia = system.GENROU.M.v.copy()
    status = {"fault_applied": False, "post_checked": False, "early_los": False,
              "restoration_ok": False, "failed_step": False}
    original_step = tds.itm_step
    solver_times: list[float] = []
    solver_states: list[np.ndarray] = []
    step_records: list[dict[str, float]] = []
    all_step_records: list[dict[str, float]] = []
    previous_sample_time = 0.0
    previous_sample_delta = np.asarray(system.GENROU.delta.v).copy()
    previous_sample_omega = np.asarray(system.GENROU.omega.v).copy()
    next_los_sample = record_start_absolute
    los_deltas: list[np.ndarray] = []
    los_omegas: list[np.ndarray] = []
    current_endpoint = 0.0
    interval_clock_start = None
    interval_clock_end = None

    def timed_step():
        nonlocal previous_sample_time, previous_sample_delta, previous_sample_omega
        nonlocal next_los_sample, current_endpoint
        nonlocal interval_clock_start
        target = float(system.dae.t)
        if target > post100_absolute + 1e-8 and interval_clock_start is None:
            interval_clock_start = time.perf_counter()
        t0 = time.perf_counter()
        ok = original_step()
        elapsed = time.perf_counter() - t0
        if not ok:
            status["failed_step"] = True
            tds.busted = True
            tds.err_msg = "timing benchmark rejects nonconverged integration step"
            return ok
        # ANDES stores the converged step at dae.t, then advances dae.t.
        current_endpoint = target
        after_delta = np.asarray(system.GENROU.delta.v).copy()
        after_omega = np.asarray(system.GENROU.omega.v).copy()
        solver_times.append(current_endpoint)
        # FOM outputs are converted to the same causal coordinates as TRAIN
        # and JointROM before alignment/accuracy metrics: delta-delta* and
        # (omega-1) * omega_s, not raw ANDES angle/pu speed.
        solver_states.append(np.r_[after_delta - equilibrium["delta"],
                                   (after_omega - 1.0) * float(event["omega_s"])])
        record = {"absolute_target_s": target, "absolute_endpoint_s": current_endpoint,
                  "seconds": elapsed}
        all_step_records.append(record)
        if current_endpoint > post100_absolute + 1e-8 and current_endpoint <= end_absolute + solver_step + 1e-8:
            step_records.append(record)
        while next_los_sample <= current_endpoint + 1e-8:
            if current_endpoint <= previous_sample_time + 1e-12:
                break
            fraction = (next_los_sample - previous_sample_time) / (current_endpoint - previous_sample_time)
            if not -1e-8 <= fraction <= 1+1e-8:
                raise ValueError("LOS sampling would extrapolate")
            los_deltas.append(previous_sample_delta + fraction * (after_delta - previous_sample_delta))
            los_omegas.append(previous_sample_omega + fraction * (after_omega - previous_sample_omega))
            next_los_sample += float(config.get("sample_interval_s", DT))
        if los_deltas and los_index(np.asarray(los_deltas), np.asarray(los_omegas), inertia, config) is not None:
            status["early_los"] = True
        previous_sample_time = current_endpoint
        previous_sample_delta = after_delta
        previous_sample_omega = after_omega
        return ok

    def physical_checks():
        nonlocal interval_clock_end
        now = float(system.dae.t)
        endpoint = current_endpoint
        if not np.isfinite(system.dae.xy).all():
            status["failed_step"] = True
            tds.busted = True
            return True
        if float(config["fault_start_s"]) + 1e-8 < endpoint < clearing - 1e-8:
            if not np.any(system.Fault.uf.v != 0):
                raise ValueError("fault_was_not_applied")
            status["fault_applied"] = True
        if endpoint > clearing + 1e-8 and not status["post_checked"]:
            if physical_parameters(system) != stored_signature or np.any(system.Fault.uf.v != 0):
                raise ValueError("postfault_system_not_restored")
            status["post_checked"] = True
            status["restoration_ok"] = True
        if endpoint <= float(config["fault_start_s"]) + 1e-8:
            if (np.max(np.abs(system.GENROU.omega.v - equilibrium["omega"]))
                    > float(config["pre_fault_speed_tolerance_pu"])
                    or np.max(np.abs(system.GENROU.delta.v - equilibrium["delta"]))
                    > float(config["pre_fault_angle_drift_rad"])):
                raise ValueError("pre_fault_not_steady")
        if endpoint >= end_absolute - 1e-8 or status["early_los"]:
            interval_clock_end = time.perf_counter()
            # TDS increments dae.t after this callback. Setting tf to the
            # current pre-increment time stops after this converged step.
            tds.config.tf = now
        return True

    tds.itm_step = timed_step
    tds.check_criteria = physical_checks
    run_start = time.perf_counter()
    success = tds.run(no_summary=True)
    run_wall_seconds = time.perf_counter() - run_start
    interval_kernel_seconds = sum(item["seconds"] for item in step_records)
    interval_wall_seconds = (interval_clock_end - interval_clock_start
                             if interval_clock_end is not None and interval_clock_start is not None else 0.)
    prelude_seconds = (interval_clock_start-run_start if interval_clock_start is not None else run_wall_seconds)
    if not success or status["failed_step"]:
        raise ValueError("numerical_TDS_failure")
    if not status["fault_applied"] or not status["post_checked"]:
        raise ValueError("missing_fault_or_restoration_check")
    times = np.asarray(solver_times)
    values = np.asarray(solver_states)
    grid = np.arange(interval["steps"] + 1) * DT + post100_absolute
    if len(times) == 0 or times[-1] < grid[-1] - 1e-8:
        aligned, boundary = None, {"coverage_failure": True,
                                   "solver_end_s": float(times[-1]) if len(times) else None,
                                   "grid_end_s": float(grid[-1])}
    else:
        aligned, boundary = interpolate_solver(times, values, grid, solver_step)
    truth = event["x"][10:11+interval["steps"], :20]
    replay_error = None if aligned is None else normalized_coi_rmse(aligned, truth, event["inertia"])
    return {"event_id": event["id"], "case": str(case), "aligned": aligned,
            "absolute_times": times, "step_records": step_records,
            "boundary": boundary, "interval": interval,
            "setup_seconds": init_timing["case_code_seconds"],
            "case_code_seconds": init_timing["case_code_seconds"],
            "pflow_seconds": init_timing["pflow_seconds"],
             "tds_init_seconds": pflow_tds_seconds,
            "prelude_seconds": prelude_seconds, "interval_wall_seconds": interval_wall_seconds,
            "interval_kernel_seconds": interval_kernel_seconds,
            "replay_vs_stored_train": replay_error,
            "run_wall_seconds": run_wall_seconds, "early_los": status["early_los"],
            "fault_applied": status["fault_applied"], "restoration_ok": status["restoration_ok"]}


def physical_parameters(system) -> str:
    values = {}
    for name, model in system.models.items():
        if not model.n or model.group == "TimedEvent":
            continue
        values[name] = {key: np.asarray(param.v).tolist() for key, param in model.params.items()}
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def benchmark_rom(model_dir: Path, event: dict[str, Any], repetitions: int = 3) -> dict[str, Any]:
    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False))
    network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]])
    network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    model = _construct(model_dir, base, network, "challenger")
    interval = matched_interval(event)
    history = event["x"][:11]
    records = []
    # One unmeasured warmup keeps import/allocator/JIT effects out of the
    # three reported repetitions while preserving the exact same workload.
    rom_rollout_timed(model, history, interval["steps"] * 2, FOM_STEP)
    for repeat in range(repetitions):
        prediction, timing = rom_rollout_timed(model, history, interval["steps"] * 2, FOM_STEP)
        records.append({"repeat": repeat + 1, **timing, "gru_seconds": timing["gru_seconds"],
                        "integration_seconds": timing["integration_seconds"],
                        "end_to_end_seconds": timing["end_to_end_seconds"],
                        "prediction": prediction[::2]})
    median = median_repetitions(records, ("gru_seconds", "integration_seconds", "end_to_end_seconds"))
    fine, _ = rom_rollout_timed(model, history, interval["steps"]*4, .0025)
    fine = fine[::4]
    discrepancy = normalized_coi_rmse(records[0]["prediction"], fine, base.mass)
    return {"model": model_dir.name, "model_sha256": sha256(model_dir/'model.pt'),
            "normalization_sha256": sha256(model_dir/'normalization.npz'),
            "records": records, "median": median, "interval": interval,
            "fine_discretization": discrepancy, "fine_all_states_finite": bool(np.isfinite(fine).all())}


def _fom_timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("case_code_seconds", "pflow_seconds", "tds_init_seconds", "prelude_seconds", "interval_wall_seconds", "interval_kernel_seconds",
              "run_wall_seconds")
    return {"records": records, "median": median_repetitions(records, fields)}


def run_formal(selected: dict[str, dict[str, Any]], events: dict[str, dict[str, Any]],
               config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run the preregistered serial timing/accuracy protocol when explicitly authorized."""
    evidence: dict[str, Any] = {"mode": "formal", "selected": selected, "test_read": False,
                                "events": {}}
    for label in ("stable", "unstable"):
        event = events[label]
        # One unmeasured FOM warmup, then exactly three measured replays.
        warmup = replay_fom(event, config, output)
        if warmup['replay_vs_stored_train'] is None or warmup['replay_vs_stored_train']['combined'] > 1e-3:
            raise ValueError('original-step TRAIN replay fails reproduction; timing not admissible')
        measured = [replay_fom(event, config, output) for _ in range(3)]
        reference = replay_fom(event, config, output, solver_step=0.0025)
        mass = event["inertia"]
        fom_metrics = []
        for record in measured:
            if record["aligned"] is None or reference["aligned"] is None:
                fom_metrics.append(None)
            else:
                fom_metrics.append({
                    "normalized_coi": normalized_coi_rmse(record["aligned"], reference["aligned"], mass),
                    "common_frequency_rmse": common_frequency_rmse(record["aligned"], reference["aligned"], mass),
                })
        model_results = {}
        for model_name, model_dir in MODEL_DIRS.items():
            rom = benchmark_rom(model_dir, event, repetitions=3)
            rom_metrics = []
            if reference["aligned"] is not None:
                for record in rom["records"]:
                    prediction = np.asarray(record["prediction"])[:, :20]
                    rom_metrics.append({
                        "normalized_coi": normalized_coi_rmse(prediction, reference["aligned"], mass),
                        "common_frequency_rmse": common_frequency_rmse(prediction, reference["aligned"], mass),
                    })
            else:
                rom_metrics = [None] * len(rom["records"])
            rom["accuracy"] = rom_metrics
            fom_valid = all(v is not None and np.isfinite(v['normalized_coi']['combined'])
                            and v['normalized_coi']['combined'] <= 1e-3 for v in fom_metrics)
            rom_valid = bool(rom['fine_all_states_finite'] and rom['fine_discretization']['combined'] <= 1e-3)
            accuracy_ok = all(v is not None and v['normalized_coi']['combined'] <= .05
                              and v['common_frequency_rmse'] <= .05 for v in rom_metrics)
            raw_ratio = float(np.median([v['interval_wall_seconds'] for v in measured])/rom['median']['end_to_end_seconds'])
            rom['runtime_comparison'] = {'FOM_over_ROM_cost_ratio': raw_ratio,
                'accuracy_qualified': bool(fom_valid and rom_valid and accuracy_ok),
                'interpretation': 'accuracy-qualified on this event' if fom_valid and rom_valid and accuracy_ok else 'cost-only; not accuracy-equivalent acceleration',
                'rom_serial_events_per_second': 1/rom['median']['end_to_end_seconds'],
                'observation_wait_s': .1}
            model_results[model_name] = rom
        evidence["events"][label] = {
            "event_id": event["id"],
            "fom": _fom_timing_summary(measured),
            "warmup": {key:value for key,value in warmup.items() if key!='aligned'},
            "fom_accuracy": fom_metrics,
            "reference": {key: value for key, value in reference.items() if key != "aligned"},
            "models": model_results,
        }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="one TRAIN event replay; not formal timing")
    parser.add_argument("--run", action="store_true", help="run the preregistered formal benchmark")
    args = parser.parse_args()
    if args.smoke == args.run:
        raise RuntimeError("choose exactly one of --smoke or --run")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "private_data").resolve()) or output.exists():
        raise ValueError("output must be a fresh private_data subdirectory")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/private_dataset.json").read_text(encoding="utf-8"))
    selected = select_events()
    base = MechanicalROM.load(BASE)
    events = {label: load_train_event(candidate, base) for label, candidate in selected.items()}
    case = Path(andes.get_case(CASE_RELATIVE))
    for event in events.values():
        verify_case_and_event(event, case)
    if args.smoke:
        # Smoke replays exactly one selected event, preserving its provenance.
        label = "stable"
        fom = replay_fom(events[label], config, output)
        evidence = {"mode": "smoke", "selected": selected, "event": events[label]["id"],
                    "fom": {key: safe_value(value) for key, value in fom.items() if key != "aligned"},
                    "case_sha256": sha256(case), "andes_version": andes.__version__,
                    "test_read": False}
    else:
        evidence = run_formal(selected, events, config, output)
        evidence.update({"case_sha256": sha256(case), "andes_version": andes.__version__})
    (output / "evidence.json").write_text(json.dumps(safe_value(evidence), indent=2, allow_nan=False), encoding="utf-8")
    if args.smoke:
        print(json.dumps({"mode": "smoke", "event": events[label]["id"], "early_los": fom["early_los"]}))
    else:
        print(json.dumps({"mode": "formal", "events": list(evidence["events"])}))


if __name__ == "__main__":
    main()
