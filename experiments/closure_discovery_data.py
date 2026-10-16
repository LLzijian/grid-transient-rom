"""Metadata-gated data adapter for the registered closure-discovery study.

Condition discovery reads only the explicit TRAIN split and its CSV index.  New
trajectories are generated in separate, local artifacts and never enter the
historical private-data stores.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT / "experiments", ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))
DATASET_ROOTS = (ROOT / "private_data/ieee39_postfault", ROOT / "private_data/ieee39_expansion")
CONFIG_PATH = ROOT / "configs/private_dataset.json"
BASE_PATH = ROOT / "private_data/task2_m3_0020/model.npz"
OMEGA_S = 2.0 * np.pi * 60.0
SCALES = np.asarray([0.47675314588683515, 1.0033667249602924], dtype=np.float64)
INTEGRATION_STEPS = (0.005, 0.0025)
RAW_KEYS = ("t", "delta", "omega", "Pm", "Pe", "te", "full_state",
            "algebraic_state", "bus_v", "bus_a", "terminal_v")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False,
                                    default=_json_default), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bus_key(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(text))
    except ValueError:
        return text


def allowed_train_conditions() -> list[dict[str, Any]]:
    """Return physical conditions attached to explicit historical TRAIN IDs.

    Only ``splits.json`` and ``dataset_index.csv`` are opened.  This function
    deliberately does not read labels, outcomes, HDF5 files, or trajectory data.
    """
    result: list[dict[str, Any]] = []
    for root in DATASET_ROOTS:
        split_path = root / "splits.json"
        index_path = root / "dataset_index.csv"
        splits = json.loads(split_path.read_text(encoding="utf-8"))
        train_ids = set(map(str, splits.get("train", ())))
        val_ids = set(map(str, splits.get("validation", splits.get("val", ()))))
        test_ids = set(map(str, splits.get("test", ())))
        if train_ids & (val_ids | test_ids) or val_ids & test_ids:
            raise ValueError(f"overlapping split IDs in {root.name}")
        with index_path.open(encoding="utf-8", newline="") as stream:
            rows = {str(row["event_id"]): row for row in csv.DictReader(stream)}
        missing = train_ids - rows.keys()
        if missing:
            raise ValueError(f"TRAIN IDs missing from {root.name} index: {sorted(missing)[:5]}")
        for event_id in sorted(train_ids):
            row = rows[event_id]
            required = ("fault_bus", "fault_line", "fault_location", "fault_duration_s", "fault_xf")
            if any(row.get(key) in (None, "") for key in required):
                raise ValueError(f"incomplete physical condition for {root.name}/{event_id}")
            bus = _bus_key(row["fault_bus"])
            location = float(row["fault_location"])
            if location not in (0.0, 1.0):
                raise ValueError(f"TRAIN condition is not a documented line endpoint: {event_id}")
            result.append({
                "event_id": event_id,
                "id": event_id,
                "root": root.name,
                "bus": int(bus) if bus.isdigit() else bus,
                "bus_id": bus,
                "line": str(row["fault_line"]),
                "location": location,
                "duration_s": float(row["fault_duration_s"]),
                "xf": float(row["fault_xf"]),
            })
    return result


def _physical_key(row: Mapping[str, Any]) -> tuple[str, str, float, float, float]:
    bus = _bus_key(row.get("bus", row.get("bus_id", row.get("train_bus"))))
    line = str(row.get("line", row.get("fault_line", "")))
    location = float(row.get("location", row.get("fault_location", 0.0)))
    duration = float(row.get("duration_s", row.get("fault_duration_s")))
    xf = float(row.get("xf", row.get("fault_xf")))
    return bus, line, location, round(duration, 12), round(xf, 12)


def _distance(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> float:
    # Fault durations are compared on their registered interval; reactance is
    # compared logarithmically because the registered grid is geometric.
    duration_scale = 0.55
    xf_ratio = abs(math.log10(float(candidate.get("xf", candidate.get("fault_xf"))) /
                              float(reference.get("xf", reference.get("fault_xf")))))
    duration_delta = (float(candidate.get("duration_s", candidate.get("fault_duration_s"))) -
                      float(reference.get("duration_s", reference.get("fault_duration_s")))) / duration_scale
    return math.hypot(duration_delta, xf_ratio)


def _score_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = next((value[key] for key in ("disagreement", "score", "discrepancy", "value")
                      if key in value), None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _condition_score(scores: Mapping[Any, Any], row: Mapping[str, Any]) -> float | None:
    event_id = str(row.get("event_id", row.get("id", "")))
    for key in (event_id, f"{row.get('root', '')}:{event_id}",
                (str(row.get("root", "")), event_id)):
        if key in scores:
            return _score_value(scores[key])
    # The existing support helper's stable condition hash is accepted too.
    try:
        from closure_discovery_support import fault_condition_key
        key = fault_condition_key({"bus_id": _bus_key(row.get("bus", row.get("bus_id"))),
                                  "duration_s": row.get("duration_s", row.get("fault_duration_s")),
                                  "xf": row.get("xf", row.get("fault_xf"))})
        if key in scores:
            return _score_value(scores[key])
    except (ImportError, KeyError, TypeError, ValueError):
        pass
    return None


def select_new_train_conditions(
    pool: Sequence[Mapping[str, Any]],
    existingtrain_disagreement: Mapping[Any, Any] | Sequence[Mapping[str, Any]],
    *,
    count: int = 24,
) -> list[dict[str, Any]]:
    """Rank new candidates by nearby historical TRAIN disagreement only.

    Scores are keyed by historical TRAIN event ID (or by the established
    condition hash).  Each candidate inherits the score of its nearest old
    TRAIN condition on the same bus; no generated trajectory is read or scored.
    """
    if count < 0:
        raise ValueError("count must be nonnegative")
    if not isinstance(existingtrain_disagreement, Mapping):
        converted: dict[Any, Any] = {}
        for row in existingtrain_disagreement:
            if not isinstance(row, Mapping):
                continue
            event_id = row.get("event_id", row.get("id"))
            if event_id is not None:
                converted[str(event_id)] = row
        existingtrain_disagreement = converted
    old = allowed_train_conditions()
    scored_old = [(row, _condition_score(existingtrain_disagreement, row)) for row in old]
    scored_old = [(row, score) for row, score in scored_old if score is not None]
    ranked: list[tuple[float, float, str, dict[str, Any]]] = []
    seen: set[tuple[str, str, float, float, float]] = set()
    for item in pool:
        candidate = _resolve_line(item, old)
        bus = _bus_key(candidate.get("bus", candidate.get("bus_id", candidate.get("train_bus"))))
        candidate_key = _physical_key(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        local = [(row, score) for row, score in scored_old if _bus_key(row["bus"]) == bus]
        if candidate.get("line", candidate.get("fault_line")) is not None:
            line = str(candidate.get("line", candidate.get("fault_line")))
            loc = float(candidate.get("location", candidate.get("fault_location", 0.0)))
            same_line = [(row, score) for row, score in local
                         if row["line"] == line and float(row["location"]) == loc]
            if same_line:
                local = same_line
        if not local:
            continue
        nearest, score = min(local, key=lambda pair: (_distance(candidate, pair[0]),
                                                       -float(pair[1]), pair[0]["event_id"]))
        candidate.update({"split": "train", "selection_score": float(score),
                          "nearest_train_event_id": nearest["event_id"],
                          "nearest_train_distance": _distance(candidate, nearest)})
        ranked.append((-float(score), _distance(candidate, nearest),
                       str(candidate.get("id", candidate.get("event_id", candidate_key))), candidate))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    if len(ranked) < count:
        raise ValueError(f"need {count} scored distinct candidate conditions; found {len(ranked)}")
    return [row[3] for row in ranked[:count]]


def _resolve_line(candidate: Mapping[str, Any], train_conditions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bus_value = candidate.get("bus", candidate.get("bus_id", candidate.get("train_bus")))
    if bus_value is None:
        raise ValueError("condition needs bus or bus_id")
    bus_key = _bus_key(bus_value)
    options = [row for row in train_conditions if _bus_key(row["bus"]) == bus_key]
    if not options:
        raise ValueError(f"bus {bus_key} is not present in historical TRAIN")
    duration = float(candidate.get("duration_s", candidate.get("fault_duration_s")))
    xf = float(candidate.get("xf", candidate.get("fault_xf")))
    if not (math.isfinite(duration) and duration > 0 and math.isfinite(xf) and xf > 0):
        raise ValueError("duration_s and xf must be finite and positive")
    if "line" in candidate or "fault_line" in candidate:
        line = str(candidate.get("line", candidate.get("fault_line")))
        location = float(candidate.get("location", candidate.get("fault_location", 0.0)))
        matches = [row for row in options if row["line"] == line and row["location"] == location]
        if not matches:
            raise ValueError(f"line endpoint {line}@{location} is not documented for TRAIN bus {bus_key}")
        reference = matches[0]
    else:
        # New protocol conditions carry only a TRAIN bus and electrical fault
        # parameters.  Reuse the lexically first documented endpoint for that bus.
        reference = min(options, key=lambda row: (row["line"], row["location"], row["event_id"]))
        line, location = reference["line"], float(reference["location"])
    bus: int | str = int(bus_key) if bus_key.isdigit() else bus_key
    normalized = {**dict(candidate), "bus": bus, "bus_id": bus_key,
                  "line": line, "location": location,
                  "duration_s": duration, "xf": xf}
    if "id" not in normalized and "event_id" not in normalized:
        payload = json.dumps([bus_key, line, location, duration, xf], separators=(",", ":"))
        normalized["id"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return normalized


def _conditions_for_split(conditions: Any, split: str) -> list[dict[str, Any]]:
    if isinstance(conditions, Mapping):
        key = "train_selected" if split == "train" else "val_conditions"
        conditions = conditions.get(key)
        if conditions is None:
            raise ValueError(f"condition manifest has no {key}")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        raise TypeError("conditions must be an explicit sequence or closure manifest")
    train = allowed_train_conditions()
    normalized = []
    seen = set()
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise TypeError("each condition must be a mapping")
        declared = condition.get("split")
        if declared in ("validation", "val"):
            declared = "val"
        if declared is not None and declared != split:
            raise ValueError(f"condition split {declared!r} does not match requested {split!r}")
        row = _resolve_line(condition, train)
        row["split"] = split
        key = _physical_key(row)
        if key in seen:
            raise ValueError(f"duplicate condition in {split}: {key}")
        seen.add(key)
        normalized.append(row)
    return normalized


def _freeze_snapshot(freeze_path: str | Path) -> dict[str, Any]:
    path = Path(freeze_path).expanduser().resolve()
    if path.is_dir():
        path = path / "frozen.json"
    if not path.is_file():
        raise FileNotFoundError(f"freeze manifest not found: {path}")
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    models = manifest.get("models")
    if isinstance(models, Mapping):
        models = list(models.values())
    if not isinstance(models, list) or not models:
        raise ValueError("freeze manifest must declare a nonempty models list")
    verified = []
    for row in models:
        if not isinstance(row, Mapping) or not row.get("path"):
            raise ValueError("each frozen model needs a path and sha256")
        expected = row.get("sha256", row.get("model_sha256"))
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("each frozen model needs a SHA256")
        declared_path = Path(str(row["path"]))
        candidates = [declared_path] if declared_path.is_absolute() else [ROOT / declared_path, path.parent / declared_path]
        model_path = next((item.resolve() for item in candidates if item.is_file()), None)
        if model_path is None:
            candidate_files = [candidate / "model.json" for candidate in candidates]
            model_path = next((item.resolve() for item in candidate_files if item.is_file()), None)
        if model_path is None:
            raise FileNotFoundError(f"frozen model file not found for {row['path']}")
        actual = _sha256(model_path)
        if actual != expected:
            raise ValueError(f"frozen model hash mismatch: {model_path}")
        verified.append({"name": str(row.get("name", model_path.stem)),
                         "path": str(model_path), "sha256": actual})
    return {"path": str(path), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "main_name": manifest.get("main_name", manifest.get("primary_model")),
            "models": verified}


def _load_base(base: Any = None) -> Any:
    if base is not None:
        return base
    from kiki.models.private_pm import MechanicalROM
    return MechanicalROM.load(BASE_PATH)


def _load_simulator() -> Callable[..., Any]:
    from generate_private_dataset import simulate
    return simulate


class _RemoteSimulationError(RuntimeError):
    def __init__(self, message: str, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def _simulation_worker(condition: Mapping[str, Any], config: Mapping[str, Any],
                       output: str, event_id: str, step: float, connection: Any) -> None:
    """Isolated default FOM worker so the stage cap can terminate a long call."""
    diagnostics: dict[str, Any] = {}
    try:
        result = _load_simulator()(condition, config, diagnostics=diagnostics)
        raw_file = _persist_raw(Path(output), event_id, step, result)
        connection.send({"status": "complete", "raw_file": raw_file,
                         "diagnostics": diagnostics})
    except BaseException as error:
        try:
            connection.send({"status": "failed", "error": f"{type(error).__name__}: {error}",
                             "diagnostics": diagnostics})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _run_bounded_simulation(condition: Mapping[str, Any], config: Mapping[str, Any],
                            output: Path, event_id: str, step: float,
                            timeout_seconds: float) -> tuple[Any, dict[str, Any], str]:
    """Run the real simulator in a child process and stop it at the cap."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_simulation_worker,
                              args=(dict(condition), dict(config), str(output), event_id, step, sender))
    deadline = time.perf_counter() + max(0.0, float(timeout_seconds))
    process.start()
    sender.close()
    message = None
    try:
        remaining = max(0.0, deadline - time.perf_counter())
        if receiver.poll(remaining):
            message = receiver.recv()
        if message is None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
                raw_name = f"{event_id}_dt{int(round(step * 10000)):04d}.npz"
                partial = output / "raw" / raw_name
                error = TimeoutError(f"FOM call exceeded remaining cap of {timeout_seconds:.6g} seconds")
                if partial.exists():
                    error.raw_file = str(partial.relative_to(output))
                raise error
            process.join(timeout=0)
            raise _RemoteSimulationError("FOM worker exited without returning an outcome")
        remaining = max(0.0, deadline - time.perf_counter())
        process.join(timeout=min(remaining, 1.0))
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        if message.get("status") != "complete":
            raise _RemoteSimulationError(message.get("error", "FOM worker failed"),
                                         message.get("diagnostics"))
        attempt = {"steps": {f"{step:g}": {"raw_file": message["raw_file"]}}}
        result = _load_result(output, attempt, f"{step:g}")
        return result, dict(message.get("diagnostics", {})), str(message["raw_file"])
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)


def _safe_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("disable_pss") is not False:
        raise ValueError("registered private_dataset.json must retain disable_pss=false")
    if float(config.get("postfault_horizon_s", -1)) != 15.0:
        raise ValueError("registered private_dataset.json must retain the 15-s postfault horizon")
    return config


def _persist_raw(output: Path, condition_id: str, step: float, result: Any) -> str:
    arrays, metadata, equilibrium, inertia, mapping, algebraic_names = result
    if not isinstance(arrays, Mapping):
        raise TypeError("simulator result arrays must be a mapping")
    attempt_dir = output / "raw"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{condition_id}_dt{int(round(step * 10000)):04d}.npz"
    path = attempt_dir / filename
    payload: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"raw array {key!r} has object dtype")
        payload[f"raw_{key}"] = array
    for key, value in equilibrium.items():
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"equilibrium array {key!r} has object dtype")
        payload[f"equilibrium_{key}"] = array
    payload["inertia"] = np.asarray(inertia, dtype=np.float64)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    os.replace(temporary, path)
    _write_json(attempt_dir / f"{condition_id}_dt{int(round(step * 10000)):04d}.json",
                {"metadata": dict(metadata), "state_mapping": mapping,
                 "algebraic_names": list(algebraic_names)})
    return str(path.relative_to(output))


def _validate_raw(result: Any, config: Mapping[str, Any]) -> None:
    arrays, metadata, equilibrium, inertia, _mapping, _names = result
    required = {"t", "delta", "omega", "Pm", "Pe", "te", "full_state"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"simulator omitted required raw arrays: {sorted(missing)}")
    from kiki.data.private_dataset import check_arrays
    check_arrays(arrays, arrays["full_state"], config)
    time_axis = np.asarray(arrays["t"], dtype=np.float64)
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
            raise ValueError(f"raw trajectory array {key!r} is nonnumeric or nonfinite")
        if array.ndim and array.shape[0] != len(time_axis):
            raise ValueError(f"raw trajectory array {key!r} has an inconsistent time length")
    if np.asarray(arrays["te"]).shape != (len(time_axis), 10):
        raise ValueError("raw mechanical torque must have ten generator columns")
    if metadata.get("label") not in ("stable", "unstable"):
        raise ValueError("simulator returned no valid physical outcome label")
    if metadata.get("solver_success", True) is not True:
        raise ValueError("simulator marked the numerical solve unsuccessful")
    inertia = np.asarray(inertia, dtype=np.float64)
    if inertia.shape != (10,) or not np.isfinite(inertia).all() or np.any(inertia <= 0):
        raise ValueError("simulator returned invalid generator inertia")
    if "delta" not in equilibrium or np.asarray(equilibrium["delta"]).shape != (10,):
        raise ValueError("simulator returned invalid equilibrium angles")
    if time_axis[-1] > float(config["postfault_horizon_s"]) + 1e-8:
        raise ValueError("trajectory exceeds the registered postfault horizon")
    if abs(float(metadata.get("trajectory_end_s", np.nan)) - float(time_axis[-1])) > 1e-8:
        raise ValueError("metadata and raw trajectory endpoints differ")
    if metadata["label"] == "stable":
        if time_axis[-1] < float(config["postfault_horizon_s"]) - 1e-8 or metadata.get("event_time_s") not in (None, ""):
            raise ValueError("stable label lacks the complete confirmed horizon")
    else:
        try:
            event_time = float(metadata["event_time_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("unstable label lacks its recorded physical event time") from error
        if abs(event_time - float(time_axis[-1])) > 1e-8:
            raise ValueError("unstable event time differs from the actual trajectory endpoint")
    if float(config["sample_interval_s"]) != 0.01:
        raise ValueError("private dataset output sample interval changed")


def _classify_result(result: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    arrays, metadata, _equilibrium, inertia, _mapping, _names = result
    event_time = metadata.get("event_time_s")
    if event_time in (None, ""):
        event_time = None
    else:
        event_time = float(event_time)
    # Use the repository's frozen physical LOS/window rules on raw ANDES units.
    from evaluate_l1_latent import classify_prediction
    raw_path = np.column_stack((np.asarray(arrays["delta"]), np.asarray(arrays["omega"])))
    result_class = classify_prediction(raw_path, np.asarray(inertia, dtype=np.float64),
                                       true_event_time_s=event_time,
                                       config={key: config[key] for key in (
                                           "sample_interval_s", "los_angle_rad",
                                           "los_relative_speed_pu", "los_hold_samples",
                                           "stable_tail_seconds", "stable_tail_speed_pu")})
    return result_class


def _pair_qualification(results: Mapping[str, Any], valid: Mapping[str, bool],
                        config: Mapping[str, Any]) -> dict[str, Any]:
    coarse, fine = results["0.005"], results["0.0025"]
    if not (valid.get("0.005") and valid.get("0.0025")):
        return {"pass": False, "reason": "raw_simulation_or_validation_failure"}
    try:
        coarse_class = _classify_result(coarse, config)
        fine_class = _classify_result(fine, config)
    except Exception as error:
        return {"pass": False, "reason": "classification_failure",
                "classification_error": f"{type(error).__name__}: {error}"}
    category_coarse, category_fine = coarse_class["category"], fine_class["category"]
    categories_match = category_coarse == category_fine
    predicted_coarse = coarse_class.get("trigger_index")
    predicted_fine = fine_class.get("trigger_index")
    classified_time_error = None
    if predicted_coarse is not None and predicted_fine is not None:
        classified_time_error = abs(int(predicted_coarse) - int(predicted_fine)) * float(config["sample_interval_s"])
        categories_match = categories_match and classified_time_error <= 0.02 + 1e-12
    meta_coarse, meta_fine = coarse[1], fine[1]
    labels_match = meta_coarse.get("label") == meta_fine.get("label")
    label_consistent = True
    if category_coarse != "latency_uncovered":
        label_consistent &= ((meta_coarse.get("label") == "unstable") ==
                             bool(coarse_class.get("predicted_los")))
    if category_fine != "latency_uncovered":
        label_consistent &= ((meta_fine.get("label") == "unstable") ==
                             bool(fine_class.get("predicted_los")))
    event_time_coarse = meta_coarse.get("event_time_s")
    event_time_fine = meta_fine.get("event_time_s")
    true_time_error = None
    if meta_coarse.get("label") == meta_fine.get("label") == "unstable":
        try:
            true_time_error = abs(float(event_time_coarse) - float(event_time_fine))
            labels_match = labels_match and true_time_error <= 0.02 + 1e-12
        except (TypeError, ValueError):
            labels_match = False
    a, b = coarse[0], fine[0]
    finite = all(np.isfinite(np.asarray(a[key])).all() and np.isfinite(np.asarray(b[key])).all()
                 for key in ("delta", "omega"))
    common = min(len(a["t"]), len(b["t"]))
    if predicted_coarse is not None and predicted_fine is not None:
        common = min(common, max(int(predicted_coarse), int(predicted_fine)) + 1)
    if common < 1:
        finite = False
    if finite:
        delta_error = np.asarray(a["delta"][:common]) - np.asarray(b["delta"][:common])
        speed_error = (np.asarray(a["omega"][:common]) - np.asarray(b["omega"][:common])) * OMEGA_S
        mass = np.asarray(coarse[3], dtype=np.float64)
        mass = mass / mass.sum()
        delta_coi = delta_error - (delta_error * mass[None, :]).sum(axis=1, keepdims=True)
        speed_coi = speed_error - (speed_error * mass[None, :]).sum(axis=1, keepdims=True)
        scaled_rms = float(np.sqrt(0.5 * np.mean((delta_coi / SCALES[0]) ** 2 +
                                                 (speed_coi / SCALES[1]) ** 2)))
    else:
        scaled_rms = float("inf")
    numerical_pass = finite and math.isfinite(scaled_rms) and scaled_rms <= 0.001
    passed = bool(numerical_pass and categories_match and labels_match and label_consistent)
    return {"pass": passed, "finite": bool(finite), "comparison_samples": int(common),
            "scaled_rms": scaled_rms, "scaled_rms_tolerance": 0.001,
            "classification_coarse": coarse_class, "classification_fine": fine_class,
            "classification_agreement": bool(categories_match),
            "classification_time_error_s": classified_time_error,
            "label_agreement": bool(labels_match), "label_consistent_with_physical_rule": bool(label_consistent),
            "physical_event_time_difference_s": true_time_error,
            "reason": None if passed else "numerical_or_classification_gate_failed"}


def _attempt_record(condition: Mapping[str, Any], split: str) -> dict[str, Any]:
    row = dict(condition)
    event_id = str(row.get("id", row.get("event_id", "")))
    if not event_id:
        identity = [split, row["bus_id"], row["line"], row["location"], row["duration_s"], row["xf"]]
        event_id = f"new{split}_{hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]}"
    else:
        event_id = f"new{split}_{event_id}"
    row.update({"event_id": event_id, "new_split": split})
    return row


def generate_conditions(
    conditions: Any,
    out: str | Path,
    split: str,
    cap_seconds: float,
    *,
    freeze_path: str | Path | None = None,
    simulator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run both registered time steps for explicit new TRAIN or VAL conditions.

    The time cap is checked before each invocation.  The default FOM runs in an
    isolated child process and is terminated if it uses the remaining cap; an
    injected simulator runs inline and is intended for deterministic tests.
    """
    if split not in ("train", "val"):
        raise ValueError("new data generation is restricted to split='train' or 'val'")
    cap_seconds = float(cap_seconds)
    if not math.isfinite(cap_seconds) or cap_seconds < 0:
        raise ValueError("cap_seconds must be finite and nonnegative")
    freeze = _freeze_snapshot(freeze_path) if split == "val" and freeze_path is not None else None
    if split == "val" and freeze is None:
        raise ValueError("VAL generation requires a verified frozen-model manifest")
    normalized = [_attempt_record(row, split) for row in _conditions_for_split(conditions, split)]
    output = Path(out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generation.json"
    if manifest_path.exists():
        raise FileExistsError(f"generation artifact already exists: {manifest_path}")
    raw_dir = output / "raw"
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise FileExistsError(f"raw attempt artifacts already exist: {raw_dir}")
    config = _safe_config()
    start = time.perf_counter()
    _write_json(manifest_path, {"schema_version": 1, "split": split,
                                "conditions": normalized, "attempts": [],
                                "freeze": freeze, "status": "RUNNING",
                                "test_read": False, "cap_seconds": cap_seconds})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_simulator = simulator
    stop = False
    stop_reason = None
    for condition in normalized:
        per_condition: dict[str, Any] = {"condition": condition, "steps": {}, "pair_qualification": None}
        results: dict[str, Any] = {}
        valid: dict[str, bool] = {}
        for step in INTEGRATION_STEPS:
            key = f"{step:g}"
            if split == "val":
                try:
                    if _freeze_snapshot(freeze_path) != freeze:
                        raise ValueError("freeze manifest/model hashes changed")
                except Exception as error:
                    per_condition["steps"][key] = {
                        "status": "freeze_invalidated",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    valid[key] = False
                    stop = True
                    stop_reason = "freeze_invalidated"
                    continue
            elapsed = time.perf_counter() - start
            if elapsed >= cap_seconds:
                per_condition["steps"][key] = {"status": "not_run_budget_exhausted",
                                               "elapsed_seconds_before_start": elapsed}
                valid[key] = False
                stop = True
                stop_reason = stop_reason or "budget_exhausted"
                continue
            step_config = dict(config)
            step_config["integration_step_s"] = step
            diagnostics: dict[str, Any] = {}
            raw_path = None
            try:
                if run_simulator is None:
                    result, diagnostics, raw_path = _run_bounded_simulation(
                        condition, step_config, output, condition["event_id"], step,
                        cap_seconds - (time.perf_counter() - start))
                else:
                    result = run_simulator(condition, step_config, diagnostics=diagnostics)
                    raw_path = _persist_raw(output, condition["event_id"], step, result)
                if time.perf_counter() - start >= cap_seconds:
                    stop = True
                    stop_reason = stop_reason or "budget_exhausted"
                results[key] = result
                try:
                    _validate_raw(result, step_config)
                    raw_validation = {"pass": True}
                    valid[key] = True
                except Exception as error:
                    raw_validation = {"pass": False, "error": f"{type(error).__name__}: {error}"}
                    valid[key] = False
                metadata = result[1]
                per_condition["steps"][key] = {
                    "status": "completed" if valid[key] else "raw_validation_failed",
                    "raw_file": raw_path, "metadata": dict(metadata),
                    "diagnostics": diagnostics, "raw_validation": raw_validation,
                }
            except Exception as error:
                error_diagnostics = getattr(error, "diagnostics", diagnostics)
                failed_status = "simulation_timeout" if isinstance(error, TimeoutError) else "simulation_failed"
                per_condition["steps"][key] = {
                    "status": failed_status, "error": f"{type(error).__name__}: {error}",
                    "diagnostics": error_diagnostics,
                }
                if raw_path is None:
                    raw_path = getattr(error, "raw_file", None)
                if raw_path is not None:
                    per_condition["steps"][key]["raw_file"] = raw_path
                valid[key] = False
        if set(results) == {"0.005", "0.0025"}:
            per_condition["pair_qualification"] = _pair_qualification(results, valid, config)
        else:
            step_statuses = [row.get("status") for row in per_condition["steps"].values()]
            reason = ("budget_exhausted_during_time_step_pair"
                      if any(status == "not_run_budget_exhausted" for status in step_statuses)
                      else "raw_simulation_or_validation_failure")
            per_condition["pair_qualification"] = {"pass": False,
                                                    "reason": reason}
        manifest["attempts"].append(per_condition)
        manifest["elapsed_seconds"] = time.perf_counter() - start
        manifest["status"] = "RUNNING"
        _write_json(manifest_path, manifest)
        if stop:
            break
    if len(manifest["attempts"]) < len(normalized):
        for condition in normalized[len(manifest["attempts"]):]:
            manifest["attempts"].append({"condition": condition,
                "steps": {f"{step:g}": {"status": "not_run_budget_exhausted"} for step in INTEGRATION_STEPS},
                "pair_qualification": {"pass": False, "reason": "budget_exhausted_before_condition"}})
    if split == "val":
        try:
            freeze_after = _freeze_snapshot(freeze_path)
            if freeze_after != freeze:
                raise ValueError("freeze manifest/model hashes changed during VAL generation")
        except Exception as error:
            manifest["freeze_error"] = f"{type(error).__name__}: {error}"
            for row in manifest["attempts"]:
                row["pair_qualification"] = {"pass": False, "reason": "freeze_invalidated"}
            stop = True
            stop_reason = "freeze_invalidated"
    manifest["elapsed_seconds"] = time.perf_counter() - start
    manifest["status"] = ("complete" if not stop and len(manifest["attempts"]) == len(normalized)
                          else stop_reason or "budget_exhausted")
    manifest["qualified_count"] = sum(bool(row["pair_qualification"].get("pass"))
                                      for row in manifest["attempts"])
    _write_json(manifest_path, manifest)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_result(output: Path, attempt: Mapping[str, Any], step: str) -> tuple[Any, ...]:
    step_record = attempt["steps"][step]
    with np.load(output / step_record["raw_file"], allow_pickle=False) as archive:
        arrays = {key.removeprefix("raw_"): archive[key].copy()
                  for key in archive.files if key.startswith("raw_")}
        equilibrium = {key.removeprefix("equilibrium_"): archive[key].copy()
                       for key in archive.files if key.startswith("equilibrium_")}
        inertia = archive["inertia"].copy()
    detail_path = output / "raw" / Path(step_record["raw_file"]).name.replace(".npz", ".json")
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    metadata = dict(step_record.get("metadata", detail.get("metadata", {})))
    return arrays, metadata, equilibrium, inertia, detail.get("state_mapping"), detail.get("algebraic_names", [])


def load_new_events(
    out: str | Path,
    split: str,
    freeze_path: str | Path | None = None,
    *,
    base: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only qualified new data in root-runner event format.

    TEST is never accepted.  VAL requires the same unchanged freeze manifest and
    every declared model file to retain its recorded hash.
    """
    if split not in ("train", "val"):
        raise ValueError("new data loading is restricted to split='train' or split='val'")
    output = Path(out).expanduser().resolve()
    manifest = json.loads((output / "generation.json").read_text(encoding="utf-8"))
    if manifest.get("split") != split:
        raise ValueError(f"artifact split {manifest.get('split')!r} does not match {split!r}")
    if split == "val":
        if freeze_path is None:
            raise ValueError("VAL loading requires the frozen-model manifest")
        current = _freeze_snapshot(freeze_path)
        stored = manifest.get("freeze")
        if stored is None or current != stored:
            raise ValueError("VAL freeze manifest/model hashes differ from generation-time freeze")
    base_model = _load_base(base)
    events: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for attempt in manifest.get("attempts", []):
        condition = attempt["condition"]
        pair = attempt.get("pair_qualification") or {}
        if not pair.get("pass"):
            excluded.append({"id": condition["event_id"], "root": "closure_discovery",
                             "new_split": split, "reason": pair.get("reason", "unqualified_pair"),
                             "steps": {key: value.get("status") for key, value in attempt.get("steps", {}).items()}})
            continue
        result = _load_result(output, attempt, "0.005")
        arrays, metadata, equilibrium, inertia, _mapping, _names = result
        t = np.asarray(arrays["t"], dtype=np.float64)
        if len(t) < 2 or float(t[-1]) <= 0.1 + 1e-9:
            excluded.append({"id": condition["event_id"], "root": "closure_discovery",
                             "new_split": split, "reason": "latency_uncovered",
                             "trajectory_end_s": float(t[-1]) if len(t) else None})
            continue
        try:
            delta = np.asarray(arrays["delta"], dtype=np.float64)
            omega = np.asarray(arrays["omega"], dtype=np.float64)
            pm = np.asarray(arrays["Pm"], dtype=np.float64)
            te = np.asarray(arrays["te"], dtype=np.float64)
            eq_delta = np.asarray(equilibrium["delta"], dtype=np.float64)
            if any(value.shape != (len(t), 10) for value in (delta, omega, pm, te)):
                raise ValueError("expected ten-generator arrays at every actual output time")
            if eq_delta.shape != (10,) or np.asarray(inertia).shape != (10,):
                raise ValueError("expected ten-generator equilibrium and inertia")
            x = np.column_stack((delta - eq_delta[None, :],
                                 (omega - 1.0) * OMEGA_S,
                                 pm - np.asarray(base_model.pm_star, dtype=np.float64)[None, :]))
            torque = te - np.asarray(base_model.pm_star, dtype=np.float64)[None, :]
            if x.shape != (len(t), 30) or not np.isfinite(x).all() or not np.isfinite(torque).all():
                raise ValueError("nonfinite or incompatible canonical states/torque")
        except Exception as error:
            excluded.append({"id": condition["event_id"], "root": "closure_discovery",
                             "new_split": split, "reason": "canonicalization_failure",
                             "error": f"{type(error).__name__}: {error}"})
            continue
        clearing = float(metadata["clearing_time_s"])
        delay = float(metadata.get("record_start_delay_s") or 0.0)
        start_absolute = clearing + delay
        event_time = metadata.get("event_time_s")
        if event_time in (None, ""):
            event_time = None
        else:
            event_time = float(event_time)
        events.append({
            "id": condition["event_id"], "group": str(condition["bus_id"]),
            "label": str(metadata["label"]), "x": x, "torque": torque,
            "root": "closure_discovery", "new_split": split,
            "t": t, "absolute_t": start_absolute + t,
            "clearing_time_s": clearing, "record_start_delay_s": delay,
            "record_start_absolute_s": start_absolute,
            "trajectory_end_s": float(t[-1]), "event_time_s": event_time,
            "classification_meta": {"equilibrium_delta": eq_delta.copy(),
                                     "omega_s": OMEGA_S,
                                     "inertia": np.asarray(inertia, dtype=np.float64).copy(),
                                     "event_time_s": event_time},
        })
    return events, excluded
