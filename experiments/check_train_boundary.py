"""Mechanical QA for the preregistered Loop0075 TRAIN boundary pairs.

This checker consumes only the two new all-TRAIN roots and selection metadata.
It never loads old event trajectories, VAL/TEST payloads, or runs a
simulation.  The optional old-index/reference-H5 inputs are metadata-only
provenance checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

S_DELTA = 0.47675314588683515
S_OMEGA = 1.0033667249602924
OMEGA_DEFAULT = 2.0 * np.pi * 60.0
CONDITION_FIELDS = ("fault_line", "fault_location", "fault_bus", "fault_type", "fault_xf", "fault_duration_s")
EVENT_ARRAYS = ("t", "delta", "omega", "Pm", "Pe", "Qe", "te", "bus_v", "bus_a", "terminal_v", "full_state", "algebraic_state")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _condition(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("fault_line")),
        float(row["fault_location"]),
        int(row["fault_bus"]),
        str(row.get("fault_type")),
        float(row["fault_xf"]),
        float(row.get("fault_duration_s", row.get("duration_s"))),
    )


def load_selection(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("selected_events"), list):
        raise ValueError("selection JSON needs the preregistered selected_events list")
    entries = payload["selected_events"]
    if len(entries) != 20:
        raise ValueError(f"selection must contain exactly 20 attempts, got {len(entries)}")
    result = []
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"selection attempt {index} is not an object")
        # This is the immutable metadata-only selection schema.  Do not
        # reinterpret endpoint IDs or search for replacement conditions.
        required = ("line", "location", "bus", "xf", "duration_s", "fault_type")
        if any(name not in raw for name in required):
            raise ValueError(f"selection attempt {index} lacks selected_events fields")
        row = {
            "event_id": f"boundary_{index:04d}",
            "fault_line": str(raw["line"]),
            "fault_location": float(raw["location"]),
            "fault_bus": int(raw["bus"]),
            "fault_xf": float(raw["xf"]),
            "fault_duration_s": float(raw["duration_s"]),
            "fault_type": str(raw["fault_type"]),
        }
        result.append(row)
    ids = [row["event_id"] for row in result]
    if len(set(ids)) != 20 or any(not event_id.startswith("boundary_") for event_id in ids):
        raise ValueError("selection IDs must be unique boundary_* IDs")
    return result


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_root(root: Path, selection: list[dict[str, Any]]) -> dict[str, Any]:
    index_path, split_path, h5_path = root / "dataset_index.csv", root / "splits.json", root / "dataset.h5"
    for path in (split_path, h5_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if index_path.exists():
        with index_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        # The generator does not emit an index when all 20 attempts fail.
        # Treat that as an explicit empty accepted set; the final evidence
        # remains UNKNOWN rather than crashing while reading provenance.
        rows = []
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    ids = [str(row["event_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate event IDs in {root}")
    split_ids = {str(event_id) for values in splits.values() for event_id in values}
    if set(ids) != split_ids or set(splits.get("validation", [])) or set(splits.get("test", [])) or set(splits.get("train", [])) != set(ids):
        raise ValueError(f"{root} is not an all-TRAIN split")
    by_id = {str(row["event_id"]): row for row in rows}
    with h5py.File(h5_path, "r") as file:
        attempted = int(file.attrs.get("attempted", -1))
        rejected = json.loads(_decode(file.attrs.get("discard_counts", "{}")))
        if attempted != len(selection):
            raise ValueError(f"{root} attempted count {attempted} != 20")
        equilibrium = np.asarray(file["equilibrium/full_state"][...], dtype=float) if "equilibrium" in file and "full_state" in file["equilibrium"] else np.empty(0, dtype=float)
        inertia = np.asarray(file["inertia"][...], dtype=float) if "inertia" in file else np.empty(0, dtype=float)
        provenance = {
            "case_sha256": _decode(file.attrs.get("case_sha256")),
            "postfault_signature": _decode(file.attrs.get("postfault_signature")),
            "omega_s": float(file.attrs.get("omega_s", OMEGA_DEFAULT)),
            "state_mapping": json.loads(_decode(file.attrs.get("state_mapping", "[]"))),
            "algebraic_names": json.loads(_decode(file.attrs.get("algebraic_names", "[]"))),
            "equilibrium_full_state": equilibrium,
            "inertia": inertia,
            "config": json.loads(_decode(file.attrs["config"])),
        }
        event_group = file.get("events")
        event_ids = set() if event_group is None else {str(event_id) for event_id in event_group}
        if event_ids != set(ids):
            raise ValueError(f"H5/index event mismatch in {root}")
        events: dict[str, dict[str, Any]] = {}
        for event_id in ids:
            group = event_group[event_id]
            attrs = {name: _decode(group.attrs[name]) for name in group.attrs}
            arrays = {name: np.asarray(group[name][...]) for name in group if isinstance(group[name], h5py.Dataset)}
            events[event_id] = {"attrs": attrs, "arrays": arrays, "row": by_id[event_id]}
    hashes = {name: digest(root / name) if (root / name).exists() else None for name in ("dataset_index.csv", "splits.json", "dataset.h5")}
    return {"root": str(root), "rows": by_id, "splits": splits, "events": events, "provenance": provenance, "attempted": attempted, "rejected": rejected, "hashes": hashes}


def _metadata_value(event: dict[str, Any], name: str) -> Any:
    return event["attrs"].get(name, event["row"].get(name))


def _coi(values: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return values - (values * mass).sum(axis=1, keepdims=True) / mass.sum()


def compare_pair(coarse: dict[str, Any] | None, fine: dict[str, Any] | None, *, mass: np.ndarray, omega_s: float) -> dict[str, Any]:
    if coarse is None or fine is None:
        return {"status": "UNKNOWN", "qualified": False, "missing_coarse": coarse is None, "missing_fine": fine is None}
    ca, fa = coarse["attrs"], fine["attrs"]
    result: dict[str, Any] = {"status": "UNKNOWN", "qualified": False, "label_coarse": ca.get("label"), "label_fine": fa.get("label")}
    checks: dict[str, Any] = {}
    checks["label_agreement"] = ca.get("label") == fa.get("label") and ca.get("label") in {"stable", "unstable"}
    checks["signature_agreement"] = ca.get("postfault_signature") == fa.get("postfault_signature")
    delay_c, delay_f = _float(ca.get("record_start_delay_s")), _float(fa.get("record_start_delay_s"))
    checks["origin_agreement"] = delay_c is not None and delay_f is not None and abs(delay_c - delay_f) <= 1e-8
    arrays_c, arrays_f = coarse["arrays"], fine["arrays"]
    required = ("t", "delta", "omega")
    checks["required_arrays"] = all(name in arrays_c and name in arrays_f for name in required)
    finite_names = set(arrays_c) | set(arrays_f)
    checks["finite_all_exported"] = checks["required_arrays"] and all(np.isfinite(arrays_c.get(name, np.array([np.nan]))).all() and np.isfinite(arrays_f.get(name, np.array([np.nan]))).all() for name in finite_names)
    shared_n = min(len(arrays_c.get("t", [])), len(arrays_f.get("t", [])))
    checks["shared_grid"] = shared_n > 0 and np.allclose(arrays_c["t"][:shared_n], arrays_f["t"][:shared_n], atol=1e-8, rtol=0)
    result["shared_points"] = int(shared_n)
    result["shared_duration_s"] = None if shared_n == 0 else float(min(arrays_c["t"][shared_n - 1], arrays_f["t"][shared_n - 1]))
    if checks["required_arrays"] and shared_n:
        delta = arrays_c["delta"][:shared_n] - arrays_f["delta"][:shared_n]
        omega = (arrays_c["omega"][:shared_n] - arrays_f["omega"][:shared_n]) * omega_s
        dcoi, wcoi = _coi(delta, mass), _coi(omega, mass)
        common = (omega * mass[None, :]).sum(axis=1) / mass.sum()
        result.update({"delta_rmse_normalized": float(np.sqrt(np.mean((dcoi / S_DELTA) ** 2))), "omega_rmse_normalized": float(np.sqrt(np.mean((wcoi / S_OMEGA) ** 2))), "combined_rmse_normalized": float(np.sqrt(0.5 * np.mean((dcoi / S_DELTA) ** 2 + (wcoi / S_OMEGA) ** 2))), "common_frequency_rmse_normalized": float(np.sqrt(np.mean((common / S_OMEGA) ** 2)))})
    label = ca.get("label")
    end_c, end_f = _float(ca.get("trajectory_end_s")), _float(fa.get("trajectory_end_s"))
    checks["future_coverage"] = end_c is not None and end_f is not None and end_c > 0.1 and end_f > 0.1
    if label == "stable":
        checks["stable_15s"] = end_c is not None and end_f is not None and end_c >= 15.0 - 1e-8 and end_f >= 15.0 - 1e-8
    elif label == "unstable":
        event_c, event_f = _float(ca.get("event_time_s")), _float(fa.get("event_time_s"))
        los_c = None if event_c is None or delay_c is None else event_c + delay_c
        los_f = None if event_f is None or delay_f is None else event_f + delay_f
        result["los_since_clearing_coarse_s"], result["los_since_clearing_fine_s"] = los_c, los_f
        result["los_difference_s"] = None if los_c is None or los_f is None else abs(los_c - los_f)
        checks["los_agreement"] = result["los_difference_s"] is not None and result["los_difference_s"] <= 0.02 + 1e-12
    else:
        checks["known_label"] = False
    result["checks"] = checks
    required_checks = [checks["label_agreement"], checks["signature_agreement"], checks["origin_agreement"], checks["required_arrays"], checks["finite_all_exported"], checks["shared_grid"], checks["future_coverage"]]
    required_checks.append(checks.get("stable_15s", True) if label == "stable" else checks.get("los_agreement", False) if label == "unstable" else False)
    metric_ok = all(result.get(name, np.inf) <= 0.001 for name in ("combined_rmse_normalized", "common_frequency_rmse_normalized"))
    result["numerical_metrics_pass"] = bool(metric_ok)
    result["qualified"] = bool(all(required_checks) and metric_ok)
    result["status"] = "QUALIFIED" if result["qualified"] else "UNKNOWN"
    return result


def _source_provenance(coarse: dict[str, Any], fine: dict[str, Any], reference_h5: Path | None) -> dict[str, Any]:
    cp, fp = coarse["provenance"], fine["provenance"]
    checks = {
        "coarse_fine_signature": cp["postfault_signature"] is not None and cp["postfault_signature"] == fp["postfault_signature"],
        "coarse_fine_case": cp["case_sha256"] is not None and cp["case_sha256"] == fp["case_sha256"],
        "coarse_fine_mapping": bool(cp["state_mapping"]) and cp["state_mapping"] == fp["state_mapping"],
        "coarse_fine_inertia": cp["inertia"].size > 0 and bool(np.array_equal(cp["inertia"], fp["inertia"])),
        "coarse_fine_equilibrium": cp["equilibrium_full_state"].size > 0 and cp["equilibrium_full_state"].shape == fp["equilibrium_full_state"].shape and bool(np.max(np.abs(cp["equilibrium_full_state"] - fp["equilibrium_full_state"])) <= 1e-9),
    }
    reference = None
    if reference_h5 is not None:
        if not reference_h5.exists():
            raise FileNotFoundError(reference_h5)
        with h5py.File(reference_h5, "r") as file:
            reference = {"case_sha256": _decode(file.attrs.get("case_sha256")), "postfault_signature": _decode(file.attrs.get("postfault_signature")), "state_mapping": json.loads(_decode(file.attrs.get("state_mapping", "[]"))), "inertia": np.asarray(file["inertia"][...], dtype=float), "equilibrium_full_state": np.asarray(file["equilibrium/full_state"][...], dtype=float)}
        checks.update({"reference_case": cp["case_sha256"] is not None and cp["case_sha256"] == reference["case_sha256"], "reference_signature": cp["postfault_signature"] is not None and cp["postfault_signature"] == reference["postfault_signature"], "reference_mapping": bool(cp["state_mapping"]) and cp["state_mapping"] == reference["state_mapping"], "reference_inertia": cp["inertia"].size > 0 and bool(np.array_equal(cp["inertia"], reference["inertia"])), "reference_equilibrium": cp["equilibrium_full_state"].size > 0 and cp["equilibrium_full_state"].shape == reference["equilibrium_full_state"].shape and bool(np.max(np.abs(cp["equilibrium_full_state"] - reference["equilibrium_full_state"])) <= 1e-9)})
    return {"checks": checks, "reference_h5": None if reference_h5 is None else str(reference_h5), "reference_read_events": False, "source_pass": bool(all(checks.values()))}


def _old_condition_keys(paths: list[Path]) -> set[tuple[Any, ...]]:
    keys = set()
    for root in paths:
        index_path, split_path = root / "dataset_index.csv", root / "splits.json"
        if not index_path.exists() or not split_path.exists():
            continue
        train_ids = set(json.loads(split_path.read_text(encoding="utf-8")).get("train", []))
        with index_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["event_id"] in train_ids:
                    keys.add(_condition(row))
    return keys


def _run_existing_qa(root: Path) -> dict[str, Any]:
    try:
        from audit_private_dataset import audit
        result = audit(root)
        return {"status": "completed", "pilot_ready": bool(result), "pilot_ready_is_not_a_gate": True}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "pilot_ready_is_not_a_gate": True}


def run(coarse_root: Path, fine_root: Path, selection_path: Path, output: Path, *, reference_h5: Path | None = None, old_roots: list[Path] | None = None, run_existing_qa: bool = True) -> dict[str, Any]:
    selection = load_selection(selection_path)
    selection_source = json.loads(selection_path.read_text(encoding="utf-8"))
    coarse = _read_root(coarse_root, selection)
    fine = _read_root(fine_root, selection)
    if coarse["provenance"]["config"].get("integration_step_s") != 0.005 or fine["provenance"]["config"].get("integration_step_s") != 0.0025:
        raise ValueError("coarse/fine integration steps must be .005/.0025")
    for source in (coarse, fine):
        cfg = source["provenance"]["config"]
        if cfg.get("sample_interval_s") != 0.01 or cfg.get("fault_start_s") != 1.0 or cfg.get("postfault_horizon_s") != 15.0 or cfg.get("andes_version") != "2.0.0":
            raise ValueError("boundary root physical configuration differs from preregistration")
    mass = coarse["provenance"]["inertia"]
    if mass.size == 0:
        mass = fine["provenance"]["inertia"]
    if mass.size == 0:
        # No accepted event can reach the metric calculation in this case;
        # keep the 20-attempt result explicitly UNKNOWN instead of crashing.
        mass = np.ones(1, dtype=float)
    provenance = _source_provenance(coarse, fine, reference_h5)
    old_keys = _old_condition_keys(old_roots or [])
    expected_buses = {2, 3, 5, 6, 8, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 28, 29, 31, 32, 33}
    pairs = []
    for config in selection:
        event_id = config["event_id"]
        c_event, f_event = coarse["events"].get(event_id), fine["events"].get(event_id)
        held_out = set(coarse["provenance"]["config"].get("validation_buses", [])) | set(coarse["provenance"]["config"].get("test_buses", []))
        condition_checks = {
            "selected_bus_expected": config["fault_bus"] in expected_buses,
            "not_held_out_bus": config["fault_bus"] not in held_out,
            "coarse_condition_match": c_event is not None and _condition({**c_event["row"], **c_event["attrs"]}) == _condition(config),
            "fine_condition_match": f_event is not None and _condition({**f_event["row"], **f_event["attrs"]}) == _condition(config),
            "duplicate_old_train_condition": _condition(config) in old_keys,
        }
        pair = compare_pair(c_event, f_event, mass=mass, omega_s=coarse["provenance"]["omega_s"])
        pair.update({
            "event_id": event_id,
            "selection": config,
            "condition_checks": condition_checks,
            "coarse_status": "accepted" if c_event is not None else "missing_or_discarded",
            "fine_status": "accepted" if f_event is not None else "missing_or_discarded",
        })
        condition_gate = all(value for name, value in condition_checks.items() if name != "duplicate_old_train_condition") and not condition_checks["duplicate_old_train_condition"]
        pair["qualified"] = bool(pair["qualified"] and condition_gate)
        if not pair["qualified"]:
            pair["status"] = "UNKNOWN"
        pairs.append(pair)
    accepted = [pair for pair in pairs if pair["status"] == "QUALIFIED"]
    raw_counts = {}
    latency_counts = {}
    for name, source in (("coarse", coarse), ("fine", fine)):
        labels = Counter(_metadata_value(event, "label") for event in source["events"].values())
        latency = sum(1 for event in source["events"].values() if (_float(_metadata_value(event, "trajectory_end_s")) or 0.0) <= 0.1)
        raw_counts[name] = dict(labels)
        latency_counts[name] = latency
    result = {"scope": {"coarse_root": str(coarse_root), "fine_root": str(fine_root), "selection": str(selection_path), "selection_sha256": digest(selection_path), "read_old_event_payloads": False, "read_validation_or_test_payloads": False}, "selection_metadata": selection_source, "source_hashes": {"coarse": coarse["hashes"], "fine": fine["hashes"]}, "provenance": provenance, "existing_dataset_qa": {"coarse": _run_existing_qa(coarse_root) if run_existing_qa else {"status": "not_run"}, "fine": _run_existing_qa(fine_root) if run_existing_qa else {"status": "not_run"}}, "interpretation": {"qualified_pairs_are_pairwise_numerical_condition_gates_only": True, "usable_data_requires_source_pass_and_existing_dataset_qa": True}, "attempts": {"configured": 20, "coarse_accepted": len(coarse["events"]), "fine_accepted": len(fine["events"]), "coarse_missing_or_discarded": 20 - len(coarse["events"]), "fine_missing_or_discarded": 20 - len(fine["events"]), "qualified_pairs": len(accepted), "qualified_stable": sum(pair.get("label_coarse") == "stable" for pair in accepted), "qualified_unstable": sum(pair.get("label_coarse") == "unstable" for pair in accepted)}, "discard_counts": {"coarse": coarse["rejected"], "fine": fine["rejected"]}, "raw_label_counts": raw_counts, "latency_counts": latency_counts, "pairs": pairs, "frozen_gate_inputs": {"qualified_at_least_16": len(accepted) >= 16, "qualified_stable_at_least_4": sum(pair.get("label_coarse") == "stable" for pair in accepted) >= 4, "qualified_unstable_at_least_4": sum(pair.get("label_coarse") == "unstable" for pair in accepted) >= 4}, "pilot_ready_is_not_a_gate": True}
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-root", type=Path, default=ROOT / "private_data/ieee39_boundary_0075")
    parser.add_argument("--fine-root", type=Path, default=ROOT / "private_data/ieee39_boundary_fine_0075")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reference-h5", type=Path, default=ROOT / "private_data/ieee39_postfault/dataset.h5")
    parser.add_argument("--old-root", type=Path, action="append", default=[ROOT / "private_data/ieee39_postfault", ROOT / "private_data/ieee39_expansion"])
    parser.add_argument("--output", type=Path, default=ROOT / "private_data/task2_boundary_integrity_0075")
    parser.add_argument("--skip-existing-qa", action="store_true")
    args = parser.parse_args()
    result = run(args.coarse_root, args.fine_root, args.selection, args.output, reference_h5=args.reference_h5, old_roots=args.old_root, run_existing_qa=not args.skip_existing_qa)
    print(json.dumps({"output": str(args.output), "qualified_pairs": result["attempts"]["qualified_pairs"], "coarse_accepted": result["attempts"]["coarse_accepted"], "fine_accepted": result["attempts"]["fine_accepted"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
