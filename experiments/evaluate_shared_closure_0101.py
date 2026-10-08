"""Bounded endpoint evaluation for the 0101 shared-closure F/S comparison.

This module reads only explicit TRAIN and validation cohorts. Checkpoint loading
is owned by ``shared_closure_rom_0101``; this wrapper does not patch legacy
loaders or call TEST payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

# Include heavyweight scientific-library initialization in the evaluator cap.
PROCESS_STARTED = time.perf_counter()
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

EXPECTED_TRAIN = (346, 23, 35, 24)
EXPECTED_VALIDATION = (107, 10, 12)
MAX_CAP_SECONDS = 750.0
MEMORY_CAP_BYTES = 8 * 1024**3
TRAIN_BATCH_SIZE = 32
SERIALIZATION_RESERVE_SECONDS = 10.0


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _progress(output: Path, *, phase: str, **details: Any) -> None:
    _write_json(output / "progress.json", {"loop": "0101", "phase": phase,
                                            "test_read": False, **details})


def _model_family(checkpoint: Path, model: Any) -> str:
    meta_path = checkpoint / "checkpoint.json"
    model_kind = getattr(model, "model_kind", None)
    if model_kind not in ("S23", "F23"):
        raise ValueError(f"loaded model must declare model_kind S23/F23, got {model_kind!r}")
    values: list[str] = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_kind = meta.get("model_kind")
        if meta_kind is None and isinstance(meta.get("config"), dict):
            meta_kind = meta["config"].get("model_kind")
        if meta_kind is not None and meta_kind != model_kind:
            raise ValueError(f"checkpoint/model kind mismatch: metadata={meta_kind!r}, loaded={model_kind!r}")
        meta_arm = str(meta.get("arm", "")).lower().replace("_", "-")
        if "f23" in meta_arm and model_kind != "F23":
            raise ValueError("checkpoint arm says F23 but loader returned S23")
        if "s23" in meta_arm and model_kind != "S23":
            raise ValueError("checkpoint arm says S23 but loader returned F23")
    arm = getattr(model, "arm", None)
    if arm is not None and arm != "L10-S":
        raise ValueError(f"0101 endpoint must preserve L10-S physical/encoder organization, got {arm!r}")
    return "F" if model_kind == "F23" else "S"


def _checkpoint_paths(checkpoint: Path) -> dict[str, dict[str, str]]:
    result = {}
    for path in sorted(p for p in checkpoint.iterdir() if p.is_file()):
        result[f"checkpoint/{path.name}"] = {"path": str(path.resolve()), "sha256": digest(path)}
    if not any(name in result for name in ("checkpoint/numeric_export.json",
                                           "checkpoint/structured_numeric_export.json")):
        raise FileNotFoundError("checkpoint directory requires numeric_export.json or structured_numeric_export.json")
    return result


def _source_paths(checkpoint: Path) -> dict[str, Path]:
    from train_local_memory_0098 import sources

    paths = {key: Path(row["path"]) for key, row in sources().items()}
    extra = {
        "root0101_context": ROOT / "reports/task2_shared_closure_FS_0101.md",
        "evaluation_code": Path(__file__),
        "model_code": ROOT / "experiments/shared_closure_rom_0101.py",
        "validation_evaluator": ROOT / "experiments/evaluate_local_memory_0098.py",
        "trajectory_evaluator": ROOT / "experiments/evaluate_structured_rom.py",
        "train_metric_code": ROOT / "experiments/summarize_train_local_memory_0098.py",
        "development_reader": ROOT / "experiments/train_joint_latent.py",
        "validation_start_reader": ROOT / "experiments/evaluate_l1_latent.py",
        "long_metric_code": ROOT / "experiments/evaluate_long_latent.py",
        "frequency_metric_code": ROOT / "experiments/evaluate_l1_latent.py",
        "mode_metric_code": ROOT / "experiments/compare_physical_modes.py",
        "canonicalization_code": ROOT / "src/kiki/data/private_dataset.py",
        "m3_model_code": ROOT / "src/kiki/models/private_pm.py",
        "network_metadata": ROOT / "private_data/task2_electrical_ports_0048/physical_network.npz",
        "checkpoint_directory": checkpoint,
    }
    for key, path in extra.items():
        paths[key] = path
    return paths


def _hash_sources(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    result = {}
    for key, path in paths.items():
        if key == "checkpoint_directory":
            result.update(_checkpoint_paths(path))
        else:
            result[key] = {"path": str(path.resolve()), "sha256": digest(path)}
    return result


def _check_cohorts(train: list[dict[str, Any]], excluded: dict[str, Any],
                   validation: list[dict[str, Any]], latency: list[dict[str, Any]]) -> None:
    excluded_train = sum(split == "train" for split in excluded.values())
    excluded_validation = sum(split == "validation" for split in excluded.values())
    got_train = (len(train), excluded_train, sum(e["label"] == "unstable" for e in train),
                 len({e["group"] for e in train}))
    if got_train != EXPECTED_TRAIN or len({e["id"] for e in train}) != EXPECTED_TRAIN[0]:
        raise ValueError(f"0101 TRAIN cohort mismatch: {got_train}")
    if (len(validation) != EXPECTED_VALIDATION[0] or len(latency) != EXPECTED_VALIDATION[1]
            or excluded_validation != EXPECTED_VALIDATION[1]):
        raise ValueError("0101 validation eligible/raw latency denominators changed")
    if sum(e["label"] == "unstable" for e in validation) != EXPECTED_VALIDATION[2]:
        raise ValueError("0101 eligible validation unstable denominator changed")


def _memory_bytes() -> int:
    from train_local_memory_0098 import process_memory_bytes
    return int(process_memory_bytes())


def _array_digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _check_deadline(deadline: float, where: str) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError(f"0101 endpoint cap reached during {where}")
    current = _memory_bytes()
    if current > MEMORY_CAP_BYTES:
        raise MemoryError(f"0101 endpoint memory cap exceeded during {where}: {current}")


def _energy_power_result(model: Any, family: str) -> dict[str, Any]:
    if family == "F":
        return {"status": "not_applicable", "pass": None,
                "reason": "L10-F23 is an ordinary RHS model with no Hamiltonian/J/R power certificate interface"}
    from evaluate_local_memory_0098 import energy_power_qa
    return energy_power_qa(model)


def _train_summary(model: Any, events: list[dict[str, Any]], output: Path,
                   deadline: float) -> dict[str, Any]:
    from evaluate_structured_rom import _rollouts
    from summarize_train_local_memory_0098 import (
        HORIZONS, build_summary, event_metrics,
    )

    records = {key: [] for key in ("records", "common_frequency", "normalized_p",
                                    "objective_point_error", "numerical")}
    arrays: dict[str, Any] = {
        "event_ids": np.asarray([e["id"] for e in events]),
        "labels": np.asarray([e["label"] for e in events]),
        "groups": np.asarray([e["group"] for e in events]),
        "full_truth_lengths": np.asarray([len(e["x"]) for e in events]),
    }
    fine_blocks, coarse_blocks, event_info, blocks = [], [], [], []
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    for first in range(0, len(events), TRAIN_BATCH_SIZE):
        _check_deadline(deadline, f"TRAIN block {first}")
        block = events[first:first + TRAIN_BATCH_SIZE]
        histories = np.stack([e["x"][:11] for e in block])
        fine = _rollouts(model, histories, 1600, .005)
        coarse = _rollouts(model, histories, 800, .01)
        _check_deadline(deadline, f"TRAIN block {first} rollout")
        for event, f, c in zip(block, fine, coarse, strict=True):
            metrics = event_metrics(model, event, f, c)
            for key, value in metrics.items():
                records[key].extend(value)
            event_info.append({"event_id": event["id"], "group": event["group"],
                               "label": event["label"], "truth_length": len(event["x"]),
                               "last_true_record_time_s": (len(event["x"]) - 1) * .01,
                               "scored_prefix_steps": min(800, len(event["x"]) - 11),
                               "canonical_train_truth_sha256": _array_digest(event["x"])})
        fine_blocks.append(fine[..., :30])
        coarse_blocks.append(coarse[..., :30])
        arrays.update({f"{e['id']}_truth": np.asarray(e["x"][:811, :30]) for e in block})
        blocks.append({"first_event_index": first, "count": len(block),
                       "elapsed_seconds": time.perf_counter() - started})
        _write_json(output / "partial_train_rows.json", records)
        _progress(output, phase="train", completed_train_events=min(first + len(block), len(events)),
                  total_train_events=len(events), block_count=len(blocks),
                  memory_bytes=_memory_bytes())
    expected_ids = [event["id"] for event in events]
    actual_ids = [row["event_id"] for row in event_info]
    if actual_ids != expected_ids or len(event_info) != len(events):
        raise AssertionError("TRAIN event metadata must preserve every event exactly once and in source order")
    arrays.update(fine_predictions=np.concatenate(fine_blocks),
                  coarse_predictions=np.concatenate(coarse_blocks))
    for key in ("records", "common_frequency", "normalized_p"):
        arrays[f"{key}_rmse"] = np.asarray([
            np.nan if r.get("rmse") is None else r["rmse"] for r in records[key]
        ]).reshape(len(events), len(HORIZONS))
        arrays[f"{key}_available"] = np.asarray([r["available"] for r in records[key]]).reshape(len(events), len(HORIZONS))
        arrays[f"{key}_numerically_admissible"] = np.asarray(
            [r["numerically_admissible"] for r in records[key]]
        ).reshape(len(events), len(HORIZONS))
    severe = {key: [r for r in records[key] if r["horizon"] == "prefix" and r["label"] == "unstable"]
              for key in ("records", "common_frequency", "normalized_p", "objective_point_error")}
    if any(len(rows) != 35 for rows in severe.values()):
        raise AssertionError("all 35 true TRAIN unstable prefixes must remain present")
    result = {
        "loop": "0101", "scope": "fixed-endpoint descriptive TRAIN capability report; not selection or confirmation",
        "test_read": False, "validation_read": False,
        "denominators": {"raw_train": 369, "eligible_train": 346, "latency_uncovered": 23,
                         "stable": 311, "unstable": 35, "fault_groups": 24},
        "complete_counts": {str(h): sum(len(e["x"]) - 11 >= h * 100 for e in events) for h in (1, 2, 5, 8)},
        "prefix_scope": "genuine available future prefix after x[10], capped at 8s; no fabricated tail",
        "dt": {"fine": .005, "coarse": .01, "finer": None},
        "refinement_executed": False,
        "numerical_rule": "per actual scored interval; full fine substeps finite and normalized COI/common discrepancy <=.001",
        "summary": build_summary(records), "unstable_prefix_event_rows": severe,
        "event_metadata": event_info, "records": records, "batch_traces": blocks,
    }
    np.savez_compressed(output / "trajectories_and_metrics.npz", **arrays)
    _write_json(output / "per_event_rows.json", records)
    _write_json(output / "evidence.json", result)
    return result


def _run_worker(checkpoint: Path, output: Path, deadline: float) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = PROCESS_STARTED
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    source_paths = _source_paths(checkpoint)
    sources_before = _hash_sources(source_paths)
    _progress(output, phase="load", source_hashes_before=sources_before)

    from evaluate_l1_latent import _read_validation_starts
    from evaluate_local_memory_0098 import modal
    from evaluate_structured_rom import _safe, evaluate
    from kiki.models.private_pm import MechanicalROM
    from train_joint_latent import BASE, NETWORK, read_development
    from shared_closure_rom_0101 import load_model

    model = load_model(checkpoint)
    family = _model_family(checkpoint, model)
    model.eval()
    if _memory_bytes() > MEMORY_CAP_BYTES:
        raise MemoryError("0101 model load exceeded 8 GiB process cap")
    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False))
    network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]])
    network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    development, excluded, identities = read_development(base, network, full_train=True)
    starts, latency = _read_validation_starts(base.pm_star)
    train = development["train"]
    validation = development["validation"]
    _check_cohorts(train, excluded, validation, latency)
    validation_hashes = {event["id"]: _array_digest(event["x"]) for event in validation}
    validation_start_hashes = {eid: _array_digest(np.asarray(row["history"]))
                               for eid, row in starts.items()}
    _progress(output, phase="cohorts_loaded", family=family,
              denominators={"train": len(train), "validation": len(validation),
                            "unstable_train": sum(e["label"] == "unstable" for e in train),
                            "unstable_validation": sum(e["label"] == "unstable" for e in validation)},
              identities=identities, excluded=excluded)

    train_result = _train_summary(model, train, output / "train", deadline)
    _check_deadline(deadline, "before validation")
    metadata = {"loop": "0101", "family": family, "checkpoint": str(checkpoint.resolve()),
                "source_hashes_before": sources_before, "identities": identities,
                "excluded": excluded, "canonical_validation_truth_sha256": validation_hashes,
                "causal_validation_start_sha256": validation_start_hashes,
                "latency_rows": [{k: v for k, v in row.items()
                                  if k not in ("inertia", "equilibrium_delta")} for row in latency]}
    _progress(output, phase="validation", family=family,
              completed_train_events=len(train), total_train_events=len(train))
    validation_result = evaluate(model, validation, starts, output=output / "validation",
                                 metadata=metadata)
    _check_deadline(deadline, "after validation")
    _progress(output, phase="modal", family=family)
    modes = modal(model, output=output / "modal")
    _check_deadline(deadline, "after modal")
    power = _energy_power_result(model, family)
    sources_after = _hash_sources(source_paths)
    unchanged = sources_before == sources_after
    result = {
        "loop": "0101", "family": family, "checkpoint": str(checkpoint.resolve()),
        "status": "complete" if unchanged else "quarantined_source_drift",
        "test_read": False, "test_enumerated": False,
        "validation_read": True, "train_read": True,
        "scope": "0101 fixed endpoint F/S evaluation; descriptive, not selection or independent confirmation",
        "cap_deadline": deadline,
        "source_hashes_before": sources_before, "source_hashes_after": sources_after,
        "sources_unchanged": unchanged,
        "denominators": {"train": EXPECTED_TRAIN[0], "validation": EXPECTED_VALIDATION[0],
                         "raw_validation": 117, "validation_latency_uncovered": 10,
                         "unstable_train": 35, "unstable_validation": 12},
        "train_evidence": "train/evidence.json", "validation_evidence": "validation/evidence.json",
        "modal_evidence": "modal/evidence.json", "train_summary": train_result["summary"],
        "validation_model_numerical_gate_pass": validation_result["model_numerical_gate_pass"],
        "validation_l1_counts": validation_result["l1_numerically_gated_counts"],
        "modal_qa_pass": modes["qa"]["pass"], "modal_match_count": modes["matched_count"],
        "energy_power_qa": power,
        "L2": "not_established",
    }
    _write_json(output / "evidence.json", result)
    return result


def run(checkpoint: str | Path, output: str | Path,
        cap_seconds: float = MAX_CAP_SECONDS) -> dict[str, Any]:
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    cap_seconds = float(cap_seconds)
    if not (0 < cap_seconds <= MAX_CAP_SECONDS):
        raise ValueError(f"cap_seconds must be in (0, {MAX_CAP_SECONDS:g}]")
    if output.exists():
        raise FileExistsError(output)
    started = PROCESS_STARTED
    deadline = started + cap_seconds
    worker_deadline = deadline - SERIALIZATION_RESERVE_SECONDS
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--checkpoint", str(checkpoint), "--output", str(output),
               "--deadline", str(worker_deadline)]
    timeout_seconds = max(0.0, worker_deadline - time.perf_counter())
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
        output.mkdir(parents=True, exist_ok=True)
        evidence = {"loop": "0101", "status": "evaluation_timeout", "test_read": False,
                    "test_enumerated": False, "checkpoint": str(checkpoint),
                    "cap_seconds": cap_seconds, "supervisor_seconds": elapsed,
                    "within_cap": elapsed <= cap_seconds,
                    "budget_exception": elapsed > cap_seconds,
                    "progress": progress, "stdout_tail": _tail(exc.stdout),
                    "stderr_tail": _tail(exc.stderr),
                    "limitation": "worker terminated at the registered evaluation deadline"}
        _write_json(output / "evidence.json", evidence)
        return evidence
    elapsed = time.perf_counter() - started
    evidence_path = output / "evidence.json"
    if completed.returncode == 0 and evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence.pop("seconds", None)
        evidence["supervisor_seconds"] = elapsed
        evidence["cap_seconds"] = cap_seconds
        evidence["within_cap"] = elapsed <= cap_seconds
        evidence["budget_exception"] = elapsed > cap_seconds
        evidence["worker_returncode"] = completed.returncode
        evidence["worker_stdout_tail"] = _tail(completed.stdout)
        _write_json(evidence_path, evidence)
        return evidence
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
    failure_path = output / "worker_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else None
    error_type = None if failure is None else failure.get("error_type")
    status = ("evaluation_timeout" if error_type == "TimeoutError" else
              "resource_limited" if error_type == "MemoryError" else "worker_failed")
    evidence = {"loop": "0101", "status": status, "test_read": False,
                "test_enumerated": False, "checkpoint": str(checkpoint),
                "cap_seconds": cap_seconds, "supervisor_seconds": elapsed,
                "within_cap": elapsed <= cap_seconds,
                "budget_exception": elapsed > cap_seconds,
                "worker_failure": failure,
                "worker_returncode": completed.returncode, "progress": progress,
                "stdout_tail": _tail(completed.stdout), "stderr_tail": _tail(completed.stderr)}
    _write_json(output / "evidence.json", evidence)
    return evidence


def _tail(text: Any, limit: int = 4000) -> str | None:
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return str(text)[-limit:]


def main() -> None:
    parser = argparse.ArgumentParser(description="0101 shared-closure endpoint evaluation; TRAIN/validation only")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=MAX_CAP_SECONDS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            result = _run_worker(args.checkpoint.resolve(), args.output.resolve(),
                                 float(args.deadline or time.perf_counter() + args.cap_seconds))
            print(json.dumps({"status": result["status"], "family": result["family"]}), flush=True)
        except Exception as exc:
            output = args.output.resolve()
            output.mkdir(parents=True, exist_ok=True)
            _write_json(output / "worker_failure.json", {
                "loop": "0101", "status": "worker_exception", "test_read": False,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            raise
    else:
        result = run(args.checkpoint, args.output, args.cap_seconds)
        print(json.dumps({"status": result["status"], "family": result.get("family"),
                          "output": str(args.output),
                          "supervisor_seconds": result.get("supervisor_seconds")}), flush=True)


if __name__ == "__main__":
    main()
