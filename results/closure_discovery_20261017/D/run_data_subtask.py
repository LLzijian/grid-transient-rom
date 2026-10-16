"""Run only the registered TRAIN or VAL data-generation leaf for stage D.

This wrapper generates/reuses the registered event cohort and records which events
loaded successfully.  It deliberately contains no fitting, evaluation, or freeze-write
path.  Running it is separate from importing or statically checking this file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
SAFETY_MARGIN_SECONDS = 1800.0
GENERATION_MARGIN_SECONDS = 60.0


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_primitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_primitive(value.tolist())
    if hasattr(value, "item"):
        return _json_primitive(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    """Atomically refresh a job record, retrying transient Windows sharing locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(_json_primitive(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.01 * (2**attempt))
    finally:
        temporary.unlink(missing_ok=True)


def write_once_json(path: Path, value: Any) -> None:
    """Create the result record once; a rerun must reproduce it exactly."""
    canonical = json.dumps(_json_primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if json.dumps(_json_primitive(existing), sort_keys=True, separators=(",", ":"), allow_nan=False) != canonical:
            raise RuntimeError(f"existing immutable data result differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_json_primitive(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        write_once_json(path, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "val"), required=True)
    parser.add_argument("--cap-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0:
        parser.error("--cap-seconds must be a finite positive number")
    return args


def _check_order(order: dict[str, Any], expected: dict[str, str], phase: str) -> None:
    if order.get("phase", phase) != phase:
        raise RuntimeError("data execution order phase mismatch")
    mismatches = [key for key, value in expected.items() if order.get(key) != value]
    if mismatches:
        raise RuntimeError("data execution order identity mismatch: " + ", ".join(mismatches))


def _verified_c_rows(confirmation, frozen: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = frozen.get("models")
    if not isinstance(rows, list) or not rows:
        raise confirmation.RecoveryUnresolved("C frozen manifest has no complete model list")
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.get('name')}:{int(row.get('seed'))}"
        if key in by_key:
            raise confirmation.RecoveryUnresolved(f"duplicate C model identity: {key}")
        model_path = confirmation._model_file(row)
        if confirmation._file_sha256(model_path) != row.get("sha256"):
            raise confirmation.RecoveryUnresolved(f"C source model hash changed: {model_path}")
        by_key[key] = row
    return rows, by_key


def main(argv: list[str] | None = None) -> int:
    process_started = time.perf_counter()
    args = parse_args(argv)
    config_path = ROOT / "configs/closure_discovery_20261017.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = (ROOT / config["output"]).resolve()
    ddir = out / "D"
    phase = args.phase
    order_path = ddir / f"data_{phase}_execution_order.json"
    conditions_path = out / "new_fault_manifest.json"
    frozen_path = out / "C/frozen.json"
    freeze_path = ddir / "confirmation_freeze.json"
    if not order_path.is_file() or not conditions_path.is_file() or not frozen_path.is_file():
        raise RuntimeError("registered data execution order, conditions manifest, and C frozen manifest are required")
    if phase == "train" and freeze_path.exists():
        raise RuntimeError("TRAIN generation is forbidden after D confirmation freeze exists")
    if phase == "val" and not freeze_path.is_file():
        raise RuntimeError("VAL generation requires an existing confirmation freeze")

    script_path = Path(__file__).resolve()
    wrapper_sha = sha(script_path)
    order_sha = sha(order_path)
    c_frozen_sha = sha(frozen_path)
    conditions_sha = sha(conditions_path)
    confirmation_freeze_sha = sha(freeze_path) if phase == "val" else None

    from closure_discovery_support import ProcessLedger

    ledger = ProcessLedger(out / "ledger")
    d_remaining = ledger.remaining("D")
    effective_cap = min(args.cap_seconds, d_remaining - SAFETY_MARGIN_SECONDS)
    if effective_cap <= 0:
        raise RuntimeError(
            f"insufficient D budget after safety margin: remaining={d_remaining:.3f}s, "
            f"margin={SAFETY_MARGIN_SECONDS:.0f}s"
        )
    deadline = process_started + effective_cap
    tag = f"{time.time_ns()}_{os.getpid()}"
    job_path = ddir / f"data_{phase}_job_{tag}.json"
    result_path = ddir / f"data_{phase}_result.json"
    command = [sys.executable, "-u", str(script_path), "--phase", phase,
               "--cap-seconds", str(args.cap_seconds)]
    job: dict[str, Any] = {
        "task": "closure-discovery-stage-D-data",
        "phase": phase,
        "pid": os.getpid(),
        "command": command,
        "cwd": str(Path.cwd()),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "started_epoch": time.time(),
        "requested_cap_seconds": args.cap_seconds,
        "D_remaining_at_start_seconds": d_remaining,
        "effective_cap_seconds": effective_cap,
        "safety_margin_seconds": SAFETY_MARGIN_SECONDS,
        "execution_order_path": str(order_path),
        "execution_order_sha256": order_sha,
        "source_identity_sha256": None,
        "wrapper_sha256": wrapper_sha,
        "c_frozen_sha256": c_frozen_sha,
        "conditions_manifest_sha256": conditions_sha,
        "confirmation_freeze_sha256": confirmation_freeze_sha,
        "status": "running",
    }

    def save_job(**updates: Any) -> None:
        job.update(updates)
        job["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(job_path, job)

    save_job()
    receipt_id: str | None = None
    try:
        with ledger.run("D", f"new-data-{phase}") as receipt_id:
            save_job(receipt_id=receipt_id)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                os.environ[key] = "1"

            import torch
            import discover_closure as dc
            import closure_discovery_confirmation as confirmation
            import closure_discovery_data as data
            from symbolic_closure_rom import SymbolicClosureROM

            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            runtime_config, runtime_out = dc.context()
            if runtime_out.resolve() != out or runtime_config != config:
                raise RuntimeError("discover_closure context differs from the registered output/config")
            source_identity = dc.bind_execution_sources(out)
            expected = {
                "source_identity_sha256": source_identity,
                "wrapper_sha256": wrapper_sha,
                "c_frozen_sha256": c_frozen_sha,
                "conditions_manifest_sha256": conditions_sha,
            }
            if phase == "val":
                expected["confirmation_freeze_sha256"] = str(confirmation_freeze_sha)
            order = json.loads(order_path.read_text(encoding="utf-8"))
            _check_order(order, expected, phase)
            save_job(source_identity_sha256=source_identity)

            # Read the immutable condition/model metadata only after all source identities match.
            manifest = json.loads(conditions_path.read_text(encoding="utf-8"))
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            frozen_models, model_rows = _verified_c_rows(confirmation, frozen, out)
            if time.perf_counter() >= deadline:
                raise TimeoutError("D data task exceeded its process cap during identity verification")

            if phase == "train":
                if freeze_path.exists():
                    raise RuntimeError("TRAIN generation is forbidden after D confirmation freeze exists")
                main_row = model_rows.get("main:2701")
                alternative_row = model_rows.get("alternative:2701")
                if main_row is None or alternative_row is None:
                    raise confirmation.RecoveryUnresolved("C frozen cohort lacks main/alternative seed 2701")
                main = SymbolicClosureROM.from_export(confirmation._model_file(main_row))
                alternative = SymbolicClosureROM.from_export(confirmation._model_file(alternative_row))
                old_train, excluded_old, _force_scale = dc.train_inputs()
                old_ids = [str(event["id"]) for event in old_train]
                if len(old_ids) != 346 or len(set(old_ids)) != 346:
                    raise confirmation.RecoveryUnresolved(
                        f"expected 346 unique old train events, found {len(old_ids)}"
                    )
                conditions = confirmation._select_or_reuse_train_conditions(
                    out, manifest, frozen_models, old_train, main, alternative
                )
                if len(conditions) != 24:
                    raise confirmation.RecoveryUnresolved("registered TRAIN selection must contain 24 conditions")
                generation_dir = ddir / "new_data/train"
                generation_cap = deadline - time.perf_counter() - GENERATION_MARGIN_SECONDS
                if generation_cap <= 0:
                    raise TimeoutError("less than 60 seconds remain for TRAIN generation")
                save_job(
                    phase_generation="requested",
                    attempted_conditions=len(conditions),
                    generation_reused=None,
                    generator_policy="generate_or_reuse owns both registered step sizes and preserves failed/unresolved attempts",
                )
                generated = confirmation.generate_or_reuse(
                    conditions, generation_dir, "train", generation_cap
                )
                generation_reused = bool(generated.get("_reused"))
                save_job(
                    phase_generation="completed",
                    attempted_conditions=len(conditions),
                    generation_reused=generation_reused,
                    fom_run=not generation_reused,
                )
                new_events, excluded_new = data.load_new_events(generation_dir, "train")
                exclusions = {"old_train": excluded_old, "new_train": excluded_new}
                result = {
                    "phase": "train",
                    "source_identity_sha256": source_identity,
                    "wrapper_sha256": wrapper_sha,
                    "c_frozen_sha256": c_frozen_sha,
                    "conditions_manifest_sha256": conditions_sha,
                    "selected_conditions_count": len(conditions),
                    "attempted_conditions": len(conditions),
                    "selected_conditions_sha256": sha(ddir / "selected_new_train.json"),
                    "old_train_event_count": len(old_train),
                    "new_train_event_count": len(new_events),
                    "exclusion_counts": {key: len(value) for key, value in exclusions.items()},
                    "exclusions": exclusions,
                    "generation_status": generated.get("status"),
                    "generation_reused": generation_reused,
                    "generator_policy": "registered dual-step-size generation; failed/unresolved attempts are preserved without replacement sampling",
                    "generation_sha256": sha(generation_dir / "generation.json"),
                    "generation_integrity_sha256": sha(generation_dir / "generation.integrity.json"),
                    "test_read": False,
                    "val_read": False,
                    "fom_run": not generation_reused,
                    "fit_run": False,
                    "evaluation_run": False,
                    "freeze_written": False,
                }
            else:
                if not freeze_path.is_file():
                    raise RuntimeError("VAL generation requires an existing confirmation freeze")
                freeze, freeze_reused = confirmation._freeze_or_reuse(
                    freeze_path, None, manifest["val24"], conditions_path,
                    source_models=frozen_models,
                )
                if not freeze_reused:
                    raise confirmation.RecoveryUnresolved("VAL data task must only reuse an existing confirmation freeze")
                if len(manifest["val24"]) != 24:
                    raise confirmation.RecoveryUnresolved("registered VAL cohort must contain exactly 24 conditions")
                confirmation._verify_model_rows(freeze["models"], frozen_models, out)
                actual_freeze_sha = sha(freeze_path)
                if actual_freeze_sha != confirmation_freeze_sha:
                    raise confirmation.RecoveryUnresolved("confirmation freeze changed after startup")
                generation_dir = ddir / "new_data/val"
                generation_cap = deadline - time.perf_counter() - GENERATION_MARGIN_SECONDS
                if generation_cap <= 0:
                    raise TimeoutError("less than 60 seconds remain for VAL generation")
                save_job(
                    phase_generation="requested",
                    attempted_conditions=len(manifest["val24"]),
                    generation_reused=None,
                    generator_policy="generate_or_reuse owns both registered step sizes and preserves failed/unresolved attempts",
                )
                generated = confirmation.generate_or_reuse(
                    manifest["val24"], generation_dir, "val", generation_cap,
                    freeze_path=freeze_path,
                )
                generation_reused = bool(generated.get("_reused"))
                save_job(
                    phase_generation="completed",
                    attempted_conditions=len(manifest["val24"]),
                    generation_reused=generation_reused,
                    fom_run=not generation_reused,
                )
                val_events, excluded_val = data.load_new_events(
                    generation_dir, "val", freeze_path=freeze_path
                )
                result = {
                    "phase": "val",
                    "source_identity_sha256": source_identity,
                    "wrapper_sha256": wrapper_sha,
                    "c_frozen_sha256": c_frozen_sha,
                    "conditions_manifest_sha256": conditions_sha,
                    "confirmation_freeze_sha256": actual_freeze_sha,
                    "val_condition_count": len(manifest["val24"]),
                    "attempted_conditions": len(manifest["val24"]),
                    "val_event_count": len(val_events),
                    "exclusion_count": len(excluded_val),
                    "exclusions": excluded_val,
                    "freeze_reused": True,
                    "generation_status": generated.get("status"),
                    "generation_reused": generation_reused,
                    "generator_policy": "registered dual-step-size generation; failed/unresolved attempts are preserved without replacement sampling",
                    "generation_sha256": sha(generation_dir / "generation.json"),
                    "generation_integrity_sha256": sha(generation_dir / "generation.integrity.json"),
                    "test_read": False,
                    "fom_run": not generation_reused,
                    "fit_run": False,
                    "evaluation_run": False,
                    "freeze_written": False,
                }

            write_once_json(result_path, result)
            save_job(result_path=str(result_path), result_sha256=sha(result_path), receipt_id=receipt_id)
        save_job(
            status="completed",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            actual_process_seconds=time.perf_counter() - process_started,
            receipt_id=receipt_id,
        )
        return 0
    except BaseException as error:
        save_job(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            finished_utc=datetime.now(timezone.utc).isoformat(),
            actual_process_seconds=time.perf_counter() - process_started,
            receipt_id=receipt_id,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
