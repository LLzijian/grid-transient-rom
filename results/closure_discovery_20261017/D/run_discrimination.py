"""Score the two preregistered C predictors on the completed D new-train cohort.

This is a score-only leaf.  It has no fitting, simulator, VAL/TEST, freeze-write,
or candidate-selection path.  A real invocation requires a separately registered
``D/discrimination_execution_order.json`` and is intentionally not part of static
preparation checks.
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
KEYS = ["main:2701", "alternative:2701"]
FULL_C_FROZEN_KEYS = frozenset(
    f"{name}:{seed}"
    for name in ("fixed", "library", "main", "alternative")
    for seed in (2701, 2702, 2703)
) | {"delete3:2701", "delete2:2701"}
SCOPE = "exploratory new train discrimination of frozen predictors, not confirmation or isolated mechanism necessity"
SAFETY_MARGIN_SECONDS = 1800.0


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    """Refresh a unique job record with bounded retry for transient Windows locks."""
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
                time.sleep(0.01 * 2**attempt)
    finally:
        temporary.unlink(missing_ok=True)


def write_once_json(path: Path, value: Any) -> None:
    """Create an immutable result or accept only a byte-equivalent JSON value."""
    primitive = _json_primitive(value)
    canonical = _canonical(primitive)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _canonical(_json_primitive(existing)) != canonical:
            raise RuntimeError(f"immutable discrimination result differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(primitive, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        write_once_json(path, primitive)


def _order_source_identity(order: dict[str, Any]) -> str | None:
    current = order.get("current_source_identity_sha256")
    regular = order.get("source_identity_sha256")
    if current is not None and regular is not None and current != regular:
        raise RuntimeError("execution order contains conflicting source identity hashes")
    value = current if current is not None else regular
    return str(value) if value is not None else None


def validate_execution_order(order: dict[str, Any], expected: dict[str, str], wrapper_sha: str) -> None:
    """Pure contract check, also exercised by static synthetic guards."""
    if order.get("keys") != KEYS:
        raise RuntimeError("discrimination execution order must authorize the two fixed keys in order")
    if order.get("wrapper_sha256") != wrapper_sha:
        raise RuntimeError("discrimination wrapper hash differs from its execution order")
    actual_source = _order_source_identity(order)
    for field, wanted in expected.items():
        actual = actual_source if field == "source_identity_sha256" else order.get(field)
        if actual != wanted:
            raise RuntimeError(f"discrimination execution order mismatch: {field}")


def assert_score_before_adaptation(ddir: Path) -> None:
    """Refuse scoring after a D fit/model/checkpoint or confirmation freeze exists."""
    freeze = ddir / "confirmation_freeze.json"
    if freeze.exists() or freeze.is_symlink():
        raise RuntimeError("discrimination must precede D confirmation freeze")
    models = ddir / "models"
    if models.exists() or models.is_symlink():
        if models.is_symlink():
            raise RuntimeError("D/models is a symlink; score-before-adaptation cannot be verified")
        for path in models.rglob("*"):
            if path.is_symlink() or path.is_file():
                raise RuntimeError(f"D/models contains possible fit/model/checkpoint evidence: {path}")


def validate_loaded_cohort(events: list[dict[str, Any]], excluded: list[dict[str, Any]], recovery_error=RuntimeError) -> bool:
    """Retain every eligible row and exclusion; reject empty or duplicate-ID cohorts."""
    if not events:
        raise recovery_error("registered new TRAIN produced no eligible events")
    ids = [str(event.get("id")) for event in events]
    if any(event.get("id") is None for event in events) or len(ids) != len(set(ids)):
        raise recovery_error("registered new TRAIN contains missing or duplicate event IDs")
    return len(events) == 24 and not excluded


def _check_frozen_models(confirmation, frozen: dict[str, Any], out: Path) -> dict[str, dict[str, Any]]:
    rows = frozen.get("models")
    if not isinstance(rows, list) or not rows:
        raise confirmation.RecoveryUnresolved("C frozen manifest has no complete model cohort")
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            key = f"{row['name']}:{int(row['seed'])}"
        except (KeyError, TypeError, ValueError) as error:
            raise confirmation.RecoveryUnresolved("C frozen model identity is incomplete") from error
        if key in by_key:
            raise confirmation.RecoveryUnresolved(f"duplicate C frozen model identity: {key}")
        by_key[key] = row
    actual_keys = set(by_key)
    if actual_keys != FULL_C_FROZEN_KEYS:
        missing = sorted(FULL_C_FROZEN_KEYS - actual_keys)
        unexpected = sorted(actual_keys - FULL_C_FROZEN_KEYS)
        raise confirmation.RecoveryUnresolved(
            f"C frozen cohort must contain the complete 14-model endpoint set; missing={missing}, unexpected={unexpected}"
        )
    for key, row in by_key.items():
        path = confirmation._model_file(row)
        if confirmation._file_sha256(path) != row.get("sha256"):
            raise confirmation.RecoveryUnresolved(f"C frozen model hash changed: {path}")
    if not set(KEYS) <= set(by_key):
        raise confirmation.RecoveryUnresolved("C frozen cohort lacks main:2701 or alternative:2701")
    return by_key


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0:
        parser.error("--cap-seconds must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    process_started = time.perf_counter()
    args = parse_args(argv)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--cap-seconds", str(args.cap_seconds)]
    config_path = ROOT / "configs/closure_discovery_20261017.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = (ROOT / config["output"]).resolve()
    ddir = out / "D"
    order_path = ddir / "discrimination_execution_order.json"
    frozen_path = out / "C/frozen.json"
    selection_path = ddir / "selected_new_train.json"
    train_dir = ddir / "new_data/train"
    generation_path = train_dir / "generation.json"
    integrity_path = train_dir / "generation.integrity.json"
    freeze_path = ddir / "confirmation_freeze.json"
    script_path = Path(__file__).resolve()
    wrapper_sha = sha256_file(script_path)

    if not order_path.is_file():
        raise RuntimeError(f"registered discrimination execution order is required: {order_path}")
    order = json.loads(order_path.read_text(encoding="utf-8"))
    order_sha = sha256_file(order_path)
    if order.get("wrapper_sha256") != wrapper_sha or order.get("keys") != KEYS:
        raise RuntimeError("discrimination order wrapper/key contract does not match")
    assert_score_before_adaptation(ddir)
    required = (frozen_path, selection_path, generation_path, integrity_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("completed C and D TRAIN artifacts are required: " + ", ".join(missing))

    c_frozen_sha = sha256_file(frozen_path)
    train_generation_sha = sha256_file(generation_path)
    train_integrity_sha = sha256_file(integrity_path)
    selected_sha = sha256_file(selection_path)
    for field, actual in (
        ("c_frozen_sha256", c_frozen_sha),
        ("train_generation_sha256", train_generation_sha),
        ("train_integrity_sha256", train_integrity_sha),
        ("selected_new_train_sha256", selected_sha),
    ):
        if order.get(field) != actual:
            raise RuntimeError(f"discrimination execution order mismatch: {field}")

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
    job_path = ddir / "discrimination" / f"job_{tag}.json"
    job: dict[str, Any] = {
        "task": "D-frozen-predictor-new-train-discrimination",
        "pid": os.getpid(),
        "command": command,
        "cwd": str(Path.cwd()),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "requested_cap_seconds": args.cap_seconds,
        "effective_cap_seconds": effective_cap,
        "D_remaining_at_start_seconds": d_remaining,
        "safety_margin_seconds": SAFETY_MARGIN_SECONDS,
        "execution_order_path": str(order_path),
        "execution_order_sha256": order_sha,
        "wrapper_sha256": wrapper_sha,
        "source_identity_sha256": None,
        "c_frozen_sha256": c_frozen_sha,
        "train_generation_sha256": train_generation_sha,
        "train_integrity_sha256": train_integrity_sha,
        "selected_new_train_sha256": selected_sha,
        "keys": KEYS,
        "status": "running",
    }
    receipt_id: str | None = None
    try:
        with ledger.run("D", "frozen-predictor-new-train-discrimination") as receipt_id:
            job["receipt_id"] = receipt_id
            atomic_json(job_path, job)
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
                raise RuntimeError("discover_closure context differs from the registered configuration")
            source_identity = dc.bind_execution_sources(out)
            job["source_identity_sha256"] = source_identity
            if sha256_file(order_path) != order_sha:
                raise RuntimeError("discrimination execution order changed during startup")
            if (sha256_file(frozen_path) != c_frozen_sha
                    or sha256_file(generation_path) != train_generation_sha
                    or sha256_file(integrity_path) != train_integrity_sha
                    or sha256_file(selection_path) != selected_sha):
                raise RuntimeError("a registered C or D TRAIN artifact changed during startup")
            expected = {
                "source_identity_sha256": source_identity,
                "c_frozen_sha256": c_frozen_sha,
                "train_generation_sha256": train_generation_sha,
                "train_integrity_sha256": train_integrity_sha,
                "selected_new_train_sha256": selected_sha,
            }
            validate_execution_order(order, expected, wrapper_sha)
            atomic_json(job_path, job)

            # Read only immutable metadata after all registered identities match.
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            model_rows = _check_frozen_models(confirmation, frozen, out)
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            conditions = selection.get("conditions")
            if not isinstance(conditions, list) or len(conditions) != 24:
                raise confirmation.RecoveryUnresolved("registered D new TRAIN must preserve all 24 selected conditions")

            generation = json.loads(generation_path.read_text(encoding="utf-8"))
            if generation.get("split") != "train" or generation.get("status") != "complete":
                raise confirmation.RecoveryUnresolved("registered new TRAIN generation is not complete")
            if freeze_path.exists():
                raise RuntimeError("new-train discrimination is forbidden after confirmation freeze exists")

            def forbid_fom(*_args, **_kwargs):
                raise RuntimeError("discrimination may reuse completed TRAIN only; FOM is forbidden")

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("D discrimination cap reached before TRAIN integrity verification")
            generated = confirmation.generate_or_reuse(
                conditions, train_dir, "train", remaining, simulator=forbid_fom
            )
            if not generated.get("_reused"):
                raise confirmation.RecoveryUnresolved("registered TRAIN was not reused; no FOM execution is allowed")
            events, excluded = data.load_new_events(train_dir, "train")
            job.update(
                selected_condition_count=len(conditions),
                eligible_event_count=len(events),
                excluded_events=excluded,
            )
            atomic_json(job_path, job)
            complete_selected_cohort = validate_loaded_cohort(
                events, excluded, confirmation.RecoveryUnresolved
            )
            if time.perf_counter() >= deadline:
                raise TimeoutError("D discrimination cap reached after loading the registered TRAIN cohort")

            result_dir = ddir / "discrimination"
            result_dir.mkdir(parents=True, exist_ok=True)
            completed: dict[str, Any] = {}
            for key in KEYS:
                assert_score_before_adaptation(ddir)
                if time.perf_counter() >= deadline:
                    raise TimeoutError(f"D discrimination deadline reached before {key}")
                row = model_rows[key]
                source_model = confirmation._model_file(row)
                source_model_sha = row["sha256"]
                if confirmation._file_sha256(source_model) != source_model_sha:
                    raise confirmation.RecoveryUnresolved(f"C source model changed before scoring: {source_model}")
                model = SymbolicClosureROM.from_export(source_model)

                # The inner evaluator cache is immutable and identity-bound by evaluate_or_reuse.
                # Visible result files wrap it with the correct exploratory new-train scope.
                cache_path = result_dir / "cache" / f"{row['name']}_{int(row['seed'])}.json"
                visible_path = result_dir / f"{row['name']}_{int(row['seed'])}.json"
                if visible_path.exists() and not cache_path.is_file():
                    raise confirmation.RecoveryUnresolved(
                        f"visible result exists without its immutable evaluator cache: {visible_path}"
                    )
                evaluation = confirmation.evaluate_or_reuse(
                    model, events, cache_path, source_model_sha, deadline=deadline
                )
                if time.perf_counter() >= deadline:
                    raise TimeoutError(f"D discrimination cap reached while evaluating {key}")
                # Require repair04 component completeness even when recovering old files.
                if any("components" not in record for record in evaluation.get("records", [])):
                    raise confirmation.RecoveryUnresolved("evaluation cache lacks repair04 component metrics")
                visible = {
                    "schema": "closure_discrimination_result_v1",
                    "key": key,
                    "scope": SCOPE,
                    "source_model_sha256": source_model_sha,
                    "source_identity_sha256": source_identity,
                    "train_generation_sha256": train_generation_sha,
                    "train_integrity_sha256": train_integrity_sha,
                    "selected_new_train_sha256": selected_sha,
                    "selected_condition_count": len(conditions),
                    "qualified_event_count": len(events),
                    "excluded_events": excluded,
                    "complete_selected_cohort_loaded": complete_selected_cohort,
                    "test_read": False,
                    "val_read": False,
                    "fom_run": False,
                    "fit_run": False,
                    "freeze_written": False,
                    "comparison_written": False,
                    "primary_improvement_computed": False,
                    "evaluation_cache_path": str(cache_path),
                    "evaluation_cache_sha256": sha256_file(cache_path),
                    "evaluation": evaluation,
                }
                # Store the generic evaluator summary without its VAL-specific prose.
                visible["evaluation"] = dict(evaluation)
                visible["evaluation"].pop("scope", None)
                write_once_json(visible_path, visible)
                completed[key] = {
                    "path": str(visible_path),
                    "sha256": sha256_file(visible_path),
                    "cache_path": str(cache_path),
                    "cache_sha256": sha256_file(cache_path),
                    "event_count": len(events),
                }
                job.update(last_key=key, completed=completed)
                atomic_json(job_path, job)

            if time.perf_counter() >= deadline:
                raise TimeoutError("D discrimination cap reached before writing its result manifest")
            assert_score_before_adaptation(ddir)
            manifest_path = result_dir / "manifest.json"
            manifest = {
                "schema": "closure_discrimination_manifest_v1",
                "scope": SCOPE,
                "source_identity_sha256": source_identity,
                "wrapper_sha256": wrapper_sha,
                "execution_order_sha256": order_sha,
                "c_frozen_sha256": c_frozen_sha,
                "train_generation_sha256": train_generation_sha,
                "train_integrity_sha256": train_integrity_sha,
                "selected_new_train_sha256": selected_sha,
                "selected_condition_count": len(conditions),
                "qualified_event_count": len(events),
                "excluded_events": excluded,
                "complete_selected_cohort_loaded": complete_selected_cohort,
                "results": completed,
                "keys": KEYS,
                "comparison_written": False,
                "primary_improvement_computed": False,
                "test_read": False,
                "val_read": False,
                "fom_run": False,
                "fit_run": False,
                "freeze_written": False,
            }
            write_once_json(manifest_path, manifest)
            job.update(
                status="completed",
                result_manifest=str(manifest_path),
                result_manifest_sha256=sha256_file(manifest_path),
                completed_utc=datetime.now(timezone.utc).isoformat(),
                actual_process_seconds=time.perf_counter() - process_started,
                completed=completed,
            )
            atomic_json(job_path, job)
        return 0
    except BaseException as error:
        job.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            completed_utc=datetime.now(timezone.utc).isoformat(),
            actual_process_seconds=time.perf_counter() - process_started,
            receipt_id=receipt_id,
        )
        atomic_json(job_path, job)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
