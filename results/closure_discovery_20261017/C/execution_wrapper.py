"""Run only the preregistered three fixed-structure C controls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments"))
EXPECTED_IMPLEMENTATION_IDENTITY = "f27cb8860a77551b07bd57749efc13c36f7566151baea49c4193155398c78428"
EXPECTED_SOURCE_SHA = "c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238"
SEEDS = (2701, 2702, 2703)
TARGET_UPDATES = 1024
SAFETY_MARGIN_SECONDS = 1800.0


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _same_or_write(path: Path, value) -> None:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != value:
            raise RuntimeError(f"immutable result differs from existing file: {path}")
        return
    _atomic_json(path, value)


def _write_job(path: Path, job: dict) -> None:
    job["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(path, job)


def main() -> int:
    cfg = json.loads((ROOT / "configs/closure_discovery_20261017.json").read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    cdir = out / "C"
    cdir.mkdir(parents=True, exist_ok=True)
    job_path = cdir / "fixed_controls_job.json"
    result_path = cdir / "predefined_fixed_controls.json"
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    previous_job = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() else {}
    prior_attempts = list(previous_job.get("prior_attempts", []))
    if previous_job:
        prior_attempts.append({
            key: previous_job.get(key)
            for key in (
                "pid", "receipt_id", "started_utc", "started_epoch", "status",
                "interrupted_utc", "actual_process_seconds", "command_execution_id",
                "command_execution_duration_seconds", "command_execution_evidence_path",
                "ledger_receipt_status", "source_identity_sha256_expected",
                "source_sha256_expected", "stage_C_remaining_at_start_seconds",
                "safety_margin_seconds", "recoverable_checkpoint", "interrupted_artifact_archive",
            )
        })
    job = {
        "task": "predefined-fixed-controls-3seeds",
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "started_epoch": time.time(),
        "command": command,
        "cwd": str(ROOT),
        "status": "running",
        "seeds": {str(seed): "pending" for seed in SEEDS},
        "target_updates_each": TARGET_UPDATES,
        "source_identity_sha256_expected": EXPECTED_IMPLEMENTATION_IDENTITY,
        "source_sha256_expected": EXPECTED_SOURCE_SHA,
        "prior_attempts": prior_attempts,
    }
    _write_job(job_path, job)

    try:
        # Importing the frozen implementation and loading train-only inputs occur
        # inside this receipt, so their startup cost is included in stage C.
        from closure_discovery_support import ProcessLedger

        ledger = ProcessLedger(out / "ledger")
        initial_remaining = ledger.remaining("C")
        if initial_remaining <= SAFETY_MARGIN_SECONDS:
            raise RuntimeError(
                f"insufficient C budget after safety margin: remaining={initial_remaining:.3f}s, "
                f"margin={SAFETY_MARGIN_SECONDS:.3f}s"
            )
        deadline = time.perf_counter() + initial_remaining - SAFETY_MARGIN_SECONDS
        with ledger.run("C", "predefined-fixed-controls-3seeds") as receipt_id:
            job["receipt_id"] = receipt_id
            job["stage_C_remaining_at_start_seconds"] = initial_remaining
            job["safety_margin_seconds"] = SAFETY_MARGIN_SECONDS
            _write_job(job_path, job)

            import numpy as np
            import torch
            import discover_closure as dc
            from symbolic_closure_rom import SymbolicClosureROM

            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)

            identity = dc.bind_execution_sources(out)
            if identity != EXPECTED_IMPLEMENTATION_IDENTITY:
                raise RuntimeError(f"frozen implementation identity mismatch: {identity}")
            if dc.sha(out / "implementation" / "identity.json") != EXPECTED_IMPLEMENTATION_IDENTITY:
                raise RuntimeError("implementation identity file hash changed")
            if dc.SOURCE_HASH != EXPECTED_SOURCE_SHA:
                raise RuntimeError("registered source checkpoint hash changed")

            events, excluded, force_scale = dc.train_inputs()
            event_ids = [str(event["id"]) for event in events]
            if len(event_ids) != 346 or len(set(event_ids)) != len(event_ids):
                raise RuntimeError(f"expected the fixed 346 train events, found {len(event_ids)}")

            base = dc.source()
            if len(base.terms) != 9:
                raise RuntimeError(f"expected the original nine source terms, found {len(base.terms)}")
            source_terms = json.loads(json.dumps(base.terms, sort_keys=True))
            source_mask = base.template_mask[3:7].detach().cpu().numpy().tolist()
            source_model_sha = dc.sha(ROOT / cfg["source"])

            # A final manifest is immutable and sufficient for a no-training resume.
            if result_path.exists():
                final = json.loads(result_path.read_text(encoding="utf-8"))
                if (final.get("implementation_identity_sha256") != EXPECTED_IMPLEMENTATION_IDENTITY
                        or final.get("source_sha256") != EXPECTED_SOURCE_SHA
                        or final.get("seeds") != list(SEEDS)
                        or final.get("target_updates_each") != TARGET_UPDATES):
                    raise RuntimeError("existing fixed-control manifest has a different identity")
                for row in final.get("models", []):
                    model_path = Path(row["model_path"])
                    seed_dir = out / "C" / "fixed" / str(row["seed"])
                    if model_path != seed_dir / "model.json" or not model_path.is_file():
                        raise RuntimeError("existing fixed-control endpoint path is invalid")
                    if dc.sha(model_path) != row.get("model_sha256"):
                        raise RuntimeError("existing fixed-control endpoint hash changed")
                    if not (seed_dir / "train_score.json").is_file() or not (seed_dir / "physics.json").is_file():
                        raise RuntimeError("existing fixed-control endpoint is missing train-only evidence")
                if len(final.get("models", [])) + len(final.get("unresolved", [])) != len(SEEDS):
                    raise RuntimeError("existing fixed-control manifest is incomplete")
                job["status"] = "completed" if not final.get("unresolved") else "completed_with_unresolved"
                job["result_manifest"] = str(result_path)
                _write_job(job_path, job)
                return 0

            models = []
            unresolved = []
            fixed_root = out / "C" / "fixed"
            for seed in SEEDS:
                seed_key = str(seed)
                path = fixed_root / seed_key
                path.mkdir(parents=True, exist_ok=True)
                job["seeds"][seed_key] = "running"
                _write_job(job_path, job)

                failure_path = path / "fit_failure.json"
                if failure_path.exists():
                    failure = json.loads(failure_path.read_text(encoding="utf-8"))
                    row = {
                        "seed": seed,
                        "path": str(path),
                        "status": "numerically_unresolved",
                        "reason": failure.get("reason"),
                        "completed_updates": failure.get("completed_updates"),
                        "failure_state_sha256": failure.get("state_sha256"),
                    }
                    unresolved.append(row)
                    job["seeds"][seed_key] = "numerically_unresolved"
                    _write_job(job_path, job)
                    continue

                checkpoint_path = path / "checkpoint.json"
                try:
                    seed_source = dc.source()
                    seed_terms = json.loads(json.dumps(seed_source.terms, sort_keys=True))
                    seed_mask = seed_source.template_mask[3:7].detach().cpu().numpy().tolist()
                    if seed_terms != source_terms or seed_mask != source_mask:
                        raise RuntimeError(f"seed {seed}: fresh source differs from the registered fixed structure")
                    if checkpoint_path.exists():
                        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        if (dc.sha(path / "model.json") != checkpoint.get("model_sha256")
                                or dc.sha(path / "training.pt") != checkpoint.get("training_sha256")):
                            raise RuntimeError(f"seed {seed}: checkpoint integrity mismatch")
                        step = int(checkpoint.get("step", -1))
                        if step < 0 or step > TARGET_UPDATES:
                            raise RuntimeError(f"seed {seed}: checkpoint update count is invalid: {step}")
                        model = SymbolicClosureROM.from_export(path / "model.json")
                        provenance = getattr(model, "provenance", {})
                        model_terms = json.loads(json.dumps(model.terms, sort_keys=True))
                        model_mask = model.template_mask[3:7].detach().cpu().numpy().tolist()
                        if model_terms != source_terms or model_mask != source_mask:
                            raise RuntimeError(f"seed {seed}: checkpoint structure or direction mask changed")
                        if (provenance.get("source_sha256") != EXPECTED_SOURCE_SHA
                                or provenance.get("implementation_identity_sha256") != EXPECTED_IMPLEMENTATION_IDENTITY
                                or provenance.get("fit_seed") != seed
                                or provenance.get("completed_updates_in_current_fit") != step):
                            raise RuntimeError(f"seed {seed}: checkpoint provenance does not match this fresh refit")
                        if step < TARGET_UPDATES:
                            model, fit_info = dc.fit(
                                model, events, force_scale, TARGET_UPDATES, path, seed,
                                final=True, deadline=deadline,
                            )
                        else:
                            fit_info = {"seconds": 0.0, "updates": step, "path": str(path), "reused_checkpoint": True}
                    else:
                        existing = [item for item in path.iterdir()]
                        if existing:
                            raise RuntimeError(f"seed {seed}: partial artifacts lack a recoverable checkpoint")
                        model, fit_info = dc.fit(
                            seed_source, events, force_scale, TARGET_UPDATES, path, seed,
                            final=True, deadline=deadline,
                        )
                except FloatingPointError as error:
                    failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {}
                    unresolved.append({
                        "seed": seed,
                        "path": str(path),
                        "status": "numerically_unresolved",
                        "reason": failure.get("reason", repr(error)),
                        "completed_updates": failure.get("completed_updates"),
                        "failure_state_sha256": failure.get("state_sha256"),
                    })
                    job["seeds"][seed_key] = "numerically_unresolved"
                    _write_job(job_path, job)
                    continue

                final_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if (int(final_checkpoint.get("step", -1)) != TARGET_UPDATES
                        or dc.sha(path / "model.json") != final_checkpoint.get("model_sha256")
                        or dc.sha(path / "training.pt") != final_checkpoint.get("training_sha256")):
                    raise RuntimeError(f"seed {seed}: final checkpoint is not an intact 1024-update endpoint")

                model_sha = dc.sha(path / "model.json")
                model_terms = json.loads(json.dumps(model.terms, sort_keys=True))
                model_mask = model.template_mask[3:7].detach().cpu().numpy().tolist()
                provenance = getattr(model, "provenance", {})
                if model_terms != source_terms or model_mask != source_mask:
                    raise RuntimeError(f"seed {seed}: fitted endpoint structure or direction mask changed")
                if (provenance.get("source_sha256") != EXPECTED_SOURCE_SHA
                        or provenance.get("implementation_identity_sha256") != EXPECTED_IMPLEMENTATION_IDENTITY
                        or provenance.get("fit_seed") != seed
                        or provenance.get("completed_updates_in_current_fit") != TARGET_UPDATES):
                    raise RuntimeError(f"seed {seed}: fitted endpoint provenance is not bound to this seed/source")
                score_path = path / "train_score.json"
                if score_path.exists():
                    score = json.loads(score_path.read_text(encoding="utf-8"))
                    if ([row.get("id") for row in score.get("events", [])] != event_ids
                            or score.get("scope") != "train selection; genuine available intervals"):
                        raise RuntimeError(f"seed {seed}: existing train score identity changed")
                else:
                    try:
                        score = dc.score(model, events, deadline=deadline)
                    except FloatingPointError as error:
                        unresolved.append({"seed": seed, "path": str(path), "status": "train_score_unresolved", "reason": repr(error), "model_sha256": model_sha})
                        job["seeds"][seed_key] = "train_score_unresolved"
                        _write_job(job_path, job)
                        continue
                    _same_or_write(score_path, score)

                physics_path = path / "physics.json"
                if physics_path.exists():
                    physics = json.loads(physics_path.read_text(encoding="utf-8"))
                    if not isinstance(physics.get("pass"), bool):
                        raise RuntimeError(f"seed {seed}: existing physics QA is malformed")
                else:
                    physics = dc.physical_qa(model)
                    _same_or_write(physics_path, physics)

                row = {
                    "name": "fixed",
                    "seed": seed,
                    "path": str(path),
                    "model_path": str(path / "model.json"),
                    "model_sha256": model_sha,
                    "source_model_sha256": source_model_sha,
                    "source_terms": source_terms,
                    "direction_mask": source_mask,
                    "updates": TARGET_UPDATES,
                    "fit_seconds_this_process": float(fit_info.get("seconds", 0.0)),
                    "train_score": {
                        "risk": score.get("risk"),
                        "group_se": score.get("group_se"),
                        "se_method": score.get("se_method"),
                        "event_count": len(score.get("events", [])),
                        "scope": score.get("scope"),
                    },
                    "physics": physics,
                    "status": "completed" if physics.get("pass") else "physics_qa_unresolved",
                }
                if physics.get("pass"):
                    models.append(row)
                    job["seeds"][seed_key] = "completed"
                else:
                    unresolved.append({"seed": seed, "path": str(path), "status": "physics_qa_unresolved", "model_sha256": model_sha})
                    job["seeds"][seed_key] = "physics_qa_unresolved"
                _write_job(job_path, job)

            result = {
                "task": "predefined-fixed-controls-3seeds",
                "implementation_identity_sha256": identity,
                "source_sha256": EXPECTED_SOURCE_SHA,
                "source_model_sha256": source_model_sha,
                "source_terms": source_terms,
                "direction_mask": source_mask,
                "split": "train",
                "train_event_ids": event_ids,
                "train_event_count": len(event_ids),
                "excluded_train_rows": excluded,
                "force_scale": force_scale,
                "seeds": list(SEEDS),
                "target_updates_each": TARGET_UPDATES,
                "models": models,
                "unresolved": unresolved,
                "test_read": False,
                "val_read": False,
                "fom_run": False,
            }
            _same_or_write(result_path, result)
            job["status"] = "completed" if not unresolved else "completed_with_unresolved"
            job["result_manifest"] = str(result_path)
            _write_job(job_path, job)
        return 0
    except BaseException as exc:
        job["status"] = "failed"
        job["error"] = repr(exc)
        job["error_type"] = type(exc).__name__
        _write_job(job_path, job)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
