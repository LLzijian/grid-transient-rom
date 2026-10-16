"""Run only registered per-key adapt/evaluate leaves of confirmation stage D."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
SOURCE_HASH = "c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238"
SAFETY_MARGIN_SECONDS = 1800.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        for attempt in range(8):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.01 * 2**attempt)
    finally:
        tmp.unlink(missing_ok=True)


def immutable_json(path: Path, value) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"immutable result differs from existing file: {path}")
    else:
        atomic_json(path, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("adapt", "evaluate"), required=True)
    parser.add_argument("--keys", nargs="+", required=True, help="name:seed endpoint keys")
    parser.add_argument("--cap-seconds", type=float, required=True)
    args = parser.parse_args()
    if len(set(args.keys)) != len(args.keys) or any(k.count(":") != 1 for k in args.keys):
        parser.error("--keys must be unique name:seed values")
    try:
        for key in args.keys:
            name, seed = key.split(":")
            if not name or int(seed) <= 0:
                raise ValueError
    except ValueError:
        parser.error("each key must be a valid name:positive_seed pair")
    if args.cap_seconds <= 0:
        parser.error("--cap-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    cfg = json.loads((ROOT / "configs/closure_discovery_20261017.json").read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    ddir = out / "D"
    order_path = ddir / f"parallel_{args.phase}_execution_order.json"
    frozen_path = out / "C/frozen.json"
    if not order_path.is_file() or not frozen_path.is_file():
        raise RuntimeError("phase execution order and C frozen endpoint manifest must exist")

    from closure_discovery_support import ProcessLedger
    ledger = ProcessLedger(out / "ledger")
    remaining = ledger.remaining("D")
    budget = min(args.cap_seconds, remaining - SAFETY_MARGIN_SECONDS)
    if budget <= 0:
        raise RuntimeError(f"insufficient D budget after safety margin: remaining={remaining:.3f}s")
    deadline = time.perf_counter() + budget
    started = time.perf_counter()
    tag = f"{time.time_ns()}_{os.getpid()}"
    joined = "_".join(k.replace(":", "-") for k in args.keys)
    job_path = ddir / f"parallel_{args.phase}_{joined}_{tag}_job.json"
    result_path = ddir / f"parallel_{args.phase}_{joined}_{tag}_result.json"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", args.phase,
               "--keys", *args.keys, "--cap-seconds", str(args.cap_seconds)]
    job = {"task": "confirmation-parallel-leaf", "phase": args.phase, "keys": args.keys,
           "pid": os.getpid(), "command": command, "cwd": str(ROOT),
           "started_utc": datetime.now(timezone.utc).isoformat(), "started_epoch": time.time(),
           "cap_seconds": args.cap_seconds, "D_remaining_at_start_seconds": remaining,
           "safety_margin_seconds": SAFETY_MARGIN_SECONDS, "status": "running"}

    def save_job(**updates) -> None:
        job.update(updates)
        job["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(job_path, job)

    save_job()
    receipt_id = None
    try:
        with ledger.run("D", f"confirmation-{args.phase}-{joined}") as receipt_id:
            save_job(receipt_id=receipt_id)
            import torch
            import discover_closure as dc
            import closure_discovery_confirmation as confirmation
            import closure_discovery_data as data
            from symbolic_closure_rom import SymbolicClosureROM

            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            identity = dc.bind_execution_sources(out)
            order = json.loads(order_path.read_text(encoding="utf-8"))
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            frozen_sha = sha(frozen_path)
            wrapper_sha = sha(Path(__file__).resolve())
            order_sha = sha(order_path)
            save_job(source_identity_sha256=identity, wrapper_sha256=wrapper_sha,
                     execution_order_sha256=order_sha, frozen_sha256=frozen_sha)
            if order.get("source_identity_sha256") != identity:
                raise RuntimeError("execution order source identity mismatch")
            if order.get("wrapper_sha256") != wrapper_sha:
                raise RuntimeError("execution order wrapper hash mismatch")
            if order.get("frozen_sha256") != frozen_sha:
                raise RuntimeError("execution order C frozen manifest hash mismatch")
            if dc.SOURCE_HASH != SOURCE_HASH:
                raise RuntimeError("registered source checkpoint hash mismatch")
            authorized = set(order.get("keys", []))
            if not set(args.keys) <= authorized:
                raise RuntimeError("requested keys are not authorized by execution order")
            rows = {f"{row['name']}:{int(row['seed'])}": row for row in frozen.get("models", [])}
            if len(rows) != len(frozen.get("models", [])) or not set(args.keys) <= set(rows):
                raise RuntimeError("requested keys are not unique entries in C frozen models")
            if args.phase == "evaluate":
                freeze_path = ddir / "confirmation_freeze.json"
                manifest_path = out / "new_fault_manifest.json"
                if not freeze_path.is_file() or not manifest_path.is_file():
                    raise RuntimeError("evaluation requires confirmation freeze and condition manifest")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                freeze, _ = confirmation._freeze_or_reuse(
                    freeze_path, None, manifest["val24"], manifest_path, source_models=frozen["models"])
                endpoints = freeze["models"]
                confirmation._verify_model_rows(endpoints, frozen["models"], out)
                if not (ddir / "new_data/val/generation.json").is_file() or not (ddir / "new_data/val/generation.integrity.json").is_file():
                    raise RuntimeError("evaluation requires completed VAL generation and integrity manifest")
                if order.get("val_generation_sha256") != sha(ddir / "new_data/val/generation.json") or order.get("val_integrity_sha256") != sha(ddir / "new_data/val/generation.integrity.json"):
                    raise RuntimeError("registered VAL generation/integrity hashes mismatch")
                if order.get("freeze_sha256") != sha(freeze_path):
                    raise RuntimeError("registered confirmation freeze hash mismatch")
                def forbid_simulator(*_args, **_kwargs):
                    raise RuntimeError("parallel evaluate may reuse completed VAL only; simulator/FOM is forbidden")

                confirmation.generate_or_reuse(manifest["val24"], ddir / "new_data/val", "val", 1.0,
                                               freeze_path=freeze_path, simulator=forbid_simulator)
                val, excluded_val = data.load_new_events(ddir / "new_data/val", "val", freeze_path=freeze_path)
                endpoint_map = {f"{r['name']}:{int(r['seed'])}": r for r in endpoints}
            else:
                if (ddir / "confirmation_freeze.json").exists():
                    raise RuntimeError("adaptation is forbidden after confirmation freeze exists")
                manifest = json.loads((out / "new_fault_manifest.json").read_text(encoding="utf-8"))
                selection_path = ddir / "selected_new_train.json"
                train_dir = ddir / "new_data/train"
                for required in (selection_path, train_dir / "generation.json", train_dir / "generation.integrity.json"):
                    if not required.is_file():
                        raise RuntimeError(f"adaptation requires completed TRAIN artifact: {required}")
                if order.get("train_generation_sha256") != sha(train_dir / "generation.json") or order.get("train_integrity_sha256") != sha(train_dir / "generation.integrity.json"):
                    raise RuntimeError("registered TRAIN generation/integrity hashes mismatch")
                if order.get("selected_new_train_sha256") != sha(selection_path):
                    raise RuntimeError("registered selected_new_train hash mismatch")
                from discover_closure import train_inputs
                old, excluded_old, scale = train_inputs()
                old_ids = [str(event["id"]) for event in old]
                if len(old_ids) != 346 or len(set(old_ids)) != 346:
                    raise confirmation.RecoveryUnresolved(f"expected 346 unique old train events, found {len(old_ids)}")
                save_job(train_exclusions={"old_train": excluded_old})
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                conditions = selection["conditions"]

                def forbid_simulator(*_args, **_kwargs):
                    raise RuntimeError("parallel adapt may reuse complete TRAIN only; simulator/FOM is forbidden")

                confirmation.generate_or_reuse(conditions, train_dir, "train", 1.0, simulator=forbid_simulator)
                new, excluded_new = data.load_new_events(train_dir, "train")
                if not new:
                    raise confirmation.RecoveryUnresolved("registered TRAIN contains no qualified events")
                combined = old + new
                train_exclusions = {"old_train": excluded_old, "new_train": excluded_new}
                save_job(train_exclusions=train_exclusions, old_train_event_count=len(old),
                         new_train_event_count=len(new))
                endpoint_map = rows
            completed, unresolved = [], []
            for key in args.keys:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("parallel phase deadline reached; remaining keys unresolved")
                row = endpoint_map[key]
                source_path = confirmation._model_file(row)
                source_sha = row["sha256"]
                if sha(source_path) != source_sha:
                    raise confirmation.RecoveryUnresolved(f"C source model hash changed: {source_path}")
                if args.phase == "adapt":
                    name, seed_text = key.split(":")
                    seed = int(seed_text)
                    model = SymbolicClosureROM.from_export(source_path)
                    dest = out / "D/models" / name / str(seed)
                    if (dest / "fit_failure.json").exists() or (dest / "candidate_failure.json").exists():
                        failure_path = dest / ("candidate_failure.json" if (dest / "candidate_failure.json").exists() else "fit_failure.json")
                        failure = json.loads(failure_path.read_text(encoding="utf-8"))
                        unresolved.append({"key": key, "status": "numerically_unresolved", "path": str(dest),
                                           "failure": failure})
                        save_job(last_key=key, completed=completed, unresolved=unresolved)
                        continue
                    fit_completed = False
                    try:
                        model, _fit_info = confirmation._fit_with_recovery(
                            dc.fit, SymbolicClosureROM, model, combined, scale, 256, dest, seed, deadline, source_sha)
                        fit_completed = True
                        model_path = dest / "model.json"
                        physics = dc.physical_qa(model)
                        immutable_json(dest / "endpoint_physics.json", physics)
                    except FloatingPointError as exc:
                        failure_path = dest / "fit_failure.json"
                        failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else {}
                        completed_updates = failure.get("completed_updates")
                        model_sha = failure.get("model_sha256")
                        if fit_completed and (dest / "model.json").is_file():
                            completed_updates = 256
                            model_sha = confirmation._file_sha256(dest / "model.json")
                        unresolved.append({"key": key, "status": "numerically_unresolved", "path": str(dest),
                                           "error": failure.get("error", failure.get("reason", repr(exc))),
                                           "completed_updates": completed_updates,
                                           "state_sha256": failure.get("state_sha256"),
                                           "model_sha256": model_sha})
                        atomic_json(dest / "candidate_failure.json", unresolved[-1])
                        save_job(last_key=key, completed=completed, unresolved=unresolved)
                        continue
                    if not physics.get("pass"):
                        failed = {"key": key, "status": "physics_qa_unresolved", "path": str(dest),
                                  "qa": physics, "completed_updates": 256,
                                  "model_sha256": confirmation._file_sha256(model_path)}
                        unresolved.append(failed)
                        atomic_json(dest / "candidate_failure.json", failed)
                        save_job(last_key=key, completed=completed, unresolved=unresolved)
                        continue
                    completed.append({**row, "source_path": str(source_path.resolve()),
                                      "source_sha256": source_sha, "path": str(model_path),
                                      "sha256": confirmation._file_sha256(model_path)})
                else:
                    endpoint = endpoint_map[key]
                    model_path = confirmation._model_file(endpoint)
                    model = SymbolicClosureROM.from_export(model_path)
                    eval_path = ddir / "evaluation" / f"{endpoint['name']}_{endpoint['seed']}.json"
                    try:
                        result = confirmation.evaluate_or_reuse(model, val, eval_path, endpoint["sha256"], deadline=deadline)
                        completed.append({"key": key, "evaluation_path": str(eval_path),
                                          "model_sha256": endpoint["sha256"], "result": result})
                    except FloatingPointError as exc:
                        unresolved.append({"key": key, "status": "numerically_unresolved", "error": repr(exc)})
                save_job(last_key=key, completed=completed, unresolved=unresolved)

            exclusions = train_exclusions if args.phase == "adapt" else {"val": excluded_val}
            result = {"phase": args.phase, "keys": args.keys, "source_identity_sha256": identity,
                      "wrapper_sha256": wrapper_sha, "execution_order_sha256": order_sha,
                      "frozen_sha256": frozen_sha, "source_hash": SOURCE_HASH,
                      "completed": completed, "unresolved": unresolved, "excluded_events": exclusions,
                      "test_read": False, "fom_run": False,
                      "primary_comparison_written": False}
            immutable_json(result_path, result)
            save_job(status="completed" if not unresolved else "completed_with_unresolved",
                     result_manifest=str(result_path), finished_utc=datetime.now(timezone.utc).isoformat(),
                     actual_process_seconds=time.perf_counter() - started, receipt_id=receipt_id,
                     completed=completed, unresolved=unresolved, excluded_events=exclusions)
        return 0
    except BaseException as exc:
        save_job(status="failed", error=repr(exc), error_type=type(exc).__name__,
                 finished_utc=datetime.now(timezone.utc).isoformat(),
                 actual_process_seconds=time.perf_counter() - started, receipt_id=receipt_id)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
