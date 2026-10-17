"""Prepare/run selected C refits for an explicitly registered name/seed subset."""
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
EXPECTED_SOURCE_HASH = "c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238"
VALID_NAMES = ("library", "main", "alternative")
VALID_SEEDS = (2701, 2702, 2703)
TARGET_UPDATES = 1024
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
                time.sleep(0.01 * (2 ** attempt))
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
    parser.add_argument("--name", required=True, choices=VALID_NAMES)
    parser.add_argument("--seeds", required=True, nargs="+", type=int, choices=VALID_SEEDS)
    parser.add_argument("--cap-seconds", required=True, type=float)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if args.cap_seconds <= 0:
        parser.error("--cap-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    cfg = json.loads((ROOT / "configs/closure_discovery_20261017.json").read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    cdir = out / "C"
    selection_path = out / "B/selection.json"
    order_path = cdir / "selected_refits_execution_order.json"
    if not selection_path.is_file() or not order_path.is_file():
        raise RuntimeError("selected-refit selection and execution-order registrations must exist before launch")

    # Import and bind the frozen implementation only inside the metered C receipt.
    from closure_discovery_support import ProcessLedger

    ledger = ProcessLedger(out / "ledger")
    initial_remaining = ledger.remaining("C")
    available = min(args.cap_seconds, initial_remaining - SAFETY_MARGIN_SECONDS)
    if available <= 0:
        raise RuntimeError(f"insufficient C budget after safety margin: remaining={initial_remaining:.3f}s")
    deadline = time.perf_counter() + available
    job_tag = f"{time.time_ns()}_{os.getpid()}"
    joined = "_".join(str(seed) for seed in args.seeds)
    job_path = cdir / f"selected_{args.name}_{joined}_{job_tag}_job.json"
    result_path = cdir / f"selected_{args.name}_{joined}_{job_tag}_result.json"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--name", args.name,
               "--seeds", *map(str, args.seeds), "--cap-seconds", str(args.cap_seconds)]
    job = {
        "task": "selected-refits", "name": args.name, "seeds": list(args.seeds),
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "started_epoch": time.time(), "command": command, "cwd": str(ROOT),
        "status": "running", "cap_seconds": args.cap_seconds,
        "stage_C_remaining_at_start_seconds": initial_remaining,
        "safety_margin_seconds": SAFETY_MARGIN_SECONDS,
        "execution_order_sha256": sha(order_path),
        "wrapper_sha256": sha(Path(__file__)),
    }
    process_started = time.perf_counter()
    receipt_id = None

    def save_job(**updates) -> None:
        job.update(updates)
        job["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(job_path, job)

    save_job()
    try:
        # Importing the ledger helper is metered as C execution; module paths
        # were prepared above so this also works when launched from this folder.
        with ledger.run("C", f"selected-refits-{args.name}-{joined}") as receipt_id:
            job["receipt_id"] = receipt_id
            save_job()
            import numpy as np
            import torch
            import discover_closure as dc
            from symbolic_closure_rom import SymbolicClosureROM

            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            identity = dc.bind_execution_sources(out)
            order = json.loads(order_path.read_text(encoding="utf-8"))
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selection_sha = sha(selection_path)
            if order.get("source_identity_sha256") != identity:
                raise RuntimeError("execution order source identity differs from bound implementation")
            if order.get("selection_sha256") != selection_sha:
                raise RuntimeError("execution order is not bound to the current B selection")
            if order.get("wrapper_sha256") != sha(Path(__file__)):
                raise RuntimeError("execution wrapper differs from the registered source")
            authorized = order.get("names", {}).get(args.name)
            if not isinstance(authorized, list) or not set(args.seeds) <= set(map(int, authorized)):
                raise RuntimeError("requested name/seeds are not covered by execution order")
            if dc.SOURCE_HASH != EXPECTED_SOURCE_HASH:
                raise RuntimeError("registered source checkpoint hash changed")
            if time.perf_counter() >= deadline:
                raise TimeoutError("execution cap exhausted during startup")

            events, excluded, force_scale = dc.train_inputs()
            event_ids = [str(event["id"]) for event in events]
            if len(event_ids) != 346 or len(set(event_ids)) != 346:
                raise RuntimeError(f"expected the fixed 346 train events, found {len(event_ids)}")
            selected = selection["library" if args.name == "library" else "generated"] if args.name != "alternative" else selection["alternative"]
            terms = json.loads(json.dumps(selected["terms"], sort_keys=True))
            mask = list(selected["direction_mask"])
            if len(mask) != 4:
                raise RuntimeError("selected direction mask must contain four entries")
            source_model_sha = dc.sha(ROOT / cfg["source"])
            models, unresolved = [], []
            for seed in args.seeds:
                seed_key = str(seed)
                path = cdir / args.name / seed_key
                path.mkdir(parents=True, exist_ok=True)
                candidate_failure_path = path / "candidate_failure.json"
                fit_failure_path = path / "fit_failure.json"
                existing_failure = candidate_failure_path if candidate_failure_path.exists() else fit_failure_path
                if candidate_failure_path.exists() or fit_failure_path.exists():
                    failure = json.loads(existing_failure.read_text(encoding="utf-8"))
                    unresolved.append({"name": args.name, "seed": seed, "path": str(path),
                                       "status": "numerically_unresolved", "reason": failure.get("error"),
                                       "completed_updates": failure.get("completed_updates"),
                                       "state_sha256": failure.get("state_sha256")})
                    save_job(last_seed=seed, seed_status="numerically_unresolved")
                    continue
                model_path, training_path, checkpoint_path = path / "model.json", path / "training.pt", path / "checkpoint.json"
                try:
                    source = dc.source().with_terms(terms)
                    with torch.no_grad():
                        source.template_mask[3:7] = torch.as_tensor(mask, dtype=torch.float64)
                    if checkpoint_path.exists():
                        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        if not model_path.is_file() or not training_path.is_file():
                            raise RuntimeError("checkpoint is missing model/training bytes")
                        if sha(model_path) != checkpoint.get("model_sha256") or sha(training_path) != checkpoint.get("training_sha256"):
                            raise RuntimeError("checkpoint model/training SHA mismatch")
                        step = int(checkpoint.get("step", -1))
                        model = SymbolicClosureROM.from_export(model_path)
                        provenance = getattr(model, "provenance", {})
                        if (step < 0 or step > TARGET_UPDATES or provenance.get("fit_seed") != seed
                                or provenance.get("source_sha256") != EXPECTED_SOURCE_HASH
                                or provenance.get("implementation_identity_sha256") != identity
                                or provenance.get("completed_updates_in_current_fit") != step
                                or json.loads(json.dumps(model.terms, sort_keys=True)) != terms
                                or model.template_mask[3:7].detach().cpu().numpy().tolist() != mask):
                            raise RuntimeError("checkpoint step, seed, provenance, terms, or mask mismatch")
                        if step < TARGET_UPDATES:
                            model, fit_info = dc.fit(model, events, force_scale, TARGET_UPDATES, path, seed, final=True, deadline=deadline)
                        else:
                            fit_info = {"seconds": 0.0, "updates": step, "reused_checkpoint": True}
                    else:
                        if any(path.iterdir()):
                            raise RuntimeError("partial artifacts exist without a recoverable checkpoint")
                        model, fit_info = dc.fit(source, events, force_scale, TARGET_UPDATES, path, seed, final=True, deadline=deadline)
                except FloatingPointError as exc:
                    existing_failure = candidate_failure_path if candidate_failure_path.exists() else fit_failure_path
                    failure = json.loads(existing_failure.read_text(encoding="utf-8")) if existing_failure.exists() else {}
                    record = {"name": args.name, "seed": seed, "path": str(path), "status": "numerically_unresolved",
                              "error": failure.get("reason", failure.get("error", repr(exc))),
                              "completed_updates": failure.get("completed_updates"),
                              "state_sha256": failure.get("state_sha256")}
                    unresolved.append(record)
                    if not candidate_failure_path.exists() and not fit_failure_path.exists():
                        atomic_json(candidate_failure_path, record)
                    save_job(last_seed=seed, seed_status="numerically_unresolved")
                    continue

                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if (int(checkpoint.get("step", -1)) != TARGET_UPDATES or sha(model_path) != checkpoint.get("model_sha256")
                        or sha(training_path) != checkpoint.get("training_sha256")):
                    raise RuntimeError("completed endpoint checkpoint integrity check failed")
                score_path, qa_path = path / "train_score.json", path / "physics.json"
                try:
                    score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else dc.score(model, events, deadline=deadline)
                    if not score_path.exists():
                        immutable_json(score_path, score)
                    if [row.get("id") for row in score.get("events", [])] != event_ids:
                        raise RuntimeError("train score event identity differs from registered train inputs")
                    qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else dc.physical_qa(model)
                    if not qa_path.exists():
                        immutable_json(qa_path, qa)
                except FloatingPointError as exc:
                    record = {"name": args.name, "seed": seed, "path": str(path),
                              "status": "train_score_unresolved", "error": repr(exc),
                              "model_sha256": sha(model_path), "completed_updates": TARGET_UPDATES}
                    unresolved.append(record)
                    immutable_json(path / "candidate_failure.json", record)
                    save_job(last_seed=seed, seed_status="train_score_unresolved")
                    continue
                row = {"name": args.name, "seed": seed, "path": str(path), "sha256": sha(model_path), "qa": qa,
                       "source_model_sha256": source_model_sha, "source_sha256": EXPECTED_SOURCE_HASH,
                       "direction_mask": mask, "updates": TARGET_UPDATES, "train_score": {
                           "risk": score.get("risk"), "group_se": score.get("group_se"),
                           "se_method": score.get("se_method")}}
                (models if qa.get("pass") else unresolved).append(row if qa.get("pass") else {**row, "status": "physics_qa_unresolved"})
                save_job(last_seed=seed, seed_status="completed" if qa.get("pass") else "physics_qa_unresolved")

            result = {"name": args.name, "seeds": list(args.seeds), "implementation_identity_sha256": identity,
                      "selection_sha256": selection_sha, "source_sha256": EXPECTED_SOURCE_HASH,
                      "source_model_sha256": source_model_sha, "split": "train", "train_event_ids": event_ids,
                      "train_event_count": len(event_ids), "excluded_train_rows": excluded, "force_scale": force_scale,
                      "target_updates_each": TARGET_UPDATES, "models": models, "unresolved": unresolved,
                      "test_read": False, "val_read": False, "fom_run": False}
            immutable_json(result_path, result)
            save_job(status="completed" if not unresolved else "completed_with_unresolved",
                     result_manifest=str(result_path), finished_utc=datetime.now(timezone.utc).isoformat(),
                     actual_process_seconds=time.perf_counter() - process_started, receipt_id=receipt_id)
        return 0
    except BaseException as exc:
        save_job(status="failed", error=repr(exc), error_type=type(exc).__name__,
                 finished_utc=datetime.now(timezone.utc).isoformat(),
                 actual_process_seconds=time.perf_counter() - process_started, receipt_id=receipt_id)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
