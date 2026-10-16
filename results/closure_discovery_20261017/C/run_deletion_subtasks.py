"""Prepare registered train-only deletion screening/refit subtasks for stage C."""
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
    parser.add_argument("--mode", choices=("screen", "fit"), required=True)
    parser.add_argument("--indices", nargs="+", type=int, choices=range(12), required=True)
    parser.add_argument("--cap-seconds", type=float, required=True)
    args = parser.parse_args()
    if len(set(args.indices)) != len(args.indices):
        parser.error("--indices must not contain duplicates")
    if args.mode == "fit" and len(args.indices) != 1:
        parser.error("fit mode accepts exactly one deletion index")
    if args.cap_seconds <= 0:
        parser.error("--cap-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    cfg = json.loads((ROOT / "configs/closure_discovery_20261017.json").read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    cdir = out / "C"
    main_path = cdir / "main/2701/model.json"
    order_path = cdir / f"deletion_execution_order_{args.mode}.json"
    if not main_path.is_file() or not order_path.is_file():
        raise RuntimeError("registered deletion order and main/2701 model must exist before launch")
    from closure_discovery_support import ProcessLedger

    ledger = ProcessLedger(out / "ledger")
    remaining = ledger.remaining("C")
    budget = min(args.cap_seconds, remaining - SAFETY_MARGIN_SECONDS)
    if budget <= 0:
        raise RuntimeError(f"insufficient C budget after safety margin: remaining={remaining:.3f}s")
    deadline = time.perf_counter() + budget
    indices = list(args.indices)
    joined = "_".join(map(str, indices))
    tag = f"{time.time_ns()}_{os.getpid()}"
    job_path = cdir / f"deletion_{args.mode}_{joined}_{tag}_job.json"
    result_path = cdir / f"deletion_{args.mode}_{joined}_{tag}_result.json"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", args.mode,
               "--indices", *map(str, indices), "--cap-seconds", str(args.cap_seconds)]
    started = time.perf_counter()
    receipt_id = None
    job = {"task": "registered-deletion-subtasks", "mode": args.mode, "indices": indices,
           "pid": os.getpid(), "command": command, "cwd": str(ROOT),
           "started_utc": datetime.now(timezone.utc).isoformat(), "started_epoch": time.time(),
           "cap_seconds": args.cap_seconds, "C_remaining_at_start_seconds": remaining,
           "safety_margin_seconds": SAFETY_MARGIN_SECONDS, "status": "running"}

    def save_job(**updates) -> None:
        job.update(updates)
        job["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(job_path, job)

    save_job()
    try:
        with ledger.run("C", f"deletion-{args.mode}-{joined}") as receipt_id:
            save_job(receipt_id=receipt_id)
            import torch
            import discover_closure as dc
            from symbolic_closure_rom import SymbolicClosureROM

            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            identity = dc.bind_execution_sources(out)
            order = json.loads(order_path.read_text(encoding="utf-8"))
            main_sha = sha(main_path)
            order_sha = sha(order_path)
            if order.get("source_identity_sha256") != identity:
                raise RuntimeError("registered source identity differs from bound implementation")
            if order.get("wrapper_sha256") != sha(Path(__file__).resolve()):
                raise RuntimeError("registered deletion wrapper hash mismatch")
            if order.get("main_model_sha256") != main_sha:
                raise RuntimeError("registered main model hash mismatch")
            if dc.SOURCE_HASH != SOURCE_HASH:
                raise RuntimeError("frozen source checkpoint hash mismatch")
            authorized = order.get("indices", [])
            if not set(indices) <= set(map(int, authorized)):
                raise RuntimeError("requested deletion indices are outside this execution order")
            main = SymbolicClosureROM.from_export(main_path)
            if len(main.terms) > 12 or any(index >= len(main.terms) for index in indices):
                raise RuntimeError("deletion index exceeds the registered main term list")
            provenance = getattr(main, "provenance", {})
            if (provenance.get("fit_seed") != 2701
                    or provenance.get("completed_updates_in_current_fit") != TARGET_UPDATES
                    or provenance.get("source_sha256") != SOURCE_HASH
                    or provenance.get("implementation_identity_sha256") != identity):
                raise RuntimeError("main/2701 model provenance is not a completed registered refit")
            main_terms = json.loads(json.dumps(main.terms, sort_keys=True))
            main_mask = main.template_mask[3:7].detach().cpu().numpy().tolist()
            common = {"source_identity_sha256": identity, "source_hash": SOURCE_HASH,
                      "wrapper_sha256": sha(Path(__file__).resolve()), "execution_order_sha256": order_sha,
                      "main_model_sha256": main_sha}
            save_job(**common)
            results, unresolved = [], []

            if args.mode == "screen":
                events, _, _ = dc.train_inputs()
                event_ids = [str(event["id"]) for event in events]
                if len(event_ids) != 346 or len(set(event_ids)) != 346:
                    raise RuntimeError(f"expected 346 unique train events, found {len(event_ids)}")
                for index in indices:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("screen deadline reached; aggregate ranking is incomplete")
                    reduced = main.with_terms(main.terms[:index] + main.terms[index + 1:])
                    try:
                        score = dc.score(reduced, events, deadline=deadline)
                        results.append({"index": index, "deleted": main_terms[index], "risk": score.get("risk"),
                                        "score": score, "status": "scored"})
                    except FloatingPointError as exc:
                        unresolved.append({"index": index, "deleted": main_terms[index],
                                           "status": "numerically_unresolved", "error": repr(exc)})
                    save_job(last_index=index, completed_indices=[r["index"] for r in results],
                             unresolved_indices=[r["index"] for r in unresolved],
                             results=results, unresolved=unresolved)
            else:
                index = indices[0]
                seed = 2701
                path = cdir / f"delete{index}" / str(seed)
                path.mkdir(parents=True, exist_ok=True)
                candidate_failure = path / "candidate_failure.json"
                fit_failure = path / "fit_failure.json"
                if candidate_failure.exists() or fit_failure.exists():
                    failure_path = candidate_failure if candidate_failure.exists() else fit_failure
                    failure = json.loads(failure_path.read_text(encoding="utf-8"))
                    unresolved.append({"index": index, "deleted": main_terms[index], "name": f"delete{index}",
                                       "seed": seed, "path": str(path), "status": "numerically_unresolved",
                                       "error": failure.get("error", failure.get("reason")),
                                       "completed_updates": failure.get("completed_updates"),
                                       "state_sha256": failure.get("state_sha256")})
                else:
                    source_terms = main_terms[:index] + main_terms[index + 1:]
                    model_path, training_path = path / "model.json", path / "training.pt"
                    checkpoint_path = path / "checkpoint.json"
                    try:
                        events, _, force_scale = dc.train_inputs()
                        event_ids = [str(event["id"]) for event in events]
                        if len(event_ids) != 346 or len(set(event_ids)) != 346:
                            raise RuntimeError(f"expected 346 unique train events, found {len(event_ids)}")
                        seed_model = dc.source().with_terms(source_terms)
                        with torch.no_grad():
                            seed_model.template_mask[3:7] = torch.as_tensor(main_mask, dtype=torch.float64)
                        if checkpoint_path.exists():
                            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                            step = int(checkpoint.get("step", -1))
                            if (not model_path.is_file() or not training_path.is_file()
                                    or sha(model_path) != checkpoint.get("model_sha256")
                                    or sha(training_path) != checkpoint.get("training_sha256")):
                                raise RuntimeError("fit checkpoint bytes/SHA mismatch")
                            model = SymbolicClosureROM.from_export(model_path)
                            provenance = getattr(model, "provenance", {})
                            if (step < 0 or step > TARGET_UPDATES or provenance.get("fit_seed") != seed
                                    or provenance.get("source_sha256") != SOURCE_HASH
                                    or provenance.get("implementation_identity_sha256") != identity
                                    or provenance.get("completed_updates_in_current_fit") != step
                                    or json.loads(json.dumps(model.terms, sort_keys=True)) != source_terms
                                    or model.template_mask[3:7].detach().cpu().numpy().tolist() != main_mask):
                                raise RuntimeError("fit checkpoint provenance, terms, mask, or step mismatch")
                            if step < TARGET_UPDATES:
                                model, fit_info = dc.fit(model, events, force_scale, TARGET_UPDATES, path, seed, final=True, deadline=deadline)
                            else:
                                fit_info = {"seconds": 0.0, "updates": step, "reused_checkpoint": True}
                        else:
                            if any(path.iterdir()):
                                raise RuntimeError("partial fit artifacts exist without a recoverable checkpoint")
                            model, fit_info = dc.fit(seed_model, events, force_scale, TARGET_UPDATES,
                                                     path, seed, final=True, deadline=deadline)
                    except FloatingPointError as exc:
                        failure = json.loads(fit_failure.read_text(encoding="utf-8")) if fit_failure.exists() else {}
                        record = {"index": index, "deleted": main_terms[index], "name": f"delete{index}",
                                  "seed": seed, "path": str(path), "status": "numerically_unresolved",
                                  "error": failure.get("error", failure.get("reason", repr(exc))),
                                  "completed_updates": failure.get("completed_updates"),
                                  "state_sha256": failure.get("state_sha256")}
                        unresolved.append(record)
                        if not fit_failure.exists() and not candidate_failure.exists():
                            atomic_json(candidate_failure, record)
                    else:
                        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        if (int(checkpoint.get("step", -1)) != TARGET_UPDATES
                                or sha(model_path) != checkpoint.get("model_sha256")
                                or sha(training_path) != checkpoint.get("training_sha256")):
                            raise RuntimeError("final 1024-update checkpoint integrity failure")
                        try:
                            score_path, qa_path = path / "train_score.json", path / "physics.json"
                            score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else dc.score(model, events, deadline=deadline)
                            if [str(row.get("id")) for row in score.get("events", [])] != event_ids:
                                raise RuntimeError("existing train score event identity/order mismatch")
                            if not score_path.exists():
                                immutable_json(score_path, score)
                            qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else dc.physical_qa(model)
                            if not qa_path.exists():
                                immutable_json(qa_path, qa)
                        except FloatingPointError as exc:
                            unresolved.append({"index": index, "name": f"delete{index}", "seed": seed,
                                               "path": str(path), "status": "numerically_unresolved",
                                               "error": repr(exc), "completed_updates": TARGET_UPDATES,
                                               "model_sha256": sha(model_path)})
                            atomic_json(path / "candidate_failure.json", unresolved[-1])
                        else:
                            row = {"name": f"delete{index}", "seed": seed, "path": str(path),
                                   "sha256": sha(model_path), "deleted": main_terms[index], "qa": qa}
                            if qa.get("pass"):
                                results.append(row)
                            else:
                                unresolved.append({**row, "status": "physics_qa_unresolved",
                                                   "completed_updates": TARGET_UPDATES})
                                atomic_json(path / "candidate_failure.json", unresolved[-1])

            result = {"mode": args.mode, "indices": indices, **common, "results": results,
                      "unresolved": unresolved, "test_read": False, "val_read": False, "fom_run": False}
            immutable_json(result_path, result)
            save_job(status="completed" if not unresolved else "completed_with_unresolved",
                     result_manifest=str(result_path), finished_utc=datetime.now(timezone.utc).isoformat(),
                     actual_process_seconds=time.perf_counter() - started, receipt_id=receipt_id)
        return 0
    except BaseException as exc:
        save_job(status="failed", error=repr(exc), error_type=type(exc).__name__,
                 finished_utc=datetime.now(timezone.utc).isoformat(),
                 actual_process_seconds=time.perf_counter() - started, receipt_id=receipt_id)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
