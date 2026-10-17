"""Thin, resumable executor for the pre-registered generated/1702/round1 B branch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


PROCESS_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[6]
OUT = ROOT / "private_data/closure_discovery_20261017"
BRANCH = OUT / "B/generated/1702/round1"
JOB = OUT / "B/predefined_generated1702_job.json"
ORDER = OUT / "B/predefined_generated1702_execution_order.json"
FIXED_IDENTITY = "f27cb8860a77551b07bd57749efc13c36f7566151baea49c4193155398c78428"
PROCESS_CAP_SECONDS = 5700.0

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
from closure_discovery_support import ProcessLedger  # noqa: E402

LEDGER = ProcessLedger(OUT / "ledger")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7: raise
            time.sleep(.01 * 2**attempt)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def record_progress(job: dict, *, status: str, phase: str, **updates) -> None:
    job.update(updates)
    job.update({
        "status": status,
        "phase": phase,
        "pid": os.getpid(),
        "elapsed_process_seconds": time.perf_counter() - PROCESS_STARTED,
        "updated_epoch": time.time(),
    })
    atomic_json(JOB, job)


def run_branch(dc, deadline: float, job: dict) -> int:
    from symbolic_closure_rom import SymbolicClosureROM

    if dc.bind_execution_sources(OUT) != FIXED_IDENTITY:
        raise RuntimeError("live bound implementation changed")
    execution_order = read_json(ORDER)
    if execution_order.get("source_identity_sha256") != FIXED_IDENTITY:
        raise RuntimeError("registered execution order has a different source identity")
    if execution_order.get("scope") != "one of the original six B search seeds, generated/1702/round1 only":
        raise RuntimeError("registered execution order does not authorize this exact branch")
    identity_path = OUT / "implementation/identity.json"
    if not identity_path.exists() or sha(identity_path) != FIXED_IDENTITY:
        raise RuntimeError("saved implementation identity does not match the registered B identity")
    order_sha = sha(ORDER)
    job["execution_order_sha256"] = order_sha
    job["implementation_identity_sha256"] = sha(identity_path)
    record_progress(job, status="running", phase="load_train_and_source")

    events, _excluded, force_scale = dc.train_inputs()
    if len(events) != 346:
        raise RuntimeError(f"expected exactly 346 registered train events, found {len(events)}")
    train_ids = [event["id"] for event in events]
    job.update({
        "train_event_count": len(events),
        "train_ids_sha256": hashlib.sha256(json.dumps(train_ids, separators=(",", ":")).encode()).hexdigest(),
        "force_scale": force_scale,
    })
    if time.perf_counter() + 120.0 >= deadline:
        record_progress(job, status="incomplete_timeout", phase="before_screening")
        return 75

    beam = [dc.source()]
    proposals, screening = dc.rank_proposals(beam, "generated", 1702, events)
    if len(proposals) < 2:
        raise RuntimeError("fewer than two distinct legal proposals; no alternative candidate will be invented")
    record_progress(job, status="running", phase="screening_complete",
                    proposal_count=len(proposals), screening_record_count=len(screening))
    dc.write(BRANCH / "screening.json", screening)
    dc.write(BRANCH / "proposals.json", proposals)

    failures = []
    fitted = []
    for index, proposal in enumerate(proposals):
        parent = int(proposal.get("parent", 0))
        candidate_path = BRANCH / f"candidate{index:02d}"
        candidate_path.mkdir(parents=True, exist_ok=True)
        failure_path = candidate_path / "candidate_failure.json"
        if failure_path.exists():
            failures.append(read_json(failure_path))
            continue
        if time.perf_counter() + 120.0 >= deadline:
            record_progress(job, status="incomplete_timeout", phase="fit32",
                            next_candidate_index=index, numerical_failures=failures)
            return 75
        model = dc.proposal_model(beam[parent], proposal)
        try:
            checkpoint_path = candidate_path / "checkpoint.json"
            step = read_json(checkpoint_path)["step"] if checkpoint_path.exists() else 0
            if step >= 32:
                model = SymbolicClosureROM.from_export(candidate_path / "model.json")
            else:
                model, _ = dc.fit(model, events, force_scale, 32, candidate_path, 1702, deadline=deadline)
            score32 = dc.score_at_level(candidate_path, events, 32, deadline=deadline)
            fitted.append((score32["risk"], index, model, score32))
            record_progress(job, status="running", phase="fit32",
                            completed_32_candidate_indices=[row[1] for row in fitted],
                            numerical_failures=failures)
        except FloatingPointError as error:
            failure = {
                "mode": "generated", "seed": 1702, "path": str(candidate_path),
                "level": 32, "status": "numerically_unresolved", "error": repr(error),
            }
            dc.write(failure_path, failure)
            failures.append(failure)
        except TimeoutError:
            record_progress(job, status="incomplete_timeout", phase="fit32_or_score32",
                            next_candidate_index=index, numerical_failures=failures)
            return 75

    shortlisted = sorted(fitted, key=lambda row: (row[0], row[1]))[:2]
    if not shortlisted:
        record_progress(job, status="numerically_unresolved", phase="no_rankable_32_candidate",
                        numerical_failures=failures)
        dc.write(BRANCH / "round1_results.json", {"status": "numerically_unresolved", "failures": failures})
        return 2
    survivor_indices = [int(row[1]) for row in shortlisted]
    record_progress(job, status="running", phase="shortlist32",
                    shortlisted_candidate_indices=survivor_indices,
                    score32_risk_by_candidate=[{"index": int(row[1]), "risk": float(row[0])} for row in fitted],
                    numerical_failures=failures)

    completed_512 = []
    for _, index, model, _ in shortlisted:
        candidate_path = BRANCH / f"candidate{index:02d}"
        level = 128
        try:
            for level in (128, 512):
                checkpoint_path = candidate_path / "checkpoint.json"
                actual = read_json(checkpoint_path)["step"]
                if actual < level:
                    model, _ = dc.fit(model, events, force_scale, level, candidate_path, 1702, deadline=deadline)
            score512 = dc.score_at_level(candidate_path, events, 512, deadline=deadline)
            completed_512.append({
                "mode": "generated", "seed": 1702, "round": 1,
                "candidate_index": int(index), "path": str(candidate_path),
                "score32_risk": float(next(row[0] for row in fitted if row[1] == index)),
                "score512": score512,
                "terms": model.terms,
                "direction_mask": model.template_mask[3:7].detach().tolist(),
                "counts": model.complexity_counts(),
            })
            record_progress(job, status="running", phase="fit512",
                            completed_512_candidate_indices=[row["candidate_index"] for row in completed_512],
                            numerical_failures=failures)
        except FloatingPointError as error:
            failure = {
                "mode": "generated", "seed": 1702, "path": str(candidate_path),
                "level": int(level), "status": "numerically_unresolved", "error": repr(error),
            }
            dc.write(candidate_path / "candidate_failure.json", failure)
            failures.append(failure)
        except TimeoutError:
            record_progress(job, status="incomplete_timeout", phase=f"fit{level}_or_score{level}",
                            current_candidate_index=int(index), numerical_failures=failures,
                            completed_512_candidate_indices=[row["candidate_index"] for row in completed_512])
            return 75

    result = {
        "format": "predefined_generated1702_round1_result_v1",
        "status": "completed" if len(completed_512) == len(shortlisted) else "completed_with_numerical_failures",
        "identity_sha256": FIXED_IDENTITY,
        "train_event_count": len(events),
        "train_ids_sha256": job["train_ids_sha256"],
        "mode": "generated", "seed": 1702, "round": 1,
        "proposal_count": len(proposals),
        "shortlisted_candidate_indices_by_score32_risk_then_index": survivor_indices,
        "completed_512_candidates": completed_512,
        "numerical_failures": failures,
        "selection_conclusion": None,
        "test_read": False,
    }
    dc.write(BRANCH / "round1_results.json", result)
    record_progress(job, status=result["status"], phase="complete",
                    completed_512_candidate_indices=[row["candidate_index"] for row in completed_512],
                    numerical_failures=failures,
                    result_path=str(BRANCH / "round1_results.json"))
    return 0


def main() -> int:
    print(json.dumps({"event": "worker_started", "pid": os.getpid(),
                      "branch": str(BRANCH), "process_cap_seconds": PROCESS_CAP_SECONDS}), flush=True)
    if not JOB.exists() or not ORDER.exists():
        raise RuntimeError("prepared job or registered execution-order file is missing")
    job = read_json(JOB)
    if job.get("identity_sha256") != FIXED_IDENTITY or job.get("status") not in {"prepared", "running", "incomplete_timeout"}:
        raise RuntimeError("job manifest identity or resumable status is invalid")
    LEDGER.require_budget("B", 6000.)
    deadline = PROCESS_STARTED + PROCESS_CAP_SECONDS
    try:
        with LEDGER.receipt("B", "predefined-generated-1702-round1",
                            details={"identity_sha256": FIXED_IDENTITY,
                                     "mode": "generated", "seed": 1702, "round": 1,
                                     "train_only": True, "process_cap_seconds": PROCESS_CAP_SECONDS}):
            import discover_closure as dc
            import torch

            torch.set_num_threads(1)
            return run_branch(dc, deadline, job)
    except BaseException as error:
        job.update({"status": "failed_implementation_or_runtime", "phase": "exception",
                    "error": repr(error), "traceback": traceback.format_exc(),
                    "pid": os.getpid(), "elapsed_process_seconds": time.perf_counter() - PROCESS_STARTED})
        atomic_json(JOB, job)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
