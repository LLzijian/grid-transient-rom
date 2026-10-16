"""Capped scratch prefit for the already registered generated/1701/round1 B branch."""
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
SCRATCH = OUT / "B/prefit_generated/1701/round1/scratch"
TARGET = OUT / "B/generated/1701/round1"
JOB = OUT / "B/predefined_generated1701_job.json"
ORDER = OUT / "B/predefined_generated1701_execution_order.json"
FIXED_IDENTITY = "f27cb8860a77551b07bd57749efc13c36f7566151baea49c4193155398c78428"
PROCESS_CAP_SECONDS = 1100.0
WORK_DEADLINE_SECONDS = PROCESS_CAP_SECONDS - 60.0

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
    # Match the production writer's bounded retry for transient Windows locks.
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.01 * (2 ** attempt))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_bounded_paths() -> tuple[Path, Path, Path]:
    b_root = (OUT / "B").resolve()
    scratch = SCRATCH.resolve()
    target = TARGET.resolve()
    if b_root not in scratch.parents or b_root not in target.parents:
        raise RuntimeError("resolved scratch and target paths must both be inside this B output")
    if scratch == target or scratch in target.parents or target in scratch.parents:
        raise RuntimeError("scratch and target must be distinct sibling output trees")
    return b_root, scratch, target


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


def _candidate_state(branch: Path) -> list[dict]:
    rows = []
    for candidate in sorted(branch.glob("candidate[0-9][0-9]")):
        checkpoint_path = candidate / "checkpoint.json"
        failure_path = candidate / "candidate_failure.json"
        if checkpoint_path.exists():
            checkpoint = read_json(checkpoint_path)
            model_path = candidate / "model.json"
            training_path = candidate / "training.pt"
            if not model_path.is_file() or not training_path.is_file():
                raise RuntimeError(f"checkpoint artifacts are incomplete for {candidate}")
            model_sha = sha(model_path)
            training_sha = sha(training_path)
            if model_sha != checkpoint.get("model_sha256"):
                raise RuntimeError(f"candidate model hash mismatch for {candidate}")
            if training_sha != checkpoint.get("training_sha256"):
                raise RuntimeError(f"candidate optimizer/RNG hash mismatch for {candidate}")
            for score_path in candidate.glob("score*.json"):
                score = read_json(score_path)
                level = score.get("adaptation_updates")
                snapshot = candidate / f"model_at_step{int(level):05d}.json" if level is not None else None
                if snapshot is not None and (not snapshot.exists() or sha(snapshot) != score.get("model_sha256")):
                    raise RuntimeError(f"immutable score snapshot mismatch for {candidate}/{score_path.name}")
            rows.append({
                "candidate_index": int(candidate.name[-2:]),
                "step": int(checkpoint["step"]),
                "path": str(candidate),
                "model_sha256": model_sha,
                "training_sha256": training_sha,
                "checkpoint_sha256": sha(checkpoint_path),
                "candidate_failure": read_json(failure_path) if failure_path.exists() else None,
                "verified": True,
            })
        elif failure_path.exists():
            fit_failure = candidate / "fit_failure.json"
            if fit_failure.exists():
                rows.append({
                    "candidate_index": int(candidate.name[-2:]),
                    "step": int(read_json(fit_failure).get("completed_updates", 0)),
                    "path": str(candidate),
                    "candidate_failure": read_json(failure_path),
                    "fit_failure_sha256": sha(fit_failure),
                    "verified": True,
                    "checkpoint_absent_after_recorded_numerical_failure": True,
                })
            else:
                raise RuntimeError(f"failure record lacks a fit_failure record for {candidate}")
        else:
            rows.append({
                "candidate_index": int(candidate.name[-2:]),
                "step": 0,
                "path": str(candidate),
                "verified": True,
                "no_checkpoint_yet": True,
            })
    return rows


def _rewrite_publish_paths(branch: Path, destination: Path, *, publish: bool) -> None:
    old_root = branch.resolve()
    new_root = destination.resolve()

    def mapped(old_path: str) -> tuple[str, str]:
        source = Path(old_path).resolve()
        relative = source.relative_to(old_root)
        return str(new_root / relative), str(source)

    for failure_path in branch.glob("candidate[0-9][0-9]/candidate_failure.json"):
        failure = read_json(failure_path)
        if failure.get("path"):
            new_path, scratch_path = mapped(failure["path"])
            failure["path"] = new_path if publish else scratch_path
            failure["scratch_path"] = scratch_path
            atomic_json(failure_path, failure)

    result_path = branch / "round1_results.json"
    if result_path.exists():
        result = read_json(result_path)
        result["branch_path"] = str(new_root if publish else old_root)
        result["scratch_path"] = str(old_root)
        for key in ("candidate_checkpoints", "completed_512_candidates", "numerical_failures"):
            for item in result.get(key, []):
                if item.get("path"):
                    new_path, scratch_path = mapped(item["path"])
                    item["path"] = new_path if publish else scratch_path
                    item["scratch_path"] = scratch_path
        result["publication_status"] = "ready_for_atomic_rename" if publish else "not_published"
        atomic_json(result_path, result)


def publish_if_safe(job: dict, *, status: str, reason: str | None = None) -> str:
    b_root, scratch, target = ensure_bounded_paths()
    if not scratch.exists():
        record_progress(job, status="not_published", phase="publish",
                        publication_status="scratch_absent", publication_reason=reason)
        return "not_published"
    try:
        candidate_rows = _candidate_state(scratch)
    except Exception as exc:
        record_progress(job, status="not_published", phase="checkpoint_verification_failed",
                        publication_status="verification_failed", publication_reason=repr(exc))
        return "verification_failed"
    job["candidate_checkpoints"] = candidate_rows
    if target.exists():
        result_path = scratch / "round1_results.json"
        if result_path.exists():
            result = read_json(result_path)
            result["publication_status"] = "not_published_target_exists"
            result["publication_reason"] = "official target already exists; scratch retained without merge or overwrite"
            atomic_json(result_path, result)
        record_progress(job, status="not_published", phase="publish_target_exists",
                        publication_status="target_exists", publication_reason="target already exists",
                        scratch_path=str(scratch), target_path=str(target), candidate_checkpoints=candidate_rows)
        return "target_exists"

    target.parent.mkdir(parents=True, exist_ok=True)
    if os.stat(b_root).st_dev != os.stat(scratch.parent).st_dev or os.stat(b_root).st_dev != os.stat(target.parent).st_dev:
        record_progress(job, status="not_published", phase="publish_volume_check_failed",
                        publication_status="different_volume", scratch_path=str(scratch), target_path=str(target))
        return "different_volume"
    _rewrite_publish_paths(scratch, target, publish=True)
    try:
        if target.exists():
            raise FileExistsError(str(target))
        os.rename(scratch, target)
    except FileExistsError:
        _rewrite_publish_paths(scratch, target, publish=False)
        result_path = scratch / "round1_results.json"
        if result_path.exists():
            result = read_json(result_path)
            result["publication_status"] = "not_published_target_exists"
            result["publication_reason"] = "target appeared before atomic rename; scratch retained"
            atomic_json(result_path, result)
        record_progress(job, status="not_published", phase="publish_race_target_exists",
                        publication_status="target_exists", publication_reason="target appeared before rename",
                        scratch_path=str(scratch), target_path=str(target), candidate_checkpoints=candidate_rows)
        return "target_exists"
    except OSError as exc:
        _rewrite_publish_paths(scratch, target, publish=False)
        record_progress(job, status="not_published", phase="atomic_rename_failed",
                        publication_status="rename_failed", publication_reason=repr(exc),
                        scratch_path=str(scratch), target_path=str(target), candidate_checkpoints=candidate_rows)
        return "rename_failed"

    record_progress(job, status=status, phase="published",
                    publication_status="published", publication_reason=reason,
                    scratch_path=str(scratch), target_path=str(target),
                    published_candidate_checkpoints=[
                        {**row, "path": str(target / Path(row["path"]).resolve().relative_to(scratch))}
                        for row in candidate_rows
                    ])
    return "published"


def finish(dc, job: dict, *, branch_status: str, reason: str | None,
          proposals: list[dict], fitted: list[tuple], shortlisted: list[tuple],
          completed_512: list[dict], failures: list[dict]) -> int:
    checkpoint_rows = _candidate_state(SCRATCH)
    result = {
        "format": "predefined_generated1701_round1_prefit_result_v1",
        "status": branch_status,
        "identity_sha256": FIXED_IDENTITY,
        "mode": "generated", "seed": 1701, "round": 1,
        "train_event_count": job.get("train_event_count"),
        "train_ids_sha256": job.get("train_ids_sha256"),
        "proposal_count": len(proposals),
        "completed_32_candidate_indices": [int(row[1]) for row in fitted],
        "score32_risk_by_candidate": [{"index": int(row[1]), "risk": float(row[0])} for row in fitted],
        "shortlisted_candidate_indices_by_score32_risk_then_index": [int(row[1]) for row in shortlisted],
        "completed_512_candidates": completed_512,
        "candidate_checkpoints": checkpoint_rows,
        "numerical_failures": failures,
        "timeout_or_stop_reason": reason,
        "selection_conclusion": None,
        "test_read": False,
    }
    dc.write(SCRATCH / "round1_results.json", result)
    publication = publish_if_safe(job, status=branch_status, reason=reason)
    return 0 if publication == "published" else 0


def run_branch(dc, deadline: float, job: dict) -> int:
    from symbolic_closure_rom import SymbolicClosureROM

    order = read_json(ORDER)
    if order.get("source_identity_sha256") != FIXED_IDENTITY:
        raise RuntimeError("registered execution order has a different identity")
    if "generated/1701/round1" not in str(order.get("scope", "")):
        raise RuntimeError("registered execution order does not authorize this exact prefit branch")
    identity_path = OUT / "implementation/identity.json"
    if not identity_path.exists() or sha(identity_path) != FIXED_IDENTITY:
        raise RuntimeError("saved implementation identity does not match the registered B identity")
    bound_identity = dc.bind_execution_sources(OUT)
    if bound_identity != FIXED_IDENTITY:
        raise RuntimeError("execution source binding does not match the registered B identity")
    if TARGET.exists():
        record_progress(job, status="not_published", phase="target_exists_before_start",
                        publication_status="target_exists", scratch_path=str(SCRATCH), target_path=str(TARGET))
        return 0
    ensure_bounded_paths()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    deadline = min(deadline, PROCESS_STARTED + WORK_DEADLINE_SECONDS)
    order_sha = sha(ORDER)
    job.update({"execution_order_sha256": order_sha,
                "implementation_identity_sha256": sha(identity_path),
                "scratch_path": str(SCRATCH.resolve()), "target_path": str(TARGET.resolve()),
                "work_deadline_seconds": WORK_DEADLINE_SECONDS})
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
    proposals: list[dict] = []
    screening = []
    fitted: list[tuple] = []
    shortlisted: list[tuple] = []
    completed_512: list[dict] = []
    failures: list[dict] = []
    stop_reason = None
    branch_status = "completed"

    beam = [dc.source()]
    proposals, screening = dc.rank_proposals(beam, "generated", 1701, events)
    if len(proposals) < 2:
        raise RuntimeError("fewer than two distinct legal proposals; no substitute is allowed")
    dc.write(SCRATCH / "screening.json", screening)
    dc.write(SCRATCH / "proposals.json", proposals)
    record_progress(job, status="running", phase="screening_complete",
                    proposal_count=len(proposals), screening_record_count=len(screening))

    for index, proposal in enumerate(proposals):
        if time.perf_counter() + 120.0 >= deadline:
            branch_status = "incomplete_timeout"
            stop_reason = "1100-second prefit cap reached before all step32 candidates completed"
            break
        parent = int(proposal.get("parent", 0))
        candidate_path = SCRATCH / f"candidate{index:02d}"
        candidate_path.mkdir(parents=True, exist_ok=True)
        failure_path = candidate_path / "candidate_failure.json"
        if failure_path.exists():
            failures.append(read_json(failure_path))
            continue
        model = dc.proposal_model(beam[parent], proposal)
        try:
            checkpoint_path = candidate_path / "checkpoint.json"
            step = read_json(checkpoint_path)["step"] if checkpoint_path.exists() else 0
            if step >= 32:
                model = SymbolicClosureROM.from_export(candidate_path / "model.json")
            else:
                model, _ = dc.fit(model, events, force_scale, 32, candidate_path, 1701, deadline=deadline)
            score32 = dc.score_at_level(candidate_path, events, 32, deadline=deadline)
            fitted.append((score32["risk"], index, model, score32))
            record_progress(job, status="running", phase="fit32",
                            completed_32_candidate_indices=[int(row[1]) for row in fitted],
                            numerical_failures=failures)
        except FloatingPointError as error:
            failure = {"mode": "generated", "seed": 1701, "path": str(candidate_path),
                       "level": 32, "status": "numerically_unresolved", "error": repr(error)}
            dc.write(failure_path, failure)
            failures.append(failure)
        except TimeoutError as error:
            branch_status = "incomplete_timeout"
            stop_reason = repr(error)
            break

    shortlisted = sorted(fitted, key=lambda row: (row[0], row[1]))[:2]
    if branch_status == "completed" and not shortlisted:
        branch_status = "numerically_unresolved"
        stop_reason = "no candidate has a rankable immutable score32"
    if branch_status == "completed" and len(shortlisted) < 2:
        branch_status = "numerically_unresolved"
        stop_reason = "fewer than two rankable candidates; no substitute candidate is permitted"
    record_progress(job, status="running" if branch_status == "completed" else branch_status,
                    phase="shortlist32", shortlisted_candidate_indices=[int(row[1]) for row in shortlisted],
                    score32_risk_by_candidate=[{"index": int(row[1]), "risk": float(row[0])} for row in fitted],
                    numerical_failures=failures)

    if branch_status == "completed":
        for _, index, model, _ in shortlisted:
            candidate_path = SCRATCH / f"candidate{index:02d}"
            for level in (128, 512):
                if time.perf_counter() + 120.0 >= deadline:
                    branch_status = "incomplete_timeout"
                    stop_reason = f"1100-second prefit cap reached before candidate{index:02d} level {level}"
                    break
                try:
                    checkpoint_path = candidate_path / "checkpoint.json"
                    actual = read_json(checkpoint_path)["step"] if checkpoint_path.exists() else 0
                    if actual < level:
                        model, _ = dc.fit(model, events, force_scale, level, candidate_path, 1701, deadline=deadline)
                    if level == 512:
                        score512 = dc.score_at_level(candidate_path, events, 512, deadline=deadline)
                        completed_512.append({
                            "mode": "generated", "seed": 1701, "round": 1,
                            "candidate_index": int(index), "path": str(candidate_path),
                            "score32_risk": float(next(row[0] for row in fitted if row[1] == index)),
                            "score512": score512, "terms": model.terms,
                            "direction_mask": model.template_mask[3:7].detach().tolist(),
                            "counts": model.complexity_counts(),
                        })
                except FloatingPointError as error:
                    failure = {"mode": "generated", "seed": 1701, "path": str(candidate_path),
                               "level": int(level), "status": "numerically_unresolved", "error": repr(error)}
                    dc.write(candidate_path / "candidate_failure.json", failure)
                    failures.append(failure)
                    break
                except TimeoutError as error:
                    branch_status = "incomplete_timeout"
                    stop_reason = repr(error)
                    break
            if branch_status == "incomplete_timeout":
                break
    if failures and branch_status == "completed":
        branch_status = "completed_with_numerical_failures"
    return finish(dc, job, branch_status=branch_status, reason=stop_reason,
                  proposals=proposals, fitted=fitted, shortlisted=shortlisted,
                  completed_512=completed_512, failures=failures)


def main() -> int:
    print(json.dumps({"event": "prefit_worker_started", "pid": os.getpid(),
                      "scratch": str(SCRATCH), "target": str(TARGET),
                      "process_cap_seconds": PROCESS_CAP_SECONDS}), flush=True)
    if not JOB.exists() or not ORDER.exists():
        raise RuntimeError("prepared job or registered execution order is missing")
    job = read_json(JOB)
    if job.get("identity_sha256") != FIXED_IDENTITY:
        raise RuntimeError("job manifest identity mismatch")
    if job.get("status") not in {"prepared_waiting_for_execution_order_and_authorization", "running", "incomplete_timeout"}:
        raise RuntimeError("job manifest is not in a resumable prepared state")
    try:
        with LEDGER.receipt("B", "prefit-generated-1701-round1",
                            details={"identity_sha256": FIXED_IDENTITY,
                                     "mode": "generated", "seed": 1701, "round": 1,
                                     "train_only": True,
                                     "process_cap_seconds": PROCESS_CAP_SECONDS,
                                     "scratch_path": str(SCRATCH), "target_path": str(TARGET)}):
            import discover_closure as dc
            import torch

            torch.set_num_threads(1)
            deadline = PROCESS_STARTED + PROCESS_CAP_SECONDS
            return run_branch(dc, deadline, job)
    except BaseException as error:
        job.update({"status": "failed_implementation_or_runtime", "phase": "exception",
                    "error": repr(error), "traceback": traceback.format_exc(),
                    "pid": os.getpid(), "elapsed_process_seconds": time.perf_counter() - PROCESS_STARTED,
                    "scratch_path": str(SCRATCH)})
        atomic_json(JOB, job)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
