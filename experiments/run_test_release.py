"""Single-pass confirmatory holdout release.

The committed preregistration is intentionally blocked for the legacy dataset
build because its test payloads have already been consumed. Once a genuinely
fresh build and refitted, accepted models are registered, this script opens each
holdout event exactly once and computes every preregistered endpoint in memory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kiki.evaluation.modal import (
    evaluate_event_modal_matches,
    summarize_event_modal_matches,
)
from kiki.evaluation.release import SinglePassHoldout
from kiki.evaluation.stability import grouped_missed_instability_audit
from kiki.models.base import sample_rollout_inputs
from kiki.models.prunable_energy_model import PrunableEnergyModel

FOLDS = ("F1", "F2", "F3", "F4", "F5")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean_git() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("confirmatory release requires a clean Git worktree")


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    preregistration = json.loads(
        (ROOT / "configs" / "test_release_v1.json").read_text(encoding="utf-8")
    )
    if preregistration["status"] != "armed":
        print(
            "[test-release] BLOCKED before holdout access: "
            f"{preregistration['status']}"
        )
        return 2

    dataset_config = json.loads(
        (ROOT / "configs" / "dataset.json").read_text(encoding="utf-8")
    )
    store = CanonicalStore.open(dataset_config["dataset_dir"])
    metadata = store.dataset_metadata
    split_by_fold = {
        fold: load_split(
            dataset_config["dataset_dir"],
            view=dataset_config["view"],
            split_name=dataset_config["split_name"],
            fold=fold,
        )
        for fold in FOLDS
    }
    ordered_ids = [event_id for fold in FOLDS for event_id in split_by_fold[fold].test_ids]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise RuntimeError("fold test populations overlap; single-pass release is invalid")
    guard = SinglePassHoldout(
        frozenset(ordered_ids),
        frozenset(preregistration["contaminated_dataset_build_ids"]),
    )
    guard.authorize(
        release_status=preregistration["status"],
        dataset_build_id=metadata.build_id,
    )

    raw_model_paths = preregistration["model_paths"]
    if set(raw_model_paths) != set(FOLDS):
        raise ValueError("preregistration must provide one final model path per fold")
    models = {}
    for fold in FOLDS:
        model_path = ROOT / raw_model_paths[fold]
        model = PrunableEnergyModel.load(model_path)
        if "module_adoption_gate" not in model.diagnostics:
            raise RuntimeError(f"{model_path} predates the total module-adoption gate")
        models[fold] = model

    output = ROOT / preregistration["output_directory"]
    if output.exists():
        raise SystemExit(f"refusing to read holdout because output already exists: {output}")
    _require_clean_git()
    release_commit = _git_commit()
    output.mkdir(parents=True, exist_ok=False)
    access_ledger = output / "holdout_access.json"

    def persist_access(audit: dict[str, Any], *, status: str = "in_progress") -> None:
        _atomic_json_write(
            access_ledger,
            {
                "release_id": preregistration["release_id"],
                "status": status,
                "git_commit": release_commit,
                "dataset_revision": metadata.dataset_revision,
                "dataset_build_id": metadata.build_id,
                "updated_at": datetime.now(UTC).isoformat(),
                **audit,
            },
        )

    guard.claim_callback = persist_access
    # The directory and initial ledger are durable before the first payload read.
    # Any interruption leaves an existing output path, so this release cannot be
    # silently restarted against the same holdout.
    persist_access(guard.audit_snapshot())

    contract = preregistration["evaluation_contract"]
    request = ModelDataRequest.initial_state_rollout(dataset_config["pm_mode"])
    per_event: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    modal_cases: list[dict[str, object]] = []
    trajectory_failures = 0
    for fold in FOLDS:
        model = models[fold]
        for event_id in split_by_fold[fold].test_ids:
            event = guard.load_event_once(store, event_id)
            sample = adapt_event(
                event,
                inertia_prior=metadata.M_prior,
                request=request,
                representation=dataset_config["representation"],
            )
            record: dict[str, Any] = {
                "event_id": event_id,
                "fold": fold,
                "source": event.metadata.source_dataset,
                "fault_line": event.metadata.fault_line,
                "true_stable": event.metadata.outcome_label == "stable",
            }

            initial, time, exogenous = sample_rollout_inputs(sample)
            steps = int(contract["trajectory_horizon_steps"]) + 1
            try:
                if len(time) < steps:
                    raise ValueError(
                        f"event has {len(time) - 1} intervals; "
                        f"the preregistered endpoint requires {steps - 1}"
                    )
                pred_delta, pred_omega = model.rollout(
                    initial,
                    time[:steps],
                    {
                        "Pm": exogenous["Pm"][:steps],
                        "interpolation": exogenous["interpolation"],
                    },
                )
                targets = sample["training_context"]["targets"]
                trajectory = trajectory_metrics(
                    targets["delta"][:steps],
                    targets["omega"][:steps],
                    pred_delta,
                    pred_omega,
                )
                trajectory_rows.append({**trajectory, "event_id": event_id})
                record["trajectory"] = trajectory
            except Exception as error:  # noqa: BLE001 - release preserves every failure
                trajectory_failures += 1
                record["trajectory_error"] = f"{type(error).__name__}: {error}"

            try:
                record["candidate_energy_screen"] = model.stability_margin(
                    sample["model_inputs"]["static"],
                    angle_limit=float(contract["candidate_energy_angle_limit_pi"])
                    * np.pi,
                )
            except Exception as error:  # noqa: BLE001 - formal verdict remains refused
                record["candidate_energy_screen"] = {
                    "certified": False,
                    "predicted_stable": False,
                    "candidate_search_completed": False,
                    "screening_predicted_stable": False,
                    "error": f"{type(error).__name__}: {error}",
                }

            if (
                record["true_stable"]
                and event.metadata.available_horizon_s
                >= float(contract["modal_horizon_seconds"])
            ):
                try:
                    modal = evaluate_event_modal_matches(
                        model,
                        sample,
                        horizon_seconds=float(contract["modal_horizon_seconds"]),
                    )
                    modal["fold"] = fold
                    modal["source"] = record["source"]
                    modal["fault_line"] = record["fault_line"]
                    modal_cases.append(modal)
                    record["modal"] = modal
                except Exception as error:  # noqa: BLE001 - release preserves every failure
                    record["modal_error"] = f"{type(error).__name__}: {error}"
            per_event.append(record)

    access_audit = guard.finalize()
    trajectory_summary = aggregate_trajectory_metrics(trajectory_rows)
    trajectory_summary["failure_rate"] = trajectory_failures / len(per_event)
    screen_rows = [
        {
            "true_stable": row["true_stable"],
            "source": row["source"],
            "fault_line": row["fault_line"],
            "candidate_search_completed": bool(
                row["candidate_energy_screen"].get("candidate_search_completed")
            ),
            "screening_predicted_stable": bool(
                row["candidate_energy_screen"].get("screening_predicted_stable")
            ),
        }
        for row in per_event
    ]
    report = _json_safe(
        {
            "release_id": preregistration["release_id"],
            "timestamp": datetime.now(UTC).isoformat(),
            "git_commit": release_commit,
            "dataset_revision": metadata.dataset_revision,
            "dataset_build_id": metadata.build_id,
            "access_audit": access_audit,
            "evaluation_contract": contract,
            "trajectory_summary": trajectory_summary,
            "formal_stability_summary": {
                "certified_events": 0,
                "stable_verdicts": 0,
                "status": "refused_without_verified_global_lower_bound",
            },
            "candidate_energy_screen_audit": grouped_missed_instability_audit(
                screen_rows,
                predicted_stable_key="screening_predicted_stable",
                eligible_key="candidate_search_completed",
            ),
            "modal_summary": summarize_event_modal_matches(modal_cases),
            "per_event": per_event,
        }
    )
    _atomic_json_write(output / "release.json", report)
    persist_access(access_audit, status="complete")
    print(json.dumps({key: value for key, value in report.items() if key != "per_event"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
