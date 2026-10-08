"""Post-selection modal diagnostic on validation data and final pruned models.

This is not an out-of-sample confirmation: the validation folds participated
in ridge/module selection. The diagnostic uses every pair from one complete
Hungarian assignment and macro-aggregates event medians; it never selects the
lowest-error pair within an event. A fresh holdout release is required for a
confirmatory modal claim.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_refit_v2 import atomic_write_json, git_provenance, sha256_file
from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.evaluation.modal import (
    evaluate_event_modal_matches,
    summarize_event_modal_matches,
)
from kiki.models.prunable_energy_model import PrunableEnergyModel

MODEL_PATHS = {
    fold: ROOT / "results" / "e10_refit_v2" / fold / "prune" / "model_pruned.npz"
    for fold in ("F1", "F2", "F3", "F4", "F5")
}
MODAL_HORIZON_SECONDS = 4.0


def main() -> int:
    out = ROOT / "results" / "e11_modal_validation_v2" / "summary.json"
    if out.parent.exists():
        raise RuntimeError(f"refusing to overwrite {out.parent}")
    provenance = git_provenance()
    provenance["models"] = {
        fold: {
            "path": str(path.relative_to(ROOT)),
            "npz_sha256": sha256_file(path),
            "json_sha256": sha256_file(path.with_suffix(".json")),
        }
        for fold, path in MODEL_PATHS.items()
    }
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    metadata = store.dataset_metadata
    provenance["dataset_build_id"] = metadata.build_id
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])

    cases: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    per_fold: dict[str, dict[str, object]] = {}
    for fold, model_path in MODEL_PATHS.items():
        print(f"Evaluating {fold} final-model modal diagnostic", flush=True)
        model = PrunableEnergyModel.load(model_path)
        if "module_adoption_gate" not in model.diagnostics:
            raise RuntimeError(
                f"{model_path} predates the total module-adoption gate; rerun pruning first"
            )
        split = load_split(
            config["dataset_dir"],
            view=config["view"],
            split_name=config["split_name"],
            fold=fold,
        )
        fold_cases: list[dict[str, object]] = []
        eligible = 0
        for event_id in split.val_ids:
            event_metadata = store.metadata(event_id)
            if event_metadata.outcome_label != "stable":
                continue
            if event_metadata.available_horizon_s < MODAL_HORIZON_SECONDS:
                continue
            eligible += 1
            event = store.load_event(event_id)
            sample = adapt_event(
                event,
                inertia_prior=metadata.M_prior,
                request=request,
                representation=config["representation"],
            )
            try:
                result = evaluate_event_modal_matches(
                    model,
                    sample,
                    horizon_seconds=MODAL_HORIZON_SECONDS,
                )
            except (FloatingPointError, RuntimeError, ValueError) as error:
                failures.append(
                    {
                        "fold": fold,
                        "event_id": event_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            result["fold"] = fold
            result["source"] = event.metadata.source_dataset
            result["fault_line"] = event.metadata.fault_line
            fold_cases.append(result)
            cases.append(result)
        per_fold[fold] = {
            "n_eligible_stable_validation_events": eligible,
            "n_failed_events": sum(row["fold"] == fold for row in failures),
            "summary": summarize_event_modal_matches(fold_cases),
        }

    report = {
        "stage": "modal_validation_v2",
        "timestamp": datetime.now(UTC).isoformat(),
        "provenance": provenance,
        "role": "post_selection_validation_diagnostic_not_confirmatory_oos",
        "population": "stable validation events reaching 4.0 s",
        "model_role": "final_pruned_after_total_module_adoption_gate",
        "modal_horizon_seconds": MODAL_HORIZON_SECONDS,
        "primary_protocol": (
            "mass-weighted speed modes; complete Hungarian assignment; macro median "
            "over each event's median of all assigned pairs"
        ),
        "per_fold": per_fold,
        "pooled": summarize_event_modal_matches(cases),
        "failures": failures,
        "per_event": cases,
    }
    if git_provenance() != {key: provenance[key] for key in ("head", "worktree")}:
        raise RuntimeError("Git changed during modal evaluation")
    out.parent.mkdir(parents=True, exist_ok=False)
    atomic_write_json(out, report)
    print(json.dumps({key: value for key, value in report["pooled"].items() if key != "per_event"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
