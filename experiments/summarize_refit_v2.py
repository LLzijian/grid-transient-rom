"""Recompute audited development evidence from saved event rows, without payload reads."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_refit_v2 import atomic_write_json, sha256_file
from kiki.evaluation.metrics import aggregate_trajectory_metrics
from kiki.evaluation.modal import summarize_event_modal_matches


def main() -> None:
    output = ROOT / "results" / "e12_refit_audit" / "summary.json"
    if output.parent.exists():
        raise RuntimeError(f"refusing to overwrite {output.parent}")
    sources = {}
    events = {stage: [] for stage in ("backbone", "full", "prune")}
    excluded = []
    folds = {}
    for fold in ("F1", "F2", "F3", "F4", "F5"):
        for stage, stem in (
            ("backbone", "model_base"), ("full", "model_full"),
            ("prune", "model_pruned"),
        ):
            directory = ROOT / "results" / "e10_refit_v2" / fold / stage
            path = directory / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            sources[str(path.relative_to(ROOT))] = sha256_file(path)
            for extension, expected in summary["model_files_sha256"].items():
                if sha256_file(directory / f"{stem}.{extension}") != expected:
                    raise ValueError(f"model hash mismatch: {fold}/{stage}/{extension}")
            for dependency in summary["provenance"]["stage_dependencies"].values():
                if sha256_file(ROOT / dependency["path"]) != dependency["sha256"]:
                    raise ValueError(f"dependency mismatch: {dependency['path']}")
            endpoint = summary["val_metrics_H100"]
            # Older snapshots included exclusions in both lists; newer ones
            # separate them. Merge by ID and verify the declared denominator.
            row_by_id = {row["event_id"]: row for row in endpoint["events"]}
            for row in endpoint["excluded_short_events"]:
                row_by_id[row["event_id"]] = row
            rows = list(row_by_id.values())
            if len(rows) != endpoint["aggregate"]["n_declared"]:
                raise ValueError(f"population accounting mismatch: {fold}/{stage}")
            successful = [row for row in rows if row["status"] == "success"]
            recalculated = aggregate_trajectory_metrics(
                [row["trajectory_metrics"] for row in successful]
            )
            for metric in ("pooled_rmse", "median_rmse", "pooled_r2", "pooled_omega_r2"):
                if not np.isclose(
                    recalculated[metric], summary["val_metrics_H100"]["aggregate"][metric],
                    rtol=1e-12, atol=1e-12,
                ):
                    raise ValueError(f"aggregate mismatch: {fold}/{stage}/{metric}")
            events[stage].extend(dict(row, fold=fold) for row in rows)
            if stage == "prune":
                excluded.extend(summary["val_metrics_H100"]["excluded_short_events"])
                folds[fold] = {
                    "module_adoption_gate": summary["module_adoption_gate"],
                    "active_blocks": summary["active_blocks"],
                    "trajectory": summary["val_metrics_H100"]["aggregate"],
                }

    def aggregate(rows):
        successful = [row for row in rows if row["status"] == "success"]
        return {
            "n_declared": len(rows),
            "n_success": len(successful),
            "outcomes_successful": dict(Counter(row["outcome"] for row in successful)),
            "metrics_successful": aggregate_trajectory_metrics(
                [row["trajectory_metrics"] for row in successful]
            ),
            "n_failures": sum(row["status"] == "failure" for row in rows),
            "n_excluded_short": sum(row["status"] == "excluded_short" for row in rows),
        }

    final = events["prune"]
    if len({row["event_id"] for row in final}) != len(final):
        raise ValueError("duplicate validation events across folds")
    modal_path = ROOT / "results" / "e11_modal_validation_v2" / "summary.json"
    modal = json.loads(modal_path.read_text(encoding="utf-8"))
    sources[str(modal_path.relative_to(ROOT))] = sha256_file(modal_path)
    modal_summary = summarize_event_modal_matches(modal["per_event"])
    if modal_summary != modal["pooled"]:
        raise ValueError("modal summary does not reproduce from all event pairs")
    eligible_modal = sum(
        row["n_eligible_stable_validation_events"] for row in modal["per_fold"].values()
    )
    modal_summary.pop("per_event")
    modal_summary.update({
        "n_eligible": eligible_modal,
        "n_failed": len(modal["failures"]),
        "successful_event_coverage_of_eligible": len(modal["per_event"]) / eligible_modal,
        "failure_records": modal["failures"],
    })
    report = {
        "role": "post_selection_validation_audit_not_confirmatory",
        "input_sha256": sources,
        "per_fold": folds,
        "pooled_stages": {stage: aggregate(rows) for stage, rows in events.items()},
        "final_by_source": {
            source: aggregate([row for row in final if row["source_dataset"] == source])
            for source in sorted({row["source_dataset"] for row in final})
        },
        "excluded_outcomes": dict(Counter(row["outcome"] for row in excluded)),
        "modal": modal_summary,
        "limitations": [
            "H100 eligibility depends on observed trajectory length; all eligible events are stable.",
            "Unscaled joint angle/speed errors mix different physical quantities; component R2 is also reported.",
            "F1 backbone/full post-reload standardized losses used unit scales; use the corrected final gate.",
            "Modal summaries are conditional on successful equilibrium solving.",
            "Validation folds participated in selection; the legacy test build is consumed.",
        ],
    }
    atomic_write_json(output, report)
    print(json.dumps(report["pooled_stages"]["prune"], indent=2))


if __name__ == "__main__":
    main()
