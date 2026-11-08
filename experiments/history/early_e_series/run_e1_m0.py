"""E1: M0 backbone reproduction on the development fold (F1).

Fits M0Model on F1 train, selects ridge on F1 validation (inside fit),
then evaluates closed-loop ZOH rollouts on validation only. Test is never
opened in this stage.

Acceptance (from kron-reduce integral-v2): validation mean RMSE of order 0.3,
COI drift ~1e-15, zero rollout failures.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kiki.models.base import RunContext, sample_rollout_inputs
from kiki.models.m0 import M0Model

HORIZON_STEPS = 100


def rollout_metrics(model, samples, horizon: int) -> dict:
    rows = []
    n_failed = 0
    for sample in samples:
        initial, t, exo = sample_rollout_inputs(sample)
        h = min(horizon + 1, len(t))
        try:
            pd, pw = model.rollout(
                initial, t[:h], {"Pm": exo["Pm"][:h], "interpolation": exo["interpolation"]}
            )
        except Exception:
            n_failed += 1
            continue
        targets = sample["training_context"]["targets"]
        row = trajectory_metrics(targets["delta"][:h], targets["omega"][:h], pd, pw)
        row["event_id"] = sample["event_id"]
        rows.append(row)
    agg = aggregate_trajectory_metrics(rows)
    agg["n_rollout_failures"] = n_failed
    return {"per_case": rows, "aggregate": agg}


def main() -> int:
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(
        config["dataset_dir"],
        view=config["view"],
        split_name=config["split_name"],
        fold=config["development_fold"],
    )
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])

    def adapt(ids):
        return [
            adapt_event(
                store.load_event(eid),
                inertia_prior=meta.M_prior,
                request=request,
                representation=config["representation"],
            )
            for eid in ids
        ]

    train = adapt(split.train_ids)
    val = adapt(split.val_ids)

    context = RunContext(
        seed=config["seed"],
        epochs=100,
        horizon_steps=HORIZON_STEPS,
        max_cases=max(len(train), len(val)),
        ridge_values=(1e-8, 1e-6, 1e-4, 1e-2),
        integral_windows=(1, 5, 20),
        rollout_refine=True,
        refinement_min_improvement=0.02,
        max_internal_step=0.01,
    )

    model = M0Model()
    fit_result = model.fit(train, val, context)

    val_eval = rollout_metrics(model, val, HORIZON_STEPS)

    # Integration convergence audit on the first 3 validation cases.
    conv = []
    for sample in val[:3]:
        initial, t, exo = sample_rollout_inputs(sample)
        conv.append(model.convergence_audit(initial, t, exo))

    report = {
        "stage": "E1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fold": config["development_fold"],
        "representation": config["representation"],
        "pm_mode": config["pm_mode"],
        "horizon_steps": HORIZON_STEPS,
        "n_train": len(train),
        "n_val": len(val),
        "fit": {
            "status": fit_result.status,
            "best_validation_loss": fit_result.best_validation_loss,
            "selected_ridge": model.selected_ridge,
            "details": fit_result.details,
        },
        "physics_audit": model.physics_audit(),
        "validation_summary": model.validation_summary(val, context),
        "val_metrics_H100": val_eval["aggregate"],
        "convergence_audit": conv,
    }

    run_dir = ROOT / "results" / "e01_m0_f1"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "summary.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    model.save(run_dir / "model_m0_f1.npz")
    print(json.dumps({k: v for k, v in report.items() if k != "validation_summary"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
