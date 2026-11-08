"""E6: 5-fold confirmation of the modular-energy pipeline (paper main table).

Runs the exact E2 pipeline (backbone -> full -> prune) for one fold into an
immutable results/e06/{fold}/{stage}/ directory.  Stage order is enforced by
checking the previous stage's outputs; each directory refuses to overwrite.

F1 was already completed under results/e02_*; F2..F5 are driven through this
script.  A later consolidation step assembles the cross-fold main table, and
only at the very end (all folds frozen) is each fold's held-out test read once.

Usage:
  python run_e6.py <fold F1..F5> <stage backbone|full|prune> [bootstrap_draws]
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kiki.models.base import RunContext, sample_rollout_inputs
from kiki.models.prunable_energy_model import PrunableEnergyModel

HORIZON_STEPS = 100
MODULE_ADOPTION_MIN_IMPROVEMENT = 0.02


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    fold = sys.argv[1].upper()
    stage = sys.argv[2].lower()
    draws = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(config["dataset_dir"], view=config["view"],
                       split_name=config["split_name"], fold=fold)
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])

    def adapt(ids):
        return [adapt_event(store.load_event(eid), inertia_prior=meta.M_prior,
                            request=request, representation=config["representation"])
                for eid in ids]

    train, val = adapt(split.train_ids), adapt(split.val_ids)
    ctx = RunContext(
        seed=int(config["seed"]), epochs=100, horizon_steps=HORIZON_STEPS,
        max_cases=max(len(train), len(val)),
        ridge_values=(1e-8, 1e-6, 1e-4, 1e-2), integral_windows=(1, 5, 20),
        rollout_refine=True, refinement_min_improvement=0.02,
        max_internal_step=0.01, weak_max_windows_per_case_horizon=120,
    )

    run_dir = ROOT / "results" / "e06" / fold / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "summary.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")

    report = {
        "stage": f"E6-{fold}-{stage}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fold": fold, "representation": config["representation"],
        "pm_mode": config["pm_mode"], "n_train": len(train),
        "n_val": len(val), "bootstrap_draws": draws,
    }

    if stage == "backbone":
        base = PrunableEnergyModel(active_blocks=(), max_internal_step=0.01)
        fr = base.fit(train, val, ctx)
        proposal = PrunableEnergyModel.propose_candidates(
            train, base.node_damping, base.coupling,
            max_steps=1001, correlation_threshold=0.20, seed=int(config["seed"]),
        )
        report["fit"] = {"status": fr.status, "best_validation_loss": fr.best_validation_loss,
                         "selected_ridge": base.selected_ridge}
        report["physics_audit"] = base.physics_audit()
        report["val_metrics_H100"] = rollout_metrics(base, val)
        report["proposal"] = proposal
        base.save(run_dir / "model_base.npz")
        (run_dir / "proposal.json").write_text(json.dumps(proposal, indent=2, default=str),
                                               encoding="utf-8")
    elif stage == "full":
        proposal = json.loads((run_dir.parent / "backbone" / "proposal.json").read_text())
        model = PrunableEnergyModel(
            active_blocks=proposal["active_blocks"],
            memory_taus=proposal["memory_taus"],
            resonances=proposal["resonances"],
            node_tau=0.2, potential_scale=0.25,
            max_internal_step=0.01, enforce_identifiability=True,
        )
        fr = model.fit(train, val, ctx)
        report["fit"] = {"status": fr.status, "best_validation_loss": fr.best_validation_loss,
                         "selected_ridge": model.selected_ridge}
        report["active_blocks"] = list(model.active_blocks)
        report["block_parameter_norms"] = fr.details.get("block_parameter_norms", {})
        report["physics_audit"] = model.physics_audit()
        report["val_metrics_H100"] = rollout_metrics(model, val)
        model.save(run_dir / "model_full.npz")
    elif stage == "prune":
        model = PrunableEnergyModel(active_blocks=None, max_internal_step=0.01,
                                    enforce_identifiability=True)
        saved = PrunableEnergyModel.load(run_dir.parent / "full" / "model_full.npz")
        model.__dict__.update(saved.__dict__)
        pruned = model.prune(train, val, ctx, tolerance=0.02,
                             closed_loop_refine_cases=0, closed_loop_refine_evaluations=0,
                             bootstrap_draws=draws, seed=int(config["seed"]),
                             minimum_module_improvement=MODULE_ADOPTION_MIN_IMPROVEMENT)
        report["module_adoption_minimum_improvement"] = MODULE_ADOPTION_MIN_IMPROVEMENT
        report["active_blocks"] = list(pruned.active_blocks)
        report["pruning_trace"] = getattr(pruned, "pruning_trace", None)
        report["physics_audit"] = pruned.physics_audit()
        report["val_metrics_H100"] = rollout_metrics(pruned, val)
        pruned.save(run_dir / "model_pruned.npz")
    else:
        raise SystemExit(f"unknown stage: {stage}")

    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


def rollout_metrics(model, samples, horizon: int = HORIZON_STEPS):
    rows, n_failed = [], 0
    for sample in samples:
        initial, t, exo = sample_rollout_inputs(sample)
        h = min(horizon + 1, len(t))
        try:
            pd, pw = model.rollout(initial, t[:h], {"Pm": exo["Pm"][:h],
                                                    "interpolation": exo["interpolation"]})
        except Exception:
            n_failed += 1
            continue
        targets = sample["training_context"]["targets"]
        row = trajectory_metrics(targets["delta"][:h], targets["omega"][:h], pd, pw)
        row["event_id"] = sample["event_id"]
        rows.append(row)
    agg = aggregate_trajectory_metrics(rows)
    agg["n_rollout_failures"] = n_failed
    return agg


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
