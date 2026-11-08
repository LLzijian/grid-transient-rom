"""E2: modular energy-structured model on the development fold (F1).

Pipeline (mirrors kron-reduce data_level_structure_selection but on the 5-fold
train/val split, generator_10):
  stage=backbone  PrunableEnergyModel(active_blocks=()) fit -> base params + candidate proposal
  stage=full      build full modular model from proposal, fit, report block norms + val metrics
  stage=prune     whole-block backward deletion with noninferiority bootstrap

Only train/val are touched; test is never opened in this stage.
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
from kiki.models.prunable_energy_model import PrunableEnergyModel

HORIZON_STEPS = 100
MODULE_ADOPTION_MIN_IMPROVEMENT = 0.02


def load_samples(config, store, meta, split):
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

    return adapt(split.train_ids), adapt(split.val_ids)


def run_context(config, train, val):
    return RunContext(
        seed=config["seed"],
        epochs=100,
        horizon_steps=HORIZON_STEPS,
        max_cases=max(len(train), len(val)),
        ridge_values=(1e-8, 1e-6, 1e-4, 1e-2),
        integral_windows=(1, 5, 20),
        rollout_refine=True,
        refinement_min_improvement=0.02,
        max_internal_step=0.01,
        weak_max_windows_per_case_horizon=120,
    )


def rollout_metrics(model, samples, horizon: int):
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
    return agg


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "backbone"
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(
        config["dataset_dir"],
        view=config["view"],
        split_name=config["split_name"],
        fold=config["development_fold"],
    )
    train, val = load_samples(config, store, meta, split)
    ctx = run_context(config, train, val)

    run_dir = ROOT / "results" / f"e02_{stage}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "summary.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")

    report = {
        "stage": f"E2-{stage}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fold": config["development_fold"],
        "representation": config["representation"],
        "pm_mode": config["pm_mode"],
        "n_train": len(train),
        "n_val": len(val),
    }

    if stage == "backbone":
        base = PrunableEnergyModel(active_blocks=(), max_internal_step=0.01)
        fr = base.fit(train, val, ctx)
        proposal = PrunableEnergyModel.propose_candidates(
            train,
            base.node_damping,
            base.coupling,
            max_steps=1001,
            correlation_threshold=0.20,
            seed=int(config["seed"]),
        )
        report["fit"] = {
            "status": fr.status,
            "best_validation_loss": fr.best_validation_loss,
            "selected_ridge": base.selected_ridge,
            "details": {k: v for k, v in fr.details.items() if k != "raw_parameters"},
        }
        report["physics_audit"] = base.physics_audit()
        report["val_metrics_H100"] = rollout_metrics(base, val, HORIZON_STEPS)
        report["proposal"] = proposal
        base.save(run_dir / "model_base.npz")
        (run_dir / "proposal.json").write_text(
            json.dumps(proposal, indent=2, default=str), encoding="utf-8"
        )
    elif stage == "full":
        proposal = json.loads(
            (ROOT / "results" / "e02_backbone" / "proposal.json").read_text()
        )
        model = PrunableEnergyModel(
            active_blocks=proposal["active_blocks"],
            memory_taus=proposal["memory_taus"],
            resonances=proposal["resonances"],
            node_tau=0.2,
            potential_scale=0.25,
            max_internal_step=0.01,
            enforce_identifiability=True,
        )
        fr = model.fit(train, val, ctx)
        report["fit"] = {
            "status": fr.status,
            "best_validation_loss": fr.best_validation_loss,
            "selected_ridge": model.selected_ridge,
            "details": {k: v for k, v in fr.details.items() if k != "raw_parameters"},
        }
        report["active_blocks"] = list(model.active_blocks)
        report["block_parameter_norms"] = fr.details.get("block_parameter_norms", {})
        report["physics_audit"] = model.physics_audit()
        report["val_metrics_H100"] = rollout_metrics(model, val, HORIZON_STEPS)
        model.save(run_dir / "model_full.npz")
    elif stage == "prune":
        # rebuild full model from diagnostics, then prune
        model = PrunableEnergyModel(
            active_blocks=None,
            max_internal_step=0.01,
            enforce_identifiability=True,
        )
        saved = PrunableEnergyModel.load(ROOT / "results" / "e02_full" / "model_full.npz")
        model.__dict__.update(saved.__dict__)
        pruned = model.prune(
            train,
            val,
            ctx,
            tolerance=0.02,
            closed_loop_refine_cases=0,
            closed_loop_refine_evaluations=0,
            bootstrap_draws=int(sys.argv[2]) if len(sys.argv) > 2 else 200,
            seed=int(config["seed"]),
            minimum_module_improvement=MODULE_ADOPTION_MIN_IMPROVEMENT,
        )
        report["module_adoption_minimum_improvement"] = MODULE_ADOPTION_MIN_IMPROVEMENT
        report["active_blocks"] = list(pruned.active_blocks)
        report["pruning_trace"] = getattr(pruned, "pruning_trace", None)
        report["physics_audit"] = pruned.physics_audit()
        report["val_metrics_H100"] = rollout_metrics(pruned, val, HORIZON_STEPS)
        pruned.save(run_dir / "model_pruned.npz")
    else:
        raise SystemExit(f"unknown stage: {stage}")

    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
