"""Segment timing inside a real full-model fit with full-size val set."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.models.base import RunContext, sample_rollout_inputs
from kiki.models.prunable_energy_model import PrunableEnergyModel

HORIZON_STEPS = 100


def main():
    ntrain = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(config["dataset_dir"], view=config["view"],
                       split_name=config["split_name"], fold="F1")
    req = ModelDataRequest.initial_state_rollout("zoh")

    def adapt(ids, n):
        out = []
        for eid in list(ids)[:n]:
            out.append(adapt_event(store.load_event(eid), inertia_prior=meta.M_prior,
                                   request=req, representation="generator_10"))
        return out

    train = adapt(split.train_ids, ntrain)
    val = adapt(split.val_ids, 43)
    proposal = json.loads((ROOT / "results" / "e02_backbone" / "proposal.json").read_text())
    model = PrunableEnergyModel(
        active_blocks=proposal["active_blocks"],
        memory_taus=proposal["memory_taus"],
        resonances=proposal["resonances"],
        node_tau=0.2, potential_scale=0.25, max_internal_step=0.01,
    )
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ctx = RunContext(seed=324, horizon_steps=HORIZON_STEPS, max_cases=10000,
                     ridge_values=(1e-8,), integral_windows=(1, 5, 20),
                     max_internal_step=0.01, weak_max_windows_per_case_horizon=cap)
    t0 = time.time()
    matrix, vector, system = model.build_weak_system(train, ctx)
    t1 = time.time()
    print(f"build_weak_system({ntrain}): {t1-t0:.1f}s rows={matrix.shape[0]} cols={matrix.shape[1]}")
    # replicate one ridge solve + validation_summary(43 val)
    ridge = 1e-8
    from kiki.models.prunable_energy_model import _pm_equilibrium
    n_nodes = len(train[0]["model_inputs"]["static"]["mass"])
    n_edges = train[0]["model_inputs"]["static"]["incidence"].shape[1]
    scale = np.sqrt(np.mean(matrix**2, axis=0)); scale = np.where(scale > 1e-14, scale, 1.0)
    theta, diag = model._solve(matrix, vector, ridge)  # noqa
    model._assign_parameters(theta, n_nodes, n_edges)
    t2 = time.time()
    print(f"solve_1ridge: {t2-t1:.1f}s")
    summary = model.validation_summary(val, ctx)
    t3 = time.time()
    print(f"validation_summary(43 val): {t3-t2:.1f}s -> {summary.get('standardized_loss')}")
    # per-event H100 rollout cost on fitted model
    t_acc = 0.0
    for s in val[:6]:
        initial, t, exo = sample_rollout_inputs(s)
        t0 = time.time(); model.rollout(initial, t[:101], {"Pm": exo["Pm"][:101], "interpolation": exo["interpolation"]}); t_acc += time.time()-t0
    print(f"mean per-event H100 rollout (6): {t_acc/6:.3f}s")


if __name__ == "__main__":
    main()
