"""E3: causal mechanical-power ODE module (Pm own equation) on F1.

Mirrors the kron-reduce M0-GMD-PM chain, rebuilt on the kiki canonical
generator_10 dev fold (ZOH causal contract).  Three explicit stages:

  m03eg : TransientEmfM0Model      -- 3rd-order M0 with nodal transient EMF
  gmd   : GraphModalDiscoveryM0Model -- low-rank graph closure, frozen backbone
  pm    : CausalPmGraphModalM0Model  -- M0-GMD + stable GraphModalPmDynamics

The Pm port dynamics live on the zero-total-power subspace::

    a   = U_p.T (Pm - Pm_equilibrium)          U_p spans 1.T p = 0
    v   = U_p.T Pi_power(omega)
    a'  = A a + B v                            A Hurwitz (spectral margin)

Deployment evolves mechanical power causally from the clearing value Pm_clear;
the exact A=B=0 ZOH policy is a candidate and the fallback.  Gate (internal):
Hurwitz + causal H100 validation improvement >= 2% over ZOH, else ZOH stays.

Every stage runs its physics audit.  H100 source-stratified trajectory error is
reported for three policies on the same M0-GMD backbone:
  ZOH    (Pm held at Pm_clear)          -- causal baseline
  PM     (Pm from the fitted ODE)       -- causal proposed
  oracle (observed Pm schedule, linear) -- non-causal upper bound

Test is never opened; outcome labels are not used in fit/selection.
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
from kiki.evaluation.metrics import trajectory_metrics
from kiki.models.base import RunContext, sample_rollout_inputs
from kiki.models.graph_modal_discovery import GraphModalDiscoveryM0Model
from kiki.models.pm_modal_discovery import CausalPmGraphModalM0Model, GraphModalPmDynamics
from kiki.models.transient_emf_m0 import TransientEmfM0Model

HORIZON_STEPS = 100
MAX_EFFECTIVE_TAU_S = 2.5  # ANDES TGOV1N T3 ~2.1 s, 2.5 s conservative cap


def load(config, smoke: int):
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(
        config["dataset_dir"], view=config["view"],
        split_name=config["split_name"], fold=config["development_fold"],
    )
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])

    def adapt(ids):
        return [
            adapt_event(store.load_event(eid), inertia_prior=meta.M_prior,
                        request=request, representation=config["representation"])
            for eid in ids
        ]

    train, val = adapt(split.train_ids), adapt(split.val_ids)
    if smoke > 0:
        train, val = train[:smoke], val[:smoke]
    return train, val


def run_context(train, val):
    return RunContext(
        seed=324, epochs=100, horizon_steps=HORIZON_STEPS,
        max_cases=max(len(train), len(val)),
        ridge_values=(1e-8, 1e-6, 1e-4, 1e-2),
        integral_windows=(1, 5, 20),
        rollout_refine=True, refinement_min_improvement=0.02,
        max_internal_step=0.01, weak_max_windows_per_case_horizon=120,
    )


def rollout_errors(model, samples, *, pm_kind: str, horizon: int = HORIZON_STEPS):
    """source-stratified H100 delta/omega rmse per event for a rollout policy."""
    rows, failures = [], 0
    for sample in samples:
        initial, t, exo = sample_rollout_inputs(sample)
        h = min(horizon + 1, len(t))
        try:
            if pm_kind == "zoh":
                pd, pw = model.rollout(initial, t[:h], {"Pm": exo["Pm"][:h], "interpolation": "zoh"})
            elif pm_kind == "oracle":
                pm_obs = np.asarray(sample["training_context"]["Pm_observed"][:h], dtype=float)
                pd, pw = model.rollout(initial, t[:h], {"Pm": pm_obs, "interpolation": "linear"})
            elif pm_kind == "pm":
                pd, pw, _pm = model.rollout_with_pm(initial, t[:h], {})
            else:
                raise ValueError(pm_kind)
        except Exception:
            failures += 1
            continue
        targets = sample["training_context"]["targets"]
        row = trajectory_metrics(targets["delta"][:h], targets["omega"][:h], pd, pw)
        row["source"] = sample["training_context"]["metadata"]["source_dataset"]
        row["event_id"] = sample["event_id"]
        rows.append(row)
    out = {"failure_rate": failures / max(len(samples), 1)}
    if rows:
        combined = np.sqrt(0.5 * (np.asarray([r["delta_rmse"] for r in rows]) ** 2
                                  + np.asarray([r["omega_rmse"] for r in rows]) ** 2))
        out["mean_rmse"] = float(np.mean(combined))
        out["mean_delta_rmse"] = float(np.mean([r["delta_rmse"] for r in rows]))
        out["mean_omega_rmse"] = float(np.mean([r["omega_rmse"] for r in rows]))
        sources = {}
        for source in sorted({r["source"] for r in rows}):
            group = [r for r in rows if r["source"] == source]
            sources[source] = {
                "n": len(group),
                "mean_rmse": float(np.mean([np.sqrt(0.5 * (r["delta_rmse"] ** 2 + r["omega_rmse"] ** 2)) for r in group])),
                "mean_delta_rmse": float(np.mean([r["delta_rmse"] for r in group])),
                "mean_omega_rmse": float(np.mean([r["omega_rmse"] for r in group])),
            }
        out["by_source"] = sources
        out["per_case"] = rows
    else:
        out["by_source"] = {}
        out["per_case"] = []
    return out


def guard(stage: str, fn):
    """Run one stage; on exception record traceback and re-raise."""
    print(f"[e3:{stage}] START {datetime.now(timezone.utc).isoformat()}", flush=True)
    try:
        result = fn()
    except Exception:
        print(f"[e3:{stage}] FAILED\n{traceback.format_exc()}", flush=True)
        raise
    print(f"[e3:{stage}] DONE {datetime.now(timezone.utc).isoformat()}", flush=True)
    return result


def main() -> int:
    args = [a for a in sys.argv[1:]]
    smoke = 0
    if "--smoke" in args:
        smoke = 6
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    train, val = load(config, smoke)
    ctx = run_context(train, val)
    run_dir = ROOT / "results" / "e03"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        raise SystemExit(f"refusing to overwrite {summary_path}")
    report = {
        "stage": "E3", "timestamp": datetime.now(timezone.utc).isoformat(),
        "fold": config["development_fold"], "representation": config["representation"],
        "pm_mode": config["pm_mode"], "n_train": len(train), "n_val": len(val),
        "smoke": bool(smoke),
        "maximum_effective_time_constant_s": MAX_EFFECTIVE_TAU_S,
        "stages": {}, "h100_policies": {},
    }

    m03eg = guard("m03eg", lambda: TransientEmfM0Model(max_internal_step=0.01))
    fr_m = guard("m03eg", lambda: m03eg.fit(train, val, ctx))
    report["stages"]["m03eg"] = {
        "method": m03eg.method_name, "status": fr_m.status,
        "best_validation_loss": fr_m.best_validation_loss,
        "physics_audit": m03eg.physics_audit(),
    }
    m03eg.save(checkpoints / "m03eg.npz")

    gmd = GraphModalDiscoveryM0Model(backbone=m03eg, max_internal_step=0.01)
    fr_g = guard("gmd", lambda: gmd.fit(train, val, ctx))
    report["stages"]["gmd"] = {
        "method": gmd.method_name, "status": fr_g.status,
        "best_validation_loss": fr_g.best_validation_loss,
        "selected_structure": {k: v for k, v in fr_g.details.items()
                               if k in {"modal_rank", "taus", "ridge",
                                        "relative_group_threshold", "active_groups",
                                        "coefficient_scale", "weak_residual_rmse"}},
        "physics_audit": gmd.physics_audit(),
    }
    gmd.save(checkpoints / "gmd.npz")

    composite = CausalPmGraphModalM0Model(
        backbone=gmd,
        pm_dynamics=GraphModalPmDynamics(
            weak_windows=(1, 5, 20),
            ridge_values=(1e-8, 1e-6, 1e-4, 1e-2, 1e-1),
            relative_thresholds=(0.0, 0.01, 0.03, 0.1),
            spectral_margins=(1e-4, 0.05, 0.2, 0.4, 0.5),
        ),
        maximum_effective_time_constant=MAX_EFFECTIVE_TAU_S,
        minimum_validation_improvement=0.02,
        standalone_shortlist=16,
        max_internal_step=0.01,
    )
    fr_p = guard("pm", lambda: composite.fit(train, val, ctx))
    diag = composite.diagnostics
    selected = diag.get("selected_candidate", {})
    report["stages"]["pm"] = {
        "method": composite.method_name, "status": fr_p.status,
        "promoted": bool(diag.get("promoted")),
        "validation_improvement_vs_ZOH": diag.get("validation_improvement_vs_ZOH"),
        "selected_candidate": {k: v for k, v in selected.items()
                               if k not in {"state_matrix", "input_matrix"}},
        "best_candidate_before_gate": {k: v for k, v in diag.get("best_candidate_before_gate", {}).items()
                                       if k not in {"state_matrix", "input_matrix"}},
        "fallback_candidate": {k: v for k, v in diag.get("fallback_candidate", {}).items()
                               if k not in {"state_matrix", "input_matrix"}},
        "eigenvalues_real": diag.get("eigenvalues_real"),
        "eigenvalues_imag": diag.get("eigenvalues_imag"),
        "effective_time_constants": diag.get("effective_time_constants"),
        "weak_system": diag.get("weak_system"),
        "physics_audit": composite.physics_audit(),
    }
    composite.save(checkpoints / "pm.npz")

    if not smoke:
        report["h100_policies"]["zoh_M0_GMD"] = rollout_errors(gmd, val, pm_kind="zoh")
        report["h100_policies"]["pm_ODE"] = rollout_errors(composite, val, pm_kind="pm")
        report["h100_policies"]["observed_oracle"] = rollout_errors(gmd, val, pm_kind="oracle")

    summary_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
