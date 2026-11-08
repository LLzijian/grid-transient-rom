"""E3b: Pm-ODE gate re-run on long stable (highorder) traces at H=800.

E3's H100 (1 s) gate rejected the Pm equation because a ~2.5 s governor
recovery time constant is far longer than the gate window.  This sub-experiment
re-runs ONLY the Pm stage (the frozen M0-GMD backbone is loaded from E3's
checkpoints) with horizon H=800 (8 s), validated on the 20 highorder validation
events whose traces are >= 18 s.  The dense (near-critical/unstable) events are
excluded from the gate because their long-horizon rollouts diverge by
construction; the train weak system still sees all 209 events (dense included).

Gate and causal contract unchanged: Hurwitz + tau<=2.5 s, exact ZOH fallback,
deploy from Pm_clear, no outcome labels in fit/selection, test never opened.
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

HORIZON = 800
MAX_EFFECTIVE_TAU_S = 2.5
HIGHORDER = "ieee39_highorder_dataset"


def main() -> int:
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(config["dataset_dir"], view=config["view"],
                       split_name=config["split_name"], fold=config["development_fold"])
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])

    def adapt(ids):
        return [adapt_event(store.load_event(eid), inertia_prior=meta.M_prior,
                            request=request, representation=config["representation"])
                for eid in ids]

    train = adapt(split.train_ids)
    val = adapt(split.val_ids)
    highorder_val = [s for s in val
                     if s["training_context"]["metadata"]["source_dataset"] == HIGHORDER]
    if len(highorder_val) != 20:
        raise SystemExit(f"expected 20 highorder val events, got {len(highorder_val)}")

    ctx = RunContext(
        seed=324, epochs=100, horizon_steps=HORIZON,
        max_cases=max(len(train), len(highorder_val)),
        ridge_values=(1e-8, 1e-6, 1e-4, 1e-2),
        integral_windows=(1, 5, 20), rollout_refine=True,
        refinement_min_improvement=0.02, max_internal_step=0.01,
        weak_max_windows_per_case_horizon=120,
    )

    gmd = GraphModalDiscoveryM0Model.load(ROOT / "results" / "e03" / "checkpoints" / "gmd.npz")
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
        standalone_shortlist=8,
        max_internal_step=0.01,
    )
    print(f"[e3b] START {datetime.now(timezone.utc).isoformat()}", flush=True)
    fr = composite.fit(train, highorder_val, ctx)
    print(f"[e3b] DONE {datetime.now(timezone.utc).isoformat()}", flush=True)

    diag = composite.diagnostics
    selected = diag.get("selected_candidate", {})
    report = {
        "stage": "E3b", "timestamp": datetime.now(timezone.utc).isoformat(),
        "fold": config["development_fold"], "horizon_steps": HORIZON,
        "gate_subset": f"{HIGHORDER} (n={len(highorder_val)}, stable long traces; "
                       "dense excluded: long-horizon rollout diverges by construction)",
        "train_for_weak_system": len(train),
        "method": composite.method_name, "status": fr.status,
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

    # external H800 policies on highorder val
    def errors(pm_kind):
        rows, failures = [], 0
        for sample in highorder_val:
            initial, t, exo = sample_rollout_inputs(sample)
            h = min(HORIZON + 1, len(t))
            try:
                if pm_kind == "zoh":
                    pd, pw = gmd.rollout(initial, t[:h], {"Pm": exo["Pm"][:h], "interpolation": "zoh"})
                elif pm_kind == "pm":
                    pd, pw, _ = composite.rollout_with_pm(initial, t[:h], {})
                elif pm_kind == "oracle":
                    pm_obs = np.asarray(sample["training_context"]["Pm_observed"][:h], dtype=float)
                    pd, pw = gmd.rollout(initial, t[:h], {"Pm": pm_obs, "interpolation": "linear"})
                else:
                    raise ValueError(pm_kind)
            except Exception:
                failures += 1
                continue
            targets = sample["training_context"]["targets"]
            row = trajectory_metrics(targets["delta"][:h], targets["omega"][:h], pd, pw)
            rows.append(row)
        combined = [np.sqrt(0.5 * (r["delta_rmse"] ** 2 + r["omega_rmse"] ** 2)) for r in rows]
        return {
            "n": len(rows), "failure_rate": failures / max(len(highorder_val), 1),
            "mean_rmse": float(np.mean(combined)) if combined else float("nan"),
            "mean_delta_rmse": float(np.mean([r["delta_rmse"] for r in rows])) if rows else float("nan"),
            "mean_omega_rmse": float(np.mean([r["omega_rmse"] for r in rows])) if rows else float("nan"),
        }

    report["h800_policies"] = {
        "zoh_M0_GMD": errors("zoh"),
        "pm_ODE": errors("pm"),
        "observed_oracle": errors("oracle"),
    }

    run_dir = ROOT / "results" / "e03b"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "summary.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    composite.save(run_dir / "pm_long.npz")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
