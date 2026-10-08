"""Exact frozen-M3 storage identity and separate full-order trajectory audit."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from run_private_m2 import development_data
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.stats import rankdata, spearmanr

from kiki.models.private_pm import MechanicalROM


def energy_record(model, state, time):
    h, supply = model.energy(state), model.dissipation(state)
    scale = max(1., abs(float(h[0])))
    difference = np.diff(h)
    balance = h-h[0]-cumulative_trapezoid(supply, time, initial=0)
    return {"initial_H": float(h[0]), "final_H": float(h[-1]),
            "net_relative_change": float((h[-1]-h[0])/scale),
            "max_positive_step_relative": float(np.max(difference, initial=0)/scale),
            "upward_variation_relative": float(np.sum(np.maximum(difference, 0))/scale),
            "positive_step_fraction": float(np.mean(difference > 1e-6*scale)),
            "quadrature_balance_max_relative": float(np.max(np.abs(balance))/scale)}, h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("private_data/ieee39_postfault"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sets, metadata, excluded = development_data(args.data)
    events = sets["train"]+sets["validation"]
    train_ids = {e["event_id"] for e in sets["train"]}
    model = MechanicalROM.load("private_data/task2_m3_0020/model.npz")
    equilibrium = np.r_[np.zeros(20), model.pm_star]
    rng = np.random.default_rng(28)
    sample = equilibrium+rng.normal(0, .2, (50, 30))
    analytic = model.energy_gradient(sample)
    difference = np.column_stack([(model.energy(sample+1e-6*v)-model.energy(sample-1e-6*v))/2e-6 for v in np.eye(30)])
    gradient_error = float(np.max(np.abs(analytic-difference)/np.maximum(1., np.abs(analytic))))
    identity = np.sum(analytic*model.field(sample), axis=1)-model.dissipation(sample)
    identity_error = float(np.max(np.abs(identity)/np.maximum(1., np.abs(model.dissipation(sample)))))
    stiffness = (model.incidence*(model.coupling*np.cos(model.delta_star@model.incidence)))@model.incidence.T
    # Fix generator10 angle; this removes exactly the rotational gauge.
    curvature = np.linalg.eigvalsh(stiffness[:9, :9])
    positive = bool(np.all(model.mass > 0) and np.all(model.gain > 0) and np.all(model.rate > 0) and np.all(model.damping >= 0))
    initial = np.array([np.r_[e["delta_sync"][0], e["omega_sync"][0], e["Pm_raw"][0]] for e in events])
    time_axis = np.arange(801)*.01
    predictions = model.rollout(initial, None, time_axis)
    records, energy_arrays = [], {}
    for pred, event in zip(predictions, events, strict=True):
        raw = np.c_[event["delta_sync"], event["omega_sync"], event["Pm_raw"]]
        reduced, h_reduced = energy_record(model, pred, time_axis)
        observed, h_observed = energy_record(model, raw, event["t"])
        row = metadata[event["event_id"]]
        records.append({"event_id": event["event_id"], "split": "train" if event["event_id"] in train_ids else "validation",
                        "group": row["fault_bus"], "label": row["label"], "duration_s": float(row["fault_duration_s"]),
                        "observed_end_s": float(event["t"][-1]), "reduced": reduced, "observed": observed})
        energy_arrays[event["event_id"]+"_reduced"] = h_reduced
        energy_arrays[event["event_id"]+"_observed"] = h_observed
    stable = [r for r in records if r["label"] == "stable"]
    selected = sorted(stable, key=lambda r: r["reduced"]["initial_H"], reverse=True)[:3]
    integration = []
    for row in selected:
        index = next(i for i, e in enumerate(events) if e["event_id"] == row["event_id"])
        ref = solve_ivp(lambda t, x: model.field(x), (0, 8), initial[index], t_eval=time_axis,
                         method="DOP853", rtol=1e-10, atol=1e-11)
        if not ref.success:
            raise RuntimeError(ref.message)
        fine = model.rollout(initial[index], None, np.arange(1601)*.005, max_step=.0025)
        fine_stats, _ = energy_record(model, fine, np.arange(1601)*.005)
        integration.append({"event_id": row["event_id"],
                            "DOP853_max_absolute_state_error": float(np.max(np.abs(ref.y.T-predictions[index]))),
                            "half_step_max_absolute_state_error": float(np.max(np.abs(fine[::2]-predictions[index]))),
                            "coarse_quadrature_balance": row["reduced"]["quadrature_balance_max_relative"],
                            "fine_quadrature_balance": fine_stats["quadrature_balance_max_relative"]})
    strata = {}
    for split in ("train", "validation"):
        subset = [r for r in records if r["split"] == split]
        for label in ("stable", "unstable"):
            cohort = [r for r in subset if r["label"] == label]
            strata[split+"_"+label] = {"n": len(cohort)}
            for source in ("reduced", "observed"):
                strata[split+"_"+label][source] = {
                    "net_increase_events": sum(r[source]["net_relative_change"] > 1e-6 for r in cohort),
                    "upward_step_events": sum(r[source]["max_positive_step_relative"] > 1e-6 for r in cohort),
                    "median_net_relative_change": float(np.median([r[source]["net_relative_change"] for r in cohort])),
                    "median_upward_variation_relative": float(np.median([r[source]["upward_variation_relative"] for r in cohort])),
                    "max_positive_step_relative": max(r[source]["max_positive_step_relative"] for r in cohort)}
        scores = np.array([r["observed"]["initial_H"] for r in subset])
        unstable = np.array([r["label"] == "unstable" for r in subset])
        n1, n0 = sum(unstable), sum(~unstable)
        auc = (rankdata(scores)[unstable].sum()-n1*(n1+1)/2)/(n1*n0)
        strata[split+"_association"] = {"initial_H_unstable_auc_descriptive": float(auc),
             "initial_H_duration_spearman": float(spearmanr(scores, [r["duration_s"] for r in subset]).statistic),
             "group_duration_spearman": {g: float(spearmanr([r["observed"]["initial_H"] for r in subset if r["group"] == g],
                         [r["duration_s"] for r in subset if r["group"] == g]).statistic) for g in sorted({r["group"] for r in subset})}}
    max_growth = max(r["reduced"]["max_positive_step_relative"] for r in stable)
    passed = positive and curvature.min() > 0 and gradient_error <= 1e-6 and identity_error <= 1e-10 and max_growth <= 1e-6
    result = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "seconds": time.perf_counter()-started, "screen_pass": bool(passed),
              "positive_parameters": positive, "gradient_relative_error": gradient_error,
              "identity_relative_error": identity_error, "gauge_fixed_potential_curvature": curvature.tolist(),
              "equilibrium_H": float(model.energy(equilibrium)), "strata": strata,
              "integration_checks": integration, "excluded_predecision": excluded, "records": records,
              "scope": "M3 local storage and reduced-ODE dissipation only; no critical energy or FOM certificate"}
    np.savez_compressed(args.output / "energies.npz", **energy_arrays)
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
