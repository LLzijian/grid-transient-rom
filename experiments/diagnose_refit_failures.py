"""Frozen-model failure diagnosis: cut feasibility, Pm input and step sensitivity.

Uses validation only. Cut tests cover every validation event. Rollout controls
cover all H100-eligible validation events in F1 and F5, the two failure folds
identified by loop 0010. Observed Pm is an oracle diagnostic, not a forecast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_refit_v2 import atomic_write_json, git_provenance, sha256_file
from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kiki.models.prunable_energy_model import NONLINEAR_POTENTIAL, PrunableEnergyModel


def sine_cut_audit(incidence, coupling, demand):
    """Necessary condition for B K sin(B.T delta)=demand, all nontrivial cuts.

    Summing over a node subset S cancels internal flows. Since abs(sin)<=1,
    abs(sum(demand[S])) cannot exceed sum(K[crossing S]). Enumerating only
    subsets excluding the last node removes complementary duplicates.
    """
    incidence = np.asarray(incidence, dtype=float)
    coupling = np.asarray(coupling, dtype=float)
    demand = np.asarray(demand, dtype=float).reshape(-1)
    if np.any(coupling < 0) or abs(demand.sum()) > 1e-8:
        raise ValueError("cut audit requires nonnegative coupling and balanced demand")
    n = len(demand)
    masks = np.asarray([
        [(subset >> node) & 1 for node in range(n)]
        for subset in range(1, 2 ** (n - 1))
    ], dtype=float)
    cut_demand = np.abs(masks @ demand)
    capacity = np.abs(masks @ incidence) @ coupling
    excess = cut_demand - capacity
    worst = int(np.argmax(excess))
    tolerance = 1e-9 * max(1.0, float(np.max(capacity)), float(np.max(cut_demand)))
    return {
        "n_cuts": len(masks),
        "violated": bool(excess[worst] > tolerance),
        "maximum_excess": float(excess[worst]),
        "tolerance": tolerance,
        "worst_cut_nodes_zero_based": np.flatnonzero(masks[worst]).tolist(),
        "worst_cut_required_absolute_flow": float(cut_demand[worst]),
        "worst_cut_capacity": float(capacity[worst]),
        "passing_does_not_prove_equilibrium_exists": True,
    }


def main():
    output = ROOT / "results" / "e13_failure_diagnosis" / "summary.json"
    if output.parent.exists():
        raise RuntimeError(f"refusing to overwrite {output.parent}")
    git = git_provenance()
    config = json.loads((ROOT / "configs/dataset.json").read_text())
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    rows, hashes = [], {}
    for fold in ("F1", "F2", "F3", "F4", "F5"):
        print(f"Diagnosing {fold}", flush=True)
        path = ROOT / "results/e10_refit_v2" / fold / "prune/model_pruned.npz"
        hashes[fold] = {
            "npz": sha256_file(path), "json": sha256_file(path.with_suffix(".json"))
        }
        model = PrunableEnergyModel.load(path)
        if NONLINEAR_POTENTIAL in model.active_blocks:
            raise ValueError("sine-only capacity bound cannot audit nonlinear potential")
        split = load_split(config["dataset_dir"], view=config["view"],
                           split_name=config["split_name"], fold=fold)
        for event_id in split.val_ids:
            event = store.load_event(event_id)
            sample = adapt_event(event, inertia_prior=meta.M_prior,
                                 request=ModelDataRequest.initial_state_rollout("zoh"),
                                 representation=config["representation"])
            static = sample["model_inputs"]["static"]
            incidence = static["incidence"]
            clear = static["Pm_clear"].reshape(-1)
            demand = clear - static["Pm_equilibrium"] + incidence @ (
                model.coupling * np.sin(incidence.T @ static["equilibrium_delta"])
            )
            _, equilibrium = model.solve_postclear_equilibrium(static)
            row = {
                "fold": fold, "event_id": event_id,
                "source": event.metadata.source_dataset,
                "fault_line": event.metadata.fault_line,
                "outcome": event.metadata.outcome_label,
                "event_signature_sha256": event.metadata.event_signature_sha256,
                "cut": sine_cut_audit(incidence, model.coupling, demand),
                "equilibrium": equilibrium,
                "equilibrium_residual_pass": equilibrium["residual_inf_norm"] <= 1e-7,
            }
            if fold in ("F1", "F5") and len(event.t) >= 101:
                time = sample["model_inputs"]["exogenous"]["t"][:101]
                target = sample["training_context"]["targets"]
                observed = sample["training_context"]["Pm_observed"][:101]
                held = sample["model_inputs"]["exogenous"]["Pm"][:101]
                predictions, metrics = {}, {}
                for name, pm, interpolation, step in (
                    ("zoh_0.01", held, "zoh", 0.01),
                    ("zoh_0.0025", held, "zoh", 0.0025),
                    ("observed_0.01", observed, "linear", 0.01),
                ):
                    try:
                        d, w, _ = model.rollout_augmented(
                            static, time, {"Pm": pm, "interpolation": interpolation},
                            max_step=step,
                        )
                        if not np.isfinite(d).all() or not np.isfinite(w).all():
                            raise FloatingPointError("nonfinite prediction")
                        predictions[name] = np.concatenate((d, w), axis=1)
                        metrics[name] = trajectory_metrics(
                            target["delta"][:101], target["omega"][:101], d, w
                        )
                    except (ValueError, RuntimeError, FloatingPointError) as error:
                        metrics[name] = {"error": str(error)}
                row["controls"] = metrics
                if all(name in predictions for name in ("zoh_0.01", "zoh_0.0025")):
                    row["step_prediction_max_difference"] = float(np.max(np.abs(
                        predictions["zoh_0.01"] - predictions["zoh_0.0025"]
                    )))
                row["pm_max_change_H100"] = float(np.max(np.abs(observed - held)))
            rows.append(row)
    aggregate = {}
    for fold in ("F1", "F5"):
        selected = [row for row in rows if row["fold"] == fold and "controls" in row]
        aggregate[fold] = {
            name: {
                "metrics": aggregate_trajectory_metrics([
                    row["controls"][name] for row in selected
                    if "error" not in row["controls"][name]
                ]),
                "n_eligible": len(selected),
                "n_failed": sum("error" in row["controls"][name] for row in selected),
            }
            for name in ("zoh_0.01", "zoh_0.0025", "observed_0.01")
        }
    if git_provenance() != git:
        raise RuntimeError("Git changed during diagnosis")
    atomic_write_json(output, {
        "role": "post_selection_failure_diagnosis",
        "git": git, "dataset_build_id": meta.build_id, "model_sha256": hashes,
        "cut_scope": "all validation events, pure sine static coupling, frozen clearing Pm",
        "controls_scope": "all H100-eligible validation events in failure folds F1 and F5",
        "observed_Pm_role": "oracle sensitivity only; future input is not a causal forecast",
        "per_event": rows, "rollout_controls": aggregate,
    })
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
