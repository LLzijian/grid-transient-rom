"""Teacher-forced impulse balance and zero-hidden initialization controls.

Frozen F1/F5 final models, all H100-eligible validation events. Force residuals
use observed states only as a diagnostic; they are not forecasting scores.
No parameters or initial states are fitted to validation outcomes.
"""

from __future__ import annotations

import copy
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
from kiki.models.prunable_energy_model import PrunableEnergyModel


def forced_impulse(model, static, time, delta, omega, pm, *, interpolation, zero_hidden=False):
    """Integrate model acceleration along prescribed states, without derivatives of data."""
    time, delta, omega, pm = map(np.asarray, (time, delta, omega, pm))
    hidden = model._pack_hidden(model.initial_hidden(omega[0], static))
    if zero_hidden:
        hidden = np.zeros_like(hidden)
    integral = np.zeros(omega.shape[1])
    integrated = [integral.copy()]
    for index, interval in enumerate(np.diff(time)):
        count = max(1, int(np.ceil(interval / 0.0025)))
        step = interval / count

        def field(h, fraction, index=index):
            d = delta[index] + fraction * (delta[index + 1] - delta[index])
            w = omega[index] + fraction * (omega[index + 1] - omega[index])
            power = pm[index] if interpolation == "zoh" else (
                pm[index] + fraction * (pm[index + 1] - pm[index])
            )
            _, acceleration, dh = model._flat_field(d, w, h, power, static)
            return acceleration, dh

        for part in range(count):
            q0, qh, q1 = part / count, (part + 0.5) / count, (part + 1) / count
            a1, h1 = field(hidden, q0)
            a2, h2 = field(hidden + step * h1 / 2, qh)
            a3, h3 = field(hidden + step * h2 / 2, qh)
            a4, h4 = field(hidden + step * h3, q1)
            integral += step * (a1 + 2 * a2 + 2 * a3 + a4) / 6
            hidden += step * (h1 + 2 * h2 + 2 * h3 + h4) / 6
        integrated.append(integral.copy())
    residual = omega - omega[[0]] - np.asarray(integrated)
    if not np.isfinite(residual).all():
        raise FloatingPointError("nonfinite forced impulse residual")
    mass = np.asarray(static["mass"])
    force_impulse = residual * mass
    late_start = max(0, int(np.searchsorted(time, 0.1, side="right")) - 1)
    late_force = (force_impulse[-1] - force_impulse[late_start]) / (
        time[-1] - time[late_start]
    )
    angle_integral = np.cumsum(
        0.5 * (omega[1:] + omega[:-1]) * np.diff(time)[:, None], axis=0
    )
    kinematic_error = delta[1:] - delta[[0]] - angle_integral
    return {
        "cumulative_speed_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "terminal_force_impulse": force_impulse[-1].tolist(),
        "terminal_force_impulse_l2": float(np.linalg.norm(force_impulse[-1])),
        "late_mean_force_residual_l2": float(np.linalg.norm(late_force)),
        "late_window_actual_seconds": [float(time[late_start]), float(time[-1])],
        "kinematic_integration_error_max": float(np.max(np.abs(kinematic_error))),
    }


def main():
    output = ROOT / "results/e14_force_balance/summary.json"
    if output.parent.exists():
        raise RuntimeError(f"refusing to overwrite {output.parent}")
    git = git_provenance()
    config = json.loads((ROOT / "configs/dataset.json").read_text())
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    rows, exclusions, hashes = [], [], {}
    for fold in ("F1", "F5"):
        path = ROOT / "results/e10_refit_v2" / fold / "prune/model_pruned.npz"
        hashes[fold] = {"npz": sha256_file(path), "json": sha256_file(path.with_suffix(".json"))}
        model = PrunableEnergyModel.load(path)
        zero_model = copy.deepcopy(model)
        initializer = zero_model.initial_hidden
        zero_model.initial_hidden = lambda omega, static, init=initializer: {
            name: np.zeros_like(value) for name, value in init(omega, static).items()
        }
        split = load_split(config["dataset_dir"], view=config["view"],
                           split_name=config["split_name"], fold=fold)
        for event_id in split.val_ids:
            metadata = store.metadata(event_id)
            if metadata.valid_length < 101:
                exclusions.append({"fold": fold, "event_id": event_id,
                                   "outcome": metadata.outcome_label})
                continue
            event = store.load_event(event_id)
            sample = adapt_event(event, inertia_prior=meta.M_prior,
                                 request=ModelDataRequest.initial_state_rollout("zoh"),
                                 representation=config["representation"])
            static = sample["model_inputs"]["static"]
            target = sample["training_context"]["targets"]
            time, delta, omega = (target[key][:101] for key in ("t", "delta", "omega"))
            powers = {
                "zoh": sample["model_inputs"]["exogenous"]["Pm"][:101],
                "observed": sample["training_context"]["Pm_observed"][:101],
            }
            row = {"fold": fold, "event_id": event_id,
                   "source": metadata.source_dataset, "fault_line": metadata.fault_line,
                   "outcome": metadata.outcome_label,
                   "event_signature_sha256": metadata.event_signature_sha256,
                   "force_balance": {}, "rollouts": {}}
            for power, pm in powers.items():
                interpolation = "zoh" if power == "zoh" else "linear"
                for hidden_name, zero in (("steady", False), ("zero", True)):
                    key = f"{power}_{hidden_name}"
                    row["force_balance"][key] = forced_impulse(
                        model, static, time, delta, omega, pm,
                        interpolation=interpolation, zero_hidden=zero,
                    )
                    rollout_model = zero_model if zero else model
                    pd, pw, _ = rollout_model.rollout_augmented(
                        static, time, {"Pm": pm, "interpolation": interpolation},
                        max_step=0.0025,
                    )
                    row["rollouts"][key] = trajectory_metrics(delta, omega, pd, pw)
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"Completed {len(rows)} event controls", flush=True)
    aggregate = {}
    for fold in ("F1", "F5"):
        local = [row for row in rows if row["fold"] == fold]
        aggregate[fold] = {}
        for key in ("zoh_steady", "zoh_zero", "observed_steady", "observed_zero"):
            aggregate[fold][key] = {
                "n": len(local),
                "rollout": aggregate_trajectory_metrics([row["rollouts"][key] for row in local]),
                "median_force_balance": {
                    metric: float(np.median([row["force_balance"][key][metric] for row in local]))
                    for metric in ("cumulative_speed_residual_rmse", "terminal_force_impulse_l2",
                                   "late_mean_force_residual_l2", "kinematic_integration_error_max")
                },
            }
    if git_provenance() != git:
        raise RuntimeError("Git changed during diagnosis")
    atomic_write_json(output, {
        "git": git, "dataset_build_id": meta.build_id, "model_sha256": hashes,
        "role": "post_selection_teacher_forcing_and_initialization_diagnostic",
        "step_seconds": 0.0025, "per_event": rows, "excluded_short_events": exclusions,
        "aggregate": aggregate,
        "limitations": ["Observed Pm and observed states are diagnostic oracle inputs.",
                        "Zero hidden states are an untuned sensitivity, not estimated physical initial states.",
                        "Residuals assume piecewise-linear observed states and do not isolate unique physical causes."],
    })
    print(json.dumps({f: {key: value["rollout"]["pooled_rmse"] for key, value in d.items()}
                      for f, d in aggregate.items()}, indent=2))


if __name__ == "__main__":
    main()
