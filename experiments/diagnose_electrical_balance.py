"""Offline Pe substitution audit; never a deployable forecast or fitted model.

Claim: replacing learned electrical force by measured Pe will distinguish
electrical-closure error from the source swing remainder (damping, losses,
sampling, and any inconsistent inertia/units). No claim of exact Pe=te.
Frozen population: F1/F5 validation, first min(101, T) points, with partial
unstable paths reported separately. No test payloads, tuning or seed search.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.run_refit_v2 import atomic_write_json, git_provenance, sha256_file
from kiki.data import CanonicalStore, load_split
from kiki.physics import coi_project, coi_project_power
from kiki.physics.units import effective_mass


def electrical_balance(time, omega, pm, pe, mass):
    """Cumulative trapezoidal impulse; remainder intentionally retains damping/losses."""
    time, omega, pm, pe, mass = map(np.asarray, (time, omega, pm, pe, mass))
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("at least two increasing sample times required")
    if omega.shape != (len(time), len(mass)) or pm.shape != omega.shape or pe.shape != omega.shape:
        raise ValueError("time/generator alignment required")
    if any(not np.isfinite(a).all() for a in (time, omega, pm, pe, mass)) or np.any(mass <= 0):
        raise ValueError("finite arrays and positive masses required")
    acceleration = coi_project_power(pm - pe, mass) / mass
    impulse = np.vstack((np.zeros(len(mass)), np.cumsum(
        np.diff(time)[:, None] * (acceleration[1:] + acceleration[:-1]) / 2, axis=0,
    )))
    residual = coi_project(omega - omega[[0]], mass) - impulse
    return {
        "speed_residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "terminal_force_impulse": (residual[-1] * mass).tolist(),
        "terminal_force_impulse_l2": float(np.linalg.norm(residual[-1] * mass)),
        "speed_change_rmse": float(np.sqrt(np.mean((omega - omega[[0]])**2))),
    }


def main():
    output = ROOT / "results/e15_electrical_balance/summary.json"
    if output.parent.exists():
        raise RuntimeError(f"refusing to overwrite {output.parent}")
    git = git_provenance()
    config = json.loads((ROOT / "configs/dataset.json").read_text())
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    mass = effective_mass(meta.M_prior, meta.omega_s)
    prior_path = ROOT / "results/e14_force_balance/summary.json"
    prior = json.loads(prior_path.read_text())
    comparisons = {(r["fold"], r["event_id"]): r for r in prior["per_event"]}
    rows = []
    for fold in ("F1", "F5"):
        split = load_split(config["dataset_dir"], view=config["view"],
                           split_name=config["split_name"], fold=fold)
        for event_id in split.val_ids:
            event = store.load_event(event_id, include_offline=True)
            n = min(101, len(event.t))
            pe = event.offline["Pe"][:n]
            result = electrical_balance(event.t[:n], event.omega[:n], event.Pm[:n], pe, mass)
            row = {"fold": fold, "event_id": event_id,
                   "source": event.metadata.source_dataset, "outcome": event.metadata.outcome_label,
                   "fault_line": event.metadata.fault_line, "n_points": n,
                   "horizon_seconds": float(event.t[n - 1]), "complete_H100": n == 101,
                   "event_signature_sha256": event.metadata.event_signature_sha256,
                   "Pe_slice_float64_sha256": hashlib.sha256(np.ascontiguousarray(pe, dtype="<f8").tobytes()).hexdigest(),
                   "power_balance_max": float(max(np.max(np.abs(pe.sum(axis=1))),
                                                   np.max(np.abs(event.Pm[:n].sum(axis=1))))),
                   "electrical_oracle": result}
            if n == 101:
                row["frozen_model_observed_Pm"] = {
                    name: comparisons[fold, event_id]["force_balance"][f"observed_{name}"]["cumulative_speed_residual_rmse"]
                    for name in ("steady", "zero")
                }
            rows.append(row)
    aggregate = {}
    for fold in ("F1", "F5"):
        for complete in (True, False):
            local = [r for r in rows if r["fold"] == fold and r["complete_H100"] == complete]
            aggregate[f"{fold}_{'H100' if complete else 'partial'}"] = {
                "n": len(local), "n_stable": sum(r["outcome"] == "stable" for r in local),
                "median_speed_residual_rmse": float(np.median([r["electrical_oracle"]["speed_residual_rmse"] for r in local])),
                "max_speed_residual_rmse": max(r["electrical_oracle"]["speed_residual_rmse"] for r in local),
            }
    if git_provenance() != git:
        raise RuntimeError("Git changed during audit")
    atomic_write_json(output, {"git": git, "dataset_build_id": meta.build_id,
                              "python": platform.python_version(), "numpy": np.__version__,
                              "schema_sha256": sha256_file(store.dataset_dir / "schema.json"),
                              "e14_sha256": sha256_file(prior_path), "per_event": rows,
                              "aggregate": aggregate,
                              "role": "offline_electrical_substitution_not_forecast",
                              "limitations": ["Pe is terminal power, not necessarily electromagnetic torque te.",
                                              "Remainder includes damping, losses, quadrature and source inconsistency.",
                                              "Partial paths do not estimate fixed-horizon unstable performance."]})
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
