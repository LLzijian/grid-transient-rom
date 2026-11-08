"""E7: non-energy baselines (SINDYc / PINN / LSTM) vs the modular energy model.

On the F1 development fold only (train=209 / val=43, test is NEVER opened),
fit each baseline from the training context (derivatives + observed Pm as
teacher) and roll it out on the full val set at H100 under the causal ZOH
deployment contract -- the exact protocol used for the modular model.  The
reference rows (M0 backbone from results/e02_backbone and the pruned modular
energy model from results/e02_prune) are re-rolled on the SAME val population
with the SAME code path so the table is internally consistent.

Results are difficulty-stratified exactly like E6/E6c (truth peak |omega| in
the first post-clear second): mild / moderate / severe.  Trajectory accuracy is
only claimed in the band where the modular model itself is accurate; severe and
unstable events are where the energy model hands over to the integration-free
margin rule, which none of these baselines can offer (certificate_claim=False).

Usage:
  python run_e7.py --methods all [--tag main] [--smoke]
  python run_e7.py --methods sindy,pinn,lstm,ref
Every output directory refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kiki.data import CanonicalStore, ModelDataRequest, adapt_event, load_split
from kiki.evaluation.metrics import aggregate_trajectory_metrics, trajectory_metrics
from kiki.models.baselines import AutoregressiveLSTMModel, PINNModel, SINDYcModel
from kiki.models.base import RunContext, sample_rollout_inputs
from kiki.models.m0 import M0Model
from kiki.models.prunable_energy_model import PrunableEnergyModel

HORIZON_STEPS = 100
MILD, MODERATE, SEVERE = 0.5, 1.5, float("inf")  # truth peak |omega| over 1 s
BANDS = [("mild", 0.0, MILD), ("moderate", MILD, MODERATE), ("severe", MODERATE, SEVERE)]

REF_MODELS = {
    "M0_backbone": ROOT / "results" / "e02_backbone" / "model_base.npz",
    "modular_pruned": ROOT / "results" / "e02_prune" / "model_pruned.npz",
}


def peak_omega_1s(sample) -> float:
    """Truth peak |omega| over the first post-clear second (the E6 difficulty key)."""
    target = sample["training_context"]["targets"]
    t = np.asarray(target["t"], dtype=np.float64)
    mask = t <= (t[0] + 1.0)
    return float(np.max(np.abs(np.asarray(target["omega"], dtype=np.float64)[mask])))


def difficulty_band(peak: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= peak < hi:
            return name
    return "severe"


def rollout_metrics(model, samples, horizon: int = HORIZON_STEPS):
    """Mirror of run_e6.rollout_metrics; single event rows carry difficulty."""
    rows, failures = [], 0
    for sample in samples:
        initial, t, exo = sample_rollout_inputs(sample)
        h = min(horizon + 1, len(t))
        try:
            pd, pw = model.rollout(initial, t[:h], {"Pm": exo["Pm"][:h],
                                                    "interpolation": exo["interpolation"]})
        except Exception:
            failures += 1
            continue
        targets = sample["training_context"]["targets"]
        row = trajectory_metrics(targets["delta"][:h], targets["omega"][:h], pd, pw)
        peak = peak_omega_1s(sample)
        row.update({
            "event_id": sample["event_id"],
            "true_stable": sample["training_context"]["label"] == "stable",
            "source": sample["training_context"]["metadata"]["source_dataset"],
            "peak_omega_1s": peak,
            "difficulty": difficulty_band(peak),
        })
        rows.append(row)
    return rows, failures


def band_table(rows):
    """Per (stability x difficulty) simple mean-RMSE summary for paper tables."""
    cells = {}
    for row in rows:
        key = ("stable" if row["true_stable"] else "unstable", row["difficulty"])
        cell = cells.setdefault(key, {"n": 0, "sd": 0.0, "sw": 0.0, "peaks": []})
        cell["n"] += 1
        cell["sd"] += row["delta_rmse"]
        cell["sw"] += row["omega_rmse"]
        cell["peaks"].append(row["peak_omega_1s"])
    table = {}
    for key, cell in sorted(cells.items()):
        n = cell["n"]
        mean_d = cell["sd"] / n
        mean_w = cell["sw"] / n
        table[f"{key[0]}_{key[1]}"] = {
            "n": n,
            "mean_delta_rmse": float(mean_d),
            "mean_omega_rmse": float(mean_w),
            "mean_rmse": float(np.sqrt(0.5 * (mean_d ** 2 + mean_w ** 2))),
            "median_peak_omega_1s": float(np.median(cell["peaks"])),
        }
    return table


def make_context(n_train: int, epochs: int) -> RunContext:
    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    return RunContext(
        seed=int(config["seed"]), epochs=epochs, horizon_steps=HORIZON_STEPS,
        max_cases=n_train, learning_rate=1e-3, integral_windows=(1, 5, 20),
    )


def write_report(dirpath: Path, method: str, report: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    out = dirpath / "summary.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


def run_reference(train_samples, val_samples, tag_dir: Path) -> None:
    """Re-roll the stored F1 models on the same val for an internally-consistent table."""
    for name, path in REF_MODELS.items():
        model = PrunableEnergyModel.load(path)
        rows, failures = rollout_metrics(model, val_samples)
        agg = aggregate_trajectory_metrics(rows)
        agg["n_rollout_failures"] = failures
        report = {
            "stage": f"E7-ref-{name}", "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": name, "model_path": str(path), "n_val": len(val_samples),
            "certificate_claim": model.certificate_claim if hasattr(model, "certificate_claim") else None,
            "aggregate": {k: v for k, v in agg.items()},
            "difficulty_stratified": band_table(rows),
        }
        write_report(tag_dir / name, name, report)
        print(f"[E7] reference {name} done.")


def run_sindy(train_samples, val_samples, ctx, tag_dir: Path) -> None:
    t0 = time.time()
    model = SINDYcModel()
    fr = model.fit(train_samples, val_samples, ctx)
    rows, failures = rollout_metrics(model, val_samples)
    agg = aggregate_trajectory_metrics(rows)
    agg["n_rollout_failures"] = failures
    report = {
        "stage": "E7-sindy", "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "SINDYc", "n_train": len(train_samples[:ctx.max_cases]), "n_val": len(val_samples),
        "fit_status": fr.status, "best_validation_loss": fr.best_validation_loss,
        "n_coefficients": int(np.count_nonzero(model.coefficients)),
        "coefficients_shape": list(model.coefficients.shape),
        "fit_seconds": round(time.time() - t0, 1),
        "physics_audit": model.physics_audit(),
        "aggregate": {k: v for k, v in agg.items()},
        "difficulty_stratified": band_table(rows),
    }
    write_report(tag_dir / "sindy", "SINDYc", report)
    model.save(tag_dir / "sindy" / "model.npz")
    print(f"[E7] SINDYc done in {report['fit_seconds']} s.")


def run_pinn(train_samples, val_samples, ctx, tag_dir: Path) -> None:
    t0 = time.time()
    model = PINNModel(hidden=24)
    fr = model.fit(train_samples, val_samples, ctx)
    rows, failures = rollout_metrics(model, val_samples)
    agg = aggregate_trajectory_metrics(rows)
    agg["n_rollout_failures"] = failures
    report = {
        "stage": "E7-pinn", "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "PINN", "n_train": len(train_samples[:ctx.max_cases]), "n_val": len(val_samples),
        "fit_status": fr.status, "best_validation_loss": fr.best_validation_loss,
        "epochs": ctx.epochs, "hidden": model.hidden, "learning_rate": ctx.learning_rate,
        "fit_seconds": round(time.time() - t0, 1),
        "base_method": model.base.method_name, "base_damping_norm": float(np.linalg.norm(model.base.damping)),
        "physics_audit": model.physics_audit(),
        "aggregate": {k: v for k, v in agg.items()},
        "difficulty_stratified": band_table(rows),
    }
    write_report(tag_dir / "pinn", "PINN", report)
    model.save(tag_dir / "pinn" / "model.pt")
    print(f"[E7] PINN done in {report['fit_seconds']} s.")


def run_lstm(train_samples, val_samples, ctx, tag_dir: Path) -> None:
    t0 = time.time()
    model = AutoregressiveLSTMModel(hidden=32)
    fr = model.fit(train_samples, val_samples, ctx)
    rows, failures = rollout_metrics(model, val_samples)
    agg = aggregate_trajectory_metrics(rows)
    agg["n_rollout_failures"] = failures
    report = {
        "stage": "E7-lstm", "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": "LSTM", "n_train": len(train_samples[:ctx.max_cases]), "n_val": len(val_samples),
        "fit_status": fr.status, "best_validation_loss": fr.best_validation_loss,
        "epochs": ctx.epochs, "hidden": model.hidden, "learning_rate": ctx.learning_rate,
        "fit_seconds": round(time.time() - t0, 1),
        "physics_audit": model.physics_audit(),
        "aggregate": {k: v for k, v in agg.items()},
        "difficulty_stratified": band_table(rows),
    }
    write_report(tag_dir / "lstm", "LSTM", report)
    model.save(tag_dir / "lstm" / "model.pt")
    print(f"[E7] LSTM done in {report['fit_seconds']} s.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="all", help="comma list of sindy,pinn,lstm,ref or all")
    ap.add_argument("--tag", default="main", help="result subdir under results/e07")
    ap.add_argument("--smoke", action="store_true", help="tiny subset + tiny epochs to time & debug")
    ap.add_argument("--pinn-epochs", type=int, default=100)
    ap.add_argument("--lstm-epochs", type=int, default=80)
    args = ap.parse_args()

    config = json.loads((ROOT / "configs" / "dataset.json").read_text(encoding="utf-8"))
    store = CanonicalStore.open(config["dataset_dir"])
    meta = store.dataset_metadata
    split = load_split(config["dataset_dir"], view=config["view"],
                       split_name=config["split_name"], fold="F1")
    request = ModelDataRequest.initial_state_rollout(config["pm_mode"])

    def adapt(ids):
        return [adapt_event(store.load_event(eid), inertia_prior=meta.M_prior,
                            request=request, representation=config["representation"])
                for eid in ids]

    train, val = adapt(split.train_ids), adapt(split.val_ids)
    if args.smoke:
        train, val = train[:6], val[:12]
    tag_dir = ROOT / "results" / "e07" / args.tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    methods = ["sindy", "pinn", "lstm", "ref"] if args.methods == "all" \
        else [m.strip() for m in args.methods.split(",")]

    if "ref" in methods:
        run_reference(train, val, tag_dir)
    if "sindy" in methods:
        run_sindy(train, val, make_context(len(train), epochs=1), tag_dir)
    if "pinn" in methods:
        ep = 2 if args.smoke else args.pinn_epochs
        run_pinn(train, val, make_context(len(train), epochs=ep), tag_dir)
    if "lstm" in methods:
        ep = 2 if args.smoke else args.lstm_epochs
        run_lstm(train, val, make_context(len(train), epochs=ep), tag_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
