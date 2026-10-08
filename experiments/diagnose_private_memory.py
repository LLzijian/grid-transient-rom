"""Post0026 numerical oracles: explicitly noncausal diagnostics, never a model."""
import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from run_private_m2 import development_data, evaluate, training_scales
from scipy.optimize import minimize_scalar

from kiki.models.private_pm import MechanicalROM
from kiki.physics import coi_project


@dataclass
class OracleStrength(MechanicalROM):
    q_rate: float
    uncentered: bool = False

    def field(self, state, pm_unused=None):
        physical = super().field(state[..., :30])
        sine = np.sin((state[..., :10]+self.delta_star)@self.incidence)-np.sin(self.delta_star@self.incidence)
        electrical = (sine*self.coupling)@self.incidence.T
        if self.uncentered:
            electrical = electrical+self.pm_star
        q = state[..., 30:]
        physical[..., 10:20] -= np.expm1(q)*electrical/self.mass
        return np.concatenate((physical, -self.q_rate*q), axis=-1)


def numerical_initial_oracle(model, events, scales):
    initial = np.array([np.r_[e["delta_sync"][0], e["omega_sync"][0], e["Pm_raw"][0], 0.] for e in events])

    def cost(q):
        initial[:, -1] = q
        predictions = model.rollout(initial, None, np.arange(801)*.01)
        losses = []
        for pred, e in zip(predictions, events, strict=True):
            k = len(e["t"])
            d = coi_project(pred[1:k, :10]-e["delta_sync"][1:], e["mass"])/scales[0]
            w = coi_project(pred[1:k, 10:20]-e["omega_sync"][1:], e["mass"])/scales[1]
            losses.append(.5*np.mean(d*d+w*w))
        return np.array(losses), predictions

    grid = np.unique(np.r_[np.linspace(-3., np.log(2.), 9), 0.])
    losses = []
    for q in grid:
        loss, _ = cost(q)
        losses.append(loss)
        print(json.dumps({"oracle_grid_q": float(q), "mean_mse": float(loss.mean())}), flush=True)
    losses = np.array(losses)
    zero_loss = losses[np.flatnonzero(grid == 0)[0]]
    index = np.argmin(losses, axis=0)
    best_q, best_loss = grid[index].copy(), np.min(losses, axis=0)
    left, right = grid[np.maximum(index-1, 0)], grid[np.minimum(index+1, len(grid)-1)]
    ratio = (np.sqrt(5)-1)/2
    x, y = right-ratio*(right-left), left+ratio*(right-left)
    fx, _ = cost(x)
    fy, _ = cost(y)
    for iteration in range(24):
        choose_left = fx < fy
        left, right = np.where(choose_left, left, x), np.where(choose_left, y, right)
        x, y = right-ratio*(right-left), left+ratio*(right-left)
        new_loss, _ = cost(np.where(choose_left, x, y))
        fx, fy = np.where(choose_left, new_loss, fy), np.where(choose_left, fx, new_loss)
        if iteration % 8 == 7:
            print(json.dumps({"golden_iteration": iteration+1, "mean_best_mse": float(np.minimum(best_loss, np.minimum(fx, fy)).mean())}), flush=True)
    for q, loss in ((x, fx), (y, fy)):
        better = loss < best_loss
        best_q, best_loss = np.where(better, q, best_q), np.minimum(best_loss, loss)
    final_loss, predictions = cost(best_q)
    if np.max(final_loss-zero_loss) > 1e-9:
        raise AssertionError("oracle must not lose to included zero competitor")
    return best_q, final_loss, zero_loss, predictions


def inversion_arrays(event, model):
    speed = np.r_[event["history_omega"][:-1], event["omega_sync"]]
    acceleration = np.gradient(speed, .01, axis=0, edge_order=2)[5:-1]
    electrical = (np.sin((event["delta_sync"][:-1]+model.delta_star)@model.incidence)-
                  np.sin(model.delta_star@model.incidence))*model.coupling@model.incidence.T
    required = event["Pm_raw"][:-1]-model.pm_star-model.damping*event["omega_sync"][:-1]-model.mass*acceleration
    return electrical, required-electrical


def time_oracle(events, model, floor, metadata):
    records, arrays = [], {}
    for event in events:
        electrical, residual = inversion_arrays(event, model)
        record = {"event_id": event["event_id"], "group": metadata[event["event_id"]]["fault_bus"],
                  "label": metadata[event["event_id"]]["label"], "n": len(electrical)}
        for name, direction in (("centered", electrical), ("uncentered", electrical+model.pm_star)):
            norm = np.sum(direction**2, axis=1)
            active = norm > floor
            modifier = np.full(len(norm), np.nan)
            modifier[active] = np.sum(direction[active]*residual[active], axis=1)/norm[active]
            valid = active & (modifier > -1)
            q = np.full(len(norm), np.nan)
            q[valid] = np.log1p(modifier[valid])
            arrays[event["event_id"]+"_"+name] = q
            total = float(np.sum(residual[active]**2))
            remaining = float(np.sum((residual[active]-modifier[active, None]*direction[active])**2))
            item = {"active_n": int(sum(active)), "valid_log_n": int(sum(valid)),
                    "nonpositive_strength_n": int(sum(active & ~valid)),
                    "residual_sse": total, "unconstrained_projection_residual_sse": remaining,
                    "explained_fraction": None if total == 0 else 1-remaining/total,
                    "q_quantiles": np.quantile(q[valid], [0, .1, .5, .9, 1]).tolist() if np.any(valid) else [],
                    "both_signs": bool(np.any(q[valid] < -.05) and np.any(q[valid] > .05))}
            if sum(valid) >= 10 and np.ptp(event["t"][:-1][valid]) >= .1:
                t, target = event["t"][:-1][valid], q[valid]
                def decay_error(rate, t=t, target=target):
                    shape = np.exp(-rate*t)
                    amplitude = np.dot(shape, target)/np.dot(shape, shape)
                    return float(np.sum((target-amplitude*shape)**2))
                fit = minimize_scalar(decay_error, bounds=(.05, 20.), method="bounded", options={"xatol": 1e-5})
                normq = max(float(np.dot(target, target)), 1e-20)
                item.update({"fixed_decay_relative_sse": decay_error(model.q_rate)/normq,
                             "best_decay_relative_sse": fit.fun/normq, "best_decay_rate": fit.x,
                             "q_total_variation": float(np.sum(np.abs(np.diff(target))))})
            record[name] = item
        records.append(record)
    return records, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("private_data/ieee39_postfault"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uncentered", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sets, metadata, excluded = development_data(args.data)
    baseline = MechanicalROM.load("private_data/task2_m3_0020/model.npz")
    rejected = json.loads(Path("reports/failed/0026_m4_evidence.json").read_text())
    model = OracleStrength(**vars(baseline), q_rate=rejected["q_rate"], uncentered=args.uncentered)
    events = sets["train"]+sets["validation"]
    scales = training_scales(sets["train"])
    best_q, loss, zero_loss, predictions = numerical_initial_oracle(model, events, scales)
    # Identity: a zero memory produces the previously accepted autonomous model.
    reference = np.load("private_data/task2_m3_0020/rollouts.npz")["predictions"]
    zero_initial = np.c_[reference[:, 0], np.zeros(len(reference))]
    zero_pred = model.rollout(zero_initial, None, np.arange(801)*.01)
    zero_error = float(np.max(np.abs(zero_pred[:, :, :30]-reference)))
    if zero_error > 1e-10:
        raise AssertionError("q=0 baseline mismatch")
    with h5py.File(args.data / "dataset.h5", "r") as file:
        config, omega_s = json.loads(file.attrs["config"]), float(file.attrs["omega_s"])
    metrics, records, classes = {}, {}, {}
    offset = 0
    for name, subset in sets.items():
        count = len(subset)
        metrics[name], records[name], classes[name] = evaluate(predictions[offset:offset+count], subset, metadata, scales, omega_s, config)
        offset += count
    norms = np.concatenate([np.sum(inversion_arrays(e, model)[0]**2, axis=1) for e in sets["train"]])
    floor = max(1e-8, 1e-4*float(np.median(norms)))
    inversion, arrays = time_oracle(events, model, floor, metadata)
    base = json.loads(Path("private_data/task2_m3_0020/evidence.json").read_text())
    clear = all(metrics["validation"][h]["mean"] <= .95*base["summary"][h]["mean"] and
                metrics["validation"][h]["pooled"] <= base["summary"][h]["pooled"] and
                metrics["validation"][h]["cvar10"] <= 1.05*base["summary"][h]["cvar10"] for h in ("2.0", "5.0", "8.0"))
    result = {"purpose": "noncausal diagnostic, not deployable or holdout evaluation",
              "coupling": "uncentered" if args.uncentered else "centered",
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "fixed_q_rate": model.q_rate, "zero_baseline_max_abs_error": zero_error,
              "oracle_clear_potential": clear, "seconds": time.perf_counter()-started,
              "index_sha256": base["index_sha256"], "split_sha256": base["split_sha256"],
              "excluded_predecision": excluded, "inversion_force_norm2_floor": floor,
              "summary": metrics, "records": records, "classification_records": classes,
              "oracle_initial": [{"event_id": e["event_id"], "q0": float(q), "prefix_mse": float(v), "zero_mse": float(z)}
                                 for e, q, v, z in zip(events, best_q, loss, zero_loss, strict=True)],
              "inversion": inversion}
    np.savez_compressed(args.output / "oracle_rollouts.npz", predictions=predictions,
                        event_ids=np.array([e["event_id"] for e in events]), q0=best_q)
    np.savez_compressed(args.output / "time_oracles.npz", **arrays)
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records", "classification_records", "oracle_initial", "inversion")}, indent=2))


if __name__ == "__main__":
    main()
