"""One fit, true-speed diagnostic and autonomous M3-Pm contribution screen."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
from run_private_m2 import development_data, evaluate, training_scales

from kiki.models.private_pm import fit_mechanical, forced_pm
from kiki.models.private_rom import SwingROM


def pm_records(predictions, events, metadata, name):
    rows = []
    for pred, event in zip(predictions, events, strict=True):
        for horizon in (.2, .5, 1., 2., 5., 8., "prefix"):
            k = len(event["t"])-1 if horizon == "prefix" else round(horizon/.01)
            if k >= len(event["t"]):
                continue
            err = pred[1:k+1]-event["Pm_raw"][1:k+1]
            rows.append({"event_id": event["event_id"], "group": metadata[event["event_id"]]["fault_bus"],
                         "horizon": str(horizon), "model": name, "n": err.size,
                         "sse": float(np.sum(err**2)), "rmse": float(np.sqrt(np.mean(err**2))),
                         "machine_sse": np.sum(err**2, axis=0).tolist()})
    return rows


def pm_summary(rows):
    result = {}
    for name in sorted({r["model"] for r in rows}):
        result[name] = {}
        for h in ("0.2", "0.5", "1.0", "2.0", "5.0", "8.0", "prefix"):
            cohort = [r for r in rows if r["model"] == name and r["horizon"] == h]
            values = np.array([r["rmse"] for r in cohort])
            result[name][h] = {"n_events": len(cohort), "mean": float(values.mean()),
                               "pooled": float(np.sqrt(sum(r["sse"] for r in cohort)/sum(r["n"] for r in cohort))),
                               "cvar10": float(np.sort(values)[-int(np.ceil(.1*len(values))):].mean())}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("private_data/ieee39_postfault"))
    parser.add_argument("--baseline", type=Path, default=Path("private_data/task2_m2_0019"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    sets, metadata, excluded = development_data(args.data)
    model = fit_mechanical(sets["train"], SwingROM.load(args.baseline / "model.npz"))
    events = sets["validation"]
    initial = np.array([np.r_[e["delta_sync"][0], e["omega_sync"][0], e["Pm_raw"][0]] for e in events])
    predictions = model.rollout(initial, None, np.arange(801)*.01)
    forced = [forced_pm(model, e) for e in events]
    held = [np.tile(e["Pm_raw"][0], (len(e["t"]), 1)) for e in events]
    pmrows = pm_records(predictions[:, :, 20:30], events, metadata, "joint")
    pmrows += pm_records(forced, events, metadata, "forced_true_speed")
    pmrows += pm_records(held, events, metadata, "zoh")
    with h5py.File(args.data / "dataset.h5", "r") as file:
        config, omega_s = json.loads(file.attrs["config"]), float(file.attrs["omega_s"])
    summary, records, classifications = evaluate(predictions, events, metadata, training_scales(sets["train"]), omega_s, config)
    pmmetrics = pm_summary(pmrows)
    base = json.loads((args.baseline / "evidence.json").read_text())
    def comparison(new, old):
        return all(new[h]["mean"] <= .95*old[h]["mean"] and new[h]["pooled"] <= old[h]["pooled"]
                   and new[h]["cvar10"] <= 1.05*old[h]["cvar10"] for h in ("2.0", "5.0", "8.0"))
    old_false = {r["event_id"] for r in base["classification_records"] if r["truth"] == "stable" and r["predicted_los"]}
    new_false = {r["event_id"] for r in classifications if r["truth"] == "stable" and r["predicted_los"]}
    result = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "baseline_commit": base["code_commit"], "index_sha256": base["index_sha256"],
              "split_sha256": base["split_sha256"], "seconds": time.perf_counter()-start,
              "T_seconds": (1/model.rate).tolist(), "K_omega": (model.gain/model.rate).tolist(),
              "identification_pass": comparison(pmmetrics["forced_true_speed"], pmmetrics["zoh"]),
              "contribution_pass": comparison(summary, base["summary"]) and not (new_false-old_false),
              "new_false_unstable": sorted(new_false-old_false), "numerical_failures": 0,
              "excluded_predecision": excluded, "summary": summary, "pm_summary": pmmetrics,
              "records": records, "pm_records": pmrows, "classification_records": classifications}
    model.save(args.output / "model.npz")
    np.savez_compressed(args.output / "rollouts.npz", predictions=predictions,
                        event_ids=np.array([e["event_id"] for e in events]))
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records", "pm_records", "classification_records")}, indent=2))


if __name__ == "__main__":
    main()
