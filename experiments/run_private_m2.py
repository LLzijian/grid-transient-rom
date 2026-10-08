"""Frozen private M2 fit and development evaluation; never load test payloads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

from kiki.data.private_dataset import load_private_event, los_index
from kiki.models.private_rom import fit_swing
from kiki.physics import coi_project


def development_data(root):
    splits = json.loads((root / "splits.json").read_text())
    assert not (set(splits["train"]) & set(splits["validation"]))
    assert not ((set(splits["train"]) | set(splits["validation"])) & set(splits["test"]))
    with (root / "dataset_index.csv").open() as stream:
        metadata = {row["event_id"]: row for row in csv.DictReader(stream)}
    sets, excluded = {}, {}
    for split in ("train", "validation"):
        sets[split], excluded[split] = [], []
        for event_id in splits[split]:
            row = metadata[event_id]
            if float(row["trajectory_end_s"]) <= .05 + 1e-9:
                excluded[split].append(event_id)
                continue
            data = load_private_event(root / "dataset.h5", event_id)
            data["history_delta"] = data["delta_sync"][:6].copy()
            data["history_omega"] = data["omega_sync"][:6].copy()
            for key in ("t", "delta", "omega", "Pm", "Pe", "delta_sync", "omega_sync", "Pm_raw"):
                data[key] = data[key][5:806].copy()
            data["t"] -= data["t"][0]
            data["event_id"] = event_id
            sets[split].append(data)
    return sets, metadata, excluded


def training_scales(events):
    return np.sqrt(np.mean([[np.mean(coi_project(e[k], e["mass"]) ** 2)
                            for k in ("delta_sync", "omega_sync")] for e in events], axis=0))


def aggregate(rows):
    if not rows:
        return {"n_events": 0}
    losses = np.array([r["rmse"] for r in rows])
    groups = sorted({r["group"] for r in rows})
    return {"n_events": len(rows), "mean": float(losses.mean()),
            "pooled": float(np.sqrt(sum(r["normalized_sse"] for r in rows) / sum(r["n"] for r in rows))),
            "cvar10": float(np.sort(losses)[-int(np.ceil(.1 * len(rows))):].mean()),
            "worst": float(losses.max()),
            "worst_group_mean": max(float(np.mean([r["rmse"] for r in rows if r["group"] == g])) for g in groups),
            "delta_rmse_rad": float(np.sqrt(sum(r["delta_sse"] for r in rows) / sum(r["n"] for r in rows))),
            "omega_rmse_rad_s": float(np.sqrt(sum(r["omega_sse"] for r in rows) / sum(r["n"] for r in rows)))}


def evaluate(predictions, events, metadata, scales, omega_s, config):
    records, classifications = [], []
    for pred, event in zip(predictions, events, strict=True):
        n = len(event["mass"])
        row = metadata[event["event_id"]]
        err_d = coi_project(pred[:len(event["t"]), :n] - event["delta_sync"], event["mass"])
        err_w = coi_project(pred[:len(event["t"]), n:2*n] - event["omega_sync"], event["mass"])
        for horizon in (.2, .5, 1., 2., 5., 8., "prefix"):
            count = len(event["t"]) - 1 if horizon == "prefix" else round(horizon / .01)
            if count >= len(event["t"]):
                continue
            ds, ws = float(np.sum(err_d[1:count+1]**2)), float(np.sum(err_w[1:count+1]**2))
            size = count*n
            normalized = .5*(ds/scales[0]**2 + ws/scales[1]**2)
            records.append({"event_id": event["event_id"], "group": row["fault_bus"],
                            "label": row["label"], "horizon": str(horizon), "n": size,
                            "delta_sse": ds, "omega_sse": ws, "normalized_sse": normalized,
                            "rmse": float(np.sqrt(normalized / size))})
        hit = los_index(pred[:, :n]+event["equilibrium_delta"], pred[:, n:2*n]/omega_s+1,
                        event["mass"], config)
        classifications.append({"event_id": event["event_id"], "truth": row["label"],
                                "predicted_los": hit is not None,
                                "predicted_los_time_s": None if hit is None else hit*.01,
                                "max_absolute_speed_rad_s": float(np.max(np.abs(pred[:, n:2*n])))})
    summary = {h: aggregate([r for r in records if r["horizon"] == h])
               for h in ("0.2", "0.5", "1.0", "2.0", "5.0", "8.0", "prefix")}
    unstable = [r for r in classifications if r["truth"] == "unstable"]
    stable = [r for r in classifications if r["truth"] == "stable"]
    summary["classification"] = {"n": len(classifications), "n_unstable": len(unstable),
        "false_stable": sum(not r["predicted_los"] for r in unstable),
        "false_unstable": sum(r["predicted_los"] for r in stable),
        "balanced_accuracy": .5*(np.mean([r["predicted_los"] for r in unstable]) +
                                  np.mean([not r["predicted_los"] for r in stable]))}
    return summary, records, classifications


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("private_data/ieee39_postfault"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    sets, metadata, excluded = development_data(args.data)
    model, fit = fit_swing(sets["train"])
    scales = training_scales(sets["train"])
    events = sets["validation"]
    initial = np.array([np.r_[e["delta_sync"][0], e["omega_sync"][0]] for e in events])
    pm = np.array([e["Pm_raw"][0] for e in events])
    predictions = model.rollout(initial, pm, np.arange(801)*.01)
    with h5py.File(args.data / "dataset.h5", "r") as file:
        config = json.loads(file.attrs["config"])
        omega_s = float(file.attrs["omega_s"])
        signature = file.attrs["postfault_signature"]
    summary, records, classifications = evaluate(predictions, events, metadata, scales, omega_s, config)
    model.save(args.output / "model.npz")
    np.savez_compressed(args.output / "rollouts.npz", predictions=predictions,
                        event_ids=np.array([e["event_id"] for e in events]))
    result = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "postfault_signature": signature,
              "index_sha256": hashlib.sha256((args.data / "dataset_index.csv").read_bytes()).hexdigest(),
              "split_sha256": hashlib.sha256((args.data / "splits.json").read_bytes()).hexdigest(),
              "numpy": np.__version__, "seconds": time.perf_counter()-start,
              "fit": fit, "training_scales": scales.tolist(), "excluded_predecision": excluded,
              "train_events": len(sets["train"]), "numerical_failures": 0,
              "summary": summary, "records": records, "classification_records": classifications}
    (args.output / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("records", "classification_records")}, indent=2))


if __name__ == "__main__":
    main()
