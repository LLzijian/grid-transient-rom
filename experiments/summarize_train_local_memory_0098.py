"""Final full-TRAIN E1 capability report; never a checkpoint selector.

Fixed .005/.01 integrations use actual TRAIN prefixes through at most eight
seconds after the causal100ms initial window. No refinement or TEST access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
from evaluate_structured_rom import _rollouts, _metric_safe, _scored_numerics, _safe
from evaluate_l1_latent import common_frequency_metrics
from evaluate_long_latent import aggregate
from train_structured_rom import _point_error

HORIZONS = ("1", "2", "5", "8", "prefix")


def scalar_summary(rows):
    if not rows:
        return None
    errors = np.asarray([r["rmse"] for r in rows])
    return {"n_events": len(rows), "mean": float(errors.mean()),
            "pooled": float(np.sqrt(sum(r["sse"] for r in rows) / sum(r["n"] for r in rows))),
            "cvar10": float(np.sort(errors)[-max(1, int(np.ceil(.1 * len(errors)))):].mean()),
            "worst": float(errors.max())}


def objective_summary(rows):
    if not rows:
        return None
    values = np.asarray([r["mean_point_error"] for r in rows])
    return {"n_events": len(rows), "mean": float(values.mean()),
            "cvar10": float(np.sort(values)[-max(1, int(np.ceil(.1 * len(values)))):].mean()),
            "worst": float(values.max())}


def summarize_rows(rows, aggregate_fn=aggregate):
    """No failing eligible row may disappear from the official denominator."""
    eligible = [r for r in rows if r["available"]]
    finite = [r for r in eligible if r["finite"]]
    admissible = [r for r in eligible if r["finite"] and r["numerically_admissible"]]
    official = aggregate_fn(eligible) if eligible and len(admissible) == len(eligible) else None
    return {"all_rows": len(rows), "eligible_events": len(eligible),
            "finite_events": len(finite), "numerically_admissible_events": len(admissible),
            "unavailable_events": len(rows) - len(eligible),
            "unresolved_event_ids": [r["event_id"] for r in eligible if r not in admissible],
            "official": official,
            "finite_subcohort_diagnostic": aggregate_fn(finite) if finite else None,
            "diagnostic_scope": "finite-only descriptive subset; does not replace unresolved official score"}


def event_metrics(model, event, fine, coarse):
    """Pure per-event calculation; keeps latent substep failures in QA."""
    mass = np.asarray(model.mass.detach().cpu() if torch.is_tensor(model.mass) else model.mass)
    if fine.shape[0] != 1601 or coarse.shape[0] != 801 or fine.shape[1] != coarse.shape[1]:
        raise ValueError("expected full-state fixed eight-second .005/.01 predictions")
    pscale = np.asarray(model.p_scale.detach().cpu() if torch.is_tensor(model.p_scale) else model.p_scale)
    if pscale.shape != (10,) or not np.isfinite(pscale).all() or np.any(pscale <= 0):
        raise ValueError("checkpoint must supply ten finite positive p normalization scales")
    numerical = _scored_numerics(fine, coarse, event, mass)
    numerics = {r["horizon"]: r for r in numerical}
    common = {r["horizon"]: r for r in common_frequency_metrics(fine[::2], event["x"][10:], mass)}
    physical, frequency, mechanical, objective = [], [], [], []
    for horizon in HORIZONS:
        qa = numerics[horizon]
        ok = bool(qa["available"] and qa["finite_to_endpoint"] and
                  qa["coi_normalized_rmse"] <= .001 and qa["common_normalized_rmse"] <= .001)
        row = _metric_safe(fine[::2], event, horizon, mass)
        if row["available"] and row.get("finite"):
            row["finite"] = all(np.isfinite(row[key]) for key in ("rmse", "normalized_sse", "delta_sse", "omega_sse"))
            if not row["finite"]:
                row.update(rmse=None, normalized_sse=None, delta_sse=None, omega_sse=None)
        row["numerically_admissible"] = ok
        physical.append(row)
        shared = {"event_id": event["id"], "group": event["group"], "label": event["label"],
                  "horizon": horizon, "available": row["available"], "finite": bool(row.get("finite")),
                  "numerically_admissible": ok}
        cr = dict(common[horizon], **shared)
        if cr['available'] and cr['finite']:
            if any(cr.get(key) is None or not np.isfinite(cr[key]) for key in ('sse','rmse')):
                cr.update(finite=False,sse=None,rmse=None)
        pr, ar = dict(shared), dict(shared)
        if row["available"] and row["finite"]:
            steps = min(800, len(event["x"]) - 11) if horizon == "prefix" else int(horizon) * 100
            pred, truth = fine[::2][1:steps+1, :30], event["x"][11:11+steps, :30]
            error = (pred[:, 20:30] - truth[:, 20:30]) / pscale
            pr.update(n=int(error.size), sse=float(np.sum(error**2)), rmse=float(np.sqrt(np.mean(error**2))))
            if not np.isfinite(pr["sse"]) or not np.isfinite(pr["rmse"]):
                pr.update(finite=False, sse=None, rmse=None)
            with torch.no_grad():
                point = _point_error(torch.as_tensor(pred, dtype=torch.float64),
                                     torch.as_tensor(truth, dtype=torch.float64), model)
            ar["mean_point_error"] = float(point.mean())
            if not np.isfinite(ar["mean_point_error"]):
                ar.update(finite=False, mean_point_error=None)
        else:
            pr.update(n=0, sse=None, rmse=None)
            ar["mean_point_error"] = None
            if row["available"]:
                cr.update(sse=None, rmse=None)
        frequency.append(cr)
        mechanical.append(pr)
        objective.append(ar)
    return {"records": physical, "common_frequency": frequency, "normalized_p": mechanical,
            "objective_point_error": objective, "numerical": numerical}


def build_summary(records):
    families = {"records": aggregate, "common_frequency": scalar_summary,
                "normalized_p": scalar_summary, "objective_point_error": objective_summary}
    result = {}
    for key, function in families.items():
        rows = records[key]
        result[key] = {h: summarize_rows([r for r in rows if r["horizon"] == h], function) for h in HORIZONS}
        result[key]["unstable_prefix"] = summarize_rows(
            [r for r in rows if r["horizon"] == "prefix" and r["label"] == "unstable"], function)
        result[key]["groups"] = {
            str(group): {h: summarize_rows([r for r in rows if r["horizon"] == h and r["group"] == group], function)
                         for h in HORIZONS}
            for group in sorted({r["group"] for r in rows})}
    return result


def _array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def run(checkpoint, output, batch_size=32):
    from train_local_memory_0098 import read_train, load_model, sources, digest
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    torch.set_num_threads(1)
    started = time.perf_counter()
    extra_paths = {"checkpoint_model": checkpoint / "model.pt", "checkpoint_config": checkpoint / "checkpoint.json",
                   "report_code": Path(__file__), "metric_code": ROOT / "experiments/evaluate_long_latent.py",
                   "numerical_code": ROOT / "experiments/evaluate_structured_rom.py",
                   "common_frequency_code": ROOT / "experiments/evaluate_l1_latent.py",
                   "aggregate_code": ROOT / "experiments/run_private_m2.py",
                   "data_canonicalization_code": ROOT / "src/kiki/data/private_dataset.py"}
    def hashes():
        return dict(sources(), **{key: {"path": str(path), "sha256": digest(path)} for key, path in extra_paths.items()})
    before = hashes()
    events, excluded = read_train()
    if (len(events), len(excluded), sum(e["label"] == "unstable" for e in events), len({e["group"] for e in events})) != (346, 23, 35, 24):
        raise AssertionError("registered TRAIN denominator changed")
    if len({e["id"] for e in events}) != 346:
        raise AssertionError("duplicate TRAIN event identity")
    counts = {str(h): sum(len(e["x"]) - 11 >= h * 100 for e in events) for h in (1, 2, 5, 8)}
    if counts != {"1": 315, "2": 311, "5": 311, "8": 311}:
        raise AssertionError("registered actual TRAIN horizons changed")
    model = load_model(checkpoint)
    model.eval()
    output.mkdir(parents=True)
    records = {key: [] for key in ("records", "common_frequency", "normalized_p", "objective_point_error", "numerical")}
    arrays = {"event_ids": np.asarray([e["id"] for e in events]),
              "labels": np.asarray([e["label"] for e in events]),
              "groups": np.asarray([e["group"] for e in events]),
              "full_truth_lengths": np.asarray([len(e["x"]) for e in events])}
    fine_blocks, coarse_blocks = [], []
    event_info, blocks = [], []
    for first in range(0, len(events), batch_size):
        block = events[first:first+batch_size]
        histories = np.stack([e["x"][:11] for e in block])
        fine = _rollouts(model, histories, 1600, .005)
        coarse = _rollouts(model, histories, 800, .01)
        for event, f, c in zip(block, fine, coarse, strict=True):
            rows = event_metrics(model, event, f, c)
            for key, value in rows.items():
                records[key].extend(value)
            event_info.append({"event_id": event["id"], "group": event["group"], "label": event["label"],
                               "truth_length": len(event["x"]), "last_true_record_time_s": (len(event["x"])-1)*.01,
                               "scored_prefix_steps": min(800, len(event["x"])-11),
                               "full_canonical_train_truth_sha256": _array_digest(event["x"])})
            arrays[f"{event['id']}_truth"] = np.asarray(event["x"][:811, :30])
        fine_blocks.append(fine[..., :30])
        coarse_blocks.append(coarse[..., :30])
        blocks.append({"first_event_index": first, "count": len(block), "elapsed_seconds": time.perf_counter()-started})
        print(json.dumps({"completed_train_events": min(first+batch_size, len(events)),
                          "elapsed_seconds": time.perf_counter()-started}), flush=True)
    arrays.update(fine_predictions=np.concatenate(fine_blocks), coarse_predictions=np.concatenate(coarse_blocks))
    # Reproducible scalar arrays accompany the physical trajectories and JSON.
    for key in ("records", "common_frequency", "normalized_p"):
        arrays[f"{key}_rmse"] = np.asarray([np.nan if r.get("rmse") is None else r["rmse"] for r in records[key]]).reshape(346, 5)
        arrays[f"{key}_available"] = np.asarray([r["available"] for r in records[key]]).reshape(346, 5)
        arrays[f"{key}_numerically_admissible"] = np.asarray([r["numerically_admissible"] for r in records[key]]).reshape(346, 5)
    np.savez_compressed(output / "trajectories_and_metrics.npz", **arrays)
    after = hashes()
    if before != after:
        raise RuntimeError("source changed during full TRAIN report")
    severe = {key: [r for r in records[key] if r["horizon"] == "prefix" and r["label"] == "unstable"]
              for key in ("records", "common_frequency", "normalized_p", "objective_point_error")}
    if any(len(rows) != 35 for rows in severe.values()):
        raise AssertionError("all 35 unstable real prefixes must remain present")
    result = {"loop": "0098", "scope": "descriptive final full-TRAIN capability report; not checkpoint selection or generalization evidence",
              "test_read": False, "validation_read": False, "checkpoint": str(checkpoint),
              "denominators": {"raw_train": 369, "eligible_train": 346, "latency_uncovered": 23,
                               "stable": 311, "unstable": 35, "fault_groups": 24},
              "complete_counts": counts, "excluded": excluded, "event_metadata": event_info,
              "dt": {"fine": .005, "coarse": .01}, "refinement_executed": False,
              "numerical_rule": "per actual scored interval; full latent-substep finite and COI/common normalized discrepancy <=.001",
              "prefix_scope": "genuine future prefix after x[10], capped at 8s; no unobserved truth tail",
              "saved_truth_scope": "original x[:811,:30], shorter events retain original length; full lengths and content hashes recorded",
              "physical_metric": "inherited normalized COI angle/speed RMSE; first future step is x[11]",
              "normalized_p_metric": "checkpoint p_scale, dimensionless RMSE",
              "objective_metric": "train_structured_rom._point_error event mean; relative/common/p terms; same checkpoint scales",
              "summary": build_summary(records), "unstable_prefix_event_rows": severe,
              "source_hashes_before": before, "source_hashes_after": after,
              "batch_traces": blocks, "seconds": time.perf_counter()-started}
    (output / "per_event_rows.json").write_text(json.dumps(_safe(records), indent=2, allow_nan=False), encoding="utf-8")
    (output / "evidence.json").write_text(json.dumps(_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = run(args.checkpoint, args.output, args.batch_size)
    print(json.dumps({"output": str(args.output), "seconds": result["seconds"], "complete_counts": result["complete_counts"]}))


if __name__ == "__main__":
    main()
