"""Fixed full-TRAIN replay of the unchanged 0098 objective, with grid QA.

No model fitting, validation reader, or TEST access occurs in this module.
All reported losses use actual .01-second truth samples; smaller RK4 steps
only change integration, including the physical .01-second latent step.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from train_joint_campaign import sample_batch
from train_structured_rom import _point_error, METRIC
from train_local_memory_0098 import NORM, process_memory_bytes, sources
from evaluate_structured_rom import _safe

HORIZONS = (1, 2, 5)
TERMS = ("risk", "mean", "local", "latent_consistency", "torque", "complexity")


class MonitorInterrupted(RuntimeError):
    pass


def _guard(deadline, reserve=3.):
    if time.perf_counter() + reserve >= deadline:
        raise MonitorInterrupted("monitor deadline reached; serialization reserve retained")
    if process_memory_bytes() > 8 * 1024**3:
        raise MonitorInterrupted("monitor 8 GiB memory ceiling reached")


def _manifest_hash(manifest):
    clean = {k: v for k, v in manifest.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_manifest(events, *, strict=True):
    """Independent seed99 sampler; no training or torch RNG is consumed."""
    if len(events) % 2 or len({e["id"] for e in events}) != len(events):
        raise ValueError("monitor coverage needs an even, unique event pool")
    if strict and (len(events), sum(e["label"] == "unstable" for e in events), len({e["group"] for e in events})) != (346, 35, 24):
        raise ValueError("registered TRAIN pool must be 346/35/24")
    rng = np.random.default_rng(99)
    order = rng.permutation(len(events))
    lookup = {e["id"]: i for i, e in enumerate(events)}
    batches = []
    for batch in range(len(events) // 2):
        chosen, _ = sample_batch(events, rng, 10)
        for index in order[2*batch:2*batch+2]:
            event = events[int(index)]
            chosen.append((event, int(rng.integers(10, len(event["x"])-1))))
        batches.append([{"event_id": e["id"], "event_index": lookup[e["id"]], "endpoint": endpoint}
                        for e, endpoint in chosen])
    manifest = {"seed": 99, "batch_size": 12, "batches": batches,
                "coverage_order": order.tolist(), "event_ids": [e["id"] for e in events],
                "lengths": [len(e["x"]) for e in events], "labels": [e["label"] for e in events],
                "groups": [str(e["group"]) for e in events], "strict_registered_pool": bool(strict)}
    manifest["sha256"] = _manifest_hash(manifest)
    return manifest


def validate_manifest(events, manifest):
    if manifest.get("sha256") != _manifest_hash(manifest):
        raise ValueError("fixed monitor manifest hash changed")
    expected = ([e["id"] for e in events], [len(e["x"]) for e in events],
                [e["label"] for e in events], [str(e["group"]) for e in events])
    actual = tuple(manifest[k] for k in ("event_ids", "lengths", "labels", "groups"))
    if actual != expected or manifest["seed"] != 99:
        raise ValueError("monitor pool identity/length/label/group changed")
    cover = []
    for batch in manifest["batches"]:
        if len(batch) != 12:
            raise ValueError("fixed batches must retain original twelve-event risk")
        for row in batch:
            i, end = row["event_index"], row["endpoint"]
            if i < 0 or i >= len(events) or events[i]["id"] != row["event_id"] or not 10 <= end < len(events[i]["x"])-1:
                raise ValueError("invalid monitor event identity or causal endpoint")
        cover.extend(r["event_index"] for r in batch[-2:])
    if sorted(cover) != list(range(len(events))) or cover != manifest["coverage_order"]:
        raise ValueError("coverage entries must cover each TRAIN event exactly once")


def _risk(values):
    values = np.asarray(values)
    return .8 * float(values.mean()) + .2 * float(np.sort(values)[-max(1, math.ceil(.1*len(values))):].mean())


def integrate_windows(model, events, windows, horizon, dt, deadline, sink, *, batch_size=64):
    """Retain .01 output plus all-substep finiteness; stop at genuine truth end."""
    substeps = round(.01 / dt)
    if substeps < 1 or abs(substeps*dt-.01) > 1e-12:
        raise ValueError("integration grid must divide the physical .01-second step")
    count, steps, dim = len(windows), round(horizon/.01), model.dim
    path = np.full((count, steps+1, dim), np.nan)
    finite = np.zeros((count, steps+1), dtype=bool)
    available = np.asarray([min(steps, len(events[i]["x"])-1-end) for i, end in windows], dtype=int)
    sink.update(path=path, finite=finite, available=available, completed_windows=0, physical_step=0)
    if np.any(available <= 0):
        raise ValueError("each monitor window requires a real future sample")
    for first in range(0, count, batch_size):
        _guard(deadline)
        stop = min(count, first+batch_size)
        history = torch.as_tensor(np.stack([events[i]["x"][end-10:end+1] for i, end in windows[first:stop]]), dtype=torch.float64)
        state = torch.cat((history[:, -1], model.encode(history)), -1)
        alive_finite = torch.isfinite(state).all(-1)
        path[first:stop, 0] = state.detach().cpu().numpy()
        finite[first:stop, 0] = alive_finite.detach().cpu().numpy()
        for physical in range(1, int(available[first:stop].max())+1):
            if physical % 5 == 1:
                _guard(deadline)
            rows = np.flatnonzero(available[first:stop] >= physical)
            ix = torch.as_tensor(rows, dtype=torch.long)
            advanced = state.index_select(0, ix)
            good = alive_finite.index_select(0, ix)
            for _ in range(substeps):
                advanced = model.rk4(advanced, dt)
                good = good & torch.isfinite(advanced).all(-1)
            state = state.index_copy(0, ix, advanced)
            alive_finite = alive_finite.index_copy(0, ix, good)
            path[first+rows, physical] = advanced.detach().cpu().numpy()
            finite[first+rows, physical] = good.detach().cpu().numpy()
            sink.update(physical_step=physical, active_first=first)
        sink["completed_windows"] = stop
    sink["complete"] = True
    return sink


def _event_statistics(model, events, auto):
    values = np.empty((len(events), 3))
    rows = []
    mass = model.mass.detach().cpu().numpy()
    pscale = model.p_scale.detach().cpu().numpy()
    for i, event in enumerate(events):
        for hindex, horizon in enumerate(HORIZONS):
            count = min(horizon*100, int(auto["available"][i]))
            pred = auto["path"][i, 1:count+1, :30]
            truth = event["x"][11:11+count]
            point = _point_error(torch.as_tensor(pred), torch.as_tensor(truth), model).detach().cpu().numpy()
            error = pred-truth
            delta = error[:, :10] - (error[:, :10]*mass).sum(-1, keepdims=True)/mass.sum()
            omega = error[:, 10:20] - (error[:, 10:20]*mass).sum(-1, keepdims=True)/mass.sum()
            common = (error[:, 10:20]*mass).sum(-1)/mass.sum()
            p = error[:, 20:30]/pscale
            coi_sse = .5*np.sum((delta/METRIC[0])**2+(omega/METRIC[1])**2)
            values[i, hindex] = float(point.mean())
            rows.append({"event_id": event["id"], "group": event["group"], "label": event["label"],
                         "horizon": str(horizon), "steps": count, "complete_horizon": count == horizon*100,
                         "point_error": values[i, hindex], "coi_sse": float(coi_sse), "coi_n": count*10,
                         "coi_rmse": float(np.sqrt(coi_sse/(count*10))),
                         "common_sse": float(np.sum(common**2)), "common_n": count,
                         "common_rmse": float(np.sqrt(np.mean(common**2))),
                         "p_sse": float(np.sum(p**2)), "p_n": count*10, "p_rmse": float(np.sqrt(np.mean(p**2))),
                         "all_substeps_finite": bool(auto["finite"][i, count])})
    return values, rows


def compute_grid(model, events, manifest, dt, force_scale, deadline, sink):
    """Unique autonomous paths + local windows, then exact original batch terms."""
    windows = [(r["event_index"], r["endpoint"]) for batch in manifest["batches"] for r in batch]
    auto = integrate_windows(model, events, [(i, 10) for i in range(len(events))], 5., dt, deadline,
                             sink.setdefault("autonomous", {}))
    values, rows = _event_statistics(model, events, auto)
    sink.update(autonomous_values=values, event_rows=rows)
    local = integrate_windows(model, events, windows, .2, dt, deadline,
                              sink.setdefault("local", {}), batch_size=128)
    n_batches = len(manifest["batches"])
    terms = np.full((n_batches, 3, len(TERMS)), np.nan)
    totals = np.full((n_batches, 3), np.nan)
    sink.update(terms=terms, totals=totals, completed_batches=0)
    for bi, batch in enumerate(manifest["batches"]):
        _guard(deadline)
        samples = [(events[r["event_index"]], r["endpoint"]) for r in batch]
        history = torch.as_tensor(np.stack([e["x"][end-10:end+1] for e, end in samples]), dtype=torch.float64)
        adjacent = torch.as_tensor(np.stack([e["x"][end-9:end+2] for e, end in samples]), dtype=torch.float64)
        state = torch.as_tensor(local["path"][bi*12:(bi+1)*12, 0])
        next_state = torch.as_tensor(local["path"][bi*12:(bi+1)*12, 1])
        latent = float(((next_state[:, 30:]-model.encode(adjacent))/.1).square().mean())
        predicted_torque = state[:, 20:30]-model.mass*model.field(state)[:, 10:20]
        truth_torque = torch.as_tensor(np.stack([e["torque"][end] for e, end in samples]), dtype=torch.float64)
        torque = float(((predicted_torque-truth_torque)/force_scale).square().mean())
        complexity = float(model.complexity(state))
        local_values = []
        for offset, (event, end) in enumerate(samples):
            wi = bi*12+offset
            count = int(local["available"][wi])
            pred = torch.as_tensor(local["path"][wi, 1:count+1])
            truth = torch.as_tensor(event["x"][end+1:end+count+1])
            local_values.append(float(_point_error(pred, truth, model).mean()))
        local_mean = float(np.mean(local_values))
        for hi in range(3):
            long = values[[r["event_index"] for r in batch], hi]
            record = (_risk(long), float(long.mean()), local_mean, latent, torque, complexity)
            terms[bi, hi] = record
            totals[bi, hi] = 10*record[0]+local_mean+.1*latent+.1*torque+1e-4*complexity
        sink["completed_batches"] = bi+1
    sink.update(J=float(totals.mean()), complete=True, dt=dt)
    return sink


def _interval_difference(model, high, low, index, count):
    mass = model.mass.detach().cpu().numpy()
    pscale = model.p_scale.detach().cpu().numpy()
    finite = bool(high["finite"][index, count] and low["finite"][index, count])
    if not finite:
        return {"pass": False, "all_substeps_finite": False, "coi": None, "common": None, "p": None}
    # Match inherited scored numerical rule: include the common initial state.
    difference = high["path"][index, :count+1, :30]-low["path"][index, :count+1, :30]
    d = difference[:, :10]-(difference[:, :10]*mass).sum(-1, keepdims=True)/mass.sum()
    w = difference[:, 10:20]-(difference[:, 10:20]*mass).sum(-1, keepdims=True)/mass.sum()
    common = (difference[:, 10:20]*mass).sum(-1)/mass.sum()
    coi = float(np.sqrt(.5*np.mean((d/METRIC[0])**2+(w/METRIC[1])**2)))
    cm = float(np.sqrt(np.mean((common/METRIC[1])**2)))
    p = float(np.sqrt(np.mean((difference[:, 20:30]/pscale)**2)))
    return {"pass": bool(max(coi, cm, p) <= .001 and np.isfinite([coi, cm, p]).all()),
            "all_substeps_finite": True, "coi": coi, "common": cm, "p": p}


def compare_grids(model, events, fine, reference):
    """Fine is always the .005 scoring grid, even for .0025 verification."""
    auto_rows, local_rows = [], []
    for i, event in enumerate(events):
        for horizon in HORIZONS:
            count = min(horizon*100, int(fine["autonomous"]["available"][i]))
            row = _interval_difference(model, fine["autonomous"], reference["autonomous"], i, count)
            row.update(event_id=event["id"], label=event["label"], horizon=str(horizon), steps=count)
            auto_rows.append(row)
    for i, count in enumerate(fine["local"]["available"]):
        row = _interval_difference(model, fine["local"], reference["local"], i, int(count))
        row["window_index"] = i
        local_rows.append(row)
    difference = np.abs(fine["terms"]-reference["terms"])
    threshold = .001*np.maximum(1., np.abs(fine["terms"]))
    finite_terms = np.isfinite(fine["terms"]) & np.isfinite(reference["terms"])
    terms_pass = finite_terms & (difference <= threshold)
    okay = all(r["pass"] for r in auto_rows+local_rows) and bool(terms_pass.all()) and np.isfinite([fine["J"], reference["J"]]).all()
    return {"pass": bool(okay), "autonomous": auto_rows, "local": local_rows,
            "term_names": list(TERMS), "term_absolute_differences": difference,
            "term_pass": terms_pass, "J_grid_difference": float(abs(fine["J"]-reference["J"]))}


def _descriptive_summary(rows, metric):
    values = np.asarray([r[metric+"_rmse"] for r in rows])
    return {"n_events": len(rows), "mean": float(values.mean()),
            "pooled": float(np.sqrt(sum(r[metric+"_sse"] for r in rows)/sum(r[metric+"_n"] for r in rows))),
            "cvar10": float(np.sort(values)[-max(1, math.ceil(.1*len(values))):].mean())}


def event_summary(rows, numeric):
    gates = {(r["event_id"], r["horizon"]): r["pass"] for r in numeric}
    def one(items):
        if not items:
            return None
        valid = all(gates.get((r["event_id"], r["horizon"]), False) and np.isfinite(r["point_error"]) for r in items)
        raw = {m: _descriptive_summary(items, m) for m in ("coi", "common", "p")}
        raw["point_error_mean"] = float(np.mean([r["point_error"] for r in items]))
        return {"n_events": len(items), "numerically_valid": valid, "official": raw if valid else None, "raw": raw}
    result = {}
    for horizon in map(str, HORIZONS):
        select = [r for r in rows if r["horizon"] == horizon]
        result[horizon] = {"genuine_prefix_all_events": one(select),
                           "complete_horizon": one([r for r in select if r["complete_horizon"]]),
                           "unstable_prefix": one([r for r in select if r["label"] == "unstable"]),
                           "groups": {str(g): one([r for r in select if r["group"] == g]) for g in sorted({r["group"] for r in select})}}
    return result


def _save(output, result, storage):
    arrays = {}
    for grid, data in storage.items():
        for kind in ("autonomous", "local"):
            for key in ("path", "finite", "available"):
                if key in data.get(kind, {}):
                    arrays[f"{grid}_{kind}_{key}"] = data[kind][key]
        for key in ("terms", "totals", "autonomous_values"):
            if key in data:
                arrays[f"{grid}_{key}"] = data[key]
    # Uncompressed arrays bound serialization work for partial deadline exits.
    np.savez(output / "monitor_arrays.npz", **arrays)
    (output / "evidence.json").write_text(json.dumps(_safe(result), indent=2, allow_nan=False), encoding="utf-8")


def _model_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str((array.dtype, array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def run_monitor(model, events, manifest, output, deadline, allow_refine=True, *, force_scale=None):
    """deadline is absolute time.perf_counter(); interrupted J is never selectable."""
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    validate_manifest(events, manifest)
    torch.set_num_threads(1)
    before_torch = torch.get_rng_state().clone()
    before_numpy = np.random.get_state()
    if force_scale is None:
        with np.load(NORM, allow_pickle=False) as data:
            force_scale = float(data["force_scale"])
    if not np.isfinite(force_scale) or force_scale <= 0:
        raise ValueError("original force normalization must be finite and positive")
    path = Path(__file__)
    own_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    source_before = sources()
    model_before = _model_hash(model)
    result = {"loop": "0099", "test_read": False, "validation_read": False,
              "complete": False, "numerical_pass": False, "refined": False,
              "J_fine": None, "J_coarse": None, "raw_J_fine": None, "raw_J_coarse": None,
              "S_fine": {str(h): None for h in HORIZONS}, "raw_S_fine": {},
              "manifest_sha256": manifest["sha256"], "force_scale": force_scale,
              "source_before": source_before, "monitor_code_sha256": own_hash,
              "model_state_sha256_before": model_before,
              "status": "running", "selection_grid": .005,
              "scope": "fixed batch replay estimate, not exact sampling expectation or independent evidence"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    storage = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if hasattr(model, "begin_graph_cache"):
                model.begin_graph_cache()
            try:
                for name, dt in (("coarse", .01), ("fine", .005)):
                    _guard(deadline)
                    start_grid = time.perf_counter()
                    grid = compute_grid(model, events, manifest, dt, force_scale, deadline, storage.setdefault(name, {}))
                    result[f"raw_J_{name}"] = grid["J"]
                    result[f"{name}_seconds"] = time.perf_counter()-start_grid
                qa = compare_grids(model, events, storage["fine"], storage["coarse"])
                result["initial_grid_qa"] = qa
                reference = "coarse"
                if not qa["pass"] and allow_refine:
                    _guard(deadline)
                    result["refined"] = True
                    start_grid = time.perf_counter()
                    compute_grid(model, events, manifest, .0025, force_scale, deadline, storage.setdefault("finer", {}))
                    result["finer_seconds"] = time.perf_counter()-start_grid
                    result["raw_J_finer"] = storage["finer"]["J"]
                    qa = compare_grids(model, events, storage["fine"], storage["finer"])
                    reference = "finer"
                result.update(complete=True, numerical_pass=qa["pass"], final_grid_qa=qa,
                              verified_reference=reference, numerical_sensitivity_d=qa["J_grid_difference"],
                              status="complete" if qa["pass"] else "numerically_unresolved")
                result["J_fine"] = storage["fine"]["J"] if qa["pass"] else None
                result["J_coarse"] = storage["coarse"]["J"] if result["initial_grid_qa"]["pass"] else None
                result["batch_terms"] = {name: {"term_names": list(TERMS), "terms": g["terms"], "totals": g["totals"]} for name, g in storage.items()}
                result["event_rows_fine"] = storage["fine"]["event_rows"]
                result["event_summary"] = event_summary(storage["fine"]["event_rows"], qa["autonomous"])
                unstable = np.asarray([e["label"] == "unstable" for e in events])
                result["S_grid_differences"] = {}
                for hi, horizon in enumerate(map(str, HORIZONS)):
                    fine_values = storage["fine"]["autonomous_values"][unstable, hi]
                    ref_values = storage[reference]["autonomous_values"][unstable, hi]
                    raw = float(fine_values.mean())
                    valid = np.isfinite(fine_values).all() and all(r["pass"] for r in qa["autonomous"] if r["label"] == "unstable" and r["horizon"] == horizon)
                    result["raw_S_fine"][horizon] = raw
                    result["S_fine"][horizon] = raw if valid else None
                    result["S_grid_differences"][horizon] = float(abs(raw-ref_values.mean()))
            finally:
                if hasattr(model, "end_graph_cache"):
                    model.end_graph_cache()
    except MonitorInterrupted as error:
        result.update(status="interrupted", interruption=str(error), complete=False, J_fine=None, J_coarse=None,
                      S_fine={str(h): None for h in HORIZONS})
    except Exception as error:
        result.update(status="implementation_failure", error_type=type(error).__name__, error=str(error),
                      J_fine=None, J_coarse=None, complete=False)
        raise
    finally:
        model.train(was_training)
        unstable = np.asarray([e["label"] == "unstable" for e in events])
        for name, grid in storage.items():
            if "autonomous_values" in grid:
                result[f"raw_S_{name}"] = {str(h): float(grid["autonomous_values"][unstable, hi].mean())
                                           for hi, h in enumerate(HORIZONS)}
        # A deadline during refinement must not erase the already completed
        # fine-grid severe diagnostics, although all selectable J/S stay null.
        if "autonomous_values" in storage.get("fine", {}):
            result["raw_S_fine"] = {str(h): float(storage["fine"]["autonomous_values"][unstable, hi].mean())
                                    for hi, h in enumerate(HORIZONS)}
            result["event_rows_fine"] = storage["fine"]["event_rows"]
        result["source_after"] = sources()
        result["model_state_sha256_after"] = _model_hash(model)
        result["sources_unchanged"] = (source_before == result["source_after"] and own_hash == hashlib.sha256(path.read_bytes()).hexdigest()
                                       and model_before == result["model_state_sha256_after"])
        result["rng_unchanged"] = bool(torch.equal(before_torch, torch.get_rng_state()) and all(
            np.array_equal(a, b) for a, b in zip(before_numpy, np.random.get_state())))
        if not result["sources_unchanged"] or not result["rng_unchanged"]:
            result.update(status="integrity_failure", complete=False, J_fine=None, J_coarse=None,
                          numerical_pass=False, S_fine={str(h): None for h in HORIZONS})
        result["progress"] = {name: {"completed_batches": g.get("completed_batches", 0),
                                     "autonomous_completed_windows": g.get("autonomous", {}).get("completed_windows", 0),
                                     "local_completed_windows": g.get("local", {}).get("completed_windows", 0)} for name, g in storage.items()}
        result["seconds"] = time.perf_counter()-started
        _save(output, result, storage)
        result["seconds"] = time.perf_counter()-started
        (output / "evidence.json").write_text(json.dumps(_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    return _safe(result)
