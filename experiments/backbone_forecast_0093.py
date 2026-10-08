"""Small, cache-only forecast/scoring helpers for Loop 0093.

This module deliberately does not load a model, a reader, or any dataset.  The
deployment worker supplies the already checked candidate and cached TRAIN
arrays.  Keeping the arithmetic here pure makes the candidate forecast and
the independent audit able to compare the same 1038 rows.
"""
from __future__ import annotations

from typing import Any, Mapping
from pathlib import Path
import time

import numpy as np

METRIC = (0.47675314588683515, 1.0033667249602924)
HORIZONS = (1.0, 2.0, 5.0)
DT = 0.01


def _finite(name: str, value: Any) -> np.ndarray:
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise FloatingPointError(f"nonfinite {name}")
    return a


def validate_cache(cache: Mapping[str, Any], *, event_ids=None, labels=None, groups=None) -> dict[str, Any]:
    """Validate the frozen 0088 cache without filtering any rows."""
    required = ("event_ids", "labels", "lengths", "truth", "qenc", "mass", "p_scale",
                "fine_baseline", "coarse_baseline")
    missing = [k for k in required if k not in cache]
    if missing:
        raise KeyError(f"missing cache fields: {missing}")
    ids = np.asarray(cache["event_ids"]); labs = np.asarray(cache["labels"])
    lengths = _finite("lengths", cache["lengths"])
    if ids.shape != (346,) or labs.shape != (346,) or lengths.shape != (346,):
        raise ValueError("0088 cache must contain 346 TRAIN events")
    if not np.equal(lengths, lengths.astype(np.int64)).all() or np.any(lengths <= 0) or np.any(lengths > 500):
        raise ValueError("invalid genuine lengths")
    if event_ids is not None and list(ids.astype(str)) != list(np.asarray(event_ids).astype(str)):
        raise AssertionError("event ordering mismatch")
    if labels is not None and list(labs.astype(str)) != list(np.asarray(labels).astype(str)):
        raise AssertionError("label ordering mismatch")
    if groups is not None:
        if "groups" not in cache: raise KeyError("missing groups")
        got = np.asarray(cache["groups"])
        expected = np.asarray(groups)
        if got.shape != (346,) or expected.shape != (346,) or not np.array_equal(got.astype(str), expected.astype(str)):
            raise AssertionError("group ordering mismatch")
        if len(np.unique(got.astype(str))) != 24: raise AssertionError("TRAIN group count")
    if sum(str(x) == "stable" for x in labs) != 311 or sum(str(x) == "unstable" for x in labs) != 35:
        raise AssertionError("TRAIN label counts")
    mass = _finite("mass", cache["mass"]); pscale = _finite("p_scale", cache["p_scale"])
    if mass.shape != (10,) or pscale.shape != (10,) or np.any(mass <= 0) or np.any(pscale <= 0):
        raise ValueError("invalid metric scales")
    qenc = _finite("qenc", cache["qenc"])
    if qenc.shape != (346, 3):
        raise ValueError("qenc shape")
    truth = np.asarray(cache["truth"])
    if not np.issubdtype(truth.dtype, np.number):
        raise TypeError("truth must be numeric")
    if truth.ndim != 3 or truth.shape[0] != 346 or truth.shape[-1] < 30:
        raise ValueError("truth shape")
    for key, width in (("fine_baseline", 33), ("coarse_baseline", 33)):
        a = np.asarray(cache[key])
        if not np.issubdtype(a.dtype, np.number): raise TypeError(f"{key} must be numeric")
        if a.ndim != 3 or a.shape[0] != 346 or a.shape[-1] != width:
            raise ValueError(f"{key} shape")
        for i, n in enumerate(lengths.astype(int)):
            end = 2 * n + 1 if key.startswith("fine") else n + 1
            if end > a.shape[1] or not np.isfinite(a[i, :end]).all():
                raise FloatingPointError(f"{key} genuine prefix {i}")
            if n + 1 > truth.shape[1] or not np.isfinite(truth[i, :n + 1, :30]).all():
                raise FloatingPointError(f"truth genuine prefix {i}")
    result = {"event_ids": ids, "labels": labs, "lengths": lengths.astype(np.int64),
            "truth": truth, "qenc": qenc, "mass": mass, "p_scale": pscale}
    if "groups" in cache: result["groups"] = np.asarray(cache["groups"])
    return result


def masked_rk4(model: Any, states: Any, lengths: np.ndarray, dt: float,
               *, deadline: float | None = None, progress: dict | None = None,
               partial=None, event_ids=None) -> np.ndarray:
    """Call the preregistered checked rollout; no local RK4 implementation."""
    from fit_coefficient_block_0082 import _checked_rollout
    lengths = np.asarray(lengths, dtype=np.int64)
    if states.ndim != 2 or states.shape[0] != len(lengths): raise ValueError("state/length shape")
    if np.any(lengths <= 0): raise ValueError("invalid rollout input")
    return _checked_rollout(model, states, lengths, dt, deadline=float('inf') if deadline is None else deadline,
                            progress=progress, partial=partial, event_ids=event_ids)


def forecast_candidate(model: Any, cache: Mapping[str, Any], *, deadline: float,
                       output: Path, progress: dict, counters: dict) -> tuple[np.ndarray, np.ndarray]:
    """Construct the frozen initial states and run exactly the paired .005/.01 forecasts."""
    import torch
    c = validate_cache(cache)
    states_np = np.concatenate((c["truth"][:, 0, :30], c["qenc"]), axis=1)
    if states_np.shape != (346, 33):
        raise ValueError("0093 requires 30 physical plus 3 latent initial coordinates")
    states = torch.as_tensor(states_np, dtype=torch.float64)
    counters['forecast_runs'] += 1
    fine = masked_rk4(model, states, c["lengths"] * 2, .005, deadline=deadline,
                      progress=progress, partial=output/'partial_fine.npz', event_ids=c['event_ids'])
    np.savez_compressed(output/'completed_fine.npz', fine_candidate=fine)
    counters['forecast_runs'] += 1
    coarse = masked_rk4(model, states, c["lengths"], .01, deadline=deadline,
                        progress=progress, partial=output/'partial_coarse.npz', event_ids=c['event_ids'])
    return fine, coarse


def point_error(pred: np.ndarray, truth: np.ndarray, mass: np.ndarray, p_scale: np.ndarray) -> np.ndarray:
    pred, truth = _finite("prediction", pred), _finite("truth segment", truth)
    if pred.ndim != 2 or truth.ndim != 2 or pred.shape[0] != truth.shape[0] or pred.shape[1] < 30 or truth.shape[1] < 30:
        raise ValueError("point shape")
    e = pred[:, :30] - truth[:, :30]
    d = e[:, :10] - (e[:, :10] * mass).sum(1, keepdims=True) / mass.sum()
    w = e[:, 10:20] - (e[:, 10:20] * mass).sum(1, keepdims=True) / mass.sum()
    c = (e[:, 10:20] * mass).sum(1) / mass.sum()
    return .5 * ((d / METRIC[0]) ** 2).mean(1) + .5 * ((w / METRIC[1]) ** 2).mean(1) + (c / METRIC[1]) ** 2 + ((e[:, 20:30] / p_scale) ** 2).mean(1)


def risk(values: np.ndarray) -> dict[str, float | int]:
    values = _finite("event risks", values).reshape(-1)
    if len(values) != 346: raise ValueError("risk requires all 346 events")
    tail = np.sort(values)[-35:]
    return {"n_events": 346, "mean": float(values.mean()), "tail35": float(tail.mean()),
            "risk": float(.8 * values.mean() + .2 * tail.mean())}


def numeric_metrics(fine: np.ndarray, coarse: np.ndarray, lengths: np.ndarray,
                    mass: np.ndarray, p_scale: np.ndarray) -> list[dict[str, float]]:
    if fine.shape[0] != 346 or coarse.shape[0] != 346: raise ValueError("numeric event count")
    rows = []
    for i, n in enumerate(np.asarray(lengths, dtype=int)):
        a, b = fine[i, :2*n+1], coarse[i, :n+1]
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise FloatingPointError(f"numeric {i}")
        x = a[::2] - b; d = x[:, :10] - (x[:, :10] * mass).sum(1, keepdims=True) / mass.sum()
        rw = x[:, 10:20]; w = rw - (rw * mass).sum(1, keepdims=True) / mass.sum(); c = (rw * mass).sum(1) / mass.sum()
        rows.append({"coi": float(np.sqrt(.5*np.mean((d/METRIC[0])**2 + (w/METRIC[1])**2))),
                     "common": float(np.sqrt(np.mean((c/METRIC[1])**2))),
                     "p": float(np.sqrt(np.mean((x[:,20:30]/p_scale)**2)))})
    return rows


def score_forecast(fine: np.ndarray, cache: Mapping[str, Any]) -> dict[str, Any]:
    c = validate_cache(cache); lengths, truth = c["lengths"], c["truth"]
    rows, horizons = [], {}
    for h in HORIZONS:
        vals = []
        for i, n in enumerate(lengths):
            k = min(round(h / DT), int(n)); vals.append(float(point_error(fine[i, 2:2*k+1:2], truth[i, 1:k+1], c["mass"], c["p_scale"]).mean()))
            rows.append({"event_id": str(c["event_ids"][i]), "label": str(c["labels"][i]), "horizon": h, "available_steps": int(k), "candidate": vals[-1]})
        horizons[str(int(h))] = risk(np.asarray(vals))
    unstable = [r["candidate"] for r in rows if r["horizon"] == 5.0 and r["label"] == "unstable"]
    return {"per_event": rows, "horizons": horizons,
            "unstable_5s": {"mean": float(np.mean(unstable)), "worst4": float(np.mean(sorted(unstable)[-4:]))}}


def score_pair(fine_candidate: np.ndarray, fine_baseline: np.ndarray,
               cache: Mapping[str, Any]) -> dict[str, Any]:
    """Score candidate and cached baseline together, preserving every 1038 row."""
    c = validate_cache(cache); rows, horizons = [], {}
    for h in HORIZONS:
        cv, bv = [], []
        for i, n in enumerate(c["lengths"]):
            k = min(round(h / DT), int(n)); truth = c["truth"][i, 1:k+1]
            cp = fine_candidate[i, 2:2*k+1:2]; bp = fine_baseline[i, 2:2*k+1:2]
            cv.append(float(point_error(cp, truth, c["mass"], c["p_scale"]).mean()))
            bv.append(float(point_error(bp, truth, c["mass"], c["p_scale"]).mean()))
            rows.append({"event_id": str(c["event_ids"][i]), "label": str(c["labels"][i]),
                         "horizon": h, "available_steps": int(k), "candidate": cv[-1], "baseline": bv[-1]})
        horizons[str(int(h))] = {"candidate": risk(np.asarray(cv)), "baseline": risk(np.asarray(bv))}
    unstable = {name: {"mean": float(np.mean([r[name] for r in rows if r["horizon"] == 5.0 and r["label"] == "unstable"])),
                       "worst4": float(np.mean(sorted(r[name] for r in rows if r["horizon"] == 5.0 and r["label"] == "unstable")[-4:]))}
                for name in ("candidate", "baseline")}
    average = {name: float(np.mean([horizons[str(int(h))][name]["risk"] for h in HORIZONS])) for name in ("candidate", "baseline")}
    return {"per_event": rows, "horizons": horizons, "unstable_5s": unstable, "average_risk": average}
