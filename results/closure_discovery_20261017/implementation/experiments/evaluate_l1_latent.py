"""L1 stability/latency baseline for cached JointROM validation forecasts.

This module deliberately consumes cached forecasts from ``evaluate_long_latent``.
It reads only validation metadata and the first eleven real samples needed to
reconstruct the causal 100-ms decision window.  It never fits, opens TEST
payloads, or launches the registered 2.5-ms refinement diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from kiki.physics import coi_project  # noqa: E402

DT = 0.01
DECISION_TIME_S = 0.10
OMEGA_S = 2.0 * np.pi * 60.0
HORIZONS = (1, 2, 5, 8)
LOS_CONFIG = {
    "los_angle_rad": np.pi,
    "los_relative_speed_pu": 0.02,
    "los_hold_samples": 5,
    "stable_tail_seconds": 1.0,
    "stable_tail_speed_pu": 0.005,
    "sample_interval_s": DT,
}
ROOTS = (
    ROOT / "private_data/ieee39_postfault",
    ROOT / "private_data/ieee39_expansion",
)
BASE = ROOT / "private_data/task2_m3_0020/model.npz"
NETWORK = ROOT / "private_data/task2_electrical_ports_0048/physical_network.npz"
SCALES = np.array([.47675314588683515, 1.0033667249602924])
SUPPORTED_STATE_DIMS = (33, 34)


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _existing_scored_rows_fail(rows: Any) -> bool:
    """Validate saved numerical rows before using them as a refinement gate.

    The cached evidence is an external artifact: missing or malformed fields
    are failures, not reasons to silently exclude a row from the gate.
    ``rows`` is already restricted to genuine prefix/complete comparisons by
    the caller, so an empty collection is itself a failure.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        return True
    for row in rows:
        if not isinstance(row, dict):
            return True
        finite_fine = row.get("finite_fine")
        finite_coarse = row.get("finite_coarse")
        if not isinstance(finite_fine, (bool, np.bool_)) or not isinstance(finite_coarse, (bool, np.bool_)):
            return True
        if not bool(finite_fine) or not bool(finite_coarse):
            return True
        try:
            step_rmse = float(row["step_rmse"])
        except (KeyError, TypeError, ValueError):
            return True
        if not np.isfinite(step_rmse) or step_rmse < 0.0 or step_rmse > 1e-3:
            return True
    return False


def restore_physical_path(
    sync_state: np.ndarray, equilibrium_delta: np.ndarray, omega_s: float = OMEGA_S
) -> np.ndarray:
    """Map [delta_sync, omega_rad_s, p, q] to raw-ANDES delta/omega units.

    The equilibrium angle is generator-specific and must be added back before
    taking ``ptp``; subtracting different equilibrium angles is not a harmless
    common shift.
    """
    x = np.asarray(sync_state, dtype=float)
    if x.ndim != 2 or x.shape[1] < 20:
        raise ValueError("state must be [time, >=20]")
    eq = np.asarray(equilibrium_delta, dtype=float).reshape(-1)
    if eq.shape != (10,):
        raise ValueError("equilibrium_delta must have ten generators")
    out = x.copy()
    out[:, :10] = x[:, :10] + eq[None, :]
    out[:, 10:20] = x[:, 10:20] / float(omega_s) + 1.0
    return out


def trigger_index(raw_path: np.ndarray, inertia: np.ndarray, config: dict[str, Any] | None = None) -> int | None:
    """Apply the frozen ANDES LOS rule to a finite raw path."""
    cfg = dict(LOS_CONFIG)
    if config:
        cfg.update(config)
    raw = np.asarray(raw_path, dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 20:
        raise ValueError("raw_path must be [time, >=20]")
    # Import the repository's authoritative implementation only when the
    # real-data evaluator is used; synthetic contract tests stay dependency
    # light and cannot accidentally open a payload.
    try:
        from kiki.data.private_dataset import los_index
    except ModuleNotFoundError as exc:  # pragma: no cover - only minimal synthetic environments
        if exc.name != "h5py":
            raise
        angle = np.ptp(raw[:, :10], axis=1)
        speed = np.max(np.abs(coi_project(raw[:, 10:20], np.asarray(inertia))), axis=1)
        exceeds = (angle > cfg["los_angle_rad"]) & (speed > cfg["los_relative_speed_pu"])
        hold = int(cfg["los_hold_samples"])
        if len(exceeds) < hold:
            return None
        hits = np.flatnonzero(np.convolve(exceeds.astype(int), np.ones(hold, dtype=int), mode="valid") == hold)
        return int(hits[0] + hold - 1) if len(hits) else None
    return los_index(raw[:, :10], raw[:, 10:20], np.asarray(inertia, dtype=float), cfg)


def _finite_prefix(path: np.ndarray) -> tuple[np.ndarray, int | None]:
    finite = np.isfinite(np.asarray(path)).all(axis=1)
    bad = np.flatnonzero(~finite)
    stop = int(bad[0]) if len(bad) else len(path)
    return np.asarray(path)[:stop], (int(bad[0]) if len(bad) else None)


def window_settled(raw_path: np.ndarray, inertia: np.ndarray, config: dict[str, Any] | None = None) -> bool:
    """Return the frozen 1-s tail screen, not a stability certificate."""
    cfg = dict(LOS_CONFIG)
    if config:
        cfg.update(config)
    raw = np.asarray(raw_path, dtype=float)
    tail = max(1, round(float(cfg["stable_tail_seconds"]) / float(cfg["sample_interval_s"])))
    if len(raw) < tail:
        return False
    tail_raw = raw[-tail:]
    speed = np.max(np.abs(coi_project(tail_raw[:, 10:20], np.asarray(inertia))), axis=1)
    return bool(
        np.all(np.ptp(tail_raw[:, :10], axis=1) < float(cfg["los_angle_rad"]))
        and np.all(speed < float(cfg["stable_tail_speed_pu"]))
    )


def classify_prediction(
    raw_path: np.ndarray,
    inertia: np.ndarray,
    *,
    true_event_time_s: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a full observed+forecast path with explicit unresolved states."""
    raw = np.asarray(raw_path, dtype=float)
    if true_event_time_s is not None and true_event_time_s <= DECISION_TIME_S + 1e-9:
        return {
            "category": "latency_uncovered",
            "predicted_los": False,
            "window_settled": False,
            "trigger_index": None,
            "finite_before_hit": None,
        }
    finite_raw, first_bad = _finite_prefix(raw)
    hit = trigger_index(finite_raw, inertia, config) if len(finite_raw) else None
    if hit is not None:
        return {
            "category": "predicted_los",
            "predicted_los": True,
            "window_settled": False,
            "trigger_index": int(hit),
            "finite_before_hit": True,
            "first_nonfinite_index": first_bad,
        }
    if first_bad is not None:
        return {
            "category": "numerical_unresolved",
            "predicted_los": False,
            "window_settled": False,
            "trigger_index": None,
            "finite_before_hit": False,
            "first_nonfinite_index": first_bad,
        }
    settled = window_settled(raw, inertia, config)
    return {
        "category": "window_settled" if settled else "window_no_los_unsettled",
        "predicted_los": False,
        "window_settled": settled,
        "trigger_index": None,
        "finite_before_hit": True,
        "first_nonfinite_index": None,
    }


def apply_numerical_gate(classification: dict[str, Any], passed: bool) -> dict[str, Any]:
    """Numerical failure dominates physical labels for eligible events."""
    if passed or classification.get("category") == "latency_uncovered":
        return classification
    return {
        **classification,
        "category": "numerical_unresolved",
        "predicted_los": False,
        "window_settled": False,
        "numerical_pair_failed": True,
        "unreliable_trigger_index": classification.get("trigger_index"),
        "trigger_index": None,
    }


def trigger_times(index: int | None, *, clearing_time_s: float, record_start_delay_s: float | None) -> dict[str, float | None]:
    """Return record-relative, decision-relative, and absolute ANDES times."""
    if index is None:
        return {
            "predicted_los_time_record_s": None,
            "predicted_los_time_decision_s": None,
            "predicted_los_time_absolute_s": None,
        }
    record = float(index) * DT
    delay = 0.0 if record_start_delay_s is None else float(record_start_delay_s)
    return {
        "predicted_los_time_record_s": record,
        "predicted_los_time_decision_s": record - DECISION_TIME_S,
        "predicted_los_time_absolute_s": float(clearing_time_s) + delay + record,
    }


def _common_frequency_row(pred: np.ndarray, truth: np.ndarray, mass: np.ndarray, steps: int, horizon: str) -> dict[str, Any]:
    steps = max(0, min(int(steps), len(pred) - 1, len(truth) - 1))
    if steps == 0:
        return {"horizon": horizon, "n": 0, "sse": None, "rmse": None, "complete": False}
    err = np.asarray(pred[1 : steps + 1, 10:20]) - np.asarray(truth[1 : steps + 1, 10:20])
    common = (err * np.asarray(mass)[None, :]).sum(axis=1) / np.asarray(mass).sum()
    sse = float(common @ common)
    return {"horizon": horizon, "n": int(steps), "sse": sse, "rmse": float(np.sqrt(sse / steps)), "complete": True}


def common_frequency_metrics(pred: np.ndarray, truth: np.ndarray, mass: np.ndarray) -> list[dict[str, Any]]:
    """Event-level common-frequency SSE/n/RMSE for fixed horizons and prefix."""
    rows = []
    for horizon in HORIZONS:
        steps = round(horizon / DT)
        if len(truth) < steps + 1:
            row = {"horizon": str(horizon), "n": 0, "sse": None, "rmse": None, "complete": False}
        else:
            row = _common_frequency_row(pred, truth, mass, steps, str(horizon))
            row["complete"] = True
        rows.append(row)
    prefix_steps = min(max(len(truth) - 1, 0), round(8.0 / DT))
    row = _common_frequency_row(pred, truth, mass, prefix_steps, "prefix")
    row["complete"] = False
    rows.append(row)
    return rows


def _read_validation_starts(base_pm_star: np.ndarray) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Read explicit validation metadata and first eleven samples only."""
    by_id: dict[str, dict[str, Any]] = {}
    latency: list[dict[str, Any]] = []
    for root in ROOTS:
        import h5py

        splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
        with (root / "dataset_index.csv").open(encoding="utf-8", newline="") as handle:
            rows = {row["event_id"]: row for row in csv.DictReader(handle)}
        with h5py.File(root / "dataset.h5", "r") as file:
            equilibrium_delta = np.asarray(file["equilibrium/delta"][:], dtype=float)
            inertia = np.asarray(file["inertia"][:], dtype=float)
            omega_s = float(file.attrs["omega_s"])
            for event_id in splits["validation"]:
                row = rows[event_id]
                meta = {
                    "event_id": event_id,
                    "group": row.get("fault_bus", row.get("group")),
                    "label": row.get("label", row.get("outcome_label")),
                    "event_time_s": _optional_float(row.get("event_time_s")),
                    "trajectory_end_s": float(row["trajectory_end_s"]),
                    "clearing_time_s": float(row["clearing_time_s"]),
                    "record_start_delay_s": _optional_float(row.get("record_start_delay_s")),
                    "source_root": str(root),
                    "inertia": inertia.copy(),
                    "equilibrium_delta": equilibrium_delta.copy(),
                    "omega_s": omega_s,
                }
                if meta["trajectory_end_s"] <= DECISION_TIME_S + 1e-9:
                    latency.append(meta)
                    continue
                event = file["events"][event_id]
                # Exactly the causal history used by JointROM; no future payload.
                raw = {key: np.asarray(event[key][:11], dtype=float) for key in ("t", "delta", "omega", "Pm")}
                state = np.column_stack(
                    (
                        raw["delta"] - equilibrium_delta[None, :],
                        (raw["omega"] - 1.0) * omega_s,
                        raw["Pm"] - np.asarray(base_pm_star, dtype=float)[None, :],
                    )
                )
                if state.shape != (11, 30) or not np.isfinite(state).all():
                    raise ValueError(f"invalid causal history for {event_id}")
                meta["history"] = state
                by_id[event_id] = meta
    return by_id, latency


def _load_cached(evaluation: Path) -> tuple[dict[str, Any], list[str], np.ndarray, np.ndarray, list[np.ndarray]]:
    evidence = json.loads((evaluation / "evidence.json").read_text(encoding="utf-8"))
    arrays = np.load(evaluation / "trajectories.npz", allow_pickle=False)
    ids = [str(v) for v in arrays["event_ids"]]
    fine = np.asarray(arrays["fine_predictions"], dtype=float)
    coarse = np.asarray(arrays["coarse_predictions"], dtype=float)
    # Unstable genuine prefixes have shorter truth arrays; retain ragged rows.
    truths = [np.asarray(arrays[f"{event_id}_truth"], dtype=float) for event_id in ids]
    if fine.ndim != 3 or coarse.ndim != 3 or fine.shape[0] != len(ids) or coarse.shape[0] != len(ids):
        raise ValueError("cached predictions have incompatible shapes")
    if fine.shape[-1] != coarse.shape[-1] or fine.shape[-1] not in SUPPORTED_STATE_DIMS:
        raise ValueError("cached predictions must share exactly 33 or 34 states")
    return evidence, ids, fine, coarse, truths


def _load_finer(path: Path | None, ids: list[str], expected_model_sha: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load an already-saved 2.5-ms refinement result; never execute it."""
    if path is None:
        return None, {"provided": False, "used": False, "status": "not_provided"}
    artifact = path / "step_0025_predictions.npz" if path.is_dir() else path
    provenance_path = artifact.with_name("evidence.json")
    if not provenance_path.exists():
        raise ValueError("saved finer artifact requires adjacent evidence.json provenance")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("source_model_sha256") != expected_model_sha or provenance.get("evaluation_model_sha256") != expected_model_sha:
        raise ValueError("saved finer source model hash mismatch")
    if (float(provenance.get("dt_old", -1.0)) != 0.005
            or float(provenance.get("dt_new", -1.0)) != 0.0025
            or float(provenance.get("output_dt", -1.0)) != 0.01):
        raise ValueError("saved finer artifact has unexpected integration/output grid")
    if not str(provenance.get("initial_source", "")).startswith("evaluation fine_predictions[:,0]"):
        raise ValueError("saved finer artifact has unexpected initial-state provenance")
    with np.load(artifact, allow_pickle=False) as data:
        finer_ids = [str(v) for v in data["event_ids"]]
        predictions = np.asarray(data["predictions"], dtype=float)
    if finer_ids != ids or predictions.shape[0] != len(ids):
        raise ValueError("saved finer predictions do not exactly cover cached IDs")
    return predictions, {
        "provided": True,
        "used": True,
        "status": "saved_sampled_output_only",
        "dt_old": 0.005,
        "dt_new": 0.0025,
        "output_dt": 0.01,
        "global_failure_step": provenance.get("nonfinite_failure_step"),
        "full_substep_per_event_first_bad": False,
    }


def _hash_audit(evidence: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    expected = {
        "model_sha256": digest(model_dir / "model.pt"),
        "normalization_sha256": digest(model_dir / "normalization.npz"),
        "base_sha256": digest(BASE),
        "network_sha256": digest(NETWORK),
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ValueError(f"cached evaluation {key} mismatch")
    return expected


def _cache_ids_and_sources(evidence: dict[str, Any], ids: list[str], metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eligible = sorted(metadata)
    if sorted(ids) != eligible or len(ids) != 107:
        raise ValueError(f"cached IDs do not exactly cover eligible validation IDs: cache={len(ids)} eligible={len(eligible)}")
    identities = evidence.get("identities", {})
    source_hashes = {}
    for root in ROOTS:
        key = str(root)
        record = identities.get(key) or identities.get(str(root.relative_to(ROOT)))
        if record is None:
            raise ValueError(f"missing cached source identity for {key}")
        actual = {name: digest(root / name) for name in ("dataset_index.csv", "splits.json")}
        if record != actual:
            raise ValueError(f"cached source identity mismatch for {key}")
        source_hashes[key] = actual
    return {"cached_event_count": len(ids), "eligible_validation_count": len(eligible), "source_hashes": source_hashes}


def _with_observed(history: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Prepend observed t=0..90 ms; forecast[0] is the 100-ms decision state."""
    history = np.asarray(history, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if history.ndim != 2 or history.shape[1] != 30:
        raise ValueError("causal history must contain 30 physical states")
    if forecast.ndim != 2 or forecast.shape[1] not in SUPPORTED_STATE_DIMS:
        raise ValueError("forecast must contain exactly 33 or 34 states")
    q_pad = forecast.shape[1] - history.shape[1]
    observed = np.pad(history[:10], ((0, 0), (0, q_pad)))
    return np.concatenate((observed, forecast), axis=0)


def _numerical_pair(
    first01: np.ndarray,
    second01: np.ndarray,
    metadata: dict[str, Any],
    mass: np.ndarray,
    *,
    first_substeps: np.ndarray | None = None,
) -> tuple[dict[str, Any], bool]:
    """Compare two 10-ms output grids, optionally auditing a 5-ms source grid."""
    if first01.ndim != 2 or second01.ndim != 2 or first01.shape != second01.shape:
        raise ValueError("comparison grids must have identical [time, 33-or-34] shapes")
    state_dim = first01.shape[1]
    if state_dim not in SUPPORTED_STATE_DIMS:
        raise ValueError("comparison grids must contain exactly 33 or 34 states")
    if first_substeps is not None and (first_substeps.ndim != 2 or first_substeps.shape[1] != state_dim):
        raise ValueError("fine substeps must use the same state width as the comparison grid")
    full_first = _with_observed(metadata["history"], first01)
    full_second = _with_observed(metadata["history"], second01)
    raw_first = restore_physical_path(full_first, metadata["equilibrium_delta"], metadata["omega_s"])
    raw_second = restore_physical_path(full_second, metadata["equilibrium_delta"], metadata["omega_s"])
    hit_f = trigger_index(raw_first, mass)
    hit_c = trigger_index(raw_second, mass)
    both = hit_f is not None and hit_c is not None
    endpoint = max(hit_f, hit_c) if both else len(full_first) - 1
    endpoint = max(0, min(endpoint, len(full_first) - 1, len(full_second) - 1))
    first_sub = _with_observed(metadata["history"], first_substeps) if first_substeps is not None else full_first
    # A full-fidelity 5-ms path has two forecast substeps per 10-ms interval.
    if first_substeps is not None:
        sub_end = max(0, min(first_sub.shape[0] - 1, 10 + 2 * max(endpoint - 10, 0)))
        finite_first_to_endpoint = bool(np.isfinite(first_sub[: sub_end + 1, :state_dim]).all())
    else:
        finite_first_to_endpoint = bool(np.isfinite(full_first[: endpoint + 1, :state_dim]).all())
    finite_second_to_endpoint = bool(np.isfinite(full_second[: endpoint + 1, :state_dim]).all())
    finite = bool(np.isfinite(first_sub[:, :state_dim]).all() and np.isfinite(full_second[:, :state_dim]).all())
    finite_to_endpoint = bool(finite_first_to_endpoint and finite_second_to_endpoint)
    # Observed history establishes the LOS hold counter, not additional
    # zero-error samples in the forecast convergence denominator.
    forecast_endpoint = max(0, endpoint - 10)
    diff = first01[: forecast_endpoint + 1, :20] - second01[: forecast_endpoint + 1, :20]
    de = coi_project(diff[:, :10], mass)
    we = coi_project(diff[:, 10:20], mass)
    discrepancy = float(np.sqrt(0.5 * np.mean((de / SCALES[0]) ** 2 + (we / SCALES[1]) ** 2))) if len(diff) else None
    hit_same = (hit_f is None) == (hit_c is None)
    hit_diff = None if not both else abs(hit_f - hit_c) * DT
    passed = bool(finite_to_endpoint and discrepancy is not None and discrepancy <= 0.001 and hit_same and (hit_diff is None or hit_diff <= 0.02))
    return {
        "state_dim": state_dim,
        "finite_all_cached_state": finite,
        # Preserve the established r3 field while making r4 state coverage explicit.
        "finite_all_cached_33": finite if state_dim == 33 else None,
        "finite_to_comparison_endpoint": finite_to_endpoint,
        "comparison_endpoint_index": endpoint,
        "comparison_endpoint_time_s": float(endpoint) * DT,
        "fine_hit_index": hit_f,
        "coarse_hit_index": hit_c,
        "same_hit_or_nonhit": hit_same,
        "hit_time_difference_s": hit_diff,
        "normalized_coi_discrepancy": discrepancy,
        "screen_pass": passed,
    }, passed


def _numerical_rows(fine: np.ndarray, coarse: np.ndarray, metadata: dict[str, Any], mass: np.ndarray) -> tuple[dict[str, Any], bool]:
    """Original cached .005-vs-.01 pair; retain all .005 substeps for finiteness."""
    if fine.ndim != 2 or coarse.ndim != 2 or fine.shape[1] != coarse.shape[1] or fine.shape[1] not in SUPPORTED_STATE_DIMS:
        raise ValueError("fine/coarse cached predictions must share [time, 33] or [time, 34] states")
    fine01 = fine[::2]
    if fine01.shape[0] != coarse.shape[0]:
        raise ValueError("fine[::2] and coarse cached grids are not aligned")
    return _numerical_pair(fine01, coarse, metadata, mass, first_substeps=fine)


def evaluate(model_dir: Path, evaluation: Path, output: Path, *, finer: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output must be a fresh directory")
    evidence, ids, fine, coarse, truths = _load_cached(evaluation)
    hashes = _hash_audit(evidence, model_dir)
    import torch
    checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    if "c" not in checkpoint or getattr(checkpoint["c"], "ndim", 0) != 2:
        raise ValueError("checkpoint must expose rank-bearing c[*, rank]")
    checkpoint_rank = int(checkpoint["c"].shape[1])
    state_dim = 30 + checkpoint_rank
    if state_dim not in SUPPORTED_STATE_DIMS:
        raise ValueError("checkpoint rank must produce exactly 33 or 34 states")
    if fine.shape[-1] != state_dim or coarse.shape[-1] != state_dim:
        raise ValueError(f"cached prediction width does not match checkpoint: expected {state_dim}")
    finer_predictions, finer_note = _load_finer(finer, ids, hashes["model_sha256"])
    if finer_predictions is not None and (finer_predictions.ndim != 3 or finer_predictions.shape[-1] != state_dim):
        raise ValueError(f"saved finer prediction width does not match checkpoint: expected {state_dim}")
    # base model is loaded only for its declared mechanical equilibrium vector.
    with np.load(BASE, allow_pickle=False) as base_data:
        base_pm_star = np.asarray(base_data["pm_star"], dtype=float)
    metadata, latency = _read_validation_starts(base_pm_star)
    if len(metadata) + len(latency) != 117 or len(latency) != 10:
        raise ValueError(
            "raw validation population must be 117 with exactly 10 latency-uncovered events"
        )
    id_audit = _cache_ids_and_sources(evidence, ids, metadata)
    rows = []
    numerical = []
    common = []
    for i, event_id in enumerate(ids):
        meta = metadata[event_id]
        original_reference = fine[i, ::2]
        reference = original_reference if finer_predictions is None else finer_predictions[i]
        observed = restore_physical_path(meta["history"], meta["equilibrium_delta"], meta["omega_s"])
        original_raw = restore_physical_path(original_reference, meta["equilibrium_delta"], meta["omega_s"])
        original_full_raw = np.concatenate((observed[:10, :20], original_raw[:, :20]), axis=0)
        original_classification = classify_prediction(original_full_raw, meta["inertia"], true_event_time_s=meta["event_time_s"])
        forecast_raw = restore_physical_path(reference, meta["equilibrium_delta"], meta["omega_s"])
        full_raw = np.concatenate((observed[:10, :20], forecast_raw[:, :20]), axis=0)
        classification = classify_prediction(full_raw, meta["inertia"], true_event_time_s=meta["event_time_s"])
        nrow, original_pair_pass = _numerical_rows(fine[i], coarse[i], meta, meta["inertia"])
        nrow["event_id"] = event_id
        nrow["pair"] = "original_005_vs_010"
        finer_pair = None
        finer_pair_pass = True
        finer_classification = None
        if finer_predictions is not None:
            finer_pair, finer_pair_pass = _numerical_pair(
                original_reference, finer_predictions[i], meta, meta["inertia"],
                first_substeps=fine[i],
            )
            if finer_note.get("global_failure_step") is not None:
                finer_pair_pass = False
                finer_pair["screen_pass"] = False
                finer_pair["global_failure_step"] = finer_note["global_failure_step"]
            finer_pair["event_id"] = event_id
            finer_pair["pair"] = "saved_0025_vs_original_005_sampled_at_010"
            finer_classification = classification
            # The saved finer artifact is selected only after its own pair
            # screen; it is never substituted blindly.
            if not finer_pair_pass and classification["category"] != "latency_uncovered":
                classification = apply_numerical_gate(classification, False)
        selected_numerical_pass = bool(finer_pair_pass if finer_predictions is not None else original_pair_pass)
        classification = apply_numerical_gate(classification, selected_numerical_pass)
        times = trigger_times(classification["trigger_index"], clearing_time_s=meta["clearing_time_s"], record_start_delay_s=meta["record_start_delay_s"])
        delay = 0.0 if meta["record_start_delay_s"] is None else float(meta["record_start_delay_s"])
        predicted_since_clear = None if times["predicted_los_time_record_s"] is None else float(times["predicted_los_time_record_s"]) + delay
        true_since_clear = None if meta["event_time_s"] is None else float(meta["event_time_s"]) + delay
        rows.append({"event_id": event_id, "label": meta["label"], "group": meta["group"], "true_event_time_s": meta["event_time_s"], "true_event_time_since_clearing_s": true_since_clear, "clearing_time_s": meta["clearing_time_s"], "record_start_delay_s": meta["record_start_delay_s"], "predicted_los_time_since_clearing_s": predicted_since_clear, "trigger_time_error_s": None if predicted_since_clear is None or true_since_clear is None else predicted_since_clear - true_since_clear, "selected_forecast_source": "saved_finer" if finer_predictions is not None and finer_pair_pass else "original_fine", "original_pair_category": original_classification["category"], "original_pair_trigger_index": original_classification["trigger_index"], "finer_pair_category": None if finer_classification is None else finer_classification["category"], "finer_pair_trigger_index": None if finer_classification is None else finer_classification["trigger_index"], **classification, **times})
        numerical.append(nrow)
        if finer_pair is not None:
            numerical.append(finer_pair)
        common.extend({"event_id": event_id, "label": meta["label"], **r} for r in common_frequency_metrics(reference, truths[i], meta["inertia"]))
    counts = {category: sum(row["category"] == category for row in rows) for category in ("predicted_los", "window_settled", "window_no_los_unsettled", "numerical_unresolved", "latency_uncovered")}
    counts["latency_uncovered"] = len(latency)
    eligible_rows = [row for row in rows if row["category"] != "latency_uncovered"]
    false_window_settled = sum(row["category"] == "window_settled" and row["label"] == "unstable" for row in eligible_rows)
    false_los = sum(row["predicted_los"] and row["label"] == "stable" for row in eligible_rows)
    unstable = [row for row in rows if row["label"] == "unstable"]
    eligible_unstable = [row for row in unstable if row["category"] != "latency_uncovered"]
    latency_unstable = sum(row["label"] == "unstable" for row in latency)
    raw_unstable_total = len(unstable) + latency_unstable
    missed = sum(not row["predicted_los"] for row in eligible_unstable)
    numerical_fail = any(not row["screen_pass"] for row in numerical)
    existing_scored_rows = [
        row for row in evidence.get("numerical", [])
        if row.get("horizon") == "prefix" or bool(row.get("complete"))
    ]
    existing_rows_fail = _existing_scored_rows_fail(existing_scored_rows)
    conditional_refinement = bool(numerical_fail or existing_rows_fail)
    group_counts = {}
    for row in rows:
        group = str(row["group"])
        cell = group_counts.setdefault(group, {category: 0 for category in counts})
        cell[row["category"]] += 1
    for row in latency:
        group = str(row["group"])
        cell = group_counts.setdefault(group, {category: 0 for category in counts})
        cell["latency_uncovered"] += 1
    raw_unstable_detected = sum(row["predicted_los"] for row in eligible_unstable)
    result = {
        "model_dir": str(model_dir), "evaluation": str(evaluation), "source_hashes": hashes,
        "cache_identity": id_audit, "event_count_raw_validation": len(ids) + len(latency), "eligible_event_count": len(ids), "latency_uncovered_count": len(latency),
        "latency_uncovered_rows": [
            {
                "event_id": row["event_id"], "label": row["label"], "group": row["group"],
                "event_time_s": row["event_time_s"], "trajectory_end_s": row["trajectory_end_s"],
                "clearing_time_s": row["clearing_time_s"], "record_start_delay_s": row["record_start_delay_s"],
            }
            for row in latency
        ],
        "categories": counts, "group_counts": group_counts, "false_window_settled": false_window_settled, "false_los": false_los,
        "unstable_total": raw_unstable_total, "unstable_eligible_denominator": len(eligible_unstable), "latency_unstable_count": latency_unstable, "missed_true_unstable": missed,
        "raw_unstable_detected": raw_unstable_detected,
        "detection_rate_eligible": None if not eligible_unstable else float(raw_unstable_detected / len(eligible_unstable)),
        "detection_rate_raw": None if not raw_unstable_total else float(raw_unstable_detected / raw_unstable_total),
        "rows": rows, "numerical": numerical, "common_frequency": common,
        "timing": {"status": "unavailable; separate same-hardware protocol required"},
        "conditional_0025_refinement": {"triggered": conditional_refinement, "reason": "numerical comparison or existing genuine prefix/complete discrepancy > 1e-3 or finite-row failure", "existing_scored_rows_fail": existing_rows_fail, "run": False, "finer": finer_note},
        "test_read": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_text(json.dumps(_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finer", type=Path, default=None, help="optional saved 2.5-ms artifact; never runs refinement")
    args = parser.parse_args()
    result = evaluate(args.model_dir, args.evaluation, args.output, finer=args.finer)
    print(json.dumps({"categories": result["categories"], "eligible": result["eligible_event_count"], "latency_uncovered": result["latency_uncovered_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
