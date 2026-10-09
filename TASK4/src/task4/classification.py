"""Pure-NumPy port of KIKI's frozen L1 classification and numerical screen.

Read-only sources (2026-09-30):
  ../kiki/src/kiki/data/private_dataset.py:los_index
  ../kiki/experiments/evaluate_l1_latent.py:restore_physical_path,
    window_settled, classify_prediction, apply_numerical_gate, trigger_times,
    _with_observed, _numerical_pair and evaluate's category/denominator summary.

TASK4 differences: physical arrays are [time, machine, (delta, omega, p)]
with arbitrary machine count; canonical delta is relative to each machine's
equilibrium and omega is in rad/s. There are no KIKI imports or I/O. Latent
finiteness is supplied as a separate per-time boolean mask, so any latent
rank is supported. A short genuine truth prefix never shortens the default
8-s *post-decision* classification window. Short predictions without a
reliable hit are unresolved. An explicit numerical gate can invalidate all
eligible physical labels. A settled window is not a stability certificate.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

DT = 0.01
DECISION_TIME_S = 0.10
DECISION_HORIZON_S = 8.0
DELTA_SCALE = 0.47675314588683515
OMEGA_SCALE = 1.0033667249602924
LOS_CONFIG = {
    "los_angle_rad": np.pi,
    "los_relative_speed_pu": 0.02,
    "los_hold_samples": 5,
    "stable_tail_seconds": 1.0,
    "stable_tail_speed_pu": 0.005,
    "sample_interval_s": DT,
}
CATEGORIES = (
    "predicted_los", "window_settled", "window_no_los_unsettled",
    "numerical_unresolved", "latency_uncovered",
)


def _config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(LOS_CONFIG)
    if config is not None:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown LOS configuration keys: {sorted(unknown)}")
        cfg.update(config)
    for name, value in cfg.items():
        if not np.isfinite(value) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if int(cfg["los_hold_samples"]) != cfg["los_hold_samples"]:
        raise ValueError("los_hold_samples must be an integer")
    return cfg


def _path(path: np.ndarray, name: str = "path") -> np.ndarray:
    result = np.asarray(path, dtype=float)
    if result.ndim != 3 or result.shape[1] < 1 or result.shape[2] != 3:
        raise ValueError(f"{name} must have shape [time, machine, 3]")
    return result


def _mass(inertia: np.ndarray, machines: int) -> np.ndarray:
    mass = np.asarray(inertia, dtype=float)
    if mass.shape != (machines,) or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("inertia must be a finite positive vector with one value per machine")
    return mass


def _grid_steps(seconds: float, dt: float, name: str) -> int:
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    steps = round(float(seconds) / dt)
    if not np.isclose(steps * dt, seconds, atol=1e-10, rtol=0):
        raise ValueError(f"{name} must be on the output sample grid")
    return steps


def _latent_mask(mask: np.ndarray | None, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    values = np.asarray(mask)
    if values.shape != (length,) or values.dtype.kind != "b":
        raise ValueError("latent_finite must be a boolean [time] mask; reduce all latent axes first")
    return values


def _finite_rows(path: np.ndarray, latent_finite: np.ndarray | None) -> np.ndarray:
    return np.isfinite(path).all(axis=(1, 2)) & _latent_mask(latent_finite, len(path))


def _prefix_length(finite: np.ndarray) -> int:
    bad = np.flatnonzero(~finite)
    return int(bad[0]) if len(bad) else len(finite)


def coi(values: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """Subtract the inertia-weighted common component on the last axis."""
    values = np.asarray(values, dtype=float)
    mass = _mass(inertia, values.shape[-1])
    return values - np.sum(values * mass, axis=-1, keepdims=True) / mass.sum()


def restore_physical_path(
    physical: np.ndarray, delta_star: np.ndarray, omega_s: float
) -> np.ndarray:
    """Restore raw angle/rad and absolute speed/pu; p is copied unchanged.

    Adding the individual equilibrium angles is essential before taking the
    angle spread. A common equilibrium shift is harmless, individual shifts
    are not. p does not enter the frozen LOS/tail criteria.
    """
    path = _path(physical, "physical")
    equilibrium = np.asarray(delta_star, dtype=float)
    if equilibrium.shape != (path.shape[1],) or not np.isfinite(equilibrium).all():
        raise ValueError("delta_star must be a finite vector with one value per machine")
    if not np.isfinite(omega_s) or omega_s <= 0:
        raise ValueError("omega_s must be finite and positive")
    raw = path.copy()
    raw[..., 0] += equilibrium[None, :]
    raw[..., 1] = raw[..., 1] / float(omega_s) + 1.0
    return raw


def trigger_index(
    raw_path: np.ndarray, inertia: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> int | None:
    """First endpoint of five sustained strict angle-AND-speed exceedances."""
    cfg = _config(config)
    raw = _path(raw_path, "raw_path")
    mass = _mass(inertia, raw.shape[1])
    # No hit after the first invalid physical state can count as evidence.
    raw = raw[:_prefix_length(np.isfinite(raw).all(axis=(1, 2)))]
    hold = int(cfg["los_hold_samples"])
    if len(raw) < hold:
        return None
    angle = np.ptp(raw[..., 0], axis=1)
    speed = np.max(np.abs(coi(raw[..., 1], mass)), axis=1)
    exceeds = (angle > cfg["los_angle_rad"]) & (speed > cfg["los_relative_speed_pu"])
    hits = np.flatnonzero(
        np.convolve(exceeds.astype(int), np.ones(hold, dtype=int), mode="valid") == hold
    )
    return int(hits[0] + hold - 1) if len(hits) else None


def window_settled(
    raw_path: np.ndarray, inertia: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Frozen last-1-s angle/speed screen; this is not a stability claim."""
    cfg = _config(config)
    raw = _path(raw_path, "raw_path")
    mass = _mass(inertia, raw.shape[1])
    tail = max(1, round(cfg["stable_tail_seconds"] / cfg["sample_interval_s"]))
    if len(raw) < tail:
        return False
    last = raw[-tail:]
    if not np.isfinite(last).all():
        return False
    speed = np.max(np.abs(coi(last[..., 1], mass)), axis=1)
    return bool(np.all(np.ptp(last[..., 0], axis=1) < cfg["los_angle_rad"])
                and np.all(speed < cfg["stable_tail_speed_pu"]))


def with_observed(
    history: np.ndarray, forecast: np.ndarray, *,
    sample_interval_s: float = DT, decision_time_s: float = DECISION_TIME_S,
) -> np.ndarray:
    """Prepend t=0..90 ms; forecast[0] supplies the t=100-ms state."""
    observed = _path(history, "history")
    predicted = _path(forecast, "forecast")
    if not np.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be finite and positive")
    index = _grid_steps(decision_time_s, sample_interval_s, "decision_time_s")
    if len(observed) < index + 1 or len(predicted) < 1:
        raise ValueError("history must reach the decision state and forecast must include it")
    if observed.shape[1:] != predicted.shape[1:]:
        raise ValueError("history and forecast must have the same machine shape")
    return np.concatenate((observed[:index], predicted), axis=0)


def classify_prediction(
    raw_path: np.ndarray, inertia: np.ndarray, *,
    true_event_time_s: float | None = None,
    latent_finite: np.ndarray | None = None,
    numerical_passed: bool | None = None,
    config: Mapping[str, Any] | None = None,
    decision_time_s: float = DECISION_TIME_S,
    decision_horizon_s: float = DECISION_HORIZON_S,
) -> dict[str, Any]:
    """Classify an observed+forecast raw path over the whole decision window.

    ``latent_finite`` covers the same full grid as raw_path. ``None`` means
    that latent finiteness was not supplied, and is recorded as such. With
    numerical_passed=None the physical category is provisional until an
    explicit apply_numerical_gate call; False always invalidates an eligible
    physical label. Events at/before 100 ms remain latency-uncovered.
    """
    cfg = _config(config)
    raw = _path(raw_path, "raw_path")
    _mass(inertia, raw.shape[1])
    dt = float(cfg["sample_interval_s"])
    decision_index = _grid_steps(decision_time_s, dt, "decision_time_s")
    horizon_steps = _grid_steps(decision_horizon_s, dt, "decision_horizon_s")
    if horizon_steps < 1:
        raise ValueError("decision_horizon_s must contain a forecast interval")
    required = decision_index + horizon_steps + 1
    finite = _finite_rows(raw, latent_finite)[:required]
    raw = raw[:required]
    base = {
        "predicted_los": False, "window_settled": False,
        "trigger_index": None, "first_nonfinite_index": None,
        "finite_before_hit": None, "prediction_complete": len(raw) == required,
        "required_samples": required, "available_samples": len(raw),
        "decision_horizon_s": float(decision_horizon_s),
        "latent_finiteness_checked": latent_finite is not None,
        "numerical_gate_passed": None,
    }
    if true_event_time_s is not None:
        if not np.isfinite(true_event_time_s) or true_event_time_s < 0:
            raise ValueError("true_event_time_s must be a nonnegative record-relative time")
        if true_event_time_s <= decision_time_s + 1e-9:
            return {**base, "category": "latency_uncovered"}
    stop = _prefix_length(finite)
    first_bad = stop if stop < len(raw) else None
    hit = trigger_index(raw[:stop], inertia, cfg)
    base["first_nonfinite_index"] = first_bad
    if hit is not None:
        result = {**base, "category": "predicted_los", "predicted_los": True,
                  "trigger_index": hit, "finite_before_hit": True}
    elif first_bad is not None or len(raw) < required:
        result = {**base, "category": "numerical_unresolved", "finite_before_hit": False,
                  "unresolved_reason": "nonfinite_state" if first_bad is not None else "incomplete_prediction"}
    else:
        settled = window_settled(raw, inertia, cfg)
        result = {**base, "category": "window_settled" if settled else "window_no_los_unsettled",
                  "window_settled": settled, "finite_before_hit": True}
    return result if numerical_passed is None else apply_numerical_gate(result, numerical_passed)


def classify_forecast(
    history: np.ndarray, forecast: np.ndarray, inertia: np.ndarray,
    delta_star: np.ndarray, omega_s: float, *,
    latent_finite: np.ndarray | None = None, **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper for canonical physical forecasts and latent masks.

    The latent mask covers forecast only, including its initial state. No
    latent state is needed for the ten observations before initialization.
    """
    cfg = _config(kwargs.get("config"))
    decision_time = kwargs.get("decision_time_s", DECISION_TIME_S)
    before = _grid_steps(decision_time, cfg["sample_interval_s"], "decision_time_s")
    predicted = _path(forecast, "forecast")
    mask = None if latent_finite is None else np.concatenate(
        (np.ones(before, dtype=bool), _latent_mask(latent_finite, len(predicted))))
    full = with_observed(history, predicted, sample_interval_s=cfg["sample_interval_s"],
                         decision_time_s=decision_time)
    raw = restore_physical_path(full, delta_star, omega_s)
    return classify_prediction(raw, inertia, latent_finite=mask, **kwargs)


def apply_numerical_gate(classification: Mapping[str, Any], passed: bool) -> dict[str, Any]:
    """An explicit failed numerical screen dominates eligible physical labels."""
    if not isinstance(passed, (bool, np.bool_)):
        raise ValueError("numerical gate must be an explicit boolean")
    result = {**classification, "numerical_gate_passed": bool(passed)}
    if passed or classification.get("category") == "latency_uncovered":
        return result
    return {**result, "category": "numerical_unresolved", "predicted_los": False,
            "window_settled": False, "numerical_pair_failed": True,
            "unresolved_reason": "numerical_pair_failed",
            "unreliable_trigger_index": classification.get("trigger_index"), "trigger_index": None}


def trigger_times(
    index: int | None, *, clearing_time_s: float,
    record_start_delay_s: float | None = None, sample_interval_s: float = DT,
    decision_time_s: float = DECISION_TIME_S,
) -> dict[str, float | None]:
    """Record-relative, decision-relative and absolute ANDES trigger times."""
    names = ("predicted_los_time_record_s", "predicted_los_time_decision_s",
             "predicted_los_time_absolute_s")
    if index is None:
        return dict.fromkeys(names)
    if isinstance(index, (bool, np.bool_)) or int(index) != index or index < 0:
        raise ValueError("trigger index must be a nonnegative integer")
    delay = 0.0 if record_start_delay_s is None else float(record_start_delay_s)
    if not all(np.isfinite(x) for x in (clearing_time_s, delay, sample_interval_s, decision_time_s)):
        raise ValueError("trigger time offsets and sample interval must be finite")
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")
    record = int(index) * float(sample_interval_s)
    return dict(zip(names, (record, record - decision_time_s, float(clearing_time_s) + delay + record)))


def numerical_pair(
    first01: np.ndarray, second01: np.ndarray, history: np.ndarray,
    inertia: np.ndarray, delta_star: np.ndarray, omega_s: float, *,
    first_latent_finite: np.ndarray | None = None,
    second_latent_finite: np.ndarray | None = None,
    first_substeps: np.ndarray | None = None,
    first_substeps_latent_finite: np.ndarray | None = None,
    substep_ratio: int = 2,
    config: Mapping[str, Any] | None = None,
    decision_time_s: float = DECISION_TIME_S,
    decision_horizon_s: float = DECISION_HORIZON_S,
) -> tuple[dict[str, Any], bool]:
    """Original RMS/LOS agreement gate on two common 10-ms output grids.

    When both paths hit, compare up to the later hit (inclusive); otherwise
    the full 8-s decision horizon is mandatory. Observed history establishes
    the hold counter, but contributes no zero-error samples to RMS. The
    forecast initial state is included, matching the original denominator.
    Every p and supplied latent state is checked for finiteness although the
    RMS uses only inertia-relative delta and omega with the frozen scales.
    Optional first_substeps audit states between output samples; their grid
    starts at the decision state and ends at the same forecast endpoint.
    """
    cfg = _config(config)
    first, second = _path(first01, "first01"), _path(second01, "second01")
    if first.shape != second.shape or len(first) < 1:
        raise ValueError("comparison grids must have identical nonempty [time, machine, 3] shapes")
    mass = _mass(inertia, first.shape[1])
    dt = float(cfg["sample_interval_s"])
    before = _grid_steps(decision_time_s, dt, "decision_time_s")
    horizon_steps = _grid_steps(decision_horizon_s, dt, "decision_horizon_s")
    if horizon_steps < 1:
        raise ValueError("decision_horizon_s must contain a forecast interval")
    required_endpoint = before + horizon_steps
    full_first = with_observed(history, first, sample_interval_s=dt, decision_time_s=decision_time_s)
    full_second = with_observed(history, second, sample_interval_s=dt, decision_time_s=decision_time_s)
    first_mask = np.concatenate((np.ones(before, dtype=bool), _latent_mask(first_latent_finite, len(first))))
    second_mask = np.concatenate((np.ones(before, dtype=bool), _latent_mask(second_latent_finite, len(second))))
    finite_first = _finite_rows(full_first, first_mask)
    finite_second = _finite_rows(full_second, second_mask)
    raw_first = restore_physical_path(full_first, delta_star, omega_s)
    raw_second = restore_physical_path(full_second, delta_star, omega_s)
    first_stop = min(_prefix_length(finite_first), required_endpoint + 1)
    second_stop = min(_prefix_length(finite_second), required_endpoint + 1)
    hit_f = trigger_index(raw_first[:first_stop], mass, cfg)
    hit_c = trigger_index(raw_second[:second_stop], mass, cfg)
    both = hit_f is not None and hit_c is not None
    endpoint = max(hit_f, hit_c) if both else required_endpoint
    covered = len(full_first) > endpoint
    finite_first_endpoint = bool(covered and finite_first[:endpoint + 1].all())
    finite_second_endpoint = bool(covered and finite_second[:endpoint + 1].all())
    substep_all_finite = True
    if first_substeps is not None:
        if isinstance(substep_ratio, bool) or int(substep_ratio) != substep_ratio or substep_ratio < 1:
            raise ValueError("substep_ratio must be a positive integer")
        substeps = _path(first_substeps, "first_substeps")
        expected_length = (len(first) - 1) * int(substep_ratio) + 1
        if substeps.shape != (expected_length, *first.shape[1:]):
            raise ValueError("first_substeps must align exactly with the first output grid")
        subfinite = _finite_rows(substeps, first_substeps_latent_finite)
        sub_endpoint = max(0, endpoint - before) * int(substep_ratio)
        finite_first_endpoint = bool(finite_first_endpoint and len(substeps) > sub_endpoint
                                     and subfinite[:sub_endpoint + 1].all())
        substep_all_finite = bool(subfinite.all())
    elif first_substeps_latent_finite is not None:
        raise ValueError("first_substeps_latent_finite requires first_substeps")
    finite_to_endpoint = finite_first_endpoint and finite_second_endpoint
    forecast_endpoint = max(0, endpoint - before)
    # An uncovered full horizon cannot pass by using a short prefix's RMS.
    difference = first[:forecast_endpoint + 1, :, :2] - second[:forecast_endpoint + 1, :, :2]
    discrepancy = delta_rms = omega_rms = None
    if covered and len(difference) and np.isfinite(difference).all():
        de = coi(difference[..., 0], mass) / DELTA_SCALE
        we = coi(difference[..., 1], mass) / OMEGA_SCALE
        delta_rms = float(np.sqrt(np.mean(de ** 2)))
        omega_rms = float(np.sqrt(np.mean(we ** 2)))
        discrepancy = float(np.sqrt(0.5 * np.mean(de ** 2 + we ** 2)))
    same_hit = (hit_f is None) == (hit_c is None)
    hit_difference = abs(hit_f - hit_c) * dt if both else None
    passed = bool(finite_to_endpoint and discrepancy is not None and np.isfinite(discrepancy)
                  and discrepancy <= 1e-3 and same_hit
                  and (hit_difference is None or hit_difference <= 0.02 + 1e-12))
    row = {
        "machine_count": first.shape[1], "physical_state_width": 3 * first.shape[1],
        "finite_all_cached_state": bool(finite_first.all() and finite_second.all() and substep_all_finite),
        "latent_finiteness_checked": {
            "first": first_latent_finite is not None, "second": second_latent_finite is not None,
            "first_substeps": first_substeps_latent_finite is not None if first_substeps is not None else None,
        },
        "finite_to_comparison_endpoint": finite_to_endpoint,
        "comparison_endpoint_index": endpoint, "comparison_endpoint_time_s": endpoint * dt,
        "comparison_endpoint_decision_s": endpoint * dt - decision_time_s,
        "comparison_interval_covered": covered,
        "full_decision_horizon_covered": len(full_first) > required_endpoint,
        "fine_hit_index": hit_f, "coarse_hit_index": hit_c,
        "same_hit_or_nonhit": same_hit, "hit_time_difference_s": hit_difference,
        "normalized_coi_discrepancy": discrepancy,
        "normalized_relative_delta_rms": delta_rms,
        "normalized_relative_omega_rms": omega_rms,
        "screen_pass": passed,
    }
    return row, passed


def summarize_classifications(
    rows: Iterable[Mapping[str, Any]], latency_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Preserve raw, covered and latency denominators and all failure classes.

    ``rows`` contain the final gated categories, original stable/unstable
    labels and optional event_id/id, group and trigger timing fields.
    Metadata-only exclusions may instead be passed as ``latency_rows``.
    Nothing is filtered by numerical quality or by original event severity.
    """
    all_rows = [dict(row) for row in rows]
    all_rows.extend({**row, "category": "latency_uncovered", "predicted_los": False,
                     "window_settled": False, "trigger_index": None} for row in latency_rows)
    seen = set()
    for row in all_rows:
        if row.get("category") not in CATEGORIES:
            raise ValueError(f"Unknown classification category: {row.get('category')}")
        if row.get("label") not in ("stable", "unstable"):
            raise ValueError("Every denominator row must preserve its original stable/unstable label")
        identity = row.get("event_id", row.get("id"))
        if identity is not None:
            if identity in seen:
                raise ValueError(f"Duplicate classification event: {identity}")
            seen.add(identity)
        if bool(row.get("predicted_los", False)) != (row["category"] == "predicted_los"):
            raise ValueError("predicted_los flag disagrees with final category")
    counts = {name: sum(row["category"] == name for row in all_rows) for name in CATEGORIES}
    eligible = [row for row in all_rows if row["category"] != "latency_uncovered"]
    eligible_unstable = [row for row in eligible if row["label"] == "unstable"]
    raw_unstable = sum(row["label"] == "unstable" for row in all_rows)
    latency_unstable = sum(row["label"] == "unstable" and row["category"] == "latency_uncovered" for row in all_rows)
    detected = sum(row["category"] == "predicted_los" for row in eligible_unstable)
    group_counts: dict[str, dict[str, int]] = {}
    for row in all_rows:
        cell = group_counts.setdefault(str(row.get("group", "unknown")), dict.fromkeys(CATEGORIES, 0))
        cell[row["category"]] += 1
    time_keys = ("event_id", "id", "true_event_time_s", "trigger_index",
                 "predicted_los_time_record_s", "predicted_los_time_decision_s",
                 "predicted_los_time_absolute_s", "predicted_los_time_since_clearing_s",
                 "true_event_time_since_clearing_s", "trigger_time_error_s")
    return {
        "event_count_raw": len(all_rows), "eligible_event_count": len(eligible),
        "latency_uncovered_count": counts["latency_uncovered"],
        "categories": counts, "group_counts": group_counts,
        "unstable_total": raw_unstable, "unstable_eligible_denominator": len(eligible_unstable),
        "latency_unstable_count": latency_unstable, "raw_unstable_detected": detected,
        "missed_true_unstable": len(eligible_unstable) - detected,
        "missed_true_unstable_raw": raw_unstable - detected,
        "false_window_settled": sum(row["category"] == "window_settled" for row in eligible_unstable),
        "false_los": sum(row["label"] == "stable" and row["category"] == "predicted_los" for row in eligible),
        "undetermined_count": counts["window_no_los_unsettled"],
        "numerical_unresolved_count": counts["numerical_unresolved"],
        "detection_rate_eligible": detected / len(eligible_unstable) if eligible_unstable else None,
        "detection_rate_raw": detected / raw_unstable if raw_unstable else None,
        "trigger_times": [{key: row.get(key) for key in time_keys if key in row} for row in all_rows],
        "latency_uncovered_rows": [row for row in all_rows if row["category"] == "latency_uncovered"],
        "rows": all_rows,
    }
