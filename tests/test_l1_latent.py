"""Synthetic contracts for the L1 cached-forecast evaluator."""
from __future__ import annotations

import numpy as np

from experiments.evaluate_l1_latent import (
    _load_cached,
    _existing_scored_rows_fail,
    _numerical_rows,
    apply_numerical_gate,
    classify_prediction,
    common_frequency_metrics,
    restore_physical_path,
    trigger_index,
    trigger_times,
    _with_observed,
)


def _scored_row(**updates):
    row = {"finite_fine": True, "finite_coarse": True, "step_rmse": 0.0}
    row.update(updates)
    return row


def test_existing_scored_rows_require_nonempty_explicit_finite_fields_and_rmse():
    assert _existing_scored_rows_fail([]) is True
    assert _existing_scored_rows_fail([_scored_row(step_rmse=None)]) is True
    assert _existing_scored_rows_fail([_scored_row(step_rmse=float("nan"))]) is True
    assert _existing_scored_rows_fail([{"finite_fine": True, "finite_coarse": True}]) is True
    assert _existing_scored_rows_fail([_scored_row(finite_fine=None)]) is True


def test_existing_scored_rows_accept_only_finite_nonnegative_rmse_at_gate():
    assert _existing_scored_rows_fail([_scored_row(step_rmse=0.0)]) is False
    assert _existing_scored_rows_fail([_scored_row(step_rmse=1e-3)]) is False
    assert _existing_scored_rows_fail([_scored_row(step_rmse=-1e-12)]) is True
    assert _existing_scored_rows_fail([_scored_row(step_rmse=1e-3 + 1e-12)]) is True
    assert _existing_scored_rows_fail([_scored_row(finite_coarse=False)]) is True


def _path(n: int, *, angle: float = 0.0, speed: float = 0.0) -> np.ndarray:
    x = np.zeros((n, 20), dtype=float)
    x[:, 0] = angle
    x[:, 10] = speed
    return x


def _los_path(n: int = 5, *, angle: float = np.pi + 1e-6, speed: float = 0.02 + 1e-6) -> np.ndarray:
    x = _path(n)
    x[:, 0] = angle
    x[:, 10] = speed
    x[:, 11] = -speed
    return x


def test_equilibrium_angle_is_restored_before_ptp():
    eq = np.zeros(10)
    eq[1] = np.pi + 0.1
    sync = np.zeros((5, 20))
    sync[:, 10] = 0.021 * 2.0 * np.pi * 60.0
    sync[:, 11] = -0.021 * 2.0 * np.pi * 60.0
    physical = restore_physical_path(sync, eq)
    assert np.isclose(np.ptp(physical[0, :10]), np.pi + 0.1)
    assert trigger_index(physical, np.ones(10)) == 4


def test_los_requires_strict_thresholds_and_five_samples():
    exact = _path(5, angle=np.pi, speed=0.02)
    exact[:, 11] = -0.02
    assert trigger_index(exact, np.ones(10)) is None
    assert trigger_index(_los_path(4), np.ones(10)) is None
    assert trigger_index(_los_path(5), np.ones(10)) == 4


def test_crossing_straddles_observation_forecast_boundary():
    path = _path(5)
    path[:4, 0] = np.pi + 1e-6
    path[:4, 10] = 0.021
    path[:4, 11] = -0.021
    path[4:] = _los_path(1)
    result = classify_prediction(path, np.ones(10), true_event_time_s=1.0)
    assert result["category"] == "predicted_los"
    assert result["trigger_index"] == 4


def test_rank4_observed_padding_and_los_crossing_preserve_physical_path():
    history = np.zeros((11, 30))
    forecast = np.zeros((5, 34))
    forecast[:4, 0] = np.pi + 1e-6
    forecast[:4, 10] = 0.021
    forecast[:4, 11] = -0.021
    forecast[4] = _los_path(1)[0].tolist() + [0.0] * 14
    full = _with_observed(history, forecast)
    assert full.shape == (15, 34)
    result = classify_prediction(full, np.ones(10), true_event_time_s=1.0)
    assert result["category"] == "predicted_los"
    assert result["trigger_index"] == 14


def test_latency_and_time_alignment_include_record_delay():
    result = classify_prediction(_los_path(), np.ones(10), true_event_time_s=0.1)
    assert result["category"] == "latency_uncovered"
    times = trigger_times(11, clearing_time_s=1.3, record_start_delay_s=0.002)
    assert np.isclose(times["predicted_los_time_record_s"], 0.11)
    assert np.isclose(times["predicted_los_time_decision_s"], 0.01)
    assert np.isclose(times["predicted_los_time_absolute_s"], 1.412)


def test_partial_prefix_is_separate_from_complete_horizons():
    pred = np.zeros((801, 20))
    truth = np.zeros((6, 20))
    rows = common_frequency_metrics(pred, truth, np.ones(10))
    one = next(row for row in rows if row["horizon"] == "1")
    prefix = next(row for row in rows if row["horizon"] == "prefix")
    assert one["complete"] is False and one["n"] == 0
    assert prefix["complete"] is False and prefix["n"] == 5


def test_finite_before_hit_allows_los_even_if_later_nonfinite():
    path = np.vstack((_los_path(5), _path(1)))
    path[-1, 0] = np.nan
    result = classify_prediction(path, np.ones(10), true_event_time_s=1.0)
    assert result["category"] == "predicted_los"
    assert result["finite_before_hit"] is True


def test_nonhit_short_window_cannot_be_called_settled():
    result = classify_prediction(_path(50), np.ones(10), true_event_time_s=1.0)
    assert result["category"] == "window_no_los_unsettled"
    assert result["window_settled"] is False


def test_nonfinite_before_hit_is_numerically_unresolved():
    path = _path(200)
    path[20, 0] = np.nan
    result = classify_prediction(path, np.ones(10), true_event_time_s=1.0)
    assert result["category"] == "numerical_unresolved"


def test_numerical_gate_blocks_physical_classification():
    classified = classify_prediction(_los_path(), np.ones(10), true_event_time_s=1.0)
    gated = apply_numerical_gate(classified, False)
    assert classified["category"] == "predicted_los"
    assert gated["category"] == "numerical_unresolved"
    assert gated["predicted_los"] is False


def test_fine_substep_33rd_state_nan_fails_to_full_comparison_endpoint():
    metadata = {
        "history": np.zeros((11, 30)),
        "equilibrium_delta": np.zeros(10),
        "omega_s": 2.0 * np.pi * 60.0,
    }
    fine = np.zeros((11, 33))
    coarse = np.zeros((6, 33))
    fine[4, 32] = np.nan
    row, passed = _numerical_rows(fine, coarse, metadata, np.ones(10))
    assert row["finite_to_comparison_endpoint"] is False
    assert passed is False


def test_fine_substep_q4_nan_fails_to_full_comparison_endpoint():
    metadata = {
        "history": np.zeros((11, 30)),
        "equilibrium_delta": np.zeros(10),
        "omega_s": 2.0 * np.pi * 60.0,
    }
    fine = np.zeros((11, 34))
    coarse = np.zeros((6, 34))
    fine[4, 33] = np.nan
    row, passed = _numerical_rows(fine, coarse, metadata, np.ones(10))
    assert row["state_dim"] == 34
    assert row["finite_all_cached_state"] is False
    assert row["finite_to_comparison_endpoint"] is False
    assert passed is False


def test_rank4_fine_coarse_width_mismatch_is_rejected():
    metadata = {
        "history": np.zeros((11, 30)),
        "equilibrium_delta": np.zeros(10),
        "omega_s": 2.0 * np.pi * 60.0,
    }
    try:
        _numerical_rows(np.zeros((11, 34)), np.zeros((6, 33)), metadata, np.ones(10))
    except ValueError as exc:
        assert "33" in str(exc) or "34" in str(exc)
    else:
        raise AssertionError("expected rank4 width mismatch")


def test_fine_coarse_grid_shape_mismatch_is_rejected():
    metadata = {
        "history": np.zeros((11, 30)),
        "equilibrium_delta": np.zeros(10),
        "omega_s": 2.0 * np.pi * 60.0,
    }
    try:
        _numerical_rows(np.zeros((11, 33)), np.zeros((5, 33)), metadata, np.ones(10))
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("expected fine/coarse grid mismatch")


def test_cached_truths_remain_ragged(tmp_path):
    evidence = {"event_count": 2}
    (tmp_path / "evidence.json").write_text(__import__("json").dumps(evidence), encoding="utf-8")
    np.savez_compressed(
        tmp_path / "trajectories.npz",
        event_ids=np.array(["a", "b"]),
        fine_predictions=np.zeros((2, 3, 33)),
        coarse_predictions=np.zeros((2, 3, 33)),
        a_truth=np.zeros((3, 30)),
        b_truth=np.zeros((2, 30)),
    )
    _, ids, _, _, truths = _load_cached(tmp_path)
    assert ids == ["a", "b"]
    assert [truth.shape for truth in truths] == [(3, 30), (2, 30)]
