"""Synthetic tests for equilibrium-anchored observer initialization (Loop0063)."""

import sys
import copy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tests"))

from audit_equilibrium_initialization import (  # noqa: E402
    aggregate_common,
    candidate_screen,
    common_omega_metric,
    initial_states,
    _force,
)
from audit_latent_order_rollout import rhs  # noqa: E402
from test_long_latent_equations import parameters  # noqa: E402


def test_initialization_arms_copy_and_modify_only_latent_tail():
    starts = np.arange(66.0).reshape(2, 33)
    original = starts.copy()
    qeq = np.array([0.2, -0.1, 0.3])
    assert np.array_equal(initial_states(starts, qeq, "original"), starts)
    anchored = initial_states(starts, qeq, "anchored")
    zero = initial_states(starts, qeq, "zero")
    assert np.array_equal(starts, original)
    assert np.array_equal(anchored[:, :30], starts[:, :30])
    assert np.array_equal(zero[:, :30], starts[:, :30])
    assert np.allclose(anchored[:, 30:], starts[:, 30:] - qeq)
    assert np.array_equal(zero[:, 30:], np.zeros((2, 3)))
    assert np.array_equal(initial_states(starts, qeq, "anchored")[:, 30:], np.array([[30, 31, 32], [63, 64, 65]]) - qeq)


def test_equal_qeq_anchored_initialization_is_exact_zero():
    starts = np.zeros((4, 33))
    qeq = np.array([0.2, -0.1, 0.3])
    starts[:, 30:] = qeq
    out = initial_states(starts, qeq, "anchored")
    assert np.array_equal(out[:, 30:], np.zeros((4, 3)))
    assert np.array_equal(out[:, :30], starts[:, :30])


def test_common_frequency_metric_excludes_initial_and_detects_uniform_error():
    truth = np.zeros((4, 33))
    pred = truth.copy()
    pred[0, 10:20] = 99.0  # must not enter the metric
    pred[1:, 10:20] = 2.0  # uniform COI error remains measurable
    metric = common_omega_metric(pred, truth, np.ones(10), 0.02)
    assert metric is not None
    assert metric["n"] == 2
    assert metric["rmse"] == pytest.approx(2.0)
    assert metric["sse"] == pytest.approx(8.0)


def test_common_frequency_event_aggregation_uses_event_rows():
    rows = [
        {"event_id": "e1", "rmse": 1.0, "sse": 2.0, "n": 2},
        {"event_id": "e2", "rmse": 3.0, "sse": 18.0, "n": 2},
    ]
    summary = aggregate_common(rows)
    assert summary["n_events"] == 2
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["pooled"] == pytest.approx(np.sqrt(5.0))
    assert summary["worst"] == 3.0
    assert summary["worst_event"] == "e2"


def test_equilibrium_force_field_is_zero_for_zero_state():
    p = parameters()
    state = np.zeros((2, 33))
    force = _force(state, p)
    assert force.shape == (2, 10)
    assert np.array_equal(force, np.zeros((2, 10)))
    assert np.array_equal(rhs(state, p), np.zeros_like(state))


def _screen_inputs():
    metrics = {"mean": 1.0, "pooled": 1.0, "cvar10": 1.0}
    complete = {arm: {"5": metrics.copy()} for arm in ("original", "anchored")}
    unstable = {arm: metrics.copy() for arm in ("original", "anchored")}
    scopes = {h: {"complete": {"max": 0.0}, "unstable_prefix": {"max": 0.0}} for h in ("5", "8")}
    numerics = {arm: {"scopes": copy.deepcopy(scopes)} for arm in ("original", "anchored")}
    equilibrium = {arm: {"physical_max_abs": 0.0, "force_max_abs": 0.0} for arm in ("anchored", "zero")}
    return complete, unstable, numerics, equilibrium


def test_candidate_screen_reports_baseline_only_numeric_failure():
    complete, unstable, numerics, equilibrium = _screen_inputs()
    numerics["original"]["scopes"]["5"]["complete"]["max"] = 0.002
    screen = candidate_screen(complete, unstable, numerics, equilibrium, [])
    assert screen["available"] is True
    assert screen["numerical_complete5_and_full_unstable_prefix_valid"] is False


def test_candidate_screen_invalid_finer_or_equilibrium_blocks_without_nan():
    complete, unstable, numerics, equilibrium = _screen_inputs()
    invalid = [{"arm": "anchored", "stage": "finer_00125", "reason": "nonfinite", "event_ids": ["e"]}]
    screen = candidate_screen(complete, unstable, numerics, equilibrium, invalid)
    assert screen["available"] is False
    equilibrium["zero"]["physical_max_abs"] = None
    screen = candidate_screen(complete, unstable, numerics, equilibrium, [])
    assert screen["available"] is False
    assert all(not isinstance(value, float) or np.isfinite(value) for value in screen.values())
