"""Synthetic contract tests for the fixed-support group-identifiability audit."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from experiments.audit_group_identifiability import (
    _spectrum,
    event_weights,
    fit_supported,
    history_indices,
    projected_metrics,
    window_average,
    window_ranges,
)


def _support() -> np.ndarray:
    mask = np.zeros((76, 3), dtype=bool)
    mask[:12, 0] = True
    mask[20:32, 1] = True
    mask[40:52, 2] = True
    return mask


def test_causal_indices_and_partial_windows():
    assert np.array_equal(history_indices(14), np.arange(10, 14))
    assert len(history_indices(10)) == 0
    assert np.array_equal(history_indices(11), np.array([10]))
    assert window_ranges(36) == [(10, 20), (20, 30), (30, 35)]
    assert window_ranges(21) == [(10, 20)]
    assert window_ranges(11) == []


def test_integrated_linear_recovery_and_no_zero_duration_window():
    t = np.arange(0.0, 1.01, 0.1)
    x = np.column_stack([np.ones_like(t), t])
    q = 2 * t - 1.5 * t**2
    rows, targets = [], []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        mean, target, duration = window_average(x[a : b + 1], q[a : b + 1], dt=0.1)
        assert duration == pytest.approx(0.1)
        rows.append(mean)
        targets.append(target)
    assert np.allclose(np.asarray(rows) @ np.array([2.0, -3.0]), targets)
    assert np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=1e-12)[0] == pytest.approx([2.0, -3.0])


def test_event_equal_weights_and_lgo_recomputation():
    ids = np.array(["a", "b", "b", "b", "b"])
    weights, counts = event_weights(ids)
    assert counts == {"a": 1, "b": 4}
    assert weights[0] == pytest.approx(1 / np.sqrt(2))
    assert np.all(weights[1:] == pytest.approx(1 / np.sqrt(8)))
    lgo, lgo_counts = event_weights(ids[1:])
    assert lgo_counts == {"b": 4}
    assert np.all(lgo == pytest.approx(0.5))


def test_supported_svd_recovers_only_saved_active_columns():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 76))
    mask = _support()
    true = np.zeros((76, 3))
    true[mask] = rng.normal(size=int(mask.sum()))
    y = X @ true
    weights = np.full(len(X), 1 / np.sqrt(len(X)))
    scale = np.maximum(np.sqrt(np.sum((weights[:, None] * X) ** 2, axis=0)), 1e-8)
    coefficient, spectra = fit_supported(X, y, weights, mask, scale)
    assert np.allclose(coefficient, true, atol=1e-10, rtol=0)
    assert all(item["rank_at_rtol"]["1e-10"] == 12 for item in spectra)

    perturbed = X.copy()
    perturbed[:, 17] += 1000.0
    changed, _ = fit_supported(perturbed, y, weights, mask, scale)
    assert np.array_equal(changed, coefficient)


def test_rank_deficient_and_wide_designs_are_nonunique():
    rng = np.random.default_rng(3)
    mask = _support()
    X = rng.normal(size=(100, 76))
    X[:, 1] = X[:, 0]
    y = X @ np.eye(76, 3, dtype=float)
    weights = np.full(len(X), 1 / np.sqrt(len(X)))
    scale = np.maximum(np.sqrt(np.sum((weights[:, None] * X) ** 2, axis=0)), 1e-8)
    _, spectra = fit_supported(X, y, weights, mask, scale)
    assert spectra[0]["rank_at_rtol"]["1e-10"] < 12
    assert spectra[0]["nonunique_at_rcond"]

    X5 = X[:5]
    y5 = y[:5]
    w5 = np.full(5, 1 / np.sqrt(5))
    scale5 = np.maximum(np.sqrt(np.sum((w5[:, None] * X5) ** 2, axis=0)), 1e-8)
    _, spectra5 = fit_supported(X5, y5, w5, mask, scale5)
    assert all(item["nonunique_at_rcond"] for item in spectra5)


def test_column_scaling_and_zero_column_floor():
    X = np.array([[2.0, 200.0], [4.0, -100.0], [-2.0, 50.0]])
    weights = np.full(3, 1 / np.sqrt(3))
    scale = np.sqrt(np.sum((weights[:, None] * X) ** 2, axis=0))
    assert np.all(scale > 0)
    assert np.allclose(np.linalg.lstsq(X / scale, (X @ [3.0, -0.5]), rcond=1e-10)[0] / scale, [3.0, -0.5])
    assert max(0.0, 1e-8) == 1e-8
    assert _spectrum(np.diag([1.0, 1e-11]))["rank_at_rtol"]["1e-12"] == 2
    assert _spectrum(np.diag([1.0, 1e-11]))["rank_at_rtol"]["1e-10"] == 1


def test_projected_force_units_and_absent_label_stratum():
    X = np.zeros((2, 2))
    y = np.array([[1.0, -2.0], [1.0, -2.0]])
    xi = np.zeros((2, 2))
    labels = np.array(["stable", "stable"])
    events = np.array(["e1", "e2"])
    result = projected_metrics(X, y, events, labels, np.diag([2.0, 3.0]), xi)
    assert result["all"]["rms_force_per_s"] == pytest.approx(np.sqrt(20))
    assert result["stable"]["n_events"] == 2
    assert result["unstable"] == {"n_windows": 0, "n_events": 0, "rms_force_per_s": None}


@pytest.mark.parametrize("which", ["X", "y", "weights"])
def test_nonfinite_inputs_raise(which):
    X = np.ones((4, 76))
    y = np.ones((4, 3))
    mask = _support()
    weights = np.full(4, 0.5)
    scale = np.ones(76)
    if which == "X":
        X[0, 0] = np.nan
    elif which == "y":
        y[0, 0] = np.inf
    else:
        weights[0] = np.nan
    with pytest.raises(FloatingPointError):
        fit_supported(X, y, weights, mask, scale)
