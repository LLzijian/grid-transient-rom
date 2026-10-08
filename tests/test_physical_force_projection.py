"""Synthetic projection and closure tests for Loop0064."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from audit_physical_force_projection import (  # noqa: E402
    _aggregate,
    _closure_summary,
    event_metrics,
    project_force,
    window_integrals,
)


def _fixture(seed=4):
    rng = np.random.default_rng(seed)
    Cq = rng.normal(size=(10, 3))
    while np.linalg.matrix_rank(Cq) < 3:
        Cq = rng.normal(size=(10, 3))
    a = rng.normal(size=(8, 3))
    qenc = rng.normal(size=(8, 3))
    qtruth = rng.normal(size=(8, 3))
    # Construct a genuine out-of-span component in null(Cq.T).
    _, _, vh = np.linalg.svd(Cq.T, full_matrices=True)
    n = vh[3]
    Fperp = rng.normal(size=(8, 1)) * n[None, :]
    F = (qtruth + a) @ Cq.T + Fperp
    return F, a, qenc, Cq


def test_inspan_outspan_decomposition_and_pythagorean_identity():
    F, a, qenc, Cq = _fixture()
    result = project_force(F, a, qenc, Cq)
    assert np.allclose(result["Fperp"] @ Cq, 0.0, atol=1e-10)
    assert np.allclose(result["e_total"], result["Fperp"] + result["e_parallel"], atol=1e-10)
    lhs = np.sum(result["e_total"] ** 2, axis=1)
    rhs = np.sum(result["Fperp"] ** 2, axis=1) + np.sum(result["e_parallel"] ** 2, axis=1)
    assert np.allclose(lhs, rhs, atol=1e-10)
    assert np.allclose(result["qF"], (F - a @ Cq.T) @ np.linalg.pinv(Cq, rcond=1e-12).T)


def test_nonorthogonal_latent_similarity_preserves_force_quantities():
    F, a, qenc, Cq = _fixture(7)
    base = project_force(F, a, qenc, Cq)
    T = np.array([[1.2, 0.3, -0.1], [0.0, 0.8, 0.2], [0.1, -0.2, 1.1]])
    assert abs(np.linalg.det(T)) > 1e-3
    transformed = project_force(F, a @ T.T, qenc @ T.T, Cq @ np.linalg.inv(T))
    for key in ("qF", "Fperp", "e_parallel", "e_total", "P"):
        expected = base[key]
        if key == "qF":
            expected = base[key] @ T.T
        assert np.allclose(transformed[key], expected, atol=1e-9)


def test_event_stats_are_equal_event_weighted_not_pooled_time():
    mass = np.ones(3)
    one = event_metrics(np.ones((1, 3)), np.ones((1, 3)), np.zeros((1, 3)), np.ones((1, 3)), mass)
    many = event_metrics(np.full((9, 3), 3.0), np.full((9, 3), 3.0), np.zeros((9, 3)), np.full((9, 3), 3.0), mass)
    aggregate = _aggregate([one, many])
    assert aggregate["n_events"] == 2
    assert aggregate["F_sq"] == pytest.approx((1.0 + 9.0) / 2)
    assert aggregate["F_sq"] != pytest.approx((1.0 + 9 * 9.0) / 10)


def test_closure_stats_are_equal_event_weighted_with_unequal_window_counts():
    defect = np.vstack([np.ones((1, 10)), np.full((3, 10), 3.0)])
    event_ids = np.array(["a", "b", "b", "b"])
    groups = np.array(["g", "g", "g", "g"])
    labels = np.array(["stable"] * 4)
    summary = _closure_summary(defect, event_ids, groups, labels)
    assert summary["all"]["n_events"] == 2
    assert summary["all"]["n_windows"] == 4
    assert summary["all"]["rms_force_per_s"] == pytest.approx(np.sqrt(5.0))


def test_common_acceleration_uses_sum_error_over_sum_mass_and_cross_term():
    mass = np.array([1.0, 2.0, 3.0])
    perp = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    parallel = np.tile(np.array([1.0, 0.0, 0.0]), (2, 1))
    total = perp + parallel
    metrics = event_metrics(np.zeros((2, 3)), perp, parallel, total, mass)
    assert metrics["Fperp_common_accel_rmse"] == pytest.approx(1 / 6)
    assert metrics["e_parallel_common_accel_rmse"] == pytest.approx(1 / 6)
    assert metrics["e_total_common_accel_rmse"] == pytest.approx(1 / 3)
    assert metrics["e_total_common_accel_rmse"] ** 2 != pytest.approx(
        metrics["Fperp_common_accel_rmse"] ** 2 + metrics["e_parallel_common_accel_rmse"] ** 2
    )


def test_aggregate_common_acceleration_is_rms_of_event_mses():
    rows = [
        {"F_sq": 0.0, "Fperp_sq": 0.0, "e_parallel_sq": 0.0, "e_total_sq": 0.0,
         "Fperp_net_force_rmse": 1.0, "e_parallel_net_force_rmse": 1.0, "e_total_net_force_rmse": 1.0,
         "Fperp_common_accel_rmse": 1.0, "e_parallel_common_accel_rmse": 1.0, "e_total_common_accel_rmse": 1.0},
        {"F_sq": 0.0, "Fperp_sq": 0.0, "e_parallel_sq": 0.0, "e_total_sq": 0.0,
         "Fperp_net_force_rmse": 3.0, "e_parallel_net_force_rmse": 3.0, "e_total_net_force_rmse": 3.0,
         "Fperp_common_accel_rmse": 3.0, "e_parallel_common_accel_rmse": 3.0, "e_total_common_accel_rmse": 3.0},
    ]
    aggregate = _aggregate(rows)
    assert aggregate["Fperp_common_accel_rmse_event_balanced_rms"] == pytest.approx(np.sqrt(5.0))


def test_net_force_rms_keeps_generator_cancellation_distinct_from_common_acceleration():
    mass = np.ones(3)
    cancel = event_metrics(np.zeros((1, 3)), np.array([[1.0, -1.0, 0.0]]),
                           np.zeros((1, 3)), np.array([[1.0, -1.0, 0.0]]), mass)
    aligned = event_metrics(np.zeros((1, 3)), np.array([[1.0, 1.0, 0.0]]),
                            np.zeros((1, 3)), np.array([[1.0, 1.0, 0.0]]), mass)
    assert cancel["Fperp_net_force_rmse"] == pytest.approx(0.0)
    assert cancel["Fperp_common_accel_rmse"] == pytest.approx(0.0)
    assert aligned["Fperp_net_force_rmse"] == pytest.approx(2.0)
    assert aligned["Fperp_common_accel_rmse"] == pytest.approx(2.0 / 3.0)


def test_nonfinite_projection_is_not_silently_filtered():
    F, a, qenc, Cq = _fixture()
    F[0, 0] = np.nan
    result = project_force(F, a, qenc, Cq)
    assert not np.isfinite(result["qF"]).all()


def test_linear_time_qf_window_integrals_keep_final_short_window():
    endpoints = np.arange(10, 36)
    t = endpoints * 0.01
    qf = np.column_stack([t, 2 * t, -3 * t + 1])
    ranges = [(10, 20), (20, 30), (30, 35)]
    result = window_integrals(qf, endpoints, ranges)
    expected_means = np.array([(qf[a - 10] + qf[b - 10]) / 2 for a, b in ranges])
    expected_y = np.tile(np.array([1.0, 2.0, -3.0]), (3, 1))
    assert np.allclose(result["qF_mean"], expected_means)
    assert np.allclose(result["yF"], expected_y)
    assert np.array_equal(result["start"], [10, 20, 30])
    assert np.array_equal(result["end"], [20, 30, 35])
    assert np.allclose(result["duration"], [0.1, 0.1, 0.05])
