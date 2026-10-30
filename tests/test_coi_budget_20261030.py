"""Offline, deterministic, CPU-only tests for the read-only COI budget probe.

No frozen prediction file and no model checkpoint is loaded here: the heavy P2 work is
exercised by the CLI, not by the unit tests.  Every test uses synthetic arrays only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]

import coi_budget_20261030 as probe  # noqa: E402

DT = probe.OUTPUT_DT
MASS = np.linspace(0.12, 3.2, 10)
DAMPING = np.linspace(1e-6, 0.24, 10)
JPI = np.linspace(0.4, 1.8, 10)
STORAGE = np.linspace(0.1, 1.2, 10)
Q_P = JPI * STORAGE


# --------------------------------------------------------------------- alignment


def _synthetic_event(steps, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "id": "synthetic",
        "label": "stable",
        "x": rng.normal(size=(probe.HISTORY_ROWS + steps, 30)),
    }


def test_prediction_and_truth_alignment():
    steps = 7
    event = _synthetic_event(steps)
    fine = np.zeros((2 * steps + 1, 40), dtype=np.float64)
    fine[::2, :30] = event["x"][10:10 + steps + 1]
    predicted = probe.prediction_window(fine, steps)
    truth = probe.truth_window(event, steps)
    assert predicted.shape == (steps, 40)
    assert truth.shape == (steps, 30)
    # fine[::2][1] is x[11]; the first compared point is the same physical time.
    np.testing.assert_array_equal(predicted[:, :30], truth)
    np.testing.assert_array_equal(predicted, fine[::2][1:steps + 1])
    np.testing.assert_array_equal(predicted, fine[2:2 * steps + 1:2])


def test_np_gradient_index_and_axis_alignment():
    steps = 9
    t = DT * np.arange(steps)
    omega = np.empty((steps, 10))
    for machine in range(10):
        omega[:, machine] = (machine + 1) * (2.0 * t ** 2 + 3.0 * t + 1.0)
    derivative = probe.omega_dot(omega)
    for machine in range(10):
        # central differences are exact for a quadratic; edge_order=2 is exact too.
        np.testing.assert_allclose(derivative[:, machine],
                                   (machine + 1) * (4.0 * t + 3.0), atol=1e-12, rtol=0)
    # axis 0 (time), never the machine axis
    assert not np.allclose(derivative[0], derivative[1])
    assert np.allclose(derivative, np.gradient(omega, DT, axis=0, edge_order=2))


def test_gradient_rejects_short_windows():
    with pytest.raises(ValueError):
        probe.omega_dot(np.zeros((2, 10)))


# ------------------------------------------------------------------ integration


def test_cumulative_trapezoid_exact_for_linear_functions():
    steps = 51
    t = DT * np.arange(steps)
    for slope, offset in ((0.0, 2.0), (3.5, -1.0), (-0.25, 0.0)):
        values = offset + slope * t
        integral = probe.cumulative_trapezoid(values, DT)
        np.testing.assert_allclose(integral, offset * t + 0.5 * slope * t ** 2,
                                   atol=1e-12, rtol=0)
    assert probe.cumulative_trapezoid(np.ones(1))[0] == 0.0


def test_cumulative_trapezoid_quadratic_is_second_order():
    steps = 200
    t = DT * np.arange(steps)
    integral = probe.cumulative_trapezoid(t ** 2, DT)
    exact = t ** 3 / 3.0
    error = np.abs(integral[-1] - exact[-1])
    assert 0.0 < error < 1e-4


def test_scalar_integrator_constant_driver_is_analytic():
    steps = 400
    w1 = np.full(steps, 1.7)
    kappa, lam, r10 = 0.9, 0.35, -0.4
    result = probe.integrate_scalar_closure(w1, r10, kappa, lam, DT)
    t = DT * np.arange(steps)
    exact = r10 * np.exp(-lam * t) + kappa * w1[0] / lam * (1.0 - np.exp(-lam * t))
    np.testing.assert_allclose(result, exact, atol=1e-12, rtol=0)


def test_scalar_integrator_linear_driver_is_exact():
    steps = 300
    t = DT * np.arange(steps)
    w1 = 0.5 + 0.8 * t
    kappa, lam, r10 = 1.3, 0.6, 0.2
    result = probe.integrate_scalar_closure(w1, r10, kappa, lam, DT)
    # analytic solution for a linear driver: R = R10 e^{-lt} + k(w0/l)(1-e^{-lt}) + k b/l^2 (lt-1+e^{-lt})
    decay = np.exp(-lam * t)
    rate = 0.8
    exact = (r10 * decay + kappa * 0.5 / lam * (1.0 - decay)
             + kappa * rate / lam ** 2 * (lam * t - 1.0 + decay))
    np.testing.assert_allclose(result, exact, atol=5e-12, rtol=0)


def test_green_column_matches_unit_driver_integration():
    steps = 120
    rng = np.random.default_rng(7)
    w1 = rng.normal(size=steps)
    grid = np.array([0.05, 0.5, 3.0])
    green = probe.scalar_closure_green(w1, grid)
    for column, lam in enumerate(grid):
        reference = probe.integrate_scalar_closure(w1, 0.0, 1.0, float(lam), DT)
        np.testing.assert_allclose(green[:, column], reference, atol=1e-12, rtol=0)


def test_pl_weights_small_lambda_limit():
    lam = 1e-3
    decay, alpha, beta = probe._pl_weights(lam, DT)
    assert decay == pytest.approx(1.0 - lam * DT + 0.5 * (lam * DT) ** 2, rel=1e-12)
    assert alpha == pytest.approx(DT * (1.0 - lam * DT / 2.0), rel=1e-10)
    assert beta == pytest.approx(DT ** 2 / 2.0 * (1.0 - lam * DT / 3.0), rel=1e-8)


# --------------------------------------------------------------- scalar fit


def test_scalar_fit_recovers_known_parameters():
    steps = 800
    t = DT * np.arange(steps)
    w1 = np.sin(2.0 * math.pi * t / 3.0) + 0.25
    true_lambda = float(probe.LAMBDA_GRID[100])
    kappa, r10 = 0.83, -0.45
    target = probe.integrate_scalar_closure(w1, r10, kappa, true_lambda, DT)
    fitted = probe.fit_scalar_closure(target, w1, r10_fixed=r10)
    free = fitted["free"]
    assert free["lambda"] == pytest.approx(true_lambda, rel=0, abs=1e-15)
    assert free["kappa"] == pytest.approx(kappa, rel=1e-8, abs=0.0)
    assert free["r10"] == pytest.approx(r10, rel=1e-8, abs=1e-12)
    assert free["rel_rms"] < 1e-6
    fixed = fitted["fixed"]
    assert fixed["lambda"] == pytest.approx(true_lambda, rel=0, abs=1e-15)
    assert fixed["kappa"] == pytest.approx(kappa, rel=1e-8, abs=0.0)
    assert fixed["rel_rms"] < 1e-6


def test_scalar_fit_detects_a_wrong_closure_form():
    """A single exponential closure cannot fit a two-timescale driver exactly."""
    steps = 800
    t = DT * np.arange(steps)
    w1 = np.sin(2.0 * math.pi * t / 1.5) + np.exp(-t / 0.4)
    target = (probe.integrate_scalar_closure(w1, 0.1, 0.8, 0.2, DT)
              + 0.4 * np.exp(-t / 0.05))  # component outside the scalar closure
    fitted = probe.fit_scalar_closure(target, w1)
    assert fitted["free"]["rel_rms"] > 1e-3


def test_scalar_fit_rejects_short_or_mismatched_windows():
    with pytest.raises(ValueError):
        probe.fit_scalar_closure(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError):
        probe.fit_scalar_closure(np.zeros(30), np.zeros(31))


# ---------------------------------------------------------- mechanical inversion


def test_collective_force_sum_recovers_known_force():
    steps = 41
    t = DT * np.arange(steps)
    rng = np.random.default_rng(11)
    a = rng.normal(size=(10,))
    b = rng.normal(size=(10,))
    c = rng.normal(size=(10,))
    d = rng.normal(size=(10,))
    e = rng.normal(size=(10,))
    omega = np.stack([a[i] * t ** 2 + b[i] * t + c[i] for i in range(10)], axis=1)
    p = np.stack([d[i] * t + e[i] for i in range(10)], axis=1)
    omega_dot = np.stack([2.0 * a[i] * t + b[i] for i in range(10)], axis=1)
    r = (-MASS[None, :] * omega_dot - DAMPING[None, :] * omega + Q_P[None, :] * p)
    recovered = probe.collective_force_sum(omega, p, MASS, DAMPING, Q_P) / 10.0
    np.testing.assert_allclose(recovered, r.sum(axis=1) / 10.0, atol=1e-10, rtol=0)


def test_r1_est_sign_units_and_linearity():
    steps = 21
    t = DT * np.arange(steps)
    omega = np.stack([np.ones(steps) * (machine + 1) * 0.1 for machine in range(10)], axis=1)
    p = np.zeros((steps, 10))
    # constant omega, no acceleration, no port: 1^T r_est = -1^T(D*omega) < 0
    baseline = probe.collective_force_sum(omega, p, MASS, DAMPING, Q_P)
    assert np.all(baseline < 0.0)
    np.testing.assert_allclose(baseline, -np.einsum("ij,j->i", omega, DAMPING))
    # linear in omega (and in its derivative): triple omega -> triple r_est
    tripled = probe.collective_force_sum(3.0 * omega, p, MASS, DAMPING, Q_P)
    np.testing.assert_allclose(tripled, 3.0 * baseline, atol=1e-12, rtol=0)
    # units: R1 = 1^T r / 10
    sum_r = probe.collective_force_sum(omega, p, MASS, DAMPING, Q_P)
    np.testing.assert_allclose(sum_r / 10.0, (-np.einsum("ij,j->i", omega, DAMPING)) / 10.0)
    assert t.shape == (steps,)


# -------------------------------------------------------------- budget identity


def _consistent_budget_fixture(steps=200, seed=3):
    """Truth/pred pair whose detrended COI budget closes to roundoff.

    Both paths are linear in time, so every difference is at most linear and the
    cumulative trapezoid is exact.  ``r_pred`` is the mechanical inversion of the *model*
    path, so the model ODE holds by construction -- exactly the cross-check situation.
    """
    t = DT * np.arange(steps)
    rng = np.random.default_rng(seed)
    omega_true = np.stack([rng.normal() + 0.1 * t for _ in range(10)], axis=1)
    p_true = np.stack([0.2 + 0.01 * t for _ in range(10)], axis=1)
    c0 = rng.normal(scale=0.5, size=10)
    slope = rng.normal(scale=0.3, size=10)
    delta_omega = np.stack([c0[i] + slope[i] * t for i in range(10)], axis=1)
    delta_p = np.stack([0.01 * i + 0.02 * t for i in range(10)], axis=1)
    omega_pred = omega_true + delta_omega
    p_pred = p_true + delta_p
    sum_r_pred = probe.collective_force_sum(omega_pred, p_pred, MASS, DAMPING, Q_P)
    r_pred = np.zeros((steps, 10))
    r_pred[:, 0] = sum_r_pred
    return omega_true, p_true, omega_pred, p_pred, r_pred


def test_budget_closes_with_consistent_synthetic_trajectory():
    omega_true, p_true, omega_pred, p_pred, r_pred = _consistent_budget_fixture()
    result = probe.budget_terms(omega_true, p_true, omega_pred, p_pred, r_pred,
                                MASS, DAMPING, Q_P)
    summary = result["summary"]
    assert summary["closure_max_abs_detrended"] < 1e-12
    # the raw residual is the initial COI offset (one 0.01 s step after initialisation)
    expected_offset = float(np.einsum("ij,j->i", omega_pred - omega_true, MASS)[0]
                            / MASS.sum())
    assert abs(expected_offset) > 1e-3
    assert summary["delta_omega_coi_first"] == pytest.approx(expected_offset, rel=1e-12)
    assert summary["closure_max_abs"] == pytest.approx(abs(expected_offset), abs=1e-12)
    np.testing.assert_allclose(np.asarray(result["closure"]) - summary["delta_omega_coi_first"],
                               np.asarray(result["closure_detrended"]), atol=1e-15, rtol=0)
    shares = summary["terminal_share"]
    abs_shares = summary["mean_abs_share"]
    assert sum(shares) == pytest.approx(1.0, abs=1e-12)
    assert sum(abs_shares) == pytest.approx(1.0, abs=1e-12)
    assert all(value >= 0.0 for value in abs_shares)


def test_budget_term_signs_are_registered():
    """Flipping the damping term's sign must break the detrended closure."""
    omega_true, p_true, omega_pred, p_pred, r_pred = _consistent_budget_fixture()
    truth = probe.budget_terms(omega_true, p_true, omega_pred, p_pred, r_pred,
                               MASS, DAMPING, Q_P)
    assert truth["summary"]["closure_max_abs_detrended"] < 1e-12
    wrong_total = (np.asarray(truth["total_integral"]) - 2.0 * np.asarray(truth["integral_d"]))
    wrong_residual = (np.asarray(truth["delta_omega_coi"])
                      - np.asarray(truth["delta_omega_coi"])[0]
                      - wrong_total / MASS.sum())
    assert np.max(np.abs(wrong_residual)) > 1e-4


def test_budget_cross_check_on_a_self_consistent_path_is_small():
    steps = 300
    rng = np.random.default_rng(5)
    t = DT * np.arange(steps)
    omega = np.stack([0.1 * np.sin(t * (1.0 + 0.05 * i)) for i in range(10)], axis=1)
    p = np.stack([0.1 * np.cos(t * 0.5) for _ in range(10)], axis=1)
    derivative = probe.omega_dot(omega)
    r = -MASS[None, :] * derivative - DAMPING[None, :] * omega + Q_P[None, :] * p
    result = probe.cross_check_terms(omega, p, r, MASS, DAMPING, Q_P)
    assert result["residual_sum_max_abs"] < 1e-10 * max(1.0, result["inversion_rms"])
    assert result["closure_max_abs"] < 1e-8


# -------------------------------------------------------------------- closure terms


def test_closure_terms_model_self_consistency():
    """Driving the closure by the model's own W1 reproduces the model's own R1."""
    steps = 400
    rng = np.random.default_rng(9)
    t = DT * np.arange(steps)
    kappa, lam = 0.83, 0.19
    w1 = np.sin(2.0 * math.pi * t / 2.5) + 0.3
    r1 = probe.integrate_scalar_closure(w1, -0.2, kappa, lam, DT)
    omega_true = np.zeros((steps, 10))
    omega_true[:] = w1[:, None]  # same collective mean for truth and model
    omega_pred = omega_true.copy()
    p_true = np.zeros((steps, 10))
    p_pred = np.zeros((steps, 10))
    r_pred = np.zeros((steps, 10))
    r_pred[:] = r1[:, None]
    sum_r_est = probe.collective_force_sum(omega_true, p_true, MASS, DAMPING, Q_P)
    terms = probe.closure_terms(omega_true, p_true, omega_pred, p_pred, r_pred,
                                sum_r_est, kappa, lam)
    assert terms["self_consistency_rel_rms"] < 1e-10
    assert terms["forward_rel_rms"] is not None
    assert terms["fit"] is not None  # 400 samples >= MIN_FIT_STEPS
    assert terms["fit_fixed_r10"] is not None


def test_closure_terms_skips_short_window_fit():
    steps = probe.MIN_FIT_STEPS - 1
    omega = np.zeros((steps, 10))
    p = np.zeros((steps, 10))
    r = np.zeros((steps, 10))
    sum_r_est = np.zeros(steps)
    terms = probe.closure_terms(omega, p, omega, p, r, sum_r_est, 0.8, 0.2)
    assert terms["fit"] is None
    assert terms["fit_fixed_r10"] is None


# ---------------------------------------------------------------- denominators


def _event(event_id, label, length):
    return {"id": event_id, "label": label, "x": np.zeros((length, 30), dtype=np.float64)}


def test_window_steps_and_denominators():
    assert (probe.EXPECTED_EVENTS, probe.EXPECTED_STABLE, probe.EXPECTED_UNSTABLE) == (297, 267, 30)
    assert probe.SCOPE_STEPS == {"5": 500, "8": 800}
    assert probe.MAX_ACTUAL_STEPS == 1500
    assert set(probe.COHORT_LABEL.values()) == {"stable", "unstable"}
    long_stable = _event("s", "stable", 1501)
    assert probe.window_steps(long_stable, "5") == 500
    assert probe.window_steps(long_stable, "8") == 800
    assert probe.window_steps(long_stable, "prefix") == 1490  # label-agnostic length
    short_stable = _event("s", "stable", 611)
    assert probe.window_steps(short_stable, "5") == 500
    assert probe.window_steps(short_stable, "8") is None
    unstable = _event("u", "unstable", 47)
    assert probe.window_steps(unstable, "prefix") == 36
    assert probe.window_steps(unstable, "5") is None
    with pytest.raises(ValueError):
        probe.window_steps(unstable, "2")
    events = [long_stable, _event("s2", "stable", 1501), unstable]
    assert len(probe.cohort(events, "5")) == 2
    assert len(probe.cohort(events, "8")) == 2
    assert [event["id"] for event in probe.cohort(events, "prefix")] == ["u"]


# --------------------------------------------------------------- frozen splits


def test_closed_split_guard():
    probe.assert_closed_splits({"status": "COMPLETED", "val_read": False, "test_read": False})
    with pytest.raises(ValueError):
        probe.assert_closed_splits({"status": "COMPLETED", "val_read": True, "test_read": False})
    with pytest.raises(ValueError):
        probe.assert_closed_splits({"status": "COMPLETED", "val_read": False, "test_read": True})
    with pytest.raises(ValueError):
        probe.assert_closed_splits({"status": "PARTIAL", "val_read": False, "test_read": False})


def test_only_fit_split_is_addressable():
    text = Path(probe.__file__).read_text(encoding="utf-8")
    assert "fit_only=True" in text
    assert "fit_only=False" not in text
    for path in probe.DATA_SOURCE_PATHS:
        parts = Path(path).resolve().relative_to(ROOT).parts
        assert not {"val", "test", "validation"} & {part.lower() for part in parts}
        assert Path(path).resolve().is_relative_to(ROOT)


# ------------------------------------------------------------- mechanical checks


def _buffers(mass=None):
    return {
        "mass": np.asarray(MASS if mass is None else mass, dtype=np.float64),
        "damping": np.asarray(DAMPING),
        "jpi_p": np.asarray(JPI),
        "storage_base": np.asarray(STORAGE),
        "p_scale": np.asarray(STORAGE),
        "q_p": np.asarray(Q_P),
    }


def test_buffer_signature_detects_any_difference():
    first = _buffers()
    second = _buffers()
    assert probe.buffer_signature(first) == probe.buffer_signature(second)
    second["damping"] = second["damping"].copy()
    second["damping"][3] = np.nextafter(second["damping"][3], np.inf)
    assert probe.buffer_signature(first) != probe.buffer_signature(second)


def test_mechanical_identity_check_reports_mismatch_without_substitution():
    identities = {}
    for name in ("A", "R_N"):
        body = _buffers()
        identities[name] = {
            "name": name,
            "buffers": {field: body[field].tolist() for field in probe.MECHANICAL_FIELDS},
            "signature": probe.buffer_signature(body),
        }
    check = probe.check_mechanical_identity(identities, canonical="A")
    assert check["bit_identical_to_canonical"] == {"A": True, "R_N": True}
    changed = dict(identities["R_N"])
    body = _buffers()
    body["mass"] = body["mass"].copy()
    body["mass"][0] = np.nextafter(body["mass"][0], np.inf)
    changed["buffers"] = {field: body[field].tolist() for field in probe.MECHANICAL_FIELDS}
    changed["signature"] = probe.buffer_signature(body)
    identities["R_N"] = changed
    check = probe.check_mechanical_identity(identities, canonical="A")
    assert check["bit_identical_to_canonical"]["R_N"] is False
    assert check["different_fields"]["R_N"] == ["mass"]
    assert check["substitution"] == "none; each model uses its own frozen buffers"


def test_uniformity_block_flags_non_uniform_vectors():
    block = probe.uniformity_block(_buffers())
    assert block["damping"]["uniform"] is False
    assert block["damping"]["std"] > 0.0
    assert block["q_p"]["distinct_values"] == 10
    uniform = _buffers(mass=np.full(10, 2.0))
    assert probe.uniformity_block(uniform)["q_p"]["uniform"] is False
    single = dict(_buffers())
    single["q_p"] = np.full(10, 0.5)
    assert probe.uniformity_block(single)["q_p"]["uniform"] is True


def _fake_row(event_id, scope, forward, w1, share_r):
    return {
        "event_id": event_id,
        "scope": scope,
        "summary": {"mean_abs_share": [share_r, 0.1, 0.9 - share_r],
                    "closure_max_abs": forward / 10.0},
        "closure": {"forward_rel_rms": forward, "oracle_r10_rel_rms": forward / 2,
                    "w1_rel_diff": w1, "w1_component_rel": w1,
                    "amplification": forward / w1, "w1_variance_explained": 0.5,
                    "r1_model_rel_diff": forward},
    }


def test_paired_model_gap_uses_matched_event_windows():
    left = [_fake_row("e1", "5", 2.0, 1.0, 0.5), _fake_row("e2", "5", 4.0, 2.0, 0.7),
            _fake_row("u1", "prefix", 0.4, 0.1, 0.9)]
    right = [_fake_row("e1", "5", 1.0, 0.5, 0.4), _fake_row("e2", "5", 3.0, 1.5, 0.6),
             _fake_row("u1", "prefix", 0.5, 0.1, 0.8)]
    gap = probe.paired_model_gap(left, right, "A", "R_N")
    assert gap["left"] == "A" and gap["right"] == "R_N"
    assert gap["5"]["pairs"] == 2
    assert gap["5"]["forward_rel_rms"]["p50"] == pytest.approx(1.0)
    assert gap["5"]["forward_rel_rms"]["positive_fraction"] == 1.0
    assert gap["prefix"]["pairs"] == 1
    assert gap["prefix"]["forward_rel_rms"]["p50"] == pytest.approx(-0.1)


# ------------------------------------------------------------------- path guards


def test_output_guard_confines_writes():
    assert probe.guard_output("private_data/coi_budget_20261030/budget.json",
                              probe.PRIVATE_ROOT) == (probe.PRIVATE_ROOT / "budget.json").resolve()
    assert probe.guard_output("results/coi_budget_20261030/public_budget.json",
                              probe.RESULTS_ROOT) == (probe.RESULTS_ROOT / "public_budget.json").resolve()
    for bad in ("experiments/coi_budget.json",
                "private_data/other_place/budget.json",
                "results/coi_budget_20261030/public_budget.json"):
        with pytest.raises(ValueError):
            probe.guard_output(bad, probe.PRIVATE_ROOT)
    with pytest.raises(ValueError):
        probe.guard_output("private_data/coi_budget_20261030/budget.txt", probe.PRIVATE_ROOT)


def test_publish_input_guard_rejects_foreign_inputs():
    with pytest.raises(ValueError):
        probe.guard_input("private_data/state_gated_memory_20261028/study01/prepare.json",
                          probe.PRIVATE_ROOT)
    with pytest.raises(ValueError):
        probe.guard_input("results/coi_budget_20261030/public_budget.json", probe.PRIVATE_ROOT)
    with pytest.raises(FileNotFoundError):
        probe.guard_input("private_data/coi_budget_20261030/missing.json", probe.PRIVATE_ROOT)


# --------------------------------------------------------------------- publish


def _minimal_budget():
    return {
        "schema": probe.SCHEMA,
        "status": "COMPLETED",
        "created_unix": 0.0,
        "models": {},
        "model_blocks": {},
        "method": {"mechanical_equation": "synthetic"},
        "frozen_preparation": {"event_count": 297, "val_read": False, "test_read": False},
        "mechanical_identity": {
            "canonical": "A", "fields": list(probe.MECHANICAL_FIELDS),
            "bit_identical_to_canonical": {"A": True},
            "different_fields": {"A": []},
            "shared_signature": "0" * 64,
            "substitution": "none",
        },
        "mechanical_variants": {},
        "cohorts": {},
        "acceptance": {"denominators": {"5": 267, "8": 267, "prefix": 30}},
        "timings": {},
        "seconds": 0.0,
        "code": {},
    }


def test_publish_rejects_wrong_denominators(tmp_path):
    budget = _minimal_budget()
    budget["acceptance"]["denominators"] = {"5": 266, "8": 267, "prefix": 30}
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(budget), encoding="utf-8")
    with pytest.raises(ValueError):
        probe.build_public(budget, path)
    budget["acceptance"]["denominators"] = {"5": 267, "8": 267, "prefix": 29}
    with pytest.raises(ValueError):
        probe.build_public(budget, path)


def test_publish_accepts_registered_denominators(tmp_path):
    budget = _minimal_budget()
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(budget), encoding="utf-8")
    public = probe.build_public(budget, path)
    assert public["schema"] == probe.PUBLIC_SCHEMA
    assert public["private_budget"]["sha256"] == probe.sha256_file(path)
    assert public["models"] == {}
    assert public["splits"] == {"val_read": False, "test_read": False}
    with pytest.raises(ValueError):
        probe.build_public(dict(budget, status="FAILED"), path)


# ------------------------------------------------------------------------- CLI


def test_cli_argument_surface(monkeypatch):
    calls = {}
    monkeypatch.setattr(probe, "run_probe", lambda models, max_events=None: calls.update(
        models=models, max_events=max_events) or {"schema": probe.SCHEMA,
                                                  "status": "COMPLETED", "seconds": 0.0,
                                                  "acceptance": {"denominators": {}},
                                                  "models": {}})
    written = {}
    monkeypatch.setattr(probe, "write_json", lambda path, payload: written.update(
        path=Path(path), payload=payload))
    exit_code = probe.main(["run", "--output", "private_data/coi_budget_20261030/budget.json",
                            "--models", "A,R_N"])
    assert exit_code == 0
    assert calls["models"] == ("A", "R_N")
    assert written["path"] == (probe.PRIVATE_ROOT / "budget.json").resolve()


def test_cli_rejects_bad_model_list():
    with pytest.raises(ValueError):
        probe._parse_models("A,Q")
    assert probe._parse_models("A, R_N") == ("A", "R_N")
