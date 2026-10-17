import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

from closure_discovery_benchmarks import (
    FIXTURE_SEED,
    PARTIAL_DT_SECONDS,
    PARTIAL_FUTURE_STEPS,
    PARTIAL_HISTORY_STEPS,
    _observe_with_noise,
    _partial_truth_fixture,
    identify,
    pool,
    toy_source,
    v,
    mul,
    power,
)


def _partial_candidates(candidates):
    return [term for term in candidates if term["channel"] == "port" and
            ((term["expression"]["op"] == "var" and
              term["expression"]["name"] in ("p", "w", "q")) or
             (term["expression"]["op"] == "tanh" and
              term["expression"]["arg"] == v("q")))]


def test_noise_is_on_state_observations_and_uses_fixed_variable_scales():
    clean = np.arange(320, dtype=np.float64).reshape(8, 40) / 7
    scales = clean.std(axis=0)
    rng = np.random.default_rng(42)
    observed = _observe_with_noise(clean, scales, .01, rng)
    draw = np.random.default_rng(42).normal(size=clean.shape)
    np.testing.assert_array_equal(observed, clean + .01 * draw * scales)
    assert np.array_equal(clean, np.arange(320, dtype=np.float64).reshape(8, 40) / 7)

    base = toy_source()
    term = {"channel": "node", "expression": mul(power(v("p"), 2), power(v("q"), 2))}
    truth = base.with_terms([term])
    with torch.no_grad():
        truth.coefficients.fill_(.2)
        states = torch.as_tensor(clean[:4] * .01, dtype=torch.float64)
        clean_labels = truth.field(states).numpy().copy()
    labels_before = clean_labels.copy()
    observed_states = _observe_with_noise(states.numpy(), scales, .01,
                                          np.random.default_rng(3))
    recovered, best = identify(base, observed_states, clean_labels, [term])
    assert best[1] == recovered.terms[0]
    np.testing.assert_array_equal(clean_labels, labels_before)
    assert not np.array_equal(observed_states, states.numpy())
    observed_tensor = torch.as_tensor(observed_states, dtype=torch.float64)
    with torch.no_grad():
        target_residual = torch.as_tensor(clean_labels) - base.field(observed_tensor)
        unit_candidate = base.with_terms([term])
        unit_candidate.coefficients.fill_(1.)
        candidate_residual = unit_candidate.field(observed_tensor) - base.field(observed_tensor)
        coefficient = (candidate_residual * target_residual).sum() / candidate_residual.square().sum()
        expected_mse = float((target_residual - coefficient * candidate_residual).square().mean())
    assert best[2] == pytest.approx(float(coefficient), abs=1e-12)
    assert best[0] == pytest.approx(expected_mse, abs=1e-12)


def test_partial_fixture_is_autonomous_ode_history_and_future_with_hidden_q():
    assert PARTIAL_HISTORY_STEPS * PARTIAL_DT_SECONDS == pytest.approx(.1)
    assert PARTIAL_FUTURE_STEPS * PARTIAL_DT_SECONDS == pytest.approx(.2)
    base = toy_source()
    saturating = {"channel": "port", "expression": {
        "op": "tanh", "scale": 2., "arg": v("q")}}
    truth = base.with_terms([saturating])
    with torch.no_grad():
        truth.coefficients.fill_(.6)
    initial, trajectory, history, future = _partial_truth_fixture(truth, FIXTURE_SEED)

    assert trajectory.shape == (8, PARTIAL_HISTORY_STEPS + PARTIAL_FUTURE_STEPS + 1, 40)
    assert history.shape == (8, PARTIAL_HISTORY_STEPS + 1, 30)
    assert future.shape == (8, PARTIAL_FUTURE_STEPS, 30)
    np.testing.assert_array_equal(history, trajectory[:, :PARTIAL_HISTORY_STEPS + 1, :30])
    np.testing.assert_array_equal(future, trajectory[:, PARTIAL_HISTORY_STEPS + 1:, :30])
    assert initial.shape == (8, 40)
    expected_next = truth.rk4(torch.as_tensor(initial, dtype=torch.float64), PARTIAL_DT_SECONDS)
    np.testing.assert_allclose(trajectory[:, 1], expected_next.detach().numpy(), atol=1e-14, rtol=0)
    from closure_discovery_benchmarks import path
    future_from_history_end = path(
        truth, torch.as_tensor(trajectory[:, PARTIAL_HISTORY_STEPS], dtype=torch.float64),
        PARTIAL_FUTURE_STEPS, dt=PARTIAL_DT_SECONDS)[:, 1:, :30]
    np.testing.assert_allclose(future, future_from_history_end.detach().numpy(), atol=1e-13, rtol=0)
    assert float(initial[:, :30].std()) > 0
    assert float(initial[:, 30:].std()) > float(initial[:, :30].std())


def test_partial_saturation_keeps_the_six_registered_blind_candidates():
    candidates = _partial_candidates(pool())
    assert len(candidates) == 6
    assert all(candidate["channel"] == "port" for candidate in candidates)
