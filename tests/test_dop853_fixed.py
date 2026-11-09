"""Qualify the fixed eighth-order step on artificial fields."""

import numpy as np
import pytest
from scipy.integrate import DOP853

from kiki.numerics import dop853_step


def oscillator(state):
    return 2.75 * np.stack((state[..., 1], -state[..., 0]), axis=-1)


def oscillator_exact(state, time):
    sine, cosine = np.sin(2.75 * time), np.cos(2.75 * time)
    return state * cosine + oscillator(state) * (sine / 2.75)


@pytest.mark.parametrize("step", [0.125, -0.125])
def test_analytic_oscillator_on_batched_states(step):
    state = np.array([[1.0, 0.0], [-2.0, 3.0]], dtype=np.float32)
    result = dop853_step(oscillator, state, step)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, oscillator_exact(state.astype(np.float64), step),
                               atol=1e-10, rtol=0)
    np.testing.assert_array_equal(state, [[1.0, 0.0], [-2.0, 3.0]])


def test_oscillator_global_error_decreases_at_order_eight():
    initial = np.array([1.0, 0.0])
    exact = oscillator_exact(initial, 2.0)
    errors = []
    for steps in (4, 8, 16):
        state = initial.copy()
        for _ in range(steps):
            state = dop853_step(oscillator, state, 2.0 / steps)
        errors.append(np.linalg.norm(state - exact))
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((7.7 < orders) & (orders < 8.3)), orders


def test_matches_scipy_accepted_step_using_only_twelve_rhs_calls():
    initial = np.array([0.5, -0.8, 1.7])
    step = 0.125
    stage_states = []

    def field(state):
        return np.sin(state) + 0.2 * np.roll(state, 1)

    def counted_field(state):
        stage_states.append(state.copy())
        return field(state)

    result = dop853_step(counted_field, initial, step)
    assert len(stage_states) == 12

    reference_states = []

    def reference_field(time, state):
        reference_states.append(state.copy())
        return field(state)

    reference = DOP853(reference_field, 0.0, initial, step, first_step=step,
                       max_step=step, rtol=1e-3, atol=1e-6)
    assert reference.step() is None
    assert reference.status == "finished" and reference.t == step
    assert reference.nfev == 13
    np.testing.assert_array_equal(stage_states, reference_states[:12])
    np.testing.assert_array_equal(result, reference.y)


@pytest.mark.parametrize(("invalid_derivative", "error"), [
    (np.ones(1), ValueError),
    (np.array([1j, 2j]), ValueError),
    (np.array([np.nan, 1.0]), FloatingPointError),
    (np.array([1.0, np.inf]), FloatingPointError),
])
def test_invalid_derivative_stops_after_failing_rhs_call(invalid_derivative, error):
    calls = []

    def field(state):
        calls.append(state)
        return invalid_derivative if len(calls) == 5 else -state

    with pytest.raises(error):
        dop853_step(field, np.ones(2), 0.125)
    assert len(calls) == 5


@pytest.mark.parametrize(("state", "step"), [
    (np.array([np.nan]), 0.125),
    (np.array([1j]), 0.125),
    (np.ones(1), 0.0),
    (np.ones(1), np.inf),
    (np.ones(1), np.nan),
    (np.ones(1), 0.125j),
    (np.ones(1), np.array([0.125])),
])
def test_invalid_inputs_fail_before_rhs(state, step):
    with pytest.raises(ValueError):
        dop853_step(lambda value: pytest.fail("unexpected RHS call"), state, step)


def test_nonfinite_stage_is_rejected_before_calling_rhs():
    calls = []

    def field(state):
        calls.append(state)
        return np.full_like(state, np.finfo(np.float64).max)

    with np.errstate(over="ignore"), pytest.raises(FloatingPointError, match="stage state"):
        dop853_step(field, np.ones(2), 32.0)
    assert len(calls) == 1
