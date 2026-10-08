import numpy as np
import pytest

from experiments.diagnose_electrical_balance import electrical_balance


def test_electrical_impulse_exact_linear_force_and_coi_invariance():
    t = np.linspace(0, 1, 101)
    mass = np.array([2.0, 3.0])
    force = t[:, None] * np.array([6.0, -6.0])
    omega = t[:, None] ** 2 * np.array([1.5, -1.0])
    result = electrical_balance(t, omega + t[:, None], force + mass, np.tile(mass, (101, 1)), mass)
    assert result["speed_residual_rmse"] < 1e-14


def test_electrical_impulse_retains_missing_force():
    t = np.linspace(0, 1, 101)
    omega = t[:, None] * np.array([3.0, -2.0])
    zero = np.zeros_like(omega)
    result = electrical_balance(t, omega, zero, zero, np.array([2.0, 3.0]))
    np.testing.assert_allclose(result["terminal_force_impulse"], [6.0, -6.0])


def test_electrical_impulse_rejects_bad_power_shape():
    with pytest.raises(ValueError, match="alignment"):
        electrical_balance([0, 1], np.zeros((2, 2)), np.zeros((2, 2)), np.zeros(2), [1, 1])
