import numpy as np

from experiments.diagnose_force_balance import forced_impulse


class ConstantAcceleration:
    def __init__(self, acceleration):
        self.acceleration = np.asarray(acceleration)

    def initial_hidden(self, omega, static):
        return np.zeros(0)

    def _pack_hidden(self, hidden):
        return hidden

    def _flat_field(self, delta, omega, hidden, pm, static):
        return omega, self.acceleration, np.zeros(0)


def test_impulse_matches_constant_acceleration_without_numerical_differentiation():
    time = np.linspace(0, 1, 101)
    acceleration = np.array([2.0, -1.0])
    omega = time[:, None] * acceleration
    delta = 0.5 * time[:, None] ** 2 * acceleration
    result = forced_impulse(ConstantAcceleration(acceleration), {"mass": np.ones(2)},
                            time, delta, omega, np.zeros_like(omega), interpolation="zoh")
    assert result["cumulative_speed_residual_rmse"] < 1e-13
    assert result["kinematic_integration_error_max"] < 1e-13


def test_impulse_reveals_persistent_missing_force():
    time = np.linspace(0, 1, 101)
    omega = time[:, None] * np.array([2.0, -1.0])
    delta = time[:, None] * omega / 2
    result = forced_impulse(ConstantAcceleration([0, 0]), {"mass": np.array([3.0, 4.0])},
                            time, delta, omega, np.zeros_like(omega), interpolation="zoh")
    np.testing.assert_allclose(result["terminal_force_impulse"], [6.0, -4.0])
    np.testing.assert_allclose(result["late_mean_force_residual_l2"], np.sqrt(52))
