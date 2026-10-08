import numpy as np
from scipy.integrate import solve_ivp

from kiki.models.private_rom import SwingROM, fit_swing


def fixture_model():
    return SwingROM(np.array([1., 2.]), np.array([[1.], [-1.]]), np.array([.1, -.1]),
                    np.array([1., 2.]), np.array([2.]), np.array([.2, .3]))


def test_equilibrium_and_independent_integration():
    model = fixture_model()
    np.testing.assert_array_equal(model.field(np.zeros(4), model.pm_star), np.zeros(4))
    initial = np.array([.2, -.1, .1, -.2])
    time = np.arange(401)*.01
    expected = solve_ivp(lambda t, x: model.field(x, model.pm_star), (0, time[-1]), initial,
                         t_eval=time, method="DOP853", rtol=1e-11, atol=1e-12).y.T
    actual = model.rollout(initial, model.pm_star, time)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    batched = model.rollout(np.stack([initial, initial]), np.stack([model.pm_star]*2), time)
    np.testing.assert_allclose(batched[0], actual)


def test_weak_fit_recovers_synthetic_parameters(tmp_path):
    model = fixture_model()
    events = []
    for seed in range(6):
        initial = np.random.default_rng(seed).normal(0, .2, 4)
        time = np.arange(401)*.005
        state = model.rollout(initial, model.pm_star, time)
        events.append({"t": time, "delta_sync": state[:, :2], "omega_sync": state[:, 2:],
                       "mass": model.mass, "incidence": model.incidence,
                       "equilibrium_delta": model.delta_star, "Pm_star": model.pm_star,
                       "Pm_raw": np.tile(model.pm_star, (len(time), 1))})
    fitted, _ = fit_swing(events, ridge=0)
    np.testing.assert_allclose(fitted.coupling, model.coupling, rtol=3e-5)
    np.testing.assert_allclose(fitted.damping, model.damping, rtol=3e-5)
    fitted.save(tmp_path / "model.npz")
    loaded = SwingROM.load(tmp_path / "model.npz")
    np.testing.assert_array_equal(loaded.coupling, fitted.coupling)
