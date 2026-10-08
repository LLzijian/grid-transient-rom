import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kiki.models.private_pm import MechanicalROM, fit_mechanical, forced_pm
from kiki.models.private_rom import SwingROM


def test_mechanical_recovery_and_causal_rollout():
    backbone = SwingROM(np.array([1., 2.]), np.array([[1.], [-1.]]), np.array([.1, -.1]),
                         np.array([1., 2.]), np.array([2.]), np.array([.2, .3]))
    model = MechanicalROM(**vars(backbone), rate=np.array([.7, 1.2]), gain=np.array([.3, .4]))
    np.testing.assert_array_equal(model.field(np.r_[np.zeros(4), model.pm_star]), np.zeros(6))
    events = []
    t = np.arange(401)*.005
    for seed in range(6):
        initial = np.r_[np.random.default_rng(seed).normal(0, .2, 4), model.pm_star+[.1, -.1]]
        state = model.rollout(initial, None, t)
        reference = solve_ivp(lambda t, x: model.field(x), (0, t[-1]), initial, t_eval=t,
                              method="DOP853", rtol=1e-11, atol=1e-12).y.T
        np.testing.assert_allclose(state, reference, atol=1e-9)
        event = {"t": t, "Pm_raw": state[:, 4:], "Pm_star": model.pm_star, "omega_sync": state[:, 2:4]}
        events.append(event)
        np.testing.assert_allclose(forced_pm(model, event), state[:, 4:], atol=2e-6)
    fitted = fit_mechanical(events, backbone, ridge=0)
    np.testing.assert_allclose(fitted.rate, model.rate, rtol=3e-5)
    np.testing.assert_allclose(fitted.gain, model.gain, rtol=3e-5)


def test_storage_identity_and_gauge():
    model = MechanicalROM(np.array([1., 2.]), np.array([[1.], [-1.]]), np.array([.1, -.1]),
                          np.array([1., 2.]), np.array([2.]), np.array([.2, .3]),
                          np.array([.7, 1.2]), np.array([.3, .4]))
    state = np.array([.2, -.1, .1, -.2, 1.1, 1.9])
    gradient = model.energy_gradient(state)
    numeric = np.array([(model.energy(state+1e-6*v)-model.energy(state-1e-6*v))/2e-6 for v in np.eye(6)])
    np.testing.assert_allclose(gradient, numeric, atol=1e-9)
    np.testing.assert_allclose(gradient@model.field(state), model.dissipation(state), atol=1e-12)
    shifted = state.copy()
    shifted[:2] += 7
    np.testing.assert_allclose(model.energy(shifted), model.energy(state), atol=1e-12)
    model.gain[0] = 0
    with pytest.raises(ValueError, match="positive governor"):
        model.energy(state)
