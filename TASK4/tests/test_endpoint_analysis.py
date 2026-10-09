"""Small linear three-machine checks; no event or model-file access."""
import importlib.util

import numpy as np
import pytest

from task4.paths import ROOT


@pytest.fixture
def analysis_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('endpoint_analysis_test_module', ROOT / 'scripts/analyze_endpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def linear_three_machine():
    n, rank = 3, 2
    mass = np.array([2., 3., 5.])
    scales = np.array([.2, .3, .07])
    laplacian = np.array([[2., -2., 0.], [-2., 5., -3.], [0., -3., 3.]])
    damping = np.array([.4, .5, .8])
    rate = np.array([.6, .7, .9])
    gain = np.array([.1, .2, .3])
    q_decay = np.array([.2, .3, .4, .5, .6, .7])

    def field(state):
        delta, pi, power, q = state[:, :3], state[:, 3:6], state[:, 6:9], state[:, 9:]
        speed = pi / mass
        return np.concatenate((speed, -delta @ laplacian.T - damping * speed + power,
                               -rate * power - gain * speed, -q_decay * q), axis=1)

    # Independently write the known scaled reduced Jacobian from component
    # equations; do not construct it using the implementation's lift/reduce.
    expected = np.zeros((14, 14))
    for i in range(n - 1):
        expected[i, 2 + i] = scales[1] / scales[0]
        expected[i, 2 + n - 1] = -scales[1] / scales[0]
    for i in range(n):
        for j in range(n - 1):
            expected[2 + i, j] = -laplacian[i, j] * scales[0] / (mass[i] * scales[1])
        expected[2 + i, 2 + i] = -damping[i] / mass[i]
        expected[2 + i, 5 + i] = scales[2] / (mass[i] * scales[1])
        expected[5 + i, 2 + i] = -gain[i] * scales[1] / scales[2]
        expected[5 + i, 5 + i] = -rate[i]
    expected[8:, 8:] = -np.diag(q_decay)
    return n, rank, mass, scales, field, expected


def test_quotient_drops_only_common_angle_and_keeps_units(analysis_script):
    n, rank, mass, scales, field, _ = linear_three_machine()
    coordinates = analysis_script.QuotientCoordinates(n, rank, mass, scales)
    full = np.arange(15, dtype=float)[None] * .01
    reduced = coordinates.reduce(full)
    shifted = full.copy()
    shifted[:, :3] += .37
    np.testing.assert_allclose(coordinates.reduce(shifted), reduced, atol=1e-15)
    lifted = coordinates.lift(reduced)
    assert lifted[0, 2] == 0
    np.testing.assert_allclose(lifted[:, :3], full[:, :3] - full[:, 2:3])
    np.testing.assert_array_equal(lifted[:, 3:], full[:, 3:])
    np.testing.assert_allclose(coordinates.reduce(field(lifted)), coordinates.reduce(field(full)), atol=1e-15)
    np.testing.assert_allclose(coordinates.scale[:2], [.2, .2])
    np.testing.assert_allclose(coordinates.scale[2:5], mass * .3)
    np.testing.assert_allclose(coordinates.scale[5:8], [.07, .07, .07])
    np.testing.assert_array_equal(coordinates.scale[8:], np.ones(6))


def test_two_central_steps_match_known_linear_three_machine_jacobian(analysis_script):
    n, rank, mass, scales, field, expected = linear_three_machine()
    coordinates = analysis_script.QuotientCoordinates(n, rank, mass, scales)
    scaled = lambda x: coordinates.scaled_field(field, x)
    jacobians = []
    for step in (1e-5, 5e-6):
        result = analysis_script.central_jacobian(scaled, coordinates.dim, step)
        assert result['complete'] and result['all_finite']
        assert result['completed_columns'] == 14
        np.testing.assert_allclose(result['jacobian'], expected, rtol=1e-13, atol=1e-14)
        jacobians.append(result['jacobian'])
    np.testing.assert_allclose(jacobians[0], jacobians[1], rtol=1e-14, atol=1e-14)


def test_scaled_physical_readout_retains_common_frequency_but_not_hidden_q(analysis_script):
    n, rank, mass, scales, _, _ = linear_three_machine()
    coordinates = analysis_script.QuotientCoordinates(n, rank, mass, scales)
    readout = coordinates.physical_readout_matrices()
    common = np.zeros(coordinates.dim)
    common[2:5] = 1.0  # pi_i=mass_i*scale_omega gives identical physical speed.
    np.testing.assert_allclose(readout['relative_omega'] @ common, 0, atol=1e-15)
    np.testing.assert_allclose(readout['common_omega'] @ common, [1.0], atol=1e-15)
    full = coordinates.scaled_to_full(common)
    np.testing.assert_allclose(full[3:6] / mass, np.full(n, scales[1]))
    latent = np.zeros(coordinates.dim)
    latent[8] = 1
    for matrix in readout.values():
        np.testing.assert_array_equal(matrix @ latent, np.zeros(matrix.shape[0]))
