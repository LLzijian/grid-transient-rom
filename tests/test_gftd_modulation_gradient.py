"""Independent regression for the previously clipped energy kernel."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kiki.physics.kuramoto_functional import StructuralGates, latent_coupling_modulation


@pytest.mark.parametrize("exponent", [-42.0, -20.1, -0.4, 19.9, 20.1, 42.0])
def test_modulation_derivative_matches_actual_kernel(exponent):
    gates = StructuralGates.overcomplete(1, 2)
    gamma = np.array([1.0, 0.7])
    state = np.array([exponent, 0.0])
    value, di, dj = latent_coupling_modulation(state, ((0, 1),), gamma, gates)
    assert value[0] > 0.0
    for node, derivative in enumerate((di, dj)):
        shift = np.eye(2)[node] * 1e-4
        high = latent_coupling_modulation(state + shift, ((0, 1),), gamma, gates)[0]
        low = latent_coupling_modulation(state - shift, ((0, 1),), gamma, gates)[0]
        np.testing.assert_allclose((high - low) / 2e-4, derivative, rtol=1e-7, atol=0.0)


def test_modulation_rejects_nonfinite_and_exponential_overflow():
    gates = StructuralGates.overcomplete(1, 2)
    with pytest.raises(ValueError):
        latent_coupling_modulation(np.array([np.nan, 0.0]), ((0, 1),), np.ones(2), gates)
    for exponent in (-1000.0, 1000.0):
        with pytest.raises(FloatingPointError):
            latent_coupling_modulation(np.array([exponent, 0.0]), ((0, 1),), np.ones(2), gates)
