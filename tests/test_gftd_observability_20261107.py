import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from gftd_observability_20261107 import condition_on_endpoint, displaced


def test_endpoint_conditioning_enforces_exact_linear_constraint():
    rng = np.random.default_rng(1107)
    jh = rng.normal(size=(20, 10))
    jp = np.column_stack((rng.normal(size=(3, 7)), np.diag([1.0, 2.0, 3.0])))
    conditional, correction, _, error = condition_on_endpoint(jh, jp)
    direction = rng.normal(size=7)
    combined = np.concatenate((direction, correction @ direction))
    np.testing.assert_allclose(jp @ combined, 0, atol=1e-14)
    np.testing.assert_allclose(jh @ combined, conditional @ direction, atol=1e-14)
    assert error < 1e-14


def test_singular_endpoint_condition_is_not_silently_inverted():
    with pytest.raises(ValueError):
        condition_on_endpoint(np.ones((20, 10)), np.zeros((3, 10)))


def test_displacement_preserves_measured_initial_angles_and_speeds():
    initial = np.arange(15, dtype=float)
    changed, modulation = displaced(initial, 0.5, np.ones(10) * 1e-3)
    np.testing.assert_array_equal(changed[:6], initial[:6])
    assert modulation == 0.5005
