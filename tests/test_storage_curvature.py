"""Synthetic contracts for the Loop0069 one-scalar curvature extension."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
import check_storage_curvature as sc  # noqa: E402


def test_curvature_is_symmetric_and_mass_coi_gauge_invariant():
    p = sc.synthetic_params(); out = sc.instantaneous_storage_curvature(p)
    assert out["gd"].shape == (52, 10)
    assert out["ad"].shape == (3, 10)
    assert np.allclose(out["Ks"], out["Ks"].T)
    assert np.max(np.abs(out["Ks"] @ np.ones(10))) < 1e-12
    assert np.allclose(out["R"] @ np.ones(10), 0.)


def test_augmented_library_has_exactly_one_new_column():
    p = sc.synthetic_params(); z = np.random.default_rng(69).normal(size=(5, 32))*.1
    values, gradients, hessians = sc.augmented_library(z, p)
    assert values.shape == (5, 225)
    assert gradients.shape == (5, 225, 32)
    assert hessians.shape == (225, 32, 32)
    assert np.allclose(values[:, -1], sc.chi_value(z, p))
    assert np.allclose(gradients[:, -1], sc.chi_gradient(z, p))


def test_chi_gradient_matches_finite_difference_away_from_origin():
    p = sc.synthetic_params(); rng = np.random.default_rng(6901); z = rng.normal(size=(3, 32))*.2
    eps = 1e-6; g = sc.chi_gradient(z, p)
    for row, coord in ((0, 0), (1, 8), (2, 31)):
        up, dn = z.copy(), z.copy(); up[row, coord] += eps; dn[row, coord] -= eps
        fd = (sc.chi_value(up, p)[row] - sc.chi_value(dn, p)[row])/(2*eps)
        assert abs(fd-g[row, coord]) < 1e-8


def test_chi_origin_hessian_matches_gradient_difference():
    p = sc.synthetic_params(); eps = 2e-5; j = 5
    up = (eps*np.eye(32)[j])[None]; dn = (-eps*np.eye(32)[j])[None]
    fd = (sc.chi_gradient(up, p)[0]-sc.chi_gradient(dn, p)[0])/(2*eps)
    assert np.allclose(fd, sc.chi_hessian_origin(p)[:, j], atol=1e-10, rtol=1e-10)


def test_synthetic_checks_are_data_free_and_finite():
    result = sc.synthetic_checks()
    assert result["library_shape"] == [4, 225]
    assert result["Ks_symmetric"]
    assert result["Ks_gauge_max"] < 1e-12
    assert result["chi_gradient_fd_max"] < 1e-8
    assert result["chi_hessian_fd_max"] < 1e-9
