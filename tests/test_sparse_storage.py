"""Synthetic contracts for the Loop0068 sparse-storage library only."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
import fit_sparse_storage as fs  # noqa: E402


def test_library_has_registered_size_and_pq_prefix():
    p = fs.synthetic_params(); z = np.zeros((2, fs.N))
    values = fs.library_values(z, p)
    assert values.shape == (2, 224)
    assert fs.library_names()[:6] == ["0.5*q0^2", "0.5*q1^2", "0.5*q2^2", "q0*q1", "q0*q2", "q1*q2"]
    assert np.array_equal(values, np.zeros_like(values))


def test_library_gradient_is_finite_difference_away_from_origin():
    p = fs.synthetic_params(); rng = np.random.default_rng(6801)
    z = rng.normal(size=(3, fs.N)) * .2
    g = fs.library_gradients(z, p)
    eps = 1e-6
    for row, coord in ((0, 0), (1, 7), (2, 31)):
        up, dn = z.copy(), z.copy(); up[row, coord] += eps; dn[row, coord] -= eps
        fd = (fs.library_values(up, p)[row] - fs.library_values(dn, p)[row]) / (2 * eps)
        assert np.max(np.abs(fd - g[row, :, coord])) < 3e-7


def test_origin_hessians_match_gradient_difference():
    p = fs.synthetic_params(); z = np.zeros(fs.N); eps = 2e-5
    hs = fs.library_hessians_origin(p)
    gp = fs.library_gradients((z + eps*np.eye(fs.N)[3])[None], p)[0]
    gm = fs.library_gradients((z - eps*np.eye(fs.N)[3])[None], p)[0]
    assert np.allclose((gp - gm)/(2*eps), hs[:, :, 3], atol=3e-8, rtol=3e-8)


def test_h3_restores_incremental_pm_and_has_exact_gradient():
    p = fs.synthetic_params(); rng = np.random.default_rng(9); z = rng.normal(size=fs.N)*.1
    eps = 1e-6; j = 11
    up, dn = z.copy(), z.copy(); up[j] += eps; dn[j] -= eps
    fd = (fs.h3_value(up, p)-fs.h3_value(dn, p))/(2*eps)
    assert np.allclose(fd, fs.h3_gradient(z, p)[j], atol=1e-8, rtol=1e-8)
    p2 = dict(p); p2["pm_star"] = p["pm_star"] + 7.
    # The incremental API is invariant to the absolute equilibrium PM value.
    assert np.allclose(fs.h3_value(z, p), fs.h3_value(z, p2))


def test_affine_ray_intersection_and_probe_domain():
    p = fs.synthetic_params(); s = np.ones(fs.N); A = np.zeros((1, fs.N)); A[0, 0] = 1.
    off = np.array([.5]); d = np.zeros(fs.N); d[0] = 1.
    assert np.isclose(fs.ray_intersection(d, s*10, A, off), np.pi - .5)
    interior, boundary = fs.domain_probes(s, A, n_interior=32, n_boundary=16, angle_offset=off)
    assert interior.shape == (32, fs.N) and boundary.shape == (16, fs.N)
    assert np.all(np.abs(interior) <= 1 + 1e-12)
    assert np.all(np.abs(boundary @ A.T + off) <= np.pi + 1e-10)


def test_cumulative_prefix_is_per_event_and_uses_registered_scales():
    c = np.zeros((2, 4, 33)); f = c.copy(); mass = np.ones(10)
    c[0, 1, 0] = .0005; c[1, 1, 0] = .01
    mask = fs.cumulative_prefix_ok(c, f, mass)
    assert mask.shape == (2, 4)
    assert mask[0, 1] and not mask[1, 1]
    assert mask[0, 0] and mask[1, 0]


def test_synthetic_smoke_does_not_open_private_payloads():
    result = fs.synthetic_smoke()
    assert result["library_shape"] == [32, 224]
    assert result["derivative_max"] < 1e-7
