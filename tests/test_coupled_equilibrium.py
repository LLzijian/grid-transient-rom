"""Synthetic correctness tests for the coupled-equilibrium audit."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))

from audit_coupled_equilibrium import analytic_jacobian, block_participation, gauge_maps, storage_probe  # noqa: E402
from audit_latent_order_rollout import rhs  # noqa: E402
from kiki.models.private_pm import MechanicalROM  # noqa: E402
from test_long_latent_equations import parameters  # noqa: E402


def _p():
    p = parameters()
    p["delta_star"] = np.linspace(-0.2, 0.3, 10)
    p["alpha"] = np.array([[0.2, -0.1, 0.3], [-0.15, 0.05, 0.1]])
    p["xi"] = np.zeros((26, 3))
    p["xi"][1:4] = np.array([[0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.3]])
    p["xi"][4:14] = np.arange(30.0).reshape(10, 3) * 0.01
    p["xi"][14:24] = np.arange(30.0).reshape(10, 3) * -0.008
    p["xi"][24:] = np.array([[0.1, 0.0, -0.1], [0.0, 0.2, 0.05]])
    p["feature_scale"] = np.linspace(.7, 1.4, 22)
    p["support"] = np.ones((26, 3)); p["support"][7, 1] = 0.0
    p["alpha_support"][1, 2] = 0.0
    return p


def test_analytic_jacobian_matches_central_difference_and_signs():
    p = _p()
    analytic = analytic_jacobian(p)
    x0 = np.zeros(33)
    fd = np.column_stack([(rhs((x0 + 1e-6 * np.eye(33)[i])[None], p)[0] - rhs((x0 - 1e-6 * np.eye(33)[i])[None], p)[0]) / 2e-6 for i in range(33)])
    assert np.allclose(analytic, fd, atol=1e-8, rtol=1e-8)
    assert np.allclose(analytic[10:20, 30:], -p["Cq"] / p["mass"][:, None])
    assert np.allclose(analytic[20:30, 10:20], -np.diag(p["gain"]))


def test_angle_gauge_quotient_keeps_common_frequency():
    p = _p(); J = analytic_jacobian(p); P, E = gauge_maps()
    g = np.r_[np.ones(10), np.zeros(23)]
    Jbar = P @ J @ E
    assert np.allclose(J @ g, 0.0, atol=1e-10)
    assert np.allclose(P @ E, np.eye(32))
    assert np.allclose(P @ J, Jbar @ P)
    assert np.array_equal(np.flatnonzero(E[:, 9]), np.array([10]))


def test_complex_block_participation_is_per_eigenvalue_and_q_similarity_invariant():
    rng = np.random.default_rng(65)
    blocks = []
    for k in range(16):
        blocks.append(np.array([[-0.1 - 0.01 * k, -(1 + .1 * k)], [1 + .1 * k, -0.1 - .01 * k]]))
    A = np.zeros((32, 32))
    for k, block in enumerate(blocks): A[2*k:2*k+2, 2*k:2*k+2] = block
    vals, _, _, records = block_participation(A)
    assert len(records) == 32
    assert all("index" in record and "eigenvalue" in record for record in records)
    assert all(abs(sum(complex(*entry) for entry in record["block_sums"]) - 1) < 1e-8 for record in records)
    T = rng.normal(size=(3, 3)); T += 4 * np.eye(3)
    S = np.eye(32); S[29:, 29:] = T
    vals2, _, _, records2 = block_participation(S @ A @ np.linalg.inv(S))
    assert np.allclose(np.sort_complex(vals), np.sort_complex(vals2))
    order = np.argsort(np.asarray([complex(*record["eigenvalue"]) for record in records]))
    order2 = np.argsort(np.asarray([complex(*record["eigenvalue"]) for record in records2]))
    assert np.allclose([records[i]["normalized_magnitudes"] for i in order], [records2[i]["normalized_magnitudes"] for i in order2])
    assert np.allclose([records[i]["block_sums"] for i in order], [records2[i]["block_sums"] for i in order2])


def test_known_hurwitz_lyapunov_fixture_has_correct_sign():
    A = np.array([[-1.0, 2.0], [-3.0, -4.0]])
    P = solve_continuous_lyapunov(A.T, -np.eye(2)); P = (P + P.T) / 2
    assert np.linalg.eigvalsh(P).min() > 0
    assert np.allclose(A.T @ P + P @ A, -np.eye(2), atol=1e-10)


def test_storage_probe_uses_incremental_p_with_pm_star_and_exact_work_identity():
    import torch

    p = _p(); p["damping"] = np.linspace(.1, .2, 10); p["rate"] = np.linspace(.2, .4, 10); p["gain"] = np.linspace(.5, 1.0, 10)
    p["pm_star"] = np.linspace(1.0, 2.0, 10)
    base = MechanicalROM(mass=p["mass"], incidence=p["incidence"], delta_star=p["delta_star"], pm_star=p["pm_star"], coupling=p["coupling"], damping=p["damping"], rate=p["rate"], gain=p["gain"])
    original_gradient = base.energy_gradient
    def recording_gradient(values):
        base.last = np.asarray(values)
        return original_gradient(values)
    base.energy_gradient = recording_gradient
    class Model:
        def field(self, states):
            x = states.detach().cpu().numpy()
            return torch.as_tensor(rhs(x, p), dtype=torch.float64)
    result = storage_probe(Model(), base, p)
    assert np.array_equal(result["states"][1:], .1 * np.random.default_rng(65).normal(size=(256, 33)))
    assert np.array_equal(result["algebraic_a"][0], np.zeros(3))
    assert np.allclose(result["dotH"], result["identity_rhs"], atol=1e-10)
    assert np.array_equal(base.last[:, 20:30], result["states"][:, 20:30] + base.pm_star)
