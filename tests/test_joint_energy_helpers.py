"""Synthetic, data-free contracts for Loop0073 energy helpers."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
import joint_energy_helpers as eh  # noqa: E402


class _SyntheticField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.Module()
        self.gru.weight_ih_l0 = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
        self.gru.weight_hh_l0 = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
        self.gru.bias_ih_l0 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.gru.bias_hh_l0 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.head = torch.nn.Module()
        self.head.weight = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
        self.head.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.c = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
        self.amplitude = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.alpha = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
        self.xi = torch.nn.Parameter(torch.tensor([.2], dtype=torch.float64))
        self.decoder = torch.nn.Module()
        self.decoder.weight = torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float64))
        self.decoder.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.rank = 3

    def field(self, state):
        first = self.xi[0] * state[..., :1] + .2 * state[..., 1:2]
        return torch.cat((first, -.5 * state[..., 1:]), dim=-1)


def test_exact_j32_is_full_field_jacobian_quotiented_by_p_e():
    model = _SyntheticField()
    actual = eh.exact_j32(model, create_graph=False).detach().numpy()
    A = np.diag(np.full(33, -.5)); A[0, 0] = .2; A[0, 1] = .2
    assert np.allclose(actual, eh.sparse.P32 @ A @ eh.sparse.E33)


def test_energy_scales_and_tie_support_are_fixed_and_deterministic():
    h0, hs, v, j = eh.synthetic_matrices()
    scales = eh.storage_scales(hs, v)
    assert scales.shape == (224,)
    assert np.all(scales >= 1e-8)
    support = eh.select_energy_support(np.zeros(224), np.ones(224))
    assert support.sum() == 24
    assert np.all(support[:6])
    assert np.array_equal(np.flatnonzero(support[6:]), np.arange(18))


def test_energy_penalty_gradient_matches_finite_difference_through_j32():
    model = _SyntheticField(); h0, hs, v, _ = eh.synthetic_matrices(7)
    gamma = torch.zeros(224, dtype=torch.float64)
    loss, eig, _ = eh.energy_lh(model, gamma, h0, hs, v)
    (gradient,) = torch.autograd.grad(loss, (model.xi,), retain_graph=True)
    eps = 1e-6
    with torch.no_grad(): model.xi[0] += eps
    plus = float(eh.energy_lh(model, gamma, h0, hs, v)[0].detach())
    with torch.no_grad(): model.xi[0] -= 2*eps
    minus = float(eh.energy_lh(model, gamma, h0, hs, v)[0].detach())
    with torch.no_grad(): model.xi[0] += eps
    assert np.isfinite(float(loss.detach())) and np.isfinite(float(eig.min().detach()))
    assert abs(float(gradient[0]) - (plus-minus)/(2*eps)) < 1e-6


def test_lambda_calibration_uses_only_declared_twelve_tensors():
    model = _SyntheticField()
    data = model.xi.square().sum()
    energy = (model.xi - .4).square().sum()
    lam, gl, gh = eh.calibrate_lambda(data, energy, model)
    assert np.isfinite(lam) and gl > 1e-12 and gh > 1e-12
    assert np.isclose(lam, .1*gl/gh)


def test_calibration_failcloses_nonpositive_energy_gradient():
    model = _SyntheticField()
    try:
        eh.calibrate_lambda(model.xi.square().sum(), torch.zeros((), dtype=torch.float64), model)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("zero energy gradient must fail closed")


def test_synthetic_clarabel_solve_and_independent_residuals():
    h0 = np.eye(32); hs = np.zeros((224, 32, 32)); v = np.eye(32); j = -np.eye(32)
    result = eh.solve_soft_storage(h0, hs, v, j)
    assert result["feasible"]
    assert result["residual_h"] <= 1e-6
    assert result["residual_d"] <= 1e-6
    check = eh.validate_soft_solution(h0, hs, v, j, result["gamma"], result["t"], result["scales"])
    assert check["valid"]


def test_noncompliant_residual_is_rejected_without_threshold_rescue():
    h0 = np.eye(32)*1e-12; hs = np.zeros((224, 32, 32)); v = np.eye(32); j = -np.eye(32)
    check = eh.validate_soft_solution(h0, hs, v, j, np.zeros(224), 0., np.ones(224))
    assert not check["valid"]
    assert check["residual_h"] > 1e-6


def test_zero_field_helper_is_strict():
    assert eh.assert_zero_field(_SyntheticField()) <= 1e-10
