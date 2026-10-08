"""Synthetic Loop0073 storage and calibration contracts; no TRAIN/H5/TEST access."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import fit_sparse_storage as sparse  # noqa: E402
from joint_energy_helpers import (  # noqa: E402
    calibrate_lambda, energy_lh_from_j32, select_energy_support,
    solve_soft_storage, storage_scales, synthetic_matrices,
)
from train_joint_campaign import JOINT, STEPS, horizon_for_step  # noqa: E402
from train_joint_energy import FINAL_LIMIT, LIBRARY_SIZE, _storage_context  # noqa: E402


def test_support_selection_keeps_pq_and_ascending_ties():
    coefficients = np.zeros(224)
    coefficients[6:] = 1.0
    scales = np.ones(224)
    support = select_energy_support(coefficients, scales, limit=18)
    assert np.flatnonzero(support).tolist() == list(range(24))
    assert np.all(support[:6])


def test_synthetic_clarabel_solve_has_finite_residuals():
    h0, hks, v, j = synthetic_matrices(73)
    result = solve_soft_storage(h0, hks, v, j, scales=storage_scales(hks, v))
    assert result["feasible"]
    assert result["gamma"].shape == (224,)
    assert result["residual_h"] <= 1e-6
    assert result["residual_d"] <= 1e-6
    assert result["inactive_max"] == 0.0


def test_energy_penalty_gradient_matches_finite_difference():
    h0, hks, v, _ = synthetic_matrices(74)
    j = torch.eye(32, dtype=torch.float64) * .5
    gamma = torch.zeros(224, dtype=torch.float64, requires_grad=True)
    loss, _, _ = energy_lh_from_j32(j, gamma, h0, hks, v)
    loss.backward()
    index = 11
    eps = 1e-6
    plus = gamma.detach().clone(); plus[index] += eps
    minus = gamma.detach().clone(); minus[index] -= eps
    fp = energy_lh_from_j32(j, plus, h0, hks, v)[0].item()
    fm = energy_lh_from_j32(j, minus, h0, hks, v)[0].item()
    finite_difference = (fp - fm) / (2 * eps)
    assert np.isfinite(finite_difference)
    assert abs(float(gamma.grad[index]) - finite_difference) <= 1e-5


def test_calibration_is_fixed_ratio_and_rejects_zero_gradient():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.Module()
            self.gru.weight_ih_l0 = torch.nn.Parameter(torch.ones(2, 2))
            self.gru.weight_hh_l0 = torch.nn.Parameter(torch.ones(2, 2))
            self.gru.bias_ih_l0 = torch.nn.Parameter(torch.ones(2))
            self.gru.bias_hh_l0 = torch.nn.Parameter(torch.ones(2))
            self.head = torch.nn.Module()
            self.head.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.head.bias = torch.nn.Parameter(torch.ones(2))
            self.c = torch.nn.Parameter(torch.ones(2))
            self.amplitude = torch.nn.Parameter(torch.ones(1))
            self.alpha = torch.nn.Parameter(torch.ones(2, 2))
            self.xi = torch.nn.Parameter(torch.ones(2, 2))
            self.decoder = torch.nn.Module()
            self.decoder.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.decoder.bias = torch.nn.Parameter(torch.ones(2))

    model = Tiny()
    values = [parameter for _, parameter in model.named_parameters()]
    data_loss = sum(parameter.square().sum() for parameter in values)
    energy_loss = sum(parameter.square().sum() for parameter in values) * 4.
    result = calibrate_lambda(data_loss, energy_loss, model)
    assert np.isclose(result[0], .025)
    assert result[1] > 1e-12 and result[2] > 1e-12
    with pytest.raises(FloatingPointError):
        calibrate_lambda(data_loss * 0., energy_loss, model)


def test_storage_context_uses_fixed_registered_dimensions_on_synthetic_params():
    context = _storage_context(sparse.synthetic_params(75))
    assert context["H0"].shape == (32, 32)
    assert context["Hks"].shape == (224, 32, 32)
    assert context["W"].shape == (32, 32)
    assert context["V"].shape == (32, 32)
    assert context["scales"].shape == (224,)
    assert np.isfinite(context["scales"]).all()


def test_runner_schedule_and_registered_joint_contract():
    assert STEPS == 800 and LIBRARY_SIZE == 224 and FINAL_LIMIT == 24
    assert horizon_for_step(1) == 2.0
    assert horizon_for_step(100) == 2.0
    assert [horizon_for_step(step) for step in (101, 102, 103, 104)] == [1.0, 2.0, 5.0, 1.0]
    assert len(JOINT) == 12 and len(set(JOINT)) == 12
