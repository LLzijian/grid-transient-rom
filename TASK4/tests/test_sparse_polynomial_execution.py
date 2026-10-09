"""Compiled sparse execution versus independent power/product expressions."""
import numpy as np
import pytest
import torch

from task4.symbolic import PolynomialFunction
from task4.symbolic_runtime import _Polynomial


def example():
    # Include constants, repeated variables, absent variables and degree >3.
    powers = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0], [0, 3, 0, 0, 0], [1, 1, 1, 0, 0],
                       [0, 0, 1, 2, 0], [5, 0, 0, 1, 0], [0, 0, 2, 0, 0]], dtype=np.int64)
    coefficients = np.random.default_rng(901).normal(0, 0.3, (len(powers), 3))
    coefficients[0, 2] = 0
    coefficients[3, 1] = 0
    coefficients[5, 1] = 0
    return powers, coefficients


def reference_value(x, powers, coefficients, support):
    return (x.unsqueeze(-2)**powers).prod(-1)@(coefficients*support)


def reference_gradient(x, powers, coefficients, support):
    result = []
    for d in range(powers.shape[1]):
        reduced = powers.clone()
        reduced[:, d] = (reduced[:, d]-1).clamp_min(0)
        atoms = (x.unsqueeze(-2)**reduced).prod(-1)*powers[:, d]
        result.append(atoms@(coefficients*support))
    return torch.stack(result, dim=-1)


@pytest.mark.parametrize('leading', [(), (4,), (2, 3)])
def test_compiled_values_input_and_coefficient_gradients_match_power_formula(leading):
    powers, coefficients = example()
    model = PolynomialFunction(powers, coefficients)
    x = torch.tensor(np.random.default_rng(52).normal(0, 0.4, (*leading, powers.shape[1])),
                     dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        x[..., 0] = 0
        x[..., 1] = 0
    independent_x = x.detach().clone().requires_grad_(True)
    independent_coefficients = model.coefficients.detach().clone().requires_grad_(True)
    expected = reference_value(independent_x, model.powers, independent_coefficients, model.support)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
    weights = torch.tensor([0.5, -0.3, 0.8], dtype=torch.float64)
    actual_grad = torch.autograd.grad((actual*weights).sum(), (x, model.coefficients))
    expected_grad = torch.autograd.grad((expected*weights).sum(), (independent_x, independent_coefficients))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=2e-12, atol=2e-13)
    value, gradient = model.value_gradient(x)
    torch.testing.assert_close(value, actual, rtol=0, atol=0)
    torch.testing.assert_close(gradient, reference_gradient(x, model.powers, model.coefficients, model.support),
                               rtol=2e-12, atol=2e-13)


def direct_hessian(x, powers, coefficients):
    """Independent closed monomial differentiation, using pow/product at zeros."""
    out = np.zeros((*x.shape[:-1], powers.shape[1], powers.shape[1]))
    for first in range(powers.shape[1]):
        for second in range(powers.shape[1]):
            for p, coefficient in zip(powers, coefficients):
                degree = p.copy()
                if degree[first] == 0:
                    continue
                factor = degree[first]
                degree[first] -= 1
                if degree[second] == 0:
                    continue
                factor *= degree[second]
                degree[second] -= 1
                out[..., first, second] += factor*coefficient*np.prod(x**degree, axis=-1)
    return out


def test_zero_input_second_derivatives_and_derivative_coefficient_gradients():
    powers, coefficients = example()
    model = PolynomialFunction(powers, coefficients)
    x = torch.tensor([[0., 0., 0.2, -0.3, 0.], [-0.5, 0., 0.4, 0., 0.2]],
                     dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.5, -0.3, 0.8], dtype=torch.float64)
    value, analytic = model.value_gradient(x)
    first = torch.autograd.grad((value*weights).sum(), x, create_graph=True)[0]
    columns = [torch.autograd.grad(first[..., d].sum(), x, retain_graph=True)[0]
               for d in range(x.shape[-1])]
    actual_hessian = torch.stack(columns, dim=-2)
    scalar_coefficients = (model.coefficients.detach()*model.support)@weights
    expected_hessian = direct_hessian(x.detach().numpy(), powers, scalar_coefficients.numpy())
    assert torch.isfinite(actual_hessian).all()
    np.testing.assert_allclose(actual_hessian.detach().numpy(), expected_hessian, rtol=2e-12, atol=2e-13)
    # Analytic derivative execution must itself be differentiable for joint ODE
    # training, including derivatives with respect to sparse scalar coefficients.
    independent_x = x.detach().clone().requires_grad_(True)
    independent_coefficients = model.coefficients.detach().clone().requires_grad_(True)
    reference = reference_gradient(independent_x, model.powers, independent_coefficients, model.support)
    actual_grad = torch.autograd.grad(analytic.square().sum(), (x, model.coefficients))
    reference_grad = torch.autograd.grad(reference.square().sum(), (independent_x, independent_coefficients))
    for actual, expected in zip(actual_grad, reference_grad):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=3e-12, atol=3e-13)


def test_degree_three_execution_uses_at_most_three_factors_per_term():
    powers = np.zeros((5, 18), dtype=np.int64)
    powers[1, 16] = 1
    powers[2, 4] = 2
    powers[3, [0, 7, 17]] = 1
    powers[4, 5] = 3
    model = PolynomialFunction(powers, np.ones((5, 2)))
    assert model._factor_indices.shape == (5, 3)
    assert model._factor_mask.sum(dim=-1).tolist() == [0, 1, 2, 3, 3]
    assert getattr(model, '_derivative_rows_2').numel() == 0
    assert getattr(model, '_derivative_indices_5').shape == (1, 2)


def test_support_does_not_revive_and_numpy_export_stays_independent():
    powers, coefficients = example()
    model = PolynomialFunction(powers, coefficients)
    with torch.no_grad():
        model.coefficients[~model.support] = 123.0
    x = torch.tensor(np.random.default_rng(914).normal(0, 0.3, (2, 3, powers.shape[1])),
                     dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        x[..., :2] = 0
    value, gradient = model.value_gradient(x)
    objective = value.square().sum()+gradient.square().sum()
    objective.backward()
    assert torch.count_nonzero(model.coefficients.grad[~model.support]) == 0
    exported = model.export()
    assert np.all(np.asarray(exported['coefficients'])[~model.support.numpy()] == 0)
    numpy_model = _Polynomial(exported, inputs=model.inputs, outputs=model.outputs)
    nv, ng = numpy_model.value_gradient(x.detach().numpy())
    np.testing.assert_allclose(nv, reference_value(x, model.powers, model.coefficients, model.support).detach().numpy(),
                               rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(ng, reference_gradient(x, model.powers, model.coefficients, model.support).detach().numpy(),
                               rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(numpy_model.value(x.detach().numpy()), nv, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('assign', [False, True])
def test_state_dict_rebuilds_same_shape_changed_powers_and_moves_buffers(device, assign):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    powers, coefficients = example()
    model = PolynomialFunction(powers, coefficients).to(device)
    assert set(model.state_dict()) == {'powers', 'coefficients', 'support'}
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state['powers'].zero_()
    state['powers'][0, 3] = 7
    state['powers'][1, 4] = 2
    state['powers'][2, 0] = 1
    state['powers'][2, 4] = 1
    model.load_state_dict(state, assign=assign)
    assert model._factor_indices.shape[1] == 7
    assert getattr(model, '_derivative_rows_1').numel() == 0
    assert getattr(model, '_derivative_rows_4').numel() == 2
    assert all(buffer.device.type == device for _, buffer in model.named_buffers())
    x = torch.linspace(-0.3, 0.4, 10, dtype=torch.float64, device=device).reshape(2, 5)
    actual, gradient = model.value_gradient(x)
    torch.testing.assert_close(actual, reference_value(x, model.powers, model.coefficients, model.support),
                               rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(gradient, reference_gradient(x, model.powers, model.coefficients, model.support),
                               rtol=2e-12, atol=2e-13)
    # Buffers are looked up dynamically rather than cached in stale tensor lists.
    model = model.cpu()
    assert all(buffer.device.type == 'cpu' for _, buffer in model.named_buffers())
    torch.testing.assert_close(model(x.cpu()), actual.cpu(), rtol=2e-12, atol=2e-13)
    assert set(model.state_dict()) == {'powers', 'coefficients', 'support'}


def test_constant_only_polynomial_keeps_exact_zero_derivatives():
    model = PolynomialFunction([[0, 0, 0]], [[2.0, -1.0]])
    x = torch.zeros((4, 3), dtype=torch.float64, requires_grad=True)
    value, gradient = model.value_gradient(x)
    torch.testing.assert_close(value, torch.tensor([[2., -1.]], dtype=torch.float64).expand(4, -1), rtol=0, atol=0)
    assert torch.count_nonzero(gradient) == 0
    np_model = _Polynomial(model.export(), inputs=3, outputs=2)
    nv, ng = np_model.value_gradient(x.detach().numpy())
    np.testing.assert_array_equal(nv, value.detach().numpy())
    np.testing.assert_array_equal(ng, np.zeros((4, 2, 3)))
