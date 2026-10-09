"""Optional degree-two executor: synthetic algebra/AD checks, not speed evidence.

Dense matrix reductions reorder floating-point arithmetic. Tolerances here
cover bounded float64 inputs; they do not certify deployment trajectories or
equal overflow/underflow boundaries for arbitrary inputs.
"""
import copy

import numpy as np
import pytest
import torch

from task4.accelerate import trace_field
from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder
from task4.integrators import rollout
from task4.symbolic import PolynomialFunction, powers_library
from task4.synthetic import fixture


def scalar_pair(inputs, outputs, degree, mask):
    powers = powers_library(inputs, degree, max_candidates=1024)
    coefficients = np.random.default_rng(638).normal(0, .2, (len(powers), outputs))
    if mask == 'sparse':
        coefficients[::2, ::2] = 0.
    elif mask == 'zero':
        coefficients[:] = 0.
    reference = PolynomialFunction(powers, coefficients)
    # Inactive stored values must never enter the formula or its gradient.
    with torch.no_grad():
        reference.coefficients[~reference.support] = 37.
    fast = copy.deepcopy(reference)
    assert reference.quadratic_execution is False
    fast.set_quadratic_execution(True)
    return reference, fast


def first_and_mixed_second(module, x):
    value, gradient = module.value_gradient(x)
    weights = torch.linspace(.3, 1.1, module.outputs, dtype=x.dtype, device=x.device)
    objective = (value*weights).sum() + .13*gradient.square().sum()
    parameters = (x, module.coefficients)
    first = torch.autograd.grad(objective, parameters, create_graph=True)
    direction = sum((g*torch.linspace(-.4, .6, g.numel(), dtype=g.dtype, device=g.device).reshape(g.shape)).sum()
                    for g in first)
    second = torch.autograd.grad(direction, parameters, allow_unused=True)
    second = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(parameters, second))
    return value, gradient, first, second


@pytest.mark.parametrize('inputs,outputs', [(1, 1), (4, 3), (18, 16)])
@pytest.mark.parametrize('degree', [0, 1, 2])
@pytest.mark.parametrize('leading', [(), (1,), (8,), (2, 3)])
@pytest.mark.parametrize('mask', ['dense', 'sparse', 'zero'])
def test_values_full_input_coefficient_and_mixed_second_gradients(inputs, outputs, degree, leading, mask):
    reference, fast = scalar_pair(inputs, outputs, degree, mask)
    values = np.random.default_rng(572).normal(0, .3, (*leading, inputs))
    values[..., 0] = 0.  # Repeated-variable and mixed derivatives at zero.
    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    expected = first_and_mixed_second(reference, x)
    actual = first_and_mixed_second(fast, y)
    assert actual[0].shape == (*leading, outputs)
    assert actual[1].shape == (*leading, outputs, inputs)
    for left, right in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(left, right, rtol=3e-12, atol=3e-13)
    for left_group, right_group in zip(actual[2:], expected[2:]):
        for left, right in zip(left_group, right_group):
            assert torch.isfinite(left).all()
            torch.testing.assert_close(left, right, rtol=3e-11, atol=3e-12)
    assert torch.count_nonzero(actual[2][1][~fast.support]) == 0
    torch.testing.assert_close(fast(y), actual[0], rtol=0, atol=0)
    # Also differentiate forward itself: value_gradient must not hide a broken
    # forward input derivative behind a correct analytic expression.
    direct = torch.autograd.grad(fast(y).sum(), y, create_graph=True)[0]
    _, analytic = fast.value_gradient(y)
    torch.testing.assert_close(direct, analytic.sum(-2), rtol=3e-12, atol=3e-13)


def test_mixed_coefficient_has_no_half_factor_and_exact_known_hessian():
    # 2 + 3*x + 5*y + 7*x*x + 11*x*y + 13*y*y.
    powers = [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
    module = PolynomialFunction(powers, [[2.], [3.], [5.], [7.], [11.], [13.]])
    module.set_quadratic_execution()
    x = torch.tensor([[0., 0.], [.2, -.3]], dtype=torch.float64, requires_grad=True)
    value, gradient = module.value_gradient(x)
    expected = 2+3*x[:, 0]+5*x[:, 1]+7*x[:, 0]**2+11*x[:, 0]*x[:, 1]+13*x[:, 1]**2
    torch.testing.assert_close(value[:, 0], expected, rtol=1e-14, atol=1e-14)
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    torch.testing.assert_close(gradient[:, 0], first, rtol=1e-14, atol=1e-14)
    hessian = torch.stack([torch.autograd.grad(first[:, i].sum(), x, retain_graph=True)[0] for i in range(2)], -2)
    expected_hessian = torch.tensor([[14., 11.], [11., 26.]], dtype=torch.float64).expand(2, -1, -1)
    torch.testing.assert_close(hessian, expected_hessian, rtol=0, atol=0)


def test_mixed_subnormal_coefficient_is_not_divided_by_two():
    tiny = np.nextafter(np.float64(0), np.float64(1))
    reference = PolynomialFunction([[1, 1]], [[tiny]])
    fast = copy.deepcopy(reference).set_quadratic_execution()
    x = torch.ones(1, 2, dtype=torch.float64)
    expected, expected_gradient = reference.value_gradient(x)
    if expected.item() == 0.:
        pytest.skip('Runtime flushes subnormals even in the default reference')
    value, gradient = fast.value_gradient(x)
    torch.testing.assert_close(value, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert fast.coefficients.data_ptr() != reference.coefficients.data_ptr()
    assert fast.coefficients.item() == tiny


def test_enable_disable_preserve_parameter_support_export_and_persistent_state():
    reference, fast = scalar_pair(4, 3, 2, 'sparse')
    before = {key: value.clone() for key, value in reference.state_dict().items()}
    pointer = reference.coefficients.data_ptr()
    original_export = reference.export()
    reference.set_quadratic_execution(True)
    x = torch.randn(8, 4, dtype=torch.float64)
    torch.testing.assert_close(reference(x), fast(x), rtol=0, atol=0)
    reference.set_quadratic_execution(False)
    assert reference.coefficients.data_ptr() == pointer
    assert set(reference.state_dict()) == {'powers', 'coefficients', 'support'}
    for key, value in before.items():
        torch.testing.assert_close(reference.state_dict()[key], value, rtol=0, atol=0)
    assert reference.export() == original_export
    restored = PolynomialFunction(before['powers'], before['coefficients'])
    restored.load_state_dict(before)
    assert restored.quadratic_execution is False
    torch.testing.assert_close(restored(x), reference(x), rtol=0, atol=0)


@pytest.mark.parametrize('powers,reason', [([[0, 0], [3, 0]], 'degree'),
                                         ([[1, 1], [1, 1]], 'unique monomial')])
def test_unsupported_degree_and_duplicates_are_explicitly_rejected(powers, reason):
    module = PolynomialFunction(powers, [[1.], [-.5]])
    x = torch.tensor([[.2, -.1]], dtype=torch.float64)
    expected = module(x)
    with pytest.raises(ValueError, match=reason):
        module.set_quadratic_execution()
    assert module.quadratic_execution is False
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
    module.quadratic_execution = True  # Direct attribute use also checks eligibility.
    with pytest.raises(ValueError, match=reason):
        module.value_gradient(x)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('assign', [False, True])
def test_same_shape_changed_powers_reload_rebuilds_transient_indices(device, assign):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    before = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]]
    after = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]]
    reference = PolynomialFunction(after, [[1., 2.], [3., 4.], [5., 6.], [7., 8.]]).to(device)
    fast = PolynomialFunction(before, np.ones((4, 2))).to(device).set_quadratic_execution()
    fast.load_state_dict(reference.state_dict(), assign=assign)
    assert fast.quadratic_execution is True
    assert all(buffer.device.type == device for buffer in fast.buffers())
    x = torch.randn(2, 3, 3, dtype=torch.float64, device=device)
    for left, right in zip(fast.value_gradient(x), reference.value_gradient(x)):
        torch.testing.assert_close(left, right, rtol=3e-12, atol=3e-13)
    assert set(fast.state_dict()) == set(reference.state_dict())
    # A later cubic load cannot silently run through stale quadratic indices.
    changed = {key: value.clone() for key, value in reference.state_dict().items()}
    changed['powers'][2, 2] = 3
    fast.load_state_dict(changed, assign=assign)
    with pytest.raises(ValueError, match='degree'):
        fast(x)
    fast.set_quadratic_execution(False)
    reference.load_state_dict(changed)
    torch.testing.assert_close(fast(x), reference(x), rtol=0, atol=0)


def compare_parameter_gradients(left, right):
    original = dict(left.named_parameters())
    candidate = dict(right.named_parameters())
    assert original.keys() == candidate.keys()
    for name, parameter in original.items():
        other = candidate[name]
        expected = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        actual = torch.zeros_like(other) if other.grad is None else other.grad
        assert torch.isfinite(actual).all(), name
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-11, msg=name)


def graph_pair(symbolic):
    _, physics = fixture(events=1, steps=12)
    torch.manual_seed(926)
    reference = PowerGraph(physics, [.2, .3, .1], rank=2, hidden=8)
    if symbolic:
        rng = np.random.default_rng(101)
        for name, module in list(reference.function_modules().items()):
            powers = powers_library(module.inputs, degree=2, max_candidates=1024)
            coefficients = rng.normal(0, .004, (len(powers), module.outputs))
            coefficients[::3, ::2] = 0.
            setattr(reference, name, PolynomialFunction(powers, coefficients))
    reference.checkpoint_steps = 0
    fast = copy.deepcopy(reference)
    for module in fast.function_modules().values():
        if isinstance(module, PolynomialFunction):
            module.set_quadratic_execution()
    fast.reuse_origin_terms = True
    return reference, fast


@pytest.mark.parametrize('symbolic', [False, True])
@pytest.mark.parametrize('batch', [1, 8])
def test_complete_field_trace_origin_reuse_and_every_parameter_and_history_gradient(symbolic, batch):
    reference, fast = graph_pair(symbolic)
    snapshot = {key: value.clone() for key, value in fast.state_dict().items()}
    pointers = {name: parameter.data_ptr() for name, parameter in fast.named_parameters()}
    trace_field(fast)
    assert fast._compiled_uses_origins
    for name, parameter in fast._compiled_field.named_parameters():
        assert parameter.data_ptr() == pointers[name.removeprefix('model.')]
    assert fast.state_dict().keys() == snapshot.keys()
    for key, value in snapshot.items():
        torch.testing.assert_close(fast.state_dict()[key], value, rtol=0, atol=0)
    encoder = HistoryEncoder(reference, hidden=8)
    candidate_encoder = copy.deepcopy(encoder)
    h = (torch.randn(batch, 11, reference.n, 3, dtype=torch.float64)*.03).requires_grad_()
    g = h.detach().clone().requires_grad_()
    x, y = reference.initial(h, encoder), fast.initial(g, candidate_encoder)
    x.retain_grad()
    y.retain_grad()
    expected, actual = reference.field(x), fast.field(y, origins=fast.origin_terms(y))
    expected_gradient, actual_gradient = reference.energy_gradient(x), fast.energy_gradient(y)
    expected_energy, actual_energy = reference.energy(x), fast.energy(y)
    for left, right in ((actual, expected), (actual_gradient, expected_gradient), (actual_energy, expected_energy)):
        torch.testing.assert_close(left, right, rtol=3e-10, atol=3e-12)
    (expected.square().sum()+.1*expected_gradient.square().sum()+.1*expected_energy.square().sum()).backward()
    (actual.square().sum()+.1*actual_gradient.square().sum()+.1*actual_energy.square().sum()).backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=2e-9, atol=2e-11)
    torch.testing.assert_close(g.grad, h.grad, rtol=2e-9, atol=2e-11)
    compare_parameter_gradients(reference, fast)
    compare_parameter_gradients(encoder, candidate_encoder)
    if symbolic:
        for module in fast.function_modules().values():
            assert torch.count_nonzero(module.coefficients.grad[~module.support]) == 0
            with pytest.raises(ValueError, match='before tracing'):
                module.set_quadratic_execution(False)


def test_two_adam_updates_keep_current_coefficients_support_and_trace_aliases():
    reference, fast = graph_pair(symbolic=True)
    trace_field(fast)
    encoder = HistoryEncoder(reference, hidden=8)
    candidate_encoder = copy.deepcopy(encoder)
    optimizer = torch.optim.Adam(list(reference.parameters())+list(encoder.parameters()), lr=1e-4)
    candidate_optimizer = torch.optim.Adam(list(fast.parameters())+list(candidate_encoder.parameters()), lr=1e-4)
    pointers = {name: parameter.data_ptr() for name, parameter in fast.named_parameters()}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)
        history = torch.randn(2, 11, reference.n, 3, dtype=torch.float64)*.04
        expected = rollout(reference, reference.initial(history, encoder), 2, max_step=.005)
        actual = rollout(fast, fast.initial(history, candidate_encoder), 2, max_step=.005)
        torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-12)
        expected.square().mean().backward()
        actual.square().mean().backward()
        compare_parameter_gradients(reference, fast)
        compare_parameter_gradients(encoder, candidate_encoder)
        optimizer.step()
        candidate_optimizer.step()
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(fast.state_dict()[key], value, rtol=3e-9, atol=3e-11)
        for name, parameter in fast._compiled_field.named_parameters():
            assert parameter.data_ptr() == pointers[name.removeprefix('model.')]
        for module in fast.function_modules().values():
            assert torch.count_nonzero(module.coefficients[~module.support]) == 0
    compare_parameter_gradients(encoder, candidate_encoder)


def test_generic_trace_cannot_silently_change_executor_mode():
    module = PolynomialFunction([[0, 0], [1, 1]], [[1.], [.3]])
    traced = torch.jit.trace(module, torch.ones(1, 2, dtype=torch.float64), check_trace=False)
    assert module.quadratic_execution is False
    with pytest.raises(ValueError, match='before tracing'):
        module.set_quadratic_execution()
    torch.testing.assert_close(traced(torch.ones(8, 2, dtype=torch.float64)), module(torch.ones(8, 2, dtype=torch.float64)))
