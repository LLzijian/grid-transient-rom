"""Optional origin reuse must preserve complete forward/backward semantics."""
import copy

import numpy as np
import pytest
import torch

from task4.accelerate import trace_field
from task4.dynamics import PowerGraph, UnstructuredGraph
from task4.encoder import HistoryEncoder
from task4.integrators import rollout
from task4.symbolic import PolynomialFunction
from task4.synthetic import fixture


def model_pair(symbolic=False, traced=False):
    _, physics = fixture(events=1, steps=12)
    torch.manual_seed(845)
    baseline = PowerGraph(physics, [0.2, 0.3, 0.1], hidden=8, rank=2)
    if symbolic:
        rng = np.random.default_rng(932)
        for name, module in list(baseline.function_modules().items()):
            powers = [np.zeros(module.inputs, dtype=np.int64)]
            for d in range(module.inputs):
                power = np.zeros(module.inputs, dtype=np.int64)
                power[d] = 1
                powers.append(power)
            for a, b in [(0, 0), (0, 1), (1, 2), (2, 2)]:
                power = np.zeros(module.inputs, dtype=np.int64)
                power[a] += 1
                power[b] += 1
                powers.append(power)
            coefficients = rng.normal(0, 0.02, (len(powers), module.outputs))
            coefficients[::4, ::2] = 0
            setattr(baseline, name, PolynomialFunction(np.stack(powers), coefficients))
    baseline.checkpoint_steps = 0
    reused = copy.deepcopy(baseline)
    assert baseline.reuse_origin_terms is False
    reused.reuse_origin_terms = True
    if traced:
        trace_field(reused)
    return baseline, reused


def compare_parameter_gradients(first, second):
    left, right = list(first.parameters()), list(second.parameters())
    assert len(left) == len(right)
    for a, b in zip(left, right):
        ga = torch.zeros_like(a) if a.grad is None else a.grad
        gb = torch.zeros_like(b) if b.grad is None else b.grad
        assert torch.isfinite(gb).all()
        torch.testing.assert_close(ga, gb, rtol=2e-8, atol=2e-11)


@pytest.mark.parametrize('symbolic', [False, True])
@pytest.mark.parametrize('batch', [1, 2, 5])
@pytest.mark.parametrize('traced', [False, True])
def test_origin_reuse_field_energy_reference_and_full_gradients(symbolic, batch, traced):
    baseline, reused = model_pair(symbolic, traced)
    x = (torch.randn(batch, baseline.dim, dtype=torch.float64)*0.1).requires_grad_()
    y = x.detach().clone().requires_grad_()
    origins = reused.origin_terms(y)
    assert len(origins) == 5 and all(term.requires_grad for term in origins)
    assert all(term.shape[0] == batch for term in origins)
    expected = baseline.field(x)
    actual = reused.field(y, origins=origins)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-12)
    independently_differentiated = torch.autograd.grad(baseline.energy(x).sum(), x, create_graph=True)[0]
    torch.testing.assert_close(reused.energy_gradient(y, origins), independently_differentiated,
                               rtol=2e-10, atol=2e-12)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=2e-9, atol=2e-11)
    compare_parameter_gradients(baseline, reused)
    # Cached and independent eager diagnostics retain the power identity.
    z = torch.zeros(batch, baseline.dim, dtype=torch.float64)
    f, details = reused.field(z, create_graph=False, details=True, origins=reused.origin_terms(z))
    torch.testing.assert_close(f, torch.zeros_like(f), rtol=0, atol=2e-12)
    torch.testing.assert_close(details['Hdot'], details['expected_Hdot'], rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize('symbolic', [False, True])
@pytest.mark.parametrize('batch', [1, 2, 5])
def test_trace_rollout_builds_origins_once_and_preserves_initial_encoder_gradients(symbolic, batch, monkeypatch):
    baseline, reused = model_pair(symbolic, traced=True)
    encoder = HistoryEncoder(baseline, hidden=8)
    cached_encoder = copy.deepcopy(encoder)
    h = (torch.randn(batch, 11, baseline.n, 3, dtype=torch.float64)*0.1).requires_grad_()
    g = h.detach().clone().requires_grad_()
    x0 = baseline.initial(h, encoder)
    y0 = reused.initial(g, cached_encoder)
    x0.retain_grad()
    y0.retain_grad()
    built = []
    origin_builder = reused.origin_terms
    def counted(state):
        built.append(tuple(state.shape))
        return origin_builder(state)
    monkeypatch.setattr(reused, 'origin_terms', counted)
    expected = rollout(baseline, x0, 3, max_step=0.005)
    actual = rollout(reused, y0, 3, max_step=0.005)
    assert built == [(batch, baseline.dim)]  # shared across all 24 RK evaluations
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-12)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(y0.grad, x0.grad, rtol=2e-9, atol=2e-11)
    torch.testing.assert_close(g.grad, h.grad, rtol=2e-9, atol=2e-11)
    compare_parameter_gradients(baseline, reused)
    compare_parameter_gradients(encoder, cached_encoder)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in cached_encoder.parameters())


@pytest.mark.parametrize('symbolic', [False, True])
@pytest.mark.parametrize('traced', [False, True])
def test_two_adam_updates_rebuild_origins_without_stale_graph_or_parameter_state(symbolic, traced, monkeypatch):
    baseline, reused = model_pair(symbolic, traced)
    encoder = HistoryEncoder(baseline, hidden=8)
    cached_encoder = copy.deepcopy(encoder)
    optimizer = torch.optim.Adam(list(baseline.parameters())+list(encoder.parameters()), lr=1e-4)
    cached_optimizer = torch.optim.Adam(list(reused.parameters())+list(cached_encoder.parameters()), lr=1e-4)
    builds = []
    origin_builder = reused.origin_terms
    def counted(state):
        builds.append(1)
        return origin_builder(state)
    monkeypatch.setattr(reused, 'origin_terms', counted)
    original_keys = set(baseline.state_dict())
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        cached_optimizer.zero_grad(set_to_none=True)
        histories = torch.randn(2, 11, baseline.n, 3, dtype=torch.float64)*0.1
        expected = rollout(baseline, baseline.initial(histories, encoder), 2)
        actual = rollout(reused, reused.initial(histories, cached_encoder), 2)
        torch.testing.assert_close(actual, expected, rtol=3e-10, atol=3e-12)
        expected.square().mean().backward()
        actual.square().mean().backward()
        compare_parameter_gradients(baseline, reused)
        compare_parameter_gradients(encoder, cached_encoder)
        optimizer.step()
        cached_optimizer.step()
        assert len(builds) == update+1
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(reused.state_dict()[name], value, rtol=3e-9, atol=3e-11)
    assert set(reused.state_dict()) == original_keys
    # The execution switch/trace metadata are scalars, not a persistent origin tuple.
    assert not any('origin' in key for key in reused.state_dict())
    assert not any(isinstance(value, tuple) and any(torch.is_tensor(item) for item in value)
                   for value in vars(reused).values())


@pytest.mark.parametrize('symbolic', [False, True])
def test_trace_origins_are_explicit_inputs_and_parameters_alias(symbolic):
    baseline, reused = model_pair(symbolic, traced=True)
    assert reused._compiled_uses_origins is True
    # self, x, node0, edge_angle0, edge_left0, edge_right0, port0
    assert len(reused._compiled_field.forward.schema.arguments) == 7
    original = dict(reused.named_parameters())
    for name, parameter in reused._compiled_field.named_parameters():
        assert parameter.data_ptr() == original[name.removeprefix('model.')].data_ptr()
    assert set(reused.state_dict()) == set(baseline.state_dict())
    # Direct calls without a prebuilt tuple recompute origins with current
    # parameters rather than using the zero-state example from trace creation.
    x = torch.randn(1, reused.dim, dtype=torch.float64)*0.1
    for _ in range(2):
        with torch.no_grad():
            baseline.port.coefficients.add_(0.01) if symbolic else baseline.port.net[0].weight.add_(0.01)
            reused.port.coefficients.add_(0.01) if symbolic else reused.port.net[0].weight.add_(0.01)
        torch.testing.assert_close(reused.field(x, create_graph=False), baseline.field(x, create_graph=False),
                                   rtol=2e-10, atol=2e-12)


def test_unsupported_combinations_are_rejected_and_default_path_stays_available():
    baseline, reused = model_pair()
    x = torch.zeros(2, baseline.dim, dtype=torch.float64)
    reused.checkpoint_steps = 2
    with pytest.raises(ValueError, match='Origin reuse and training checkpoint'):
        rollout(reused, x, 2, training=True)
    # Inference does not use checkpoint recomputation and may reuse origins.
    inferred = rollout(reused, x, 2, training=False)
    torch.testing.assert_close(inferred, torch.zeros_like(inferred), rtol=0, atol=1e-12)
    _, physics = fixture(events=1, steps=12)
    unstructured = UnstructuredGraph(physics, [0.2, 0.3, 0.1], hidden=8)
    rollout(unstructured, x, 1, training=False)  # unchanged default U path
    unstructured.reuse_origin_terms = True
    for action in (lambda: unstructured.origin_terms(x), lambda: trace_field(unstructured),
                   lambda: rollout(unstructured, x, 1), lambda: unstructured.field(x)):
        with pytest.raises(ValueError, match='Origin reuse'):
            action()
    trace_field(baseline)
    assert baseline._compiled_uses_origins is False
    baseline.reuse_origin_terms = True
    with pytest.raises(ValueError, match='explicit origin inputs'):
        rollout(baseline, x, 1)
    with pytest.raises(ValueError, match='mode changed after tracing'):
        trace_field(baseline)
