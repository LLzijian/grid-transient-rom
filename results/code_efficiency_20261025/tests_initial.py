"""CPU-only exactness checks. No optimizer updates, GPU or project data."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
from compiled_expression_runtime_20261025 import ExpressionKernel, CompiledExpressionROM, adapt_base
from symbolic_closure_rom import _torch_feature_eval, _VARIABLES
from closure_discovery_benchmarks import toy_source
from closure_causal_recovery_20261019 import copy_runtime, trajectory

torch.set_num_threads(1)


def var(name):
    return {'op': 'var', 'name': name}


def current_terms():
    binding = json.loads((ROOT / 'private_data/closure_mechanism_20261019/P1/horizon_pair01/binding.json').read_text())
    pool = json.loads((ROOT / binding['pool_manifest']['path']).read_text())
    return [pool['selected'][2], pool['selected'][3]]


EXPRESSIONS = [
    ('node', {'op': 'mul', 'args': [var('p'), {'op': 'pow2', 'arg': var('q')}]}),
    ('node', {'op': 'add', 'args': [var('p'), {'op': 'const', 'value': 2.}]}),
    ('port', {'op': 'tanh', 'scale': .5, 'arg': var('q')}),
    ('edge', {'op': 'pow3', 'arg': var('s')}),
] + [(t['channel'], t['expression']) for t in current_terms()]


@pytest.mark.parametrize('channel,expression', EXPRESSIONS)
def test_expression_value_derivative_and_backprop_exact(channel, expression):
    torch.manual_seed(2501)
    a = torch.randn(3, 7, len(_VARIABLES[channel]), dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    expected = _torch_feature_eval(expression, a, _VARIABLES[channel], channel)
    kernel = ExpressionKernel(expression, channel)
    actual = kernel(b)
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    le = sum(t.square().sum() for t in expected)
    la = sum(t.square().sum() for t in actual)
    le.backward(); la.backward()
    assert torch.equal(a.grad, b.grad)
    cached_ids = [id(t) for t in kernel._constants.values()]
    again = kernel(b.detach())
    assert all(torch.equal(x.detach(), y) for x, y in zip(expected, again))
    assert cached_ids == [id(t) for t in kernel._constants.values()]
    cloned = copy.deepcopy(kernel)
    assert not cloned._constants
    cloned(b.detach())
    assert not ({id(t) for t in cloned._constants.values()} & set(cached_ids))
    kernel.clear()
    assert not kernel._constants and not kernel._origins


def gradients(model, state):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    result = model.field(state)
    loss = result.square().sum()
    values = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
    return result.detach(), {n: None if v is None else v.detach() for (n, _), v in zip(parameters, values)}


@pytest.mark.parametrize('term', current_terms())
def test_actual_terms_nonunit_physics_field_trajectory_and_gradients_exact(term, tmp_path):
    reference = copy_runtime(toy_source(), [term], 'cpu')
    # Apply the same non-unit configuration to both registered physical copies.
    with torch.no_grad():
        mass = torch.linspace(.7, 2.3, 10, dtype=torch.float64)
        reference.mass.copy_(mass)
        reference.structured.mass.copy_(mass)
        reference.phi.add_(.17)
        reference.structured.phi.copy_(reference.phi)
        reference.coefficients.fill_(.15)
    rng_before = torch.get_rng_state().clone()
    actual = adapt_base(reference)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert actual._expression_enabled
    state = torch.randn(3, 40, dtype=torch.float64) * .1
    fr, gr = gradients(reference, state)
    fa, ga = gradients(actual, state)
    assert torch.equal(fr, fa)
    for name in gr:
        assert (gr[name] is None and ga[name] is None) or torch.equal(gr[name], ga[name]), name
    assert torch.equal(reference.rk4(state, .01), actual.rk4(state, .01))
    a = trajectory(reference, state, 8)
    b = trajectory(actual, state, 8)
    assert torch.equal(a, b)
    assert all(not k._constants and not k._origins for k in actual._expression_kernels)
    path = tmp_path / 'model.json'
    actual.export(path)
    restored = CompiledExpressionROM.from_export(path)
    assert torch.equal(actual.field(state), restored.field(state))
    switched = actual.with_terms([{'channel': 'port', 'expression': var('q')}])
    assert not switched._expression_enabled  # Explicit reference fallback.


def test_only_edge_features_are_built_and_constant_allocations_reused(monkeypatch):
    import compiled_expression_runtime_20261025 as module
    model = adapt_base(copy_runtime(toy_source(), [current_terms()[0]], 'cpu'))
    original = torch.stack
    widths = []
    def observed_stack(items, *args, **kwargs):
        widths.append(len(items))
        return original(items, *args, **kwargs)
    monkeypatch.setattr(module.torch, 'stack', observed_stack)
    model.begin_graph_cache()
    try:
        state = torch.zeros(2, 40, dtype=torch.float64)
        model.field(state)
        caches = [id(t) for k in model._expression_kernels for t in k._constants.values()]
        model.field(state)
        assert caches == [id(t) for k in model._expression_kernels for t in k._constants.values()]
    finally:
        model.end_graph_cache()
    # No 2-column node feature, no extra 6-column port feature. Edge plus
    # endpoint reversal and reversed derivative account for three stacks/RHS.
    assert 2 not in widths
    assert widths.count(6) == 6
