"""Synthetic evaluation-executor parity, with no real assets or metrics."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from task4.accelerate import trace_field
from task4.dynamics import PowerGraph, UnstructuredGraph
from task4.evaluation import _safe_field, forecast_batch
from task4.paths import ROOT
from task4.synthetic import fixture


def synthetic_graph(kind):
    torch.manual_seed(1287)
    if kind == 'S':
        # Reuse the existing complete PolynomialFunction synthetic fixture;
        # importing its module does not execute any of that module's tests.
        spec = importlib.util.spec_from_file_location('trace_symbolic_fixture', ROOT / 'tests/test_symbolic_runtime.py')
        helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helpers)
        eager, _, _ = helpers.polynomial_model(n=3, rank=2, hidden=8)
    else:
        _, physics = fixture(n=3, events=1, steps=12)
        cls = PowerGraph if kind == 'N' else UnstructuredGraph
        eager = cls(physics, [.2, .3, .1], rank=2, hidden=8)
    eager.eval()
    return eager


@pytest.fixture(scope='module', params=('N', 'S', 'U'))
def graph_pair(request):
    eager = synthetic_graph(request.param)
    traced = copy.deepcopy(eager)
    before = {key: value.detach().clone() for key, value in traced.state_dict().items()}
    assert eager.reuse_origin_terms is traced.reuse_origin_terms is False
    trace_field(traced)
    assert not traced._compiled_uses_origins
    assert set(before) == set(traced.state_dict())
    for key, value in traced.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    return eager, traced


def compare_forecasts(eager, traced):
    for key in ('states', 'physical', 'substep_states', 'substep_physical'):
        np.testing.assert_allclose(eager[key], traced[key], rtol=2e-10, atol=2e-12, equal_nan=True, err_msg=key)
    for key in ('state_finite', 'substep_state_finite'):
        np.testing.assert_array_equal(eager[key], traced[key], err_msg=key)
    assert eager['failures'] == traced['failures']
    assert eager['substeps_per_output'] == traced['substeps_per_output']
    assert eager['actual_step_s'] == traced['actual_step_s']


@pytest.mark.parametrize('batch', (1, 8))
@pytest.mark.parametrize('step', (.0025, .005))
def test_frozen_rk4_eager_trace_all_states_and_physical_paths(graph_pair, batch, step):
    eager, traced = graph_pair
    z0 = torch.as_tensor(np.random.default_rng(983 + batch).normal(0., .025, (batch, eager.dim)), dtype=torch.float64)
    original = z0.clone()
    first = forecast_batch(eager, z0, steps=4, max_step=step)
    second = forecast_batch(traced, z0, steps=4, max_step=step)
    compare_forecasts(first, second)
    torch.testing.assert_close(z0, original, rtol=0, atol=0)
    assert first['failures'] == [None] * batch
    assert first['substep_state_finite'].all()
    assert first['states'].shape == (batch, 5, eager.dim)


@pytest.mark.parametrize('batch', (1, 8))
@pytest.mark.parametrize('step', (.0025, .005))
def test_initial_latent_nan_keeps_identical_first_failure_and_finite_neighbors(graph_pair, batch, step):
    eager, traced = graph_pair
    z0 = torch.zeros(batch, eager.dim, dtype=torch.float64)
    z0[0, -1] = float('nan')  # Physical coordinates are all finite.
    first = forecast_batch(eager, z0, steps=3, max_step=step)
    second = forecast_batch(traced, z0, steps=3, max_step=step)
    compare_forecasts(first, second)
    assert first['failures'][0]['stage'] == 'initial_state'
    assert first['failures'][0]['first_bad_substep'] == 0
    assert not first['state_finite'][0].any()
    assert np.isnan(first['states'][0]).all()
    if batch > 1:
        assert first['substep_state_finite'][1:].all()
        assert first['failures'][1:] == [None] * (batch - 1)


class LatentFailureClock(torch.nn.Module):
    """Synthetic wrapper: keep the N/S/U field, add a tensor-only hidden trap.

    The penultimate q is a clock and the last q is a fixed event marker. Only
    the marked event's latent derivative becomes NaN after 11ms; physical
    derivatives remain finite. This is not a proposed scientific model.
    """
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.n, self.dim, self.mass = graph.n, graph.dim, graph.mass
        self.reuse_origin_terms = False

    def physical(self, state):
        return self.graph.physical(state)

    def field(self, state, create_graph=True):
        compiled = getattr(self, '_compiled_field', None)
        if compiled is not None:
            with torch.set_grad_enabled(create_graph):
                return compiled(state)
        with torch.set_grad_enabled(create_graph):
            result = self.graph.field(state, create_graph=create_graph).clone()
            result[:, -2] = 1.
            bad = (state[:, -2] > .011) & (state[:, -1] > .5)
            result[:, -1] = torch.where(bad, torch.full_like(state[:, -1], float('nan')), torch.zeros_like(state[:, -1]))
        return result if create_graph else result.detach()


@pytest.fixture(scope='module')
def clock_pair(graph_pair):
    eager = LatentFailureClock(copy.deepcopy(graph_pair[0]))
    traced = copy.deepcopy(eager)
    trace_field(traced)
    assert not traced._compiled_uses_origins
    return eager, traced


@pytest.mark.parametrize('batch', (1, 8))
@pytest.mark.parametrize('step', (.0025, .005))
def test_late_latent_derivative_failure_and_batch_shrinking_are_identical(clock_pair, batch, step):
    eager, traced = clock_pair
    z0 = torch.zeros(batch, eager.dim, dtype=torch.float64)
    z0[0, -1] = 1.
    first = forecast_batch(eager, z0, steps=4, max_step=step)
    second = forecast_batch(traced, z0, steps=4, max_step=step)
    compare_forecasts(first, second)
    failed = first['failures'][0]
    assert failed['stage'] == 'k2_field'
    assert failed['first_missing_output_index'] == 2
    assert failed['first_bad_substep'] == (5 if step == .0025 else 3)
    assert first['state_finite'][0, :2].all() and not first['state_finite'][0, 2:].any()
    assert np.isfinite(first['states'][0, :2]).all() and np.isnan(first['states'][0, 2:]).all()
    if batch > 1:
        assert first['substep_state_finite'][1:].all()
        assert first['failures'][1:] == [None] * (batch - 1)


@pytest.fixture
def cli_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('evaluation_trace_cli', ROOT / 'scripts/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m3_trace_is_rejected_before_source_reads(cli_script, monkeypatch):
    monkeypatch.setattr(cli_script, 'check_authorized', lambda synthetic: None)
    monkeypatch.setattr(cli_script, 'read_json', lambda path: pytest.fail('M3 trace must reject before source reads'))
    monkeypatch.setattr(cli_script, 'trace_field', lambda model: pytest.fail('Fixed M3 must not trace'))
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', '--baseline-m3', '--trace-field', '--output', 'unused.json'])
    with pytest.raises(ValueError, match='fixed M3 uses the eager'):
        cli_script.main()


@pytest.mark.parametrize('traced', (False, True))
def test_cli_option_preserves_identity_and_adds_execution_metadata(cli_script, monkeypatch, tmp_path, traced):
    checkpoint_path, output = tmp_path / 'constructed.txt', tmp_path / 'result.json'
    checkpoint_path.write_text('synthetic placeholder, not checkpoint data', encoding='utf-8')
    model = SimpleNamespace(reuse_origin_terms=False)
    metadata = {'synthetic': True, 'config': {'arm': 'P'}, 'kind': 'T4-N', 'step': 7}
    calls = []
    monkeypatch.setattr(cli_script, 'check_authorized', lambda synthetic: None)
    monkeypatch.setattr(cli_script.torch, 'set_num_threads', lambda value: None)
    monkeypatch.setattr(cli_script, 'project_path', lambda value: Path(value))
    monkeypatch.setattr(cli_script, 'digest', lambda path: 'constructed-source-hash')
    monkeypatch.setattr(cli_script, 'load_checkpoint', lambda path: (model, object(), object(), metadata))
    monkeypatch.setattr(cli_script, 'fixture', lambda: ([SimpleNamespace(id='constructed')], SimpleNamespace()))
    monkeypatch.setattr(cli_script, 'read_json', lambda path: pytest.fail('Synthetic CLI must not read source configuration'))
    def compile_stub(given):
        assert given is model and not given.reuse_origin_terms
        calls.append('trace')
        given.trace_build_seconds = .125
    def evaluate_stub(given, encoder, events, destination, **kwargs):
        assert given is model
        assert kwargs['fine_dt'] == .0025 and kwargs['coarse_dt'] == .005
        calls.append('evaluate')
        return {'summary': {}, 'model_identity': copy.deepcopy(given.evaluation_identity)}
    monkeypatch.setattr(cli_script, 'trace_field', compile_stub)
    monkeypatch.setattr(cli_script, 'evaluate', evaluate_stub)
    monkeypatch.setattr(cli_script, 'write_json', lambda path, result: Path(path).write_text(json.dumps(result, allow_nan=False), encoding='utf-8'))
    args = ['evaluate.py', '--synthetic', '--endpoint', 'P-N', '--checkpoint', str(checkpoint_path), '--output', str(output)]
    if traced:
        args.append('--trace-field')
    monkeypatch.setattr(sys, 'argv', args)
    cli_script.main()
    result = json.loads(output.read_text(encoding='utf-8'))
    assert calls == (['trace', 'evaluate'] if traced else ['evaluate'])
    assert result['model_identity'] == {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': 'constructed-source-hash', 'kind': 'T4-N', 'step': 7}
    assert result['source_identity_unchanged']
    assert result['execution'] == {'field_executor': 'torchscript_trace' if traced else 'eager',
                                   'trace_field_requested': traced, 'trace_build_seconds': .125 if traced else 0.,
                                   'origin_terms_reused': False}


@pytest.fixture(scope='module', params=('N', 'S'))
def origin_graph_pair(request):
    eager = synthetic_graph(request.param)
    reused = copy.deepcopy(eager)
    reused.reuse_origin_terms = True
    trace_field(reused)
    assert reused._compiled_uses_origins
    return eager, reused


def count_origin_builds(model, monkeypatch):
    original, calls = model.origin_terms, []
    def counted(state):
        calls.append(len(state))
        return original(state)
    monkeypatch.setattr(model, 'origin_terms', counted)
    return calls


@pytest.mark.parametrize('batch', (1, 8))
@pytest.mark.parametrize('step', (.0025, .005))
def test_origin_reuse_builds_once_per_rollout_and_preserves_complete_paths(origin_graph_pair, monkeypatch, batch, step):
    eager, reused = origin_graph_pair
    calls = count_origin_builds(reused, monkeypatch)
    z0 = torch.as_tensor(np.random.default_rng(551 + batch).normal(0., .025, (batch, eager.dim)), dtype=torch.float64)
    reference = forecast_batch(eager, z0, steps=4, max_step=step)
    actual = forecast_batch(reused, z0, steps=4, max_step=step)
    compare_forecasts(reference, actual)
    assert calls == [batch]
    # A second (fine/coarse) rollout receives a freshly computed local tuple.
    second_step = .005 if step == .0025 else .0025
    compare_forecasts(forecast_batch(eager, z0, steps=4, max_step=second_step),
                      forecast_batch(reused, z0, steps=4, max_step=second_step))
    assert calls == [batch, batch]


@pytest.mark.parametrize('batch', (1, 8))
@pytest.mark.parametrize('step', (.0025, .005))
def test_origin_reuse_keeps_initial_latent_failure_and_valid_neighbors(origin_graph_pair, monkeypatch, batch, step):
    eager, reused = origin_graph_pair
    calls = count_origin_builds(reused, monkeypatch)
    z0 = torch.zeros(batch, eager.dim, dtype=torch.float64)
    z0[0, -1] = float('nan')
    expected = forecast_batch(eager, z0, steps=3, max_step=step)
    actual = forecast_batch(reused, z0, steps=3, max_step=step)
    compare_forecasts(expected, actual)
    assert calls == [batch]
    assert actual['failures'][0]['stage'] == 'initial_state'
    if batch == 8:
        assert actual['substep_state_finite'][1:].all()


def tensor_clock_field(self, state, create_graph=True, details=False, origins=None):
    """Patch exact PowerGraph objects only in synthetic failure tests."""
    assert not details
    result = PowerGraph.field(self, state, create_graph=create_graph, origins=origins).clone()
    result[:, -2] = 1.
    bad = (state[:, -2] > .011) & (state[:, -1] > .5)
    result[:, -1] = torch.where(bad, torch.full_like(state[:, -1], float('nan')), torch.zeros_like(state[:, -1]))
    return result if create_graph else result.detach()


@pytest.fixture(scope='module', params=('N', 'S'))
def origin_clock_pair(request):
    eager = synthetic_graph(request.param)
    object.__setattr__(eager, 'field', MethodType(tensor_clock_field, eager))
    reused = copy.deepcopy(eager)
    reused.reuse_origin_terms = True
    trace_field(reused)
    assert type(reused) is PowerGraph and reused._compiled_uses_origins
    return eager, reused


@pytest.mark.parametrize('batch', (1, 8))
@pytest.mark.parametrize('step', (.0025, .005))
@pytest.mark.parametrize('exceptional', (False, True))
def test_reused_origins_survive_late_failure_active_shrink_and_exception_singletons(origin_clock_pair, monkeypatch, batch, step, exceptional):
    eager, reused = origin_clock_pair
    origin_calls = count_origin_builds(reused, monkeypatch)
    cached_batches = []
    def failure_dispatch(self, state, create_graph=True, details=False, origins=None):
        if self is reused:
            assert origins is not None and all(term.shape[0] == len(state) for term in origins)
            cached_batches.append(len(state))
        bad = (state[:, -2] > .011) & (state[:, -1] > .5)
        if exceptional and bool(bad.any()):
            if len(state) > 1:
                raise RuntimeError('constructed batch retry')
            raise FloatingPointError('constructed singleton latent failure')
        return tensor_clock_field(self, state, create_graph, details, origins)
    monkeypatch.setattr(eager, 'field', MethodType(failure_dispatch, eager))
    monkeypatch.setattr(reused, 'field', MethodType(failure_dispatch, reused))
    z0 = torch.zeros(batch, eager.dim, dtype=torch.float64)
    z0[0, -1] = 1.
    reference = forecast_batch(eager, z0, steps=4, max_step=step)
    actual = forecast_batch(reused, z0, steps=4, max_step=step)
    compare_forecasts(reference, actual)
    assert origin_calls == [batch]
    failure = actual['failures'][0]
    assert failure['stage'] == 'k2_field'
    assert failure['first_bad_substep'] == (5 if step == .0025 else 3)
    assert failure['first_missing_output_index'] == 2
    assert actual['state_finite'][0, :2].all() and not actual['state_finite'][0, 2:].any()
    if batch == 8:
        assert 7 in cached_batches
        assert actual['substep_state_finite'][1:].all()
        if exceptional:
            assert 1 in cached_batches  # _safe_field's exceptional single-row retry.


def test_safe_field_origin_singletons_use_local_indices_after_noncontiguous_active_slice():
    all_states = torch.arange(8, dtype=torch.float64)[:, None].expand(-1, 3).clone()
    all_origins = tuple(torch.arange(8, dtype=torch.float64)[:, None] + offset for offset in range(5))
    active = torch.tensor([1, 3, 7])
    calls = []
    class ExceptionalRows:
        def field(self, state, create_graph=False, origins=None):
            assert not create_graph and origins is not None
            calls.append(len(state))
            for offset, term in enumerate(origins):
                torch.testing.assert_close(term[:, 0], state[:, 0] + offset, rtol=0, atol=0)
            if len(state) > 1:
                raise RuntimeError('constructed batch exception')
            if state[0, 0] == 3:
                raise FloatingPointError('constructed bad row')
            return state * 2
    result, errors = _safe_field(ExceptionalRows(), all_states[active], tuple(term[active] for term in all_origins))
    assert calls == [3, 1, 1, 1]
    assert errors == [None, 'FloatingPointError: constructed bad row', None]
    torch.testing.assert_close(result[[0, 2]], all_states[[1, 7]] * 2, rtol=0, atol=0)
    assert torch.isnan(result[1]).all()


@pytest.mark.parametrize('batch', (1, 8))
def test_non_batch_first_origins_reject_before_any_field_execution(origin_graph_pair, monkeypatch, batch):
    _, reused = origin_graph_pair
    z0 = torch.zeros(batch, reused.dim, dtype=torch.float64)
    origins = reused.origin_terms(z0)
    corrupted = (origins[0][0] if batch == 1 else origins[0][:-1], *origins[1:])
    monkeypatch.setattr(reused, 'origin_terms', lambda state: corrupted)
    monkeypatch.setattr(reused, 'field', lambda *args, **kwargs: pytest.fail('Malformed origin layout must reject before fields'))
    with pytest.raises(ValueError, match='batch-first'):
        forecast_batch(reused, z0, steps=2)


def test_batch_first_but_wrong_origin_layout_also_rejects(origin_graph_pair, monkeypatch):
    _, reused = origin_graph_pair
    z0 = torch.zeros(8, reused.dim, dtype=torch.float64)
    origins = reused.origin_terms(z0)
    corrupted = (*origins[:4], origins[4][..., :1])
    monkeypatch.setattr(reused, 'origin_terms', lambda state: corrupted)
    monkeypatch.setattr(reused, 'field', lambda *args, **kwargs: pytest.fail('Wrong full origin layout must reject before fields'))
    with pytest.raises(ValueError, match='layout does not match'):
        forecast_batch(reused, z0, steps=2)


def test_origin_reuse_requires_correct_trace_and_rejects_unstructured_graph():
    graph = synthetic_graph('N')
    graph.reuse_origin_terms = True
    with pytest.raises(ValueError, match='requires a trace'):
        forecast_batch(graph, torch.zeros(1, graph.dim, dtype=torch.float64), steps=1)
    graph.reuse_origin_terms = False
    trace_field(graph)
    graph.reuse_origin_terms = True
    with pytest.raises(ValueError, match='explicit origin inputs'):
        forecast_batch(graph, torch.zeros(1, graph.dim, dtype=torch.float64), steps=1)
    unstructured = synthetic_graph('U')
    unstructured.reuse_origin_terms = True
    with pytest.raises(ValueError, match='exact PowerGraph'):
        forecast_batch(unstructured, torch.zeros(1, unstructured.dim, dtype=torch.float64), steps=1)


@pytest.mark.parametrize('args,message', [
    (['--checkpoint', 'unread.txt', '--endpoint', 'P-N', '--reuse-origin-terms'], 'requires --trace-field'),
    (['--checkpoint', 'unread.txt', '--endpoint', 'U', '--trace-field', '--reuse-origin-terms'], 'limited to structured'),
    (['--baseline-m3', '--trace-field', '--reuse-origin-terms'], 'limited to structured'),
])
def test_invalid_origin_cli_modes_reject_before_source_reads(cli_script, monkeypatch, args, message):
    monkeypatch.setattr(cli_script, 'check_authorized', lambda synthetic: None)
    monkeypatch.setattr(cli_script, 'read_json', lambda path: pytest.fail('Unsupported origin mode must reject before source reads'))
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', *args, '--output', 'unused.json'])
    with pytest.raises(ValueError, match=message):
        cli_script.main()


@pytest.mark.parametrize('endpoint,arm,kind', [('P-N', 'P', 'T4-N'), ('P-S', 'P', 'T4-S'), ('A', 'A', 'T4-N')])
def test_origin_cli_sets_reuse_before_compiling_and_reports_actual_mode(cli_script, monkeypatch, tmp_path, endpoint, arm, kind):
    class FakePowerGraph(SimpleNamespace):
        pass
    model = FakePowerGraph(reuse_origin_terms=False)
    checkpoint_path, output = tmp_path / 'constructed.txt', tmp_path / 'result.json'
    checkpoint_path.write_text('placeholder only', encoding='utf-8')
    metadata = {'synthetic': True, 'config': {'arm': arm}, 'kind': kind, 'step': 7}
    calls = []
    monkeypatch.setattr(cli_script, 'PowerGraph', FakePowerGraph)
    monkeypatch.setattr(cli_script, 'check_authorized', lambda synthetic: None)
    monkeypatch.setattr(cli_script.torch, 'set_num_threads', lambda value: None)
    monkeypatch.setattr(cli_script, 'project_path', lambda value: Path(value))
    monkeypatch.setattr(cli_script, 'digest', lambda path: 'constructed-source-hash')
    monkeypatch.setattr(cli_script, 'load_checkpoint', lambda path: (model, object(), object(), metadata))
    monkeypatch.setattr(cli_script, 'fixture', lambda: ([SimpleNamespace(id='constructed')], SimpleNamespace()))
    monkeypatch.setattr(cli_script, 'read_json', lambda path: pytest.fail('No source configuration in synthetic CLI test'))
    def compile_stub(given):
        assert given is model and given.reuse_origin_terms
        calls.append('trace_with_origins')
        given.trace_build_seconds = .125
    def evaluate_stub(given, encoder, events, destination, **kwargs):
        assert given is model and given.reuse_origin_terms
        assert kwargs['fine_dt'] == .0025 and kwargs['coarse_dt'] == .005
        calls.append('evaluate')
        return {'summary': {}, 'model_identity': copy.deepcopy(given.evaluation_identity)}
    monkeypatch.setattr(cli_script, 'trace_field', compile_stub)
    monkeypatch.setattr(cli_script, 'evaluate', evaluate_stub)
    monkeypatch.setattr(cli_script, 'write_json', lambda path, result: Path(path).write_text(json.dumps(result, allow_nan=False), encoding='utf-8'))
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', '--synthetic', '--endpoint', endpoint, '--checkpoint', str(checkpoint_path),
                                    '--trace-field', '--reuse-origin-terms', '--output', str(output)])
    cli_script.main()
    result = json.loads(output.read_text(encoding='utf-8'))
    assert calls == ['trace_with_origins', 'evaluate']
    assert result['model_identity']['checkpoint_sha256'] == 'constructed-source-hash'
    assert result['execution']['origin_terms_reused'] is True
    assert result['execution']['field_executor'] == 'torchscript_trace'
    assert result['execution']['origin_build_scope'] == 'once per fine/coarse rollout for the full initial batch; active/singleton slices'
