"""Focused synthetic opt-in, fit ordering and optimizer-resume regression."""
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import task4.accelerate as accelerate
import task4.training as training
from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder, PredictiveHead
from task4.objectives import ShootingBank
from task4.symbolic import PolynomialFunction
from task4.synthetic import fixture


def config():
    return dict(name='synthetic_quadratic_fit', seed=41, arm='P', latent_per_machine=2,
                hidden=8, device='cpu', batch_size=2, lr=.0003, grad_clip=100.,
                integration_dt=.01, steps=2, horizons=[.02], milestones=[1],
                anchor_horizons=[.02], anchor_weight=.2, latent_weight=.1,
                torque_weight=.1, complexity_weight=1e-5, shooting_fraction=0.,
                final_autonomous_fraction=.35, segment_schedule=[.25],
                continuity_weight=1., checkpoint_steps=0, save_every=1,
                monitor_count=1, synthetic=True)


@pytest.mark.parametrize('arm,autonomous,anchor_weight,expected_use,expected_membership', (
    ('P', False, 0., False, True),  # C: unused head remains an Adam member.
    ('P', False, .2, True, True),  # A false metadata hint cannot hide actual use.
    ('A', False, .2, False, True),
    ('P', True, 0., False, False),
))
def test_fit_summary_distinguishes_head_membership_use_and_optimization(
        tmp_path, monkeypatch, synthetic_source, arm, autonomous, anchor_weight,
        expected_use, expected_membership):
    source, cfg, _, _, _ = synthetic_source
    requested = {**cfg, 'arm': arm, 'autonomous_only': autonomous,
                 'anchor_weight': anchor_weight, 'prediction_head_used_or_deployed': False,
                 'teacher_usage': 'TRAIN_only_compression_adaptation'}
    if autonomous:
        requested.update(latent_weight=0., torque_weight=0.)
    elif anchor_weight == 0. and arm == 'P':
        requested['teacher_checkpoint'] = str(source)
    groups = training.optimizer_groups
    observed = {}
    forward_calls = []
    hooks = []
    def observe_groups(model, encoder, head, configuration):
        result = groups(model, encoder, head, configuration)
        observed['head'] = head
        observed['before'] = {name: value.detach().clone() for name, value in head.named_parameters()}
        observed['membership'] = any(id(p) in {id(h) for h in head.parameters()}
                                     for group in result for p in group['params'])
        hooks.append(head.register_forward_hook(lambda *args: forward_calls.append(True)))
        return result
    monkeypatch.setattr(training, 'optimizer_groups', observe_groups)
    # Use an available point in this constructed TRAIN event so this reporting
    # regression observes anchor gradients, independently of random late masks.
    targets = training.AnchorTargets
    def available_targets(*args):
        instance = targets(*args)
        original_get = instance.get
        instance.get = lambda event, index: original_get(event, 10)
        return instance
    monkeypatch.setattr(training, 'AnchorTargets', available_targets)
    output = tmp_path / f'adapt_{arm}_{autonomous}_{anchor_weight}'
    try:
        report = training.fit(requested, {}, output, warm_start=source)
    finally:
        for hook in hooks:
            hook.remove()
    saved = json.loads((output / 'training_summary.json').read_text(encoding='utf-8'))
    assert saved == report
    assert observed['membership'] is expected_membership
    assert bool(forward_calls) is expected_use
    assert saved['prediction_head_in_optimizer'] is expected_membership
    assert saved['prediction_head_used'] is expected_use
    assert saved['prediction_head_used_or_deployed'] is expected_use
    assert saved['prediction_head_deployed'] is False
    assert saved['prediction_head_optimized'] is expected_use
    assert saved['prediction_head_used_or_optimized'] is expected_membership
    assert saved['prediction_head_used_or_optimized_semantics'] == 'legacy_optimizer_membership_label'
    assert requested['prediction_head_used_or_deployed'] is False
    if not expected_use:
        assert all(parameter.grad is None for parameter in observed['head'].parameters())
        for name, parameter in observed['head'].named_parameters():
            torch.testing.assert_close(parameter, observed['before'][name], rtol=0, atol=0)


def polynomial_graph(physics):
    torch.manual_seed(143)
    graph = PowerGraph(physics, [.2, .3, .1], rank=2, hidden=8)
    rng = np.random.default_rng(812)
    for name, fn in list(graph.function_modules().items()):
        powers = np.zeros((5, fn.inputs), dtype=np.int64)
        powers[1, 0], powers[2, 0], powers[3, 1] = 1, 2, 1
        powers[4, 0:2] = 1
        coefficients = rng.normal(0., .005, (5, fn.outputs))
        coefficients[3, 0] = 0.
        replacement = PolynomialFunction(powers, coefficients)
        # A supported coefficient reaching zero and an unsupported stored
        # nonzero coefficient distinguish the true mask from final nonzeros.
        with torch.no_grad():
            replacement.coefficients[1, 0] = 0.
            replacement.coefficients[3, 0] = .125
        setattr(graph, name, replacement)
    return graph


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    events, physics = fixture(events=2, steps=14)
    monkeypatch.setattr(training, 'project_path', lambda value: Path(value))
    monkeypatch.setattr(training, 'write_json', lambda path, value: Path(path).write_text(json.dumps(value, allow_nan=False), encoding='utf-8'))
    monkeypatch.setattr(training, 'fixture', lambda: (events, physics))
    cfg = config()
    graph = polynomial_graph(physics)
    # Source units/scales are exactly those of this constructed TRAIN set.
    graph.scales.copy_(torch.as_tensor(training.train_scales(events, physics), dtype=torch.float64))
    encoder, head = HistoryEncoder(graph, 8), PredictiveHead(graph, [.02], 8)
    saved_cfg = {**cfg, '_optimizer_blocks': True, '_event_ids': [event.id for event in events]}
    saved_cfg['torque_scale'] = max(1e-5, float(np.sqrt(np.mean([np.mean(event.torque[10:]**2) for event in events]))))
    optimizer = torch.optim.Adam(training.optimizer_groups(graph, encoder, head, saved_cfg), lr=cfg['lr'])
    source = tmp_path / 'constructed.pt'
    training.checkpoint(source, graph, encoder, head, ShootingBank(), optimizer, saved_cfg, physics,
                        0, np.random.default_rng(cfg['seed']), [])
    return source, cfg, graph, events, physics


def assert_persistent_state_unchanged(before, model):
    assert set(before) == set(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize('mode,selected', [('off', []), ('energy_only', ['node_energy', 'edge_energy']),
                                        ('all_poly', list(('node_energy', 'edge_energy', 'node_exchange', 'node_dissipation', 'edge_exchange', 'edge_dissipation', 'port')))])
def test_executor_switch_keeps_parameters_support_and_adam_state(synthetic_source, mode, selected):
    _, _, graph, _, _ = synthetic_source
    parameters = {name: (id(param), param.data_ptr()) for name, param in graph.named_parameters()}
    before = {key: value.clone() for key, value in graph.state_dict().items()}
    reference = copy.deepcopy(graph)
    optimizer, reference_optimizer = torch.optim.Adam(graph.parameters(), lr=.0003), torch.optim.Adam(reference.parameters(), lr=.0003)
    # Seed Adam moments using explicitly equal raw gradients, independently of
    # either function executor's multiplication/reduction order.
    for index, (first, second) in enumerate(zip(graph.parameters(), reference.parameters())):
        first.grad = torch.full_like(first, .01 * (index + 1)); second.grad = first.grad.clone()
    optimizer.step(); reference_optimizer.step()
    before = {key: value.clone() for key, value in graph.state_dict().items()}
    report = training.configure_quadratic_functions(graph, mode)
    assert report['enabled_function_modules'] == selected
    assert not report['scientific_definition_changed_by_executor']
    assert_persistent_state_unchanged(before, graph)
    assert parameters == {name: (id(param), param.data_ptr()) for name, param in graph.named_parameters()}
    for index, (first, second) in enumerate(zip(graph.parameters(), reference.parameters())):
        first.grad = torch.full_like(first, -.02 * (index + 1)); second.grad = first.grad.clone()
    optimizer.step(); reference_optimizer.step()
    for first, second in zip(graph.parameters(), reference.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for key, state in optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, reference_optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
    assert [name for name, fn in graph.function_modules().items() if fn.quadratic_execution] == selected


@pytest.mark.parametrize('mode', ('off', 'energy_only', 'all_poly'))
def test_fit_sets_executor_before_trace_and_does_not_rewrite_source(tmp_path, monkeypatch, synthetic_source, mode):
    source, cfg, _, _, _ = synthetic_source
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []
    original_data, original_load = training.fixture, training.load_checkpoint
    def data():
        calls.append('data')
        return original_data()
    def load(path, device='cpu'):
        calls.append('load')
        return original_load(path, device)
    def trace(model):
        enabled = [name for name, fn in model.function_modules().items() if fn.quadratic_execution]
        expected = [] if mode == 'off' else ['node_energy', 'edge_energy'] if mode == 'energy_only' else list(model.function_modules())
        assert enabled == expected
        calls.append('trace')
    monkeypatch.setattr(training, 'fixture', data)
    monkeypatch.setattr(training, 'load_checkpoint', load)
    monkeypatch.setattr(accelerate, 'trace_field', trace)
    requested = {**cfg, 'steps': 0, 'milestones': [], 'trace_field': True}
    if mode != 'off':
        requested['quadratic_functions'] = mode
    report = training.fit(requested, {}, tmp_path / mode, warm_start=source)
    assert calls == (['data', 'load', 'trace'] if mode == 'off' else ['load', 'data', 'load', 'trace'])
    assert report['function_execution']['quadratic_functions'] == mode
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert requested == {**cfg, 'steps': 0, 'milestones': [], 'trace_field': True, **({} if mode == 'off' else {'quadratic_functions': mode})}
    loaded, _, _, saved = original_load(tmp_path / mode / 'last.pt')
    assert saved['config']['quadratic_functions'] == mode
    assert all(not fn.quadratic_execution for fn in loaded.function_modules().values())
    assert saved['config']['steps'] == 0 and saved['kind'] == 'T4-S'


@pytest.mark.parametrize('reject_source', (False, True))
def test_preflight_restores_torch_python_numpy_rng_even_if_loader_consumes_them(monkeypatch, synthetic_source, reject_source):
    source, cfg, _, _, _ = synthetic_source
    original = training.load_checkpoint
    def consuming(path, device='cpu'):
        random.random(); np.random.random(); torch.rand(3)
        if reject_source:
            raise ValueError('constructed source rejection after random allocation')
        return original(path, device)
    monkeypatch.setattr(training, 'load_checkpoint', consuming)
    random.seed(91); np.random.seed(92); torch.manual_seed(93)
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    if reject_source:
        with pytest.raises(ValueError, match='constructed source rejection'):
            training.preflight_quadratic_source({**cfg, 'quadratic_functions': 'energy_only'}, source)
    else:
        training.preflight_quadratic_source({**cfg, 'quadratic_functions': 'energy_only'}, source)
    assert random.getstate() == py
    after = np.random.get_state()
    assert after[0] == np_state[0] and after[2:] == np_state[2:]
    np.testing.assert_array_equal(after[1], np_state[1])
    torch.testing.assert_close(torch.get_rng_state(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ('energy_only', 'all_poly'))
def test_exact_resume_changes_only_executor_and_preserves_optimizer_and_rng(tmp_path, synthetic_source, mode):
    source, cfg, _, _, _ = synthetic_source
    training.fit(cfg, {}, tmp_path / 'generic', warm_start=source)
    step_one = tmp_path / 'generic/step_000001.pt'
    original_bytes = step_one.read_bytes()
    training.fit({**cfg, 'quadratic_functions': mode}, {}, tmp_path / 'resumed', resume=step_one)
    generic_model, _, _, generic = training.load_checkpoint(tmp_path / 'generic/last.pt')
    accelerated_model, _, _, resumed = training.load_checkpoint(tmp_path / 'resumed/last.pt')
    assert step_one.read_bytes() == original_bytes
    for block in ('model', 'encoder', 'head'):
        assert generic[block].keys() == resumed[block].keys()
        for name, value in generic[block].items():
            if value.dtype.is_floating_point:
                torch.testing.assert_close(resumed[block][name], value, rtol=2e-8, atol=2e-11)
            else:
                torch.testing.assert_close(resumed[block][name], value, rtol=0, atol=0)
    assert generic['rng_numpy'] == resumed['rng_numpy']
    torch.testing.assert_close(generic['rng_torch'], resumed['rng_torch'], rtol=0, atol=0)
    assert generic['optimizer']['param_groups'] == resumed['optimizer']['param_groups']
    for key, state in generic['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(resumed['optimizer']['state'][key][name], value, rtol=2e-8, atol=2e-11)
    for graph in (generic_model, accelerated_model):
        assert all(not fn.quadratic_execution for fn in graph.function_modules().values())


@pytest.mark.parametrize('bad', ('fresh_nn', 'source_nn', 'cubic', 'duplicate', 'source_config', 'source_architecture', 'source_mode', 'u'))
def test_illegal_opt_in_is_rejected_before_any_event_read(tmp_path, monkeypatch, synthetic_source, bad):
    source, cfg, graph, _, physics = synthetic_source
    requested = {**cfg, 'quadratic_functions': 'energy_only'}
    old_model, old_encoder, old_head, obj = training.load_checkpoint(source)
    if bad == 'source_nn':
        old_model = PowerGraph(physics, [.2, .3, .1], rank=2, hidden=8)
    elif bad in ('cubic', 'duplicate'):
        inputs = old_model.node_energy.inputs
        powers = np.zeros((2, inputs), dtype=np.int64)
        if bad == 'cubic':powers[1, 0] = 3
        old_model.node_energy = PolynomialFunction(powers, np.ones((2, 1)))
    elif bad == 'source_config':
        obj['config']['quadratic_functions'] = 'unchecked_future_mode'
    elif bad == 'source_architecture':
        obj['config']['latent_per_machine'] = 4
    elif bad == 'source_mode':
        obj['synthetic'] = False
    elif bad == 'u':
        requested['arm'] = 'U'
    untouched = copy.deepcopy(obj['config'])
    monkeypatch.setattr(training, 'load_checkpoint', lambda *args, **kwargs: (old_model, old_encoder, old_head, obj))
    monkeypatch.setattr(training, 'fixture', lambda: pytest.fail('Illegal executor opt-in must reject before data'))
    monkeypatch.setattr(training, 'load_events', lambda *args, **kwargs: pytest.fail('Illegal executor opt-in must reject before TRAIN payloads'))
    with pytest.raises(ValueError):
        training.fit(requested, {}, tmp_path / 'illegal', warm_start=None if bad == 'fresh_nn' else source)
    assert obj['config'] == untouched
    assert not (tmp_path / 'illegal').exists()


@pytest.mark.parametrize('invalid', ('quadratic', True, None, 1))
def test_unknown_or_nonstring_execution_mode_rejects_before_source_and_data(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(training, 'load_checkpoint', lambda *args, **kwargs: pytest.fail('Invalid option must not read checkpoint'))
    monkeypatch.setattr(training, 'fixture', lambda: pytest.fail('Invalid option must not read data'))
    with pytest.raises(ValueError, match='quadratic_functions'):
        training.fit({**config(), 'quadratic_functions': invalid}, {}, tmp_path / 'illegal')


def test_new_execution_option_does_not_relax_scientific_resume_config(tmp_path, synthetic_source):
    source, cfg, _, _, _ = synthetic_source
    training.fit(cfg, {}, tmp_path / 'generic', warm_start=source)
    with pytest.raises(ValueError, match='Exact resume config changed: lr'):
        training.fit({**cfg, 'quadratic_functions': 'energy_only', 'lr': .0006}, {}, tmp_path / 'illegal',
                     resume=tmp_path / 'generic/step_000001.pt')


@pytest.mark.parametrize('mode', ('off', 'energy_only'))
@pytest.mark.parametrize('source_kind', ('warm_start', 'resume'))
def test_relative_training_source_uses_project_root_under_unrelated_cwd(tmp_path, monkeypatch, synthetic_source, mode, source_kind):
    source, cfg, _, _, _ = synthetic_source
    # Construct a zero-step protocol for this path-only regression; no loss,
    # gradient, optimizer update, monitor, or model-field execution is needed.
    obj = torch.load(source, weights_only=True)
    obj['config'].update(steps=0, milestones=[])
    torch.save(obj, source)
    requested = {**cfg, 'steps': 0, 'milestones': [], 'quadratic_functions': mode}
    relative = 'sources/constructed.pt'
    original_load, loaded_paths = training.load_checkpoint, []
    def rooted(value):
        return source if str(value).replace('\\', '/') == relative else Path(value)
    def load(path, device='cpu'):
        loaded_paths.append(Path(path))
        assert Path(path).is_absolute() and Path(path) == source
        return original_load(path, device)
    monkeypatch.setattr(training, 'project_path', rooted)
    monkeypatch.setattr(training, 'load_checkpoint', load)
    unrelated = tmp_path / 'unrelated-cwd'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    report = training.fit(requested, {}, tmp_path / 'output', **{source_kind: relative})
    assert report['steps_completed'] == 0
    assert loaded_paths == [source] * (1 if mode == 'off' else 2)
    assert not (unrelated / relative).exists()
