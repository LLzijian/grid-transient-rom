"""Synthetic contracts only: no real checkpoint, HDF5 or event payload access."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from task4.data import train_scales
from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder, PredictiveHead
from task4.paths import ROOT
from task4.symbolic import PolynomialFunction
from task4.synthetic import fixture
from task4.training import optimizer_groups


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('quadratic_training_probe_test', ROOT/'scripts/probe_quadratic_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_source():
    events, physics = fixture(events=2, steps=13)
    events[1].label = 'unstable'  # Synthetic selection label, never scientific evidence.
    scales = train_scales(events, physics)
    cfg = {'arm': 'P', 'synthetic': False, 'hidden': 32, 'latent_per_machine': 2, 'dtype': 'float64',
           'autonomous_only': True, 'horizons': [5.], 'integration_dt': .01, 'steps': 400,
           'shooting_fraction': 0., 'final_autonomous_fraction': 1., 'batch_size': 2, '_optimizer_blocks': True,
           'lr': 1e-4, 'grad_clip': 100., 'anchor_horizons': [.02, .05], 'complexity_weight': 1e-5,
           'anchor_weight': 0., 'latent_weight': 0., 'torque_weight': 0., 'continuity_weight': 0.,
           '_event_ids': [e.id for e in events]}
    torch.manual_seed(340)
    model = PowerGraph(physics, scales, rank=2, hidden=32)
    rng = np.random.default_rng(424)
    for name, module in list(model.function_modules().items()):
        powers = np.zeros((4, module.inputs), dtype=np.int64)
        powers[1, 0], powers[2, 1], powers[3, :2] = 1, 2, 1
        coefficients = rng.normal(0, .001, (4, module.outputs))
        coefficients[0, ::2] = 0.
        setattr(model, name, PolynomialFunction(powers, coefficients))
    encoder = HistoryEncoder(model, hidden=32)
    head = PredictiveHead(model, cfg['anchor_horizons'], hidden=32)
    head.requires_grad_(False)
    optimizer = torch.optim.Adam(optimizer_groups(model, encoder, head, cfg), lr=cfg['lr'])
    # Populate nonempty Adam moments so an accidental optimizer reset is detectable.
    sum(p.square().sum() for group in optimizer.param_groups for p in group['params']).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved = {'config': cfg, 'kind': 'T4-S', 'synthetic': False, 'step': 400, 'exact_resume_eligible': True,
             'scales': scales.tolist(), 'physics': physics.as_dict(), 'shooting_keys': [],
             'event_ids': [e.id for e in events], 'symbolic_functions': {name: fn.export() for name, fn in model.function_modules().items()},
             'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng_torch': torch.get_rng_state(),
             'rng_numpy': np.random.default_rng(591).bit_generator.state}
    return model, encoder, head, saved, events, physics


def install_synthetic_io(script, monkeypatch, tmp_path):
    model, encoder, head, saved, events, physics = synthetic_source()
    checkpoint = tmp_path/'mock.pt'
    checkpoint.write_bytes(b'synthetic checkpoint identity; tensor loader is mocked')
    calls = {'authorize': [], 'splits': [], 'loads': [], 'traces': [], 'adam_steps_before': []}
    monkeypatch.setattr(script, 'check_authorized', lambda flag: calls['authorize'].append(flag))
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', '1000000000000')
    monkeypatch.setattr(script, 'project_path', lambda path: Path(path))
    monkeypatch.setattr(script, 'write_json', lambda path, value: Path(path).write_text(json.dumps(value, allow_nan=False), encoding='utf-8'))
    monkeypatch.setattr(script, 'read_json', lambda path: {'synthetic_io_contract_test': True})

    def load(path, device):
        assert Path(path) == checkpoint and device == 'cpu'
        calls['loads'].append(str(path))
        return copy.deepcopy((model, encoder, head, saved))

    def data(config, split):
        assert config == {'synthetic_io_contract_test': True}
        calls['splits'].append(split)
        assert split == 'train'
        return events, physics, {'split': 'train', 'test_payload_read': False}

    def trace(candidate):
        calls['traces'].append([name for name, fn in candidate.function_modules().items() if fn.quadratic_execution])
        assert candidate.reuse_origin_terms and candidate.checkpoint_steps == 0

    objective = script.batch_objective

    def checked_objective(model, encoder, head, panel, horizon, cfg, targets, rng, bank, optimizer, segment, factor, teacher):
        assert horizon == 5. and segment is None and factor == 0. and teacher is None and targets is None
        assert cfg == saved['config']
        assert [e.id for e in panel] == [e.id for e in events]
        assert all(p.requires_grad for module in (model, encoder) for p in module.parameters())
        assert not any(p.requires_grad for p in head.parameters())
        calls['adam_steps_before'].append([float(state['step']) for state in optimizer.state.values()])
        return objective(model, encoder, head, panel, horizon, cfg, targets, rng, bank, optimizer, segment, factor, teacher)

    monkeypatch.setattr(script, 'load_checkpoint', load)
    monkeypatch.setattr(script, 'load_events', data)
    monkeypatch.setattr(script, 'trace_field', trace)  # Trace itself is covered by executor QA.
    monkeypatch.setattr(script, 'batch_objective', checked_objective)
    return checkpoint, calls, saved


def test_panel_is_first_by_label_without_loss_or_event_substitution(script):
    events = [SimpleNamespace(id='u1', label='unstable'), SimpleNamespace(id='s1', label='stable'),
              SimpleNamespace(id='s2', label='stable'), SimpleNamespace(id='u2', label='unstable')]
    assert [event.id for event in script.fixed_panel(events)] == ['s1', 'u1']
    with pytest.raises(ValueError, match='both a stable and an unstable'):
        script.fixed_panel(events[1:3])


@pytest.mark.parametrize('mutation', ['ids', 'scales', 'physics', 'kind', 'step', 'auxiliary', 'degree'])
def test_source_contract_rejects_mismatch_before_any_probe(script, mutation):
    model, _, _, saved, events, physics = synthetic_source()
    if mutation == 'ids': saved['event_ids'].reverse()
    if mutation == 'scales': saved['scales'][0] *= 2
    if mutation == 'physics': saved['physics']['gain'][0] *= 2
    if mutation == 'kind': saved['kind'] = 'T4-N'
    if mutation == 'step': saved['step'] = 200
    if mutation == 'auxiliary': saved['config']['latent_weight'] = .1
    if mutation == 'degree':
        model.port = PolynomialFunction([[3]+[0]*(model.port.inputs-1)], [[1., 1.]])
    with pytest.raises(ValueError):
        script.validate_source(model, saved, events, physics, train_scales(events, physics))


def test_three_modes_restore_same_adam_and_record_complete_parameter_comparisons(script, monkeypatch, tmp_path):
    checkpoint, calls, saved = install_synthetic_io(script, monkeypatch, tmp_path)
    output = tmp_path/'probe'
    result = script.probe(str(checkpoint), str(output))
    assert result['complete'] and result['checkpoint_unchanged'] and result['all_modes_completed']
    assert result['scientific_candidate_saved'] is False
    assert calls['authorize'] == [False] and calls['splits'] == ['train'] and len(calls['loads']) == 4
    assert calls['traces'] == [[], ['node_energy', 'edge_energy'], list(saved['symbolic_functions'])]
    assert len(calls['adam_steps_before']) == 3
    assert all(steps and set(steps) == {1.} for steps in calls['adam_steps_before'])
    assert [row['mode'] for row in result['modes']] == list(script.MODES)
    for row in result['modes']:
        assert row['status'] == 'completed'
        assert set(('loss', 'backward', 'clip', 'adam')).issubset(row['seconds'])
        assert all(row['terms'][key] == 0 for key in ('continuity', 'anchor', 'latent', 'torque', 'teacher_field', 'teacher_gradient'))
        with np.load(output/row['artifact'], allow_pickle=False) as arrays:
            assert len(arrays['parameter_names']) == len(row['parameters'])
            assert all(f'gradient_{i}' in arrays and f'update_{i}' in arrays for i in range(len(row['parameters'])))
        if row['mode'] != 'reference':
            comparison = row['comparison_to_reference']
            assert comparison['available'] and comparison['initial_field']['within_tolerance'] and comparison['loss']['within_tolerance']
            assert len(comparison['parameters']) == len(row['parameters'])
            assert all(item[key]['within_tolerance'] for item in comparison['parameters'] for key in ('gradient', 'update'))
    assert not list(output.glob('*.pt'))
    json.loads((output/'probe_summary.json').read_text(encoding='utf-8'))
    with pytest.raises(FileExistsError):
        script.probe(str(checkpoint), str(output))


def test_nonfinite_gradient_failure_preserves_fixed_mode_artifact(script, monkeypatch, tmp_path):
    checkpoint, _, _ = install_synthetic_io(script, monkeypatch, tmp_path)
    def failed(source, digest, mode, panel, deadline, row, arrays):
        arrays['gradient_0'] = np.array([np.inf])
        row['raw_global_gradient_norm'] = float('inf')
        raise FloatingPointError('synthetic nonfinite gradient')
    monkeypatch.setattr(script, 'run_mode', failed)
    output = tmp_path/'failed'
    result = script.probe(str(checkpoint), str(output))
    assert result['status'] == 'probe_complete_with_failures' and not result['all_modes_completed']
    assert len(result['modes']) == 3
    saved = json.loads((output/'probe_summary.json').read_text(encoding='utf-8'))
    assert saved['modes'][0]['raw_global_gradient_norm'] is None and saved['nonfinite_json_fields']
    with np.load(output/'reference_diagnostics.npz', allow_pickle=False) as arrays:
        assert np.isinf(arrays['gradient_0']).all()


def test_deadline_keeps_partial_diagnostics_and_checks_source(script, monkeypatch, tmp_path):
    checkpoint, _, _ = install_synthetic_io(script, monkeypatch, tmp_path)
    def interrupted(source, digest, mode, panel, deadline, row, arrays):
        arrays['initial_field'] = np.zeros((2, 15))
        raise TimeoutError('synthetic stop during one fixed mode')
    monkeypatch.setattr(script, 'run_mode', interrupted)
    output = tmp_path/'partial'
    result = script.probe(str(checkpoint), str(output))
    assert not result['complete'] and result['stop_reason'] == 'save_deadline'
    assert result['checkpoint_unchanged'] and len(result['modes']) == 1
    assert (output/'reference_diagnostics.npz').exists()


def test_final_checkpoint_hash_mismatch_cannot_be_called_complete(script, monkeypatch, tmp_path):
    checkpoint, _, _ = install_synthetic_io(script, monkeypatch, tmp_path)
    monkeypatch.setattr(script, 'sha256', lambda path, deadline=None: 'before' if deadline is not None else 'after')
    def failed(*args):
        raise FloatingPointError('synthetic failure avoids unnecessary numerical work')
    monkeypatch.setattr(script, 'run_mode', failed)
    result = script.probe(str(checkpoint), str(tmp_path/'changed'))
    assert not result['checkpoint_unchanged'] and not result['complete'] and result['status'] == 'source_changed'


def test_authorization_and_finite_deadline_precede_input_access(script, monkeypatch):
    def denied(synthetic):
        assert synthetic is False
        raise PermissionError('not managed')
    def forbidden(*args, **kwargs):
        raise AssertionError('input/output access before authorization/deadline')
    monkeypatch.setattr(script, 'check_authorized', denied)
    monkeypatch.setattr(script, 'project_path', forbidden)
    with pytest.raises(PermissionError, match='not managed'):
        script.probe('unused', 'unused')
    monkeypatch.setattr(script, 'check_authorized', lambda synthetic: None)
    monkeypatch.delenv('TASK4_SAVE_DEADLINE_UNIX', raising=False)
    with pytest.raises(PermissionError, match='finite managed'):
        script.probe('unused', 'unused')


def test_overflow_in_parameter_difference_has_null_reason_and_fails_comparison(script):
    result = script.difference([1e308], [-1e308])
    assert not result['finite'] and not result['within_tolerance']
    serialized = script.report_for_json({'comparison': result})
    assert serialized['comparison']['max_absolute'] is None and serialized['nonfinite_json_fields']
    json.dumps(serialized, allow_nan=False)
