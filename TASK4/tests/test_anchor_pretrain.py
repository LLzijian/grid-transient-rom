"""Bounded synthetic engineering checks; no real event payloads are opened."""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from task4.paths import ROOT, read_json, write_json
from task4.synthetic import fixture
import task4.training as training


@pytest.fixture
def anchor_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('anchor_pretrain_test_module', ROOT/'scripts/pretrain_anchor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_windows_keep_real_prefix_and_unique_short_events(anchor_script):
    np.testing.assert_array_equal(anchor_script.window_indices(101),
                                  np.unique(np.linspace(10, 99, 8, dtype=np.int64)))
    np.testing.assert_array_equal(anchor_script.window_indices(12), [10])
    with pytest.raises(ValueError):
        anchor_script.window_indices(11)


def test_each_tau_is_normalized_by_its_own_availability(anchor_script):
    values = torch.tensor([[1., 3., 5.], [7., 9., 11.]], dtype=torch.float64)
    prediction = values[:, None, :, None].expand(-1, 1, -1, 3).clone().requires_grad_(True)
    available = torch.tensor([[True, False, True], [True, True, False]])
    target = torch.zeros_like(prediction)
    target[~available[:, None, :, None].expand_as(target)] = float('nan')
    loss, per_tau, counts = anchor_script.masked_anchor_loss(prediction, target, available,
                                                            torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(per_tau, torch.tensor([25., 81., 25.], dtype=torch.float64))
    torch.testing.assert_close(loss, per_tau.mean())
    assert counts.tolist() == [2, 1, 1]
    loss.backward()
    assert torch.count_nonzero(prediction.grad[~available[:, None, :, None].expand_as(prediction)]) == 0
    scales = torch.tensor([0.1, 0.3, 0.02], dtype=torch.float64)
    loss, _, _ = anchor_script.masked_anchor_loss(torch.zeros_like(prediction),
                                                  scales.expand_as(prediction), available, scales)
    torch.testing.assert_close(loss, torch.ones((), dtype=torch.float64))


def test_cache_records_all_train_sources_and_real_horizon_masks(anchor_script, tmp_path, monkeypatch):
    events, physics = fixture(events=2, steps=15)
    monkeypatch.setattr(anchor_script, 'write_json', lambda path, obj: write_json(path, obj, root=tmp_path))
    arrays, manifest = anchor_script.build_anchor_cache(events, physics, [0.02, 0.05], 8,
                                                        tmp_path, time.time()+120)
    assert manifest['complete'] and manifest['all_train_source_ids'] == [e.id for e in events]
    assert manifest['cache_wall_seconds'] >= 0
    assert not manifest['mask_is_model_input'] and not manifest['test_payload_read']
    for k, (ei, index) in enumerate(zip(arrays['event_index'], arrays['sample_index'])):
        np.testing.assert_array_equal(arrays['history'][k], events[ei].y[index-10:index+1])
        np.testing.assert_array_equal(arrays['mask'][k], [index+2 < 15, index+5 < 15])
    # Last true-prefix windows remain registered even if no tau is available.
    assert not arrays['mask'][arrays['sample_index'] == 13].any()
    assert (tmp_path/'anchor_cache.npz').is_file()
    assert read_json(tmp_path/'anchor_cache_manifest.json')['complete']


def test_partial_cache_deadline_preserves_explicit_incomplete_cache(anchor_script, tmp_path, monkeypatch):
    events, physics = fixture(events=2, steps=15)
    monkeypatch.setattr(anchor_script, 'write_json', lambda path, obj: write_json(path, obj, root=tmp_path))
    calls = []
    def bounded(deadline, estimated_seconds=0):
        calls.append(deadline)
        if len(calls) == 3:
            raise TimeoutError('synthetic deadline fixture')
    monkeypatch.setattr(anchor_script, 'check_deadline', bounded)
    with pytest.raises(TimeoutError):
        anchor_script.build_anchor_cache(events, physics, [0.02, 0.05], 8, tmp_path, time.time()+120)
    manifest = read_json(tmp_path/'anchor_cache_manifest.json')
    assert not manifest['complete'] and manifest['windows_cached'] == 2
    assert manifest['all_train_source_ids'] == [e.id for e in events]
    with np.load(tmp_path/'anchor_cache.npz', allow_pickle=False) as cache:
        assert cache['history'].shape[0] == 2


def test_unauthorized_command_cannot_read_payload_or_create_output(anchor_script, tmp_path, monkeypatch):
    visited = []
    def denied(synthetic):
        assert synthetic is False
        visited.append('authorization')
        raise PermissionError('synthetic denied authorization fixture')
    monkeypatch.setattr(anchor_script, 'check_authorized', denied)
    monkeypatch.setattr(anchor_script, 'load_events', lambda *a, **k: pytest.fail('payload opened'))
    monkeypatch.setattr(anchor_script, 'read_json', lambda *a, **k: pytest.fail('configuration read before gate'))
    with pytest.raises(PermissionError):
        anchor_script.pretrain(tmp_path/'forbidden')
    assert visited == ['authorization'] and not (tmp_path/'forbidden').exists()


def _synthetic_environment(module, monkeypatch, tmp_path):
    events, physics = fixture(events=2, steps=18)
    authorized = []
    monkeypatch.setattr(module, 'check_authorized', lambda synthetic: authorized.append(synthetic))
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', str(time.time()+120))
    def local_path(value):
        path = Path(value).resolve()
        if not path.is_relative_to(tmp_path.resolve()):
            raise PermissionError('test output outside isolated fixture')
        return path
    monkeypatch.setattr(module, 'project_path', local_path)
    monkeypatch.setattr(training, 'project_path', local_path)
    monkeypatch.setattr(module, 'write_json', lambda path, obj: write_json(path, obj, root=tmp_path))
    def configuration(path):
        if Path(path).name == 'project.json':
            return {'history_ms': 100, 'sample_dt': 0.01}
        if Path(path).name == 'experiment.json':
            return {'anchor_horizons': [0.02, 0.05], 'grad_clip': 100.0}
        return read_json(path)
    monkeypatch.setattr(module, 'read_json', configuration)
    def load(config, split):
        assert authorized == [False] and split == 'train'
        return events, physics, {'split': 'train', 'test_payload_read': False,
                                 'fixture_notice': 'synthetic engineering test data only'}
    monkeypatch.setattr(module, 'load_events', load)
    return events, physics


def test_pretrain_checkpoint_warm_start_compatible_and_dynamics_unchanged(anchor_script, tmp_path, monkeypatch):
    events, _ = _synthetic_environment(anchor_script, monkeypatch, tmp_path)
    snapshots = []
    original_checkpoint = anchor_script.checkpoint
    def record_snapshot(path, model, encoder, *args, **kwargs):
        snapshots.append(({key: value.detach().clone() for key, value in model.state_dict().items()},
                          {key: value.detach().clone() for key, value in encoder.state_dict().items()}))
        return original_checkpoint(path, model, encoder, *args, **kwargs)
    monkeypatch.setattr(anchor_script, 'checkpoint', record_snapshot)
    output = tmp_path/'pretrain'
    report = anchor_script.pretrain(output, steps=2, windows=2, batch_size=2, device='cpu', save_every=1)
    model, encoder, head, obj = training.load_checkpoint(output/'last.pt')
    assert report['steps_completed'] == 2 and report['cache_complete']
    assert not report['dynamics_optimized'] and not report['autonomous_performance_evaluated']
    assert obj['event_ids'] == [event.id for event in events]
    assert obj['synthetic'] is False and obj['config']['arm'] == 'P'
    assert obj['config']['hidden'] == 32 and obj['config']['latent_per_machine'] == 2
    assert obj['config']['warm_start_only'] and not obj['exact_resume_eligible']
    assert obj['step'] == 2 and len(obj['records']) == 2
    assert all('q_rms' in record and 'loss_per_tau' in record for record in obj['records'])
    for key, value in snapshots[0][0].items():
        torch.testing.assert_close(snapshots[-1][0][key], value, rtol=0, atol=0)
    assert any(not torch.equal(value, snapshots[-1][1][key]) for key, value in snapshots[0][1].items())
    assert all(parameter.requires_grad for parameter in model.parameters())  # joint-fit warm start is trainable
    assert encoder.gru.hidden_size == 32 and head.horizons == [0.02, 0.05]
    with pytest.raises(FileExistsError):
        anchor_script.pretrain(output, steps=1, device='cpu')


def test_cache_timeout_keeps_valid_step_zero_last_checkpoint(anchor_script, tmp_path, monkeypatch):
    _synthetic_environment(anchor_script, monkeypatch, tmp_path)
    def timeout(*args, **kwargs):
        raise TimeoutError('synthetic cache deadline fixture')
    monkeypatch.setattr(anchor_script, 'build_anchor_cache', timeout)
    output = tmp_path/'expired_cache'
    report = anchor_script.pretrain(output, steps=2, windows=2, batch_size=2, device='cpu')
    _, _, _, obj = training.load_checkpoint(output/'last.pt')
    assert report['stop_reason'] == 'save_deadline' and report['steps_completed'] == 0
    assert obj['step'] == 0 and not obj['exact_resume_eligible']
    assert (output/'pretraining_summary.json').is_file()
