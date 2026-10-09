"""Synthetic contracts for strict-autonomous N -> TRAIN-only C adaptation."""
import copy
import importlib.util
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder
from task4.objectives import batch_objective
from task4.paths import ROOT, write_json
from task4.synthetic import fixture


@pytest.fixture
def compression(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('compression_autonomous_test', ROOT/'scripts/compress_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ForbiddenHead(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError('C must not call the prediction head')


class ForbiddenAuxiliary:
    def get(self, *args, **kwargs):
        raise AssertionError('C must not query anchors or shooting states')


def setup_source():
    events, physics = fixture(events=1, steps=14)
    torch.manual_seed(401)
    model = PowerGraph(physics, [.2, .3, .1], rank=2, hidden=32)
    encoder = HistoryEncoder(model, hidden=32)
    cfg = {'autonomous_only': True, 'synthetic': False, 'arm': 'P',
           'hidden': 32, 'latent_per_machine': 2, 'integration_dt': .01,
           'anchor_weight': 0., 'latent_weight': 0., 'torque_weight': 0.,
           'teacher_field_weight': 0., 'teacher_gradient_weight': 0.,
           'complexity_weight': 1e-5, 'continuity_weight': 0., 'torque_scale': 1.}
    return events, physics, model, encoder, cfg


def assert_c_contract(cfg):
    assert cfg['autonomous_only'] is False
    assert cfg['compression_source_autonomous_only'] is True
    assert cfg['anchor_weight'] == cfg['latent_weight'] == cfg['torque_weight'] == 0.
    assert cfg['teacher_field_weight'] == cfg['teacher_gradient_weight'] == .1
    assert cfg['teacher_usage'] == 'TRAIN_only_compression_adaptation'
    assert cfg['prediction_head_used_or_deployed'] is False
    assert cfg['final_single_initialization_5s_finetune_completed'] is False


def test_autonomous_source_enables_teacher_losses_without_head(compression):
    events, _, model, encoder, source = setup_source()
    original = copy.deepcopy(source)
    cfg = compression.adaptation_config(source, ROOT/'unused_teacher.pt', 16, 'node_energy', 'cpu', 419)
    assert_c_contract(cfg)
    assert source == original  # N stays strict-autonomous; only C changes mode.
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        teacher.log_coupling.add_(.2)
    teacher.requires_grad_(False)
    forbidden = ForbiddenAuxiliary()
    loss, terms = batch_objective(model, encoder, ForbiddenHead(), events, .02, cfg,
                                  forbidden, np.random.default_rng(91), forbidden, None,
                                  segment_seconds=None, teacher=teacher)
    assert torch.isfinite(loss)
    assert terms['teacher_field'].item() > 0
    assert terms['teacher_gradient'].item() > 0
    assert terms['anchor'].item() == terms['continuity'].item() == 0
    expected = (terms['rollout']+cfg['complexity_weight']*terms['regularization']
                +.1*terms['teacher_field']+.1*terms['teacher_gradient'])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in encoder.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in encoder.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.isfinite(p.grad).all() for module in (model, encoder)
               for p in module.parameters() if p.grad is not None)


def test_saved_c_asset_cannot_claim_final_autonomous_d(compression, monkeypatch, tmp_path):
    events, physics, model, encoder, source = setup_source()
    original = copy.deepcopy(source)
    obj = {'config': source, 'kind': 'T4-N', 'synthetic': False,
           'event_ids': [event.id for event in events], 'scales': model.scales.tolist(),
           'physics': physics.as_dict()}
    authorized, saved = [], []
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', str(time.time()+120))
    monkeypatch.setattr(compression, 'project_path', Path)
    monkeypatch.setattr(compression, 'check_authorized', lambda synthetic: authorized.append(synthetic))
    monkeypatch.setattr(compression, 'read_json', lambda path: (
        {'history_ms': 100, 'sample_dt': .01} if Path(path).name == 'project.json' else {}))
    monkeypatch.setattr(compression, 'write_json',
                        lambda path, value: write_json(path, value, root=tmp_path))

    def load_synthetic_checkpoint(*args, **kwargs):
        assert authorized == [False]
        return copy.deepcopy((model, encoder, ForbiddenHead(), obj))

    def load_synthetic_events(config, split):
        assert authorized == [False] and split == 'train'
        return events, physics, {'synthetic_fixture': True, 'test_payload_read': False}

    def capture_checkpoint(path, m, e, head, bank, optimizer, config, *args, **kwargs):
        saved.append(copy.deepcopy(config))
        assert kwargs['boundary_complete'] is False
        Path(path).write_bytes(b'synthetic-placeholder-no-real-checkpoint')

    def stop_before_collection(*args, **kwargs):
        raise TimeoutError('synthetic stop after asset metadata capture')

    monkeypatch.setattr(compression, 'load_checkpoint', load_synthetic_checkpoint)
    monkeypatch.setattr(compression, 'load_events', load_synthetic_events)
    monkeypatch.setattr(compression, 'train_scales', lambda *args: model.scales.numpy())
    monkeypatch.setattr(compression, 'save_checkpoint', capture_checkpoint)
    monkeypatch.setattr(compression, 'collect_teacher_states', stop_before_collection)
    result = compression.compress(tmp_path/'unused_teacher.pt', tmp_path/'output')
    assert len(saved) == 1
    assert_c_contract(saved[0])
    assert saved[0]['warm_start_only'] is True
    assert source == original
    assert result['complete'] is False
    assert result['final_single_initialization_5s_finetune_completed'] is False
    assert result['stop_reason'] == 'save_deadline'
