"""Synthetic update and exact-resume checks for the bounded joint trainer."""
import json
from pathlib import Path
import numpy as np
import torch
from task4.dynamics import PowerGraph, UnstructuredGraph
from task4.encoder import HistoryEncoder, PredictiveHead
from task4.synthetic import fixture
from task4.paths import write_json
import task4.training as training


def test_named_blocks_preserve_adam_and_report_actual_updates():
    _, physics = fixture(events=1, steps=12)
    torch.manual_seed(19)
    model = PowerGraph(physics, [.2, .3, .1], hidden=8)
    encoder = HistoryEncoder(model, 8)
    head = PredictiveHead(model, [.02], 8)
    groups = training.optimizer_groups(model, encoder, head, {'lr': .0003})
    assert {'energy', 'coupling', 'exchange', 'dissipation', 'port', 'encoder', 'head'} <= {g['name'] for g in groups}
    params = [p for g in groups for p in g['params']]
    assert len({id(p) for p in params}) == len(params)
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
    grouped = torch.optim.Adam(groups, lr=.0003)
    flat = torch.optim.Adam(reference, lr=.0003)
    for _ in range(2):
        for p, q in zip(params, reference):
            p.grad = torch.randn_like(p); q.grad = p.grad.clone()
        before, stats = training.capture_updates(grouped)
        grouped.step(); flat.step()
        report = training.summarize_updates(before, stats)
        for p, q in zip(params, reference):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        for name, row in report.items():
            exact = sum(float((p.detach()-old).square().sum()) for key, p, old in before if key == name)**.5
            np.testing.assert_allclose(row['update_l2'], exact, rtol=1e-14, atol=0)
            assert row['gradient_l2'] > 0


def test_shooting_learning_rate_is_independent_of_coupling():
    from task4.objectives import ShootingBank
    events, physics = fixture(events=1, steps=21)
    model = PowerGraph(physics, [.2, .3, .1], hidden=8)
    encoder = HistoryEncoder(model, 8)
    head = PredictiveHead(model, [.02], 8)
    cfg = {'lr': .0003, 'block_lr_scales': {'coupling': 5.}}
    optim = torch.optim.Adam(training.optimizer_groups(model, encoder, head, cfg), lr=cfg['lr'])
    optim.task4_shooting_lr = .0006
    ShootingBank().get(events[0], 15, encoder, optim, 'cpu')
    assert optim.param_groups[-1]['lr'] == .0006 and optim.param_groups[-1]['name'] == 'shooting'


def test_exact_resume_new_and_legacy_optimizer_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'project_path', lambda path: Path(path))
    monkeypatch.setattr(training, 'write_json', lambda path, value: write_json(path, value, root=tmp_path))
    cfg = dict(name='synthetic_resume', seed=41, arm='P', latent_per_machine=2,
               hidden=8, device='cpu', batch_size=2, lr=.0003, grad_clip=100.,
               integration_dt=.01, steps=2, horizons=[.02], milestones=[1,2],
               anchor_horizons=[.02], anchor_weight=.2, latent_weight=.1,
               torque_weight=.1, complexity_weight=1e-5, shooting_fraction=0.,
               final_autonomous_fraction=.35, segment_schedule=[.25],
               continuity_weight=1., checkpoint_steps=0, save_every=1,
               monitor_count=1, synthetic=True)
    training.fit(cfg, {}, tmp_path/'full')
    training.fit(cfg, {}, tmp_path/'resume', resume=tmp_path/'full/step_000001.pt')
    _, _, _, full = training.load_checkpoint(tmp_path/'full/last.pt')
    _, _, _, resumed = training.load_checkpoint(tmp_path/'resume/last.pt')
    for block in ['model','encoder','head']:
        for key, value in full[block].items():
            torch.testing.assert_close(value, resumed[block][key], rtol=0, atol=0)
    # Produce the former flat layout in its original module parameter order.
    original_groups = training.optimizer_groups
    monkeypatch.setattr(training, 'optimizer_groups', lambda m,e,h,c: [{'params': [p for module in [m,e,h] for p in module.parameters() if p.requires_grad]}])
    training.fit(cfg, {}, tmp_path/'legacy_full')
    monkeypatch.setattr(training, 'optimizer_groups', original_groups)
    legacy = torch.load(tmp_path/'legacy_full/step_000001.pt', weights_only=True)
    legacy['config'].pop('_optimizer_blocks')
    torch.save(legacy, tmp_path/'legacy.pt')
    training.fit(cfg, {}, tmp_path/'legacy_resume', resume=tmp_path/'legacy.pt')
    _, _, _, old_resumed = training.load_checkpoint(tmp_path/'legacy_resume/last.pt')
    for block in ['model','encoder','head']:
        for key, value in full[block].items():
            torch.testing.assert_close(value, old_resumed[block][key], rtol=0, atol=0)
    rows = [json.loads(line) for line in (tmp_path/'resume/steps.jsonl').read_text().splitlines()]
    assert rows[0]['blocks']['energy']['update_l2'] > 0


def test_u_width_keeps_shared_encoder_architecture():
    _, physics = fixture(events=1, steps=12)
    p = PowerGraph(physics, [.2, .3, .1], hidden=32)
    u = UnstructuredGraph(physics, [.2, .3, .1], hidden=32, u_hidden=64)
    pe, ue = HistoryEncoder(p, 32), HistoryEncoder(u, 32)
    ue.load_state_dict(pe.state_dict())
    h = torch.randn(2, 11, p.n, 3, dtype=torch.float64)
    torch.testing.assert_close(pe(h), ue(h), rtol=0, atol=0)
    assert ue.gru.hidden_size == 32 and u.rank == 2
