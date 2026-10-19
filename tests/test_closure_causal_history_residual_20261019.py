"""Targeted disposable QA; no project val/test and no formal scientific fit."""
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
from closure_discovery_benchmarks import toy_source
import closure_causal_history_residual_20261019 as implementation
from closure_causal_history_residual_20261019 import (
    CausalHistoryResidual, history_features, make_optimizer, checkpoint, restore,
    runtime_model, prediction, train_update, fixture, truths, synchronize, atomic_json, sha, source_identity,
)


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def small():
    base = runtime_model(toy_source(), [truths()['node'][0]], 5101, 'cpu')
    h = torch.tensor(np.random.default_rng(17).normal(size=(16, 11, 30)) * .04,
                     dtype=torch.float64)
    return base, h


def test_zero_head_encode_rollout_shared_gradients():
    base, h = small()
    model = CausalHistoryResidual.from_train(copy.deepcopy(base), h)
    assert torch.equal(model.encode(h), base.encode(h))
    obs = torch.cat((h, h[:, -1:].expand(-1, 2, -1)), 1)
    a, b = prediction(base, obs), prediction(model, obs)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, p in base.named_parameters():
        q = dict(model.base.named_parameters())[name]
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad), name


def test_public_omega_nonunit_mass_angle_equivariance_permutation():
    _, h = small()
    mass = torch.arange(1, 11, dtype=torch.float64)
    f = history_features(h, mass)
    assert torch.equal(f[..., 1], h[:, -1, 10:20])
    assert not torch.equal(f[..., 1], h[:, -1, 10:20] / mass)
    shifted = h.clone()
    shifted[..., :10] += torch.linspace(0, 2, 11)[None, :, None]
    assert torch.allclose(history_features(shifted, mass), f, atol=1e-13, rtol=1e-12)
    perm = torch.randperm(10)
    reordered = h.reshape(16, 11, 3, 10)[..., perm].reshape(16, 11, 30)
    assert torch.allclose(history_features(reordered, mass[perm]), f[:, perm], atol=1e-14)
    coeff = torch.arange(6, dtype=torch.float64)
    assert torch.allclose(history_features(reordered, mass[perm]) @ coeff, (f @ coeff)[:, perm])
    assert torch.equal(history_features(torch.zeros_like(h), mass), torch.zeros_like(f))


def test_p_causality_and_future_observations_are_independent():
    base, h = small()
    model = CausalHistoryResidual.from_train(base, h)
    with torch.no_grad():
        model.history_coefficients[2] = .4
    changed = h.clone()
    changed[..., 20:] += .02
    assert not torch.equal(model.encode(h), model.encode(changed))
    obs = torch.cat((h, h[:, -1:].expand(-1, 2, -1)), 1)
    future = obs.clone()
    future[:, 11:] += 500
    assert torch.equal(prediction(model, obs), prediction(model, future))


def test_train_rms_only_frozen_and_original_scales_unchanged():
    base, h = small()
    old = {k: v.clone() for k, v in base.state_dict().items()}
    model = CausalHistoryResidual.from_train(base, h)
    expected = history_features(h, base.mass).square().mean((0, 1)).sqrt().clamp_min(1e-6)
    assert torch.equal(model.history_scales, expected)
    for k, v in old.items():
        assert torch.equal(base.state_dict()[k], v)
    fixed = model.source_mass.clone()
    with torch.no_grad():
        base.mass.mul_(2)
    assert torch.equal(model.source_mass, fixed)
    assert not model.history_scales.requires_grad


def test_head_optimizer_gradient_and_export_restore(tmp_path):
    base, h = small()
    model = CausalHistoryResidual.from_train(base, h)
    opt = make_optimizer(model)
    assert [g['lr'] for g in opt.param_groups] == [g['lr'] for g in make_optimizer(copy.deepcopy(base)).param_groups] + [.01]
    model.encode(h).square().mean().backward()
    assert torch.isfinite(model.history_coefficients.grad).all()
    assert model.history_coefficients.grad.abs().max() > 0
    opt.step()
    model.export(tmp_path / 'model.json')
    restored = CausalHistoryResidual.from_export(tmp_path / 'model.json')
    for k, v in model.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k]), k
    assert torch.equal(model.encode(h), restored.encode(h))
    with pytest.raises(ValueError):
        type(base).from_export(tmp_path / 'model.json')


def test_exact_next_adam_rng_update(tmp_path):
    base, h = small()
    model = CausalHistoryResidual.from_train(base, h)
    opt = make_optimizer(model)
    rng = np.random.default_rng(45)
    obs = torch.cat((h, h[:, -1:].expand(-1, 2, -1)), 1)
    scale = torch.ones(30, dtype=torch.float64)
    trace = []
    for _ in range(32):
        ix = rng.permutation(16)[:12]
        opt.zero_grad(set_to_none=True)
        value = model.encode(h[ix]).square().mean()
        value.backward()
        opt.step()
        trace.append(float(value.detach()))
    checkpoint(tmp_path, model, opt, rng, 32, trace, {'fixed': 1})
    # Next draw and next full rollout update must match exact continuation.
    expected_torch = torch.rand(4)
    train_update(model, opt, rng, obs, scale)
    other = CausalHistoryResidual.from_train(copy.deepcopy(base), h)
    other_opt = make_optimizer(other)
    other_rng = np.random.default_rng(9)
    restore(tmp_path, other, other_opt, other_rng, {'fixed': 1})
    assert torch.equal(torch.rand(4), expected_torch)
    train_update(other, other_opt, other_rng, obs, scale)
    for k, v in model.state_dict().items():
        assert torch.equal(v, other.state_dict()[k]), k
    assert rng.bit_generator.state == other_rng.bit_generator.state
    checkpoint_file = tmp_path / 'checkpoint_00032' / 'model.json'
    checkpoint_file.write_text(checkpoint_file.read_text() + ' ')
    with pytest.raises(ValueError, match='bytes changed'):
        restore(tmp_path, other, other_opt, other_rng, {'fixed': 1})


def test_profiles_and_single_initialization():
    all18, node6 = source_identity('all18'), source_identity('node6')
    assert all18['effective_caps']['item_fit'] == 540
    assert node6['effective_caps']['item_fit'] == 1620
    assert node6['effective_caps']['item_eval_reserved'] == 180
    for prior in (all18, node6):
        assert prior['effective_caps']['fit_global'] == 9720
        assert prior['effective_caps']['total'] == 10800
        assert prior['effective_caps']['eval_global_reserved'] == 1080
    node_stage = source_identity('node6_15600')
    assert node_stage['profile'] == 'node6_15600'
    assert node_stage['effective_caps'] == {'item_total': 2600., 'item_fit': 2480.,
        'item_eval_reserved': 120., 'items': 6, 'fit_global': 14880., 'total': 15600.,
        'eval_global_reserved': 720.}
    node_order = [x for x in implementation.ORDER if x[1] == 'node']
    assert node_order == [(s, 'node', g) for s in (5101, 5102, 5103)
                          for g in ('control', 'residual')]
    assert node_stage['config']['updates'] == 512
    base, h = small()
    model = CausalHistoryResidual.from_train(base, h)
    calls = []
    original = model.encode
    def counted(history):
        calls.append(history.clone())
        return original(history)
    model.encode = counted
    obs = torch.cat((h, h[:, -1:].expand(-1, 4, -1)), 1)
    prediction(model, obs, dt=.005)
    assert len(calls) == 1
    assert torch.equal(calls[0], h)


def test_edge_only_profile_order_budget_and_restore_isolation(tmp_path):
    edge = source_identity('edge6_6600')
    assert implementation.study_order('edge6_6600') == [
        (seed, 'edge', group) for seed in (5101, 5102, 5103)
        for group in ('control', 'residual')]
    assert implementation.study_order('node6_15600') == [
        (seed, 'node', group) for seed in (5101, 5102, 5103)
        for group in ('control', 'residual')]
    assert implementation.study_order('all18') == implementation.ORDER
    assert edge['cases'] == ['edge'] and edge['config']['updates'] == 512
    assert edge['effective_caps'] == {'item_total': 1100., 'item_fit': 980.,
        'item_eval_reserved': 120., 'items': 6, 'fit_global': 5880.,
        'total': 6600., 'eval_global_reserved': 720.}
    assert json.loads(json.dumps(edge)) == edge
    base, h = small()
    model = CausalHistoryResidual.from_train(base, h)
    opt = make_optimizer(model); rng = np.random.default_rng(11)
    node = source_identity('node6_15600')
    checkpoint(tmp_path, model, opt, rng, 0, [], node)
    with pytest.raises(ValueError, match='provenance'):
        restore(tmp_path, model, opt, rng, edge)
    assert (tmp_path / 'checkpoint_00000' / 'training.pt').exists()


def test_checkpoint_pointer_interruption_rejects_orphan(tmp_path, monkeypatch):
    base, h = small()
    model = CausalHistoryResidual.from_train(base, h)
    opt = make_optimizer(model)
    rng = np.random.default_rng(1)
    checkpoint(tmp_path, model, opt, rng, 0, [], {'fixed': 1})
    original = implementation.atomic_json
    def interrupt_pointer(path, value):
        if Path(path).name == 'latest.json':
            raise KeyboardInterrupt('injected after committed directory rename')
        return original(path, value)
    monkeypatch.setattr(implementation, 'atomic_json', interrupt_pointer)
    with pytest.raises(KeyboardInterrupt):
        checkpoint(tmp_path, model, opt, rng, 32, [0.] * 32, {'fixed': 1})
    assert (tmp_path / 'checkpoint_00032' / 'training.pt').exists()
    with pytest.raises(ValueError, match='uncommitted checkpoint'):
        restore(tmp_path, model, opt, rng, {'fixed': 1})


@pytest.mark.skipif(os.environ.get('CAUSAL_HISTORY_GPU_QA') != '1', reason='explicit disposable GPU timing only')
def test_gpu_feasibility_receipt():
    started = time.perf_counter()
    assert torch.cuda.is_available()
    device = torch.device('cuda:0')
    times = {}
    for case in ('node', 'edge', 'saturation'):
        source, _, observed, scales, meta = fixture(case, 5101, 0., device, started + 60)
        train = torch.tensor(observed[0], dtype=torch.float64, device=device)
        scale = torch.tensor(scales, dtype=torch.float64, device=device)
        model = CausalHistoryResidual.from_train(runtime_model(source, [truths()[case][0]], 5101, device), train[:, :11])
        opt = make_optimizer(model)
        rng = np.random.default_rng(np.random.SeedSequence([5101, 7]))
        times[case] = []
        for _ in range(3):
            synchronize(device)
            before = time.perf_counter()
            train_update(model, opt, rng, train, scale, started + 60)
            synchronize(device)
            times[case].append(time.perf_counter() - before)
        assert time.perf_counter() - started < 60
    tau = max(x for group in times.values() for x in group)
    c = 1.25 * (512 * tau + 120)
    receipt = {'kind': 'disposable implementation QA; no formal scientific endpoint',
               'pid': os.getpid(), 'children': [], 'includes_all_children_once': True,
               'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__,
               'updates': 9, 'updates_per_case': 3, 'horizon_steps': 50,
               'synchronized_update_seconds': times, 'tau_max_seconds': tau, 'C_seconds': c,
               'feasible_all18': bool(18 * c <= 10800), 'feasible_node6': bool(6 * c <= 10800),
               'benchmark_seconds': time.perf_counter() - started,
               'source_sha256': sha(ROOT / 'experiments/closure_causal_history_residual_20261019.py')}
    target = ROOT / 'private_data/closure_mechanism_20261019/P1/history_residual/implementation_qa_receipt.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, receipt)
