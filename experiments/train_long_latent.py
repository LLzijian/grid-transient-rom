"""Loop0060 Xi-only long-horizon coefficient challenger.

This module deliberately reuses the Loop0058 JointROM and its saved
normalization/state.  Only the existing masked Xi coefficients are trainable.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from train_joint_latent import (DT, NETWORK, BASE, JointROM, read_development,
                                 tensor)
from kiki.models.private_pm import MechanicalROM


def _copy_state(model, state):
    """Load a 0058 state and verify the architecture is exactly compatible."""
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name]), name


def freeze_xi_only(model):
    """Freeze every parameter and buffer except the fitted Xi tensor."""
    before = {name: value.detach().clone() for name, value in model.state_dict().items()
              if name != 'xi'}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == 'xi')
    return before


def assert_non_xi_unchanged(model, before):
    for name, value in model.state_dict().items():
        if name != 'xi':
            assert torch.equal(value, before[name]), name


def assert_inactive_xi_unchanged(model, source):
    inactive = model.support == 0
    assert torch.equal(model.xi.detach()[inactive], source['xi'][inactive]), 'inactive Xi'


def select_training_events(events, exclude_train_group=None):
    """Optionally remove one declared TRAIN group from the optimization pool only."""
    original = list(events)
    if exclude_train_group is None:
        return original, []
    group = str(exclude_train_group)
    excluded = [event for event in original if str(event['group']) == group]
    if not excluded:
        raise ValueError(f'unknown TRAIN group: {group}')
    fit = [event for event in original if str(event['group']) != group]
    if not fit:
        raise ValueError('TRAIN group exclusion leaves an empty optimization pool')
    return fit, excluded


def _group_counts(events):
    counts = {}
    for event in events:
        key = str(event['group'])
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_failure_evidence(output, args, step, reason, endpoints, event_ids, losses, provenance, model):
    """Persist the mandatory numerical-failure trace without changing samples."""
    coverage = {'count': len(endpoints)}
    if endpoints:
        coverage.update({'min': min(endpoints), 'max': max(endpoints),
                         'count_gt_700': sum(i > 700 for i in endpoints),
                         'count_gt_810': sum(i > 810 for i in endpoints)})
    payload = {'start_model': str(args.start_model), 'source_model_sha256': hashlib.sha256(
        (args.start_model / 'model.pt').read_bytes()).hexdigest(), 'code_sha256': hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest(), 'horizon_s': args.horizon, 'step': step,
        'reason': reason, 'sampled_endpoint_coverage': coverage,
        'sampled_event_ids': event_ids, 'losses': losses, 'training_pool': provenance,
        'source_hashes_after_failure': {'model': hashlib.sha256((args.start_model / 'model.pt').read_bytes()).hexdigest(),
                                        'normalization': hashlib.sha256((args.start_model / 'normalization.npz').read_bytes()).hexdigest()},
        'test_read': False}
    (output / 'failure_evidence.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    torch.save(model.state_dict(), output / 'failure_model.pt')


def sample_batch(events, rng, count=32):
    """Sample genuine prefixes and endpoints across each event's full TRAIN."""
    samples = []
    for _ in range(count):
        event = events[int(rng.integers(len(events)))]
        # endpoint is the final history index; one future point must exist.
        endpoint = int(rng.integers(10, len(event['x']) - 1))
        samples.append((event, endpoint))
    histories = tensor(np.stack([e['x'][i-10:i+1] for e, i in samples]))
    return samples, histories


def _error(pred, truth, model):
    err = pred[..., :20] - tensor(truth[..., :20])
    delta = model.coi(err[..., :10]) / model.metric_scale[0]
    omega = model.coi(err[..., 10:20]) / model.metric_scale[1]
    return .5 * (delta.square().mean(dim=-1) + omega.square().mean(dim=-1))


def campaign_loss(model, samples, histories, horizon_s):
    """Return preregistered total and component losses without padded targets."""
    # Local endpoint states are sampled prefixes; the long forecast always
    # starts from the genuine shared initial 100-ms history.
    local_states = torch.cat((histories[:, -1], model.encode(histories)), dim=-1)
    initial = tensor(np.stack([e['x'][:11] for e, _ in samples]))
    states = torch.cat((initial[:, -1], model.encode(initial)), dim=-1)
    count = len(samples)
    device = local_states.device
    local_available = np.array([min(20, len(e['x']) - 1 - endpoint)
                                for e, endpoint in samples], dtype=int)
    if np.any(local_available <= 0):
        raise ValueError('sampled endpoint has no future point')
    local_sum = torch.zeros(count, dtype=torch.float64, device=device)
    local_count = torch.zeros(count, dtype=torch.float64, device=device)
    current = local_states
    # Only active trajectories are advanced. Truth is materialized solely for
    # active observations, never as a padded batch.
    for offset in range(1, int(local_available.max()) + 1):
        active_np = np.flatnonzero(local_available >= offset)
        active = torch.as_tensor(active_np, dtype=torch.long, device=device)
        current = current.index_copy(0, active, model.rk4(current.index_select(0, active), min(DT, .02)))
        truth = np.stack([samples[row][0]['x'][samples[row][1] + offset] for row in active_np])
        values = _error(current.index_select(0, active)[:, :20], truth, model)
        local_sum = local_sum.index_add(0, active, values)
        local_count = local_count.index_add(0, active, torch.ones_like(values))
    local_mean = (local_sum / local_count).mean()

    adjacent = tensor(np.stack([e['x'][endpoint-9:endpoint+2] for e, endpoint in samples]))
    dyn_mean = ((model.rk4(local_states, DT)[:, 30:] - model.encode(adjacent)) / .1).square().mean()

    # Long forecasts are deliberately independent of local endpoints: every
    # event starts from its genuine 0--100 ms history and targets index 10+k.
    max_steps = max(1, round(float(horizon_s) / DT))
    long_available = np.array([min(max_steps, len(e['x']) - 11) for e, _ in samples], dtype=int)
    if np.any(long_available <= 0):
        raise ValueError('event has no future after the initial 100-ms history')
    long_sum = torch.zeros(count, dtype=torch.float64, device=device)
    long_count = torch.zeros(count, dtype=torch.float64, device=device)
    current = states
    for offset in range(1, int(long_available.max()) + 1):
        active_np = np.flatnonzero(long_available >= offset)
        active = torch.as_tensor(active_np, dtype=torch.long, device=device)
        current = current.index_copy(0, active, model.rk4(current.index_select(0, active), min(DT, .02)))
        observed_np = active_np[(offset == 1) | (offset == 2) | (offset % 2 == 0) |
                                (long_available[active_np] == offset)]
        if len(observed_np):
            observed = torch.as_tensor(observed_np, dtype=torch.long, device=device)
            truth = np.stack([samples[row][0]['x'][10 + offset] for row in observed_np])
            values = _error(current.index_select(0, observed)[:, :20], truth, model)
            long_sum = long_sum.index_add(0, observed, values)
            long_count = long_count.index_add(0, observed, torch.ones_like(values))
    event_loss = long_sum / long_count
    mean_long = event_loss.mean()
    tail = event_loss.topk(max(1, int(np.ceil(.1 * len(event_loss))))).values.mean()
    risk = .8 * mean_long + .2 * tail
    xi_l1 = (model.xi * model.support).abs().mean()
    total = 10 * risk + local_mean + dyn_mean + 1e-4 * xi_l1
    # Frozen electrical diagnostics use real local truth.  They are detached
    # because their parameters and mappings are outside this Xi-only fit.
    with torch.no_grad():
        q = local_states[:, 30:]
        force_pred = model.memory(q + model.algebraic_coordinate(local_states))
        force_truth = tensor(np.stack([e['force'][endpoint] for e, endpoint in samples]))
        zpred, load_pred = model.electrical_output(local_states[:, :20], q)
        ztruth = tensor(np.stack([e['z'][endpoint] for e, endpoint in samples])) / model.z_scale
        load_truth = tensor(np.array([e['load'][endpoint] for e, endpoint in samples]))[:, None] / model.load_scale
        electrical = {'force_frozen': ((force_pred-force_truth)/model.force_scale).square().mean(),
                      'pe_frozen': (zpred[:, :10]-ztruth[:, :10]).square().mean(),
                      'voltage_frozen': (zpred[:, 10:]-ztruth[:, 10:]).square().mean(),
                      'load_frozen': (load_pred-load_truth).square().mean()}
    return total, {'risk_long': risk, 'long_mean': mean_long, 'long_tail': tail,
                   'local_mean': local_mean, 'dyn': dyn_mean, 'xi_l1': xi_l1,
                   **electrical}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizon', type=float, choices=(2, 5), default=2)
    parser.add_argument('--start-model', type=Path, default=Path('private_data/task2_joint_0058'))
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--exclude-train-group', type=str, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(60)
    rng = np.random.default_rng(60)
    base, network = MechanicalROM.load(BASE), dict(np.load(NETWORK))
    network['observed_buses'] = np.unique(np.r_[network['load_buses'], network['generator_buses']])
    network['load_positions'] = np.searchsorted(network['observed_buses'], network['load_buses'])
    events, excluded, identities = read_development(base, network, full_train=True)
    fit_events, excluded_fit_events = select_training_events(events['train'], args.exclude_train_group)
    provenance = {'excluded_train_group': args.exclude_train_group,
                  'original_train_event_ids': [event['id'] for event in events['train']],
                  'fit_train_event_ids': [event['id'] for event in fit_events],
                  'excluded_train_event_ids': [event['id'] for event in excluded_fit_events],
                  'group_counts_original': _group_counts(events['train']),
                  'group_counts_fit': _group_counts(fit_events),
                  'group_counts_excluded': _group_counts(excluded_fit_events)}
    source_hashes_before = {'model': hashlib.sha256((args.start_model / 'model.pt').read_bytes()).hexdigest(),
                            'normalization': hashlib.sha256((args.start_model / 'normalization.npz').read_bytes()).hexdigest()}
    provenance['source_hashes_before'] = source_hashes_before
    norm = np.load(args.start_model / 'normalization.npz')
    # 0058 normalization is immutable; do not recompute or call conditioning.
    electrical = (norm['z_scale'], norm['voltage_star'], norm['load_positions'],
                  network['load_g'], network['equilibrium_components'][0])
    ports = network['port_incidence']
    port_ids = norm['port_ids'].astype(int)
    model = JointROM(base, ports[:, port_ids], norm['input_scale'], norm['feature_scale'],
                     float(norm['force_scale']), float(norm['load_scale']),
                  electrical=electrical, latent_dim=3)
    source = torch.load(args.start_model / 'model.pt', map_location='cpu', weights_only=True)
    # Confirm immutable physical construction agrees with the checkpoint before
    # loading learned observer coordinates and Xi.
    physical_buffers = ('mass', 'incidence', 'delta_star', 'coupling', 'damping',
                        'rate', 'gain', 'angle_incidence', 'ports', 'input_scale',
                        'feature_scale', 'force_scale', 'load_scale', 'metric_scale',
                        'z_scale', 'voltage_star', 'load_positions', 'load_g', 'load_star')
    for name in physical_buffers:
        assert name in source and torch.equal(model.state_dict()[name], source[name]), name
    _copy_state(model, source)
    before = freeze_xi_only(model)
    optimizer = torch.optim.Adam([model.xi], lr=.0003)
    steps = 1 if args.smoke else args.steps
    losses = []
    sampled_endpoints = []
    sampled_event_ids = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        samples, histories = sample_batch(fit_events, rng)
        sampled_endpoints.extend(i for _, i in samples)
        sampled_event_ids.extend(e['id'] for e, _ in samples)
        loss, terms = campaign_loss(model, samples, histories, args.horizon)
        if not torch.isfinite(loss):
            write_failure_evidence(args.output, args, step + 1, 'nonfinite loss', sampled_endpoints, sampled_event_ids, losses, provenance, model)
            raise FloatingPointError(f'nonfinite loss at step {step + 1}')
        loss.backward()
        if model.xi.grad is None or not torch.isfinite(model.xi.grad).all():
            write_failure_evidence(args.output, args, step + 1, 'nonfinite Xi gradient', sampled_endpoints, sampled_event_ids, losses, provenance, model)
            raise FloatingPointError(f'nonfinite gradient at step {step + 1}')
        grad_norm = float(torch.linalg.vector_norm(model.xi.grad))
        optimizer.step()
        if not torch.isfinite(model.xi).all():
            write_failure_evidence(args.output, args, step + 1, 'nonfinite Xi parameter', sampled_endpoints, sampled_event_ids, losses, provenance, model)
            raise FloatingPointError(f'nonfinite Xi at step {step + 1}')
        if step % 50 == 0 or step + 1 == steps:
            row = {'step': step + 1, 'loss': float(loss.detach()),
                   'xi_grad_norm': grad_norm, **{k: float(v.detach()) for k, v in terms.items()}}
            losses.append(row)
            print(json.dumps(row), flush=True)
    assert_non_xi_unchanged(model, before)
    assert_inactive_xi_unchanged(model, source)
    source_hashes_after = {'model': hashlib.sha256((args.start_model / 'model.pt').read_bytes()).hexdigest(),
                           'normalization': hashlib.sha256((args.start_model / 'normalization.npz').read_bytes()).hexdigest()}
    assert source_hashes_after == source_hashes_before, 'source artifacts changed during fit'
    torch.save(model.state_dict(), args.output / 'model.pt')
    shutil.copyfile(args.start_model / 'normalization.npz', args.output / 'normalization.npz')
    source_model = args.start_model / 'model.pt'
    sampled_lengths = {e['id']: len(e['x']) for e, _ in samples}
    evidence = {'start_model': str(args.start_model), 'seed': 60, 'horizon_s': args.horizon,
                'steps': steps, 'full_train': True, 'excluded': excluded, 'training_pool': provenance,
                'identities': identities, 'train_lengths': {e['id']: len(e['x']) for e in events['train']},
                'validation_lengths': {e['id']: len(e['x']) for e in events['validation']},
                'sampled_event_lengths': sampled_lengths,
                'sampled_endpoint_coverage': {'min': min(sampled_endpoints), 'max': max(sampled_endpoints),
                                               'count': len(sampled_endpoints),
                                               'count_gt_700': sum(i > 700 for i in sampled_endpoints),
                                               'count_gt_810': sum(i > 810 for i in sampled_endpoints)},
                'sampled_event_ids': sampled_event_ids,
                'losses': losses, 'nonfinite': False, 'frozen_equal': True, 'inactive_xi_equal': True,
                'source_hashes_before': source_hashes_before, 'source_hashes_after': source_hashes_after,
                'source_model_sha256': hashlib.sha256(source_model.read_bytes()).hexdigest(),
                'normalization_sha256': hashlib.sha256((args.output / 'normalization.npz').read_bytes()).hexdigest(),
                'model_sha256': hashlib.sha256((args.output / 'model.pt').read_bytes()).hexdigest(),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'seconds': time.perf_counter() - started, 'test_read': False}
    (args.output / 'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding='utf-8')
    if args.smoke:
        print('SMOKE ONLY: one Xi-only step; no scientific evaluation', flush=True)


if __name__ == '__main__':
    main()
