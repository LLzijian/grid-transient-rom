"""TRAIN-only predictive-anchor initialization, followed by required joint ODE fit.

This command never optimizes the PowerGraph parameters. Its checkpoint is for
``scripts/train.py --warm-start`` only; it is not autonomous prediction evidence.
All real execution must be inside the project's accounted managed runner.
"""
import _bootstrap  # noqa: F401; resolves TASK4 independently of the caller's cwd

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from task4.data import load_events, train_scales
from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder, PredictiveHead
from task4.objectives import AnchorTargets, ShootingBank
from task4.paths import ROOT, project_path, read_json, write_json
from task4.training import check_authorized, checkpoint


STAGE = 'TRAIN-only GRU/head predictive-anchor pretraining; dynamics not optimized'


def window_indices(length, windows=8):
    """Fixed integer linspace, including the initial and last real-prefix window."""
    if windows < 1 or length < 12:
        raise ValueError('Positive windows and at least 12 real samples are required')
    return np.unique(np.linspace(10, length-2, num=windows, dtype=np.int64))


def check_deadline(deadline, estimated_seconds=0.0):
    if time.time() + max(10.0, 1.5*estimated_seconds) >= deadline:
        raise TimeoutError('Managed save deadline reached')


def _save_cache(output, arrays, manifest):
    """Atomic local cache; incomplete caches are explicitly marked and not trained."""
    temporary = output/'anchor_cache.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(output/'anchor_cache.npz')
    write_json(output/'anchor_cache_manifest.json', manifest)


def build_anchor_cache(events, physics, horizons, windows, output, deadline):
    """Build only from the already-authorized explicit TRAIN event list.

    Availability masks are per window/tau and are never model features. The
    existing AnchorTargets implementation supplies real residuals from M3.
    """
    started = time.monotonic()
    target_builder = AnchorTargets(physics, horizons, dt=0.01, max_cache=1)
    history, target, mask, event_index, sample_index = [], [], [], [], []
    manifest = {
        'format': 1, 'stage': STAGE, 'split': 'train', 'test_payload_read': False,
        'complete': False, 'history_samples': 11, 'history_ms': 100,
        'horizons_seconds': list(horizons), 'windows_requested_per_event': windows,
        'window_rule': 'unique integer linspace(10, len(event.y)-2, windows)',
        'target_definition': 'real y(index+tau) minus frozen M3 rollout from real y(index)',
        'mask_is_model_input': False, 'target_scaling': 'raw physical residual; divide by full TRAIN scales in loss',
        'all_train_source_ids': [event.id for event in events],
        'events': [{'id': e.id, 'group': e.group, 'label': e.label, 'source': e.source,
                    'real_samples': len(e.y), 'indices': window_indices(len(e.y), windows).tolist()}
                   for e in events],
    }
    failure = None
    previous_cost = 0.0
    try:
        for ei, event in enumerate(events):
            for index in window_indices(len(event.y), windows):
                check_deadline(deadline, previous_cost)
                tick = time.monotonic()
                truth, available = target_builder.get(event, int(index))
                h = np.array(event.y[index-10:index+1], dtype=np.float64, copy=True)
                if not np.isfinite(h).all() or not np.isfinite(truth).all():
                    raise FloatingPointError(f'Nonfinite real history/anchor: {event.id} at {index}')
                history.append(h)
                target.append(truth)
                mask.append(available)
                event_index.append(ei)
                sample_index.append(int(index))
                previous_cost = time.monotonic()-tick
        manifest['complete'] = True
    except BaseException as exc:
        failure = exc
        manifest['stop_reason'] = type(exc).__name__ + ': ' + str(exc)
    count, n, taus = len(history), physics.n, len(horizons)
    arrays = {
        'history': np.asarray(history, dtype=np.float64).reshape(count, 11, n, 3),
        'target': np.asarray(target, dtype=np.float64).reshape(count, n, taus, 3),
        'mask': np.asarray(mask, dtype=np.bool_).reshape(count, taus),
        'event_index': np.asarray(event_index, dtype=np.int64),
        'sample_index': np.asarray(sample_index, dtype=np.int64),
        'event_ids': np.asarray([event.id for event in events], dtype=np.str_),
        'horizons': np.asarray(horizons, dtype=np.float64),
    }
    manifest.update(windows_cached=count, available_count_per_tau=arrays['mask'].sum(0).tolist(),
                    windows_with_any_available_tau=int(arrays['mask'].any(1).sum()))
    manifest['events_without_available_anchor'] = [
        event.id for ei, event in enumerate(events)
        if not arrays['mask'][arrays['event_index'] == ei].any()
    ]
    # Persistence is part of cache wall time, and all of it is within the same
    # externally accounted process/deadline as the later optimization phase.
    _save_cache(output, arrays, manifest)
    manifest['cache_wall_seconds'] = time.monotonic()-started
    write_json(output/'anchor_cache_manifest.json', manifest)
    if failure is not None:
        raise failure
    return arrays, manifest


def masked_anchor_loss(prediction, raw_target, available, scales):
    """Average each tau over its available windows, then average legal taus."""
    if prediction.shape != raw_target.shape or prediction.ndim != 4:
        raise ValueError('Expected matching [batch,N,tau,3] prediction and target')
    if available.shape != (prediction.shape[0], prediction.shape[2]) or available.dtype != torch.bool:
        raise ValueError('Expected boolean [batch,tau] availability mask')
    scaled_target = raw_target/scales
    # Clear absent positions before subtraction so missing targets cannot affect
    # values or gradients; they remain absent and are never fed into the GRU.
    legal = available[:, None, :, None]
    difference = torch.where(legal, prediction, 0)-torch.where(legal, scaled_target, 0)
    per_window_tau = difference.square().mean(dim=(1, 3))
    counts = available.sum(0)
    per_tau = per_window_tau.sum(0)/counts.clamp_min(1)
    used = counts > 0
    loss = per_tau[used].mean() if used.any() else prediction.sum()*0
    return loss, per_tau, counts


def pretrain(output, *, steps=500, windows=8, batch_size=64, lr=3e-4,
             seed=401, device='auto', save_every=25):
    # This gate precedes directory creation, configuration reads, and especially
    # any call that can open real event payloads. There is no synthetic bypass.
    check_authorized(False)
    started = time.monotonic()
    if steps < 1 or windows < 1 or batch_size < 1 or save_every < 1:
        raise ValueError('steps/windows/batch/save_every must be positive')
    if lr not in (3e-4, 1e-4):
        raise ValueError('Registered anchor learning rates are 3e-4 and 1e-4')
    deadline = float(os.environ.get('TASK4_SAVE_DEADLINE_UNIX', 'nan'))
    if not np.isfinite(deadline):
        raise PermissionError('Anchor pretraining requires a finite managed save deadline')
    check_deadline(deadline)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    project_config = read_json(ROOT/'configs/project.json')
    cfg = read_json(ROOT/'configs/experiment.json')
    if project_config['history_ms'] != 100 or project_config['sample_dt'] != 0.01:
        raise ValueError('Required observation contract is 100 ms with 11 samples')
    horizons = list(cfg['anchor_horizons'])
    if (not horizons or any(not np.isfinite(h) or h <= 0 or abs(h/0.01-round(h/0.01)) > 1e-8
                            for h in horizons)):
        raise ValueError('Anchor horizons must be positive multiples of the sample interval')
    cfg.update(name='TRAIN_anchor_pretrain', arm='P', latent_per_machine=2, hidden=32,
               dtype='float64', synthetic=False, steps=steps, batch_size=batch_size,
               lr=lr, seed=seed, device=device, anchor_horizons=horizons,
               anchor_windows_per_event=windows, stage=STAGE, dynamics_optimized=False,
               required_followup='Complete joint autonomous training using --warm-start',
               warm_start_only=True, save_every=save_every)
    write_json(output/'run_config.json', cfg)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['resolved_device'] = device

    # load_events indexes only splits.json TRAIN IDs and checks all protected
    # split/data/unit boundaries before exposing the source events.
    check_deadline(deadline)
    data_started = time.monotonic()
    events, physics, facts = load_events(project_config, 'train')
    if not events:
        raise ValueError('No eligible TRAIN events')
    cfg['_event_ids'] = [event.id for event in events]
    if len(set(cfg['_event_ids'])) != len(events):
        raise ValueError('Duplicate TRAIN source IDs')
    scales = train_scales(events, physics)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise FloatingPointError('Invalid full-TRAIN scales')
    facts.update(all_train_source_ids=cfg['_event_ids'], scale_source='all eligible TRAIN events',
                 scales=scales.tolist(), stage=STAGE, test_payload_read=False,
                 data_and_scale_wall_seconds=time.monotonic()-data_started)
    write_json(output/'data_facts.json', facts)
    write_json(output/'run_config.json', cfg)

    model = PowerGraph(physics, scales, rank=2, hidden=32,
                       strength=cfg.get('model_strength', 0.01)).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    encoder = HistoryEncoder(model, hidden=32).to(device)
    head = PredictiveHead(model, horizons, hidden=32).to(device)
    parameters = list(encoder.parameters())+list(head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    bank = ShootingBank().to(device)
    rng = np.random.default_rng(seed)
    records = []
    last, safe_to_save, failure = 0, True, None
    seen = set()
    cache_info = None
    stop_reason = 'steps_completed'

    def save(name='last.pt'):
        # Shared checkpoint format loads directly for joint-fit warm_start.
        # The optimizer contains only GRU/head and cannot exact-resume joint fit.
        checkpoint(output/name, model, encoder, head, bank, optimizer, cfg,
                   physics, last, rng, records, boundary_complete=False)

    # Always retain a valid pre-cache starting point if cache construction times
    # out. Future timeout checkpoints contain only complete optimizer updates.
    save()
    train_started = None
    try:
        cache, cache_info = build_anchor_cache(events, physics, horizons, windows, output, deadline)
        usable = np.flatnonzero(cache['mask'].any(1))
        if not len(usable):
            raise ValueError('No TRAIN window has an available anchor horizon')
        check_deadline(deadline)
        train_started = time.monotonic()
        permutation, cursor, previous_cost = rng.permutation(usable), 0, 0.0
        with (output/'steps.jsonl').open('x', encoding='utf-8') as log:
            for step in range(1, steps+1):
                check_deadline(deadline, previous_cost)
                tick = time.monotonic()
                pieces, needed = [], batch_size
                while needed:
                    take = min(needed, len(permutation)-cursor)
                    pieces.append(permutation[cursor:cursor+take])
                    cursor += take
                    needed -= take
                    if cursor == len(permutation):
                        permutation, cursor = rng.permutation(usable), 0
                indices = np.concatenate(pieces)
                histories = torch.as_tensor(cache['history'][indices], dtype=torch.float64, device=device)
                targets = torch.as_tensor(cache['target'][indices], dtype=torch.float64, device=device)
                masks = torch.as_tensor(cache['mask'][indices], dtype=torch.bool, device=device)
                optimizer.zero_grad(set_to_none=True)
                # Only q, neighboring q, and mu reach the predictive head.
                q = encoder(histories)
                loss, per_tau, counts = masked_anchor_loss(head(q), targets, masks, model.scales)
                if not torch.isfinite(loss) or not torch.isfinite(q).all():
                    raise FloatingPointError('Nonfinite anchor loss or latent state')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.get('grad_clip', 100.0),
                                                          error_if_nonfinite=True)
                # An interrupted/failed optimizer update must not overwrite the
                # most recent complete-update last.pt.
                safe_to_save = False
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError('Nonfinite GRU/head parameter after update')
                safe_to_save = True
                last = step
                ids = [events[int(i)].id for i in cache['event_index'][indices]]
                seen.update(ids)
                previous_cost = time.monotonic()-tick
                row = {'step': step, 'loss': float(loss.detach()),
                       'loss_per_tau': [float(v) if int(c) else None
                                        for v, c in zip(per_tau.detach().cpu(), counts.cpu())],
                       'available_count_per_tau': counts.cpu().tolist(),
                       'q_rms': float(q.detach().square().mean().sqrt()),
                       'q_max_abs': float(q.detach().abs().max()), 'gradient_norm': float(grad_norm),
                       'ids': ids, 'window_indices': cache['sample_index'][indices].tolist(),
                       'step_seconds': previous_cost, 'elapsed_seconds': time.monotonic()-started}
                records.append(row)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                log.flush()
                if step % save_every == 0:
                    save()
                if step % 50 == 0 or step == steps:
                    print(f'anchor_step={step} loss={row["loss"]:.6g} q_rms={row["q_rms"]:.6g}', flush=True)
    except TimeoutError:
        stop_reason = 'save_deadline'
    except BaseException as exc:
        failure = exc
        stop_reason = type(exc).__name__ + ': ' + str(exc)
    finally:
        if safe_to_save:
            save()
        else:
            save('interrupted.pt')

    # Read only our own manifest after a partial-cache timeout/failure.
    if cache_info is None and (output/'anchor_cache_manifest.json').exists():
        cache_info = read_json(output/'anchor_cache_manifest.json')
    report = {
        'stage': STAGE, 'synthetic': False, 'test_payload_read': False,
        'steps_completed': last, 'steps_requested': steps, 'stop_reason': stop_reason,
        'cache_complete': bool(cache_info and cache_info['complete']),
        'cache_wall_seconds': None if cache_info is None else cache_info.get('cache_wall_seconds'),
        'cache_windows': None if cache_info is None else cache_info['windows_cached'],
        'train_source_ids': cfg['_event_ids'], 'optimized_source_ids': sorted(seen),
        'train_events': len(events), 'optimized_events': len(seen),
        'scales': scales.tolist(), 'mask_is_model_input': False,
        'dynamics_optimized': False, 'autonomous_performance_evaluated': False,
        'warm_start_only': True, 'required_followup': cfg['required_followup'],
        'training_wall_seconds': 0 if train_started is None else time.monotonic()-train_started,
        'wall_seconds_including_data_cache_and_saving': time.monotonic()-started,
        'acceptance': 'Requires complete joint autonomous training and independent evaluation',
    }
    write_json(output/'pretraining_summary.json', report)
    if failure is not None:
        raise failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--batch', '--batch-size', dest='batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, choices=[3e-4, 1e-4], default=3e-4)
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--save-every', type=int, default=25)
    args = vars(parser.parse_args())
    print(json.dumps(pretrain(**args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
