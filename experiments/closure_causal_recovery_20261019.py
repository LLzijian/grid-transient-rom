"""Bounded causal synthetic recovery, with no project dataset readers.

Search enumerates a declared syntax-only subset of single-term production
proposals; it is not the full multi-round discovery algorithm. Failure recovery
is an explicit whole-run restart in a new immutable attempt directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import numpy as np
import torch
from closure_discovery_benchmarks import toy_source, pool, v, mul, power
from closure_discovery_support import term_key
from closure_expression_identity import mechanism_signature as term_signature
from closure_training_runtime import TrainingClosureROM, resolve_device, synchronize, export_cpu


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def truths():
    return {
        'node': ({'channel': 'node', 'expression': mul(power(v('p'), 2), power(v('q'), 2))}, .15),
        'edge': ({'channel': 'edge', 'expression': mul({'op': 'add', 'args': [v('qi'), mul({'op': 'const', 'value': -1}, v('qj'))]}, v('s'))}, .15),
        'saturation': ({'channel': 'port', 'expression': {'op': 'tanh', 'scale': 2., 'arg': v('q')}}, .4),
    }


def proposals(per_channel=6):
    """Balanced finite syntax subset; ordering never consults the case/targets."""
    if per_channel < 1:
        raise ValueError('positive proposal count required')
    candidates = pool()
    result = []
    for channel in ('node', 'edge', 'port'):
        choices = [t for t in candidates if t['channel'] == channel]
        choices.sort(key=lambda t: hashlib.sha256(term_key(t).encode()).hexdigest())
        result.extend(choices[:per_channel])
    return result


def copy_runtime(source, terms, device):
    base = source.with_terms(terms)
    model = TrainingClosureROM.from_config(base._export_payload()['config'], terms=base.terms)
    model.load_state_dict(base.state_dict())
    trainable = {name for name, p in base.named_parameters() if p.requires_grad}
    for name, p in model.named_parameters():
        p.requires_grad_(name in trainable)
    return model.to(device)


def runtime_model(source, terms, seed, device):
    # from_source copies the registered skeleton; GRU is reset identically for
    # every candidate, independently of synthetic data generation.
    model = copy_runtime(source, terms, device)
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'gru' in name or 'head' in name:
                parameter.uniform_(-.05, .05)
        model.coefficients.zero_()
    return model


def trajectory(model, initial, steps, dt=.01, deadline=None):
    state = initial
    rows = [state[..., :30]]
    model.begin_graph_cache()
    try:
        for i in range(steps):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('trajectory stopped at process deadline')
            state = model.rk4(state, dt)
            rows.append(state[..., :30])
        return torch.stack(rows, 1)
    finally:
        model.end_graph_cache()


def fixture(case, seed, noise, device, deadline=None, count=16, train_steps=60, heldout_steps=110):
    """Return observations only. Hidden truth coordinates never leave this function."""
    term, coefficient = truths()[case]
    source = toy_source()
    truth = copy_runtime(source, [term], device)
    with torch.no_grad():
        truth.coefficients.fill_(coefficient)
    streams = np.random.SeedSequence([seed, 20261019]).spawn(4)
    arrays, integration, differences = [], [], []
    for stream, steps in zip(streams[:2], (train_steps, heldout_steps)):
        rng = np.random.default_rng(stream)
        strength = np.where(np.arange(count)[:, None] < count // 2, .08, .3)
        initial = rng.normal(size=(count, 40)) * strength
        initial[:, 30:] = rng.normal(size=(count, 10)) * np.where(np.arange(count)[:, None] < count // 2, .2, 1.2)
        z = torch.tensor(initial, dtype=torch.float64, device=device)
        with torch.no_grad():
            coarse = trajectory(truth, z, steps, deadline=deadline).cpu().numpy()
            fine = trajectory(truth, z, steps * 2, dt=.005, deadline=deadline).cpu().numpy()[:, ::2]
        if not np.isfinite(coarse).all() or not np.isfinite(fine).all():
            raise FloatingPointError('synthetic truth integration nonfinite; no event removed')
        arrays.append(coarse)
        differences.append(coarse - fine)
        integration.append(float(np.max(np.abs(coarse - fine))))
    scales = np.maximum(arrays[0].std(axis=(0, 1)), 1e-6)
    noisy = [a + noise * scales * np.random.default_rng(s).normal(size=a.shape)
             for a, s in zip(arrays, streams[2:])]
    meta = {'seed': seed, 'noise': noise, 'case': case, 'truth_term': term,
            'truth_coefficient': coefficient, 'train_count': count, 'heldout_count': count,
            'dt': .01, 'history_samples': 11, 'train_future_seconds': (train_steps - 10) * .01,
            'heldout_future_seconds': (heldout_steps - 10) * .01,
            'dual_step_observation_max_abs': integration, 'scales': scales.tolist(),
            'dual_step_normalized_max_abs': [float(np.max(np.abs(d) / scales)) for d in differences],
            'data_streams': [s.spawn_key for s in streams],
            'observed_coordinates': 30, 'q_supervision': False, 'rhs_supervision': False,
            'clean_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays]}
    return source, arrays, noisy, scales, meta


def predict(model, observations, steps, deadline=None):
    history = observations[:, :11]
    initial = torch.cat((history[:, -1], model.encode(history)), -1)
    return trajectory(model, initial, steps, deadline=deadline)[:, 1:]


def trajectory_loss(model, observed, scales, deadline=None):
    prediction = predict(model, observed, observed.shape[1] - 11, deadline)
    return ((prediction - observed[:, 11:]) / scales).square().mean()


def optimizer(model):
    groups = {'encoder': [], 'closure': [], 'skeleton': []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            group = 'encoder' if ('gru' in name or 'head' in name) else 'closure' if name in ('coefficients', '_template_scalars.3', '_template_scalars.4') else 'skeleton'
            groups[group].append(p)
    rates = {'encoder': 3e-4, 'closure': 1e-3, 'skeleton': 1e-2}
    return torch.optim.Adam([{'params': ps, 'lr': rates[name]} for name, ps in groups.items() if ps])


def save_checkpoint(folder, model, opt, rng, step, trace):
    pending = folder / ('pending_' + uuid.uuid4().hex)
    pending.mkdir()
    export_cpu(model, pending / 'model.json')
    torch.save({'optimizer': opt.state_dict(), 'fit_numpy_rng': rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if model.mass.is_cuda else [],
                'completed_updates': step, 'trace': trace}, pending / 'training.pt')
    target = folder / f'checkpoint_{step:05d}'
    os.replace(pending, target)
    atomic_json(folder / 'latest.json', {'checkpoint': target.name, 'updates': step,
                'recovery': 'explicit whole-run restart only; retain checkpoints and failed attempts'})


def fit(source, terms, seed, device, observed, scales, updates, folder, deadline):
    folder.mkdir()
    model = runtime_model(source, terms, seed, device)
    opt = optimizer(model)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    trace = []
    try:
        for step in range(1, updates + 1):
            if time.perf_counter() >= deadline:
                raise TimeoutError('fit incomplete; no partial endpoint scored')
            indices = torch.as_tensor(rng.permutation(len(observed))[:12], device=device)
            opt.zero_grad(set_to_none=True)
            loss = trajectory_loss(model, observed[indices], scales, deadline)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite trajectory loss')
            loss.backward()
            if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('nonfinite trajectory gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            synchronize(device)
            trace.append(float(loss.detach()))
            if step % 32 == 0 or step == updates:
                save_checkpoint(folder, model, opt, rng, step, trace)
        with torch.no_grad():
            score = float(trajectory_loss(model, observed, scales, deadline))
        row = {'terms': model.terms, 'updates': updates, 'train_normalized_mse': score,
               'trace': trace, 'checkpoint': f'checkpoint_{updates:05d}', 'status': 'complete'}
        atomic_json(folder / 'result.json', row)
        return model, row
    except BaseException as error:
        atomic_json(folder / 'failure.json', {'error': repr(error), 'completed_updates': len(trace),
                    'traceback': traceback.format_exc(), 'recovery': 'restart whole run explicitly; this directory is retained'})
        raise


def run(args):
    started = time.perf_counter()
    deadline = started + args.max_seconds
    device = resolve_device(args.device)
    torch.set_num_threads(1)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.restart:
        raise ValueError('nonempty output requires --restart; all old attempts retained')
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / ('attempt_' + uuid.uuid4().hex)
    attempt.mkdir()
    files = [Path(__file__), ROOT/'experiments/closure_discovery_benchmarks.py',
             ROOT/'experiments/closure_discovery_support.py', ROOT/'experiments/closure_expression_identity.py',
             ROOT/'experiments/closure_training_runtime.py', ROOT/'experiments/symbolic_closure_rom.py']
    identity = {'config': {k: v for k, v in vars(args).items() if k != 'restart'},
                'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'device': str(device), 'torch': torch.__version__, 'dtype': 'float64',
                'scope': 'synthetic only; bounded balanced single-term proposal enumeration, not full structure search'}
    atomic_json(attempt/'identity.json', identity)
    try:
        source, clean, noisy, scales, metadata = fixture(args.case, args.seed, args.noise, device, deadline)
        candidates = proposals(args.per_channel)
        metadata['proposal_terms'] = candidates
        metadata['truth_in_bounded_pool'] = any(term_signature(t) == term_signature(truths()[args.case][0]) for t in candidates)
        metadata['recovery_scope'] = ('syntax permits this truth; fit adequacy still untested' if metadata['truth_in_bounded_pool']
                                      else 'truth absent: this bounded subset cannot establish formula recovery')
        atomic_json(attempt/'fixture.json', metadata)
        if max(metadata['dual_step_normalized_max_abs']) > 1e-3:
            raise FloatingPointError('synthetic truth dual-step qualification failed; no fitting allowed')
        np.savez_compressed(attempt/'observations.npz', train_clean=clean[0], heldout_clean=clean[1], train_observed=noisy[0], heldout_observed=noisy[1], scales=scales)
        observed = torch.tensor(noisy[0], dtype=torch.float64, device=device)
        scale = torch.tensor(scales, dtype=torch.float64, device=device)
        updates = 1 if args.command == 'pilot' else args.updates
        term = truths()[args.case][0]
        jobs = [('known_support', [term])]
        if args.command == 'run' and not args.known_support_only:
            jobs += [(f'proposal_{i:03d}', [t]) for i, t in enumerate(candidates)]
        rows, models = [], {}
        for name, terms in jobs:
            before = time.perf_counter()
            model, row = fit(source, terms, args.seed, device, observed, scale, updates, attempt/name, deadline)
            row.update(name=name, elapsed_seconds=time.perf_counter()-before,
                       same_coordinate_term_signature=term_signature(terms[0]) == term_signature(term))
            rows.append(row)
            models[name] = model.cpu()
        selected = min((r for r in rows if r['name'] != 'known_support'), key=lambda r: r['train_normalized_mse'], default=None)
        evaluation = []
        for name in ['known_support'] + ([selected['name']] if selected else []):
            model = models[name].to(device)
            heldout = torch.tensor(noisy[1], dtype=torch.float64, device=device)
            with torch.no_grad():
                prediction = predict(model, heldout, 100, deadline).cpu().numpy()
            clean_errors = np.mean(((prediction-clean[1][:,11:])/scales)**2, axis=(1,2))
            noisy_errors = np.mean(((prediction-noisy[1][:,11:])/scales)**2, axis=(1,2))
            evaluation.append({'name': name, 'heldout_clean_normalized_mse': float(clean_errors.mean()),
                               'heldout_observed_normalized_mse': float(noisy_errors.mean()),
                               'heldout_clean_event_mse': clean_errors.tolist(),
                               'heldout_observed_event_mse': noisy_errors.tolist()})
        result = {'status': 'complete', 'command': args.command, 'fits': rows,
                  'selection': selected, 'heldout': evaluation, 'elapsed_seconds': time.perf_counter()-started,
                  'causal_interface': 'one GRU initialization from 11 samples; autonomous thereafter',
                  'interpretation': 'same-coordinate signatures only; no unique hidden mechanism claim',
                  'identity_sha256': digest(identity)}
        atomic_json(attempt/'result.json', result)
        atomic_json(output/'latest.json', {'attempt': attempt.name, 'status': 'complete'})
        return result
    except BaseException as error:
        atomic_json(attempt/'failure.json', {'status': 'unresolved', 'error': repr(error), 'elapsed_seconds': time.perf_counter()-started, 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['pilot', 'run'])
    parser.add_argument('--case', choices=list(truths()), required=True)
    parser.add_argument('--seed', type=int, choices=[5101, 5102, 5103], required=True)
    parser.add_argument('--noise', type=float, choices=[0., .01], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--updates', type=int, default=128)
    parser.add_argument('--per-channel', type=int, default=6)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--known-support-only', action='store_true', help='fit the registered control only; no proposal scoring')
    args = parser.parse_args()
    if not np.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.updates < 1 or args.per_channel < 1:
        parser.error('positive caps/counts required')
    print(json.dumps(run(args), allow_nan=False))


if __name__ == '__main__':
    main()
