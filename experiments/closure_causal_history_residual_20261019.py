"""Fixed paired causal-history study. Synthetic observations only; no project readers.

All 18 fit endpoints are frozen before any heldout model metric is computed.
This entry point never changes an existing model, historical result, or default.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

PROCESS_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import numpy as np
import torch
from torch import nn
from closure_causal_recovery_20261019 import (
    fixture, truths, runtime_model, optimizer, trajectory, atomic_json,
    resolve_device, synchronize,
)
from closure_training_runtime import TrainingClosureROM

FORMAT = 'causal_history_residual_v1'
CONFIG = {'updates': 512, 'checkpoint_every': 32, 'noise': 0.,
          'fit_global_seconds': 9720., 'total_seconds': 10800.,
          'fit_item_seconds': 540., 'item_seconds': 600.,
          'head_lr': .01, 'batch_size': 12, 'dtype': 'float64'}
ORDER = [(seed, case, group) for seed in (5101, 5102, 5103)
         for case in ('node', 'edge', 'saturation') for group in ('control', 'residual')]
PROFILES = {'all18': (600., 60., 9720., 10800.),
            'node6': (1800., 180., 9720., 10800.),
            'node6_15600': (2600., 120., 14880., 15600.)}


def history_features(history, mass):
    """Public delta, omega, p; only delta uses frozen source-mass COI removal."""
    if history.ndim != 3 or history.shape[1:] != (11, 30):
        raise ValueError('requires batch x 11 x 30 public histories')
    d, w, p = history.split(10, -1)
    d = d - (d * mass).sum(-1, keepdim=True) / mass.sum()
    x = torch.stack((d, w, p), -1)
    t = torch.arange(11, dtype=history.dtype, device=history.device) * .01
    t = t - t.mean()
    slope = (x * t[None, :, None, None]).sum(1) / t.square().sum()
    return torch.cat((x[:, -1], slope), -1)


class CausalHistoryResidual(nn.Module):
    def __init__(self, base, scales, source_mass=None):
        super().__init__()
        self.base = base
        self.register_buffer('history_scales', torch.as_tensor(scales, dtype=base.mass.dtype,
                                                             device=base.mass.device).clone())
        self.register_buffer('source_mass', (base.mass if source_mass is None else
                             torch.as_tensor(source_mass, dtype=base.mass.dtype,
                                             device=base.mass.device)).detach().clone())
        if self.history_scales.shape != (6,) or not bool(torch.isfinite(self.history_scales).all()) or not bool((self.history_scales >= 1e-6).all()):
            raise ValueError('six finite train RMS scales with floor 1e-6 required')
        if self.source_mass.shape != (10,) or not bool(torch.isfinite(self.source_mass).all()) or not bool((self.source_mass > 0).all()):
            raise ValueError('positive finite frozen source masses required')
        self.history_coefficients = nn.Parameter(base.mass.new_zeros(6))

    @classmethod
    def from_train(cls, base, train_histories):
        if train_histories.shape != (16, 11, 30):
            raise ValueError('RMS must pool exactly 16 train histories and 10 machines')
        features = history_features(train_histories, base.mass.detach())
        scales = features.square().mean((0, 1)).sqrt().clamp_min(1e-6).detach()
        return cls(base, scales)

    @property
    def mass(self):
        return self.base.mass

    def encode(self, history):
        history = torch.as_tensor(history, dtype=self.mass.dtype, device=self.mass.device)
        return self.base.encode(history) + (history_features(history, self.source_mass) /
                                          self.history_scales) @ self.history_coefficients

    def field(self, z, u=None):
        return self.base.field(z, u)

    def rk4(self, z, dt, u=None):
        return self.base.rk4(z, dt, u) if u is not None else self.base.rk4(z, dt)

    def begin_graph_cache(self):
        self.base.begin_graph_cache()

    def end_graph_cache(self):
        self.base.end_graph_cache()

    def export(self, path):
        if getattr(self.base, '_runtime_cache', None) is not None:
            raise RuntimeError('close graph before wrapper export')
        payload = {'format': FORMAT, 'base': copy.deepcopy(self.base).cpu()._export_payload(),
                   'history_scales': self.history_scales.detach().cpu().tolist(),
                   'source_mass': self.source_mass.detach().cpu().tolist(),
                   'history_coefficients': self.history_coefficients.detach().cpu().tolist()}
        atomic_json(path, payload)
        return payload

    @classmethod
    def from_export(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get('format') != FORMAT:
            raise ValueError('expected versioned causal-history wrapper')
        b = payload['base']
        if b.get('format') != 'symbolic_closure_v1':
            raise ValueError('unknown nested base format')
        base = TrainingClosureROM.from_config(b['config'], terms=b['terms'])
        state = {k: torch.as_tensor(v, dtype=base.state_dict()[k].dtype) for k, v in b['state_dict'].items()}
        base.load_state_dict(state, strict=True)
        trainable = set(b['trainable_parameters'])
        if not trainable <= set(dict(base.named_parameters())):
            raise ValueError('unknown trainable base parameter')
        for name, p in base.named_parameters():
            p.requires_grad_(name in trainable)
        base.provenance = b.get('provenance', {})
        model = cls(base, payload['history_scales'], payload['source_mass'])
        with torch.no_grad():
            model.history_coefficients.copy_(torch.tensor(payload['history_coefficients'], dtype=base.mass.dtype))
        return model


def make_optimizer(model):
    opt = optimizer(model.base if isinstance(model, CausalHistoryResidual) else model)
    if isinstance(model, CausalHistoryResidual):
        opt.add_param_group({'params': [model.history_coefficients], 'lr': .01})
    return opt


def prediction(model, observations, dt=.01, deadline=None):
    # Construct exactly one initial state for each autonomous rollout.
    h = observations[:, :11]
    initial = torch.cat((h[:, -1], model.encode(h)), -1)
    factor = round(.01 / dt)
    return trajectory(model, initial, (observations.shape[1] - 11) * factor,
                      dt, deadline)[:, factor::factor]


def loss(model, observed, scales, deadline=None):
    return ((prediction(model, observed, deadline=deadline) - observed[:, 11:]) / scales).square().mean()


def train_update(model, opt, rng, observed, scales, deadline=None):
    ix = torch.as_tensor(rng.permutation(len(observed))[:12], device=observed.device)
    opt.zero_grad(set_to_none=True)
    value = loss(model, observed[ix], scales, deadline)
    if not bool(torch.isfinite(value)):
        raise FloatingPointError('nonfinite fit loss')
    value.backward()
    gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
    if not all(math.isfinite(v) for v in gradients.values()):
        raise FloatingPointError('nonfinite gradient')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
    opt.step()
    synchronize(observed.device)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('nonfinite updated parameter')
    return float(value.detach()), gradients


def checkpoint(folder, model, opt, rng, step, trace, identity):
    target = folder / f'checkpoint_{step:05d}'
    if target.exists():
        raise ValueError('checkpoint already exists; refusing to overwrite evidence')
    pending = folder / ('pending_' + uuid.uuid4().hex)
    pending.mkdir()
    if isinstance(model, CausalHistoryResidual):
        model.export(pending / 'model.json')
    else:
        copy.deepcopy(model).cpu().export(pending / 'model.json')
    torch.save({'optimizer': opt.state_dict(), 'fit_numpy_rng': rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if model.mass.is_cuda else [],
                'completed_updates': step, 'trace': trace, 'identity': identity}, pending / 'training.pt')
    os.replace(pending, target)
    atomic_json(folder / 'latest.json', {'checkpoint': target.name, 'updates': step,
                'sha256': {p.name: sha(p) for p in target.iterdir()}})


def restore(folder, model, opt, rng, identity):
    latest = json.loads((folder / 'latest.json').read_text())
    step = latest['updates']
    if type(step) is not int or step not in range(0, 513, 32) or latest['checkpoint'] != f'checkpoint_{step:05d}':
        raise ValueError('invalid checkpoint identity')
    directory = folder / latest['checkpoint']
    if list(folder.glob('pending_*')) or any(p.name > directory.name for p in folder.glob('checkpoint_*')):
        raise ValueError('uncommitted checkpoint evidence; root reconciliation required')
    if latest['sha256'] != {p.name: sha(p) for p in directory.iterdir()}:
        raise ValueError('checkpoint bytes changed')
    restored = type(model).from_export(directory / 'model.json').to(model.mass.device)
    model.load_state_dict(restored.state_dict(), strict=True)
    state = torch.load(directory / 'training.pt', map_location='cpu', weights_only=False)
    if state['identity'] != identity or state['completed_updates'] != step or len(state['trace']) != step:
        raise ValueError('checkpoint provenance/count mismatch')
    opt.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['fit_numpy_rng']
    torch.set_rng_state(state['torch_rng'])
    if model.mass.is_cuda:
        if len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('CUDA identity changed')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    elif state['cuda_rng']:
        raise ValueError('CUDA checkpoint cannot resume on CPU')
    return step, state['trace']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(profile='all18'):
    # Explicit transitive local dependencies of this observation-only runner.
    names = ('closure_causal_history_residual_20261019.py',
             'closure_causal_recovery_20261019.py', 'closure_training_runtime.py',
             'symbolic_closure_rom.py', 'local_memory_rom_0098.py', 'structured_rom.py',
             'structure_grammar.py', 'closure_discovery_benchmarks.py',
             'closure_discovery_support.py', 'closure_expression_identity.py')
    paths = [ROOT / 'experiments' / name for name in names]
    item, reserve, fit_global, total = PROFILES[profile]
    return {'files': {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
            'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'cuda': torch.version.cuda, 'config': CONFIG, 'profile': profile,
            'effective_caps': {'item_total': item, 'item_fit': item - reserve,
                               'item_eval_reserved': reserve, 'items': 18 if profile == 'all18' else 6,
                               'fit_global': fit_global, 'total': total,
                               'eval_global_reserved': total - fit_global}}


def checked_fixture(case, seed, device, folder, deadline):
    source, clean, observed, scales, meta = fixture(case, seed, 0., device, deadline)
    original = ROOT / f'private_data/closure_mechanism_20261019/P1/known_controls/{case}_noise0_seed{seed}'
    pointer = json.loads((original / 'latest.json').read_text())['attempt']
    historical = original / pointer / 'fixture.json'
    old = json.loads(historical.read_text())
    if meta['clean_sha256'] != old['clean_sha256'] or meta['scales'] != old['scales']:
        raise ValueError('observations/scales differ from original fixture')
    if max(meta['dual_step_normalized_max_abs']) > 1e-3:
        raise FloatingPointError('truth dual-step qualification failed')
    meta['original_fixture_sha256'] = sha(historical)
    meta['source_model_sha256'] = hashlib.sha256(json.dumps(source._export_payload(),
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()
    meta = json.loads(json.dumps(meta))
    if (folder / 'fixture.json').exists() and json.loads((folder / 'fixture.json').read_text()) != meta:
        raise ValueError('fixture identity changed on resume')
    atomic_json(folder / 'fixture.json', meta)
    return source, observed, scales, meta


def evaluate(model, observed, scales, deadline):
    rows = {}
    for split, arr in zip(('train', 'heldout'), observed):
        obs = torch.tensor(arr, dtype=torch.float64, device=model.mass.device)
        scale = torch.as_tensor(scales, device=model.mass.device)
        with torch.no_grad():
            h = obs[:, :11]
            initial = torch.cat((h[:, -1], model.encode(h)), -1)
            steps = len(arr[0]) - 11
            coarse = trajectory(model, initial, steps, .01, deadline)[:, 1:]
            fine = trajectory(model, initial, steps * 2, .005, deadline)[:, 2::2]
        if not bool(torch.isfinite(coarse).all() and torch.isfinite(fine).all()):
            raise FloatingPointError('nonfinite evaluation; all events retained')
        errors = ((coarse - obs[:, 11:]) / scale).square().mean((1, 2)).cpu().numpy()
        rows[split] = {'event_mse': errors.tolist(), 'mse': float(errors.mean()),
                       'weak_mse': float(errors[:8].mean()), 'strong_mse': float(errors[8:].mean()),
                       'dual_step_max_normalized': float(((coarse - fine) / scale).abs().max())}
    rows['numerical_qualified'] = all(rows[s]['dual_step_max_normalized'] <= 1e-3 for s in ('train', 'heldout'))
    rows['prediction_gate'] = bool(rows['numerical_qualified'] and rows['heldout']['mse'] <= .01)
    return rows


def run(output, device, resume=False, profile='all18'):
    output = Path(output)
    if profile not in PROFILES:
        raise ValueError('root must lock a registered profile before launch')
    order = ORDER if profile == 'all18' else [x for x in ORDER if x[1] == 'node']
    item_cap, eval_reserve, fit_global, total_cap = PROFILES[profile]
    global_eval_reserve = total_cap - fit_global
    fit_cap = item_cap - eval_reserve
    identity = source_identity(profile)
    if resume:
        state = json.loads((output / 'state.json').read_text())
        # Only caught, fully accounted runtime interrupts are accepted. Abrupt
        # kill/power loss has unknown elapsed cost and deliberately needs root reconciliation.
        if state['status'] != 'INTERRUPTED_RUNTIME' or state['resumes'] != 0 or state['identity'] != identity or state['device'] != str(device):
            raise ValueError('unknown interruption/source/device or resume already used')
        state['resumes'] = 1
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = {'identity': identity, 'device': str(device), 'resumes': 0,
                 'elapsed_seconds': time.perf_counter() - PROCESS_START, 'fit_seconds': 0., 'phase': 'fit',
                 'items': [{'seed': s, 'case': c, 'group': g, 'fit_status': 'pending',
                            'fit_seconds': 0., 'eval_seconds': 0.} for s, c, g in order]}
        atomic_json(output / 'identity.json', identity)
    prior = state['elapsed_seconds'] if resume else 0.
    started = PROCESS_START
    state.update(status='RUNNING', pid=os.getpid(), command=sys.argv, children=[],
                 child_accounting='no subprocesses; process elapsed includes all in-process work')
    active = None
    phase_start = None

    def persist():
        state['elapsed_seconds'] = prior + time.perf_counter() - started
        atomic_json(output / 'state.json', state)

    try:
        persist()
        if state['phase'] == 'fit':
            for item in state['items']:
                if item['fit_status'] not in ('pending', 'running'):
                    continue
                name = f"{item['case']}_{item['seed']}_{item['group']}"
                folder = output / name
                folder.mkdir(exist_ok=resume)
                active = item
                state['current'] = name
                item['fit_status'] = 'running'
                phase_start = time.perf_counter()
                deadline = phase_start + max(0., min(fit_cap - item['fit_seconds'], fit_global - state['fit_seconds'],
                            total_cap - (prior + phase_start - started) - global_eval_reserve))
                persist()
                try:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('no fit budget remains')
                    source, observations, scales, meta = checked_fixture(item['case'], item['seed'], device, folder, deadline)
                    train = torch.tensor(observations[0], dtype=torch.float64, device=device)
                    scale = torch.tensor(scales, dtype=torch.float64, device=device)
                    model = runtime_model(source, [truths()[item['case']][0]], item['seed'], device)
                    if item['group'] == 'residual':
                        model = CausalHistoryResidual.from_train(model, train[:, :11])
                    opt = make_optimizer(model)
                    rng = np.random.default_rng(np.random.SeedSequence([item['seed'], 7]))
                    key = {'run': identity, 'item': name, 'fixture_sha256': sha(folder / 'fixture.json')}
                    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
                    if not (folder / 'latest.json').exists() and (list(folder.glob('pending_*')) or list(folder.glob('checkpoint_*'))):
                        raise ValueError('orphan checkpoint without authoritative pointer')
                    completed, trace = (restore(folder, model, opt, rng, key) if (folder / 'latest.json').exists()
                                        else (0, []))
                    if not (folder / 'latest.json').exists():
                        checkpoint(folder, model, opt, rng, 0, trace, key)
                    milestones = json.loads((folder / 'milestones.json').read_text()) if (folder / 'milestones.json').exists() else {}
                    gradients = {}
                    for step in range(completed + 1, 513):
                        if time.perf_counter() >= deadline:
                            raise TimeoutError('fit cap; partial endpoint not scored')
                        value, gradients = train_update(model, opt, rng, train, scale, deadline)
                        trace.append(value)
                        if step in (128, 256, 512):
                            with torch.no_grad():
                                milestones[str(step)] = float(loss(model, train, scale, deadline))
                            atomic_json(folder / 'milestones.json', milestones)
                        if step % 32 == 0:
                            checkpoint(folder, model, opt, rng, step, trace, key)
                    item.update(fit_status='complete', updates=512, train_milestones=milestones,
                                last_gradient_norms=gradients,
                                parameter_changes={n: float((p - initial[n]).detach().norm()) for n, p in model.named_parameters()})
                except Exception as error:
                    item.update(fit_status='unresolved', error=repr(error))
                    atomic_json(folder / ('failure_' + uuid.uuid4().hex + '.json'), {'error': repr(error), 'traceback': traceback.format_exc()})
                elapsed = time.perf_counter() - phase_start
                item['fit_seconds'] += elapsed
                state['fit_seconds'] += elapsed
                phase_start = None
                persist()
            state['phase'] = 'evaluation'
            atomic_json(output / 'fits_frozen.json', state['items'])
            persist()
        # No heldout model scores exist before this durable freeze barrier.
        for item in state['items']:
            if item['fit_status'] != 'complete' or 'evaluation' in item or 'evaluation_error' in item:
                continue
            name = f"{item['case']}_{item['seed']}_{item['group']}"
            folder = output / name
            state['current'] = name
            active = item
            phase_start = time.perf_counter()
            deadline = phase_start + max(0., min(item_cap - item['fit_seconds'] - item['eval_seconds'],
                         total_cap - (prior + phase_start - started)))
            persist()
            try:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('no evaluation budget remains')
                _, observed, scales, _ = checked_fixture(item['case'], item['seed'], device, folder, deadline)
                cls = CausalHistoryResidual if item['group'] == 'residual' else TrainingClosureROM
                model = cls.from_export(folder / 'checkpoint_00512' / 'model.json').to(device)
                item['evaluation'] = evaluate(model, observed, scales, deadline)
                atomic_json(folder / 'result.json', item['evaluation'])
            except Exception as error:
                item['evaluation_error'] = repr(error)
                atomic_json(folder / ('evaluation_failure_' + uuid.uuid4().hex + '.json'), {'error': repr(error), 'traceback': traceback.format_exc()})
            item['eval_seconds'] += time.perf_counter() - phase_start
            phase_start = None
            persist()
        pairs = []
        for control, residual in zip(state['items'][::2], state['items'][1::2]):
            if 'evaluation' in control and 'evaluation' in residual:
                pairs.append({'case': control['case'], 'seed': control['seed'],
                    'residual_minus_control': {s: {k: residual['evaluation'][s][k] - control['evaluation'][s][k]
                    for k in ('mse', 'weak_mse', 'strong_mse')} for s in ('train', 'heldout')}})
        atomic_json(output / 'paired_differences.json', pairs)
        state.update(status='CLOSED', phase='closed', current=None)
    except KeyboardInterrupt:
        if active is not None and phase_start is not None:
            elapsed = time.perf_counter() - phase_start
            if state['phase'] == 'fit':
                active['fit_seconds'] += elapsed
                state['fit_seconds'] += elapsed
            else:
                active['eval_seconds'] += elapsed
        state['status'] = 'INTERRUPTED_RUNTIME'
        raise
    except BaseException as error:
        state.update(status='UNRESOLVED', error=repr(error))
        raise
    finally:
        persist()
        atomic_json(output / 'receipt.json', {'pid': os.getpid(), 'status': state['status'],
                    'cumulative_seconds': state['elapsed_seconds'], 'this_invocation_seconds': time.perf_counter() - PROCESS_START,
                    'prior_seconds': prior, 'children': [], 'includes_all_children_once': True,
                    'fit_seconds': state['fit_seconds'], 'current': state.get('current')})
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output, resolve_device(args.device), args.resume, args.profile)
