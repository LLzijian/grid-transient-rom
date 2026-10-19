"""Registered six-item saturated-port comparison; synthetic observations only."""
from __future__ import annotations
import time
PROCESS_START = time.perf_counter()  # Includes heavy imports and all startup work.
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import numpy as np
import torch
import closure_causal_history_residual_20261019 as core
from closure_causal_recovery_20261019 import copy_runtime
from closure_discovery_benchmarks import toy_source
from symbolic_closure_rom import SymbolicClosureROM
from structured_rom import StructuredROM
from tanh_port_runtime_20261019 import TERM, TanhPortTrainingROM, select_runtime

ORDER = [(s, 'saturation', g) for s in (5101, 5102, 5103)
         for g in ('control', 'residual')]
CAPS = dict(startup=50., fit=535., evaluation=120., item=655., items=3930., total=4000.)
PINNED = {'closure_causal_history_residual_20261019.py': 'abe887a7c4758f0ad21fa0d9fb97282ce22349a03eb70a09bbb478a8f54f364a',
          'tanh_port_runtime_20261019.py': 'd0e46fb0894713ba0123ad2957f48abe7f65893a049f0bd50138f5902356faa5'}


def source_identity():
    identity = core.source_identity()
    files = identity['files']
    for name, expected in PINNED.items():
        actual = core.sha(ROOT / 'experiments' / name)
        if actual != expected:
            raise ValueError('registered source changed: ' + name)
        files['experiments/' + name] = actual
    files[str(Path(__file__).relative_to(ROOT))] = core.sha(__file__)
    return {'files': files, 'python': sys.version, 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda': torch.version.cuda, 'caps': CAPS,
            'order': ORDER, 'updates': 512, 'checkpoint_every': 32,
            'dtype': 'float64', 'noise': 0., 'batch_size': 12, 'head_lr': .01}


def structure(model):
    """No mutable learned values: admission remains valid after optimization."""
    s = model.structured
    return {'attrs': (model.arm, model.n, model.rank, model.dim, model.seed,
                      s.n, s.n_p, s.n_q),
            'terms': model.terms, 'kinds': model._term_kinds,
            'specs': (s.storage_specs, s.exchange_specs, s.dissipation_specs, s.port_specs),
            'modules': [(n, type(m).__module__, type(m).__name__, m.extra_repr())
                        for n, m in model.named_modules() if n],
            'parameters': [(n, tuple(p.shape), str(p.dtype), p.requires_grad)
                           for n, p in model.named_parameters()]}


def strict_admit(model, reference):
    if type(model) not in (core.TrainingClosureROM, TanhPortTrainingROM):
        raise TypeError('only exact unwrapped registered training runtime admitted')
    if type(model.structured) is not StructuredROM or model.terms != [TERM]:
        raise ValueError('not the unique canonical tanh(2q) toy')
    if model.structured.D.shape != (10, 10) or model.structured.D.numel() == 0:
        raise ValueError('registered nonempty D required')
    if getattr(model, '_runtime_cache', None) is not None:
        raise ValueError('open graph at admission')
    if structure(model) != structure(reference):
        raise ValueError('registered structure/order/trainability changed')
    a, b = dict(model.named_buffers()), dict(reference.named_buffers())
    if list(a) != list(b) or any(a[k].dtype != b[k].dtype or
            not torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a):
        raise ValueError('registered immutable toy buffers changed')
    return model


def fast_model(model, reference):
    strict_admit(model, reference)
    selected = select_runtime(model, fast=True)
    strict_admit(selected, reference)
    if not selected.supports_fast_field():
        raise ValueError('specialization rejected; no fallback allowed')
    return selected


def evaluation_model(path, group, device, reference):
    cls = core.CausalHistoryResidual if group == 'residual' else core.TrainingClosureROM
    model = cls.from_export(path).to(device)
    if group == 'residual':
        model.base = fast_model(model.base, reference)
    else:
        model = fast_model(model, reference)
    return model


def cache_digest(observations, scales):
    return [hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
            for a in [*observations, scales]]


def load_cache(output, seed, manifest):
    folder = output / 'fixtures' / str(seed)
    record = manifest[str(seed)]
    if {n: core.sha(folder / n) for n in record['files']} != record['files']:
        raise ValueError('fixture cache bytes changed')
    with np.load(folder / 'observations.npz', allow_pickle=False) as data:
        if set(data.files) != {'train', 'val', 'scales'}:
            raise ValueError('unexpected cache contents')
        observed = [data['train'].copy(), data['val'].copy()]
        scales = data['scales'].copy()
    meta = json.loads((folder / 'fixture.json').read_text())
    if cache_digest(observed, scales) != record['arrays'] or meta != record['meta']:
        raise ValueError('fixture identity mismatch')
    if [a.shape for a in observed] != [(16, 61, 30), (16, 111, 30)] or scales.shape != (30,):
        raise ValueError('registered fixture shapes changed')
    if any(a.dtype != np.float64 or not np.isfinite(a).all() for a in [*observed, scales]):
        raise ValueError('nonfinite/wrong dtype fixture')
    if cache_digest(observed, scales)[:2] != meta['clean_sha256'] or scales.tolist() != meta['scales']:
        raise ValueError('cached observations differ from registered noise-zero fixture')
    if max(meta['dual_step_normalized_max_abs']) > 1e-3:
        raise ValueError('unqualified truth integration')
    return observed, scales, meta


def prepare(output, device, deadline):
    fixtures = output / 'fixtures'
    fixtures.mkdir()
    # This independent source factory is not a fixture or a hidden truth export.
    source = toy_source()
    source.export(fixtures / 'source.json')
    records = {}
    for seed in (5101, 5102, 5103):
        if time.perf_counter() >= deadline:
            raise TimeoutError('shared fixture startup cap before any updates')
        folder = fixtures / str(seed)
        folder.mkdir()
        generated, obs, scales, meta = core.checked_fixture('saturation', seed, device, folder, deadline)
        if generated._export_payload() != source._export_payload():
            raise ValueError('fixture source does not match independent original factory')
        np.savez(folder / 'observations.npz', train=obs[0], val=obs[1], scales=scales)
        records[str(seed)] = {'files': {n: core.sha(folder / n) for n in ('fixture.json', 'observations.npz')},
                             'arrays': cache_digest(obs, scales), 'meta': meta}
    manifest = {'source_sha256': core.sha(fixtures / 'source.json'), 'seeds': records}
    core.atomic_json(fixtures / 'manifest.json', manifest)
    for seed in (5101, 5102, 5103):
        load_cache(output, seed, records)
    if time.perf_counter() >= deadline:
        raise TimeoutError('shared fixture startup cap before any updates')
    return manifest


def phase_budget(state, item, phase, elapsed):
    spent = sum(i['fit_seconds'] + i['eval_seconds'] for i in state['items'])
    key = 'fit_seconds' if phase == 'fit' else 'eval_seconds'
    return max(0., min(CAPS['fit' if phase == 'fit' else 'evaluation'] - item[key],
                       CAPS['item'] - item['fit_seconds'] - item['eval_seconds'],
                       CAPS['items'] - spent, CAPS['total'] - elapsed))


def resume_allowed(state, identity, device):
    if (state['status'] != 'INTERRUPTED_RUNTIME' or state['resumes'] != 0 or
            state['identity'] != identity or state['device'] != str(device)):
        raise ValueError('unknown interruption/source/device or resume already used')
    if state['phase'] == 'startup' or state['elapsed_seconds'] >= CAPS['total']:
        raise ValueError('incomplete fixture startup or exhausted process cannot resume')


def run(output, device, resume=False):
    output = Path(output)
    # JSON normalization makes identities stable over disk round trips.
    identity = json.loads(json.dumps(source_identity()))
    if resume:
        state = json.loads((output / 'state.json').read_text())
        resume_allowed(state, identity, device)
        state['resumes'] = 1
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = {'identity': identity, 'device': str(device), 'resumes': 0,
                 'elapsed_seconds': 0., 'phase': 'startup',
                 'items': [dict(seed=s, case=c, group=g, fit_status='pending',
                                fit_seconds=0., eval_seconds=0.) for s, c, g in ORDER]}
        core.atomic_json(output / 'identity.json', identity)
    prior = state['elapsed_seconds'] if resume else 0.
    state.update(status='RUNNING', pid=os.getpid(), command=sys.argv, cwd=str(Path.cwd()), children=[])
    active = None
    phase_start = None
    active_phase = None

    def elapsed():
        return prior + time.perf_counter() - PROCESS_START

    def persist():
        state['elapsed_seconds'] = elapsed()
        core.atomic_json(output / 'state.json', state)

    def account():
        nonlocal phase_start
        if active is not None and phase_start is not None:
            active['fit_seconds' if active_phase == 'fit' else 'eval_seconds'] += time.perf_counter() - phase_start
            phase_start = None

    try:
        persist()
        if not resume:
            state['manifest'] = prepare(output, device, PROCESS_START + CAPS['startup'])
            state['startup_seconds'] = time.perf_counter() - PROCESS_START
            state['phase'] = 'fit'
            persist()
        manifest = state['manifest']
        manifest_path = output / 'fixtures' / 'manifest.json'
        if json.loads(manifest_path.read_text()) != manifest or core.sha(output / 'fixtures' / 'source.json') != manifest['source_sha256']:
            raise ValueError('source factory/manifest identity changed')
        source = SymbolicClosureROM.from_export(output / 'fixtures' / 'source.json')
        reference = copy_runtime(toy_source(), [TERM], 'cpu')
        strict_admit(copy_runtime(source, [TERM], 'cpu'), reference)
        for seed in (5101, 5102, 5103):
            load_cache(output, seed, manifest['seeds'])
        if state['phase'] == 'fit':
            for item in state['items']:
                if item['fit_status'] not in ('pending', 'running'):
                    continue
                name = f"saturation_{item['seed']}_{item['group']}"
                folder = output / name
                folder.mkdir(exist_ok=resume)
                active, active_phase, phase_start = item, 'fit', time.perf_counter()
                deadline = phase_start + phase_budget(state, item, 'fit', elapsed())
                state['current'] = name
                item['fit_status'] = 'running'
                persist()
                try:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('no registered fit budget remains')
                    obs, scales, _ = load_cache(output, item['seed'], manifest['seeds'])
                    train = torch.tensor(obs[0], dtype=torch.float64, device=device)
                    scale = torch.tensor(scales, dtype=torch.float64, device=device)
                    model = fast_model(core.runtime_model(source, [TERM], item['seed'], device), reference)
                    if item['group'] == 'residual':
                        model = core.CausalHistoryResidual.from_train(model, train[:, :11])
                    opt = core.make_optimizer(model)
                    rng = np.random.default_rng(np.random.SeedSequence([item['seed'], 7]))
                    key = {'run': identity, 'item': name, 'fixture': manifest['seeds'][str(item['seed'])],
                           'factory': manifest['source_sha256']}
                    if not (folder / 'latest.json').exists() and (list(folder.glob('pending_*')) or list(folder.glob('checkpoint_*'))):
                        raise ValueError('orphan checkpoint; no automatic restart')
                    completed, trace = core.restore(folder, model, opt, rng, key) if (folder / 'latest.json').exists() else (0, [])
                    strict_admit(model.base if item['group'] == 'residual' else model, reference)
                    if not (folder / 'latest.json').exists():
                        core.checkpoint(folder, model, opt, rng, 0, trace, key)
                    milestones = json.loads((folder / 'milestones.json').read_text()) if (folder / 'milestones.json').exists() else {}
                    for step in range(completed + 1, 513):
                        if time.perf_counter() >= deadline:
                            raise TimeoutError('fit cap; partial endpoint never scored')
                        value, gradients = core.train_update(model, opt, rng, train, scale, deadline)
                        trace.append(value)
                        if step in (128, 256, 512):
                            with torch.no_grad():
                                milestones[str(step)] = float(core.loss(model, train, scale, deadline))
                            core.atomic_json(folder / 'milestones.json', milestones)
                        if step % 32 == 0:
                            core.checkpoint(folder, model, opt, rng, step, trace, key)
                        if time.perf_counter() >= deadline:
                            raise TimeoutError('fit serialization/update exceeded cap')
                    item.update(fit_status='complete', updates=512, train_milestones=milestones,
                                last_gradient_norms=gradients if completed < 512 else {},
                                checkpoint=str(folder / 'checkpoint_00512'))
                except Exception as error:
                    item.update(fit_status='unresolved', error=repr(error))
                    core.atomic_json(folder / ('failure_' + uuid.uuid4().hex + '.json'),
                                     {'error': repr(error), 'traceback': traceback.format_exc()})
                finally:
                    account()
                persist()
            core.atomic_json(output / 'fits_frozen.json', state['items'])
            state.update(phase='evaluation', freeze_sha256=core.sha(output / 'fits_frozen.json'))
            persist()
        if core.sha(output / 'fits_frozen.json') != state['freeze_sha256']:
            raise ValueError('fit freeze changed')
        for item in state['items']:
            if item['fit_status'] != 'complete' or 'evaluation' in item or 'evaluation_error' in item:
                continue
            name = f"saturation_{item['seed']}_{item['group']}"
            folder = output / name
            active, active_phase, phase_start = item, 'evaluation', time.perf_counter()
            state['current'] = name
            deadline = phase_start + phase_budget(state, item, 'evaluation', elapsed())
            persist()
            try:
                if time.perf_counter() >= deadline:
                    raise TimeoutError('no registered evaluation budget remains')
                obs, scales, _ = load_cache(output, item['seed'], manifest['seeds'])
                # Validate authoritative checkpoint bytes and identity before export load.
                latest = json.loads((folder / 'latest.json').read_text())
                endpoint = folder / 'checkpoint_00512'
                if latest['updates'] != 512 or latest['checkpoint'] != endpoint.name or latest['sha256'] != {p.name: core.sha(p) for p in endpoint.iterdir()}:
                    raise ValueError('endpoint identity changed')
                model = evaluation_model(endpoint / 'model.json', item['group'], device, reference)
                result = core.evaluate(model, obs, scales, deadline)
                result['val'] = result.pop('heldout')
                if time.perf_counter() >= deadline:
                    raise TimeoutError('evaluation exceeded cap')
                core.atomic_json(folder / 'result.json', result)
                item['evaluation'] = result
            except Exception as error:
                item['evaluation_error'] = repr(error)
                core.atomic_json(folder / ('evaluation_failure_' + uuid.uuid4().hex + '.json'),
                                 {'error': repr(error), 'traceback': traceback.format_exc()})
            finally:
                account()
            persist()
        pairs = []
        for control, residual in zip(state['items'][::2], state['items'][1::2]):
            if 'evaluation' in control and 'evaluation' in residual:
                pairs.append({'seed': control['seed'], 'case': 'saturation',
                              'residual_minus_control': {s: {k: residual['evaluation'][s][k] - control['evaluation'][s][k]
                              for k in ('mse', 'weak_mse', 'strong_mse')} for s in ('train', 'val')}})
        core.atomic_json(output / 'paired_differences.json', pairs)
        state.update(status='CLOSED', phase='closed', current=None, complete_pairs=len(pairs))
    except KeyboardInterrupt:
        account()
        state['status'] = 'INTERRUPTED_RUNTIME'
        raise
    except BaseException as error:
        account()
        state.update(status='UNRESOLVED', error=repr(error))
        raise
    finally:
        persist()
        core.atomic_json(output / 'receipt.json', {'pid': os.getpid(), 'status': state['status'],
                         'cumulative_seconds': elapsed(), 'this_invocation_seconds': time.perf_counter() - PROCESS_START,
                         'prior_seconds': prior, 'children': [], 'includes_all_children_once': True,
                         'command': sys.argv, 'current': state.get('current')})
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output, core.resolve_device(args.device), args.resume)
