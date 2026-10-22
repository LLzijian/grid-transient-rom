"""Prospective uncapped entry to the unchanged finite grammar experiment.

No external resume is offered. Internal 32/128 continuation is the original
serializer. Validation reads actual states, never infers updates from a name.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import finite_grammar_discovery_20261021 as science

SEEDS = science.SEEDS
SCIENCE_MODULE = science.__name__
POOL_SHA256 = '00040fbf83705eb660474de660ef96fec0424a299fd3650df4c1f77b605a8925'
TEACHER_SHA256 = '5e31e4a454c45d56aa8ca2586cb39cde1ce96cbcb485fd7316fdbda6af36157c'


def frozen_pool(binding):
    if binding['pool_manifest']['sha256'] != POOL_SHA256:
        raise ValueError('pool identity changed')
    if binding['teacher_identity']['sha256'] != TEACHER_SHA256:
        raise ValueError('teacher identity changed')
    return science.read_pool(binding)


def validate_state(state, identity, updates):
    """Count, Adam and RNG checks shared by real endpoints and small CPU QA."""
    import numpy as np
    import torch
    if type(updates) is not int or type(state.get('completed_updates')) is not int:
        raise ValueError('integer actual update count required')
    if state['completed_updates'] != updates or state.get('identity') != identity:
        raise ValueError('actual count or checkpoint identity mismatch')
    trace = state.get('trace')
    if not isinstance(trace, list) or len(trace) != updates or not np.isfinite(trace).all():
        raise ValueError('finite full actual loss trace required')
    opt = state.get('optimizer', {})
    groups, slots = opt.get('param_groups', []), opt.get('state', {})
    params = [p for group in groups for p in group['params']]
    if not params or len(set(params)) != len(params) or not set(slots) <= set(params):
        raise ValueError('invalid optimizer parameter identity')
    if updates and not slots:
        raise ValueError('nonempty actual Adam state required')
    if not updates and slots:
        raise ValueError('initial Adam state must be empty')
    for slot in slots.values():
        step = slot.get('step')
        if not torch.is_tensor(step) or step.numel() != 1 or float(step) != updates:
            raise ValueError('actual per-parameter Adam step mismatch')
        for name in ('exp_avg', 'exp_avg_sq'):
            value = slot.get(name)
            if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
                raise ValueError('finite Adam moments required')
        if slot['exp_avg'].shape != slot['exp_avg_sq'].shape:
            raise ValueError('Adam moment shapes differ')
    # Validate without installing any checkpoint RNG into the running process.
    np.random.default_rng().bit_generator.state = state['fit_numpy_rng']
    cpu = state.get('torch_rng')
    if not torch.is_tensor(cpu) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError('actual CPU RNG required')
    torch.Generator(device='cpu').set_state(cpu)
    cuda = state.get('cuda_rng')
    if not isinstance(cuda, list) or any(not torch.is_tensor(x) or x.dtype != torch.uint8 or x.ndim != 1 or not x.numel() for x in cuda):
        raise ValueError('actual CUDA RNG records required')
    return {'updates': updates, 'adam_parameters': len(slots), 'trace_length': len(trace),
            'cuda_rng_count': len(cuda)}


def validate_checkpoint(directory, identity, updates, files=None, require_cuda=True):
    """Check serialized bytes and actual state, leaving global RNG untouched."""
    import torch
    directory = Path(directory)
    parent = directory.parent
    if directory.name != f'checkpoint_{updates:05d}' or not directory.is_dir():
        raise ValueError('checkpoint directory/count mismatch')
    if set(p.name for p in directory.iterdir()) != {'model.json', 'training.pt'}:
        raise ValueError('unknown checkpoint contents')
    hashes = {n: science.sha(directory / n) for n in ('model.json', 'training.pt')}
    if files is not None and files != hashes:
        raise ValueError('checkpoint bytes changed')
    if list(parent.glob('pending_*')):
        raise ValueError('pending checkpoint evidence')
    state = torch.load(directory / 'training.pt', map_location='cpu', weights_only=False)
    proof = validate_state(state, identity, updates)
    if require_cuda and proof['cuda_rng_count'] != torch.cuda.device_count():
        raise ValueError('CUDA RNG device identity mismatch')
    if require_cuda and not proof['cuda_rng_count']:
        raise ValueError('formal endpoint must carry CUDA RNG')
    # Validate export/Adam compatibility with the actual wrapper, not a rational model.
    from closure_causal_history_residual_20261019 import CausalHistoryResidual, make_optimizer
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        model = CausalHistoryResidual.from_export(directory / 'model.json')
        optimizer = make_optimizer(model)
        current = optimizer.state_dict()['param_groups']
        saved = state['optimizer']['param_groups']
        if len(current) != len(saved) or any(len(a['params']) != len(b['params']) for a, b in zip(current, saved)):
            raise ValueError('optimizer group/export mismatch')
        optimizer.load_state_dict(state['optimizer'])
        for param, slot in optimizer.state.items():
            if slot['exp_avg'].shape != param.shape or slot['exp_avg_sq'].shape != param.shape:
                raise ValueError('optimizer state/export shape mismatch')
        if any(not bool(torch.isfinite(x).all()) for x in model.state_dict().values()):
            raise ValueError('nonfinite model state')
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
    if hashes != {n: science.sha(directory / n) for n in hashes}:
        raise ValueError('checkpoint changed during state validation')
    return {**proof, 'files': hashes, 'trace': state['trace']}


def validate_layout(candidate_folder, steps):
    candidate_folder = Path(candidate_folder)
    if {p.name for p in candidate_folder.iterdir()} != {'latest.json', *(f'checkpoint_{s:05d}' for s in steps)}:
        raise ValueError('unknown/pending/future candidate evidence')
    latest = json.loads((candidate_folder / 'latest.json').read_text())
    last = steps[-1]
    if type(latest['updates']) is not int or latest['updates'] != last or latest['checkpoint'] != f'checkpoint_{last:05d}':
        raise ValueError('latest checkpoint differs from actual ranking path')
    return latest


def _validate_row(row, pool, commit):
    import numpy as np
    if row.get('status') != 'FROZEN' or row.get('seed') not in SEEDS:
        raise ValueError('all three real frozen endpoints required before val')
    raw = row['train_observations']
    if science.sha(raw['path']) != raw['sha256']:
        raise ValueError('train observations changed')
    with np.load(raw['path'], allow_pickle=False) as arr:
        if set(arr.files) != {'coarse', 'fine', 'scales'}:
            raise ValueError('unexpected train arrays')
        coarse, fine, scales = (arr[k] for k in ('coarse', 'fine', 'scales'))
        qualification = science.qualify_teacher(coarse, fine, scales)
        if not qualification['valid'] or qualification != row['teacher_qualification']:
            raise ValueError('teacher full interval qualification changed')
        if not np.array_equal(scales, np.maximum(coarse[:, :61].std((0, 1)), 1e-6)):
            raise ValueError('scales must come from full train histories')
        train_hash = hashlib.sha256(coarse.tobytes()).hexdigest()
    folder = Path(raw['path']).parent
    history = row['history']
    if [stage['updates'] for stage in history] != [32, 128, 512]:
        raise ValueError('missing selection stage')
    expected = list(range(6)); by_candidate = {i: [0] for i in expected}
    final = None
    for stage, target, keep in zip(history, (32, 128, 512), (2, 1, 1)):
        if [r['index'] for r in stage['rows']] != expected:
            raise ValueError('candidate path changed')
        if json.loads((folder / f'ranking_{target}.json').read_text()) != stage:
            raise ValueError('ranking evidence changed')
        for candidate in stage['rows']:
            index = candidate['index']
            if candidate['key'] != pool['keys'][index] or not candidate['score']['valid']:
                raise ValueError('invalid or changed candidate ranking')
            directory = folder / f'candidate_{index}' / f'checkpoint_{target:05d}'
            if Path(candidate['path']).resolve() != directory.resolve():
                raise ValueError('candidate checkpoint source changed')
            by_candidate[index].append(target)
        selected, ambiguous = science.choose(copy.deepcopy(stage['rows']), keep)
        if ambiguous != stage['ambiguous_boundary'] or not selected:
            raise ValueError('selection boundary changed')
        expected = [r['index'] for r in selected]; final = selected[0]
    if final != row['selected']:
        raise ValueError('frozen selection differs from train-only rule')
    checkpoint_count = updates = 0
    proofs = []
    if {p.name for p in folder.glob('candidate_*')} != {f'candidate_{i}' for i in range(6)}:
        raise ValueError('unexpected candidate directory')
    for index, steps in by_candidate.items():
        candidate_folder = folder / f'candidate_{index}'
        latest = validate_layout(candidate_folder, steps)
        last = steps[-1]
        identity = {'source_commit': commit, 'seed': row['seed'], 'train_sha256': train_hash,
                    'candidate_key': pool['keys'][index]}
        previous_trace = []
        for step in steps:
            files = latest['sha256'] if step == last else None
            proof = validate_checkpoint(candidate_folder / f'checkpoint_{step:05d}', identity, step, files)
            if proof['trace'][:len(previous_trace)] != previous_trace:
                raise ValueError('continuation loss trace prefix changed')
            previous_trace = proof.pop('trace')
            if step:
                stage_row = next(r for s in history if s['updates'] == step for r in s['rows'] if r['index'] == index)
                if stage_row['files'] != proof['files']:
                    raise ValueError('ranked snapshot differs from actual state')
            proofs.append({'candidate': index, **proof}); checkpoint_count += 1
        updates += last
    return {'seed': row['seed'], 'successful_updates': updates,
            'checkpoint_directories': checkpoint_count, 'checkpoints': proofs}


def validate_selection(selection, binding, commit):
    pool = frozen_pool(binding)
    if selection.get('source_commit') != commit or selection.get('selection') != 'train_only' or not selection.get('all_three_frozen'):
        raise ValueError('all three same-source frozen endpoints required before val')
    if [r['seed'] for r in selection['rows']] != list(SEEDS):
        raise ValueError('all three registered seeds required')
    proofs = [_validate_row(row, pool, commit) for row in selection['rows']]
    total = {'successful_updates': sum(p['successful_updates'] for p in proofs),
             'checkpoint_directories': sum(p['checkpoint_directories'] for p in proofs),
             'fully_published_candidate_rankings': sum(len(s['rows']) for r in selection['rows'] for s in r['history'])}
    if total != {'successful_updates': 2304, 'checkpoint_directories': 45, 'fully_published_candidate_rankings': 27}:
        raise ValueError('actual workload incomplete')
    if selection['recorded_completed_counts'] != total or selection['planned_counts'] != science.counts():
        raise ValueError('reported counts differ from actual states')
    return {'status': 'PASS', 'counts': total, 'rows': proofs}


def run_study(output, binding, commit):
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    pool = frozen_pool(binding); truth = science.teacher(binding); rows = []
    for seed in SEEDS:
        folder = output / str(seed); folder.mkdir()
        row = {'seed': seed, 'status': 'UNRESOLVED', 'selected': None, 'history': []}
        try:
            coarse, fine, scales = science.fixture(truth['term'], seed, 0, None)
            np.savez(folder / 'train_observations.npz', coarse=coarse, fine=fine, scales=scales)
            row['train_observations'] = {'path': str(folder / 'train_observations.npz'),
                                         'sha256': science.sha(folder / 'train_observations.npz')}
            qualification = science.qualify_teacher(coarse, fine, scales)
            row['teacher_qualification'] = qualification
            science.atomic(folder / 'teacher_qualification.json', qualification)
            if qualification['valid']:
                result = science.search_seed(pool['selected'], pool['keys'], seed, coarse, fine, scales, folder, commit, None)
                row.update(result)
        except (RuntimeError, FloatingPointError, ValueError) as error:
            row.update(status='UNRESOLVED', selected=None, error=repr(error), traceback=traceback.format_exc())
            science.atomic(folder / 'failure.json', row)
        rows.append(row)
        science.atomic(output / 'progress.json', {'rows': rows, 'source_commit': commit})
    updates = directories = 0
    for folder in output.glob('*/candidate_*'):
        latest = json.loads((folder / 'latest.json').read_text()) if (folder / 'latest.json').exists() else {'updates': 0}
        count = latest['updates']
        for failure in folder.glob('failure_before_*.json'):
            count = max(count, json.loads(failure.read_text()).get('actual_updates', 0))
        updates += count; directories += len(list(folder.glob('checkpoint_*')))
    selection = {'status': 'COMPLETED', 'all_three_frozen': len(rows) == 3 and all(r['status'] == 'FROZEN' for r in rows),
                 'source_commit': commit, 'rows': rows, 'selection': 'train_only', 'planned_counts': science.counts(),
                 'recorded_completed_counts': {'successful_updates': updates, 'checkpoint_directories': directories,
                    'fully_published_candidate_rankings': sum(len(s['rows']) for r in rows for s in r['history'])}}
    if selection['all_three_frozen']:
        try:
            science.atomic(output / 'freeze_validation.json', validate_selection(selection, binding, commit))
        except (RuntimeError, FloatingPointError, ValueError, KeyError, OSError) as error:
            selection['all_three_frozen'] = False
            selection['freeze_error'] = repr(error)
    science.atomic(output / 'selection.json', selection)
    return selection


def evaluate_study(output, binding, commit):
    """The original evaluator is called once only after a real three-seed barrier."""
    output = Path(output)
    selection = science.bound(binding, 'selection_manifest')
    before = validate_selection(selection, binding, commit)
    output.mkdir(parents=True, exist_ok=False)
    # Complete train windows were not computed in the original ranking routine.
    # Report them after the selection is frozen, without altering the selector.
    import numpy as np
    import torch
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    train_rows = []
    for row in selection['rows']:
        with np.load(row['train_observations']['path'], allow_pickle=False) as arr:
            coarse, fine, scales = (arr[k] for k in ('coarse', 'fine', 'scales'))
            model = CausalHistoryResidual.from_export(Path(row['selected']['path']) / 'model.json').cuda()
            mc = science.predict(model, coarse, .01, None); mf = science.predict(model, coarse, .005, None)
            train_rows.append({'seed': row['seed'], 'windows': {name: science.score(mc, mf, coarse[:, 11:], fine[:, 11:], scales, a, b)
                              for name, a, b in [('one_second', 0, 100), ('five_seconds', 0, 500), ('primary', 200, 500)]}})
            del model
    science.atomic(output / 'train_windows.json', {'rows': train_rows})
    result = science.evaluate_study(output, binding, commit, None)
    after = validate_selection(selection, binding, commit)
    if before != after or science.bound(binding, 'selection_manifest') != selection:
        raise ValueError('frozen endpoints changed during evaluation')
    result.update(source_commit=commit, train_windows=train_rows, freeze_validation=after,
                  project_val_accessed=False, project_test_accessed=False)
    science.atomic(output / 'result.json', result)
    return result
