"""Fixed 12-model train-only window comparison; no data generation or val API."""
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
import finite_grammar_execution_20261022 as checks

SEEDS = (6301, 6302, 6303)
TRAIN_SEEDS = (6201, 6202, 6203)
CANDIDATES = (2, 3)
WINDOWS = (50, 200)
STEPS = (0, 512, 2048)
OLD_SOURCE = 'c9ec666fc7e8dd5ad04f63d9bea3ada33d4ec7a2'
OLD_SELECTION_SHA = 'c4f0c7864100218d472a2b0abf474c3ab87e74d19654704fa9f2a9f66d816ea9'


def counts():
    # RK4 counts are batched rollout time steps: no batch12 multiplier,
    # no four-RHS multiplier, and no evaluation or qualification work.
    return {'models': 12, 'updates_per_model': 2048, 'formal_updates': 24576,
            'checkpoints': 36, 'restores': 12, 'dual_step_model_evaluations': 24,
            'optimization_rk4_steps_short': 6 * 2048 * 50,
            'optimization_rk4_steps_long': 6 * 2048 * 200}


def order():
    return [(seed, candidate, future) for seed in SEEDS for candidate in CANDIDATES for future in WINDOWS]


def key(seed, candidate, future): return f'{seed}_candidate{candidate}_future{future}'


def fit_slice(observations, future):
    import numpy as np
    if future not in WINDOWS or observations.shape != (16, 511, 30) or not np.isfinite(observations).all():
        raise ValueError('full16x511x30 observations and registered window required')
    return observations[:, :11 + future]


def checked_inputs(binding):
    """Read frozen syntax and train only. No trajectory generator is invoked."""
    import numpy as np
    old = science.bound(binding, 'old_selection')
    complete = science.bound(binding, 'old_complete'); native = science.bound(binding, 'old_native')
    if binding['old_selection']['sha256'] != OLD_SELECTION_SHA or old.get('source_commit') != OLD_SOURCE or not old.get('all_eighteen_frozen'):
        raise ValueError('original complete eighteen-model selection required')
    if complete.get('source_commit') != OLD_SOURCE or complete.get('status') != 'COMPLETED' or not complete.get('all_eighteen_frozen'):
        raise ValueError('original completion required')
    if complete['selection_manifest']['sha256'] != binding['old_selection']['sha256']:
        raise ValueError('original selection source mismatch')
    if (native.get('source_commit') != OLD_SOURCE or native.get('status') != 'COMPLETED'
        or native.get('phase') != 'bounded_edge_run' or native.get('cleanup_verified') is not True
        or native.get('actual_worker_state') != 'GONE' or native.get('orphans_cleaned') is not False
        or type(native.get('job_active_processes')) is not int or native['job_active_processes'] != 0
        or not native.get('source_before') or native['source_before'] != native.get('source_after')
        or any(type(native.get(k)) is not int or native[k] != 0 for k in ('exit_code', 'worker_exit_code', 'launcher_exit_code'))):
        raise ValueError('original clean native completion required')
    pool = science.bound(binding, 'pool_manifest'); teacher = science.bound(binding, 'teacher_identity')
    if teacher['pool_sha256'] != binding['pool_manifest']['sha256'] or teacher['index'] != 2 or teacher['term'] != pool['selected'][2] or teacher['coefficient'] != .15:
        raise ValueError('original teacher2 identity required')
    if [r['seed'] for r in old['rows']] != list(TRAIN_SEEDS): raise ValueError('old train streams changed')
    loaded = {}
    for seed, old_seed, row in zip(SEEDS, TRAIN_SEEDS, old['rows']):
        ref = binding['train_inputs'][str(old_seed)]
        if science.sha(ROOT / ref['path']) != ref['sha256'] or ref['sha256'] != row['train_observations']['sha256']:
            raise ValueError('original train bytes changed')
        if (ROOT / ref['path']).resolve() != Path(row['train_observations']['path']).resolve():
            raise ValueError('original train location changed')
        qualification_ref = binding['train_qualifications'][str(old_seed)]
        if science.sha(ROOT / qualification_ref['path']) != qualification_ref['sha256']:
            raise ValueError('original teacher qualification bytes changed')
        qualification = json.loads((ROOT / qualification_ref['path']).read_text())
        if qualification != row['teacher_qualification'] or qualification.get('valid') is not True or qualification['count'] != 16:
            raise ValueError('old full teacher qualification required')
        with np.load(ROOT / ref['path'], allow_pickle=False) as arrays:
            if set(arrays.files) != {'coarse', 'fine', 'scales'}: raise ValueError('unexpected train arrays')
            c, f, sc = (arrays[name].copy() for name in ('coarse', 'fine', 'scales'))
        if c.shape != (16, 511, 30) or f.shape != c.shape or sc.shape != (30,) or not all(np.isfinite(a).all() for a in (c, f, sc)) or (sc < 1e-6).any():
            raise ValueError('original train shapes/scales invalid')
        loaded[seed] = {'coarse': c, 'fine': f, 'scales': sc, 'old_seed': old_seed,
                        'ref': ref, 'coarse_sha256': hashlib.sha256(c.tobytes()).hexdigest()}
    return pool, loaded


def identity(commit, seed, candidate, future, pool, data):
    return {'source_commit': commit, 'seed': seed, 'train_seed': data['old_seed'],
            'train_file_sha256': data['ref']['sha256'], 'train_sha256': data['coarse_sha256'],
            'candidate_key': pool['keys'][candidate], 'candidate_index': candidate, 'fit_future_steps': future}


def model_metrics(model, data):
    c, f, sc = (data[n] for n in ('coarse', 'fine', 'scales'))
    mc = science.predict(model, c, .01, None); mf = science.predict(model, c, .005, None)
    return {name: science.score(mc, mf, c[:, 11:], f[:, 11:], sc, a, b)
            for name, a, b in [('one_second', 0, 100), ('five_seconds', 0, 500),
                               ('primary', 200, 500), ('half_to_two_seconds', 50, 200)]}


def require_model_rows(rows):
    if [(r['seed'], r['candidate'], r['future']) for r in rows] != order(): raise ValueError('all12 fixed models required')
    if any(r.get('status') != 'FROZEN' for r in rows): raise ValueError('all12 actual endpoints required')


def validate_selection(result, binding, commit):
    if result.get('source_commit') != commit or result.get('planned_counts') != counts(): raise ValueError('source/count identity changed')
    rows = result['rows']; require_model_rows(rows)
    pool, data = checked_inputs(binding)
    proofs = []
    for row in rows:
        folder = Path(row['path']); latest = checks.validate_layout(folder, list(STEPS))
        expected = identity(commit, row['seed'], row['candidate'], row['future'], pool, data[row['seed']])
        prefix = []; stage_proofs = []
        for step in STEPS:
            # validate_checkpoint/state accept actual integer2048 and contain
            # no legacy restore range guard. The shared restore is not called.
            proof = checks.validate_checkpoint(folder / f'checkpoint_{step:05d}', expected, step,
                                                latest['sha256'] if step == 2048 else None)
            trace = proof.pop('trace')
            if trace[:len(prefix)] != prefix: raise ValueError('loss trace prefix changed')
            prefix = trace
            if proof['files'] != row['checkpoints'][str(step)]: raise ValueError('published checkpoint source changed')
            stage_proofs.append(proof)
        if row.get('actual_updates') != 2048 or row.get('actual_restores') != 1: raise ValueError('actual workload mismatch')
        if set(row['metrics']) != {'512', '2048'}: raise ValueError('missing fixed endpoint metrics')
        proofs.append({'key': key(row['seed'], row['candidate'], row['future']), 'checkpoints': stage_proofs})
    return {'status': 'PASS', 'models': len(proofs), 'actual_updates': 24576,
            'checkpoints': 36, 'restores': 12, 'rows': proofs}


def paired_profile(rows):
    profile = []
    for seed in SEEDS:
        for future in WINDOWS:
            for step in (512, 2048):
                selected = {r['candidate']: r for r in rows if r['seed'] == seed and r['future'] == future}
                scores = {i: selected[i].get('metrics', {}).get(str(step), {}).get('primary') for i in CANDIDATES if i in selected}
                valid = len(scores) == 2 and all(s is not None and s['valid'] for s in scores.values())
                difference = scores[3]['mean'] - scores[2]['mean'] if valid else None
                error = scores[3]['B'] + scores[2]['B'] if valid else None
                profile.append({'seed': seed, 'future': future, 'updates': step, 'valid': valid,
                    'competitor_minus_teacher_mse': difference, 'numerical_error_bound': error,
                    'interval': [difference-error, difference+error] if valid else None,
                    'decision': ('TEACHER' if difference > error else 'COMPETITOR' if difference < -error else 'UNSEPARATED') if valid else 'UNRESOLVED'})
    return profile


def paired_changes(profile):
    """Predeclared arithmetic contrasts, retaining summed numerical bounds.

    These intervals describe numerical error only; they are not sampling CIs.
    """
    lookup = {(r['seed'], r['future'], r['updates']): r for r in profile}
    changes = []
    def contrast(kind, seed, left_key, right_key):
        left, right = lookup[left_key], lookup[right_key]
        valid = left['valid'] and right['valid']
        value = right['competitor_minus_teacher_mse'] - left['competitor_minus_teacher_mse'] if valid else None
        bound = left['numerical_error_bound'] + right['numerical_error_bound'] if valid else None
        return {'comparison': kind, 'seed': seed, 'from': list(left_key[1:]), 'to': list(right_key[1:]),
                'valid': valid, 'relative_teacher_gain_change': value, 'numerical_error_bound': bound,
                'interval': [value-bound, value+bound] if valid else None,
                'direction': ('IMPROVED' if value > bound else 'WORSENED' if value < -bound else 'UNSEPARATED') if valid else 'UNRESOLVED'}
    for seed in SEEDS:
        for future in WINDOWS:
            changes.append(contrast('fixed_window_512_to2048', seed, (seed,future,512), (seed,future,2048)))
        for step in (512,2048):
            changes.append(contrast('fixed_endpoint_short_to_long', seed, (seed,50,step), (seed,200,step)))
    return {'contrasts': changes,
            'long2048_teacher_separated_all_three': all(lookup[(seed,200,2048)]['decision']=='TEACHER' for seed in SEEDS)}


def run_study(output, binding, commit):
    import numpy as np
    import torch
    from closure_causal_history_residual_20261019 import checkpoint, restore, train_update
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    pool, inputs = checked_inputs(binding); rows = []
    for seed, candidate, future in order():
        folder = output / key(seed, candidate, future); folder.mkdir()
        data = inputs[seed]; ident = identity(commit, seed, candidate, future, pool, data)
        row = {'seed': seed, 'candidate': candidate, 'future': future, 'path': str(folder),
               'status': 'UNRESOLVED', 'actual_updates': 0, 'attempted_updates': 0, 'actual_restores': 0, 'checkpoints': {}, 'metrics': {}}
        trace = []; step = 0
        try:
            model, opt = science.make(pool['selected'][candidate], seed, data['coarse'])
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
            observed = torch.as_tensor(fit_slice(data['coarse'], future), device='cuda')
            scales = torch.as_tensor(data['scales'], device='cuda')
            checkpoint(folder, model, opt, rng, 0, trace, ident)
            row['checkpoints']['0'] = {n: science.sha(folder / 'checkpoint_00000' / n) for n in ('model.json', 'training.pt')}
            for target in (512, 2048):
                while step < target:
                    row['attempted_updates'] += 1
                    value, _ = train_update(model, opt, rng, observed, scales, None)
                    trace.append(value); step += 1; row['actual_updates'] = step
                checkpoint(folder, model, opt, rng, step, trace, ident)
                row['checkpoints'][str(step)] = {n: science.sha(folder / f'checkpoint_{step:05d}' / n) for n in ('model.json', 'training.pt')}
                try: metrics = model_metrics(model, data)
                except (RuntimeError, FloatingPointError, ValueError) as error:
                    metrics = {'error': repr(error), 'category': 'numerical_or_implementation_failure'}
                row['metrics'][str(step)] = metrics
                science.atomic(output / f'metrics_{key(seed,candidate,future)}_{step}.json', metrics)
                if target == 512:
                    model, opt = science.make(pool['selected'][candidate], seed, data['coarse'])
                    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
                    step, trace = restore(folder, model, opt, rng, ident)
                    if step != 512: raise ValueError('actual512 restore required')
                    row['actual_restores'] += 1
            row['status'] = 'FROZEN'
        except (RuntimeError, FloatingPointError, ValueError) as error:
            row.update(error=repr(error), traceback=traceback.format_exc(), trace=trace)
            science.atomic(output / f'failure_{key(seed,candidate,future)}.json', row)
        rows.append(row); science.atomic(output / 'progress.json', {'source_commit': commit, 'rows': rows})
    result = {'status': 'COMPLETED', 'source_commit': commit, 'planned_counts': counts(), 'rows': rows,
              'all_twelve_frozen': all(r['status'] == 'FROZEN' for r in rows), 'paired_profile': paired_profile(rows),
              'actual_updates': sum(r['actual_updates'] for r in rows), 'actual_restores': sum(r['actual_restores'] for r in rows),
              'project_val_accessed': False, 'project_test_accessed': False, 'synthetic_val_accessed': False,
              'freeze_scope': 'training state only; numerical validity is reported per metrics/pair and may remain UNRESOLVED',
              'optimization_step_convention': 'batched RK4 time steps; excludes batch12 multiplier, four RHS calls per step, evaluation and QA'}
    result['paired_changes'] = paired_changes(result['paired_profile'])
    if result['all_twelve_frozen']:
        try: result['freeze_validation'] = validate_selection(result, binding, commit)
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            result.update(all_twelve_frozen=False, freeze_error=repr(error))
    science.atomic(output / 'selection.json', result)
    return result
