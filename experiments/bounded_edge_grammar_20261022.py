"""Finite q-linear edge grammar comparison; new synthetic streams, no quotas.

No external resume. Every candidate receives the same registered 512 updates;
selection sees train only. Numerical failures are retained and prevent val.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
import finite_grammar_discovery_20261021 as science
import finite_grammar_execution_20261022 as checks

SEEDS = (6201, 6202, 6203)
SCIENCE_MODULE = science.__name__
STEPS = (0, 32, 128, 512)


def polynomial(expression):
    """Exact expansion; no opaque function atoms and no forbidden variables."""
    op = expression['op']
    if op == 'var':
        if expression['name'] not in {'qi', 'qj', 's', 'c'}:
            raise ValueError('only qi,qj,s,c allowed')
        return {(expression['name'],): Fraction(1)}
    if op == 'const':
        value = Fraction(str(expression['value']))
        return {(): value} if value else {}
    def add(left, right):
        result = dict(left)
        for key, value in right.items(): result[key] = result.get(key, Fraction(0)) + value
        return {key: value for key, value in result.items() if value}
    def multiply(left, right):
        result = {}
        for a, u in left.items():
            for b, v in right.items():
                key = tuple(sorted(a + b)); result[key] = result.get(key, Fraction(0)) + u * v
        return {key: value for key, value in result.items() if value}
    if op in {'add', 'mul'}:
        result = {} if op == 'add' else {(): Fraction(1)}
        for child in expression['args']:
            result = (add if op == 'add' else multiply)(result, polynomial(child))
        return result
    if op in {'pow2', 'pow3', 'pow4'}:
        result = {(): Fraction(1)}; base = polynomial(expression['arg'])
        for _ in range(int(op[-1])): result = multiply(result, base)
        return result
    raise ValueError('finite polynomial operators only')


def centered_edge_polynomial(expression):
    """Exact endpoint symmetry and physical-origin mechanism centering.

    The c feature has physical angular curvature, so its linear feature term
    is retained just as in the production mechanism signature.
    """
    raw = polynomial(expression); combined = {}
    swap = {'qi': 'qj', 'qj': 'qi'}
    for monomial, coefficient in raw.items():
        reverse = tuple(sorted(swap.get(v, v) for v in monomial))
        combined[monomial] = combined.get(monomial, Fraction(0)) + coefficient / 2
        combined[reverse] = combined.get(reverse, Fraction(0)) + coefficient * (-1 if monomial.count('s') % 2 else 1) / 2
    return {k: v for k, v in combined.items() if v and k and (len(k) > 1 or k == ('c',))}


def eligible(term):
    if term.get('channel') != 'edge': return False
    try: expanded = polynomial(term['expression'])
    except (ValueError, KeyError, TypeError, ZeroDivisionError): return False
    if not expanded or max(sum(v in {'qi', 'qj'} for v in monomial) for monomial in expanded) != 1:
        return False
    centered = centered_edge_polynomial(term['expression'])
    return bool(centered) and max(sum(v in {'qi', 'qj'} for v in monomial) for monomial in centered) == 1 and any(
        's' in monomial or 'c' in monomial for monomial in centered)


def generate_pool():
    from closure_discovery_benchmarks import pool
    from closure_discovery_support import term_key
    from closure_expression_identity import mechanism_signature
    order = lambda term: (hashlib.sha256(term_key(term).encode()).hexdigest(), term_key(term))
    unique = {}
    for term in sorted(pool(), key=order):
        if eligible(term):
            signature = mechanism_signature(term)
            if signature is not None: unique.setdefault(signature, term)
    terms = sorted(unique.values(), key=order)
    return {'eligible_pool': terms, 'selected': terms[:6], 'keys': [term_key(t) for t in terms[:6]],
            'generation_rule': 'edge; exact q total degree1 raw and centered endpoint symmetry; no p; finite qi,qj,s,c polynomial; nonzero signature dedup; SHA256(term_key) first6'}


def freeze(output):
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    pool = generate_pool(); science.atomic(output / 'pool.json', pool)
    pool_ref = {'path': str(output / 'pool.json'), 'sha256': science.sha(output / 'pool.json')}
    if len(pool['selected']) != 6:
        result = {'status': 'INSUFFICIENT_POOL', 'pool_manifest': pool_ref, 'eligible_count': len(pool['eligible_pool'])}
    else:
        index = int(np.random.default_rng(np.random.SeedSequence([20261022, 6200])).integers(6))
        teacher = {'index': index, 'term': pool['selected'][index], 'key': pool['keys'][index],
                   'coefficient': .15, 'pool_sha256': pool_ref['sha256']}
        science.atomic(output / 'teacher.json', teacher)
        result = {'status': 'FROZEN', 'pool_manifest': pool_ref,
                  'teacher_identity': {'path': str(output / 'teacher.json'), 'sha256': science.sha(output / 'teacher.json')}}
    science.atomic(output / 'result.json', result)
    return result


def read_pool(binding):
    frozen = science.bound(binding, 'pool_manifest')
    if frozen != generate_pool() or len(frozen['selected']) != 6:
        raise ValueError('frozen production subgrammar differs or insufficient')
    return frozen


def teacher(binding):
    import numpy as np
    pool = read_pool(binding); actual = science.bound(binding, 'teacher_identity')
    index = int(np.random.default_rng(np.random.SeedSequence([20261022, 6200])).integers(6))
    expected = {'index': index, 'term': pool['selected'][index], 'key': pool['keys'][index],
                'coefficient': .15, 'pool_sha256': binding['pool_manifest']['sha256']}
    if actual != expected: raise ValueError('teacher identity changed')
    return actual


def fixture(term, seed, stream):
    import numpy as np
    import torch
    from closure_discovery_benchmarks import toy_source
    from closure_causal_recovery_20261019 import copy_runtime, trajectory
    if seed not in SEEDS or stream not in (0, 1): raise ValueError('registered streams only')
    model = copy_runtime(toy_source(), [term], 'cuda')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261022, 6200]).spawn(2)[stream])
    initial = rng.normal(size=(16, 40)) * np.where(np.arange(16)[:, None] < 8, .08, .3)
    initial[:, 30:] = rng.normal(size=(16, 10)) * np.where(np.arange(16)[:, None] < 8, .2, 1.2)
    state = torch.as_tensor(initial, device='cuda', dtype=torch.float64)
    with torch.no_grad():
        model.coefficients.fill_(.15)
        coarse = trajectory(model, state, 510, .01, None).cpu().numpy()
        fine = trajectory(model, state, 1020, .005, None).cpu().numpy()[:, ::2]
    return coarse, fine, np.maximum(coarse[:, :61].std((0, 1)), 1e-6)


def counts():
    return {'updates_per_seed': 3072, 'total_updates': 9216, 'total_endpoints': 18,
            'total_checkpoints': 72, 'total_restores': 36, 'total_rankings': 54}


def select_seed(terms, keys, seed, coarse, fine, sc, output, commit):
    """No teacher identity, hidden truth or val may enter this function."""
    import numpy as np
    import torch
    from closure_causal_history_residual_20261019 import checkpoint, restore, train_update
    output = Path(output)
    observed = torch.as_tensor(coarse[:, :61], device='cuda'); scale = torch.as_tensor(sc, device='cuda')
    base_identity = {'source_commit': commit, 'seed': seed, 'train_sha256': hashlib.sha256(coarse.tobytes()).hexdigest()}
    stages = {target: [] for target in STEPS[1:]}
    for index, term in enumerate(terms):
        folder = output / f'candidate_{index}'; folder.mkdir()
        identity = {**base_identity, 'candidate_key': keys[index]}
        trace = []; step = 0
        try:
            model, optimizer = science.make(term, seed, coarse)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
            checkpoint(folder, model, optimizer, rng, 0, trace, identity)
            for target in STEPS[1:]:
                while step < target:
                    loss, _ = train_update(model, optimizer, rng, observed, scale, None)
                    trace.append(loss); step += 1
                checkpoint(folder, model, optimizer, rng, step, trace, identity)
                try: score = science.rank_model(model, coarse, fine, sc, None)
                except (RuntimeError, FloatingPointError, ValueError) as error:
                    score = {'valid': False, 'count': 16, 'mean': None, 'B': None, 'error': repr(error), 'category': 'numerical_or_implementation_failure'}
                row = {'index': index, 'key': keys[index], 'score': score, 'path': str(folder / f'checkpoint_{step:05d}'),
                       'files': {n: science.sha(folder / f'checkpoint_{step:05d}' / n) for n in ('model.json', 'training.pt')}}
                stages[target].append(row)
                science.atomic(output / f'ranking_candidate_{index}_{target}.json', row)
                if target != 512:
                    # Actually reconstruct and restore the registered state, not
                    # a fake resume marker or an extra training trajectory.
                    model, optimizer = science.make(term, seed, coarse)
                    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
                    step, trace = restore(folder, model, optimizer, rng, identity)
                    if step != target: raise ValueError('internal restore count mismatch')
        except (RuntimeError, FloatingPointError, ValueError) as error:
            science.atomic(output / f'failure_candidate_{index}.json', {'index': index, 'actual_updates': step, 'trace': trace,
                'error': repr(error), 'traceback': traceback.format_exc(), 'status': 'UNRESOLVED'})
    history = []
    for target in STEPS[1:]:
        chosen, ambiguous = science.choose(copy.deepcopy(stages[target]), 1)
        stage = {'updates': target, 'rows': stages[target], 'ambiguous_boundary': ambiguous}
        science.atomic(output / f'ranking_{target}.json', stage); history.append(stage)
    chosen, _ = science.choose(copy.deepcopy(stages[512]), 1)
    complete = all(len(stages[target]) == 6 for target in STEPS[1:])
    return {'seed': seed, 'status': 'FROZEN' if complete and chosen else 'UNRESOLVED',
            'selected': chosen[0] if complete and chosen else None, 'history': history}


def _validate_row(row, pool, commit):
    import numpy as np
    if row.get('status') != 'FROZEN': raise ValueError('all18 actual endpoints required')
    raw = row['train_observations']; folder = Path(raw['path']).parent
    if science.sha(raw['path']) != raw['sha256']: raise ValueError('train bytes changed')
    with np.load(raw['path'], allow_pickle=False) as arrays:
        if set(arrays.files) != {'coarse', 'fine', 'scales'}: raise ValueError('train arrays changed')
        coarse, fine, scales = (arrays[k] for k in ('coarse', 'fine', 'scales'))
        qualification = science.qualify_teacher(coarse, fine, scales)
        if not qualification['valid'] or qualification != row['teacher_qualification']: raise ValueError('teacher qualification invalid')
        if not np.array_equal(scales, np.maximum(coarse[:, :61].std((0, 1)), 1e-6)): raise ValueError('train scales changed')
        train_sha = hashlib.sha256(coarse.tobytes()).hexdigest()
    history = row['history']
    if [s['updates'] for s in history] != list(STEPS[1:]): raise ValueError('missing stage')
    for stage in history:
        if [r['index'] for r in stage['rows']] != list(range(6)): raise ValueError('all six candidates required at each stage')
        if stage != json.loads((folder / f"ranking_{stage['updates']}.json").read_text()): raise ValueError('ranking changed')
        _, ambiguous = science.choose(copy.deepcopy(stage['rows']), 1)
        if ambiguous != stage['ambiguous_boundary']: raise ValueError('boundary changed')
    chosen, _ = science.choose(copy.deepcopy(history[-1]['rows']), 1)
    if not chosen or row['selected'] != chosen[0]: raise ValueError('final train-only selection changed')
    if {p.name for p in folder.glob('candidate_*')} != {f'candidate_{i}' for i in range(6)}: raise ValueError('unknown candidates')
    proofs = []
    for index in range(6):
        candidate = folder / f'candidate_{index}'; latest = checks.validate_layout(candidate, list(STEPS))
        identity = {'source_commit': commit, 'seed': row['seed'], 'train_sha256': train_sha, 'candidate_key': pool['keys'][index]}
        prefix = []
        for step in STEPS:
            proof = checks.validate_checkpoint(candidate / f'checkpoint_{step:05d}', identity, step, latest['sha256'] if step == 512 else None)
            trace = proof.pop('trace')
            if trace[:len(prefix)] != prefix: raise ValueError('actual loss trace continuation changed')
            prefix = trace
            if step:
                record = next(stage for stage in history if stage['updates'] == step)['rows'][index]
                if record['key'] != pool['keys'][index] or Path(record['path']).resolve() != (candidate / f'checkpoint_{step:05d}').resolve() or record['files'] != proof['files']:
                    raise ValueError('ranked checkpoint source differs')
                if step == 512 and not record['score']['valid']:
                    raise ValueError('final candidate numerical qualification failed')
            proofs.append({'candidate': index, **proof})
    return {'seed': row['seed'], 'checkpoints': proofs}


def require_all_candidate_layouts(rows):
    """Reject a missing candidate before deserializing any optimizer state.

    This is only a layout barrier; actual state counts are checked afterwards.
    """
    for row in rows:
        if row.get('status') != 'FROZEN': raise ValueError('all18 actual endpoints required')
        folder = Path(row['train_observations']['path']).parent
        if {p.name for p in folder.glob('candidate_*')} != {f'candidate_{i}' for i in range(6)}:
            raise ValueError('all18 candidate layouts required')
        for index in range(6): checks.validate_layout(folder / f'candidate_{index}', list(STEPS))


def validate_selection(selection, binding, commit):
    pool = read_pool(binding)
    if selection.get('source_commit') != commit or selection.get('selection') != 'train_only' or not selection.get('all_eighteen_frozen'):
        raise ValueError('all18 same-source endpoints required before val')
    if [r['seed'] for r in selection['rows']] != list(SEEDS): raise ValueError('three registered seeds required')
    require_all_candidate_layouts(selection['rows'])
    proofs = [_validate_row(row, pool, commit) for row in selection['rows']]
    actual = {'successful_updates': sum(p['updates'] for r in proofs for p in r['checkpoints'] if p['updates'] == 512),
              'checkpoint_directories': sum(len(r['checkpoints']) for r in proofs),
              'rankings': sum(len(s['rows']) for r in selection['rows'] for s in r['history'])}
    if actual != {'successful_updates': 9216, 'checkpoint_directories': 72, 'rankings': 54} or actual != selection['recorded_completed_counts'] or selection.get('planned_counts') != counts():
        raise ValueError('actual workload incomplete')
    return {'status': 'PASS', 'counts': actual, 'rows': proofs}


def run_study(output, binding, commit):
    import numpy as np
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    pool = read_pool(binding); truth = teacher(binding); rows = []
    for seed in SEEDS:
        folder = output / str(seed); folder.mkdir()
        row = {'seed': seed, 'status': 'UNRESOLVED', 'selected': None, 'history': []}
        try:
            c, f, sc = fixture(truth['term'], seed, 0)
            np.savez(folder / 'train_observations.npz', coarse=c, fine=f, scales=sc)
            row['train_observations'] = {'path': str(folder / 'train_observations.npz'), 'sha256': science.sha(folder / 'train_observations.npz')}
            q = science.qualify_teacher(c, f, sc); row['teacher_qualification'] = q
            science.atomic(folder / 'teacher_qualification.json', q)
            if q['valid']: row.update(select_seed(pool['selected'], pool['keys'], seed, c, f, sc, folder, commit))
        except (RuntimeError, FloatingPointError, ValueError) as error:
            row.update(status='UNRESOLVED', selected=None, error=repr(error), traceback=traceback.format_exc())
            science.atomic(folder / 'failure.json', row)
        rows.append(row); science.atomic(output / 'progress.json', {'rows': rows, 'source_commit': commit})
    updates = directories = 0
    for folder in output.glob('*/candidate_*'):
        latest = json.loads((folder / 'latest.json').read_text()) if (folder / 'latest.json').exists() else {'updates': 0}
        count = latest['updates']; failure = folder.parent / f'failure_{folder.name}.json'
        if failure.exists(): count = max(count, json.loads(failure.read_text())['actual_updates'])
        updates += count; directories += len(list(folder.glob('checkpoint_*')))
    result = {'status': 'COMPLETED', 'source_commit': commit, 'selection': 'train_only', 'rows': rows, 'planned_counts': counts(),
              'all_eighteen_frozen': len(rows) == 3 and all(r['status'] == 'FROZEN' for r in rows),
              'recorded_completed_counts': {'successful_updates': updates, 'checkpoint_directories': directories,
                 'rankings': sum(len(s['rows']) for r in rows for s in r['history'])}}
    if result['all_eighteen_frozen']:
        try: science.atomic(output / 'freeze_validation.json', validate_selection(result, binding, commit))
        except (RuntimeError, ValueError, KeyError, OSError) as error:
            result.update(all_eighteen_frozen=False, freeze_error=repr(error))
    science.atomic(output / 'selection.json', result)
    return result


def load_for_evaluation(path):
    """Count the export on CPU before moving the model used for prediction.

    The inherited complexity counter contains NumPy conversions, so it must
    never receive CUDA tensors. Loading here does not perform any update.
    """
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    model = CausalHistoryResidual.from_export(path)
    complexity = model.base.complexity_counts()
    return model.cuda(), complexity


def evaluate_study(output, binding, commit):
    import numpy as np
    from closure_expression_identity import mechanism_signature
    frozen = science.bound(binding, 'selection_manifest'); before = validate_selection(frozen, binding, commit)
    truth = teacher(binding); pool = read_pool(binding)
    output = Path(output); output.mkdir(parents=True, exist_ok=False); rows = []
    for row in frozen['rows']:
        folder = output / str(row['seed']); folder.mkdir(); record = {'seed': row['seed'], 'valid': False}
        try:
            with np.load(row['train_observations']['path'], allow_pickle=False) as train:
                tc, tf, sc = (train[k] for k in ('coarse', 'fine', 'scales'))
                model, complexity = load_for_evaluation(Path(row['selected']['path']) / 'model.json')
                record['selected_expression'] = pool['selected'][row['selected']['index']]
                record['selected_complexity'] = complexity
                train_mc = science.predict(model, tc, .01, None); train_mf = science.predict(model, tc, .005, None)
                record['train_windows'] = {name: science.score(train_mc, train_mf, tc[:, 11:], tf[:, 11:], sc, a, b)
                    for name, a, b in [('one_second', 0, 100), ('five_seconds', 0, 500), ('primary', 200, 500)]}
            c, f, _ = fixture(truth['term'], row['seed'], 1)
            np.savez(folder / 'val_observations.npz', coarse=c, fine=f)
            qualification = science.qualify_teacher(c, f, sc); record['teacher_qualification'] = qualification
            if qualification['valid']:
                mc = science.predict(model, c, .01, None); mf = science.predict(model, c, .005, None)
                np.savez(folder / 'predictions.npz', coarse=mc, fine=mf)
                record['windows'] = {name: science.score(mc, mf, c[:, 11:], f[:, 11:], sc, a, b)
                    for name, a, b in [('one_second', 0, 100), ('five_seconds', 0, 500), ('primary', 200, 500)]}
                record['valid'] = all(w['valid'] for w in record['windows'].values())
            record['signature_match'] = mechanism_signature(pool['selected'][row['selected']['index']]) == mechanism_signature(truth['term'])
            final_order = science.choose(copy.deepcopy(row['history'][-1]['rows']), 6)[0]
            record['teacher_final_position'] = next(i + 1 for i, r in enumerate(final_order) if r['key'] == truth['key'])
            record['teacher_final_interval_group'] = next(r for r in final_order if r['key'] == truth['key'])['overlap_group']
        except (RuntimeError, FloatingPointError, ValueError) as error:
            record.update(valid=False, error=repr(error), category='teacher_or_model_failure')
        record['files'] = {n: {'path': str(folder / n), 'sha256': science.sha(folder / n)} for n in ('val_observations.npz', 'predictions.npz') if (folder / n).is_file()}
        science.atomic(folder / 'result.json', record); rows.append(record)
    after = validate_selection(frozen, binding, commit)
    if after != before or science.bound(binding, 'selection_manifest') != frozen: raise ValueError('frozen sources changed during evaluation')
    result = {'status': 'COMPLETED', 'source_commit': commit, 'rows': rows, 'freeze_validation': after,
        'subgrammar_consistent': all(r['valid'] and r.get('signature_match', False) for r in rows) and not any(r['history'][-1]['ambiguous_boundary'] for r in frozen['rows']),
        'project_val_accessed': False, 'project_test_accessed': False}
    science.atomic(output / 'result.json', result); return result
