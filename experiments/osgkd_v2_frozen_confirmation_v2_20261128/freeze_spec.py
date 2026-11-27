"""Versioned extension of the locked IEEE39 holdout comparison inventory."""

from collections import Counter
from pathlib import Path

from experiments.osgkd_v2_frozen_confirmation_20261127 import freeze_spec as v1
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io


ROOT = Path(__file__).resolve().parents[2]
RUN = 'osgkd_v2_frozen_confirmation_v2_20261128'
PUBLIC = ROOT / 'results' / RUN
LOCAL = ROOT / 'artifacts' / RUN
EVALUATION_ID = 'ieee39-independent-expansion45-supplement41-20261128-v2'
PROTOCOL = 'campaigns/osgkd_v2/wp3_frozen_confirmation_v2_protocol.md'
FLAT = 'osgkd_v2_wp3_2_flat_seeds_20261128'
V1_SHA = 'e4895159fbb7de1769ab6cb27c8a7328258b1dbdd511904ef57976e3a58cb892'
SEEDS = v1.SEEDS
REFITS = ('full', 'no_coupling', 'no_memory', 'no_beta', 'frozen_p', 'top30', 'top20')
ROLE_COUNTS = {'primary': 1, 'matched_open_model': 1, 'matched_encoder': 3, 'paired_refit': 18,
               'registered_parent': 23, 'matched_static_mask': 6}
READ_CLAIM = ROOT / 'artifacts' / 'osgkd_ieee39_expansion45_supplement41_exposure.json'


def prior_v1():
    path = v1.PUBLIC / 'freeze.json'
    ref, book = v1.checked_json(path)
    io.require(ref['sha256'] == V1_SHA and book['status'] == 'frozen'
               and book['user_test_authorized'] is False and book['test_trajectory_access'] is False,
               'frozen v1 identity differs')
    private = ROOT / 'artifacts' / v1.RUN
    io.require(not any(p.exists() for p in (v1.PUBLIC/'authorized_release.json', v1.PUBLIC/'summary.json',
                                          private/'test_access_receipt.json', private/'data_complete.json')),
               'v1 approval or data exposure cannot be carried into v2')
    return ref, book


def supplemental():
    ref, run = v1.checked_json(ROOT/'harness'/'runs'/f'{FLAT}.json')
    io.require(run['execution']['status'] == 'COMPLETED' and run['scientific_work']['budget_status'] == 'closed',
               'flat seed supplement must be independently closed')
    audit = run['evidence']['final_independent_review']
    io.require(audit['status'] == 'passed', 'flat supplement independent review required')
    io.verified(audit['reference'])
    eval_ref = run['execution']['completed_jobs']['evaluate01']['summary']
    io.require(eval_ref['path'] == f'results/{FLAT}/evaluate01/summary.json', 'flat supplemental score path differs')
    scored = io.read(io.verified(eval_ref))
    io.require(scored['run_id'] == FLAT and scored['status'] == 'completed' and scored['six_endpoints_qualified']
               and scored['original_val_passes'] == 1 and scored['test_trajectory_access'] is False
               and scored['new_fom'] == 0 and scored['default_model'] == 'none', 'flat score qualification differs')
    training = io.read(io.verified(scored['training_summary']))
    train_release = io.read(io.verified(training['fit_release']))
    io.require(train_release['batch_plan_sha256'] ==
               'b11463b14f01ed1c26f60b6227f7ce0072a90d5f465e61593fbc056878e8ae55',
               'flat batch authority differs')
    return dict(run_id=FLAT, registry=ref, review=audit['reference']), eval_ref, scored, training, train_release


def variants(seed):
    return REFITS + ('static',) if seed != 24121 else REFITS


def fixed_endpoints():
    old, alias = v1.fixed_endpoints()
    orig = {row['id']: row for row in old}
    _, flat_ref, flat, training, _ = supplemental()
    for seed in (24121, 24131):
        name = f'flat_{seed}'
        row = flat['val'][name]
        summary = io.read(io.verified(training['jobs'][name]['summary_ref']))
        r = summary['result']
        io.require(row['status'] == 'scored' and row['seed'] == seed and r['valid_endpoint']
                   and r['attempts'] == 2048 and r['accepted'] >= 2033
                   and row['checkpoint_ref'] == r['checkpoints']['2048'], 'new flat endpoint differs: ' + name)
        orig[name] = dict(id=name, kind='flat', attempt=2048, checkpoint=row['checkpoint_ref'], source=flat_ref)
    ordered = [f'{kind}_{seed}' for kind in ('graph', 'flat') for seed in SEEDS]
    result = [orig[name] for name in (*ordered, 'neural_ode_24111', 'closed_passive_24111')]
    refit_ref, refit = v1.checked_json(ROOT/'results'/v1.WP1A/'evaluate01/summary.json')
    for seed in SEEDS:
        for variant in variants(seed):
            name = f'{variant}_{seed}'
            row = refit['val'][name]
            io.require(row['status'] == 'scored' and row['qualified_events'] == 107, 'refit eligibility differs')
            new = dict(id=name, kind='restricted', attempt=128, checkpoint=row['checkpoint_ref'], source=refit_ref)
            io.require(name not in orig or new == orig[name], 'original frozen refit identity differs: ' + name)
            result.append(new)
    io.require(len(result) == len({x['id'] for x in result}) == len({x['checkpoint']['sha256'] for x in result}) == 31,
               '31 distinct frozen endpoints required')
    return result, alias


def fixed_comparisons():
    result = [dict(reference='neural_ode_24111', candidate='graph_24111', role='primary'),
              dict(reference='closed_passive_24111', candidate='graph_24111', role='matched_open_model')]
    result += [dict(reference=f'flat_{s}', candidate=f'graph_{s}', role='matched_encoder') for s in SEEDS]
    result += [dict(reference=f'full_{s}', candidate=f'{kind}_{s}', role='paired_refit')
               for s in SEEDS for kind in REFITS[1:]]
    result += [dict(reference=f'graph_{s}', candidate=f'{kind}_{s}', role='registered_parent')
               for s in SEEDS for kind in variants(s)]
    result += [dict(reference=f"{'full' if s == 24121 else 'static'}_{s}", candidate=f'{kind}_{s}', role='matched_static_mask')
               for s in SEEDS for kind in ('top30', 'top20')]
    io.require(len(result) == 52 and Counter(x['role'] for x in result) == ROLE_COUNTS
               and len({(x['reference'], x['candidate'], x['role']) for x in result}) == 52,
               'frozen comparison counts differ')
    return result


def closed_prerequisites():
    extra, *_ = supplemental()
    return [*v1.closed_prerequisites(), extra]


def evidence_sources():
    _, historic = prior_v1()
    *_, latest = supplemental()
    return set(historic['source_hashes']) | set(latest['source_hashes']) | {
        PROTOCOL, f'experiments/{RUN}/freeze_spec.py', f'experiments/{RUN}/run.py',
    }
