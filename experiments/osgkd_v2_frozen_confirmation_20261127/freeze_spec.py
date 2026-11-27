"""Prospective identities for the one-time frozen confirmation; no trajectory I/O."""

from pathlib import Path

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io


ROOT = Path(__file__).resolve().parents[2]
RUN = 'osgkd_v2_frozen_confirmation_20261127'
PUBLIC = ROOT/'results'/RUN
SEEDS = (24111, 24121, 24131)
REFITS = ('full', 'no_coupling', 'no_memory', 'top30', 'top20')
SKIP_PILOT_SHA = '02990d0ddb22e4153824a28f1e99fc0d22a0afb70a4d197e6a294e6da253dba8'
PARENT = 'osgkd_v2_sprint2a_20261125'
BASELINES = 'osgkd_v2_sprint2b_baselines_20261126'
WP1A = 'osgkd_v2_wp1a_refits_20261127'
PREREQUISITES = ('osgkd_v2_wp0_20261127', WP1A, 'osgkd_v2_wp2_layer1_20261127',
                 'osgkd_v2_wp3_pendulum_20261127')
CONSTANTS = 'results/osgkd_fit297_20261113/port_audit.json'
WP0_REVIEW_SHA256 = '9281bd5c2651049e65238970f0d7f4c4a41b0771cc7b125bb200eeb44dd0b072'


def checked_json(path):
    ref = io.identity(path)
    return ref, io.read(io.verified(ref))


def fixed_endpoints():
    parent_ref, parent = checked_json(ROOT/'results'/PARENT/'evaluate01/summary.json')
    baseline_ref, base = checked_json(ROOT/'results'/BASELINES/'evaluate01/summary.json')
    refit_ref, re = checked_json(ROOT/'results'/WP1A/'evaluate01/summary.json')
    io.require(all(s['status']=='completed' for s in (parent, base, re)) and
               base['parent_evaluation']==parent_ref, 'frozen original comparison lineage differs')
    result = []
    original_models = [(f'graph_{seed}', 'graph') for seed in SEEDS] + [('flat_24111','flat')]
    for name, kind in original_models:
        row = parent['result']['val'][name]
        result.append(dict(id=name,kind=kind,attempt=2048,checkpoint=row['checkpoint_ref'],source=parent_ref))
    for name,kind in (('neural_ode_24111','neural_ode'),('closed_passive_24111','closed_passive')):
        row = base['val'][name]
        result.append(dict(id=name,kind=kind,attempt=2048,checkpoint=row['checkpoint_ref'],source=baseline_ref))
    for seed in SEEDS:
        variants = REFITS+('static',) if seed!=24121 else REFITS
        for variant in variants:
            name = f'{variant}_{seed}'
            row = re['val'][name]
            io.require(row['status']=='scored' and row['qualified_events']==107, 'refit terminal differs')
            result.append(dict(id=name,kind='restricted',attempt=128,checkpoint=row['checkpoint_ref'],source=refit_ref))
    io.require(len(result)==len({x['id'] for x in result})==len({x['checkpoint']['sha256'] for x in result})==23,
               '23 unique fixed endpoints required')
    duplicate_ref, duplicate = checked_json(ROOT/'results'/WP1A/'duplicate_24121.json')
    io.require(duplicate['same_restrictions'] and duplicate['matched_attempts']==128 and
               duplicate['matched_model_optimizer_rng_at']==[0,64,128], 'static alias proof differs')
    return result, dict(alias='static_24121', endpoint='full_24121', proof=duplicate_ref)


def fixed_comparisons():
    pairs = [dict(reference='neural_ode_24111', candidate='graph_24111', role='primary'),
             dict(reference='flat_24111', candidate='graph_24111', role='matched_encoder'),
             dict(reference='closed_passive_24111', candidate='graph_24111', role='matched_open_model')]
    for seed in SEEDS:
        for kind in ('no_coupling','no_memory','top30','top20'):
            pairs.append(dict(reference=f'full_{seed}', candidate=f'{kind}_{seed}', role='paired_refit'))
        variants = REFITS+('static',) if seed!=24121 else REFITS
        for kind in variants:
            pairs.append(dict(reference=f'graph_{seed}',candidate=f'{kind}_{seed}',role='registered_parent'))
        for kind in ('top30','top20'):
            anchor = f'full_{seed}' if seed==24121 else f'static_{seed}'
            pairs.append(dict(reference=anchor, candidate=f'{kind}_{seed}', role='matched_static_mask'))
    return pairs


def evidence_sources():
    reference = io.read(ROOT/'results'/BASELINES/'evaluate01/summary.json')['release_ref']
    paths = set(io.read(ROOT/'results'/WP1A/'release.json')['source_hashes'])
    paths.update(io.read(io.verified(reference))['source_hashes'])
    return set(paths) | {
        'campaigns/osgkd_v2/wp3_fresh_holdout_identity.md', 'campaigns/osgkd_v2/wp3_frozen_confirmation_protocol.md',
        *('experiments/'+RUN+'/'+name for name in ('freeze_spec.py','data.py','run.py','score.py')),
        'src/kiki/evaluation/osgkd_autonomous.py', CONSTANTS,
    }


def closed_prerequisites():
    refs = []
    for name in PREREQUISITES:
        ref, row = checked_json(ROOT/'harness/runs'/f'{name}.json')
        io.require(row['execution']['status']=='COMPLETED' and row['scientific_work']['budget_status']=='closed',
                   'previous stage not closed: '+name)
        audit = row['evidence']['final_independent_review']
        review = audit['reference']
        io.verified(review)
        accepted = (review['sha256']==WP0_REVIEW_SHA256 if name==PREREQUISITES[0] else audit['status']=='passed')
        io.require(accepted, 'independent evidence review did not pass: '+name)
        refs.append(dict(run_id=name, registry=ref, review=review))
    return refs
