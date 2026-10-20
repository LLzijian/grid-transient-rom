"""One pure-stdlib, in-memory QA invocation; never reads study results."""
import copy
import hashlib
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'private_data/closure_mechanism_20261019/P2/quadratic_memory_edge01/extract_fixed_results.py'
m = runpy.run_path(str(SOURCE))


def rejected(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError('expected rejection')


def event(eid, label, n=1, rmse=2.):
    pair = {'finite_to_endpoint': True, 'los_pair_pass': True,
            'coi_normalized_rmse': .0001, 'common_normalized_rmse': .0001,
            'fine_classification': {'category': 'window_no_los_unsettled', 'predicted_los': False}}
    physical = {'available': True, 'frequency': {'finite': True, 'n_scalar_points': n, 'physical_sse': rmse*rmse*n},
                'mechanical_power': {'finite': True, 'n_scalar_points': n*10, 'physical_sse': .01*n*10}}
    return {'event_id': eid, 'label': label, 'group': 'synthetic_bus', 'scientific_numeric_8s': True,
            'classification_8s': copy.deepcopy(pair), 'classification_15s': copy.deepcopy(pair),
            'actual_interval15': {'available': True, 'finite': True, 'numerically_admissible': True,
                'coi_normalized_rmse': .0001, 'common_normalized_rmse': .0001, 'n_time_points': n,
                'mean_point_error': rmse, 'frequency_rmse_rad_s': rmse, 'power_rmse_system_pu': .1},
            'physical_1_2_5_8s': {h: copy.deepcopy(physical) for h in ('1', '2', '5', '8', 'prefix')}}


def evidence(full):
    extras = [event(eid, identity['label']) for eid, identity in full.items()]
    rows = [{'event_id': e['event_id'], 'group': e['group'], 'label': e['label'], 'horizon': h,
             'available': True, 'finite': True, 'numerically_admissible': True, 'mean_point_error': 2.}
            for e in extras for h in ('1', '2', '5', '8', 'prefix')]
    return {'event_count': len(extras), 'extras': extras, 'rows': {'objective_point_error': rows}, 'test_read': False,
            'cohorts': {'original366': {'event_count': 346}, 'new19': {'event_count': 39}}}


old_ids = [f'original_{i:03d}' for i in range(346)]+[f'newtrain_OLD_{i:02d}' for i in range(20)]
new_ids = [f'newtrain_NEW_{i:02d}' for i in range(19)]
rows = [{'id': eid, 'group': 'synthetic_bus', 'label': 'stable' if i < 342 else 'unstable'}
        for i, eid in enumerate(old_ids+new_ids)]
groups, full = m['cohorts'](rows[:366], rows)
assert len(groups['original366']) == 366 and len(groups['new19']) == 19
assert sum(eid.startswith('newtrain_') for eid in groups['original366']) == 20
assert groups['new19'] == set(new_ids)
assert len([eid for eid in full if eid.startswith('newtrain_')]) == 39
rejected(m['cohorts'], rows[:365], rows)
rejected(m['cohorts'], rows[:366], rows[:-1]+[rows[0]])
e = evidence(full); m['validate_evidence'](e, full)
missing = copy.deepcopy(e); missing['extras'].pop()
rejected(m['validate_evidence'], missing, full)
missing = copy.deepcopy(e); missing['rows']['objective_point_error'].pop()
rejected(m['validate_evidence'], missing, full)
assert abs(m['risk']([1., 2., 3.])-2.2) < 1e-12
assert m['risk']([None]) is None
a, b = event('a', 'unstable', 1, 2.), event('b', 'unstable', 3, 4.)
pooled = m['pooled']([a, b], 'frequency')
assert pooled['sse'] == 52. and pooled['n_scalar_points'] == 4
assert abs(pooled['rmse']-math.sqrt(13.)) < 1e-12
power = m['pooled']([a, b], 'mechanical_power')
assert power['n_scalar_points'] == 40 and abs(power['rmse']-.1) < 1e-12 and abs(power['rmse_MW']-10.) < 1e-12
b['classification_15s']['los_pair_pass'] = False
assert m['pooled']([a, b], 'frequency')['rmse'] is None
all_summary = m['summarize'](e, groups['common385'])
assert all_summary['event_count'] == 385 and all_summary['stable_count'] == 342 and all_summary['unstable_count'] == 43
assert all_summary['stable']['5']['ceil_risk'] == 2.
assert m['summarize'](e, groups['original366'])['event_count'] == 366
assert m['summarize'](e, groups['new19'])['event_count'] == 19
e['extras'][0]['scientific_numeric_8s'] = False
failed = m['summarize'](e, groups['common385'])
assert failed['stable']['5']['ceil_risk'] is None
assert failed['classification_8s']['numerical_unresolved'] == 1
assert failed['classification_15s']['numerical_unresolved'] == 0
assert m['differences'](failed, all_summary)['stable_5.ceil_risk'] is None
assert failed['unstable_actual_interval15']['frequency']['rmse'] == all_summary['unstable_actual_interval15']['frequency']['rmse']
print('PASS: exact identity cohorts despite39 shared prefixes; missing/duplicate IDs rejected; ceil risk; pooled SSE/count; 8/15 qualification separation; None-preserving differences')
print('extractor_sha256='+hashlib.sha256(SOURCE.read_bytes()).hexdigest())
print('qa_sha256='+hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
