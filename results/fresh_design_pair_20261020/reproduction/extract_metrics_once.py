"""Stdlib JSON-only facts; no inference, dataset loading, or statistical tests."""
import argparse
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
KEYS = [f'{s}_{t}' for s in (6101, 6102, 6103) for t in ('full8', 'delete7')]
SCOPES = {'original_after': 366, 'new_after': 19, 'new_before': 19}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def risk(values):
    n = len(values)
    return None if not n or not all(finite(v) for v in values) else .8*sum(values)/n+.2*sum(sorted(values, reverse=True)[:math.ceil(n*.1)])/math.ceil(n*.1)


def pair_ok(p):
    return (p.get('finite_to_endpoint') is True and p.get('los_pair_pass') is True
            and all(finite(p.get(k)) and p[k] <= .001 for k in ('coi_normalized_rmse', 'common_normalized_rmse')))


def qualified15(e):
    a = e['actual_interval15']
    return (a.get('available') is True and a.get('finite') is True and a.get('numerically_admissible') is True and pair_ok(e['classification_15s'])
            and all(finite(a.get(k)) and a[k] <= .001 for k in ('coi_normalized_rmse', 'common_normalized_rmse')))


def pooled(rows, family, horizon, actual=False):
    sse = 0.; count = 0; bad = []; eligible = 0
    for e in rows:
        p = e['actual_interval15'] if actual else e['physical_1_2_5_8s'][horizon]
        available = p.get('available') is True
        eligible += available
        valid = qualified15(e) if actual else e['scientific_numeric_8s'] is True
        if actual:
            n = p.get('n_time_points', 0)*(10 if family == 'mechanical_power' else 1)
            rmse = p.get('power_rmse_system_pu' if family == 'mechanical_power' else 'frequency_rmse_rad_s')
            value = rmse*rmse*n if finite(rmse) else None
        else:
            f = p[family]; n = f.get('n_scalar_points', 0); value = f.get('physical_sse')
            valid = valid and f.get('finite') is True
        if not available or not valid or not finite(value) or value < 0 or n <= 0:
            bad.append(e['event_id'])
        else:
            sse += value; count += n
    complete = bool(rows) and not bad
    rmse = math.sqrt(sse/count) if complete and count else None
    return {'all_events': len(rows), 'eligible_events': eligible, 'unresolved_or_unavailable_ids': bad,
            'physical_sse': sse if complete else None, 'n_scalar_points': count if complete else None,
            'rmse': rmse, 'unit': 'system pu (100 MVA)' if family == 'mechanical_power' else 'rad/s',
            'rmse_MW': 100*rmse if family == 'mechanical_power' and rmse is not None else None,
            'sse_origin': 'sum(event_rmse^2*actual_scalar_points)' if actual else 'sum(stored_physical_sse)'}


def classifications(extras, seconds):
    result = {k: 0 for k in ('stable_denominator', 'unstable_denominator', 'false_los', 'missed', 'undetermined', 'numerical_unresolved')}
    result['categories'] = {}
    for e in extras:
        result[e['label']+'_denominator'] += 1
        p = e[f'classification_{seconds}s']; c = p['fine_classification']
        valid = qualified15(e) if seconds == 15 else e['scientific_numeric_8s'] is True
        category = c['category'] if valid else 'numerical_unresolved'
        result['categories'][category] = result['categories'].get(category, 0)+1
        if not valid:
            result['numerical_unresolved'] += 1
            continue
        if category == 'window_no_los_unsettled': result['undetermined'] += 1
        if c.get('predicted_los') is None or category in ('latency_uncovered', 'undetermined', 'nonfinite'):
            result['undetermined'] += 1
            continue
        result['false_los'] += int(e['label'] == 'stable' and c['predicted_los'] is True)
        result['missed'] += int(e['label'] == 'unstable' and c['predicted_los'] is False)
    return result


def summarize(evidence):
    extras = evidence['extras']; byid = {r['event_id']: r for r in extras}
    assert len(byid) == evidence['event_count'], 'duplicate/missing event IDs'
    out = {'event_count': evidence['event_count'], 'primary_original_ceil_risk': evidence['primary'], 'cells': {}}
    for label, horizons in (('stable', ('1', '2', '5', '8')), ('unstable', ('prefix',))):
        selected = [e for e in extras if e['label'] == label]
        for h in horizons:
            rows = [r for r in evidence['rows']['objective_point_error'] if r['label'] == label and str(r['horizon']) == h]
            assert len(rows) == len(selected) and {r['event_id'] for r in rows} == {e['event_id'] for e in selected}
            bad = [r['event_id'] for r in rows if not (r.get('available') is True and r.get('finite') is True and r.get('numerically_admissible') is True and byid[r['event_id']]['scientific_numeric_8s'] is True and finite(r.get('mean_point_error')))]
            out['cells'][label+'_'+h] = {'events': len(rows), 'unresolved_or_unavailable_ids': bad,
                'point_risk': None if bad else risk([r['mean_point_error'] for r in rows]),
                **{f: pooled(selected, f, h) for f in ('frequency', 'mechanical_power')}}
    unstable = [r for r in extras if r['label'] == 'unstable']
    out['unstable_actual_interval15'] = {f: pooled(unstable, f, None, True) for f in ('frequency', 'mechanical_power')}
    bad15 = [e['event_id'] for e in unstable if not qualified15(e) or not finite(e['actual_interval15'].get('mean_point_error'))]
    out['unstable_actual_interval15_point_error'] = {'events': len(unstable), 'unresolved_ids': bad15,
        'mean_event_point_error': None if bad15 or not unstable else sum(e['actual_interval15']['mean_point_error'] for e in unstable)/len(unstable), 'event_equal_weight': True}
    out['classification_8s'] = classifications(extras, 8); out['classification_15s'] = classifications(extras, 15)
    return out


def event_rows(evidence):
    points = {(r['event_id'], str(r['horizon'])): r for r in evidence['rows']['objective_point_error']}
    result = []
    for e in evidence['extras']:
        h = '5' if e['label'] == 'stable' else 'prefix'
        result.append({k: e[k] for k in ('event_id', 'group', 'label', 'scientific_numeric_8s')} |
                      {'point_error': points[(e['event_id'], h)], 'physical': e['physical_1_2_5_8s'][h],
                       'actual_interval15': e['actual_interval15'] if e['label'] == 'unstable' else None})
    return result


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def cohort_ids(extras, expected, reference=None):
    ids = {r['event_id'] for r in extras}
    assert len(extras) == len(ids) == expected, 'missing/duplicate event IDs'
    assert reference is None or ids == reference, 'cohort mismatch'
    return ids


def val_coverage(expected, models):
    complete = expected == 24 and len(models) == 6 and all(m['new_val']['primary_original_ceil_risk']['value'] is not None and all(m['new_val'][f'classification_{s}s']['numerical_unresolved'] == 0 for s in (8, 15)) for m in models.values())
    return {'attempted_count': 24, 'qualified_event_count': expected, 'missing_count': 24-expected,
            'whole24_resolved': complete, 'scope': 'numerical coverage only; no confirmation or improvement claim; N set by root FOM qualification before model scores'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--split', choices=('train', 'val'), default='train')
    p.add_argument('--allow-new-val', action='store_true'); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-val-events', type=int, choices=range(1, 25))
    args = p.parse_args()
    if args.split == 'val' and not args.allow_new_val:
        p.error('new val requires explicit --allow-new-val; all historical val/test forbidden')
    if args.split == 'val' and args.expected_val_events is None:
        p.error('new val requires explicit --expected-val-events from root FOM qualification; zero coverage is archived separately')
    sibling = args.output.with_name(args.output.stem+'_event_rows.json')
    if args.output.exists() or sibling.exists():
        p.error('both outputs must be new paths')
    models = {}; tables = {}; sources = {}; memberships = {}
    for key in KEYS:
        models[key] = {}; tables[key] = {}
        for scope, expected in (SCOPES.items() if args.split == 'train' else [('new_val', args.expected_val_events)]):
            folder = BASE/'evaluation_train'/key/scope if args.split == 'train' else BASE/'evaluation_val'/key
            marker = read(folder/'complete.json'); raw = (folder/'evidence.json').read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            assert marker['status'] == 'COMPLETED' and marker['event_count'] == expected and marker['evidence_sha256'] == sha
            e = json.loads(raw); assert e['event_count'] == expected and e['test_read'] is False
            ids = cohort_ids(e['extras'], expected, memberships.get(scope))
            memberships[scope] = ids
            models[key][scope] = summarize(e); tables[key][scope] = event_rows(e)
            sources[key+'/'+scope] = {'path': str(folder/'evidence.json'), 'sha256': sha}
            if scope in ('original_after', 'new_val'):
                local = read(folder/'local.json')
                models[key]['local'] = {k: local[k] for k in ('source_sha256', 'origin', 'quotient', 'linear_field', 'energy_curvature')}
                models[key]['local']['origin'] = {k: local['origin'][k] for k in ('equilibrium_pass', 'energy_stationary_pass', 'quotient_eligible', 'f0_norm_inf', 'gradH0_norm_inf')}
    if args.split == 'val':
        write_new(sibling, tables)
        write_new(args.output, {'split': 'val', 'models': models, 'sources': sources, 'coverage': val_coverage(args.expected_val_events, models), 'event_rows_file': str(sibling), 'scope': 'only exact current-study new val; conditional qualified-cohort metrics; no inference/tests or historical val/test'})
        return
    baseline_path = ROOT/'results/fixed_edge_pair_20261019/final.json'; baseline = read(baseline_path)
    attempts_path = ROOT/'results/fresh_design_pair_20261020/train_fom_attempts.json'; attempts = read(attempts_path)
    assert attempts['denominator'] == 24 and len(attempts['attempts']) == 24
    exclusions = [{k: r.get(k) for k in ('condition', 'reason', 'status', 'qualification')} for r in attempts['attempts'] if r.get('qualification', {}).get('pass') is not True]
    result = {'split': 'train', 'models': models, 'sources': sources, 'event_rows_file': str(sibling),
        'coverage': {'old_original': 393, 'old_eligible': 366, 'old_excluded': 27, 'new_attempted': 24, 'new_qualified': 19, 'new_failed': 5, 'new_exclusions': exclusions},
        'P2_before_original366': {k: v for k, v in baseline['capability_table'].items() if k.endswith('_1024')},
        'P2_before_source': str(baseline_path), 'new_attempt_source': str(attempts_path),
        'scope': 'JSON facts only; conditional qualified cohorts; no significance, model promotion or L2 conclusion'}
    assert memberships['new_before'] == memberships['new_after'], 'before/after new-train cohort mismatch'
    assert len(exclusions) == 5, 'new qualification exclusions must preserve five failed attempts'
    paired = {}
    for scope in ('new_before', 'new_after'):
        joined = {}
        for key in KEYS:
            for row in tables[key][scope]:
                eid = row['event_id']
                joined.setdefault(eid, {'event_id': eid, 'group': row['group'], 'label': row['label'], 'models': {}})['models'][key] = row
        assert len(joined) == 19 and all(len(r['models']) == 6 for r in joined.values())
        paired[scope] = list(joined.values())
    write_new(sibling, {'by_model_scope': tables, 'paired_newtrain19': paired}); write_new(args.output, result)


if __name__ == '__main__':
    main()
