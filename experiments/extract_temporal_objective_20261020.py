"""Read-only JSON arithmetic for exactly nine fixed 512-update train queues.

No model imports, binary artifacts, data readers, inference or statistical tests.
Cohorts use the frozen385 identity file, never event-name prefixes.
Only JSON is read; no training/evaluation/model code is imported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'private_data/closure_mechanism_20261019/P2/temporal_objective01'
COHORT_PATH = ROOT/'results/quadratic_memory_edge_20261020/cohort_identity.json'
COHORT_SHA = '0d2b3ed030cd742ae0809114008c31e3a99c35c56d0238d2d03c5473abcecbf5'
SEEDS = (6101, 6102, 6103)
VARIANTS = ('O', 'F', 'T')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def risk(values):
    if not values or not all(finite(v) for v in values):
        return None
    n = len(values); tail = math.ceil(.1*n)
    return .8*sum(values)/n+.2*sum(sorted(values, reverse=True)[:tail])/tail


def identity_map(rows):
    result = {r['id']: {'group': str(r['group']), 'label': r['label']} for r in rows}
    require(len(result) == len(rows), 'duplicate source identity')
    return result


def load_cohorts(payload):
    rows = payload['cohorts']
    original, new = identity_map(rows['original366']), identity_map(rows['new19'])
    require(len(original) == 366 and len(new) == 19 and not set(original) & set(new), 'frozen 366/19 cohorts invalid')
    full = original | new
    require(sum(r['label'] == 'stable' for r in full.values()) == 342 and sum(r['label'] == 'unstable' for r in full.values()) == 43, '385 labels must be342/43')
    return {'common385': set(full), 'original366': set(original), 'new19': set(new)}, full


def validate_evidence(evidence, full):
    extras = evidence['extras']; ids = [r['event_id'] for r in extras]
    require(evidence['event_count'] == 385 and len(ids) == len(set(ids)) == 385 and set(ids) == set(full), 'queue IDs missing/extra/duplicated')
    require(evidence.get('test_read') is False, 'evidence must declare test_read=false')
    require(all({'group': str(r['group']), 'label': r['label']} == full[r['event_id']] for r in extras), 'queue source label/group mismatch')
    for horizon in ('1', '2', '5', '8', 'prefix'):
        rows = [r for r in evidence['rows']['objective_point_error'] if str(r['horizon']) == horizon]
        require(len(rows) == 385 and {r['event_id'] for r in rows} == set(full), 'point-error horizon IDs missing or duplicated')


def pair_ok(p):
    return (p.get('finite_to_endpoint') is True and p.get('los_pair_pass') is True and
            all(finite(p.get(k)) and p[k] <= .001 for k in ('coi_normalized_rmse', 'common_normalized_rmse')))


def qualified15(e):
    a = e['actual_interval15']
    return (a.get('available') is True and a.get('finite') is True and a.get('numerically_admissible') is True
            and pair_ok(e['classification_15s']) and
            all(finite(a.get(k)) and a[k] <= .001 for k in ('coi_normalized_rmse', 'common_normalized_rmse')))


def pooled(extras, family, horizon=None):
    sse = 0.; count = 0; bad = []
    for e in extras:
        actual = horizon is None
        p = e['actual_interval15'] if actual else e['physical_1_2_5_8s'][horizon]
        valid = qualified15(e) if actual else e['scientific_numeric_8s'] is True
        if actual:
            n = p.get('n_time_points', 0)*(10 if family == 'mechanical_power' else 1)
            rmse = p.get('power_rmse_system_pu' if family == 'mechanical_power' else 'frequency_rmse_rad_s')
            value = rmse*rmse*n if finite(rmse) else None
        else:
            f = p[family]; n = f.get('n_scalar_points', 0); value = f.get('physical_sse')
            valid = valid and f.get('finite') is True
        if not valid or p.get('available') is not True or not finite(value) or value < 0 or not isinstance(n, int) or n <= 0:
            bad.append(e['event_id'])
        else:
            sse += value; count += n
    complete = bool(extras) and not bad
    rmse = math.sqrt(sse/count) if complete and count else None
    return {'events': len(extras), 'unresolved_ids': bad, 'sse': sse if complete else None,
            'n_scalar_points': count if complete else None, 'rmse': rmse,
            'rmse_MW': 100*rmse if family == 'mechanical_power' and rmse is not None else None,
            'unit': 'system pu (100 MVA)' if family == 'mechanical_power' else 'rad/s',
            'sse_origin': 'event_rmse_squared_times_actual_scalar_points' if horizon is None else 'stored_physical_sse'}


def classifications(extras, seconds):
    out = {k: 0 for k in ('stable_denominator', 'unstable_denominator', 'missed', 'false_los', 'undetermined', 'numerical_unresolved')}
    out['categories'] = {}
    for e in extras:
        out[e['label']+'_denominator'] += 1
        c = e[f'classification_{seconds}s']['fine_classification']
        ok = qualified15(e) if seconds == 15 else e['scientific_numeric_8s'] is True
        category = c['category'] if ok else 'numerical_unresolved'
        out['categories'][category] = out['categories'].get(category, 0)+1
        if not ok or category == 'numerical_unresolved':
            out['numerical_unresolved'] += 1
            continue
        if category == 'window_no_los_unsettled': out['undetermined'] += 1
        if c.get('predicted_los') is None or category in ('latency_uncovered', 'undetermined', 'nonfinite'):
            out['undetermined'] += 1
            continue
        out['missed'] += int(e['label'] == 'unstable' and c['predicted_los'] is False)
        out['false_los'] += int(e['label'] == 'stable' and c['predicted_los'] is True)
    out['note'] = 'unsettled no-LOS counts as undetermined and, for unstable truth, missed; counts can overlap'
    return out


def base_summary(evidence, ids):
    extras = [e for e in evidence['extras'] if e['event_id'] in ids]
    byid = {e['event_id']: e for e in extras}
    require(set(byid) == ids, 'summary cohort incomplete')
    stable = [e for e in extras if e['label'] == 'stable']; unstable = [e for e in extras if e['label'] == 'unstable']
    out = {'event_count': len(ids), 'stable_count': len(stable), 'unstable_count': len(unstable), 'stable': {}}
    for h in ('1', '2', '5', '8'):
        rows = [r for r in evidence['rows']['objective_point_error'] if r['event_id'] in ids and r['label'] == 'stable' and str(r['horizon']) == h]
        require(len(rows) == len(stable) and {r['event_id'] for r in rows} == {r['event_id'] for r in stable}, 'stable rows incomplete')
        bad = [r['event_id'] for r in rows if not (r.get('available') is True and r.get('finite') is True and r.get('numerically_admissible') is True and byid[r['event_id']]['scientific_numeric_8s'] is True and finite(r.get('mean_point_error')))]
        values = [r['mean_point_error'] for r in rows]
        out['stable'][h] = {'events': len(rows), 'unresolved_ids': bad,
            'mean_event_point_error': None if bad or not values else sum(values)/len(values),
            **{f: pooled(stable, f, h) for f in ('frequency', 'mechanical_power')}}
        if h == '5': out['stable'][h]['ceil_risk'] = None if bad else risk(values)
    bad = [e['event_id'] for e in unstable if not qualified15(e) or not finite(e['actual_interval15'].get('mean_point_error')) or e['actual_interval15'].get('n_time_points', 0) <= 0]
    points = sum(e['actual_interval15']['n_time_points'] for e in unstable) if not bad else 0
    out['unstable_actual_interval15'] = {'events': len(unstable), 'unresolved_ids': bad,
        'mean_event_point_error': None if bad or not unstable else sum(e['actual_interval15']['mean_point_error'] for e in unstable)/len(unstable),
        'time_pooled_point_error': None if bad or not points else sum(e['actual_interval15']['mean_point_error']*e['actual_interval15']['n_time_points'] for e in unstable)/points,
        **{f: pooled(unstable, f) for f in ('frequency', 'mechanical_power')}}
    out['classification_8s'] = classifications(extras, 8); out['classification_15s'] = classifications(extras, 15)
    return out


def metric_values(summary):
    out = {}
    for h, cell in summary['stable'].items():
        out[f'stable_{h}.mean_event_point_error'] = cell['mean_event_point_error']
        if h == '5': out['stable_5.ceil_risk'] = cell['ceil_risk']
        for f in ('frequency', 'mechanical_power'): out[f'stable_{h}.{f}.rmse'] = cell[f]['rmse']
    cell = summary['unstable_actual_interval15']
    for k in ('mean_event_point_error', 'time_pooled_point_error'): out['unstable_actual_interval15.'+k] = cell[k]
    for f in ('frequency', 'mechanical_power'): out['unstable_actual_interval15.'+f+'.rmse'] = cell[f]['rmse']
    out['unstable_actual_interval15.mechanical_power.rmse_MW'] = cell['mechanical_power']['rmse_MW']
    for s in (8, 15):
        for k in ('missed', 'false_los', 'undetermined', 'numerical_unresolved'): out[f'classification_{s}s.{k}'] = summary[f'classification_{s}s'][k]
    for family in ('temporal_actual_interval15', 'unstable_temporal_actual_interval15'):
        for group, cell in summary[family].items():
            for metric in ('mean_event_point_error', 'mean_event_tail20_point_error'):
                out[f'{family}.{group}.{metric}'] = cell[metric]
    return out


def differences(left, right):
    a, b = metric_values(left), metric_values(right)
    return {k: a[k]-b[k] if finite(a[k]) and finite(b[k]) else None for k in a}


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def temporal_facts(extras):
    """Fixed actual end groups, including finite but unqualified tail facts."""
    result = {}
    for group in ('all', 'le5s', 'gt5s'):
        selected = [e for e in extras if group == 'all' or
                    ('le5s' if e['actual_interval15']['n_time_points'] <= 500 else 'gt5s') == group]
        bad = []; gaps = []; time_values = []; tail_values = []
        for e in selected:
            n = e['actual_interval15']['n_time_points']; t = e.get('temporal')
            require(isinstance(n,int) and not isinstance(n,bool) and 1 <= n <= 1500, 'actual future point count must be genuine1..1500')
            if not isinstance(t, dict):
                gaps.append(e['event_id']); bad.append(e['event_id']); continue
            expected_group = 'le5s' if n <= 500 else 'gt5s'
            require(t['actual_future_points'] == n and t['tail_points'] == (n+4)//5 and t['endpoint_group'] == expected_group, 'temporal true endpoint facts inconsistent')
            if not qualified15(e) or t.get('finite') is not True or not finite(t.get('mean_point_error')) or not finite(t.get('tail_point_error')):
                bad.append(e['event_id']); continue
            time_values.append(t['mean_point_error']); tail_values.append(t['tail_point_error'])
        good = bool(selected) and not bad
        result[group] = {'event_count': len(selected), 'stable_count': sum(e['label'] == 'stable' for e in selected),
            'unstable_count': sum(e['label'] == 'unstable' for e in selected), 'unresolved_ids': bad,
            'missing_temporal_fact_ids': gaps,
            'mean_event_point_error': sum(time_values)/len(time_values) if good else None,
            'mean_event_tail20_point_error': sum(tail_values)/len(tail_values) if good else None,
            'qualification': 'same-event actual_interval15 and classification_15s guards; finite alone is insufficient',
            'classification_15s': classifications(selected,15)}
    return result


def summarize(evidence, ids):
    result = base_summary(evidence, ids)
    extras = [e for e in evidence['extras'] if e['event_id'] in ids]
    result['temporal_actual_interval15'] = temporal_facts(extras)
    result['unstable_temporal_actual_interval15'] = temporal_facts([e for e in extras if e['label'] == 'unstable'])
    return result


def compact_local(local):
    return {'source_sha256': local['source_sha256'],
            'origin': {k:local['origin'][k] for k in ('equilibrium_pass','energy_stationary_pass','quotient_eligible','f0_norm_inf','gradH0_norm_inf')},
            'quotient': local['quotient'], 'linear_field': local['linear_field'], 'energy_curvature': local['energy_curvature']}


def extract(output):
    output = Path(output)
    local_events = BASE/'extracted_event_rows_step000512.json'
    require(not output.exists() and not local_events.exists(), 'both output files must be new; no overwrite')
    require(sha(COHORT_PATH) == COHORT_SHA, 'frozen cohort identity hash changed')
    groups, full = load_cohorts(read(COHORT_PATH))
    queues = {f'{s}_{v}': BASE/f'evaluation/{s}_{v}/step000512/queue' for s in SEEDS for v in VARIANTS}
    # Preflight every completion marker before parsing any scientific evidence.
    for key, queue in queues.items():
        marker = read(queue/'complete.json')
        require(marker['status'] == 'COMPLETED' and marker['event_count'] == 385 and marker['evidence_sha256'] == sha(queue/'evidence.json'), 'all nine complete385 evidence hashes required: '+key)
    models = {}; events = {}; sources = {}; endpoint_counts = None
    for key, queue in queues.items():
        seed, variant = key.split('_'); evidence = read(queue/'evidence.json')
        validate_evidence(evidence,full)
        p = evidence['provenance']
        require((p['seed'],p['variant'],p['stage_step']) == (int(seed),variant,512), 'wrong queue provenance')
        counts = {e['event_id']:e['actual_interval15']['n_time_points'] for e in evidence['extras']}
        require(endpoint_counts is None or counts == endpoint_counts, 'actual end groups must be identical across all models')
        endpoint_counts = counts
        models[key] = {name:summarize(evidence,ids) for name,ids in groups.items()}
        models[key]['local'] = compact_local(read(queue/'local.json'))
        events[key] = {'objective_point_error':evidence['rows']['objective_point_error'], 'extras':evidence['extras']}
        sources[key] = {'path':str(queue/'evidence.json'),'sha256':sha(queue/'evidence.json'), 'local_sha256':sha(queue/'local.json')}
    pairs = {str(s):{f'{a}-{b}':{name:differences(models[f'{s}_{a}'][name],models[f'{s}_{b}'][name]) for name in groups}
             for a,b in (('F','O'),('T','F'),('T','O'))} for s in SEEDS}
    result = {'status':'COMPLETED','split':'train','stage_step':512,'models':models,'paired_differences':pairs,
              'cohort_source':{'path':str(COHORT_PATH),'sha256':COHORT_SHA},'queue_sources':sources,
              'event_rows_local_only':str(local_events),
              'scope':'JSON arithmetic on fixed train cohorts; original scores retained; no scientific adoption or significance claim'}
    write_new(local_events,events); write_new(output,result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        extract(args.output)
    except (OSError,ValueError,KeyError,TypeError) as error:
        if args.output.exists(): raise
        write_new(args.output,{'status':'UNRESOLVED','split':'train','error':repr(error),'scope':'no incomplete queue/event omitted; no new inference'})
        raise SystemExit(2)


if __name__ == '__main__':
    main()


