"""Read-only JSON arithmetic for exactly nine fixed 512-update train queues.

No model imports, binary artifacts, data readers, inference or statistical tests.
The evaluator's prefix-based `cohorts` field is deliberately never used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
FRESH = BASE.parents[1]/'P3/fresh_design_pair01'
SEEDS = (6101, 6102, 6103)
VARIANTS = ('A', 'B', 'C')


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


def cohorts(original_rows, all_rows):
    original, full = identity_map(original_rows), identity_map(all_rows)
    require(len(original) == 366 and len(full) == 385, 'source cohort sizes must be 366/385')
    require(set(original) <= set(full), 'original IDs absent from full385')
    require(all(original[k] == full[k] for k in original), 'source label/group mismatch')
    require(sum(r['label'] == 'stable' for r in full.values()) == 342 and sum(r['label'] == 'unstable' for r in full.values()) == 43, 'full385 labels must be 342/43')
    new = set(full)-set(original)
    require(len(new) == 19, 'new cohort must be exact difference of size19')
    return {'common385': set(full), 'original366': set(original), 'new19': new}, full


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


def summarize(evidence, ids):
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
    for s in (8, 15):
        for k in ('missed', 'false_los', 'undetermined', 'numerical_unresolved'): out[f'classification_{s}s.{k}'] = summary[f'classification_{s}s'][k]
    return out


def differences(left, right):
    a, b = metric_values(left), metric_values(right)
    return {k: a[k]-b[k] if finite(a[k]) and finite(b[k]) else None for k in a}


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def extract(output):
    event_output = BASE/'extracted_event_rows_step000512.json'
    require(not output.exists() and not event_output.exists(), 'new output paths required; no overwrite')
    source_records = {}; reference = None; groups = None
    for seed in SEEDS:
        path = FRESH/f'models/{seed}_full8/checkpoints/step001280/checkpoint.json'
        meta = read(path)
        require(meta['step'] == 1280 and meta['identity']['seed'] == seed and meta['identity']['structure'] == 'full8', 'source endpoint identity mismatch')
        candidate_groups, full = cohorts(meta['identity']['source_identity']['common']['events'], meta['identity']['events'])
        if reference is not None: require(reference == full and groups == candidate_groups, 'source cohorts differ across seeds')
        reference, groups = full, candidate_groups
        source_records[str(seed)] = {'path': str(path), 'sha256': sha(path)}
    summaries = {}; queue_sources = {}; events = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            key = f'{seed}_{variant}'; queue = BASE/f'evaluation/{key}/step000512/queue'
            marker = read(queue/'complete.json'); path = queue/'evidence.json'
            require(marker['status'] == 'COMPLETED' and marker['event_count'] == 385 and marker['evidence_sha256'] == sha(path), 'queue completion/hash failure')
            e = read(path); validate_evidence(e, reference)
            provenance = e['provenance']
            require(provenance['seed'] == seed and provenance['variant'] == variant and provenance['stage_step'] == 512, 'queue provenance mismatch')
            summaries[key] = {name: summarize(e, ids) for name, ids in groups.items()}
            queue_sources[key] = {'path': str(path), 'sha256': sha(path)}
            events[key] = {'objective_point_error': e['rows']['objective_point_error'], 'extras': e['extras']}
    pairs = {}
    for seed in SEEDS:
        pairs[str(seed)] = {f'{a}-{b}': {name: differences(summaries[f'{seed}_{a}'][name], summaries[f'{seed}_{b}'][name]) for name in groups} for a, b in (('C', 'B'), ('A', 'B'), ('C', 'A'))}
    result = {'status': 'COMPLETED', 'step': 512, 'split': 'train', 'source_checkpoint_json': source_records, 'queue_sources': queue_sources,
        'cohort_ids': {name: sorted(ids) for name, ids in groups.items()}, 'models': summaries, 'paired_differences': pairs,
        'event_rows_local_only': str(event_output), 'ignored_field': 'evidence.cohorts (prefix-derived346/39 is invalid for these385)',
        'scope': 'fixed train factual arithmetic; no adoption, significance, independent confirmation, or stability claim'}
    write_new(event_output, events); write_new(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        extract(args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if args.output.exists(): raise
        write_new(args.output, {'status': 'UNRESOLVED', 'step': 512, 'split': 'train', 'error': repr(error), 'scope': 'no missing queue/event filled; no partial aggregate'})
        raise SystemExit(2)


if __name__ == '__main__':
    main()
