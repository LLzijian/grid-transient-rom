"""Apply the frozen adoption criteria to completed archives, without model execution."""

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from experiments.osgkd_v2_fast_zero_sum_20261124 import run
from experiments.osgkd_v2_train385_risk_alignment_20261123 import analyze as earlier

require, identity, read = run.require, run.identity, run.read
SPLITS = ('train', 'val')
GROUPS = ('control', 'candidate')


def audit_progress(result):
    plan = read(result['plan_ref'])
    batches = plan['batches']
    require(len(batches) == 200 and len(plan['coverage']) == 385, 'plan denominator differs')
    require(sum(plan['coverage'].values()) == 1600, 'plan exposure count differs')
    sizes, accepted = {}, {}
    for name in GROUPS:
        group = result['treatments'][name]
        log = read(group['attempts_ref'])
        sizes[name] = len(log)
        accepted[name] = group['accepted']
        require([row['event_ids'] for row in log] == batches[:len(log)], 'paired order differs')
        for i, row in enumerate(log):
            require(row['index'] == i and row['attempt'] == 2061 + i and
                    row['lr_multiplier'] == run.multiplier(i) and
                    row['fast_lr_multiplier'] == run.fast_multiplier(i) and
                    (not row['accepted'] or row['loss'] is not None and row['shared_norm'] is not None and
                     earlier.finite(row)), 'attempt log differs')
        for key in ('accepted', 'shared_clipped', 'fast_clipped'):
            require(sum(bool(row.get(key)) for row in log) == group[key], 'attempt tally differs: ' + key)
        require(group['accepted'] + group['rejected'] == len(log) == group['attempts'], 'attempt count differs')
        require(group['work']['rhs_forward'] <= run.CAP and group['work']['rhs_recompute'] == 0, 'training work differs')
        run.verified(group['checkpoint'])
    require(set(sizes.values()) == {result['paired_attempts']}, 'paired count differs')
    valid = (result['paired_attempts'] == 200 and result['fixed_endpoint'] == 2260 and not result['stopped_early'] and
             min(accepted.values()) >= 196 and abs(accepted['control'] - accepted['candidate']) <= 2)
    require(valid == result['valid_endpoint'], 'endpoint eligibility differs')
    return valid


def audit_short(short, inputs):
    profile = read(inputs['conflict_profile']['result']['rows_ref'])
    expected = set(profile['cohort_event_ids']['parent_misses'])
    selected = {row['event_id']: row['selected_machine'] for row in profile['signed_electrical']
                if all(item['persistent_80pct'] is True for item in row['endpoints'].values())}
    rows = read(short['rows_ref'])
    require(len(rows) == 27 and {row['event_id'] for row in rows} == expected and len(selected) == 22 and
            {row['event_id']: row['selected_machine'] for row in rows if row['selected_machine'] is not None} == selected,
            'frozen cohort differs')
    output = {}
    for window in ('0.03', 'capped_0.25'):
        counts, means = {}, {}
        for name in GROUPS:
            items = [row['models'][name]['windows'][window] for row in rows if row['event_id'] in selected]
            good = [item for item in items if item['complete'] and item['numerical_qualified']]
            counts[name] = len(good)
            aggregate = short['aggregates'][name][window]['persistent_22']
            means[name] = sum(item['electrical_normalized_mse'] for item in good) / len(good) if good else None
            reference = aggregate['event_equal_electrical_normalized_mse']
            require(aggregate['raw_denominator'] == len(items) == 22 and
                    aggregate['numerical_qualified_events'] == len(good) and
                    (reference is means[name] is None or reference is not None and means[name] is not None and
                     math.isclose(reference, means[name], rel_tol=1e-12, abs_tol=1e-12)), 'short-window tally differs')
        output[window] = {'qualified': counts, 'mean_normalized_mse': means,
                          'candidate_reduction': earlier.reduction(means['candidate'], means['control'])}
    return output


def endpoint_decision(summary, inputs):
    result = summary['result']
    fixed = audit_progress(result)
    if result['stopped_early']:
        return {'gates': {'eligible_fixed_endpoint': False, 'all_working_parent_gates': False}}
    parent_rows, model_rows, comparisons, changes = {}, {name: {} for name in GROUPS}, {}, {}
    for split in SPLITS:
        parent_end = inputs['parent_summary']['result']['treatments']['control']['evaluation'][split]
        parent_rows[split], parent_metrics = earlier.read_score(parent_end, split)
        comparisons[split] = {'parent_2060': earlier.describe(parent_rows[split], parent_metrics, split)}
        comparisons[split]['parent_2060']['timing'] = parent_end['timing']
        for name in GROUPS:
            ref = result['treatments'][name]['evaluation'][split]
            require(ref['data_identity'] == parent_end['data_identity'], 'scored data identity differs')
            rows, metrics = earlier.read_score(ref, split)
            cohort = lambda sequence: {(row['event_id'], row['label'], row['actual_future_points']) for row in sequence}
            require(cohort(rows) == cohort(parent_rows[split]), 'scored population differs')
            model_rows[name][split] = rows
            comparisons[split][name] = earlier.describe(rows, metrics, split)
            comparisons[split][name]['timing'] = ref['timing']
        a, b = (earlier.hits(model_rows[name][split]) for name in GROUPS)
        parent = earlier.hits(parent_rows[split])
        changes[split] = {'candidate_gained_vs_control': len(b - a), 'candidate_lost_vs_control': len(a - b),
                          'parent_hits': len(parent), 'parent_hits_retained': len(parent & b)}
    numerical = all(earlier.qualification(model_rows[name][split], split) for name in GROUPS for split in SPLITS)
    val = comparisons['val']
    control, candidate = (val[name]['stable_risks']['8']['official'] for name in GROUPS)
    stable_rows = [row for row in model_rows['candidate']['val'] if row['label'] == 'stable']
    protection = (len(stable_rows) == 95 and all(row['classification_8s']['truth_supported'] and
                   row['classification_8s']['los_pair_pass'] and
                   not row['classification_8s']['classification']['predicted_los'] for row in stable_rows) and
                  candidate is not None and control is not None and candidate['motion_risk_v1'] < 1 and
                  candidate['motion_risk_v1'] <= 1.05 * control['motion_risk_v1'])
    short = audit_short(result['short_diagnostic'], inputs)
    gains = {window: item['candidate_reduction'] for window, item in short.items()}
    gates = {'eligible_fixed_endpoint': fixed, 'matched_numerical_coverage': numerical,
             'val_stable_protection': protection, 'new_val_actual_hit': changes['val']['candidate_gained_vs_control'] >= 1,
             'retains_control_and_parent_hits': all(item['candidate_lost_vs_control'] == 0 and
                                                   item['parent_hits_retained'] == item['parent_hits'] for item in changes.values()),
             'fixed_22_complete_and_qualified': all(list(item['qualified'].values()) == [22, 22] for item in short.values()),
             'short_capped_250ms_mse_reduced': gains['capped_0.25'] is not None and gains['capped_0.25'] >= .05,
             'short_30ms_mse_protected': gains['0.03'] is not None and gains['0.03'] >= -.05}
    gates['all_working_parent_gates'] = all(gates.values())
    return {'gates': gates, 'endpoint_metrics': comparisons, 'actual_detection_changes': changes,
            'short_diagnostic_audit': short}


def main():
    result_path = ROOT / 'results' / run.RUN
    summary_ref = identity(result_path / 'pair01' / 'summary.json')
    summary = read(summary_ref)
    require((summary['run_id'], summary['job_id'], summary['status']) == (run.RUN, 'pair01', 'completed'), 'pair incomplete')
    require(read(summary['release_ref']) == summary['release'], 'release identity differs')
    done = ROOT / 'artifacts' / run.RUN / 'pair01' / 'completed.json'
    require(read(identity(done)) == {'status': 'completed', 'summary_ref': summary_ref}, 'completion differs')
    inputs = run.validate(summary['release'], 'pair01')
    output = endpoint_decision(summary, inputs)
    output.update(run_id=run.RUN, summary_ref=summary_ref, analysis_source_ref=identity(Path(__file__).resolve()),
                  automatic_adoption=False)
    ref = run.baseline.atomic_write(result_path / 'analysis.json', output, public=True)
    print(json.dumps({'analysis_ref': ref, 'gates': output['gates']}))


if __name__ == '__main__':
    main()
