"""Read the saved six-endpoint val rows and assemble the registered matched comparison."""

from collections import Counter
from math import isfinite
from statistics import fmean, stdev

import numpy as np

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
from kiki.evaluation.osgkd import aggregate_metrics
from kiki.evaluation.osgkd_autonomous import autonomous_outcome


SEEDS = (24111, 24121, 24131)
SCOPES = tuple(f'stable_{h}s' for h in (1, 2, 5, 8)) + ('unstable_actual',)


def vector(endpoint, timing):
    scored = endpoint['score']
    metrics = scored['summary']['splits']['val']
    values = {}
    for scope in SCOPES:
        m = metrics[scope]['covered_only_diagnostic' if scope == 'unstable_actual' else 'official'] or {}
        values.update({
            scope + '_risk': m.get('motion_risk_v1'),
            scope + '_speed': m.get('common_speed_rmse_rad_s'),
            scope + '_tau_m': (m.get('tau_m') or {}).get('normalized_rmse'),
            scope + '_tau_e': (m.get('tau_e') or {}).get('normalized_rmse'),
        })
    auto = scored['autonomous']
    stable, unstable = auto['stable_false_alarm_8s'], auto['unstable_recall_8s']
    io.require((stable['raw_denominator'], stable['input_eligible_denominator'],
                unstable['raw_denominator'], unstable['input_eligible_denominator']) == (95, 95, 22, 12),
               'symmetric val populations differ')
    lag = auto['trigger_lag_s']
    spectrum = endpoint['spectrum']
    return dict(values, qualified=scored['numerical_qualified_events'],
                stable_false_alarms=stable['autonomous_trigger'], unstable_autonomous_hits=unstable['autonomous_trigger'],
                unstable_actual_covered=metrics['unstable_actual']['covered'],
                unstable_actual_hits=auto['unstable_actual_interval_hits'],
                lag_count=lag['count'], lag_median_s=lag['median'], lag_iqr_s=lag['iqr'],
                lag_q1_s=lag['q1'], lag_q3_s=lag['q3'],
                alpha=spectrum['max_real'], positive_modes=spectrum['positive_modes'],
                inference_s=timing['inference_wall_s'])


def sample_statistics(values):
    available = [value for value in values if value is not None]
    if len(available) != 3:
        return dict(n=len(available), mean=None, sample_sd=None, minimum=None, maximum=None)
    return dict(n=3, mean=fmean(available), sample_sd=stdev(available),
                minimum=min(available), maximum=max(available))


def lag_summary(seconds):
    q1, median, q3 = (float(x) for x in np.quantile(seconds, [.25, .5, .75])) if seconds else (None, None, None)
    return dict(count=len(seconds), q1=q1, median=median, q3=q3, iqr=q3-q1 if seconds else None)


def paired_lag(graph, flat):
    io.require(len(graph) == len(flat) == 117, 'paired raw val count differs')
    counts = Counter()
    seconds, actual = [], []
    for a, b in zip(graph, flat, strict=True):
        io.require((a['event_id'], a['label'], a['group']) == (b['event_id'], b['label'], b['group']),
                   'paired val event identity differs')
        if a['label'] != 'unstable':
            continue
        graph_hit = autonomous_outcome(a) == 'autonomous_trigger'
        flat_hit = autonomous_outcome(b) == 'autonomous_trigger'
        counts['graph_only'] += graph_hit and not flat_hit
        counts['flat_only'] += flat_hit and not graph_hit
        if not (graph_hit and flat_hit):
            continue
        counts['common'] += 1
        l, r = a['classification_8s'], b['classification_8s']
        true = l['true_time_record_s']
        io.require(true == r['true_time_record_s'], 'paired true trigger time differs')
        if true is not None and isfinite(true):
            difference = l['predicted_time_record_s'] - r['predicted_time_record_s']
            seconds.append(difference)
            if l['timing_comparable'] and r['timing_comparable']:
                actual.append(difference)
    return dict(**lag_summary(seconds), common=counts['common'], graph_only=counts['graph_only'],
                flat_only=counts['flat_only'], actual_comparable=lag_summary(actual))


def critical_event(rows):
    stable = [row for row in rows if row['label'] == 'stable']
    focal = [row for row in stable if row['event_id'] == 'pilot_0232']
    io.require(len(stable) == 95 and len(focal) == 1, 'critical stable val event differs')
    event = focal[0]
    item = event['classification_8s'] or {}
    horizons = {}
    for h in (1, 2, 5, 8):
        m = event['scopes'][str(h)]['metrics'] or {}
        horizons[str(h)] = m.get('mean_motion_error')
    omit = [row for row in stable if row['event_id'] != 'pilot_0232']
    reduced = aggregate_metrics(omit, '8')
    return dict(outcome=autonomous_outcome(event), category=item.get('classification', {}).get('category'),
                predicted_record_s=item.get('predicted_time_record_s'), qualification=item.get('los_pair_pass'),
                event_mean_motion_error=horizons, omitted_stable_count=94,
                sensitivity_stable_8s_risk=(reduced['official'] or {}).get('motion_risk_v1'))


def compare(endpoints, timing):
    order = [f'{kind}_{seed}' for kind in ('graph', 'flat') for seed in SEEDS]
    io.require(set(endpoints) == set(timing) == set(order), 'six fixed endpoints required')
    readout = {name: vector(endpoints[name], timing[name]) for name in order}
    rows = {name: io.read(io.verified(endpoints[name]['score']['rows_ref'])) for name in order}
    keys = tuple(readout[order[0]])
    stats = {
        kind: {key: sample_statistics([readout[f'{kind}_{seed}'][key] for seed in SEEDS]) for key in keys}
        for kind in ('graph', 'flat')
    }
    differences = {}
    trigger = {}
    for seed in SEEDS:
        left, right = f'graph_{seed}', f'flat_{seed}'
        a, b = readout[left], readout[right]
        differences[str(seed)] = {
            key: a[key]-b[key] if a[key] is not None and b[key] is not None else None for key in keys
        }
        trigger[str(seed)] = paired_lag(rows[left], rows[right])
    delta_stats = {}
    for key in keys:
        values = [row[key] for row in differences.values()]
        sign = Counter('lower' if val < 0 else 'higher' if val > 0 else 'equal'
                       for val in values if val is not None)
        delta_stats[key] = dict(**sample_statistics(values), lower=sign['lower'], equal=sign['equal'], higher=sign['higher'])
    return dict(seed_order=list(SEEDS), per_endpoint=readout, group_statistics=stats,
                graph_minus_flat=differences, graph_minus_flat_statistics=delta_stats,
                common_trigger_graph_minus_flat_s=trigger, pilot_0232={name: critical_event(rows[name]) for name in order})
