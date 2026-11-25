"""Score the four frozen Sprint 2A endpoints with the unchanged val evaluator."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import numpy as np
import torch

from experiments.osgkd_v2_p0_20261123.run import spectrum
from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import run as graph_run
from experiments.osgkd_v2_sprint2a_20261125 import run
from kiki.data.osgkd_train_val import load_val
from kiki.evaluation.osgkd_autonomous import autonomous_outcome
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_matched_flat import OSGKDMatchedFlat


def endpoint_spectrum(model, directory):
    summary, detail = spectrum(model)
    eigenvalues = detail.get('eigenvalues')
    summary['positive_modes'] = (
        sum(pole['real'] > max(1e-7, summary['max_real_fd_difference']) for pole in eigenvalues)
        if eigenvalues else None
    )
    summary['detail_ref'] = graph_run.baseline.atomic_write(directory / 'spectrum.json', detail)
    return summary


def statistics(values):
    if any(value is None for value in values):
        return {'mean': None, 'sample_sd': None}
    return {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))}


def metric_vector(item):
    auto = item['score']['autonomous']
    return {
        **{f'stable_{h}s_risk': risk for h, risk in item['stable_motion_risk'].items()},
        'stable_8s_false_alarms': auto['stable_false_alarm_8s']['autonomous_trigger'],
        'unstable_8s_hits': auto['unstable_recall_8s']['autonomous_trigger'],
        'unstable_actual_risk_covered': item['unstable_actual_risk_covered'],
        'unstable_actual_hits': auto['unstable_actual_interval_hits'],
        'trigger_lag_median_s': auto['trigger_lag_s']['median'],
        'max_real_per_s': item['spectrum'].get('max_real'),
    }


def comparison(results):
    graph = [results[job] for job, (kind, _) in run.JOBS.items() if kind == 'graph']
    vectors = [metric_vector(item) for item in graph]
    aggregate = {key: statistics([row[key] for row in vectors]) for key in vectors[0]}
    pair_graph, pair_flat = results['graph_24111'], results['flat_24111']
    graph_vector, flat_vector = metric_vector(pair_graph), metric_vector(pair_flat)
    graph_hit_times, flat_hit_times = [], []
    n_graph_only = n_flat_only = 0
    left = run.read(pair_graph['score']['rows_ref'])
    right = run.read(pair_flat['score']['rows_ref'])
    run.require(len(left) == len(right) == 117, 'val row count differs in the pair')
    for a, b in zip(left, right, strict=True):
        run.require((a['event_id'], a['label']) == (b['event_id'], b['label']), 'val order differs in the pair')
        if a['label'] != 'unstable':
            continue
        ag, bf = [autonomous_outcome(r) == 'autonomous_trigger' for r in (a, b)]
        n_graph_only += ag and not bf
        n_flat_only += bf and not ag
        if ag and bf:
            graph_hit_times.append(a['classification_8s']['predicted_time_record_s'])
            flat_hit_times.append(b['classification_8s']['predicted_time_record_s'])
    common_deltas = np.asarray(graph_hit_times) - np.asarray(flat_hit_times)
    quantiles = [float(x) for x in np.quantile(common_deltas, (.25, .5, .75))] if common_deltas.size else [None] * 3
    paired = {
        'seed': 24111,
        'graph_minus_flat': {key: value - flat_vector[key] if value is not None and flat_vector[key] is not None else None
                             for key, value in graph_vector.items()},
        'unstable_shared_triggers': int(common_deltas.size),
        'unstable_graph_only_triggers': n_graph_only,
        'unstable_flat_only_triggers': n_flat_only,
        'shared_trigger_time_graph_minus_flat_s': dict(zip(('q1', 'median', 'q3'), quantiles, strict=True)),
    }
    risk = aggregate['stable_8s_risk']['mean']
    gates = {
        'all_four_numerically_qualified': all(item['score']['gate_passed'] for item in results.values()),
        'mean_graph_stable_8s_risk_below_0_50': risk is not None and risk < .50,
        'mean_graph_autonomous_hits_at_least_9_of_12': aggregate['unstable_8s_hits']['mean'] >= 9,
        'each_graph_stable_8s_false_alarms_at_most_1_of_95': all(row['stable_8s_false_alarms'] <= 1 for row in vectors),
    }
    accepted = all(gates.values())
    selected = min(graph, key=lambda item: (item['stable_motion_risk']['8'],
                                            -item['score']['autonomous']['unstable_recall_8s']['autonomous_trigger'],
                                            item['seed']))['seed'] if accepted else None
    return {'graph_mean_and_sample_sd': aggregate, 'matched_pair': paired, 'adoption_gates': gates,
            'adoptable': accepted, 'working_graph_seed': selected}


def evaluate(_raw, constants, _release, inputs, local, context):
    val = load_val(constants)
    val_identity = graph_run.baseline.data_identity(val)
    run.require(val_identity == inputs['anchor_analysis']['evaluation']['val']['data_identity'], 'original val digest differs')
    results = {}
    for job, (kind, seed) in run.JOBS.items():
        fit_receipt = inputs['fits'][job]
        fit = fit_receipt['result']
        checkpoint_ref = fit['checkpoints']['2048']
        saved = torch.load(run.verified(checkpoint_ref), map_location='cpu', weights_only=True)
        run.require(saved['attempt'] == 2048 and saved['release'] == fit_receipt['release'], 'endpoint release binding differs')
        cls = OSGKDGraph if kind == 'graph' else OSGKDMatchedFlat
        model = cls.from_checkpoint(saved['model']).eval().requires_grad_(False)
        directory = local / job
        scored = graph_run.score(model, val, constants, 'val', directory, context)
        pop = scored['autonomous']
        for key, raw_n, eligible_n in (('stable_false_alarm_8s', 95, 95), ('unstable_recall_8s', 22, 12)):
            run.require((pop[key]['raw_denominator'], pop[key]['input_eligible_denominator']) == (raw_n, eligible_n), 'val denominators differ')
        metrics = scored['summary']['splits']['val']
        stable = {}
        for h in (1, 2, 5, 8):
            official = metrics[f'stable_{h}s']['official']
            stable[str(h)] = official['motion_risk_v1'] if official is not None else None
        actual = metrics['unstable_actual']['covered_only_diagnostic']
        results[job] = {'seed': seed, 'encoder': kind, 'encoder_parameters': fit['encoder_parameters'],
                        'checkpoint_ref': checkpoint_ref, 'score': scored, 'stable_motion_risk': stable,
                        'unstable_actual_risk_covered': actual['motion_risk_v1'] if actual is not None else None,
                        'unstable_actual_risk_coverage': metrics['unstable_actual']['covered'],
                        'spectrum': endpoint_spectrum(model, directory)}
    return {'val': results, **comparison(results), 'endpoint': 2048, 'val_identity': val_identity, 'new_fom': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    run.execute(args.release, 'evaluate01', evaluate)
