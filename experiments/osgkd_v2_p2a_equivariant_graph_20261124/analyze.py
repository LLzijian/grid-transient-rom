"""Audit the scored endpoints and compare their initial states without new rollouts."""

import json
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import torch

from experiments.osgkd_v2_early_closure_20261123.run import identity, read, require, verified
from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import diagnostics
from experiments.osgkd_v2_train385_baseline_20261123 import run as baseline
from experiments.osgkd_v2_train385_risk_alignment_20261123 import analyze as earlier
from kiki.data.osgkd_train_val import load_train, load_val
from kiki.evaluation.osgkd_autonomous import summarize_autonomous_8s
from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_graph import OSGKDGraph

RUN = 'osgkd_v2_p2a_equivariant_graph_20261124'
ANCHOR = 'results/osgkd_v2_sprint0_20261124/analysis.json'
ANCHOR_SHA = '82b87d068a884143a95ccc800a133575a14ffa146d857753943039085028538c'


def read_job(job):
    ref = identity(ROOT / 'results' / RUN / job / 'summary.json')
    summary = read(ref)
    require((summary['run_id'], summary['job_id'], summary['status']) == (RUN, job, 'completed'), 'job incomplete')
    done = identity(ROOT / 'artifacts' / RUN / job / 'completed.json')
    require(read(done) == {'status': 'completed', 'summary_ref': ref}, 'completion identity differs')
    require(read(summary['release_ref']) == summary['release'], 'release identity differs')
    return ref, summary


def native_score(record, split, reference):
    rows, metrics = earlier.read_score(record, split)
    ref_rows = read(reference['rows_ref'])
    def population(items):
        return {(row['event_id'], row['label'], row['actual_future_points']) for row in items}
    require(population(rows) == population(ref_rows) and record['data_identity'] == reference['data_identity'],
            'paired evaluation population differs')
    autonomous = summarize_autonomous_8s(rows)
    require(autonomous == record['autonomous'], 'autonomous metrics differ')
    values = metrics['splits'][split]
    actual = values['unstable_actual']['official'] or values['unstable_actual']['covered_only_diagnostic']
    details = earlier.describe(rows, metrics, split)
    details.update(autonomous=autonomous, stable_8s_motion_risk=record['stable_8s_motion_risk'],
                   unstable_actual_motion_risk=actual['motion_risk_v1'] if actual else None,
                   paired_trigger_lag=diagnostics.paired_trigger_lag(ref_rows, rows),
                   rows_ref=record['rows_ref'])
    full_8s = sum(row['classification_8s'] is not None and row['classification_8s']['los_pair_pass'] for row in rows)
    qualified = earlier.qualification(rows, split) and full_8s == (385 if split == 'train' else 107)
    details['numerical_qualified_all'] = qualified
    return details


def val_gates(details, param_count, flat_params):
    score = details['autonomous']
    unstable, stable = score['unstable_recall_8s'], score['stable_false_alarm_8s']
    require((stable['raw_denominator'], unstable['raw_denominator'], unstable['input_eligible_denominator']) == (95,22,12),
            'original val denominators differ')
    risk = details['stable_8s_motion_risk']
    return dict(numerical=details['numerical_qualified_all'], stable_risk=risk is not None and risk <= .55,
                false_alarm=stable['autonomous_trigger'] <= 1, autonomous_recall=unstable['autonomous_trigger'] >= 8,
                encoder_complexity=param_count <= .4 * flat_params)


def main():
    wall, cpu = perf_counter(), process_time()
    torch.set_num_threads(1)
    fit_ref, fit = read_job('fit01')
    evaluation_ref, evaluation = read_job('evaluate01')
    require(evaluation['release']['fit_ref'] == fit_ref, 'score fit identity differs')
    fit_result, result = fit['result'], evaluation['result']
    require(fit_result['valid_endpoint'] and fit_result['actual_endpoint'] == 2048 and
            fit_result['accepted'] >= 2033 and set(fit_result['checkpoints']) == {'1792','2048'}, 'ineligible fit')
    anchor_ref = identity(ROOT / ANCHOR)
    require(anchor_ref['sha256'] == ANCHOR_SHA, 'flat anchor differs')
    anchor = read(anchor_ref)
    nparam = fit_result['encoder_parameters']
    val_points = {}
    for step in (1792, 2048):
        score = result['val'][str(step)]
        point = native_score(score, 'val', anchor['evaluation']['val'])
        point['gates'] = val_gates(point, nparam, anchor['encoder_parameters'])
        require(score['gate_passed'] == point['numerical_qualified_all'] and
                score['selection_gate_passed'] == all(point['gates'].values()), 'val endpoint gate differs')
        val_points[step] = point
    candidates = [step for step, score in val_points.items() if all(score['gates'].values())]
    picked = min(candidates, key=lambda step: (val_points[step]['stable_8s_motion_risk'],
                 -val_points[step]['autonomous']['unstable_recall_8s']['autonomous_trigger'], step)) if candidates else None
    require(result['selected_endpoint'] == picked and result['primary_endpoint'] == (picked or 2048), 'validation selection differs')
    primary = result['primary_endpoint']
    train_score = native_score(result['train'], 'train', anchor['evaluation']['train'])
    adoptable = picked is not None and train_score['numerical_qualified_all']
    require(result['adoptable'] == adoptable, 'adoption qualification differs')

    train, constants = load_train()
    data = {'train': train, 'val': load_val(constants)}
    for split, events in data.items():
        require(baseline.data_identity(events) == anchor['evaluation'][split]['data_identity'], 'diagnostic population differs')
    raw_flat = torch.load(verified(anchor['anchor']), map_location='cpu', weights_only=True)
    raw_graph = torch.load(verified(fit_result['checkpoints'][str(primary)]), map_location='cpu', weights_only=True)
    flat = OSGKD.from_checkpoint(raw_flat['model']).eval()
    graph = OSGKDGraph.from_checkpoint(raw_graph['model']).eval()
    require(sum(p.numel() for p in graph.encoder.parameters()) == nparam == 1928, 'graph size differs')
    init = {name: {split: diagnostics.initial_amplitude(model, events) for split, events in data.items()}
            for name, model in (('flat', flat), ('graph', graph))}
    output = dict(run_id=RUN, fit_ref=fit_ref, evaluation_ref=evaluation_ref, anchor_ref=anchor_ref,
                  analysis_source=identity(Path(__file__).resolve()), diagnostic_source=identity(Path(diagnostics.__file__).resolve()),
                  primary_endpoint=primary, selected_endpoint=picked, gate_eligible=adoptable,
                  encoder_parameters=nparam, encoder_reduction=1 - nparam / anchor['encoder_parameters'],
                  train=train_score, val={str(step): value for step, value in val_points.items()}, initial_amplitude=init,
                  new_rollout_rhs=0, optimizer_updates=0, cost=dict(wall_s=perf_counter()-wall,cpu_s=process_time()-cpu))
    print(json.dumps(baseline.atomic_write(ROOT/'results'/RUN/'analysis.json', output, public=True)))


if __name__ == '__main__':
    main()
