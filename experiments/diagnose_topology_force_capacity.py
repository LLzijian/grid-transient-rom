"""Noncausal forcing controls isolate geometry, coordinate and ODE failures."""
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from run_private_m2 import aggregate

from kiki.data.private_dataset import load_private_event
from kiki.models.private_pm import MechanicalROM
from kiki.physics import coi_project


def forced_rollout(base, initial, forcing, true_pm=None, step=.005):
    state = initial.copy()
    output = [state.copy()]
    for k in range(forcing.shape[1]-1):
        def field(value, fraction, k=k):
            value = value.copy()
            if true_pm is not None:
                value[..., 20:30] = (1-fraction)*true_pm[:, k]+fraction*true_pm[:, k+1]
            derivative = base.field(value)
            derivative[..., 10:20] += ((1-fraction)*forcing[:, k]+fraction*forcing[:, k+1])/base.mass
            if true_pm is not None:
                derivative[..., 20:30] = (true_pm[:, k+1]-true_pm[:, k])/.01
            return derivative
        count = round(.01/step)
        dt = .01/count
        for j in range(count):
            a = field(state, j/count)
            b = field(state+.5*dt*a, (j+.5)/count)
            c = field(state+.5*dt*b, (j+.5)/count)
            d = field(state+dt*c, (j+1)/count)
            state += dt*(a+2*b+2*c+d)/6
        if true_pm is not None:
            state[..., 20:30] = true_pm[:, k+1]
        output.append(state.copy())
    result = np.stack(output, axis=1)
    assert np.isfinite(result).all()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = MechanicalROM.load('private_data/task2_m3_0020/model.npz')
    coordinate = np.load('private_data/task2_topology_passive_0042/fit.npz')
    v = coordinate['direction']
    atlas = json.loads(Path('private_data/task2_residual_atlas_0040/evidence.json').read_text())
    events, excluded, metadata = [], [], {}
    for root in (Path('private_data/ieee39_postfault'), Path('private_data/ieee39_expansion')):
        split = json.loads((root/'splits.json').read_text())
        with (root/'dataset_index.csv').open() as stream:
            rows = {r['event_id']: r for r in csv.DictReader(stream)}
        metadata.update(rows)
        with h5py.File(root/'dataset.h5', 'r') as file:
            eq = file['equilibrium/full_state'][:]
            for eid in split['train']:
                assert eid not in split['validation'] and eid not in split['test']
                if float(rows[eid]['trajectory_end_s']) <= .05+1e-9:
                    excluded.append(eid)
                    continue
                event = load_private_event(root/'dataset.h5', eid)
                for key in ('t', 'delta_sync', 'omega_sync', 'Pm_raw'):
                    event[key] = event[key][5:806]
                event['t'] = event['t']-event['t'][0]
                event['event_id'] = eid
                raw = file['events'][eid]
                te = raw['te'][5:806]
                sine = np.sin((event['delta_sync']+base.delta_star)@base.incidence)-np.sin(base.delta_star@base.incidence)
                residual = base.pm_star-te+(sine*base.coupling)@base.incidence.T+base.damping*event['omega_sync']
                hidden = raw['full_state'][5:806]
                amplitude = (hidden[:, coordinate['columns']]-eq[coordinate['columns']])@coordinate['W']
                event['forces'] = {'M3': np.zeros_like(residual), 'rank1_true': (residual@v)[:, None]*v,
                    'rank1_hidden': -amplitude[:, None]*v, 'balanced_true': residual-residual.mean(axis=1, keepdims=True),
                    'full_true': residual, 'full_true_pm': residual}
                events.append(event)
    assert len(events) == 351 and len(excluded) == 18
    cohorts = defaultdict(list)
    for event in events:
        cohorts[len(event['t'])].append(event)
    records, artifacts = [], {}
    scales = np.array([.47675314588683515, 1.0033667249602924])
    for length, cohort in sorted(cohorts.items()):
        initial = np.array([np.r_[e['delta_sync'][0], e['omega_sync'][0], e['Pm_raw'][0]] for e in cohort])
        for arm in cohort[0]['forces']:
            forcing = np.array([e['forces'][arm] for e in cohort])
            pm = np.array([e['Pm_raw'] for e in cohort]) if arm == 'full_true_pm' else None
            predictions = forced_rollout(base, initial, forcing, pm)
            for pred, event in zip(predictions, cohort, strict=True):
                eid = event['event_id']
                artifacts[eid+'_'+arm] = pred
                artifacts[eid+'_'+arm+'_force'] = event['forces'][arm]
                artifacts[eid+'_t'] = event['t']
                err_d = coi_project(pred[:, :10]-event['delta_sync'], base.mass)
                err_w = coi_project(pred[:, 10:20]-event['omega_sync'], base.mass)
                for horizon in (.2, .5, 1., 2., 5., 8., 'prefix'):
                    count = length-1 if horizon == 'prefix' else round(horizon/.01)
                    if count >= length:
                        continue
                    ds, ws = float(np.sum(err_d[1:count+1]**2)), float(np.sum(err_w[1:count+1]**2))
                    sse = .5*(ds/scales[0]**2+ws/scales[1]**2)
                    records.append({'event_id': eid, 'group': metadata[eid]['fault_bus'], 'label': metadata[eid]['label'],
                        'arm': arm, 'horizon': str(horizon), 'n': count*10, 'delta_sse': ds, 'omega_sse': ws,
                        'normalized_sse': sse, 'rmse': float(np.sqrt(sse/(count*10)))})
        print('completed available-prefix cohort', length, len(cohort), flush=True)
    summaries = {arm: {h: aggregate([r for r in records if r['arm'] == arm and r['horizon'] == h])
                      for h in ('0.2', '0.5', '1.0', '2.0', '5.0', '8.0', 'prefix')} for arm in events[0]['forces']}
    controls = [r['rmse'] for r in records if r['arm'] == 'full_true_pm' and r['horizon'] == 'prefix']
    quantiles = np.quantile(controls, [.5, .95, 1])
    representatives = [next(e for e in events if e['event_id'].startswith(prefix) and metadata[e['event_id']]['label'] == 'stable') for prefix in ('pilot', 'expansion')]
    representatives.append(max((e for e in events if metadata[e['event_id']]['label'] == 'unstable'), key=lambda e: len(e['t'])))
    half_records = []
    for event in representatives:
        initial = np.array([np.r_[event['delta_sync'][0], event['omega_sync'][0], event['Pm_raw'][0]]])
        for arm, force in event['forces'].items():
            pm = event['Pm_raw'][None] if arm == 'full_true_pm' else None
            half = forced_rollout(base, initial, force[None], pm, step=.0025)[0]
            difference = float(np.max(abs(half-artifacts[event['event_id']+'_'+arm]))/max(1., np.max(abs(half))))
            half_records.append({'event_id': event['event_id'], 'arm': arm, 'relative_error': difference})
    capacity = {arm: all(summaries[arm][h]['mean'] <= .95*summaries['M3'][h]['mean']
                         and summaries[arm][h]['pooled'] <= summaries['M3'][h]['pooled']
                         and summaries[arm][h]['cvar10'] <= 1.05*summaries['M3'][h]['cvar10'] for h in ('2.0', '5.0', '8.0'))
                for arm in summaries if arm != 'M3'}
    passed = bool(quantiles[0] < .01 and quantiles[1] < .05 and max(r['relative_error'] for r in half_records) < 1e-3)
    result = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dataset_hashes': atlas['dataset_hashes'], 'baseline_sha256': atlas['model_sha256'], 'events': len(events),
        'groups': len({metadata[e['event_id']]['fault_bus'] for e in events}), 'excluded_predecision': excluded,
        'numerical_control_prefix_rmse_quantiles': quantiles.tolist(), 'half_step': half_records,
        'diagnostic_pass': passed, 'noncausal_capacity_screen': capacity, 'summary': summaries, 'records': records,
        'scope': 'TRAIN-only noncausal forcing diagnostic; no autonomous memory or stability classification claim'}
    np.savez_compressed(args.output/'trajectories.npz', **artifacts)
    (args.output/'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'summary', 'half_step', 'excluded_predecision')}, indent=2))
    print(json.dumps({arm: summaries[arm]['8.0'] for arm in summaries}, indent=2))


if __name__ == '__main__':
    main()
