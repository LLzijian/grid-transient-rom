"""Thin frozen-metric test dispatch; no loading, endpoint selection or authority."""

from collections import defaultdict
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import torch

from experiments.osgkd_v2_fast_zero_sum_20261124 import scoring
from kiki.evaluation.osgkd_autonomous import summarize_autonomous_8s
from kiki.models.osgkd_fitting import IntegrationWork


def aggregate(rows, events, constants):
    """Map unchanged metric keys to test, retaining every original denominator."""
    mapped = scoring.summarize(rows, {**constants, 'fit_ids': [], 'check_ids': []})
    metrics = mapped['splits']['train_all']
    for name, metric in metrics.items():
        if name.startswith('stable_'):
            scope = name.split('_')[-1][:-1]
            if any(not row['scopes'][scope]['full_horizon_truth'] or row['scopes'][scope]['metrics'] is None
                   for row in rows if row['label'] == 'stable'):
                metric['official'] = None
    summary = dict(metric=mapped['metric'], splits={'test': metrics},
                   classification={'test': mapped['classification']['train_all']},
                   coverage=dict(denominator=len(events), input_eligible=sum(event['history'] is not None for event in events),
                                 missing_history_ids=[event['id'] for event in events if event['history'] is None],
                                 stable=sum(event['label'] == 'stable' for event in events),
                                 unstable=sum(event['label'] == 'unstable' for event in events)))
    return dict(summary=scoring.baseline.public_summary(summary), autonomous=summarize_autonomous_8s(rows),
                timing=scoring.baseline.timing_summary(rows, events))


@torch.no_grad()
def score_endpoint(model, events, constants, directory):
    """Traverse an already cached bank once; never re-open trajectory sources."""
    if not events or len({event['id'] for event in events}) != len(events):
        raise ValueError('nonempty unique confirmation events required')
    if model.n != 10:
        raise ValueError('ten-machine endpoint required')
    if model.mass.device.type != 'cpu' or model.mass.dtype != torch.float64:
        raise ValueError('frozen scoring requires CPU float64')
    directory = Path(directory).resolve()
    if not directory.is_relative_to((scoring.baseline.ROOT/'artifacts').resolve()):
        raise ValueError('confirmation rows must be written below the private artifacts directory')
    directory.mkdir(parents=True, exist_ok=False)
    work, indexed, cohorts = IntegrationWork(), {}, defaultdict(list)
    started, cpu = perf_counter(), process_time()
    expected = 0
    was_training = model.training
    model.eval()
    try:
        for event in events:
            if event['history'] is None:
                indexed[event['id']] = scoring.missing_row(event)
            else:
                count = max(800, min(1500, len(event['t'])-1)) if event['label'] == 'unstable' else 800
                cohorts[count].append(event)
        for count, cohort in cohorts.items():
            for start in range(0, len(cohort), 16):
                batch = cohort[start:start+16]
                theta, omega = (torch.as_tensor(np.stack([event['history'][key] for event in batch]),
                                                dtype=torch.float64, device='cpu') for key in ('theta', 'omega'))
                initial = model.initialize(theta, omega)
                work.encoder_events += len(batch)
                wanted_shape = (len(batch), 6*model.n if hasattr(model, 'fast_gamma') else 5*model.n)
                if initial.shape != wanted_shape:
                    raise ValueError('state shape differs from frozen scoring')
                for actual, observed in ((initial[:, :10], theta[:, -1]), (initial[:, 10:20], omega[:, -1])):
                    if not torch.equal(actual.contiguous().view(torch.int64), observed.contiguous().view(torch.int64)):
                        raise ValueError('observed initialization endpoints changed')
                main = scoring.integrate(model, initial, count, internal_step=.005, work=work)
                fine = scoring.integrate(model, initial, count, internal_step=.0025, work=work)
                ports = model.ports(fine)
                main, fine = main.numpy(), fine.numpy()
                output = {key: getattr(ports, key).numpy() for key in ('tau_m', 'tau_e')}
                for index, event in enumerate(batch):
                    indexed[event['id']] = scoring.augmented_row(event, main[index], fine[index],
                                            {key: value[index] for key, value in output.items()}, constants)
                expected += 24*count*len(batch)
    finally:
        model.train(was_training)
    if work.rhs['forward'] != expected or work.rhs['recompute'] != 0:
        raise ValueError('frozen confirmation RHS accounting differs')
    rows = [indexed[event['id']] for event in events]
    result = aggregate(rows, events, constants)
    sources, groups = defaultdict(list), defaultdict(list)
    for event in events:
        sources[event['metadata']['source_root']].append(event)
        groups[event['group']].append(event)
    result['by_source'] = {name: aggregate([indexed[event['id']] for event in cohort], cohort, constants)
                           for name, cohort in sorted(sources.items())}
    result['by_group'] = {name: aggregate([indexed[event['id']] for event in cohort], cohort, constants)
                          for name, cohort in sorted(groups.items())}
    result.update(data_identity=scoring.baseline.data_identity(events), work=work.snapshot(),
                  cost=dict(wall_s=perf_counter()-started, cpu_s=process_time()-cpu))
    rows_ref = scoring.baseline.atomic_write(directory/'rows.json', rows)
    metrics_ref = scoring.baseline.atomic_write(directory/'metrics.json', result)
    completed_ref = scoring.baseline.atomic_write(directory/'completed.json',
                          dict(status='completed', rows_ref=rows_ref, metrics_ref=metrics_ref))
    result.update(rows_ref=rows_ref, metrics_ref=metrics_ref, completed_ref=completed_ref)
    scoring.baseline.atomic_write(directory/'endpoint_summary.json', result)
    return result
