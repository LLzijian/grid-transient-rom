"""Post-fit initial-amplitude and paired trigger-lag diagnostics for saved scores."""

from collections import defaultdict

import numpy as np
import torch

from kiki.evaluation.osgkd_autonomous import autonomous_outcome


def quantiles(values):
    if not len(values):
        return dict(count=0, q1=None, median=None, q3=None, iqr=None)
    q1, median, q3 = (float(x) for x in np.quantile(values, (.25, .5, .75), method='linear'))
    return dict(count=len(values), q1=q1, median=median, q3=q3, iqr=q3 - q1)


@torch.no_grad()
def initial_amplitude(model, events):
    """Only motion histories enter initialize; labels are used after inference."""
    cohorts = defaultdict(list)
    initialized = [event for event in events if event['history'] is not None]
    edges = model.c.clamp_min(0)
    modulated = (edges > 0) & (model.beta > 0)
    for start in range(0, len(initialized), 16):
        batch = initialized[start:start+16]
        theta, omega = (torch.as_tensor(np.stack([e['history'][key] for e in batch]), dtype=torch.float64)
                        for key in ('theta', 'omega'))
        a = model.initialize(theta, omega)[..., 3*model.n:4*model.n]
        ratio = model.edge_kernel(a)[..., modulated] / edges[modulated]
        for index, event in enumerate(batch):
            cohorts[event['label']].append((a[index].numpy(), ratio[index].numpy()))

    result = {}
    for label in ('stable', 'unstable'):
        group = cohorts[label]
        amplitude = np.stack([row[0] for row in group])
        coupling = np.stack([row[1] for row in group])
        if not np.isfinite(amplitude).all() or not np.isfinite(coupling).all():
            raise ValueError('nonfinite initial diagnostic')
        result[label] = dict(initialized_events=len(group), raw_events=sum(event['label'] == label for event in events),
            a0_mean=float(amplitude.mean()), a0_fraction_negative=float((amplitude < 0).mean()),
            a0_events_mean_negative=int((amplitude.mean(axis=1) < 0).sum()),
            a0_fraction_at_99pct_bound=float((np.abs(amplitude) >= .99 * model.config.a_init_bound).mean()),
            a0_event_mean_quantiles=quantiles(amplitude.mean(axis=1)),
            modulated_edges=int(modulated.sum()), candidate_edges=int(edges.numel()),
            modulated_edge_k_over_c_mean=float(coupling.mean()) if coupling.size else None,
            modulated_edge_fraction_weakened=float((coupling < 1 - 1e-12).mean()) if coupling.size else None)
    return result


def paired_trigger_lag(anchor_rows, graph_rows):
    def lag(rows):
        out = {}
        for row in rows:
            if row['label'] != 'unstable' or autonomous_outcome(row) != 'autonomous_trigger':
                continue
            choice = row['classification_8s']
            true = choice['true_time_record_s']
            if true is not None and np.isfinite(true):
                out[row['event_id']] = (true, choice['predicted_time_record_s'])
        return out

    anchor, graph = lag(anchor_rows), lag(graph_rows)
    common = sorted(anchor.keys() & graph.keys())
    if any(anchor[key][0] != graph[key][0] for key in common):
        raise ValueError('paired lag references differ')
    return dict(common_triggered=len(common), only_anchor=len(anchor.keys() - graph.keys()),
                only_graph=len(graph.keys() - anchor.keys()),
                graph_minus_flat_same_event_s=quantiles([graph[key][1] - anchor[key][1] for key in common]))
