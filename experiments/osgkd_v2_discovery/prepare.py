"""Rank, refit, and summarize fixed OSGKD structural restrictions.

Scientific data and release decisions are supplied by the experiment harness.
The table command aggregates saved scores and performs no fitting or evaluation.
"""

import argparse
from dataclasses import asdict
from itertools import combinations
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import fit, run as graph_run
from experiments.osgkd_v2_p2a_post_audit_20261124.contributions import measure
from experiments.osgkd_v2_sprint2a_20261125 import train as matched
from kiki.models.osgkd_baselines import ClosedPassiveKuramoto, GraphNeuralODE
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events
from kiki.models.osgkd_restricted import RestrictedOSGKDGraph, Restrictions


GRAPH_SEEDS = (24111, 24121, 24131)
EDGE_BUDGETS = (30, 20, 12)
STATIC_THRESHOLD = 0.0002
REFIT_ATTEMPTS = 128
VARIANTS = ('full', 'static', 'top30', 'top20', 'top12', 'no_beta', 'no_memory', 'no_coupling', 'frozen_p')
BASELINES = {'graph_neural_ode': GraphNeuralODE, 'closed_passive_kuramoto': ClosedPassiveKuramoto}


def contribution(value):
    score = value['C_g']
    if score is None or not math.isfinite(score) or score < 0:
        raise ValueError('finite nonnegative functional contribution required')
    return score


def restriction_plan(parent, edge_rows, groups):
    pairs = tuple(map(tuple, parent.edge_index.T.tolist()))
    indices = [row['edge_index'] for row in edge_rows]
    if sorted(indices) != list(range(len(pairs))) or len(pairs) < max(EDGE_BUDGETS):
        raise ValueError('a complete candidate graph ranking is required')
    if any((row['left'], row['right']) != pairs[row['edge_index']] for row in edge_rows):
        raise ValueError('ranked edge identity differs from the parent')
    ranked = sorted(edge_rows, key=lambda row: (-contribution(row['exchange']), row['left'], row['right']))
    static = tuple(row for row, name in enumerate(('environment_a', 'environment_a2'))
                   if contribution(groups[name]) < STATIC_THRESHOLD)
    return {
        'full': Restrictions(),
        'static': Restrictions(static_environment_rows=static),
        **{f'top{k}': Restrictions(tuple(row['edge_index'] for row in ranked[:k]), static) for k in EDGE_BUDGETS},
        'no_beta': Restrictions(remove_beta=True),
        'no_memory': Restrictions(remove_memory=True),
        'no_coupling': Restrictions(remove_coupling=True),
        'frozen_p': Restrictions(freeze_mechanical=True),
    }


def rank_train(parent, train_events, directory):
    """Measure the frozen parent on the supplied full train and persist the plan."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    work = IntegrationWork()
    measured = measure(parent, train_events, work, directory)
    rows = graph_run.read(measured['local_edge_ref'])
    restrictions = restriction_plan(parent, rows, measured['groups'])
    record = {'contributions': measured, 'work': work.snapshot(),
              'restrictions': {name: asdict(spec) for name, spec in restrictions.items()}}
    graph_run.baseline.atomic_write(directory / 'restriction_plan.json', record)
    return restrictions, record


def new_baseline(name, constants, seed):
    template = matched.new_model(constants, 'graph', seed)
    names = ('mass', 'damping', 'F', 'Sp', 'Se', 'omega_scale', 'theta_ref', 'edge_index')
    model = BASELINES[name](template.config, **{key: getattr(template, key) for key in names})
    model.encoder.load_state_dict(template.encoder.state_dict())
    return model


def _fit_planned(model, train_events, constants, common_step, recipe, start):
    if common_step not in (.01, .005):
        raise ValueError('the matched fits require a common RK4 mesh')
    batches = matched.plan(train_events)['batches']
    lookup = {event['id']: event for event in tensor_events(train_events, 'cpu', max_steps=800)}
    optimizer = recipe.new_optimizer(model)
    work, guard_work = IntegrationWork(), IntegrationWork()
    rows, guards, rejected, streak = [], [], 0, 0
    for k in range(start, recipe.ATTEMPTS):
        batch = [lookup[event_id] for event_id in batches[k]]
        horizon = recipe.horizon(k)
        if k == start or horizon != recipe.horizon(k - 1):
            guard_batch = batch if k == 0 else [batch[0], batch[-1]]
            guard = recipe.guard(model, guard_batch, constants, guard_work, limits=(800 if k == 0 else horizon,))
            guards.append({'attempt': k + 1, **guard})
            if common_step == .01 and not guard['passed']:
                break
        before = work.rhs['forward']
        row = recipe.attempt(model, optimizer, batch, k, constants, work, common_step)
        expected = 4 * round(.01 / common_step) * sum(min(horizon, len(e['targets']['theta']) - 1) for e in batch)
        if work.rhs['forward'] - before != expected:
            raise ValueError('integration count differs from the matched batch')
        rows.append(row)
        rejected += not row['accepted']
        streak = 0 if row['accepted'] else streak + 1
        if rejected >= 16 or streak >= 3:
            break
    return model, {'step': common_step, 'guards': guards, 'attempts': rows, 'rejected': rejected,
                   'valid_endpoint': len(rows) == recipe.ATTEMPTS - start and rejected < 16 and streak < 3,
                   'work': work.snapshot(), 'guard_work': guard_work.snapshot()}


def refit_variant(parent, restrictions, train_events, constants, *, common_step):
    """Refit from one frozen seed with the terminal 128 data-loss updates."""
    model = RestrictedOSGKDGraph.from_parent(parent, restrictions).train()
    return _fit_planned(model, train_events, constants, common_step, fit, fit.ATTEMPTS - REFIT_ATTEMPTS)


def fit_baseline(name, seed, train_events, constants, *, common_step):
    """Fit a fresh baseline with the entire Sprint 2A recipe and shared batches."""
    model = new_baseline(name, constants, seed)
    return _fit_planned(model, train_events, constants, common_step, matched, 0)


def _score_row(model, val_events, constants, directory, seed, variant):
    context = {'work': {}}
    score = graph_run.score(model.eval(), val_events, constants, 'val', directory, context)
    stable = score['autonomous']['stable_false_alarm_8s']
    unstable = score['autonomous']['unstable_recall_8s']
    if (stable['raw_denominator'], unstable['input_eligible_denominator'], unstable['raw_denominator']) != (95, 12, 22):
        raise ValueError('the original val coverage is required')
    return {'seed': seed, 'variant': variant, 'qualified': score['gate_passed'],
            'stable_risk_8s': score['stable_8s_motion_risk'],
            'stable_false_alarms': stable['autonomous_trigger'], 'unstable_8s_hits': unstable['autonomous_trigger'],
            'audit': score}


def score_variant(model, val_events, constants, directory, *, seed, variant):
    """Use the existing dual-grid evaluator on an explicitly supplied val."""
    row = _score_row(model, val_events, constants, directory, seed, variant)
    exchange_edges = model.edge_index.T[model.c.detach() > 0].tolist()
    return {**row, 'exchange_support': exchange_edges, **model.complexity()}


def score_baseline(model, val_events, constants, directory, *, seed, name):
    if not isinstance(model, BASELINES[name]):
        raise ValueError('baseline name and model must agree')
    row = _score_row(model, val_events, constants, directory, seed, name)
    encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    total = sum(parameter.numel() for parameter in model.parameters())
    return {**row, 'encoder_parameters': encoder,
            'effective_encoder_parameters': 0 if isinstance(model, ClosedPassiveKuramoto) else encoder,
            'ode_parameters': total - encoder, 'embedded_state_dimension': 5 * model.n,
            'effective_state_dimension': getattr(model, 'effective_state_dimension', 5 * model.n)}


def support_agreement(first, second):
    a, b = (set(map(tuple, edges)) for edges in (first, second))
    intersection = len(a & b)
    return {'jaccard': intersection / len(a | b) if a or b else None,
            'support_f1': 2 * intersection / (len(a) + len(b)) if a or b else None}


def _pareto(rows):
    axes = {row['variant']: (row['active_edge_paths'], row['active_module_groups'], row['parameters'], row['stable_risk_8s'],
                             row['stable_false_alarms'], -row['unstable_8s_hits']) for row in rows}
    return [name for name, a in axes.items() if not any(
        all(x <= y for x, y in zip(b, a)) and any(x < y for x, y in zip(b, a)) for b in axes.values())]


def tables(rows):
    """Assemble Pareto, paired module effects, and all three seed comparisons."""
    indexed = {(row['seed'], row['variant']): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {(seed, name) for seed in GRAPH_SEEDS for name in VARIANTS}:
        raise ValueError('one row for each registered seed and variant is required')
    metrics = ('stable_risk_8s', 'stable_false_alarms', 'unstable_8s_hits', 'active_edge_paths', 'active_module_groups', 'parameters')
    summary, agreement, differences, by_seed = {}, {}, {}, {}
    for name in VARIANTS:
        matched = [indexed[seed, name] for seed in GRAPH_SEEDS]
        for row in matched:
            if row['qualified'] and any(row[key] is None or not math.isfinite(row[key]) for key in metrics):
                raise ValueError('qualified comparisons require finite metrics')
        qualified = [row for row in matched if row['qualified']]
        summary[name] = {
            'qualified_seeds': len(qualified),
            'mean_and_sample_sd': {key: {'mean': statistics.mean(row[key] for row in matched),
                                         'sample_sd': statistics.stdev(row[key] for row in matched)}
                                   for key in metrics} if len(qualified) == len(GRAPH_SEEDS) else None,
        }
        agreement[name] = [{'seeds': [a['seed'], b['seed']], **support_agreement(a['exchange_support'], b['exchange_support'])}
                           for a, b in combinations(qualified, 2)]
    for seed in GRAPH_SEEDS:
        dense = indexed[seed, 'full']
        qualified = [indexed[seed, name] for name in VARIANTS if indexed[seed, name]['qualified']]
        by_seed[seed] = _pareto(qualified)
        differences[seed] = {row['variant']: {key: row[key] - dense[key] for key in metrics[:3]}
                             for row in qualified if dense['qualified']}
    return {'denominators': {'stable': 95, 'unstable_eligible': 12, 'unstable_raw': 22},
            'rows': [{key: row[key] for key in ('seed', 'variant', 'qualified', *metrics)} for row in rows],
            'seed_summary': summary, 'support_agreement': agreement, 'pareto_by_seed': by_seed,
            'paired_deltas_from_refit_full': differences}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path, help='JSON list of already scored, fixed endpoints')
    args = parser.parse_args()
    print(json.dumps(tables(json.loads(args.scores.read_bytes())), indent=2, allow_nan=False))
