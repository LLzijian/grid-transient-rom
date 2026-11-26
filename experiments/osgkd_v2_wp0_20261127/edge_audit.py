"""Cross-seed exchange contribution agreement on supplied train events."""

from dataclasses import asdict
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from experiments.osgkd_v2_discovery import prepare
from experiments.osgkd_v2_p2a_post_audit_20261124 import contributions
from kiki.models.osgkd_fitting import IntegrationWork
from kiki.models.osgkd_graph import OSGKDGraph


MODEL_KEYS = tuple(f'graph_{seed}' for seed in prepare.GRAPH_SEEDS)
PERMUTATIONS = 9999
PERMUTATION_SEED = 27031
PAIRS = tuple(combinations(range(10), 2))


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (45,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('45 finite nonnegative aligned exchange contributions required')
    order = np.argsort(values, kind='stable')
    ranks = np.empty(45, dtype=float)
    start = 0
    while start < 45:
        end = start + 1
        while end < 45 and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def top_indices(values, k):
    # Canonical pair order breaks boundary ties exactly as restriction_plan.
    return np.argsort(-np.asarray(values), kind='stable')[:k]


def overlap_null(k, intersection):
    denominator = comb(45, k)
    support = range(max(0, 2*k - 45), k + 1)
    probabilities = [(i, comb(k, i)*comb(45-k, k-i)/denominator) for i in support]
    return {'p_one_sided': sum(p for i, p in probabilities if i >= intersection),
            'expected_I': k*k/45, 'expected_F1': k/45,
            'expected_J': sum(p*i/(2*k-i) for i, p in probabilities)}


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [None] * len(pvalues)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(order)-rank)*pvalues[i]))
        adjusted[i] = running
    return adjusted


def statistics(scores):
    """Compute the fixed 12-test family without touching models or events."""
    if set(scores) != set(MODEL_KEYS):
        raise ValueError('the three registered graph seeds are required')
    values = {key: np.asarray(scores[key], dtype=float) for key in MODEL_KEYS}
    ranks = {key: average_ranks(value) for key, value in values.items()}
    centered = {key: value-value.mean() for key, value in ranks.items()}
    rng = np.random.default_rng(PERMUTATION_SEED)
    lookup = {edge: i for i, edge in enumerate(PAIRS)}
    # Relabel the second whole graph, preserving all incident-edge dependencies.
    mappings = np.empty((PERMUTATIONS, 45), dtype=int)
    for draw in range(PERMUTATIONS):
        permutation = rng.permutation(10)
        for i, (a, b) in enumerate(PAIRS):
            mappings[draw, i] = lookup[tuple(sorted((int(permutation[a]), int(permutation[b]))))]
    rows, family = [], []
    for first, second in combinations(MODEL_KEYS, 2):
        a, b = centered[first], centered[second]
        denominator = np.linalg.norm(a)*np.linalg.norm(b)
        rho = float(a @ b / denominator) if denominator else None
        permuted = values[second][mappings]
        null_rho = (centered[second][mappings] @ a / denominator
                    if denominator else np.zeros(PERMUTATIONS))
        p = ((1 + int(np.count_nonzero(null_rho >= rho - 1e-12))) / (PERMUTATIONS+1)
             if rho is not None else 1.)
        rank = {'rho': rho, 'permutation_p': p}
        family.append(rank)
        tops = {}
        for k in prepare.EDGE_BUDGETS:
            left, right = top_indices(values[first], k), top_indices(values[second], k)
            intersection = int(np.intersect1d(left, right).size)
            null_top = np.argsort(-permuted, axis=1, kind='stable')[:, :k]
            null_intersections = np.isin(null_top, left).sum(axis=1)
            top = {'I': intersection, 'F1': intersection/k, 'J': intersection/(2*k-intersection),
                   'hypergeometric': overlap_null(k, intersection),
                   'permutation_p': (1+int(np.count_nonzero(null_intersections >= intersection)))/(PERMUTATIONS+1),
                   'boundary_tie_first': bool(np.sort(values[first])[45-k] == np.sort(values[first])[44-k]),
                   'boundary_tie_second': bool(np.sort(values[second])[45-k] == np.sort(values[second])[44-k])}
            tops[str(k)] = top
            family.append(top)
        rows.append({'first': first, 'second': second, 'rank': rank, 'top': tops})
    for entry, adjusted in zip(family, holm([entry['permutation_p'] for entry in family]), strict=True):
        entry['holm_p'] = adjusted
    rank_gate = all(row['rank']['rho'] is not None and row['rank']['rho'] >= .70
                    and row['rank']['holm_p'] < .05 for row in rows)
    stable = {str(k): bool(rank_gate and all(row['top'][str(k)]['F1'] >= .75
                                           and row['top'][str(k)]['holm_p'] < .05 for row in rows))
              for k in prepare.EDGE_BUDGETS}
    return {'pairs': rows, 'strong_rank': rank_gate, 'stable_K': stable,
            'permutation_draws': PERMUTATIONS, 'permutation_seed': PERMUTATION_SEED,
            'holm_family_size': len(family), 'holm_family': '3 rank correlations and 9 top-set overlaps; node-label permutation p values',
            'ties': 'Spearman average ranks; top boundaries use ascending (left,right), including after node relabeling',
            'constant_rank': 'rho=null, permutation_p=1; rank gate fails',
            'interpretation': 'cross-seed agreement of functional contributions; failure of a gate does not establish true unidentifiability'}


def audit(models, train_events, local):
    """Write private diagnostics; return plans, work references, and agreement."""
    if set(models) != set(MODEL_KEYS):
        raise ValueError('the three registered graph seeds are required')
    for model in models.values():
        if type(model) is not OSGKDGraph or model.n != 10:
            raise ValueError('dense ten-node OSGKDGraph models required')
        edges = list(map(tuple, model.edge_index.T.tolist()))
        if len(edges) != 45 or set(edges) != set(PAIRS):
            raise ValueError('complete canonical undirected graph required')
    local = Path(local).resolve()
    local.mkdir(parents=True, exist_ok=False)
    per_seed, scores = {}, {}
    for key in MODEL_KEYS:
        directory = local / key
        directory.mkdir()
        work = IntegrationWork()
        measured = contributions.measure(models[key], train_events, work, directory)
        rows = prepare.graph_run.read(measured['local_edge_ref'])
        plans = prepare.restriction_plan(models[key], rows, measured['groups'])
        aligned = {(row['left'], row['right']): prepare.contribution(row['exchange']) for row in rows}
        scores[key] = [aligned[pair] for pair in PAIRS]
        details_ref = prepare.graph_run.baseline.atomic_write(directory / 'details.json', measured)
        work_ref = prepare.graph_run.baseline.atomic_write(directory / 'work.json', work.snapshot())
        per_seed[key] = {'details_ref': details_ref, 'work_ref': work_ref,
                         'diagnostic_rhs': measured['diagnostic_rhs'],
                         'restriction_plan': {name: asdict(spec) for name, spec in plans.items()},
                         'edge_ranking': [{'left': row['left'], 'right': row['right'],
                                           'edge_index': row['edge_index'], 'C_g': row['exchange']['C_g']}
                                          for row in sorted(rows, key=lambda row: (-row['exchange']['C_g'], row['left'], row['right']))]}
    return {'statistics': statistics(scores), 'per_seed': per_seed}
