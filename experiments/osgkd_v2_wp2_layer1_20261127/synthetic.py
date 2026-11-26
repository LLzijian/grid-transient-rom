"""Independent NumPy truth and legal observations for WP2; no learner imports."""

from copy import deepcopy
from itertools import combinations

import numpy as np

from kiki.data import osgkd_synthetic as truth_model


SYSTEM_SEED = 31
NOISE_LEVELS = (0., .01, .03)
EXCITATIONS = ('distributed', 'local', 'mechanical', 'environment')
CONDITIONS = ('static', 'modulated', 'open')
FORECAST_S = 4.
HISTORY_S = .1
TRAIN_SEED = 1000
VAL_SEED = 2000


def noise_key(level):
    return f'{level:.2f}'


def full_offset_lift(copies, source_n=8):
    """Map every candidate edge, irrespective of truth or learned support.

    Node i + r*source_n has role i. Same-role clone edges have no source edge.
    Cross-role weights divide by copies, preserving per-clone total coupling.
    """
    if type(copies) is not int or copies not in (1, 2, 4) or source_n != 8:
        raise ValueError('WP2 uses eight roles and 1/2/4 offset copies')
    n = source_n*copies
    node_source = np.arange(n) % source_n
    source_edges = np.array(list(combinations(range(source_n), 2)), dtype=int)
    target_edges = np.array(list(combinations(range(n), 2)), dtype=int)
    lookup = {tuple(edge): index for index, edge in enumerate(source_edges.tolist())}
    edge_source = np.array([lookup.get(tuple(sorted((int(node_source[a]), int(node_source[b])))), -1)
                            for a, b in target_edges], dtype=int)
    return dict(node_source=node_source, source_edges=source_edges, target_edges=target_edges,
                edge_source=edge_source, edge_weight=np.where(edge_source >= 0, 1/copies, 0.))


def lift_system(base, copies):
    """Private truth lift: k squared clone pairs per true role edge."""
    if base['n'] != 8:
        raise ValueError('lift requires the eight-role source system')
    mapping = full_offset_lift(copies)
    source_lookup = {tuple(edge): i for i, edge in enumerate(base['edges'].tolist())}
    edges, c, beta = [], [], []
    for pair, source, weight in zip(mapping['target_edges'], mapping['edge_source'], mapping['edge_weight'], strict=True):
        if source < 0:
            continue
        index = source_lookup.get(tuple(mapping['source_edges'][source]))
        if index is not None:
            edges.append(pair)
            c.append(base['c'][index]*weight)
            beta.append(base['beta'][index]*weight)
    result = deepcopy(base)
    result.update(n=8*copies, edges=np.asarray(edges, dtype=int), c=np.asarray(c), beta=np.asarray(beta),
                  candidate_edges=mapping['target_edges'].copy())
    for name in ('mass', 'damping', 'reference', 'environment_gain'):
        result[name] = base[name][mapping['node_source']].copy()
    result['lift'] = dict(copies=copies, node_source=mapping['node_source'].copy(),
                          convention='node=role+8*copy; every cross-role clone pair weighted 1/copies')
    return result


def make_system(n=8, condition='open', seed=SYSTEM_SEED):
    """Three conditions share graph, mass, damping and baseline c exactly."""
    if n not in (8, 16, 32) or condition not in CONDITIONS:
        raise ValueError('WP2 requires N=8/16/32 and static/modulated/open')
    base = truth_model.make_system(n=8, seed=seed, condition='open')
    base['condition'] = condition
    if condition == 'static':
        base['beta'].fill(0.)
    if condition != 'open':
        base['environment_gain'].fill(0.)
    return base if n == 8 else lift_system(base, n//8)


def generate_episode(system, episode_seed, excitation, noise_levels=NOISE_LEVELS,
                     forecast_s=FORECAST_S, internal_dt=.0025):
    """Integrate once, then reuse truth and Gaussian noise direction at all levels.

    Noise units are the frozen unit angle/speed scales. Only 11-frame history
    is corrupted; all future targets and port supervision remain exact.
    """
    if excitation not in EXCITATIONS:
        raise ValueError('one of the four stratified WP2 excitations required')
    if not np.isfinite(forecast_s) or forecast_s <= 0:
        raise ValueError('positive forecast duration required')
    levels = tuple(float(level) for level in noise_levels)
    if not levels or any(not np.isfinite(level) or level < 0 for level in levels) or len(set(levels)) != len(levels):
        raise ValueError('distinct finite nonnegative noise levels required')
    if len({noise_key(level) for level in levels}) != len(levels):
        raise ValueError('noise levels must have distinct two-decimal keys')
    event, truth = truth_model.generate_event(system, seed=episode_seed, excitation=excitation,
                   duration=HISTORY_S+forecast_s, internal_dt=internal_dt, noise_std=0.)
    rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 270311]))
    direction = {name: rng.standard_normal(event['history'][name].shape) for name in ('theta', 'omega')}
    events = {}
    for level in levels:
        observed = deepcopy(event)
        for name in direction:
            observed['history'][name] += level*direction[name]
        events[noise_key(level)] = observed
    truth['noise_direction'] = direction
    truth['noise_levels'] = levels
    truth['forecast_s'] = forecast_s
    return events, truth


def make_dataset(n=8, condition='open', train_count=48, val_count=24,
                 noise_levels=NOISE_LEVELS, forecast_s=FORECAST_S, internal_dt=.0025,
                 system_seed=SYSTEM_SEED, train_seed=TRAIN_SEED, val_seed=VAL_SEED):
    """Return (public_pack, private_truth), with independent stratified episodes.

    Public train/val maps noise strings to lists of conventional events;
    constants come exclusively from motion_packet. Private structure/state is
    returned separately for locked scoring and never needed to construct input.
    """
    for count in (train_count, val_count):
        if type(count) is not int or count <= 0 or count % len(EXCITATIONS):
            raise ValueError('episode counts must be positive multiples of four')
    train_ids = set(range(train_seed, train_seed+train_count))
    val_ids = set(range(val_seed, val_seed+val_count))
    if train_ids & val_ids:
        raise ValueError('train and independent val episode seeds overlap')
    system = make_system(n, condition, system_seed)
    public = dict(train={}, val={}, constants=None,
                  contract=dict(history_points=11, history_s=.1, forecast_s=forecast_s,
                                output_step=.01, truth_internal_step=internal_dt,
                                noise_scope='history_only; unit angle and speed scales',
                                noise_levels=list(noise_levels)))
    private = dict(system=deepcopy(system), train=[], val=[])
    for split, count, seed in (('train', train_count, train_seed), ('val', val_count, val_seed)):
        for i in range(count):
            observations, truth = generate_episode(system, seed+i, EXCITATIONS[i // (count//4)],
                                                   noise_levels, forecast_s, internal_dt)
            for key, event in observations.items():
                public[split].setdefault(key, []).append(event)
                if public['constants'] is None:
                    public['constants'] = truth_model.model_constants(truth_model.motion_packet(event))
            private[split].append(truth)
    return public, private


def forecast_prefix(public_pack, forecast_s=2.):
    """Use the identical legal episodes for short/long resource measurements."""
    steps = int(round(forecast_s/.01))
    if (not np.isfinite(forecast_s) or forecast_s <= 0 or
            not np.isclose(steps*.01, forecast_s, atol=1e-12, rtol=0) or
            forecast_s > public_pack['contract']['forecast_s']):
        raise ValueError('positive sampled forecast prefix within the existing package required')
    result = deepcopy(public_pack)
    result['contract']['forecast_s'] = forecast_s
    for split in ('train', 'val'):
        for events in result[split].values():
            for event in events:
                event['t'] = event['t'][:steps+1].copy()
                for family in ('targets', 'masks'):
                    event[family] = {key: value[:steps+1].copy() for key, value in event[family].items()}
    return result


def smoke():
    """One .11s artificial episode; no formal dataset or learner is created."""
    systems = {condition: make_system(condition=condition) for condition in CONDITIONS}
    for name in ('edges', 'mass', 'damping', 'c'):
        assert all(np.array_equal(system[name], systems['open'][name]) for system in systems.values())
    assert not systems['static']['beta'].any()
    assert not systems['static']['environment_gain'].any()
    assert not systems['modulated']['environment_gain'].any()
    for copies in (2, 4):
        lifted = lift_system(systems['open'], copies)
        mapping = full_offset_lift(copies)
        assert len(lifted['edges']) == len(systems['open']['edges'])*copies**2
        source_state = truth_model.initial_state(systems['open'], 9)
        clone_state = source_state[:, mapping['node_source']]
        np.testing.assert_allclose(truth_model.rhs(lifted, clone_state),
                                   truth_model.rhs(systems['open'], source_state)[:, mapping['node_source']],
                                   atol=1e-14, rtol=1e-13)
    events, truth = generate_episode(systems['open'], 9, 'distributed', forecast_s=.01)
    for name in ('theta', 'omega'):
        np.testing.assert_allclose(events['0.03']['history'][name]-events['0.00']['history'][name],
                                   3*(events['0.01']['history'][name]-events['0.00']['history'][name]), atol=1e-15)
    for event in events.values():
        for name in event['targets']:
            assert np.array_equal(event['targets'][name], events['0.00']['targets'][name])
        assert set(truth_model.motion_packet(event)) == {'history', 'constants'}
    return dict(status='passed', artificial_truth_duration_s=.11, lift_sizes=[16, 32],
                scientific_datasets_created=False, truth_frames=len(truth['t']))
