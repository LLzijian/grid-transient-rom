"""Synthetic interface checks for the planned Sprint 2B experiments."""

import numpy as np
import pytest
import torch

from experiments.osgkd_v2_discovery import prepare
from experiments.osgkd_v2_sprint2a_20261125 import train
from kiki.data.osgkd_synthetic import generate_event, make_system, model_constants, motion_packet
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events
from kiki.models.osgkd_restricted import RestrictedOSGKDGraph, Restrictions


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def parent():
    n = 10
    constants = {key: np.ones(n) for key in ('mass', 'damping', 'Sp', 'Se', 'omega_scale', 'S_m')}
    constants.update(F=np.eye(n), theta_ref=np.linspace(-.2, .2, n), s_coi=1., edge_index=np.stack(np.triu_indices(n, 1)))
    model = train.new_model(constants)
    with torch.no_grad():
        model.environment_coeff.fill_(.1)
        model.memory_coeff[1:].fill_(.1)
        model.edge_coeff.fill_(.12)
    return model


def test_nested_rank_and_static_threshold(parent):
    rows = [dict(edge_index=j, left=i, right=k, exchange={'C_g': .005 * j})
            for j, (i, k) in enumerate(parent.edge_index.T.tolist())]
    specs = prepare.restriction_plan(parent, rows[::-1], {'environment_a': {'C_g': .000199},
                                                         'environment_a2': {'C_g': .0002}})
    assert specs['static'].static_environment_rows == (0,)
    assert set(specs['top12'].edge_ids) < set(specs['top20'].edge_ids) < set(specs['top30'].edge_ids)
    assert specs['top12'].edge_ids == tuple(range(33, 45))
    assert specs['no_beta'].remove_beta and not specs['no_beta'].static_environment_rows


def test_dense_and_hard_edge_mask_checkpoint(parent):
    x = torch.linspace(-.7, .7, 50, dtype=torch.float64)
    dense = RestrictedOSGKDGraph.from_parent(parent, Restrictions())
    torch.testing.assert_close(dense.rhs(None, x), parent.rhs(None, x), rtol=0, atol=0)
    spec = Restrictions(edge_ids=(0, 2), static_environment_rows=(0,))
    reduced = RestrictedOSGKDGraph.from_parent(parent, spec)
    without = torch.ones(45, dtype=torch.bool)
    without[[0, 2]] = False
    before = reduced.rhs(None, x).detach().clone()
    with torch.no_grad():
        reduced.parametrizations.c.original[without] = 100
        reduced.parametrizations.beta.original[without] = 100
        reduced.parametrizations.edge_coeff.original[without] = 100
        reduced.parametrizations.environment_coeff.original[0] = 100
    torch.testing.assert_close(reduced.rhs(None, x), before, rtol=0, atol=0)
    reduced.rhs(None, x).square().sum().backward()
    assert not reduced.parametrizations.c.original.grad[without].count_nonzero()
    assert not reduced.parametrizations.edge_coeff.original.grad[without].count_nonzero()
    assert reduced.complexity()['active_edge_paths'] == 2
    clone = RestrictedOSGKDGraph.from_checkpoint(reduced.checkpoint())
    torch.testing.assert_close(clone.rhs(None, x), before, rtol=0, atol=0)
    assert clone.restrictions == spec


def test_module_restrictions(parent):
    x = torch.linspace(-.7, .7, 50, dtype=torch.float64)
    no_memory = RestrictedOSGKDGraph.from_parent(parent, Restrictions(remove_memory=True))
    changed = x.clone()
    changed[40:] += .5
    torch.testing.assert_close(no_memory.rhs(None, x), no_memory.rhs(None, changed), atol=0, rtol=0)
    assert not no_memory.rhs(None, x)[40:].count_nonzero()
    assert not no_memory.initialize(x[:10].expand(11, 10), x[10:20].expand(11, 10))[40:].count_nonzero()
    assert no_memory.complexity()['evolved_state_dimension'] == 40
    no_beta = RestrictedOSGKDGraph.from_parent(parent, Restrictions(remove_beta=True))
    assert not no_beta.beta.count_nonzero() and no_beta.c.count_nonzero() == 45
    no_exchange = RestrictedOSGKDGraph.from_parent(parent, Restrictions(remove_coupling=True))
    assert not no_exchange.ports(x).Fc.count_nonzero() and no_exchange.edge_coeff.count_nonzero()
    frozen_p = RestrictedOSGKDGraph.from_parent(parent, Restrictions(freeze_mechanical=True))
    assert not frozen_p.rhs(None, x)[20:30].count_nonzero()
    torch.testing.assert_close(frozen_p.ports(x).tau_m, parent.ports(x).tau_m)


def test_three_seed_tables():
    rows = [dict(seed=seed, variant=variant, qualified=True, stable_risk_8s=.5+j*.1,
                 stable_false_alarms=0, unstable_8s_hits=9, exchange_support=[[0, 1], [0, 2]],
                 active_edge_paths=2, active_module_groups=3, parameters=100)
            for seed in prepare.GRAPH_SEEDS for j, variant in enumerate(prepare.VARIANTS)]
    report = prepare.tables(rows)
    assert len(report['rows']) == 27
    assert report['pareto_by_seed'][24111] == ['full']
    assert report['support_agreement']['top12'][0]['support_f1'] == 1.
    assert report['paired_deltas_from_refit_full'][24131]['top12']['stable_risk_8s'] == pytest.approx(.4)
    assert prepare.support_agreement(((0, 1), (0, 2)), ((0, 2), (0, 3))) == {'jaccard': 1/3, 'support_f1': .5}
    assert prepare.support_agreement((), ()) == {'jaccard': None, 'support_f1': None}


@pytest.mark.parametrize('name', tuple(prepare.BASELINES))
def test_matched_baseline_uses_synthetic_packet_and_original_loss(name):
    event, _ = generate_event(make_system(8), duration=.21, noise_std=.01)
    packet = motion_packet(event)
    assert set(packet) == {'history', 'constants'}
    assert not {'edges', 'beta', 'state', 'id'} & set(packet['constants'])
    constants = model_constants(packet)
    baseline = prepare.new_baseline(name, constants, 24111)
    graph = train.new_model(constants, 'graph', 24111)
    for key, value in graph.encoder.state_dict().items():
        torch.testing.assert_close(baseline.encoder.state_dict()[key], value, atol=0, rtol=0)
    work = IntegrationWork()
    batch = tensor_events([event], 'cpu', max_steps=11)
    terms = train.components(baseline, batch, constants, work, .01, 11)
    train.objective(terms).backward()
    assert terms.shape == (1, 3) and torch.isfinite(terms).all()
    assert work.rhs['forward'] == 44
    assert any(parameter.grad is not None for parameter in baseline.parameters())


def test_baseline_scores_keep_fixed_coverage(monkeypatch):
    event, _ = generate_event(make_system())
    constants = model_constants(motion_packet(event))
    baseline = prepare.new_baseline('closed_passive_kuramoto', constants, 24111)
    def saved_score(model, events, given_constants, split, directory, context):
        assert model is baseline and events == [] and given_constants is constants and split == 'val'
        return {'gate_passed': True, 'stable_8s_motion_risk': .45, 'autonomous': {
            'stable_false_alarm_8s': {'raw_denominator': 95, 'autonomous_trigger': 0},
            'unstable_recall_8s': {'input_eligible_denominator': 12, 'raw_denominator': 22, 'autonomous_trigger': 9},
        }}
    monkeypatch.setattr(prepare.graph_run, 'score', saved_score)
    row = prepare.score_baseline(baseline, [], constants, None, seed=24111, name='closed_passive_kuramoto')
    assert row['effective_encoder_parameters'] == 0 and row['effective_state_dimension'] == 16
    assert row['unstable_8s_hits'] == 9 and row['qualified']
