"""Synthetic short-trajectory regression; no scientific data are consumed."""

import pytest
import torch
from torch.nn.utils import parametrize

from experiments.osgkd_v2_discovery import prepare
from experiments.osgkd_v2_refits.optimizer import VARIANTS, new_optimizer
from kiki.models.osgkd import OSGKDConfig
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_restricted import RestrictedOSGKDGraph


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def parent_and_plan():
    torch.manual_seed(721)
    n = 9  # 36 edges exercise all three registered edge budgets.
    parent = OSGKDGraph(OSGKDConfig('C1', hidden_size=20),
                        mass=torch.ones(n), damping=torch.ones(n) * .2,
                        F=torch.eye(n), Sp=torch.ones(n), Se=torch.ones(n),
                        omega_scale=torch.ones(n), theta_ref=torch.linspace(-.2, .2, n),
                        edge_index=torch.triu_indices(n, n, 1))
    with torch.no_grad():
        for name in ('mechanical_coeff', 'amplitude_coeff', 'environment_coeff',
                     'memory_coeff', 'edge_coeff'):
            getattr(parent, name).normal_(0, .05)
        parent.c.fill_(.2)
        parent.beta.fill_(.04)
    rows = [{'edge_index': i, 'left': a, 'right': b, 'exchange': {'C_g': float(i + 1)}}
            for i, (a, b) in enumerate(parent.edge_index.T.tolist())]
    groups = {'environment_a': {'C_g': 0.}, 'environment_a2': {'C_g': 0.}}
    return parent, prepare.restriction_plan(parent, rows, groups)


def expected_masks(model, spec):
    edges = torch.ones_like(model.c, dtype=torch.bool)
    if spec.edge_ids is not None:
        edges.zero_()
        edges[list(spec.edge_ids)] = True
    masks = {name: torch.ones_like(getattr(model, name), dtype=torch.bool)
             for name in ('c', 'beta', 'edge_coeff', 'environment_coeff',
                          'memory_coeff', 'mechanical_coeff')}
    masks['c'] &= edges & (not spec.remove_coupling)
    masks['beta'] &= edges & (not spec.remove_coupling) & (not spec.remove_beta)
    masks['edge_coeff'] &= edges[:, None, None]
    masks['environment_coeff'][list(spec.static_environment_rows)] = False
    if spec.remove_memory:
        masks['memory_coeff'].zero_()
        masks['edge_coeff'][..., 2:4] = False
    if spec.freeze_mechanical:
        masks['mechanical_coeff'].zero_()
    return masks


@pytest.mark.parametrize('variant', VARIANTS)
def test_registered_variant_optimizer_step_and_masks(variant):
    assert VARIANTS is prepare.VARIANTS
    parent, plan = parent_and_plan()
    assert tuple(plan) == VARIANTS
    spec = plan[variant]
    model = RestrictedOSGKDGraph.from_parent(parent, spec)
    optimizer = new_optimizer(model)
    owned = [p for group in optimizer.param_groups for p in group['params']]
    assert len(owned) == len({id(p) for p in owned})
    assert {id(p) for p in owned} == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(p.is_leaf and p.requires_grad for p in owned)
    assert {id(p) for p in optimizer.param_groups[0]['params']} == {
        id(p) for p in model.encoder.parameters() if p.requires_grad}
    assert [group['lr'] for group in optimizer.param_groups] == [.001, .0003]
    before = [p.detach().clone() for p in owned]
    theta = model.theta_ref + torch.randn(2, 11, model.n, dtype=torch.float64) * .05
    omega = torch.randn_like(theta) * .05
    path = model.rollout(model.initialize(theta, omega), [0., .001, .002])
    assert torch.isfinite(path).all()
    path.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in owned)
    optimizer.step()
    model.project_parameters_()
    assert all(torch.isfinite(p).all() for p in owned)
    assert any(not torch.equal(p, old) for p, old in zip(owned, before))
    assert {id(p) for p in optimizer.state} == {id(p) for p in owned}
    for name, keep in expected_masks(model, spec).items():
        actual = getattr(model, name)
        assert torch.count_nonzero(actual[~keep]) == 0
        if not keep.all():
            mapping = model.parametrizations[name]
            assert torch.equal(mapping[0].keep, keep)
            assert torch.count_nonzero(mapping.original[~keep]) == 0
        else:
            assert not parametrize.is_parametrized(model, name)
    if spec.remove_memory:
        mapping = model.encoder.readout.parametrizations.weight
        assert not mapping[0].keep[2].any()
        assert mapping[0].keep[:2].all()
        assert torch.count_nonzero(model.encoder.readout.weight[2]) == 0
        assert torch.count_nonzero(mapping.original[2]) == 0
    assert (model.c >= 0).all()
    assert ((model.beta >= 0) & (model.beta <= model.c)).all()


def test_optimizer_excludes_frozen_leaves():
    parent, plan = parent_and_plan()
    model = RestrictedOSGKDGraph.from_parent(parent, plan['no_memory'])
    next(model.encoder.parameters()).requires_grad_(False)
    model.amplitude_coeff.requires_grad_(False)
    optimizer = new_optimizer(model)
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {
        id(p) for p in model.parameters() if p.requires_grad}
