"""Synthetic implementation checks; no scientific data or qualification runs."""

import pytest
import torch

from kiki.models.osgkd import OSGKD, OSGKDConfig
from kiki.models.osgkd_fast_zero_sum import OSGKDFastZeroSum


def pair(n=3):
    torch.manual_seed(83)
    vector = torch.linspace(.2, .7, n, dtype=torch.float64)
    mass = torch.diag(torch.linspace(1., 2., n, dtype=torch.float64)) + vector.outer(vector)
    parent = OSGKD(OSGKDConfig("C1", hidden_size=5), mass=mass,
                   damping=torch.linspace(.1, .4, n), F=torch.eye(n),
                   Sp=torch.linspace(.7, 1.2, n), Se=torch.linspace(.4, 2.1, n),
                   omega_scale=torch.linspace(.8, 1.5, n),
                   theta_ref=torch.linspace(-.3, .4, n),
                   edge_index=torch.triu_indices(n, n, 1))
    with torch.no_grad():
        for name, parameter in parent.named_parameters():
            if "coeff" in name:
                parameter.normal_(std=.03)
    return parent, OSGKDFastZeroSum.from_parent_checkpoint(parent.checkpoint())


def bits(a, b):
    assert torch.equal(a.contiguous().view(torch.int64), b.contiguous().view(torch.int64))


@pytest.mark.parametrize("n", [1, 3, 5])
def test_exact_nesting(n):
    parent, model = pair(n)
    theta, omega = torch.randn(2, 11, n, dtype=torch.float64), torch.randn(2, 11, n, dtype=torch.float64)
    old, new = parent.initialize(theta, omega), model.initialize(theta, omega)
    bits(old, new[..., :5*n])
    bits(parent.rhs(None, old), model(None, new)[..., :5*n])
    for old_port, new_port in zip(parent.ports(old), model.ports(new)):
        bits(old_port, new_port)
    bits(parent.ports(old).tau_e, model.ports(new).tau_e)
    for step in (.005, .0025):
        bits(parent.rollout(old, [0, .01, .02], max_step=step),
             model.rollout(new, [0, .01, .02], max_step=step)[..., :5*n])
    for name, parameter in parent.state_dict().items():
        bits(parameter, model.state_dict()[name])
    torch.testing.assert_close(model.fast_rate, torch.full((n,), 10., dtype=torch.float64), rtol=0, atol=2e-15)


def test_active_torque_coi_and_power():
    parent, model = pair()
    with torch.no_grad():
        model.fast_gamma.copy_(torch.tensor([.3, -.2, .7]))
        model.fast_edge_coeff.normal_(std=.1)
    state = torch.randn(4, 18, dtype=torch.float64, requires_grad=True)
    ports = model.ports(state)
    assert ports.F_fast.norm() > 0
    torch.testing.assert_close(ports.F_fast.sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-15, rtol=0)
    correction = model.rhs(None, state)[..., 3:6] - parent.rhs(None, state[..., :15])[..., 3:6]
    torch.testing.assert_close((correction @ model.mass).sum(-1), torch.zeros(4, dtype=torch.float64), atol=2e-15, rtol=0)
    torch.testing.assert_close(ports.tau_e, ports.Fc + ports.tau_env + ports.F_fast)
    power = model.storage_power(state)
    gradient, = torch.autograd.grad(power.H.sum(), state)
    torch.testing.assert_close((gradient * model.rhs(None, state)).sum(-1), power.total_power, atol=2e-14, rtol=2e-14)
    # General symmetric mass uses column sums, not its diagonal, for the COI.
    with torch.no_grad():
        model.fast_edge_coeff.zero_()
    omega, f = state[..., 3:6], state[..., 15:]
    coi = (omega @ model.mass).sum(-1, keepdim=True) / model.mass.sum()
    expected = -model.fast_rate * f + model.fast_kappa * (omega - coi) / model.omega_scale
    torch.testing.assert_close(model.rhs(None, state)[..., 15:], expected)


def test_gradient_and_checkpoint():
    _, model = pair()
    theta, omega = torch.randn(2, 11, 3, dtype=torch.float64), torch.randn(2, 11, 3, dtype=torch.float64)
    state = model.initialize(theta, omega)
    model.rollout(state, [0, .01, .02], max_step=.005)[..., 3:6].square().sum().backward()
    assert torch.isfinite(model.fast_gamma.grad).all() and model.fast_gamma.grad.norm() > 0
    model.zero_grad()
    with torch.no_grad():
        model.fast_gamma.copy_(torch.tensor([.3, -.2, .7]))
    state = model.initialize(theta, omega)
    model.rollout(state, [0, .01, .02], max_step=.005)[..., 3:6].square().sum().backward()
    new = list(model.fast_parameters())
    assert len(new) == 6 and len({id(p) for p in new}) == 6
    for parameter in new:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
    snapshot = model.checkpoint()
    restored = OSGKDFastZeroSum.from_checkpoint(snapshot)
    bits(model.initialize(theta, omega), restored.initialize(theta, omega))
    bits(model(None, state), restored(None, state))
    with torch.no_grad():
        model.fast_gamma.add_(1)
    assert not torch.equal(snapshot["state_dict"]["fast_gamma"], model.fast_gamma)
    with pytest.raises(ValueError):
        OSGKD.from_checkpoint(snapshot)
