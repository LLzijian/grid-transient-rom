"""Synthetic symmetry and ABI gates; no data or checkpoints are consumed."""

import torch
import pytest

from kiki.models.osgkd import OSGKDConfig
from kiki.models.osgkd_graph import OSGKDGraph


def model(n=8):
    torch.manual_seed(308 + n)
    return OSGKDGraph(OSGKDConfig('C1', hidden_size=20, p_init_bound=4,
                                 a_init_bound=2, e_init_bound=4),
                      mass=1 + torch.rand(n, dtype=torch.float64),
                      damping=torch.ones(n, dtype=torch.float64) * .2,
                      F=torch.eye(n, dtype=torch.float64),
                      Sp=torch.ones(n, dtype=torch.float64),
                      Se=torch.ones(n, dtype=torch.float64),
                      omega_scale=1 + torch.rand(n, dtype=torch.float64),
                      theta_ref=torch.randn(n, dtype=torch.float64) * .2,
                      edge_index=torch.triu_indices(n, n, 1))


def history(m):
    return (m.theta_ref + .3 * torch.randn(2, 3, 11, m.n, dtype=torch.float64),
            .2 * torch.randn(2, 3, 11, m.n, dtype=torch.float64))


def close(actual, expected):
    deviation = (actual - expected).abs().max().item()
    assert deviation < 1e-12, deviation


def permuted(m, perm):
    inverse = torch.argsort(perm)
    mapped = inverse[m.edge_index]
    flipped = mapped[0] > mapped[1]
    edges = mapped.sort(dim=0).values
    result = OSGKDGraph(m.config, mass=m.mass[perm][:, perm],
                        damping=m.damping[perm][:, perm], F=m.F[perm][:, perm],
                        Sp=m.Sp[perm], Se=m.Se[perm], omega_scale=m.omega_scale[perm],
                        theta_ref=m.theta_ref[perm], edge_index=edges)
    result.encoder.load_state_dict(m.encoder.state_dict())
    with torch.no_grad():
        for name in ('mechanical_coeff', 'amplitude_coeff', 'environment_coeff', 'memory_coeff'):
            getattr(result, name).copy_(getattr(m, name)[:, perm])
        result.c.copy_(m.c)
        result.beta.copy_(m.beta)
        coeff = m.edge_coeff.clone()
        coeff[flipped] = coeff[flipped].flip(1)
        coeff[flipped, :, 0] *= -1
        coeff[flipped, :, 2] *= -1
        result.edge_coeff.copy_(coeff)
    assert flipped.any()
    return result


def test_permutation_encoder_and_nonzero_ode():
    m = model()
    with torch.no_grad():
        for name in ('mechanical_coeff', 'amplitude_coeff', 'environment_coeff', 'memory_coeff', 'edge_coeff'):
            getattr(m, name).normal_(0, .2)
        m.c.uniform_(.2, .4)
        m.beta.uniform_(.02, .1)
    perm = torch.randperm(m.n)
    other = permuted(m, perm)
    theta, omega = history(m)
    state = m.initialize(theta, omega)
    indices = torch.cat([perm + k * m.n for k in range(5)])
    close(other.initialize(theta[..., perm], omega[..., perm]), state[..., indices])
    rhs = m.rhs(None, state)
    assert rhs.abs().max() > .01
    close(other.rhs(None, state[..., indices]), rhs[..., indices])
    close(other.rk4_step(state[..., indices], .001), m.rk4_step(state, .001)[..., indices])
    close(other.storage_power(state[..., indices]).H, m.storage_power(state).H)
    for actual, expected in zip(other.ports(state[..., indices]), m.ports(state)):
        close(actual, expected[..., perm])


def test_global_angle_shift():
    m = model()
    theta, omega = history(m)
    initial = m.initialize(theta, omega)
    shifted = m.initialize(theta + 1.75, omega)
    close(initial[..., 2*m.n:], shifted[..., 2*m.n:])
    close(m.rhs(None, initial), m.rhs(None, shifted))
    assert torch.equal(initial[..., :m.n], theta[..., -1, :])
    assert torch.equal(initial[..., m.n:2*m.n], omega[..., -1, :])


def test_zero_reference_and_shifted_equilibrium():
    m = model()
    zero = torch.zeros(2, 3, 11, m.n, dtype=torch.float64)
    reference = m.theta_ref.expand_as(zero)
    for shift in (0., 1.25):
        initial = m.initialize(reference + shift, zero)
        close(initial[..., 2*m.n:], torch.zeros_like(initial[..., 2*m.n:]))
        close(m.rhs(None, initial), torch.zeros_like(initial))
        if shift == 0:
            assert torch.count_nonzero(initial[..., 2*m.n:]) == 0


@pytest.mark.parametrize('n', [1, 8, 10, 16])
def test_size_independent_encoder_checkpoint_and_gradients(n):
    source = model()
    m = model(n)
    m.encoder.load_state_dict(source.encoder.state_dict(), strict=True)
    assert sum(p.numel() for p in m.encoder.parameters()) == 1928
    assert not list(m.initial_head.parameters())
    theta, omega = history(m)
    theta.requires_grad_()
    initial = m.initialize(theta, omega)
    assert initial.shape == (2, 3, 5*n)
    assert torch.isfinite(initial).all()
    restored = OSGKDGraph.from_checkpoint(m.checkpoint())
    assert m.checkpoint()['version'] == 4
    close(restored.initialize(theta, omega), initial)
    close(restored.rhs(None, initial), m.rhs(None, initial))
    initial.square().sum().backward()
    assert torch.isfinite(theta.grad).all()
    for name, p in m.encoder.named_parameters():
        if n == 1 and name.startswith('edge_mlp.'):
            assert p.grad is None  # No pairs exist in the single-node system.
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all()
    assert m.initialize(theta[0, 0], omega[0, 0]).shape == (5*n,)
