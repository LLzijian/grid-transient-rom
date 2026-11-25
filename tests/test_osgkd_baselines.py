"""Synthetic, seconds-scale baseline interface and physical qualifications."""
import pytest
import torch
from kiki.models.osgkd import OSGKDConfig
from kiki.models.osgkd_baselines import GraphNeuralODE, ClosedPassiveKuramoto


@pytest.fixture(scope="module", autouse=True)
def single_thread_synthetic_checks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(cls, n=3, permutation=None):
    order = torch.arange(n) if permutation is None else permutation
    constants = dict(mass=(torch.arange(n).double() + 1)[order],
                     damping=(torch.arange(n).double() * .1 + .2)[order],
                     F=torch.eye(n, dtype=torch.float64), Sp=torch.ones(n), Se=torch.ones(n),
                     omega_scale=torch.ones(n), theta_ref=torch.linspace(-.2, .3, n)[order],
                     edge_index=torch.triu_indices(n, n, offset=1))
    torch.manual_seed(321)
    return cls(OSGKDConfig('C1', hidden_size=20), **constants)


@pytest.mark.parametrize('cls', [GraphNeuralODE, ClosedPassiveKuramoto])
@pytest.mark.parametrize('n', [1, 3, 5])
def test_interfaces_symmetry_and_checkpoint(cls, n):
    m = model(cls, n)
    theta = torch.randn(2, 3, 11, n, dtype=torch.float64) * .1
    omega = torch.randn_like(theta) * .1
    initial = m.initialize(theta, omega)
    assert initial.shape == (2, 3, 5*n)
    torch.testing.assert_close(initial[..., :n], theta[..., -1, :])
    torch.testing.assert_close(initial[..., n:2*n], omega[..., -1, :])
    assert (initial[..., 2*n:].abs() <= m.init_bounds).all()
    shifted = m.initialize(theta + .7, omega)
    torch.testing.assert_close(shifted[..., n:], initial[..., n:])
    torch.testing.assert_close(m.rhs(None, shifted), m.rhs(None, initial))
    reference = torch.cat((m.theta_ref, torch.zeros(4*n, dtype=torch.float64)))
    torch.testing.assert_close(m.rhs(None, reference), torch.zeros_like(reference), atol=1e-14, rtol=0)
    reference_history = m.theta_ref.expand(11, n)
    torch.testing.assert_close(m.initialize(reference_history, torch.zeros_like(reference_history)), reference)
    path = m.rollout(initial, [0, .01, .02])
    assert path.shape == (2, 3, 3, 5*n)
    clone = cls.from_checkpoint(m.checkpoint())
    torch.testing.assert_close(clone.rollout(initial, [0, .01, .02]), path)
    other = ClosedPassiveKuramoto if cls is GraphNeuralODE else GraphNeuralODE
    with pytest.raises(ValueError):
        other.from_checkpoint(m.checkpoint())
    perm = torch.arange(n-1, -1, -1)
    p = model(cls, n, perm)
    z = initial.reshape(2, 3, 5, n)[..., perm].reshape(2, 3, 5*n)
    expected = m.rhs(None, initial).reshape(2, 3, 5, n)[..., perm].reshape_as(z)
    torch.testing.assert_close(p.rhs(None, z), expected)
    torch.testing.assert_close(p.initialize(theta[..., perm], omega[..., perm]), z)
    ports = m.ports(initial)
    acceleration = m.rhs(None, initial)[..., n:2*n]
    torch.testing.assert_close(acceleration @ m.mass.T,
                               ports.tau_m - ports.tau_e - omega[..., -1, :] @ m.damping.T)
    variable = initial.detach().requires_grad_()
    m.rk4_step(variable, .01).square().sum().backward()
    assert torch.isfinite(variable.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_neural_counts_full_speed_and_port_gradients():
    m = model(GraphNeuralODE)
    assert sum(p.numel() for p in m.encoder.parameters()) == 1928
    assert sum(p.numel() for name, p in m.named_parameters() if not name.startswith('encoder.')) == 671
    reference = torch.cat((m.theta_ref, torch.zeros(12, dtype=torch.float64)))
    uniform_speed = reference.clone(); uniform_speed[3:6] = .2
    assert not torch.equal(m.rhs(None, reference), m.rhs(None, uniform_speed))
    state = torch.randn(15, dtype=torch.float64) * .1
    acceleration = m.rhs(None, state)[3:6].sum()
    gradients = torch.autograd.grad(acceleration, m.node_mlp[-1].weight)[0]
    assert gradients[:2].abs().sum(dim=1).gt(0).all()
    m.project_parameters_()


def test_closed_power_identity_and_inactive_latents():
    m = model(ClosedPassiveKuramoto)
    assert sum(p.numel() for p in m.parameters()) == 1928 + 3
    assert m.effective_state_dimension == 6
    history = torch.randn(11, 3, dtype=torch.float64)
    velocity = torch.randn_like(history)
    changed_history = history.clone(); changed_history[:-1] += .4
    torch.testing.assert_close(m.initialize(history, velocity), m.initialize(changed_history, velocity))
    state = torch.randn(4, 15, dtype=torch.float64, requires_grad=True)
    storage = m.storage_power(state)
    gradient = torch.autograd.grad(storage.H.sum(), state, create_graph=True)[0]
    derivative = (gradient * m.rhs(None, state)).sum(-1)
    torch.testing.assert_close(derivative, storage.total_power)
    assert (storage.total_power <= 0).all()
    assert (storage.H >= 0).all()
    changed = torch.cat((state[..., :6], state[..., 6:] + 7), -1)
    changed_rhs = m.rhs(None, changed)
    torch.testing.assert_close(changed_rhs[..., :2 * m.n], m.rhs(None, state)[..., :2 * m.n])
    torch.testing.assert_close(changed_rhs[..., 2 * m.n:], -changed[..., 2 * m.n:])
    assert torch.count_nonzero(m.ports(state).tau_m) == 0
    assert torch.count_nonzero(m.ports(state).tau_env) == 0
    with torch.no_grad(): m.c.fill_(-1)
    m.project_parameters_()
    assert (m.c == 0).all()
