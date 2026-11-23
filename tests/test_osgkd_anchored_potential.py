import pytest
import torch

from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_anchored_potential import OSGKDAnchoredPotential
from kiki.models.osgkd_fitting import event_objective, joint_loss, tensor_events
from test_osgkd_model import model, sample_state
from test_osgkd_fitting import event


def pair(variant="C1"):
    torch.manual_seed(41)
    parent = model(variant, c_init=1.2, beta_fraction_init=.7)
    return parent, OSGKDAnchoredPotential.from_parent_checkpoint(parent.checkpoint())


def same_bits(left, right):
    return torch.equal(left.contiguous().view(torch.int64), right.contiguous().view(torch.int64))


@pytest.mark.parametrize("variant", ["C0", "C1"])
def test_zero_embedding_exact_functions_and_snapshot(variant):
    parent, candidate = pair(variant)
    old_parameters = dict(parent.named_parameters())
    new_parameters = dict(candidate.named_parameters())
    assert new_parameters.keys() - old_parameters.keys() == {"rho"}
    assert candidate.rho.shape == (3, 2) and candidate.rho.count_nonzero() == 0
    for name, value in old_parameters.items():
        assert torch.equal(value, new_parameters[name])
    state = torch.stack((sample_state(), -sample_state()))
    theta, omega = state[:, None, :3].expand(-1, 11, -1), state[:, None, 3:6].expand(-1, 11, -1)
    assert torch.equal(parent.initialize(theta, omega), candidate.initialize(theta, omega))
    for a, b in zip(parent.ports(state), candidate.ports(state)):
        assert torch.equal(a, b)
    assert same_bits(parent.rhs(None, state), candidate.rhs(None, state))
    assert torch.equal(parent(state.new_tensor(0), state), candidate(state.new_tensor(0), state))
    assert same_bits(parent.rollout(state, [0., .01, .03]), candidate.rollout(state, [0., .01, .03]))
    for a, b in zip(parent.storage_power(state), candidate.storage_power(state)):
        assert torch.equal(a, b)
    payload = candidate.checkpoint()
    assert payload["version"] != parent.checkpoint()["version"]
    loaded = OSGKDAnchoredPotential.from_checkpoint(payload)
    assert torch.equal(candidate.rhs(None, state), loaded.rhs(None, state))
    with pytest.raises(ValueError):
        OSGKD.from_checkpoint(payload)
    with pytest.raises(ValueError):
        OSGKDAnchoredPotential.from_checkpoint(parent.checkpoint())


def test_zero_embedding_exact_old_objective_and_gradients():
    parent, candidate = pair()
    events = tensor_events([event(10)], "cpu")
    losses = [event_objective(rom, events, 10, [.4, .5, .6]) for rom in (parent, candidate)]
    assert torch.equal(*losses)
    for loss in losses:
        joint_loss(loss).backward()
    for name, parameter in parent.named_parameters():
        assert parameter.grad is not None
        assert same_bits(parameter.grad, dict(candidate.named_parameters())[name].grad), name
    assert candidate.rho.grad is not None and candidate.rho.grad.norm() > 1e-9


@pytest.mark.parametrize("rho", [0., .37])
def test_rho_direction_derivative_at_two_step_sizes(rho):
    _, candidate = pair()
    with torch.no_grad():
        candidate.rho.fill_(rho)
    state = sample_state()
    direction = torch.linspace(-.8, .7, 6, dtype=torch.float64).reshape(3, 2)
    def objective():
        return candidate.rollout(state, [0., .02])[-1, 3:6].square().sum()
    derivative = (torch.autograd.grad(objective(), candidate.rho)[0] * direction).sum()
    assert derivative.abs() > 1e-7
    center = candidate.rho.detach().clone()
    for step in (1e-4, 5e-5):
        with torch.no_grad():
            candidate.rho.copy_(center + step * direction)
        plus = objective().item()
        with torch.no_grad():
            candidate.rho.copy_(center - step * direction)
        minus = objective().item()
        torch.testing.assert_close(derivative, derivative.new_tensor((plus-minus)/(2*step)),
                                   rtol=2e-6, atol=1e-10)
    with torch.no_grad():
        candidate.rho.copy_(center)


def test_nonzero_rho_full_reference_jacobian_and_mixed_amplitude_derivatives():
    parent, candidate = pair()
    with torch.no_grad():
        candidate.rho.copy_(torch.tensor([[.3, -.5], [-.2, .4], [.7, .1]], dtype=torch.float64))
    for amplitude in (torch.zeros(3, dtype=torch.float64), sample_state()[9:12]):
        reference = torch.cat((candidate.theta_ref, torch.zeros(6, dtype=torch.float64),
                               amplitude, torch.zeros(3, dtype=torch.float64)))
        assert torch.equal(parent.rhs(None, reference), candidate.rhs(None, reference))
        jacobians = [torch.autograd.functional.jacobian(lambda s: rom.rhs(None, s), reference)
                     for rom in (parent, candidate)]
        torch.testing.assert_close(*jacobians, rtol=0, atol=1e-15)
        hessians = [torch.autograd.functional.hessian(lambda s: rom.storage_power(s).Uc, reference)
                    for rom in (parent, candidate)]
        torch.testing.assert_close(*hessians, rtol=0, atol=1e-15)


def test_nonzero_rho_potential_power_zero_sum_gauge_and_edge_reversal():
    parent, candidate = pair()
    with torch.no_grad():
        candidate.rho.copy_(torch.tensor([[.3, -.5], [-.2, .4], [.7, .1]], dtype=torch.float64))
    state = torch.stack((sample_state(), -sample_state())).requires_grad_()
    storage = candidate.storage_power(state)
    gradient = torch.autograd.grad(storage.H.sum(), state)[0]
    field = candidate.rhs(None, state)
    torch.testing.assert_close(gradient[..., :3], candidate.ports(state).Fc, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close((gradient[..., 9:12] * field[..., 9:12]).sum(-1), storage.amplitude_power)
    torch.testing.assert_close((gradient * field).sum(-1), storage.total_power, rtol=1e-12, atol=1e-13)
    assert (storage.amplitude_power - parent.storage_power(state).amplitude_power).abs().min() > 1e-5
    delta = candidate.ports(state).Fc - parent.ports(state).Fc
    torch.testing.assert_close(delta.sum(-1), torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-14)
    assert delta.norm() > 1e-3
    shifted = state.detach().clone()
    shifted[..., :3] += 2.7
    torch.testing.assert_close(candidate.rhs(None, shifted), field)
    torch.testing.assert_close(candidate.storage_power(shifted).H, storage.H)
    # Reverse the computational edge orientation; the public parent constructor enforces left < right.
    reversed_model = OSGKDAnchoredPotential.from_checkpoint(candidate.checkpoint())
    with torch.no_grad():
        reversed_model.edge_index.copy_(reversed_model.edge_index.flip(0))
        reversed_model.edge_nodes.copy_(reversed_model.edge_index.T.flatten())
        reversed_model.incidence.neg_()
        reversed_model.phi_ref.neg_()
        reversed_model.sin_ref.neg_()
        reversed_model.edge_coeff.copy_(reversed_model.edge_coeff.flip(1))
        reversed_model.edge_coeff[..., 0].neg_()
        reversed_model.edge_coeff[..., 2].neg_()
    torch.testing.assert_close(reversed_model.rhs(None, state), field)
    for actual, expected in zip(reversed_model.storage_power(state), storage):
        torch.testing.assert_close(actual, expected)
