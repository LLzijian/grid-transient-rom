"""Synthetic contracts; do not require or read private trajectories."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
spec = importlib.util.spec_from_file_location('joint', Path(__file__).resolve().parents[1]/'experiments/train_joint_latent.py')
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)


def setup_model(rank=1):
    from kiki.models.private_pm import MechanicalROM
    from kiki.physics import complete_incidence
    b = complete_incidence(10)[0]
    k = np.r_[np.ones(26), np.zeros(19)]
    base = MechanicalROM(np.linspace(1, 2, 10), b, np.linspace(0, .2, 10),
                         np.ones(10), k, np.ones(10)*.1, np.ones(10)*2, np.ones(10)*.3)
    torch.manual_seed(53)
    model = joint.JointROM(base, np.eye(10), np.ones(20), np.ones(72), 1., 1., latent_dim=rank)
    return base, model


def test_frozen_field_sign_and_mass_convention():
    base, model = setup_model()
    x = np.random.default_rng(1).normal(scale=.1, size=(4, 31))
    original = x[:, :30].copy()
    original[:, 20:] += base.pm_star
    expected = base.field(original)
    expected[:, 10:20] -= model.memory(joint.tensor(x[:, 30:])).detach().numpy()/base.mass
    assert np.max(abs(model.field(joint.tensor(x)).detach().numpy()[:, :30]-expected)) < 1e-12
    assert all(not getattr(model, key).requires_grad for key in ('mass', 'coupling', 'damping', 'rate', 'gain'))
    assert np.isclose(float(torch.linalg.vector_norm(model.c/torch.linalg.vector_norm(model.c)).detach()), 1)


def test_encoder_once_and_future_perturbations_cannot_affect_prediction():
    _, model = setup_model()
    raw = joint.tensor(np.random.default_rng(2).normal(scale=.1, size=(3, 20, 30)))
    calls = []
    hook = model.gru.register_forward_hook(lambda *args: calls.append(1))
    a = model.rollout(raw[:, :11], 4, .02)
    assert len(calls) == 1
    raw[:, 11:] += 1e5
    b = model.rollout(raw[:, :11], 4, .02)
    assert torch.equal(a, b)
    hook.remove()


def test_rollout_gradients_reach_joint_parameters_and_rk4_accuracy():
    _, model = setup_model()
    raw = joint.tensor(np.random.default_rng(3).normal(scale=.1, size=(3, 11, 30)))
    fine = model.rollout(raw, 8, .01)
    coarse = model.rollout(raw, 4, .02)
    coarse[:, -1, :20].square().sum().backward()
    for parameter in (model.gru.weight_ih_l0, model.c, model.xi):
        assert torch.isfinite(parameter.grad).all()
        assert torch.linalg.vector_norm(parameter.grad) > 0
    assert float((fine[:, ::2]-coarse).abs().max().detach()) < 1e-6


def test_zero_memory_is_exact_m3_and_shared_support():
    base, model = setup_model()
    raw = np.random.default_rng(4).normal(scale=.1, size=(2, 11, 30))
    with torch.no_grad():
        model.amplitude.zero_()
        model.xi.fill_(.1)
        model.support[2:] = 0
    original = raw[:, -1].copy()
    original[:, 20:] += base.pm_star
    pred = model.rollout(joint.tensor(raw), 10, .01).detach().numpy()
    expected = base.rollout(original, None, np.arange(11)*.01, max_step=.01)
    expected[:, :, 20:] -= base.pm_star
    assert np.max(abs(pred[:, :, :30]-expected)) < 1e-12
    assert np.allclose(model.field(joint.tensor(pred[:, 0]))[:, 30:].detach().numpy(),
                       .1+.1*pred[:, 0, 30:])


def test_algebraic_force_and_voltage_power_are_instantaneous():
    base, model = setup_model()
    x = joint.tensor(np.random.default_rng(5).normal(scale=.1, size=(3, 31)))
    with torch.no_grad():
        model.alpha[3] = .2
        model.memory_enabled.zero_()
    a = model.field(x)
    modified = x.clone()
    modified[:, 30:] += 100
    assert torch.equal(a[:, :30], model.field(modified)[:, :30])
    expected = base.field(np.column_stack((x[:, :20], x[:, 20:30]+joint.tensor(base.pm_star))))
    expected[:, 10:20] -= model.memory(model.algebraic_coordinate(x)).detach().numpy()/base.mass
    assert np.max(abs(a[:, :30].detach().numpy()-expected)) < 1e-12
    z, load = model.electrical_output(x[:, :30], x[:, 30:])
    v = model.voltage_star+z[:, 10:]*model.z_scale[10:]
    assert torch.all(v >= 0)
    assert torch.allclose(load.ravel()*model.load_scale+model.load_star,
                          v[:, model.load_positions].square()@model.load_g)
    # Signed polar voltage and its physical magnitude yield identical load.
    raw = np.array([-.3, 0., .5])
    assert np.array_equal(raw**2, np.abs(raw)**2)


@pytest.mark.parametrize('rank', [2, 3])
def test_multiple_coordinates_keep_one_shared_explicit_vector_ode(rank):
    _, model = setup_model(rank=rank)
    history = joint.tensor(np.random.default_rng(6).normal(scale=.1, size=(3, 11, 30)))
    pred = model.rollout(history, 5, .01)
    assert pred.shape == (3, 6, 30+rank)
    assert model.features(pred[:, 0]).shape == (3, 73+rank)
    assert model.algebraic_coordinate(pred[:, 0]).shape == (3, rank)
    assert model.xi.shape == (73+rank, rank)
    assert torch.allclose(torch.linalg.vector_norm(model.c/torch.linalg.vector_norm(model.c, dim=0), dim=0),
                          torch.ones(rank, dtype=torch.float64))
    pred[:, -1, :20].square().mean().backward()
    for parameter in (model.c, model.alpha, model.xi):
        assert torch.all(torch.linalg.vector_norm(parameter.grad, dim=0) > 0)


def test_train_reference_conditioning_and_frozen_query_batch_independence():
    _, model = setup_model(rank=3)
    rng = np.random.default_rng(7)
    reference = joint.tensor(rng.normal(size=(50, 11, 30)))
    weights = joint.tensor(np.arange(1, 51)/np.arange(1, 51).sum())
    model.condition_coordinates(reference, weights)
    q = model.encode(reference)
    assert torch.max(torch.abs((q*weights[:, None]).sum(dim=0))) < 1e-12
    raw = model.raw_encode(reference)
    center = raw-model.q_center
    covariance = center.T@(center*weights[:, None])+torch.eye(3, dtype=torch.float64)*1e-6
    assert torch.max(torch.abs(model.q_transform.T@covariance@model.q_transform-
                               torch.eye(3, dtype=torch.float64))) < 1e-10
    q.square().sum().backward()
    assert torch.isfinite(model.gru.weight_ih_l0.grad).all()
    assert torch.linalg.vector_norm(model.gru.weight_ih_l0.grad) > 0
    with torch.no_grad():
        model.condition_coordinates(reference, weights)
        first = model.encode(reference[:1])
        batch_first = model.encode(reference[:4])[:1]
        assert torch.allclose(first, batch_first, atol=1e-12, rtol=0)
        calls = []
        hook = model.gru.register_forward_hook(lambda *args: calls.append(1))
        model.rollout(reference[:1], 2, .01)
        hook.remove()
        assert len(calls) == 1


def test_event_tail_risk_keeps_every_observed_event_in_gradient():
    _, model = setup_model()
    rng = np.random.default_rng(8)
    events = [{'x': rng.normal(scale=.1, size=(35, 30)), 'force': np.zeros((35, 10)),
               'load': np.zeros(35), 'z': np.zeros((35, 20))} for _ in range(10)]
    samples = joint.batch(events, rng, count=10)
    losses = joint.loss_terms(model, *samples, .2)
    assert torch.allclose(losses['roll'], .8*losses['roll_mean']+.2*losses['roll_tail'])
    losses['roll'].backward()
    assert torch.isfinite(model.xi.grad).all()
    event_loss = torch.arange(1., 11., dtype=torch.float64, requires_grad=True)
    risk = .8*event_loss.mean()+.2*event_loss.topk(1).values.mean()
    risk.backward()
    assert torch.all(event_loss.grad > 0)
    assert torch.allclose(event_loss.grad, joint.tensor([.08]*9+[.28]))
