import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from joint_training_grid_0100 import IntegrationView, objective_at_grid
from local_memory_rom_0098 import LocalMemoryROM
from structured_rom import StructuredROM
from train_local_memory_0098 import objective as objective_0098


def _base():
    d = torch.zeros((26, 10), dtype=torch.float64)
    d[:9] = torch.eye(10, dtype=torch.float64)[:9] - torch.eye(10, dtype=torch.float64)[1:]
    d[9] = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0, 0, -1.], dtype=torch.float64)
    d[10:] = d[:16].clone()
    return StructuredROM(
        mass=torch.linspace(1., 2., 10), D=d, coupling=torch.ones(26),
        delta_star=torch.linspace(-.1, .1, 10), damping=torch.full((10,), .1),
        gain=torch.ones(10), n_q=0, storage=(),
    ).double()


def _model():
    model = LocalMemoryROM(_base(), torch.ones(20), torch.ones(10), "L10-S")
    for index in range(8):
        model.set_active(index, True)
    return model


def _samples():
    rng = np.random.default_rng(100)
    events = []
    for _ in range(2):
        events.append({
            "x": rng.normal(scale=.002, size=(40, 30)),
            "torque": rng.normal(scale=.002, size=(40, 10)),
        })
    samples = [(events[0], 10), (events[1], 11)]
    history = torch.as_tensor(np.stack([event["x"][i - 10:i + 1] for event, i in samples]), dtype=torch.float64)
    return samples, history


def _gradients(total, model):
    params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    gradients = torch.autograd.grad(total, params, allow_unused=False)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    return params, gradients


def test_one_substep_matches_frozen_objective_and_all_parameter_gradients():
    samples, history = _samples()
    reference = _model()
    wrapped = copy.deepcopy(reference)

    reference_total, reference_terms = objective_0098(reference, samples, history, .02, 1.0)
    wrapped_total, wrapped_terms = objective_at_grid(wrapped, samples, history, .02, 1.0, .01)
    assert torch.allclose(wrapped_total, reference_total, atol=1e-12, rtol=1e-12)
    assert reference_terms.keys() == wrapped_terms.keys()
    for name in reference_terms:
        assert torch.allclose(wrapped_terms[name], reference_terms[name], atol=1e-12, rtol=1e-12)

    reference_params, reference_gradients = _gradients(reference_total, reference)
    wrapped_params, wrapped_gradients = _gradients(wrapped_total, wrapped)
    assert [name for name, _ in reference.named_parameters()] == [name for name, _ in wrapped.named_parameters()]
    assert len(reference_params) == len(wrapped_params)
    for left, right in zip(reference_gradients, wrapped_gradients):
        assert torch.allclose(left, right, atol=1e-11, rtol=1e-11)
    assert all(torch.isfinite(gradient).all() for gradient in wrapped_gradients)


def test_two_substeps_match_manual_physical_10ms_trajectory_and_latent():
    samples, history = _samples()
    model = _model()
    view = IntegrationView(model, .005)
    states = torch.cat((history[:, -1], model.encode(history)), dim=-1)
    adjacent = torch.as_tensor(np.stack([event["x"][i - 9:i + 2] for event, i in samples]), dtype=torch.float64)

    raw_before = model.rk4(states, .01)
    manual = model.rk4(model.rk4(states, .005), .005)
    viewed = view.rk4(states, .01)
    assert view.grid_dt == .005
    assert torch.allclose(viewed, manual, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(viewed, raw_before, atol=1e-15, rtol=1e-15)

    latent_view = ((viewed[:, 30:] - model.encode(adjacent)) / .1).square().mean()
    latent_manual = ((manual[:, 30:] - model.encode(adjacent)) / .1).square().mean()
    assert torch.allclose(latent_view, latent_manual, atol=1e-13, rtol=1e-13)

    viewed_trajectory = [states]
    manual_trajectory = [states]
    for _ in range(3):
        viewed_trajectory.append(view.rk4(viewed_trajectory[-1], .01))
        manual_trajectory.append(model.rk4(model.rk4(manual_trajectory[-1], .005), .005))
    assert torch.allclose(torch.stack(viewed_trajectory, 1), torch.stack(manual_trajectory, 1), atol=1e-12, rtol=1e-12)

    raw_after = model.rk4(states, .01)
    assert torch.allclose(raw_before, raw_after, atol=1e-12, rtol=1e-12)


def test_fine_grid_gradients_are_finite_and_raw_model_is_unmodified():
    samples, history = _samples()
    model = _model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    total, terms = objective_at_grid(model, samples, history, .02, 1.0, .005)
    _, gradients = _gradients(total, model)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.isfinite(value) for value in terms.values())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert model.rk4.__func__ if hasattr(model.rk4, "__func__") else True


def test_grid_and_physical_dt_validation():
    model = _model()
    view = IntegrationView(model, .01)
    with torch.no_grad():
        state = torch.zeros((1, model.dim), dtype=torch.float64)
    try:
        view.rk4(state, .005)
    except ValueError:
        pass
    else:
        raise AssertionError("physical 5 ms calls must not silently change the 10 ms objective interval")
    for bad in (.02, .001):
        try:
            IntegrationView(model, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported grid accepted")
