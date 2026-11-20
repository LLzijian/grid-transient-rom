from io import BytesIO
import json
import math

import pytest

torch = pytest.importorskip("torch")

from kiki.models.osgkd import OSGKD, OSGKDConfig


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def constants():
    return {
        "mass": [[2.0, 0.2, 0.0], [0.2, 3.0, 0.0], [0.0, 0.0, 4.0]],
        "damping": [0.1, 0.15, 0.2],
        "F": [[1.0, 0.1, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.8]],
        "Sp": [0.4, 0.5, 0.6],
        "Se": [0.7, 0.8, 0.9],
        "omega_scale": [1.0, 2.0, 3.0],
        "theta_ref": [0.35, -0.2, 0.1],
        "edge_index": [[0, 0, 1], [1, 2, 2]],
    }


def model(variant="C1", **config):
    result = OSGKD(OSGKDConfig(variant, hidden_size=5, **config), **constants())
    with torch.no_grad():
        for name in ("mechanical_coeff", "amplitude_coeff", "environment_coeff", "memory_coeff", "edge_coeff"):
            coeff = getattr(result, name)
            coeff.copy_(torch.linspace(-0.3, 0.4, coeff.numel(), dtype=coeff.dtype).reshape_as(coeff))
    return result


def sample_state():
    return tensor([[0.7, -0.4, 0.05], [0.5, -0.3, 0.2], [0.1, 0.2, -0.1],
                   [0.4, 0.1, -0.2], [-0.2, -0.15, 0.1]]).flatten()


def test_full_reference_and_independent_fixed_reference_force():
    rom = model(c_init=1.2, beta_fraction_init=0.7)
    reference = torch.cat((rom.theta_ref, torch.zeros(12, dtype=torch.float64))).requires_grad_()
    torch.testing.assert_close(rom.rhs(None, reference), torch.zeros_like(reference), rtol=0, atol=0)
    storage = rom.storage_power(reference)
    assert storage.H.item() == storage.total_power.item() == 0
    torch.testing.assert_close(torch.autograd.grad(storage.H, reference)[0],
                               torch.zeros_like(reference), rtol=0, atol=1e-15)

    theta_at_ref = torch.cat((rom.theta_ref, sample_state()[3:]))
    ports = rom.ports(theta_at_ref)
    a = theta_at_ref[9:12]
    left, right = rom.edge_index
    kernel = rom.c - rom.beta + rom.beta * torch.exp(torch.tanh(a[left] + a[right]))
    expected_coupling = rom.incidence @ ((kernel - rom.c) * rom.sin_ref)
    torch.testing.assert_close(ports.Fc, expected_coupling)
    assert ports.Fc.norm() > 0
    assert abs(ports.Fc.sum().item()) < 1e-15
    torch.testing.assert_close(ports.tau_m, rom.F @ (rom.Sp * theta_at_ref[6:9]))
    assert abs(ports.tau_env.sum().item()) > 1e-3


def test_storage_gradient_and_signed_open_power_chain():
    rom = model(c_init=1.2, beta_fraction_init=0.7)
    state = torch.stack((sample_state(), -sample_state())).requires_grad_()
    storage = rom.storage_power(state)
    gradient = torch.autograd.grad(storage.H.sum(), state)[0]
    rhs = rom.rhs(None, state)
    torch.testing.assert_close(gradient[..., :3], rom.ports(state).Fc, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close((gradient[..., 9:12] * rhs[..., 9:12]).sum(-1), storage.amplitude_power)
    assert (storage.amplitude_power.abs() > 1e-6).all()
    torch.testing.assert_close((gradient * rhs).sum(dim=-1), storage.total_power, rtol=1e-12, atol=1e-13)


def test_exact_static_limit_and_nonzero_gradient_at_zero_modulation():
    c0, c1 = model("C0"), model("C1", beta_fraction_init=0)
    c1.load_state_dict({**c0.state_dict(), "beta": c1.beta.detach().clone()})
    state = sample_state()
    times = [0.0, 0.01, 0.03]
    assert "beta" not in dict(c0.named_parameters())
    assert torch.equal(c0.rollout(state, times), c1.rollout(state, times))
    assert c0.rhs(None, state)[9:12].norm() > 0
    assert c0.ports(state).tau_env.norm() > 0
    c1.rollout(state, times)[-1, 3:6].square().sum().backward()
    assert c1.beta.grad is not None and (c1.beta.grad.abs() > 1e-10).any()


def test_encoder_gauge_invariance_full_speed_and_batched_rollout_gradient():
    rom = model()
    theta = tensor([[0.2, -0.1, 0.3]]) + tensor([[0.03], [-0.04]])[:, None] * tensor(
        list(range(11))
    )[None, :, None]
    omega = tensor([[0.9, 0.8, 1.1]]).expand_as(theta)
    initial = rom.initialize(theta, omega)
    shifted = rom.initialize(theta + 7, omega)
    assert torch.equal(initial[:, :3], theta[:, -1])
    assert torch.equal(initial[:, 3:6], omega[:, -1])
    assert torch.equal(shifted[:, :3], theta[:, -1] + 7)
    assert (initial[:, 6:].abs() <= rom.init_bounds).all()
    torch.testing.assert_close(initial[:, 6:], shifted[:, 6:])
    torch.testing.assert_close(rom.ports(initial).Fc, rom.ports(shifted).Fc)
    torch.testing.assert_close(rom.rhs(None, initial), rom.rhs(None, shifted))
    with pytest.raises(TypeError):
        rom.initialize(theta, omega, p0=torch.zeros(3))

    times = [0, 0.01, 0.035]
    trajectory = rom.rollout(initial, times)
    assert trajectory.shape == (2, 3, 15)
    assert trajectory.dtype == torch.float64
    assert (trajectory[:, -1, :3].mean(-1) - trajectory[:, 0, :3].mean(-1) > 0.02).all()
    torch.testing.assert_close(trajectory[0], rom.rollout(initial[0], times))
    trajectory[:, -1, 3:6].square().sum().backward()
    assert rom.encoder.weight_ih_l0.grad.norm() > 0


def test_projection_and_bounded_kernel_at_extreme_amplitudes():
    rom = model(c_init=1.5)
    with torch.no_grad():
        rom.c.copy_(tensor([-1.0, 1.0, 2.0]))
        rom.beta.copy_(tensor([1.0, -1.0, 4.0]))
    rom.project_parameters_()
    torch.testing.assert_close(rom.c, tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(rom.beta, tensor([0.0, 0.0, 2.0]))
    strengths = rom.edge_kernel(tensor([[-1e3, -1e3, -1e3], [1e3, 1e3, 1e3]]))
    torch.testing.assert_close(strengths, tensor([[0.0, 1.0, 2 / math.e], [0.0, 1.0, 2 * math.e]]))


@pytest.mark.parametrize("variant", ["C0", "C1"])
def test_checkpoint_roundtrip(variant):
    rom = model(variant)
    payload = rom.checkpoint()
    assert json.loads(json.dumps(payload["config"])) == payload["config"]
    assert set(constants()) <= payload["state_dict"].keys()
    serialized = BytesIO()
    torch.save(payload, serialized)
    serialized.seek(0)
    loaded = OSGKD.from_checkpoint(torch.load(serialized, weights_only=True))
    state = sample_state()
    for name in constants():
        assert torch.equal(getattr(rom, name), getattr(loaded, name))
    assert torch.equal(rom.rollout(state, [0.0, 0.01]), loaded.rollout(state, [0.0, 0.01]))
    theta = state[:3].expand(11, 3)
    omega = state[3:6].expand(11, 3)
    assert torch.equal(rom.initialize(theta, omega), loaded.initialize(theta, omega))
    with pytest.raises(ValueError, match="contract or version"):
        OSGKD.from_checkpoint({**payload, "contract": "known_p0_reference"})


def test_dimension_and_time_grid_validation():
    for name, value in (
        ("F", [[1.0]]), ("mass", [1.0, 0.0, 2.0]), ("damping", [-0.1, 0.1, 0.1]),
        ("Sp", [1.0, -1.0, 1.0]), ("Se", [1.0, float("nan"), 1.0]),
        ("edge_index", [[0, 0], [1, 1]]),
    ):
        with pytest.raises(ValueError, match=name):
            OSGKD(OSGKDConfig("C0"), **{**constants(), name: value})
    rom = model()
    with pytest.raises(ValueError, match="history"):
        rom.initialize(tensor([[1, 2, 3]]), tensor([[1, 2, 3]]))
    with pytest.raises(ValueError, match="5N"):
        rom.rollout(tensor([1.0]), [0.0, 0.01])
    with pytest.raises(ValueError, match="strictly increasing"):
        rom.rollout(sample_state(), [0.0, 0.01, 0.01])


def test_empty_candidate_graph_preserves_node_dynamics():
    rom = OSGKD(OSGKDConfig("C1"), **{
        **constants(), "edge_index": torch.empty((2, 0), dtype=torch.int64),
    })
    state = sample_state()
    assert torch.equal(rom.ports(state).Fc, torch.zeros_like(state[:3]))
    assert rom.storage_power(state).Uc.item() == 0
    assert torch.isfinite(rom.rhs(None, state)).all()
