"""Model-only contracts for the Loop0078 trainer; no private events are read."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
spec = importlib.util.spec_from_file_location("structured_train", ROOT / "experiments" / "train_structured_rom.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


def source_fixture():
    from kiki.models.private_pm import MechanicalROM
    from kiki.physics import complete_incidence
    incidence = complete_incidence(10)[0]
    coupling = np.r_[np.ones(26), np.zeros(19)]
    base = MechanicalROM(np.linspace(1, 2, 10), incidence, np.zeros(10), np.zeros(10),
                         coupling, np.ones(10) * .1, np.ones(10) * .2,
                         np.ones(10) * .3)
    source = screen.JointROM(base, np.eye(10), np.ones(20), np.ones(72), 1., 1.,
                             electrical=(np.ones(20), np.ones(10), np.arange(10), np.ones(10), 10.), latent_dim=3)
    return base, source


def test_coordinate_translation_preserves_old_decoder_at_initialization():
    _, source = source_fixture()
    model = screen.CFreeROM(source)
    h = torch.as_tensor(np.random.default_rng(1).normal(scale=.1, size=(4, 11, 30)), dtype=torch.float64)
    old_q = source.encode(h)
    new_q = model.encode(h)
    old_z, old_load = source.electrical_output(h[:, -1], old_q)
    new_z, new_load = model.electrical_output(h[:, -1], new_q)
    assert torch.allclose(old_z, new_z, atol=1e-12, rtol=0)
    assert torch.allclose(old_load, new_load, atol=1e-12, rtol=0)
    assert model.head.bias is None or not model.head.bias.requires_grad


@pytest.mark.parametrize("arm", ["G-auto", "G-fixed"])
def test_g_owns_only_structured_generator_and_cached_field_is_exact(arm):
    base, source = source_fixture()
    model = screen.GROM(base, source, arm)
    assert not any("xi" in name for name, _ in model.named_parameters())
    z = torch.as_tensor(np.random.default_rng(2).normal(scale=.03, size=(3, 33)), dtype=torch.float64)
    direct = model.field(z)
    model.begin_graph_cache(); cached = model.field(z); loss = cached.square().mean(); model.end_graph_cache(); loss.backward()
    assert torch.allclose(direct, cached.detach(), atol=1e-12, rtol=0)
    assert torch.isfinite(model.structured.coupling.grad).all()
    assert torch.isfinite(model.structured.port_direction[0].grad).all()
    for name, parameter in model.named_parameters():
        if any(key in name for key in ("exchange_a", "exchange_b", "dissipation_a")):
            assert not parameter.requires_grad


def test_growth_adds_a_new_adam_group_without_resetting_existing_moments():
    base, source = source_fixture(); model = screen.GROM(base, source, "G-auto")
    optimizer = screen.optimizer_for(model)
    z = torch.as_tensor(np.random.default_rng(3).normal(scale=.02, size=(2, 33)), dtype=torch.float64)
    model.field(z).square().mean().backward(); optimizer.step()
    old = model.structured.coupling
    before = optimizer.state[old]["exp_avg"].detach().clone()
    new = model.add_storage(screen.PrimitiveSpec("product", (0, 10), 1e-3, (1., 1.)), normalized=99.)[0]
    screen.add_growth_parameters(optimizer, model, [new])
    assert torch.equal(optimizer.state[old]["exp_avg"], before)
    assert any(any(p is new for p in group["params"]) and group["lr"] == pytest.approx(1e-3)
               for group in optimizer.param_groups)
    assert float(new.detach()) == pytest.approx(1e-3)


def test_fixed_schedule_and_parameter_type_learning_rates():
    assert [screen.horizon_for_step(i) for i in (1, 100, 101, 102, 300, 301, 302, 303)] == [1., 1., 1., 2., 2., 1., 2., 5.]
    base, source = source_fixture(); model = screen.CFreeROM(source)
    assert screen._parameter_lr(model, "gru.weight_ih_l0") == 1e-4
    assert screen._parameter_lr(model, "damping_raw") == 1e-2
    assert screen._parameter_lr(model, "core.xi") == 1e-3


def test_random_endpoint_rollout_targets_its_own_future_not_index_ten():
    class Static:
        mass = torch.ones(10, dtype=torch.float64)
        p_scale = torch.ones(10, dtype=torch.float64)
        def encode(self, h): return torch.zeros((len(h), 3), dtype=h.dtype)
        def rk4(self, x, dt): return x
    event = {"x": np.zeros((24, 30))}
    event["x"][21, 20] = 2.0  # endpoint 20's next state; x[11] remains zero.
    history = torch.as_tensor(event["x"][10:21][None], dtype=torch.float64)
    error = screen._event_rollout_errors(Static(), [(event, 20)], history, .01)
    assert float(error) == pytest.approx(.4)  # mean over ten p components


def test_model_only_reload_rebuilds_structured_parameter_lists(tmp_path):
    base, source = source_fixture(); model = screen.GROM(base, source, "G-auto")
    model.add_storage(screen.PrimitiveSpec("product", (0, 10), 1e-3, (1., 1.)), 1.)
    model.add_exchange(0, 20, -1e-3); model.add_dissipation(22, 1e-3)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    model.structured.export(tmp_path / "structured_numeric_export.json")
    (tmp_path / "checkpoint.json").write_text(__import__("json").dumps({
        "arm": "G-auto", "source": "unused", "extra_groups": model.extra_groups,
        "requires_grad": {name: bool(p.requires_grad) for name, p in model.named_parameters()},
    }))
    loaded = screen.load_model(tmp_path, base, source)
    z = torch.as_tensor(np.random.default_rng(9).normal(scale=.01, size=(2, 33)), dtype=torch.float64)
    assert torch.allclose(model.field(z), loaded.field(z), atol=1e-12, rtol=0)
    assert loaded.extra_groups == model.extra_groups


def test_force_truth_buffers_are_immutable_when_g_empirical_coupling_moves():
    base, source = source_fixture(); model = screen.GROM(base, source, "G-auto")
    before = model.frozen_coupling.detach().clone()
    with torch.no_grad(): model.structured.coupling.add_(1.)
    assert torch.equal(model.frozen_coupling, before)
    assert not torch.equal(model.structured._pos("coupling"), before[before > 0])
