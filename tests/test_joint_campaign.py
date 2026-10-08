"""Synthetic Loop0067 contracts; no private data or TEST payloads."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
spec = importlib.util.spec_from_file_location("joint_campaign", Path(__file__).resolve().parents[1] / "experiments/train_joint_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def model_fixture(rank=3):
    from kiki.models.private_pm import MechanicalROM
    from kiki.physics import complete_incidence
    b = complete_incidence(10)[0]
    base = MechanicalROM(np.linspace(1, 2, 10), b, np.linspace(0, .2, 10),
                         np.ones(10), np.r_[np.ones(26), np.zeros(19)],
                         np.ones(10) * .1, np.ones(10) * 2,
                         np.ones(10) * .3)
    from train_joint_latent import JointROM
    model = JointROM(base, np.eye(10), np.ones(20), np.ones(72), 1., 1., latent_dim=rank)
    # The production source has the constant Xi row inactive; preserve that
    # invariant in the synthetic fixture.
    with torch.no_grad():
        model.support[0].zero_()
    return model


def event(name, label, group, length=60):
    stable_seed = sum((index + 1) * ord(character) for index, character in enumerate(name)) % (2**32)
    rng = np.random.default_rng(stable_seed)
    return {"id": name, "label": label, "group": group,
            "x": rng.normal(scale=.03, size=(length, 30)),
            "force": rng.normal(scale=.02, size=(length, 10)),
            "z": rng.normal(scale=.02, size=(length, 20)),
            "load": rng.normal(scale=.02, size=length)}


def events():
    return [event(f"s{g}{i}", "stable", str(g)) for g in (1, 2) for i in range(2)] + \
           [event(f"u{g}{i}", "unstable", str(g)) for g in (3, 4) for i in range(2)]


def test_sampler_is_deterministic_stratified_and_reaches_full_endpoint_range():
    pool = events()
    a, ha = campaign.sample_batch(pool, np.random.default_rng(67), count=32)
    b, hb = campaign.sample_batch(pool, np.random.default_rng(67), count=32)
    assert [(e["id"], i) for e, i in a] == [(e["id"], i) for e, i in b]
    assert torch.equal(ha, hb)
    assert [_label for _label in [e["label"] for e, _ in a]] == ["stable", "unstable"] * 16
    assert all(10 <= i <= len(e["x"]) - 2 for e, i in a)


def test_sampler_uses_all_available_endpoint_range():
    pool = [event("s", "stable", "1", 1000), event("u", "unstable", "2", 1000)]
    samples, _ = campaign.sample_batch(pool, np.random.default_rng(8), count=4096)
    endpoints = np.asarray([i for _, i in samples])
    assert endpoints.min() >= 10 and endpoints.max() <= 998
    assert endpoints.max() > 810


def test_schedule_is_fixed_cycle():
    assert campaign.horizon_for_step(1) == 2.
    assert campaign.horizon_for_step(100) == 2.
    assert [campaign.horizon_for_step(i) for i in (101, 102, 103, 104, 105)] == [1., 2., 5., 1., 2.]


def test_configure_c_and_j_exact_parameter_sets_and_constant_support_rejection():
    model = model_fixture()
    assert set(campaign.configure_trainable(model, "C")) == {"xi"}
    assert [name for name, p in model.named_parameters() if p.requires_grad] == ["xi"]
    assert set(campaign.configure_trainable(model, "J")) == set(campaign.JOINT)
    assert {name for name, p in model.named_parameters() if p.requires_grad} == set(campaign.JOINT)
    with torch.no_grad():
        model.support[0, 0] = 1.
    with pytest.raises(ValueError, match="constant active support"):
        campaign.configure_trainable(model, "C")


def test_joint_objective_reaches_all_twelve_parameter_families_and_freezes_masks():
    model = model_fixture()
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    allowed = set(campaign.configure_trainable(model, "J"))
    samples = [(e, 20 + i) for i, e in enumerate(events())]
    histories = campaign.tensor(np.stack([e["x"][i - 10:i + 1] for e, i in samples]))
    loss, _ = campaign.objective(model, samples, histories, 0.02)
    loss.backward()
    for name in campaign.JOINT:
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert float(torch.linalg.vector_norm(parameter.grad)) > 0., name
    # The electrical objective must reach the decoder itself, not merely the
    # mechanical/latent branches.
    assert float(torch.linalg.vector_norm(model.decoder.weight.grad)) > 0.
    assert float(torch.linalg.vector_norm(model.decoder.bias.grad)) > 0.
    optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if n in allowed], lr=1e-4)
    optimizer.step()
    assert campaign._frozen_equal(model, source, allowed)
    assert campaign._masked_inactive_equal(model, source)


def test_c_only_has_xi_gradient_and_all_other_tensors_are_frozen():
    model = model_fixture()
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    allowed = set(campaign.configure_trainable(model, "C"))
    samples = [(e, 20 + i) for i, e in enumerate(events())]
    histories = campaign.tensor(np.stack([e["x"][i - 10:i + 1] for e, i in samples]))
    loss, _ = campaign.objective(model, samples, histories, 0.02)
    loss.backward()
    assert model.xi.grad is not None and torch.isfinite(model.xi.grad).all()
    assert all((p.grad is None) for n, p in model.named_parameters() if n not in allowed)
    optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if n in allowed], lr=1e-4)
    optimizer.step()
    assert campaign._frozen_equal(model, source, allowed)
    for name, value in model.state_dict().items():
        if name != "xi":
            assert torch.equal(value, source[name]), name


def test_common_frequency_term_detects_uniform_omega_error_after_coi_removal():
    model = model_fixture()
    truth = torch.zeros((2, 30), dtype=torch.float64)
    pred = truth.clone()
    pred[:, 10:20] = .2
    err = campaign._point_error(pred, truth, model)
    coi_only = model.coi(pred[:, 10:20]).square().mean()
    assert float(coi_only) == pytest.approx(0.)
    assert torch.all(err > 0)


def test_zero_field_contract_and_no_future_padding_in_objective():
    model = model_fixture()
    campaign.configure_trainable(model, "J")
    assert campaign._assert_zero_field(model) <= 1e-10
    short = event("short", "stable", "1", 14)
    long = event("long", "unstable", "2", 60)
    samples = [(short, 10), (long, 20)]
    histories = campaign.tensor(np.stack([e["x"][i - 10:i + 1] for e, i in samples]))
    total, terms = campaign.objective(model, samples, histories, .2)
    assert torch.isfinite(total) and torch.isfinite(terms["risk"])
