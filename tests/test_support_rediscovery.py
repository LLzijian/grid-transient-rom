"""Synthetic Loop0070 support-opening/pruning contracts; no real data access."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import train_support_rediscovery as rediscovery  # noqa: E402
from train_joint_latent import JointROM, tensor  # noqa: E402
from kiki.models.private_pm import MechanicalROM  # noqa: E402
from kiki.physics import complete_incidence  # noqa: E402


def model_fixture():
    base = MechanicalROM(np.linspace(1, 2, 10), complete_incidence(10)[0], np.linspace(0, .2, 10),
                         np.ones(10), np.r_[np.ones(26), np.zeros(19)], np.ones(10) * .1,
                         np.ones(10) * 2, np.ones(10) * .3)
    model = JointROM(base, np.eye(10), np.ones(20), np.ones(72), 1., 1., latent_dim=3)
    with torch.no_grad():
        model.support[0].zero_()
        model.xi[0].zero_()
    return model


def event(name, label, group, length=90):
    seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=.02, size=(length, 30))
    return {"id": name, "label": label, "group": group, "x": x,
            "force": rng.normal(size=(length, 10)) * .01,
            "z": rng.normal(size=(length, 20)) * .01,
            "load": rng.normal(size=length) * .01}


def events():
    return [event("s1", "stable", "1"), event("s2", "stable", "2"),
            event("u1", "unstable", "3"), event("u2", "unstable", "4")]


def test_open_support_zeroes_stale_inactive_entries_and_preserves_constant_row():
    model = model_fixture()
    with torch.no_grad():
        model.support[5, 2] = 0.
        model.xi[5, 2] = 3.5
        model.xi[0] = torch.tensor([4.6652092440666174e-08, -2.3338410058036076e-07, 1.258795766104651e-07])
        states = torch.randn(9, 33, dtype=torch.float64)
        before_field = model.field(states).detach().clone()
    before_support, before_xi = rediscovery.open_nonconstant_support(model)
    assert before_support[5, 2] == 0 and before_xi[5, 2] == 3.5
    assert torch.all(model.support[1:] == 1)
    assert model.support[0].eq(0).all() and model.xi[0].eq(0).all()
    assert model.xi[5, 2] == 0
    torch.testing.assert_close(model.field(states), before_field, rtol=0., atol=0.)


def test_open_support_rejects_active_constant_mask():
    model = model_fixture()
    with torch.no_grad():
        model.support[0, 1] = 1.
    with pytest.raises(ValueError, match="constant Xi row"):
        rediscovery.open_nonconstant_support(model)


def test_top48_positive_scores_have_row_major_tie_break_and_exclude_constant():
    scores = np.zeros((76, 3))
    scores[1:] = 1.
    selected, indices = rediscovery.rank_support(scores, np.ones_like(scores), np.zeros_like(scores), keep=48)
    np.testing.assert_array_equal(indices, np.arange(48))
    assert selected[0].sum() == 0
    assert int(selected.sum()) == 48


def test_actual_prune_artifact_matches_encode_endpoint_rms_and_zeroes_dropped(tmp_path):
    model = model_fixture()
    rediscovery.open_nonconstant_support(model)
    with torch.no_grad():
        model.xi[1:] = .01
    pool = events()
    before = model.xi.detach().cpu().numpy().copy()
    report = rediscovery.prune_support(model, pool, tmp_path)
    with np.load(tmp_path / "support_rediscovery.npz", allow_pickle=False) as artifact:
        ids = [str(value) for value in artifact["sample_event_ids"]]
        endpoints = artifact["sample_endpoints"].tolist()
        theta_rms = artifact["theta_rms"]
        scores = artifact["scores"]
        coeff_before = artifact["coefficients_before"]
        coeff_after = artifact["coefficients_after"]
        support_before = artifact["support_before"]
        support_after = artifact["support_after"]
        assert report["candidate_count"] == 225
        assert len(scores) == len(coeff_before) == len(coeff_after) == 225
        assert len(ids) == len(endpoints) == 512
        assert int(support_before[1:].sum()) == 225
        assert int(support_after.sum()) <= 48
        np.testing.assert_allclose(scores, np.abs(coeff_before) * np.repeat(theta_rms[1:], 3))
        np.testing.assert_allclose(coeff_after[support_after[1:].reshape(-1).astype(bool)],
                                   coeff_before[support_after[1:].reshape(-1).astype(bool)])
        assert np.all(coeff_after[~support_after[1:].reshape(-1).astype(bool)] == 0.)
        event_map = {item["id"]: item for item in pool}
        histories = tensor(np.stack([event_map[event_id]["x"][endpoint - 10:endpoint + 1]
                                     for event_id, endpoint in zip(ids, endpoints, strict=True)]))
        with torch.no_grad():
            q = model.encode(histories)
            state = torch.cat((histories[:, -1], q), dim=-1)
            expected_rms = torch.sqrt(model.features(state).square().mean(dim=0)).numpy()
        np.testing.assert_allclose(theta_rms, expected_rms)
    assert np.allclose(before[1:], np.asarray(coeff_before).reshape(75, 3))


def test_phase_contract_masks_and_frozen_buffers():
    model = model_fixture()
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rediscovery.open_nonconstant_support(model)
    rediscovery.configure_trainable(model, "J")
    checks = rediscovery._phase_contract(model, source)
    assert all(checks.values())
    mask = torch.zeros_like(model.support)
    mask[1, 0] = 1.
    with torch.no_grad():
        model.support.copy_(mask)
        model.xi[~mask.bool()] = 0.
    checks = rediscovery._phase_contract(model, source, support_expected=mask)
    assert all(checks.values())
    assert torch.equal(model.alpha_support, source["alpha_support"])


def test_joint_gradient_reaches_electrical_decoder_after_support_opening():
    model = model_fixture()
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    rediscovery.open_nonconstant_support(model)
    rediscovery.configure_trainable(model, "J")
    pool = events()
    samples = [(pool[index], 30 + index) for index in range(4)]
    histories = tensor(np.stack([item[0]["x"][item[1] - 10:item[1] + 1] for item in samples]))
    loss, _ = rediscovery.objective(model, samples, histories, 2.)
    loss.backward()
    for name in rediscovery.JOINT:
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
    assert torch.linalg.vector_norm(model.decoder.weight.grad) > 0
    assert torch.linalg.vector_norm(model.decoder.bias.grad) > 0
    assert rediscovery._phase_contract(model, source)["alpha_support_fixed"]


def test_pruner_rng_does_not_advance_training_rng():
    pool = events()
    train_a = np.random.default_rng(67)
    train_b = np.random.default_rng(67)
    rediscovery.sample_batch(pool, train_a, count=32)
    rediscovery.sample_batch(pool, np.random.default_rng(rediscovery.PRUNE_SEED), count=512)
    next_a, _ = rediscovery.sample_batch(pool, train_a, count=32)
    rediscovery.sample_batch(pool, train_b, count=32)
    next_b, _ = rediscovery.sample_batch(pool, train_b, count=32)
    assert [(item[0]["id"], item[1]) for item in next_a] == [(item[0]["id"], item[1]) for item in next_b]


def test_fixed_schedule_is_reused():
    assert [rediscovery.horizon_for_step(step) for step in (1, 100, 101, 102, 103, 104)] == [2., 2., 1., 2., 5., 1.]
