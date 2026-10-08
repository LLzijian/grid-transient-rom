"""Synthetic Loop0074 rank-4 mapping/support contracts; no TRAIN/H5/TEST access."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from train_joint_campaign import JOINT, horizon_for_step  # noqa: E402
from train_joint_latent import JointROM  # noqa: E402
from train_rank4_sparse import (  # noqa: E402
    KEEP_COUNT, NONCONSTANT_COUNT, RANK, RANK_OLD, XI_Q4_COL, XI_Q4_ROW,
    _initial_equivalence, _phase_contract, _rank4_preflight, expand_rank3_state,
    open_rank4_support, prune_support, rank4_objective, rank_support,
)
import train_rank4_sparse as rank4_sparse  # noqa: E402
from kiki.models.private_pm import MechanicalROM  # noqa: E402
from kiki.physics import complete_incidence  # noqa: E402


def base_fixture():
    return MechanicalROM(np.linspace(1, 2, 10), complete_incidence(10)[0], np.linspace(0, .2, 10),
                         np.ones(10), np.r_[np.ones(26), np.zeros(19)], np.ones(10) * .1,
                         np.ones(10) * 2, np.ones(10) * .3)


def rank_fixture():
    base = base_fixture()
    ports = np.zeros((10, 11)); ports[:, :10] = np.eye(10)
    old = JointROM(base, ports, np.ones(20), np.ones(72), 1., 1., latent_dim=RANK_OLD)
    with torch.no_grad():
        old.c.zero_(); old.c[:3, :3] = torch.eye(3)
        old.support[0].zero_(); old.xi[0].zero_()
    new = JointROM(base, ports, np.ones(20), np.ones(72), 1., 1., latent_dim=RANK)
    mapped, init = expand_rank3_state(old.state_dict(), new.ports, new.mass)
    new.load_state_dict(mapped, strict=True)
    return old, new, init


def event(name: str, length: int = 90):
    rng = np.random.default_rng(sum((i + 1) * ord(c) for i, c in enumerate(name)))
    return {"id": name, "label": "stable", "group": name, "x": rng.normal(size=(length, 30)) * .01,
            "force": rng.normal(size=(length, 10)) * .01,
            "z": rng.normal(size=(length, 20)) * .01, "load": rng.normal(size=length) * .01}


def test_rank4_state_mapping_preserves_old_channels_and_initializes_new_channel():
    old, new, init = rank_fixture()
    assert new.c.shape == (11, 4)
    assert new.head.weight.shape[0] == 4 and new.decoder.weight.shape[1] == 14
    assert new.xi.shape == (77, 4) and new.alpha.shape == (52, 4)
    assert torch.equal(new.head.weight[:3], old.head.weight)
    assert torch.equal(new.decoder.weight[:, :13], old.decoder.weight)
    assert torch.equal(new.q_transform[:3, :3], old.q_transform)
    assert torch.equal(new.q_transform[3, 3], torch.ones((), dtype=torch.float64))
    assert torch.equal(new.alpha[:, 3], torch.zeros(52, dtype=torch.float64))
    assert torch.equal(new.alpha_support[:, 3], torch.zeros(52, dtype=new.alpha_support.dtype))
    assert new.xi[XI_Q4_ROW, XI_Q4_COL] == -1.
    assert init["port_residual"] <= 1e-10 and init["mass_span_relative_distance"] > 1e-6
    assert init["amplitude4"] > 1e-8


def test_initial_field_and_auxiliary_outputs_are_exact_on_q4_zero_embedding():
    old, new, _ = rank_fixture()
    errors = _initial_equivalence(old, new)
    assert max(errors.values()) <= 1e-10
    states = torch.zeros((1, 34), dtype=torch.float64)
    assert float(new.field(states).detach().abs().max()) <= 1e-10


def test_open_support_exposes_exact_304_nonconstant_entries_and_keeps_constant_closed():
    _, new, _ = rank_fixture()
    open_rank4_support(new)
    assert int(new.support.sum()) == NONCONSTANT_COUNT
    assert torch.equal(new.support[0], torch.zeros_like(new.support[0]))
    assert new.xi[XI_Q4_ROW, XI_Q4_COL] == -1.


def test_rank4_support_selection_uses_positive_row_major_ties():
    scores = np.zeros((77, 4)); coefficients = np.ones((77, 4)); support = np.ones((77, 4), dtype=bool)
    scores[1:] = 1.
    selected, flat = rank_support(scores, coefficients, support)
    assert len(flat) == KEEP_COUNT
    assert flat.tolist() == list(range(KEEP_COUNT))
    assert np.flatnonzero(selected.reshape(-1)).tolist() == [index + 4 for index in range(KEEP_COUNT)]


def test_rank4_objective_applies_fixed_component_denominators():
    old, model, _ = rank_fixture()
    samples = [(event("a"), 20), (event("b"), 21), (event("c"), 22), (event("d"), 23)]
    histories = torch.as_tensor(np.stack([item["x"][endpoint - 10:endpoint + 1] for item, endpoint in samples]), dtype=torch.float64)
    old_loss, old_terms = rank4_sparse.objective(old, samples, histories, horizon_for_step(1))
    loss, terms = rank4_objective(model, samples, histories, horizon_for_step(1))
    assert torch.isfinite(loss)
    assert torch.equal(terms["xi_l1"], (model.xi * model.support).abs().sum() / (76 * 3))
    assert terms["latent_consistency"] >= 0 and terms["equilibrium"] >= 0
    assert torch.allclose(terms["latent_consistency"], old_terms["latent_consistency"], atol=1e-12, rtol=0.)
    assert torch.allclose(terms["equilibrium"], old_terms["equilibrium"], atol=1e-12, rtol=0.)
    assert torch.allclose(loss, old_loss + torch.as_tensor(1e-4 / (76 * 3)), atol=1e-12, rtol=0.)


def test_prune_support_end_to_end_zeroes_inactive_and_writes_aligned_artifact(monkeypatch, tmp_path):
    _, model, _ = rank_fixture()
    open_rank4_support(model)
    with torch.no_grad():
        model.xi[1:].fill_(1.)
        model.xi[0].zero_()
        model.xi[XI_Q4_ROW, XI_Q4_COL] = -1.
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    samples = [(event("prune"), 20)] * 512
    histories = torch.as_tensor(np.stack([samples[0][0]["x"][10:21]] * 512), dtype=torch.float64)

    def fake_sample_batch(train_events, rng, *, count):
        assert count == 512
        return samples, histories

    monkeypatch.setattr(rank4_sparse, "sample_batch", fake_sample_batch)
    report = prune_support(model, [samples[0][0]], tmp_path)
    artifact = np.load(tmp_path / "support_rediscovery.npz")
    assert report["candidate_count"] == NONCONSTANT_COUNT
    assert 0 < report["selected_count"] <= KEEP_COUNT
    assert artifact["scores"].shape == (NONCONSTANT_COUNT,)
    assert artifact["candidate_rows"].shape == (NONCONSTANT_COUNT,)
    assert artifact["candidate_cols"].shape == (NONCONSTANT_COUNT,)
    scores77 = np.zeros((77, 4)); scores77[1:] = artifact["scores"].reshape(76, 4)
    coefficients77 = np.zeros((77, 4)); coefficients77[1:] = artifact["coefficients_before"].reshape(76, 4)
    expected, expected_flat = rank_support(scores77, coefficients77, artifact["support_before"])
    assert report["selected_row_major_indices"] == expected_flat.tolist()
    assert np.array_equal(artifact["support_after"], expected)
    assert np.all(artifact["coefficients_after"][artifact["support_after"][1:].reshape(-1) == 0.] == 0.)
    assert torch.equal(model.xi[model.support == 0], torch.zeros_like(model.xi[model.support == 0]))
    assert all(_phase_contract(model, source, model.support).values())


def test_rank4_preflight_jacobian_and_spectrum_extension_are_finite():
    old, new, _ = rank_fixture()
    preflight = _rank4_preflight(old, new)
    assert preflight["jacobian_fd_max_abs"] < 1e-6
    assert preflight["jacobian_block_max_abs"] < 1e-10
    assert len(preflight["old_eigenvalues"]) == 33
    assert len(preflight["new_eigenvalues"]) == 34


def test_rank4_parameter_contract_and_schedule_constants():
    _, new, _ = rank_fixture()
    assert {name for name, _ in new.named_parameters()} == set(JOINT)
    assert RANK == 4 and RANK_OLD == 3 and NONCONSTANT_COUNT == 304
    assert [horizon_for_step(i) for i in (1, 100, 101, 102, 103)] == [2., 2., 1., 2., 5.]


def test_new_head_and_xi_paths_reach_physical_rollout_loss():
    _, model, _ = rank_fixture()
    open_rank4_support(model)
    rng = np.random.default_rng(7401)
    histories = torch.as_tensor(rng.normal(size=(4, 11, 30)) * .02, dtype=torch.float64)
    q = model.encode(histories)
    state = torch.cat((histories[:, -1], q), dim=-1)
    rollout = model.rk4(model.rk4(state, .01), .01)
    physical_loss = rollout[:, :20].square().mean()
    head_grad, xi_grad = torch.autograd.grad(physical_loss, (model.head.weight, model.xi), allow_unused=True)
    assert head_grad is not None and torch.linalg.vector_norm(head_grad[3]) > 0.
    assert xi_grad is not None and torch.linalg.vector_norm(xi_grad[:, 3]) > 0.
