import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).parents[1] / 'experiments'))
from train_joint_latent import BASE, NETWORK
from train_teacher_latent import (TeacherROM, construct_teacher, freeze_teacher_only)
from kiki.models.private_pm import MechanicalROM


ANCHOR = Path('private_data/task2_joint_0058')


def teacher():
    base, network = MechanicalROM.load(BASE), dict(np.load(NETWORK, allow_pickle=False))
    network['observed_buses'] = np.unique(np.r_[network['load_buses'], network['generator_buses']])
    network['load_positions'] = np.searchsorted(network['observed_buses'], network['load_buses'])
    return construct_teacher(ANCHOR, base, network)


def test_zero_initialized_teacher_field_and_rollout_match_anchor_exactly():
    model, _ = teacher()
    state = torch.randn(3, 33, dtype=torch.float64)
    assert torch.equal(model.field(state), TeacherROM.__mro__[1].field(model, state))
    history = torch.randn(3, 11, 30, dtype=torch.float64)
    teacher_rollout = model.rollout(history, 4, .01)
    state = torch.cat((history[:, -1], model.encode(history)), dim=-1)
    anchor_rollout = [state]
    for _ in range(4):
        a = TeacherROM.__mro__[1].field(model, state)
        b = TeacherROM.__mro__[1].field(model, state+.01/2*a)
        c = TeacherROM.__mro__[1].field(model, state+.01/2*b)
        d = TeacherROM.__mro__[1].field(model, state+.01*c)
        state = state+.01/6*(a+2*b+2*c+d)
        anchor_rollout.append(state)
    assert torch.equal(teacher_rollout, torch.stack(anchor_rollout, dim=1))


def test_only_1264_teacher_parameters_train_and_anchor_state_stays_fixed():
    model, anchor = teacher()
    before = freeze_teacher_only(model, anchor)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1264
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0003)
    state = torch.randn(2, 33, dtype=torch.float64)
    optimizer.zero_grad(); model.field(state)[:, 30:].square().mean().backward(); optimizer.step()
    for name, value in before.items():
        assert torch.equal(model.state_dict()[name], value), name


def test_first_step_has_finite_nonzero_w2_gradient_and_zero_origin_correction():
    model, _ = teacher()
    state = torch.randn(2, 33, dtype=torch.float64)
    model.field(state)[:, 30:].square().mean().backward()
    assert torch.isfinite(model.W2.weight.grad).all() and torch.linalg.vector_norm(model.W2.weight.grad) > 0
    with torch.no_grad():
        model.W2.weight.normal_()
    origin = torch.zeros(1, 33, dtype=torch.float64)
    assert torch.equal(model.field(origin), TeacherROM.__mro__[1].field(model, origin))


def test_analytic_complete_aq_matches_finite_difference():
    model, _ = teacher()
    with torch.no_grad():
        model.W2.weight.normal_()
    origin = torch.zeros(1, 33, dtype=torch.float64)
    epsilon, finite_difference = 1e-6, []
    for coordinate in range(3):
        plus, minus = origin.clone(), origin.clone()
        plus[0, 30+coordinate] += epsilon; minus[0, 30+coordinate] -= epsilon
        finite_difference.append(((model.field(plus)[0, 30:]-model.field(minus)[0, 30:])/(2*epsilon)).detach())
    assert torch.allclose(model.latent_jacobian_at_reference(), torch.stack(finite_difference, dim=1), atol=1e-7, rtol=1e-7)


def test_rollout_encodes_initial_history_once_and_has_no_future_argument(monkeypatch):
    model, _ = teacher()
    calls, encode = {'n': 0}, model.encode
    def counted(history):
        calls['n'] += 1
        return encode(history)
    monkeypatch.setattr(model, 'encode', counted)
    history = torch.randn(2, 11, 30, dtype=torch.float64)
    rollout = model.rollout(history, 3, .01)
    assert calls['n'] == 1
    assert torch.equal(rollout[:, 0, :30], history[:, -1])
