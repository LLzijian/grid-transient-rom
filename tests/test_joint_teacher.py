"""Synthetic Loop0072 contracts; no real TRAIN/H5/TEST access."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from train_joint_latent import JointROM, tensor  # noqa: E402
from train_joint_teacher import (  # noqa: E402
    JOINT, TEACHER_NAMES, _assert_zero_field, _contract, configure_trainable, objective,
)
from train_teacher_latent import TeacherROM  # noqa: E402
from kiki.models.private_pm import MechanicalROM  # noqa: E402
from kiki.physics import complete_incidence  # noqa: E402


def base_fixture():
    return MechanicalROM(np.linspace(1, 2, 10), complete_incidence(10)[0], np.linspace(0, .2, 10),
                         np.ones(10), np.r_[np.ones(26), np.zeros(19)], np.ones(10) * .1,
                         np.ones(10) * 2, np.ones(10) * .3)


def teacher_fixture():
    base = base_fixture()
    ports = np.eye(10)
    anchor = JointROM(base, ports, np.ones(20), np.ones(72), 1., 1., latent_dim=3)
    with torch.no_grad():
        anchor.support[0].zero_()
        anchor.xi[0].zero_()
    teacher = TeacherROM(base, ports, np.ones(20), np.ones(72), 1., 1., latent_dim=3)
    missing, unexpected = teacher.load_state_dict(anchor.state_dict(), strict=False)
    assert set(missing) == {"W1.weight", "W1.bias", "W2.weight"} and not unexpected
    return anchor, teacher


def event(name, label, group, length=90):
    seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=.02, size=(length, 30))
    return {"id": name, "label": label, "group": group, "x": x,
            "force": rng.normal(size=(length, 10)) * .01,
            "z": rng.normal(size=(length, 20)) * .01,
            "load": rng.normal(size=length) * .01}


def batch():
    pool = [event("s1", "stable", "1"), event("s2", "stable", "2"),
            event("u1", "unstable", "3"), event("u2", "unstable", "4")]
    samples = [(item, 30 + index) for index, item in enumerate(pool)]
    histories = tensor(np.stack([item["x"][endpoint - 10:endpoint + 1] for item, endpoint in samples]))
    return samples, histories


def test_teacher_initial_field_equals_anchor_and_f0_is_zero():
    anchor, teacher = teacher_fixture()
    states = torch.arange(4 * 33, dtype=torch.float64).reshape(4, 33) / 1000.
    with torch.no_grad():
        torch.testing.assert_close(anchor.field(states), teacher.field(states), rtol=0., atol=0.)
    assert _assert_zero_field(teacher) <= 1e-10


def test_allowed_set_is_exact_joint_plus_teacher_tensors_and_buffers_freeze():
    anchor, teacher = teacher_fixture()
    source = {name: value.detach().clone() for name, value in teacher.state_dict().items()
              if name in anchor.state_dict()}
    assert set(configure_trainable(teacher)) == set(JOINT) | {"W1.weight", "W1.bias", "W2.weight"}
    assert {name for name, parameter in teacher.named_parameters() if parameter.requires_grad} == set(TEACHER_NAMES)
    assert len(list(teacher.named_parameters())) == 15
    assert sum(parameter.numel() for name, parameter in teacher.named_parameters()
               if name in {"W1.weight", "W1.bias", "W2.weight"}) == 1264
    assert sum(parameter.numel() for parameter in teacher.parameters()) == 2708
    assert all(not parameter.requires_grad for name, parameter in teacher.named_parameters() if name not in TEACHER_NAMES)
    checks = _contract(teacher, source, set(TEACHER_NAMES))
    assert all(checks.values())


def test_teacher_gradients_have_zero_initial_w1_then_downstream_w1_gradient():
    _, teacher = teacher_fixture()
    source = {name: value.detach().clone() for name, value in teacher.state_dict().items()
              if name not in {"W1.weight", "W1.bias", "W2.weight"}}
    configure_trainable(teacher)
    samples, histories = batch()
    loss, _ = objective(teacher, samples, histories, 2.)
    loss.backward()
    assert torch.linalg.vector_norm(teacher.W1.weight.grad) <= 1e-14
    assert torch.linalg.vector_norm(teacher.W1.bias.grad) <= 1e-14
    assert torch.linalg.vector_norm(teacher.W2.weight.grad) > 0
    optimizer = torch.optim.Adam([parameter for name, parameter in teacher.named_parameters()
                                  if name in TEACHER_NAMES], lr=1e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = objective(teacher, samples, histories, 2.)
    loss.backward()
    for name in TEACHER_NAMES:
        parameter = dict(teacher.named_parameters())[name]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert torch.linalg.vector_norm(teacher.W1.weight.grad) > 0
    assert torch.linalg.vector_norm(teacher.W1.bias.grad) > 0
    assert all(_contract(teacher, source, set(TEACHER_NAMES)).values())


def test_mutated_teacher_buffer_fails_frozen_contract():
    _, teacher = teacher_fixture()
    source = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    configure_trainable(teacher)
    with torch.no_grad():
        teacher.support[1].add_(1.0)
    checks = _contract(teacher, source, set(TEACHER_NAMES))
    assert not checks["support_fixed"]
    assert not all(checks.values())
