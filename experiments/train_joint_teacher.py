"""Loop0072 joint nonlinear-memory capacity probe.

The TeacherROM constructor and nonlinear correction remain unchanged in
``train_teacher_latent.py``.  This runner only supplies the preregistered
S-start, Loop0067 objective/sampler/schedule, joint parameter set, and
failure-preserving evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_long_latent import _construct
from train_joint_campaign import (
    BATCH_SIZE, JOINT, LR, SOURCE, SOURCE_MODEL_SHA256, STEPS, _assert_zero_field,
    horizon_for_step, objective, sample_batch,
)
from train_joint_latent import BASE, NETWORK, read_development
from train_teacher_latent import TeacherROM, construct_teacher
from kiki.models.private_pm import MechanicalROM

ROOT = Path(__file__).resolve().parents[1]
TEACHER_NAMES = JOINT + ("W1.weight", "W1.bias", "W2.weight")
INITIAL_W1_TOL = 1e-14


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        "model": digest(SOURCE / "model.pt"),
        "normalization": digest(SOURCE / "normalization.npz"),
        "base": digest(BASE), "network": digest(NETWORK),
        "code": digest(Path(__file__)),
        "campaign_code": digest(ROOT / "experiments/train_joint_campaign.py"),
        "latent_code": digest(ROOT / "experiments/train_joint_latent.py"),
        "teacher_code": digest(ROOT / "experiments/train_teacher_latent.py"),
        "evaluate_code": digest(ROOT / "experiments/evaluate_long_latent.py"),
    }


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def configure_trainable(model: TeacherROM) -> tuple[str, ...]:
    """Enable exactly J0067's 12 parameters plus the three teacher tensors."""
    observed = {name for name, _ in model.named_parameters()}
    expected = set(JOINT) | {"W1.weight", "W1.bias", "W2.weight"}
    if observed != expected:
        raise ValueError(f"unexpected teacher parameter names: {sorted(observed ^ expected)}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in expected)
    return TEACHER_NAMES


def _frozen_equal(model: TeacherROM, source: dict[str, torch.Tensor], allowed: set[str]) -> bool:
    for name, value in model.state_dict().items():
        if name not in allowed and name in source and not torch.equal(value, source[name]):
            return False
    return True


def _inactive_equal(model: TeacherROM, source: dict[str, torch.Tensor]) -> bool:
    for name, mask_name in (("xi", "support"), ("alpha", "alpha_support")):
        inactive = source[mask_name] == 0
        if not torch.equal(model.state_dict()[name][inactive], source[name][inactive]):
            return False
    return True


def _contract(model: TeacherROM, source: dict[str, torch.Tensor], allowed: set[str]) -> dict[str, bool]:
    return {
        "frozen_source": _frozen_equal(model, source, allowed),
        "inactive_xi_alpha": _inactive_equal(model, source),
        "support_fixed": torch.equal(model.support, source["support"]),
        "alpha_support_fixed": torch.equal(model.alpha_support, source["alpha_support"]),
        "f0_zero": _assert_zero_field(model) <= 1e-10,
    }


def _initial_field_check(anchor_model, teacher: TeacherROM) -> float:
    values = torch.arange(4 * 33, dtype=torch.float64).reshape(4, 33) / 1000.
    with torch.no_grad():
        anchor_field = anchor_model.field(values)
        teacher_field = teacher.field(values)
    error = float(torch.abs(anchor_field - teacher_field).max())
    if error != 0.0:
        raise AssertionError(f"initial TeacherROM field differs from S: {error}")
    return error


def _failure(output: Path, mode: str, step: int, phase: str, reason: str,
             traces: list[dict[str, Any]], source_before: dict[str, str], model: TeacherROM,
             failed_provenance: dict[str, Any] | None) -> None:
    torch.save(model.state_dict(), output / "failure_model.pt")
    (output / "failure_evidence.json").write_text(json.dumps({
        "arm": "J-teacher", "mode": mode, "step": step, "phase": phase,
        "reason": reason, "traces": traces,
        "failed_provenance": failed_provenance,
        "source_hashes_before": source_before,
        "source_hashes_after_failure": source_hashes(), "test_read": False,
    }, indent=2, allow_nan=False), encoding="utf-8")


def run(output: Path, *, smoke: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    np_rng = np.random.default_rng(67)
    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False))
    network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]])
    network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    events, excluded, identities = read_development(base, network, full_train=True)
    train_events = events["train"]
    if len(train_events) != 346:
        raise AssertionError(f"unexpected explicit TRAIN count: {len(train_events)}")
    source_before = source_hashes()
    if source_before["model"] != SOURCE_MODEL_SHA256:
        raise AssertionError("unexpected independent S checkpoint")
    teacher, anchor_state = construct_teacher(SOURCE, base, network)
    anchor = _construct(SOURCE, base, network, "challenger")
    initial_field_error = _initial_field_check(anchor, teacher)
    source_state = {name: value.detach().clone() for name, value in teacher.state_dict().items() if name in anchor_state}
    allowed = set(configure_trainable(teacher))
    if allowed != set(TEACHER_NAMES):
        raise AssertionError("teacher trainable parameter set mismatch")
    initial_f0 = _assert_zero_field(teacher)
    optimizer = torch.optim.Adam([parameter for name, parameter in teacher.named_parameters() if name in allowed], lr=LR)
    steps = 2 if smoke else STEPS
    traces: list[dict[str, Any]] = []
    failed_step = 0
    failed_provenance: dict[str, Any] | None = None
    try:
        for step in range(1, steps + 1):
            failed_step = step
            horizon = horizon_for_step(step)
            samples, histories = sample_batch(train_events, np_rng, count=BATCH_SIZE)
            provenance = {"step": step, "horizon_s": horizon,
                          "event_ids": [event["id"] for event, _ in samples],
                          "endpoints": [int(endpoint) for _, endpoint in samples]}
            failed_provenance = provenance
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective(teacher, samples, histories, horizon)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            gradients = []
            for name, parameter in teacher.named_parameters():
                if name in allowed:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"nonfinite gradient: {name}")
                    gradients.append(parameter.grad.detach().reshape(-1))
            w1_grad_norm = float(torch.linalg.vector_norm(torch.cat([teacher.W1.weight.grad.reshape(-1), teacher.W1.bias.grad.reshape(-1)])))
            w2_grad_norm = float(torch.linalg.vector_norm(teacher.W2.weight.grad))
            if step == 1 and w1_grad_norm > INITIAL_W1_TOL:
                raise AssertionError(f"W1 must have zero initial gradient, got {w1_grad_norm}")
            if step == 1 and w2_grad_norm <= 0.:
                raise AssertionError("W2 initial gradient is zero")
            if step == 2 and w1_grad_norm <= 0.:
                raise AssertionError("W1 gradient did not become nonzero after one update")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for name, parameter in teacher.named_parameters() if name in allowed):
                raise FloatingPointError("nonfinite parameter")
            checks = _contract(teacher, source_state, allowed)
            if not all(checks.values()):
                raise AssertionError(f"teacher frozen/zero-field contract: {checks}")
            traces.append({**provenance, "loss": float(loss.detach()),
                           "gradient_norm": float(torch.linalg.vector_norm(torch.cat(gradients))),
                           "W1_gradient_norm": w1_grad_norm, "W2_gradient_norm": w2_grad_norm,
                           "terms": {name: float(value.detach()) for name, value in terms.items()},
                           "allowed_tensor_checks": checks})
            if step == 1 or step % 50 == 0:
                print(json.dumps({"step": step, "horizon_s": horizon, "loss": traces[-1]["loss"],
                                  "gradient_norm": traces[-1]["gradient_norm"],
                                  "W1_gradient_norm": w1_grad_norm, "W2_gradient_norm": w2_grad_norm}), flush=True)
        source_after = source_hashes()
        if source_after != source_before:
            raise AssertionError("source artifacts changed during fit")
    except Exception as exc:
        _failure(output, "SMOKE" if smoke else "formal", failed_step, "joint", str(exc), traces,
                 source_before, teacher, failed_provenance)
        raise
    torch.save(teacher.state_dict(), output / "model.pt")
    shutil.copyfile(SOURCE / "normalization.npz", output / "normalization.npz")
    evidence = {
        "arm": "J-teacher", "mode": "SMOKE ONLY" if smoke else "formal", "seed": 67,
        "teacher_init_seed": 61, "steps": steps, "configured_steps": STEPS,
        "lr": LR, "batch_size": BATCH_SIZE, "full_train": True,
        "allowed_parameters": sorted(allowed), "source_hashes_before": source_before,
        "source_hashes_after": source_after, "model_sha256": digest(output / "model.pt"),
        "normalization_sha256": digest(output / "normalization.npz"),
        "excluded": excluded, "identities": identities,
        "train_event_ids": [event["id"] for event in train_events],
        "initial_field_max_abs": initial_field_error, "initial_f0_max_abs": initial_f0,
        "traces": traces, "seconds": time.perf_counter() - started,
        "nonfinite": False, "test_read": False,
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="two TRAIN steps only; not a scientific fit")
    args = parser.parse_args()
    run(args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
