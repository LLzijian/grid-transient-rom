"""Loop0070 J-S support-rediscovery candidate.

This runner is a bounded implementation of the preregistered support search.
It reuses the Loop0067 JointROM, sampler, objective, schedule, and exactly
the same twelve joint parameters.  It never enumerates TEST data.
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

from train_joint_campaign import (
    BATCH_SIZE, JOINT, LR, SOURCE, SOURCE_MODEL_SHA256, STEPS, _assert_zero_field,
    _construct_source,
    configure_trainable, horizon_for_step, objective, sample_batch,
)
from train_joint_latent import NETWORK, BASE, read_development
from kiki.models.private_pm import MechanicalROM

ROOT = Path(__file__).resolve().parents[1]
PRUNE_SEED = 7000
OPEN_STEPS = 200
SMOKE_STEPS = 2
SMOKE_PRUNE_AFTER = 1
XI_ROWS = 76
LATENT_RANK = 3
NONCONSTANT_COUNT = (XI_ROWS - 1) * LATENT_RANK
KEEP_COUNT = 48


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
    }


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int, np.floating, np.integer)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def open_nonconstant_support(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Open all 225 nonconstant Xi entries; newly opened coefficients start zero."""
    with torch.no_grad():
        support_before = model.support.detach().clone()
        xi_before = model.xi.detach().clone()
        if torch.any(support_before[0] != 0):
            raise ValueError("constant Xi row must remain inactive")
        newly_opened = (support_before == 0)
        newly_opened[0] = False
        model.xi[newly_opened] = 0.
        # Historical serialization can retain masked constant coefficients;
        # canonicalizing them is functionally neutral because support[0]==0.
        model.xi[0].zero_()
        model.support[1:].fill_(1.)
        model.support[0].zero_()
    return support_before, xi_before


def _source_buffers_equal(model, source: dict[str, torch.Tensor], *, allow_support: bool) -> bool:
    parameter_names = {name for name, _ in model.named_parameters()}
    for name, value in model.state_dict().items():
        if name in parameter_names or (allow_support and name == "support"):
            continue
        if not torch.equal(value, source[name]):
            return False
    return True


def _phase_contract(model, source: dict[str, torch.Tensor], *, support_expected: torch.Tensor | None = None) -> dict[str, bool]:
    support_ok = torch.equal(model.support[0], torch.zeros_like(model.support[0]))
    if support_expected is not None:
        support_ok = support_ok and torch.equal(model.support, support_expected)
    alpha_ok = torch.equal(model.alpha_support, source["alpha_support"])
    xi_inactive = model.support == 0
    alpha_inactive = model.alpha_support == 0
    inactive_ok = (torch.equal(model.xi[xi_inactive], torch.zeros_like(model.xi[xi_inactive]))
                   and torch.equal(model.alpha[alpha_inactive], source["alpha"][alpha_inactive]))
    return {
        "frozen_buffers": _source_buffers_equal(model, source, allow_support=True),
        "inactive_xi_alpha": inactive_ok,
        "alpha_support_fixed": alpha_ok,
        "support_contract": support_ok,
        "f0_zero": _assert_zero_field(model) <= 1e-10,
    }


def rank_support(scores: np.ndarray, coefficients: np.ndarray, support: np.ndarray,
                 keep: int = KEEP_COUNT) -> tuple[np.ndarray, np.ndarray]:
    """Select positive standardized contributions with row-major tie breaking."""
    scores = np.asarray(scores, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    support = np.asarray(support)
    if scores.shape != (XI_ROWS, LATENT_RANK) or coefficients.shape != scores.shape or support.shape != scores.shape:
        raise ValueError("Xi score/support shape mismatch")
    if np.any(~np.isfinite(scores)) or np.any(~np.isfinite(coefficients)):
        raise FloatingPointError("nonfinite Xi score")
    flat = scores[1:].reshape(-1)
    positive = np.flatnonzero(flat > 0.)
    order = sorted((int(index) for index in positive), key=lambda index: (-float(flat[index]), index))
    selected_flat = np.asarray(order[:keep], dtype=int)
    selected = np.zeros_like(support, dtype=bool)
    if len(selected_flat):
        rows = selected_flat // LATENT_RANK + 1
        cols = selected_flat % LATENT_RANK
        selected[rows, cols] = True
    return selected, selected_flat


def contribution_scores(model, samples: list[tuple[dict[str, Any], int]], histories: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Score Xi rows using current q and true endpoint physical states."""
    with torch.no_grad():
        q = model.encode(histories)
        endpoint = histories[:, -1]
        state = torch.cat((endpoint, q), dim=-1)
        theta = model.features(state)
        rms = torch.sqrt(theta.square().mean(dim=0))
        scores = model.xi.abs() * rms[:, None]
    return scores.detach().cpu().numpy(), theta.detach().cpu().numpy()


def prune_support(model, train_events: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """Use an independent pruning RNG and write the complete 225-entry artifact."""
    prune_rng = np.random.default_rng(PRUNE_SEED)
    samples, histories = sample_batch(train_events, prune_rng, count=512)
    scores, theta = contribution_scores(model, samples, histories)
    before_support = model.support.detach().cpu().numpy().copy()
    before_coefficients = model.xi.detach().cpu().numpy().copy()
    if not np.all(before_support[1:] == 1):
        raise AssertionError("all nonconstant Xi entries must be open before pruning")
    selected, selected_flat = rank_support(scores, before_coefficients, before_support)
    if len(selected_flat) == 0:
        raise RuntimeError("support rediscovery found zero positive entries")
    after_support = np.zeros_like(before_support)
    after_support[selected] = 1.
    after_coefficients = before_coefficients.copy()
    after_coefficients[~selected] = 0.
    with torch.no_grad():
        model.support.copy_(torch.as_tensor(after_support, dtype=model.support.dtype))
        model.xi.copy_(torch.as_tensor(after_coefficients, dtype=model.xi.dtype))
    np.savez_compressed(
        output / "support_rediscovery.npz",
        sample_event_ids=np.asarray([event["id"] for event, _ in samples]),
        sample_endpoints=np.asarray([endpoint for _, endpoint in samples], dtype=np.int64),
        theta_rms=np.sqrt(np.mean(theta * theta, axis=0)),
        candidate_rows=np.repeat(np.arange(1, XI_ROWS), LATENT_RANK),
        candidate_cols=np.tile(np.arange(LATENT_RANK), XI_ROWS - 1),
        scores=scores[1:].reshape(-1),
        coefficients_before=before_coefficients[1:].reshape(-1),
        coefficients_after=after_coefficients[1:].reshape(-1),
        support_before=before_support,
        support_after=after_support,
    )
    return {
        "seed": PRUNE_SEED, "sample_count": len(samples),
        "sample_event_ids": [event["id"] for event, _ in samples],
        "sample_endpoints": [int(endpoint) for _, endpoint in samples],
        "candidate_count": NONCONSTANT_COUNT, "selected_count": int(len(selected_flat)),
        "selected_row_major_indices": selected_flat.tolist(),
        "artifact": "support_rediscovery.npz",
    }


def _failure(output: Path, arm: str, traces: list[dict[str, Any]], reason: str,
             source_before: dict[str, str], model, phase: str) -> None:
    torch.save(model.state_dict(), output / "failure_model.pt")
    (output / "failure_evidence.json").write_text(json.dumps({
        "arm": "J-S", "mode": arm, "phase": phase, "reason": reason,
        "traces": traces, "source_hashes_before": source_before,
        "source_hashes_after_failure": source_hashes(), "test_read": False,
    }, indent=2, allow_nan=False), encoding="utf-8")


def run(output: Path, *, smoke: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(67)
    train_rng = np.random.default_rng(67)
    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False))
    network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]])
    network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    events, excluded, identities = read_development(base, network, full_train=True)
    train_events = events["train"]
    if len(train_events) != 346:
        raise AssertionError(f"unexpected explicit TRAIN count: {len(train_events)}")
    model = _construct_source(base, network)
    source_before = source_hashes()
    if source_before["model"] != SOURCE_MODEL_SHA256:
        raise AssertionError("unexpected independent S checkpoint")
    source_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    open_nonconstant_support(model)
    open_support_expected = model.support.detach().clone()
    allowed = set(configure_trainable(model, "J"))
    if allowed != set(JOINT):
        raise AssertionError("J-S trainable parameter set mismatch")
    zero_field = _assert_zero_field(model)
    optimizer = torch.optim.Adam([parameter for name, parameter in model.named_parameters() if name in allowed], lr=LR)
    steps = SMOKE_STEPS if smoke else STEPS
    prune_after = SMOKE_PRUNE_AFTER if smoke else OPEN_STEPS
    traces: list[dict[str, Any]] = []
    prune_report = None
    support_after = None
    phase = "open"
    current_provenance: dict[str, Any] | None = None
    try:
        for step in range(1, steps + 1):
            horizon = horizon_for_step(step)
            samples, histories = sample_batch(train_events, train_rng, count=BATCH_SIZE)
            provenance = {"step": step, "horizon_s": horizon,
                          "event_ids": [event["id"] for event, _ in samples],
                          "endpoints": [int(endpoint) for _, endpoint in samples]}
            current_provenance = {**provenance, "phase": phase}
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective(model, samples, histories, horizon)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            gradients = []
            for name, parameter in model.named_parameters():
                if name in allowed:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f"nonfinite gradient: {name}")
                    gradients.append(parameter.grad.detach().reshape(-1))
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for name, parameter in model.named_parameters() if name in allowed):
                raise FloatingPointError("nonfinite parameter")
            checks = _phase_contract(
                model, source_state,
                support_expected=open_support_expected if support_after is None else support_after,
            )
            if not all(checks.values()):
                raise AssertionError(f"phase/support contract: {checks}")
            traces.append({**provenance, "phase": phase, "loss": float(loss.detach()),
                           "gradient_norm": float(torch.linalg.vector_norm(torch.cat(gradients))),
                           "terms": {name: float(value.detach()) for name, value in terms.items()},
                           "allowed_tensor_checks": checks})
            if step == 1 or step % 50 == 0:
                print(json.dumps({"step": step, "phase": phase, "horizon_s": horizon,
                                  "loss": float(loss.detach()),
                                  "gradient_norm": traces[-1]["gradient_norm"]}), flush=True)
            if step == prune_after:
                prune_report = prune_support(model, train_events, output)
                support_after = model.support.detach().clone()
                post_prune_checks = _phase_contract(model, source_state, support_expected=support_after)
                if not all(post_prune_checks.values()):
                    raise AssertionError(f"post-pruning support contract: {post_prune_checks}")
                if int(model.support.sum().item()) > KEEP_COUNT:
                    raise AssertionError("post-pruning support exceeds 48")
                print(json.dumps({"event": "support_prune", "selected_count": prune_report["selected_count"],
                                  "phase": "open_to_refit"}), flush=True)
                optimizer = torch.optim.Adam([parameter for name, parameter in model.named_parameters() if name in allowed], lr=LR)
                phase = "refit"
        source_after = source_hashes()
        if source_after != source_before:
            raise AssertionError("source artifacts changed during fit")
        if prune_report is None:
            raise AssertionError("support pruning did not occur")
    except Exception as exc:
        failed = [] if current_provenance is None else [{**current_provenance, "failed": True, "reason": str(exc)}]
        _failure(output, "SMOKE" if smoke else "formal", traces + failed, str(exc), source_before, model, phase)
        raise
    shutil.copyfile(SOURCE / "normalization.npz", output / "normalization.npz")
    torch.save(model.state_dict(), output / "model.pt")
    evidence = {
        "arm": "J-S", "mode": "SMOKE ONLY" if smoke else "formal", "seed": 67,
        "steps": steps, "configured_steps": STEPS, "open_steps": prune_after,
        "refit_start_step": prune_after + 1, "lr": LR, "batch_size": BATCH_SIZE,
        "full_train": True, "allowed_parameters": sorted(allowed),
        "source_hashes_before": source_before, "source_hashes_after": source_after,
        "model_sha256": digest(output / "model.pt"),
        "normalization_sha256": digest(output / "normalization.npz"),
        "excluded": excluded, "identities": identities,
        "train_event_ids": [event["id"] for event in train_events],
        "zero_field_max_abs": zero_field, "support_phase": prune_report,
        "traces": traces, "seconds": time.perf_counter() - started,
        "nonfinite": False, "test_read": False,
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="one open TRAIN step + pruning + one refit step")
    args = parser.parse_args()
    run(args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
