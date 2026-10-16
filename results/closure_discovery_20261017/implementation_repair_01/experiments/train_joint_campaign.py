"""Loop0067 matched Xi-only (C) and joint (J) development campaigns.

This is intentionally a thin campaign runner around the frozen Loop0060
``JointROM``.  It does not alter either historical trainer or evaluator.
Only explicit TRAIN/validation IDs are read by the inherited development
reader; TEST is never enumerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from evaluate_long_latent import BASE, NETWORK, _construct
from train_joint_latent import DT, JointROM, read_development, tensor
from kiki.models.private_pm import MechanicalROM

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "private_data/task2_long_0060_5s"
SOURCE_MODEL_SHA256 = "d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c"
SEED = 67
BATCH_SIZE = 32
STEPS = 800
LR = 1e-4
SCALES = (0.47675314588683515, 1.0033667249602924)

XI_ONLY = ("xi",)
JOINT = (
    "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0",
    "head.weight", "head.bias", "c", "amplitude", "alpha", "xi",
    "decoder.weight", "decoder.bias",
)
ALL_PARAMETER_NAMES = set(JOINT)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def horizon_for_step(step: int) -> float:
    """Return the preregistered 1-indexed horizon schedule."""
    if step < 1 or step > STEPS:
        raise ValueError(f"step outside fixed campaign: {step}")
    if step <= 100:
        return 2.0
    return (1.0, 2.0, 5.0)[(step - 101) % 3]


def _label_key(value: Any) -> str:
    return str(value).strip().lower()


def _stratified_pools(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for event in events:
        label = _label_key(event["label"])
        group = str(event["group"])
        pools.setdefault(label, {}).setdefault(group, []).append(event)
    required = {"stable", "unstable"}
    if set(pools) != required:
        raise ValueError(f"expected stable/unstable TRAIN strata, got {sorted(pools)}")
    if any(not groups for groups in pools.values()):
        raise ValueError("empty label stratum")
    return pools


def sample_batch(events: list[dict[str, Any]], rng: np.random.Generator,
                 count: int = BATCH_SIZE) -> tuple[list[tuple[dict[str, Any], int]], torch.Tensor]:
    """Sample alternating labels, then group/event, then a full-TRAIN endpoint.

    Endpoint ``len(x)-1`` is excluded because one future state is required;
    consequently the exact inclusive range is ``10..len(x)-2``.  Sampling is
    with replacement and the returned tuples retain the actual event object
    and endpoint for provenance.
    """
    if count <= 0 or count % 2:
        raise ValueError("count must be a positive even number")
    pools = _stratified_pools(events)
    samples: list[tuple[dict[str, Any], int]] = []
    labels = ("stable", "unstable")
    for index in range(count):
        label = labels[index % 2]
        groups = sorted(pools[label])
        group = groups[int(rng.integers(len(groups)))]
        choices = pools[label][group]
        event = choices[int(rng.integers(len(choices)))]
        if len(event["x"]) <= 11:
            raise ValueError(f"event {event.get('id')} has no valid endpoint")
        endpoint = int(rng.integers(10, len(event["x"]) - 1))
        samples.append((event, endpoint))
    histories = tensor(np.stack([event["x"][endpoint - 10:endpoint + 1]
                                  for event, endpoint in samples]))
    return samples, histories


def configure_trainable(model: JointROM, arm: str) -> tuple[str, ...]:
    """Configure exactly C's Xi or J's twelve declared tensors.

    Buffers are never made trainable.  The constant feature of Xi must remain
    inactive: a nonzero constant support would violate the required ``f(0)=0``
    contract and is rejected before fitting.
    """
    arm = str(arm).upper()
    if arm not in {"C", "J"}:
        raise ValueError(f"unknown arm: {arm}")
    names = tuple(XI_ONLY if arm == "C" else JOINT)
    observed = {name for name, _ in model.named_parameters()}
    if observed != ALL_PARAMETER_NAMES:
        raise ValueError(f"unexpected JointROM parameter names: {sorted(observed)}")
    for mask_name in ("support", "alpha_support"):
        mask = getattr(model, mask_name)
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{mask_name} must be a binary frozen mask")
    if torch.any(model.support[0] != 0):
        raise ValueError("nonzero constant active support in Xi")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    return names


def _masked_inactive_equal(model: JointROM, source: dict[str, torch.Tensor]) -> bool:
    for name, mask_name in (("xi", "support"), ("alpha", "alpha_support")):
        mask = getattr(model, mask_name)
        inactive = mask == 0
        if not torch.equal(model.state_dict()[name][inactive], source[name][inactive]):
            return False
    return True


def _frozen_equal(model: JointROM, source: dict[str, torch.Tensor], allowed: set[str]) -> bool:
    parameter_names = {name for name, _ in model.named_parameters()}
    for name, value in model.state_dict().items():
        if name not in parameter_names or name not in allowed:
            if not torch.equal(value, source[name]):
                return False
    return True


def _point_error(pred: torch.Tensor, truth: torch.Tensor, model: JointROM) -> torch.Tensor:
    """Common-motion-aware per-point error, retaining physical COI terms."""
    err = pred[..., :20] - truth[..., :20]
    delta = model.coi(err[..., :10]) / SCALES[0]
    omega = model.coi(err[..., 10:20]) / SCALES[1]
    common = (err[..., 10:20] * model.mass).sum(dim=-1) / model.mass.sum() / SCALES[1]
    return 0.5 * (delta.square().mean(dim=-1) + omega.square().mean(dim=-1)) + common.square()


def objective(model: JointROM, samples: list[tuple[dict[str, Any], int]],
              histories: torch.Tensor, horizon_s: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the complete preregistered C/J objective without target padding."""
    if horizon_s <= 0:
        raise ValueError("horizon must be positive")
    device = histories.device
    count = len(samples)
    local_states = torch.cat((histories[:, -1], model.encode(histories)), dim=-1)

    local_available = np.asarray([min(round(0.2 / DT), len(event["x"]) - 1 - endpoint)
                                  for event, endpoint in samples], dtype=int)
    if np.any(local_available <= 0):
        raise ValueError("sampled endpoint has no local future")
    local_sum = torch.zeros(count, dtype=torch.float64, device=device)
    local_count = torch.zeros(count, dtype=torch.float64, device=device)
    current = local_states
    for offset in range(1, int(local_available.max()) + 1):
        active_np = np.flatnonzero(local_available >= offset)
        active = torch.as_tensor(active_np, dtype=torch.long, device=device)
        current = current.index_copy(0, active, model.rk4(current.index_select(0, active), DT))
        truth = tensor(np.stack([samples[row][0]["x"][samples[row][1] + offset]
                                 for row in active_np])).to(device)
        value = _point_error(current.index_select(0, active), truth, model)
        local_sum = local_sum.index_add(0, active, value)
        local_count = local_count.index_add(0, active, torch.ones_like(value))
    local = (local_sum / local_count).mean()

    adjacent = tensor(np.stack([event["x"][endpoint - 9:endpoint + 2]
                                for event, endpoint in samples])).to(device)
    latent = ((model.rk4(local_states, DT)[:, 30:] - model.encode(adjacent)) / 0.1).square().mean()

    initial = tensor(np.stack([event["x"][:11] for event, _ in samples])).to(device)
    current = torch.cat((initial[:, -1], model.encode(initial)), dim=-1)
    max_steps = round(float(horizon_s) / DT)
    long_available = np.asarray([min(max_steps, len(event["x"]) - 11)
                                 for event, _ in samples], dtype=int)
    if np.any(long_available <= 0):
        raise ValueError("event has no future after initial history")
    long_sum = torch.zeros(count, dtype=torch.float64, device=device)
    long_count = torch.zeros(count, dtype=torch.float64, device=device)
    for offset in range(1, int(long_available.max()) + 1):
        active_np = np.flatnonzero(long_available >= offset)
        active = torch.as_tensor(active_np, dtype=torch.long, device=device)
        current = current.index_copy(0, active, model.rk4(current.index_select(0, active), DT))
        truth = tensor(np.stack([samples[row][0]["x"][10 + offset]
                                 for row in active_np])).to(device)
        value = _point_error(current.index_select(0, active), truth, model)
        long_sum = long_sum.index_add(0, active, value)
        long_count = long_count.index_add(0, active, torch.ones_like(value))
    event_error = long_sum / long_count
    long_mean = event_error.mean()
    long_tail = event_error.topk(max(1, int(np.ceil(0.1 * count)))).values.mean()
    risk = 0.8 * long_mean + 0.2 * long_tail

    q = local_states[:, 30:]
    force_pred = model.memory(q + model.algebraic_coordinate(local_states))
    force_truth = tensor(np.stack([event["force"][endpoint] for event, endpoint in samples])).to(device)
    force_loss = ((force_pred - force_truth) / model.force_scale).square().mean()
    z_pred, load_pred = model.electrical_output(local_states[:, :30], q)
    z_truth = tensor(np.stack([event["z"][endpoint] for event, endpoint in samples])).to(device) / model.z_scale
    load_truth = tensor(np.asarray([event["load"][endpoint] for event, endpoint in samples])[:, None]).to(device) / model.load_scale
    pe_loss = (z_pred[:, :10] - z_truth[:, :10]).square().mean()
    voltage_loss = (z_pred[:, 10:] - z_truth[:, 10:]).square().mean()
    load_loss = (load_pred - load_truth).square().mean()
    electrical = (pe_loss + voltage_loss + load_loss) / 3

    zero = torch.zeros((1, 11, 30), dtype=histories.dtype, device=device)
    equilibrium = model.encode(zero).square().mean()
    xi_l1 = (model.xi * model.support).abs().mean()
    total = (10 * risk + local + 0.1 * latent + 0.1 * (force_loss + electrical)
             + 1e-4 * xi_l1 + equilibrium)
    return total, {
        "risk": risk, "long_mean": long_mean, "long_tail": long_tail,
        "local": local, "latent_consistency": latent, "force": force_loss,
        "electrical": electrical, "pe": pe_loss, "voltage": voltage_loss,
        "load": load_loss, "xi_l1": xi_l1, "equilibrium": equilibrium,
    }


# Keep the historical vocabulary available to small diagnostics while making
# ``objective`` the public preregistered helper for this campaign.
campaign_loss = objective


def _assert_zero_field(model: JointROM, tolerance: float = 1e-10) -> float:
    zero = torch.zeros((1, 30 + model.rank), dtype=torch.float64)
    value = float(model.field(zero).abs().max().detach())
    if not np.isfinite(value) or value > tolerance:
        raise AssertionError(f"f(0)={value} exceeds {tolerance}")
    return value


def _failure(output: Path, args: argparse.Namespace, model: JointROM,
             source: dict[str, torch.Tensor], traces: list[dict[str, Any]],
             reason: str, step: int, source_hashes: dict[str, str]) -> None:
    torch.save(model.state_dict(), output / "failure_model.pt")
    (output / "failure_evidence.json").write_text(json.dumps({
        "arm": args.arm, "step": step, "reason": reason, "traces": traces,
        "source_hashes_before": source_hashes,
        "source_hashes_after_failure": {
            "model": digest(SOURCE / "model.pt"),
            "normalization": digest(SOURCE / "normalization.npz"),
            "base": digest(BASE), "network": digest(NETWORK),
            "code": digest(Path(__file__)),
        },
        "test_read": False,
    }, indent=2), encoding="utf-8")


def _construct_source(base: MechanicalROM, network: dict[str, np.ndarray]) -> JointROM:
    model = _construct(SOURCE, base, network, "challenger")
    if digest(SOURCE / "model.pt") != SOURCE_MODEL_SHA256:
        raise AssertionError("unexpected Loop0060 source checkpoint")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("C", "J"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False))
    network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]])
    network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    events, excluded, identities = read_development(base, network, full_train=True)
    train_events = events["train"]
    model = _construct_source(base, network)
    source_hashes_before = {
        "model": digest(SOURCE / "model.pt"),
        "normalization": digest(SOURCE / "normalization.npz"),
        "base": digest(BASE), "network": digest(NETWORK),
        "code": digest(Path(__file__)),
    }
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    allowed = set(configure_trainable(model, args.arm))
    try:
        zero_field = _assert_zero_field(model)
    except Exception as exc:
        _failure(args.output, args, model, source,
                 [{"step": 0, "horizon_s": None, "event_ids": [], "endpoints": [],
                   "source_hashes": source_hashes_before, "failed": True,
                   "reason": str(exc)}], str(exc), 0, source_hashes_before)
        raise
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if name in allowed], lr=LR)
    traces: list[dict[str, Any]] = []
    steps = 1 if args.smoke else STEPS
    for step in range(1, steps + 1):
        horizon = horizon_for_step(step)
        samples, histories = sample_batch(train_events, rng)
        failed_provenance = {
            "step": step, "horizon_s": horizon,
            "event_ids": [event["id"] for event, _ in samples],
            "endpoints": [endpoint for _, endpoint in samples],
            "source_hashes": source_hashes_before,
        }
        try:
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
            checks = {
                "frozen_buffers": _frozen_equal(model, source, allowed),
                "inactive_xi_alpha": _masked_inactive_equal(model, source),
                "f0_zero": _assert_zero_field(model) <= 1e-10,
                "allowed_names": sorted(allowed),
            }
            if not checks["frozen_buffers"] or not checks["inactive_xi_alpha"] or not checks["f0_zero"]:
                raise AssertionError(f"frozen/zero-field contract: {checks}")
            traces.append({
                **failed_provenance,
                "loss": float(loss.detach()),
                "gradient_norm": float(torch.linalg.vector_norm(torch.cat(gradients))),
                "terms": {name: float(value.detach()) for name, value in terms.items()},
                "allowed_tensor_checks": checks,
            })
        except Exception as exc:
            failed_provenance.update({"failed": True, "reason": str(exc)})
            _failure(args.output, args, model, source, traces + [failed_provenance],
                     str(exc), step, source_hashes_before)
            raise
        if step == 1 or step % 50 == 0:
            print(json.dumps({
                "step": step, "horizon_s": horizon,
                "loss": traces[-1]["loss"],
                "gradient_norm": traces[-1]["gradient_norm"],
            }), flush=True)
    source_hashes_after = {
        "model": digest(SOURCE / "model.pt"),
        "normalization": digest(SOURCE / "normalization.npz"),
        "base": digest(BASE), "network": digest(NETWORK),
        "code": digest(Path(__file__)),
    }
    if source_hashes_after != source_hashes_before:
        raise AssertionError("source artifacts changed during fit")
    torch.save(model.state_dict(), args.output / "model.pt")
    shutil.copyfile(SOURCE / "normalization.npz", args.output / "normalization.npz")
    evidence = {
        "arm": args.arm, "seed": SEED, "steps": steps, "configured_steps": STEPS,
        "lr": LR, "batch_size": BATCH_SIZE, "full_train": True,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "model_sha256": digest(args.output / "model.pt"),
        "normalization_sha256": digest(args.output / "normalization.npz"),
        "allowed_parameters": sorted(allowed),
        "excluded": excluded, "identities": identities,
        "train_event_ids": [event["id"] for event in train_events],
        "zero_field_max_abs": zero_field, "traces": traces,
        "seconds": time.perf_counter() - started, "nonfinite": False,
        "test_read": False,
    }
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    if args.smoke:
        print("SMOKE ONLY: one preregistered C/J step; no scientific evaluation", flush=True)


if __name__ == "__main__":
    main()
