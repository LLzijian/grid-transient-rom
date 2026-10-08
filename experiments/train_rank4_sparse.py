"""Loop0074 rank-4 sparse-memory candidate.

This runner constructs a rank-4 checkpoint from the independent rank-3 S
checkpoint, then reuses the fixed Loop0067 objective and sampler with only the
three preregistered rank-dependent denominator corrections.
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
    configure_trainable, horizon_for_step, objective, sample_batch,
)
from train_joint_latent import BASE, NETWORK, JointROM, read_development
from kiki.models.private_pm import MechanicalROM

ROOT = Path(__file__).resolve().parents[1]
PRUNE_SEED = 7000
OPEN_STEPS = 200
SMOKE_STEPS = 2
SMOKE_PRUNE_AFTER = 1
RANK_OLD = 3
RANK = 4
XI_ROWS_OLD = 76
XI_ROWS = 77
NONCONSTANT_COUNT = (XI_ROWS - 1) * RANK
KEEP_COUNT = 48
XI_Q4_ROW = 4
XI_Q4_COL = 3
Q4_RELAXATION_SECONDS = 1.0
EXPECTED_INITIAL_DATA = 7.157914720643666
EXPECTED_INITIAL_TOTAL = EXPECTED_INITIAL_DATA + 1e-4 / (76 * 3)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        "model": digest(SOURCE / "model.pt"),
        "normalization": digest(SOURCE / "normalization.npz"),
        "base": digest(BASE),
        "network": digest(NETWORK),
        "physical_equilibrium": digest(ROOT / "private_data/task2_coupled_equilibrium_0065/coupled_equilibrium.npz"),
        "code": digest(Path(__file__)),
        "campaign_code": digest(ROOT / "experiments/train_joint_campaign.py"),
        "latent_code": digest(ROOT / "experiments/train_joint_latent.py"),
        "evaluate_code": digest(ROOT / "experiments/evaluate_long_latent.py"),
        "l1_evaluator_code": digest(ROOT / "experiments/evaluate_l1_latent.py"),
    }


def _jsonable(value: Any, *, allow_nonfinite: bool = False) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v, allow_nonfinite=allow_nonfinite) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, allow_nonfinite=allow_nonfinite) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), allow_nonfinite=allow_nonfinite)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            if allow_nonfinite:
                return None
            raise FloatingPointError("nonfinite evidence value")
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _has_nonfinite(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_nonfinite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nonfinite(v) for v in value)
    if isinstance(value, np.ndarray):
        return not bool(np.isfinite(value).all())
    if isinstance(value, (np.floating, float)):
        return not bool(np.isfinite(value))
    return False


def _state_copy(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def expand_rank3_state(old_state: dict[str, torch.Tensor], ports: torch.Tensor,
                       mass: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Map an S rank-3 state into the exact rank-4 initialization contract."""
    state = {name: value.detach().clone() for name, value in old_state.items()}
    dtype = old_state["c"].dtype
    device = old_state["c"].device
    new = {}
    for name, value in old_state.items():
        if name not in {"c", "amplitude", "xi", "alpha", "q_center", "q_transform",
                        "support", "alpha_support", "memory_enabled", "head.weight",
                        "head.bias", "decoder.weight"}:
            new[name] = value.detach().clone()
    new["c"] = torch.zeros((old_state["c"].shape[0], RANK), dtype=dtype, device=device)
    new["c"][:, :RANK_OLD] = old_state["c"]
    target = mass / torch.linalg.vector_norm(mass)
    c4 = torch.linalg.lstsq(ports, target[:, None]).solution[:, 0]
    c4_svd = torch.linalg.pinv(ports) @ target
    minnorm_error = float(torch.linalg.vector_norm(c4 - c4_svd))
    new["c"][:, 3] = c4
    new["amplitude"] = torch.cat((old_state["amplitude"], old_state["amplitude"].abs().median().reshape(1)))
    new["q_center"] = torch.cat((old_state["q_center"], torch.zeros(1, dtype=dtype, device=device)))
    new["q_transform"] = torch.zeros((RANK, RANK), dtype=dtype, device=device)
    new["q_transform"][:3, :3] = old_state["q_transform"]
    new["q_transform"][3, 3] = 1.
    new["memory_enabled"] = torch.cat((old_state["memory_enabled"], torch.ones(1, dtype=dtype, device=device)))
    new["head.weight"] = torch.zeros((RANK, old_state["head.weight"].shape[1]), dtype=dtype, device=device)
    new["head.weight"][:3] = old_state["head.weight"]
    new["head.bias"] = torch.zeros(RANK, dtype=dtype, device=device)
    new["head.bias"][:3] = old_state["head.bias"]
    new["decoder.weight"] = torch.zeros((old_state["decoder.weight"].shape[0], 10 + RANK), dtype=dtype, device=device)
    new["decoder.weight"][:, :13] = old_state["decoder.weight"]
    new["xi"] = torch.zeros((XI_ROWS, RANK), dtype=dtype, device=device)
    new["xi"][0, :3] = old_state["xi"][0]
    new["xi"][1:4, :3] = old_state["xi"][1:4]
    new["xi"][5:, :3] = old_state["xi"][4:]
    new["xi"][XI_Q4_ROW, XI_Q4_COL] = -1. / Q4_RELAXATION_SECONDS
    new["support"] = torch.zeros((XI_ROWS, RANK), dtype=old_state["support"].dtype, device=device)
    new["support"][0, :3] = old_state["support"][0]
    new["support"][1:4, :3] = old_state["support"][1:4]
    new["support"][5:, :3] = old_state["support"][4:]
    new["support"][XI_Q4_ROW, XI_Q4_COL] = 1.
    new["alpha"] = torch.zeros((old_state["alpha"].shape[0], RANK), dtype=dtype, device=device)
    new["alpha"][:, :3] = old_state["alpha"]
    new["alpha_support"] = torch.zeros((old_state["alpha_support"].shape[0], RANK), dtype=old_state["alpha_support"].dtype, device=device)
    new["alpha_support"][:, :3] = old_state["alpha_support"]
    new["xi"][new["support"] == 0] = 0.
    old_cq = (ports @ (old_state["c"] / torch.linalg.vector_norm(old_state["c"], dim=0, keepdim=True))) * old_state["amplitude"][None, :]
    projection = old_cq @ torch.linalg.pinv(old_cq)
    mass_distance = float(torch.linalg.vector_norm(mass - projection @ mass) / torch.linalg.vector_norm(mass))
    residual = float(torch.linalg.vector_norm(ports @ c4 - target) / torch.linalg.vector_norm(target))
    amplitude = float(new["amplitude"][3])
    diagnostics = (residual, minnorm_error, mass_distance, amplitude)
    if not all(np.isfinite(value) for value in diagnostics):
        raise FloatingPointError(f"nonfinite rank-4 initialization diagnostics: {diagnostics}")
    if not torch.isfinite(new["c"]).all() or not torch.isfinite(new["amplitude"]).all():
        raise FloatingPointError("nonfinite rank-4 initialization state")
    if amplitude <= 1e-8:
        raise ValueError("rank-4 amplitude initialization is unusable")
    if residual > 1e-10:
        raise ValueError(f"rank-4 port residual exceeds contract: {residual}")
    if mass_distance <= 1e-6:
        raise ValueError(f"mass direction is too close to old Cq span: {mass_distance}")
    if minnorm_error > 1e-10:
        raise AssertionError(f"rank-4 least-squares solution is not min norm: {minnorm_error}")
    return new, {"port_residual": residual, "minnorm_svd_error": minnorm_error,
                 "mass_span_relative_distance": mass_distance,
                 "amplitude4": amplitude, "c4": c4.detach().cpu().numpy()}


def construct_rank4(base: MechanicalROM, network: dict[str, np.ndarray], source_dir: Path = SOURCE):
    old = _construct(source_dir, base, network, "challenger")
    old_state = _state_copy(old.state_dict())
    model = JointROM(base, old.ports.detach().cpu().numpy(), old.input_scale.detach().cpu().numpy(),
                     old.feature_scale.detach().cpu().numpy(), float(old.force_scale), float(old.load_scale),
                     electrical=(old.z_scale.detach().cpu().numpy(), old.voltage_star.detach().cpu().numpy(),
                                 old.load_positions.detach().cpu().numpy(), old.load_g.detach().cpu().numpy(),
                                 old.load_star.detach().cpu().numpy()), latent_dim=RANK)
    new_state, init = expand_rank3_state(old_state, model.ports, model.mass)
    missing = set(model.state_dict()) - set(new_state)
    if missing:
        raise AssertionError(f"rank-4 state mapping missing: {sorted(missing)}")
    model.load_state_dict(new_state, strict=True)
    model.eval()
    return old, model, init


def open_rank4_support(model) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        before = model.support.detach().clone()
        xi_before = model.xi.detach().clone()
        if torch.any(before[0] != 0):
            raise ValueError("constant Xi support must be inactive")
        newly_opened = before == 0
        newly_opened[0] = False
        model.xi[newly_opened] = 0.
        model.xi[0].zero_()
        model.support[1:].fill_(1.)
        model.support[0].zero_()
        model.xi[XI_Q4_ROW, XI_Q4_COL] = -1. / Q4_RELAXATION_SECONDS
    return before, xi_before


def rank4_objective(model, samples, histories, horizon_s):
    """Exact 0067 objective with fixed rank-3 component denominators."""
    loss, terms = objective(model, samples, histories, horizon_s)
    latent = terms["latent_consistency"] * (RANK / RANK_OLD)
    equilibrium = terms["equilibrium"] * (RANK / RANK_OLD)
    xi_l1 = (model.xi * model.support).abs().sum() / float(76 * 3)
    corrected = loss + .1 * (latent - terms["latent_consistency"])
    corrected = corrected + (equilibrium - terms["equilibrium"])
    corrected = corrected + 1e-4 * (xi_l1 - terms["xi_l1"])
    terms = dict(terms, latent_consistency=latent, equilibrium=equilibrium, xi_l1=xi_l1)
    return corrected, terms


def rank_support(scores: np.ndarray, coefficients: np.ndarray, support: np.ndarray,
                 keep: int = KEEP_COUNT) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float); coefficients = np.asarray(coefficients, dtype=float)
    support = np.asarray(support)
    if scores.shape != (XI_ROWS, RANK) or coefficients.shape != scores.shape or support.shape != scores.shape:
        raise ValueError("rank-4 Xi shape mismatch")
    if not np.isfinite(scores).all() or not np.isfinite(coefficients).all():
        raise FloatingPointError("nonfinite rank-4 Xi score")
    flat = scores[1:].reshape(-1)
    positive = np.flatnonzero(flat > 0.)
    order = sorted((int(index) for index in positive), key=lambda index: (-float(flat[index]), index))
    selected_flat = np.asarray(order[:keep], dtype=int)
    selected = np.zeros_like(support, dtype=bool)
    if len(selected_flat):
        rows = selected_flat // RANK + 1; cols = selected_flat % RANK
        selected[rows, cols] = True
    return selected, selected_flat


def contribution_scores(model, samples, histories):
    with torch.no_grad():
        q = model.encode(histories)
        state = torch.cat((histories[:, -1], q), dim=-1)
        theta = model.features(state)
        rms = torch.sqrt(theta.square().mean(dim=0))
        scores = model.xi.abs() * rms[:, None]
    return scores.cpu().numpy(), theta.cpu().numpy()


def prune_support(model, train_events, output: Path) -> dict[str, Any]:
    rng = np.random.default_rng(PRUNE_SEED)
    samples, histories = sample_batch(train_events, rng, count=512)
    scores, theta = contribution_scores(model, samples, histories)
    before_support = model.support.detach().cpu().numpy().copy()
    before_coefficients = model.xi.detach().cpu().numpy().copy()
    if not np.all(before_support[1:] == 1):
        raise AssertionError("rank-4 open phase did not expose all 304 nonconstant Xi entries")
    selected, selected_flat = rank_support(scores, before_coefficients, before_support)
    if len(selected_flat) == 0:
        raise RuntimeError("rank-4 support rediscovery selected no positive entries")
    after_support = np.zeros_like(before_support); after_support[selected] = 1.
    after_coefficients = before_coefficients.copy(); after_coefficients[~selected] = 0.
    with torch.no_grad():
        model.support.copy_(torch.as_tensor(after_support, dtype=model.support.dtype))
        model.xi.copy_(torch.as_tensor(after_coefficients, dtype=model.xi.dtype))
    np.savez_compressed(
        output / "support_rediscovery.npz",
        sample_event_ids=np.asarray([event["id"] for event, _ in samples]),
        sample_endpoints=np.asarray([endpoint for _, endpoint in samples], dtype=np.int64),
        theta_rms=np.sqrt(np.mean(theta * theta, axis=0)),
        candidate_rows=np.repeat(np.arange(1, XI_ROWS), RANK),
        candidate_cols=np.tile(np.arange(RANK), XI_ROWS - 1),
        scores=scores[1:].reshape(-1), coefficients_before=before_coefficients[1:].reshape(-1),
        coefficients_after=after_coefficients[1:].reshape(-1), support_before=before_support,
        support_after=after_support,
    )
    return {"seed": PRUNE_SEED, "sample_count": len(samples),
            "sample_event_ids": [event["id"] for event, _ in samples],
            "sample_endpoints": [int(endpoint) for _, endpoint in samples],
            "candidate_count": NONCONSTANT_COUNT, "selected_count": int(len(selected_flat)),
            "selected_row_major_indices": selected_flat.tolist(),
            "q4_memory_entry_selected": bool(selected[XI_Q4_ROW, XI_Q4_COL]),
            "fourth_xi_column_selected": bool(np.any(selected[1:, 3])),
            "added_state_dynamics_removed": not bool(np.any(selected[1:, 3])),
            "artifact": "support_rediscovery.npz"}


def _phase_contract(model, source: dict[str, torch.Tensor], expected_support: torch.Tensor) -> dict[str, bool]:
    parameter_names = {name for name, _ in model.named_parameters()}
    frozen = all(torch.equal(value, source[name]) for name, value in model.state_dict().items()
                 if name not in parameter_names and name != "support")
    inactive = model.support == 0
    alpha_inactive = model.alpha_support == 0
    return {"frozen_buffers": frozen,
            "inactive_xi_zero": torch.equal(model.xi[inactive], torch.zeros_like(model.xi[inactive])),
            "inactive_alpha_source": torch.equal(model.alpha[alpha_inactive], source["alpha"][alpha_inactive]),
            "alpha_support_fixed": torch.equal(model.alpha_support, source["alpha_support"]),
            "support_expected": torch.equal(model.support, expected_support),
            "f0_zero": _assert_zero_field(model) <= 1e-10}


def _jacobian(model) -> torch.Tensor:
    zero = torch.zeros(34, dtype=torch.float64, requires_grad=True)
    return torch.autograd.functional.jacobian(lambda x: model.field(x[None])[0], zero)


def _initial_equivalence(old, new) -> dict[str, float]:
    rng = torch.Generator().manual_seed(74)
    physical = torch.randn((8, 30), generator=rng, dtype=torch.float64) * .05
    histories = physical[:, None, :].repeat(1, 11, 1)
    with torch.no_grad():
        old_q = old.encode(histories)
        new_q = new.encode(histories)
        old_state = torch.cat((physical, old_q), dim=-1)
        new_state = torch.cat((physical, new_q), dim=-1)
        old_field = old.field(old_state)
        new_field = new.field(new_state)
        old_z, old_load = old.electrical_output(physical, old_q)
        new_z, new_load = new.electrical_output(physical, new_q)
        old_roll = old_state
        new_roll = new_state
        rollout_error = 0.0
        q4_error = 0.0
        for _ in range(20):
            old_roll = old.rk4(old_roll, .01)
            new_roll = new.rk4(new_roll, .01)
            rollout_error = max(rollout_error, float((old_roll - new_roll[:, :33]).abs().max()))
            q4_error = max(q4_error, float(new_roll[:, 33].abs().max()))
    return {"field_max_abs": float((old_field - new_field[:, :33]).abs().max()),
            "q_old_max_abs": float((old_q - new_q[:, :3]).abs().max()),
            "electrical_max_abs": float((old_z - new_z).abs().max()),
            "load_max_abs": float((old_load - new_load).abs().max()),
            "rollout20_max_abs": rollout_error, "rollout20_q4_max_abs": q4_error}


def _rank4_preflight(old, new) -> dict[str, Any]:
    equivalence = _initial_equivalence(old, new)
    if _has_nonfinite(equivalence):
        raise FloatingPointError("nonfinite rank-4 initial equivalence diagnostic")
    if max(equivalence.values()) > 1e-10:
        raise AssertionError(f"rank-4 initial equivalence failed: {equivalence}")
    old_j = torch.autograd.functional.jacobian(lambda x: old.field(x[None])[0], torch.zeros(33, dtype=torch.float64))
    new_j = _jacobian(new)
    eps = 1e-6
    plus = torch.eye(34, dtype=torch.float64) * eps; minus = -plus
    with torch.no_grad():
        fd = (new.field(plus) - new.field(minus)) / (2 * eps)
    fd_error = float((new_j - fd.T).abs().max())
    old_j_np = old_j.detach().numpy(); new_j_np = new_j.detach().numpy()
    if not np.isfinite(old_j_np).all() or not np.isfinite(new_j_np).all() or not np.isfinite(fd.detach().numpy()).all():
        raise FloatingPointError("nonfinite rank-4 Jacobian preflight")
    top_left_error = float(np.max(np.abs(new_j_np[:33, :33] - old_j_np)))
    q4_row_error = float(np.max(np.abs(new_j_np[33, :33])))
    q4_diag_error = float(abs(new_j_np[33, 33] + 1.))
    q4_output = (new.spatial()[:, 3] * new.amplitude[3] / new.mass).detach().cpu().numpy()
    omega_block_error = float(np.max(np.abs(new_j_np[10:20, 33] + q4_output)))
    block_error = max(top_left_error, q4_row_error, q4_diag_error, omega_block_error)
    if fd_error > 1e-6 or block_error > 1e-10:
        raise AssertionError(f"rank-4 Jacobian preflight failed: fd={fd_error}, block={block_error}")
    actual = np.linalg.eigvals(new_j_np)
    return {"equivalence": equivalence, "jacobian_fd_max_abs": fd_error,
            "jacobian_block_max_abs": block_error, "top_left_error": top_left_error,
            "q4_row_error": q4_row_error, "q4_diag_error": q4_diag_error,
            "q4_omega_column_error": omega_block_error,
            "spectrum_has_rank4": bool(len(actual) == 34),
            "old_eigenvalues": [[float(v.real), float(v.imag)] for v in np.linalg.eigvals(old_j.detach().numpy())],
            "new_eigenvalues": [[float(v.real), float(v.imag)] for v in actual]}


def _failure(output: Path, traces, reason: str, source_before: dict[str, str], model, phase: str, current):
    torch.save(model.state_dict(), output / "failure_model.pt")
    evidence = {"arm": "J-rank4-S", "mode": "failure", "phase": phase, "reason": reason,
                "traces": traces, "current_failed_batch": current,
                "source_hashes_before": source_before, "source_hashes_after_failure": source_hashes(),
                "test_read": False}
    evidence["nonfinite"] = _has_nonfinite(evidence)
    (output / "failure_evidence.json").write_text(
        json.dumps(_jsonable(evidence, allow_nonfinite=True), indent=2, allow_nan=False), encoding="utf-8")


def run(output: Path, *, smoke: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(1); torch.manual_seed(67)
    train_rng = np.random.default_rng(67)
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
    old, model, init = construct_rank4(base, network)
    open_rank4_support(model)
    open_support = model.support.detach().clone()
    source = _state_copy(model.state_dict())
    allowed = set(configure_trainable(model, "J"))
    if allowed != set(JOINT):
        raise AssertionError("rank-4 trainable parameter set mismatch")
    preflight = _rank4_preflight(old, model)
    initial_f0 = _assert_zero_field(model)
    if not np.isfinite(initial_f0):
        raise FloatingPointError("nonfinite rank-4 initial f(0) diagnostic")
    if _has_nonfinite(init) or _has_nonfinite(preflight):
        raise FloatingPointError("nonfinite rank-4 initialization/preflight diagnostics")
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if name in allowed], lr=LR)
    steps = SMOKE_STEPS if smoke else STEPS
    prune_after = SMOKE_PRUNE_AFTER if smoke else OPEN_STEPS
    phase = "open"
    support_after = None
    prune_report = None
    traces: list[dict[str, Any]] = []
    current = None
    try:
        first_samples, first_histories = sample_batch(train_events, train_rng, count=BATCH_SIZE)
        first_horizon = horizon_for_step(1)
        current = {"step": 1, "phase": phase, "horizon_s": first_horizon,
                   "event_ids": [event["id"] for event, _ in first_samples],
                   "endpoints": [int(endpoint) for _, endpoint in first_samples]}
        old_loss, old_terms = objective(old, first_samples, first_histories, first_horizon)
        first_loss, first_terms = rank4_objective(model, first_samples, first_histories, first_horizon)
        initial_old_loss = float(old_loss.detach())
        initial_total = float(first_loss.detach())
        if not np.isfinite(initial_old_loss) or not np.isfinite(initial_total) or _has_nonfinite(first_terms):
            raise FloatingPointError("nonfinite rank-4 initial objective diagnostics")
        if abs(initial_old_loss - EXPECTED_INITIAL_DATA) > 1e-12:
            raise AssertionError(f"initial S data loss mismatch: {initial_old_loss}")
        if abs(initial_total - (initial_old_loss + 1e-4 / (76 * 3))) > 1e-12:
            raise AssertionError(f"initial rank-4 total loss mismatch: {initial_total}")
        initial_term_errors = {name: float((first_terms[name] - value).abs().detach())
                               for name, value in old_terms.items() if name != "xi_l1"}
        if max(initial_term_errors.values()) > 1e-12:
            raise AssertionError(f"initial S objective term mismatch: {initial_term_errors}")
        del old_loss, old_terms, first_loss, first_terms
        for step in range(1, steps + 1):
            horizon = horizon_for_step(step)
            if step == 1:
                samples, histories = first_samples, first_histories
            else:
                samples, histories = sample_batch(train_events, train_rng, count=BATCH_SIZE)
            provenance = {"step": step, "phase": phase, "horizon_s": horizon,
                          "event_ids": [event["id"] for event, _ in samples],
                          "endpoints": [int(endpoint) for _, endpoint in samples]}
            current = provenance
            optimizer.zero_grad(set_to_none=True)
            loss, terms = rank4_objective(model, samples, histories, horizon)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite rank-4 loss")
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
            expected = open_support if support_after is None else support_after
            checks = _phase_contract(model, source, expected)
            if not all(checks.values()):
                raise AssertionError(f"rank-4 phase contract: {checks}")
            traces.append({**provenance, "loss": float(loss.detach()),
                           "gradient_norm": float(torch.linalg.vector_norm(torch.cat(gradients))),
                           "terms": {name: float(value.detach()) for name, value in terms.items()},
                           "allowed_tensor_checks": checks})
            if step == 1 or step % 50 == 0:
                print(json.dumps({"step": step, "phase": phase, "horizon_s": horizon,
                                  "loss": traces[-1]["loss"], "gradient_norm": traces[-1]["gradient_norm"]}), flush=True)
            if step == prune_after:
                prune_report = prune_support(model, train_events, output)
                support_after = model.support.detach().clone()
                post = _phase_contract(model, source, support_after)
                if not all(post.values()):
                    raise AssertionError(f"post-prune rank-4 contract: {post}")
                optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if name in allowed], lr=LR)
                phase = "refit"
                print(json.dumps({"event": "support_prune", "selected_count": prune_report["selected_count"],
                                  "q4_memory_entry_selected": prune_report["q4_memory_entry_selected"]}), flush=True)
        source_after = source_hashes()
        if source_after != source_before:
            raise AssertionError("source artifacts changed during fit")
        if prune_report is None:
            raise AssertionError("rank-4 support pruning did not occur")
    except Exception as exc:
        failed = [] if current is None else [{**current, "failed": True, "reason": str(exc)}]
        _failure(output, traces + failed, str(exc), source_before, model, phase, current)
        raise
    shutil.copyfile(SOURCE / "normalization.npz", output / "normalization.npz")
    torch.save(model.state_dict(), output / "model.pt")
    evidence = {
        "arm": "J-rank4-S", "mode": "SMOKE ONLY" if smoke else "formal", "seed": 67,
        "steps": steps, "configured_steps": STEPS, "open_steps": prune_after,
        "refit_start_step": prune_after + 1, "lr": LR, "batch_size": BATCH_SIZE, "full_train": True,
        "allowed_parameters": sorted(allowed), "source_hashes_before": source_before,
        "source_hashes_after": source_after, "model_sha256": digest(output / "model.pt"),
        "normalization_sha256": digest(output / "normalization.npz"), "excluded": excluded,
        "identities": identities, "train_event_ids": [event["id"] for event in train_events],
        "initialization": init, "preflight": preflight, "initial_f0_max_abs": initial_f0,
        "initial_S_total_loss": initial_old_loss, "initial_total_loss": initial_total,
        "initial_S_term_max_abs_errors": initial_term_errors,
        "expected_initial_total": EXPECTED_INITIAL_TOTAL,
        "support_phase": prune_report, "traces": traces, "seconds": time.perf_counter() - started,
        "rank": RANK, "xi_nonconstant_count": NONCONSTANT_COUNT, "nonfinite": False, "test_read": False,
    }
    (output / "evidence.json").write_text(json.dumps(_jsonable(evidence), indent=2, allow_nan=False), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="one open step, prune, one refit step; not a scientific fit")
    args = parser.parse_args()
    run(args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
