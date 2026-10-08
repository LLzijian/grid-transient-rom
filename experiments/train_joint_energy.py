"""Loop0073 matched J training with the registered sparse-storage bias.

The 224-term storage algebra and Clarabel solve live in
``joint_energy_helpers``.  This file only orchestrates the fixed S-start,
0067 data objective/sampler, alternating solves, and auditable artifacts.
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
from fit_sparse_storage import _w_matrix, h3_hessian_origin, library_hessians_origin, load_local_sources
from joint_energy_helpers import (
    FIT_MARGIN, JOINT_NAMES, RESIDUAL_TOL, assert_zero_field, calibrate_lambda,
    energy_lh, exact_j32, select_energy_support, solve_soft_storage, storage_scales,
)
from train_joint_campaign import (
    BATCH_SIZE, JOINT, LR, SOURCE, SOURCE_MODEL_SHA256, STEPS, _assert_zero_field,
    _frozen_equal, _masked_inactive_equal, configure_trainable, horizon_for_step,
    objective, sample_batch,
)
from train_joint_latent import BASE, NETWORK, read_development
from kiki.models.private_pm import MechanicalROM

ROOT = Path(__file__).resolve().parents[1]
SEED = 67
LIBRARY_SIZE = 224
PQ_COUNT = 6
FINAL_LIMIT = 24


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
        "energy_helper_code": digest(ROOT / "experiments/joint_energy_helpers.py"),
        "sparse_code": digest(ROOT / "experiments/fit_sparse_storage.py"),
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
            raise FloatingPointError("nonfinite value in evidence")
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


class StorageSolveError(RuntimeError):
    def __init__(self, message: str, record: dict[str, Any]):
        super().__init__(message)
        self.record = record


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _storage_context(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the fixed physical H3/library basis and unit scaling once."""
    h0 = h3_hessian_origin(params)
    hks = library_hessians_origin(params)
    w = _w_matrix(h0)
    v = np.diag(1.0 / np.sqrt(np.diag(w)))
    scales = storage_scales(hks, v)
    return {"H0": h0, "Hks": hks, "W": w, "V": v, "scales": scales}


def _solver_record(result: dict[str, object], *, step: int, phase: str,
                   support: np.ndarray, J32: np.ndarray) -> dict[str, Any]:
    failure_record = {"step": step, "phase": phase, "status": result.get("status"),
                      "feasible": bool(result.get("feasible", False)),
                      "solver_result": result,
                      "support_indices": np.flatnonzero(support).astype(int).tolist(),
                      "J32": np.asarray(J32, dtype=float)}
    if not bool(result.get("feasible", False)):
        raise StorageSolveError(f"storage solver failed at step {step} ({phase})", failure_record)
    gamma = np.asarray(result["gamma"], dtype=float)
    if gamma.shape != (LIBRARY_SIZE,) or not np.isfinite(gamma).all():
        raise FloatingPointError("invalid storage coefficients")
    record = {
        "step": step, "phase": phase, "status": str(result.get("status")),
        "iterations": result.get("iterations"), "t": float(result["t"]),
        "support_count": int(np.count_nonzero(support)),
        "support_indices": np.flatnonzero(support).astype(int).tolist(),
        "residual_h": float(result["residual_h"]),
        "residual_d": float(result["residual_d"]),
        "inactive_max": float(result["inactive_max"]),
        "objective": float(result["t"]) + 1e-4 * float(np.abs(np.asarray(result["beta"], dtype=float)).sum()),
        "finite": True,
    }
    if record["residual_h"] > RESIDUAL_TOL or record["residual_d"] > RESIDUAL_TOL:
        raise StorageSolveError(f"storage residual exceeds tolerance: {record}", failure_record)
    if record["inactive_max"] > RESIDUAL_TOL:
        raise StorageSolveError(f"inactive storage coefficient is nonzero: {record}", failure_record)
    if np.any(gamma[~support] != 0.0):
        raise StorageSolveError("inactive storage coefficients are not exactly zero", failure_record)
    return {**record, "gamma": gamma, "beta": np.asarray(result["beta"], dtype=float),
            "H_normalized": np.asarray(result["H_normalized"], dtype=float),
            "D_normalized": np.asarray(result["D_normalized"], dtype=float),
            "dual_H": np.asarray(result["dual_H"], dtype=float) if result.get("dual_H") is not None else np.empty((0, 0)),
            "dual_D": np.asarray(result["dual_D"], dtype=float) if result.get("dual_D") is not None else np.empty((0, 0)),
            "J32": np.asarray(J32, dtype=float)}


def _solve_storage(model, params: dict[str, np.ndarray], context: dict[str, np.ndarray],
                   support: np.ndarray, *, step: int, phase: str) -> dict[str, Any]:
    j32 = exact_j32(model, create_graph=False).detach().cpu().numpy()
    result = solve_soft_storage(context["H0"], context["Hks"], context["V"], j32,
                                scales=context["scales"], support=support)
    record = _solver_record(result, step=step, phase=phase, support=support, J32=j32)
    record["H_raw"] = context["H0"] + np.tensordot(record["gamma"], context["Hks"], axes=(0, 0))
    record["D_raw"] = -(j32.T @ record["H_raw"] + record["H_raw"] @ j32)
    return record


def _gradient_alignment(loss_data, loss_energy, model) -> dict[str, Any]:
    data_norm, data_grad = _gradient_vector(loss_data, model)
    energy_norm, energy_grad = _gradient_vector(loss_energy, model)
    d = torch.cat(data_grad).detach()
    e = torch.cat(energy_grad).detach()
    dot = float(torch.dot(d, e))
    denom = float(torch.linalg.vector_norm(d) * torch.linalg.vector_norm(e))
    return {"gL": float(data_norm), "gH": float(energy_norm),
            "dot": dot, "cosine": None if denom == 0. else dot / denom}


def _gradient_vector(loss, model) -> tuple[float, list[torch.Tensor]]:
    by_name = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, [by_name[name] for name in JOINT_NAMES],
                                    retain_graph=True, allow_unused=True)
    zero = torch.zeros((), dtype=loss.dtype, device=loss.device)
    flat: list[torch.Tensor] = []
    total = zero
    for name, gradient in zip(JOINT_NAMES, gradients):
        if gradient is None:
            flat.append(torch.zeros_like(by_name[name]).reshape(-1))
        else:
            flat.append(gradient.reshape(-1))
            total = total + gradient.square().sum()
    return float(torch.sqrt(total).detach()), flat


def _contract(model, source: dict[str, torch.Tensor], allowed: set[str]) -> dict[str, bool]:
    return {
        "frozen_source": _frozen_equal(model, source, allowed),
        "inactive_xi_alpha": _masked_inactive_equal(model, source),
        "support_fixed": torch.equal(model.support, source["support"]),
        "alpha_support_fixed": torch.equal(model.alpha_support, source["alpha_support"]),
        "f0_zero": _assert_zero_field(model) <= 1e-10,
    }


def _final_storage_diagnostics(model, gamma: np.ndarray, context: dict[str, np.ndarray]) -> dict[str, Any]:
    j32 = exact_j32(model, create_graph=False).detach().cpu().numpy()
    h_raw = context["H0"] + np.tensordot(gamma, context["Hks"], axes=(0, 0))
    d_raw = -(j32.T @ h_raw + h_raw @ j32)
    h_norm = context["V"] @ h_raw @ context["V"]
    d_norm = context["V"] @ d_raw @ context["V"]
    return {"step": "model_final_no_solve", "gamma": np.asarray(gamma, dtype=float),
            "J32": j32, "H_raw": h_raw, "D_raw": d_raw,
            "H_normalized": h_norm, "D_normalized": d_norm,
            "hessian_min_eig": float(np.linalg.eigvalsh(h_norm).min()),
            "lyapunov_min_eig": float(np.linalg.eigvalsh(d_norm).min())}


def _failure(output: Path, mode: str, step: int, reason: str, model,
             source_before: dict[str, str], traces: list[dict[str, Any]],
             storage_records: list[dict[str, Any]], failed_provenance: dict[str, Any] | None) -> None:
    torch.save(model.state_dict(), output / "failure_model.pt")
    evidence = {"arm": "J-energy", "mode": mode, "step": step, "reason": reason,
                "traces": traces, "storage_traces": storage_records,
                "failed_provenance": failed_provenance,
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
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
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
    model = _construct(SOURCE, base, network, "challenger")
    source = {name: value.detach().clone() for name, value in model.state_dict().items()}
    allowed = set(JOINT)
    configured = set(configure_trainable(model, "J"))
    if configured != allowed:
        raise AssertionError(f"unexpected JointROM parameter names: {sorted(configured ^ allowed)}")
    params = load_local_sources(ROOT)
    context = _storage_context(params)
    support = np.ones(LIBRARY_SIZE, dtype=bool)
    storage_records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    failed_provenance: dict[str, Any] | None = None
    optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if name in allowed], lr=LR)
    steps = 2 if smoke else STEPS
    try:
        initial_f0 = assert_zero_field(model)
        first_solver = _solve_storage(model, params, context, support, step=0, phase="full")
        storage_records.append(first_solver)
        gamma = first_solver["gamma"].copy()
        first_samples, first_histories = sample_batch(train_events, rng, count=BATCH_SIZE)
        first_horizon = horizon_for_step(1)
        failed_provenance = {"step": 1, "horizon_s": first_horizon,
                             "event_ids": [event["id"] for event, _ in first_samples],
                             "endpoints": [int(endpoint) for _, endpoint in first_samples],
                             "phase": "calibration"}
        first_data_loss, _ = objective(model, first_samples, first_histories, first_horizon)
        first_energy_loss, first_eigs, _ = energy_lh(
            model, torch.as_tensor(gamma, dtype=torch.float64), context["H0"], context["Hks"], context["V"])
        calibration = _gradient_alignment(first_data_loss, first_energy_loss, model)
        lambda_h, g_l, g_h = calibrate_lambda(first_data_loss, first_energy_loss, model)
        calibration.update({"lambdaH": lambda_h, "gL": g_l, "gH": g_h,
                            "first_data_loss": float(first_data_loss.detach()),
                            "first_energy_loss": float(first_energy_loss.detach()),
                            "first_energy_min_eig": float(first_eigs.min().detach())})
        del first_data_loss, first_energy_loss, first_eigs
        for step in range(1, steps + 1):
            horizon = horizon_for_step(step)
            if step == 1:
                samples, histories = first_samples, first_histories
            else:
                samples, histories = sample_batch(train_events, rng, count=BATCH_SIZE)
            failed_provenance = {"step": step, "horizon_s": horizon,
                                 "event_ids": [event["id"] for event, _ in samples],
                                 "endpoints": [int(endpoint) for _, endpoint in samples]}
            optimizer.zero_grad(set_to_none=True)
            data_loss, terms = objective(model, samples, histories, horizon)
            energy_loss, eigs, _ = energy_lh(
                model, torch.as_tensor(gamma, dtype=torch.float64), context["H0"], context["Hks"], context["V"])
            loss = data_loss + lambda_h * energy_loss
            if not _finite_tensor(loss):
                raise FloatingPointError("nonfinite combined loss")
            loss.backward()
            gradients = []
            for name, parameter in model.named_parameters():
                if name in allowed:
                    if parameter.grad is None or not _finite_tensor(parameter.grad):
                        raise FloatingPointError(f"nonfinite gradient: {name}")
                    gradients.append(parameter.grad.detach().reshape(-1))
            optimizer.step()
            if any(not _finite_tensor(parameter) for name, parameter in model.named_parameters() if name in allowed):
                raise FloatingPointError("nonfinite parameter")
            checks = _contract(model, source, allowed)
            if not all(checks.values()):
                raise AssertionError(f"frozen/zero-field contract: {checks}")
            traces.append({**failed_provenance, "loss": float(loss.detach()),
                           "data_loss": float(data_loss.detach()), "energy_loss": float(energy_loss.detach()),
                           "energy_min_eig": float(eigs.min().detach()),
                           "gradient_norm": float(torch.linalg.vector_norm(torch.cat(gradients))),
                           "terms": {name: float(value.detach()) for name, value in terms.items()},
                           "allowed_tensor_checks": checks})
            if step == 1 or step % 50 == 0:
                print(json.dumps({"step": step, "horizon_s": horizon, "loss": traces[-1]["loss"],
                                  "data_loss": traces[-1]["data_loss"], "energy_loss": traces[-1]["energy_loss"],
                                  "gradient_norm": traces[-1]["gradient_norm"]}), flush=True)
            if step % 100 == 0 and step <= STEPS:
                phase = "full" if step <= 200 else "sparse"
                full = _solve_storage(model, params, context, support, step=step, phase=phase)
                storage_records.append(full)
                if step == 200:
                    support = select_energy_support(full["gamma"], context["scales"], limit=18)
                    if np.count_nonzero(support) != FINAL_LIMIT or not np.all(support[:PQ_COUNT]):
                        raise AssertionError("invalid registered 6+18 support")
                    sparse_refit = _solve_storage(model, params, context, support, step=step, phase="sparse_refit")
                    storage_records.append(sparse_refit)
                    gamma = sparse_refit["gamma"].copy()
                else:
                    gamma = full["gamma"].copy()
        source_after = source_hashes()
        if source_after != source_before:
            raise AssertionError("source artifacts changed during fit")
    except Exception as exc:
        if isinstance(exc, StorageSolveError):
            storage_records.append(exc.record)
        _failure(output, "SMOKE" if smoke else "formal", failed_provenance["step"] if failed_provenance else 0,
                 str(exc), model, source_before, traces, storage_records, failed_provenance)
        raise
    if len(storage_records) == 0:
        raise AssertionError("missing storage records")
    final = storage_records[-1]
    final_current = _final_storage_diagnostics(model, gamma, context)
    np.savez_compressed(
        output / "storage.npz", gamma=np.stack([r["gamma"] for r in storage_records]),
        beta=np.stack([r["beta"] for r in storage_records]),
        H_normalized=np.stack([r["H_normalized"] for r in storage_records]),
        D_normalized=np.stack([r["D_normalized"] for r in storage_records]),
        H_raw=np.stack([r["H_raw"] for r in storage_records]),
        D_raw=np.stack([r["D_raw"] for r in storage_records]),
        J32=np.stack([r["J32"] for r in storage_records]), H0=context["H0"], Hks=context["Hks"],
        W=context["W"], V=context["V"], scales=context["scales"], support=support,
        t=np.asarray([r["t"] for r in storage_records], dtype=float),
        steps=np.asarray([r["step"] for r in storage_records], dtype=int),
        dual_H=np.stack([r["dual_H"] for r in storage_records]),
        dual_D=np.stack([r["dual_D"] for r in storage_records]),
        final_gamma=final_current["gamma"], final_J32=final_current["J32"],
        final_H_raw=final_current["H_raw"], final_D_raw=final_current["D_raw"],
        final_H_normalized=final_current["H_normalized"], final_D_normalized=final_current["D_normalized"])
    torch.save(model.state_dict(), output / "model.pt")
    shutil.copyfile(SOURCE / "normalization.npz", output / "normalization.npz")
    evidence = {
        "arm": "J-energy", "mode": "SMOKE ONLY" if smoke else "formal", "seed": SEED,
        "steps": steps, "configured_steps": STEPS, "lr": LR, "batch_size": BATCH_SIZE,
        "full_train": True, "allowed_parameters": sorted(allowed),
        "source_hashes_before": source_before, "source_hashes_after": source_after,
        "model_sha256": digest(output / "model.pt"), "normalization_sha256": digest(output / "normalization.npz"),
        "excluded": excluded, "identities": identities,
        "train_event_ids": [event["id"] for event in train_events],
        "initial_f0_max_abs": initial_f0, "calibration": calibration,
        "storage_traces": [{k: v for k, v in r.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
                           for r in storage_records],
        "storage_solve_final_step": int(final["step"]),
        "final_storage_terms": int(np.count_nonzero(np.abs(final_current["gamma"]) > 0)),
        "final_hessian_min_eig": final_current["hessian_min_eig"],
        "final_lyapunov_min_eig": final_current["lyapunov_min_eig"],
        "traces": traces, "seconds": time.perf_counter() - started,
        "nonfinite": False, "test_read": False,
    }
    (output / "evidence.json").write_text(json.dumps(_jsonable(evidence), indent=2, allow_nan=False), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="two TRAIN steps only; not a scientific fit")
    args = parser.parse_args()
    run(args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
