"""Loop0062 frozen-encoder, fixed-support TRAIN group identifiability audit.

This diagnostic never writes a model and deliberately constructs windows only
from genuine causal 11-sample histories in the explicit development TRAIN set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_long_latent import BASE, NETWORK, _construct
from kiki.models.private_pm import MechanicalROM
from train_joint_latent import DT, read_development, tensor


RCOND = 1e-10
RANK_TOLS = (1e-12, 1e-10, 1e-8)
EXPECTED_MODEL_SHA = "d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c"


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_ranges(length: int) -> list[tuple[int, int]]:
    """Successive [a,b] windows over genuine endpoints 10..length-1."""
    if length <= 11:
        return []
    result: list[tuple[int, int]] = []
    start, last = 10, length - 1
    while start < last:
        end = min(start + 10, last)
        result.append((start, end))
        start = end
    return result


def history_indices(length: int) -> np.ndarray:
    """All causal encoder endpoints; every returned i owns x[i-10:i+1]."""
    if length <= 10:
        return np.empty(0, dtype=int)
    return np.arange(10, length, dtype=int)


def window_average(features: np.ndarray, q: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray, float]:
    """Integrated feature mean and q increment derivative on one genuine window."""
    if len(features) != len(q) or len(features) < 2:
        raise ValueError("window requires matching endpoint-inclusive arrays")
    duration = (len(features) - 1) * dt
    return np.trapezoid(features, dx=dt, axis=0) / duration, (q[-1] - q[0]) / duration, duration


def event_weights(event_ids: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """sqrt(1/(events * windows-per-event)): equal total weight per event."""
    ids = np.asarray(event_ids).astype(str)
    unique, counts = np.unique(ids, return_counts=True)
    if not len(unique) or np.any(counts <= 0):
        raise ValueError("empty event-weight input")
    count_map = {str(key): int(value) for key, value in zip(unique, counts, strict=True)}
    weights = np.array([np.sqrt(1.0 / (len(unique) * count_map[key])) for key in ids], dtype=float)
    return weights, count_map


def _spectrum(design: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(design).all():
        raise FloatingPointError("nonfinite scaled design")
    singular = np.linalg.svd(design, compute_uv=False)
    largest = float(singular[0]) if len(singular) else 0.0
    ranks = {str(tol): int(np.count_nonzero(singular > tol * largest)) for tol in RANK_TOLS}
    rank_rcond = ranks[str(RCOND)]
    full_rank = rank_rcond == design.shape[1]
    condition = (float(largest / singular[-1])
                 if len(singular) == design.shape[1] and len(singular) and singular[-1] > 0 else None)
    if condition is not None and not np.isfinite(condition):
        condition = None
    return {
        "singular_values": singular.tolist(),
        "rank_at_rtol": ranks, "condition": condition,
        "singular_or_underdetermined": bool(not full_rank),
        "nonunique_at_rcond": bool(not full_rank),
    }


def fit_supported(X: np.ndarray, y: np.ndarray, weights: np.ndarray, mask: np.ndarray,
                  scale: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Weighted SVD minimum-norm fits restricted exactly to saved support."""
    if X.shape[1] != mask.shape[0] or y.shape[1] != mask.shape[1]:
        raise ValueError("design/mask dimensions")
    if (not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(weights).all()
            or not np.isfinite(scale).all() or np.any(weights <= 0) or np.any(scale <= 0)):
        raise FloatingPointError("nonfinite or invalid fixed-support fit input")
    coefficient = np.zeros_like(mask, dtype=float)
    diagnostics: list[dict[str, Any]] = []
    for output in range(y.shape[1]):
        active = np.flatnonzero(mask[:, output])
        if len(active) != 12:
            raise AssertionError(f"output {output}: expected 12 active support columns, got {len(active)}")
        design = X[:, active] / scale[active]
        weighted = design * weights[:, None]
        solved, *_ = np.linalg.lstsq(weighted, y[:, output] * weights, rcond=RCOND)
        coefficient[active, output] = solved / scale[active]
        info = _spectrum(weighted)
        info.update(output=int(output), active_indices=active.tolist())
        diagnostics.append(info)
    if not np.isfinite(coefficient).all():
        raise FloatingPointError("nonfinite diagnostic coefficient")
    return coefficient, diagnostics


def _rms_event_balanced(values: np.ndarray, event_ids: np.ndarray) -> float:
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite projected values")
    weights, _ = event_weights(event_ids)
    result = float(np.sqrt(np.sum((weights[:, None] * values) ** 2) / values.shape[1]))
    if not np.isfinite(result):
        raise FloatingPointError("nonfinite event-balanced RMS")
    return result


def projected_metrics(X: np.ndarray, y: np.ndarray, event_ids: np.ndarray, labels: np.ndarray,
                      Cq: np.ndarray, xi: np.ndarray, pooled: np.ndarray | None = None) -> dict[str, Any]:
    """Output-force projected reference defect and optional pooled-disagreement numerator."""
    if not all(np.isfinite(value).all() for value in (X, y, Cq, xi)):
        raise FloatingPointError("nonfinite projected-metric input")
    if pooled is not None and not np.isfinite(pooled).all():
        raise FloatingPointError("nonfinite pooled comparison coefficient")
    def subset(which: np.ndarray) -> dict[str, float | int]:
        if not np.any(which):
            return {"n_windows": 0, "n_events": 0, "rms_force_per_s": None}
        residual = (y[which] - X[which] @ xi) @ Cq.T
        return {"n_windows": int(which.sum()), "n_events": int(len(np.unique(event_ids[which]))),
                "rms_force_per_s": _rms_event_balanced(residual, event_ids[which])}
    all_rows = np.ones(len(X), dtype=bool)
    result: dict[str, Any] = {"all": subset(all_rows), "stable": subset(labels == "stable"),
                              "unstable": subset(labels == "unstable")}
    if pooled is not None:
        def disagreement(which: np.ndarray) -> dict[str, float | int | None]:
            if not np.any(which):
                return {"n_windows": 0, "n_events": 0, "numerator_force_per_s": None,
                        "denominator_force_per_s": None, "ratio": None}
            delta = (X[which] @ (xi - pooled)) @ Cq.T
            field = (X[which] @ pooled) @ Cq.T
            numerator = _rms_event_balanced(delta, event_ids[which])
            denominator = _rms_event_balanced(field, event_ids[which])
            ratio = numerator / max(denominator, 1e-8)
            if not np.isfinite(ratio):
                raise FloatingPointError("nonfinite field-disagreement ratio")
            return {"n_windows": int(which.sum()), "n_events": int(len(np.unique(event_ids[which]))),
                    "numerator_force_per_s": numerator, "denominator_force_per_s": denominator,
                    "ratio": ratio}
        result["field_difference"] = {"all": disagreement(all_rows),
                                      "stable": disagreement(labels == "stable"),
                                      "unstable": disagreement(labels == "unstable")}
    return result


def _safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, np.ndarray): return _safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite evidence value")
        return float(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    return value


def _encode_event(model: Any, event: dict[str, Any], batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoints = history_indices(len(event["x"]))
    histories = np.stack([event["x"][i - 10:i + 1] for i in endpoints])
    encoded, features = [], []
    with torch.no_grad():
        for start in range(0, len(endpoints), batch_size):
            stop = min(start + batch_size, len(endpoints))
            q = model.encode(tensor(histories[start:stop]))
            state = torch.cat((tensor(event["x"][endpoints[start:stop]]), q), dim=1)
            encoded.append(q.cpu().numpy())
            features.append(model.features(state).cpu().numpy())
    return endpoints, np.concatenate(encoded), np.concatenate(features)


def _event_rows(model: Any, event: dict[str, Any]) -> dict[str, np.ndarray]:
    endpoints, q, features = _encode_event(model, event)
    by_endpoint = {int(index): position for position, index in enumerate(endpoints)}
    X, y, starts, ends, durations = [], [], [], [], []
    for start, end in window_ranges(len(event["x"])):
        positions = np.array([by_endpoint[i] for i in range(start, end + 1)], dtype=int)
        mean, target, duration = window_average(features[positions], q[positions])
        X.append(mean); y.append(target); starts.append(start); ends.append(end); durations.append(duration)
    return {"X": np.asarray(X), "y": np.asarray(y), "start": np.asarray(starts),
            "end": np.asarray(ends), "duration": np.asarray(durations)}


def _design_report(X: np.ndarray, event_ids: np.ndarray, mask: np.ndarray, scale: np.ndarray) -> list[dict[str, Any]]:
    weights, _ = event_weights(event_ids)
    reports = []
    for output in range(mask.shape[1]):
        active = np.flatnonzero(mask[:, output])
        report = _spectrum((X[:, active] / scale[active]) * weights[:, None])
        report.update(output=int(output), active_indices=active.tolist())
        reports.append(report)
    return reports


def _aq_spectrum(xi: np.ndarray) -> list[list[float]]:
    """Descriptive latent linear-block poles, with no ordering or gate."""
    poles = np.linalg.eigvals(xi[1:4].T)
    return [[float(pole.real), float(pole.imag)] for pole in poles]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("private_data/task2_long_0060_5s"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial_model_hash = digest(args.model_dir / "model.pt")
    if initial_model_hash != EXPECTED_MODEL_SHA:
        raise AssertionError("unexpected frozen Loop0060 5s checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False))
    network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]])
    network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    model = _construct(args.model_dir, base, network)
    model.eval()
    if model.rank != 3:
        raise AssertionError(f"expected r=3, got {model.rank}")
    development, excluded, identities = read_development(base, network, full_train=True)
    events = development["train"]
    groups = sorted({str(event["group"]) for event in events})
    if len(events) != 346 or len(groups) != 24:
        raise AssertionError(f"expected 346 events / 24 TRAIN groups, got {len(events)} / {len(groups)}")
    train_lengths = {str(event["id"]): int(len(event["x"])) for event in events}

    rows = [_event_rows(model, event) for event in events]
    X = np.concatenate([row["X"] for row in rows])
    y = np.concatenate([row["y"] for row in rows])
    event_ids = np.concatenate([np.repeat(event["id"], len(row["X"])) for event, row in zip(events, rows, strict=True)])
    row_groups = np.concatenate([np.repeat(str(event["group"]), len(row["X"])) for event, row in zip(events, rows, strict=True)])
    labels = np.concatenate([np.repeat(str(event["label"]), len(row["X"])) for event, row in zip(events, rows, strict=True)])
    starts = np.concatenate([row["start"] for row in rows]); ends = np.concatenate([row["end"] for row in rows])
    durations = np.concatenate([row["duration"] for row in rows])
    weights, counts = event_weights(event_ids)
    with torch.no_grad():
        support_raw = model.support.cpu().numpy()
        if not np.all((support_raw == 0) | (support_raw == 1)):
            raise AssertionError("support must remain binary")
        mask = support_raw != 0
        source_xi = (model.xi * model.support).cpu().numpy()
        cunit = model.c / torch.linalg.vector_norm(model.c, dim=0, keepdim=True)
        Cq = (model.ports @ cunit * model.amplitude).cpu().numpy()
        zero_history = torch.zeros(1, 11, 30, dtype=torch.float64)
        zero_q = model.encode(zero_history)
        zero_state = torch.cat((torch.zeros(1, 30, dtype=torch.float64), zero_q), dim=1)
        zero_force = model.memory(model.memory_enabled * zero_q + model.algebraic_coordinate(zero_state))
        zero_field = model.field(zero_state)
    if X.shape[1] != 76 or mask.shape != (76, 3) or not np.all(mask.sum(axis=0) == 12):
        raise AssertionError("expected frozen 76-column / 12-active-per-output support")
    scales = np.maximum(np.sqrt(np.sum((weights[:, None] * X) ** 2, axis=0)), 1e-8)
    if not all(np.isfinite(value).all() for value in (X, y, weights, scales, mask, Cq, source_xi)):
        raise FloatingPointError("nonfinite frozen diagnostic input before SVD")
    pooled, pooled_spectra = fit_supported(X, y, weights, mask, scales)

    group_spectra = {group: _design_report(X[row_groups == group], event_ids[row_groups == group], mask, scales)
                     for group in groups}
    lgo_coefficients, lgo_spectra, group_metrics = [], {}, {}
    for group in groups:
        held = row_groups == group
        train = ~held
        lgo_weights, _ = event_weights(event_ids[train])
        coefficient, spectra = fit_supported(X[train], y[train], lgo_weights, mask, scales)
        lgo_coefficients.append(coefficient); lgo_spectra[group] = spectra
        deployed = projected_metrics(X[held], y[held], event_ids[held], labels[held], Cq, source_xi)
        pooled_metrics = projected_metrics(X[held], y[held], event_ids[held], labels[held], Cq, pooled)
        lgo_metrics = projected_metrics(X[held], y[held], event_ids[held], labels[held], Cq, coefficient, pooled)
        denom = max(float(pooled_metrics["all"]["rms_force_per_s"]), 1e-8)
        group_metrics[group] = {
            "deployed": deployed, "pooled": pooled_metrics, "leave_group_out": lgo_metrics,
            "E_lgo_over_pooled": float(lgo_metrics["all"]["rms_force_per_s"]) / denom,
            "screens_raw": {"E_ratio_le_1p10": bool(float(lgo_metrics["all"]["rms_force_per_s"]) / denom <= 1.10),
                            "D_le_0p10": bool(lgo_metrics["field_difference"]["all"]["ratio"] <= .10)},
        }
    lgo_coefficients_array = np.stack(lgo_coefficients)
    if digest(args.model_dir / "model.pt") != initial_model_hash:
        raise AssertionError("frozen checkpoint changed during diagnostic")
    rank_ok = all(report["rank_at_rtol"][str(RCOND)] == 12 for report in pooled_spectra)
    rank_ok &= all(report["rank_at_rtol"][str(RCOND)] == 12 for reports in lgo_spectra.values() for report in reports)
    all_screens = all(value["screens_raw"]["E_ratio_le_1p10"] and value["screens_raw"]["D_le_0p10"] for value in group_metrics.values())

    arrays = {"X": X, "y": y, "event_ids": event_ids, "group": row_groups, "label": labels,
              "start": starts, "end": ends, "duration": durations, "weights": weights,
              "eventwindowcounts": np.asarray([counts[event] for event in event_ids]), "Cq": Cq,
              "mask": mask, "sourceXi": source_xi, "scales": scales,
              "diagnosticcoeffs_pooled": pooled, "diagnosticcoeffs_leave_group_out": lgo_coefficients_array,
              "leave_group_out_groups": np.asarray(groups)}
    np.savez_compressed(args.output / "windows.npz", **arrays)
    evidence = {
        "test_read": False, "validation_loaded_by_existing_reader_not_used": True,
        "train_events": len(events), "train_groups": groups, "window_count": int(len(X)),
        "train_lengths": train_lengths, "event_window_counts": counts,
        "excluded_predecision": excluded, "identities": identities,
        "window_definition": {"endpoint_start": 10, "max_steps": 10, "dt": DT,
                              "partial_final_window_retained": True},
        "weight_definition": "sqrt(1/(N_events*N_windows_in_event)); LGO recomputes N_events",
        "feature_columns": int(X.shape[1]), "active_per_output": mask.sum(axis=0).astype(int).tolist(),
        "pooled_column_rms_scale_floor_1e-8": scales.tolist(), "mask": mask.tolist(),
        "source_xi": source_xi.tolist(), "Cq_physical_force_per_q": Cq.tolist(),
        "encoder_zero_history_q": zero_q.cpu().numpy().tolist(),
        "encoder_zero_history_actual_force": zero_force.cpu().numpy().tolist(),
        "encoder_zero_history_full_field": zero_field.cpu().numpy().tolist(),
        "pooled": {"coefficients": pooled.tolist(), "change_from_deployed": (pooled-source_xi).tolist(),
                   "spectra": pooled_spectra, "Aq_spectrum": _aq_spectrum(pooled)},
        "source_Aq_spectrum": _aq_spectrum(source_xi), "group_alone_design_spectra": group_spectra,
        "leave_group_out": {"groups": groups, "coefficients": lgo_coefficients_array.tolist(),
                              "change_from_deployed": (lgo_coefficients_array-source_xi).tolist(),
                              "spectra": lgo_spectra,
                              "Aq_spectra": {group: _aq_spectrum(coefficient) for group, coefficient in
                                              zip(groups, lgo_coefficients, strict=True)}},
        "heldout_group_metrics": group_metrics,
        "raw_boolean_checks_not_scientific_decision": {"all_pooled_and_lgo_rank12_at_1e-10": bool(rank_ok),
                                                         "all_common_group_screens": bool(all_screens)},
        "source_sha256": {"model": initial_model_hash,
                          "normalization": digest(args.model_dir / "normalization.npz"),
                          "base": digest(BASE), "network": digest(NETWORK), "audit_code": digest(__file__),
                          "evaluator_helper": digest(Path(__file__).with_name("evaluate_long_latent.py")),
                          "joint_helper": digest(Path(__file__).with_name("train_joint_latent.py"))},
    }
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(lgo_coefficients_array).all():
        raise FloatingPointError("nonfinite frozen diagnostic arrays")
    (args.output / "evidence.json").write_text(json.dumps(_safe(evidence), indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
