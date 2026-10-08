"""Loop0063 equilibrium-anchored frozen observer-initialization diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT / "src"))
import audit_latent_order_rollout as core
import audit_long_latent_rollout as stageb
from evaluate_long_latent import NETWORK, _construct
from kiki.models.private_pm import MechanicalROM


DT = .01
MODEL = ROOT / "private_data/task2_long_0060_5s"
CACHE = ROOT / "private_data/task2_long_0060_5s_eval"
AUDIT = ROOT / "private_data/task2_long_sparse_audit_0061"
BASELINE = ROOT / "private_data/task2_long_sparse_rollout_0061"
EXPECTED_MODEL_SHA = "d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c"


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_states(starts: np.ndarray, qeq: np.ndarray, arm: str) -> np.ndarray:
    """Return a copy whose physical 30-state prefix is bitwise untouched."""
    result = np.asarray(starts, dtype=float).copy()
    original = result[:, :30].copy()
    qeq = np.asarray(qeq, dtype=float).reshape(3)
    if arm == "original":
        pass
    elif arm == "anchored":
        result[:, 30:] -= qeq
    elif arm == "zero":
        result[:, 30:] = 0.0
    else:
        raise ValueError(f"unknown initialization arm {arm}")
    if not np.array_equal(result[:, :30], original):
        raise AssertionError("initialization altered physical state")
    return result


def common_omega_metric(pred: np.ndarray, truth: np.ndarray, mass: np.ndarray, horizon: int | str) -> dict[str, float | int] | None:
    steps = len(truth) - 1 if horizon == "prefix" else round(float(horizon) / DT)
    if steps <= 0 or (horizon != "prefix" and len(truth) <= steps):
        return None
    p = (pred[1:steps + 1, 10:20] * mass).sum(axis=1) / mass.sum()
    t = (truth[1:steps + 1, 10:20] * mass).sum(axis=1) / mass.sum()
    sse = float(np.sum((p - t) ** 2))
    return {"n": int(steps), "sse": sse, "rmse": float(np.sqrt(sse / steps))}


def aggregate_common(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([row["rmse"] for row in rows], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise FloatingPointError("nonfinite or empty common-frequency rows")
    worst = int(values.argmax())
    return {"n_events": len(rows), "mean": float(values.mean()),
            "pooled": float(np.sqrt(sum(row["sse"] for row in rows) / sum(row["n"] for row in rows))),
            "worst": float(values[worst]), "worst_event": rows[worst]["event_id"]}


def _force(state: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    q = state[..., 30:]
    af = core.angle_features(state[..., :10], p)
    algebraic = af @ (p["alpha"] * p["alpha_support"])
    return (q + algebraic) @ p["Cq"].T


def _invalid(name: str, stage: str, array: np.ndarray, ids: list[str]) -> dict[str, Any] | None:
    if array.ndim == 3:
        return stageb.finite_status(name, stage, array, ids)
    if not np.isfinite(array).all():
        return {"arm": name, "stage": stage, "reason": "nonfinite", "event_ids": ids}
    return None


def _metadata(cache_evidence: dict[str, Any], ids: list[str]) -> dict[str, dict[str, str]]:
    result = {row["event_id"]: {"group": str(row["group"]), "label": str(row["label"])}
              for row in cache_evidence["records"] if row["horizon"] == "prefix"}
    if set(result) != set(ids):
        raise AssertionError("cache prefix metadata/event IDs mismatch")
    return result


def _metric_rows(name: str, prediction: np.ndarray, truth: list[np.ndarray], ids: list[str],
                 meta: dict[str, dict[str, str]], mass: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, common = [], []
    for index, (event_id, actual) in enumerate(zip(ids, truth, strict=True)):
        for horizon in (1, 2, 5, 8, "prefix"):
            metric = stageb.metric(prediction[index], actual, mass, horizon)
            if metric is not None:
                rows.append({"arm": name, "event_id": event_id, **meta[event_id], "horizon": str(horizon),
                             "scope": "available_prefix" if horizon == "prefix" else "complete",
                             "complete": horizon != "prefix", **metric})
            frequency = common_omega_metric(prediction[index], actual, mass, horizon)
            if frequency is not None:
                common.append({"arm": name, "event_id": event_id, **meta[event_id], "horizon": str(horizon),
                               "scope": "available_prefix" if horizon == "prefix" else "complete",
                               "complete": horizon != "prefix", **frequency})
    return rows, common


def _summary(rows: list[dict[str, Any]], arm: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    complete = {str(h): stageb.aggregate([row for row in rows if row["arm"] == arm and row["horizon"] == str(h)])
                for h in (1, 2, 5, 8)}
    prefix_rows = [row for row in rows if row["arm"] == arm and row["horizon"] == "prefix"]
    prefix = stageb.aggregate(prefix_rows)
    unstable = stageb.aggregate([row for row in prefix_rows if row["label"] == "unstable"])
    per_group = {group: {str(h): stageb.aggregate([row for row in rows if row["arm"] == arm and row["group"] == group and row["horizon"] == str(h)])
                         for h in (1, 2, 5, 8, "prefix")}
                 for group in sorted({row["group"] for row in rows if row["arm"] == arm})}
    return complete, prefix, unstable, per_group


def _common_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    summary = {str(h): aggregate_common([row for row in rows if row["arm"] == arm and row["horizon"] == str(h)])
               for h in (1, 2, 5, 8, "prefix")}
    summary["unstable_prefix"] = aggregate_common([row for row in rows if row["arm"] == arm and row["horizon"] == "prefix" and row["label"] == "unstable"])
    return summary


def _equilibrium(p: dict[str, np.ndarray], qeq: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    equilibrium_start = np.zeros((1, 33), dtype=float)
    equilibrium_start[:, 30:] = np.asarray(qeq, dtype=float).reshape(3)
    initial = initial_states(equilibrium_start, qeq, name)
    prediction = core.decoded_rollout(initial, 8.0, p, dt=.0025)[0]
    return prediction, _force(prediction, p)


def candidate_screen(complete: dict[str, Any], unstable: dict[str, Any], numerics: dict[str, Any],
                     equilibrium: dict[str, Any], invalid: list[dict[str, Any]]) -> dict[str, Any]:
    """Preregistered raw booleans only; invalid original/anchored artifacts block it."""
    relevant_invalid = [item for item in invalid if item["arm"] in ("original", "anchored")]
    if relevant_invalid or any(name not in complete or name not in numerics for name in ("original", "anchored")):
        return {"available": False, "reason": "invalid_or_missing_original_or_anchored", "invalid": relevant_invalid}
    if any(equilibrium[name][key] is None for name in ("anchored", "zero") for key in ("physical_max_abs", "force_max_abs")):
        return {"available": False, "reason": "invalid_equilibrium_reference"}
    def ratios(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
        return {key: float(candidate[key] / reference[key]) for key in ("mean", "pooled", "cvar10")}
    complete_ratio = ratios(complete["anchored"]["5"], complete["original"]["5"])
    unstable_ratio = ratios(unstable["anchored"], unstable["original"])
    numerical_valid = all(numerics[name]["scopes"]["5"]["complete"]["max"] < .001 and
                          numerics[name]["scopes"]["8"]["unstable_prefix"]["max"] < .001
                          for name in ("original", "anchored"))
    return {"available": True, "complete5_ratios": complete_ratio, "unstable_prefix_ratios": unstable_ratio,
            "equilibrium_anchored_max_le_1e-10": bool(max(equilibrium["anchored"]["physical_max_abs"], equilibrium["anchored"]["force_max_abs"]) <= 1e-10),
            "equilibrium_zero_max_le_1e-10": bool(max(equilibrium["zero"]["physical_max_abs"], equilibrium["zero"]["force_max_abs"]) <= 1e-10),
            "numerical_complete5_and_full_unstable_prefix_valid": bool(numerical_valid),
            "ratio_booleans": {"complete_mean": bool(complete_ratio["mean"] <= 1.02), "complete_pooled": bool(complete_ratio["pooled"] <= 1.02), "complete_cvar10": bool(complete_ratio["cvar10"] <= 1.05),
                               "unstable_mean": bool(unstable_ratio["mean"] <= 1.05), "unstable_pooled": bool(unstable_ratio["pooled"] <= 1.05), "unstable_cvar10": bool(unstable_ratio["cvar10"] <= 1.05)}}


def _source_and_baseline(ids: list[str], starts: np.ndarray, p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    hashes = stageb.source_contract(AUDIT, ids)
    if hashes["model"] != EXPECTED_MODEL_SHA:
        raise AssertionError("unexpected source model")
    baseline_evidence = json.loads((BASELINE / "evidence.json").read_text())
    if any(baseline_evidence["source_sha256"][key] != value for key, value in hashes.items()):
        raise AssertionError("0061 baseline source hash mismatch")
    data = np.load(BASELINE / "predictions.npz", allow_pickle=False)
    needed = ("event_ids", "full", "full_005", "full_00125")
    if any(key not in data for key in needed) or [str(value) for value in data["event_ids"]] != ids:
        raise AssertionError("0061 baseline event ID/array mismatch")
    arrays = {key: np.asarray(data[key]) for key in needed if key != "event_ids"}
    for key, value in arrays.items():
        if value.shape != (len(ids), 801, 33) or not np.isfinite(value).all():
            raise FloatingPointError(f"invalid saved baseline {key}")
        if not np.array_equal(value[:, 0], starts):
            raise AssertionError(f"saved baseline {key} initial-state mismatch")
    field_error, step_error = stageb.joint_equivalence(p, starts)
    if field_error >= 1e-10 or step_error >= 1e-10:
        raise AssertionError("frozen cached RHS mismatch")
    arrays["field_equivalence_max_abs"] = np.asarray(field_error)
    arrays["one_step_equivalence_max_abs"] = np.asarray(step_error)
    return arrays


def _safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(item) for item in value]
    if isinstance(value, np.ndarray): return _safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value): raise FloatingPointError("nonfinite evidence value")
        return float(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    return value


def run(model_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    if sha(model_dir / "model.pt") != EXPECTED_MODEL_SHA: raise AssertionError("model hash")
    torch.set_num_threads(1)
    matrices_file = np.load(AUDIT / "matrices.npz", allow_pickle=False)
    matrices = {key: matrices_file[key] for key in matrices_file.files}
    p = stageb.params(model_dir, matrices)
    cache = np.load(CACHE / "trajectories.npz", allow_pickle=False)
    ids = [str(value) for value in cache["event_ids"]]
    starts = np.asarray(cache["fine_predictions"][:, 0])
    truth = [np.asarray(cache[f"{event_id}_truth"]) for event_id in ids]
    cache_evidence = json.loads((CACHE / "evidence.json").read_text())
    meta = _metadata(cache_evidence, ids)
    if len(ids) != 107 or [sum(len(item) > round(h / DT) for item in truth) for h in (1, 2, 5, 8)] != [96, 95, 95, 95]:
        raise AssertionError("cached cohort counts")
    if sum(value["label"] == "unstable" for value in meta.values()) != 12: raise AssertionError("unstable prefix count")
    if starts.shape != (107, 33) or not np.isfinite(starts).all() or any(not np.isfinite(item).all() for item in truth):
        raise FloatingPointError("nonfinite cached starts/truth")
    baseline = _source_and_baseline(ids, starts, p)
    network = dict(np.load(NETWORK, allow_pickle=False)); network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]]); network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    frozen = _construct(model_dir, MechanicalROM.load(stageb.BASE), network, "challenger")
    frozen.eval()
    with torch.no_grad(): qeq = frozen.encode(torch.zeros(1, 11, 30, dtype=torch.float64)).cpu().numpy()[0]
    if not np.isfinite(qeq).all(): raise FloatingPointError("nonfinite equilibrium encoder state")
    initial = {name: initial_states(starts, qeq, name) for name in ("original", "anchored", "zero")}
    if not np.array_equal(initial["original"], starts): raise AssertionError("original initial-state mismatch")

    output.mkdir(parents=True)
    invalid: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {"original": {"primary": baseline["full"], "companion": baseline["full_005"], "finer": baseline["full_00125"]}}
    def save_arm(name: str) -> None:
        np.savez_compressed(output / f"{name}.npz", event_ids=np.asarray(ids), **predictions[name])
    # Preserve the primary baseline immediately, before any companion handling.
    save_arm("original")
    for name in ("anchored", "zero"):
        predictions[name] = {"primary": core.decoded_rollout(initial[name], 8.0, p, dt=.0025)}
        save_arm(name)
        status = _invalid(name, "primary_0025", predictions[name]["primary"], ids)
        if status is not None:
            invalid.append(status)
            continue
        predictions[name]["companion"] = core.decoded_rollout(initial[name], 8.0, p, dt=.005)
        save_arm(name)
        status = _invalid(name, "companion_005", predictions[name]["companion"], ids)
        if status is not None:
            invalid.append(status)
            continue
        scopes = {str(h): stageb.scope_values(predictions[name]["primary"], predictions[name]["companion"], truth, ids, meta, p["mass"], h) for h in (1, 2, 5, 8)}
        if any(scope["all_fixed"]["max"] >= .001 for scope in scopes.values()):
            predictions[name]["finer"] = core.decoded_rollout(initial[name], 8.0, p, dt=.00125)
            save_arm(name)
            status = _invalid(name, "finer_00125", predictions[name]["finer"], ids)
            if status is not None: invalid.append(status)
    for name, arrays in predictions.items():
        # Candidate arms have already been checked as each trajectory was saved.
        if name != "original": continue
        for stage, values in arrays.items():
            stage_name = {"primary": "primary_0025", "companion": "companion_005", "finer": "finer_00125"}[stage]
            status = _invalid(name, stage_name, values, ids)
            if status is not None: invalid.append(status)

    rows: list[dict[str, Any]] = []; common_rows: list[dict[str, Any]] = []; numerics: dict[str, Any] = {}
    complete: dict[str, Any] = {}; prefix: dict[str, Any] = {}; unstable: dict[str, Any] = {}; per_group: dict[str, Any] = {}; common: dict[str, Any] = {}
    for name, arrays in predictions.items():
        if any(item["arm"] == name and item["stage"] == "primary_0025" for item in invalid): continue
        primary_rows, primary_common = _metric_rows(name, arrays["primary"], truth, ids, meta, p["mass"])
        rows.extend(primary_rows); common_rows.extend(primary_common)
        complete[name], prefix[name], unstable[name], per_group[name] = _summary(rows, name)
        common[name] = _common_summary(common_rows, name)
        if not any(item["arm"] == name and item["stage"] == "companion_005" for item in invalid):
            record = {"scopes": {str(h): stageb.scope_values(arrays["primary"], arrays["companion"], truth, ids, meta, p["mass"], h) for h in (1, 2, 5, 8)}}
            if "finer" in arrays and not any(item["arm"] == name and item["stage"] == "finer_00125" for item in invalid):
                record["finer_00125"] = {"available": True, "scopes": {str(h): stageb.scope_values(arrays["primary"], arrays["finer"], truth, ids, meta, p["mass"], h) for h in (1, 2, 5, 8)}}
            numerics[name] = record

    equilibrium: dict[str, Any] = {}
    for name in ("original", "anchored", "zero"):
        prediction, force = _equilibrium(p, qeq, name)
        equilibrium_start = np.r_[np.zeros(30), qeq][None, :]
        companion = core.decoded_rollout(initial_states(equilibrium_start, qeq, name), 8.0, p, dt=.005)[0]
        companion_force = _force(companion, p)
        status = _invalid(name, "equilibrium_prediction", prediction, ["equilibrium"])
        if status is not None: invalid.append(status)
        status = _invalid(name, "equilibrium_force", force, ["equilibrium"])
        if status is not None: invalid.append(status)
        status = _invalid(name, "equilibrium_companion_005", companion, ["equilibrium"])
        if status is not None: invalid.append(status)
        status = _invalid(name, "equilibrium_companion_force", companion_force, ["equilibrium"])
        if status is not None: invalid.append(status)
        np.savez_compressed(output / f"equilibrium_{name}.npz", prediction=prediction, force=force, q=prediction[:, 30:],
                            companion_005=companion, companion_force_005=companion_force, companion_q_005=companion[:, 30:])
        finite = all(np.isfinite(array).all() for array in (prediction, force, companion, companion_force))
        equilibrium[name] = ({"physical_max_abs": float(np.abs(prediction[:, :30]).max()),
                              "force_max_abs": float(np.abs(force).max()), "q_max_abs": float(np.abs(prediction[:, 30:]).max()),
                              "companion_physical_max_abs": float(np.abs(companion[:, :30]).max()),
                              "normalized_step_rmse_8s": float(stageb.numerical(prediction[None], companion[None], p["mass"], 8)[0])}
                             if finite else {"physical_max_abs": None, "force_max_abs": None, "q_max_abs": None,
                                             "companion_physical_max_abs": None, "normalized_step_rmse_8s": None})
    screen = candidate_screen(complete, unstable, numerics, equilibrium, invalid)
    if sha(model_dir / "model.pt") != EXPECTED_MODEL_SHA: raise AssertionError("frozen model changed")
    evidence = {"test_read": False, "model_sha256": EXPECTED_MODEL_SHA, "normalization_sha256": sha(model_dir / "normalization.npz"),
                "source_sha256": {"model": sha(model_dir / "model.pt"), "normalization": sha(model_dir / "normalization.npz"), "base": sha(stageb.BASE), "network": sha(NETWORK), "cache": sha(CACHE / "trajectories.npz"), "baseline": sha(BASELINE / "predictions.npz"), "code": sha(__file__),
                                  "core_helper": sha(Path(core.__file__)), "stageb_helper": sha(Path(stageb.__file__)), "evaluator_helper": sha(Path(_construct.__code__.co_filename))},
                "qeq": qeq.tolist(), "event_count": len(ids), "event_ids": ids, "initial_physical_prefix_bitwise_unchanged": True,
                "baseline_contract": {"field_equivalence_max_abs": float(baseline["field_equivalence_max_abs"]), "one_step_equivalence_max_abs": float(baseline["one_step_equivalence_max_abs"])},
                "source_params": {"Cq": p["Cq"].tolist()},
                "invalid": invalid, "complete": complete, "prefix": prefix, "unstable_prefix": unstable, "per_group": per_group,
                "common_synchronous_omega": {"units": "rad/s", "exclude_initial": True, "summaries": common},
                "rows": rows, "common_omega_rows": common_rows, "numerical": numerics, "equilibrium": equilibrium,
                "candidate_screen_raw_booleans_not_scientific_decision": screen}
    (output / "evidence.json").write_text(json.dumps(_safe(evidence), indent=2, allow_nan=False), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.model_dir, args.output)


if __name__ == "__main__":
    main()
