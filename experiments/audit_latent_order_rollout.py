"""Loop0059 frozen, no-refit latent-order rollout audit.

The normal invocation is validation-only and writes one new immutable directory.
``--synthetic`` is deliberately self-contained: it does not read private data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DT, FINE_DT = 0.01, 0.005
SCALES = np.array([0.47675314588683515, 1.0033667249602924])
BASE = Path("private_data/task2_m3_0020/model.npz")
FROZEN = Path("private_data/task2_joint_0056")
NETWORK = Path("private_data/task2_electrical_ports_0048/physical_network.npz")
ROOTS = (Path("private_data/ieee39_postfault"), Path("private_data/ieee39_expansion"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coi(x: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return x - np.sum(x * mass, axis=-1, keepdims=True) / np.sum(mass)


def state_dict_numpy(path: Path) -> dict[str, np.ndarray]:
    """Load tensor-only weights; this never instantiates a trainer/model."""
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("frozen model.pt is not a state dictionary")
    return {key: np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value,
                            dtype=float) for key, value in raw.items()}


def load_contract(directory: Path) -> dict[str, np.ndarray]:
    path = directory / "matrices.npz"
    if not path.exists():
        raise FileNotFoundError(f"expected algebra contract: {path}")
    with np.load(path, allow_pickle=False) as z:
        out = {key: np.asarray(z[key]) for key in z.files}
    for key in ("Cq", "C_unit", "balanced_V_T1", "balanced_W_T1"):
        if key not in out:
            raise KeyError(f"matrix contract missing {key}")
    return out


def frozen_parameters(contract: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Use saved 0056 buffers and verify the physical Cq contract exactly."""
    state = state_dict_numpy(FROZEN / "model.pt")
    with np.load(BASE, allow_pickle=False) as base, np.load(FROZEN / "normalization.npz", allow_pickle=False) as norm, np.load(NETWORK, allow_pickle=False) as network:
        p = {key: np.asarray(base[key], dtype=float) for key in
             ("mass", "incidence", "delta_star", "coupling", "damping", "rate", "gain", "pm_star")}
        p["feature_scale"] = np.asarray(norm["feature_scale"], dtype=float)
        ports = np.asarray(network["port_incidence"], dtype=float)
        port_ids = np.asarray(norm["port_ids"], dtype=int) if "port_ids" in norm.files else np.arange(ports.shape[1])
    for key in ("xi", "support", "alpha", "alpha_support", "c", "amplitude", "angle_incidence"):
        if key not in state:
            raise KeyError(f"frozen state dictionary missing {key}")
        p[key] = state[key]
    for key in ("mass", "incidence", "delta_star", "coupling", "damping", "rate", "gain"):
        if key not in state or not np.allclose(state[key], p[key], atol=1e-12, rtol=1e-12):
            raise AssertionError(f"frozen {key} buffer disagrees with M3 base artifact")
    if "feature_scale" not in state or not np.allclose(state["feature_scale"], p["feature_scale"], atol=1e-12, rtol=1e-12):
        raise AssertionError("frozen feature_scale buffer disagrees with normalization artifact")
    active_incidence = p["incidence"][:, p["coupling"] > 0]
    if not np.allclose(p["angle_incidence"], active_incidence, atol=1e-12, rtol=1e-12):
        raise AssertionError("frozen angle_incidence buffer disagrees with active M3 lines")
    if "ports" not in state or not np.allclose(state["ports"], ports[:, port_ids], atol=1e-12, rtol=1e-12):
        raise AssertionError("frozen port buffer disagrees with saved physical network")
    c_norm = p["c"] / np.linalg.norm(p["c"], axis=0, keepdims=True)
    expected_cunit = ports[:, port_ids] @ c_norm
    expected_cq = expected_cunit * p["amplitude"][None, :]
    if not np.allclose(contract["C_unit"], expected_cunit, atol=1e-10, rtol=1e-10):
        raise AssertionError("contract C_unit disagrees with saved ports and normalized c")
    if not np.allclose(contract["Cq"], expected_cq, atol=1e-10, rtol=1e-10):
        raise AssertionError("contract Cq disagrees with saved ports, c, and amplitude")
    p["Cq"] = np.asarray(contract["Cq"], dtype=float)
    if p["Cq"].shape != (10, 3) or p["xi"].shape[1] != 3:
        raise ValueError("0056 rollout requires ten outputs and exactly three latent coordinates")
    return p


def angle_features(delta: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    phase = p["delta_star"] @ p["angle_incidence"]
    eta = delta @ p["angle_incidence"] + phase
    raw = np.concatenate((np.sin(eta) - np.sin(phase), np.cos(eta) - np.cos(phase)), axis=-1)
    return raw / p["feature_scale"][-raw.shape[-1]:]


def rhs(y: np.ndarray, p: dict[str, np.ndarray], *, electrical: bool = True,
        memory: bool = True, memory_mask: np.ndarray | None = None,
        V: np.ndarray | None = None, W: np.ndarray | None = None) -> np.ndarray:
    """Batched frozen field. V/W define q=Vz, zdot=W f(x,Vz)."""
    d, w, pm, z = y[..., :10], y[..., 10:20], y[..., 20:30], y[..., 30:]
    q = z @ V.T if V is not None else z
    af = angle_features(d, p)
    a = af @ (p["alpha"] * p["alpha_support"])
    mask = np.ones(3) if memory_mask is None else np.asarray(memory_mask, dtype=float)
    latent_action = q * mask if memory else np.zeros_like(q)
    action = (latent_action + a) @ p["Cq"].T
    phase = p["delta_star"] @ p["incidence"]
    e0 = ((np.sin(d @ p["incidence"] + phase) - np.sin(phase)) * p["coupling"]) @ p["incidence"].T
    if not electrical:
        action = np.zeros_like(action)
    features = np.concatenate((np.ones(q.shape[:-1] + (1,)), q,
                               w / p["feature_scale"][:10], pm / p["feature_scale"][10:20], af), axis=-1)
    qdot_full = features @ (p["xi"] * p["support"])
    qdot = qdot_full @ W.T if W is not None else qdot_full
    return np.concatenate((w, (pm - e0 - p["damping"] * w - action) / p["mass"],
                           -p["rate"] * pm - p["gain"] * w, qdot), axis=-1)


def rk4(y: np.ndarray, dt: float, fun) -> np.ndarray:
    k1 = fun(y)
    k2 = fun(y + dt * k1 / 2)
    k3 = fun(y + dt * k2 / 2)
    k4 = fun(y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rollout(initial: np.ndarray, seconds: float, p: dict[str, np.ndarray], *,
            dt: float = FINE_DT, **kwargs) -> np.ndarray:
    """Return [event,time,state], retaining every 10-ms point for comparison."""
    steps, stride = round(seconds / dt), round(DT / dt)
    if abs(stride * dt - DT) > 1e-12:
        raise ValueError("rollout dt must divide 10 ms")
    y = np.asarray(initial, dtype=float).copy()
    if y.ndim == 1:
        y = y[None, :]
    rows = [y.copy()]
    fun = lambda x: rhs(x, p, **kwargs)
    for step in range(steps):
        y = rk4(y, dt, fun)
        if (step + 1) % stride == 0:
            rows.append(y.copy())
    return np.stack(rows, axis=1)


def decoded_rollout(starts: np.ndarray, seconds: float, p: dict[str, np.ndarray], *,
                    V: np.ndarray | None = None, W: np.ndarray | None = None,
                    **kwargs) -> np.ndarray:
    initial = starts if V is None else np.concatenate((starts[:, :30], starts[:, 30:] @ W.T), axis=1)
    values = rollout(initial, seconds, p, V=V, W=W, **kwargs)
    return values if V is None else np.concatenate((values[..., :30], values[..., 30:] @ V.T), axis=-1)


def _error_metric(pred: np.ndarray, truth: np.ndarray, mass: np.ndarray) -> dict[str, float | int]:
    err = pred[1:, :20] - truth[1:, :20]
    de, we = coi(err[:, :10], mass), coi(err[:, 10:20], mass)
    sse = .5 * np.sum(de ** 2 / SCALES[0] ** 2 + we ** 2 / SCALES[1] ** 2)
    n = int((len(pred) - 1) * 10)
    return {"n": n, "normalized_sse": float(sse), "rmse": float(np.sqrt(sse / n))}


def event_metrics(pred: np.ndarray, truth: np.ndarray, mass: np.ndarray) -> list[dict]:
    """Fixed horizons are complete-only; the available truth prefix is separate."""
    out = []
    for horizon in (1, 2, 5, 8):
        points = round(horizon / DT)
        if len(truth) - 1 >= points:
            out.append({"horizon": str(horizon), "scope": "complete", "complete": True,
                        **_error_metric(pred[:points + 1], truth[:points + 1], mass)})
        else:
            out.append({"horizon": str(horizon), "scope": "complete", "complete": False,
                        "n": 0, "available": False})
    points = min(len(pred), len(truth)) - 1
    if points > 0:
        out.append({"horizon": "prefix", "scope": "available_prefix", "complete": False,
                    **_error_metric(pred[:points + 1], truth[:points + 1], mass)})
    return out


def trajectory_discrepancy(coarse: np.ndarray, fine: np.ndarray, mass: np.ndarray, horizon: int) -> np.ndarray:
    """Eventwise COI-normalized RMSE over the complete prefix 0..h."""
    points = round(horizon / DT)
    err = coarse[:, :points + 1, :20] - fine[:, :points + 1, :20]
    de, we = coi(err[..., :10], mass), coi(err[..., 10:20], mass)
    sse = .5 * np.sum(de ** 2 / SCALES[0] ** 2 + we ** 2 / SCALES[1] ** 2, axis=(1, 2))
    return np.sqrt(sse / ((points + 1) * 10))


def validation_truth(event_ids: list[str], base) -> list[np.ndarray]:
    """Read only split-listed validation IDs, never HDF5 TEST payloads."""
    from kiki.data.private_dataset import canonical_arrays

    wanted, found = set(event_ids), {}
    for root in ROOTS:
        splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
        selected = [str(eid) for eid in splits["validation"] if str(eid) in wanted]
        if not selected:
            continue
        with __import__("h5py").File(root / "dataset.h5", "r") as h5:
            eq = {key: h5["equilibrium"][key][:] for key in ("delta", "omega", "Pm", "Pe")}
            for eid in selected:
                raw = {key: h5["events"][eid][key][:811] for key in ("t", "delta", "omega", "Pm", "Pe")}
                event = canonical_arrays(raw, eq, h5["inertia"][:], h5.attrs["omega_s"])
                found[eid] = np.column_stack((event["delta_sync"], event["omega_sync"], event["Pm_raw"] - base.pm_star))[10:]
    if set(found) != wanted:
        raise RuntimeError(f"validation IDs missing from explicit split payload: {sorted(wanted - set(found))[:5]}")
    return [found[eid] for eid in event_ids]


def evaluation_metadata(event_ids: list[str]) -> tuple[dict[str, dict], dict]:
    evidence = json.loads((FROZEN / "evidence.json").read_text(encoding="utf-8"))
    info: dict[str, dict] = {}
    for row in evidence["records"]:
        eid = str(row["event_id"])
        current = {"group": str(row["group"]), "label": str(row["label"])}
        if eid in info and info[eid] != current:
            raise ValueError(f"inconsistent saved evaluation metadata for {eid}")
        info[eid] = current
    if set(info) != set(event_ids):
        raise ValueError("saved 0056 evaluation records do not exactly cover rollout event IDs")
    return info, evidence


def aggregate(rows: list[dict]) -> dict:
    values = np.asarray([row["rmse"] for row in rows], dtype=float)
    total_sse, total_n = sum(row["normalized_sse"] for row in rows), sum(row["n"] for row in rows)
    tail = max(1, int(np.ceil(.1 * len(values))))
    groups = {}
    for group in sorted({row["group"] for row in rows}):
        group_values = np.asarray([row["rmse"] for row in rows if row["group"] == group], dtype=float)
        groups[group] = {"count": len(group_values), "mean": float(group_values.mean()), "worst": float(group_values.max())}
    return {"count": len(rows), "mean": float(values.mean()), "pooled": float(np.sqrt(total_sse / total_n)),
            "cvar10": float(np.mean(np.sort(values)[-tail:])), "worst": float(values.max()), "groups": groups}


def add_metrics(rows: list[dict], arm: str, preds: np.ndarray, truths: list[np.ndarray], info: dict[str, dict], mass: np.ndarray, ids: list[str]) -> None:
    for eid, pred, truth in zip(ids, preds, truths, strict=True):
        for metric in event_metrics(pred, truth, mass):
            rows.append({"arm": arm, "event_id": eid, **info[eid], **metric})


def verify_first_second(starts: np.ndarray, stored: np.ndarray, truths: list[np.ndarray], p: dict[str, np.ndarray], mass: np.ndarray, ids: list[str], evidence: dict) -> np.ndarray:
    full = decoded_rollout(starts, 1, p)
    replica = float(np.max(np.abs(full - stored[:, :101])))
    if replica >= 1e-8:
        raise AssertionError(f"0056 frozen first-second replication failed: maxabs={replica:.3g}")
    saved = {str(row["event_id"]): row for row in evidence["records"] if str(row["horizon"]) == "1.0"}
    actual = []
    for eid, pred, truth in zip(ids, full, truths, strict=True):
        if len(truth) - 1 >= 100:
            row = _error_metric(pred, truth[:101], mass)
            reference = saved[eid]
            if abs(row["normalized_sse"] - float(reference["normalized_sse"])) >= 1e-10:
                raise AssertionError(f"0056 saved one-second metric mismatch for {eid}")
            actual.append(row)
    if len(actual) != 96 or len(saved) != 96:
        raise AssertionError(f"expected 96 complete frozen one-second events, found {len(actual)}/{len(saved)}")
    ref = evidence["summary"]["1.0"]
    summary = aggregate([{**row, "group": "all"} for row in actual])
    for key in ("mean", "pooled", "cvar10"):
        if abs(summary[key] - float(ref[key])) >= 1e-10:
            raise AssertionError(f"0056 saved one-second {key} mismatch")
    return full


def modal_arms(contract: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Remove declared real modal blocks, retaining only their fixed complement."""
    required = ("modal_V", "modal_W", "modal_eigenvalues")
    if all(key in contract for key in required):
        V, W, eig = contract["modal_V"], contract["modal_W"], contract["modal_eigenvalues"]
        if (not np.iscomplexobj(V) and not np.iscomplexobj(W) and not np.iscomplexobj(eig)
                and V.shape == (3, 3) and W.shape == (3, 3)
                and np.allclose(W @ V, np.eye(3), atol=1e-10)
                and min(abs(float(eig[i]) - float(eig[j])) for i in range(3) for j in range(i)) > 1e-8):
            # Preserve the pre-existing simple-real arm names exactly.
            return [(f"modal_off_{i + 1}", V[:, np.arange(3) != i], W[np.arange(3) != i]) for i in range(3)]

    required = ("modal_real_block_V", "modal_real_block_W", "modal_block_column_groups", "modal_block_kinds")
    if not all(key in contract for key in required):
        return []
    V, W = contract["modal_real_block_V"], contract["modal_real_block_W"]
    groups = np.asarray(contract["modal_block_column_groups"], dtype=int)
    kinds = np.asarray(contract["modal_block_kinds"]).astype(str)
    if (np.iscomplexobj(V) or np.iscomplexobj(W) or V.shape != (3, 3) or W.shape != (3, 3)
            or groups.ndim != 2 or groups.shape[1:] != (2,) or len(groups) != len(kinds)
            or not np.allclose(W @ V, np.eye(3), atol=1e-10)):
        return []
    covered = np.concatenate([np.arange(start, stop) for start, stop in groups]) if len(groups) else np.array([], dtype=int)
    if not np.array_equal(covered, np.arange(3)) or any(stop <= start for start, stop in groups):
        return []
    arms = []
    for block_index, ((start, stop), kind) in enumerate(zip(groups, kinds, strict=True), start=1):
        keep = np.ones(3, dtype=bool)
        keep[start:stop] = False
        if not np.any(keep):
            continue
        # A singleton simple block retains the historic coordinate-derived name.
        name = f"modal_off_{start + 1}" if kind == "simple_real" and stop == start + 1 else f"modal_block_{block_index}_off"
        arms.append((name, V[:, keep], W[keep]))
    return arms


def stop_for_nonfinite(output: Path, arrays: dict[str, np.ndarray], ids: list[str], arm: str) -> None:
    """Persist computed arrays and a strict-JSON validity failure; never impute/drop."""
    value = arrays[arm]
    invalid = {"arm": arm, "nonfinite_count": int(np.size(value) - np.isfinite(value).sum()),
               "shape": list(value.shape), "action": "preserved_and_stopped_without_clipping_or_event_removal"}
    saved = {**arrays, "event_ids": np.asarray(ids)}
    np.savez_compressed(output / "predictions.npz", **saved)
    report = {"test_read": False, "all_predictions_finite": False, "validity_failure": invalid,
              "replication_first_second_max_abs": None, "primary_rank2_1s": None,
              "complete_horizon_summaries": None, "available_prefix_summaries": None,
              "unstable_prefix_summaries": None, "rows": None}
    (output / "per_event.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    raise FloatingPointError(f"non-finite frozen rollout in {arm}; preserved invalid result and stopped")


def execute(audit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output}")
    contract = load_contract(audit)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_started.json").write_text(json.dumps({"test_read": False, "status": "started",
        "source_sha256": {"model": sha(FROZEN / "model.pt"), "base": sha(BASE), "network": sha(NETWORK), "matrices": sha(audit / "matrices.npz"), "code": sha(Path(__file__))}}, indent=2), encoding="utf-8")
    with np.load(FROZEN / "trajectories.npz", allow_pickle=False) as saved:
        ids = [str(x) for x in saved["event_ids"]]
        starts, stored = np.asarray(saved["predictions"][:, 0], dtype=float), np.asarray(saved["predictions"], dtype=float)
    from kiki.models.private_pm import MechanicalROM
    base = MechanicalROM.load(BASE)
    truths, p = validation_truth(ids, base), frozen_parameters(contract)
    info, evidence = evaluation_metadata(ids)
    if starts.shape != (107, 33) or stored.shape[1] != 101:
        raise ValueError(f"expected frozen 107x101x33 prediction evidence, got {starts.shape}/{stored.shape}")
    first_second = verify_first_second(starts, stored, truths, p, base.mass, ids, evidence)
    V, W = contract["balanced_V_T1"][:, :2], contract["balanced_W_T1"][:2]
    if V.shape != (3, 2) or W.shape != (2, 3) or not np.allclose(W @ V, np.eye(2), atol=1e-10):
        raise ValueError("balanced rank-2 contract must satisfy W@V=I")
    arrays, rows = {}, []
    arms: list[tuple[str, dict]] = [("full", {}), ("memory_off", {"memory": False})]
    arms += [(f"force_coordinate_{i + 1}_off", {"memory_mask": np.eye(3)[np.arange(3) != i].sum(axis=0)}) for i in range(3)]
    for name, kwargs in arms:
        arrays[name] = decoded_rollout(starts, 8, p, **kwargs)
        if not np.isfinite(arrays[name]).all():
            stop_for_nonfinite(output, arrays, ids, name)
        add_metrics(rows, name, arrays[name], truths, info, base.mass, ids)
    arrays["balanced_rank2"] = decoded_rollout(starts, 8, p, V=V, W=W)
    if not np.isfinite(arrays["balanced_rank2"]).all():
        stop_for_nonfinite(output, arrays, ids, "balanced_rank2")
    add_metrics(rows, "balanced_rank2", arrays["balanced_rank2"], truths, info, base.mass, ids)
    prescribed_modal_arms = modal_arms(contract)
    modal_status = {"executed_arms": [arm[0] for arm in prescribed_modal_arms],
                    "scope": "whole invariant real modal blocks; no arbitrary split of repeated or conjugate poles"}
    for name, mv, mw in prescribed_modal_arms:
        arrays[name] = decoded_rollout(starts, 8, p, V=mv, W=mw)
        if not np.isfinite(arrays[name]).all():
            stop_for_nonfinite(output, arrays, ids, name)
        add_metrics(rows, name, arrays[name], truths, info, base.mass, ids)
    T = np.array([[2., .4, -.2], [0., .5, .3], [.1, 0., 1.5]])
    similar = decoded_rollout(starts, 1, p, V=np.linalg.inv(T), W=T)
    similarity_error = float(np.max(np.abs(similar - first_second)))
    if similarity_error >= 1e-8:
        raise AssertionError(f"rank-3 similarity physical trajectory mismatch: {similarity_error:.3g}")
    for name, kwargs in (("full", {}), ("balanced_rank2", {"V": V, "W": W})):
        arrays[f"{name}_coarse"] = decoded_rollout(starts, 8, p, dt=DT, **kwargs)
        if not np.isfinite(arrays[f"{name}_coarse"]).all():
            stop_for_nonfinite(output, arrays, ids, f"{name}_coarse")
    numerical = {}
    for arm in ("full", "balanced_rank2"):
        numerical[arm] = {}
        for horizon in (1, 2, 5, 8):
            values = trajectory_discrepancy(arrays[f"{arm}_coarse"], arrays[arm], base.mass, horizon)
            numerical[arm][str(horizon)] = {"eventwise_rmse": values.tolist(), "mean": float(values.mean()), "worst": float(values.max())}
        if numerical[arm]["1"]["worst"] >= .001:
            raise AssertionError(f"{arm} RK4 .01/.005 one-second discrepancy exceeds .001")
    arrays["event_ids"] = np.asarray(ids)
    names = sorted({row["arm"] for row in rows})
    complete = {arm: {h: aggregate([row for row in rows if row["arm"] == arm and row["horizon"] == h and row["complete"]]) for h in ("1", "2", "5", "8") if any(row["arm"] == arm and row["horizon"] == h and row["complete"] for row in rows)} for arm in names}
    prefixes = {arm: aggregate([row for row in rows if row["arm"] == arm and row["horizon"] == "prefix"]) for arm in names}
    unstable = {arm: aggregate([row for row in rows if row["arm"] == arm and row["horizon"] == "prefix" and row["label"] == "unstable"]) for arm in names}
    full1, rank21 = complete["full"]["1"], complete["balanced_rank2"]["1"]
    primary = {"full": full1, "balanced_rank2": rank21, "mean_ratio": rank21["mean"] / full1["mean"], "pooled_ratio": rank21["pooled"] / full1["pooled"], "cvar10_ratio": rank21["cvar10"] / full1["cvar10"], "thresholds": {"mean_max": 1.02, "pooled_max": 1.02, "cvar10_max": 1.05}}
    finite = bool(all(np.isfinite(value).all() for key, value in arrays.items() if key != "event_ids"))
    np.savez_compressed(output / "predictions.npz", **arrays)
    report = {"test_read": False, "replication_first_second_max_abs": float(np.max(np.abs(first_second - stored))), "similarity_physical_all107_max_abs": similarity_error, "primary_rank2_1s": primary, "complete_horizon_summaries": complete, "available_prefix_summaries": prefixes, "unstable_prefix_summaries": unstable, "numerical_coarse_vs_fine": numerical, "modal_reductions": modal_status, "rows": rows, "all_predictions_finite": finite, "source_sha256": {"model": sha(FROZEN / "model.pt"), "base": sha(BASE), "network": sha(NETWORK), "matrices": sha(audit / "matrices.npz"), "code": sha(Path(__file__))}}
    (output / "per_event.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def synthetic() -> None:
    """Independent fixtures for RHS, time alignment, projection, and RK4 checks."""
    rng = np.random.default_rng(59)
    incidence = rng.normal(size=(10, 45))
    angle_incidence = rng.normal(size=(10, 26))
    p = {"mass": np.arange(1., 11.), "incidence": incidence, "angle_incidence": angle_incidence, "delta_star": rng.normal(size=10) * .02, "coupling": rng.uniform(.2, 1., size=45), "damping": np.linspace(.1, .3, 10), "rate": np.linspace(.2, .4, 10), "gain": np.linspace(.3, .5, 10), "feature_scale": np.r_[np.linspace(1.3, 2.2, 10), np.linspace(2.4, 3.3, 10), np.linspace(3.5, 5.1, 52)], "xi": np.zeros((76, 3)), "support": np.ones((76, 3)), "alpha": rng.normal(size=(52, 3)) * .02, "alpha_support": np.ones((52, 3)), "Cq": rng.normal(size=(10, 3)) * .3}
    p["xi"][0] = [.2, -.1, .3]
    p["xi"][1:4] = np.diag([-.3, -.5, -.7])
    p["xi"][4:] = rng.normal(size=(72, 3)) * .01
    initial = rng.normal(size=33) * .03
    field = rhs(initial, p)
    af = angle_features(initial[None, :10], p)[0]
    features = np.r_[1., initial[30:], initial[10:20] / p["feature_scale"][:10], initial[20:30] / p["feature_scale"][10:20], af]
    expected_qdot = features @ p["xi"]
    action = (initial[30:] + af @ p["alpha"]) @ p["Cq"].T
    phase = p["delta_star"] @ p["incidence"]
    e0 = ((np.sin(initial[:10] @ p["incidence"] + phase) - np.sin(phase)) * p["coupling"]) @ p["incidence"].T
    expected_wdot = (initial[20:30] - e0 - p["damping"] * initial[10:20] - action) / p["mass"]
    assert np.max(abs(field[30:] - expected_qdot)) < 1e-13 and np.max(abs(field[10:20] - expected_wdot)) < 1e-13
    T = np.array([[2., .4, -.2], [0., .5, .3], [.1, 0., 1.5]])
    direct = decoded_rollout(initial[None, :], 1, p)
    changed = decoded_rollout(initial[None, :], 1, p, V=np.linalg.inv(T), W=T)
    assert np.max(abs(direct - changed)) < 1e-10
    V = np.array([[1., .2], [0., 1.], [.1, -.3]])
    W = np.linalg.inv(V.T @ V) @ V.T
    assert np.max(abs(W @ V - np.eye(2))) < 1e-12
    block_v = np.array([[1., .2, -.1], [.1, 1., .3], [.2, -.3, 1.]])
    block_w = np.linalg.inv(block_v)
    for kinds in (np.array(["conjugate_block", "simple_real"]), np.array(["repeated_block", "simple_real"])):
        block_contract = {"modal_real_block_V": block_v, "modal_real_block_W": block_w,
                          "modal_block_column_groups": np.array([[0, 2], [2, 3]]), "modal_block_kinds": kinds}
        arms = modal_arms(block_contract)
        assert [name for name, _, _ in arms] == ["modal_block_1_off", "modal_off_3"]
        assert all(np.max(abs(mw @ mv - np.eye(mv.shape[1]))) < 1e-12 for _, mv, mw in arms)
    raw = np.zeros((61, 33)); truth = raw[10:]; pred = np.zeros((801, 33)); pred[1] = truth[1]
    checked = event_metrics(pred, truth, p["mass"])
    assert not next(row for row in checked if row["horizon"] == "1")["complete"]
    prefix = next(row for row in checked if row["horizon"] == "prefix")
    assert prefix["n"] == 500 and prefix["normalized_sse"] == 0.
    coarse = decoded_rollout(initial[None, :], 1, p, dt=DT)
    assert np.isfinite(trajectory_discrepancy(coarse, direct, p["mass"], 1)).all()
    print(json.dumps({"synthetic": True, "rhs_qdot_max_abs": float(np.max(abs(field[30:] - expected_qdot))), "rank3_similarity_max_abs": float(np.max(abs(direct - changed))), "rank2_wv_max_abs": float(np.max(abs(W @ V - np.eye(2))))}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        synthetic()
        return 0
    if args.audit is None or args.output is None:
        parser.error("--audit and --output are required unless --synthetic is used")
    execute(args.audit, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
