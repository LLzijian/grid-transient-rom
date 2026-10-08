"""Loop0066: fixed paired group31 deployment-loss sensitivity, no TEST."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]
from kiki.models.private_pm import MechanicalROM
from train_joint_latent import read_development, tensor
from train_long_latent import campaign_loss
from evaluate_long_latent import NETWORK, _construct, _safe
import audit_latent_order_rollout as core
import audit_long_latent_rollout as stageb
from audit_coupled_equilibrium import analytic_jacobian, gauge_maps

SOURCE = ROOT / "private_data/task2_long_0060_5s"
ADEFAULT = ROOT / "private_data/task2_group31_all_0066"
BDEFAULT = ROOT / "private_data/task2_group31_omit_0066"
BASE = ROOT / "private_data/task2_m3_0020/model.npz"
MATRICES = ROOT / "private_data/task2_long_sparse_audit_0061/matrices.npz"
EXPECTED_SOURCE = "d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c"
DT = .01


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def midpoint_samples(events):
    return [(e, (10 + len(e["x"]) - 2) // 2) for e in events]


def panel_loss(model, events):
    samples = midpoint_samples(events)
    histories = tensor(np.stack([e["x"][i-10:i+1] for e, i in samples]))
    with torch.no_grad():
        total, terms = campaign_loss(model, samples, histories, 5.)
    values = {"total": float(total), **{k: float(v) for k, v in terms.items()}}
    return {**values, "finite": all(np.isfinite(v) for v in values.values()),
            "event_count": len(events), "event_ids": [e["id"] for e in events],
            "endpoints": [i for _, i in samples]}


def aggregate(rows):
    return stageb.aggregate(rows)


def safe_aggregate(rows):
    return aggregate(rows) if rows else {"n_events": 0, "mean": None, "pooled": None, "cvar10": None}


def score(pred, truth, mass, h=5, *, complete=False):
    requested = round(h / DT)
    if complete and len(truth)-1 < requested:
        return None
    k = min(requested, len(truth)-1)
    if k <= 0:
        return None
    if len(pred) <= k:
        raise ValueError("forecast shorter than requested genuine prefix")
    e = pred[1:k+1, :20] - truth[1:k+1, :20]
    d, w = stageb.coi(e[:, :10], mass), stageb.coi(e[:, 10:20], mass)
    ds, ws = float((d*d).sum()), float((w*w).sum())
    ss = .5 * (ds / stageb.SCALES[0]**2 + ws / stageb.SCALES[1]**2)
    return {"n": k*10, "delta_sse": ds, "omega_sse": ws,
            "normalized_sse": float(ss), "rmse": float(np.sqrt(ss/(k*10)))}


def common(pred, truth, mass, h):
    k = min(round(h / DT), len(truth)-1)
    if k <= 0:
        return None
    e = (pred[1:k+1, 10:20] - truth[1:k+1, 10:20]) @ mass / mass.sum()
    sse = float(e @ e)
    return {"n": k, "sse": sse, "rmse": float(np.sqrt(sse/k))}


def frozen_check(source, arm):
    s = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    a = torch.load(arm / "model.pt", map_location="cpu", weights_only=True)
    if s.keys() != a.keys():
        raise AssertionError("state keys")
    for k in s:
        if k != "xi" and not torch.equal(s[k], a[k]):
            raise AssertionError(f"frozen tensor {k}")
    inactive = s["support"] == 0
    if not torch.equal(a["xi"][inactive], s["xi"][inactive]):
        raise AssertionError("inactive Xi")
    if sha(source / "normalization.npz") != sha(arm / "normalization.npz"):
        raise AssertionError("normalization")
    return s, a


def live_equivalence(model, p, initial):
    x = initial[:3]
    with torch.no_grad():
        actual = model.field(tensor(x)).numpy()
        step = model.rk4(tensor(x), .005).numpy()
    return {"field_max_abs": float(np.abs(actual-core.rhs(x, p)).max()),
            "one_rk4_max_abs": float(np.abs(step-core.rk4(x, .005, lambda y: core.rhs(y, p))).max())}


def registered_classification(numerical_valid, optimization_adequate, ratios, disagreement):
    if not numerical_valid:
        return "numerical_unresolved"
    if not optimization_adequate:
        return "optimization_unresolved"
    return "supported" if all(v <= 1.10 for v in ratios.values()) and disagreement <= .10 else "rejected"


def scored_rows(selected, group, truths, mass):
    rows, commonrows = [], []
    for name, prediction in selected.items():
        for i, (event, truth) in enumerate(zip(group, truths, strict=True)):
            meta = {"arm": name, "event_id": event["id"],
                    "group": str(event["group"]), "label": event["label"]}
            for h in (1, 2, 5, 8):
                r = score(prediction[i], truth, mass, h, complete=True)
                if r is not None:
                    rows.append({**meta, "scope": "complete", "horizon": str(h), **r})
            for h, label in ((5, "capped5_prefix"), (8, "capped8_prefix")):
                rows.append({**meta, "scope": label, "horizon": label, **score(prediction[i], truth, mass, h)})
                commonrows.append({**meta, "scope": label, "horizon": label, **common(prediction[i], truth, mass, h)})
    return rows, commonrows


def run(source, arm_a, arm_b, output):
    if output.exists():
        raise FileExistsError(output)
    directories = {"S": source, "A": arm_a, "B": arm_b}
    paths = {name: directory / "model.pt" for name, directory in directories.items()}
    paths.update({f"{name}_normalization": directory / "normalization.npz" for name, directory in directories.items()})
    paths.update(base=BASE, network=NETWORK, matrices=MATRICES)
    before = {k: sha(v) for k, v in paths.items()}
    if before["S"] != EXPECTED_SOURCE:
        raise AssertionError("frozen source model")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False))
    network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]])
    network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    dev, excluded, identities = read_development(base, network, full_train=True)
    events = dev["train"]
    group = [e for e in events if str(e["group"]) == "31"]
    complement = [e for e in events if str(e["group"]) != "31"]
    if (len(events), len(group), len(complement)) != (346, 13, 333):
        raise AssertionError("TRAIN/group31 identity counts")
    p = stageb.params(source, dict(np.load(MATRICES, allow_pickle=False)))
    models = {n: _construct(d, base, network, "challenger") for n, d in directories.items()}
    states = {n: frozen_check(source, d)[1] for n, d in directories.items()}
    arm_p = {n: {**p, "xi": s["xi"].numpy()} for n, s in states.items()}
    panels = {n: {key: panel_loss(model, es) for key, es in
                 {"all": events, "complement31": complement, "group31": group}.items()}
              for n, model in models.items()}
    (output / "panels.json").write_text(json.dumps(_safe(panels), indent=2, allow_nan=False))
    mass = p["mass"]
    truths = [np.asarray(e["x"])[10:] for e in group]
    ids = [e["id"] for e in group]
    meta = {e["id"]: {"group": str(e["group"]), "label": e["label"]} for e in group}
    arrays = {"event_ids": np.asarray(ids), "mass": mass, "metric_scales": stageb.SCALES,
              "support": p["support"], **{f"{eid}_truth": t for eid, t in zip(ids, truths, strict=True)}}
    arrays.update({f"param_{k}": v for k, v in p.items()})
    selected, selected_grid, numerical, equivalence, finite = {}, {}, {}, {}, {}
    for name, model in models.items():
        with torch.no_grad():
            initial = np.c_[np.stack([e["x"][10] for e in group]),
                            model.encode(tensor(np.stack([e["x"][:11] for e in group]))).numpy()]
        arrays[f"{name}_initial"] = initial
        if name != "S" and not np.array_equal(initial, arrays["S_initial"]):
            raise AssertionError("frozen causal initial state mismatch")
        arrays[f"{name}_Xi"] = arm_p[name]["xi"]
        equivalence[name] = live_equivalence(model, arm_p[name], initial)
        if not all(np.isfinite(v) and v <= 1e-10 for v in equivalence[name].values()):
            np.savez_compressed(output / "predictions_partial.npz", **arrays)
            (output / "failure_evidence.json").write_text(json.dumps({"arm": name, "reason": "live_numpy_mismatch", "equivalence": equivalence[name]}))
            raise AssertionError("live NumPy equation mismatch")
        fine = core.decoded_rollout(initial, 8, arm_p[name], dt=.0025)
        arrays[f"{name}_fine"] = fine
        coarse = core.decoded_rollout(initial, 8, arm_p[name], dt=.005)
        arrays[f"{name}_coarse"] = coarse
        np.savez_compressed(output / "predictions_partial.npz", **arrays)
        numerical[name] = {str(h): stageb.scope_values(fine, coarse, truths, ids, meta, mass, h) for h in (1, 2, 5, 8)}
        discrepancy = numerical[name]["5"]["available_prefix"]["max"]
        selected[name], selected_grid[name] = fine, ".0025"
        primary_pair_finite = all(np.isfinite(a[i, :min(500, len(t)-1)+1]).all()
                                  for a in (fine, coarse) for i, t in enumerate(truths))
        if not primary_pair_finite or not np.isfinite(discrepancy) or discrepancy > 1e-3:
            finer = core.decoded_rollout(initial, 8, arm_p[name], dt=.00125)
            arrays[f"{name}_finer"] = finer
            numerical[name]["finer_00125"] = {str(h): stageb.scope_values(fine, finer, truths, ids, meta, mass, h) for h in (1, 2, 5, 8)}
            selected[name], selected_grid[name] = finer, ".00125"
        comparison = fine if selected_grid[name] == ".00125" else coarse
        finite[name] = {"all_fixed8": bool(np.isfinite(selected[name]).all()),
                        "primary_prefix5": all(np.isfinite(selected[name][i, :min(500, len(t)-1)+1]).all() for i, t in enumerate(truths)),
                        "comparison_prefix5": all(np.isfinite(comparison[i, :min(500, len(t)-1)+1]).all() for i, t in enumerate(truths))}
        np.savez_compressed(output / "predictions_partial.npz", **arrays)
    rows, commonrows = scored_rows(selected, group, truths, mass)
    scopes = ("1", "2", "5", "8", "capped5_prefix", "capped8_prefix")
    summary = {n: {h: safe_aggregate([r for r in rows if r["arm"] == n and r["horizon"] == h]) for h in scopes} for n in models}
    labels = {n: {label: {h: safe_aggregate([r for r in rows if r["arm"] == n and r["label"] == label and r["horizon"] == h])
                         for h in scopes[-2:]} for label in ("stable", "unstable")} for n in models}
    disrows = [{"event_id": ids[i], **score(selected["B"][i]-selected["A"][i], np.zeros_like(t), mass, 5)} for i, t in enumerate(truths)]
    disagreement = aggregate(disrows)
    ratios = {k: summary["B"]["capped5_prefix"][k] / max(summary["A"]["capped5_prefix"][k], 1e-8) for k in ("mean", "pooled", "cvar10")}
    disratio = disagreement["pooled"] / max(summary["A"]["capped5_prefix"]["pooled"], 1e-8)
    selected_num = {n: numerical[n].get("finer_00125", numerical[n])["5"]["available_prefix"]["max"] for n in models}
    numerical_valid = all(finite[n]["primary_prefix5"] and finite[n]["comparison_prefix5"] and np.isfinite(v) and v <= 1e-3 for n, v in selected_num.items())
    adequate = all(panel["finite"] for arm in panels.values() for panel in arm.values()) and (
        panels["A"]["all"]["total"] <= panels["S"]["all"]["total"]*(1+1e-6) and
        panels["B"]["complement31"]["total"] <= panels["S"]["complement31"]["total"]*(1+1e-6))
    project, embed = gauge_maps()
    j32 = {n: project @ analytic_jacobian(arm_p[n]) @ embed for n in models}
    arrays.update({f"J32_{n}": j for n, j in j32.items()})
    np.savez_compressed(output / "group31_predictions.npz", **arrays)
    after = {k: sha(v) for k, v in paths.items()}
    if before != after:
        raise AssertionError("source artifacts changed during audit")
    poles = lambda matrix: np.c_[np.linalg.eigvals(matrix).real, np.linalg.eigvals(matrix).imag]
    report = {"test_read": False, "source_sha256": {**before, "code": sha(__file__)},
              "source_hashes_before": before, "source_hashes_after": after,
              "train_counts": {"all": len(events), "complement31": len(complement), "group31": len(group)},
              "excluded": excluded, "identities": identities, "panels": panels,
              "rows": rows, "common_omega_rows": commonrows, "summary": summary,
              "stable_unstable": labels, "numerical": numerical, "selected_grid": selected_grid,
              "finite": finite, "equivalence": equivalence, "ratios_B_over_A": ratios,
              "trajectory_disagreement": {**disagreement, "ratio_to_A": disratio, "rows": disrows},
              "adequacy": {"numerical_valid": numerical_valid, "optimization_adequate": adequate},
              "classification": registered_classification(numerical_valid, adequate, ratios, disratio),
              "numeric_thresholds": {"step_max": .001, "ratio": 1.10, "disagreement": .10},
              "J32_poles": {n: poles(j) for n, j in j32.items()},
              "Aq_poles": {n: poles((arm_p[n]["xi"]*p["support"])[1:4].T) for n in models},
              "Xi": {**{n: s["xi"].numpy() for n, s in states.items()}, "support": p["support"]}}
    (output / "evidence.json").write_text(json.dumps(_safe(report), indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--arm-a", type=Path, default=ADEFAULT)
    parser.add_argument("--arm-b", type=Path, default=BDEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.arm_a, args.arm_b, args.output)


if __name__ == "__main__":
    main()
