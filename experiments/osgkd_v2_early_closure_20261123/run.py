"""Read-only early momentum/port closure probe on frozen P1a saved states."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
import torch
from kiki.models.osgkd import OSGKD

RUN = "osgkd_v2_early_closure_20261123"
PAIR = "results/osgkd_v2_p1a_20261123/pair01/summary.json"
ANALYSIS = "results/osgkd_v2_p1a_20261123/analysis.json"
AUDIT = "results/osgkd_fit297_20261113/port_audit_release.json"
SOURCES = {f"experiments/{RUN}/run.py", "campaigns/osgkd_v2/early_closure_protocol.md",
           "src/kiki/models/osgkd.py", "src/kiki/data/osgkd.py"}
WINDOWS = {"fixed_0.03": 3, "fixed_0.10": 10, "capped_0.25": 25}
TERMS = ("E", "A_m", "A_e", "A_d", "R_pred", "R_obs")


def require(value, message):
    if not value:
        raise ValueError(message)


def repo_path(name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "relative repository path required")
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT) and path.relative_to(ROOT).as_posix() == name, "canonical path required")
    return path


def identity(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = stream.tell()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": size, "sha256": digest}


def verified(ref):
    require(set(ref) == {"path", "bytes", "sha256"}, "typed file reference required")
    path = repo_path(ref["path"])
    require(identity(path) == ref, "file identity differs: " + ref["path"])
    return path


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return identity(path)


def read(ref):
    return json.loads(verified(ref).read_bytes())


def validate(release):
    require((release.get("run_id"), release.get("job_id"), release.get("released")) ==
            (RUN, "probe01", True), "fixed released probe01 required")
    require(SOURCES <= release.get("source_hashes", {}).keys(), "required source hashes missing")
    for name, digest in release["source_hashes"].items():
        require(identity(repo_path(name))["sha256"] == digest, "source changed: " + name)
    for key, name in (("audit_ref", AUDIT), ("parent_pair_ref", PAIR), ("parent_analysis_ref", ANALYSIS)):
        require(release[key]["path"] == name, "noncanonical parent: " + key)
        verified(release[key])
    parent, analysis = read(release["parent_pair_ref"]), read(release["parent_analysis_ref"])
    require(parent["status"] == "completed" and parent["result"]["fixed_endpoint"] == 1700,
            "completed fixed P1a endpoint required")
    require(parent["release"]["audit_ref"] == release["audit_ref"], "parent audit binding differs")
    require(analysis["endpoint_receipt"] == release["parent_pair_ref"], "analysis parent binding differs")
    require(set(release["models"]) == {"long", "short"}, "exact long/short pair required")
    for name, treatment in (("long", "8s"), ("short", "2s")):
        refs = release["models"][name]
        require(set(refs) == {"checkpoint_ref", "predictions_ref", "rows_ref"}, "exact model references required")
        base = f"artifacts/osgkd_v2_p1a_20261123/pair01/{treatment}"
        for key, suffix in (("checkpoint_ref", "/attempt_1700.pt"),
                            ("predictions_ref", "/evaluation/predictions.npz"), ("rows_ref", "/evaluation/rows.json")):
            require(refs[key]["path"] == base + suffix, "noncanonical P1a file")
            verified(refs[key])
        require(refs["checkpoint_ref"] == parent["result"]["treatments"][treatment]["checkpoint"], "checkpoint parent mismatch")
        bound = {r["path"]: r for r in analysis["artifacts"]}
        require(all(bound.get(ref["path"]) == ref for ref in refs.values()), "analysis file binding differs")
    return parent


def decomposition(obs, pred, constants, mask, steps):
    """One identical all-time node mask is supplied for every term and model."""
    h = .01 * steps
    scale = h * np.maximum(constants["S_m"], constants["Se"])
    mass, damping = np.asarray(constants["mass"]), np.asarray(constants["damping"])
    q = lambda x: .01 * (x[0] / 2 + x[-1] / 2 + x[1:-1].sum(axis=0))
    jo = mass * (obs["omega"][-1] - obs["omega"][0])
    jp = mass * (pred["omega"][-1] - pred["omega"][0])
    out = {"E": (jp-jo)/scale,
           "A_m": q(pred["tau_m"]-obs["tau_m"])/scale,
           "A_e": -q(pred["tau_e"]-obs["tau_e"])/scale,
           "A_d": -q(damping*(pred["omega"]-obs["omega"]))/scale,
           "R_pred": (jp-q(pred["tau_m"]-pred["tau_e"]-damping*pred["omega"]))/scale,
           "R_obs": (jo-q(obs["tau_m"]-obs["tau_e"]-damping*obs["omega"]))/scale}
    rhs = out["A_m"] + out["A_e"] + out["A_d"] + out["R_pred"] - out["R_obs"]
    tolerance = 256*np.finfo(np.float64).eps*np.maximum(1., sum(abs(v) for v in out.values()))
    require(np.all(abs(out["E"]-rhs)[mask] <= tolerance[mask]), "momentum component identity failed")
    selected = {key: value[mask] for key, value in out.items()}
    n = int(mask.sum())
    row = {"nodes": np.flatnonzero(mask).tolist(), "node_count": n, "steps": steps, "h_s": h,
           "terms": {key: value.tolist() for key, value in selected.items()},
           "squared_means": {key: float(np.mean(value**2)) if n else None for key, value in selected.items()},
           "signed_projections": {key: float(np.mean(selected["E"]*value))*(-1 if key == "R_obs" else 1) if n else None
                                  for key, value in selected.items()},
           "identity_max_abs_error": float(np.max(abs(out["E"]-rhs)[mask])) if n else None}
    row["ports"] = {}
    for key, scale_name in (("tau_m", "S_m"), ("tau_e", "Se")):
        difference = ((pred[key]-obs[key])/constants[scale_name])[:, mask]
        mse = float(np.mean(difference**2)) if n else None
        row["ports"][key] = {"normalized_mse": mse, "normalized_rmse": np.sqrt(mse).item() if n else None,
                             "point_node_count": (steps+1)*n}
    weights = mass/mass.sum()
    error = float((pred["omega"][-1]-obs["omega"][-1]) @ weights) if n == len(mass) else None
    row["common_speed_endpoint_error_rad_s"] = error
    row["common_speed_endpoint_squared_error"] = error**2 if error is not None else None
    return row


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    require(np.isfinite(values).all(), "nonfinite aggregate")
    q25, med, q75 = np.quantile(values, [.25, .5, .75])
    return {"count": len(values), "mean": float(values.mean()), "median": float(med), "q25": float(q25), "q75": float(q75)}


def aggregate(rows, indices, window, model, mode, *, complete_only=False):
    selected = [rows[i]["windows"][window] for i in indices]
    valid = [x["models"][model][mode] for x in selected
             if x.get("node_count", 0) and (not complete_only or x["node_count"] == 10)]
    result = {"event_denominator": len(indices), "node_denominator": 10*len(indices),
              "covered_events": len(valid), "covered_nodes": sum(x["node_count"] for x in valid),
              "fully_covered_events": sum(x.get("node_count") == 10 for x in selected),
              "full_0.25_time_events": sum(rows[i]["full_0.25_time_coverage"] for i in indices),
              "full_0.25_coverage_events": sum(rows[i]["full_0.25_time_coverage"] and
                  rows[i]["windows"]["capped_0.25"].get("node_count") == 10 for i in indices),
              "h_s": distribution([x["h_s"] for x in valid]),
              "squared_means": {key: distribution([x["squared_means"][key] for x in valid]) for key in TERMS},
              "signed_projections": {key: distribution([x["signed_projections"][key] for x in valid]) for key in TERMS}}
    v = result["squared_means"]["E"]["mean"]
    result["signed_contributions"] = {key: result["signed_projections"][key]["mean"]/v if v is not None and v > 0 else None for key in TERMS if key != "E"}
    result["signed_contribution_sum"] = sum(result["signed_contributions"].values()) if v is not None and v > 0 else None
    result["ports"] = {key: distribution([x["ports"][key]["normalized_mse"] for x in valid]) for key in ("tau_m", "tau_e")}
    for item in result["ports"].values():
        item["normalized_rmse"] = float(np.sqrt(item["mean"])) if item["mean"] is not None else None
    speeds = [x["common_speed_endpoint_squared_error"] for x in valid if x["node_count"] == 10]
    result["common_speed_endpoint"] = {"event_count": len(speeds), "rmse_rad_s": float(np.sqrt(np.mean(speeds))) if speeds else None}
    return result


def summarize(rows, groups):
    result = {}
    for name, indices in groups.items():
        result[name] = {}
        for window in WINDOWS:
            item = {model: {mode: aggregate(rows, indices, window, model, mode) for mode in ("main", "fine")}
                    for model in ("long", "short")}
            item["paired_long_minus_short"] = {}
            for mode in ("main", "fine"):
                differences = []
                port_differences = {key: [] for key in ("tau_m", "tau_e")}
                nodes = 0
                for i in indices:
                    w = rows[i]["windows"][window]
                    if w.get("node_count", 0):
                        differences.append(w["models"]["long"][mode]["squared_means"]["E"]-
                                           w["models"]["short"][mode]["squared_means"]["E"])
                        nodes += w["node_count"]
                        for key in port_differences:
                            port_differences[key].append(w["models"]["long"][mode]["ports"][key]["normalized_mse"]-
                                                         w["models"]["short"][mode]["ports"][key]["normalized_mse"])
                item["paired_long_minus_short"][mode] = {**distribution(differences), "event_denominator": len(indices),
                    "node_denominator": 10*len(indices), "matched_nodes": nodes,
                    "better": sum(d < 0 for d in differences), "worse": sum(d > 0 for d in differences),
                    "same": sum(d == 0 for d in differences),
                    "port_normalized_mse": {key: {**distribution(values), "better": sum(d < 0 for d in values),
                        "worse": sum(d > 0 for d in values), "same": sum(d == 0 for d in values)} for key, values in port_differences.items()}}
            result[name][window] = item
    return result


def gate(rows, groups):
    checks, candidates = {}, []
    for window in ("fixed_0.03", "capped_0.25"):
        values = {mode: aggregate(rows, groups["long_misses"], window, "long", mode, complete_only=True)
                  for mode in ("main", "fine")}
        v = [values[mode]["squared_means"]["E"]["mean"] for mode in ("main", "fine")]
        deviation = abs(v[0]-v[1])/v[1] if all(x is not None and x > 1e-12 for x in v) else None
        checks[window] = {"main_fine_V_relative_deviation": deviation, "modes": {}}
        for mode in ("main", "fine"):
            a = values[mode]; means = a["squared_means"]; vv = means["E"]["mean"]
            eligible = a["fully_covered_events"] >= 22 and vv is not None and vv > 1e-12
            rp = np.sqrt(means["R_pred"]["mean"]/vv).item() if eligible else None
            ro = np.sqrt(means["R_obs"]["mean"]/vv).item() if eligible else None
            checks[window]["modes"][mode] = {"coverage_pass": a["fully_covered_events"] >= 22,
                "V": vv, "R_pred_to_E_rms": rp, "R_obs_to_E_rms": ro,
                "passed": bool(eligible and rp <= .10 and ro <= .20 and deviation is not None and deviation <= .05),
                "signed_contributions": a["signed_contributions"]}
    for candidate, other in (("A_m", "A_e"), ("A_e", "A_m")):
        if all(item["passed"] and item["signed_contributions"][candidate] >= .60 and
               item["signed_contributions"][candidate] >= 1.5*max(item["signed_contributions"][other], 0.)
               for check in checks.values() for item in check["modes"].values()):
            candidates.append(candidate)
    return {"checks": checks, "selected": candidates[0] if len(candidates) == 1 else None,
            "interpretation": "signed momentum/port error accounting; no causal or internal/electrical attribution"}


def evaluate(events, constants, models, saved, work):
    rows = []
    for i, event in enumerate(events):
        maximum = min(25, len(event["t"])-1)
        require(np.allclose(event["t"][:maximum+1], np.arange(maximum+1)*.01, atol=1e-12, rtol=0), "nonuniform original endpoints")
        predictions = {}
        for name, model in models.items():
            predictions[name] = {}
            for mode in ("main", "fine"):
                state = np.asarray(saved[name][f"event_{i:04d}_{mode}"][:maximum+1], dtype=np.float64)
                require(state.shape == (maximum+1, 50), "saved state prefix shape differs")
                endpoint = np.r_[event["history"]["theta"][-1], event["history"]["omega"][-1]]
                require(np.allclose(state[0, :20], endpoint, atol=1e-12, rtol=0), "motion endpoint differs")
                with torch.no_grad():
                    ports = model.ports(torch.as_tensor(state, dtype=torch.float64))
                work["port_state_points"] += len(state)
                work["port_calls"] += 1
                predictions[name][mode] = {"state": state, "omega": state[:, 10:20],
                                          "tau_m": ports.tau_m.numpy(), "tau_e": ports.tau_e.numpy()}
        row = {"event_id": event["id"], "label": event["label"], "true_horizon_s": float(event["t"][-1]),
               "full_0.25_time_coverage": maximum == 25, "windows": {}}
        for window, requested in WINDOWS.items():
            steps = min(requested, maximum) if window == "capped_0.25" else requested
            if steps < 1 or steps > maximum:
                row["windows"][window] = {"node_count": 0, "reason": "original_time_coverage", "full_0.25_coverage": False}
                continue
            obs = {key: np.asarray(event["targets"][key][:steps+1]) for key in ("omega", "tau_m", "tau_e")}
            mask = np.ones(10, dtype=bool)
            mask &= (np.asarray(event["masks"]["theta"][:steps+1], dtype=bool) &
                     np.isfinite(event["targets"]["theta"][:steps+1])).all(axis=0)
            for key, value in obs.items():
                mask &= np.all(np.asarray(event["masks"][key][:steps+1], dtype=bool) & np.isfinite(value), axis=0)
            for prediction in predictions.values():
                for p in prediction.values():
                    mask &= np.isfinite(p["state"][:steps+1].reshape(steps+1, 5, 10)).all(axis=(0, 1))
                    for key in obs:
                        mask &= np.isfinite(p[key][:steps+1]).all(axis=0)
            row["windows"][window] = {"node_count": int(mask.sum()), "full_0.25_coverage": maximum == 25,
                "models": {name: {mode: decomposition(obs, {k: p[k][:steps+1] for k in obs}, constants, mask, steps)
                                  for mode, p in prediction.items()} for name, prediction in predictions.items()}}
        rows.append(row)
    return rows


def execute(path):
    start, cpu = perf_counter(), process_time()
    path = path.resolve()
    require(path.is_relative_to(ROOT), "release must be inside repository")
    release = json.loads(path.read_bytes()); validate(release)
    local, public = [ROOT/side/RUN/"probe01" for side in ("artifacts", "results")]
    require(not local.exists() and not public.exists(), "exclusive probe already claimed")
    local.mkdir(parents=True, exist_ok=False)
    receipt = {"run_id": RUN, "job_id": "probe01", "release": release, "release_ref": identity(path),
               "status": "started", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": [sys.executable, *sys.argv]}
    write(local/"started.json", receipt)
    try:
        public.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
        from kiki.data.osgkd import load_fit297
        events, constants = load_fit297(release["audit_ref"])
        require(len(events) == 297 and sum(e["label"] == "stable" for e in events) == 267, "fit297 denominator differs")
        require(np.array_equal(constants["damping"], np.zeros(10)), "audited D=0 required")
        for key in ("mass", "S_m", "Se", "Sp", "omega_scale"):
            require(np.asarray(constants[key]).shape == (10,) and np.all(np.isfinite(constants[key])) and
                    np.all(np.asarray(constants[key]) > 0), "ten positive frozen constants required: " + key)
        models, score_rows = {}, {}
        for name, refs in release["models"].items():
            checkpoint = torch.load(verified(refs["checkpoint_ref"]), map_location="cpu", weights_only=True)
            require(checkpoint["attempt"] == 1700 and checkpoint["model"]["config"]["variant"] == "C1" and
                    checkpoint["provenance"]["audit_ref"] == release["audit_ref"], "fixed audited P1a C1 required")
            model = OSGKD.from_checkpoint(checkpoint["model"]).cpu().double().eval().requires_grad_(False)
            require(model.n == 10 and torch.count_nonzero(model.damping).item() == 0, "ten-node model D=0 required")
            for key in ("mass", "damping", "Sp", "Se", "omega_scale", "theta_ref"):
                actual = getattr(model, key).numpy(); expected = np.asarray(constants[key])
                if key in ("mass", "damping"):
                    expected = np.diag(expected)
                require(np.array_equal(actual, expected), "frozen model/bundle constant differs: " + key)
            models[name] = model
            score_rows[name] = read(refs["rows_ref"])
            require([r["event_id"] for r in score_rows[name]] == [e["id"] for e in events], "frozen event order differs")
        unstable = [i for i, e in enumerate(events) if e["label"] == "unstable"]
        hits = [i for i in unstable if score_rows["long"][i]["classification_actual"]["truth_supported"] and
                score_rows["long"][i]["classification_actual"]["classification"]["predicted_los"]]
        misses = [i for i in unstable if score_rows["long"][i]["classification_actual"]["truth_supported"] and
                  not score_rows["long"][i]["classification_actual"]["classification"]["predicted_los"]]
        require(len(misses) == 27 and len(hits) == 3, "frozen long miss/hit denominator differs")
        groups = {"train_all": list(range(297)), "stable": [i for i,e in enumerate(events) if e["label"] == "stable"],
                  "long_misses": misses, "long_hits": hits, "unstable": unstable,
                  **{f"train_{side}": [i for i,e in enumerate(events) if e["id"] in constants[f"{side}_ids"]] for side in ("fit", "check")}}
        work = Counter()
        from contextlib import ExitStack
        with ExitStack() as stack:
            saved = {name: stack.enter_context(np.load(verified(refs["predictions_ref"]), allow_pickle=False))
                     for name, refs in release["models"].items()}
            rows = evaluate(events, constants, models, saved, work)
        rows_ref = write(local/"rows.json", rows)
        summary = summarize(rows, groups)
        receipt.update(status="completed", result={"groups": summary, "gate": gate(rows, groups), "rows_ref": rows_ref,
            "group_event_ids": {name: [events[i]["id"] for i in indices] for name, indices in groups.items()},
            "resource_counts": {**dict(work), "parameter_updates": 0, "rhs_evaluations": 0, "integrations": 0,
                                "encoder_calls": 0, "new_fom": 0, "maximum_saved_points_per_event_path": 26},
            "cost": {"wall_s": perf_counter()-start, "cpu_s": process_time()-cpu}},
            environment={"torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu", "dtype": "float64", "threads": 1})
        summary_ref = write(public/"summary.json", receipt)
        write(local/"completed.json", {"status": "completed", "summary_ref": summary_ref})
        return 0
    except BaseException as error:
        write(local/"failed.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
                                   "wall_s": perf_counter()-start, "cpu_s": process_time()-cpu})
        raise


def self_check():
    constants = {key: np.ones(2) for key in ("mass", "S_m", "Se")}
    constants["damping"] = np.zeros(2)
    t = np.arange(4)*.01
    obs = {"omega": np.zeros((4,2)), "tau_m": np.zeros((4,2)), "tau_e": np.zeros((4,2))}
    pred = {"omega": np.column_stack((t, -t)), "tau_m": np.tile([2., -2.], (4,1)), "tau_e": np.tile([1., -1.], (4,1))}
    row = decomposition(obs, pred, constants, np.ones(2, bool), 3)
    assert np.allclose(row["terms"]["E"], [1,-1])
    assert np.allclose(row["terms"]["A_m"], [2,-2]) and np.allclose(row["terms"]["A_e"], [-1,1])
    assert row["signed_projections"]["A_m"] == 2 and row["signed_projections"]["A_e"] == -1
    assert row["squared_means"]["A_d"] == 0 and row["squared_means"]["R_pred"] < 1e-28
    pred["omega"] = pred["omega"]*2
    residual = decomposition(obs, pred, constants, np.ones(2, bool), 3)
    assert np.allclose(residual["terms"]["R_pred"], [1,-1])
    obs["tau_m"] = np.ones((4,2))
    observed_residual = decomposition(obs, pred, constants, np.ones(2, bool), 3)
    assert np.allclose(observed_residual["terms"]["R_obs"], [-1,-1])
    pred["tau_e"][1,0] = np.nan
    mask = np.isfinite(pred["tau_e"]).all(axis=0)
    missing = decomposition(obs, pred, constants, mask, 3)
    assert missing["nodes"] == [1] and missing["common_speed_endpoint_error_rad_s"] is None
    short = decomposition({k:v[:2] for k,v in obs.items()}, {k:v[:2] for k,v in pred.items()}, constants, mask, 1)
    assert short["h_s"] == .01 and short["steps"] == 1
    assert distribution([])["mean"] is None
    from types import SimpleNamespace
    class SyntheticPorts:
        def ports(self, state):
            return SimpleNamespace(tau_m=state[:, 20:30], tau_e=state[:, 30:40])
    c = {key: np.ones(10) for key in ("mass", "S_m", "Se")}
    c["damping"] = np.zeros(10)
    state = np.zeros((4, 50)); state[:, 10:20] = t[:, None]; state[:, 20:30] = 1.
    event = {"id": "synthetic", "label": "unstable", "t": t,
             "history": {key: np.zeros((11,10)) for key in ("theta", "omega")},
             "targets": {key: np.zeros((4,10)) for key in ("theta", "omega", "tau_m", "tau_e")},
             "masks": {key: np.ones((4,10), bool) for key in ("theta", "omega", "tau_m", "tau_e")}}
    saved = {name: {f"event_0000_{mode}": state.copy() for mode in ("main", "fine")} for name in ("long", "short")}
    models = {name: SyntheticPorts() for name in saved}
    work = Counter(); rows = evaluate([event], c, models, saved, work)
    assert work["port_state_points"] == 16 and rows[0]["windows"]["fixed_0.10"]["node_count"] == 0
    assert not rows[0]["full_0.25_time_coverage"]
    groups = {"long_misses": list(range(27))}
    repeated = rows*27
    summary = summarize(repeated, groups)
    assert summary["long_misses"]["fixed_0.03"]["paired_long_minus_short"]["main"]["same"] == 27
    assert gate(repeated, groups)["selected"] == "A_m"
    assert abs(summary["long_misses"]["fixed_0.03"]["long"]["main"]["signed_contribution_sum"]-1) < 1e-14
    event["masks"]["theta"][1, 0] = False
    missing_rows = evaluate([event], c, models, saved, Counter())
    assert missing_rows[0]["windows"]["fixed_0.03"]["node_count"] == 9
    assert gate(missing_rows*27, groups)["selected"] is None
    print("synthetic self-check passed: signs, cancellation, both residuals, missing interior, short window, joint support, paired aggregation and gate; no private assets read")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--release", type=Path)
    group.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        raise SystemExit(execute(args.release))
