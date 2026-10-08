"""Loop0064 frozen physical-force span and encoder-closure audit."""
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
try:
    from audit_group_identifiability import _encode_event, event_weights, window_average, window_ranges
except ModuleNotFoundError:  # Importable from repository-root synthetic tests.
    from experiments.audit_group_identifiability import _encode_event, event_weights, window_average, window_ranges
from evaluate_long_latent import BASE, NETWORK, _construct
from kiki.models.private_pm import MechanicalROM
from train_joint_latent import read_development, tensor


MODEL_SHA = "d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c"
SOURCE = Path("private_data/task2_long_0060_5s")
WINDOWS = Path("private_data/task2_group_identifiability_0062")


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_force(F: np.ndarray, a: np.ndarray, qenc: np.ndarray, Cq: np.ndarray) -> dict[str, np.ndarray]:
    """Exact fixed-Cq force decomposition at arbitrary endpoint rows."""
    pinv = np.linalg.pinv(Cq, rcond=1e-12)
    singular = np.linalg.svd(Cq, compute_uv=False)
    if np.count_nonzero(singular > 1e-12 * singular[0]) != 3:
        raise AssertionError("Cq must have rank three")
    qF = (F - a @ Cq.T) @ pinv.T
    Fperp = F - (qF + a) @ Cq.T
    parallel = (qF - qenc) @ Cq.T
    total = F - (qenc + a) @ Cq.T
    return {"qF": qF, "Fperp": Fperp, "e_parallel": parallel, "e_total": total,
            "P": Cq @ pinv}


def event_metrics(F: np.ndarray, Fperp: np.ndarray, parallel: np.ndarray, total: np.ndarray,
                  mass: np.ndarray) -> dict[str, float]:
    """Endpoint/generator mean-square force and common-acceleration errors."""
    values = {"F_sq": float(np.mean(F * F)), "Fperp_sq": float(np.mean(Fperp * Fperp)),
              "e_parallel_sq": float(np.mean(parallel * parallel)), "e_total_sq": float(np.mean(total * total))}
    for name, value in (("Fperp", Fperp), ("e_parallel", parallel), ("e_total", total)):
        values[f"{name}_net_force_rmse"] = float(np.sqrt(np.mean(np.sum(value, axis=1) ** 2)))
        acceleration = value.sum(axis=1) / mass.sum()
        values[f"{name}_common_accel_rmse"] = float(np.sqrt(np.mean(acceleration * acceleration)))
    return values


def window_integrals(qF: np.ndarray, endpoints: np.ndarray, ranges: list[tuple[int, int]], dt: float = .01) -> dict[str, np.ndarray]:
    """Exact qF averages and integrated derivatives on saved 0062 window mappings."""
    location = {int(endpoint): row for row, endpoint in enumerate(endpoints)}
    means, derivatives, starts, ends, durations = [], [], [], [], []
    for start, end in ranges:
        rows = np.asarray([location[index] for index in range(start, end + 1)], dtype=int)
        duration = (end - start) * dt
        means.append(np.trapezoid(qF[rows], dx=dt, axis=0) / duration)
        derivatives.append((qF[rows[-1]] - qF[rows[0]]) / duration)
        starts.append(start); ends.append(end); durations.append(duration)
    return {"qF_mean": np.asarray(means), "yF": np.asarray(derivatives), "start": np.asarray(starts),
            "end": np.asarray(ends), "duration": np.asarray(durations)}


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError("nonfinite or mismatched verification arrays")
    return float(np.abs(actual - expected).max() / max(1.0, float(np.linalg.norm(actual)), float(np.linalg.norm(expected))))


def _aggregate(stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats: return {"n_events": 0}
    keys = ("F_sq", "Fperp_sq", "e_parallel_sq", "e_total_sq", "Fperp_net_force_rmse", "e_parallel_net_force_rmse", "e_total_net_force_rmse", "Fperp_common_accel_rmse", "e_parallel_common_accel_rmse", "e_total_common_accel_rmse")
    result = {"n_events": len(stats), **{key: float(np.mean([row[key] for row in stats])) for key in keys}}
    for key in ("F_sq", "Fperp_sq", "e_parallel_sq", "e_total_sq"):
        result[key.replace("_sq", "_rms")] = float(np.sqrt(result[key]))
    for key in ("Fperp_common_accel_rmse", "e_parallel_common_accel_rmse", "e_total_common_accel_rmse"):
        result[key + "_event_balanced_rms"] = float(np.sqrt(np.mean([row[key] ** 2 for row in stats])))
    for key in ("Fperp_net_force_rmse", "e_parallel_net_force_rmse", "e_total_net_force_rmse"):
        result[key + "_event_balanced_rms"] = float(np.sqrt(np.mean([row[key] ** 2 for row in stats])))
    result["net_force_error_rms"] = result["e_total_net_force_rmse_event_balanced_rms"]
    return result


def _closure_summary(defect: np.ndarray, event_ids: np.ndarray, groups: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    def rms(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any(): return {"n_events": 0, "n_windows": 0, "rms_force_per_s": None}
        selected_ids, selected = event_ids[mask], defect[mask]
        per_event = [float(np.mean(selected[selected_ids == event] ** 2)) for event in np.unique(selected_ids)]
        return {"n_events": int(len(np.unique(event_ids[mask]))), "n_windows": int(mask.sum()),
                "rms_force_per_s": float(np.sqrt(np.mean(per_event)))}
    return {"all": rms(np.ones(len(event_ids), dtype=bool)),
            "stable": rms(labels == "stable"), "unstable": rms(labels == "unstable"),
            "groups": {str(group): rms(groups == group) for group in sorted(set(groups.astype(str)))}}


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


def _preserve_failure(output: Path, event_id: str, stage: str, arrays: dict[str, np.ndarray]) -> None:
    """Persist the offending real arrays before invalidating an audit run."""
    np.savez_compressed(output / f"invalid_{event_id}_{stage}.npz", **arrays)
    (output / "invalid.json").write_text(json.dumps({"event_id": event_id, "stage": stage}, indent=2), encoding="utf-8")


def run(model_dir: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError(output)
    if digest(model_dir / "model.pt") != MODEL_SHA: raise AssertionError("unexpected frozen model")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = np.load(WINDOWS / "windows.npz", allow_pickle=False)
    cached = {key: np.asarray(saved[key]) for key in saved.files}
    prior = json.loads((WINDOWS / "evidence.json").read_text())
    expected_hashes = {"model": MODEL_SHA, "normalization": digest(model_dir / "normalization.npz"),
                       "base": digest(BASE), "network": digest(NETWORK)}
    if any(prior["source_sha256"][key] != value for key, value in expected_hashes.items()):
        raise AssertionError("0062 source hash mismatch")
    base = MechanicalROM.load(BASE)
    network = dict(np.load(NETWORK, allow_pickle=False)); network["observed_buses"] = np.unique(np.r_[network["load_buses"], network["generator_buses"]]); network["load_positions"] = np.searchsorted(network["observed_buses"], network["load_buses"])
    model = _construct(model_dir, base, network); model.eval()
    development, excluded, identities = read_development(base, network, full_train=True)
    if identities != prior["identities"]:
        raise AssertionError("0062 dataset/split identity mismatch")
    events = development["train"]
    if len(events) != 346 or len({event["group"] for event in events}) != 24 or model.rank != 3:
        raise AssertionError("expected 346 TRAIN events, 24 groups, r=3")
    if not np.array_equal(cached["Cq"], (model.ports @ (model.c / torch.linalg.vector_norm(model.c, dim=0, keepdim=True)) * model.amplitude).detach().numpy()): raise AssertionError("saved Cq mismatch")
    source_xi = (model.xi * model.support).detach().numpy()
    if not np.array_equal(cached["sourceXi"], source_xi): raise AssertionError("saved Xi mismatch")
    Cq = cached["Cq"]; ids = [str(event["id"]) for event in events]
    F_all=[]; a_all=[]; q_all=[]; qf_all=[]; fp_all=[]; ep_all=[]; et_all=[]; event_index=[]; endpoint=[]; endpoint_groups=[]; endpoint_labels=[]
    Xenc=[]; yenc=[]; window_ids=[]; window_groups=[]; window_labels=[]; starts=[]; ends=[]; durations=[]; qf_mean=[]; yF=[]; per_event=[]; bins: dict[str, list[dict[str, Any]]] = {"[0.1,1)":[], "[1,2)":[], "[2,5)":[], "[5,8)":[], "[8,infinity)":[]}
    initial_event=[]
    for number, event in enumerate(events):
        endpoints, qenc, features = _encode_event(model, event)
        with torch.no_grad():
            state = torch.cat((tensor(event["x"][endpoints]), tensor(qenc)), dim=1)
            algebraic = model.algebraic_coordinate(state).numpy()
        force = np.asarray(event["force"])[endpoints]
        projected = project_force(force, algebraic, qenc, Cq)
        if not all(np.isfinite(array).all() for array in (force, algebraic, qenc, *projected.values())):
            _preserve_failure(output, str(event["id"]), "endpoint", {"F": force, "a": algebraic, "qenc": qenc, **projected})
            raise FloatingPointError(f"nonfinite endpoint force arrays: {event['id']}")
        F_all.append(force); a_all.append(algebraic); q_all.append(qenc); qf_all.append(projected["qF"]); fp_all.append(projected["Fperp"]); ep_all.append(projected["e_parallel"]); et_all.append(projected["e_total"])
        event_index.append(np.full(len(endpoints), number)); endpoint.append(endpoints); endpoint_groups.append(np.repeat(str(event["group"]), len(endpoints))); endpoint_labels.append(np.repeat(str(event["label"]), len(endpoints)))
        stats = {"event_id": event["id"], "group": str(event["group"]), "label": str(event["label"]), "n_endpoints": int(len(endpoints)), **event_metrics(force, projected["Fperp"], projected["e_parallel"], projected["e_total"], base.mass)}
        per_event.append(stats)
        initial_event.append({"event_id": event["id"], "group": str(event["group"]), "label": str(event["label"]),
                              **event_metrics(force[:1], projected["Fperp"][:1], projected["e_parallel"][:1], projected["e_total"][:1], base.mass)})
        for label, lower, upper in (("[0.1,1)", .1, 1.), ("[1,2)", 1., 2.), ("[2,5)", 2., 5.), ("[5,8)", 5., 8.), ("[8,infinity)", 8., np.inf)):
            choose = (endpoints * .01 >= lower) & (endpoints * .01 < upper)
            if choose.any(): bins[label].append({"event_id": event["id"], "group": str(event["group"]), "label": str(event["label"]), "n_endpoints": int(choose.sum()), **event_metrics(force[choose], projected["Fperp"][choose], projected["e_parallel"][choose], projected["e_total"][choose], base.mass)})
        positions = {int(value): index for index, value in enumerate(endpoints)}
        local_X=[]; local_y=[]
        for start, end in window_ranges(len(event["x"])):
            take=np.asarray([positions[index] for index in range(start, end+1)],dtype=int)
            mean,target,duration=window_average(features[take],qenc[take]); local_X.append(mean); local_y.append(target)
        Xenc.append(np.asarray(local_X)); yenc.append(np.asarray(local_y)); closure=window_integrals(projected["qF"], endpoints, window_ranges(len(event["x"])))
        if not all(np.isfinite(array).all() for array in (np.asarray(local_X), np.asarray(local_y), closure["qF_mean"], closure["yF"])):
            _preserve_failure(output, str(event["id"]), "window", {"Xenc": np.asarray(local_X), "yenc": np.asarray(local_y), **closure})
            raise FloatingPointError(f"nonfinite window arrays: {event['id']}")
        count=len(closure["start"]); window_ids.append(np.repeat(event["id"],count)); window_groups.append(np.repeat(str(event["group"]),count)); window_labels.append(np.repeat(str(event["label"]),count)); starts.append(closure["start"]); ends.append(closure["end"]); durations.append(closure["duration"]); qf_mean.append(closure["qF_mean"]); yF.append(closure["yF"])
    Xnew=np.concatenate(Xenc); ynew=np.concatenate(yenc); wids=np.concatenate(window_ids); wgroups=np.concatenate(window_groups); wlabels=np.concatenate(window_labels); wstart=np.concatenate(starts); wend=np.concatenate(ends); wduration=np.concatenate(durations); qfmean=np.concatenate(qf_mean); yf=np.concatenate(yF)
    for key, actual in (("event_ids", wids), ("group", wgroups), ("label", wlabels), ("start", wstart), ("end", wend), ("duration", wduration)):
        if not np.array_equal(actual, cached[key]): raise AssertionError(f"0062 window {key} ordering mismatch")
    x_error, y_error = _relative_error(Xnew, cached["X"]), _relative_error(ynew, cached["y"])
    if max(x_error, y_error) > 1e-9: raise AssertionError("0062 cached encoder window reproduction")
    XF=cached["X"].copy(); XF[:,1:4]=qfmean
    enc_defect=(cached["y"]-cached["X"]@source_xi)@Cq.T
    force_defect=(yf-XF@source_xi)@Cq.T
    if not all(np.isfinite(values).all() for values in (XF, enc_defect, force_defect)):
        bad = ~np.isfinite(XF).all(axis=1) | ~np.isfinite(enc_defect).all(axis=1) | ~np.isfinite(force_defect).all(axis=1)
        _preserve_failure(output, "closure", "projected_defect", {
            "event_ids": wids[bad], "start": wstart[bad], "end": wend[bad],
            "XF": XF[bad], "yF": yf[bad], "qenc_defect": enc_defect[bad],
            "qF_defect": force_defect[bad]})
        raise FloatingPointError("nonfinite closure arrays")
    F=np.concatenate(F_all); a=np.concatenate(a_all); qenc=np.concatenate(q_all); qF=np.concatenate(qf_all); Fperp=np.concatenate(fp_all); parallel=np.concatenate(ep_all); total=np.concatenate(et_all); ei=np.concatenate(event_index); ep=np.concatenate(endpoint); eg=np.concatenate(endpoint_groups); el=np.concatenate(endpoint_labels)
    scale=max(1.,float(np.linalg.norm(total)))
    pythagorean_residual = (np.sum(total * total, axis=1) - np.sum(Fperp * Fperp, axis=1)
                           - np.sum(parallel * parallel, axis=1))
    decomposition_error = float(np.abs(total - Fperp - parallel).max() / scale)
    orthogonality_error = float(np.abs(Fperp @ Cq).max() / max(1., np.linalg.norm(Fperp) * np.linalg.norm(Cq)))
    pythagorean_error = float(np.abs(pythagorean_residual).max() / max(1., np.linalg.norm(total) ** 2))
    checks={"decomposition_max_relative": decomposition_error,
            "orthogonality_max_relative": orthogonality_error,
            "pythagorean_max_relative": pythagorean_error}
    if max(checks.values())>1e-9: raise AssertionError("force decomposition identity")
    if digest(model_dir/"model.pt") != MODEL_SHA: raise AssertionError("frozen model changed")
    current_hashes = {"model": digest(model_dir / "model.pt"), "normalization": digest(model_dir / "normalization.npz"),
                      "base": digest(BASE), "network": digest(NETWORK)}
    if current_hashes != expected_hashes: raise AssertionError("frozen source changed during audit")
    decomposition=_aggregate(per_event); denominator=decomposition["e_parallel_sq"]+decomposition["Fperp_sq"]; share=None if denominator==0 else decomposition["e_parallel_sq"]/denominator
    by_group={str(group):_aggregate([row for row in per_event if row["group"]==str(group)]) for group in sorted({row["group"] for row in per_event})}
    by_label={str(label):_aggregate([row for row in per_event if row["label"]==str(label)]) for label in sorted({row["label"] for row in per_event})}
    event_groups=np.asarray([str(event["group"]) for event in events]); event_labels=np.asarray([str(event["label"]) for event in events])
    evidence={"test_read":False,"validation_loaded_by_existing_reader_not_used":True,"event_count":len(events),"endpoint_count":int(len(F)),"window_count":int(len(wids)),"mass":base.mass.tolist(),"event_ids":ids,"event_groups":event_groups.tolist(),"event_labels":event_labels.tolist(),"source_sha256":{"model":MODEL_SHA,"normalization":digest(model_dir/"normalization.npz"),"base":digest(BASE),"network":digest(NETWORK),"source0062_windows":digest(WINDOWS/"windows.npz"),"source0062_evidence":digest(WINDOWS/"evidence.json"),"audit_code":digest(__file__),"group_helper":digest(Path(__file__).with_name("audit_group_identifiability.py")),"evaluator_helper":digest(Path(__file__).with_name("evaluate_long_latent.py")),"joint_helper":digest(Path(__file__).with_name("train_joint_latent.py"))},"projection_checks":checks,"cached_window_reproduction":{"X_relative_max":x_error,"y_relative_max":y_error},"force_decomposition":{**decomposition,"denominator_parallel_plus_perp_sq":denominator,"parallel_share":share,"parallel_share_gt_0p5_raw":None if share is None else bool(share>.5),"by_group":by_group,"by_label":by_label},"per_event":per_event,"initial_endpoint":{"stats":initial_event,"aggregate":_aggregate(initial_event)},"time_bins":{key:{"stats":values,"aggregate":_aggregate(values)} for key,values in bins.items()},"common_acceleration":{"units":"rad/s^2","event_balanced_rms":{key:decomposition[key+"_event_balanced_rms"] for key in ("Fperp_common_accel_rmse","e_parallel_common_accel_rmse","e_total_common_accel_rmse")},"mean_event_rmse":{key:decomposition[key] for key in ("Fperp_common_accel_rmse","e_parallel_common_accel_rmse","e_total_common_accel_rmse")}},"closure":{"qenc":_closure_summary(enc_defect,wids,wgroups,wlabels),"qF":_closure_summary(force_defect,wids,wgroups,wlabels),"warning":"qF is an instantaneous truth-force projection oracle, not a causal latent state or forecast input."},"excluded_predecision":excluded,"identities":identities}
    np.savez_compressed(output/"force_arrays.npz",F=F,a=a,qenc=qenc,qF=qF,Fperp=Fperp,e_parallel=parallel,e_total=total,event_ids=np.asarray(ids),event_groups=event_groups,event_labels=event_labels,mass=base.mass,event_index=ei,endpoint=ep,Cq=Cq,sourceXi=source_xi)
    np.savez_compressed(output/"closure_windows.npz",event_ids=wids,group=wgroups,label=wlabels,start=wstart,end=wend,duration=wduration,qF_mean=qfmean,yF=yf)
    (output/"evidence.json").write_text(json.dumps(_safe(evidence),indent=2,allow_nan=False),encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--model-dir",type=Path,default=SOURCE); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); run(args.model_dir,args.output)


if __name__=="__main__": main()
