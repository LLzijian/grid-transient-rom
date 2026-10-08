"""Loop0081 frozen conditional latent-consistency diagnostic (TRAIN-only CLI)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
from train_joint_latent import BASE, NETWORK, DT, ROOTS, read_development  # noqa: E402
from train_structured_rom import load_model  # noqa: E402
from kiki.models.private_pm import MechanicalROM  # noqa: E402

CHECKPOINT = ROOT / "private_data/task2_structure_0078_extend_G-fixed/step3000"
FORCE = ROOT / "private_data/task2_structure_force_0079"
FINE, COARSE = .005, .01
TORQUE_SCALE = 16.534910857679173
EXPECTED_ENDPOINTS, EXPECTED_WINDOWS = 465317, 46514


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(checkpoint: Path, force: Path) -> dict[str, str]:
    paths = {"model": checkpoint / "model.pt", "checkpoint": checkpoint / "checkpoint.json",
             "normalization": checkpoint / "normalization.npz", "export": checkpoint / "structured_numeric_export.json",
             "base": Path(BASE), "network": Path(NETWORK), "projection_arrays": force / "projection_arrays.npz",
             "projection_evidence": force / "evidence.json", "audit": Path(__file__),
             "trainer": ROOT / "experiments/train_structured_rom.py", "reader": ROOT / "experiments/train_joint_latent.py",
             "core": ROOT / "experiments/structured_rom.py", "grammar": ROOT / "experiments/structure_grammar.py"}
    for root in ROOTS:
        paths[f"{root.name}_csv"] = root / "dataset_index.csv"; paths[f"{root.name}_splits"] = root / "splits.json"
    return {name: sha(path) for name, path in paths.items()}


def windows(length: int) -> list[tuple[int, int]]:
    if length <= 11: raise ValueError("event has no genuine post-100ms diagnostic window")
    return [(start, min(start + 10, length - 1)) for start in range(10, length - 1, 10)]


def _physical(xsegments: torch.Tensor, time_s: float, duration_steps: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear true physical state at one shared RK stage time."""
    step = torch.floor(torch.full_like(duration_steps, time_s / DT + 1e-12, dtype=torch.float64)).to(torch.long)
    step = torch.minimum(step, duration_steps - 1)
    fraction = ((time_s - step.to(torch.float64) * DT) / DT).clamp(0., 1.)
    row = torch.arange(len(step), device=xsegments.device)
    return xsegments[row, step] + fraction[:, None] * (xsegments[row, step + 1] - xsegments[row, step])


def _rhs(model, xsegments: torch.Tensor, duration_steps: torch.Tensor, q: torch.Tensor, time_s: float) -> torch.Tensor:
    if not torch.isfinite(q).all(): raise FloatingPointError("nonfinite intermediate conditional q")
    x = _physical(xsegments, time_s, duration_steps)
    value = model.field(torch.cat((x, q), -1))[:, 30:]
    if not torch.isfinite(value).all(): raise FloatingPointError("nonfinite conditional q field")
    return value


def integrate_q(model, xsegments: torch.Tensor, durations: torch.Tensor, q0: torch.Tensor, dt: float) -> torch.Tensor:
    """RK4 q-only integration; physical x is interpolated at every stage."""
    if xsegments.ndim != 3 or xsegments.shape[2] != 30 or q0.shape != (len(xsegments), 3): raise ValueError("conditional batch shape")
    if not torch.isfinite(xsegments).all() or not torch.isfinite(q0).all() or torch.any(durations <= 0): raise FloatingPointError("nonfinite conditional input")
    q = q0.clone(); nsteps = torch.round(durations.to(torch.float64) * DT / dt).to(torch.long)
    if not torch.allclose(nsteps.to(torch.float64) * dt, durations.to(torch.float64) * DT, atol=1e-12, rtol=0): raise ValueError("window duration incompatible with RK step")
    for step in range(int(nsteps.max())):
        rows = torch.nonzero(nsteps > step, as_tuple=False).ravel()
        duration = durations.index_select(0, rows); x = xsegments.index_select(0, rows); state = q.index_select(0, rows); t = step * dt
        a = _rhs(model, x, duration, state, t)
        b = _rhs(model, x, duration, state + dt * a / 2, t + dt / 2)
        c = _rhs(model, x, duration, state + dt * b / 2, t + dt / 2)
        d = _rhs(model, x, duration, state + dt * c, t + dt)
        nxt = state + dt * (a + 2 * b + 2 * c + d) / 6
        if not torch.isfinite(nxt).all(): raise FloatingPointError("nonfinite conditional q stage")
        q = q.index_copy(0, rows, nxt)
    return q


def mapped_defect(L: np.ndarray, qhat: np.ndarray, qref: np.ndarray, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.einsum("nij,nj->ni", L, qhat - qref) / duration[:, None]
    if not (np.isfinite(value).all() and np.all(duration > 0)): raise FloatingPointError("nonfinite mapped conditional defect")
    return value, np.mean(value * value, axis=1)


def _summary(event_errors: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    picked = event_errors[indices]
    return {"n_events": int(len(indices)), "rms": float(np.sqrt(np.mean(picked))),
            "event_mean_squared": float(np.mean(picked))}


def summarize(window_event: np.ndarray, window_label: list[str], window_group: list[str], values: np.ndarray) -> dict[str, Any]:
    per_event = np.array([values[window_event == event].mean() for event in range(346)])
    labels = np.asarray(window_label, dtype=str); groups = np.asarray(window_group, dtype=str)
    result = {"overall": _summary(per_event, np.arange(346)),
              "subsets": {}, "groups": {}, "initial_window": {}}
    for label, expected in (("stable", 311), ("unstable", 35)):
        event_ids = np.unique(window_event[labels == label])
        if len(event_ids) != expected: raise AssertionError(f"{label} event count")
        result["subsets"][label] = _summary(per_event, event_ids)
    for group in sorted(set(groups)):
        event_ids = np.unique(window_event[groups == group]); result["groups"][group] = _summary(per_event, event_ids)
    result["event_mean_squared"] = per_event.tolist()
    return result


def _window_table(train: list[dict[str, Any]], force_arrays, force_evidence: dict[str, Any]):
    ids = force_evidence["train_event_ids"]
    if ids != [event["id"] for event in train]: raise AssertionError("0079 event ordering")
    index, endpoint = force_arrays["G-fixed_event_index"], force_arrays["G-fixed_endpoint"]
    qenc, qf, L = force_arrays["G-fixed_qenc"], force_arrays["G-fixed_qF"], force_arrays["G-fixed_L"]
    lookup = {(int(i), int(e)): row for row, (i, e) in enumerate(zip(index, endpoint, strict=True))}
    expected = sum(len(event["x"]) - 10 for event in train)
    if len(index) != EXPECTED_ENDPOINTS or expected != EXPECTED_ENDPOINTS or len(lookup) != len(index): raise AssertionError("0079 endpoint mapping/count")
    event_i=[]; starts=[]; ends=[]; q0e=[]; q1e=[]; q0f=[]; q1f=[]; maps=[]; xsegments=[]; labels=[]; groups=[]; ids_out=[]
    for i, event in enumerate(train):
        saved = np.sort(endpoint[index == i])
        expected_endpoints = np.arange(10, len(event["x"]), dtype=int)
        if not np.array_equal(saved, expected_endpoints): raise AssertionError("0079 full per-event endpoint mapping")
        for start, end in windows(len(event["x"])):
            try: a, b = lookup[(i, start)], lookup[(i, end)]
            except KeyError as error: raise AssertionError("missing 0079 window endpoint") from error
            event_i.append(i); starts.append(start); ends.append(end); q0e.append(qenc[a]); q1e.append(qenc[b]); q0f.append(qf[a]); q1f.append(qf[b]); maps.append(L[b])
            xsegments.append(event["x"][start:end + 1]); labels.append(event["label"]); groups.append(str(event["group"])); ids_out.append(event["id"])
    if len(event_i) != EXPECTED_WINDOWS: raise AssertionError(f"window count {len(event_i)}")
    return {"event_index":np.asarray(event_i), "start":np.asarray(starts), "end":np.asarray(ends), "duration":(np.asarray(ends)-np.asarray(starts))*DT,
            "qenc_start":np.asarray(q0e), "qenc_end":np.asarray(q1e), "qF_start":np.asarray(q0f), "qF_end":np.asarray(q1f), "L_end":np.asarray(maps),
            "xsegments":xsegments, "label":np.asarray(labels), "group":np.asarray(groups), "event_id":np.asarray(ids_out)}


def _run_batches(model, table, q_start: np.ndarray, step: float, name: str, batch_size: int = 256) -> np.ndarray:
    out = np.empty_like(q_start)
    with torch.no_grad():
        for low in range(0, len(q_start), batch_size):
            high=min(low+batch_size,len(q_start)); duration_steps=torch.as_tensor(table["end"][low:high]-table["start"][low:high],dtype=torch.long)
            padded=np.zeros((high-low, 11, 30));
            for local, segment in enumerate(table["xsegments"][low:high]): padded[local,:len(segment)]=segment; padded[local,len(segment):]=segment[-1]
            try:
                out[low:high]=integrate_q(model,torch.as_tensor(padded,dtype=torch.float64),duration_steps,torch.as_tensor(q_start[low:high],dtype=torch.float64),step).cpu().numpy()
            except Exception as error:
                raise type(error)(f"{name} conditional batch [{low},{high}) dt={step}: {error}") from error
    return out


def run(checkpoint: Path, force: Path, output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1); started=time.perf_counter(); before=hashes(checkpoint,force); current={"stage":"provenance"}
    try:
        meta=json.loads((checkpoint/"checkpoint.json").read_text());
        if meta.get("arm") != "G-fixed" or meta.get("step") != 3000: raise AssertionError("frozen G-fixed3000 required")
        base=MechanicalROM.load(BASE); net=dict(np.load(NETWORK,allow_pickle=False)); net["observed_buses"]=np.unique(np.r_[net["load_buses"],net["generator_buses"]]); net["load_positions"]=np.searchsorted(net["observed_buses"],net["load_buses"])
        events, excluded, identities=read_development(base,net,full_train=True); train=events["train"]
        if len(train)!=346 or sum(e["label"]=="stable" for e in train)!=311 or sum(e["label"]=="unstable" for e in train)!=35 or len({str(e["group"]) for e in train})!=24 or sum(v=="train" for v in excluded.values())!=23: raise AssertionError("TRAIN identity")
        extension=json.loads((checkpoint.parent/"evidence.json").read_text())
        if identities != extension["identities"]: raise AssertionError("extension dataset identity")
        evidence=json.loads((force/"evidence.json").read_text()); arrays=np.load(force/"projection_arrays.npz",allow_pickle=False)
        if evidence["identities"] != identities or evidence["hashes_before"]["G"]["model"] != sha(checkpoint/"model.pt") or evidence["hashes_before"]["G"]["normalization"] != sha(checkpoint/"normalization.npz") or evidence["hashes_before"]["G"]["base"] != sha(BASE) or evidence["hashes_before"]["G"]["network"] != sha(NETWORK): raise AssertionError("0079 provenance")
        table=_window_table(train,arrays,evidence); output.mkdir(); model=load_model(checkpoint,base=base); model.eval(); frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
        current={"stage":"conditional_q_only","windows":len(table["event_index"])}
        if hasattr(model,"begin_graph_cache"):
            with torch.no_grad(): model.begin_graph_cache()
        try:
            fine_e=_run_batches(model,table,table["qenc_start"],FINE,"qenc_fine"); fine_f=_run_batches(model,table,table["qF_start"],FINE,"qF_fine")
            coarse_e=_run_batches(model,table,table["qenc_start"],COARSE,"qenc_coarse"); coarse_f=_run_batches(model,table,table["qF_start"],COARSE,"qF_coarse")
        finally:
            if hasattr(model,"end_graph_cache"): model.end_graph_cache()
        values={}; numerics={}
        for name,qhat,qref,qcoarse in (("qenc",fine_e,table["qenc_end"],coarse_e),("qF",fine_f,table["qF_end"],coarse_f)):
            defect,squared=mapped_defect(table["L_end"],qhat,qref,table["duration"]); numeric_defect,_=mapped_defect(table["L_end"],qhat-qcoarse,np.zeros_like(qhat),table["duration"])
            numeric_rms=np.sqrt(np.mean(numeric_defect*numeric_defect,axis=1))/TORQUE_SCALE
            values[name]=(defect,squared); numerics[name]=numeric_rms
        # Preserve all decisive conditional forecasts and window mappings even
        # if the subsequent fixed numerical gate fails.
        np.savez_compressed(output/"windows.npz",**{key:value for key,value in table.items() if key!="xsegments"},qhat_qenc_fine=fine_e,qhat_qF_fine=fine_f,qhat_qenc_coarse=coarse_e,qhat_qF_coarse=coarse_f,defect_qenc=values["qenc"][0],defect_qF=values["qF"][0],window_mse_qenc=values["qenc"][1],window_mse_qF=values["qF"][1],numeric_qenc=numerics["qenc"],numeric_qF=numerics["qF"])
        if any(not np.isfinite(value).all() or np.any(value>.001) for value in numerics.values()): raise FloatingPointError("conditional numerical comparison")
        summaries={name:summarize(table["event_index"],table["label"],table["group"],squared) for name,(_,squared) in values.items()}
        # Explicit initial-window summaries and duration inventory are separate
        # from the equal-window-within-event primary aggregation.
        initial=table["start"]==10
        for name,(_,squared) in values.items():
            summaries[name]["initial_window"]={"n_windows":int(initial.sum()),"rms":float(np.sqrt(np.mean(squared[initial])))}
        after=hashes(checkpoint,force)
        if before != after: raise RuntimeError("source provenance changed")
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()): raise AssertionError("frozen model changed")
        result={"loop":"0081","test_read":False,"validation_materialized_by_reader":True,"conditional_only":True,"autonomous_claim":False,"identities":identities,"excluded":excluded,"hashes_before":before,"hashes_after":after,"n_train":346,"endpoint_count":EXPECTED_ENDPOINTS,"window_count":EXPECTED_WINDOWS,"duration_counts":{str(k):int(v) for k,v in Counter(table["end"]-table["start"]).items()},"summaries":summaries,"numerical":{name:{"max":float(value.max()),"n_windows":len(value)} for name,value in numerics.items()},"seconds":time.perf_counter()-started,"model_tensors_unchanged":True}
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False)); return result
    except Exception as error:
        if output.exists(): (output/"failure_evidence.json").write_text(json.dumps({"type":type(error).__name__,"message":str(error),"current":current,"hashes_before":before,"test_read":False},indent=2))
        raise


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",type=Path,default=CHECKPOINT); parser.add_argument("--force",type=Path,default=FORCE); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); run(args.checkpoint,args.force,args.output)


if __name__ == "__main__": main()
