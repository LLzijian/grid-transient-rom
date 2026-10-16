"""Fixed-horizon, development-only evaluator for the Loop0060 latent campaign."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from kiki.models.private_pm import MechanicalROM  # noqa: E402
from kiki.physics import coi_project  # noqa: E402
from train_joint_latent import BASE, DT, NETWORK, JointROM, aggregate, read_development, tensor  # noqa: E402

SCALES = np.array([.47675314588683515, 1.0033667249602924]); HORIZONS = (1, 2, 5, 8)
DEFAULT_MODEL = ROOT / "private_data/task2_joint_0058"

def _safe(v: Any) -> Any:
    if isinstance(v, dict): return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_safe(x) for x in v]
    if isinstance(v, np.ndarray): return _safe(v.tolist())
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, (np.bool_, bool)): return bool(v)
    return v

def _model_class(model_name: str):
    """Import the nonlinear teacher only when explicitly requested."""
    if model_name != "teacher": return JointROM
    from train_teacher_latent import TeacherROM
    return TeacherROM

def _construct(model_dir: Path, base: MechanicalROM, network: dict[str, np.ndarray], model_name: str="0058") -> JointROM:
    """Use the original constructor; prove frozen buffers before state loading."""
    import torch
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    with np.load(model_dir / "normalization.npz", allow_pickle=False) as z:
        # Keep load_star's saved one-element tensor shape: casting it to a
        # scalar would fail the checkpoint-buffer identity contract.
        electrical = (z["z_scale"], z["voltage_star"], z["load_positions"].astype(int), network["load_g"], network["equilibrium_components"][0])
        cls = _model_class(model_name)
        model = cls(base, network["port_incidence"][:, z["port_ids"].astype(int)], z["input_scale"], z["feature_scale"], float(z["force_scale"]), float(z["load_scale"]), electrical=electrical, latent_dim=int(state["c"].shape[1]))
    buffers = ("mass", "incidence", "delta_star", "coupling", "damping", "rate", "gain", "angle_incidence", "ports", "input_scale", "feature_scale", "force_scale", "load_scale", "metric_scale", "z_scale", "voltage_star", "load_positions", "load_g", "load_star")
    for name in buffers:
        if name not in state or not torch.equal(model.state_dict()[name], state[name]): raise AssertionError(f"checkpoint/constructor buffer mismatch: {name}")
    model.load_state_dict(state, strict=True); model.eval(); return model

def _metric_row(pred: np.ndarray, event: dict[str, Any], horizon: int | str, mass: np.ndarray) -> dict[str, Any] | None:
    """COI score aligned at x[10]; scored errors begin at the first future step."""
    available = len(event["x"]) - 11
    steps = available if horizon == "prefix" else round(float(horizon) / DT)
    if steps <= 0 or (horizon != "prefix" and available < steps): return None
    steps = min(steps, len(pred) - 1)
    err = pred[1:steps + 1, :20] - event["x"][11:11 + steps, :20]
    de, we = coi_project(err[:, :10], mass), coi_project(err[:, 10:20], mass)
    ds, ws = float((de*de).sum()), float((we*we).sum()); sse = .5*(ds/SCALES[0]**2 + ws/SCALES[1]**2)
    return {"event_id": event["id"], "group": event["group"], "label": event["label"], "horizon": str(horizon), "scope": "available_prefix" if horizon == "prefix" else "complete", "complete": horizon != "prefix", "n": steps*10, "delta_sse": ds, "omega_sse": ws, "normalized_sse": sse, "rmse": float(np.sqrt(sse/(steps*10)))}

def _groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(g): aggregate([r for r in rows if str(r["group"]) == str(g)]) for g in sorted({r["group"] for r in rows})}

def _m3(base: MechanicalROM, histories: np.ndarray, dt: float, steps: int) -> np.ndarray:
    initial = histories[:, -1].copy(); initial[:, 20:30] += base.pm_star
    result = base.rollout(initial, None, np.arange(steps+1)*dt, max_step=dt); result[..., 20:30] -= base.pm_star
    return result

def _reported_aq(model: JointROM, model_name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return actual teacher RHS Jacobian plus the separately labeled anchor."""
    anchor = (model.xi*model.support).detach().numpy()[1:1 + model.rank].T
    if model_name == "teacher":
        return model.latent_jacobian_at_reference().detach().cpu().numpy(), anchor, "complete_teacher_rhs_reference_derivative"
    return anchor, anchor, "masked_sparse_xi"

def _diagnostic(model: JointROM | None, state: np.ndarray, event: dict[str, Any], h: int | str, mass: np.ndarray) -> dict[str, Any]:
    steps = min(len(state)-1, len(event["x"])-11 if h == "prefix" else round(float(h)/DT)); x = state[:steps+1]
    row = {"event_id":event["id"], "group":event["group"], "label":event["label"], "horizon":str(h), "points":steps+1, "finite":bool(np.isfinite(x).all()), "max_coi_delta":float(np.abs(coi_project(x[:,:10],mass)).max()), "max_coi_omega":float(np.abs(coi_project(x[:,10:20],mass)).max()), "max_p":float(np.abs(x[:,20:30]).max()), "max_q":None, "max_actual_memory_force":0., "max_total_force":None}
    if model is None: return row
    import torch
    with torch.no_grad():
        tx=tensor(x); q=tx[:,30:]; memory_q=model.memory(model.memory_enabled*q); action=model.memory(model.memory_enabled*q+model.algebraic_coordinate(tx)); phase=model.delta_star@model.incidence; e0=((torch.sin(tx[:,:10]@model.incidence+phase)-torch.sin(phase))*model.coupling)@model.incidence.T; total=tx[:,20:30]-e0-model.damping*tx[:,10:20]-action
    row.update(max_q=float(np.abs(x[:,30:]).max()), max_actual_memory_force=float(memory_q.abs().max()), max_combined_action_force=float(action.abs().max()), max_total_force=float(total.abs().max())); return row

def _q_discrepancy(model: JointROM, autonomous: np.ndarray, event: dict[str, Any], h: int | str) -> dict[str, Any]:
    """Batched genuine 11-point encoder windows: no truth/future feedback."""
    import torch
    requested = len(event["x"])-11 if h == "prefix" else round(float(h)/DT)
    steps = min(len(autonomous)-1, len(event["x"])-11, requested)
    if steps < 0: return {"event_id":event["id"], "horizon":str(h), "points":0, "rms":None}
    windows=np.stack([event["x"][j:j+11] for j in range(steps+1)])
    with torch.no_grad(): encoded=model.encode(tensor(windows)).numpy()
    diff=autonomous[:steps+1,30:]-encoded
    return {"event_id":event["id"], "group":event["group"], "label":event["label"], "horizon":str(h), "points":steps+1, "rms":float(np.sqrt(np.mean(diff*diff))), "max_abs":float(np.abs(diff).max())}

def _electrical(model: JointROM, event: dict[str, Any], h: int | str) -> dict[str, Any]:
    """Auxiliary genuine-window decoder error; it is not a rollout input."""
    import torch
    steps=len(event["x"])-11 if h == "prefix" else round(float(h)/DT)
    if steps < 0 or len(event["x"])-11 < steps: return {"event_id":event["id"],"horizon":str(h),"points":0,"available":False}
    w=np.stack([event["x"][j:j+11] for j in range(steps+1)])
    with torch.no_grad(): q=model.encode(tensor(w)); z, load=model.electrical_output(tensor(w[:,-1]),q)
    ze=z.numpy()-event["z"][10:11+steps]/model.z_scale.detach().numpy(); le=load.numpy().ravel()-event["load"][10:11+steps]/float(model.load_scale)
    return {"event_id":event["id"],"group":event["group"],"label":event["label"],"horizon":str(h),"points":steps+1,"available":True,"pe_mse":float(np.mean(ze[:,:10]*ze[:,:10])),"voltage_mse":float(np.mean(ze[:,10:]*ze[:,10:])),"load_mse":float(np.mean(le*le))}

def _verify_0058(model: JointROM, histories: np.ndarray, events: list[dict[str, Any]], base: MechanicalROM, directory: Path) -> dict[str,float]:
    import torch
    saved=np.load(directory/"trajectories.npz",allow_pickle=False)["predictions"]
    with torch.no_grad(): current=model.rollout(tensor(histories),200,.005)[:,::2].numpy()
    trajectory_error=float(np.abs(current-saved).max())
    if trajectory_error >= 1e-8: raise AssertionError(f"0058 trajectory reproduction: {trajectory_error}")
    rows=[]
    for p,e in zip(current,events,strict=True):
        if len(e["x"]) < 111: continue
        err=p[1:101,:20]-e["x"][11:111,:20]; de,we=coi_project(err[:,:10],base.mass),coi_project(err[:,10:20],base.mass); ds,ws=float((de*de).sum()),float((we*we).sum()); s=.5*(ds/SCALES[0]**2+ws/SCALES[1]**2); rows.append({"event_id":e["id"],"group":e["group"],"label":e["label"],"n":1000,"delta_sse":ds,"omega_sse":ws,"normalized_sse":s,"rmse":float(np.sqrt(s/1000))})
    old=json.loads((directory/"evidence.json").read_text())["summary"]["1.0"]; now=aggregate(rows); aggregate_error=max(abs(float(now[k])-float(old[k])) for k in ("mean","pooled","cvar10","worst"))
    if aggregate_error >= 1e-10: raise AssertionError(f"0058 aggregate reproduction: {aggregate_error}")
    return {"trajectory_max_abs":trajectory_error,"aggregate_max_abs":aggregate_error}

def benchmark(model_name: str="0058", model_dir: Path=DEFAULT_MODEL, output: Path|None=None) -> dict[str,Any]:
    """Development-only execution; caller, not this module, authorizes a run."""
    import torch
    torch.set_num_threads(1); base=MechanicalROM.load(BASE); network=dict(np.load(NETWORK,allow_pickle=False)); network["observed_buses"]=np.unique(np.r_[network["load_buses"],network["generator_buses"]]); network["load_positions"]=np.searchsorted(network["observed_buses"],network["load_buses"])
    development, excluded, identities=read_development(base,network); events=development["validation"]; histories=np.stack([e["x"][:11] for e in events]); model=None if model_name=="M3" else _construct(model_dir,base,network,model_name)
    with torch.no_grad():
        if model is None: fine=_m3(base,histories,.005,1600); coarse=_m3(base,histories,.01,800); reproduction=None
        else: fine=model.rollout(tensor(histories),1600,.005).numpy(); coarse=model.rollout(tensor(histories),800,.01).numpy(); reproduction=_verify_0058(model,histories,events,base,model_dir) if model_name=="0058" else None
    reference=fine[:,::2]
    # M3 is always evaluated from precisely the same x[10] physical state.
    m3fine = fine if model is None else _m3(base,histories,.005,1600)
    m3reference = m3fine[:,::2]
    complete={str(h):[] for h in HORIZONS}; prefix=[]; records=[]; m3complete={str(h):[] for h in HORIZONS}; m3prefix=[]; m3records=[]; diagnostics=[]; qrows=[]; electrical=[]
    for i,e in enumerate(events):
        for h in HORIZONS:
            r=_metric_row(reference[i],e,h,base.mass)
            if r is not None: records.append(r); complete[str(h)].append(r)
            b=_metric_row(m3reference[i],e,h,base.mass)
            if b is not None: m3complete[str(h)].append(b); m3records.append(b)
            diagnostics.append(_diagnostic(model,reference[i],e,h,base.mass))
            if model is not None: qrows.append(_q_discrepancy(model,reference[i],e,h)); electrical.append(_electrical(model,e,h))
        r=_metric_row(reference[i],e,"prefix",base.mass)
        if r is not None: records.append(r); prefix.append(r)
        b=_metric_row(m3reference[i],e,"prefix",base.mass)
        if b is not None: m3prefix.append(b); m3records.append(b)
        diagnostics.append(_diagnostic(model,reference[i],e,"prefix",base.mass))
        if model is not None: qrows.append(_q_discrepancy(model,reference[i],e,"prefix")); electrical.append(_electrical(model,e,"prefix"))
    numerical=[]; numerical_summary={}
    for i,e in enumerate(events):
        diff=fine[i,::2,:20]-coarse[i,:,:20]; de,we=coi_project(diff[:,:10],base.mass),coi_project(diff[:,10:20],base.mass)
        for h in HORIZONS:
            k=round(h/DT); d=diff[:k+1]; nde,nwe=coi_project(d[:,:10],base.mass),coi_project(d[:,10:20],base.mass)
            numerical.append({"event_id":e["id"],"group":e["group"],"label":e["label"],"horizon":str(h),"complete":len(e["x"])-11>=k,"finite_fine":bool(np.isfinite(fine[i,:2*k+1]).all()),"finite_coarse":bool(np.isfinite(coarse[i,:k+1]).all()),"step_rmse":float(np.sqrt(.5*np.mean((nde/SCALES[0])**2+(nwe/SCALES[1])**2)))})
        pk=min(len(e["x"])-11,800); pd=diff[:pk+1]; pde,pwe=coi_project(pd[:,:10],base.mass),coi_project(pd[:,10:20],base.mass)
        numerical.append({"event_id":e["id"],"group":e["group"],"label":e["label"],"horizon":"prefix","complete":False,"finite_fine":bool(np.isfinite(fine[i,:2*pk+1]).all()),"finite_coarse":bool(np.isfinite(coarse[i,:pk+1]).all()),"step_rmse":float(np.sqrt(.5*np.mean((pde/SCALES[0])**2+(pwe/SCALES[1])**2)))})
    for h in HORIZONS:
        rows=[r for r in numerical if r["horizon"]==str(h)]
        full=[r for r in rows if r["complete"]]
        numerical_summary[str(h)]={"all_fixed_forecasts":{"n_events":len(rows),"max_step_rmse":max(r["step_rmse"] for r in rows)},"complete_cohort":{"n_events":len(full),"max_step_rmse":max((r["step_rmse"] for r in full),default=None)}}
    evidence={"model":model_name,"model_sha256":hashlib.sha256((model_dir/"model.pt").read_bytes()).hexdigest() if model else None,"normalization_sha256":hashlib.sha256((model_dir/"normalization.npz").read_bytes()).hexdigest() if model else None,"base_sha256":hashlib.sha256(Path(BASE).read_bytes()).hexdigest(),"network_sha256":hashlib.sha256(Path(NETWORK).read_bytes()).hexdigest(),"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"test_read":False,"history_points":11,"initial_truth_index":10,"dt_reference":.005,"dt_coarse":.01,"horizons":list(HORIZONS),"event_count":len(events),"excluded_predecision":excluded,"identities":identities,"complete_counts":{h:len(v) for h,v in complete.items()},"summary":{h:aggregate(v) for h,v in complete.items()},"group_summary":{h:_groups(v) for h,v in complete.items()},"available_prefix":aggregate(prefix),"available_prefix_groups":_groups(prefix),"unstable_available_prefix":aggregate([r for r in prefix if r["label"]=="unstable"]),"M3_summary":{h:aggregate(v) for h,v in m3complete.items()},"M3_group_summary":{h:_groups(v) for h,v in m3complete.items()},"M3_available_prefix":aggregate(m3prefix),"M3_unstable_available_prefix":aggregate([r for r in m3prefix if r["label"]=="unstable"]),"records":records,"M3_records":m3records,"diagnostics":diagnostics,"q_autonomous_encoder":qrows,"electrical_auxiliary_errors":electrical,"numerical":numerical,"numerical_summary":numerical_summary,"reproduction":reproduction,"truth_lengths":[len(e["x"])-10 for e in events]}
    if model is not None:
        xi=(model.xi*model.support).detach().numpy()
        # Teacher reports the derivative of its full sparse-plus-neural RHS,
        # never the sparse anchor alone.
        aq,anchor_aq,aq_source=_reported_aq(model,model_name)
        poles=np.linalg.eigvals(aq); c=model.c.detach().numpy().copy(); c=c/np.linalg.norm(c,axis=0,keepdims=True); cq=model.amplitude.detach().numpy()[None,:]*(model.ports.detach().numpy()@c)
        if model_name == "teacher":
            evidence.update(masked_xi_anchor=xi, support_anchor=model.support.detach().numpy(), Aq_anchor_masked_xi=anchor_aq)
        else:
            # Preserve the established sparse-artifact field names exactly.
            evidence.update(masked_xi=xi, support=model.support.detach().numpy())
        evidence.update(Aq=aq,Aq_source=aq_source,Aq_poles=np.column_stack((poles.real,poles.imag)),Cq_singular_values=np.linalg.svd(cq,compute_uv=False))
    if output is not None:
        output.mkdir(parents=True,exist_ok=False)
        arrays={"event_ids":np.array([e["id"] for e in events]),"fine_predictions":fine,"coarse_predictions":coarse,"M3":m3fine}
        # Truth is aligned with forecast t=0, not padded to a synthetic length.
        arrays.update({f"{e['id']}_truth":e["x"][10:821] for e in events})
        np.savez_compressed(output/"trajectories.npz",**arrays); (output/"per_event_rows.json").write_text(json.dumps(_safe(records),allow_nan=False)); (output/"evidence.json").write_text(json.dumps(_safe(evidence),indent=2,allow_nan=False))
    return evidence

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=("0058","challenger","teacher","M3"),default="0058"); p.add_argument("--model-dir",type=Path,default=DEFAULT_MODEL); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); benchmark(a.model,a.model_dir,a.output); return 0
if __name__ == "__main__": raise SystemExit(main())
