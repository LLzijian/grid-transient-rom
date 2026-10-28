"""Finite matched p/r-exchange study; import performs no science or data access.

Native ownership, archival-commit verification and external-interruption authority
belong to the separate controller. No legacy qualification is relabelled here.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"src")]

import numpy as np
import torch

import force_graph_study_20261028 as fg
from force_graph_model_20261028 import ForceGraphROM
from force_exchange_model_20261028 import ForceExchangeROM
from closure_training_runtime import resolve_device, synchronize
from train_local_memory_0098 import graph, risk
from train_structured_rom import _point_error

ROOT=Path(__file__).resolve().parents[1]
old=fg.old
sha,read,write,safe=fg.sha,fg.read,fg.write,fg.safe
SEEDS=(28211,28212)
VARIANTS=("C0","C1")
PROTOCOL=ROOT/"reports/force_exchange_20261028_protocol.md"
PARENT=ROOT/"private_data/force_graph_20261028/study01"
SOURCE=PARENT/"study/A512_M_28101/checkpoints/step000192"
SOURCE_SHA={"model.json":"01d5220dce065ec186e8fe9f98d91eb9c8805bed75bbc2caf81db9b25de3e1d8",
 "checkpoint.json":"56e08f8d2e33d4bf14f109a580f8ef445f8db24c4457b26f3afc141ddb9a23b4",
 "training.pt":"68ed6d20093a6500345d2c5b51cf5526bb0c34373e2a1d8cb5ef72f7d480f72e"}
LIMITS={"qa":72,"formal":768,"repair":72,"replay":64}
EventStore=fg.EventStore
objective=fg.objective
horizon_for=fg.horizon_for


def code_identity():
    paths=(Path(__file__),ROOT/"experiments/force_exchange_model_20261028.py",PROTOCOL,
      ROOT/"results/force_exchange_20261028/matched_control_cpu_qualification.json",
      ROOT/"results/force_exchange_20261028/model_pre_freeze_qa.json")
    return {**fg.code_identity(),**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}


def source_identity():
    actual={name:sha(SOURCE/name) for name in SOURCE_SHA}
    if actual!=SOURCE_SHA: raise ValueError("fixed F assets changed")
    return {"directory":str(SOURCE),"files":actual,"parent_implementation":"ForceGraphROM"}


def build_model(variant,device="cpu"):
    source_identity()
    if variant=="F": model=ForceGraphROM.from_export(SOURCE/"model.json").requires_grad_(False)
    elif variant in VARIANTS:
        model=ForceExchangeROM.from_force_graph_export(SOURCE/"model.json",train_gamma=variant=="C1")
    else: raise ValueError("F/C0/C1 only")
    return model.to(resolve_device(device)).double()


def make_schedule(events,seed):
    if seed not in SEEDS or len(events)!=297: raise ValueError("fixed seed and fit297 required")
    rng=np.random.default_rng(seed)
    coverage={"cover_order":rng.permutation(len(events)).tolist(),"cursor":0}
    initial={"rng":copy.deepcopy(rng.bit_generator.state),"coverage":copy.deepcopy(coverage)}
    rows=[]
    for step in range(1,193):
        _,sampling=fg.sample_step(events,rng,coverage)
        rows.append({"step":step,"horizon_s":horizon_for(step),**sampling,
          "rng_after":copy.deepcopy(rng.bit_generator.state),"coverage_after":copy.deepcopy(coverage)})
    assert {e["id"] for e in events}=={events[i]["id"] for r in rows[:75] for i in r["coverage_indices"]}
    return {"initial":initial,"rows":rows}


def prepare(base):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    if (base/"prepare.json").exists(): raise FileExistsError("immutable preparation exists")
    source=source_identity()
    parent,fit=fg.checked_preparation(PARENT,fit_only=True)
    if len(fit)!=297 or sum(e["label"]=="unstable" for e in fit)!=30: raise ValueError("fit267/30 changed")
    panel=fg.panel_events(fit)
    selected=sorted([e for e in panel if e["label"]=="stable"],key=lambda e:(-len(e["x"]),e["id"]))[:12]
    if len(selected)!=12 or any(len(e["x"])-11<800 for e in selected): raise ValueError("12 full8s stable unavailable")
    schedule={str(seed):make_schedule(fit,seed) for seed in SEEDS}
    write(base/"schedule.json",schedule,exclusive=True)
    record={k:parent[k] for k in ("fit_ids","no_update_ids","original366_ids","fresh19_ids","fit_groups","no_update_groups",
                                  "fit_cache_sha256","no_update_cache_sha256")}
    record.update(status="COMPLETED",source=source,code=code_identity(),parent_prepare_sha256=sha(PARENT/"prepare.json"),
        schedule_sha256=sha(base/"schedule.json"),panel_ids=[e["id"] for e in panel],cost12_ids=[e["id"] for e in selected],
        old_legacy_drop_in_qualification="FAILED_NOT_ADOPTED",no_update_payload_read=False,
        source_exposed_to_train385=True,val_read=False,test_read=False,training_admission=False)
    write(base/"prepare.json",record,exclusive=True)
    return record


def checked_preparation(base,*,fit_only=True):
    base=Path(base);r=read(base/"prepare.json")
    if r["source"]!=source_identity() or r["code"]!=code_identity(): raise ValueError("source/code binding changed")
    if r["parent_prepare_sha256"]!=sha(PARENT/"prepare.json") or r["schedule_sha256"]!=sha(base/"schedule.json"):
        raise ValueError("parent/schedule changed")
    _,events=fg.checked_preparation(PARENT,fit_only=fit_only)
    return r,events


def make_optimizer(model):
    return torch.optim.Adam([{"params":[p],"block":name,"lr":.03} for name,p in model.named_parameters() if p.requires_grad],
                            betas=(.9,.999),eps=1e-8,weight_decay=0.)


def call_counts(base):
    path=Path(base)/"optimizer_calls.jsonl"
    calls={};counts={"attempts":0,"completed":0,"by_category":{},"pending":[]}
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);cid=r["call_id"];event=r["event"]
            if event=="optimizer_step_attempt":
                if cid in calls: raise ValueError("duplicate optimizer attempt")
                calls[cid]=r;key="attempts"
            elif event=="optimizer_step_completed":
                if cid not in calls or calls[cid].get("completed"): raise ValueError("unpaired optimizer completion")
                if any(r[k]!=calls[cid][k] for k in ("category","run_id","step")): raise ValueError("completion ownership differs")
                calls[cid]["completed"]=True;key="completed"
            else: raise ValueError("unknown ledger record")
            counts[key]+=1;counts["by_category"].setdefault(r["category"],{"attempts":0,"completed":0})[key]+=1
    counts["pending"]=[r for r in calls.values() if not r.get("completed")]
    return counts


def _journal(base,event,category,*,call_id,run_id,step,reserved_reason=None):
    if category not in LIMITS or category in ("repair","replay") and not reserved_reason:
        raise ValueError("reserved repair/replay requires explicit actual reason; never automatic")
    counts=call_counts(base)
    if event=="optimizer_step_attempt":
        if any(r["run_id"]==run_id for r in counts["pending"]):
            raise ValueError("unreconciled actual optimizer attempt in this run")
        if counts["attempts"]>=976 or counts["by_category"].get(category,{}).get("attempts",0)>=LIMITS[category]:
            raise ValueError("registered optimizer-call bound exhausted")
        if category=="formal" and not 1<=step<=192: raise ValueError("formal step outside finite run")
    elif event!="optimizer_step_completed": raise ValueError("unknown optimizer event")
    row=dict(event=event,category=category,call_id=call_id,run_id=run_id,step=step,time_unix=time.time(),reserved_reason=reserved_reason)
    with (Path(base)/"optimizer_calls.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()


def one_update(model,opt,store,samples,horizon,*,base,category,run_id,step):
    synchronize(model.mass.device);tick=time.perf_counter();opt.zero_grad(set_to_none=True)
    with graph(model):
        loss,terms=objective(model,store,samples,horizon,method="rk4")
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite J; no physical classification")
        loss.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params): raise FloatingPointError("nonfinite/missing gradient")
    norm=torch.nn.utils.clip_grad_norm_(params,100.)
    if not torch.isfinite(norm): raise FloatingPointError("nonfinite gradient norm")
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    cid=uuid.uuid4().hex
    _journal(base,"optimizer_step_attempt",category,call_id=cid,run_id=run_id,step=step)
    opt.step()
    _journal(base,"optimizer_step_completed",category,call_id=cid,run_id=run_id,step=step)
    model.project_nonnegative_();synchronize(model.mass.device)
    if any(not torch.isfinite(p).all() for p in params) or not old._adam_states_finite(opt.state.values()):
        raise FloatingPointError("nonfinite post-update model/Adam")
    changes={n:{"relative_l2":float((p.detach()-before[n]).norm()/before[n].norm().clamp_min(1.))}
             for n,p in model.named_parameters() if p.requires_grad}
    return {"step":step,"horizon_s":horizon,"method":"rk4",**{k:float(v.detach()) for k,v in terms.items()},
            "gradient_norm":float(norm),"block_updates":changes,"seconds":time.perf_counter()-tick,
            "gamma_physical_1_per_second":float(model.gamma.detach())}


def save_checkpoint(*args,**kwargs):
    # Parent saver is model-agnostic and invokes the actual subclass export.
    return fg.save_checkpoint(*args,**kwargs)


def restore_checkpoint(output,device,identity):
    output=Path(output);pointer=read(output/"latest.json");name=pointer["directory"]
    if Path(name).name!=name: raise ValueError("checkpoint must be immediate child")
    folder=output/name
    if sha(folder/"checkpoint.json")!=pointer["checkpoint_sha256"]: raise ValueError("metadata hash changed")
    meta=read(folder/"checkpoint.json");device=resolve_device(device)
    if meta["identity"]!=identity or not meta["resumable"] or meta["runtime"]!=old.runtime_identity(device):
        raise ValueError("resume identity/runtime differs")
    for filename,key in (("model.json","model_sha256"),("training.pt","training_sha256")):
        if sha(folder/filename)!=meta[key]: raise ValueError("checkpoint asset changed")
    model=ForceExchangeROM.from_export(folder/"model.json").to(device).double();opt=make_optimizer(model)
    if old._parameter_layout(model,opt)!=meta["parameter_layout"]: raise ValueError("Adam layout/LR mismatch")
    state=torch.load(folder/"training.pt",map_location="cpu",weights_only=False)
    if state["step"]!=meta["step"]: raise ValueError("saved step mismatch")
    opt.load_state_dict(state["optimizer"])
    if meta["step"] and (not opt.state or any(int(s["step"])!=meta["step"] for s in opt.state.values())):
        raise ValueError("real Adam counters mismatch")
    if not old._adam_states_finite(opt.state.values()): raise FloatingPointError("restored Adam nonfinite")
    rng=np.random.default_rng();rng.bit_generator.state=state["numpy_rng"]
    np.random.set_state(state["numpy_global_rng"]);random.setstate(state["python_rng"]);torch.set_rng_state(state["torch_rng"])
    if device.type=="cuda":
        if state["cuda_rng"] is None: raise ValueError("missing CUDA RNG")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return model,opt,rng,state["coverage"],meta


def profile(records,extras,events,seconds,parameter_count):
    p=fg.panel_summary(records,extras,seconds=seconds,parameter_count=parameter_count)
    rows=[r for r in records["objective_point_error"] if str(r["horizon"])=="prefix" and r["label"]=="unstable" and r["available"]]
    byid={r["event_id"]:r for r in extras}
    unresolved=[r["event_id"] for r in rows if not r.get("finite") or not r.get("numerically_admissible")
        or not byid[r["event_id"]]["scientific_numeric_8s"] or byid[r["event_id"]]["coarse_point_errors"].get("prefix") is None]
    fine=coarse=None
    if rows and not unresolved:
        fine=float(risk(torch.tensor([r["mean_point_error"] for r in rows],dtype=torch.float64)))
        coarse=float(risk(torch.tensor([byid[r["event_id"]]["coarse_point_errors"]["prefix"] for r in rows],dtype=torch.float64)))
    p.update(unstable_interval_risk=fine,unstable_interval_coarse_risk=coarse,
        unstable_interval_numerical_score_perturbation=abs(fine-coarse) if fine is not None else None,
        unstable_interval_unresolved_ids=unresolved,unstable_interval_eligible=len(rows),
        unstable_interval_definition="same actual per-event interval up to8s; original 0.8mean+0.2ceil-top10 risk",
        unstable_interval_steps={e["id"]:min(800,len(e["x"])-11) for e in events if e["label"]=="unstable"})
    allrows=[r for r in records["objective_point_error"] if str(r["horizon"])=="prefix" and r["available"]]
    allok=bool(allrows) and all(r.get("finite") and r.get("numerically_admissible") and byid[r["event_id"]]["scientific_numeric_8s"]
           and byid[r["event_id"]]["coarse_point_errors"].get("prefix") is not None for r in allrows)
    p["J_actual_intervals_up_to8_fine"]=10*float(risk(torch.tensor([r["mean_point_error"] for r in allrows],dtype=torch.float64))) if allok else None
    p["J_actual_intervals_up_to8_coarse"]=10*float(risk(torch.tensor([byid[r["event_id"]]["coarse_point_errors"]["prefix"] for r in allrows],dtype=torch.float64))) if allok else None
    return p


def _coarse_points(model,event,c):
    points={}
    for h in ("1","2","5","8","prefix"):
        n=min(800,len(event["x"])-11) if h=="prefix" else int(h)*100
        points[h]=None
        if n>0 and len(event["x"])-11>=n and np.isfinite(c[:n+1]).all():
            with torch.no_grad():
                value=float(_point_error(torch.as_tensor(c[1:n+1,:30]),torch.as_tensor(event["x"][11:11+n]),model).mean())
            if math.isfinite(value): points[h]=value
    return points


def score_panel(model,events):
    from evaluate_structured_rom import _rollouts,_pair_row
    from summarize_train_local_memory_0098 import event_metrics
    from fixed_edge_pair_20261019 import scientific_qualification
    torch.set_num_threads(1);tick=time.perf_counter();candidate=copy.deepcopy(model).cpu().double().eval()
    records={k:[] for k in ("records","common_frequency","normalized_p","objective_point_error","numerical")};extras=[]
    for start in range(0,len(events),16):
        block=events[start:start+16];histories=np.stack([e["x"][:11] for e in block]);error=None
        try: fine=_rollouts(candidate,histories,1600,.005);coarse=_rollouts(candidate,histories,800,.01)
        except Exception as exc:
            error=repr(exc);fine=np.full((len(block),1601,40),np.nan);coarse=np.full((len(block),801,40),np.nan)
        for e,h,f,c in zip(block,histories,fine,coarse):
            metrics=event_metrics(candidate,e,f,c)
            for key,rows in metrics.items(): records[key].extend(rows)
            pair=_pair_row(f,c,h,e,e["classification_meta"],candidate.mass.numpy())
            extras.append(dict(event_id=e["id"],label=e["label"],classification_8s=pair,
              scientific_numeric_8s=scientific_qualification(pair,metrics["numerical"]),coarse_point_errors=_coarse_points(candidate,e,c),error=error))
    return {**profile(records,extras,events,time.perf_counter()-tick,candidate.complexity_counts()["raw_trainable_scalar_count"]),
            "rows":safe(records),"extras":safe(extras),"event_ids":[e["id"] for e in events],"val_read":False,"test_read":False}


def evaluate(model,events,output,prepared,*,noisy=False):
    # Reliable parent path supplies 1/2/5/8s, actual15s qualification, stored
    # coarse/fine predictions, original366/new19 strata and per-event noise.
    result=old.evaluate_model(model,events,output,prepared,noisy=noisy)
    candidate=copy.deepcopy(model).cpu().double().eval();byid={e["id"]:e for e in events};extra={r["event_id"]:r for r in result["extras"]}
    for entry in result["prediction_files"]:
        if sha(entry["path"])!=entry["sha256"]: raise ValueError("prediction artifact changed")
        with np.load(entry["path"],allow_pickle=False) as data:
            for eid,c in zip(data["event_ids"],data["coarse"]):
                eid=str(eid);extra[eid]["coarse_point_errors"]=_coarse_points(candidate,byid[eid],c)
    p=profile(result["rows"],result["extras"],events,result["seconds"],candidate.complexity_counts()["raw_trainable_scalar_count"])
    p["all_required_rows_qualified"]=all(r["all_required_rows_qualified"] for r in result["profiles"].values())
    result["decision_profile"]=p
    result["original_evidence_sha256"]=sha(Path(output)/"evidence.json")
    write(Path(output)/"exchange_profile.json",{**p,"original_evidence_sha256":result["original_evidence_sha256"]},exclusive=True)
    return result


def valid(p):
    return fg._valid_profile(p) and p.get("unstable_interval_numerical_score_perturbation") is not None and not p.get("unstable_interval_unresolved_ids")


def optimization(trace):
    recent=trace[-48:];groups={n for r in recent for n in r["block_updates"]}
    clip=sum(r["gradient_norm"]>100 for r in recent)
    jumps={n:sum(r["block_updates"].get(n,{}).get("relative_l2",0)>.25 for r in recent) for n in groups}
    return dict(optimization_unresolved=len(recent)!=48 or clip>12 or any(v>=12 for v in jumps.values()),
                clip_count_last48=clip,large_change_counts_last48=jumps)


def extension_decision(before,after,trace):
    o=optimization(trace);r={**o,"extend":False}
    if o["optimization_unresolved"] or not valid(before) or not valid(after): return {**r,"reason":"optimization_or_numeric_unresolved"}
    a,b=before["unstable_interval_risk"],after["unstable_interval_risk"]
    gain=(a-b>=.01*a and a-b>before["unstable_interval_numerical_score_perturbation"]+after["unstable_interval_numerical_score_perturbation"])
    miss="missed_resolved_unstable";false="false_los_resolved_stable"
    gain=gain or after["classification_8s"][miss]<before["classification_8s"][miss]
    safe_rows=(b<=1.05*a and all(after["classification_8s"][k]<=before["classification_8s"][k] for k in (miss,false))
      and all(after["horizon_risks"][h]["value"]<=1.05*before["horizon_risks"][h]["value"] for h in ("5","8")))
    return {**r,"extend":bool(gain and safe_rows),"registered_key_event_gain":bool(gain),"nonworsening_guards":bool(safe_rows)}


def retention(F,C0,C1,*,optimization_unresolved=False):
    result=dict(A=False,B=False,second_seed=False,comparison="common96",optimization_unresolved=optimization_unresolved)
    if optimization_unresolved or not all(valid(p) and p.get("all_required_rows_qualified",False) for p in (F,C0,C1)):
        return {**result,"reason":"required_evidence_unresolved"}
    miss="missed_resolved_unstable";false="false_los_resolved_stable"
    metrics=lambda p:[p["horizon_risks"][h]["value"] for h in ("5","8")]+[p["common_frequency_rmse_8s"],p["normalized_p_rmse_8s"]]
    if any(v is None for p in (F,C0,C1) for v in metrics(p)): return {**result,"reason":"missing_RMSE_or_risk"}
    if any(p.get("parameter_count") is None or p.get("qualified_inference_seconds_per_event") is None for p in (F,C0,C1)):
        return {**result,"reason":"missing_complexity_or_qualified_cost"}
    guards=all(C1["classification_8s"][false]<=p["classification_8s"][false] and all(a<=1.05*b for a,b in zip(metrics(C1),metrics(p))) for p in (F,C0))
    A=guards and C1["classification_8s"][miss]<=C0["classification_8s"][miss]-1 and C1["classification_8s"][miss]<=F["classification_8s"][miss] and all(C1["unstable_interval_risk"]<=p["unstable_interval_risk"] for p in (F,C0))
    B=guards and all(C1["classification_8s"][miss]<=p["classification_8s"][miss] and C1["unstable_interval_risk"]<=.95*p["unstable_interval_risk"]
       and p["unstable_interval_risk"]-C1["unstable_interval_risk"]>p["unstable_interval_numerical_score_perturbation"]+C1["unstable_interval_numerical_score_perturbation"] for p in (F,C0))
    # C1 has one additional free scalar; C0 cannot dominate a strict A/B gain.
    return {**result,"A":bool(A),"B":bool(B),"second_seed":bool(A or B),"C0_all_capability_cost_complexity_dominance":False if A or B else None,
            "claim":"missed-event benefit" if A else "unstable-trajectory benefit" if B else "no registered retention"}


def device_comparison(variant,events,device):
    device=resolve_device(device)
    if device.type!="cuda": raise ValueError("actual CUDA required; CPU is not a substitute")
    models=[build_model(variant,"cpu"),build_model(variant,device)]
    h=np.stack([e["x"][:11] for e in events]);states=np.column_stack((h[:,-1],np.full((len(h),10),1e-5)))
    values=[];gradients=[]
    for m in models:
        tensors={"field":m.field(m._tensor(states)),"initialize":m.initialize(h),"short_RK4":m.rollout(h,4,.01)}
        with graph(m):
            J,_=objective(m,EventStore(events,m.mass.device),[(e,10) for e in events],.02)
            J.backward()
        tensors["J"]=J
        values.append({k:v.detach().cpu() for k,v in tensors.items()})
        gradients.append({n:p.grad.detach().cpu() for n,p in m.named_parameters() if p.requires_grad and p.grad is not None})
        if set(gradients[-1])!={n for n,p in m.named_parameters() if p.requires_grad}: raise ValueError("missing raw gradient")
    errors={}
    for name in values[0]:
        torch.testing.assert_close(values[0][name],values[1][name],rtol=1e-8,atol=1e-10)
        errors[name]=float((values[0][name]-values[1][name]).abs().max())
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=1e-8,atol=1e-10)
        if not torch.isfinite(gradients[1][name]).all(): raise ValueError("nonfinite raw gradient")
        errors["gradient_"+name]=float((gradients[0][name]-gradients[1][name]).abs().max())
    return {"variant":variant,"status":"PASS","differences":errors,"atol":1e-10,"rtol":1e-8,"optimizer_calls":0}


def _source_delta(F,C0,events):
    rows=[]
    for start in range(0,len(events),16):
        block=events[start:start+16];h=np.stack([e["x"][:11] for e in block])
        with torch.no_grad():
            f,c=F.history_system(h),C0.history_system(h)
            rf=f["s"]@f["A"].T-f["bscaled"];rc=c["s"]@c["A"].T-c["bscaled"]
            rows.append({"event_ids":[e["id"] for e in block],"initial_state_maxabs_delta":float((F.initialize(h)-C0.initialize(h)).abs().max()),
                "scaled_momentum_residual_F":rf.tolist(),"scaled_momentum_residual_C0":rc.tolist(),
                "physical_force_integral_residual_F":(rf.reshape(-1,5,10)*(.02*F.rstar)).tolist(),
                "physical_force_integral_residual_C0":(rc.reshape(-1,5,10)*(.02*C0.rstar)).tolist()})
    return rows


def _exact_event_ids(actual,expected):
    if (len(expected)!=297 or len(set(expected))!=297 or len(actual)!=297
            or len(set(actual))!=297 or set(actual)!=set(expected)):
        raise ValueError("cross-model cache must contain exactly the fixed 297 event IDs once")


def _fine_cache(evidence,events,expected_ids):
    """Read each existing prediction artifact once; never invoke a model."""
    _exact_event_ids([e["id"] for e in events],expected_ids)
    if evidence.get("event_count")!=297: raise ValueError("cross-model evidence count differs")
    byid={e["id"]:e for e in events};result={}
    for entry in evidence["prediction_files"]:
        if sha(entry["path"])!=entry["sha256"]: raise ValueError("cross-model prediction hash changed")
        with np.load(entry["path"],allow_pickle=False) as data:
            ids=[str(eid) for eid in data["event_ids"]]
            paths=data["fine"];histories=data["histories"]
            if len(paths)!=len(ids) or len(histories)!=len(ids): raise ValueError("cache row alignment differs")
            for eid,h,path in zip(ids,histories,paths):
                if eid not in byid or eid in result: raise ValueError("wrong or duplicate cache event ID")
                if h.shape!=(11,30) or not np.array_equal(h,byid[eid]["x"][:11]):
                    raise ValueError("cached causal history or offset differs")
                if path.shape!=(3001,40): raise ValueError("expected complete .005 RK4 path through15s")
                result[eid]=path.copy()
    _exact_event_ids(list(result),expected_ids)
    return result


def _cross_event(model,event,Ffine,C0fine):
    from evaluate_structured_rom import _pair_row,_scored_numerics
    from fixed_edge_pair_20261019 import scientific_qualification
    from active_fom_pair_fit_20261020 import interval15
    from frequency_dissipation_20261021 import actual15_qualified
    mass=model.mass.detach().cpu().numpy();history=event["x"][:11]
    n=min(1500,len(event["x"])-11)
    if n<=0 or Ffine.shape!=(3001,40) or C0fine.shape!=(3001,40):
        raise ValueError("cross-model actual interval/path shape unavailable")
    intervals={}
    for scope,count in (("8s",800),("actual15",n)):
        a,b=Ffine[:2*count+1],C0fine[:2*count+1]
        pair_ab=_pair_row(a,b[::2],history,event,event["classification_meta"],mass)
        pair_ba=_pair_row(b,a[::2],history,event,event["classification_meta"],mass)
        scores_ab=_scored_numerics(a,b,event,mass,stride=2,coarse_stride=2)
        scores_ba=_scored_numerics(b,a,event,mass,stride=2,coarse_stride=2)
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        qualified=(finite and scientific_qualification(pair_ab,scores_ab)
                    and scientific_qualification(pair_ba,scores_ba))
        row=dict(actual_output_steps=count,full_F_fine_finite=bool(np.isfinite(a).all()),
            full_C0_fine_finite=bool(np.isfinite(b).all()),F_to_C0_pair=pair_ab,C0_to_F_pair=pair_ba,
            F_to_C0_scored_numerics=scores_ab,C0_to_F_scored_numerics=scores_ba)
        if scope=="actual15":
            physical_ab=interval15(model,event,a,b[::2]);physical_ba=interval15(model,event,b,a[::2])
            qualified=qualified and actual15_qualified(physical_ab,pair_ab) and actual15_qualified(physical_ba,pair_ba)
            row.update(F_to_C0_actual_interval=physical_ab,C0_to_F_actual_interval=physical_ba)
        intervals[scope]={**row,"qualified":bool(qualified)}
    return dict(event_id=event["id"],label=event["label"],actual_steps=n,intervals=intervals,
                qualified=all(r["qualified"] for r in intervals.values()))


def cross_model_qualification(model,events,F_evidence,C0_evidence,prepared,output):
    expected=prepared["fit_ids"]
    result=dict(status="UNRESOLVED",qualified=False,record_count=0,event_ids=list(expected),source=prepared["source"],
        scope="F_C0_same_source_fit297_bidirectional_RK4_fine005_base01_8s_actual15",
        bidirectional=True,predictions_recomputed=False,alignment=dict(fine_dt=.005,output_dt=.01,stride=2,coarse_stride=2,
            history_rows=11,history_dt=.01,initial_observation_index=10,eight_second_shapes=[1601,801],
            actual_interval="n=min(1500,len(x)-11); 2n+1 fine and n+1 decimated; original classification/history offset"),
        F_prediction_files=F_evidence.get("prediction_files",[]),C0_prediction_files=C0_evidence.get("prediction_files",[]))
    rows=[]
    try:
        f=_fine_cache(F_evidence,events,expected);c=_fine_cache(C0_evidence,events,expected)
        byid={e["id"]:e for e in events}
        for eid in expected:
            try: row=_cross_event(model,byid[eid],f[eid],c[eid])
            except Exception as exc: row=dict(event_id=eid,qualified=False,error=repr(exc))
            rows.append(row)
        result["record_count"]=len(rows)
        result["aggregates"]={scope:dict(event_count=len(rows),qualified_count=sum(r.get("intervals",{}).get(scope,{}).get("qualified",False) for r in rows)) for scope in ("8s","actual15")}
        result["qualified"]=len(rows)==297 and all(r["qualified"] for r in rows)
        if result["qualified"]: result["status"]="PASS"
    except Exception as exc: result["error"]=repr(exc)
    result["rows"]=rows
    write(output,safe(result),exclusive=True)
    return result


def qualify(base,device="cuda"):
    device=resolve_device(device)
    if device.type!="cuda": raise ValueError("real CUDA qualification required")
    base=Path(base);prepared,fit=checked_preparation(base);output=base/"qualification";output.mkdir(exist_ok=False)
    cpu_record=read(ROOT/"results/force_exchange_20261028/matched_control_cpu_qualification.json")
    if cpu_record["status"]!="PASS_FOR_MATCHED_CONTROLS" or not cpu_record["immutable_verified"]:
        raise ValueError("independent CPU physical reference qualification missing")
    fixture=old.synthetic_fixture()
    comparisons=[device_comparison(v,fixture,device) for v in VARIANTS]
    # Same-device gamma-zero matching remains tighter than device parity.
    matched=[]
    for dev in ("cpu",device):
        c0,c1=[build_model(v,dev) for v in VARIANTS];h=np.stack([e["x"][:11] for e in fixture]);grads=[];losses=[]
        for m in (c0,c1):
            with graph(m):
                J,_=objective(m,EventStore(fixture,m.mass.device),[(e,10) for e in fixture],.02);J.backward()
            losses.append(J.detach());grads.append({n:p.grad for n,p in m.named_parameters() if p.requires_grad})
        for a,b in ((c0.initialize(h),c1.initialize(h)),(c0.field(c0.initialize(h)),c1.field(c1.initialize(h))),
                    (c0.rollout(h,4,.01),c1.rollout(h,4,.01)),tuple(losses)):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        for name in grads[0]: torch.testing.assert_close(grads[0][name],grads[1][name],rtol=1e-12,atol=1e-12)
        if not torch.isfinite(grads[1]["raw_gamma"]).all() or float(grads[1]["raw_gamma"].abs())<=1e-12:
            raise ValueError("nonfinite or locked gamma derivative on qualification fixture")
        matched.append({"device":str(dev),"status":"PASS","atol":1e-12,"rtol":1e-12})
    F,C0=build_model("F"),build_model("C0")
    delta=_source_delta(F,C0,fit)
    f=evaluate(F,fit,output/"F_fit297",prepared)
    c=evaluate(C0,fit,output/"C0_zero_fit297",prepared)
    cross_path=output/"F_C0_cross_model.json"
    cross=cross_model_qualification(F,fit,f,c,prepared,cross_path)
    selected=[e for eid in prepared["cost12_ids"] for e in fit if e["id"]==eid]
    costs=[]
    for variant in VARIANTS:
        m=build_model(variant,device);store=EventStore(selected,device);torch.cuda.reset_peak_memory_stats(device)
        synchronize(device);tick=time.perf_counter()
        with graph(m):
            J,_=objective(m,store,[(e,10) for e in selected],8.);J.backward()
        synchronize(device)
        ok=bool(torch.isfinite(J)) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        costs.append(dict(variant=variant,finite_complete_8s_backward=ok,seconds=time.perf_counter()-tick,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device),optimizer_calls=0,event_ids=prepared["cost12_ids"],objective_J=float(J.detach())))
    zero_ok=cross["qualified"] and all(valid(e["decision_profile"]) and e["decision_profile"]["all_required_rows_qualified"] for e in (f,c))
    required=zero_ok and all(r["finite_complete_8s_backward"] for r in costs)
    result=dict(status="COMPLETED" if required else "UNRESOLVED",training_method="rk4",device_comparisons=comparisons,
        same_device_matched=matched,cost12=costs,F_profile=f["decision_profile"],C0_zero_profile=c["decision_profile"],
        initializer_comparison=delta,old_legacy_drop_in_qualification="FAILED_NOT_ADOPTED",strict_legacy_equivalence_claim=False,
        C0_zero_minus_F="separate finite-precision initialization comparison; not a gamma effect",optimizer_calls=0,
        no_update_payload_read=False,val_read=False,test_read=False,training_admission=bool(required),
        cpu_cuda_pair_pass=True,matched_zero_pass=True,F_C0_zero_qualified=bool(zero_ok),
        cross_model_F_C0_qualified=bool(cross["qualified"]),cross_model_F_C0=dict(path=str(cross_path),sha256=sha(cross_path)),
        independent_physics_reference_reused=prepared["code"]["results/force_exchange_20261028/matched_control_cpu_qualification.json"],
        fullfit297_qualified=all(e["decision_profile"]["all_required_rows_qualified"] for e in (f,c)),
        finite_12x8_backward=all(r["finite_complete_8s_backward"] for r in costs))
    write(output/"summary.json",result,exclusive=True);return result


def qa(base,device="cuda"):
    base=Path(base);device=resolve_device(device)
    if device.type!="cuda" or not read(base/"qualification/summary.json")["training_admission"]: raise ValueError("actual CUDA qualification prerequisite")
    output=base/"qa";output.mkdir(exist_ok=False);events=old.synthetic_fixture();store=EventStore(events,device);results=[]
    for variant in VARIANTS:
        folder=output/variant;folder.mkdir();identity=dict(variant=variant,fixture_only=True,source=source_identity(),code=code_identity())
        def initial():
            m=build_model(variant,device);opt=make_optimizer(m);fg._seed(SEEDS[0],device);rng=np.random.default_rng(SEEDS[0])
            return m,opt,rng,{"cover_order":rng.permutation(len(events)).tolist(),"cursor":0}
        def advance(m,opt,rng,cov,start,end,label):
            trace=[]
            for step in range(start+1,end+1):
                samples,_=fg.sample_step(events,rng,cov)
                random.random();np.random.random();torch.rand((),device=device)
                trace.append(one_update(m,opt,store,samples,.02,base=base,category="qa",run_id=f"qa_{variant}_{label}",step=step))
            return trace
        a,oa,ra,ca=initial();current=(a,oa,ra,ca);trace=[]
        try:
            trace=advance(a,oa,ra,ca,0,18,"continuous");rng_final=old._cpu_snapshot(fg._rng_state(ra,ca))
            b,ob,rb,cb=initial();current=(b,ob,rb,cb);trace=advance(b,ob,rb,cb,0,16,"restored")
            snapshot=old._cpu_snapshot(dict(model=b.state_dict(),Adam=ob.state_dict(),RNG=fg._rng_state(rb,cb),layout=old._parameter_layout(b,ob)))
            if not any(bool(torch.count_nonzero(s["exp_avg"])) for s in ob.state.values()): raise ValueError("real Adam moment required")
            checkpoint=save_checkpoint(folder/"checkpoints",b,ob,rb,cb,16,identity,trace)
            c,oc,rc,cc,meta=restore_checkpoint(folder/"checkpoints",device,identity);current=(c,oc,rc,cc)
            old._exact_nested(snapshot,old._cpu_snapshot(dict(model=c.state_dict(),Adam=oc.state_dict(),RNG=fg._rng_state(rc,cc),layout=old._parameter_layout(c,oc))))
            trace+=advance(c,oc,rc,cc,16,18,"restored")
            pe=old._qa_compare(old._cpu_snapshot(a.state_dict()),old._cpu_snapshot(c.state_dict()))
            sa,sc=old._cpu_snapshot(oa.state_dict()),old._cpu_snapshot(oc.state_dict())
            old._exact_nested(sa["param_groups"],sc["param_groups"])
            ae=max(old._qa_compare(sa["state"][k],sc["state"][k]) for k in sa["state"])
            old._exact_nested(rng_final,old._cpu_snapshot(fg._rng_state(rc,cc)))
            row=dict(variant=variant,status="PASS",actual_optimizer_calls=36,checkpoint=str(checkpoint),parameter_maxabs=pe,Adam_maxabs=ae,
                     snapshot_model_Adam_RNG_layout_exact=True,nonzero_Adam_moment=True,continuation_atol=2e-9,continuation_rtol=2e-9)
            results.append(row);write(folder/"result.json",row,exclusive=True)
        except BaseException as exc:
            quarantine=old.quarantine_failure(folder,*current,len(trace),trace,exc)
            write(folder/"failure.json",dict(error=repr(exc),quarantine=quarantine,calls=call_counts(base)),exclusive=True);raise
    counts=call_counts(base)
    if counts["pending"] or counts["by_category"].get("qa")!={"attempts":72,"completed":72}: raise ValueError("QA actual ledger differs")
    result=dict(status="PASS",models=results,calls=counts,optimizer_calls=72,actual_optimizer_updates=72,fixture_only=True,training_admission=True,
                admission_scope="formal matched C0/C1 only after qualification; no legacy adoption",val_read=False,test_read=False)
    write(output/"summary.json",result,exclusive=True);return result


def formal_run(base,variant,seed,prepared,fit,panel,device):
    if variant not in VARIANTS or seed not in SEEDS: raise ValueError("unregistered run")
    base=Path(base);run_id=f"{variant}_{seed}";folder=base/"study"/run_id;folder.mkdir(parents=True,exist_ok=False)
    m=build_model(variant,device);opt=make_optimizer(m);fg._seed(seed,m.mass.device)
    schedule=read(base/"schedule.json")[str(seed)];rng=np.random.default_rng();rng.bit_generator.state=schedule["initial"]["rng"]
    cov=copy.deepcopy(schedule["initial"]["coverage"]);trace=[];profiles={};full={}
    identity=dict(variant=variant,seed=seed,source=prepared["source"],code=prepared["code"],schedule_sha256=prepared["schedule_sha256"],prepare_sha256=sha(base/"prepare.json"))
    store=EventStore(fit,m.mass.device);lookup={e["id"]:e for e in fit}
    try:
        profiles["0"]=score_panel(m,panel);write(folder/"panel000000.json",profiles["0"],exclusive=True)
        if not valid(profiles["0"]): raise ValueError("initial panel unresolved; no training")
        save_checkpoint(folder/"checkpoints",m,opt,rng,cov,0,identity,trace);end=96
        for row in schedule["rows"]:
            step=row["step"]
            if step>end: break
            trace.append(one_update(m,opt,store,[(lookup[eid],10) for eid in row["sample_ids"]],row["horizon_s"],
                                    base=base,category="formal",run_id=run_id,step=step))
            rng.bit_generator.state=row["rng_after"];cov=copy.deepcopy(row["coverage_after"])
            if step%16==0: save_checkpoint(folder/"checkpoints",m,opt,rng,cov,step,identity,trace)
            if step in (48,96,192):
                profiles[str(step)]=score_panel(m,panel);write(folder/f"panel{step:06d}.json",profiles[str(step)],exclusive=True)
            if step in (96,192):
                evidence=evaluate(m,fit,folder/f"fit{step:06d}",prepared)
                full[str(step)]=evidence["decision_profile"]
            if step==96:
                decision=extension_decision(profiles["48"],profiles["96"],trace)
                write(folder/"extension_decision.json",decision,exclusive=True)
                if decision["extend"]: end=192
        result=dict(status="COMPLETED",run_id=run_id,variant=variant,seed=seed,updates=len(trace),
          endpoint=str(folder/"checkpoints"/f"step{len(trace):06d}"),fit_endpoints=full,extension_decision=decision,
          optimization_unresolved_at96=optimization(trace[:96])["optimization_unresolved"],
          upper_bound_still_improving=extension_decision(profiles["96"],profiles["192"],trace)["extend"] if len(trace)==192 else False,
          finite_optimization_scope=True,no_update_payload_read=False,val_read=False,test_read=False)
        write(folder/"result.json",result,exclusive=True);return result
    except BaseException as exc:
        quarantine=old.quarantine_failure(folder,m,opt,rng,cov,len(trace),trace,exc)
        result=dict(status="UNRESOLVED",run_id=run_id,variant=variant,seed=seed,updates=len(trace),error=repr(exc),quarantine=quarantine,
                    failure_is_physical_instability=False,resumable=False)
        write(folder/"result.json",result,exclusive=True);return result


def study(base,device="cuda"):
    base=Path(base)
    if resolve_device(device).type!="cuda" or not read(base/"qa/summary.json")["training_admission"]:
        raise ValueError("real CUDA/QA admission missing")
    prepared,fit=checked_preparation(base,fit_only=True);panel=[e for eid in prepared["panel_ids"] for e in fit if e["id"]==eid]
    output=base/"study";output.mkdir(exist_ok=False);runs=[];decisions=[]
    F=read(base/"qualification/summary.json")["F_profile"]
    for seed in SEEDS:
        pair=[formal_run(base,v,seed,prepared,fit,panel,device) for v in VARIANTS];runs.extend(pair)
        if all(r["status"]=="COMPLETED" for r in pair):
            decision=retention(F,*[r["fit_endpoints"]["96"] for r in pair],
                               optimization_unresolved=any(r["optimization_unresolved_at96"] for r in pair))
            if all("192" in r["fit_endpoints"] and valid(r["fit_endpoints"]["192"])
                   and r["fit_endpoints"]["192"]["all_required_rows_qualified"] for r in pair):
                decision["common192_descriptive_only"]={v:r["fit_endpoints"]["192"] for v,r in zip(VARIANTS,pair)}
        else: decision=dict(second_seed=False,reason="required_common96_unresolved")
        decisions.append(dict(seed=seed,**decision));write(output/f"retention_{seed}.json",decisions[-1],exclusive=True)
        if seed==SEEDS[0] and not decision["second_seed"]: break
    # Freeze every fit-only seed/extension decision before any train88 payload.
    write(output/"fit_decisions_closed.json",dict(status="COMPLETED",decisions=decisions,run_ids=[r["run_id"] for r in runs]),exclusive=True)
    _,all_events=checked_preparation(base,fit_only=False);held=[e for e in all_events if e["id"] in prepared["no_update_ids"]]
    if len(held)!=88 or sum(e["label"]=="unstable" for e in held)!=13: raise ValueError("report75/13 changed")
    reports=[]
    for r in runs:
        if r["status"]!="COMPLETED": continue
        m=ForceExchangeROM.from_export(Path(r["endpoint"])/"model.json").cpu().double()
        folder=output/r["run_id"]
        full=evaluate(m,all_events,folder/"evaluation385",prepared)
        noisy=evaluate(m,held,folder/"noise88",prepared,noisy=True)
        reports.append(dict(run_id=r["run_id"],full385=str(folder/"evaluation385/evidence.json"),
          noise88=str(folder/"noise88/evidence.json"),all_required_rows_qualified=full["decision_profile"]["all_required_rows_qualified"],
          noise_event_count=noisy["event_count"],noise_selects_nothing=True))
    result=dict(status="COMPLETED" if all(r["status"]=="COMPLETED" for r in runs) else "UNRESOLVED",runs=runs,
        common96_decisions=decisions,reports=reports,calls=call_counts(base),default_model=None,scientific_model_adoption=False,
        old_legacy_drop_in_qualification="FAILED_NOT_ADOPTED",val_read=False,test_read=False)
    write(output/"summary.json",result,exclusive=True);return result


def summary(base):
    base=Path(base);paths={"prepare":"prepare.json","qualification":"qualification/summary.json","qa":"qa/summary.json","study":"study/summary.json"}
    stages={k:dict(path=str(base/p),sha256=sha(base/p),status=read(base/p).get("status")) if (base/p).exists() else None for k,p in paths.items()}
    result=dict(status=stages["study"]["status"] if stages["study"] else "INCOMPLETE",stages=stages,calls=call_counts(base),
      default_model=None,scientific_model_adoption=False,old_legacy_drop_in_qualification="FAILED_NOT_ADOPTED",val_read=False,test_read=False,
      no_process_count_or_runtime_limit=True)
    write(base/"summary.json",result);return result


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=("prepare","qualify","qa","study","summary"))
    parser.add_argument("--base",type=Path,default=ROOT/"private_data/force_exchange_20261028/study01")
    parser.add_argument("--device",default="cuda");args=parser.parse_args(argv)
    fn=globals()[args.phase]
    result=fn(args.base) if args.phase in ("prepare","summary") else fn(args.base,args.device)
    print(json.dumps(safe(result),allow_nan=False))
    return 0 if result["status"] in ("PASS","COMPLETED") else 1


if __name__=="__main__": raise SystemExit(main())
