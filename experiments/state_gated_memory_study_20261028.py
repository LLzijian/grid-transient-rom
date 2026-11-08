"""Finite state-gated memory comparison. Import never reads event payloads or runs optimizers."""
from __future__ import annotations
import copy,json,math,random,sys,time,uuid,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"src")]
import numpy as np
import torch
import force_graph_study_20261028 as fg
import force_exchange_study_20261028 as fx
from force_graph_model_20261028 import ForceGraphROM
from state_gated_memory_model_20261028 import StateGatedMemoryROM,ControlDegenerate
from closure_training_runtime import resolve_device,synchronize
from train_local_memory_0098 import graph,risk
from train_structured_rom import _point_error
old=fg.old
sha,read,write,safe=fg.sha,fg.read,fg.write,fg.safe
SEEDS=(28311,28312)
VARIANTS=("A","B","C")
PROTOCOL=ROOT/"reports/state_gated_memory_20261028_protocol.md"
PARENT=ROOT/"private_data/force_graph_20261028/study01"
SOURCE=PARENT/"study/N6101_M_28102/checkpoints/step000192"
SOURCE_SHA={"model.json":"937f00b740875913923487b8d772f40d2a8145a88eba2cdaf61f2d889715bfc9","checkpoint.json":"2083fcbd3f25679350b7b63f2d7285412995a9ae141be1fb2a4a5cb8f60d0ff9","training.pt":"197ab52b8061de7319f08345ffe3829036caa38c9fcbd7594a0c6936f0ea210b"}
LIMITS={"qa":108,"formal":1152,"repair":108,"replay":96}
EventStore=fg.EventStore
objective=fg.objective
horizon_for=fg.horizon_for

def code_identity():
    paths=(Path(__file__),ROOT/"experiments/state_gated_memory_model_20261028.py",PROTOCOL,ROOT/"results/state_gated_memory_20261028/model_pre_freeze_qa.json")
    return {**fg.code_identity(),**{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}

def source_identity():
    actual={n:sha(SOURCE/n) for n in SOURCE_SHA}
    if actual!=SOURCE_SHA:raise ValueError("fixed R_N assets changed")
    return {"name":"R_N","directory":str(SOURCE),"files":actual,"parent_implementation":"ForceGraphROM"}

def build_model(variant,device="cpu"):
    source_identity()
    if variant=="R_N":model=ForceGraphROM.from_export(SOURCE/"model.json")
    elif variant in VARIANTS:model=StateGatedMemoryROM.from_force_graph_export(SOURCE/"model.json",variant,history_dt=HISTORY_DT_MAIN)
    else:raise ValueError("R_N/A/B/C only")
    return model.to(resolve_device(device)).double()

def prepare(base):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    if (base/"prepare.json").exists():raise FileExistsError("immutable preparation")
    parent,events=fg.checked_preparation(PARENT,fit_only=True)
    if len(events)!=297 or sum(e["label"]=="unstable" for e in events)!=30:raise ValueError("fit297 changed")
    panel=fg.panel_events(events);selected=sorted([e for e in panel if e["label"]=="stable"],key=lambda e:(-len(e["x"]),e["id"]))[:12]
    if len(selected)!=12 or any(len(e["x"])-11<800 for e in selected):raise ValueError("fixed12 complete8s unavailable")
    write(base/"schedule.json",{str(s):make_schedule(events,s) for s in SEEDS},exclusive=True)
    record={k:parent[k] for k in ("fit_ids","no_update_ids","original366_ids","fresh19_ids","fit_groups","no_update_groups","fit_cache_sha256","no_update_cache_sha256")}
    record.update(status="COMPLETED",source=source_identity(),code=code_identity(),parent_prepare_sha256=sha(PARENT/"prepare.json"),schedule_sha256=sha(base/"schedule.json"),panel_ids=[e["id"] for e in panel],cost12_ids=[e["id"] for e in selected],source_exposed_to_train385=True,no_update_payload_read=False,val_read=False,test_read=False)
    write(base/"prepare.json",record,exclusive=True);return record

def checked_preparation(base,*,fit_only=True):
    base=Path(base);p=read(base/"prepare.json")
    if p["source"]!=source_identity() or p["code"]!=code_identity() or p["schedule_sha256"]!=sha(base/"schedule.json") or p["parent_prepare_sha256"]!=sha(PARENT/"prepare.json"):raise ValueError("source/code/schedule changed")
    _,events=fg.checked_preparation(PARENT,fit_only=fit_only);return p,events


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
        if counts["attempts"]>=1464 or counts["by_category"].get(category,{}).get("attempts",0)>=LIMITS[category]:
            raise ValueError("registered optimizer-call bound exhausted")
        if category=="formal" and not 1<=step<=192: raise ValueError("formal step outside finite run")
    elif event!="optimizer_step_completed": raise ValueError("unknown optimizer event")
    row=dict(event=event,category=category,call_id=call_id,run_id=run_id,step=step,time_unix=time.time(),reserved_reason=reserved_reason)
    with (Path(base)/"optimizer_calls.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()

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
    model=StateGatedMemoryROM.from_export(folder/"model.json").to(device).double()
    if model.history_dt!=HISTORY_DT_MAIN:raise ValueError("checkpoint must retain registered main history step")
    opt=make_optimizer(model)
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

def optimization(trace):
    recent=trace[-48:];groups={n for r in recent for n in r["block_updates"]}
    clip=sum(r["gradient_norm"]>100 for r in recent)
    jumps={n:sum(r["block_updates"].get(n,{}).get("relative_l2",0)>.25 for r in recent) for n in groups}
    return dict(optimization_unresolved=len(recent)!=48 or clip>12 or any(v>=12 for v in jumps.values()),
                clip_count_last48=clip,large_change_counts_last48=jumps)

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

def _cross_event(model,event,R_N_fine,A_fine):
    from evaluate_structured_rom import _pair_row,_scored_numerics
    from fixed_edge_pair_20261019 import scientific_qualification
    from active_fom_pair_fit_20261020 import interval15
    from frequency_dissipation_20261021 import actual15_qualified
    mass=model.mass.detach().cpu().numpy();history=event["x"][:11]
    n=min(1500,len(event["x"])-11)
    if n<=0 or R_N_fine.shape!=(3001,40) or A_fine.shape!=(3001,40):
        raise ValueError("cross-model actual interval/path shape unavailable")
    intervals={}
    for scope,count in (("8s",800),("actual15",n)):
        a,b=R_N_fine[:2*count+1],A_fine[:2*count+1]
        pair_ab=_pair_row(a,b[::2],history,event,event["classification_meta"],mass)
        pair_ba=_pair_row(b,a[::2],history,event,event["classification_meta"],mass)
        scores_ab=_scored_numerics(a,b,event,mass,stride=2,coarse_stride=2)
        scores_ba=_scored_numerics(b,a,event,mass,stride=2,coarse_stride=2)
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        qualified=(finite and scientific_qualification(pair_ab,scores_ab)
                    and scientific_qualification(pair_ba,scores_ba))
        row=dict(actual_output_steps=count,full_R_N_fine_finite=bool(np.isfinite(a).all()),
            full_A_fine_finite=bool(np.isfinite(b).all()),R_N_to_A_pair=pair_ab,A_to_R_N_pair=pair_ba,
            R_N_to_A_scored_numerics=scores_ab,A_to_R_N_scored_numerics=scores_ba)
        if scope=="actual15":
            physical_ab=interval15(model,event,a,b[::2]);physical_ba=interval15(model,event,b,a[::2])
            qualified=qualified and actual15_qualified(physical_ab,pair_ab) and actual15_qualified(physical_ba,pair_ba)
            row.update(R_N_to_A_actual_interval=physical_ab,A_to_R_N_actual_interval=physical_ba)
        intervals[scope]={**row,"qualified":bool(qualified)}
    return dict(event_id=event["id"],label=event["label"],actual_steps=n,initial_state_maxabs_delta=float(np.max(np.abs(R_N_fine[0]-A_fine[0]))),intervals=intervals,
                qualified=all(r["qualified"] for r in intervals.values()))

def cross_model_qualification(model,events,R_N_evidence,A_evidence,prepared,output):
    expected=prepared["fit_ids"]
    result=dict(status="UNRESOLVED",qualified=False,record_count=0,event_ids=list(expected),source=prepared["source"],
        scope="R_N_A_same_source_fit297_bidirectional_RK4_fine005_base01_8s_actual15",
        bidirectional=True,predictions_recomputed=False,alignment=dict(fine_dt=.005,output_dt=.01,stride=2,coarse_stride=2,
            history_rows=11,history_dt=.01,initial_observation_index=10,eight_second_shapes=[1601,801],
            actual_interval="n=min(1500,len(x)-11); 2n+1 fine and n+1 decimated; original classification/history offset"),
        R_N_prediction_files=R_N_evidence.get("prediction_files",[]),A_prediction_files=A_evidence.get("prediction_files",[]))
    rows=[]
    try:
        f=_fine_cache(R_N_evidence,events,expected);c=_fine_cache(A_evidence,events,expected)
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
            if step%16==0 or trace[-1].get("control_degenerate",False): save_checkpoint(folder/"checkpoints",m,opt,rng,cov,step,identity,trace)
            if trace[-1].get("control_degenerate",False): raise ControlDegenerate("B control degenerate after durable completed update")
            if step in (48,96,192):
                profiles[str(step)]=score_panel(m,panel);write(folder/f"panel{step:06d}.json",profiles[str(step)],exclusive=True)
            if step in (96,192):
                evidence=evaluate(m,fit,folder/f"fit{step:06d}",prepared,mechanism=(variant=="C" and step==96))
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
        result=dict(status="CONTROL_DEGENERATE" if isinstance(exc,ControlDegenerate) else "UNRESOLVED",optimization_unresolved=True,run_id=run_id,variant=variant,seed=seed,updates=len(trace),error=repr(exc),quarantine=quarantine,
                    failure_is_physical_instability=False,resumable=False)
        write(folder/"result.json",result,exclusive=True);return result


def one_update(model,opt,store,samples,horizon,*,base,category,run_id,step):
    synchronize(model.mass.device);tick=time.perf_counter();opt.zero_grad(set_to_none=True)
    with graph(model):
        J,terms=objective(model,store,samples,horizon,method="rk4")
        if not torch.isfinite(J):raise FloatingPointError("nonfinite J")
        J.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        if model.gate_mode=="B":raise ControlDegenerate("B weighted-ratio gradient unqualified")
        raise FloatingPointError("nonfinite/missing gradient")
    norm=torch.nn.utils.clip_grad_norm_(params,100.)
    if not torch.isfinite(norm):raise FloatingPointError("nonfinite norm")
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};cid=uuid.uuid4().hex
    _journal(base,"optimizer_step_attempt",category,call_id=cid,run_id=run_id,step=step);opt.step()
    _journal(base,"optimizer_step_completed",category,call_id=cid,run_id=run_id,step=step)
    degenerate=False
    try:model.project_runtime_()
    except ControlDegenerate:degenerate=True
    synchronize(model.mass.device)
    if any(not torch.isfinite(p).all() for p in params) or not old._adam_states_finite(opt.state.values()):raise FloatingPointError("post-update nonfinite state")
    return dict(step=step,horizon_s=horizon,method="rk4",**{k:float(v.detach()) for k,v in terms.items()},gradient_norm=float(norm),
      block_updates={n:{"relative_l2":float((p.detach()-before[n]).norm()/before[n].norm().clamp_min(1.))} for n,p in model.named_parameters() if p.requires_grad},
      seconds=time.perf_counter()-tick,control_degenerate=degenerate)


def confirmation(pair,numeric):
    if not numeric:return dict(time_s=None,reason="numerical_unresolved")
    if pair.get("category")=="latency_uncovered":return dict(time_s=None,reason="latency_uncovered")
    k=pair.get("trigger_index")
    if not pair.get("predicted_los") or k is None:return dict(time_s=None,reason="no_confirmed_trigger")
    return dict(time_s=.01*(int(k)-10),confirmation_index=int(k),reason=None)


def actual_extra(model,event,history,fine,coarse):
    from evaluate_structured_rom import _pair_row,_classification
    from active_fom_pair_fit_20261020 import interval15
    from frequency_dissipation_20261021 import actual15_qualified
    n=min(1500,len(event["x"])-11);f,c=fine[:2*n+1],coarse[:n+1];mass=model.mass.detach().numpy()
    pair=_pair_row(f,c,history,event,event["classification_meta"],mass)
    actual=interval15(model,event,f,c);qualified=actual15_qualified(actual,pair)
    value=None
    if qualified:
        with torch.no_grad():value=float(_point_error(torch.as_tensor(c[1:n+1,:30]),torch.as_tensor(event["x"][11:n+11]),model).mean())
    truth=event["x"][10:n+11]
    tclass=_classification(truth,event["x"][:11],event["classification_meta"],mass)
    return dict(classification_15s=pair,actual_interval15=actual,actual15_coarse_point_error=value,
      timing_actual15={"truth":confirmation(tclass,bool(np.isfinite(truth).all())),
        "fine":confirmation(pair["fine_classification"],qualified),"coarse":confirmation(pair["coarse_classification"],qualified),
        "numerically_qualified":qualified,"actual_output_steps":n})


def profile(records,extras,events,seconds,parameter_count):
    from frequency_dissipation_20261021 import actual15_qualified
    p=fx.profile(records,extras,events,seconds,parameter_count);byid={x["event_id"]:x for x in extras}
    unstable=[e for e in events if e["label"]=="unstable"]
    bad=[e["id"] for e in unstable if not actual15_qualified(byid[e["id"]]["actual_interval15"],byid[e["id"]]["classification_15s"]) or byid[e["id"]].get("actual15_coarse_point_error") is None]
    fine=coarse=None
    if unstable and not bad:
        fine=float(risk(torch.tensor([byid[e["id"]]["actual_interval15"]["mean_point_error"] for e in unstable],dtype=torch.float64)))
        coarse=float(risk(torch.tensor([byid[e["id"]]["actual15_coarse_point_error"] for e in unstable],dtype=torch.float64)))
    p.update(classification_actual15s=old._classification_counts(extras,"classification_15s",{}),
      unstable_actual15_risk=fine,unstable_actual15_coarse_risk=coarse,
      unstable_actual15_numerical_score_perturbation=abs(fine-coarse) if fine is not None else None,
      unstable_actual15_eligible=len(unstable),unstable_actual15_unresolved_ids=bad,
      unstable_actual15_steps={e["id"]:min(1500,len(e["x"])-11) for e in unstable},
      timing_actual15={x["event_id"]:x["timing_actual15"] for x in extras},
      all_actual15_rows_qualified=all(actual15_qualified(x["actual_interval15"],x["classification_15s"]) for x in extras))
    p["all_required_rows_qualified"]=p["all_8s_rows_qualified"] and p["all_actual15_rows_qualified"]
    p["timing_scope"]="CPU float64 one thread; 15s RK4 .005/.01 including causal initialization, outputs and metrics"
    return p


def score_panel(model,events):
    from evaluate_structured_rom import _rollouts,_pair_row
    from summarize_train_local_memory_0098 import event_metrics
    from fixed_edge_pair_20261019 import scientific_qualification
    torch.set_num_threads(1);tick=time.perf_counter();m=copy.deepcopy(model).cpu().double().eval()
    records={k:[] for k in ("records","common_frequency","normalized_p","objective_point_error","numerical")};extras=[]
    for start in range(0,len(events),16):
        block=events[start:start+16];h=np.stack([e["x"][:11] for e in block]);f=_paired_rollouts(m,h,3000,.005);c=_paired_rollouts(m,h,1500,.01)
        for e,hh,ff,cc in zip(block,h,f,c):
            metrics=event_metrics(m,e,ff[:1601],cc[:801])
            for key,rows in metrics.items():records[key].extend(rows)
            pair=_pair_row(ff[:1601],cc[:801],hh,e,e["classification_meta"],m.mass.numpy())
            extras.append(dict(event_id=e["id"],label=e["label"],classification_8s=pair,scientific_numeric_8s=scientific_qualification(pair,metrics["numerical"]),coarse_point_errors=_coarse_points(m,e,cc),**actual_extra(m,e,hh,ff,cc)))
    return {**profile(records,extras,events,time.perf_counter()-tick,m.complexity_counts()["raw_trainable_scalar_count"]),"rows":safe(records),"extras":safe(extras),"event_ids":[e["id"] for e in events],"val_read":False,"test_read":False}


def mechanism_event(model,event,fine):
    n=min(1500,len(event["x"])-11);z=model._tensor(fine[::2][1:n+1]);d=z[:,:10];r=z[:,30:];a=copy.deepcopy(model);b=copy.deepcopy(model);a.gate_mode="A";b.gate_mode="B"
    with torch.no_grad():
        qc,qa,qb=model.memory_matrix(d),a.memory_matrix(d),b.memory_matrix(d)
        cb=((qb-qc)@r[...,None]).squeeze(-1)/model.rstar;ca=((qa-qc)@r[...,None]).squeeze(-1)/model.rstar;g=model.gate_values(d)
    return dict(event_id=event["id"],label=event["label"],actual_steps=n,C_minus_Bmatch_mean_abs=float(cb.abs().mean()),C_minus_Bmatch_max_abs=float(cb.abs().max()),C_minus_A_mean_abs=float(ca.abs().mean()),C_minus_A_max_abs=float(ca.abs().max()),g_edge_range_mean=float((g.max(-1).values-g.min(-1).values).mean()),g_min=float(g.min()),g_max=float(g.max()))


def evaluate(model,events,output,prepared,*,noisy=False,mechanism=False):
    result=paired_evaluate_model(model,events,output,prepared,noisy=noisy);m=copy.deepcopy(model).cpu().double().eval();byid={e["id"]:e for e in events};extras={e["event_id"]:e for e in result["extras"]};mechanisms=[]
    for entry in result["prediction_files"]:
        if sha(entry["path"])!=entry["sha256"]:raise ValueError("cached prediction changed")
        with np.load(entry["path"],allow_pickle=False) as data:
            for eid,h,f,c in zip(data["event_ids"],data["histories"],data["fine"],data["coarse"]):
                eid=str(eid);e=byid[eid];extras[eid].update(actual_extra(m,e,h,f,c));extras[eid]["coarse_point_errors"]=_coarse_points(m,e,c)
                if mechanism:mechanisms.append(mechanism_event(m,e,f))
    p=profile(result["rows"],list(extras.values()),events,result["seconds"],m.complexity_counts()["raw_trainable_scalar_count"])
    result["decision_profile"]=p;write(Path(output)/"state_gated_profile.json",p,exclusive=True)
    if mechanism:
        write(Path(output)/"mechanism.json",dict(scope="same C predicted states and C parameters; genuine up-to15 intervals; force-rate /rstar per-second",rows=mechanisms,groups={label:{key:float(np.mean([r[key] for r in mechanisms if r["label"]==label])) for key in ("C_minus_Bmatch_mean_abs","C_minus_Bmatch_max_abs","C_minus_A_mean_abs","C_minus_A_max_abs","g_edge_range_mean")} for label in ("stable","unstable")}),exclusive=True)
    return result


def valid(p):
    return fg._valid_profile(p) and p.get("all_required_rows_qualified") is True and p.get("unstable_actual15_risk") is not None and p.get("unstable_actual15_numerical_score_perturbation") is not None and not p.get("unstable_actual15_unresolved_ids")


def extension_decision(before,after,trace):
    opt=optimization(trace);guards=valid(before) and valid(after)
    if not guards:return {**opt,"extend":False,"reason":"required_numerics_unresolved"}
    a,b=before["unstable_actual15_risk"],after["unstable_actual15_risk"]
    gain=(after["classification_actual15s"]["missed_resolved_unstable"]<=before["classification_actual15s"]["missed_resolved_unstable"]-1 or (b<=.99*a and a-b>before["unstable_actual15_numerical_score_perturbation"]+after["unstable_actual15_numerical_score_perturbation"]))
    for key in ("classification_8s","classification_actual15s"):
        guards &= all(after[key][k]<=before[key][k] for k in ("missed_resolved_unstable","false_los_resolved_stable"))
    guards &= b<=1.05*a and all(after["horizon_risks"][h]["value"]<=1.05*before["horizon_risks"][h]["value"] for h in ("5","8"))
    guards &= all(after[k]<=1.05*before[k] for k in ("common_frequency_rmse_8s","normalized_p_rmse_8s"))
    return {**opt,"extend":bool(gain and guards and not opt["optimization_unresolved"]),"registered_actual15_gain":bool(gain),"nonworsening_guards":bool(guards)}


def timing_comparison(profiles):
    if set(profiles)!={"R_N","A","B","C"}:return dict(status="UNRESOLVED",common_event_ids=[],reason="missing_matched_control")
    ids=set.intersection(*(set(p["timing_actual15"]) for p in profiles.values()));common=[]
    for eid in sorted(ids):
        rows=[p["timing_actual15"][eid] for p in profiles.values()]
        if all(x["numerically_qualified"] and all(x[k]["time_s"] is not None for k in ("truth","fine","coarse")) for x in rows):common.append(eid)
    if not common:return dict(status="UNRESOLVED",common_event_ids=[],reason="empty_common_T")
    metrics={name:{"E":sum(abs(p["timing_actual15"][i]["fine"]["time_s"]-p["timing_actual15"][i]["truth"]["time_s"]) for i in common)/len(common),"u":max(abs(p["timing_actual15"][i]["fine"]["time_s"]-p["timing_actual15"][i]["coarse"]["time_s"]) for i in common)} for name,p in profiles.items()}
    ok=all(metrics["C"]["E"]<=metrics[j]["E"]+.01+metrics["C"]["u"]+metrics[j]["u"] for j in ("R_N","A","B"))
    return dict(status="PASS" if ok else "FAIL",common_event_ids=common,event_count=len(common),metrics=metrics,nonworsening=ok)


def retention(profiles,*,optimization_unresolved=False):
    result=dict(second_seed=False,comparison="common96_actual15",locality="UNRESOLVED")
    if set(profiles)!={"R_N","A","B","C"} or optimization_unresolved or not all(valid(p) for p in profiles.values()):return {**result,"reason":"missing_or_unresolved_control"}
    c=profiles["C"];ok=True
    for name in ("R_N","A","B"):
        p=profiles[name]
        ok &= c["classification_actual15s"]["missed_resolved_unstable"]<=p["classification_actual15s"]["missed_resolved_unstable"]-1
        ok &= c["classification_actual15s"]["false_los_resolved_stable"]<=p["classification_actual15s"]["false_los_resolved_stable"]
        ok &= c["classification_8s"]["missed_resolved_unstable"]<=p["classification_8s"]["missed_resolved_unstable"] and c["unstable_actual15_risk"]<=p["unstable_actual15_risk"]
        ok &= all(c["horizon_risks"][h]["value"]<=1.05*p["horizon_risks"][h]["value"] for h in ("5","8"))
        ok &= all(c[k]<=1.05*p[k] for k in ("common_frequency_rmse_8s","normalized_p_rmse_8s"))
    timing=timing_comparison(profiles)
    # Strictly fewer actual15 misses than every control prevents full capability dominance by A/B.
    return {**result,"second_seed":bool(ok and timing.get("nonworsening",False)),"locality":"PASS" if ok and timing.get("nonworsening",False) else "NOT_RETAINED","timing":timing,"actual15_and_accuracy_guards":bool(ok)}


def qa(base,device="cuda"):
    base=Path(base);device=resolve_device(device)
    if device.type!="cuda": raise ValueError("actual CUDA required")
    eligible=read(base/"qualification/summary.json")["eligible_variants"]
    output=base/"qa";output.mkdir(exist_ok=False);events=old.synthetic_fixture();store=EventStore(events,device);results=[]
    for variant in eligible:
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
                if trace[-1].get("control_degenerate",False):
                    save_checkpoint(folder/"control_degenerate_checkpoints",m,opt,rng,cov,step,identity,trace)
                    raise ControlDegenerate("QA B control degenerate after saved completed update")
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
            quarantine=old.quarantine_failure(folder,*current[:3],current[3],len(trace),trace,exc)
            write(folder/"failure.json",dict(error=repr(exc),quarantine=quarantine,calls=call_counts(base)),exclusive=True)
            results.append(dict(variant=variant,status="UNRESOLVED",error=repr(exc)))
    counts=call_counts(base)
    passed=[r["variant"] for r in results if r["status"]=="PASS"]
    result=dict(status="COMPLETED",models=results,calls=counts,actual_optimizer_updates=counts["by_category"].get("qa",{}).get("completed",0),eligible_variants=passed,training_admission=bool(passed),val_read=False,test_read=False)
    write(output/"summary.json",result,exclusive=True);return result


HISTORY_DT_MAIN=.0025
HISTORY_DT_REFERENCE=.00125

def _paired_rollouts(model,histories,steps,future_dt):
    from evaluate_structured_rom import _rollouts
    if future_dt not in (.01,.005):raise ValueError("registered future steps only")
    if not hasattr(model,"history_dt"):return _rollouts(model,histories,steps,future_dt)
    if getattr(model,"_cache",None) is not None:raise RuntimeError("history refinement cannot reuse active graph cache")
    history_dt=HISTORY_DT_MAIN if future_dt==.01 else HISTORY_DT_REFERENCE
    h=torch.as_tensor(np.asarray(histories),dtype=torch.float64)
    with torch.no_grad():
        if hasattr(model,"begin_graph_cache"):model.begin_graph_cache()
        try:path=model.rollout(h,int(steps),float(future_dt),history_dt=history_dt)
        finally:
            if hasattr(model,"end_graph_cache"):model.end_graph_cache()
    path=path.detach().cpu().numpy() if hasattr(path,"detach") else np.asarray(path,dtype=np.float64)
    if path.ndim!=3 or path.shape[:2]!=(len(h),steps+1):raise ValueError("paired inference shape differs")
    return path

def paired_evaluate_model(model, events, output, prepared, *, noisy=False):
    """P2 arithmetic with model-agnostic orchestration, no forced class switch.

    evaluate_queue itself forces LinearCacheRuntime and rejects the new family.
    Every numerical/statistical function below is the existing P2 function.
    """
    from evaluate_structured_rom import _rollouts, _pair_row
    from summarize_train_local_memory_0098 import event_metrics
    from fixed_edge_pair_20261019 import scientific_qualification, physical_metrics
    from active_fom_pair_fit_20261020 import interval15, actual_interval_predictions
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    candidate = copy.deepcopy(model).cpu().double().eval()
    events = sorted(events, key=lambda e: e["id"])
    timing = {"initialization_seconds": 0., "initialization_calls": 0, "rollout_seconds": 0.}
    original_encode = candidate.encode

    def timed_encode(*args, **kwargs):
        tick = time.perf_counter()
        result = original_encode(*args, **kwargs)
        timing["initialization_seconds"] += time.perf_counter() - tick
        timing["initialization_calls"] += 1
        return result

    candidate.encode = timed_encode
    mass = candidate.mass.detach().numpy(); pscale = candidate.p_scale.detach().numpy()
    records = {key: [] for key in ("records", "common_frequency", "normalized_p", "objective_point_error", "numerical")}
    extras, prediction_files = [], []
    start = time.perf_counter()
    for first in range(0, len(events), 16):
        block = events[first:first + 16]
        histories = np.stack([old.noise_history(e, candidate.input_scale_delta.detach().numpy(),
            candidate.input_scale_omega.detach().numpy()) if noisy else e["x"][:11] for e in block])
        failure = None
        tick = time.perf_counter()
        try:
            fine = _paired_rollouts(candidate, histories, 3000, .005)
            coarse = _paired_rollouts(candidate, histories, 1500, .01)
        except Exception as error:
            failure = {"error": repr(error), "traceback": traceback.format_exc()}
            fine = np.full((len(block), 3001, 40), np.nan)
            coarse = np.full((len(block), 1501, 40), np.nan)
        finally:
            timing["rollout_seconds"] += time.perf_counter() - tick
        path = output / f"predictions_{first:04d}.npz"
        np.savez_compressed(path, event_ids=np.asarray([e["id"] for e in block]),
            histories=histories, fine=fine, coarse=coarse)
        prediction_files.append({"path": str(path.resolve()), "sha256": sha(path)})
        for event, history, f, c in zip(block, histories, fine, coarse):
            metrics = event_metrics(candidate, event, f[:1601], c[:801])
            for key, rows in metrics.items():
                records[key].extend(rows)
            pair8 = _pair_row(f[:1601], c[:801], history, event, event["classification_meta"], mass)
            actual_f, actual_c, n = actual_interval_predictions(event, f, c)
            pair15 = _pair_row(actual_f, actual_c, history, event, event["classification_meta"], mass)
            pair15.update(scope="actual_interval_up_to_15s", observed_future_steps=n,
                observed_future_seconds=n * .01)
            coarse_points = {}
            for h in ("1", "2", "5", "8"):
                steps = int(h) * 100
                if len(event["x"]) - 11 < steps or not np.isfinite(c[:steps + 1]).all():
                    coarse_points[h] = None
                else:
                    with torch.no_grad():
                        value = float(_point_error(torch.as_tensor(c[1:steps + 1, :30]),
                            torch.as_tensor(event["x"][11:11 + steps]), candidate).mean())
                    coarse_points[h] = value if np.isfinite(value) else None
            extras.append({"event_id": event["id"], "group": event["group"], "label": event["label"],
                "exception": failure, "classification_8s": pair8, "classification_15s": pair15,
                "coarse_point_errors": coarse_points,
                "actual_interval15": interval15(candidate, event, f, c),
                "scientific_numeric_8s": scientific_qualification(pair8, metrics["numerical"]),
                "physical_1_2_5_8s": physical_metrics(f[::2], event, mass, pscale)})
        write(output / "partial.json", {"rows": records, "extras": extras})
    all_ids = {e["id"] for e in events}
    original_ids = {e["id"] for e in events if e["id"] not in prepared.get("fresh19_ids", [])}
    scopes = {"common385" if not noisy else "no_update_noisy": all_ids,
        "fit": all_ids & set(prepared["fit_ids"]),
        "no_update": all_ids & set(prepared["no_update_ids"]),
        "original366": original_ids, "fresh19": all_ids - original_ids}
    timing["integration_and_rollout_overhead_seconds"] = timing["rollout_seconds"] - timing["initialization_seconds"]
    result = {"event_count": len(events), "rows": records, "extras": extras,
        "profiles": {name: old.group_summary(records, extras, ids) for name, ids in scopes.items() if ids},
        "prediction_files": prediction_files, "seconds": time.perf_counter() - start,
        "timing": timing,
        "timing_scope": "CPU double one-thread; paired history and future RK4 includes initialization/output/metrics", "history_dt_main":HISTORY_DT_MAIN,"history_dt_reference":HISTORY_DT_REFERENCE,
        "complexity": candidate.complexity_counts(),
        "noisy": noisy, "truth_unchanged": True, "val_read": False, "test_read": False,
        "helper_sources": {name: sha(ROOT / "experiments" / name) for name in (
            "evaluate_structured_rom.py", "summarize_train_local_memory_0098.py",
            "fixed_edge_pair_20261019.py", "active_fom_pair_fit_20261020.py",
            "closure_discovery_confirmation.py")}}
    write(output / "evidence.json", result, exclusive=True)
    write(output / "complete.json", {"status": "COMPLETED", "event_count": len(events),
        "evidence_sha256": sha(output / "evidence.json")}, exclusive=True)
    return result



def qualify(base,device="cuda"):
    dev=resolve_device(device)
    if dev.type!="cuda":raise ValueError("real CUDA required")
    base=Path(base)
    cpu=read(ROOT/"results/state_gated_memory_20261028/model_pre_freeze_qa.json")
    if cpu.get("shared_checks_qualified") is not True or any(cpu.get("per_variant_qualified",{}).get(v) is not True for v in VARIANTS):
        raise ValueError("the once-refined full CPU mathematical qualification must pass before real payload access")
    prepared,fit=checked_preparation(base);output=base/"qualification";output.mkdir(exist_ok=False)
    cpu_ok=cpu.get("per_variant_qualified",{}) if cpu.get("shared_checks_qualified") is True else {}
    results={};reference=None;eligible=[]
    if any(cpu_ok.get(v) is True for v in VARIANTS):reference=evaluate(build_model("R_N"),fit,output/"R_N_fit297",prepared)
    selected=[e for eid in prepared["cost12_ids"] for e in fit if e["id"]==eid]
    for variant in VARIANTS:
        row=dict(status="NUMERICAL_UNRESOLVED",training_admission=False,cpu_qualified=cpu_ok.get(variant) is True)
        if not row["cpu_qualified"]:results[variant]=row;continue
        try:
            parity=device_comparison(variant,old.synthetic_fixture(),dev)
            model=build_model(variant);evidence=evaluate(model,fit,output/f"{variant}_zero_fit297",prepared,mechanism=variant=="C")
            cross=None
            if variant=="A":
                cross_path=output/"R_N_A_cross_model.json"
                cross=cross_model_qualification(build_model("R_N"),fit,reference,evidence,prepared,cross_path)
                row["R_N_A_cross_model"]={"path":str(cross_path),"sha256":sha(cross_path)}
                row["R_N_A_cross_model_qualified"]=cross["qualified"]
            m=build_model(variant,dev);store=EventStore(selected,dev);init_timing={"seconds":0.,"calls":0};original_initialize=m.initialize
            def timed_initialize(*args,**kwargs):
                synchronize(dev);t=time.perf_counter()
                try:return original_initialize(*args,**kwargs)
                finally:
                    synchronize(dev);init_timing["seconds"]+=time.perf_counter()-t;init_timing["calls"]+=1
            m.initialize=timed_initialize
            torch.cuda.reset_peak_memory_stats(dev);synchronize(dev);tick=time.perf_counter()
            with graph(m):J,_=objective(m,store,[(e,10) for e in selected],8.,method="rk4");J.backward()
            synchronize(dev);finite=bool(torch.isfinite(J)) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
            row.update(cpu_cuda_pair_pass=True,device_comparison=parity,finite_12x8_backward=finite,cost12_seconds=time.perf_counter()-tick,peak_cuda_bytes=torch.cuda.max_memory_allocated(dev),initialization_seconds=init_timing["seconds"],initialization_calls=init_timing["calls"],fullfit297_qualified=valid(evidence["decision_profile"]),zero_profile=evidence["decision_profile"])
            row["training_admission"]=bool(finite and row["fullfit297_qualified"] and (cross is None or cross["qualified"]))
            row["status"]="PASS" if row["training_admission"] else "NUMERICAL_UNRESOLVED"
            if row["training_admission"]:eligible.append(variant)
        except Exception as exc:row.update(error=repr(exc),traceback=traceback.format_exc())
        results[variant]=row
    result=dict(status="COMPLETED",per_variant=results,eligible_variants=eligible,R_N_profile=reference["decision_profile"] if reference else None,training_method="rk4",history_dt_main=HISTORY_DT_MAIN,history_dt_reference=HISTORY_DT_REFERENCE,full_matched_comparison_qualified=eligible==list(VARIANTS),val_read=False,test_read=False)
    write(output/"summary.json",result,exclusive=True);return result


def study(base,device="cuda"):
    base=Path(base);prepared,fit=checked_preparation(base,fit_only=True);panel=[e for eid in prepared["panel_ids"] for e in fit if e["id"]==eid]
    eligible=read(base/"qa/summary.json")["eligible_variants"];reference=read(base/"qualification/summary.json")["R_N_profile"]
    output=base/"study";output.mkdir(exist_ok=False);runs=[];decisions=[]
    for seed in SEEDS:
        group=[formal_run(base,v,seed,prepared,fit,panel,device) for v in eligible];runs.extend(group)
        profiles={r["variant"]:r["fit_endpoints"]["96"] for r in group if r["status"]=="COMPLETED"}
        if reference is not None:profiles["R_N"]=reference
        decision=retention(profiles,optimization_unresolved=any(r.get("optimization_unresolved_at96",True) for r in group));decisions.append(dict(seed=seed,**decision))
        write(output/f"retention_{seed}.json",decisions[-1],exclusive=True)
        if not decision["second_seed"]:break
    write(output/"fit_decisions_closed.json",dict(status="COMPLETED",decisions=decisions,run_ids=[r["run_id"] for r in runs]),exclusive=True)
    _,all_events=checked_preparation(base,fit_only=False);held=[e for e in all_events if e["id"] in prepared["no_update_ids"]];reports=[]
    for r in runs:
        if r["status"]!="COMPLETED":continue
        m=StateGatedMemoryROM.from_export(Path(r["endpoint"])/"model.json");folder=output/r["run_id"]
        evaluate(m,all_events,folder/"evaluation385",prepared);evaluate(m,held,folder/"noise88",prepared,noisy=True)
        reports.append(dict(run_id=r["run_id"],evaluation385=str(folder/"evaluation385/evidence.json"),noise88=str(folder/"noise88/evidence.json")))
    result=dict(status="COMPLETED" if len(eligible)==3 and all(r["status"]=="COMPLETED" for r in runs) else "UNRESOLVED",runs=runs,common96_decisions=decisions,reports=reports,calls=call_counts(base),default_model=None,scientific_model_adoption=False,val_read=False,test_read=False)
    write(output/"summary.json",result,exclusive=True);return result


def summary(base):
    base=Path(base);result=dict(status=read(base/"study/summary.json")["status"],calls=call_counts(base),model_adopted=False,val_read=False,test_read=False,no_process_quotas=True)
    write(base/"summary.json",result,exclusive=True);return result
