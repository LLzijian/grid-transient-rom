"""Loop0085 one analytic locally guarded coefficient update; cached TRAIN scoring only."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, shutil, sys, time
from pathlib import Path
from typing import Any
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
import torch
from scipy.linalg import solve_continuous_lyapunov
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"src")]
from audit_force_initialization_0080 import numeric,point,risk  # noqa:E402
from fit_coefficient_block_0082 import (AUTONOMOUS_METRIC,CONDITIONAL_SPEED_SCALE,_checked_rollout,_theta,affinity_qa,conditional_summary,deterministic_test_z,set_theta,structural_qa)  # noqa:E402
from train_joint_latent import BASE,NETWORK  # noqa:E402
from train_structured_rom import SOURCE,load_model  # noqa:E402

BASELINE=ROOT/"private_data/task2_structure_0078_extend_G-fixed/step3000"
FIT=ROOT/"private_data/task2_coefficient_block_0082_attempt2"
POLES=ROOT/"private_data/task2_coefficient_poles_0083/matrices.npz"
FORCE_BASELINE=ROOT/"private_data/task2_force_initialization_0080"
BASELINE_SHA="2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b"
PROPOSAL_SHA="72fce608287d9e13dd23957e860ad5d20abb2e3130cd883e0f493e1021348db5"
WALL,FORECAST=1800.,1200.
CACHE_SHA={"fit_arrays":"b23928f3075dc88966e2f1e51d6b689056ff72ea54b7e64ce88ff90b3dbe5294","trajectories":"5b09a22e8c5cdadf8a6710db3164d97fc0ad15051d7397ffcdb178e00d73997d","poles":"46bcaa3480fdb422f3f98ca069243ca8f911bd44ae7c061f4eba75b77199d212","fit_evidence":"1fb5a1146f90bfe24455021914525cf69d7366e24a318f6384cb37b10ab0adad","force_baseline_evidence":"38b206d5316846596da8670489f3714fedb866f9f92dd2f9759cf1fb8e931209"}

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes(baseline:Path,fit:Path,poles:Path)->dict[str,str]:
    candidate=fit/"candidate"; paths={"baseline_model":baseline/"model.pt","baseline_meta":baseline/"checkpoint.json","baseline_norm":baseline/"normalization.npz","baseline_export":baseline/"structured_numeric_export.json","proposal_model":candidate/"model.pt","proposal_meta":candidate/"checkpoint.json","proposal_norm":candidate/"normalization.npz","proposal_export":candidate/"structured_numeric_export.json","fit_arrays":fit/"fit_arrays.npz","fit_evidence":fit/"evidence.json","trajectories":fit/"trajectories.npz","force_baseline_evidence":FORCE_BASELINE/"evidence.json","force_baseline_trajectories":FORCE_BASELINE/"trajectories.npz","poles":poles,"poles_evidence":poles.parent/"evidence.json","base":Path(BASE),"network":Path(NETWORK),"audit":Path(__file__),"fit_audit":ROOT/"experiments/fit_coefficient_block_0082.py","poles_audit":ROOT/"experiments/audit_coefficient_poles_0083.py","trainer":ROOT/"experiments/train_structured_rom.py","core":ROOT/"experiments/structured_rom.py","grammar":ROOT/"experiments/structure_grammar.py","force_audit":ROOT/"experiments/audit_force_initialization_0080.py","conditional_audit":ROOT/"experiments/audit_conditional_latent_0081.py","joint_latent":ROOT/"experiments/train_joint_latent.py","joint_campaign":ROOT/"experiments/train_joint_campaign.py"}
    for label,directory in (("baseline",baseline),("proposal",candidate)):
        source=Path(json.loads((directory/"checkpoint.json").read_text()).get("source") or SOURCE);paths[f"{label}_source_model"]=source/"model.pt";paths[f"{label}_source_norm"]=source/"normalization.npz"
    return {key:sha(value) for key,value in paths.items()}
def require_time(deadline:float,stage:str)->None:
    if time.perf_counter()>deadline:raise TimeoutError(f"0085 wall cap during {stage}")

def guard(A0:np.ndarray,delta:np.ndarray)->dict[str,Any]:
    A0,delta=np.asarray(A0,float),np.asarray(delta,float)
    if A0.shape!=(32,32) or delta.shape!=(32,32) or not(np.isfinite(A0).all() and np.isfinite(delta).all()):raise ValueError("invalid guard matrices")
    P=solve_continuous_lyapunov(A0.T,-np.eye(32));P=(P+P.T)/2; residual=np.linalg.norm(A0.T@P+P@A0+np.eye(32),"fro")/np.sqrt(32);eig=np.linalg.eigvalsh(P)
    if eig.min()<=1e-10 or residual>1e-9:raise AssertionError("invalid fixed P")
    S=(delta.T@P+P@delta);S=(S+S.T)/2;mu=float(np.linalg.eigvalsh(S).max());alpha=1. if mu<=.49 else .49/mu
    if not(np.isfinite(alpha) and 0<alpha<=1):raise FloatingPointError("invalid alpha")
    return {"P":P,"P_eigenvalues":eig,"P_condition":float(np.linalg.cond(P)),"P_residual":float(residual),"S":S,"mu":mu,"alpha":float(alpha)}

def local_qa(A:np.ndarray,P:np.ndarray)->dict[str,Any]:
    A,P=np.asarray(A,float),np.asarray(P,float)
    if A.shape!=(32,32) or P.shape!=(32,32) or not(np.isfinite(A).all() and np.isfinite(P).all()):raise ValueError("invalid local matrix")
    decay=(A.T@P+P@A); decay=(decay+decay.T)/2;values,vectors=np.linalg.eig(A);norm=np.linalg.norm(A,2);res=max(float(np.linalg.norm(A@vectors[:,i]-values[i]*vectors[:,i])/max(1e-300,(norm+abs(values[i]))*np.linalg.norm(vectors[:,i]))) for i in range(32))
    return {"decay_max_eigenvalue":float(np.linalg.eigvalsh(decay).max()),"spectral_abscissa":float(values.real.max()),"eigen_residual_max":res}

def conditional(table:dict[str,np.ndarray],alpha:float)->tuple[dict[str,Any],dict[str,np.ndarray]]:
    if not(np.isfinite(alpha) and 0<alpha<=1):raise ValueError("conditional alpha")
    rb,rc,bb,bc=(np.asarray(table[key],float) for key in ("residual_baseline_fine","residual_candidate_fine","residual_baseline_coarse","residual_candidate_coarse"))
    if any(value.shape!=(46514,31) or not np.isfinite(value).all() for value in (rb,rc,bb,bc)):raise ValueError("conditional residual shape/finite")
    if any(np.asarray(table[key]).shape!=(46514,) for key in ("event_index","label","group","start")):raise ValueError("conditional mapping shape")
    guard_f=rb+alpha*(rc-rb);guard_c=bb+alpha*(bc-bb)
    mapping={"event_index":table["event_index"],"label":table["label"].astype(str).tolist(),"group":table["group"].astype(str).tolist(),"start":table["start"]};summary={"baseline":conditional_summary(mapping,rb),"guard":conditional_summary(mapping,guard_f)}; numerical=np.linalg.norm(guard_f-guard_c,axis=1)
    return summary,{"guard_fine":guard_f,"guard_coarse":guard_c,"numerical":numerical}

def _finite(value:Any)->bool:
    if isinstance(value,dict):return all(_finite(item) for item in value.values())
    if isinstance(value,(list,tuple)):return all(_finite(item) for item in value)
    if isinstance(value,np.ndarray):return bool(np.isfinite(value).all())
    return not isinstance(value,(float,np.floating)) or bool(np.isfinite(value))

def _candidate_hashes(candidate:Path)->dict[str,str]:
    return {name:sha(candidate/name) for name in ("model.pt","normalization.npz","structured_numeric_export.json","checkpoint.json")}

def _cached_baseline(fit:Path,evidence:dict[str,Any],table:dict[str,np.ndarray],model,before:dict[str,str])->dict[str,np.ndarray]:
    """Materialize once, then prove the immutable 0082 cache is usable before forecasts."""
    with np.load(fit/"trajectories.npz",allow_pickle=False) as source:
        keys=("event_ids","labels","lengths","truth","qenc","mass","p_scale","fine_baseline","coarse_baseline","autonomous_metric","conditional_speed_scale")
        cache={key:np.asarray(source[key]) for key in keys}
    ids,labels,lengths,truth,qenc=(cache[key] for key in ("event_ids","labels","lengths","truth","qenc"))
    if ids.shape!=(346,) or len(set(ids.astype(str)))!=346 or labels.shape!=(346,) or lengths.shape!=(346,) or qenc.shape!=(346,3):raise AssertionError("cached event dimensions")
    if int((labels.astype(str)=="stable").sum())!=311 or int((labels.astype(str)=="unstable").sum())!=35 or not np.issubdtype(lengths.dtype,np.integer) or np.any((lengths<1)|(lengths>500)):raise AssertionError("cached event coverage")
    if truth.ndim!=3 or truth.shape[0]!=346 or truth.shape[2]!=30 or any(not(np.isfinite(truth[i,:n+1]).all() and np.isnan(truth[i,n+1:]).all()) for i,n in enumerate(lengths)):raise AssertionError("cached physical padding")
    fine,coarse=cache["fine_baseline"],cache["coarse_baseline"]
    if fine.shape!=(346,int(lengths.max())*2+1,33) or coarse.shape!=(346,int(lengths.max())+1,33):raise AssertionError("cached forecast shape")
    if any(not(np.isfinite(fine[i,:2*n+1]).all() and np.isnan(fine[i,2*n+1:]).all() and np.isfinite(coarse[i,:n+1]).all() and np.isnan(coarse[i,n+1:]).all()) for i,n in enumerate(lengths)):raise AssertionError("cached forecast padding")
    if not(np.array_equal(fine[:,0,:30],truth[:,0]) and np.array_equal(coarse[:,0,:30],truth[:,0]) and np.array_equal(fine[:,0,30:],qenc) and np.array_equal(coarse[:,0,30:],qenc)):raise AssertionError("cached initial state")
    if tuple(cache["autonomous_metric"].tolist())!=AUTONOMOUS_METRIC or float(cache["conditional_speed_scale"])!=CONDITIONAL_SPEED_SCALE or not(np.array_equal(cache["mass"],model.mass.detach().cpu().numpy()) and np.array_equal(cache["p_scale"],model.p_scale.detach().cpu().numpy())):raise AssertionError("cached scale provenance")
    event=np.asarray(table["event_index"]); counts=np.bincount(event,minlength=346)
    if event.shape!=(46514,) or np.any(event<0) or np.any(event>=346) or counts.shape!=(346,) or np.any(counts==0) or len(set(np.asarray(table["group"]).astype(str)))!=24 or int(np.sum(table["start"]==10))!=346:raise AssertionError("fit window coverage")
    for i in range(346):
        hit=event==i
        if not(np.all(np.asarray(table["event_id"])[hit].astype(str)==str(ids[i])) and np.all(np.asarray(table["label"])[hit].astype(str)==str(labels[i]))):raise AssertionError("fit/cache event mapping")
    if table["theta"].shape!=(24,) or table["theta0"].shape!=(24,) or not(np.isfinite(table["theta"]).all() and np.isfinite(table["theta0"]).all()):raise AssertionError("cached theta")
    saved=evidence["per_event"]; expected={(str(row["event_id"]),int(row["horizon"])):float(row["baseline"]) for row in saved}
    if len(saved)!=1038 or len(expected)!=1038:raise AssertionError("0082 baseline row coverage")
    numeric_rows=numeric(fine,coarse,lengths,cache["mass"],cache["p_scale"])
    force_evidence=json.loads((FORCE_BASELINE/"evidence.json").read_text());old_numeric=force_evidence["numerics"]["baseline"]
    if any(abs(numeric_rows[i][key]-old_numeric[i][key])>1e-11 for i in range(346) for key in ("coi","common","p")):raise AssertionError("0082 baseline numerical reproduction")
    reproduced={}
    for h in (1,2,5):
        values=[]
        for i,n in enumerate(lengths):
            k=min(round(h/.01),int(n));value=float(point(fine[i,2:2*k+1:2],truth[i,1:k+1],cache["mass"],cache["p_scale"]).mean());values.append(value)
            if abs(value-expected[(str(ids[i]),h)])>1e-11:raise AssertionError("0082 baseline event reproduction")
        reproduced[str(h)]=risk(values)
        for key in ("mean","tail35","risk"):
            if abs(reproduced[str(h)][key]-evidence["autonomous_horizons"][str(h)]["baseline"][key])>1e-11:raise AssertionError("0082 baseline risk reproduction")
    old=evidence["hashes_before"]; after=evidence["hashes_after"]
    required={"checkpoint_model":"baseline_model","normalization":"baseline_norm","base":"base","network":"network","reader":"joint_latent","audit0081_helper":"conditional_audit","baseline_evidence":"force_baseline_evidence","baseline_trajectories":"force_baseline_trajectories"}
    if any(old[key]!=before[value] or after[key]!=old[key] for key,value in required.items()) or evidence["candidate_hashes"]["model.pt"]!=before["proposal_model"]:raise AssertionError("0082 hash linkage")
    if any(evidence["candidate_hashes"][name]!=before[f"proposal_{key}"] for name,key in (("normalization.npz","norm"),("structured_numeric_export.json","export"),("checkpoint.json","meta"))):raise AssertionError("0082 candidate hash linkage")
    poles=json.loads((POLES.parent/"evidence.json").read_text());pold=poles["hashes_before"]
    if pold["baseline_model"]!=before["baseline_model"] or pold["candidate_model"]!=before["proposal_model"] or any(pold[key]!=before[key] for key in ("base","network","trainer","core","grammar","joint_latent","joint_campaign")) or poles["hashes_after"]!=pold:raise AssertionError("0083 hash linkage")
    cache["baseline_numerics"]=np.asarray([[row[key] for key in ("coi","common","p")] for row in numeric_rows])
    return cache

def run(baseline:Path,fit:Path,poles:Path,output:Path)->dict[str,Any]:
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+WALL;before=hashes(baseline,fit,poles);output.mkdir();current={"stage":"load"};arrays={};timers={}
    try:
        if before["baseline_model"]!=BASELINE_SHA or before["proposal_model"]!=PROPOSAL_SHA or any(before[key]!=value for key,value in CACHE_SHA.items()):raise AssertionError("preregistered input SHA mismatch")
        with np.load(fit/"fit_arrays.npz",allow_pickle=False) as saved: table={key:np.asarray(saved[key]) for key in ("theta","theta0","event_index","event_id","label","group","start","residual_baseline_fine","residual_candidate_fine","residual_baseline_coarse","residual_candidate_coarse")}
        with np.load(poles,allow_pickle=False) as saved: Q=np.asarray(saved["baseline_Q"],float);A33=np.asarray(saved["baseline_A33"],float);proposalA=np.asarray(saved["candidate_A33"],float)
        if Q.shape!=(33,32) or A33.shape!=(33,33) or proposalA.shape!=(33,33) or not(np.isfinite(Q).all() and np.isfinite(A33).all() and np.isfinite(proposalA).all()):raise AssertionError("0083 matrix provenance")
        A0=Q.T@A33@Q;delta33=proposalA-A33;g=guard(A0,Q.T@delta33@Q);alpha=g["alpha"];summary_cond,cond_arrays=conditional(table,alpha);theta_guard=table["theta0"]+alpha*(table["theta"]-table["theta0"])
        conditional_ratios={"total_ratio":summary_cond["guard"]["overall"]["total"]/summary_cond["baseline"]["overall"]["total"],"physical_ratio":summary_cond["guard"]["overall"]["physical"]/summary_cond["baseline"]["overall"]["physical"],"unstable_total_ratio":summary_cond["guard"]["subsets"]["unstable"]["total"]/summary_cond["baseline"]["subsets"]["unstable"]["total"]}
        arrays.update({"Q":Q,"reference_A33":A33,"delta_A33":delta33,"A0":A0,"P":g["P"],"P_eigenvalues":g["P_eigenvalues"],"S":g["S"],"theta0":table["theta0"],"theta_proposal":table["theta"],"theta_guard":theta_guard,"conditional_event_index":table["event_index"],"conditional_event_id":table["event_id"],"conditional_label":table["label"],"conditional_group":table["group"],"conditional_start":table["start"],**cond_arrays})
        (output/"guard.json").write_text(json.dumps({key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in g.items() if key not in ("P","S")},indent=2,allow_nan=False));np.savez_compressed(output/"guard_arrays.npz",**arrays)
        if cond_arrays["numerical"].max()>.001:raise FloatingPointError("conditional numerical gate")
        require_time(deadline,"guard")
        current={"stage":"candidate"};model=load_model(baseline);model.eval();frozen={key:value.detach().clone() for key,value in model.state_dict().items()};theta0=_theta(model)
        if not np.array_equal(theta0,table["theta0"]):raise AssertionError("baseline theta/cache mismatch")
        affinity=affinity_qa(model,deterministic_test_z(),theta_guard-theta0)
        set_theta(model,theta_guard);changed=[key for key,value in model.state_dict().items() if not torch.equal(value,frozen[key])];allowed=[f"structured.storage_gamma.{i}" for i in range(8)]+["structured.port_coefficient.0","structured.port_coefficient.1"]
        if not set(changed).issubset(allowed) or any(not torch.equal(value,model.state_dict()[key]) for key,value in frozen.items() if key not in allowed) or not np.array_equal(_theta(model),theta_guard):raise AssertionError("illegal guard state change")
        candidate=output/"candidate";candidate.mkdir();torch.save(model.state_dict(),candidate/"model.pt");shutil.copyfile(baseline/"normalization.npz",candidate/"normalization.npz");model.structured.export(candidate/"structured_numeric_export.json");meta=json.loads((baseline/"checkpoint.json").read_text());(candidate/"checkpoint.json").write_text(json.dumps({"arm":"G-fixed","step":3000,"source":meta.get("source") or str(SOURCE),"parent":str(baseline),"loop":"0085"},indent=2));candidate_hashes=_candidate_hashes(candidate)
        structural=structural_qa(model,candidate,deterministic_test_z());reloaded=load_model(candidate);reloaded.eval();zero=torch.zeros(33,dtype=torch.float64)
        with torch.enable_grad():actual=torch.autograd.functional.jacobian(lambda z:reloaded.field(z[None]).squeeze(0),zero).detach().cpu().numpy()
        expected=A33+alpha*delta33;actual_relative=float(np.linalg.norm(actual-expected,"fro")/max(1.,np.linalg.norm(expected,"fro")));actual_A32=Q.T@actual@Q;actual_local=local_qa(actual_A32,g["P"])
        actual_decay=(actual_A32.T@g["P"]+g["P"]@actual_A32);actual_decay=(actual_decay+actual_decay.T)/2
        arrays.update({"actual_A33":actual,"actual_A32":actual_A32,"actual_decay_eigenvalues":np.linalg.eigvalsh(actual_decay)})
        if actual_relative>1e-10 or actual_local["decay_max_eigenvalue"]>-.5 or actual_local["spectral_abscissa"]>=-1e-6 or actual_local["eigen_residual_max"]>1e-9:raise AssertionError("actual guard QA")
        qa={"alpha":alpha,"mu":g["mu"],"actual_relative":actual_relative,"actual_local":actual_local,"affinity":affinity,"structural":structural,"allowed_state_keys_changed":changed};(output/"guard_qa.json").write_text(json.dumps(qa,indent=2,allow_nan=False));np.savez_compressed(output/"guard_arrays.npz",**arrays)
        require_time(deadline,"candidate QA")
        current={"stage":"cached baseline verification"};evidence=json.loads((fit/"evidence.json").read_text());cache=_cached_baseline(fit,evidence,table,model,before);require_time(deadline,"cached baseline verification")
        current={"stage":"candidate forecast"};forecast_started=time.perf_counter();forecast_deadline=min(deadline,forecast_started+FORECAST);ids,labels,lengths,truth,qenc,mass,pscale=(cache[key] for key in ("event_ids","labels","lengths","truth","qenc","mass","p_scale"));fine_base,coarse_base=cache["fine_baseline"],cache["coarse_baseline"]
        states=torch.as_tensor(np.c_[truth[:,0],qenc],dtype=torch.float64);fine=_checked_rollout(model,states,lengths*2,.005,deadline=forecast_deadline,progress=current,partial=output/"partial_fine.npz",event_ids=ids);coarse=_checked_rollout(model,states,lengths,.01,deadline=forecast_deadline,progress=current,partial=output/"partial_coarse.npz",event_ids=ids)
        np.savez_compressed(output/"trajectories.npz",event_ids=ids,labels=labels,mass=mass,p_scale=pscale,conditional_speed_scale=CONDITIONAL_SPEED_SCALE,autonomous_metric=np.asarray(AUTONOMOUS_METRIC),lengths=lengths,truth=truth,qenc=qenc,fine_baseline=fine_base,coarse_baseline=coarse_base,fine_candidate=fine,coarse_candidate=coarse)
        timers["forecast_seconds"]=time.perf_counter()-forecast_started;require_time(forecast_deadline,"candidate forecast cap");require_time(deadline,"candidate forecast")
        num=numeric(fine,coarse,lengths,mass,pscale)
        if not _finite(num) or any(value>.001 for row in num for value in row.values()):raise FloatingPointError("autonomous numerical gate")
        rows=[];horizons={}
        for h in (1,2,5):
            candidate_values=[];baseline_values=[]
            for i,n in enumerate(lengths):
                k=min(round(h/.01),int(n));candidate_values.append(float(point(fine[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean()));baseline_values.append(float(point(fine_base[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean()));rows.append({"event_id":str(ids[i]),"label":str(labels[i]),"horizon":h,"candidate":candidate_values[-1],"baseline":baseline_values[-1],"available_steps":int(k)})
            horizons[str(h)]={"candidate":risk(candidate_values),"baseline":risk(baseline_values)}
        unstable=[row for row in rows if row["horizon"]==5 and row["label"]=="unstable"];severe={name:{"mean":float(np.mean([row[name] for row in unstable])),"worst4":float(np.mean(sorted(row[name] for row in unstable)[-4:]))} for name in ("baseline","candidate")};average={name:float(np.mean([horizons[str(h)][name]["risk"] for h in (1,2,5)])) for name in ("baseline","candidate")}
        after=hashes(baseline,fit,poles);candidate_after=_candidate_hashes(candidate);timers["global_seconds"]=time.perf_counter()-started;require_time(deadline,"final")
        if before!=after or candidate_hashes!=candidate_after or not np.array_equal(_theta(model),theta_guard) or any(not torch.equal(value,model.state_dict()[key]) for key,value in frozen.items() if key not in allowed):raise RuntimeError("source/candidate/frozen state changed")
        result={"loop":"0085","test_read":False,"h5_read":False,"cached_train_read":True,"hashes_before":before,"hashes_after":after,"candidate_hashes":candidate_hashes,"alpha":alpha,"mu":g["mu"],"conditional":summary_cond,"conditional_numeric_max":float(cond_arrays["numerical"].max()),"conditional_original_reference_gates":{"total_ratio":{"reference":.90,"observed":conditional_ratios["total_ratio"]},"physical_ratio":{"reference":1.05,"observed":conditional_ratios["physical_ratio"]},"unstable_total_ratio":{"reference":1.05,"observed":conditional_ratios["unstable_total_ratio"]}},"guard_qa":qa,"autonomous_numerics":num,"autonomous_horizons":horizons,"autonomous_average_risk":average,"unstable_5s":severe,"per_event":rows,"timers":timers,"seconds":timers["global_seconds"],"allowed_state_keys_changed":changed}
        if not _finite(result):raise FloatingPointError("nonfinite final metric")
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False));return result
    except Exception as error:
        timers["global_seconds"]=time.perf_counter()-started
        if arrays:np.savez_compressed(output/"partial_guard_arrays.npz",**arrays)
        (output/"failure_evidence.json").write_text(json.dumps({"type":type(error).__name__,"message":str(error),"current":current,"timers":timers,"hashes_before":before,"test_read":False,"h5_read":False,"cached_train_read":True,"seconds":time.perf_counter()-started},indent=2));raise

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--baseline",type=Path,default=BASELINE);p.add_argument("--fit",type=Path,default=FIT);p.add_argument("--poles",type=Path,default=POLES);p.add_argument("--output",type=Path,required=True);a=p.parse_args();run(a.baseline,a.fit,a.poles,a.output)
if __name__=="__main__":main()
