"""Loop0087 fixed-budget direct objective updates in the existing 24-amplitude block."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, time
from pathlib import Path
from typing import Any
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
from audit_force_initialization_0080 import numeric,point,risk  # noqa:E402
from audit_lyapunov_guard_0085 import (BASELINE,BASELINE_SHA,FORCE_BASELINE,_cached_baseline,_candidate_hashes,_finite,local_qa)  # noqa:E402
from fit_coefficient_block_0082 import (AUTONOMOUS_METRIC,CONDITIONAL_SPEED_SCALE,_theta,affinity_qa,deterministic_test_z,set_theta,structural_qa)  # noqa:E402
from fit_guarded_regression_0086 import (ANCHORS,FIT,FORCE,GUARD,NULL,POLES,SOLVE_CAP,coefficient_lmis,hashes as hashes86,jsonable,solve_sdp)  # noqa:E402
from train_joint_campaign import sample_batch  # noqa:E402
from train_joint_latent import BASE,ROOTS,read_development  # noqa:E402
from train_structured_rom import SOURCE,load_model,objective  # noqa:E402
from kiki.models.private_pm import MechanicalROM  # noqa:E402

PRE=ROOT/"private_data/task2_guarded_rollout_0087_preflight"
OUT=ROOT/"private_data/task2_guarded_rollout_0087"
STEPS,BATCH,SEED=300,32,87
PREFLIGHT_CAP,TRAIN_CAP,MAIN_CAP,FORECAST_CAP=600.,2700.,3600.,1200.

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes(baseline:Path)->dict[str,str]:
    result=hashes86(baseline,FIT,POLES,NULL,GUARD);result["audit0087"]=sha(Path(__file__));return result
def require_time(deadline:float,stage:str)->None:
    if time.perf_counter()>deadline:raise TimeoutError(f"0087 cap during {stage}")
def theta_parameters(model)->list[torch.nn.Parameter]:
    names=[f"structured.storage_gamma.{i}" for i in range(8)]+["structured.port_coefficient.0","structured.port_coefficient.1"]
    found=dict(model.named_parameters())
    if set(names)-set(found):raise AssertionError("missing frozen 24-amplitude block")
    for name,param in found.items():param.requires_grad_(name in names)
    selected=[found[name] for name in names]
    if sum(param.numel() for param in selected)!=24 or any(param.requires_grad is False for param in selected):raise AssertionError("trainable amplitude block")
    return selected
def _snapshot(model)->dict[str,torch.Tensor]:return {key:value.detach().clone() for key,value in model.state_dict().items()}
def _frozen_equal(model,source:dict[str,torch.Tensor],allowed:set[str])->bool:return all(torch.equal(value,model.state_dict()[key]) for key,value in source.items() if key not in allowed)
def _trace_value(value:Any)->Any:
    if isinstance(value,torch.Tensor):return float(value.detach())
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value
def estimated_seconds(times:list[float])->float:
    if len(times)!=3 or not all(np.isfinite(value) and value>=0 for value in times):raise ValueError("three finite preflight times")
    return 100*sum(times)+STEPS*.5
def allow_final_forecast(completed_updates:int,now:float,deadline:float)->bool:
    return completed_updates==STEPS and now<=deadline
def _load_fixed() -> tuple[dict[str,np.ndarray],np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    with np.load(FIT/"fit_arrays.npz",allow_pickle=False) as saved:cache={key:np.asarray(saved[key]) for key in ("theta0","theta","scales","K","g","active","event_index","event_id","label","group","start")}
    with np.load(POLES,allow_pickle=False) as saved:Q=np.asarray(saved["baseline_Q"],float);A33=np.asarray(saved["baseline_A33"],float)
    with np.load(NULL,allow_pickle=False) as saved:D=np.asarray(saved["D"],float);null_theta=np.asarray(saved["theta_delta"],float)
    with np.load(GUARD,allow_pickle=False) as saved:P=np.asarray(saved["P"],float);q85=np.asarray(saved["Q"],float);a085=np.asarray(saved["A0"],float)
    if not(np.array_equal(cache["active"],np.ones(24,dtype=bool)) and np.array_equal(null_theta,cache["theta"]-cache["theta0"]) and np.array_equal(q85,Q) and np.array_equal(a085,Q.T@A33@Q)):raise AssertionError("prior amplitude/tensor order")
    L=coefficient_lmis(D,Q,cache["scales"],P);res=float(np.linalg.norm((Q.T@A33@Q).T@P+P@(Q.T@A33@Q)+np.eye(32),"fro")/np.sqrt(32))
    if np.linalg.eigvalsh(P).min()<=1e-10 or res>1e-9:raise AssertionError("fixed P")
    return cache,Q,A33,D,P,L
def _project(beta_hat:np.ndarray,beta_old:np.ndarray,L:np.ndarray,deadline:float)->dict[str,Any]:
    beta_hat,beta_old,L=np.asarray(beta_hat,float),np.asarray(beta_old,float),np.asarray(L,float)
    if beta_hat.shape!=(24,) or beta_old.shape!=(24,) or L.shape!=(24,32,32) or not(np.isfinite(beta_hat).all() and np.isfinite(beta_old).all() and np.isfinite(L).all()):return {"status":"invalid reference","qa_pass":False,"projection_pass":False}
    old_f=.49*np.eye(32)-np.einsum("j,jab->ab",beta_old,L)
    if np.linalg.eigvalsh(old_f).min()<-1e-8:return {"status":"infeasible reference","qa_pass":False,"projection_pass":False,"old_F":old_f}
    solved=solve_sdp(np.eye(24),beta_hat,L,deadline)
    reference=float(beta_old@beta_old-2*beta_hat@beta_old)
    solved["reference_objective"]=reference
    if not solved.get("qa_pass",False) or solved.get("primal",np.inf)>reference+1e-7*max(1.,abs(reference)):solved["projection_pass"]=False
    else:solved["projection_pass"]=True
    return solved
def _save_projection(output:Path,step:int,old:np.ndarray,proposal:np.ndarray,gradient:np.ndarray,solved:dict[str,Any])->None:
    arrays={"old_beta":old,"proposal_beta":proposal,"theta_gradient":gradient,**{key:solved[key] for key in ("beta","Y","Y_raw","F","ell","old_F") if key in solved}}
    np.savez_compressed(output/f"projection_{step:04d}.npz",**arrays)
def _events() -> tuple[list[dict[str,Any]],dict[str,Any],dict[str,Any]]:
    base=MechanicalROM.load(BASE);events,excluded,identities=read_development(base,__import__("fit_coefficient_block_0082")._network(),full_train=True);train=events["train"]
    if len(train)!=346 or sum(event["label"]=="stable" for event in train)!=311 or sum(event["label"]=="unstable" for event in train)!=35 or len({event["group"] for event in train})!=24 or sum(value=="train" for value in excluded.values())!=23:raise AssertionError("TRAIN identity")
    fit=json.loads((FIT/"evidence.json").read_text());force=json.loads((FORCE/"evidence.json").read_text())
    if identities!=fit["identities"] or identities!=force["identities"] or excluded!=fit["excluded"] or excluded!=force["excluded"]:raise AssertionError("0079/0082 identities")
    return train,excluded,identities
def _write_failure(output:Path,error:Exception,current:dict[str,Any],before:dict[str,str],model,optimizer,traces,did_h5:bool)->None:
    if model is not None:torch.save(model.state_dict(),output/"failure_model.pt")
    if optimizer is not None:torch.save(optimizer.state_dict(),output/"failure_optimizer.pt")
    (output/"failure_evidence.json").write_text(json.dumps(jsonable({"type":type(error).__name__,"message":str(error),"current":current,"hashes_before":before,"test_read":False,"h5_read":did_h5,"traces":traces}),indent=2,allow_nan=False))

def preflight(baseline:Path,output:Path)->dict[str,Any]:
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+PREFLIGHT_CAP;before=hashes(baseline);output.mkdir();model=None;optimizer=None;traces=[];current={"stage":"load"};did_h5=False
    try:
        if before["baseline_model"]!=BASELINE_SHA or any(before[key]!=value for key,value in ANCHORS.items()):raise AssertionError("input SHA")
        torch.manual_seed(SEED);model=load_model(baseline);model.eval();params=theta_parameters(model);source=_snapshot(model);did_h5=True;train,excluded,identities=_events();rng=np.random.default_rng(SEED);samples,histories=sample_batch(train,rng,BATCH)
        for horizon in (1.,2.,5.):
            current={"stage":"backward","horizon":horizon};begin=time.perf_counter();model.zero_grad(set_to_none=True);model.begin_graph_cache()
            try:loss,parts=objective(model,samples,histories,horizon,complexity_weight=0);loss.backward()
            finally:model.end_graph_cache()
            grad=torch.cat([param.grad.detach().reshape(-1) for param in params]);
            if not(torch.isfinite(loss) and torch.isfinite(grad).all()) or not _frozen_equal(model,source,{name for name,_ in model.named_parameters() if _.requires_grad}):raise FloatingPointError("preflight finite/frozen")
            traces.append({"horizon":horizon,"seconds":time.perf_counter()-begin,"loss":float(loss.detach()),"gradient_l2":float(torch.linalg.vector_norm(grad)),"event_ids":[event["id"] for event,_ in samples]});require_time(deadline,"preflight backward")
        after=hashes(baseline)
        if before!=after or not all(torch.equal(value,model.state_dict()[key]) for key,value in source.items()):raise AssertionError("preflight model/source mutation")
        estimate=estimated_seconds([row["seconds"] for row in traces]);result={"loop":"0087","mode":"preflight","completed":True,"test_read":False,"h5_read":did_h5,"validation_materialized_by_reader":did_h5,"hashes_before":before,"hashes_after":after,"identities":identities,"excluded":excluded,"traces":traces,"estimate_seconds":estimate,"authorized":bool(estimate<=2400),"seconds":time.perf_counter()-started}
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False));return result
    except Exception as error:_write_failure(output,error,current,before,model,optimizer,traces,did_h5);raise

def train(baseline:Path,preflight_dir:Path,output:Path)->dict[str,Any]:
    if output.exists():raise FileExistsError(output)
    pre=json.loads((preflight_dir/"evidence.json").read_text())
    times=[float(row["seconds"]) for row in pre.get("traces",[])];horizons=[float(row["horizon"]) for row in pre.get("traces",[])]
    if not(pre.get("completed") and horizons==[1.,2.,5.] and len(times)==3 and np.isclose(estimated_seconds(times),pre.get("estimate_seconds"),rtol=0,atol=1e-12) and estimated_seconds(times)<=2400):raise AssertionError("preflight resource authorization")
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+MAIN_CAP;train_deadline=started+TRAIN_CAP;before=hashes(baseline);output.mkdir();model=None;optimizer=None;traces=[];current={"stage":"load"};did_h5=False;arrays={}
    try:
        if before["baseline_model"]!=BASELINE_SHA or any(before[key]!=value for key,value in ANCHORS.items()):raise AssertionError("input SHA")
        if pre.get("hashes_before")!=before or pre.get("hashes_after")!=before:raise AssertionError("preflight input provenance")
        cache,Q,A33,D,P,L=_load_fixed();torch.manual_seed(SEED);model=load_model(baseline);model.train();params=theta_parameters(model);source=_snapshot(model);allowed={name for name,param in model.named_parameters() if param.requires_grad};np.savez_compressed(output/"fixed_arrays.npz",P=P,Q=Q,A33=A33,D=D,L=L,scales=cache["scales"],theta0=cache["theta0"])
        if not np.array_equal(_theta(model),cache["theta0"]):raise AssertionError("baseline theta")
        optimizer=torch.optim.Adam(params,lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False);did_h5=True;train_events,excluded,identities=_events();rng=np.random.default_rng(SEED)
        for step in range(1,STEPS+1):
            require_time(min(deadline,train_deadline),"training");horizon=(1.,2.,5.)[(step-1)%3];samples,histories=sample_batch(train_events,rng,BATCH);optimizer.zero_grad(set_to_none=True);model.begin_graph_cache()
            try:loss,parts=objective(model,samples,histories,horizon,complexity_weight=0);loss.backward()
            finally:model.end_graph_cache()
            grads=torch.cat([param.grad.detach().reshape(-1) for param in params])
            if not(torch.isfinite(loss) and torch.isfinite(grads).all()):raise FloatingPointError("nonfinite train loss/gradient")
            before_theta=_theta(model);norm=float(torch.nn.utils.clip_grad_norm_(params,100.));optimizer.step();proposal_theta=_theta(model);proposal_beta=cache["scales"]*(proposal_theta-cache["theta0"]);old_beta=cache["scales"]*(before_theta-cache["theta0"]);current={"stage":"projection","step":step};projection_started=time.perf_counter();projection_deadline=min(deadline,train_deadline,projection_started+SOLVE_CAP);solved=_project(proposal_beta,old_beta,L,projection_deadline);projection_seconds=time.perf_counter()-projection_started;_save_projection(output,step,old_beta,proposal_beta,grads.detach().cpu().numpy(),solved)
            trace={"step":step,"horizon":horizon,"loss":float(loss.detach()),"loss_components":{key:float(value.detach()) for key,value in parts.items()},"gradient_l2_before_clip":float(torch.linalg.vector_norm(grads)),"clip_return":norm,"proposal_distance":float(np.linalg.norm(proposal_beta-old_beta)),"projection_distance":float(np.linalg.norm(solved.get("beta",proposal_beta)-proposal_beta)),"projection_seconds":projection_seconds,"solver":jsonable({key:value for key,value in solved.items() if key not in ("K","g","L","F","ell","Y","Y_raw","beta")}),"elapsed":time.perf_counter()-started};traces.append(trace)
            with (output/"trace.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(trace,allow_nan=False)+"\n")
            if not solved.get("projection_pass",False):raise AssertionError("projection KKT/objective")
            set_theta(model,cache["theta0"]+solved["beta"]/cache["scales"])
            if not(np.array_equal(_theta(model),cache["theta0"]+solved["beta"]/cache["scales"]) and _frozen_equal(model,source,allowed) and all(torch.isfinite(value).all() for value in model.state_dict().values())):raise AssertionError("post projection invariant")
            require_time(min(deadline,train_deadline),"post update")
            if step%25==0:
                (output/"progress.json").write_text(json.dumps({"step":step,"traces":traces},indent=2,allow_nan=False));print(f"0087 step {step}/300 loss={trace['loss']:.6g}",flush=True)
        if not allow_final_forecast(STEPS,time.perf_counter(),min(deadline,train_deadline)):raise TimeoutError("incomplete updates cannot forecast")
        model.eval();candidate=output/"candidate";candidate.mkdir();torch.save(model.state_dict(),candidate/"model.pt");shutil.copyfile(baseline/"normalization.npz",candidate/"normalization.npz");model.structured.export(candidate/"structured_numeric_export.json");meta=json.loads((baseline/"checkpoint.json").read_text());(candidate/"checkpoint.json").write_text(json.dumps({"arm":"G-fixed","step":3300,"source":meta.get("source") or str(SOURCE),"parent":str(baseline),"loop":"0087"},indent=2));candidate_hashes=_candidate_hashes(candidate)
        affinity=affinity_qa(model,deterministic_test_z(),_theta(model)-cache["theta0"]);structural=structural_qa(model,candidate,deterministic_test_z());reloaded=load_model(candidate);reloaded.eval();zero=torch.zeros(33,dtype=torch.float64)
        with torch.enable_grad():actual=torch.autograd.functional.jacobian(lambda z:reloaded.field(z[None]).squeeze(0),zero).detach().cpu().numpy()
        expected=A33+np.einsum("j,rjs->rs",_theta(model)-cache["theta0"],D);actual_relative=float(np.linalg.norm(actual-expected,"fro")/max(1.,np.linalg.norm(expected,"fro")));actual_local=local_qa(Q.T@actual@Q,P)
        theta_final=_theta(model);beta_final=cache["scales"]*(theta_final-cache["theta0"]);qa={"affinity":affinity,"structural":structural,"actual_relative":actual_relative,"actual_local":actual_local};(output/"candidate_qa.json").write_text(json.dumps(qa,indent=2,allow_nan=False));np.savez_compressed(output/"fixed_arrays.npz",P=P,Q=Q,A33=A33,D=D,L=L,scales=cache["scales"],theta0=cache["theta0"],theta_final=theta_final,beta_final=beta_final,actual_A33=actual,actual_A32=Q.T@actual@Q);torch.save(optimizer.state_dict(),output/"final_optimizer.pt");torch.save({"numpy":rng.bit_generator.state,"torch":torch.get_rng_state()},output/"final_rng.pt")
        if actual_relative>1e-10 or actual_local["decay_max_eigenvalue"]>-.5 or actual_local["spectral_abscissa"]>=-1e-6 or actual_local["eigen_residual_max"]>1e-9:raise AssertionError("final AD guard")
        current={"stage":"cached baseline"};fit_evidence=json.loads((FIT/"evidence.json").read_text());table={**cache,"theta":_theta(model)};cache_base=_cached_baseline(FIT,fit_evidence,table,model,before)
        current={"stage":"final forecast"};forecast_deadline=min(deadline,time.perf_counter()+FORECAST_CAP);ids,labels,lengths,truth,qenc,mass,pscale=(cache_base[key] for key in ("event_ids","labels","lengths","truth","qenc","mass","p_scale"));from fit_coefficient_block_0082 import _checked_rollout
        states=torch.as_tensor(np.c_[truth[:,0],qenc],dtype=torch.float64);fine=_checked_rollout(model,states,lengths*2,.005,deadline=forecast_deadline,progress=current,partial=output/"partial_fine.npz",event_ids=ids);coarse=_checked_rollout(model,states,lengths,.01,deadline=forecast_deadline,progress=current,partial=output/"partial_coarse.npz",event_ids=ids);fine_base,coarse_base=cache_base["fine_baseline"],cache_base["coarse_baseline"]
        np.savez_compressed(output/"trajectories.npz",event_ids=ids,labels=labels,mass=mass,p_scale=pscale,conditional_speed_scale=CONDITIONAL_SPEED_SCALE,autonomous_metric=np.asarray(AUTONOMOUS_METRIC),lengths=lengths,truth=truth,qenc=qenc,fine_baseline=fine_base,coarse_baseline=coarse_base,fine_candidate=fine,coarse_candidate=coarse);require_time(forecast_deadline,"forecast")
        numerics=numeric(fine,coarse,lengths,mass,pscale)
        if not _finite(numerics) or any(value>.001 for row in numerics for value in row.values()):raise FloatingPointError("forecast numerical")
        rows=[];horizons={}
        for horizon in (1,2,5):
            values=[];base_values=[]
            for i,n in enumerate(lengths):
                k=min(round(horizon/.01),int(n));values.append(float(point(fine[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean()));base_values.append(float(point(fine_base[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean()));rows.append({"event_id":str(ids[i]),"label":str(labels[i]),"horizon":horizon,"candidate":values[-1],"baseline":base_values[-1],"available_steps":int(k)})
            horizons[str(horizon)]={"candidate":risk(values),"baseline":risk(base_values)}
        severe={name:{"mean":float(np.mean([row[name] for row in rows if row["horizon"]==5 and row["label"]=="unstable"])),"worst4":float(np.mean(sorted(row[name] for row in rows if row["horizon"]==5 and row["label"]=="unstable")[-4:]))} for name in ("baseline","candidate")};average={name:float(np.mean([horizons[str(horizon)][name]["risk"] for horizon in (1,2,5)])) for name in ("baseline","candidate")}
        require_time(deadline,"before final evidence");after=hashes(baseline);candidate_after=_candidate_hashes(candidate)
        if before!=after or candidate_after!=candidate_hashes or not np.array_equal(_theta(model),cache["theta0"]+solved["beta"]/cache["scales"]) or not _frozen_equal(model,source,allowed):raise AssertionError("final provenance/frozen")
        result={"loop":"0087","mode":"train","completed_updates":STEPS,"projection_attempts":len(traces),"test_read":False,"h5_read":did_h5,"validation_materialized_by_reader":did_h5,"hashes_before":before,"hashes_after":after,"identities":identities,"excluded":excluded,"candidate_hashes":candidate_hashes,"traces":traces,"candidate_qa":qa,"autonomous_numerics":numerics,"autonomous_horizons":horizons,"autonomous_average_risk":average,"unstable_5s":severe,"per_event":rows,"seconds":time.perf_counter()-started}
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False));return result
    except Exception as error:_write_failure(output,error,current,before,model,optimizer,traces,did_h5);raise

def main()->None:
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument("--preflight",action="store_true");group.add_argument("--train",action="store_true");parser.add_argument("--baseline",type=Path,default=BASELINE);parser.add_argument("--preflight-dir",type=Path,default=PRE);parser.add_argument("--output",type=Path);args=parser.parse_args()
    if args.preflight:preflight(args.baseline,args.output or args.preflight_dir)
    else:train(args.baseline,args.preflight_dir,args.output or OUT)
if __name__=="__main__":main()
