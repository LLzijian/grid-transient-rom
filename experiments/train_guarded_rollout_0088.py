"""Loop0088: the sole compute-granted, otherwise unchanged Loop0087 training run."""
from __future__ import annotations
import argparse, hashlib, json, shutil, time
from pathlib import Path
from typing import Any
import os
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
from audit_force_initialization_0080 import numeric,point,risk  # noqa:E402
from audit_lyapunov_guard_0085 import BASELINE,BASELINE_SHA,_cached_baseline,_candidate_hashes,_finite,local_qa  # noqa:E402
from fit_coefficient_block_0082 import AUTONOMOUS_METRIC,CONDITIONAL_SPEED_SCALE,_theta,affinity_qa,deterministic_test_z,set_theta,structural_qa  # noqa:E402
from fit_guarded_regression_0086 import ANCHORS,FIT,GUARD,SOLVE_CAP,hashes as hashes86,jsonable  # noqa:E402
from train_guarded_rollout_0087 import (BATCH,FORCE,NULL,POLES,SEED,STEPS,_events,_frozen_equal,_load_fixed,_project,_save_projection,_snapshot,_write_failure,allow_final_forecast,theta_parameters)  # noqa:E402
from train_joint_campaign import sample_batch  # noqa:E402
from train_structured_rom import SOURCE,load_model,objective  # noqa:E402

PRE=ROOT/"private_data/task2_guarded_rollout_0087_preflight";OUT=ROOT/"private_data/task2_guarded_rollout_0088"
PRE_SHA="5db2032a784eec51179d00f50cd71e33c7ded33bb687925d7123976b001719dd";SOURCE_SHA="607892c92afbe5f2babab61c23d183ea47c1e569b36a4326f4ab45dc39808115"
PRIOR_SECONDS=35.83270050000283;TRAIN_CAP=3600.;MAIN_CAP=5400.-PRIOR_SECONDS;FORECAST_CAP=1200.
EXPECTED_ESTIMATE=2900.435699999798

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes(baseline:Path)->dict[str,str]:
    result=hashes86(baseline,FIT,POLES,NULL,GUARD);result["audit0087"]=sha(ROOT/"experiments/train_guarded_rollout_0087.py");result["audit0088"]=sha(Path(__file__));return result
def _same_inputs(old:dict[str,str],current:dict[str,str])->bool:
    now=dict(current);now.pop("audit0088",None);return old==now
def grant_record()->dict[str,float|str]:return {"preflight_evidence_sha256":PRE_SHA,"source0087_sha256":SOURCE_SHA,"previous_actual_seconds":PRIOR_SECONDS,"previous_estimate_seconds":EXPECTED_ESTIMATE,"new_training_cap_seconds":TRAIN_CAP,"new_main_cap_seconds":MAIN_CAP,"forecast_cap_seconds":FORECAST_CAP,"aggregate_cap_seconds":5400.}
def validate_preflight(path:Path,current:dict[str,str])->dict[str,Any]:
    if sha(path/"evidence.json")!=PRE_SHA or sha(ROOT/"experiments/train_guarded_rollout_0087.py")!=SOURCE_SHA:raise AssertionError("0087 anchor SHA")
    pre=json.loads((path/"evidence.json").read_text());traces=pre.get("traces",[]);times=tuple(float(row["seconds"]) for row in traces)
    recomputed=100*sum(times)+150
    if not(pre.get("completed") and pre.get("authorized") is False and [float(row["horizon"]) for row in traces]==[1.,2.,5.] and len(times)==3 and all(np.isfinite(value) and value>=0 for value in times) and recomputed==float(pre.get("estimate_seconds")) and recomputed>2400 and pre.get("hashes_before")==pre.get("hashes_after") and _same_inputs(pre["hashes_before"],current)):raise AssertionError("honest failed0087 preflight provenance")
    return pre
def require_time(deadline:float,stage:str)->None:
    if time.perf_counter()>deadline:raise TimeoutError(f"0088 cap during {stage}")

def train(baseline:Path,preflight_dir:Path,output:Path)->dict[str,Any]:
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);started=time.perf_counter();before=hashes(baseline);pre=validate_preflight(preflight_dir,before);deadline=started+MAIN_CAP;train_deadline=started+TRAIN_CAP;output.mkdir();model=None;optimizer=None;traces=[];current={"stage":"load"};did_h5=False
    try:
        if before["baseline_model"]!=BASELINE_SHA or any(before[key]!=value for key,value in ANCHORS.items()):raise AssertionError("input SHA")
        cache,Q,A33,D,P,L=_load_fixed();torch.manual_seed(SEED);model=load_model(baseline);model.train();params=theta_parameters(model);source=_snapshot(model);allowed={name for name,param in model.named_parameters() if param.requires_grad};np.savez_compressed(output/"fixed_arrays.npz",P=P,Q=Q,A33=A33,D=D,L=L,scales=cache["scales"],theta0=cache["theta0"])
        if not np.array_equal(_theta(model),cache["theta0"]):raise AssertionError("baseline theta")
        optimizer=torch.optim.Adam(params,lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False);did_h5=True;train_events,excluded,identities=_events();rng=np.random.default_rng(SEED)
        for step in range(1,STEPS+1):
            require_time(min(deadline,train_deadline),"training");horizon=(1.,2.,5.)[(step-1)%3];samples,histories=sample_batch(train_events,rng,BATCH);optimizer.zero_grad(set_to_none=True);model.begin_graph_cache()
            try:loss,parts=objective(model,samples,histories,horizon,complexity_weight=0);loss.backward()
            finally:model.end_graph_cache()
            grads=torch.cat([param.grad.detach().reshape(-1) for param in params])
            if not(torch.isfinite(loss) and torch.isfinite(grads).all()):raise FloatingPointError("nonfinite train loss/gradient")
            before_theta=_theta(model);norm=float(torch.nn.utils.clip_grad_norm_(params,100.));optimizer.step();proposal_theta=_theta(model);proposal_beta=cache["scales"]*(proposal_theta-cache["theta0"]);old_beta=cache["scales"]*(before_theta-cache["theta0"]);current={"stage":"projection","step":step};projection_started=time.perf_counter();solved=_project(proposal_beta,old_beta,L,min(deadline,train_deadline,projection_started+SOLVE_CAP));projection_seconds=time.perf_counter()-projection_started;_save_projection(output,step,old_beta,proposal_beta,grads.detach().cpu().numpy(),solved)
            trace={"step":step,"horizon":horizon,"loss":float(loss.detach()),"loss_components":{key:float(value.detach()) for key,value in parts.items()},"gradient_l2_before_clip":float(torch.linalg.vector_norm(grads)),"clip_return":norm,"proposal_distance":float(np.linalg.norm(proposal_beta-old_beta)),"projection_distance":float(np.linalg.norm(solved.get("beta",proposal_beta)-proposal_beta)),"projection_seconds":projection_seconds,"solver":jsonable({key:value for key,value in solved.items() if key not in ("K","g","L","F","ell","Y","Y_raw","beta")}),"elapsed":time.perf_counter()-started};traces.append(trace)
            with (output/"trace.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(trace,allow_nan=False)+"\n")
            if not solved.get("projection_pass",False):raise AssertionError("projection KKT/objective")
            set_theta(model,cache["theta0"]+solved["beta"]/cache["scales"])
            if not(np.array_equal(_theta(model),cache["theta0"]+solved["beta"]/cache["scales"]) and _frozen_equal(model,source,allowed) and all(torch.isfinite(value).all() for value in model.state_dict().values())):raise AssertionError("post projection invariant")
            require_time(min(deadline,train_deadline),"post update")
            if step%25==0:(output/"progress.json").write_text(json.dumps({"step":step,"traces":traces},indent=2,allow_nan=False));print(f"0088 step {step}/300 loss={trace['loss']:.6g}",flush=True)
        if not allow_final_forecast(STEPS,time.perf_counter(),min(deadline,train_deadline)):raise TimeoutError("incomplete updates cannot forecast")
        training_seconds=time.perf_counter()-started
        model.eval();candidate=output/"candidate";candidate.mkdir();torch.save(model.state_dict(),candidate/"model.pt");shutil.copyfile(baseline/"normalization.npz",candidate/"normalization.npz");model.structured.export(candidate/"structured_numeric_export.json");meta=json.loads((baseline/"checkpoint.json").read_text());(candidate/"checkpoint.json").write_text(json.dumps({"arm":"G-fixed","step":3300,"source":meta.get("source") or str(SOURCE),"parent":str(baseline),"loop":"0088","resource_reauthorization":grant_record()},indent=2));candidate_hashes=_candidate_hashes(candidate)
        affinity=affinity_qa(model,deterministic_test_z(),_theta(model)-cache["theta0"]);structural=structural_qa(model,candidate,deterministic_test_z());reloaded=load_model(candidate);reloaded.eval();zero=torch.zeros(33,dtype=torch.float64)
        with torch.enable_grad():actual=torch.autograd.functional.jacobian(lambda z:reloaded.field(z[None]).squeeze(0),zero).detach().cpu().numpy()
        expected=A33+np.einsum("j,rjs->rs",_theta(model)-cache["theta0"],D);actual_relative=float(np.linalg.norm(actual-expected,"fro")/max(1.,np.linalg.norm(expected,"fro")));actual_local=local_qa(Q.T@actual@Q,P);theta_final=_theta(model);beta_final=cache["scales"]*(theta_final-cache["theta0"]);qa={"affinity":affinity,"structural":structural,"actual_relative":actual_relative,"actual_local":actual_local};(output/"candidate_qa.json").write_text(json.dumps(qa,indent=2,allow_nan=False));np.savez_compressed(output/"fixed_arrays.npz",P=P,Q=Q,A33=A33,D=D,L=L,scales=cache["scales"],theta0=cache["theta0"],theta_final=theta_final,beta_final=beta_final,actual_A33=actual,actual_A32=Q.T@actual@Q);torch.save(optimizer.state_dict(),output/"final_optimizer.pt");torch.save({"numpy":rng.bit_generator.state,"torch":torch.get_rng_state()},output/"final_rng.pt")
        if actual_relative>1e-10 or actual_local["decay_max_eigenvalue"]>-.5 or actual_local["spectral_abscissa"]>=-1e-6 or actual_local["eigen_residual_max"]>1e-9:raise AssertionError("final AD guard")
        current={"stage":"cached baseline"};fit_evidence=json.loads((FIT/"evidence.json").read_text());table={**cache,"theta":_theta(model)};cache_base=_cached_baseline(FIT,fit_evidence,table,model,before)
        current={"stage":"final forecast"};forecast_started=time.perf_counter();forecast_deadline=min(deadline,forecast_started+FORECAST_CAP);ids,labels,lengths,truth,qenc,mass,pscale=(cache_base[key] for key in ("event_ids","labels","lengths","truth","qenc","mass","p_scale"));from fit_coefficient_block_0082 import _checked_rollout
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
        forecast_seconds=time.perf_counter()-forecast_started;require_time(deadline,"before final evidence");after=hashes(baseline);candidate_after=_candidate_hashes(candidate)
        if before!=after or candidate_after!=candidate_hashes or not np.array_equal(_theta(model),cache["theta0"]+solved["beta"]/cache["scales"]) or not _frozen_equal(model,source,allowed):raise AssertionError("final provenance/frozen")
        new_seconds=time.perf_counter()-started;result={"loop":"0088","mode":"train","resource_reauthorization":grant_record(),"timers":{"previous_preflight_seconds":PRIOR_SECONDS,"training_seconds":training_seconds,"forecast_seconds":forecast_seconds,"new_main_seconds":new_seconds,"aggregate_seconds":PRIOR_SECONDS+new_seconds,"new_main_cap_seconds":MAIN_CAP,"training_cap_seconds":TRAIN_CAP,"forecast_cap_seconds":FORECAST_CAP},"completed_updates":STEPS,"projection_attempts":len(traces),"test_read":False,"h5_read":did_h5,"validation_materialized_by_reader":did_h5,"hashes_before":before,"hashes_after":after,"identities":identities,"excluded":excluded,"candidate_hashes":candidate_hashes,"traces":traces,"candidate_qa":qa,"autonomous_numerics":numerics,"autonomous_horizons":horizons,"autonomous_average_risk":average,"unstable_5s":severe,"per_event":rows,"seconds":new_seconds}
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False));return result
    except Exception as error:
        _write_failure(output,error,current,before,model,optimizer,traces,did_h5);failure=output/"failure_evidence.json"
        if failure.exists():
            payload=json.loads(failure.read_text());new_seconds=time.perf_counter()-started
            payload.update({"loop":"0088","resource_reauthorization":grant_record(),"new_main_seconds":new_seconds,"aggregate_seconds":PRIOR_SECONDS+new_seconds})
            failure.write_text(json.dumps(payload,indent=2,allow_nan=False))
        raise

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--train",action="store_true",required=True);parser.add_argument("--baseline",type=Path,default=BASELINE);parser.add_argument("--preflight-dir",type=Path,default=PRE);parser.add_argument("--output",type=Path,default=OUT);args=parser.parse_args();train(args.baseline,args.preflight_dir,args.output)
if __name__=="__main__":main()
