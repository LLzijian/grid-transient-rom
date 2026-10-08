"""Loop0086: one fixed-P, 24-amplitude convex guarded regression."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
from typing import Any
os.environ["OMP_NUM_THREADS"]="1";os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import torch

ROOT=Path(__file__).resolve().parents[1]
# NumPy/SciPy above intentionally come from the main runtime before these isolated solver wheels.
DEPS=ROOT/".loop-work/research_deps_0067"
if not DEPS.is_dir(): raise RuntimeError("registered Loop0067 solver dependency is absent")
sys.path[:0]=[str(DEPS),str(ROOT/"experiments"),str(ROOT/"src")]
from audit_force_initialization_0080 import numeric,point,risk  # noqa:E402
from audit_lyapunov_guard_0085 import (BASELINE,BASELINE_SHA,CACHE_SHA,FORCE_BASELINE,PROPOSAL_SHA,_cached_baseline,_candidate_hashes,_finite,hashes as hashes85,local_qa)  # noqa:E402
from fit_coefficient_block_0082 import (AUTONOMOUS_METRIC,CONDITIONAL_SPEED_SCALE,_design,_table,_theta,affinity_qa,conditional_summary,deterministic_test_z,set_theta,structural_qa)  # noqa:E402
from train_joint_latent import BASE,NETWORK,ROOTS,read_development  # noqa:E402
from train_structured_rom import SOURCE,load_model  # noqa:E402
from kiki.models.private_pm import MechanicalROM  # noqa:E402

FIT=ROOT/"private_data/task2_coefficient_block_0082_attempt2"
POLES=ROOT/"private_data/task2_coefficient_poles_0083/matrices.npz"
NULL=ROOT/"private_data/task2_coefficient_nullspace_0084/derivatives.npz"
GUARD=ROOT/"private_data/task2_lyapunov_guard_0085/guard_arrays.npz"
FORCE=ROOT/"private_data/task2_structure_force_0079"
WALL,SOLVE_CAP,REBUILD_CAP,FORECAST_CAP=3600.,300.,1800.,1200.
ANCHORS={**CACHE_SHA,"null":"8817cff75d4d6308c561f8eee4f2362292ba40c119a096f10bb611dcf58aaf53","guard":"eeca4bce9c3233f6bdcc81e94ac8f24a8e2f5805ad4d20be151abd092800d640"}

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def hashes(baseline:Path,fit:Path,poles:Path,null:Path,guard:Path)->dict[str,str]:
    result=hashes85(baseline,fit,poles);result.update({"null":sha(null),"guard":sha(guard),"force_arrays":sha(FORCE/"projection_arrays.npz"),"force_evidence":sha(FORCE/"evidence.json"),"audit0086":sha(Path(__file__)),"null_audit":sha(ROOT/"experiments/audit_coefficient_nullspace_0084.py"),"guard_audit":sha(ROOT/"experiments/audit_lyapunov_guard_0085.py")})
    for root in ROOTS:result[f"{root.name}_csv"]=sha(root/"dataset_index.csv");result[f"{root.name}_splits"]=sha(root/"splits.json")
    return result
def require_time(deadline:float,stage:str)->None:
    if time.perf_counter()>deadline:raise TimeoutError(f"0086 wall cap during {stage}")
def jsonable(value:Any)->Any:
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:jsonable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(item) for item in value]
    return value

def coefficient_lmis(D:np.ndarray,Q:np.ndarray,scales:np.ndarray,P:np.ndarray)->np.ndarray:
    D,Q,scales,P=np.asarray(D,float),np.asarray(Q,float),np.asarray(scales,float),np.asarray(P,float)
    if D.shape!=(33,24,33) or Q.shape!=(33,32) or scales.shape!=(24,) or P.shape!=(32,32) or np.any(scales<=0) or not all(np.isfinite(x).all() for x in (D,Q,scales,P)):raise ValueError("coefficient LMI input")
    E=np.stack([Q.T@D[:,j,:]@Q/scales[j] for j in range(24)])
    L=np.asarray([((e.T@P+P@e)+(e.T@P+P@e).T)/2 for e in E])
    return L

def kkt(K:np.ndarray,g:np.ndarray,L:np.ndarray,beta:np.ndarray,Y:np.ndarray)->dict[str,Any]:
    K,g,L,beta,Y=(np.asarray(v,float) for v in (K,g,L,beta,Y));n=len(g)
    if g.ndim!=1 or Y.ndim!=2 or K.shape!=(n,n) or L.shape!=(n,Y.shape[0],Y.shape[1]) or beta.shape!=(n,) or Y.shape[0]!=Y.shape[1] or not all(np.isfinite(v).all() for v in (K,g,L,beta,Y)):raise ValueError("KKT shape/finite")
    K=(K+K.T)/2;Y=(Y+Y.T)/2;F=.49*np.eye(Y.shape[0])-np.einsum("j,jab->ab",beta,L);ell=np.einsum("ab,jab->j",Y,L);primal=float(beta@K@beta-2*g@beta);dual=float(-(g-.5*ell)@np.linalg.solve(K,g-.5*ell)-.49*np.trace(Y));stationarity=float(np.linalg.norm(2*(K@beta-g)+ell)/max(1.,2*np.linalg.norm(g)));complementarity=float(abs(np.trace(Y@F))/max(1.,abs(primal),abs(dual)));gap=float(abs(primal-dual)/max(1.,abs(primal),abs(dual)))
    return {"F":F,"ell":ell,"primal":primal,"dual":dual,"stationarity":stationarity,"complementarity":complementarity,"gap":gap,"F_min_eigenvalue":float(np.linalg.eigvalsh(F).min()),"Y_min_eigenvalue":float(np.linalg.eigvalsh(Y).min())}

def solve_sdp(K:np.ndarray,g:np.ndarray,L:np.ndarray,deadline:float=float("inf"))->dict[str,Any]:
    require_time(deadline,"solver start");K=np.asarray(K,float);g=np.asarray(g,float);L=np.asarray(L,float);n=len(g)
    if g.ndim!=1 or K.shape!=(n,n) or L.shape!=(n,32,32) or not(np.isfinite(K).all() and np.isfinite(g).all() and np.isfinite(L).all()):raise ValueError("SDP input")
    K=(K+K.T)/2
    if np.linalg.eigvalsh(K).min()<=0:raise AssertionError("K_s must be positive definite")
    import cvxpy as cp
    beta=cp.Variable(n);F=.49*np.eye(32)-sum(beta[j]*L[j] for j in range(n));constraint=F >> 0
    problem=cp.Problem(cp.Minimize(cp.quad_form(beta,K)-2*g@beta),[constraint])
    kwargs={"solver":"CLARABEL","max_iter":200,"time_limit":SOLVE_CAP,"tol_gap_abs":1e-9,"tol_gap_rel":1e-9,"tol_feas":1e-9,"warm_start":False}
    versions={"numpy":np.__version__,"scipy":__import__("scipy").__version__}
    try:
        problem.solve(solver="CLARABEL",max_iter=200,time_limit=SOLVE_CAP,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,warm_start=False,verbose=False)
    except Exception as exc:return {"K":K,"g":g,"L":L,"status":"exception","reason":str(exc),"qa_pass":False,"versions":versions,"solver_kwargs":kwargs}
    versions.update({"cvxpy":cp.__version__,"clarabel":__import__("clarabel").__version__})
    if problem.status not in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE) or beta.value is None or constraint.dual_value is None:
        partial={"K":K,"g":g,"L":L,"status":str(problem.status),"reason":"missing optimal primal/dual","qa_pass":False,"versions":versions,"solver_kwargs":kwargs,"iterations":getattr(problem.solver_stats,"num_iters",None),"solve_time":getattr(problem.solver_stats,"solve_time",None)}
        if beta.value is not None:partial["beta"]=np.asarray(beta.value,float)
        if constraint.dual_value is not None:
            partial["Y_raw"]=np.asarray(constraint.dual_value,float);partial["Y"]=(partial["Y_raw"]+partial["Y_raw"].T)/2
        return partial
    raw_beta=np.asarray(beta.value,float);raw_y=np.asarray(constraint.dual_value,float);sym_y=(raw_y+raw_y.T)/2;common={"beta":raw_beta,"Y":sym_y,"Y_raw":raw_y,"K":K,"g":g,"L":L,"status":str(problem.status),"objective":float(problem.value),"versions":versions,"solver_kwargs":kwargs,"iterations":getattr(problem.solver_stats,"num_iters",None),"solve_time":getattr(problem.solver_stats,"solve_time",None)}
    try:out=kkt(K,g,L,raw_beta,sym_y);out.update(common)
    except Exception as exc:return {**common,"reason":f"KKT construction: {exc}","qa_pass":False}
    timed_out=time.perf_counter()>deadline
    out["qa_pass"]=bool(not timed_out and _finite(out) and out["F_min_eigenvalue"]>=-1e-8 and out["Y_min_eigenvalue"]>=-1e-8 and max(out["stationarity"],out["complementarity"],out["gap"])<=1e-6)
    if timed_out:out["reason"]="solver deadline exceeded after Clarabel return"
    return out

def _relative(actual:np.ndarray,reference:np.ndarray)->float:return float(np.linalg.norm(np.asarray(actual)-np.asarray(reference))/max(1.,np.linalg.norm(np.asarray(reference))))
def _maxrelative(actual:np.ndarray,reference:np.ndarray)->float:return float(np.max(np.abs(np.asarray(actual)-np.asarray(reference)))/max(1.,float(np.max(np.abs(actual))),float(np.max(np.abs(reference)))))
def _conditional(table:dict[str,Any],design:dict[str,np.ndarray],delta:np.ndarray)->tuple[dict[str,Any],dict[str,np.ndarray]]:
    c,A,cc,Ac=(design[k] for k in ("c","A","coarse_c","coarse_A"));fine=c+np.einsum("wkj,j->wk",A,delta);coarse=cc+np.einsum("wkj,j->wk",Ac,delta)
    arrays={"fine_baseline":c,"fine_candidate":fine,"coarse_baseline":cc,"coarse_candidate":coarse,"numerical":np.linalg.norm(fine-coarse,axis=1)}
    if not all(np.isfinite(value).all() for value in (fine,coarse)):return {"finite":False,"reason":"nonfinite conditional residual"},arrays
    mapping={key:table[key] for key in ("event_index","label","group","start")};summary={"baseline":conditional_summary(mapping,c),"candidate":conditional_summary(mapping,fine)};ratios={"total_ratio":summary["candidate"]["overall"]["total"]/summary["baseline"]["overall"]["total"],"physical_ratio":summary["candidate"]["overall"]["physical"]/summary["baseline"]["overall"]["physical"],"unstable_total_ratio":summary["candidate"]["subsets"]["unstable"]["total"]/summary["baseline"]["subsets"]["unstable"]["total"]}
    return {"finite":True,"summary":summary,"original_reference_gates":{"total_ratio":{"reference":.90,"observed":ratios["total_ratio"]},"physical_ratio":{"reference":1.05,"observed":ratios["physical_ratio"]},"unstable_total_ratio":{"reference":1.05,"observed":ratios["unstable_total_ratio"]}}},arrays

def _rebuild(model,deadline:float,current:dict[str,Any])->tuple[dict[str,Any],dict[str,np.ndarray],dict[str,Any]]:
    base=MechanicalROM.load(BASE);events,excluded,identities=read_development(base, __import__("fit_coefficient_block_0082")._network(),full_train=True);train=events["train"]
    if len(train)!=346 or sum(e["label"]=="stable" for e in train)!=311 or sum(e["label"]=="unstable" for e in train)!=35 or len({e["group"] for e in train})!=24 or sum(v=="train" for v in excluded.values())!=23:raise AssertionError("TRAIN identity")
    force_evidence=json.loads((FORCE/"evidence.json").read_text())
    with np.load(FORCE/"projection_arrays.npz",allow_pickle=False) as saved:table=_table(train,saved,force_evidence)
    design=_design(model,table,deadline,current);return table,design,{"identities":identities,"excluded":excluded,"force_evidence":force_evidence}

def run(baseline:Path,fit:Path,poles:Path,null:Path,guard:Path,output:Path)->dict[str,Any]:
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+WALL;before=hashes(baseline,fit,poles,null,guard);output.mkdir();current={"stage":"provenance"};arrays={};timers={};did_read_h5=False
    try:
        if before["baseline_model"]!=BASELINE_SHA or before["proposal_model"]!=PROPOSAL_SHA or any(before[key]!=value for key,value in ANCHORS.items()):raise AssertionError("preregistered input SHA mismatch")
        with np.load(fit/"fit_arrays.npz",allow_pickle=False) as saved:cached={key:np.asarray(saved[key]) for key in ("c","A","weights","event_index","start","end","event_id","label","group","L_end","theta0","theta","scales","K","g","active")}
        with np.load(poles,allow_pickle=False) as saved:Q=np.asarray(saved["baseline_Q"],float);A33=np.asarray(saved["baseline_A33"],float);proposal_a33=np.asarray(saved["candidate_A33"],float)
        with np.load(null,allow_pickle=False) as saved:D=np.asarray(saved["D"],float);null_theta=np.asarray(saved["theta_delta"],float);null_reference=np.asarray(saved["reference_A_difference"],float)
        with np.load(guard,allow_pickle=False) as saved:P=np.asarray(saved["P"],float);guard_theta=np.asarray(saved["theta_guard"],float);guard_q=np.asarray(saved["Q"],float);guard_a0=np.asarray(saved["A0"],float);guard_t0=np.asarray(saved["theta0"],float);guard_tp=np.asarray(saved["theta_proposal"],float)
        K=(cached["K"]+cached["K"].T)/2;g=np.asarray(cached["g"],float);scales=np.asarray(cached["scales"],float)
        if not(np.array_equal(cached["active"],np.ones(24,dtype=bool)) and np.array_equal(guard_q,Q) and np.array_equal(guard_t0,cached["theta0"]) and np.array_equal(guard_tp,cached["theta"]) and np.array_equal(null_theta,cached["theta"]-cached["theta0"]) and _maxrelative(null_reference,proposal_a33-A33)<=1e-10 and _maxrelative(np.einsum("j,rjs->rs",null_theta,D),null_reference)<=1e-10 and np.linalg.eigvalsh(K).min()>0):raise AssertionError("0082/0084/0085 provenance")
        A0=Q.T@A33@Q;pres=float(np.linalg.norm(A0.T@P+P@A0+np.eye(32),"fro")/np.sqrt(32))
        if not np.array_equal(guard_a0,A0) or np.linalg.eigvalsh(P).min()<=1e-10 or pres>1e-9:raise AssertionError("fixed P provenance")
        L=coefficient_lmis(D,Q,scales,P)
        beta85=scales*(guard_theta-cached["theta0"]);objective85=float(beta85@K@beta85-2*g@beta85);F85=.49*np.eye(32)-np.einsum("j,jab->ab",beta85,L)
        if np.linalg.eigvalsh(F85).min()<-1e-8:raise AssertionError("guarded0085 feasibility QA")
        arrays={"K":K,"g":g,"scales":scales,"P":P,"Q":Q,"A0":A0,"L":L,"D":D,"theta0":cached["theta0"],"theta85":guard_theta,"beta85":beta85,"F85":F85,"objective85":np.asarray(objective85)}
        solve_started=time.perf_counter();current={"stage":"single SDP"};solved=solve_sdp(K,g,L,min(deadline,solve_started+SOLVE_CAP));timers["solve_seconds"]=time.perf_counter()-solve_started
        arrays.update({key:solved[key] for key in ("beta","Y","Y_raw") if key in solved});np.savez_compressed(output/"solve_arrays.npz",**arrays);(output/"solve_evidence.json").write_text(json.dumps(jsonable({key:value for key,value in solved.items() if key not in ("beta","Y","Y_raw","K","g","L","F","ell")}),indent=2,allow_nan=False))
        if not solved.get("qa_pass",False):raise AssertionError("SDP status/KKT QA")
        if solved["primal"]>objective85+1e-7*max(1.,abs(objective85)):raise AssertionError("guarded0085 objective QA")
        delta=solved["beta"]/scales;theta=cached["theta0"]+delta;arrays["theta"]=theta;np.savez_compressed(output/"solve_arrays.npz",**arrays)
        current={"stage":"quadrature rebuild"};model=load_model(baseline);model.eval();frozen={key:value.detach().clone() for key,value in model.state_dict().items()}
        if not np.array_equal(_theta(model),cached["theta0"]):raise AssertionError("baseline theta/cache mismatch")
        did_read_h5=True;rebuild_started=time.perf_counter();table,design,rebuild_info=_rebuild(model,min(deadline,rebuild_started+REBUILD_CAP),current);timers["rebuild_seconds"]=time.perf_counter()-rebuild_started;require_time(min(deadline,rebuild_started+REBUILD_CAP),"quadrature rebuild")
        fit_evidence=json.loads((fit/"evidence.json").read_text())
        fit_before=fit_evidence["hashes_before"]
        if rebuild_info["identities"]!=fit_evidence["identities"] or rebuild_info["identities"]!=rebuild_info["force_evidence"]["identities"] or rebuild_info["excluded"]!=fit_evidence["excluded"] or rebuild_info["excluded"]!=rebuild_info["force_evidence"]["excluded"] or fit_before["force_arrays"]!=before["force_arrays"] or fit_before["force_evidence"]!=before["force_evidence"] or any(fit_before[f"{root.name}_{kind}"]!=before[f"{root.name}_{kind}"] for root in ROOTS for kind in ("csv","splits")):raise AssertionError("0082/0079 split provenance")
        for key,reference in (("c",cached["c"]),("A",cached["A"]),("weights",cached["weights"]),("L",cached["L_end"])):
            actual=design[key] if key in design else table[key]
            if _maxrelative(actual,reference)>1e-10:raise AssertionError(f"0082 quadrature reproduction {key}")
        for key in ("event_index","start","end"):
            if not np.array_equal(np.asarray(table[key]),cached[key]):raise AssertionError(f"0082 exact integer mapping {key}")
        if not(np.array_equal(np.asarray(table["event_id"]).astype(str),cached["event_id"].astype(str)) and np.array_equal(np.asarray(table["label"]).astype(str),cached["label"].astype(str)) and np.array_equal(np.asarray(table["group"]).astype(str),cached["group"].astype(str))):raise AssertionError("0082 mapping reproduction")
        conditional,conditional_arrays=_conditional(table,design,delta);np.savez_compressed(output/"conditional_arrays.npz",**conditional_arrays,fine_A=design["A"],fine_c=design["c"],coarse_A=design["coarse_A"],coarse_c=design["coarse_c"],weights=design["weights"],event_index=table["event_index"],event_id=np.asarray(table["event_id"]),label=np.asarray(table["label"]),group=np.asarray(table["group"]),start=table["start"],end=table["end"],L=table["L"])
        (output/"conditional_evidence.json").write_text(json.dumps(jsonable(conditional),indent=2,allow_nan=False));
        if not conditional["finite"] or not np.isfinite(conditional_arrays["numerical"]).all() or conditional_arrays["numerical"].max()>.001:raise FloatingPointError("conditional numerical gate")
        require_time(deadline,"conditional")
        affinity=affinity_qa(model,deterministic_test_z(),delta);set_theta(model,theta);allowed=[f"structured.storage_gamma.{i}" for i in range(8)]+["structured.port_coefficient.0","structured.port_coefficient.1"];changed=[key for key,value in model.state_dict().items() if not torch.equal(value,frozen[key])]
        if not set(changed).issubset(allowed) or any(not torch.equal(value,model.state_dict()[key]) for key,value in frozen.items() if key not in allowed):raise AssertionError("illegal candidate state change")
        candidate=output/"candidate";candidate.mkdir();torch.save(model.state_dict(),candidate/"model.pt");shutil.copyfile(baseline/"normalization.npz",candidate/"normalization.npz");model.structured.export(candidate/"structured_numeric_export.json");meta=json.loads((baseline/"checkpoint.json").read_text());(candidate/"checkpoint.json").write_text(json.dumps({"arm":"G-fixed","step":3000,"source":meta.get("source") or str(SOURCE),"parent":str(baseline),"loop":"0086"},indent=2));candidate_hashes=_candidate_hashes(candidate)
        structural=structural_qa(model,candidate,deterministic_test_z());reloaded=load_model(candidate);reloaded.eval();zero=torch.zeros(33,dtype=torch.float64)
        with torch.enable_grad():actual=torch.autograd.functional.jacobian(lambda z:reloaded.field(z[None]).squeeze(0),zero).detach().cpu().numpy()
        expected=A33+np.einsum("j,rjs->rs",delta,D);actual_relative=_relative(actual,expected);actual_local=local_qa(Q.T@actual@Q,P)
        arrays.update({"actual_A33":actual,"actual_A32":Q.T@actual@Q});np.savez_compressed(output/"solve_arrays.npz",**arrays)
        qa={"actual_relative":actual_relative,"actual_local":actual_local,"affinity":affinity,"structural":structural,"allowed_state_keys_changed":changed};(output/"candidate_qa.json").write_text(json.dumps(qa,indent=2,allow_nan=False))
        if actual_relative>1e-10 or actual_local["decay_max_eigenvalue"]>-.5 or actual_local["spectral_abscissa"]>=-1e-6 or actual_local["eigen_residual_max"]>1e-9:raise AssertionError("candidate AD/guard QA")
        current={"stage":"cached baseline verification"};cache=_cached_baseline(fit,fit_evidence,{**cached,"theta":theta},model,before)
        current={"stage":"candidate forecast"};forecast_started=time.perf_counter();forecast_deadline=min(deadline,forecast_started+FORECAST_CAP);ids,labels,lengths,truth,qenc,mass,pscale=(cache[key] for key in ("event_ids","labels","lengths","truth","qenc","mass","p_scale"));fine_base,coarse_base=cache["fine_baseline"],cache["coarse_baseline"]
        from fit_coefficient_block_0082 import _checked_rollout
        states=torch.as_tensor(np.c_[truth[:,0],qenc],dtype=torch.float64);fine=_checked_rollout(model,states,lengths*2,.005,deadline=forecast_deadline,progress=current,partial=output/"partial_fine.npz",event_ids=ids);coarse=_checked_rollout(model,states,lengths,.01,deadline=forecast_deadline,progress=current,partial=output/"partial_coarse.npz",event_ids=ids)
        np.savez_compressed(output/"trajectories.npz",event_ids=ids,labels=labels,mass=mass,p_scale=pscale,conditional_speed_scale=CONDITIONAL_SPEED_SCALE,autonomous_metric=np.asarray(AUTONOMOUS_METRIC),lengths=lengths,truth=truth,qenc=qenc,fine_baseline=fine_base,coarse_baseline=coarse_base,fine_candidate=fine,coarse_candidate=coarse);timers["forecast_seconds"]=time.perf_counter()-forecast_started;require_time(forecast_deadline,"forecast cap")
        numerics=numeric(fine,coarse,lengths,mass,pscale)
        if not _finite(numerics) or any(value>.001 for row in numerics for value in row.values()):raise FloatingPointError("autonomous numerical gate")
        rows=[];horizons={}
        for h in (1,2,5):
            values=[];base_values=[]
            for i,n in enumerate(lengths):
                k=min(round(h/.01),int(n));values.append(float(point(fine[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean()));base_values.append(float(point(fine_base[i,2:2*k+1:2],truth[i,1:k+1],mass,pscale).mean()));rows.append({"event_id":str(ids[i]),"label":str(labels[i]),"horizon":h,"candidate":values[-1],"baseline":base_values[-1],"available_steps":int(k)})
            horizons[str(h)]={"candidate":risk(values),"baseline":risk(base_values)}
        severe={name:{"mean":float(np.mean([row[name] for row in rows if row["horizon"]==5 and row["label"]=="unstable"])),"worst4":float(np.mean(sorted(row[name] for row in rows if row["horizon"]==5 and row["label"]=="unstable")[-4:]))} for name in ("baseline","candidate")};average={name:float(np.mean([horizons[str(h)][name]["risk"] for h in (1,2,5)])) for name in ("baseline","candidate")}
        after=hashes(baseline,fit,poles,null,guard);candidate_after=_candidate_hashes(candidate);timers["global_seconds"]=time.perf_counter()-started;require_time(deadline,"final")
        if before!=after or candidate_hashes!=candidate_after or not np.array_equal(_theta(model),theta) or any(not torch.equal(value,model.state_dict()[key]) for key,value in frozen.items() if key not in allowed):raise RuntimeError("source/candidate/frozen state changed")
        result={"loop":"0086","test_read":False,"h5_read":did_read_h5,"cached_train_read":True,"validation_materialized_by_reader":did_read_h5,"rebuild":{"identities":rebuild_info["identities"],"excluded":rebuild_info["excluded"]},"hashes_before":before,"hashes_after":after,"candidate_hashes":candidate_hashes,"solver":{key:value for key,value in solved.items() if key not in ("beta","Y","Y_raw","K","g","L","F","ell")},"conditional":conditional,"candidate_qa":qa,"autonomous_numerics":numerics,"autonomous_horizons":horizons,"autonomous_average_risk":average,"unstable_5s":severe,"per_event":rows,"timers":timers,"seconds":timers["global_seconds"],"allowed_state_keys_changed":changed}
        if not _finite(result):raise FloatingPointError("nonfinite final metric")
        (output/"evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False));return result
    except Exception as error:
        timers["global_seconds"]=time.perf_counter()-started
        if arrays:np.savez_compressed(output/"partial_solve_arrays.npz",**arrays)
        (output/"failure_evidence.json").write_text(json.dumps(jsonable({"type":type(error).__name__,"message":str(error),"current":current,"timers":timers,"hashes_before":before,"test_read":False,"h5_read":did_read_h5,"cached_train_read":True,"seconds":time.perf_counter()-started}),indent=2));raise

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--baseline",type=Path,default=BASELINE);parser.add_argument("--fit",type=Path,default=FIT);parser.add_argument("--poles",type=Path,default=POLES);parser.add_argument("--null",type=Path,default=NULL);parser.add_argument("--guard",type=Path,default=GUARD);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();run(args.baseline,args.fit,args.poles,args.null,args.guard,args.output)
if __name__=="__main__":main()
