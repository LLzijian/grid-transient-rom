"""One frozen 40ms estimate per machine, then read-only oracle diagnostics.

prepare -> estimate -> analyze. Wrapper logs belong in the separate _execution
directory. No optimizer, model export or autonomous ROM rollout is provided.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for _thread_variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_thread_variable]="1"
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"src")]
import analytic_p_backward_qualification_20261031 as qualified
import p_port_diagnostic_20261031 as filters
import p_reverse_precision_20261031 as utility

PRIVATE=ROOT/"private_data/p_form_discrimination_20261031"
PUBLIC=ROOT/"results/p_form_discrimination_20261031"
BINDINGS=PUBLIC/"source_bindings.json"
PARENT_COMMIT="9768acc47395a2126b07d095824f10cd72abe404"
BINDINGS_SHA="d840ea0d604ec42d42769e85b95d66ba607fb9857489d61b7046126e5f583546"
DESIGN_SHA="d881c2acbe67a5bedd0c203e6b46920f589bd933ba944eac98f2d80fd38f1f5e"
FORMAT="p_form_discrimination_20261031_v1"
MODELS=("A96_analytic","A16_analytic")
GROUPS=("stable8","unstable_actual","all297","stable5")
PRIMARY=("stable8","unstable_actual")
LENGTHS=(2,4,8)
sha,read,ident,safe,write_once=utility.sha,utility.read,utility.ident,utility.safe,utility.write_once


def source_closure():
    return {**qualified.source_closure(),Path(__file__).name:sha(Path(__file__))}


def shared_parameters(parameters):
    if set(parameters)!=set(MODELS) or any(utility.array_hash(np.asarray(parameters[MODELS[0]][key]))!=utility.array_hash(np.asarray(parameters[MODELS[1]][key])) for key in ("g","a","p_scale")):
        raise ValueError("fixed shared-constant premise fails; no expanded model set allowed")
    return parameters[MODELS[0]]


def digest(value):
    return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def finite_number(x):
    return isinstance(x,(float,int,np.floating,np.integer)) and not isinstance(x,(bool,np.bool_)) and np.isfinite(x)


def scope_steps(event,group):
    available=len(event["x"])-11
    primary=800 if event["label"]=="stable" else min(1500,available)
    if group=="all297":return primary
    if group=="stable8" and event["label"]=="stable":return 800
    if group=="stable5" and event["label"]=="stable":return 500
    if group=="unstable_actual" and event["label"]=="unstable":return primary
    return None


def window_design(truth,steps,scale,*,common80=False):
    """Composite Simpson/trapezoid on original .01 samples, disjoint windows."""
    truth=np.asarray(truth,dtype=np.float64)
    n=len(truth)-1;coverage=n//8*8 if common80 else n
    count=coverage//steps;used=count*steps
    meta={"original_future_steps":n,"requested_common80":common80,"windows":count,"coverage_steps":used,"coverage_seconds":used*.01,"tail_steps":n-used,"tail_seconds":(n-used)*.01}
    if truth.ndim!=2 or truth.shape[1]!=2 or not np.isfinite(truth).all():
        return {**meta,"available":False,"reason":"nonfinite_or_invalid_event","X_S":None,"X_T":None,"y":None}
    if count==0:return {**meta,"available":False,"reason":"no_complete_window","X_S":None,"X_T":None,"y":None}
    indices=np.arange(0,used,steps)[:,None]+np.arange(steps+1)[None,:]
    windows=truth[indices]
    s=np.ones(steps+1);s[1:-1:2]=4;s[2:-1:2]=2
    t=np.full(steps+1,2.);t[[0,-1]]=1.
    is_=.01/3*np.einsum('k,nkj->nj',s,windows)
    it=.01/2*np.einsum('k,nkj->nj',t,windows)
    denominator=steps*.01*scale
    return {**meta,"available":True,"reason":None,"X_S":is_/denominator,"X_T":it/denominator,"y":-(windows[:,-1,1]-windows[:,0,1])/denominator}


def collection(events,machine,group,steps,scale,*,common80=False):
    result=[]
    for event in events:
        n=scope_steps(event,group)
        if n is None:continue
        if n<=0 or len(event["x"])-11<n:
            row={"available":False,"reason":"incomplete_registered_interval","windows":0,"coverage_steps":0,"tail_steps":n,"X_S":None,"X_T":None,"y":None}
        else:row=window_design(event["x"][10:n+11][:,[10+machine,20+machine]],steps,scale,common80=common80)
        result.append({"event_id":event["id"],**row})
    return result


def weighted_target(rows,expected):
    if len(rows)!=expected or any(not r["available"] for r in rows):
        return None,{"status":"unknown","denominator":expected,"recorded_events":len(rows),"available_events":sum(r["available"] for r in rows),"unavailable_event_ids":[r["event_id"] for r in rows if not r["available"]]}
    weights=np.concatenate([np.full(r["windows"],1/(expected*r["windows"])) for r in rows])
    xs=np.concatenate([r["X_S"] for r in rows]);xt=np.concatenate([r["X_T"] for r in rows]);y=np.concatenate([r["y"] for r in rows])
    norms=np.linalg.norm(xs*np.sqrt(weights[:,None]),axis=0);scales=np.where(norms>0,norms,1.)
    a=xs*np.sqrt(weights[:,None])/scales;b=y*np.sqrt(weights)
    return {"A":a,"b":b,"X_S":xs,"X_T":xt,"weights":weights,"column_norms":norms,"scales":scales}, {"status":"available","denominator":expected,"recorded_events":len(rows),"available_events":len(rows),"row_count":len(y),"event_weight_sums":[1/expected]*expected}


def primary_svd(target):
    """The only new parameter-estimation call for this machine."""
    a,b=target["A"],target["b"]
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    threshold=max(a.shape)*np.finfo(np.float64).eps*s[0]
    keep=s>threshold;rank=int(keep.sum())
    gamma=vt[keep].T@((u[:,keep].T@b)/s[keep]);beta=gamma/target["scales"]
    fitted=a@gamma;r=b-fitted
    norm=np.linalg.norm(a.T@r)/max(1.,s[0]*np.linalg.norm(b))
    reconstruction=float(np.max(np.abs(a-(u*s)@vt)))
    delta=(target["X_S"]-target["X_T"])*np.sqrt(target["weights"][:,None])/target["scales"]
    spectral=float(np.sqrt(max(0.,np.linalg.eigvalsh(delta.T@delta)[-1])))
    condition=float(s[0]/s[-1]) if rank==2 else None;eta=float(spectral/s[-1]) if rank==2 else None
    return {"status":"estimated","beta40":beta,"gamma":gamma,"rank":rank,"rank_threshold":threshold,"singular_values":s,"column_weighted_L2_norms":target["column_norms"],"condition":condition,"eta":eta,
        "independent_identifiability":bool(rank==2 and condition<=1000 and eta<=.1),"signs_legal":bool(beta[0]>0 and beta[1]>=0),"normalized_normal_equation_residual":float(norm),"reconstruction_maxabs":reconstruction,
        "svd_numerical_checks_passed":bool(norm<=1e-12 and np.allclose(a,(u*s)@vt,atol=1e-12,rtol=1e-12) and np.allclose(fitted,u[:,keep]@(u[:,keep].T@b),atol=1e-12,rtol=1e-12)),
        "weighted_optimal_residual_L2":float(np.linalg.norm(r)),"target_sha256":digest({"A":a,"b":b}),"parameter_estimates":1}


def qr_check(target,primary):
    """One independent QR verification of the same objective, never used as beta."""
    from scipy.linalg import qr,solve_triangular
    a,b=target["A"],target["b"];q,r,piv=qr(a,mode="economic",pivoting=True)
    rank=primary["rank"];gamma=np.zeros(2)
    if rank:gamma[piv[:rank]]=solve_triangular(r[:rank,:rank],(q.T@b)[:rank])
    beta=gamma/target["scales"];svd_beta=np.asarray(primary["beta40"])
    fitted_ok=np.allclose(a@gamma,a@np.asarray(primary["gamma"]),atol=1e-12,rtol=1e-12)
    beta_ok=np.allclose(beta,svd_beta,atol=1e-12,rtol=1e-12) if rank==2 else None
    return {"status":"verified" if fitted_ok and (beta_ok or rank<2) else "numerical_unresolved","same_target_sha256":digest({"A":a,"b":b}),"beta_equal_1e12":bool(beta_ok) if beta_ok is not None else None,
        "fit_vector_equal_1e12":bool(fitted_ok),"beta_maxabs_difference":float(np.max(np.abs(beta-svd_beta))) if rank==2 else None,"parameter_candidate_created":False,"qr_checks":1}


def equal_event_stats(left,right,expected,check_error=0.,kind="response"):
    good=[(np.asarray(a),np.asarray(b)) for a,b in zip(left,right) if a is not None and b is not None and np.size(a)>0 and np.isfinite(a).all() and np.isfinite(b).all()]
    def unknown(reason,fields=None):
        return {"status":"unknown","reason":reason,"nonfinite_fields":fields or [],"denominator":expected,"available_events":len(good),"error_L":None,"error_R":None,"epsilon":None,"Q_S":None,"Q_T":None,"nu":None,"compatible":None,"discernible":None,"representation_difference_rms":None,"analytic_check_error_normalized":None}
    if len(left)!=expected or len(right)!=expected or len(good)!=expected:
        return unknown("incomplete_or_nonfinite_event_values")
    rms=lambda values:float(np.sqrt(np.mean([np.mean(v*v) for v in values])))
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        l=rms([a for a,b in good]);r=rms([b for a,b in good]);difference=rms([a-b for a,b in good])
        if kind=="response" and not finite_number(check_error):return unknown("nonfinite_analytic_check_error",["analytic_check_error_normalized"])
        eps=max(1e-10,difference,check_error) if kind=="response" else max(1e-10,difference)
        threshold=10*eps
    derived={"error_L":l,"error_R":r,"representation_difference_rms":difference,"epsilon":eps,"comparison_threshold":threshold}
    nonfinite=[name for name,value in derived.items() if not finite_number(value)]
    if nonfinite:return unknown("nonfinite_derived_statistics",nonfinite)
    return {"status":"computed","denominator":expected,"available_events":expected,"error_L":l,"error_R":r,"epsilon":eps,"Q_S":l,"Q_T":r,"nu":eps,
        "compatible":bool(max(l,r)<=threshold),"discernible":bool(min(l,r)>threshold),"representation_difference_rms":difference,"analytic_check_error_normalized":check_error if kind=="response" else None}


def weak_stats(rows,beta,expected):
    left=[];right=[]
    for row in rows:
        if beta is None or not row["available"]:left.append(None);right.append(None)
        else:left.append(row["y"]-row["X_S"]@beta);right.append(row["y"]-row["X_T"]@beta)
    return equal_event_stats(left,right,expected,kind="weak"),left,right


def weak_pair(old,new):
    if old["status"]!="computed" or new["status"]!="computed":return {"old_discernible":None,"new_compatible":None,"remaining":None}
    return {"old_discernible":old["discernible"],"new_compatible":new["compatible"],"remaining":bool(new["discernible"] and min(new["Q_S"],new["Q_T"])>=.2*max(old["Q_S"],old["Q_T"]))}


def relative_change(old,new,key):
    if not finite_number(old.get(key)) or not finite_number(new.get(key)):return {"ratio":None,"difference":None}
    return {"ratio":new[key]/old[key] if old[key]!=0 else None,"difference":new[key]-old[key]}


def branch_decision(solver_ok,identifiable,signs_legal,full40,groups):
    def main_good(group,mode):
        if mode=="C":return all(group["weak"][str(n)]["new_compatible"] is True and group["local"][str(n)]["beta40"]["compatible"] is True for n in LENGTHS) and group["persistent"]["beta40"]["compatible"] is True
        return all(group["weak"][str(n)]["remaining"] is True and group["local"][str(n)]["beta40"]["discernible"] is True for n in LENGTHS) and group["persistent"]["beta40"]["discernible"] is True
    old_resolved=any(all(groups[g]["weak"][str(n)]["old_discernible"] is True for n in LENGTHS) for g in PRIMARY)
    c=solver_ok and identifiable and signs_legal and old_resolved and all(main_good(groups[g],"C") for g in PRIMARY)
    full_resolved=full40["beta40"].get("discernible") is True
    supporting=[g for g in PRIMARY if main_good(groups[g],"F")]
    f=solver_ok and identifiable and full_resolved and bool(supporting)
    return {"status":"C_compatible" if c else "F_remaining_within_frozen_coordinates" if f else "mixed_or_unresolved","C_compatible":bool(c),"F_remaining_evidence":bool(f),"F_supporting_groups":supporting,"signs_legal_for_adoption":bool(signs_legal),
        "unconstrained_reference_only":not signs_legal,"full297_40ms_optimal_target_discernible":full_resolved,"main40_compatible_other_lengths_unresolved":full40["beta40"].get("compatible") is True and not c,
        "model_adopted":False,"physical_law_falsified":False,"majority_vote_used":False}


def coefficient_check(g,a,scale):
    from scipy.linalg import expm
    if not all(finite_number(v) for v in (g,a,scale)) or scale<=0:return {"status":"unknown","normalized_error":None,"reason":"nonfinite_coefficient"}
    error=0.
    try:
        with np.errstate(over="raise",invalid="raise",divide="raise"):
            for slope in (0.,.4):
                for h in (.01,-.01):
                    w0=-.3;p0=.2;omega=np.array([[w0],[w0+slope*h]])
                    path=filters.linear_filter(omega,[p0],[g],[a],h)
                    reference=(expm(h*np.array([[-a,-g,0],[0,0,1],[0,0,0]],dtype=np.float64))@[p0,w0,slope])[0]
                    back=filters.linear_filter(omega[::-1],path[-1],[g],[a],-h)[-1,0]
                    if not np.isfinite([path[-1,0],reference,back]).all():raise FloatingPointError("nonfinite scalar check")
                    np.testing.assert_allclose(path[-1,0],reference,atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(back,p0,atol=1e-12,rtol=1e-12)
                    error=max(error,abs(path[-1,0]-reference)/scale,abs(back-p0)/scale)
        return {"status":"PASS","normalized_error":error,"signs_legal":bool(g>0 and a>=0)}
    except Exception as exc:return {"status":"unknown","normalized_error":None,"reason":repr(exc)}


def local_response(truth,steps,beta,scale,*,common80=False):
    row=window_design(truth,steps,scale,common80=common80)
    if not row["available"] or beta is None:return None,None
    count=row["windows"];indices=np.arange(count)[:,None]*steps+np.arange(steps+1)
    windows=np.asarray(truth)[indices];omega=windows[:,:,0].T;p0=windows[:,0,1]
    try:
        with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
            l=filters.linear_filter(omega,p0,np.full(count,beta[0]),np.full(count,beta[1]),.01)[-1]
            r=filters.right_endpoint_filter(omega,p0,np.full(count,beta[0]),np.full(count,beta[1]),.01)[-1]
        if not np.isfinite(l).all() or not np.isfinite(r).all():return None,None
        return (l-windows[:,-1,1])/scale,(r-windows[:,-1,1])/scale
    except (ValueError,FloatingPointError):return None,None


def reject_unknown(started,complete):
    if Path(started).exists() and not Path(complete).exists():raise RuntimeError("unreconciled started stage; no automatic retry or re-estimation")


def verify_source_bindings(bindings,root=ROOT):
    def visit(value):
        if isinstance(value,dict):
            if "path" in value and "sha256" in value:
                path=(Path(root)/value["path"]).resolve();path.relative_to(Path(root).resolve())
                if sha(path)!=value["sha256"]:raise ValueError("frozen source binding hash differs: "+value["path"])
            else:
                for item in value.values():visit(item)
    if bindings.get("parent_commit")!=PARENT_COMMIT:raise ValueError("frozen parent commit differs")
    visit(bindings)


def checked_events(manifest):
    entry=manifest["bindings"]["data"]["fit_cache"];path=ROOT/entry["path"]
    if sha(path)!=entry["sha256"]:raise ValueError("frozen fit297 cache changed")
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data["metadata_json"]));events=[{**e,"x":data[f"event_{i:04d}_x"].copy()} for i,e in enumerate(meta)]
    labels={e["id"]:e["label"] for e in events}
    if len(events)!=297 or len(labels)!=297 or labels!=manifest["event_labels"] or list(labels.values()).count("stable")!=267 or list(labels.values()).count("unstable")!=30:raise ValueError("fit297 cohort changed")
    return sorted(events,key=lambda e:e["id"])


def checked_manifest():
    manifest=read(PRIVATE/"manifest.json");proof=read(PRIVATE/"prepare_complete.json")
    if proof["manifest_sha256"]!=sha(PRIVATE/"manifest.json") or manifest["source"]!=source_closure():raise ValueError("prepared source identity differs")
    for entry in (manifest["protocol"],manifest["design"],manifest["source_bindings"],manifest["authorization"]):
        if sha(ROOT/entry["path"])!=entry["sha256"]:raise ValueError("frozen protocol or source binding changed")
    return manifest


def verify_stage_release(stage,path,protocol,authorization):
    path=Path(path) if path is not None else PUBLIC/f"{stage}_execution_release.json"
    release=read(path)
    expected={"source_sha256":sha(Path(__file__)),"tests_sha256":sha(ROOT/"tests/test_p_form_discrimination_20261031.py"),"protocol_sha256":sha(protocol),"authorization_sha256":sha(authorization),"source_bindings_sha256":sha(BINDINGS),
        "maximum_main_parameter_estimates":10 if stage=="estimate" else 0,"same_objective_independent_qr_checks_upper_bound":10 if stage=="estimate" else 0,"optimizer_upper_bound":0,"new_autonomous_rom_rollouts_upper_bound":0,"model_writes_upper_bound":0}
    if release.get("scientific_computation_released") is not True or release.get("allowed_stage")!=stage or any(release.get(key)!=value for key,value in expected.items()):
        raise PermissionError("root stage execution release is missing or does not match the frozen source/tests/protocol/auth/budget")
    return ident(path)


def prepare(protocol,design,authorization,release=None):
    auth=read(authorization)
    if sha(BINDINGS)!=BINDINGS_SHA or sha(design)!=DESIGN_SHA or auth["protocol"]["sha256"]!=sha(protocol) or auth["scientist"]["sha256"]!=sha(design):raise ValueError("frozen design/protocol/source bindings differ")
    release_identity=verify_stage_release("prepare",release,protocol,authorization)
    if PRIVATE.exists():raise FileExistsError("tool output exists; reconcile before replay; use separate _execution logs")
    PRIVATE.mkdir(parents=True);started=time.perf_counter()
    write_once(PRIVATE/"prepare_started.json",{"source":source_closure(),"execution_release":release_identity,"optimizer_calls":0,"new_ROM_rollouts":0})
    try:
        bindings=read(BINDINGS);verify_source_bindings(bindings)
        proof=read(ROOT/bindings["parent_archive"]["path"])
        if proof.get("local_commit")!=PARENT_COMMIT or proof.get("remote_commit")!=PARENT_COMMIT or proof.get("ordinary_push_verified") is not True:raise ValueError("parent archive proof differs")
        qmanifest=read(ROOT/bindings["qualified_manifest"]["path"])
        qcomplete=read(ROOT/bindings["qualified_prepare_complete"]["path"])
        if qcomplete["status"]!="COMPLETED" or qcomplete["manifest_sha256"]!=bindings["qualified_manifest"]["sha256"]:raise ValueError("parent preparation receipt differs")
        for item in [qcomplete["baselines"],qcomplete["science_anchors"],*qcomplete["entries"].values()]:
            if sha(ROOT/item["path"])!=item["sha256"]:raise ValueError("parent preparation member changed")
        parents={};parameters={};model_proof={}
        for name in MODELS:
            candidate=qualified.checked_candidate_complete(qualified.PRIVATE/name)
            if candidate["implementation_qualified"] is not True or candidate["original_numerical_qualification"] is not True or candidate["causal_entry"]["passed"] is not True:raise ValueError("parent candidate qualification unresolved")
            model=qualified.load_overlay(ROOT/bindings["candidates"][name]["candidate_overlay"]["path"])
            values={"g":model.jpi_p.numpy().copy(),"a":(model.rpp*model.storage_base).numpy().copy(),"p_scale":model.p_scale.numpy().copy()}
            parameters[name]=values
            parents[name]={"qualification":bindings["candidates"][name]["qualification"],"complete":bindings["candidates"][name]["complete"],"prediction_files":[{"path":item["path"],"sha256":item["sha256"]} for item in read(ROOT/bindings["candidates"][name]["complete"]["path"])["artifacts"] if Path(item["path"]).name.startswith("predictions_")]}
            for entry in parents[name]["prediction_files"]:entry["path"]=str(ROOT/entry["path"])
            if not parents[name]["prediction_files"]:raise ValueError("parent full prediction members missing")
            model_proof[name]={"all_original_qualification_passed":True,"science_entry_eligible":candidate["science_entry"]["science_entry_eligible"],"parameter_hashes":{key:utility.array_hash(value) for key,value in values.items()},"candidate_state_identity":qualified.state_identity(model)}
        values=shared_parameters(parameters)
        import torch,scipy
        manifest={"format":FORMAT,"protocol":ident(protocol),"design":ident(design),"authorization":ident(authorization),"execution_release":release_identity,"source_bindings":ident(BINDINGS),"bindings":bindings,"source":source_closure(),"event_labels":qmanifest["data"]["event_labels"],"fit_ids":qmanifest["data"]["fit_ids"],"parents":parents,"parent_qualification":model_proof,"parameters":{"beta0":np.stack((values["g"],values["a"]),axis=1),"p_scale":values["p_scale"]},"runtime":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"torch":torch.__version__,"dtype":"float64","device":"cpu","threads":1},"maximum_main_SVD_estimates":10,"maximum_same_target_QR_checks":10,"optimizer_calls":0,"new_ROM_rollouts":0,"new_fom":0,"val_read":False,"test_read":False}
        write_once(PRIVATE/"manifest.json",manifest)
        events=checked_events(manifest)
        coverage=[]
        for event in events:
            groups={}
            for group in GROUPS:
                n=scope_steps(event,group)
                if n is None:continue
                groups[group]={"future_steps":n,"common80_steps":n//8*8,"registered_interval_available":len(event["x"])-11>=n,"windows":{str(s):{"full_windows":n//s,"full_tail_steps":n%s,"common80_windows":(n//8*8)//s,"common80_tail_steps":n-n//8*8} for s in LENGTHS}}
            coverage.append({"event_id":event["id"],"label":event["label"],"groups":groups})
        write_once(PRIVATE/"coverage.json",coverage)
        write_once(PRIVATE/"prepare_complete.json",{"status":"COMPLETED","manifest_sha256":sha(PRIVATE/"manifest.json"),"coverage":ident(PRIVATE/"coverage.json"),"wall_seconds":time.perf_counter()-started,"parameter_estimates":0,"new_ROM_rollouts":0})
    except BaseException as exc:
        write_once(PRIVATE/"prepare_failure.json",{"error":repr(exc),"traceback":traceback.format_exc(),"wall_seconds":time.perf_counter()-started,"parameter_estimates":0})
        raise


def verify_estimate(machine):
    folder=PRIVATE/"estimates"/f"machine_{machine:02d}"
    complete=read(folder/"complete.json")
    if complete["status"]!="COMPLETED" or complete["manifest_sha256"]!=sha(PRIVATE/"manifest.json") or complete["parameters"]["sha256"]!=sha(folder/"parameters.json") or complete["result"]["sha256"]!=sha(folder/"result.json"):raise ValueError("frozen per-machine estimate receipt changed")
    primary,result=read(folder/"parameters.json"),read(folder/"result.json")
    if primary["machine_column"]!=machine or result["machine_column"]!=machine or primary["manifest_sha256"]!=complete["manifest_sha256"]:raise ValueError("per-machine estimate identity differs")
    return primary,result,complete


def estimate(release=None):
    manifest=checked_manifest();started_path=PRIVATE/"estimate_started.json";complete_path=PRIVATE/"estimate_complete.json"
    release_identity=verify_stage_release("estimate",release,ROOT/manifest["protocol"]["path"],ROOT/manifest["authorization"]["path"])
    if complete_path.exists():
        for i in range(10):verify_estimate(i)
        return read(complete_path)
    reject_unknown(started_path,complete_path)
    write_once(started_path,{"manifest_sha256":sha(PRIVATE/"manifest.json"),"execution_release":release_identity,"maximum_main_estimates":10,"automatic_repair":False})
    started=time.perf_counter()
    try:
        events=checked_events(manifest);scales=np.asarray(manifest["parameters"]["p_scale"]);receipts=[]
        for machine in range(10):
            folder=PRIVATE/"estimates"/f"machine_{machine:02d}"
            if (folder/"complete.json").exists():
                receipts.append(verify_estimate(machine)[2]);continue
            reject_unknown(folder/"started.json",folder/"complete.json")
            write_once(folder/"started.json",{"machine_column":machine,"manifest_sha256":sha(PRIVATE/"manifest.json"),"max_estimates":1})
            rows=collection(events,machine,"all297",4,scales[machine]);target,coverage=weighted_target(rows,297)
            stage=time.perf_counter()
            if target is None:
                primary={"status":"unknown","beta40":None,"parameter_estimates":0,"coverage":coverage,"reason":"incomplete_full297_target_no_reweighting"}
            else:
                try:primary=primary_svd(target)
                except (np.linalg.LinAlgError,FloatingPointError) as exc:primary={"status":"numerical_unresolved","beta40":None,"parameter_estimates":1,"reason":repr(exc)}
            # Persist the new parameter set BEFORE QR or any follow-on analysis.
            write_once(folder/"parameters.json",{"machine_column":machine,"manifest_sha256":sha(PRIVATE/"manifest.json"),**primary})
            parameter_identity=ident(folder/"parameters.json")
            if target is not None and primary["beta40"] is not None:
                try:check=qr_check(target,primary)
                except (np.linalg.LinAlgError,FloatingPointError) as exc:check={"status":"numerical_unresolved","reason":repr(exc),"qr_checks":1}
            else:check={"status":"unavailable","qr_checks":0}
            result={"machine_column":machine,"parameters":parameter_identity,"QR":check,"coverage":coverage,"numeric_optimality_passed":primary.get("svd_numerical_checks_passed") is True and check["status"]=="verified","wall_seconds":time.perf_counter()-stage,"main_estimate_attempts":primary["parameter_estimates"],"same_target_QR_checks":check["qr_checks"]}
            write_once(folder/"result.json",result)
            receipt={"status":"COMPLETED","manifest_sha256":sha(PRIVATE/"manifest.json"),"parameters":parameter_identity,"result":ident(folder/"result.json")}
            write_once(folder/"complete.json",receipt);receipts.append(receipt)
        result={"status":"COMPLETED","machine_receipts":receipts,"main_SVD_estimates":sum(verify_estimate(i)[1]["main_estimate_attempts"] for i in range(10)),"same_target_QR_checks":sum(verify_estimate(i)[1]["same_target_QR_checks"] for i in range(10)),"wall_seconds":time.perf_counter()-started,"optimizer_calls":0,"new_ROM_rollouts":0}
        write_once(complete_path,result)
        return result
    except BaseException as exc:
        write_once(PRIVATE/"estimate_failure.json",{"error":repr(exc),"traceback":traceback.format_exc(),"wall_seconds":time.perf_counter()-started,"previous_parameters_must_be_reused":True,"automatic_reestimate_allowed":False})
        raise


def persistent_oracles(events,beta0,beta40,scales,checks):
    """Batch valid scalar columns; invalid columns never remove another event."""
    columns=[(kind,machine) for kind in ("beta0","beta40") for machine in range(10) if checks[kind][machine]["status"]=="PASS"]
    parameters={"beta0":beta0,"beta40":beta40};data={};paths={};counts={"oracle_scalar_trajectories_L":0,"oracle_scalar_trajectories_R":0,"oracle_future_steps_L":0,"oracle_future_steps_R":0}
    for event in events:
        n=scope_steps(event,"all297");truth=event["x"][10:n+11]
        usable=[(kind,m) for kind,m in columns if len(truth)==n+1 and np.isfinite(truth[:,10+m]).all() and np.isfinite(truth[0,20+m])]
        outputs={}
        if usable:
            omega=np.stack([truth[:,10+m] for kind,m in usable],axis=1)
            initial=np.array([truth[0,20+m] for kind,m in usable]);g=np.array([parameters[kind][m][0] for kind,m in usable]);a=np.array([parameters[kind][m][1] for kind,m in usable])
            with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
                left=filters.linear_filter(omega,initial,g,a,.01);right=filters.right_endpoint_filter(omega,initial,g,a,.01)
            for index,(kind,machine) in enumerate(usable):
                if np.isfinite(left[:,index]).all() and np.isfinite(right[:,index]).all():outputs[(kind,machine)]=(left[:,index],right[:,index])
            paths[event["id"]]={"columns":usable,"L":left,"R":right}
            for representation in ("L","R"):
                counts[f"oracle_scalar_trajectories_{representation}"]+=len(usable);counts[f"oracle_future_steps_{representation}"]+=len(usable)*n
        for kind in ("beta0","beta40"):
            for machine in range(10):
                for group in GROUPS:
                    count=scope_steps(event,group)
                    if count is None:continue
                    key=(kind,machine,group);data.setdefault(key,[])
                    if (kind,machine) not in outputs or not np.isfinite(truth[:count+1,20+machine]).all():data[key].append((event["id"],None,None));continue
                    l,r=outputs[(kind,machine)]
                    data[key].append((event["id"],(l[1:count+1]-truth[1:count+1,20+machine])/scales[machine],(r[1:count+1]-truth[1:count+1,20+machine])/scales[machine]))
    return data,paths,counts


def compact_stats(cell):
    return {key:value for key,value in cell.items() if key not in ("Q_S","Q_T","nu")}


def analyze_tables(events,beta0,beta40,scales,checks,persistent):
    machines=[];private_rows={};arrays={};local_counts={"oracle_local_scalar_trajectories_L":0,"oracle_local_scalar_trajectories_R":0};local_cache={}
    for machine in range(10):
        weak={};local={};sustained={};decision_groups={}
        for group in GROUPS:
            denominator=297 if group=="all297" else 30 if group=="unstable_actual" else 267
            weak[group]={};local[group]={};sustained[group]={}
            for kind in ("beta0","beta40"):
                items=persistent[(kind,machine,group)];error=checks[kind][machine].get("normalized_error")
                sustained[group][kind]=equal_event_stats([x[1] for x in items],[x[2] for x in items],denominator,check_error=error if error is not None else 0.)
                for rep,index in (("L",1),("R",2)):
                    good=[value[index] for value in items if value[index] is not None]
                    arrays[f"m{machine}_{group}_{kind}_persistent_{rep}"]=np.concatenate(good) if good else np.empty(0)
            for common,name in ((False,"full_each_L"),(True,"common80")):
                weak[group][name]={};local[group][name]={}
                for steps in LENGTHS:
                    table=collection(events,machine,group,steps,scales[machine],common80=common)
                    byid={r["event_id"]:r for r in table};weak[group][name][str(steps)]={};local[group][name][str(steps)]={}
                    for kind,parameters in (("beta0",beta0),("beta40",beta40)):
                        beta=parameters[machine]
                        ws,qs,qt=weak_stats(table,beta,denominator)
                        weak[group][name][str(steps)][kind]=ws
                        ls=[];rs=[];ids=[];counts=[]
                        for event in events:
                            n=scope_steps(event,group)
                            if n is None:continue
                            ids.append(event["id"]);counts.append(byid[event["id"]]["windows"])
                            coverage=byid[event["id"]]
                            if checks[kind][machine]["status"]!="PASS" or not coverage["available"]:l=r=None
                            else:
                                cache_key=(kind,machine,event["id"],steps,coverage["coverage_steps"])
                                if cache_key not in local_cache:
                                    local_cache[cache_key]=local_response(event["x"][10:n+11][:,[10+machine,20+machine]],steps,beta,scales[machine],common80=common)
                                    local_counts["oracle_local_scalar_trajectories_L"]+=coverage["windows"];local_counts["oracle_local_scalar_trajectories_R"]+=coverage["windows"]
                                l,r=local_cache[cache_key]
                            ls.append(l);rs.append(r)
                        epsilon=checks[kind][machine].get("normalized_error")
                        local[group][name][str(steps)][kind]=equal_event_stats(ls,rs,denominator,check_error=epsilon if epsilon is not None else 0.)
                        tag=f"m{machine}_{group}_{name}_{steps}_{kind}"
                        for suffix,values in (("qS",qs),("qT",qt),("localL",ls),("localR",rs)):
                            good=[v for v in values if v is not None];arrays[tag+"_"+suffix]=np.concatenate(good) if good else np.empty(0)
                        private_rows[tag]={"event_ids":ids,"window_counts":counts,"available_qS":[v is not None for v in qs],"available_local":[v is not None for v in ls],"coverage":[{k:v for k,v in row.items() if k not in ("X_S","X_T","y")} for row in table]}
                    pair=weak_pair(weak[group][name][str(steps)]["beta0"],weak[group][name][str(steps)]["beta40"])
                    weak[group][name][str(steps)]["comparison"]=pair
            decision_groups[group]={"weak":{str(n):weak[group]["common80"][str(n)]["comparison"] for n in LENGTHS},"local":{str(n):local[group]["common80"][str(n)] for n in LENGTHS},"persistent":sustained[group]}
        machines.append({"machine_column":machine,"weak":weak,"local_oracle":local,"persistent_oracle":sustained,"decision_groups":decision_groups,
            "persistent_change":{group:{key:relative_change(sustained[group]["beta0"],sustained[group]["beta40"],key) for key in ("error_L","error_R")} for group in GROUPS}})
    return machines,private_rows,arrays,local_counts


def m2_bridge(manifest,events,persistent_paths):
    lookup={e["id"]:e for e in events};beta0=np.asarray(manifest["parameters"]["beta0"]);scale=np.asarray(manifest["parameters"]["p_scale"])
    private={};aggregate={};counts={"candidate_cache_passes":0,"model_omega_scalar_filter_trajectories":0}
    for candidate in MODELS:
        rows=[];seen=[]
        for ids,h,f,c in qualified.prediction_blocks(manifest["parents"][candidate]["prediction_files"]):
            for eid,history,ff,cc in zip(ids,h,f,c):
                event=lookup[eid];seen.append(eid);n=scope_steps(event,"all297")
                if not np.array_equal(history,event["x"][:11]):raise ValueError("M2 source history identity changed")
                groups=[("stable5",500),("stable8",800)] if event["label"]=="stable" else [("unstable_actual15",n)]
                ff,cc=ff[:2*n+1],cc[:n+1]
                try:
                    truth=event["x"][10:n+11]
                    source=persistent_paths.get(eid,{})
                    columns=source.get("columns",[])
                    indices=[columns.index(("beta0",machine)) for machine in range(10)]
                    tl,tr=source["L"][:,indices],source["R"][:,indices]
                    fm=filters.linear_filter(ff[:,10:20],history[-1,20:30],beta0[:,0],beta0[:,1],.005)[::2]
                    cm=filters.linear_filter(cc[:,10:20],history[-1,20:30],beta0[:,0],beta0[:,1],.01)
                    dm=filters.linear_filter(ff[::2,10:20],history[-1,20:30],beta0[:,0],beta0[:,1],.01)
                    counts["model_omega_scalar_filter_trajectories"]+=30
                    for scope,steps in groups:
                        reference=truth[1:steps+1,20:30]
                        fine=filters.squared_decomposition(ff[::2][1:steps+1,20:30],fm[1:steps+1],tl[1:steps+1],reference,scale)
                        coarse=filters.squared_decomposition(cc[1:steps+1,20:30],cm[1:steps+1],tl[1:steps+1],reference,scale)
                        r=filters.normalized_rms(ff[::2][1:steps+1,20:30]-fm[1:steps+1],scale)
                        s=max(filters.normalized_rms(ff[::2][1:steps+1,20:30]-cc[1:steps+1,20:30],scale),filters.normalized_rms(fm[1:steps+1]-cm[1:steps+1],scale),filters.normalized_rms(fm[1:steps+1]-dm[1:steps+1],scale))
                        tau=max(1e-10,r,s,filters.normalized_rms(tl[1:steps+1]-tr[1:steps+1],scale))
                        fixed=filters.normalized_rms(tl[1:steps+1]-reference,scale)
                        rows.append({"event_id":eid,"label":event["label"],"scope":scope,"qualified":True,"fine":fine,"coarse":coarse,"r_e":r,"s_e":s,"tau_e":tau,"self_closure_passed":filters.self_closure_gate(r,s),"fixed_error_normalized_rms":fixed,"fixed_residual_resolved":fixed>10*tau})
                except (ValueError,FloatingPointError,AssertionError) as exc:
                    for scope,steps in groups:rows.append({"event_id":eid,"label":event["label"],"scope":scope,"qualified":False,"fine":None,"coarse":None,"reason":repr(exc)})
        if len(seen)!=297 or len(set(seen))!=297 or set(seen)!=set(lookup):raise ValueError("M2 did not retain the original 297 events")
        private[candidate]=rows;aggregate[candidate]=filters.summarize_decompositions(rows);counts["candidate_cache_passes"]+=1
    return aggregate,private,counts


def analyze(release=None):
    manifest=checked_manifest();started_path=PRIVATE/"analyze_started.json";complete_path=PRIVATE/"analyze_complete.json"
    release_identity=verify_stage_release("analyze",release,ROOT/manifest["protocol"]["path"],ROOT/manifest["authorization"]["path"])
    if complete_path.exists():
        receipt=read(complete_path)
        for entry in receipt["artifacts"]:
            if sha(ROOT/entry["path"])!=entry["sha256"]:raise ValueError("completed analysis artifact changed")
        return read(PRIVATE/"diagnostic.json")
    reject_unknown(started_path,complete_path)
    if not (PRIVATE/"estimate_complete.json").exists():raise RuntimeError("all frozen per-machine estimate receipts required")
    primary=[];results=[];receipts=[]
    for machine in range(10):
        p,r,c=verify_estimate(machine);primary.append(p);results.append(r);receipts.append(c)
    write_once(started_path,{"manifest_sha256":sha(PRIVATE/"manifest.json"),"execution_release":release_identity,"estimate_receipts_sha256":digest(receipts),"new_parameter_estimates":0,"optimizer_calls":0,"new_ROM_rollouts":0})
    started=time.perf_counter()
    try:
        stage=time.perf_counter();events=checked_events(manifest);read_seconds=time.perf_counter()-stage
        beta0=np.asarray(manifest["parameters"]["beta0"]);beta40=[np.asarray(p["beta40"]) if p["beta40"] is not None else None for p in primary];scales=np.asarray(manifest["parameters"]["p_scale"])
        fixed_artificial=filters.artificial_qualification();checks={"beta0":[],"beta40":[]};check_cache={}
        for kind,parameters in (("beta0",beta0),("beta40",beta40)):
            for machine,beta in enumerate(parameters):
                if beta is None:cell={"status":"unknown","normalized_error":None,"reason":"no_frozen_coefficient"}
                else:
                    key=tuple(beta)+(scales[machine],)
                    if key not in check_cache:check_cache[key]=coefficient_check(beta[0],beta[1],scales[machine])
                    cell=check_cache[key]
                checks[kind].append(cell)
        stage=time.perf_counter()
        persistent,oracle_paths,persistent_counts=persistent_oracles(events,beta0,beta40,scales,checks)
        machines,private_rows,arrays,local_counts=analyze_tables(events,beta0,beta40,scales,checks,persistent)
        filter_seconds=time.perf_counter()-stage
        for machine,row in enumerate(machines):
            full40={kind:row["weak"]["all297"]["full_each_L"]["4"][kind] for kind in ("beta0","beta40")}
            decision=branch_decision(results[machine]["numeric_optimality_passed"],primary[machine].get("independent_identifiability",False),primary[machine].get("signs_legal",False),full40,row["decision_groups"])
            row.update(parameters=ident(PRIVATE/"estimates"/f"machine_{machine:02d}"/"parameters.json"),primary_numerics={key:value for key,value in primary[machine].items() if key not in ("beta40","gamma","coverage")},QR=results[machine]["QR"],numeric_optimality_passed=results[machine]["numeric_optimality_passed"],decision=decision)
        stage=time.perf_counter();m2,m2_private,m2_counts=m2_bridge(manifest,events,oracle_paths);bridge_seconds=time.perf_counter()-stage
        write_once(PRIVATE/"per_event.json",{"windows":private_rows,"M2":m2_private,"persistent_event_order":{f"{kind}_{m}_{group}":[row[0] for row in values] for (kind,m,group),values in persistent.items()}})
        with (PRIVATE/"residuals_and_endpoints.npz").open("xb") as f:np.savez_compressed(f,**arrays)
        path_arrays={};path_members={}
        for index,(eid,paths) in enumerate(oracle_paths.items()):
            path_arrays[f"event_{index:04d}_L"]=paths["L"];path_arrays[f"event_{index:04d}_R"]=paths["R"];path_members[str(index)]={"event_id":eid,"columns":paths["columns"]}
        with (PRIVATE/"oracle_p_paths.npz").open("xb") as f:np.savez_compressed(f,**path_arrays)
        write_once(PRIVATE/"oracle_path_members.json",path_members)
        estimates=read(PRIVATE/"estimate_complete.json")
        result={"format":FORMAT,"status":"COMPLETED","manifest_sha256":sha(PRIVATE/"manifest.json"),"machines":machines,"coefficient_numerical_checks":checks,"fixed_artificial_qualification":fixed_artificial,"M2_bridge":m2,"oracle_usage":"future true omega and local true p resets are offline oracle diagnostics, never deployment inputs","old_M3_unlocked":False,"model_adopted":False,
            "counts":{"main_SVD_estimates":estimates["main_SVD_estimates"],"same_target_QR_checks":estimates["same_target_QR_checks"],"scalar_parameter_check_sets":len(check_cache),**persistent_counts,**local_counts,**m2_counts,"new_autonomous_ROM_rollouts":0,"optimizer_calls":0,"model_writes":0,"new_fom":0},
            "p_coordinate_interpretation":"frozen numeric p coordinate; upstream tm torque label and physical unit/energy limitations retained; oracle is not deployment",
            "five_axes":{"evidence":"per-machine source/coverage/numerical checks retained","optimizer_health":"not_applicable; zero optimizer calls","optimization_sufficiency":"same 40ms linear objective only where SVD/QR gates pass; source training unknown","mechanism_testability":"cross-scale weak and oracle response conditions; no majority vote","effect_and_confirmation":"consumed train and frozen .01s sampling representations; no new-data confirmation"},
            "cost":{"read_seconds":read_seconds,"estimation_stage_wall_seconds":estimates["wall_seconds"],"filter_and_fixed_weak_analysis_seconds":filter_seconds,"cached_M2_bridge_seconds":bridge_seconds,"analysis_wall_before_summary_publication":time.perf_counter()-started,"cpu_process_seconds":None,"LLM_cost":None,"project_total_cost":None}}
        write_once(PRIVATE/"diagnostic.json",result)
        public={key:value for key,value in result.items() if key!="machines"}
        public["machines"]=[]
        for row in machines:
            cell={key:value for key,value in row.items() if key not in ("parameters","primary_numerics")}
            cell["parameters_sha256"]=row["parameters"]["sha256"]
            cell["primary_numerics"]={key:value for key,value in row["primary_numerics"].items() if key not in ("unavailable_event_ids","reason")}
            public["machines"].append(cell)
        # No raw beta or event IDs are present in the public machine tables.
        write_once(PUBLIC/"public_diagnostic.json",public)
        artifacts=[ident(PRIVATE/name) for name in ("diagnostic.json","per_event.json","residuals_and_endpoints.npz","oracle_p_paths.npz","oracle_path_members.json")]+[ident(PUBLIC/"public_diagnostic.json")]
        write_once(complete_path,{"status":"COMPLETED","artifacts":artifacts,"actual_analysis_total_wall_seconds":time.perf_counter()-started,"new_parameter_estimates":0,"optimizer_calls":0,"new_ROM_rollouts":0})
        return result
    except BaseException as exc:
        write_once(PRIVATE/"analyze_failure.json",{"error":repr(exc),"traceback":traceback.format_exc(),"wall_seconds":time.perf_counter()-started,"new_parameter_estimates":0,"frozen_beta_must_not_be_reestimated":True})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","estimate","analyze"))
    parser.add_argument("--protocol",type=Path,default=ROOT/"reports/p_form_discrimination_20261031_protocol.md")
    parser.add_argument("--design",type=Path,default=ROOT/"reports/p_form_discrimination_20261031_scientist.md")
    parser.add_argument("--authorization",type=Path,default=PUBLIC/"authorization.json")
    parser.add_argument("--release",type=Path,help="root's separate stage execution release; defaults to PUBLIC/<stage>_execution_release.json")
    args=parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    if args.command=="prepare":prepare(args.protocol,args.design,args.authorization,args.release)
    elif args.command=="estimate":estimate(args.release)
    else:analyze(args.release)


if __name__=="__main__":
    main()
