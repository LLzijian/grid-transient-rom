"""Bounded weak regression for one shared coupling and event-specific memory."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import scipy

from gftd_initialization_20261107 import ROOT, digest, functional, write_json
from gftd_observability_20261107 import observations, rk4

OUTPUT = ROOT/"results/gftd_coupling_memory_20261107"
PROTOCOL_SHA256 = "d0f17289ec3382e751c591ac48e72131d577acac071fbbd709dabf46041f05db"
STEPS = (.0025,.00125)
SIGMAS = (0.,1e-6,1e-4)
VARIANTS = ("complete","omit_p","omit_r","single_test")
THRESHOLDS = (1e-8,1e-10,1e-12)


def test_functions(t):
    tau=np.asarray(t,dtype=np.float64)/.1
    x=2*tau-1
    polynomials=(np.ones_like(x),x,(3*x*x-1)/2,(5*x**3-3*x)/2)
    derivatives=(np.zeros_like(x),np.ones_like(x),3*x,(15*x*x-3)/2)
    v=np.stack([tau*(1-tau)*p for p in polynomials])
    dv=np.stack([((1-2*tau)*p+2*tau*(1-tau)*dp)/.1
                 for p,dp in zip(polynomials,derivatives)])
    return v,dv


def boole_weights(t):
    t=np.asarray(t,dtype=np.float64)
    if t.ndim!=1 or len(t)<5 or (len(t)-1)%4 or not np.isfinite(t).all():
        raise ValueError("Boole requires a finite uniform grid with 4k intervals")
    steps=np.diff(t)
    if steps[0]<=0 or not np.allclose(steps,steps[0],atol=1e-14,rtol=1e-12):
        raise ValueError("Boole grid must be strictly increasing and uniform")
    weights=np.zeros(len(t))
    for start in range(0,len(t)-1,4):
        weights[start:start+5]+=np.array([7.,32.,12.,32.,7.])*2*steps[0]/45
    return weights


def validate_observation(packet):
    allowed={"theta_history","omega_history","history_times","p_end"}
    if set(packet)!=allowed:
        raise ValueError("only theta/omega history, history_times and endpoint p are allowed")
    result={key:np.asarray(value,dtype=np.float64).copy() for key,value in packet.items()}
    if result["theta_history"].shape!=(11,3) or result["omega_history"].shape!=(11,3) or result["p_end"].shape!=(3,):
        raise ValueError("eleven three-node observations required")
    if result["history_times"].shape!=(11,) or not np.allclose(result["history_times"],np.linspace(0.,.1,11),atol=1e-12,rtol=0):
        raise ValueError("history must cover 0 through .1 seconds")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("observations must be finite")
    return result


def linear_rk4(rhs,t,state,h):
    a=rhs(t,state)
    b=rhs(t+h/2,state+h*a/2)
    c=rhs(t+h/2,state+h*b/2)
    d=rhs(t+h,state+h*c)
    result=state+h*(a+2*b+2*c+d)/6
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite port reconstruction")
    return result


def reconstruct_ports(t,omega,p_end,f):
    """Only supplied speed forcing and endpoint p enter these linear equations."""
    t=np.asarray(t,dtype=np.float64)
    omega=np.asarray(omega,dtype=np.float64)
    def forcing(time):
        return np.array([np.interp(time,t,omega[:,node]) for node in range(3)])
    p=np.zeros_like(omega)
    rf=np.zeros_like(omega)
    p[-1]=p_end
    alpha=f.rate_p*f.storage_p
    # R_m Lambda_m = diag(Lambda_m / tau_mem_sync), giving lambda=.3 here.
    decay=f.lambda_mem/f.tau_mem_sync
    for index in range(len(t)-1,0,-1):
        p[index-1]=linear_rk4(lambda time,value:-alpha*value-f.gain_p*forcing(time),
                              t[index],p[index],t[index-1]-t[index])
    for index in range(len(t)-1):
        rf[index+1]=linear_rk4(lambda time,value:f.j_omega_mem*forcing(time)-decay*value,
                               t[index],rf[index],t[index+1]-t[index])
    return p,rf


def weak_terms(t,theta,omega,p,memory,f):
    weights=boole_weights(t)
    v,dv=test_functions(t)
    phase=np.asarray(theta)@f.incidence
    force=((np.sin(phase)-np.sin(f.eta_edge))*f.k_edge)@f.incidence.T
    weighted=v*weights
    return dict(A=(weighted@force).reshape(-1)/.1,
                inertial=((dv*weights)@(omega*f.mass)).reshape(-1)/.1,
                damping=(weighted@(omega*f.damping)).reshape(-1)/.1,
                p=(weighted@(p*f.gain_p*f.storage_p)).reshape(-1)/.1,
                r=(weighted@memory).reshape(-1)/.1)


def build_system(packet,step):
    packet=validate_observation(packet)
    if step not in STEPS:
        raise ValueError("unregistered quadrature step")
    f=functional(0.)
    t=np.linspace(0.,.1,int(round(.1/step))+1)
    theta=np.column_stack([np.interp(t,packet["history_times"],packet["theta_history"][:,i]) for i in range(3)])
    omega=np.column_stack([np.interp(t,packet["history_times"],packet["omega_history"][:,i]) for i in range(3)])
    p,rf=reconstruct_ports(t,omega,packet["p_end"],f)
    terms=weak_terms(t,theta,omega,p,rf,f)
    v,_=test_functions(t)
    exponential=np.exp(-t[:,None]*(f.lambda_mem/f.tau_mem_sync)[None,:])
    integrated=(v*boole_weights(t))@exponential
    Z=np.vstack([np.diag(row) for row in integrated]) # .1 parameter scale cancels .1 force scale.
    b=terms["inertial"]-terms["damping"]+terms["p"]-terms["r"]
    return dict(A=terms["A"],Z=Z,b=b,p_term=terms["p"],rf_term=terms["r"],
                reconstructed_p=p,reconstructed_rf=rf,t=t)


def combine(systems,variant):
    if variant not in VARIANTS:
        raise ValueError("unknown fixed model variant")
    matrices=[]
    for system in systems:
        rows=slice(0,3) if variant=="single_test" else slice(None)
        b=system["b"].copy()
        if variant=="omit_p":
            b-=system["p_term"]
        elif variant=="omit_r":
            b+=system["rf_term"]
        matrices.append((system["A"][rows],system["Z"][rows],b[rows]))
    A=np.concatenate([item[0] for item in matrices])
    b=np.concatenate([item[2] for item in matrices])
    Z=np.zeros((len(b),0 if variant=="omit_r" else 3*len(matrices)))
    start=0
    if variant!="omit_r":
        for index,(a,z,_) in enumerate(matrices):
            Z[start:start+len(a),3*index:3*index+3]=z
            start+=len(a)
    return A,Z,b


def solve_linear(A,Z,b):
    A,Z,b=(np.asarray(value,dtype=np.float64) for value in (A,Z,b))
    if A.ndim!=1 or b.shape!=A.shape or Z.ndim!=2 or Z.shape[0]!=len(A) or not all(np.isfinite(v).all() for v in (A,Z,b)):
        raise ValueError("invalid finite weak linear system")
    U,s,Vt=np.linalg.svd(Z,full_matrices=False)
    full_s=np.linalg.svd(np.column_stack((A,Z)),compute_uv=False)
    norm_A=float(np.linalg.norm(A))
    diagnostics=[]
    for threshold in THRESHOLDS:
        rank=int(np.sum(s>s[0]*threshold)) if s.size and s[0]>0 else 0
        basis=U[:,:rank]
        ap=A-basis@(basis.T@A)
        bp=b-basis@(basis.T@b)
        ratio=float(np.linalg.norm(ap)/norm_A) if norm_A else 0.
        diagnostics.append(dict(threshold=threshold,memory_rank=rank,
            full_rank=int(np.sum(full_s>full_s[0]*threshold)) if full_s.size and full_s[0]>0 else 0,
            residual_column_ratio=ratio,identifiable=bool(norm_A>0 and ratio>threshold)))
    main=diagnostics[1]
    rank=main["memory_rank"]
    basis=U[:,:rank]
    ap=A-basis@(basis.T@A)
    bp=b-basis@(basis.T@b)
    quadratic=float(ap@ap)
    def rho(c):
        return Vt[:rank].T@((basis.T@(b-A*c))/s[:rank]) if rank else np.zeros(Z.shape[1])
    formal_c=float(ap@bp/quadratic) if quadratic>0 else None
    c_hat=formal_c if main["identifiable"] else None
    profile=[]
    for c in np.linspace(0.,2.,9):
        profile.append(dict(c=float(c),loss=float(np.linalg.norm(bp-ap*c)**2),rho=.1*rho(c)))
    residual=None if c_hat is None else A*c_hat+Z@rho(c_hat)-b
    return dict(c_hat=c_hat,rho_hat=None if c_hat is None else .1*rho(c_hat),
        column_norms=np.linalg.norm(np.column_stack((A,Z)),axis=0),singular_values=full_s,memory_singular_values=s,
        threshold_diagnostics=diagnostics,rank_threshold_sensitive=len({(d["memory_rank"],d["full_rank"],d["identifiable"]) for d in diagnostics})>1,
        a_perp=ap,b_perp=bp,memory_projection_A=A-ap,memory_projection_b=b-bp,
        profile_curvature=2*quadratic,profile_minimum=float(np.linalg.norm(bp-ap*formal_c)**2) if formal_c is not None else float(bp@bp),
        profile=profile,parameter_residual=residual,
        normal_residual=None if residual is None else np.column_stack((A,Z)).T@residual,
        identified=main["identifiable"],residual_column_ratio=main["residual_column_ratio"])


def initial_conditions():
    E0=np.zeros(15)
    E1=np.r_[[.1,-.05,0.],[0.,0.,0.],[.02,-.01,.005],np.zeros(3),[.015,-.02,.005]]
    E2=np.r_[[.08,.02,-.06],[.05,-.02,.01],[-.01,.015,0.],np.zeros(3),[.02,.01,-.015]]
    return [(c,name,state.copy()) for c in (.7,1.3) for name,state in (("E0",E0),("E1",E1),("E2",E2),("E3",-E2))]+[(0.,"E1",E1.copy())]


def generate(output,counts):
    truth=[]
    observed={}
    plan=[]
    for index,(c,name,initial) in enumerate(initial_conditions()):
        record=dict(index=index,c=c,excitation=name,initial=initial,grids={})
        truth.append(record)
        for step in STEPS:
            counts["generation_attempts"]+=1
            f=functional(0.)
            f.k_edge*=c
            t=np.linspace(0.,.1,int(round(.1/step))+1)
            state=initial.copy()
            states=[state.copy()]
            for _ in range(len(t)-1):
                state=rk4(f,state,step)
                states.append(state.copy())
            record["grids"][str(step)]=dict(t=t,states=np.asarray(states))
            counts["generated_trajectories"]+=1
            write_json(output/"generation.partial.json",dict(truth=truth,actual_counts=counts))
        fine=record["grids"][str(STEPS[1])]["states"][::8]
        noise=np.random.Generator(np.random.PCG64(140000+index)).standard_normal((11,6))
        for sigma in SIGMAS:
            key=f"obs_{len(observed):03d}"
            measurement=fine[:,:6]+.1*sigma*noise
            observed[key]=dict(theta_history=measurement[:,:3],omega_history=measurement[:,3:6],
                               history_times=np.linspace(0.,.1,11),p_end=fine[-1,6:9].copy())
            plan.append(dict(input_id=key,trajectory=index,sigma=sigma,seed=140000+index))
            counts["observation_packages"]+=1
    write_json(output/"isolated_truth.json",truth)
    write_json(output/"observations.json",observed)
    write_json(output/"observation_plan.json",plan)
    return output/"isolated_truth.json",output/"observations.json",output/"observation_plan.json"


def qualify_A(truth,counts):
    f=functional(0.)
    records=[]
    for event in truth:
        grid_records=[]
        residuals=[]
        for step in STEPS:
            counts["A_grid_attempts"]+=1
            grid=event["grids"][str(step)]
            t=np.asarray(grid["t"])
            states=np.asarray(grid["states"])
            terms=weak_terms(t,states[:,:3],states[:,3:6],states[:,6:9],states[:,12:15],f)
            residual=event["c"]*terms["A"]-(terms["inertial"]-terms["damping"]+terms["p"]-terms["r"])
            omit_p=residual+terms["p"]
            omit_r=residual-terms["r"]
            p,rf=reconstruct_ports(t,states[:,3:6],states[-1,6:9],f)
            decay=f.lambda_mem/f.tau_mem_sync
            r=rf+np.exp(-t[:,None]*decay)*np.asarray(event["initial"])[None,12:15]
            p_error=float(np.max(np.abs(p-states[:,6:9]))/.1)
            r_error=float(np.max(np.abs(r-states[:,12:15]))/.1)
            algebra=bool(np.allclose(omit_p-residual,terms["p"],atol=1e-12,rtol=1e-10)
                         and np.allclose(omit_r-residual,-terms["r"],atol=1e-12,rtol=1e-10))
            equilibrium=event["excitation"]!="E0" or (np.all(terms["A"]==0) and np.all(residual==0))
            passed=bool(np.max(np.abs(residual))<=1e-7 and p_error<=1e-7 and r_error<=1e-7 and algebra and equilibrium)
            grid_records.append(dict(step=step,residual=residual,terms=terms,p_reconstruction_max=p_error,
                r_reconstruction_max=r_error,omit_p_residual=omit_p,omit_r_residual=omit_r,algebra_passed=algebra,passed=passed))
            residuals.append(residual)
            counts["A_grid_checks"]+=1
        coarse=np.asarray(event["grids"][str(STEPS[0])]["states"])[::4]
        fine=np.asarray(event["grids"][str(STEPS[1])]["states"])[::8]
        difference=observations(coarse,f.mass)-observations(fine,f.mass)
        generation=dict(rms=float(np.sqrt(np.mean(difference**2))),peak=float(np.max(np.abs(difference))),
                        endpoint_p_scaled_max=float(np.max(np.abs(coarse[-1,6:9]-fine[-1,6:9]))/.1))
        cross=float(np.max(np.abs(residuals[0]-residuals[1])))
        passed=all(row["passed"] for row in grid_records) and cross<=1e-7 and generation["rms"]<=1e-7 and generation["peak"]<=1e-6 and generation["endpoint_p_scaled_max"]<=1e-7
        records.append(dict(trajectory=event["index"],excitation=event["excitation"],grids=grid_records,
                            generation=generation,weak_main_fine_max=cross,passed=bool(passed)))
    excitation={term:any(np.max(np.abs(grid["terms"][term]))*.1>1e-8 for record in records
        if record["excitation"] in ("E1","E2") for grid in record["grids"]) for term in ("p","r")}
    return dict(records=records,port_excitation=excitation,passed=all(row["passed"] for row in records) and all(excitation.values()))


def fit_B(observed,plan,output,counts):
    cached={}
    for key,packet in observed.items():
        for step in STEPS:
            counts["B_reconstruction_attempts"]+=1
            cached[(key,step)]=build_system(packet,step)
            counts["B_reconstructions"]+=1
    results=[]
    for sigma in SIGMAS:
        ids={row["trajectory"]:row["input_id"] for row in plan if row["sigma"]==sigma}
        problems=[(f"single_{i}",[ids[i]]) for i in range(9)]+[("shared_A",[ids[i] for i in (1,2,3)]),("shared_B",[ids[i] for i in (5,6,7)])]
        for step in STEPS:
            for name,keys in problems:
                for variant in VARIANTS:
                    A,Z,b=combine([cached[(key,step)] for key in keys],variant)
                    counts["linear_attempts"]+=1
                    result=solve_linear(A,Z,b)
                    counts["linear_problems"]+=1
                    results.append(dict(problem=name,packet_ids=keys,sigma=sigma,step=step,variant=variant,A=A,Z=Z,b=b,result=result))
                write_json(output/"estimates.partial.json",dict(results=results,actual_counts=counts))
                print(json.dumps(dict(problem=name,sigma=sigma,step=step,linear_problems=counts["linear_problems"])),flush=True)
    return dict(results=results,actual_counts=counts)


def score_frozen(estimates_path,expected_hash,truth_path,plan_path):
    if digest(estimates_path)!=expected_hash:
        raise ValueError("estimates changed before truth scoring")
    estimates=json.loads(Path(estimates_path).read_text(encoding="utf-8"))
    truth=json.loads(Path(truth_path).read_text(encoding="utf-8"))
    plan={row["input_id"]:row for row in json.loads(Path(plan_path).read_text(encoding="utf-8"))}
    rows=[]
    for record in estimates["results"]:
        result=record["result"]
        events=[truth[plan[key]["trajectory"]] for key in record["packet_ids"]]
        c_true=events[0]["c"]
        if any(event["c"]!=c_true for event in events):
            raise ValueError("inconsistent predefined shared-parameter batch")
        rho_true=np.concatenate([np.asarray(event["initial"])[12:15] for event in events])
        identifiable=result["c_hat"] is not None
        row=dict(problem=record["problem"],packet_ids=record["packet_ids"],sigma=record["sigma"],step=record["step"],variant=record["variant"],
                 c_true=c_true,c_error=None if not identifiable else result["c_hat"]-c_true,
                 c_absolute_error=None if not identifiable else abs(result["c_hat"]-c_true),
                 zero_c_absolute_bias=None if c_true!=0. or not identifiable else abs(result["c_hat"]),
                 rho_scaled_error=None if result["rho_hat"] is None or len(result["rho_hat"])!=len(rho_true)
                    else (np.asarray(result["rho_hat"])-rho_true)/.1,
                 noise_pseudo_identification=record["sigma"]>0 and all(event["excitation"]=="E0" for event in events) and identifiable,
                 identified=identifiable,rank_threshold_sensitive=result["rank_threshold_sensitive"],
                 no_noise_accuracy_passed=None if record["sigma"]!=0. else identifiable and abs(result["c_hat"]-c_true)<=.01,
                 single_test_negative_calibration=record["variant"]!="single_test" or not result["identified"])
        rows.append(row)
    for row in rows:
        baseline=next(candidate for candidate in rows if all(candidate[k]==row[k] for k in ("problem","step","variant")) and candidate["sigma"]==0.)
        row["c_absolute_error_change_from_noiseless"]=(None if row["c_absolute_error"] is None or baseline["c_absolute_error"] is None
            else row["c_absolute_error"]-baseline["c_absolute_error"])
    paired=[]
    for row in rows:
        if row["step"]!=STEPS[0]:
            continue
        other=next(candidate for candidate in rows if all(candidate[k]==row[k] for k in ("problem","sigma","variant")) and candidate["step"]==STEPS[1])
        dc=None if row["c_error"] is None or other["c_error"] is None else abs(row["c_error"]-other["c_error"])
        dr=None if row["rho_scaled_error"] is None or other["rho_scaled_error"] is None else float(np.max(np.abs(row["rho_scaled_error"]-other["rho_scaled_error"])))
        paired.append(dict(problem=row["problem"],sigma=row["sigma"],variant=row["variant"],delta_c=dc,delta_rho_scaled_max=dr,
            integration_unresolved=(row["identified"]!=other["identified"] or (dc is not None and dc>1e-3) or (dr is not None and dr>1e-3)),
            rank_threshold_sensitive=row["rank_threshold_sensitive"] or other["rank_threshold_sensitive"]))
    targets=[row for row in rows if row["problem"] in ("shared_A","shared_B") and row["sigma"]==0. and row["variant"]=="complete"]
    return dict(rows=rows,grid_comparisons=paired,
        no_noise_shared_claim=all(row["no_noise_accuracy_passed"] and not row["rank_threshold_sensitive"] for row in targets)
            and len(targets)==4 and not any(row["integration_unresolved"] for row in paired
                if row["problem"] in ("shared_A","shared_B") and row["sigma"]==0. and row["variant"]=="complete"),
        all_single_test_negative_calibrations=all(row["single_test_negative_calibration"] for row in rows),
        scope="fixed interpolated observations; one shared scale, not graph recovery")


def main():
    if digest(ROOT/"reports/gftd_coupling_memory_20261107_protocol.md")!=PROTOCOL_SHA256:
        raise ValueError("weak coupling protocol changed")
    release=json.loads((OUTPUT/"execution_release.json").read_text(encoding="utf-8-sig"))
    if release.get("released") is not True:
        raise PermissionError("root execution release required")
    for path,expected in release["source_sha256"].items():
        if digest(ROOT/path)!=expected:
            raise ValueError(f"frozen source changed: {path}")
    with (OUTPUT/"started.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(action_id=release["action_id"],status="RUNNING"),stream)
    counts=dict(generation_attempts=0,generated_trajectories=0,observation_packages=0,A_grid_attempts=0,A_grid_checks=0,
                B_reconstruction_attempts=0,B_reconstructions=0,linear_attempts=0,linear_problems=0)
    begin=time.perf_counter()
    receipt=dict(action_id=release["action_id"],status="FAILED",actual_counts=counts,numpy=np.__version__,scipy=scipy.__version__)
    try:
        truth_path,obs_path,plan_path=generate(OUTPUT,counts)
        gate=qualify_A(json.loads(truth_path.read_text(encoding="utf-8")),counts)
        write_json(OUTPUT/"qualification_A.json",gate)
        if not gate["passed"]:
            receipt["status"]="STOPPED_A_FAILED_B_NOT_RUN"
            return
        observed=json.loads(obs_path.read_text(encoding="utf-8"))
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        obs_hash,plan_hash=digest(obs_path),digest(plan_path)
        estimates=fit_B(observed,plan,OUTPUT,counts)
        estimates.update(observations_sha256=obs_hash,plan_sha256=plan_hash,source_sha256=release["source_sha256"])
        estimates_path=OUTPUT/"estimates.json"
        write_json(estimates_path,estimates)
        estimates_hash=digest(estimates_path)
        write_json(OUTPUT/"estimates_frozen.json",dict(estimates_sha256=estimates_hash,observations_sha256=obs_hash,plan_sha256=plan_hash))
        if digest(obs_path)!=obs_hash or digest(plan_path)!=plan_hash:
            raise ValueError("legal inputs changed during regression")
        summary=score_frozen(estimates_path,estimates_hash,truth_path,plan_path)
        write_json(OUTPUT/"summary.json",summary)
        if counts["linear_problems"]!=264 or counts["generated_trajectories"]!=18 or counts["B_reconstructions"]!=54:
            raise AssertionError("registered finite workload changed")
        receipt.update(status="COMPLETED",estimates_sha256=estimates_hash,summary_sha256=digest(OUTPUT/"summary.json"))
    except Exception as exc:
        receipt["error"]=repr(exc)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin,iterative_optimization=False,real_data_reads=False,
            source_unchanged=all(digest(ROOT/path)==sha for path,sha in release["source_sha256"].items()))
        write_json(OUTPUT/"execution_receipt.json",receipt)
        print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
