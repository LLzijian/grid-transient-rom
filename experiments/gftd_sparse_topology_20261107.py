"""K4 sparse static-edge reference, with complete ports and event memory elimination."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import nnls

from gftd_initialization_20261107 import ROOT, digest, write_json
from gftd_coupling_memory_20261107 import STEPS, SIGMAS, boole_weights, test_functions, linear_rk4
from gftd_observability_20261107 import rk4
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates

OUTPUT = ROOT/"results/gftd_sparse_topology_20261107"
PROTOCOL_SHA256 = "040b03bec2bf2021df7bb4ff1d18ba769095d5023aa73e6178ce38924e510368"
TAU_ABS = 1e-6
THRESHOLDS = (1e-8,1e-10,1e-12)


def functional():
    n=4
    edges=n*(n-1)//2
    return GeneratingFunctionalKuramoto(
        mass=np.array([1.,1.3,1.7,2.]),damping=np.full(n,.2),k_edge=np.ones(edges),eta_edge=np.zeros(edges),
        gamma_amp=np.zeros(n),lambda_amp=np.ones(n),lambda_mem=np.full(n,1/.7),tau_amp=np.full(n,5.),
        tau_mem_sync=np.full(n,1/(.7*.3)),tau_mem_coi=1/(.7*.3),j_omega_mem=np.full(n,.7),
        gates=StructuralGates.static_baseline_limit(np.ones(edges),n),storage_p=np.ones(n),
        gain_p=np.ones(n),rate_p=np.full(n,.3))


def validate_observation(packet,n):
    if set(packet)!={"theta_history","omega_history","history_times","p_end","sigma"}:
        raise ValueError("only theta/omega history, times, endpoint p and known sigma are allowed")
    result={name:np.asarray(value,dtype=np.float64).copy() for name,value in packet.items()}
    if result["theta_history"].shape!=(11,n) or result["omega_history"].shape!=(11,n) or result["p_end"].shape!=(n,):
        raise ValueError("eleven node-aligned observations required")
    if result["history_times"].shape!=(11,) or not np.allclose(result["history_times"],np.linspace(0.,.1,11),atol=1e-12,rtol=0):
        raise ValueError("history_times must span 0 through .1 seconds")
    if result["sigma"].shape!=() or float(result["sigma"]) not in SIGMAS:
        raise ValueError("unregistered noise scale")
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("observations must be finite")
    return result


def reconstruct_ports(t,omega,p_end,f):
    t=np.asarray(t,dtype=np.float64)
    omega=np.asarray(omega,dtype=np.float64)
    n=f.n_nodes
    def forcing(time):
        return np.array([np.interp(time,t,omega[:,node]) for node in range(n)])
    p=np.zeros_like(omega)
    rf=np.zeros_like(omega)
    p[-1]=p_end
    for index in range(len(t)-1,0,-1):
        p[index-1]=linear_rk4(lambda time,value:-f.rate_p*f.storage_p*value-f.gain_p*forcing(time),
                              t[index],p[index],t[index-1]-t[index])
    for index in range(len(t)-1):
        rf[index+1]=linear_rk4(lambda time,value:f.j_omega_mem*forcing(time)-f.lambda_mem/f.tau_mem_sync*value,
                               t[index],rf[index],t[index+1]-t[index])
    return p,rf


def weak_terms(t,theta,omega,p,memory,f):
    v,dv=test_functions(t)
    weights=boole_weights(t)
    weighted=v*weights
    sine=np.sin(np.asarray(theta)@f.incidence)-np.sin(f.eta_edge)
    edge_integrals=weighted@sine
    A=np.vstack([f.incidence*row[None,:] for row in edge_integrals])/.1
    return dict(A=A,inertial=((dv*weights)@(omega*f.mass)).reshape(-1)/.1,
                damping=(weighted@(omega*f.damping)).reshape(-1)/.1,
                p=(weighted@(p*f.gain_p*f.storage_p)).reshape(-1)/.1,
                r=(weighted@memory).reshape(-1)/.1)


def build_system(packet,step,f=None):
    f=functional() if f is None else f
    packet=validate_observation(packet,f.n_nodes)
    if step not in STEPS:
        raise ValueError("unregistered reconstruction grid")
    t=np.linspace(0.,.1,int(round(.1/step))+1)
    dense={name:np.column_stack([np.interp(t,packet["history_times"],packet[name][:,node])
                               for node in range(f.n_nodes)]) for name in ("theta_history","omega_history")}
    p,rf=reconstruct_ports(t,dense["omega_history"],packet["p_end"],f)
    terms=weak_terms(t,dense["theta_history"],dense["omega_history"],p,rf,f)
    v,_=test_functions(t)
    integral=(v*boole_weights(t))@np.exp(-t[:,None]*(f.lambda_mem/f.tau_mem_sync)[None,:])
    Z=np.vstack([np.diag(row) for row in integral])
    return dict(A=terms["A"],Z=Z,b=terms["inertial"]-terms["damping"]+terms["p"]-terms["r"],
                p_reconstructed=p,rf_reconstructed=rf,t=t)


def combine(systems):
    A=np.vstack([system["A"] for system in systems])
    b=np.concatenate([system["b"] for system in systems])
    Z=np.zeros((len(b),sum(system["Z"].shape[1] for system in systems)))
    row=column=0
    for system in systems:
        rows,columns=system["Z"].shape
        Z[row:row+rows,column:column+columns]=system["Z"]
        row+=rows
        column+=columns
    return A,Z,b


def ranks(singular):
    return {str(t):int(np.sum(singular>t*singular[0])) if singular.size and singular[0]>0 else 0 for t in THRESHOLDS}


def project_memory(A,Z,b):
    A,Z,b=(np.asarray(value,dtype=np.float64) for value in (A,Z,b))
    if A.ndim!=2 or A.shape[1]!=6 or Z.ndim!=2 or b.shape!=(A.shape[0],) or Z.shape[0]!=len(b) or not all(np.isfinite(v).all() for v in (A,Z,b)):
        raise ValueError("finite six-edge weak system required")
    U,s,Vt=np.linalg.svd(Z,full_matrices=False)
    rank=ranks(s)[str(1e-10)]
    basis=U[:,:rank]
    residual_A=A-basis@(basis.T@A)
    residual_b=b-basis@(basis.T@b)
    scale=np.sqrt(len(b))
    return dict(X=residual_A/scale,y=residual_b/scale,U=U,s=s,Vt=Vt,memory_rank=rank,
        memory_ranks=ranks(s),orthogonality_A=basis.T@residual_A,orthogonality_b=basis.T@residual_b)


def information_diagnostics(projection):
    X=projection["X"]
    norms=np.linalg.norm(X,axis=0)
    singular=np.linalg.svd(X,compute_uv=False)
    correlation=np.divide(X.T@X,norms[:,None]*norms[None,:],out=np.zeros((6,6)),where=(norms[:,None]*norms[None,:])>0)
    conditional=[]
    for edge in range(6):
        U,s,_=np.linalg.svd(np.delete(X,edge,axis=1),full_matrices=False)
        rank_info=ranks(s)
        rank=rank_info[str(1e-10)]
        residual=X[:,edge]-U[:,:rank]@(U[:,:rank].T@X[:,edge])
        norm=float(np.linalg.norm(residual))
        conditional.append(dict(norm=norm,ratio=norm/float(norms[edge]) if norms[edge]>0 else 0.,other_ranks=rank_info))
    off_diagonal=np.abs(correlation-np.diag(np.diag(correlation)))
    norm=float(np.linalg.norm(X))
    rank_info=ranks(singular)
    rank_sensitive=len(set(rank_info.values()))>1 or len(set(projection["memory_ranks"].values()))>1 or any(
        len(set(row["other_ranks"].values()))>1 for row in conditional)
    return dict(frobenius_norm=norm,column_norms=norms,singular_values=singular,ranks=rank_info,
        conditional_columns=conditional,correlation=correlation,max_absolute_correlation=float(np.max(off_diagonal)),
        confounded=any(row["ratio"]<=1e-8 for row in conditional) or bool(np.max(off_diagonal)>=1-1e-8),
        unexcited_edges=np.flatnonzero(norms<=TAU_ABS).tolist(),rank_threshold_sensitive=rank_sensitive,
        information_status="insufficient" if norm<=TAU_ABS else "candidate_information")


def kkt_audit(X,y,k):
    finite=all(np.isfinite(value).all() for value in (X,y,k))
    gradient=X.T@(X@k-y)
    violation=np.where(k>1e-10,np.abs(gradient),np.maximum(0.,-gradient))
    v=float(np.max(violation)) if len(k) else 0.
    norm=float(np.linalg.norm(X,2)) if X.size else 0.
    d=norm*(norm*float(np.linalg.norm(k))+float(np.linalg.norm(y)))
    feasible=finite and bool(np.all(k>=-1e-12))
    relative=v/d if d>0 else None
    qualified=feasible and (relative<=1e-9 if relative is not None else bool(np.all(gradient==0)))
    return dict(gradient=gradient,violation=v,data_scale=d,relative_violation=relative,
                feasible=feasible,qualified=bool(qualified),trivial_optimum=bool(qualified and d==0))


def solve_subset(X,y,subset,counts):
    counts["algebra_terms_actual"]+=1
    subset=list(subset)
    if not subset:
        return dict(subset=[],actual=True,k=np.zeros(6),loss=float(y@y),status="empty_algebra",nnls_called=False,
                    audit=dict(qualified=True,trivial_optimum=True))
    counts["nnls_attempts"]+=1
    try:
        values,_=nnls(X[:,subset],y,maxiter=200)
        counts["nnls_completed"]+=1
        audit=kkt_audit(X[:,subset],y,values)
        full=np.zeros(6)
        full[subset]=values
        if audit["qualified"]:
            counts["nnls_qualified"]+=1
        return dict(subset=subset,actual=True,k=full,loss=float(np.linalg.norm(X@full-y)**2),
                    status="qualified" if audit["qualified"] else "solver_unresolved",nnls_called=True,audit=audit)
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        return dict(subset=subset,actual=True,k=None,loss=None,status="solver_unresolved",nnls_called=True,error=repr(exc))


def infer_setting(A,Z,b,sigma,counts):
    if sigma not in SIGMAS:
        raise ValueError("unregistered known noise scale")
    projection=project_memory(A,Z,b)
    information=information_diagnostics(projection)
    X,y=projection["X"],projection["y"]
    subsets=[list(itertools.compress(range(6),[(mask>>edge)&1 for edge in range(6)])) for mask in range(64)]
    if information["information_status"]=="insufficient":
        counts["algebra_terms_skipped"]+=65
        return dict(A=A,Z=Z,b=b,projection=projection,information=information,
            enumeration=[dict(subset=subset,actual=False,reason="insufficient_information") for subset in subsets],
            debias=dict(actual=False,reason="insufficient_information"),raw_support=None,raw_k=None,raw_rho=None,
            algorithm_unique=False,enumeration_unresolved=False,support_threshold_inconsistent=False,
            support=None,support_qualified=False)
    penalty=(1e-6+.1*sigma)**2
    enumeration=[]
    for subset in subsets:
        result=solve_subset(X,y,subset,counts)
        result["J"]=None if result["loss"] is None else result["loss"]+penalty*len(subset)
        result["contains_unexcited_column"]=bool(set(subset)&set(information["unexcited_edges"]))
        enumeration.append(result)
    available=[row for row in enumeration if row["J"] is not None]
    minimum=min(row["J"] for row in available)
    tied=[row for row in available if row["J"]-minimum<=1e-14*max(1.,abs(minimum))]
    representative=min(tied,key=lambda row:(len(row["subset"]),row["subset"]))
    support=np.flatnonzero(np.asarray(representative["k"])>=.05).tolist()
    debias=solve_subset(X,y,support,counts)
    unresolved=any(row["status"]=="solver_unresolved" for row in enumeration) or debias["status"]=="solver_unresolved"
    raw_k=debias["k"]
    threshold_bad=raw_k is not None and any(raw_k[edge]<.05 for edge in support)
    rho=None
    if raw_k is not None:
        rank=projection["memory_rank"]
        rho=.1*(projection["Vt"][:rank].T@((projection["U"][:,:rank].T@(b-A@raw_k))/projection["s"][:rank]))
    return dict(A=A,Z=Z,b=b,projection=projection,information=information,penalty=penalty,
        enumeration=enumeration,J_min=minimum,tied_supports=[row["subset"] for row in tied],representative_subset=representative["subset"],
        pre_debias_k=representative["k"],debias=debias,raw_support=support,raw_k=raw_k,raw_rho=rho,
        algorithm_unique=len(tied)==1 and not unresolved,enumeration_unresolved=unresolved,
        support_threshold_inconsistent=bool(threshold_bad),support=None,support_qualified=False)


def qualify_pair(coarse,fine):
    insufficient=coarse["information"]["information_status"]==fine["information"]["information_status"]=="insufficient"
    same_support=coarse["raw_support"]==fine["raw_support"]
    delta=None if coarse["raw_k"] is None or fine["raw_k"] is None else float(np.max(np.abs(np.asarray(coarse["raw_k"])-fine["raw_k"])))
    def rank_signature(result):
        rank_sets=[result["information"]["ranks"],result["projection"]["memory_ranks"]]
        rank_sets.extend(row["other_ranks"] for row in result["information"]["conditional_columns"])
        return tuple(tuple(rank_set[str(threshold)] for threshold in THRESHOLDS) for rank_set in rank_sets)
    numerical=insufficient or (same_support and delta is not None and delta<=1e-3
        and coarse["information"]["rank_threshold_sensitive"]==fine["information"]["rank_threshold_sensitive"]
        and rank_signature(coarse)==rank_signature(fine)
        and not coarse["enumeration_unresolved"] and not fine["enumeration_unresolved"])
    reasons=[]
    for name,result in (("coarse",coarse),("fine",fine)):
        info=result["information"]
        checks=dict(global_excitation=info["frobenius_norm"]>TAU_ABS,all_edges_excited=len(info["unexcited_edges"])==0,
            full_rank_at_all_thresholds=all(value==6 for value in info["ranks"].values()),
            rank_insensitive=not info["rank_threshold_sensitive"],conditional_information=all(row["ratio"]>1e-8 for row in info["conditional_columns"]),
            correlations=info["max_absolute_correlation"]<1-1e-8,enumeration_resolved=not result["enumeration_unresolved"],
            algorithm_unique=result["algorithm_unique"],threshold_consistent=not result["support_threshold_inconsistent"])
        reasons.extend(name+"."+key for key,passed in checks.items() if not passed)
    if not numerical:
        reasons.append("integration_or_solver_unresolved")
    qualified=not reasons
    for result in (coarse,fine):
        result["support_qualified"]=qualified
        result["support"]=result["raw_support"] if qualified else None
        result["qualification_reasons"]=reasons.copy()
    return dict(numerically_qualified=bool(numerical),support_qualified=qualified,reasons=reasons,delta_k_max=delta,
                same_raw_support=same_support,insufficient_negative_calibration=insufficient)


def conditions():
    graphs=(np.array([0.,0.,0.,0.,0.,0.]),np.array([.7,0.,0.,.9,0.,1.1]),
            np.array([.8,1.,1.2,0.,0.,0.]),np.array([.7,0.,1.1,.9,0.,1.2]))
    states=[np.zeros(20),np.r_[[.12,-.08,.03,-.04],[.03,-.02,.01,0.],[.02,-.01,.005,0.],np.zeros(4),[.015,-.02,.005,.01]],
        np.r_[[-.03,.11,-.07,.05],[-.01,.04,-.03,.02],[-.01,.015,0.,.005],np.zeros(4),[.02,.01,-.015,-.005]],
        np.r_[[.04,-.02,.13,-.09],[.02,.01,-.04,.03],[.005,0.,-.015,.02],np.zeros(4),[-.01,.02,.01,-.015]]]
    return [(graph,excitation,k.copy(),state.copy()) for graph,k in enumerate(graphs) for excitation,state in enumerate(states)]


def generate(output,counts):
    truth=[]
    observed={}
    plan=[]
    for index,(graph,excitation,k,initial) in enumerate(conditions()):
        record=dict(index=index,graph=graph,excitation=excitation,k=k,initial=initial,grids={})
        truth.append(record)
        for step in STEPS:
            counts["generation_attempts"]+=1
            f=functional()
            f.k_edge=k.copy()
            t=np.linspace(0.,.1,int(round(.1/step))+1)
            state=initial.copy()
            states=[state.copy()]
            for _ in range(len(t)-1):
                state=rk4(f,state,step)
                states.append(state.copy())
            record["grids"][str(step)]=dict(t=t,states=np.asarray(states))
            counts["generated_grids"]+=1
            write_json(output/"generation.partial.json",dict(truth=truth,actual_counts=counts))
        fine=record["grids"][str(STEPS[1])]["states"][::8]
        noise=np.random.Generator(np.random.PCG64(150000+index)).standard_normal((11,8))
        for sigma in SIGMAS:
            key=f"obs_{len(observed):03d}"
            measurement=fine[:,:8]+.1*sigma*noise
            observed[key]=dict(theta_history=measurement[:,:4],omega_history=measurement[:,4:8],
                p_end=fine[-1,8:12].copy(),history_times=np.linspace(0.,.1,11),sigma=sigma)
            plan.append(dict(input_id=key,batch=graph,excitation=excitation,trajectory=index,sigma=sigma,seed=150000+index))
            counts["observation_packages"]+=1
    write_json(output/"isolated_truth.json",truth)
    write_json(output/"observations.json",observed)
    write_json(output/"observation_plan.json",plan)
    return output/"isolated_truth.json",output/"observations.json",output/"observation_plan.json"


def qualify_A(truth,counts):
    f=functional()
    records=[]
    for event in truth:
        grids=[]
        residuals=[]
        for step in STEPS:
            counts["A_grid_attempts"]+=1
            grid=event["grids"][str(step)]
            t=np.asarray(grid["t"])
            states=np.asarray(grid["states"])
            terms=weak_terms(t,states[:,:4],states[:,4:8],states[:,8:12],states[:,16:20],f)
            residual=terms["A"]@event["k"]-(terms["inertial"]-terms["damping"]+terms["p"]-terms["r"])
            omit_p=residual+terms["p"]
            omit_r=residual-terms["r"]
            p,rf=reconstruct_ports(t,states[:,4:8],states[-1,8:12],f)
            r=rf+np.exp(-t[:,None]*(f.lambda_mem/f.tau_mem_sync))*np.asarray(event["initial"])[None,16:20]
            p_error=float(np.max(np.abs(p-states[:,8:12]))/.1)
            r_error=float(np.max(np.abs(r-states[:,16:20]))/.1)
            algebra=bool(np.allclose(omit_p-residual,terms["p"],atol=1e-12,rtol=1e-10)
                and np.allclose(omit_r-residual,-terms["r"],atol=1e-12,rtol=1e-10))
            passed=bool(np.max(np.abs(residual))<=1e-7 and algebra
                and (step!=STEPS[1] or (p_error<=1e-7 and r_error<=1e-7)))
            grids.append(dict(step=step,residual=residual,terms=terms,p_reconstruction_max=p_error,
                r_reconstruction_max=r_error,algebra_passed=algebra,passed=passed))
            residuals.append(residual)
            counts["A_grid_checks"]+=1
        coarse=np.asarray(event["grids"][str(STEPS[0])]["states"])[::4]
        fine=np.asarray(event["grids"][str(STEPS[1])]["states"])[::8]
        dtheta=coarse[:,:4]-fine[:,:4]
        dtheta-= (dtheta@f.coi_weights)[:,None]
        difference=np.column_stack((dtheta,coarse[:,4:8]-fine[:,4:8]))/.1
        generation=dict(rms=float(np.sqrt(np.mean(difference**2))),peak=float(np.max(np.abs(difference))),
                        endpoint_p_scaled_max=float(np.max(np.abs(coarse[-1,8:12]-fine[-1,8:12]))/.1))
        cross=float(np.max(np.abs(residuals[0]-residuals[1])))
        passed=all(row["passed"] for row in grids) and cross<=1e-7 and generation["rms"]<=1e-7 and generation["peak"]<=1e-6 and generation["endpoint_p_scaled_max"]<=1e-7
        records.append(dict(trajectory=event["index"],excitation=event["excitation"],grids=grids,
                            generation=generation,weak_main_fine_max=cross,passed=bool(passed)))
    excitation={term:any(np.max(np.abs(grid["terms"][term]))*.1>1e-8 for record in records
        if record["excitation"]!=0 for grid in record["grids"]) for term in ("p","r")}
    return dict(records=records,port_excitation=excitation,passed=all(row["passed"] for row in records) and all(excitation.values()))


def fit_B(observed,plan,output,counts):
    cached={}
    for key,packet in observed.items():
        for step in STEPS:
            counts["reconstruction_attempts"]+=1
            cached[(key,step)]=build_system(packet,step)
            counts["reconstructions"]+=1
    results=[]
    for batch in range(4):
        for sigma in SIGMAS:
            packets={row["excitation"]:row["input_id"] for row in plan if row["batch"]==batch and row["sigma"]==sigma}
            for group,keys in (("excited",[packets[i] for i in (1,2,3)]),("equilibrium",[packets[0]])):
                pair=[]
                for step in STEPS:
                    A,Z,b=combine([cached[(key,step)] for key in keys])
                    counts["settings_attempted"]+=1
                    result=infer_setting(A,Z,b,sigma,counts)
                    row=dict(batch_id=batch,group=group,sigma=sigma,step=step,packet_ids=keys,result=result)
                    results.append(row)
                    pair.append(result)
                    counts["settings_completed"]+=1
                check=qualify_pair(*pair)
                for row in results[-2:]:
                    row["pair_qualification"]=check
                write_json(output/"estimates.partial.json",dict(results=results,actual_counts=counts))
                print(json.dumps(dict(batch_id=batch,group=group,sigma=sigma,settings_completed=counts["settings_completed"],
                                      nnls_attempts=counts["nnls_attempts"])),flush=True)
    return dict(results=results,actual_counts=counts,edges=functional().edges)


def score_frozen(estimates_path,expected_hash,truth_path,plan_path):
    if digest(estimates_path)!=expected_hash:
        raise ValueError("estimates changed before truth scoring")
    estimates=json.loads(Path(estimates_path).read_text(encoding="utf-8"))
    truth=json.loads(Path(truth_path).read_text(encoding="utf-8"))
    plan={row["input_id"]:row for row in json.loads(Path(plan_path).read_text(encoding="utf-8"))}
    rows=[]
    for row in estimates["results"]:
        result=row["result"]
        events=[truth[plan[key]["trajectory"]] for key in row["packet_ids"]]
        k_true=np.asarray(events[0]["k"])
        if any(not np.array_equal(k_true,event["k"]) for event in events):
            raise ValueError("predefined graph batch is inconsistent")
        true_support=np.flatnonzero(k_true>0).tolist()
        support=result["support"]
        raw=result["raw_support"]
        rho=np.concatenate([np.asarray(event["initial"])[16:20] for event in events])
        error=None if result["raw_k"] is None else np.asarray(result["raw_k"])-k_true
        rows.append(dict(batch_id=row["batch_id"],group=row["group"],sigma=row["sigma"],step=row["step"],
            support=support,raw_support=raw,support_qualified=result["support_qualified"],true_support=true_support,
            exact_support=result["support_qualified"] and support==true_support,
            false_positive=None if support is None else len(set(support)-set(true_support)),
            false_negative=None if support is None else len(set(true_support)-set(support)),
            raw_false_positive=None if raw is None else len(set(raw)-set(true_support)),
            raw_false_negative=None if raw is None else len(set(true_support)-set(raw)),
            k_error=error,max_k_error=None if error is None else float(np.max(np.abs(error))),
            rho_scaled_error=None if result["raw_rho"] is None else (np.asarray(result["raw_rho"])-rho)/.1,
            noise_pseudo_discovery=row["sigma"]>0 and all(event["excitation"]==0 for event in events) and support is not None,
            information_status=result["information"]["information_status"],qualification_reasons=result["qualification_reasons"],
            pair_qualification=row["pair_qualification"]))
    targets=[row for row in rows if row["group"]=="excited" and row["sigma"]==0.]
    return dict(rows=rows,noiseless_four_batch_claim=len(targets)==8 and all(
        row["exact_support"] and row["max_k_error"] is not None and row["max_k_error"]<=.05
        and row["pair_qualification"]["numerically_qualified"] for row in targets),
        scope="K4 finite subset reference, not generic topology recovery")


def main():
    if digest(ROOT/"reports/gftd_sparse_topology_20261107_protocol.md")!=PROTOCOL_SHA256:
        raise ValueError("sparse topology protocol changed")
    release=json.loads((OUTPUT/"execution_release.json").read_text(encoding="utf-8-sig"))
    if release.get("released") is not True:
        raise PermissionError("root execution release required")
    for path,expected in release["source_sha256"].items():
        if digest(ROOT/path)!=expected:
            raise ValueError(f"frozen source changed: {path}")
    with (OUTPUT/"started.json").open("x",encoding="utf-8") as stream:
        json.dump(dict(action_id=release["action_id"],status="RUNNING"),stream)
    counts=dict(generation_attempts=0,generated_grids=0,observation_packages=0,A_grid_attempts=0,A_grid_checks=0,
        reconstruction_attempts=0,reconstructions=0,settings_attempted=0,settings_completed=0,
        algebra_terms_planned=3120,algebra_terms_actual=0,algebra_terms_skipped=0,nnls_attempts=0,nnls_completed=0,nnls_qualified=0)
    begin=time.perf_counter()
    receipt=dict(action_id=release["action_id"],status="FAILED",actual_counts=counts,numpy=np.__version__,scipy=scipy.__version__)
    try:
        if np.__version__!="2.5.2" or scipy.__version__!="1.18.0":
            raise ValueError("registered NumPy/SciPy versions required")
        truth_path,obs_path,plan_path=generate(OUTPUT,counts)
        gate=qualify_A(json.loads(truth_path.read_text(encoding="utf-8")),counts)
        write_json(OUTPUT/"qualification_A.json",gate)
        if not gate["passed"]:
            receipt["status"]="STOPPED_A_FAILED_B_NOT_RUN"
            return
        observed=json.loads(obs_path.read_text(encoding="utf-8"))
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        hashes=dict(observations=digest(obs_path),plan=digest(plan_path))
        estimates=fit_B(observed,plan,OUTPUT,counts)
        estimates.update(input_sha256=hashes,source_sha256=release["source_sha256"])
        estimates_path=OUTPUT/"estimates.json"
        write_json(estimates_path,estimates)
        estimates_hash=digest(estimates_path)
        write_json(OUTPUT/"estimates_frozen.json",dict(estimates_sha256=estimates_hash,input_sha256=hashes))
        if digest(obs_path)!=hashes["observations"] or digest(plan_path)!=hashes["plan"]:
            raise ValueError("legal observations changed during estimation")
        summary=score_frozen(estimates_path,estimates_hash,truth_path,plan_path)
        write_json(OUTPUT/"summary.json",summary)
        if (counts["generated_grids"]!=32 or counts["observation_packages"]!=48 or counts["reconstructions"]!=96
                or counts["settings_completed"]!=48 or counts["algebra_terms_actual"]+counts["algebra_terms_skipped"]!=3120
                or counts["nnls_attempts"]>3072):
            raise AssertionError("finite sparse-reference workload changed")
        receipt.update(status="COMPLETED",estimates_sha256=estimates_hash,summary_sha256=digest(OUTPUT/"summary.json"))
    except Exception as exc:
        receipt["error"]=repr(exc)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin,real_data_reads=False,source_unchanged=all(
            digest(ROOT/path)==sha for path,sha in release["source_sha256"].items()))
        write_json(OUTPUT/"execution_receipt.json",receipt)
        print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
