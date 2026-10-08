"""One fixed convex harmonic/conditioned-balance compatibility screen (Loop0095)."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import numpy as np
import scipy  # Deliberately load main numerical runtime before isolated solver wheels.

ROOT=Path(__file__).resolve().parents[1]
P94=ROOT/'private_data/task2_conditional_balance_0094'; A94=P94/'arrays.npz'; M94=P94/'metrics.npz'; E94=P94/'evidence.json'; I94=ROOT/'private_data/task2_conditional_balance_0094_independent/independent.json'; C94=ROOT/'private_data/task2_conditional_balance_0094_independent/check_balance.py'
P92=ROOT/'private_data/task2_backbone_span_0092'; A92=P92/'arrays.npz'; E92=P92/'evidence.json'; I92=ROOT/'private_data/task2_backbone_span_0092_independent/independent.json'; C92=ROOT/'private_data/task2_backbone_span_0092_independent/check_span.py'
A94_SHA='d33cbb83b656e1275ac3af01aa779444db2f33e6a13bbeb285eed2dcf8dae2ec';M94_SHA='e50396f14a1d01f937ddb31f3bba8fe0ac21fc32789a73c39d04cd0278496dbb';E94_SHA='61373d0088cdd7b0271918cbb55a0291d08db753a8344ff6a323bace16134225';I94_SHA='37bbb42ad09f3b23768ee747a45b256a9d9b88a9ebf75f805d1eafffe2908060'
A92_SHA='2b4c0725339c360ea2502e8059e03b8d4f4c9c96197a58701eccd47e26983a8f';E92_SHA='b5b9f7cc6f673fbd6dd4b5d59f6b3cb8d26361c68e4aad590141e72f3dc0a89b';I92_SHA='29f1d15d9c797537ada125f41baee45642184d9c98e894987864c112c8fb6535'
DEPS=ROOT/'.loop-work/research_deps_0067'; METADATA_SECONDS=20.; SOLVE_CAP=600.

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))))
def require(deadline,stage):
    if time.perf_counter()>deadline:raise TimeoutError('0095 '+stage+' cap')
def finite(*values):return all(np.isfinite(np.asarray(value)).all() for value in values)

def _paths(e94,i94,i92):
    anchors=((A94,A94_SHA),(M94,M94_SHA),(E94,E94_SHA),(I94,I94_SHA),(A92,A92_SHA),(E92,E92_SHA),(I92,I92_SHA))
    if any(sha(p)!=h for p,h in anchors):raise AssertionError('prior anchor')
    if e94.get('hashes_before')!=e94.get('hashes_after') or not i94.get('qa_pass') or i94.get('errors')!=[] or i94.get('hashes_before')!=i94.get('hashes_after'):raise AssertionError('0094 QA')
    if not i92.get('qa_pass') or i92.get('errors')!=[] or i92.get('hashes_before')!=i92.get('hashes_after'):raise AssertionError('0092 QA')
    inherited={k:Path(v) for k,v in e94['source_paths'].items()}
    paths={**inherited,'arrays94':A94,'metrics94':M94,'evidence94':E94,'independent94':I94,'checker94':C94,'arrays92':A92,'evidence92':E92,'independent92':I92,'checker92':C92,'code95':Path(__file__),'metric_helper95':ROOT/'experiments/backbone_compatibility_0095.py'}
    before={k:sha(v) for k,v in paths.items()}
    for k,v in e94['hashes_before'].items():
        if k in before and before[k]!=v:raise AssertionError('changed inherited '+k)
    ih=i94['hashes_before']
    if ih.get('arrays')!=A94_SHA or ih.get('evidence')!=E94_SHA or ih.get('checker')!=sha(C94):raise AssertionError('0094 independent links')
    jh=i92['hashes_before']
    if jh.get('arrays')!=A92_SHA or jh.get('evidence')!=E92_SHA or jh.get('checker')!=sha(C92):raise AssertionError('0092 independent links')
    return paths,before

def _candidate(e94):
    paths={k:Path(v) for k,v in e94['candidate_paths'].items()};before={k:sha(v) for k,v in paths.items()}
    if before!=e94.get('candidate_hashes_before') or before!=e94.get('candidate_hashes_after'):raise AssertionError('0094 candidate bytes')
    return paths,before

def _tail_constraints(cp,e,unstable,baseline):
    """Original unsorted tail epigraphs; event scores are normalized by baseline average."""
    tau=cp.Variable(3);u=cp.Variable((3,346));constraints=[u>=0];risks=[]
    for h in range(3):
        constraints += [u[h]>=e[h]-tau[h]]
        risks.append(.8*cp.sum(e[h])/346+.2*(tau[h]+cp.sum(u[h])/35))
    tau_sev=cp.Variable();u_sev=cp.Variable(35);sev=e[2,unstable]
    constraints += [u_sev>=0,u_sev>=sev-tau_sev]
    constraints += [cp.sum(cp.hstack(risks))/3<=1.,cp.sum(sev)/35<=baseline['severe_mean']/baseline['average_risk'],tau_sev+cp.sum(u_sev)/4<=baseline['severe_worst4']/baseline['average_risk']]
    return constraints,{'tau':tau,'u':u,'tau_severe':tau_sev,'u_severe':u_sev,'risks':risks}

def solve_once(X,y,k0,d0,scales,factors,labels,baseline,deadline,counters):
    """The sole permitted CVXPY/Clarabel solve; returns raw variables even on QA failure."""
    if not DEPS.is_dir():raise RuntimeError('registered solver wheels absent')
    if str(DEPS) not in sys.path:sys.path.insert(0,str(DEPS))
    import cvxpy as cp
    X,y,k0,d0,scales,factors=(np.asarray(a,float) for a in (X,y,k0,d0,scales,factors));J0=float(y@y)
    if X.shape!=(2600,36) or y.shape!=(2600,) or k0.shape!=(26,) or d0.shape!=(10,) or scales.shape!=(36,) or factors.shape!=(3,346,37,37) or not finite(X,y,k0,d0,scales,factors) or J0<=0 or np.any(scales<=0):raise ValueError('0095 solver inputs')
    v=cp.Variable(36);h=cp.Variable(2600);z=cp.hstack([v,1.]);e=cp.Variable((3,346));A=X*scales[None]/np.sqrt(J0);b=y/np.sqrt(J0)
    constraints=[h==A@v-b,k0+cp.multiply(scales[:26],v[:26])>=1e-8,d0+cp.multiply(scales[26:],v[26:])>=1e-8,e>=0]
    for hi in range(3):
        for event in range(346):constraints.append(e[hi,event]>=cp.sum_squares(factors[hi,event]@z)/baseline['average_risk'])
    unstable=np.flatnonzero(np.asarray(labels).astype(str)=='unstable')
    if unstable.shape!=(35,):raise AssertionError('unstable set')
    tails,tailvars=_tail_constraints(cp,e,unstable,baseline);constraints += tails
    problem=cp.Problem(cp.Minimize(cp.sum_squares(h)),constraints);counters['solve_attempts']+=1
    kwargs={'solver':'CLARABEL','max_iter':200,'time_limit':SOLVE_CAP,'tol_gap_abs':1e-9,'tol_gap_rel':1e-9,'tol_feas':1e-9,'warm_start':False,'verbose':False}
    partial={'status':None,'solver_kwargs':kwargs,'J0':J0,'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'cvxpy':cp.__version__,'clarabel':__import__('clarabel').__version__}}
    try:problem.solve(**kwargs)
    except Exception as exc:return {**partial,'status':'exception','reason':str(exc),'qa_pass':False,'v':None,'h':None,'e':None}
    partial.update(status=str(problem.status),objective=None if problem.value is None else float(problem.value),v=None if v.value is None else np.asarray(v.value),h=None if h.value is None else np.asarray(h.value),e=None if e.value is None else np.asarray(e.value))
    for key,var in tailvars.items():
        partial[key]=None if not hasattr(var,'value') else (None if var.value is None else np.asarray(var.value))
    rawduals=[None if c.dual_value is None else np.asarray(c.dual_value) for c in constraints]
    partial.update(dual_h_equality=rawduals[0],dual_k_positive=rawduals[1],dual_d_positive=rawduals[2],dual_e_nonnegative=rawduals[3],
                   dual_event_epigraph=None if any(x is None for x in rawduals[4:1042]) else np.asarray(rawduals[4:1042]).reshape(3,346),
                   dual_u_nonnegative=rawduals[1042],dual_tail_u=None if any(x is None for x in rawduals[1043:1046]) else np.asarray(rawduals[1043:1046]),
                   dual_severe_u_nonnegative=rawduals[1046],dual_severe_u=rawduals[1047],dual_average=rawduals[1048],dual_severe_mean=rawduals[1049],dual_severe_worst4=rawduals[1050])
    stats=problem.solver_stats;partial.update(iterations=getattr(stats,'num_iters',None),solve_time=getattr(stats,'solve_time',None))
    solver=problem._solver_cache.get('CLARABEL');info=None if solver is None else solver.get_info();solution=None if solver is None else solver.get_solution()
    partial['native']={} if info is None else {k:getattr(info,k) for k in ('cost_primal','cost_dual','gap_abs','gap_rel','res_primal','res_dual','iterations','solve_time','status')}
    partial['native_solution']={} if solution is None else {k:getattr(solution,k) for k in ('obj_val','obj_val_dual','r_prim','r_dual','iterations','solve_time','status')}
    if partial['v'] is not None:
        beta=scales*partial['v'];direct=float(np.sum((X@beta-y)**2)/J0);partial.update(beta=beta,direct_objective=direct)
        if not finite(beta,direct) or rel(direct,partial['objective'])>1e-7 or (info is not None and rel(direct,float(info.cost_primal))>1e-7):partial['qa_pass']=False;partial['reason']='objective reconstruction'
        else:partial['qa_pass']=bool(problem.status==cp.OPTIMAL and time.perf_counter()<=deadline)
    else:partial['qa_pass']=False
    return partial

def _anchor_summary(zero,full,prior):
    """Fail closed on the registered 0094 base/full score reproductions."""
    for grid in ('fine','coarse'):
        for got,arm in ((zero['grids'][grid],'base'),(full['grids'][grid],'full')):
            for key in ('average_risk','severe_mean','severe_worst4'):
                if rel(got[key],prior['grids'][grid][arm][key])>1e-10:raise AssertionError('0094 score anchor '+grid+'/'+arm+'/'+key)
    return True

def _postchecks(grams,solver,labels,baseline_summary,k0,d0,scales):
    if solver.get('beta') is None:return {'qa_pass':False,'reason':'no primal'}
    from backbone_compatibility_0095 import score,risk_summary
    candidate_scores=score(grams['grams'],solver['v'])
    scores=risk_summary(candidate_scores,labels)
    z=np.r_[solver['v'],1.]
    step=np.einsum('...ij,i,j->...',grams['difference_grams'],z,z)
    if not finite(step): raise FloatingPointError('nonfinite paired energy')
    # Clipping affects only roundoff in a PSD quadratic; independent direct
    # time-domain norms must still agree with every saved un-clipped energy.
    if np.min(step)<-1e-10*max(1.,float(np.max(np.abs(step)))):
        raise FloatingPointError('negative paired energy')
    rms=np.sqrt(np.maximum(step[2],0.))  # full genuine prefix, not shorter horizons
    ratios={};slacks={};feasible=True
    for grid in ('fine','coarse'):
        arm=scores['grids'][grid];base=baseline_summary['grids'][grid]['base']
        ratios[grid]={k:arm[k]/base[k] for k in ('average_risk','severe_mean','severe_worst4')}
        slacks[grid]={k:base[k]-arm[k] for k in ratios[grid]}
        feasible &= all(np.isfinite(value) and value<=1+1e-7 for value in ratios[grid].values())
    native=solver.get('native',{})
    required=('cost_primal','cost_dual','gap_abs','gap_rel','res_primal','res_dual')
    native_finite=all(key in native and native[key] is not None and np.isfinite(native[key]) for key in required)
    absolute_gap=abs(native['cost_primal']-native['cost_dual']) if native_finite else None
    normalized_gap=(absolute_gap/max(1.,abs(native['cost_primal']),abs(native['cost_dual']))
                    if native_finite else None)
    numerical=bool(solver.get('qa_pass') and solver.get('status')=='optimal' and native_finite
                   and absolute_gap<=1e-7 and normalized_gap<=1e-7
                   and abs(native['res_primal'])<=1e-7 and abs(native['res_dual'])<=1e-7)
    amplitudes=np.r_[k0,d0]+scales*solver['v']
    positive=bool(np.isfinite(amplitudes).all() and np.min(amplitudes)>=1e-8-1e-10)
    return {'qa_pass':bool(numerical and feasible and positive and np.max(rms)<=.001),
            'scores':scores,'scores_array':candidate_scores,'step_energy':step,
            'ratios':ratios,'constraint_slacks':slacks,
            'fullprefix_weighted_rms_max':float(np.max(rms)),
            'numerical_optimality':numerical,'normalized_objective_gap':normalized_gap,
            'absolute_objective_gap':absolute_gap,'constraints_feasible':bool(feasible),
            'amplitude_minimum':float(np.min(amplitudes)),'positivity':positive,
            'capture':float(1-solver['direct_objective'])}

def run(output):
    if Path(output).exists():raise FileExistsError(output)
    output=Path(output);output.mkdir(parents=True);start=time.perf_counter();deadline=start+900.;arrays={};paths={};before={};candidate_paths={};candidate_before={};counters={'gram_attempts':0,'eighs':0,'solve_attempts':0};current={}
    try:
        e94=json.loads(E94.read_text());i94=json.loads(I94.read_text());i92=json.loads(I92.read_text());paths,before=_paths(e94,i94,i92);candidate_paths,candidate_before=_candidate(e94)
        prior_summary=e94.get('metrics_summary',{})
        if prior_summary.get('numeric_gate') is not True or prior_summary.get('term_identity_max_relative',float('inf'))>1e-10:raise AssertionError('0094 metric numeric QA')
        for grid in ('fine','coarse'):
            denom=prior_summary.get('denominator',{}).get(grid,{})
            if denom.get('primary_passes') is not True or denom.get('worst4',{}).get('passes') is not True:raise AssertionError('0094 denominator QA '+grid)
        with np.load(A92,allow_pickle=False) as z:X,y,k0,d0=(np.asarray(z[k]) for k in ('X','y','coupling0','damping0'))
        J0=float(y@y);scales=np.sqrt(J0)/np.linalg.norm(X,axis=0)
        if not finite(X,y,k0,d0,scales) or np.any(scales<=0) or np.any(k0<1e-8) or np.any(d0<1e-8):raise AssertionError('harmonic columns')
        with np.load(A94,allow_pickle=False) as z:data={k:np.asarray(z[k]) for k in z.files}
        with np.load(M94,allow_pickle=False) as z:data['metrics94']={k:np.asarray(z[k]) for k in z.files}
        for key in ('mass','D','phi','delta_k','delta_d','qenc'):
            if key not in data or not finite(data[key]):raise AssertionError('0094 finite '+key)
        if data['labels'].shape!=(346,) or data['lengths'].shape!=(346,) or data['event_ids'].shape!=(346,) or len(set(data['event_ids'].astype(str)))!=346:raise AssertionError('0094 event identities')
        # The registered helper builds both grids' Simpson columns/factors once; no model/q replay here.
        from backbone_compatibility_0095 import build_grams, factors, score, risk_summary
        current={'stage':'grams'};counters['gram_attempts']+=1;grams=build_grams(data,scales,deadline=deadline,progress=current,partial_path=output/'partial_grams.npz');arrays.update(X=X,y=y,k0=k0,d0=d0,scales=scales,grams=grams['grams'],difference_grams=grams['difference_grams']);np.savez_compressed(output/'arrays.npz',**arrays)
        current={'stage':'fine factors'};fine_factors,eigmins,factor_errors=factors(grams['grams'][0],deadline=deadline,progress=current,counters=counters)
        if counters['eighs']!=1038 or np.max(factor_errors)>1e-10:raise ArithmeticError('fine Gram factors')
        arrays.update(fine_factors=fine_factors,factor_eigenminimum=eigmins,factor_reconstruction_error=factor_errors);np.savez_compressed(output/'arrays.npz',**arrays)
        baseline=e94['metrics_summary']['grids']['fine']['base']
        vzero=np.zeros(36);vfull=np.r_[data['delta_k'],data['delta_d']]/scales;scores_zero=score(grams['grams'],vzero);scores_full=score(grams['grams'],vfull);anchors=risk_summary(scores_zero,data['labels']);fullanchors=risk_summary(scores_full,data['labels']);_anchor_summary(anchors,fullanchors,e94['metrics_summary'])
        prior_scores=data['metrics94']['scores']
        if prior_scores.shape!=(2,4,3,346) or rel(scores_zero,prior_scores[:,0])>1e-10 or rel(scores_full,prior_scores[:,3])>1e-10:raise AssertionError('0094 all-event score anchors')
        if not (anchors['grids']['fine']['average_risk']<=baseline['average_risk']*(1+1e-7) and anchors['grids']['fine']['severe_mean']<=baseline['severe_mean']*(1+1e-7) and anchors['grids']['fine']['severe_worst4']<=baseline['severe_worst4']*(1+1e-7)):raise AssertionError('zero feasibility')
        if grams['anchors']['increment_reconstruction_max_relative'] is None or grams['anchors']['increment_reconstruction_max_relative']>1e-10:raise AssertionError('known increment anchor')
        arrays.update(scores_zero=scores_zero,scores_full=scores_full);require(deadline,'pre-solve');current={'stage':'solve'};solver=solve_once(X,y,k0,d0,scales,fine_factors,data['labels'],baseline,deadline,counters);arrays.update(**{k:v for k,v in solver.items() if isinstance(v,np.ndarray)})
        np.savez_compressed(output/'arrays.npz',**arrays)
        checks=_postchecks(grams,solver,data['labels'],e94['metrics_summary'],k0,d0,scales)
        if isinstance(checks.get('scores_array'),np.ndarray):arrays.update(scores_candidate=checks.pop('scores_array'),candidate_step_energy=checks.pop('step_energy'));np.savez_compressed(output/'arrays.npz',**arrays)
        after={k:sha(v) for k,v in paths.items()};candidate_after={k:sha(v) for k,v in candidate_paths.items()};require(deadline,'final')
        if before!=after or candidate_before!=candidate_after:raise RuntimeError('source/candidate mutation')
        evidence={'loop':'0095','test_read':False,'h5_read':False,'models_loaded':False,'metadata_seconds':METADATA_SECONDS,'seconds':time.perf_counter()-start,'source_paths':{k:str(v.resolve()) for k,v in paths.items()},'hashes_before':before,'hashes_after':after,'candidate_paths':{k:str(v.resolve()) for k,v in candidate_paths.items()},'candidate_hashes_before':candidate_before,'candidate_hashes_after':candidate_after,'counters':counters,'gram_anchors':grams['anchors'],'solver':{k:v for k,v in solver.items() if not isinstance(v,np.ndarray)},'postchecks':checks}
        (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)));return evidence
    except Exception as exc:
        if arrays:np.savez_compressed(output/'arrays.npz',**arrays)
        (output/'failure_evidence.json').write_text(json.dumps({'loop':'0095','type':type(exc).__name__,'message':str(exc),'current':current,'counters':counters,'seconds':time.perf_counter()-start,'source_paths':{k:str(v) for k,v in paths.items()},'hashes_before':before},indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
