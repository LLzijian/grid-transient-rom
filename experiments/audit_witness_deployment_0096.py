"""Deploy exactly one feasible numerical point retained by Loop0095 (no solve)."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1];P95=ROOT/'private_data/task2_backbone_compatibility_0095';A95=P95/'arrays.npz';E95=P95/'evidence.json';I95=ROOT/'private_data/task2_backbone_compatibility_0095_independent/independent.json';C95=ROOT/'private_data/task2_backbone_compatibility_0095_independent/check_compatibility.py'
BASE=ROOT/'private_data/task2_structure_0078_extend_G-fixed/step3000';P83=ROOT/'private_data/task2_coefficient_poles_0083/matrices.npz'
A95_SHA='ef147edab2eb4c20aa962ede1e28a9fea80b0416b0544a3826e573e14074e75d';E95_SHA='668d82bf9d7361308563d5c4f2fb478b4f4d933c8f45ec67522c0d1ff5354128';I95_SHA='9639b8f03710cd3f67cf82da5c059deacfdedc562d9555b71168edfc0cb3136a';C95_SHA='94a1c06fb5e2ebeae4f7bde1a44731d3d64279c88fbe2b7a9e501ff39d094a87';BASE_SHA='2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b';METADATA_SECONDS=20.

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))))
def require(deadline,stage):
    if time.perf_counter()>deadline:raise TimeoutError('0096 '+stage+' cap')

def partition_witness(beta,k0,d0):
    beta,k0,d0=map(lambda x:np.asarray(x,float),(beta,k0,d0))
    if beta.shape!=(36,) or k0.shape!=(26,) or d0.shape!=(10,) or not all(np.isfinite(x).all() for x in (beta,k0,d0)):raise ValueError('witness shape/finite')
    k,d=k0+beta[:26],d0+beta[26:]
    if not(np.isfinite(k).all() and np.isfinite(d).all() and np.all(k>0) and np.all(d>0)):raise ValueError('nonpositive witness')
    return k,d

def _prior(e95,i95):
    if len(C95_SHA)!=64:raise AssertionError('checker anchor length')
    if any(sha(p)!=h for p,h in ((A95,A95_SHA),(E95,E95_SHA),(I95,I95_SHA),(C95,C95_SHA))):raise AssertionError('0095 anchor')
    if not i95.get('qa_pass') or i95.get('errors')!=[] or i95.get('hashes_before')!=i95.get('hashes_after'):raise AssertionError('0095 independent')
    ih=i95['hashes_before']
    if ih.get('arrays')!=A95_SHA or ih.get('evidence')!=E95_SHA or ih.get('checker')!=C95_SHA:raise AssertionError('0095 independent links')
    if e95.get('solver',{}).get('status')!='optimal_inaccurate' or e95.get('counters',{}).get('solve_attempts')!=1 or e95.get('counters',{}).get('eighs')!=1038:raise AssertionError('0095 retained status/count')
    post=e95.get('postchecks',{})
    if post.get('constraints_feasible') is not True or post.get('positivity') is not True or post.get('fullprefix_weighted_rms_max',float('inf'))>.001:raise AssertionError('0095 point infeasible')
    if e95.get('hashes_before')!=e95.get('hashes_after'):raise AssertionError('0095 source mutation')
    paths={k:Path(v) for k,v in e95['source_paths'].items()}
    if 'code93' not in paths:raise AssertionError('inherited code93 absent')
    oldcandidate={f'prior95_{k}':Path(v) for k,v in e95.get('candidate_paths',{}).items()}
    oldhash=e95.get('candidate_hashes_before',{})
    if set(oldcandidate)!={f'prior95_{k}' for k in oldhash} or any(sha(path)!=oldhash[name.removeprefix('prior95_')] for name,path in oldcandidate.items()) or oldhash!=e95.get('candidate_hashes_after'):raise AssertionError('0095 candidate chain')
    paths.update(oldcandidate);paths.update(arrays95=A95,evidence95=E95,independent95=I95,checker95=C95,code96=Path(__file__),deployment_helper93=ROOT/'experiments/audit_backbone_deployment_0093.py',forecast_helper93=ROOT/'experiments/backbone_forecast_0093.py')
    before={k:sha(v) for k,v in paths.items()}
    for k,v in e95['hashes_before'].items():
        if k in before and before[k]!=v:raise AssertionError('changed inherited '+k)
    return paths,before

def _witness(e95):
    with np.load(A95,allow_pickle=False) as z:
        required=('beta','k0','d0','scales');
        if any(k not in z.files for k in required):raise KeyError('missing 0095 witness')
        beta,k0,d0,scales=(np.asarray(z[k]) for k in required)
    k,d=partition_witness(beta,k0,d0)
    if scales.shape!=(36,) or not np.isfinite(scales).all():raise ValueError('witness scales')
    return beta,k0,d0,scales,k,d

def run(output):
    if Path(output).exists():raise FileExistsError(output)
    import torch
    torch.set_num_threads(1)
    from train_structured_rom import load_model
    import audit_backbone_deployment_0093 as d93
    from fit_coefficient_block_0082 import deterministic_test_z,structural_qa
    from compare_physical_modes import ordinary_modes,physical_output
    from backbone_forecast_0093 import forecast_candidate,numeric_metrics,score_pair,validate_cache
    output=Path(output);output.mkdir(parents=True);start=time.perf_counter();deadline=start+1200.;arrays={};current={};paths={};before={};counters={'ad':0,'fd_rhs':0,'eig':0,'forecast_runs':0,'candidate_raw_writes':0};candidate_paths={};candidate_hashes={}
    try:
        e95=json.loads(E95.read_text());i95=json.loads(I95.read_text());paths,before=_prior(e95,i95);beta,k0,d0,scales,k,d=_witness(e95);arrays.update(beta=beta,scales=scales,k0=k0,d0=d0,k=k,d=d)
        if sha(BASE/'model.pt')!=BASE_SHA:raise AssertionError('baseline model SHA')
        model=load_model(BASE);basek=np.asarray(model.structured._pos('coupling').detach());based=np.asarray(model.structured._pos('damping').detach())
        if rel(basek,k0)>1e-12 or rel(based,d0)>1e-12:raise AssertionError('witness baseline amplitudes')
        candidate=output/'candidate';state_before,changed=d93.write_candidate(model,BASE,candidate,k,d,counters);candidate_paths={'deployment_candidate_model':candidate/'model.pt','deployment_candidate_meta':candidate/'checkpoint.json','deployment_candidate_norm':candidate/'normalization.npz','deployment_candidate_export':candidate/'structured_numeric_export.json'};candidate_hashes['after_write']={x:sha(y) for x,y in candidate_paths.items()}
        reloaded=load_model(candidate);d93._unit_masks(reloaded.structured);d93._freeze_check(state_before,reloaded.state_dict())
        ck=np.asarray(reloaded.structured._pos('coupling').detach());cd=np.asarray(reloaded.structured._pos('damping').detach())
        if rel(ck,k)>1e-12 or rel(cd,d)>1e-12:raise AssertionError('witness reload')
        baseline=load_model(BASE);zqa=deterministic_test_z();structural=structural_qa(reloaded,candidate,zqa)
        with np.load(d93.A92,allow_pickle=False) as z:mass,D,phi=(np.asarray(z[x]) for x in ('mass','D','phi'))
        arrays.update(mass=mass,D=D,phi=phi)
        A0=d93.ad_jacobian(baseline,counters);arrays['baseline_A33']=A0;A=d93.ad_jacobian(reloaded,counters);arrays['candidate_A33']=A;expected=d93.expected_delta_A(mass,D,phi,k-k0,d-d0);arrays['expected_delta_A33']=expected
        with np.load(P83,allow_pickle=False) as p:anchor,Q=np.asarray(p['baseline_A33']),np.asarray(p['baseline_Q'])
        if rel(A0,anchor)>1e-10 or rel(A,A0+expected)>1e-10:raise AssertionError('AD identity')
        fd1=d93.finite_difference(reloaded,1e-5,deadline,counters);arrays['FD_1e_5']=fd1;fd2=d93.finite_difference(reloaded,1e-6,deadline,counters);arrays['FD_1e_6']=fd2;fdqa={str(h):float(np.linalg.norm(A-fd,'fro')/max(1.,np.linalg.norm(A,'fro'))) for h,fd in ((1e-5,fd1),(1e-6,fd2))}
        if max(fdqa.values())>1e-6 or counters['fd_rhs']!=132:raise ArithmeticError('FD')
        qb,rb,bb=(x.detach().cpu().numpy() for x in baseline.structured.matrices()+(baseline.structured.port_matrix(),));qc,rc,bc=(x.detach().cpu().numpy() for x in reloaded.structured.matrices()+(reloaded.structured.port_matrix(),));rdelta=np.zeros((33,33));rdelta[10:20,10:20]=np.diag(d-d0)
        if rel(qb,qc)>1e-10 or rel(bb,bc)>1e-10 or rel(rc-rb,rdelta)>1e-10:raise AssertionError('JRB')
        qqa=d93.quotient_qa(Q,A);A32=Q.T@A@Q;arrays.update(Q=Q,A32=A32,J_baseline=qb,J_candidate=qc,R_baseline=rb,R_candidate=rc,B_baseline=bb,B_candidate=bc);counters['eig']+=1;values,vectors,eigres=ordinary_modes(A32);arrays.update(eigenvalues=values,eigenvectors=vectors)
        if eigres>1e-10:raise ArithmeticError('eigen residual')
        alpha=float(np.max(values.real));sens=float(np.linalg.cond(vectors)*max(np.linalg.norm(Q.T@(A-fd1)@Q,2),np.linalg.norm(Q.T@(A-fd2)@Q,2),np.finfo(float).eps*np.linalg.norm(A32,2)))
        verified=json.loads(paths['verified'].read_text());targets=verified['fullorder_electromechanical_modes']
        if [int(row['index']) for row in targets]!=d93.TARGETS:raise AssertionError('verified target IDs')
        with np.load(paths['m78'],allow_pickle=False) as m:ix=np.asarray(m['target_recomputed_indices'],int);raw=np.asarray(m['descriptor_right_vectors'])[:,ix];saved=np.asarray(m['target_outputs']);ids=np.asarray(m['target_ids'],int)
        tos=[physical_output(raw[:,i],mass,omega_scale=2*np.pi*60) for i in range(13)]
        if not np.array_equal(ids,np.asarray(d93.TARGETS)) or rel(np.column_stack(tos),saved)>1e-10:raise AssertionError('modal target identity')
        matching,outputs=d93.modal_screen(Q,mass,targets,tos,values,vectors);arrays.update(target_vectors=raw,target_outputs=saved,modal_outputs=outputs)
        numerical_ok=bool(np.isfinite(sens) and np.isfinite(eigres) and max(fdqa.values())<=1e-6 and sens<.1*abs(alpha));local_ok=bool(alpha<=-.01);launch,reason=d93.decision(numerical_ok,local_ok,len(matching['matched']));np.savez_compressed(output/'arrays.npz',**arrays)
        result={'loop':'0096','test_read':False,'h5_read':False,'models_loaded':True,'metadata_seconds':METADATA_SECONDS,'source_paths':{x:str(y.resolve()) for x,y in paths.items()},'hashes_before':before,'candidate_paths':{x:str(y.resolve()) for x,y in candidate_paths.items()},'candidate_hashes':candidate_hashes,'witness_status_0095':e95['solver']['status'],'candidate_changed_state_keys':changed,'counters':counters,'structural':structural,'quotient':qqa,'fd':fdqa,'eigen_residual':eigres,'alpha':alpha,'sensitivity':sens,'numerical_ok':numerical_ok,'local_ok':local_ok,'modal':matching,'forecast_launch':launch,'forecast_reason':reason}
        if launch:
            with np.load(d93.T88,allow_pickle=False) as z:cache=d93._cache_mapping(z)
            fi=json.loads(d93.I88.read_text())
            if not fi.get('qa_pass') or fi.get('errors')!=[] or fi.get('hashes_before')!=fi.get('hashes_after') or fi['hashes_before'].get('checker')!=sha(d93.C88) or fi['hashes_before'].get('trajectories')!=sha(d93.T88) or fi['hashes_before'].get('evidence0082')!=sha(d93.E88):raise AssertionError('0088 forecast provenance')
            c=validate_cache(cache);base_score=score_pair(cache['fine_baseline'],cache['fine_baseline'],cache);e88=json.loads(d93.E88.read_text());d93._same_rows(base_score['per_event'],e88['per_event'])
            if abs(base_score['average_risk']['baseline']-.8938195532229601)>1e-10 or abs(base_score['unstable_5s']['baseline']['mean']-.6654646977500597)>1e-10 or abs(base_score['unstable_5s']['baseline']['worst4']-2.303873238746948)>1e-10:raise AssertionError('cached baseline aggregates')
            fine,coarse=forecast_candidate(reloaded,cache,deadline=deadline,output=output,progress=current,counters=counters);np.savez_compressed(output/'trajectories.npz',**cache,fine_candidate=fine,coarse_candidate=coarse)
            nums=numeric_metrics(fine,coarse,c['lengths'],c['mass'],c['p_scale']);
            if any(not np.isfinite(list(row.values())).all() or max(row.values())>.001 for row in nums):raise ArithmeticError('numerics')
            scores=score_pair(fine,cache['fine_baseline'],cache);ratios={'average':scores['average_risk']['candidate']/scores['average_risk']['baseline'],'severe_mean':scores['unstable_5s']['candidate']['mean']/scores['unstable_5s']['baseline']['mean'],'severe_worst4':scores['unstable_5s']['candidate']['worst4']/scores['unstable_5s']['baseline']['worst4']};result.update(trajectory=scores,trajectory_numerics=nums,trajectory_ratios=ratios,trajectory_gates=bool(ratios['average']<=.90 and ratios['severe_mean']<=1.05 and ratios['severe_worst4']<=1.05),cached_baseline_score=base_score)
        candidate_hashes['after_forecast']={x:sha(y) for x,y in candidate_paths.items()};d93._freeze_check(state_before,reloaded.state_dict());d93._freeze_check(state_before,load_model(candidate).state_dict())
        if rel(np.asarray(reloaded.structured._pos('coupling').detach()),k)>1e-12 or rel(np.asarray(reloaded.structured._pos('damping').detach()),d)>1e-12:raise AssertionError('in-memory witness mutation')
        if candidate_hashes['after_write']!=candidate_hashes['after_forecast']:raise AssertionError('candidate mutation')
        after={x:sha(y) for x,y in paths.items()};require(deadline,'final')
        if before!=after:raise RuntimeError('source mutation')
        result.update(candidate_hashes=candidate_hashes,hashes_after=after,counters=counters,seconds=time.perf_counter()-start,joint_gate=bool(numerical_ok and local_ok and len(matching['matched'])>=7 and result.get('trajectory_gates',False)))
        (output/'evidence.json').write_text(json.dumps(d93._json(result),indent=2,allow_nan=False));return result
    except Exception as exc:
        if arrays:np.savez_compressed(output/'arrays.npz',**arrays)
        (output/'failure_evidence.json').write_text(json.dumps({'loop':'0096','type':type(exc).__name__,'message':str(exc),'current':current,'counters':counters,'seconds':time.perf_counter()-start,'source_paths':{x:str(y) for x,y in paths.items()},'hashes_before':before,'hashes_after':{x:sha(y) for x,y in paths.items() if y.exists()},'candidate_paths':{x:str(y) for x,y in candidate_paths.items()},'candidate_hashes':candidate_hashes},indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
