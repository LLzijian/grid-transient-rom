"""One fixed, response-supervised backbone deployment screen for Loop0093."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, shutil, time
from pathlib import Path
for _thread_key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_thread_key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; P92=ROOT/'private_data/task2_backbone_span_0092'; A92=P92/'arrays.npz'; E92=P92/'evidence.json'; I92=ROOT/'private_data/task2_backbone_span_0092_independent/independent.json'
BASE=ROOT/'private_data/task2_structure_0078_extend_G-fixed/step3000'; P83=ROOT/'private_data/task2_coefficient_poles_0083/matrices.npz'
E88=ROOT/'private_data/task2_guarded_rollout_0088/evidence.json'; T88=ROOT/'private_data/task2_guarded_rollout_0088/trajectories.npz'
I88=ROOT/'private_data/task2_guarded_rollout_0088_independent/forecast_independent.json'; C88=ROOT/'private_data/task2_guarded_rollout_0088_independent/check_forecast.py'
C92=ROOT/'private_data/task2_backbone_span_0092_independent/check_span.py'
A92_SHA='2b4c0725339c360ea2502e8059e03b8d4f4c9c96197a58701eccd47e26983a8f'; E92_SHA='b5b9f7cc6f673fbd6dd4b5d59f6b3cb8d26361c68e4aad590141e72f3dc0a89b'; I92_SHA='29f1d15d9c797537ada125f41baee45642184d9c98e894987864c112c8fb6535'
BASE_SHA='2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b'; SDELTA=.47675314588683515; SOMEGA=1.0033667249602924; METADATA_SECONDS=20.
TARGETS=[62,64,66,68,71,73,77,79,84,91,98,100,104]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))))
def require(deadline,stage):
    if time.perf_counter()>deadline:raise TimeoutError('0093 '+stage+' cap')
def inverse_softplus(value):
    value=np.asarray(value,float)
    if not(np.isfinite(value).all() and np.all(value>0)):raise ValueError('positive amplitude')
    return np.where(value>20,value,np.log(np.expm1(value)))

def expected_delta_A(mass,D,phi,delta_k,delta_d):
    mass,D,phi,delta_k,delta_d=map(np.asarray,(mass,D,phi,delta_k,delta_d));out=np.zeros((33,33))
    out[10:20,:10]=-np.diag(1/mass)@D.T@np.diag(delta_k*np.cos(phi))@D
    out[10:20,10:20]=-np.diag(delta_d/mass)
    return out

def finite_difference(model,h,deadline,counters):
    import torch
    z=torch.zeros(33,dtype=torch.float64);columns=[]
    with torch.no_grad():
        model.begin_graph_cache()
        try:
            for j in range(33):
                require(deadline,'FD');zp=z.clone();zm=z.clone();zp[j]+=h;zm[j]-=h;counters['fd_rhs']+=1;fp=model.field(zp[None])[0];counters['fd_rhs']+=1;fm=model.field(zm[None])[0];columns.append(((fp-fm)/(2*h)).detach().cpu().numpy())
        finally:model.end_graph_cache()
    return np.column_stack(columns)

def decision(numerical_ok,local_ok,modal_count):
    """Only numerical/local failure skips forecast; modal failure remains binding later."""
    if not numerical_ok:return False,'numerical_inconclusive'
    if not local_ok:return False,'local_candidate_rejected'
    return True,'forecast_required_modal_count_'+str(modal_count)

def trajectory_summary(event_error,labels):
    event_error=np.asarray(event_error,float);labels=np.asarray(labels)
    if event_error.shape!=(3,346) or labels.shape!=(346,) or not np.isfinite(event_error).all() or np.sum(labels=='unstable')!=35:raise ValueError('trajectory metric input')
    risk=np.array([.8*np.mean(row)+.2*np.mean(np.sort(row)[-35:]) for row in event_error])
    severe=event_error[2,labels=='unstable'];return {'risks':risk,'average_risk':float(np.mean(risk)),'severe_mean_5s':float(np.mean(severe)),'severe_worst4_5s':float(np.mean(np.sort(severe)[-4:]))}

def trajectory_gates(candidate,baseline):
    c,b=trajectory_summary(**candidate),trajectory_summary(**baseline)
    ratios={'average':c['average_risk']/b['average_risk'],'severe_mean':c['severe_mean_5s']/b['severe_mean_5s'],'severe_worst4':c['severe_worst4_5s']/b['severe_worst4_5s']}
    return c,b,ratios, bool(ratios['average']<=.90 and ratios['severe_mean']<=1.05 and ratios['severe_worst4']<=1.05)

def quotient_qa(Q,A):
    Q,A=np.asarray(Q,float),np.asarray(A,float);common=np.r_[np.ones(10),np.zeros(23)];P=np.zeros((33,33));P[:10,:10]=np.eye(10)-np.ones((10,10))/10;P[10:,10:]=np.eye(23)
    qa={'qtq':rel(Q.T@Q,np.eye(32)),'qqt':rel(Q@Q.T,P),'common':float(np.max(np.abs(A@common)))}
    if max(qa.values())>1e-10:raise ArithmeticError('quotient QA')
    return qa

def modal_screen(Q,mass,targets,target_outputs,values,vectors):
    """Reuse the frozen 0089 output/assignment convention, without FOM solving."""
    from compare_physical_modes import assignment, physical_output
    matched=assignment(targets,target_outputs,values,vectors,mass,Q)
    outputs=np.column_stack([physical_output(vectors[:,i],mass,Q) for i in range(32)])
    return matched,outputs

def source_paths(main,ind):
    if sha(A92)!=A92_SHA or sha(E92)!=E92_SHA or sha(I92)!=I92_SHA:raise AssertionError('0092 anchors')
    if main.get('hashes_before')!=main.get('hashes_after') or main.get('fit_count')!=1 or main.get('svd_attempts')!=1 or main.get('rhs_evaluations')!=6:raise AssertionError('0092 main')
    if (not ind.get('qa_pass') or ind.get('errors')!=[] or ind.get('fit_count')!=1 or
        ind.get('svd_attempts')!=0 or ind.get('qr_attempts')!=1 or ind.get('rhs_evaluations')!=0 or
        ind.get('hashes_before')!=ind.get('hashes_after')):raise AssertionError('0092 independent')
    paths={k:Path(v) for k,v in main['source_paths'].items()}
    if set(paths)!=set(main['hashes_before']):raise AssertionError('0092 source map')
    for key,path in paths.items():
        if sha(path)!=main['hashes_before'][key]:raise AssertionError('changed92 '+key)
    if 'code' not in paths or 'code91' not in paths or 'code92' not in paths:raise AssertionError('inherited code key')
    ih=ind.get('hashes_before',{})
    if ih.get('arrays')!=A92_SHA or ih.get('evidence')!=E92_SHA or ih.get('checker')!=sha(C92):raise AssertionError('0092 independent links')
    paths.update(checker92=C92)
    return paths

def _unit_masks(structured):
    import torch
    for name in ('coupling_mask','damping_mask'):
        mask=getattr(structured,name)
        if not torch.equal(mask,torch.ones_like(mask)):raise AssertionError('nonunit base mask '+name)

def _apply_amplitudes(structured,k,d):
    import torch
    _unit_masks(structured)
    with torch.no_grad():
        structured.coupling.copy_(torch.as_tensor(inverse_softplus(k),dtype=structured.coupling.dtype,device=structured.coupling.device))
        structured.damping.copy_(torch.as_tensor(inverse_softplus(d),dtype=structured.damping.dtype,device=structured.damping.device))

def write_candidate(model, baseline, candidate, k, d, counters):
    import torch
    candidate.mkdir();before={key:value.detach().clone() for key,value in model.state_dict().items()}
    _apply_amplitudes(model.structured,k,d);counters['candidate_raw_writes']+=2
    changed=[key for key,value in model.state_dict().items() if not torch.equal(value,before[key])]
    allowed={'structured.coupling','structured.damping'}
    if set(changed)!=allowed:raise AssertionError('changed frozen state keys '+str(changed))
    torch.save(model.state_dict(),candidate/'model.pt');shutil.copyfile(baseline/'normalization.npz',candidate/'normalization.npz');shutil.copyfile(baseline/'checkpoint.json',candidate/'checkpoint.json');model.structured.export(candidate/'structured_numeric_export.json')
    return before,changed

def ad_jacobian(model,counters):
    import torch
    z=torch.zeros(33,dtype=torch.float64,requires_grad=True);counters['ad']+=1
    return torch.autograd.functional.jacobian(lambda x:model.field(x[None])[0],z).detach().cpu().numpy()

def _json(value):
    if isinstance(value,dict): return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json(v) for v in value]
    if isinstance(value,np.generic): return value.item()
    return value

def _freeze_check(before, after):
    allowed={'structured.coupling','structured.damping'}
    changed=[key for key,value in before.items() if not __import__('torch').equal(value,after[key])]
    if set(changed)!=allowed: raise AssertionError('frozen state keys '+str(changed))
    return changed

def _cache_mapping(data):
    # The 0088 cache also retains its historical candidate arrays.  They are
    # evidence, not the 0093 baseline, and must never collide on save.
    keys=('event_ids','labels','lengths','truth','qenc','mass','p_scale','fine_baseline','coarse_baseline','groups')
    return {key:np.asarray(data[key]) for key in keys if key in data.files}

def _same_rows(rows, reference):
    if len(rows)!=1038 or len(reference)!=1038: raise AssertionError('baseline 1038 rows')
    for a,b in zip(rows,reference,strict=True):
        if (str(a['event_id'])!=str(b['event_id']) or str(a['label'])!=str(b['label']) or
            int(a['available_steps'])!=int(b['available_steps']) or abs(float(a['horizon'])-float(b['horizon']))>0 or
            abs(float(a['baseline'])-float(b['baseline']))>1e-10):raise AssertionError('baseline per-event mismatch')

def run(output):
    """Actual invocation only; no data reader, fitting, or retry path."""
    if Path(output).exists():raise FileExistsError(output)
    os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='1'
    import torch
    torch.set_num_threads(1)
    from train_structured_rom import load_model
    from fit_coefficient_block_0082 import deterministic_test_z, structural_qa
    from compare_physical_modes import ordinary_modes, physical_output
    from backbone_forecast_0093 import forecast_candidate, numeric_metrics, score_pair, validate_cache
    output=Path(output);output.mkdir(parents=True);start=time.perf_counter();deadline=start+1200.;arrays={};current={};counters={'ad':0,'fd_rhs':0,'eig':0,'forecast_runs':0,'candidate_raw_writes':0};paths={};before={};candidate_hashes={}
    try:
        main=json.loads(E92.read_text());ind=json.loads(I92.read_text());paths=source_paths(main,ind)
        paths.update(arrays92=A92,evidence92=E92,independent92=I92,code93=Path(__file__),base_model=BASE/'model.pt',
                     fit82=ROOT/'experiments/fit_coefficient_block_0082.py',force80=ROOT/'experiments/audit_force_initialization_0080.py',
                     forecast_helper93=ROOT/'experiments/backbone_forecast_0093.py',forecast_cache88=T88,forecast_evidence88=E88,
                     forecast_independent88=I88,forecast_checker88=C88)
        before={k:sha(v) for k,v in paths.items()}
        if before['base_model']!=BASE_SHA:raise AssertionError('baseline model SHA')
        if before['fit82']!='1c680b4705a67ffa9999dc6de2651a4b8f5419d22f558469b4b8f7c06eb0fe1e' or before['force80']!='4617bfd1ea3e90468d74ba7fc10854a8a7f748da58c327fe13965554b2fbb1d9':raise AssertionError('helper SHA')
        if before['forecast_cache88']!='c466a551740404390ef967f82dd116df1543e8ee166356d348c3c9cc02ab0c5e' or before['forecast_independent88']!='67940d769feb3aa15b58127bee73cdbde75ccc8e329444e7f43d4b32688f2da5':raise AssertionError('0088 forecast anchors')
        forecast_ind=json.loads(I88.read_text())
        if not forecast_ind.get('qa_pass') or forecast_ind.get('errors')!=[] or forecast_ind.get('hashes_before')!=forecast_ind.get('hashes_after'):raise AssertionError('0088 forecast independent')
        fih=forecast_ind['hashes_before']
        if fih.get('checker')!=sha(C88) or fih.get('trajectories')!=before['forecast_cache88'] or fih.get('evidence0082')!=before['forecast_evidence88']:raise AssertionError('0088 forecast links')
        with np.load(A92,allow_pickle=False) as z:k,d,mass,D,phi=(np.asarray(z[x]) for x in ('k_plus_delta','d_plus_delta','mass','D','phi'))
        if k.shape!=(26,) or d.shape!=(10,) or not(np.isfinite(k).all() and np.isfinite(d).all() and np.all(k>0) and np.all(d>0)):raise AssertionError('0092 positive targets')
        arrays.update(k=k,d=d,mass=mass,D=D,phi=phi)
        model=load_model(BASE);_unit_masks(model.structured);k0=np.asarray(model.structured._pos('coupling').detach());d0=np.asarray(model.structured._pos('damping').detach());arrays.update(k0=k0,d0=d0)
        candidate=output/'candidate';state_before,changed=write_candidate(model,BASE,candidate,k,d,counters);candidate_paths={'deployment_candidate_model':candidate/'model.pt','deployment_candidate_meta':candidate/'checkpoint.json','deployment_candidate_norm':candidate/'normalization.npz','deployment_candidate_export':candidate/'structured_numeric_export.json'};candidate_hashes['after_write']={x:sha(y) for x,y in candidate_paths.items()}
        reloaded=load_model(candidate);_unit_masks(reloaded.structured);_freeze_check(state_before,reloaded.state_dict())
        ck=np.asarray(reloaded.structured._pos('coupling').detach());cd=np.asarray(reloaded.structured._pos('damping').detach())
        if rel(ck,k)>1e-12 or rel(cd,d)>1e-12:raise AssertionError('reload amplitude')
        baseline=load_model(BASE);zqa=deterministic_test_z();structural=structural_qa(reloaded,candidate,zqa)
        A0=ad_jacobian(baseline,counters);arrays['baseline_A33']=A0
        A=ad_jacobian(reloaded,counters);arrays['candidate_A33']=A
        expected=expected_delta_A(mass,D,phi,k-k0,d-d0);arrays['expected_delta_A33']=expected
        with np.load(P83,allow_pickle=False) as p83:anchor=np.asarray(p83['baseline_A33']);Q=np.asarray(p83['baseline_Q'])
        if rel(A0,anchor)>1e-10 or rel(A,A0+expected)>1e-10:raise AssertionError('AD backbone identity')
        fd1=finite_difference(reloaded,1e-5,deadline,counters);arrays['FD_1e_5']=fd1
        fd2=finite_difference(reloaded,1e-6,deadline,counters);arrays['FD_1e_6']=fd2
        fdqa={str(h):float(np.linalg.norm(A-fd,'fro')/max(1.,np.linalg.norm(A,'fro'))) for h,fd in ((1e-5,fd1),(1e-6,fd2))}
        if max(fdqa.values())>1e-6 or counters['fd_rhs']!=132:raise ArithmeticError('candidate FD')
        qb,rb,bb=(x.detach().cpu().numpy() for x in baseline.structured.matrices()+ (baseline.structured.port_matrix(),));qc,rc,bc=(x.detach().cpu().numpy() for x in reloaded.structured.matrices()+ (reloaded.structured.port_matrix(),));rdelta=np.zeros((33,33));rdelta[10:20,10:20]=np.diag(d-d0)
        arrays.update(Q=Q,J_baseline=qb,J_candidate=qc,R_baseline=rb,R_candidate=rc,B_baseline=bb,B_candidate=bc)
        if rel(qb,qc)>1e-10 or rel(bb,bc)>1e-10 or rel(rc-rb,rdelta)>1e-10:raise AssertionError('J/R/B deployment identity')
        qqa=quotient_qa(Q,A);A32=Q.T@A@Q;arrays['A32']=A32;counters['eig']+=1;values,vectors,eigres=ordinary_modes(A32);arrays.update(eigenvalues=values,eigenvectors=vectors)
        if eigres>1e-10:raise ArithmeticError('candidate eigen residual')
        alpha=float(np.max(values.real));sensitivity=float(np.linalg.cond(vectors)*max(np.linalg.norm(Q.T@(A-fd1)@Q,2),np.linalg.norm(Q.T@(A-fd2)@Q,2),np.finfo(float).eps*np.linalg.norm(A32,2)))
        verified=json.loads(paths['verified'].read_text());targets=verified['fullorder_electromechanical_modes']
        if [int(row['index']) for row in targets]!=TARGETS:raise AssertionError('verified target IDs')
        with np.load(paths['m78'],allow_pickle=False) as m78:
            target_ix=np.asarray(m78['target_recomputed_indices'],int);rawvec=np.asarray(m78['descriptor_right_vectors'])[:,target_ix];saved_outputs=np.asarray(m78['target_outputs']);saved_ids=np.asarray(m78['target_ids'],int)
        target_outputs=[physical_output(rawvec[:,i],mass,omega_scale=2*np.pi*60) for i in range(13)]
        if not np.array_equal(saved_ids,np.asarray(TARGETS)) or rel(np.column_stack(target_outputs),saved_outputs)>1e-10:raise AssertionError('target output identity')
        matching,outputs=modal_screen(Q,mass,targets,target_outputs,values,vectors);arrays.update(target_vectors=rawvec,target_outputs=saved_outputs,modal_outputs=outputs)
        numerical_ok=bool(np.isfinite(sensitivity) and np.isfinite(eigres) and max(fdqa.values())<=1e-6 and sensitivity<.1*abs(alpha));local_ok=bool(alpha<=-.01);launch,reason=decision(numerical_ok,local_ok,len(matching['matched']))
        np.savez_compressed(output/'arrays.npz',**arrays)
        result={'loop':'0093','test_read':False,'models_loaded':True,'metadata_seconds':METADATA_SECONDS,'source_paths':{k:str(v.resolve()) for k,v in paths.items()},'hashes_before':before,'candidate_paths':{k:str(v.resolve()) for k,v in candidate_paths.items()},'candidate_hashes':candidate_hashes,'candidate_changed_state_keys':changed,'counters':counters,'structural':_json(structural),'quotient':qqa,'fd':fdqa,'eigen_residual':eigres,'alpha':alpha,'sensitivity':sensitivity,'numerical_ok':numerical_ok,'local_ok':local_ok,'modal':_json(matching),'forecast_launch':launch,'forecast_reason':reason}
        if launch:
            current={'stage':'forecast'}
            with np.load(T88,allow_pickle=False) as cached:cache=_cache_mapping(cached)
            c=validate_cache(cache);baseline_score=score_pair(cache['fine_baseline'],cache['fine_baseline'],cache)
            eb=json.loads(E88.read_text())
            _same_rows(baseline_score['per_event'],eb['per_event'])
            if abs(baseline_score['average_risk']['baseline']-.8938195532229601)>1e-10 or abs(baseline_score['unstable_5s']['baseline']['mean']-.6654646977500597)>1e-10 or abs(baseline_score['unstable_5s']['baseline']['worst4']-2.303873238746948)>1e-10:raise AssertionError('cached baseline score')
            fine,coarse=forecast_candidate(reloaded,cache,deadline=deadline,output=output,progress=current,counters=counters)
            np.savez_compressed(output/'trajectories.npz',**cache,fine_candidate=fine,coarse_candidate=coarse)
            numerics=numeric_metrics(fine,coarse,c['lengths'],c['mass'],c['p_scale'])
            if any(not np.isfinite(list(row.values())).all() or max(row.values())>.001 for row in numerics):raise ArithmeticError('candidate integration numerics')
            scores=score_pair(fine,cache['fine_baseline'],cache);ratios={'average':scores['average_risk']['candidate']/scores['average_risk']['baseline'],'severe_mean':scores['unstable_5s']['candidate']['mean']/scores['unstable_5s']['baseline']['mean'],'severe_worst4':scores['unstable_5s']['candidate']['worst4']/scores['unstable_5s']['baseline']['worst4']}
            result.update(trajectory=scores,trajectory_numerics=numerics,trajectory_ratios=ratios,trajectory_gates=bool(ratios['average']<=.90 and ratios['severe_mean']<=1.05 and ratios['severe_worst4']<=1.05),cached_baseline_score=baseline_score)
        candidate_hashes['after_forecast']={x:sha(y) for x,y in candidate_paths.items()}
        if candidate_hashes['after_write']!=candidate_hashes['after_forecast']:raise AssertionError('candidate artifact mutation')
        _freeze_check(state_before,reloaded.state_dict());_freeze_check(state_before,load_model(candidate).state_dict())
        if rel(np.asarray(reloaded.structured._pos('coupling').detach()),k)>1e-12 or rel(np.asarray(reloaded.structured._pos('damping').detach()),d)>1e-12:raise AssertionError('in-memory candidate mutation')
        after={k:sha(v) for k,v in paths.items()};require(deadline,'final')
        if before!=after:raise RuntimeError('source changed')
        result.update(candidate_hashes=candidate_hashes,seconds=time.perf_counter()-start,counters=counters,hashes_after=after,
                      joint_gate=bool(numerical_ok and local_ok and len(matching['matched'])>=7 and result.get('trajectory_gates',False)))
        (output/'evidence.json').write_text(json.dumps(_json(result),indent=2,allow_nan=False));return result
    except Exception as exc:
        if arrays:np.savez_compressed(output/'arrays.npz',**arrays)
        after={k:sha(v) for k,v in paths.items() if v.exists()}
        (output/'failure_evidence.json').write_text(json.dumps(_json({'loop':'0093','type':type(exc).__name__,'message':str(exc),'current':current,'counters':counters,'seconds':time.perf_counter()-start,'source_paths':{k:str(v) for k,v in paths.items()},'hashes_before':before,'hashes_after':after,'candidate_hashes':candidate_hashes}),indent=2,allow_nan=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
