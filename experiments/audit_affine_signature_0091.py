"""Cached, fixed-form frequency-affine discrepancy signature for Loop0091."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'private_data/task2_harmonic_reaction_0090'
ARRAYS=PRIOR/'arrays.npz'; EVIDENCE=PRIOR/'evidence.json'
INDEPENDENT=ROOT/'private_data/task2_harmonic_reaction_0090_independent/independent.json'
ARRAYS_SHA='a997503d4c935291ba9330328e5cc19215aa94f9b35090740bb9ebb3bb4d368c'
EVIDENCE_SHA='e94429dac4b333ec51ee267add7b574b444c2acd545a26b5b8328f45c98c6f2a'
INDEPENDENT_SHA='85ef82af4921e50ab33177ef496613bc574eeaa1eab6686c87748867b869d836'
S_DELTA=.47675314588683515; S_OMEGA=1.0033667249602924; METADATA_SECONDS=1.5360985
INDEPENDENT_CODE=ROOT/'private_data/task2_harmonic_reaction_0090_independent/check_reaction.py'

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))))
def deadline_ok(deadline):
    if time.perf_counter()>deadline:raise TimeoutError('0091 main cap')

def _check_input(TF,TG,mass,frequency):
    if TF.shape!=(13,10,10) or TG.shape!=(13,10,10) or mass.shape!=(10,) or frequency.shape!=(13,):raise ValueError('cached array shape')
    if not all(np.isfinite(x).all() for x in (TF,TG,mass,frequency)) or np.any(mass<=0) or np.any(frequency<=0):raise ValueError('cached array finite')

def normalized_discrepancy(TF,TG,mass):
    scale=S_DELTA/(mass*S_OMEGA)
    return (TF-TG)*scale[None,:,None]

def fit(Z,omega,indices):
    """The registered closed-form real U/V solution, with right-only gauge."""
    chosen=np.asarray(indices,bool)
    if Z.shape!=(13,10,10) or omega.shape!=(13,) or chosen.shape!=(13,) or chosen.sum()<1:raise ValueError('fit shape/index')
    if not(np.isfinite(Z).all() and np.isfinite(omega).all()) or np.any(omega<=0):raise ValueError('fit finite')
    one=np.ones(10); P0=np.eye(10)-np.outer(one,one)/10
    real=np.real(Z[chosen]); imag=np.imag(Z[chosen]); w=omega[chosen]
    U=np.mean(real,axis=0)@P0
    V=np.einsum('f,fij->ij',w,imag)/np.dot(w,w)
    prediction=U[None]+1j*omega[:,None,None]*V[None]
    train_residual=Z[chosen]-prediction[chosen]
    zero_objective=float(np.sum(np.abs(Z[chosen])**2)); objective=float(np.sum(np.abs(train_residual)**2))
    normal_u=(U-np.mean(real,axis=0))@P0
    normal_v=np.dot(w,w)*V-np.einsum('f,fij->ij',w,imag)
    u_scale=max(1.,float(np.max(np.abs(U))),float(np.max(np.abs(np.mean(real,axis=0)))) )
    v_left=np.dot(w,w)*V; v_right=np.einsum('f,fij->ij',w,imag)
    v_scale=max(1.,float(np.max(np.abs(v_left))),float(np.max(np.abs(v_right))))
    qa={'gauge_u1':float(np.max(np.abs(U@one))/max(1.,float(np.max(np.abs(U))))),
        'normal_u':float(np.max(np.abs(normal_u))/u_scale),
        'normal_v':float(np.max(np.abs(normal_v))/v_scale),
        'objective':objective,'zero_objective':zero_objective,
        'objective_excess':objective-zero_objective,'finite':bool(np.isfinite(U).all() and np.isfinite(V).all() and np.isfinite(prediction).all())}
    if not qa['finite'] or max(qa['gauge_u1'],qa['normal_u'],qa['normal_v'])>1e-10 or objective>zero_objective+1e-10*max(1.,zero_objective):raise ArithmeticError('closed-form fit QA')
    return U,V,prediction,qa

def _capture(Z,prediction):
    denominator=float(np.sum(np.abs(Z)**2))
    if not np.isfinite(denominator) or denominator<=1e-24:raise ArithmeticError('zero global discrepancy')
    residual=Z-prediction
    return float(1.-np.sum(np.abs(residual)**2)/denominator),np.linalg.norm(Z,axis=(1,2)),np.linalg.norm(residual,axis=(1,2))

def _prior_paths(main, independent):
    if sha(ARRAYS)!=ARRAYS_SHA or sha(EVIDENCE)!=EVIDENCE_SHA or sha(INDEPENDENT)!=INDEPENDENT_SHA:raise AssertionError('0090 anchor')
    if not main.get('completed') or main.get('solve_attempts')!=26 or main.get('svd_attempts')!=26 or len(main.get('numerical',[]))!=26 or not all(row.get('qa_pass') for row in main['numerical']):raise AssertionError('0090 main numerical QA')
    ih=independent.get('hashes_before',{})
    if not independent.get('qa_pass') or independent.get('errors')!=[] or independent.get('solve_attempts')!=26 or independent.get('svd_attempts')!=0 or independent.get('hashes_before')!=independent.get('hashes_after') or ih.get('arrays')!=ARRAYS_SHA or ih.get('evidence')!=EVIDENCE_SHA or sha(INDEPENDENT_CODE)!=ih.get('checker'):raise AssertionError('0090 independent QA')
    paths={k:Path(v) for k,v in main['source_paths'].items()}
    if main.get('hashes_before')!=main.get('hashes_after') or set(paths)!=set(main['hashes_before']):raise AssertionError('0090 source identity')
    for key,path in paths.items():
        if sha(path)!=main['hashes_before'][key]:raise AssertionError('0090 changed source '+key)
    return paths

def run(output:Path):
    if output.exists():raise FileExistsError(output)
    start=time.perf_counter(); deadline=start+60.; output.mkdir(parents=True,exist_ok=False); arrays={}; fits=[]; current={}; paths={}; before={}
    try:
        main=json.loads(EVIDENCE.read_text()); independent=json.loads(INDEPENDENT.read_text()); paths=_prior_paths(main,independent)
        paths.update(arrays90=ARRAYS,evidence90=EVIDENCE,independent90=INDEPENDENT,independent90_code=INDEPENDENT_CODE,code91=Path(__file__))
        before={k:sha(v) for k,v in paths.items()}; deadline_ok(deadline)
        with np.load(ARRAYS,allow_pickle=False) as saved:
            TF,TG,mass,frequency=(np.asarray(saved[k]) for k in ('T_F','T_G','mass','frequencies'))
        _check_input(TF,TG,mass,frequency); Z=normalized_discrepancy(TF,TG,mass); omega=2*np.pi*frequency
        if not(np.isfinite(Z).all() and np.linalg.norm(Z)>1e-12):raise ArithmeticError('global discrepancy')
        U=np.full((14,10,10),np.nan); V=np.full_like(U,np.nan); masks=np.zeros((14,13),bool)
        full_prediction=np.full_like(Z,np.nan+0j); loo_prediction=np.full_like(Z,np.nan+0j); qas=[]
        arrays={'Z':Z,'omega':omega,'mass':mass,'frequencies':frequency,'U':U,'V':V,'full_prediction':full_prediction,'loo_prediction':loo_prediction,'fit_indices':masks}
        for slot in range(14):
            deadline_ok(deadline); mask=np.ones(13,bool) if slot==0 else np.arange(13)!=(slot-1); current={'fit':slot,'held_out':None if slot==0 else slot-1}
            u,v,pred,qa=fit(Z,omega,mask); U[slot]=u; V[slot]=v; masks[slot]=mask; qas.append(qa)
            if slot==0:full_prediction[:]=pred
            else:loo_prediction[slot-1]=pred[slot-1]
            fits.append({'fit':'full' if slot==0 else f'loo_{slot-1}','held_out':None if slot==0 else slot-1,'indices':np.flatnonzero(mask).tolist(),**qa})
        full_capture,discrepancy_norm,full_residual_norm=_capture(Z,full_prediction)
        loo_capture,_,loo_residual_norm=_capture(Z,loo_prediction)
        if not(np.isfinite(full_capture) and np.isfinite(loo_capture) and np.isfinite(discrepancy_norm).all() and np.isfinite(full_residual_norm).all() and np.isfinite(loo_residual_norm).all()):raise ArithmeticError('nonfinite capture')
        np.savez_compressed(output/'arrays.npz',**arrays); deadline_ok(deadline)
        after={k:sha(v) for k,v in paths.items()}
        if before!=after:raise AssertionError('sources changed')
        deadline_ok(deadline)
        evidence={'loop':'0091','test_read':False,'models_loaded':False,'fit_count':14,'source_paths':{k:str(v.resolve()) for k,v in paths.items()},'hashes_before':before,'hashes_after':after,'metadata_seconds':METADATA_SECONDS,'seconds':time.perf_counter()-start,'fit_qa':fits,'full_capture':full_capture,'loo_capture':loo_capture,'per_frequency':[{'index':i,'frequency_hz':float(frequency[i]),'discrepancy_norm':float(discrepancy_norm[i]),'full_residual_norm':float(full_residual_norm[i]),'held_out_residual_norm':float(loo_residual_norm[i])} for i in range(13)]}
        (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)); return evidence
    except Exception as exc:
        if arrays:np.savez_compressed(output/'arrays.npz',**arrays)
        failure={'loop':'0091','type':type(exc).__name__,'message':str(exc),'current':current,'fit_count':len(fits),'fit_qa':fits,'source_paths':{k:str(v) for k,v in paths.items()},'hashes_before':before,'hashes_after':{k:sha(v) for k,v in paths.items()},'metadata_seconds':METADATA_SECONDS,'seconds':time.perf_counter()-start,'test_read':False,'models_loaded':False}
        (output/'failure_evidence.json').write_text(json.dumps(failure,indent=2,allow_nan=False)); raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
