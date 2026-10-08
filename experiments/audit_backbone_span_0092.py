"""Cached fixed-span least-squares audit for Loop0092; no candidate is made."""
from __future__ import annotations
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; PRIOR=ROOT/'private_data/task2_affine_signature_0091'
ARRAYS=PRIOR/'arrays.npz'; EVIDENCE=PRIOR/'evidence.json'; INDEPENDENT=ROOT/'private_data/task2_affine_signature_0091_independent/independent.json'
ARRAYS_SHA='e38c779c3d80f059c184c60e7c69aebeabfdd3fc4d74aa8ce55713738271ffcb'; EVIDENCE_SHA='4e96d916630ba1a3234ded0d4c33654645aa772e8c61577fd7ffbba65e2dfc40'; INDEPENDENT_SHA='898f06fddf969a2bfe5938a4e7b90142b612cd2797103d62921f34eebf2ebce7'
EXPORT_SHA='4407f45bbafc367a912c09f5f4fd5757b8722e5418a06eb46c0f6adee60d78e9'; CORE_SHA='921b6dedb2e8b5218c5f4ec84fc6a35ac4844ebc0e310f8a6639bb0c7bcf88f3'; GRAMMAR_SHA='0325a0a1f50d7f22510cd3d7da5dbc28d8c3e186ea7288e7717f7a3a8704d28d'
S_DELTA=.47675314588683515; S_OMEGA=1.0033667249602924; METADATA_SECONDS=10.
CHECKER91=ROOT/'private_data/task2_affine_signature_0091_independent/check_affine.py'

class SpanFailure(ArithmeticError):
    def __init__(self,message,scale,singular):
        super().__init__(message);self.scale=scale;self.singular=singular

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))))
def cap(deadline):
    if time.perf_counter()>deadline:raise TimeoutError('0092 cap')

def basis(omega,mass,D,phi):
    omega=np.asarray(omega,float);mass=np.asarray(mass,float);D=np.asarray(D,float);phi=np.asarray(phi,float)
    if omega.shape!=(13,) or mass.shape!=(10,) or D.shape!=(26,10) or phi.shape!=(26,) or not all(np.isfinite(x).all() for x in (omega,mass,D,phi)) or np.any(mass<=0):raise ValueError('basis input')
    L=S_DELTA/(mass*S_OMEGA); C=np.zeros((13,10,10,36),complex)
    for e in range(26):C[:,:,:,e]=-np.cos(phi[e])*(L[:,None]*np.outer(D[e],D[e]))
    for j in range(10):C[:,j,j,26+j]=-1j*omega*L[j]
    return C

def stack(C,Z):
    if C.shape!=(13,10,10,36) or Z.shape!=(13,10,10) or not(np.isfinite(C).all() and np.isfinite(Z).all()):raise ValueError('stack input')
    return np.r_[C.real.reshape(-1,36),C.imag.reshape(-1,36)],np.r_[Z.real.ravel(),Z.imag.ravel()]

def solve_span(X,y,deadline,attempts):
    cap(deadline); scale=np.linalg.norm(X,axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale<=1e-12):raise ArithmeticError('column norm')
    xs=X/scale; attempts['svd_attempts']+=1; left,singular,right=np.linalg.svd(xs,full_matrices=False);cap(deadline)
    threshold=max(X.shape)*np.finfo(float).eps*singular[0]; rank=int(np.count_nonzero(singular>threshold)); condition=float(singular[0]/singular[-1])
    if rank!=36 or not np.isfinite(condition) or condition*np.finfo(float).eps*2600>1e-8:raise SpanFailure('rank/condition',scale,singular)
    beta=(right.T@((left.T@y)/singular))/scale; prediction=X@beta; residual=prediction-y
    lhs=X.T@residual; rhs1=X.T@y;rhs2=X.T@(X@beta); normal=float(np.max(np.abs(lhs))/max(1.,float(np.max(np.abs(rhs1))),float(np.max(np.abs(rhs2)))))
    objective=float(np.dot(residual,residual)); zero=float(np.dot(y,y))
    if not np.isfinite(zero) or zero<=1e-24:raise SpanFailure('zero target norm',scale,singular)
    capture=float(1.-objective/zero)
    if not(np.isfinite(beta).all() and np.isfinite(prediction).all() and np.isfinite(normal) and np.isfinite(capture)) or zero<=1e-24 or normal>1e-10 or objective>zero+1e-10*max(1.,zero):raise ArithmeticError('least-squares QA')
    return beta,prediction,residual,{'column_scales':scale,'singular_values':singular,'rank':rank,'condition':condition,'normal_equation':normal,'objective':objective,'zero_objective':zero,'capture':capture}

def export_qa(payload):
    D=np.asarray(payload['D'],float);mass=np.asarray(payload['mass'],float);k=np.asarray(payload['coupling'],float);d=np.asarray(payload['damping'],float);J=np.asarray(payload['J'],float);R=np.asarray(payload['R'],float);B=np.asarray(payload['B'],float);delta=np.asarray(payload['delta_star'],float);phi=D@delta
    if D.shape!=(26,10) or mass.shape!=(10,) or k.shape!=(26,) or d.shape!=(10,) or J.shape!=(33,33) or R.shape!=(33,33) or B.ndim!=2:raise ValueError('export shape')
    if not all(np.isfinite(x).all() for x in (D,mass,k,d,J,R,B,delta)) or np.any(mass<=0) or np.max(np.abs(D@np.ones(10)))>1e-12:raise ArithmeticError('export finite/gauge')
    expected=np.zeros((10,33));expected[:,10:20]=np.eye(10)
    if rel(J[:10],expected)>1e-12 or rel(J[10:20,:10],-np.eye(10))>1e-12 or np.max(np.abs(J[20:,:10]))>1e-12 or np.max(np.abs(R[:10]))>1e-12 or np.max(np.abs(R[:,:10]))>1e-12 or (B.size and np.max(np.abs(B[:10]))>1e-12):raise ArithmeticError('export canonical/extra')
    return D,mass,k,d,phi

def nonlinear_check(payload,D,mass,k,d,phi,attempts):
    from structured_rom import numpy_field
    perturbed=copy.deepcopy(payload); perturbed['coupling']=(1.01*k).tolist(); perturbed['damping']=(1.01*d).tolist(); Rp=np.asarray(perturbed['R'],float).copy();Rp[10:20,10:20]+=np.diag(.01*d);perturbed['R']=Rp.tolist()
    states=[];base=[];changed=[];expected=[]
    for t in (1.,2.,3.):
        delta=.01*t*np.sin(np.arange(1,11));omega=.01*t*np.cos(np.arange(1,11));x=np.r_[delta,mass*omega,np.zeros(13)];attempts['rhs_evaluations']+=1;f0=numpy_field(payload,x);attempts['rhs_evaluations']+=1;f1=numpy_field(perturbed,x);extra=np.zeros(33);extra[10:20]=-((.01*k*(np.sin(phi+D@delta)-np.sin(phi)))@D)-.01*d*omega
        states.append(x);base.append(f0);changed.append(f1);expected.append(extra)
    states,base,changed,expected=map(np.asarray,(states,base,changed,expected));difference=changed-base
    if not all(np.isfinite(x).all() for x in (states,base,changed,difference,expected)):raise ArithmeticError('nonfinite nonlinear export response')
    if rel(difference,expected)>1e-10:raise ArithmeticError('nonlinear export response')
    return states,base,changed,difference,expected,perturbed

def prior_paths(main,ind):
    if sha(ARRAYS)!=ARRAYS_SHA or sha(EVIDENCE)!=EVIDENCE_SHA or sha(INDEPENDENT)!=INDEPENDENT_SHA:raise AssertionError('0091 anchors')
    if main.get('hashes_before')!=main.get('hashes_after') or main.get('fit_count')!=14:raise AssertionError('0091 main')
    ih=ind.get('hashes_before',{})
    if not ind.get('qa_pass') or ind.get('errors')!=[] or ind.get('fit_count')!=14 or ind.get('hashes_before')!=ind.get('hashes_after') or ih.get('arrays')!=ARRAYS_SHA or ih.get('evidence')!=EVIDENCE_SHA or sha(CHECKER91)!=ih.get('checker'):raise AssertionError('0091 independent')
    paths={k:Path(v) for k,v in main['source_paths'].items()}
    if set(paths)!=set(main['hashes_before']):raise AssertionError('0091 source map')
    for k,p in paths.items():
        if sha(p)!=main['hashes_before'][k]:raise AssertionError('changed0091 '+k)
    p83e=json.loads(paths['p83e'].read_text())
    if p83e.get('hashes_before',{}).get('baseline_export')!=EXPORT_SHA or p83e.get('hashes_after',{}).get('baseline_export')!=EXPORT_SHA:raise AssertionError('0083 export link')
    if sha(paths['base_export'])!=EXPORT_SHA or sha(paths['old78_structured_core'])!=CORE_SHA or sha(paths['grammar'])!=GRAMMAR_SHA:raise AssertionError('export/core anchor')
    if 'code' not in paths or 'code91' not in paths:raise AssertionError('inherited code keys')
    return paths

def run(output):
    if Path(output).exists():raise FileExistsError(output)
    started=time.perf_counter();deadline=started+120.;output=Path(output);output.mkdir(parents=True,exist_ok=False);arrays={};paths={};before={};current={};attempts={'fit_attempts':0,'svd_attempts':0,'rhs_evaluations':0}
    try:
        main=json.loads(EVIDENCE.read_text());ind=json.loads(INDEPENDENT.read_text());paths=prior_paths(main,ind);paths.update(arrays91=ARRAYS,evidence91=EVIDENCE,independent91=INDEPENDENT,independent91_code=CHECKER91,core92=ROOT/'experiments/structured_rom.py',code92=Path(__file__))
        if sha(paths['core92'])!=CORE_SHA:raise AssertionError('core92 anchor')
        before={k:sha(v) for k,v in paths.items()};cap(deadline)
        with np.load(ARRAYS,allow_pickle=False) as z: Z,omega,mass,frequency=(np.asarray(z[k]) for k in ('Z','omega','mass','frequencies'))
        if Z.shape!=(13,10,10) or omega.shape!=(13,) or mass.shape!=(10,) or frequency.shape!=(13,) or rel(omega,2*np.pi*frequency)>1e-12:raise ValueError('0091 cache')
        payload=json.loads(paths['base_export'].read_text());D,mass_export,k,d,phi=export_qa(payload)
        if rel(mass,mass_export)>1e-12:raise AssertionError('mass identity')
        states,f0,f1,diff,exact,perturbed=nonlinear_check(payload,D,mass,k,d,phi,attempts);cap(deadline)
        C=basis(omega,mass,D,phi);X,y=stack(C,Z);arrays={'Z':Z,'omega':omega,'mass':mass,'frequencies':frequency,'D':D,'phi':phi,'X':X,'y':y,'states':states,'rhs_base':f0,'rhs_perturbed':f1,'rhs_difference':diff,'rhs_expected':exact,'coupling0':k,'damping0':d,'coupling_perturbed':np.asarray(perturbed['coupling']),'damping_perturbed':np.asarray(perturbed['damping'])}
        current={'stage':'svd'};attempts['fit_attempts']+=1
        try:beta,pvec,rvec,qa=solve_span(X,y,deadline,attempts)
        except SpanFailure as exc:
            arrays.update(column_scales=exc.scale,singular_values=exc.singular);raise
        prediction=(C@beta);residual=prediction-Z;arrays.update(column_scales=qa.pop('column_scales'),singular_values=qa.pop('singular_values'),beta=beta,prediction_vector=pvec,prediction=prediction,residual=residual,k_plus_delta=k+beta[:26],d_plus_delta=d+beta[26:]);np.savez_compressed(output/'arrays.npz',**arrays)
        after={k:sha(v) for k,v in paths.items()};cap(deadline)
        if before!=after:raise AssertionError('source changed')
        ev={'loop':'0092','test_read':False,'models_loaded':False,'fit_count':attempts['fit_attempts'],'svd_attempts':attempts['svd_attempts'],'rhs_evaluations':attempts['rhs_evaluations'],'metadata_seconds':METADATA_SECONDS,'seconds':time.perf_counter()-started,'source_paths':{k:str(v.resolve()) for k,v in paths.items()},'hashes_before':before,'hashes_after':after,'qa':qa,'per_frequency':[{'index':i,'frequency_hz':float(frequency[i]),'discrepancy_norm':float(np.linalg.norm(Z[i])),'residual_norm':float(np.linalg.norm(residual[i]))} for i in range(13)]}
        ev['fit_attempts']=attempts['fit_attempts']
        (output/'evidence.json').write_text(json.dumps(ev,indent=2,allow_nan=False));return ev
    except Exception as exc:
        if arrays:np.savez_compressed(output/'arrays.npz',**arrays)
        (output/'failure_evidence.json').write_text(json.dumps({'loop':'0092','type':type(exc).__name__,'message':str(exc),'current':current,**attempts,'metadata_seconds':METADATA_SECONDS,'seconds':time.perf_counter()-started,'source_paths':{k:str(v) for k,v in paths.items()},'hashes_before':before,'hashes_after':{k:sha(v) for k,v in paths.items()},'test_read':False,'models_loaded':False},indent=2,allow_nan=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
