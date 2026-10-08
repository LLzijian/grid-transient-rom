"""Frozen complex-direction balance diagnostic at the 13 cached FOM poles."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import numpy as np
from harmonic_solve_0090 import solve_scaled

ROOT=Path(__file__).resolve().parents[1];P96=ROOT/'private_data/task2_witness_deployment_0096';A96=P96/'arrays.npz';E96=P96/'evidence.json';I96=ROOT/'private_data/task2_witness_deployment_0096_independent/independent.json';C96=ROOT/'private_data/task2_witness_deployment_0096_independent/check_witness.py'
P93=ROOT/'private_data/task2_backbone_deployment_0093';A93=P93/'arrays.npz';P90=ROOT/'private_data/task2_harmonic_reaction_0090';A90=P90/'arrays.npz';E90=P90/'evidence.json'
A96_SHA='ad90932963116b973753305aba7db2ca0149168cb2308088d10db49fe215d783';E96_SHA='04e8b5700b6271e983436f36f707c2b2240f41e114aca8d1084b720eace057c0';I96_SHA='73190689f5d6c661520c9e13ceca06f1383eeff866ccf57d98c6420e56f7d290';TARGETS=[62,64,66,68,71,73,77,79,84,91,98,100,104];OMEGA=2*np.pi*60.;SDELTA=.47675314588683515;SOMEGA=1.0033667249602924;METADATA_SECONDS=10.

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))))
def require(deadline,stage):
    if time.perf_counter()>deadline:raise TimeoutError('0097 '+stage+' cap')

def physical_output(v,mass):
    delta,omega=np.asarray(v[:10]),np.asarray(v[10:20]);coi=delta-mass@delta/mass.sum()
    return np.r_[coi/SDELTA,OMEGA*omega/SOMEGA]

def normalize_target(v,mass):
    v=np.asarray(v,np.complex128);out=physical_output(v,mass);norm=float(np.linalg.norm(out))
    if not np.isfinite(norm) or norm<=0:raise ValueError('zero physical output')
    return v/norm,norm

def controls(AF,EF,A,mass,lam,v):
    v=np.asarray(v);delta,omega=v[:10],v[10:20]
    ek=np.asarray(EF)[10:20]
    kd=float(np.linalg.norm(lam*delta)+np.linalg.norm(OMEGA*omega))
    gd=float(np.linalg.norm(lam*ek*omega)+np.linalg.norm(AF[10:20]@v))
    ed=float((np.linalg.norm(AF,'fro')+abs(lam)*np.max(np.abs(EF)))*np.linalg.norm(v))
    if not np.isfinite([kd,gd,ed]).all() or min(kd,gd)<=1e-12 or ed<=0:
        raise ArithmeticError('control denominator')
    kin=float(np.linalg.norm(lam*delta-OMEGA*omega)/kd)
    gen=float(np.linalg.norm(lam*ek*omega-AF[10:20]@v)/gd)
    common=np.r_[np.ones(10),np.zeros(23)];romdelta=rel(A[:10],np.c_[np.zeros((10,10)),np.eye(10),np.zeros((10,13))]);gauge=rel(A@common,np.zeros(33));expected=np.zeros((10,699));expected[:,10:20]=OMEGA*np.eye(10);fomdelta=rel(AF[:10],expected);fomEdelta=rel(EF[:10],np.ones(10));fomEomega=rel(ek,mass*OMEGA)
    backward=float(np.linalg.norm(AF@v-lam*np.asarray(EF)*v)/ed)
    if not np.isfinite([kin,gen,backward,romdelta,gauge,fomdelta,fomEdelta,fomEomega]).all() or min(np.linalg.norm(lam*delta)+np.linalg.norm(OMEGA*omega),np.linalg.norm(lam*ek*omega)+np.linalg.norm(AF[10:20]@v))<=1e-12:raise ArithmeticError('control denominator')
    if backward>1e-8 or max(kin,gen)>1e-6 or max(romdelta,gauge,fomdelta,fomEdelta,fomEomega)>1e-10:raise ArithmeticError('unit/kinematic control')
    return {'fom_backward':backward,'fom_kinematic':kin,'fom_generator':gen,'rom_delta_rows':romdelta,'rom_common_angle':gauge,'fom_delta_rows':fomdelta,'fom_Edelta':fomEdelta,'fom_Eomega':fomEomega}

def complex_balance(A,mass,lam,v,deadline):
    """One 13x13 hidden solve and the registered complex reaction metric."""
    A=np.asarray(A,float);mass=np.asarray(mass,float);v=np.asarray(v,np.complex128);H=lam*np.eye(13)-A[20:,20:];B=A[20:,:10]+lam*A[20:,10:20]
    solved=solve_scaled(H,B,deadline)
    if not solved['qa_pass']:return {'solve':solved,'qa_pass':False}
    X=solved['X'];T=np.diag(mass)@(A[10:20,:10]+lam*A[10:20,10:20]+A[10:20,20:]@X);R=np.diag(mass)*(lam**2)-T;L=np.diag(SDELTA/(mass*SOMEGA));K=L@R;a=v[:10]/SDELTA;ka=K@a;lt=L@T@a;ld=L@(np.diag(mass)*(lam**2))@a;den1,den2=float(np.linalg.norm(ld)),float(np.linalg.norm(lt));den=den1+den2
    if not(np.isfinite(K).all() and np.isfinite(T).all() and np.isfinite(R).all() and np.isfinite(den) and den>1e-12):raise ArithmeticError('complex balance denominator')
    singular=np.linalg.svd(K,compute_uv=False);condition=None if singular[-1]==0 else float(singular[0]/singular[-1]);eta=float(np.linalg.norm(ka)/den)
    return {'qa_pass':True,'H':H,'B':B,'X':X,'T':T,'R':R,'K':K,'inertia_vector':L@(np.diag(mass)*(lam**2))@a,'force_vector':L@T@a,'defect_vector':ka,'reaction_singular_values':singular,'reaction_condition':condition,'reaction_singular':bool(singular[-1]==0),'eta':eta,'numerator':float(np.linalg.norm(ka)),'den_lambda2':den1,'den_torque':den2,'solve':solved}

def _paths(e96,i96):
    if any(sha(p)!=h for p,h in ((A96,A96_SHA),(E96,E96_SHA),(I96,I96_SHA))):raise AssertionError('0096 anchors')
    if not i96.get('qa_pass') or i96.get('errors')!=[] or i96.get('hashes_before')!=i96.get('hashes_after') or i96['hashes_before'].get('arrays')!=A96_SHA or i96['hashes_before'].get('evidence')!=E96_SHA or i96['hashes_before'].get('checker')!=sha(C96):raise AssertionError('0096 independent')
    if e96.get('hashes_before')!=e96.get('hashes_after'):raise AssertionError('0096 source mutation')
    cpaths={k:Path(v) for k,v in e96.get('candidate_paths',{}).items()};ch=e96.get('candidate_hashes',{}).get('after_forecast',{})
    if set(cpaths)!=set(ch) or any(sha(v)!=ch[k] for k,v in cpaths.items()) or e96.get('candidate_hashes',{}).get('after_write')!=ch:raise AssertionError('0096 candidate artifacts')
    paths={k:Path(v) for k,v in e96['source_paths'].items()};paths.update({f'candidate96_{k}':v for k,v in cpaths.items()});paths.update(arrays96=A96,evidence96=E96,independent96=I96,checker96=C96,arrays93=A93,arrays90=A90,evidence90=E90,code97=Path(__file__),solver90=ROOT/'experiments/harmonic_solve_0090.py')
    before={k:sha(v) for k,v in paths.items()}
    for k,v in e96['hashes_before'].items():
        if k in before and before[k]!=v:raise AssertionError('changed inherited '+k)
    return paths,before

def run(output):
    if Path(output).exists():raise FileExistsError(output)
    output=Path(output);output.mkdir(parents=True);start=time.perf_counter();deadline=start+120.;paths={};before={};arrays={};rows=[];current={};counters={'hidden_solves':0,'hidden_svds':0,'reaction_svds':0}
    try:
        e96=json.loads(E96.read_text());i96=json.loads(I96.read_text());paths,before=_paths(e96,i96)
        with np.load(A90,allow_pickle=False) as z:AF,EF,A90g,mass90=(np.asarray(z[k]) for k in ('A_F','E_F','A_G','mass'))
        with np.load(A96,allow_pickle=False) as z:base,A96c,mass=(np.asarray(z[k]) for k in ('baseline_A33','candidate_A33','mass'))
        with np.load(A93,allow_pickle=False) as z:A93c,A93base,mass93=(np.asarray(z[k]) for k in ('candidate_A33','baseline_A33','mass'))
        with np.load(paths['m78'],allow_pickle=False) as z:ix=np.asarray(z['target_recomputed_indices'],int);raw=np.asarray(z['descriptor_right_vectors'])[:,ix];lam=np.asarray(z['descriptor_eigenvalues'])[ix];ids=np.asarray(z['target_ids'],int);saved=np.asarray(z['target_outputs'])
        if not(np.array_equal(ids,TARGETS) and AF.shape==(699,699) and EF.shape==(699,) and raw.shape==(699,13) and lam.shape==(13,) and saved.shape==(20,13) and base.shape==(33,33) and A93c.shape==(33,33) and A93base.shape==(33,33) and A96c.shape==(33,33) and mass.shape==(10,) and all(np.isfinite(x).all() for x in (AF,EF,A90g,mass90,base,A93c,A93base,A96c,mass,mass93,raw,lam,saved)) and np.all(mass>0) and rel(mass,mass90)<=1e-10 and rel(mass,mass93)<=1e-10 and rel(base,A90g)<=1e-10 and rel(base,A93base)<=1e-10):raise AssertionError('cached dimensions/identities')
        vectors=np.empty_like(raw);outnorm=np.empty(13)
        for i in range(13):
            if rel(physical_output(raw[:,i],mass),saved[:,i])>1e-10:raise AssertionError('target output')
            vectors[:,i],outnorm[i]=normalize_target(raw[:,i],mass)
            if not np.isfinite(vectors[:,i]).all():raise FloatingPointError('normalized target')
        arms={'baseline':base,'failed0093':A93c,'witness0096':A96c};arrays.update(A_F=AF,E_F=EF,baseline_A33=base,failed0093_A33=A93c,witness0096_A33=A96c,mass=mass,lambda_values=lam,raw_vectors=raw,normalized_vectors=vectors,output_norms=outnorm,target_ids=ids)
        for name in arms:arrays.update({f'{name}_H':np.full((13,13,13),np.nan+0j),f'{name}_B':np.full((13,13,10),np.nan+0j),f'{name}_X':np.full((13,13,10),np.nan+0j),f'{name}_Y':np.full((13,13,10),np.nan+0j),f'{name}_equilibrated_H':np.full((13,13,13),np.nan+0j),f'{name}_equilibrated_B':np.full((13,13,10),np.nan+0j),f'{name}_row_scale':np.full((13,13),np.nan),f'{name}_column_scale':np.full((13,13),np.nan),f'{name}_T':np.full((13,10,10),np.nan+0j),f'{name}_R':np.full((13,10,10),np.nan+0j),f'{name}_K':np.full((13,10,10),np.nan+0j),f'{name}_inertia_vector':np.full((13,10),np.nan+0j),f'{name}_force_vector':np.full((13,10),np.nan+0j),f'{name}_defect_vector':np.full((13,10),np.nan+0j),f'{name}_hidden_singular_values':np.full((13,13),np.nan),f'{name}_reaction_singular_values':np.full((13,10),np.nan)})
        np.savez_compressed(output/'arrays.npz',**arrays)
        for arm,A in arms.items():
            for i in range(13):
                require(deadline,'target start');current={'arm':arm,'target':int(ids[i])};ctrl=controls(AF,EF,A,mass,lam[i],vectors[:,i]);result=complex_balance(A,mass,lam[i],vectors[:,i],deadline);counters['hidden_solves']+=result['solve']['solve_attempts'];counters['hidden_svds']+=result['solve']['svd_attempts']
                for key in ('H','B','X','Y','equilibrated_H','equilibrated_B','row_scale','column_scale','singular_values'):
                    if result['solve'].get(key) is not None:arrays[f'{arm}_{"hidden_singular_values" if key=="singular_values" else key}'][i]=result['solve'][key]
                if not result['qa_pass']:
                    np.savez_compressed(output/'arrays.npz',**arrays);raise ArithmeticError('hidden solve '+str(result['solve']['error']))
                counters['reaction_svds']+=1
                for key in ('H','B','X','T','R','K','inertia_vector','force_vector','defect_vector','reaction_singular_values'):arrays[f'{arm}_{key}'][i]=result[key]
                arrays[f'{arm}_hidden_singular_values'][i]=result['solve']['singular_values'];rows.append({'arm':arm,'target_id':int(ids[i]),**ctrl,'eta':result['eta'],'numerator':result['numerator'],'den_lambda2':result['den_lambda2'],'den_torque':result['den_torque'],'reaction_condition':result['reaction_condition'],'reaction_singular':result['reaction_singular'],**result['solve']['metrics']})
                np.savez_compressed(output/'arrays.npz',**arrays);require(deadline,'target loop')
        if (counters['hidden_solves'],counters['hidden_svds'],counters['reaction_svds'])!=(39,39,39):raise AssertionError('solve/SVD counts')
        after={k:sha(v) for k,v in paths.items()};require(deadline,'final')
        if before!=after:raise RuntimeError('source mutation')
        evidence={'loop':'0097','test_read':False,'models_loaded':False,'metadata_seconds':METADATA_SECONDS,'seconds':time.perf_counter()-start,'source_paths':{k:str(v.resolve()) for k,v in paths.items()},'hashes_before':before,'hashes_after':after,'counters':counters,'per_target':rows}
        (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False));return evidence
    except Exception as exc:
        if arrays:np.savez_compressed(output/'arrays.npz',**arrays)
        (output/'failure_evidence.json').write_text(json.dumps({'loop':'0097','type':type(exc).__name__,'message':str(exc),'current':current,'counters':counters,'seconds':time.perf_counter()-start,'source_paths':{k:str(v) for k,v in paths.items()},'hashes_before':before,'hashes_after':{k:sha(v) for k,v in paths.items() if v.exists()},'per_target':rows},indent=2));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
