"""Frozen physical harmonic reaction from the original descriptor; no index reduction."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from compare_physical_modes import sha, S_DELTA, S_OMEGA, OMEGA_S
from harmonic_solve_0090 import solve_scaled

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'private_data/task2_structured_modes_0089/evidence.json'
INDEPENDENT = ROOT/'private_data/task2_structured_modes_0089_independent/independent.json'
PRIOR_SHA = '3bb21720fd768a01172b1a74bcd7faf3f65c496456c004648f29b16b31e9d9de'
INDEPENDENT_SHA = 'e6d36d063bed926b648588c5228d87d16dd49becadc4f5e0d8aaa831fe59d8a0'
TARGETS = [62,64,66,68,71,73,77,79,84,91,98,100,104]
METADATA_SECONDS = 3.6526071


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a-b))/max(1., np.max(np.abs(a)), np.max(np.abs(b))))


def source_paths():
    if sha(PRIOR) != PRIOR_SHA or sha(INDEPENDENT) != INDEPENDENT_SHA:
        raise AssertionError('0089 evidence anchors')
    prior = json.loads(PRIOR.read_text())
    check = json.loads(INDEPENDENT.read_text())
    if not check.get('qa_pass') or check.get('errors') != [] or check['hashes_before'] != check['hashes_after'] or check['hashes_before']['evidence'] != PRIOR_SHA:
        raise AssertionError('inherited0089 independent QA')
    paths = {k: Path(v) for k,v in prior['source_paths'].items()}
    if prior['source_sha256_before'] != prior['source_sha256_after'] or set(paths) != set(prior['source_sha256_before']):
        raise AssertionError('0089 source identities')
    for key, path in paths.items():
        if sha(path) != prior['source_sha256_before'][key]: raise AssertionError('0089 changed source '+key)
    paths.update(prior89=PRIOR, independent89=INDEPENDENT,
                 independent89_code=ROOT/'private_data/task2_structured_modes_0089_independent/check_modal.py',
                 code90=Path(__file__), solver90=Path(__file__).with_name('harmonic_solve_0090.py'))
    if sha(paths['independent89_code']) != check['hashes_before']['checker']:
        raise AssertionError('0089 checker source')
    return paths


def assemble_inputs(paths):
    with np.load(paths['descriptor'], allow_pickle=False) as z:
        A = np.block([[z['fx'],z['fy']],[z['gx'],z['gy']]])
        E = np.r_[z['time_constants'],np.zeros(479)]
        names = [str(v) for v in z['state_names'][:20]]
    with np.load(paths['p83'], allow_pickle=False) as z:
        G = z['baseline_A33']
    with np.load(paths['m3'], allow_pickle=False) as z:
        mass = z['mass']
    expected_names = [f'delta GENROU {i}' for i in range(1,11)]+[f'omega GENROU {i}' for i in range(1,11)]
    if names != expected_names or A.shape != (699,699) or E.shape != (699,) or G.shape != (33,33) or mass.shape != (10,):
        raise AssertionError('physical shape/order')
    if not all(np.isfinite(v).all() for v in (A,E,G,mass)) or not np.all(mass>0) or np.any(E<0):
        raise AssertionError('physical finite/mass')
    expected_delta = np.zeros((10,699)); expected_delta[:,10:20] = OMEGA_S*np.eye(10)
    g_delta = np.zeros((10,33)); g_delta[:,10:20] = np.eye(10)
    qa = {'fom_delta_row':relative(A[:10],expected_delta), 'fom_delta_mass':relative(E[:10],np.ones(10)),
          'fom_omega_mass':relative(E[10:20],mass*OMEGA_S), 'rom_delta_row':relative(G[:10],g_delta),
          'rom_common_angle':relative(G@np.r_[np.ones(10),np.zeros(23)],np.zeros(33))}
    if any(v>1e-10 for v in qa.values()): raise AssertionError('physical units/kinematics '+str(qa))
    verified = json.loads(paths['verified'].read_text())
    identity = json.loads(paths['old78_descriptor_evidence'].read_text())
    if any(identity[k] != verified['fullorder_identity'][k] for k in ('case_sha256','physical_signature','equilibrium_state_error','equilibrium_algebraic_error')):
        raise AssertionError('FOM equilibrium/signature')
    targets = verified['fullorder_electromechanical_modes']
    if [int(t['index']) for t in targets] != TARGETS: raise AssertionError('target list')
    frequencies = np.array([t['frequency_hz'] for t in targets],float)
    if not np.isfinite(frequencies).all() or np.any(frequencies<=0): raise AssertionError('frequency grid')
    qa.update(nonzero_time_constants=int(np.count_nonzero(E)), zero_time_constants=int(np.count_nonzero(E==0)))
    return A,E,G,mass,frequencies,qa


def harmonic_system(A, E, s, omega_factor):
    """Hidden descriptor system for delta=I, native omega=s/Omega I."""
    H = s*np.diag(E[20:])-A[20:,20:]
    B = A[20:,:10]+(s/omega_factor)*A[20:,10:20]
    return H, B


def reconstruct(A, E, mass, s, X, omega_factor, fom):
    full = np.vstack((np.eye(10),s/omega_factor*np.eye(10),X))
    endogenous = A[10:20]@full
    T = endogenous if fom else mass[:,None]*endogenous
    R = mass[:,None]*s*s*np.eye(10)-T
    expected = np.zeros_like(full)
    expected[10:20] = R if fom else R/mass[:,None]
    left, right = s*E[:,None]*full, A@full
    residue = left-right-expected
    error = float(np.max(np.abs(residue))/max(1.,np.max(np.abs(left)),np.max(np.abs(right)),np.max(np.abs(expected))))
    if not all(np.isfinite(v).all() for v in (full,T,R)) or not np.isfinite(error) or error>1e-10:
        raise AssertionError('original descriptor reconstruction')
    return T,R,error


def metric_report(TF,TG,RF,RG,mass,frequencies):
    scale = S_DELTA/(mass*S_OMEGA)
    KF,KG = RF*scale[None,:,None],RG*scale[None,:,None]
    NF,NG = TF*scale[None,:,None],TG*scale[None,:,None]
    normf = np.linalg.norm(KF,axis=(1,2)); normt = np.linalg.norm(NF,axis=(1,2))
    if np.any(normf<=1e-12) or np.linalg.norm(KF)<=1e-12: raise AssertionError('reaction denominator unresolved')
    w = mass/mass.sum(); P = np.eye(10)-np.ones((10,1))*w[None,:]
    rows = []
    for i,f in enumerate(frequencies):
        diff = KG[i]-KF[i]
        rows.append({'index':i,'frequency_hz':float(f),
                     'relative_reaction_error':float(np.linalg.norm(diff)/normf[i]),
                     'relative_torque_error':float(np.linalg.norm(NG[i]-NF[i])/normt[i]) if normt[i]>1e-12 else None,
                     'coi_error_absolute':float(np.linalg.norm(P@diff)),
                     'common_error_absolute':float(np.linalg.norm(w@diff))})
    aggregate = float(np.linalg.norm(KG-KF)/np.linalg.norm(KF))
    torque = float(np.linalg.norm(NG-NF)/np.linalg.norm(NF)) if np.linalg.norm(NF)>1e-12 else None
    return KF,KG,rows,aggregate,torque


def run(output):
    if output.exists(): raise FileExistsError(output)
    started = time.perf_counter(); deadline = started+600.; paths={}; before={}; arrays={}; numerical=[]; current={}; solves=0; svds=0
    output.mkdir(parents=True,exist_ok=False)
    try:
        paths=source_paths(); before={k:sha(p) for k,p in paths.items()}
        A,E,G,mass,frequencies,qa=assemble_inputs(paths)
        arrays.update(A_F=A,E_F=E,A_G=G,mass=mass,frequencies=frequencies)
        for name,n in (('F',679),('G',13)):
            arrays['X_'+name]=np.full((13,n,10),np.nan+0j)
            for kind in ('T','R'): arrays[kind+'_'+name]=np.full((13,10,10),np.nan+0j)
        np.savez_compressed(output/'arrays.npz',**arrays)
        for i,f in enumerate(frequencies):
            s=2j*np.pi*f
            for name,mat,diag,omega in (('F',A,E,OMEGA_S),('G',G,np.ones(33),1.)):
                current={'index':i,'system':name,'frequency_hz':float(f)}
                H,B=harmonic_system(mat,diag,s,omega)
                solved=solve_scaled(H,B,deadline)
                solves+=solved['solve_attempts']; svds+=solved['svd_attempts']
                row={**current,**solved['metrics'],'qa_pass':solved['qa_pass'],'error':solved['error'],
                     'solve_attempts':solved['solve_attempts'],'svd_attempts':solved['svd_attempts']}
                numerical.append(row)
                for key in ('X','Y','row_scale','column_scale','singular_values'):
                    if solved.get(key) is not None: arrays[f'{name}_{i}_{key}']=solved[key]
                if not solved['qa_pass']:
                    np.savez_compressed(output/'failed_solve.npz',H=H,B=B,**{k:solved[k] for k in ('X','Y','row_scale','column_scale','singular_values') if solved.get(k) is not None})
                    raise ArithmeticError('clamped solve QA: '+str(solved['error']))
                arrays['X_'+name][i]=solved['X']
                T,R,residual=reconstruct(mat,diag,mass,s,solved['X'],omega,name=='F')
                row['equation_residual']=residual
                arrays['T_'+name][i]=T; arrays['R_'+name][i]=R
            if time.perf_counter()>deadline: raise TimeoutError('0090 main cap')
            print(f'0090 frequency {i+1}/13 complete',flush=True)
        KF,KG,rows,aggregate,torque=metric_report(arrays['T_F'],arrays['T_G'],arrays['R_F'],arrays['R_G'],mass,frequencies)
        arrays.update(K_F=KF,K_G=KG)
        np.savez_compressed(output/'arrays.npz',**arrays)
        after={k:sha(p) for k,p in paths.items()}
        if before!=after: raise AssertionError('sources changed')
        if time.perf_counter()>deadline: raise TimeoutError('0090 final main cap')
        evidence={'loop':'0090','completed':True,'test_read':False,'models_loaded':False,
                  'source_paths':{k:str(p.resolve()) for k,p in paths.items()},'hashes_before':before,'hashes_after':after,
                  'seconds':time.perf_counter()-started,'metadata_seconds':METADATA_SECONDS,'solve_attempts':solves,'svd_attempts':svds,
                  'frequencies':frequencies.tolist(),'per_frequency':rows,'aggregate_reaction_error':aggregate,
                  'aggregate_torque_error':torque,'numerical':numerical,'qa':qa}
        (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)); return evidence
    except Exception as exc:
        if arrays: np.savez_compressed(output/'arrays.npz',**arrays)
        failure={'loop':'0090','completed':False,'type':type(exc).__name__,'message':str(exc),'current':current,
                 'numerical':numerical,'solve_attempts':solves,'svd_attempts':svds,'seconds':time.perf_counter()-started,
                 'metadata_seconds':METADATA_SECONDS,'hashes_before':before,'hashes_after':{k:sha(p) for k,p in paths.items()},
                 'test_read':False,'models_loaded':False}
        (output/'failure_evidence.json').write_text(json.dumps(failure,indent=2,allow_nan=False)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); run(parser.parse_args().output)
