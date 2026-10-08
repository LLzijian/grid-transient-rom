"""Fixed-path conditional speed-balance diagnostic for the rejected 0093 correction."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P93=ROOT/'private_data/task2_backbone_deployment_0093'; E93=P93/'evidence.json'; A93=P93/'arrays.npz'; I93=ROOT/'private_data/task2_backbone_deployment_0093_independent/independent.json'
C93=ROOT/'private_data/task2_backbone_deployment_0093_independent/check_deployment.py'
T88=ROOT/'private_data/task2_guarded_rollout_0088/trajectories.npz'; BASE=ROOT/'private_data/task2_structure_0078_extend_G-fixed/step3000'
E93_SHA='ec4ab06c662dadfd0aff6dfc71c23c8db11d27af0375d0560789965665ae107f'; A93_SHA='82659802566986e76bcb5cdd11b289f5adc3110ae0895790c5c586d5c97d8f51'; I93_SHA='99bd9c74d71dd0dafa727018cf32bc8f99b91a37d03b9479fc72b0b8d28c843d'
CANDIDATE_SHA='8f699a7524c71529227f01a2876e9d4ec4633cea095dad3edfc93914ca33d46e'; BASE_SHA='2667480a75d7672ad49ed66c79731e754676bae52574229dbb5f98c581ac890b'; T88_SHA='c466a551740404390ef967f82dd116df1543e8ee166356d348c3c9cc02ab0c5e'; METADATA_SECONDS=20.

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(deadline,stage):
    if time.perf_counter()>deadline: raise TimeoutError('0094 '+stage+' cap')
def rel(a,b): return float(np.max(np.abs(np.asarray(a)-np.asarray(b)))/max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))))

def physical_at(truth,i,t):
    """True physical path, linearly interpolated only inside its genuine prefix."""
    x=np.asarray(truth[i]); u=float(t)/.01; lo=int(np.floor(u+1e-12)); frac=u-lo
    if lo<0 or lo>=len(x): raise ValueError('physical interpolation extrapolation')
    if abs(frac)<1e-12:return x[lo]
    if lo+1>=len(x): raise ValueError('physical endpoint extrapolation')
    return (1-frac)*x[lo]+frac*x[lo+1]

def known_terms(x,mass,D,phi,dk,dd):
    delta,omega=np.asarray(x[:10]),np.asarray(x[10:20]); phase=phi+D@delta
    fk=-(D.T@(dk*(np.sin(phase)-np.sin(phi))))/mass
    fd=-dd*omega/mass
    if not np.isfinite(fk).all() or not np.isfinite(fd).all():raise FloatingPointError('nonfinite known increment')
    return fk,fd

def packed_rhs(model,x,y,mass,D,phi,dk,dd):
    import torch
    if not np.isfinite(y).all(): raise FloatingPointError('nonfinite packed state')
    if not np.isfinite(x).all(): raise FloatingPointError('nonfinite physical input')
    state=np.c_[x,y[:,:3]]
    with torch.no_grad(): f=np.asarray(model.field(torch.as_tensor(state,dtype=torch.float64)).detach().cpu())
    if not np.isfinite(f).all(): raise FloatingPointError('nonfinite model field')
    delta,omega=x[:,:10],x[:,10:20]; phase=phi[None]+delta@D.T
    fk=-(dk[None]*(np.sin(phase)-np.sin(phi)[None]))@D/mass[None]
    fd=-dd[None]*omega/mass[None]
    out=np.c_[f[:,30:33],f[:,10:20],fk,fd]
    if not np.isfinite(out).all(): raise FloatingPointError('nonfinite RHS')
    return out

def _physical_batch(truth,active,node,fraction):
    x=np.asarray(truth[active,node],float)
    if fraction:
        x=(1-fraction)*x+fraction*np.asarray(truth[active,node+1],float)
    if not np.isfinite(x).all():raise FloatingPointError('nonfinite genuine physical batch')
    return x

def replay(model,truth,qenc,lengths,mass,D,phi,dk,dd,dt,deadline,progress,partial):
    """One packed RK4 path; NaN tail preserves each event's genuine endpoint."""
    ratio=2 if dt==.005 else 1;steps=ratio*500;out=np.full((346,steps+1,33),np.nan);y=np.c_[qenc,np.zeros((346,30))];out[:,0]=y
    model.begin_graph_cache()
    try:
        for j in range(steps):
            active=np.flatnonzero(np.asarray(lengths,int)*ratio>j)
            if not len(active):break
            progress.update(stage='replay',step=int(j),active=int(len(active)),dt=float(dt));require(deadline,'RK4')
            def stage_input(stage):
                u=(j+stage)/ratio; node=int(np.floor(u)); return _physical_batch(truth,active,node,u-node)
            x1,x2,x4=stage_input(0.),stage_input(.5),stage_input(1.)
            ya=y[active];k1=packed_rhs(model,x1,ya,mass,D,phi,dk,dd);k2=packed_rhs(model,x2,ya+dt*k1/2,mass,D,phi,dk,dd);k3=packed_rhs(model,x2,ya+dt*k2/2,mass,D,phi,dk,dd);k4=packed_rhs(model,x4,ya+dt*k3,mass,D,phi,dk,dd)
            yn=ya+dt*(k1+2*k2+2*k3+k4)/6
            if not np.isfinite(yn).all():raise FloatingPointError('nonfinite RK4 state')
            y[active]=yn;out[active,j+1]=yn
    except Exception:
        np.savez_compressed(partial,packed=out)
        raise
    finally:model.end_graph_cache()
    return out

def expected_field_difference(x,mass,D,phi,dk,dd):
    fk,fd=known_terms(x,mass,D,phi,dk,dd);out=np.zeros(33);out[10:20]=fk+fd;return out

def probe_qa(base_fields,candidate_fields,expected):
    base_fields,candidate_fields,expected=map(np.asarray,(base_fields,candidate_fields,expected))
    if not(np.isfinite(base_fields).all() and np.isfinite(candidate_fields).all() and np.isfinite(expected).all()):raise FloatingPointError('nonfinite fixed-state probe')
    return rel(candidate_fields-base_fields,expected)

def _cache(data):
    keys=('event_ids','labels','lengths','truth','qenc','mass','p_scale')
    return {k:np.asarray(data[k]) for k in keys}

def validate_cache(cache):
    ids,labs,n=np.asarray(cache['event_ids']),np.asarray(cache['labels']),np.asarray(cache['lengths'])
    if ids.shape!=(346,) or len(set(ids.astype(str)))!=346 or labs.shape!=(346,) or n.shape!=(346,):raise AssertionError('event identities')
    if np.sum(labs.astype(str)=='stable')!=311 or np.sum(labs.astype(str)=='unstable')!=35:raise AssertionError('label counts')
    if not np.equal(n,n.astype(int)).all() or np.any(n<1) or np.any(n>500):raise AssertionError('genuine lengths')
    if cache['truth'].shape!=(346,501,30) or cache['qenc'].shape!=(346,3) or cache['mass'].shape!=(10,) or not(np.isfinite(cache['qenc']).all() and np.isfinite(cache['mass']).all() and np.all(cache['mass']>0)):raise AssertionError('conditional cache shapes')
    for i,end in enumerate(n.astype(int)+1):
        if not np.isfinite(cache['truth'][i,:end]).all():raise FloatingPointError('nonfinite genuine truth')
        if end<501 and np.isfinite(cache['truth'][i,end:]).any():raise AssertionError('finite truth padding')
    return n.astype(int)

def _source_paths(e93,i93):
    if sha(E93)!=E93_SHA or sha(A93)!=A93_SHA or sha(I93)!=I93_SHA or sha(T88)!=T88_SHA:raise AssertionError('0093/0088 anchors')
    if not i93.get('qa_pass') or i93.get('errors')!=[] or i93.get('hashes_before')!=i93.get('hashes_after'):raise AssertionError('0093 independent')
    if e93.get('hashes_before')!=e93.get('hashes_after') or not e93.get('numerical_ok') or not e93.get('local_ok'):raise AssertionError('0093 prior QA')
    ih=i93.get('hashes_before',{})
    if ih.get('arrays')!=A93_SHA or ih.get('evidence')!=E93_SHA or ih.get('checker')!=sha(C93):raise AssertionError('0093 independent links')
    paths={k:Path(v) for k,v in e93['source_paths'].items()};paths.update(evidence93=E93,arrays93=A93,independent93=I93,checker93=C93,cache88=T88,code94=Path(__file__),metric_helper94=ROOT/'experiments/conditional_balance_metrics_0094.py')
    before={k:sha(v) for k,v in paths.items()}
    for k,v in e93['hashes_before'].items():
        if k in before and before[k]!=v:raise AssertionError('changed inherited '+k)
    return paths,before

def _candidate(e93):
    paths={k:Path(v) for k,v in e93['candidate_paths'].items()}
    if sha(paths['deployment_candidate_model'])!=CANDIDATE_SHA:raise AssertionError('candidate SHA')
    recorded=e93.get('candidate_hashes',{}).get('after_forecast',{})
    if set(recorded)!=set(paths) or any(sha(paths[k])!=recorded[k] for k in paths):raise AssertionError('0093 candidate artifact link')
    return paths,{k:sha(v) for k,v in paths.items()}

def run(output):
    if Path(output).exists():raise FileExistsError(output)
    import torch
    torch.set_num_threads(1)
    from train_structured_rom import load_model
    output=Path(output);output.mkdir(parents=True);start=time.perf_counter();deadline=start+900.;arrays={};progress={};paths={};before={};counters={'paths':0,'probes':0,'model_field_probes':0};candidate_paths={};candidate_before={}
    try:
        e93=json.loads(E93.read_text());i93=json.loads(I93.read_text());paths,before=_source_paths(e93,i93);candidate_paths,candidate_before=_candidate(e93)
        with np.load(A93,allow_pickle=False) as z:
            mass,D,phi,k,k0,d,d0=(np.asarray(z[k]) for k in ('mass','D','phi','k','k0','d','d0'))
        dk,dd=k-k0,d-d0
        with np.load(T88,allow_pickle=False) as z: cache=_cache(z)
        validate_cache(cache)
        if sha(BASE/'model.pt')!=BASE_SHA:raise AssertionError('baseline SHA')
        arrays.update(truth=cache['truth'],event_ids=cache['event_ids'],labels=cache['labels'],lengths=cache['lengths'],qenc=cache['qenc'],mass=mass,D=D,phi=phi,delta_k=dk,delta_d=dd)
        base=load_model(BASE);candidate=load_model(candidate_paths['deployment_candidate_model'].parent)
        if rel(mass,np.asarray(cache['mass']))>1e-10 or rel(mass,np.asarray(base.mass.detach()))>1e-10 or rel(mass,np.asarray(candidate.mass.detach()))>1e-10:raise AssertionError('inertia identity')
        from fit_coefficient_block_0082 import deterministic_test_z
        zqa=np.asarray(deterministic_test_z());probes=np.vstack((np.zeros(33),zqa[:2]));bf=[];cf=[];ef=[]
        for x in probes:
            require(deadline,'fixed-state probe')
            counters['probes']+=1;counters['model_field_probes']+=1
            with torch.no_grad():b=np.asarray(base.field(torch.as_tensor(x[None],dtype=torch.float64))[0].detach())
            counters['model_field_probes']+=1
            with torch.no_grad():c=np.asarray(candidate.field(torch.as_tensor(x[None],dtype=torch.float64))[0].detach())
            bf.append(b);cf.append(c);ef.append(expected_field_difference(x,mass,D,phi,dk,dd))
        arrays.update(probe_states=probes,probe_base_fields=np.asarray(bf),probe_candidate_fields=np.asarray(cf),probe_expected_diff=np.asarray(ef))
        if probe_qa(np.asarray(bf),np.asarray(cf),np.asarray(ef))>1e-10:raise AssertionError('omega-only field probes')
        counters['paths']+=1;fine=replay(base,cache['truth'],cache['qenc'],cache['lengths'],mass,D,phi,dk,dd,.005,deadline,progress,output/'partial_fine.npz');arrays['fine']=fine;np.savez_compressed(output/'arrays.npz',**arrays)
        counters['paths']+=1;coarse=replay(base,cache['truth'],cache['qenc'],cache['lengths'],mass,D,phi,dk,dd,.01,deadline,progress,output/'partial_coarse.npz');arrays['coarse']=coarse;np.savez_compressed(output/'arrays.npz',**arrays)
        # Metrics are intentionally delegated to the one registered interface helper.
        from conditional_balance_metrics_0094 import compute_metrics
        metrics=compute_metrics(fine,coarse,cache['truth'],cache['lengths'],mass,event_ids=cache['event_ids'],labels=cache['labels'])
        np.savez_compressed(output/'metrics.npz',scores=metrics['scores'],terms=metrics['terms'],numeric_energy=metrics['numeric_energy'],fullprefixrms=metrics['fullprefixrms'])
        after={k:sha(v) for k,v in paths.items()};require(deadline,'final')
        if before!=after:raise RuntimeError('source changed')
        candidate_after={k:sha(v) for k,v in candidate_paths.items()}
        if candidate_before!=candidate_after:raise RuntimeError('candidate bytes changed')
        evidence={'loop':'0094','test_read':False,'h5_read':False,'conditional_truth_access':True,'metadata_seconds':METADATA_SECONDS,'seconds':time.perf_counter()-start,'source_paths':{k:str(v.resolve()) for k,v in paths.items()},'hashes_before':before,'hashes_after':after,'candidate_paths':{k:str(v.resolve()) for k,v in candidate_paths.items()},'candidate_hashes_before':candidate_before,'candidate_hashes_after':candidate_after,'counters':counters,'metrics_summary':metrics['summary']}
        (output/'evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False));return evidence
    except Exception as exc:
        if arrays:np.savez_compressed(output/'arrays.npz',**arrays)
        (output/'failure_evidence.json').write_text(json.dumps({'loop':'0094','type':type(exc).__name__,'message':str(exc),'progress':progress,'counters':counters,'seconds':time.perf_counter()-start,'source_paths':{k:str(v) for k,v in paths.items()},'hashes_before':before,'hashes_after':{k:sha(v) for k,v in paths.items() if v.exists()},'candidate_paths':{k:str(v) for k,v in candidate_paths.items()},'candidate_hashes_before':candidate_before},indent=2,allow_nan=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
