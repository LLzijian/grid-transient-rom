"""Frozen three-structure five-second comparison; no training or project readers."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

START=time.perf_counter()
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import edge_memory_discrimination_20261021 as prior
SEEDS=(5101,5102,5103)
NAMES=('sin','cos','empty')
WINDOWS={'primary_2_5':(200,500),'secondary_1_2':(100,200),
         'cumulative_0_2':(0,200),'cumulative_0_5':(0,500),'reproduction_0_1':(0,100)}
CAPS={'qa':210.,'cost':210.,'evaluate':240.}


def validate_manifest(value):
    if value.get('seeds')!=list(SEEDS) or [r.get('seed') for r in value.get('rows',[])]!=list(SEEDS):
        raise ValueError('all three registered seeds required')
    for row in value['rows']:
        scales=row.get('scales',[])
        if len(scales)!=30 or any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<1e-6 for x in scales):
            raise ValueError('fixed 30 train scales required')
        for key in ('old_val_clean_sha256','teacher_source_model_sha256'):
            digest=row.get(key,'')
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('missing teacher provenance')
        if set(row.get('models',{}))!=set(NAMES): raise ValueError('all three structures required')
        for name,model in row['models'].items():
            for pathkey,hashkey in (('path','model_sha256'),('training_path','training_sha256')):
                path=Path(model[pathkey])
                if not path.is_absolute() or not path.is_file() or prior.sha_file(path)!=model[hashkey]:
                    raise ValueError('frozen endpoint bytes changed')
            payload=json.loads(Path(model['path']).read_text(encoding='utf-8'))
            expected={'sin':[prior.canonical_terms()[0]],'cos':[prior.canonical_terms()[1]],'empty':[]}[name]
            if (payload.get('format')!='causal_history_residual_v1'
                    or payload.get('base',{}).get('terms')!=expected):
                raise ValueError('wrong frozen structure type')
    for path,digest in value.get('files',{}).items():
        if prior.sha_file(ROOT/path)!=digest: raise ValueError('frozen evidence bytes changed')
    return value


def read_manifest(binding):
    identity=binding.get('endpoint_manifest',{})
    expected=(ROOT/'private_data/closure_mechanism_20261019/P0/long_horizon_discrimination01/endpoints.json').resolve()
    path=(ROOT/identity.get('path','')).resolve()
    if path!=expected or not path.is_file() or prior.sha_file(path)!=identity.get('sha256'):
        raise ValueError('missing root-frozen nine-endpoint manifest')
    return validate_manifest(json.loads(path.read_text(encoding='utf-8')))


def save_raw_pair(output,label,coarse,fine):
    """Retain only actually computed public-coordinate arrays, never hidden q."""
    import numpy as np
    if coarse.shape[-1]!=30 or fine.shape[-1]!=30: raise ValueError('only observed30 raw coordinates allowed')
    path=Path(output)/(label+'.npz')
    if path.exists(): raise ValueError('raw trajectory already exists')
    np.savez(path,coarse=coarse,fine=fine)
    return {'path':str(path.resolve()),'sha256':prior.sha_file(path),'bytes':path.stat().st_size,
            'coarse_shape':list(coarse.shape),'fine_shape':list(fine.shape),'observed_coordinates':30,
            'observation_dt_seconds':.01,'coarse_integration_dt_seconds':.01,
            'fine_integration_dt_seconds':.005,'fine_observation_stride':2}


def future_fixture(seed,deadline,formal=False,row=None,output=None):
    """Hidden teacher coordinates stay inside; only 30 observed columns return."""
    import numpy as np
    import torch
    from closure_discovery_benchmarks import toy_source
    from closure_causal_recovery_20261019 import copy_runtime,trajectory
    source=toy_source()
    source_hash=hashlib.sha256(json.dumps(source._export_payload(),sort_keys=True,allow_nan=False).encode()).hexdigest()
    if formal and source_hash!=row['teacher_source_model_sha256']: raise ValueError('teacher source changed')
    teacher=copy_runtime(source,[prior.SIN_TERM],'cuda')
    if formal:
        stream=np.random.SeedSequence([seed,20261019]).spawn(4)[1]
    else: stream=np.random.SeedSequence([seed,20261021,78019])
    rng=np.random.default_rng(stream)
    strength=np.where(np.arange(16)[:,None]<8,.08,.3)
    initial=rng.normal(size=(16,40))*strength
    initial[:,30:]=rng.normal(size=(16,10))*np.where(np.arange(16)[:,None]<8,.2,1.2)
    z=torch.tensor(initial,dtype=torch.float64,device='cuda')
    with torch.no_grad():
        teacher.coefficients.fill_(.15)
        coarse=trajectory(teacher,z,510,.01,deadline).cpu().numpy()
        fine=trajectory(teacher,z,1020,.005,deadline).cpu().numpy()[:,::2]
    prefix_hash=hashlib.sha256(coarse[:,:111].copy().tobytes()).hexdigest()
    raw_start=time.perf_counter()
    raw=save_raw_pair(output,'teacher',coarse,fine) if output is not None else None
    raw_io_seconds=time.perf_counter()-raw_start
    if formal and np.isfinite(coarse[:,:111]).all() and prefix_hash!=row['old_val_clean_sha256']:
        raise ValueError('old stream1 first111 observations changed')
    return source,coarse,fine,{'formal':formal,'prefix_111_sha256':prefix_hash,
                              'teacher_source_model_sha256':source_hash,'steps_coarse':510,'steps_fine':1020,
                              'raw':raw,'raw_io_seconds':raw_io_seconds}


def predict(model,history,dt,deadline,steps=500):
    import torch
    from closure_causal_recovery_20261019 import trajectory
    if tuple(history.shape)!=(16,11,30): raise ValueError('exact 100ms causal history required')
    with torch.no_grad():
        h=torch.as_tensor(history,device=model.mass.device,dtype=model.mass.dtype)
        initial=torch.cat((h[:,-1],model.encode(h)),-1)
        factor=round(.01/dt)
        if factor not in (1,2) or dt not in (.01,.005): raise ValueError('registered steps only')
        return trajectory(model,initial,steps*factor,dt,deadline)[:,factor::factor].cpu().numpy()


def event_qualification(coarse,fine,scales,end):
    import numpy as np
    finite=np.isfinite(coarse[:,:end]).all((1,2)) & np.isfinite(fine[:,:end]).all((1,2))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        delta=np.max(np.abs(coarse[:,:end]-fine[:,:end])/scales,axis=(1,2))
    eligible=finite & np.isfinite(delta) & (delta<=1e-3)
    return [{'eligible':bool(ok),'normalized_step_max':float(d) if np.isfinite(d) else None,
             'failure_category':None if ok else ('nonfinite' if not f or not np.isfinite(d) else 'step_difference')}
            for ok,d,f in zip(eligible,delta,finite)]


def summarize(events,eligible):
    if len(events)!=16 or len(eligible)!=16: raise ValueError('all16 denominator required')
    ok=all(eligible) and all(x is not None and math.isfinite(x) for x in events)
    if not ok:
        return {'status':'UNRESOLVED','count':16,'eligible_count':sum(eligible),
                'mean':None,'median':None,'worst':None,'weak_mean':None,'strong_mean':None,'above_0_01':None}
    ordered=sorted(events)
    return {'status':'QUALIFIED','count':16,'eligible_count':16,'mean':math.fsum(x/16 for x in events),
            'median':ordered[7]/2+ordered[8]/2,'worst':max(events),'weak_mean':math.fsum(x/8 for x in events[:8]),
            'strong_mean':math.fsum(x/8 for x in events[8:]),'above_0_01':sum(x>.01 for x in events)}


def score_model(mc,mf,tc,tf,scales):
    import numpy as np
    if any(a.shape!=(16,500,30) for a in (mc,mf,tc,tf)): raise ValueError('full16 x500 x30 arrays required')
    scales=np.asarray(scales)
    windows={}
    for name,(start,end) in WINDOWS.items():
        # Late-window scores require qualification over the entire preceding
        # autonomous interval, not only a numerically convenient late subset.
        tq=event_qualification(tc,tf,scales,end);mq=event_qualification(mc,mf,scales,end)
        eligible=[a['eligible'] and b['eligible'] for a,b in zip(tq,mq)]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            e=(mc[:,start:end]-tc[:,start:end])/scales
            d=(np.abs(mc[:,start:end]-mf[:,start:end])+np.abs(tc[:,start:end]-tf[:,start:end]))/scales
            mse=np.mean(e*e,axis=(1,2));bound=np.mean(2*np.abs(e)*d+d*d,axis=(1,2))
        clean=lambda arr:[float(x) if np.isfinite(x) else None for x in arr]
        windows[name]={'indices':[start,end],'time_points':end-start,'per_event_mse':clean(mse),
                       'per_event_B':clean(bound),'teacher_qualification':tq,'model_qualification':mq,
                       'eligible':eligible,**summarize(clean(mse),eligible)}
    return windows


def pairwise(models):
    pairs={}
    for other in ('cos','empty'):
        windows={}
        for name in WINDOWS:
            a=models['sin'][name];b=models[other][name]
            eligible=[x and y for x,y in zip(a['eligible'],b['eligible'])]
            D=[None if x is None or y is None else y-x for x,y in zip(a['per_event_mse'],b['per_event_mse'])]
            B=[None if x is None or y is None or not math.isfinite(x+y) else x+y for x,y in zip(a['per_event_B'],b['per_event_B'])]
            valid=all(eligible) and all(x is not None and y is not None for x,y in zip(D,B))
            meanD=math.fsum(x/16 for x in D) if valid else None;meanB=math.fsum(x/16 for x in B) if valid else None
            count=sum(x>y for x,y in zip(D,B)) if valid else None
            windows[name]={'status':'QUALIFIED' if valid else 'UNRESOLVED','count':16,
                           'per_event_D':D,'per_event_B_sum':B,'eligible':eligible,
                           'mean_D':meanD,'mean_B_sum':meanB,'events_D_gt_B':count,
                           'sin_advantage_rule':bool(valid and meanD>meanB and meanD>0 and count>=9)}
        pairs[other+'_minus_sin']=windows
    return pairs


def evaluate_seed(row,deadline,output):
    import numpy as np
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    try:
        _,coarse,fine,fixture=future_fixture(row['seed'],deadline,formal=True,row=row,output=output)
    except TimeoutError: raise
    except (RuntimeError,FloatingPointError,ValueError) as error:
        failed=np.full((16,500,30),np.nan)
        models={name:score_model(failed,failed,failed,failed,row['scales']) for name in NAMES}
        teacher_path=Path(output)/'teacher.npz'
        rawfiles={'teacher':{'path':str(teacher_path.resolve()),'sha256':prior.sha_file(teacher_path),
                            'bytes':teacher_path.stat().st_size}} if teacher_path.is_file() else {}
        return {'status':'COMPLETED','seed_status':'UNRESOLVED','seed':row['seed'],
                'primary_status':'UNRESOLVED','registered_seed_rule':False,
                'fixture':None,'models':models,'pairs':pairwise(models),'event_count':16,'weak_count':8,'strong_count':8,
                'rawfiles':rawfiles,
                'failures':{'teacher':{'category':'source_identity_failure' if isinstance(error,ValueError)
                            else 'teacher_execution_failure','error':repr(error)}}}
    tc,tf=coarse[:,11:],fine[:,11:];models={};errors={};rawfiles={'teacher':fixture['raw']}
    for name in NAMES:
        try:
            model=CausalHistoryResidual.from_export(row['models'][name]['path']).cuda()
            mc=predict(model,coarse[:,:11],.01,deadline)
            mf=predict(model,coarse[:,:11],.005,deadline)
            rawfiles[name]=save_raw_pair(output,name,mc,mf)
        except TimeoutError: raise
        except (RuntimeError,FloatingPointError,ValueError) as error:
            # Preserve the full denominator and all other candidates. This is
            # a numerical/implementation failure, never a physical label.
            errors[name]={'category':'model_execution_failure','error':repr(error)}
            mc=np.full((16,500,30),np.nan);mf=mc.copy()
        models[name]=score_model(mc,mf,tc,tf,row['scales'])
    pairs=pairwise(models)
    return {'status':'COMPLETED','seed':row['seed'],'fixture':fixture,'models':models,
            'pairs':pairs,'failures':errors,'event_count':16,'weak_count':8,'strong_count':8,
            'rawfiles':rawfiles,
            'primary_status':'QUALIFIED' if all(m['primary_2_5']['status']=='QUALIFIED' for m in models.values()) else 'UNRESOLVED',
            'registered_seed_rule':all(p['primary_2_5']['sin_advantage_rule'] for p in pairs.values())}


def run_cost(output,deadline,manifest):
    import numpy as np
    import torch
    from closure_causal_recovery_20261019 import runtime_model
    from closure_causal_history_residual_20261019 import CausalHistoryResidual
    rows=[]
    for seed in SEEDS:
        rawdir=output/f'cost_only_raw_{seed}';rawdir.mkdir()
        begin=time.perf_counter();compute=0.;torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize();start=time.perf_counter()
        source,coarse,fine,fixture=future_fixture(seed,deadline,formal=False,output=rawdir)
        torch.cuda.synchronize();compute+=max(0.,time.perf_counter()-start-fixture['raw_io_seconds'])
        timed_metrics={};rawfiles={'teacher':fixture['raw']}
        for name,terms in zip(NAMES,([prior.SIN_TERM],[prior.COS_TERM],[])):
            model=CausalHistoryResidual.from_train(runtime_model(source,terms,seed,'cuda'),
                                                   torch.as_tensor(coarse[:,:11],device='cuda'))
            with torch.no_grad():model.base.coefficients.fill_(.01)
            path=output/f'cost_only_{seed}_{name}.json';model.export(path)
            model=CausalHistoryResidual.from_export(path).cuda()
            torch.cuda.synchronize();start=time.perf_counter()
            mc=predict(model,coarse[:,:11],.01,deadline);mf=predict(model,coarse[:,:11],.005,deadline)
            torch.cuda.synchronize();compute+=time.perf_counter()-start
            rawfiles[name]=save_raw_pair(rawdir,name,mc,mf)
            # Timing output is not a scientific score. Same-shape serialization
            # also covers the full compact result and its per-event bound work.
            timed_metrics[name]=score_model(mc,mf,coarse[:,11:],fine[:,11:],
                next(r for r in manifest['rows'] if r['seed']==seed)['scales'])
        shaped={'status':'COMPLETED','seed':seed,'fixture':fixture,'models':timed_metrics,
                'pairs':pairwise(timed_metrics),'failures':{},'rawfiles':rawfiles,
                'event_count':16,'weak_count':8,'strong_count':8,'source_commit':'0'*40,
                'primary_status':'QUALIFIED','registered_seed_rule':False,'worker_phase_elapsed_seconds':0.}
        size=len(json.dumps(shaped,indent=2,sort_keys=True,allow_nan=False).encode('utf-8'))+65536
        # Exercise result calculation and same-size atomic serialization, but
        # discard all timing-only errors/qualifications rather than publish scores.
        for suffix in ('result','complete'):
            prior._atomic_json(output/f'cost_only_output_{seed}_{suffix}.json',
                               {'cost_only':True,'serialized_size_allowance_bytes':size,'padding':'0'*size})
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        fixed=max(0.,elapsed-compute);estimate=1.5*compute+fixed+20.
        rows.append({'seed':seed,'synchronized_compute_seconds':compute,'fixed_seconds':fixed,
                     'estimated_native_seconds':estimate,'admitted':estimate<=280,
                     'teacher_coarse_steps':510,'teacher_fine_steps':1020,'model_count':3,
                     'model_coarse_steps':500,'model_fine_steps':1000,
                     'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
    return {'status':'PASS' if all(r['admitted'] for r in rows) else 'REJECT',
            'seeds':rows,'P1_complete_cap_seconds':1400,'science_scores':None,
            'supervision_reserve_seconds':20.,'cost_only':True}


def execute(phase,output,deadline,binding,seed):
    import torch
    torch.set_num_threads(1)
    manifest=read_manifest(binding)  # all9 bytes before any future
    if phase=='qa':
        proc=subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_long_horizon_discrimination_20261021.py')],
                            cwd=ROOT,capture_output=True,text=True,timeout=max(.1,deadline-time.perf_counter()))
        (output/'pytest_stdout.txt').write_text(proc.stdout,encoding='utf-8')
        (output/'pytest_stderr.txt').write_text(proc.stderr,encoding='utf-8')
        result={'status':'PASS' if proc.returncode==0 else 'FAIL','pytest_returncode':proc.returncode,
                'stdout':proc.stdout,'stderr':proc.stderr,'frozen_model_count':9}
    elif phase=='cost':
        qa=prior._bound_pass(binding,'qa_result')
        if qa.get('frozen_model_count')!=9: raise ValueError('missing actual nine-model QA')
        result=run_cost(output,deadline,manifest)
    elif phase=='evaluate':
        admission=prior._bound_pass(binding,'admission')
        if admission.get('P1_complete_cap_seconds')!=1400: raise ValueError('admission cap differs')
        result=evaluate_seed(next(r for r in manifest['rows'] if r['seed']==seed),deadline,output)
    else: raise ValueError('unknown phase')
    read_manifest(binding)  # all9 bytes after execution
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=CAPS)
    for name in ('output','binding'):parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--max-seconds',required=True,type=float)
    parser.add_argument('--seed',choices=SEEDS,type=int);a=parser.parse_args(argv)
    if a.max_seconds!=CAPS[a.phase] or len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):
        parser.error('exact registered cap and full source commit required')
    if a.phase=='evaluate' and a.seed not in SEEDS:parser.error('registered seed required')
    output=a.output.resolve();category='P1' if a.phase=='evaluate' else 'P0'
    allowed=(ROOT/'private_data/closure_mechanism_20261019'/category/'long_horizon_discrimination01').resolve()
    if not output.is_relative_to(allowed) or output==allowed:raise ValueError('exclusive stage output required')
    output.mkdir(parents=True,exist_ok=False);prior._worker_receipt(output,a.phase,a.source_commit,a.binding)
    deadline=START+a.max_seconds
    def timeout():
        prior._atomic_json(output/'timeout.json',{'status':'TIMEOUT','max_seconds':a.max_seconds});os._exit(124)
    timer=threading.Timer(max(.001,deadline-time.perf_counter()),timeout);timer.daemon=True;timer.start()
    try:
        binding=json.loads(a.binding.read_text(encoding='utf-8'))
        result=execute(a.phase,output,deadline,binding,a.seed)
        result.update(source_commit=a.source_commit,worker_phase_elapsed_seconds=time.perf_counter()-START)
        prior._atomic_json(output/'result.json',result);prior._atomic_json(output/'complete.json',{'status':'COMPLETED','result':result})
        return 0 if result['status'] in ('PASS','COMPLETED') else 2
    except BaseException as error:
        import traceback
        prior._atomic_json(output/'failure.json',{'status':'UNRESOLVED','error':repr(error),'traceback':traceback.format_exc()});raise
    finally:timer.cancel()


if __name__=='__main__':raise SystemExit(main())
