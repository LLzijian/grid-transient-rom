"""Sync-merge engineering benchmark. No scientific fitting or val entry point."""
from __future__ import annotations
import copy,hashlib,json,math,statistics,time,sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import closure_causal_history_residual_20261019 as old
import horizon_pair_20261023 as horizon

CASES=tuple((2,w) for w in (50,200))+tuple((3,w) for w in (50,200))


def _update(model,opt,rng,observed,scales,deadline=None,*,merged=True,profile=False,loss_fn=None):
    scope=lambda name:torch.profiler.record_function('training/'+name) if profile else nullcontext()
    ix=torch.as_tensor(rng.permutation(len(observed))[:12],device=observed.device)
    opt.zero_grad(set_to_none=True)
    with scope('forward'):
        value=(old.loss if loss_fn is None else loss_fn)(model,observed[ix],scales,deadline)
        if not bool(torch.isfinite(value)):raise FloatingPointError('nonfinite fit loss')
    with scope('backward'):value.backward()
    with scope('gradient_checks'):
        named=[(n,p) for n,p in model.named_parameters() if p.grad is not None]
        if merged:
            norms=torch.stack([p.grad.norm() for _,p in named]).detach().cpu().tolist() if named else []
            gradients=dict(zip((n for n,_ in named),norms))
        else:gradients={n:float(p.grad.norm()) for n,p in named}
        if not all(math.isfinite(v) for v in gradients.values()):raise FloatingPointError('nonfinite gradient')
    with scope('clip'):torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
    with scope('adam'):opt.step()
    old.synchronize(observed.device)
    with scope('parameter_checks'):
        if merged:
            parameters=list(model.parameters())
            finite=bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all()) if parameters else True
        else:finite=not any(not bool(torch.isfinite(p).all()) for p in model.parameters())
        if not finite:raise FloatingPointError('nonfinite updated parameter')
    return float(value.detach()),gradients


def train_update(model,opt,rng,observed,scales,deadline=None):
    return _update(model,opt,rng,observed,scales,deadline,merged=True)


def exact(a,b):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()):raise ValueError('tensor exactness failed')
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('NumPy exactness failed')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('mapping identity failed')
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('sequence identity failed')
        for x,y in zip(a,b):exact(x,y)
    elif a!=b:raise ValueError('scalar exactness failed')


def capture(model,opt,rng):
    return copy.deepcopy({'model':model.state_dict(),'optimizer':opt.state_dict(),'fit_numpy_rng':rng.bit_generator.state,
        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})


def verify_ref(ref):
    path=ROOT/ref['path']
    if old.sha(path)!=ref['sha256']:raise ValueError('source SHA changed: '+str(path))
    return path


def load_snapshot(case,step):
    item=case['snapshots'][str(step)]
    model_path=verify_ref(item['model']);state_path=verify_ref(item['training'])
    model=old.CausalHistoryResidual.from_export(model_path).cuda();opt=old.make_optimizer(model)
    state=torch.load(state_path,map_location='cpu',weights_only=False)
    horizon.checks.validate_state(state,case['identity'],step)
    opt.load_state_dict(state['optimizer'])
    rng=np.random.default_rng();rng.bit_generator.state=state['fit_numpy_rng']
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    return model,opt,rng


def inspect_model(model,observed,scales):
    """No optimizer step: outputs, full backprop and both gradient snapshots."""
    model.zero_grad(set_to_none=True)
    history=observed[:,:11];initial=torch.cat((history[:,-1],model.encode(history)),-1)
    field=model.field(initial);rk4=model.rk4(initial,.01)
    path=old.prediction(model,observed)
    loss=((path-observed[:,11:])/scales).square().mean();loss.backward()
    unclipped={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
    clipped={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    if getattr(model.base,'_runtime_cache',None) is not None:raise ValueError('graph cache leaked')
    return {'field':field.detach(),'rk4':rk4.detach(),'trajectory':path.detach(),'loss':loss.detach(),
            'unclipped':unclipped,'clipped':clipped}


def model_qualification(case,data,observed,scales,output):
    proofs=[]
    for step in (0,2048):
        first,_,_=load_snapshot(case,step);a=inspect_model(first,observed,scales)
        second,_,_=load_snapshot(case,step);b=inspect_model(second,observed,scales);exact(a,b)
        if step==2048:
            metrics=horizon.model_metrics(second,data)
            if not all(v['valid'] for v in metrics.values()):raise ValueError('original full16 dual-step qualification failed')
        else:metrics=None
        # trajectory() owns a finally block, including exception cleanup.
        from closure_causal_recovery_20261019 import trajectory
        original=second.rk4
        def injected_failure(*args,**kwargs):raise RuntimeError('engineering cache cleanup probe')
        second.rk4=injected_failure
        try:
            try:trajectory(second,torch.zeros((1,40),device='cuda',dtype=torch.float64),1,.01,None)
            except RuntimeError as error:
                if str(error)!='engineering cache cleanup probe':raise
            else:raise ValueError('cache cleanup probe did not execute')
        finally:second.rk4=original
        if getattr(second.base,'_runtime_cache',None) is not None:raise ValueError('cache invalidation failed')
        exported=Path(output)/f'qualification_export_{step}.json';second.export(exported)
        third=old.CausalHistoryResidual.from_export(exported).cuda()
        exact(second.state_dict(),third.state_dict())
        if getattr(third.base,'_runtime_cache',None) is not None:raise ValueError('export restored transient cache')
        proofs.append({'step':step,'exact_outputs_and_gradients':True,'train_metrics':metrics,
            'term_kinds':list(second.base._term_kinds),'terms':second.base.terms})
        del first,second,third,a,b
    return proofs


def read_inputs(binding):
    selection=json.loads(verify_ref(binding['selection']).read_text())
    if selection.get('source_commit')!='ebd45248a25905dbf5af4a3cea8c4073a02994db' or not selection.get('all_twelve_frozen') or selection.get('actual_updates')!=24576:
        raise ValueError('completed frozen source required')
    with np.load(verify_ref(binding['train']),allow_pickle=False) as arrays:
        data={name:arrays[name].copy() for name in ('coarse','fine','scales')}
    if data['coarse'].shape!=(16,511,30) or data['fine'].shape!=(16,511,30):raise ValueError('full train arrays required')
    pool=json.loads(verify_ref(binding['pool']).read_text())
    for case in binding['cases']:
        expected=horizon.identity(selection['source_commit'],6301,case['candidate'],case['future'],pool,
            {'old_seed':6201,'ref':binding['train'],'coarse_sha256':hashlib.sha256(data['coarse'].tobytes()).hexdigest()})
        if case['identity']!=expected:raise ValueError('old model identity changed')
        row=next(r for r in selection['rows'] if r['seed']==6301 and r['candidate']==case['candidate'] and r['future']==case['future'])
        for step in (0,2048):
            for name in ('model','training'):
                ref=case['snapshots'][str(step)][name];file='model.json' if name=='model' else 'training.pt'
                if Path(ROOT/ref['path']).resolve()!=(Path(row['path'])/f'checkpoint_{step:05d}'/file).resolve() or ref['sha256']!=row['checkpoints'][str(step)][file]:raise ValueError('snapshot source changed')
    if [(c['candidate'],c['future']) for c in binding['cases']]!=list(CASES):raise ValueError('fixed four cases required')
    return data


def benchmark(output,binding,commit,implementation):
    if implementation not in ('reference','opt1'):raise ValueError('optimization2 requires separate evidence/admission')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    data=read_inputs(binding);rows=[];ledger={'attempted_updates':0,'completed_updates':0}
    updater=old.train_update if implementation=='reference' else train_update
    def update(m,o,r,obs,sc,profile=False):
        ledger['attempted_updates']+=1
        value=(_update(m,o,r,obs,sc,merged=False,profile=True) if profile else updater(m,o,r,obs,sc,None))
        ledger['completed_updates']+=1
        return value
    try:
        for case in binding['cases']:
            case_out=out/horizon.key(6301,case['candidate'],case['future']);case_out.mkdir()
            observed=torch.as_tensor(horizon.fit_slice(data['coarse'],case['future']),device='cuda')
            scales=torch.as_tensor(data['scales'],device='cuda')
            start=time.perf_counter();m,opt,rng=load_snapshot(case,0);torch.cuda.synchronize();cold=time.perf_counter()-start
            start=time.perf_counter()
            for _ in range(2):update(m,opt,rng,observed,scales)
            torch.cuda.synchronize();warmup=time.perf_counter()-start
            groups=[]
            for _ in range(3):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();start=time.perf_counter()
                for _ in range(4):update(m,opt,rng,observed,scales)
                end.record();torch.cuda.synchronize()
                groups.append({'wall_seconds':time.perf_counter()-start,'cuda_ms':begin.elapsed_time(end),
                    'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
            del m,opt,rng
            # Profiler updates are separate, reset to the exact initial state.
            if implementation=='reference':
                m,opt,rng=load_snapshot(case,0)
                # Python call counts complement the operator/CUDA trace without
                # contaminating the separate timed measurements.
                import sys
                from collections import Counter
                calls=Counter();watched={'_torch_eval','_torch_feature_eval','_torch_value','energy_and_gradient','port_values','encode'}
                previous_profile=sys.getprofile()
                def count_calls(frame,event,arg):
                    if event=='call' and frame.f_code.co_name in watched:
                        calls[str(Path(frame.f_code.co_filename).name)+':'+frame.f_code.co_name]+=1
                try:
                    sys.setprofile(count_calls)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                        for _ in range(2):update(m,opt,rng,observed,scales,True)
                finally:sys.setprofile(previous_profile)
                prof.export_chrome_trace(str(case_out/'profile.json'))
                (case_out/'profile.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=80),encoding='utf-8')
                old.atomic_json(case_out/'expression_calls.json',dict(calls))
                del prof
                del m,opt,rng
            qualification=model_qualification(case,data,observed,scales,case_out)
            m,opt,rng=load_snapshot(case,0);trace=[];ident={'engineering_only':True,'source_commit':commit,'implementation':implementation,'case':case['identity']}
            for step in range(1,35):
                value,grads=update(m,opt,rng,observed,scales);trace.append(value)
                if step==32:
                    start=time.perf_counter();old.checkpoint(case_out,m,opt,rng,32,trace,ident);torch.cuda.synchronize();save_seconds=time.perf_counter()-start
            expected=capture(m,opt,rng);expected_trace=list(trace)
            expected_batch=copy.deepcopy(rng).permutation(16)[:12]
            start=time.perf_counter();m,opt,rng=load_snapshot(case,0)
            actual_step,trace=old.restore(case_out,m,opt,rng,ident);torch.cuda.synchronize();restore_seconds=time.perf_counter()-start
            if actual_step!=32:raise ValueError('actual32 restore required')
            for _ in range(2):value,grads=update(m,opt,rng,observed,scales);trace.append(value)
            actual=capture(m,opt,rng);exact(expected,actual);exact(trace,expected_trace)
            exact(expected_batch,copy.deepcopy(rng).permutation(16)[:12])
            torch.save(actual,case_out/'final_state.pt')
            row={'candidate':case['candidate'],'future':case['future'],'cold_seconds':cold,'warmup_seconds':warmup,
                'timing_groups':groups,'median_wall':statistics.median(g['wall_seconds'] for g in groups),
                'wall_range':[min(g['wall_seconds'] for g in groups),max(g['wall_seconds'] for g in groups)],
                'save_seconds':save_seconds,'restore_seconds':restore_seconds,'trace':trace,'last_gradient_norms':grads,
                'qualification':qualification,'final_state':{'path':str(case_out/'final_state.pt'),'sha256':old.sha(case_out/'final_state.pt')},'resume_exact':True}
            rows.append(row);old.atomic_json(out/'progress.json',{'rows':rows,'ledger':ledger})
            del m,opt,rng,expected,actual
        if ledger['completed_updates']!=(208 if implementation=='reference' else 200):raise ValueError('registered workload mismatch')
        result={'status':'PASS','source_commit':commit,'implementation':implementation,'rows':rows,'ledger':ledger,
            'optimization2':'NOT_IMPLEMENTED_REQUIRES_REFERENCE_PROFILE_DECISION','project_val_test_accessed':False,
            'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
                'other_gpu_workloads':'NOT_ESTABLISHED_BY_BENCHMARK; root must record actual concurrent jobs'}}
        if implementation=='opt1':
            reference=json.loads(verify_ref(binding['reference_result']).read_text());speed=[];safe=True
            if reference['source_commit']!=commit or reference['implementation']!='reference' or reference['status']!='PASS':raise ValueError('same-source reference required')
            for a,b in zip(reference['rows'],rows):
                if (a['candidate'],a['future'])!=(b['candidate'],b['future']):raise ValueError('matched case ordering required')
                state_a=torch.load(verify_ref(a['final_state']),map_location='cpu',weights_only=False)
                state_b=torch.load(verify_ref(b['final_state']),map_location='cpu',weights_only=False)
                exact(state_a,state_b);exact(a['trace'],b['trace']);exact(a['last_gradient_norms'],b['last_gradient_norms']);exact(a['qualification'],b['qualification'])
                speed.append(a['median_wall']/b['median_wall'])
                safe &= b['median_wall']<=a['median_wall']*1.05 and max(g['peak_allocated'] for g in b['timing_groups'])<=max(g['peak_allocated'] for g in a['timing_groups'])*1.10
            result.update(geometric_speedup=math.exp(sum(math.log(x) for x in speed)/4),case_speedups=speed)
            result['engineering_acceptance']=bool(safe and result['geometric_speedup']>=1.10)
        old.atomic_json(out/'result.json',result);return result
    except BaseException as error:
        import traceback
        old.atomic_json(out/'failure.json',{'status':'UNRESOLVED','ledger':ledger,'rows':rows,'error':repr(error),'traceback':traceback.format_exc()});raise
