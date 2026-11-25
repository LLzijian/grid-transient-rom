"""Release-bound graph qualification, from-scratch fit, and fixed-point evaluation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from experiments.osgkd_v2_sprint2a_20261125 import train as fit
from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import run as previous
from kiki.models.osgkd_spectral import reference_geometry, spectral_penalty
from experiments.osgkd_v2_fast_zero_sum_20261124 import run as old, scoring
from experiments.osgkd_v2_train385_risk_alignment_20261123 import analyze
from kiki.evaluation.osgkd_autonomous import summarize_autonomous_8s
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events
baseline=old.baseline
require,read,verified,identity,repo_path=old.require,old.read,old.verified,old.identity,old.repo_path
RUN='osgkd_v2_sprint2a_20261125'
JOBS={'graph_24111':('graph',24111),'graph_24121':('graph',24121),'graph_24131':('graph',24131),'flat_24111':('flat',24111)}
PROTOCOL='campaigns/osgkd_v2/sprint2a_spectral_matched_protocol.md'
PROBE='results/osgkd_v2_p2a_post_audit_20261124/probe01/summary.json'
PROBE_SHA='5381143c686e4739b749ffcb76a48d322f4332fa3720f5a8425160346c62e6a1'
INPUTS=previous.INPUTS|{'probe_summary':PROBE}
SOURCES=previous.SOURCES|{f'experiments/{RUN}/{name}.py' for name in ('run','train','score')}|{PROTOCOL,'designs/'+RUN+'.md','src/kiki/models/osgkd_matched_flat.py','src/kiki/models/osgkd_spectral.py','tests/test_osgkd_spectral.py','experiments/osgkd_v2_p2a_post_audit_20261124/run.py','experiments/osgkd_v2_p2a_post_audit_20261124/contributions.py','campaigns/osgkd_v2/p2a_post_audit_protocol.md','experiments/osgkd_v2_p0_20261123/run.py'}
COMMON_KEYS=previous.COMMON_KEYS

def validate(release,job):
    require(job in JOBS or job in ('qualify01','evaluate01'),'unknown job')
    extra={'qualification_ref'} if job in JOBS else {'fit_refs'} if job=='evaluate01' else set()
    require(set(release)==COMMON_KEYS|extra,'exact release keys required')
    require(tuple(release[k] for k in ('run_id','job_id','released','device','threads','recompute'))==(RUN,job,True,'cpu',1,False),'release identity differs')
    require(set(release['source_hashes'])==SOURCES,'exact source coverage required')
    for name,sha in release['source_hashes'].items(): require(identity(repo_path(name))['sha256']==sha,'source differs: '+name)
    require(set(release['input_refs'])==set(INPUTS),'exact input references required')
    inputs={}
    for key,path in INPUTS.items():
        require(release['input_refs'][key]['path']==path,'input path differs'); inputs[key]=read(release['input_refs'][key])
    old.prior.validate_baseline(inputs['baseline_summary'],inputs['baseline_archive'],release['input_refs']['baseline_summary'])
    anchor=inputs['anchor_analysis']
    require(release['input_refs']['anchor_analysis']['sha256']==previous.ANCHOR_SHA and anchor['status']=='completed' and anchor['encoder_parameters']==6174,'Sprint0 identity differs')
    require(anchor['evaluator']==identity(ROOT/'src/kiki/evaluation/osgkd_autonomous.py'),'anchor evaluator identity differs')
    for split in ('train','val'): require(anchor['evaluation'][split]['data_identity']==inputs['baseline_summary']['result'][split]['data_identity'],'anchor data identity differs')
    probe=inputs['probe_summary']; require(release['input_refs']['probe_summary']['sha256']==PROBE_SHA and probe['status']=='completed','probe identity differs')
    spectrum=probe['result']['spectra']['2048']
    require(spectrum['qualified'] and spectrum['dimension']==49 and spectrum['max_real']>0 and spectrum['positive_count']>=1 and probe['result']['spectral_branch']=='enable_spectral_hinge','positive qualified spectral branch required')
    def prerequisite(ref,required_job):
        require(ref['path']==f'results/{RUN}/{required_job}/summary.json','canonical prerequisite required')
        prior=read(ref); require(prior['status']=='completed' and prior['run_id']==RUN and prior['job_id']==required_job,'completed prerequisite required')
        for key in COMMON_KEYS-{'job_id'}: require(prior['release'][key]==release[key],'prerequisite binding differs: '+key)
        return prior
    if job in JOBS:
        prior=prerequisite(release['qualification_ref'],'qualify01'); require(prior['result']['passed'],'qualification failed'); inputs['prerequisite']=prior
    if job=='evaluate01':
        require(set(release['fit_refs'])==set(JOBS),'all four fits required')
        inputs['fits']={key:prerequisite(ref,key) for key,ref in release['fit_refs'].items()}
        qualifications=[]
        for key,prior in inputs['fits'].items():
            require(prior['result']['valid_endpoint'] and prior['result']['attempts']==2048 and prior['result']['accepted']>=2033 and set(prior['result']['checkpoints'])=={'2048'},'invalid fixed endpoint: '+key)
            q=prerequisite(prior['release']['qualification_ref'],'qualify01'); require(q['result']['passed'],'qualification failed')
            qualifications.append(prior['release']['qualification_ref'])
        require(all(q==qualifications[0] for q in qualifications),'common qualification differs')
    return inputs

def load_train(inputs): return previous.load_train(inputs)


def peak_memory_bytes():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(key,ctypes.c_size_t) for key in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        process=ctypes.windll.kernel32.GetCurrentProcess
        process.restype=wintypes.HANDLE
        query=ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        require(bool(query(process(),ctypes.byref(counters),counters.cb)),'peak memory query failed')
        return counters.PeakWorkingSetSize
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)

def checkpoint(path,model,opt,k,release,plan_ref,work,guard_work,rejected):
    payload=dict(spectral_jacobian_calls=max(0,k-896),attempt=k,model=model.checkpoint(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state().clone(),release=release,plan_ref=plan_ref,work=work.snapshot(),guard_work=guard_work.snapshot(),rejected=rejected)
    require(not path.exists(),'checkpoint exists'); temp=path.with_suffix('.pt.tmp')
    with temp.open('xb') as stream: torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    os.rename(temp,path); return identity(path)

def qualify(raw,constants,release,inputs,local,context):
    sequence=fit.plan(raw); lookup={e['id']:e for e in tensor_events(raw,'cpu',max_steps=800)}
    batch=[lookup[x] for x in sequence['batches'][0]]; results={}; context['work']={}
    for job,(kind,seed) in JOBS.items():
        started,cpu=perf_counter(),process_time(); model=fit.new_model(constants,kind,seed); opt=fit.new_optimizer(model); work=IntegrationWork()
        context.update(phase='qualification',model=model,opt=opt); context['work'][job]=work
        result=fit.guard(model,batch,constants,work,limits=(800,))
        model.zero_grad(set_to_none=True); penalty,alpha=spectral_penalty(model,reference_geometry(model)); penalty.backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        finite=bool(torch.isfinite(alpha)) and bool(torch.isfinite(penalty)) and all(bool(torch.isfinite(g).all()) for g in gradients)
        norm=float(torch.cat([g.flatten() for g in gradients]).norm())
        require(finite and (not bool(alpha>-.05) or norm>0),'spectral training gradient failed')
        result.update(alpha=float(alpha.detach()),spec=float(penalty.detach()),spectral_gradient_norm=norm,spectral_gradient_finite=finite,encoder_parameters=fit.encoder_parameters(model),spectral_jacobian_calls=3,cost=dict(wall_s=perf_counter()-started,cpu_s=process_time()-cpu,peak_memory_bytes=peak_memory_bytes()))
        results[job]=result
    coarse=all(row['passed'] for row in results.values())
    return dict(passed=True,coarse_passed=coarse,training_internal_step=.01 if coarse else .005,jobs=results,official_updates=0,spectral_branch='enable_spectral_hinge')

def train(raw,constants,release,inputs,local,context):
    sequence=fit.plan(raw); plan_ref=baseline.atomic_write(local/'batch_plan.json',sequence)
    lookup={e['id']:e for e in tensor_events(raw,'cpu',max_steps=800)}; kind,seed=JOBS[release['job_id']]; model=fit.new_model(constants,kind,seed); opt=fit.new_optimizer(model)
    work,guard_work=IntegrationWork(),IntegrationWork(); rows=[]; guards=[]; refs={}; rejected=consecutive=0
    h=inputs['prerequisite']['result']['training_internal_step']
    context.update(phase='training',work={'training':work,'guard':guard_work},model=model,opt=opt,rows=rows)
    stopped=False
    for k,bids in enumerate(sequence['batches']):
        context.update(attempt=k+1); batch=[lookup[x] for x in bids]
        if k in (512,896):
            context['phase']='stage_guard'; guard_started,guard_cpu=perf_counter(),process_time(); g=fit.guard(model,[batch[0],batch[-1]],constants,guard_work,limits=(fit.horizon(k),)); g['cost']=dict(wall_s=perf_counter()-guard_started,cpu_s=process_time()-guard_cpu); guards.append(dict(attempt=k+1,**g))
            if h==.01 and not g['passed']:
                stopped=True; context['phase']='training'; break
            context['phase']='training'
        expected=round(.01/h)*4*sum(min(fit.horizon(k),len(e['targets']['theta'])-1) for e in batch)
        require(work.rhs['forward']+expected<=fit.CAP,'training cap would exceed')
        started,cpu=perf_counter(),process_time(); before=work.rhs['forward']; row=fit.attempt(model,opt,batch,k,constants,work,h)
        require(work.rhs['forward']-before==expected,'training RHS accounting differs')
        row.update(wall_s=perf_counter()-started,cpu_s=process_time()-cpu,peak_memory_bytes=peak_memory_bytes()); rows.append(row)
        rejected+=not row['accepted']; consecutive=0 if row['accepted'] else consecutive+1
        baseline.atomic_write(local/f'attempt_{k+1:04d}.json',row); stopped=consecutive>=3 or rejected>=16
        if (k+1)%64==0 or stopped:
            ref=checkpoint(local/f'attempt_{k+1}.pt',model,opt,k+1,release,plan_ref,work,guard_work,rejected)
            if k+1==2048: refs[str(k+1)]=ref
            progress=dict(spectral_jacobian_calls=max(0,k+1-896),peak_memory_bytes=peak_memory_bytes(),attempt=k+1,accepted=len(rows)-rejected,rejected=rejected,stopped_early=stopped,checkpoint=ref,work=work.snapshot(),guard_work=guard_work.snapshot(),training_internal_step=h,cost=dict(wall_s=sum(r['wall_s'] for r in rows),cpu_s=sum(r['cpu_s'] for r in rows)))
            baseline.atomic_write(local/f'progress_{k+1:04d}.json',progress); print(json.dumps(progress),flush=True)
        if stopped: break
    return dict(fixed_endpoint=2048 if not stopped else None,actual_endpoint=len(rows),attempts=len(rows),accepted=len(rows)-rejected,rejected=rejected,valid_endpoint=not stopped and k==2047 and len(rows)-rejected>=2033,stopped_early=stopped,checkpoints=refs,plan_ref=plan_ref,attempts_ref=baseline.atomic_write(local/'attempts.json',rows),work=work.snapshot(),guard_work=guard_work.snapshot(),guards=guards,train_cost=dict(wall_s=sum(r['wall_s'] for r in rows),cpu_s=sum(r['cpu_s'] for r in rows)),encoder_parameters=fit.encoder_parameters(model),state_dimension=50,job=release['job_id'],seed=seed,training_internal_step=h,spectral_jacobian_calls=sum(r['horizon_s']==8 for r in rows),guard_spectral_jacobian_calls=sum(g['spectral_jacobian_calls'] for g in guards),cost=dict(peak_memory_bytes=peak_memory_bytes(),training_wall_s=sum(r['wall_s'] for r in rows),training_cpu_s=sum(r['cpu_s'] for r in rows),guard_wall_s=sum(g['cost']['wall_s'] for g in guards),guard_cpu_s=sum(g['cost']['cpu_s'] for g in guards)),health=dict(model_finite=old.prior.finite_tree(model.state_dict()),adam_finite=old.prior.finite_tree(opt.state_dict()),clipped=sum(r.get('clipped',False) for r in rows)))

def execute(path,job,action=None):
    started,cpu=perf_counter(),process_time(); path=path.resolve(); require(path.is_relative_to(ROOT),'repository release required')
    release=json.loads(path.read_bytes()); inputs=validate(release,job)
    local,public=[ROOT/side/RUN/job for side in ('artifacts','results')]; require(not local.exists() and not public.exists(),'job already claimed')
    local.mkdir(parents=True); public.mkdir(parents=True)
    receipt=dict(run_id=RUN,job_id=job,status='started',release=release,release_ref=identity(path),pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),environment=dict(device='cpu',dtype='float64',threads=1,recompute=False,torch=str(torch.__version__),numpy=np.__version__,python=sys.version))
    baseline.atomic_write(local/'started.json',receipt); context=dict(phase='loading',work={})
    try:
        torch.set_num_threads(1); raw,constants=load_train(inputs)
        handler=action or (qualify if job=='qualify01' else train)
        require(action is not None or job!='evaluate01','evaluation action required')
        result=handler(raw,constants,release,inputs,local,context)
        receipt.update(status='completed',result=result,completed_utc=datetime.now(timezone.utc).isoformat(),cost=dict(wall_s=perf_counter()-started,cpu_s=process_time()-cpu,peak_memory_bytes=peak_memory_bytes()))
        ref=baseline.atomic_write(public/'summary.json',receipt,public=True); baseline.atomic_write(local/'completed.json',dict(status='completed',summary_ref=ref)); return receipt
    except BaseException as error:
        snapshot=None
        if 'model' in context:
            target=local/'failure_snapshot.pt'; temporary=local/'failure_snapshot.pt.tmp'
            with temporary.open('xb') as stream:
                torch.save(dict(attempt=context.get('attempt'),model=context['model'].checkpoint(),optimizer=context['opt'].state_dict(),torch_rng=torch.get_rng_state().clone()),stream); stream.flush(); os.fsync(stream.fileno())
            os.rename(temporary,target); snapshot=identity(target)
        baseline.atomic_write(local/'failed.json',dict(status='failed',error=repr(error),phase=context['phase'],attempt=context.get('attempt'),active_split=context.get('active_split'),snapshot_ref=snapshot,recorded_attempts=len(context.get('rows',[])),last_attempt=context.get('rows',[None])[-1] if context.get('rows') else None,model_finite=old.prior.finite_tree(context['model'].state_dict()) if 'model' in context else None,adam_finite=old.prior.finite_tree(context['opt'].state_dict()) if 'opt' in context else None,work={k:v.snapshot() for k,v in context['work'].items()},cost=dict(wall_s=perf_counter()-started,cpu_s=process_time()-cpu,peak_memory_bytes=peak_memory_bytes()))); raise

def self_check():
    torch.set_num_threads(1)
    previous.self_check()
    import pytest
    result=pytest.main(['-q',str(ROOT/'tests/test_osgkd_graph.py'),str(ROOT/'tests/test_osgkd_spectral.py')])
    require(result==0,'synthetic model and spectral tests failed')
    events=[dict(id=str(i),label='stable' if i<342 else 'unstable') for i in range(385)]
    shared_plan=fit.plan(events)
    original_plan=previous.fit.plan(events)
    assert shared_plan['batches']==original_plan['batches'] and shared_plan['coverage']==original_plan['coverage']
    assert shared_plan['seeds']=={'stable':24112,'unstable':24113}
    constants={key:np.ones(10) for key in ('mass','damping','Sp','Se','omega_scale','S_m')}
    constants.update(F=np.eye(10),theta_ref=np.zeros(10),candidate_graph='undirected_complete_K10',s_coi=2.)
    models=[fit.new_model(constants,*spec) for spec in JOBS.values()]
    ode=lambda m:{k:v for k,v in m.state_dict().items() if not k.startswith(('encoder.','initial_head.'))}
    for m in models[1:]: assert old.prior.p1a.tree_equal(ode(models[0]),ode(m))
    for model in models:
        penalty,alpha=spectral_penalty(model,reference_geometry(model)); penalty.backward()
        assert torch.isfinite(alpha) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    small={key:np.ones(2) for key in ('mass','damping','Sp','Se','omega_scale','S_m')}
    small.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]),s_coi=2.)
    rng=np.random.default_rng(11)
    event=dict(id='synthetic',history={key:rng.normal(0,.001,(11,2)) for key in ('theta','omega')},t=np.arange(12)*.01,targets={key:rng.normal(0,.001,(12,2)) for key in ('theta','omega','tau_m','tau_e')},masks={key:np.ones((12,2),bool) for key in ('theta','omega','tau_m','tau_e')})
    batch=tensor_events([event],'cpu',max_steps=800)
    for kind in ('graph','flat'):
        model=fit.new_model(small,kind)
        with torch.no_grad(): model.mechanical_coeff[0,0]=.4
        work=IntegrationWork(); row=fit.attempt(model,fit.new_optimizer(model),batch,896,small,work,.01)
        assert row['accepted'] and row['hinge_active'] and row['spec']>0 and work.rhs['forward']==44
    print(json.dumps(dict(self_check='passed',scientific_payloads_opened=False,jobs=list(JOBS),encoder_parameters=[fit.encoder_parameters(m) for m in models])))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True); sub.add_parser('self-check')
    sub.add_parser('qualify').add_argument('--release',type=Path,required=True)
    fitting=sub.add_parser('fit'); fitting.add_argument('--release',type=Path,required=True); fitting.add_argument('--job',choices=JOBS,required=True)
    args=parser.parse_args(); self_check() if args.command=='self-check' else execute(args.release,'qualify01' if args.command=='qualify' else args.job)
