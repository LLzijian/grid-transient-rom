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
from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import fit
from experiments.osgkd_v2_fast_zero_sum_20261124 import run as old, scoring
from experiments.osgkd_v2_train385_risk_alignment_20261123 import analyze
from kiki.evaluation.osgkd_autonomous import summarize_autonomous_8s
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events
baseline=old.baseline
require,read,verified,identity,repo_path=old.require,old.read,old.verified,old.identity,old.repo_path
RUN='osgkd_v2_p2a_equivariant_graph_20261124'
PROTOCOL='campaigns/osgkd_v2/p2a_equivariant_graph_initializer_protocol.md'
INPUTS={key:old.prior.INPUTS[key] for key in ('baseline_summary','baseline_archive')}|{'anchor_analysis':'results/osgkd_v2_sprint0_20261124/analysis.json'}
SOURCES=old.SOURCES|{f'experiments/{RUN}/run.py',f'experiments/{RUN}/fit.py',PROTOCOL,'designs/'+RUN+'.md','src/kiki/models/osgkd_graph.py','src/kiki/evaluation/osgkd_autonomous.py','tests/test_osgkd_graph.py','experiments/osgkd_v2_train385_risk_alignment_20261123/analyze.py'}
COMMON_KEYS={'run_id','job_id','released','device','threads','recompute','source_hashes','input_refs'}
ANCHOR_SHA='82b87d068a884143a95ccc800a133575a14ffa146d857753943039085028538c'

def validate(release,job):
    extra={'qualification_ref'} if job=='fit01' else {'fit_ref'} if job=='evaluate01' else set()
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
    require(release['input_refs']['anchor_analysis']['sha256']==ANCHOR_SHA and anchor['status']=='completed' and anchor['encoder_parameters']==6174,'Sprint0 identity differs')
    require(anchor['evaluator']==identity(ROOT/'src/kiki/evaluation/osgkd_autonomous.py'),'anchor evaluator identity differs')
    for split in ('train','val'):
        require(anchor['evaluation'][split]['data_identity']==inputs['baseline_summary']['result'][split]['data_identity'],'anchor data identity differs')
    if extra:
        key=next(iter(extra)); required_job='qualify02' if job=='fit01' else 'fit01'
        require(release[key]['path']==f'results/{RUN}/{required_job}/summary.json','canonical prerequisite required')
        previous=read(release[key]); require(previous['status']=='completed' and previous['run_id']==RUN and previous['job_id']==required_job,'completed prerequisite required')
        for name in COMMON_KEYS-{'job_id'}: require(previous['release'][name]==release[name],'prerequisite binding differs: '+name)
        require(previous['result']['passed'] if job=='fit01' else previous['result']['valid_endpoint'],'prerequisite did not pass')
        inputs['prerequisite']=previous
    return inputs

def load_train(inputs):
    from kiki.data.osgkd_train_val import load_train as loader
    events,constants=loader()
    require(baseline.data_identity(events)==inputs['anchor_analysis']['evaluation']['train']['data_identity'],'train digest differs')
    return events,constants

def checkpoint(path,model,opt,k,release,plan_ref,work,guard_work,rejected):
    payload=dict(attempt=k,model=model.checkpoint(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state().clone(),release=release,plan_ref=plan_ref,work=work.snapshot(),guard_work=guard_work.snapshot(),rejected=rejected)
    require(not path.exists(),'checkpoint exists'); temp=path.with_suffix('.pt.tmp')
    with temp.open('xb') as stream: torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    os.rename(temp,path); return identity(path)

def qualify(raw,constants,context):
    model=fit.new_model(constants); sequence=fit.plan(raw); lookup={e['id']:e for e in tensor_events(raw,'cpu',max_steps=800)}
    first=sequence['batches'][0]; work=IntegrationWork(); context.update(phase='qualification',work={'guard':work},model=model,opt=fit.new_optimizer(model))
    result=fit.guard(model,[lookup[first[0]],lookup[first[-1]]],constants,work)
    # A coarse failure selects the registered fine mesh; it does not invalidate finite fine qualification.
    return dict(passed=True,coarse_passed=result['passed'],training_internal_step=.01 if result['passed'] else .005,guard=result,official_updates=0,work=work.snapshot())

def train(raw,constants,release,inputs,local,context):
    sequence=fit.plan(raw); plan_ref=baseline.atomic_write(local/'batch_plan.json',sequence)
    lookup={e['id']:e for e in tensor_events(raw,'cpu',max_steps=800)}; model=fit.new_model(constants); opt=fit.new_optimizer(model)
    work,guard_work=IntegrationWork(),IntegrationWork(); rows=[]; guards=[]; refs={}; rejected=consecutive=0
    h=inputs['prerequisite']['result']['training_internal_step']
    context.update(phase='training',work={'training':work,'guard':guard_work},model=model,opt=opt,rows=rows)
    stopped=False
    for k,bids in enumerate(sequence['batches']):
        context.update(attempt=k+1); batch=[lookup[x] for x in bids]
        if k in (512,896):
            context['phase']='stage_guard'; g=fit.guard(model,[batch[0],batch[-1]],constants,guard_work,limits=(fit.horizon(k),)); guards.append(dict(attempt=k+1,**g))
            if not g['passed']: h=.005
            context['phase']='training'
        expected=round(.01/h)*4*sum(min(fit.horizon(k),len(e['targets']['theta'])-1) for e in batch)
        require(work.rhs['forward']+expected<=fit.CAP,'training cap would exceed')
        started,cpu=perf_counter(),process_time(); before=work.rhs['forward']; row=fit.attempt(model,opt,batch,k,constants,work,h)
        require(work.rhs['forward']-before==expected,'training RHS accounting differs')
        row.update(wall_s=perf_counter()-started,cpu_s=process_time()-cpu); rows.append(row)
        rejected+=not row['accepted']; consecutive=0 if row['accepted'] else consecutive+1
        baseline.atomic_write(local/f'attempt_{k+1:04d}.json',row); stopped=consecutive>=3 or rejected>=16
        if (k+1)%64==0 or stopped:
            ref=checkpoint(local/f'attempt_{k+1}.pt',model,opt,k+1,release,plan_ref,work,guard_work,rejected)
            if k+1 in (1792,2048): refs[str(k+1)]=ref
            progress=dict(attempt=k+1,accepted=len(rows)-rejected,rejected=rejected,stopped_early=stopped,checkpoint=ref,work=work.snapshot(),guard_work=guard_work.snapshot(),training_internal_step=h,cost=dict(wall_s=sum(r['wall_s'] for r in rows),cpu_s=sum(r['cpu_s'] for r in rows)))
            baseline.atomic_write(local/f'progress_{k+1:04d}.json',progress); print(json.dumps(progress),flush=True)
        if stopped: break
    return dict(fixed_endpoint=2048 if not stopped else None,actual_endpoint=k+1,attempts=k+1,accepted=len(rows)-rejected,rejected=rejected,valid_endpoint=not stopped and k==2047 and len(rows)-rejected>=2033,stopped_early=stopped,checkpoints=refs,plan_ref=plan_ref,attempts_ref=baseline.atomic_write(local/'attempts.json',rows),work=work.snapshot(),guard_work=guard_work.snapshot(),guards=guards,train_cost=dict(wall_s=sum(r['wall_s'] for r in rows),cpu_s=sum(r['cpu_s'] for r in rows)),encoder_parameters=1928,state_dimension=50)

def score(model,events,constants,split,directory,context):
    work=IntegrationWork(); context.update(phase='evaluation',active_split=split); context['work'][str(directory)]=work
    result=scoring.score_split(model,events,constants,split,directory,work=work)
    rows=read(result['rows_ref']); metrics=read(result['metrics_ref']); autonomous=summarize_autonomous_8s(rows)
    qualified=sum(r['actual_future_points'] is not None and r['classification_8s'] is not None and r['classification_8s']['los_pair_pass'] for r in rows)
    risk=metrics['splits'][split]['stable_8s']['official']; risk=None if risk is None else risk['motion_risk_v1']
    expected=sum(24*(max(800,min(1500,len(e['t'])-1)) if e['label']=='unstable' else 800) for e in events if e['history'] is not None)
    require(work.rhs['forward']==expected,'evaluation workload differs')
    gate=qualified==(107 if split=='val' else 385) and analyze.qualification(rows,split)
    result.update(autonomous=autonomous,stable_8s_motion_risk=risk,numerical_qualified_events=qualified,gate_passed=gate)
    return result

def evaluate(raw,constants,release,inputs,local,context):
    completed=inputs['prerequisite']['result']; require(completed['attempts']==2048 and completed['accepted']>=2033 and set(completed['checkpoints'])=={'1792','2048'},'fixed fit incomplete')
    anchor=inputs['anchor_analysis']
    for split in ('train','val'):
        previous=anchor['evaluation'][split]; rows=read(previous['rows_ref']); metrics=read(previous['metrics_ref'])
        require(analyze.qualification(rows,split),'anchor score qualification differs')
        require(summarize_autonomous_8s(rows)=={key:previous[key] for key in summarize_autonomous_8s(rows)},'anchor autonomous summary differs')
        require(metrics['splits'][split]['stable_8s']['official']['motion_risk_v1']==previous['stable_8s_motion_risk'],'anchor risk differs')
    from kiki.data.osgkd_train_val import load_val
    val=load_val(constants); require(baseline.data_identity(val)==inputs['anchor_analysis']['evaluation']['val']['data_identity'],'val digest differs')
    scores={}; eligible=[]; context['work']={}
    for endpoint in (1792,2048):
        saved=torch.load(verified(completed['checkpoints'][str(endpoint)]),map_location='cpu',weights_only=True)
        require(saved['attempt']==endpoint and saved['release']==inputs['prerequisite']['release'],'checkpoint binding differs')
        model=OSGKDGraph.from_checkpoint(saved['model']).eval().requires_grad_(False)
        result=score(model,val,constants,'val',local/f'val_{endpoint}',context); scores[str(endpoint)]=result
        auto=result['autonomous']; risk=result['stable_8s_motion_risk']
        passing=result['gate_passed'] and risk is not None and risk<=.55 and auto['stable_false_alarm_8s']['autonomous_trigger']<=1 and auto['unstable_recall_8s']['autonomous_trigger']>=8 and 1-1928/6174>=.60
        result['selection_gate_passed']=passing
        if passing: eligible.append(endpoint)
    selected=min(eligible,key=lambda e:(scores[str(e)]['stable_8s_motion_risk'],-scores[str(e)]['autonomous']['unstable_recall_8s']['autonomous_trigger'],e)) if eligible else None
    primary=2048 if selected is None else selected
    saved=torch.load(verified(completed['checkpoints'][str(primary)]),map_location='cpu',weights_only=True)
    model=OSGKDGraph.from_checkpoint(saved['model']).eval().requires_grad_(False)
    train_result=score(model,raw,constants,'train',local/f'train_{primary}',context)
    return dict(val=scores,primary_endpoint=primary,selected_endpoint=selected,selection_passed=selected is not None,train=train_result,adoptable=selected is not None and train_result['gate_passed'],result='selected' if selected is not None and train_result['gate_passed'] else 'no_selection' if selected is None else 'failure_train_numerical',anchor_analysis_ref=release['input_refs']['anchor_analysis'],new_fom=0)

def execute(path,job):
    started,cpu=perf_counter(),process_time(); path=path.resolve(); require(path.is_relative_to(ROOT),'repository release required')
    release=json.loads(path.read_bytes()); inputs=validate(release,job)
    local,public=[ROOT/side/RUN/job for side in ('artifacts','results')]; require(not local.exists() and not public.exists(),'job already claimed')
    local.mkdir(parents=True); public.mkdir(parents=True)
    receipt=dict(run_id=RUN,job_id=job,status='started',release=release,release_ref=identity(path),pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),environment=dict(device='cpu',dtype='float64',threads=1,recompute=False,torch=str(torch.__version__),numpy=np.__version__,python=sys.version))
    baseline.atomic_write(local/'started.json',receipt); context=dict(phase='loading',work={})
    try:
        torch.set_num_threads(1); raw,constants=load_train(inputs)
        result=qualify(raw,constants,context) if job=='qualify02' else train(raw,constants,release,inputs,local,context) if job=='fit01' else evaluate(raw,constants,release,inputs,local,context)
        receipt.update(status='completed',result=result,completed_utc=datetime.now(timezone.utc).isoformat(),cost=dict(wall_s=perf_counter()-started,cpu_s=process_time()-cpu))
        ref=baseline.atomic_write(public/'summary.json',receipt,public=True); baseline.atomic_write(local/'completed.json',dict(status='completed',summary_ref=ref))
    except BaseException as error:
        snapshot=None
        if 'model' in context:
            target=local/'failure_snapshot.pt'; temporary=local/'failure_snapshot.pt.tmp'
            with temporary.open('xb') as stream:
                torch.save(dict(attempt=context.get('attempt'),model=context['model'].checkpoint(),optimizer=context['opt'].state_dict(),torch_rng=torch.get_rng_state().clone()),stream); stream.flush(); os.fsync(stream.fileno())
            os.rename(temporary,target); snapshot=identity(target)
        baseline.atomic_write(local/'failed.json',dict(status='failed',error=repr(error),phase=context['phase'],attempt=context.get('attempt'),active_split=context.get('active_split'),snapshot_ref=snapshot,recorded_attempts=len(context.get('rows',[])),last_attempt=context.get('rows',[None])[-1] if context.get('rows') else None,model_finite=old.prior.finite_tree(context['model'].state_dict()) if 'model' in context else None,adam_finite=old.prior.finite_tree(context['opt'].state_dict()) if 'opt' in context else None,work={k:v.snapshot() for k,v in context['work'].items()},cost=dict(wall_s=perf_counter()-started,cpu_s=process_time()-cpu))); raise

def self_check():
    torch.set_num_threads(1)
    events=[dict(id=str(i),label='stable' if i<342 else 'unstable') for i in range(385)]
    assert fit.plan(events)==fit.plan(events) and [fit.horizon(k) for k in (0,511,512,895,896,2047)]==[200,200,400,400,800,800]
    assert fit.multiplier(0)==1. and fit.multiplier(2047)==.15
    constants={k:np.ones(2) for k in ('mass','damping','Sp','Se','omega_scale','S_m')}; constants.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]),s_coi=2.)
    k10={key:np.ones(10) for key in ('mass','damping','Sp','Se','omega_scale','S_m')}
    k10.update(F=np.eye(10),theta_ref=np.zeros(10),candidate_graph='undirected_complete_K10',s_coi=2.)
    canonical=fit.new_model(k10)
    assert canonical.n==10 and canonical.c.numel()==45
    k10['edge_index']=np.stack(np.triu_indices(10,k=1)); fit.new_model(k10)
    k10['edge_index']=k10['edge_index'][:,::-1].copy()
    try: fit.new_model(k10)
    except (ValueError,RuntimeError): pass
    else: raise AssertionError('noncanonical K10 ordering accepted')
    model=fit.new_model(constants); opt=fit.new_optimizer(model); rng=np.random.default_rng(1)
    event=dict(id='synthetic',history={k:rng.normal(0,.001,(11,2)) for k in ('theta','omega')},t=np.arange(12)*.01,targets={k:rng.normal(0,.001,(12,2)) for k in ('theta','omega','tau_m','tau_e')},masks={k:np.ones((12,2),bool) for k in ('theta','omega','tau_m','tau_e')})
    batch=tensor_events([event],'cpu',max_steps=800); work=IntegrationWork(); row=fit.attempt(model,opt,batch,0,constants,work,.01)
    assert row['accepted'] and work.rhs['forward']==44 and work.rhs['recompute']==0
    restored=OSGKDGraph.from_checkpoint(model.checkpoint()); assert old.prior.p1a.tree_equal(model.checkpoint(),restored.checkpoint())
    guard=fit.guard(model,batch,constants,IntegrationWork(),limits=(11,)); assert guard['checks'][0]['fine_finite']
    assert fit.gradient_compare(torch.zeros(2),torch.zeros(2))['passed']
    print(json.dumps(dict(self_check='passed',scientific_payloads_opened=False,encoder_parameters=1928,actual_batch_rhs=True)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True); sub.add_parser('self-check')
    for command in ('qualify','fit','evaluate'): sub.add_parser(command).add_argument('--release',type=Path,required=True)
    args=parser.parse_args(); self_check() if args.command=='self-check' else execute(args.release,{'qualify':'qualify02','fit':'fit01','evaluate':'evaluate01'}[args.command])
