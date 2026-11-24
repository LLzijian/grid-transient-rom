"""Release-bound zero-sum fast closure qualification and fixed paired execution."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import torch
from experiments.osgkd_v2_train385_risk_alignment_20261123 import run as prior
from experiments.osgkd_v2_fast_zero_sum_20261124 import scoring
from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events
baseline,p1a=prior.baseline,prior.p1a
require,read,verified,identity,repo_path=prior.require,prior.read,prior.verified,prior.identity,prior.repo_path
RUN='osgkd_v2_fast_zero_sum_20261124'
PROTOCOL='campaigns/osgkd_v2/fast_zero_sum_protocol.md'
INPUTS=prior.INPUTS|{'parent_summary':'results/osgkd_v2_train385_risk_alignment_20261123/pair01/summary.json'}
SOURCES=prior.SOURCES|{f'experiments/{RUN}/run.py',f'experiments/{RUN}/scoring.py',PROTOCOL,
 'designs/osgkd_v2_fast_zero_sum_20261124.md','src/kiki/models/osgkd_fast_zero_sum.py'}
PARENT_PATH='artifacts/osgkd_v2_train385_risk_alignment_20261123/pair01/control/attempt_2060.pt'
PARENT_SHA='77c5907f3097d92c97a04e869ffce0591d8b7be4a6b3f752bc9f5d114369d683'
ATTEMPTS,CAP,QUALIFY_CAP=200,10_240_000,170_000
COMMON_KEYS={'run_id','job_id','released','device','threads','recompute','source_hashes','input_refs','checkpoint_ref'}


def multiplier(k):
    require(type(k) is int and 0<=k<200,'invalid attempt')
    return .15+.175*(1+math.cos(math.pi*k/199))


def fast_multiplier(k): return 1+.75*(1+math.cos(math.pi*min(k,40)/40))


def plan(events):
    stable=sorted(e['id'] for e in events if e['label']=='stable')
    unstable=sorted(e['id'] for e in events if e['label']=='unstable')
    require((len(stable),len(unstable))==(342,43) and len(set(stable+unstable))==385,'train population differs')
    s=np.random.default_rng(24101).permutation(stable).tolist()
    u=np.random.default_rng(24102).permutation(unstable).tolist()
    batches=[[s[(7*k+j)%342] for j in range(7)]+[u[k%43]] for k in range(200)]
    require(all(len(set(b))==8 for b in batches) and set(x for b in batches for x in b)==set(stable+unstable),'batch coverage differs')
    return dict(seeds=dict(stable=24101,unstable=24102,fast=24103),batches=batches,coverage=dict(Counter(x for b in batches for x in b)))


def restore(saved,candidate=False):
    parent=OSGKD.from_checkpoint(saved['model']).cpu().double()
    original=p1a.old.new_optimizer(parent)
    original.load_state_dict(copy.deepcopy(saved['optimizer']))
    if not candidate: return parent,original
    from kiki.models.osgkd_fast_zero_sum import OSGKDFastZeroSum
    model=OSGKDFastZeroSum.from_parent_checkpoint(saved['model'],fast_seed=24103).cpu().double()
    names={id(p):n for n,p in parent.named_parameters()}; mapped=dict(model.named_parameters())
    groups=[{**{k:v for k,v in g.items() if k!='params'},'params':[mapped[names[id(p)]] for p in g['params']]} for g in original.param_groups]
    opt=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8,weight_decay=0)
    opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    opt.add_param_group({'params':list(model.fast_parameters()),'lr':.0003})
    require(not any(p in opt.state for p in model.fast_parameters()),'new Adam group must start empty')
    return model,opt


def components(model,events,constants,work,finite):
    th,om=[torch.stack([e['history'][key] for e in events]) for key in ('theta','omega')]
    initial=model.initialize(th,om); work.encoder_events+=len(events)
    groups=defaultdict(list); rows=[None]*len(events)
    for i,e in enumerate(events):
        count=min(800,len(e['targets']['theta'])-1); require(count>0,'missing future'); groups[count].append(i)
    scale=torch.as_tensor(constants['S_m'],dtype=torch.float64)
    for count,indices in groups.items():
        path=scoring.integrate(model,initial[indices],count,work=work); ports=model.ports(path[:,1:])
        for local,index in enumerate(indices):
            e=events[index]; target=e['targets']; finite.append(torch.isfinite(path[local]).all())
            motion=prior.segmented_motion(path[local],target,model.mass,model.omega_scale,constants['s_coi'])
            terms=[]
            for key,pred,s in (('tau_m',ports.tau_m[local],scale),('tau_e',ports.tau_e[local],model.Se)):
                mask=e['masks'][key][1:count+1]
                if mask.any(): terms.append(((pred-target[key][1:count+1])/s).square()[mask].mean())
            port=torch.stack(terms).mean() if terms else motion.new_zeros(())
            consistency=motion.new_zeros(())
            if count>=10:
                encoded=model.initialize(target['theta'][:11],target['omega'][:11]); work.encoder_events+=1
                consistency=((path[local,10,2*model.n:5*model.n]-encoded[2*model.n:5*model.n])/model.init_bounds).square().mean()
            rows[index]=torch.stack((motion,port,consistency))
    return torch.stack(rows)


def objective(comp): return (comp[:,0]+.1*comp[:,1]+.01*comp[:,2]).mean()


def attempt(model,opt,batch,k,constants,work):
    for g,base in zip(opt.param_groups[:2],(.001,.0003),strict=True): g['lr']=base*multiplier(k)
    if len(opt.param_groups)==3: opt.param_groups[2]['lr']=.0003*multiplier(k)*fast_multiplier(k)
    opt.zero_grad(set_to_none=True); finite=[]; comp=components(model,batch,constants,work,finite); loss=objective(comp)
    value=float(loss.detach()); row=dict(attempt=2061+k,index=k,event_ids=[e['id'] for e in batch],accepted=False,
        loss=value if math.isfinite(value) else None,shared_norm=None,fast_norm=None,lr_multiplier=multiplier(k),fast_lr_multiplier=fast_multiplier(k))
    if not bool(torch.stack(finite).all()) or not bool(torch.isfinite(loss)): return {**row,'rejection':'nonfinite_forward'}
    loss.backward(); shared=[p for g in opt.param_groups[:2] for p in g['params']]
    norm=float(torch.nn.utils.clip_grad_norm_(shared,5.)); row['shared_norm']=norm if math.isfinite(norm) else None
    fast=0.
    if len(opt.param_groups)==3:
        fast=float(torch.nn.utils.clip_grad_norm_(opt.param_groups[2]['params'],5.)); row['fast_norm']=fast if math.isfinite(fast) else None
    if not math.isfinite(norm) or not math.isfinite(fast): return {**row,'rejection':'nonfinite_gradient'}
    opt.step(); require(prior.finite_tree(model.state_dict()) and prior.finite_tree(opt.state_dict()),'nonfinite model or Adam: immediate stop')
    with torch.no_grad(): model.project_parameters_()
    require(prior.finite_tree(model.state_dict()),'nonfinite projected model')
    return {**row,'accepted':True,'shared_clipped':norm>5,'fast_clipped':fast>5,
            'fast_gamma_norm':float(model.fast_gamma.detach().norm()) if hasattr(model,'fast_gamma') else None}


def validate(release,job):
    require(set(release)==COMMON_KEYS|({'qualification_ref'} if job=='pair01' else set()),'exact release keys required')
    require(tuple(release[k] for k in ('run_id','job_id','released','device','threads','recompute'))==(RUN,job,True,'cpu',1,False),'release identity differs')
    require(set(release['source_hashes'])==SOURCES,'exact source coverage required')
    for name,sha in release['source_hashes'].items(): require(identity(repo_path(name))['sha256']==sha,'source differs: '+name)
    require(set(release['input_refs'])==set(INPUTS),'exact input references required')
    inputs={}
    for key,path in INPUTS.items():
        require(release['input_refs'][key]['path']==path,'input path differs'); inputs[key]=read(release['input_refs'][key])
    prior.validate_baseline(inputs['baseline_summary'],inputs['baseline_archive'],release['input_refs']['baseline_summary'])
    parent=inputs['parent_summary']
    require(parent['status']=='completed' and parent['run_id']==prior.RUN and parent['job_id']=='pair01' and
            parent['result']['fixed_endpoint']==2060 and not parent['result']['stopped_early'],'completed fixed parent required')
    prior.validate(parent['release'])
    ref=release['checkpoint_ref']
    require(ref==parent['result']['treatments']['control']['checkpoint'] and ref['path']==PARENT_PATH and ref['sha256']==PARENT_SHA,'exact control2060 parent required')
    verified(ref)
    if job=='pair01':
        qref=release['qualification_ref']; require(qref['path']==f'results/{RUN}/qualify01/summary.json','canonical qualification required')
        q=read(qref)
        require(q['status']=='completed' and q['run_id']==RUN and q['job_id']=='qualify01' and q['result']['passed'] is True,'passed qualification required')
        for key in COMMON_KEYS-{'job_id'}: require(q['release'][key]==release[key],'qualification binding differs: '+key)
    return inputs


def load_bound(release,inputs):
    from kiki.data.osgkd_train_val import load_train
    raw,constants=load_train()
    require(baseline.data_identity(raw)==inputs['baseline_summary']['result']['train']['data_identity'],'train digest differs')
    saved=torch.load(verified(release['checkpoint_ref']),map_location='cpu',weights_only=True)
    require(saved['attempt']==2060 and saved['model']['config']['variant']=='C1','parent checkpoint content differs')
    model=OSGKD.from_checkpoint(saved['model'])
    for key in ('mass','damping','F','Sp','Se','omega_scale','theta_ref'):
        expected=np.asarray(constants[key])
        if key in ('mass','damping'): expected=np.diag(expected)
        require(np.array_equal(getattr(model,key).detach().numpy(),expected),'frozen parent constant differs: '+key)
    require(model.n==10,'ten-machine parent required')
    return raw,constants,saved


def bits(a,b): return torch.equal(a.detach().contiguous().view(torch.int64),b.detach().contiguous().view(torch.int64))


def qualify(saved,raw,constants,sequence,context=None):
    a,ao=restore(saved); b,bo=restore(saved,True); work=IntegrationWork()
    if context is not None: context.update(phase='qualification',work={'qualification':work})
    lookup={e['id']:e for e in raw}; first=[lookup[x] for x in sequence['batches'][0]]
    for event in (first[0],first[-1]):
        th,om=[torch.as_tensor(event['history'][key],dtype=torch.float64) for key in ('theta','omega')]
        x=a.initialize(th,om); y=b.initialize(th,om); work.encoder_events+=2
        require(bits(x,y[:50]),'initial prefix differs')
        require(bits(a.rhs(None,x),b.rhs(None,y)[:50]),'RHS prefix differs'); work.rhs['forward']+=2
        for key in ('tau_m','tau_e','Fc','tau_env'): require(bits(getattr(a.ports(x),key),getattr(b.ports(y),key)),'readout prefix differs')
        count=min(25,len(event['t'])-1)
        for h in (.005,.0025): require(bits(scoring.integrate(a,x,count,internal_step=h,work=work),scoring.integrate(b,y,count,internal_step=h,work=work)[:,:50]),'trajectory prefix differs')
    batch=tensor_events(first,'cpu',max_steps=800)
    ca=components(a,batch,constants,work,[]); cb=components(b,batch,constants,work,[])
    la,lb=objective(ca),objective(cb)
    require(torch.allclose(la,lb,atol=1e-12,rtol=1e-12),'first loss differs')
    la.backward(); lb.backward()
    shared=dict(a.named_parameters()); new=dict(b.named_parameters())
    for name,p in shared.items():
        q=new[name]
        require((p.grad is None)==(q.grad is None),'shared gradient availability differs')
        if p.grad is not None: require(torch.allclose(p.grad,q.grad,atol=1e-12,rtol=1e-12),'shared gradient differs: '+name)
    require(b.fast_gamma.grad is not None and bool(torch.isfinite(b.fast_gamma.grad).all()) and float(b.fast_gamma.grad.norm())>0,'fast gate gradient must be usable')
    gate_norm=float(b.fast_gamma.grad.norm())
    for opt in (ao,bo):
        for g,base in zip(opt.param_groups[:2],(.001,.0003),strict=True): g['lr']=base*multiplier(0)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups[:2] for p in g['params']],5.)
    bo.param_groups[2]['lr']=.0003*multiplier(0)*fast_multiplier(0)
    torch.nn.utils.clip_grad_norm_(bo.param_groups[2]['params'],5.)
    ao.step(); bo.step()
    with torch.no_grad(): a.project_parameters_(); b.project_parameters_()
    for name,p in a.named_parameters(): require(torch.allclose(p,new[name],atol=1e-12,rtol=1e-12),'discarded shared update differs')
    for ga,gb in zip(ao.param_groups,bo.param_groups[:2],strict=True):
        for pa,pb in zip(ga['params'],gb['params'],strict=True):
            sa,sb=ao.state[pa],bo.state[pb]
            require(sa.keys()==sb.keys(),'shared Adam state keys differ')
            for key in sa:
                if isinstance(sa[key],torch.Tensor): require(torch.allclose(sa[key],sb[key],atol=1e-12,rtol=1e-12),'shared Adam moments differ')
                else: require(sa[key]==sb[key],'shared Adam metadata differs')
    bo.zero_grad(set_to_none=True); objective(components(b,batch,constants,work,[])).backward()
    gradients={}
    for name,p in b.named_parameters():
        if name not in shared:
            require(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm())>0,'fast parameter gradient unusable: '+name)
            gradients[name]=float(p.grad.norm())
    state=b.initialize(batch[0]['history']['theta'],batch[0]['history']['omega']).detach().requires_grad_(True)
    ports=b.ports(state); force=ports.F_fast
    require(float(force.abs().max())>0,'active branch required')
    zero=float(force.sum().abs()); direct=float(((-force)@b.inverse_mass.T @ b.mass.diag()/b.mass.diag().sum()).abs())
    ledger=b.storage_power(state); derivative=torch.autograd.grad(ledger.H,state,create_graph=False)[0]@b.rhs(None,state)
    work.rhs['forward']+=1
    residual=float((derivative-ledger.total_power).detach().abs())
    require(zero<1e-12 and direct<1e-12 and residual<1e-10,'active physical ledger failed')
    require(work.rhs['forward']<=QUALIFY_CAP,'qualification RHS cap exceeded')
    return dict(passed=True,official_updates=0,discarded_updates_per_group=1,gate_gradient_norm=gate_norm,
                fast_gradient_norms=gradients,zero_sum_residual=zero,direct_coi_residual=direct,power_residual=residual,work=work.snapshot())


def checkpoint(path,state,k,release,plan_ref):
    payload=dict(attempt=2061+k,model=state['model'].checkpoint(),optimizer=state['opt'].state_dict(),torch_rng=state['rng'],
                 completed_pairs=k+1,release=release,plan_ref=plan_ref,work=state['work'].snapshot(),rejected=state['rejected'])
    require(not path.exists(),'checkpoint exists')
    temp=path.with_suffix('.pt.tmp')
    with temp.open('xb') as stream: torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    os.rename(temp,path); return identity(path)


@torch.no_grad()
def short_diagnostics(models,events,constants,inputs,directory,*,work=None):
    profile=read(inputs['conflict_profile']['result']['rows_ref'])
    misses=profile['cohort_event_ids']['parent_misses']
    selected={r['event_id']:r['selected_machine'] for r in profile['signed_electrical'] if all(v['persistent_80pct'] is True for v in r['endpoints'].values())}
    require(len(set(misses))==27 and len(selected)==22 and set(selected)<=set(misses),'fixed short cohorts differ')
    lookup={e['id']:e for e in events}; rows=[]; work=IntegrationWork() if work is None else work
    for eid in misses:
        e=lookup[eid]; stop=min(25,len(e['t'])-1); chosen=selected.get(eid)
        row=dict(event_id=eid,selected_machine=chosen,models={})
        for name,model in models.items():
            th,om=[torch.as_tensor(e['history'][key],dtype=torch.float64) for key in ('theta','omega')]
            initial=model.initialize(th,om); work.encoder_events+=1
            main=scoring.integrate(model,initial,stop,work=work)
            fine=scoring.integrate(model,initial,stop,internal_step=.0025,work=work)
            ports=model.ports(fine); windows={}
            for label,count in (('0.03',3),('capped_0.25',stop)):
                if count>stop:
                    windows[label]=dict(complete=False,numerical_qualified=False,reason='short_record'); continue
                gate=scoring.numerical_gate(main[:count+1,:50].numpy(),fine[:count+1,:50].numpy(),constants['omega_scale'])
                fg=scoring.fast_gate(main[:count+1,50:].numpy(),fine[:count+1,50:].numpy()) if fine.shape[-1]>50 else None
                numerical=gate['passed'] and (fg is None or fg['passed'])
                total=ports.tau_e[1:count+1].numpy(); force=ports.F_fast[1:count+1].numpy() if hasattr(ports,'F_fast') else np.zeros_like(total)
                complete=bool(np.isfinite(total).all() and np.asarray(e['masks']['tau_e'][1:count+1]).all())
                item=dict(complete=complete,numerical_qualified=numerical,numerical_gate=gate,fast_numerical_gate=fg,points=count,end_s=.01*count)
                if complete:
                    error=total-e['targets']['tau_e'][1:count+1]
                    item.update(electrical_normalized_mse=float(np.mean((error/constants['Se'])**2)),
                        geometry=prior.electrical.decompose(error,np.asarray(constants['Se']),np.asarray(constants['mass']),chosen),
                        same_state_without_fast_geometry=prior.electrical.decompose(error-force,np.asarray(constants['Se']),np.asarray(constants['mass']),chosen),
                        same_state_without_fast_electrical_normalized_mse=float(np.mean(((error-force)/constants['Se'])**2)),
                        fast_force_rms=float(np.sqrt(np.mean(force**2))),fast_force_max=float(np.abs(force).max()),
                        fast_force_zero_sum_max=float(np.abs(force.sum(-1)).max()))
                windows[label]=item
            env=fine[:,40:50]; coeff=model.memory_coeff
            rate=-(coeff[0]+coeff[3]*(1-env.tanh().square()))
            row['models'][name]=dict(windows=windows,environment_local_rate_min=float(rate.min()),environment_local_rate_max=float(rate.max()),
                fast_rate_min=float(model.fast_rate.min()) if hasattr(model,'fast_rate') else None,
                fast_rate_max=float(model.fast_rate.max()) if hasattr(model,'fast_rate') else None)
        rows.append(row)
    aggregates={}
    for name in models:
        aggregates[name]={}
        for window in ('0.03','capped_0.25'):
            aggregates[name][window]={}
            for cohort,ids in (('parent_misses',set(misses)),('persistent_22',set(selected))):
                items=[r['models'][name]['windows'][window] for r in rows if r['event_id'] in ids]
                good=[x for x in items if x['complete'] and x['numerical_qualified']]
                mean=lambda values:float(np.mean(values)) if values else None
                aggregates[name][window][cohort]=dict(raw_denominator=len(ids),complete_events=sum(x['complete'] for x in items),
                    numerical_qualified_events=len(good),event_equal_electrical_normalized_mse=mean([x['electrical_normalized_mse'] for x in good]),
                    same_state_without_fast_electrical_normalized_mse=mean([x['same_state_without_fast_electrical_normalized_mse'] for x in good]),
                    selected_signed_mean=mean([x['geometry']['selected_total'] for x in good if 'selected_total' in x['geometry']]),
                    same_state_without_fast_selected_signed_mean=mean([x['same_state_without_fast_geometry']['selected_total'] for x in good if 'selected_total' in x['geometry']]),
                    same_state_selected_signed_direct_difference=mean([x['geometry']['selected_total']-x['same_state_without_fast_geometry']['selected_total'] for x in good if 'selected_total' in x['geometry']]),
                    coi_geometry={key:mean([x['geometry'][key] for x in good]) for key in ('accel_common_M','accel_relative_M','accel_total_M')},
                    same_state_without_fast_coi_geometry={key:mean([x['same_state_without_fast_geometry'][key] for x in good]) for key in ('accel_common_M','accel_relative_M','accel_total_M')},
                    same_state_coi_geometry_direct_difference={key:mean([x['geometry'][key]-x['same_state_without_fast_geometry'][key] for x in good]) for key in ('accel_common_M','accel_relative_M','accel_total_M')},
                    fast_force_zero_sum_max=max((x['fast_force_zero_sum_max'] for x in good),default=None),
                    fast_force_rms_event_mean=mean([x['fast_force_rms'] for x in good]),fast_force_max=max((x['fast_force_max'] for x in good),default=None))
        aggregates[name]['rates']={key:min(r['models'][name][key] for r in rows) if key.endswith('min') else max(r['models'][name][key] for r in rows)
            for key in ('environment_local_rate_min','environment_local_rate_max')}
        if hasattr(models[name],'fast_rate'): aggregates[name]['rates'].update(fast_min=float(models[name].fast_rate.min()),fast_max=float(models[name].fast_rate.max()))
    require(work.rhs['forward']<=32_400 and work.rhs['recompute']==0,'short diagnostic cap exceeded')
    return dict(rows_ref=baseline.atomic_write(directory/'short_rows.json',rows),aggregates=aggregates,work=work.snapshot(),
                interpretation='F_fast removed only from readout at identical saved candidate states; not a dynamical ablation')


def pair(saved,raw,constants,inputs,release,local,context=None):
    sequence=plan(raw); plan_ref=baseline.atomic_write(local/'batch_plan.json',sequence)
    fit=tensor_events(raw,'cpu',max_steps=800); lookup={e['id']:e for e in fit}; states={}
    for name in ('control','candidate'):
        model,opt=restore(saved,name=='candidate'); target=local/name; target.mkdir()
        states[name]=dict(model=model,opt=opt,target=target,rng=saved['torch_rng'].clone(),work=IntegrationWork(),rows=[],rejected=0,consecutive=0)
    if context is not None: context.update(phase='training',states=states,work={name:s['work'] for name,s in states.items()})
    stopped=False; progress=[]
    for k in range(200):
        for name,s in states.items():
            if context is not None: context.update(active_treatment=name,attempt=2061+k)
            torch.set_rng_state(s['rng']); batch=[lookup[x] for x in sequence['batches'][k]]
            expected=8*sum(min(800,len(e['targets']['theta'])-1) for e in batch)
            before=s['work'].rhs['forward']; require(before+expected<=CAP,'training cap would exceed')
            started,cpu=perf_counter(),process_time(); row=attempt(s['model'],s['opt'],batch,k,constants,s['work'])
            require(s['work'].rhs['forward']-before==expected,'actual training RHS differs')
            row.update(wall_s=perf_counter()-started,cpu_s=process_time()-cpu)
            s['rows'].append(row); s['rng']=torch.get_rng_state().clone(); s['rejected']+=not row['accepted']
            s['consecutive']=0 if row['accepted'] else s['consecutive']+1
            baseline.atomic_write(s['target']/f'log_{k+1:03d}.json',row)
            stopped|=s['rejected']>=5 or s['consecutive']>=3
        if (k+1)%20==0 or stopped:
            refs={name:checkpoint(s['target']/f'attempt_{2061+k}.pt',s,k,release,plan_ref) for name,s in states.items()}
            for name,s in states.items(): s['checkpoint']=refs[name]
            progress.append(baseline.atomic_write(local/f'progress_{k+1:03d}.json',dict(completed_pairs=k+1,checkpoints=refs,stopped_early=stopped)))
            print(json.dumps(dict(completed_pairs=k+1,stopped_early=stopped)),flush=True)
        if stopped: break
    accepted={name:len(s['rows'])-s['rejected'] for name,s in states.items()}
    result=dict(fixed_endpoint=None if stopped else 2260,actual_endpoint=2061+k,stopped_early=stopped,paired_attempts=k+1,
                valid_endpoint=not stopped and min(accepted.values())>=196 and abs(accepted['control']-accepted['candidate'])<=2,
                plan_ref=plan_ref,progress_refs={str(i):r for i,r in enumerate(progress)},new_fom=0,treatments={})
    for name,s in states.items():
        result['treatments'][name]=dict(attempts=k+1,accepted=accepted[name],rejected=s['rejected'],checkpoint=s['checkpoint'],
            work=s['work'].snapshot(),shared_clipped=sum(r.get('shared_clipped',False) for r in s['rows']),fast_clipped=sum(r.get('fast_clipped',False) for r in s['rows']),
            attempts_ref=baseline.atomic_write(s['target']/'attempts.json',s['rows']),parameters=sum(p.numel() for p in s['model'].parameters()),
            state_dimension=60 if name=='candidate' else 50,train_cost=dict(wall_s=sum(r['wall_s'] for r in s['rows']),cpu_s=sum(r['cpu_s'] for r in s['rows'])))
    if not stopped:
        from kiki.data.osgkd_train_val import load_val
        val=load_val(constants)
        require(baseline.data_identity(val)==inputs['baseline_summary']['result']['val']['data_identity'],'val digest differs')
        require(baseline.data_identity(raw)==inputs['baseline_summary']['result']['train']['data_identity'],'train mutated')
        for name,s in states.items():
            model=s['model'].eval().requires_grad_(False)
            result['treatments'][name]['evaluation']={}
            for split,events in (('train',raw),('val',val)):
                evaluation_work=IntegrationWork()
                if context is not None:
                    context.update(phase='evaluation',active_treatment=name,active_split=split)
                    context['work'][name+'_'+split]=evaluation_work
                result['treatments'][name]['evaluation'][split]=scoring.score_split(model,events,constants,split,s['target']/split,work=evaluation_work)
            require(sum(result['treatments'][name]['evaluation'][split]['summary']['work']['rhs_forward'] for split in ('train','val'))<=10_370_400,'evaluation cap exceeded')
        short_work=IntegrationWork()
        if context is not None: context.update(phase='short_diagnostic'); context['work']['short']=short_work
        result['short_diagnostic']=short_diagnostics({n:s['model'] for n,s in states.items()},raw,constants,inputs,local,work=short_work)
    return result


def execute(path,job):
    started,cpu=perf_counter(),process_time(); path=path.resolve(); require(path.is_relative_to(ROOT),'repository release required')
    release=json.loads(path.read_bytes()); inputs=validate(release,job)
    local,public=[ROOT/side/RUN/job for side in ('artifacts','results')]
    require(not local.exists() and not public.exists(),'job already claimed')
    local.mkdir(parents=True,exist_ok=False); public.mkdir(parents=True,exist_ok=False)
    receipt=dict(run_id=RUN,job_id=job,status='started',release=release,release_ref=identity(path),pid=os.getpid(),
                 started_utc=datetime.now(timezone.utc).isoformat(),environment=dict(device='cpu',dtype='float64',threads=1,torch=str(torch.__version__),numpy=np.__version__,python=sys.version))
    baseline.atomic_write(local/'started.json',receipt)
    context=dict(phase='loading',work={})
    try:
        torch.set_num_threads(1); raw,constants,saved=load_bound(release,inputs)
        result=qualify(saved,raw,constants,plan(raw),context) if job=='qualify01' else pair(saved,raw,constants,inputs,release,local,context)
        receipt.update(status='completed',result=result,completed_utc=datetime.now(timezone.utc).isoformat(),cost=dict(wall_s=perf_counter()-started,cpu_s=process_time()-cpu))
        ref=baseline.atomic_write(public/'summary.json',receipt,public=True); baseline.atomic_write(local/'completed.json',dict(status='completed',summary_ref=ref))
    except BaseException as error:
        baseline.atomic_write(local/'failed.json',dict(status='failed',error=repr(error),
            phase=context['phase'],active_treatment=context.get('active_treatment'),active_split=context.get('active_split'),attempt=context.get('attempt'),
            cost=dict(wall_s=perf_counter()-started,cpu_s=process_time()-cpu),work={key:value.snapshot() for key,value in context['work'].items()},
            treatments={name:dict(recorded_attempts=len(s['rows']),accepted=sum(r['accepted'] for r in s['rows']),rejected=s['rejected'],
                model_finite=prior.finite_tree(s['model'].state_dict()),adam_finite=prior.finite_tree(s['opt'].state_dict())) for name,s in context.get('states',{}).items()},
            unmeasured_cost=None)); raise


def self_check():
    torch.set_num_threads(1)
    events=[dict(id=str(i),label='stable' if i<342 else 'unstable') for i in range(385)]
    assert plan(events)==plan(events)
    assert multiplier(0)==.5 and multiplier(199)==.15 and fast_multiplier(0)==2.5 and fast_multiplier(40)==1.
    constants={k:np.ones(2) for k in ('mass','damping','Sp','Se','omega_scale','S_m')}
    constants.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]),s_coi=2.)
    model=p1a.old.new_model('C1',constants,torch.device('cpu'))
    from kiki.models.osgkd_fitting import integrate as original
    for dim in (5,6):
        class Linear:
            n=2
            def rhs(self,t,x): return -.3*x
        state=torch.arange(dim*2*3,dtype=torch.float64).reshape(3,dim*2); work=IntegrationWork()
        result=scoring.integrate(Linear(),state,3,work=work)
        assert work.rhs['forward']==3*3*8 and result.shape==(3,4,dim*2)
    state=torch.zeros((2,10),dtype=torch.float64)
    assert bits(original(model,state,3),scoring.integrate(model,state,3))
    assert scoring.fast_gate(np.zeros((4,10)),np.zeros((4,10)))['passed']
    assert not scoring.fast_gate(np.zeros((4,10)),np.ones((4,10)))['passed']
    rng=np.random.default_rng(24103)
    event=dict(id='synthetic',history={k:rng.normal(0,.02,(11,2)) for k in ('theta','omega')},t=np.arange(12)*.01,
               targets={k:rng.normal(0,.02,(12,2)) for k in ('theta','omega','tau_m','tau_e')},
               masks={k:np.ones((12,2),bool) for k in ('theta','omega','tau_m','tau_e')})
    batch=tensor_events([event],'cpu',max_steps=800)
    opt=p1a.old.new_optimizer(model)
    attempt(model,opt,batch,0,constants,IntegrationWork())
    saved=dict(model=model.checkpoint(),optimizer=copy.deepcopy(opt.state_dict()),torch_rng=torch.get_rng_state().clone())
    a,ao=restore(saved); b,bo=restore(saved,True)
    ca=components(a,batch,constants,IntegrationWork(),[]); cb=components(b,batch,constants,IntegrationWork(),[])
    assert bits(ca,cb)
    la,lb=objective(ca),objective(cb); la.backward(); lb.backward()
    mapped=dict(b.named_parameters())
    for name,p in a.named_parameters():
        if p.grad is not None: assert torch.allclose(p.grad,mapped[name].grad,atol=1e-12,rtol=1e-12),name
    assert b.fast_gamma.grad is not None and b.fast_gamma.grad.norm()>0
    rowa=attempt(a,ao,batch,0,constants,IntegrationWork()); rowb=attempt(b,bo,batch,0,constants,IntegrationWork())
    assert rowa['accepted'] and rowb['accepted']
    for name,p in a.named_parameters(): assert torch.allclose(p,mapped[name],atol=1e-12,rtol=1e-12),name
    bo.zero_grad(set_to_none=True); objective(components(b,batch,constants,IntegrationWork(),[])).backward()
    for p in b.fast_parameters(): assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0
    x=b.initialize(batch[0]['history']['theta'],batch[0]['history']['omega']).detach().requires_grad_(True)
    ledger=b.storage_power(x); force=b.ports(x).F_fast
    derivative=torch.autograd.grad(ledger.H,x)[0]@b.rhs(None,x)
    assert force.abs().max()>0 and force.sum().abs()<1e-12
    assert torch.allclose(derivative,ledger.total_power,atol=1e-10,rtol=1e-10)
    print(json.dumps(dict(self_check='passed',scientific_payloads_opened=False,actual_batch_rhs=True,old_rk4_bit_parity=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check')
    for command in ('qualify','pair'): sub.add_parser(command).add_argument('--release',type=Path,required=True)
    args=parser.parse_args()
    self_check() if args.command=='self-check' else execute(args.release,'qualify01' if args.command=='qualify' else 'pair01')
