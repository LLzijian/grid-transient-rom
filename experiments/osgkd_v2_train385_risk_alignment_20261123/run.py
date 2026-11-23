"""Release-bound fixed 200-pair strategy comparison; synthetic checks open no data."""
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
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import torch
from experiments.osgkd_v2_train385_baseline_20261123 import run as baseline
from experiments.osgkd_v2_p1a_20261123 import run as p1a
from experiments.osgkd_v2_electrical_decomposition_20261123 import run as electrical
from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_fitting import IntegrationWork, integrate, event_objective, joint_loss, tensor_events
from kiki.evaluation.osgkd import numerical_gate
identity, verified, read, require, repo_path = (baseline.identity, baseline.verified, baseline.read, baseline.require, baseline.repo_path)
RUN = 'osgkd_v2_train385_risk_alignment_20261123'
PROTOCOL = 'campaigns/osgkd_v2/train385_risk_alignment_protocol.md'
INPUTS = {'baseline_summary':'results/osgkd_v2_train385_baseline_20261123/eval02/summary.json',
          'baseline_archive':'harness/runs/osgkd_v2_train385_baseline_20261123.json',
          'conflict_profile':'results/osgkd_v2_conflict_profile_20261123/probe01/summary.json',
          'decomposition_summary':'results/osgkd_v2_electrical_decomposition_20261123/probe01/summary.json'}
SOURCES = baseline.SOURCES | {f'experiments/{RUN}/run.py', PROTOCOL,
    'designs/osgkd_v2_train_risk_alignment_20261123.md',
    'experiments/osgkd_v2_p1a_20261123/run.py', 'experiments/osgkd_fit297_20261113/run.py',
    'experiments/osgkd_v2_electrical_decomposition_20261123/run.py',
    'experiments/osgkd_v2_p0_20261123/run.py', 'src/kiki/models/osgkd_anchored_potential.py'}
ATTEMPTS, CAP = 200, 10_240_000
SEEDS = {'control':23101, 'candidate_stable':23102, 'candidate_unstable':23103}


def multiplier(k):
    require(type(k) is int and 0 <= k < ATTEMPTS, 'invalid attempt index')
    return .25 + .375 * (1 + math.cos(math.pi*k/199))


def plan(events):
    def cycle(ids, seed, take):
        order = np.random.default_rng(seed).permutation(ids).tolist()
        return [[order[(k*take+j) % len(order)] for j in range(take)] for k in range(ATTEMPTS)]
    ids = sorted(e['id'] for e in events)
    stable = sorted(e['id'] for e in events if e['label']=='stable')
    unstable = sorted(e['id'] for e in events if e['label']=='unstable')
    require(len(ids)==385 and len(set(ids))==385 and (len(stable),len(unstable))==(342,43), 'train population differs')
    a = cycle(ids, SEEDS['control'], 8)
    b = [s+u for s,u in zip(cycle(stable,SEEDS['candidate_stable'],6),cycle(unstable,SEEDS['candidate_unstable'],2),strict=True)]
    for batches in (a,b):
        require(all(len(set(batch))==8 for batch in batches), 'duplicate ID within batch')
    ca, cb = Counter(x for row in a for x in row), Counter(x for row in b for x in row)
    require(set(ca)==set(cb)==set(ids) and set(ca.values())=={4,5}, 'control coverage differs')
    require({cb[x] for x in stable}=={3,4} and {cb[x] for x in unstable}=={9,10}, 'candidate coverage differs')
    return {'seeds':SEEDS, 'cycle_rule':'one fixed seeded permutation repeated; immutable phase order stable then unstable',
            'control':a,'candidate':b,'coverage':{'control':dict(ca),'candidate':dict(cb)}}


def segmented_motion(path, target, mass, omega_scale, s_coi):
    mass = mass.diag() if mass.ndim == 2 else mass
    n = len(mass)
    error = path[1:,:2*n] - torch.cat((target['theta'][1:len(path)],target['omega'][1:len(path)]),dim=-1)
    weights = mass / mass.sum()
    angle, speed = error[:,:n], error[:,n:]
    ca, co = angle@weights, speed@weights
    point = .5*(angle-ca[:,None]).square().mean(-1) + .5*((speed-co[:,None])/omega_scale).square().mean(-1) + (co/s_coi).square()
    terms, weights = [], []
    for lo,hi,w in ((0,100,.4),(100,300,.3),(300,800,.3)):
        if lo < len(point):
            terms.append(point[lo:hi].mean()); weights.append(w)
    require(bool(terms), 'no available future')
    return sum(w*x for w,x in zip(weights,terms,strict=True))/sum(weights)


def candidate_components(model, events, scale, s_coi, work, path_finite):
    theta,omega = [torch.stack([e['history'][k] for e in events]) for k in ('theta','omega')]
    initial = model.initialize(theta,omega); work.encoder_events += len(events)
    groups = defaultdict(list)
    for i,e in enumerate(events):
        count = min(800,len(e['targets']['theta'])-1)
        require(count>=1, 'unavailable train future')
        groups[count].append(i)
    rows, finite = [None]*len(events), [None]*len(events)
    scale = torch.as_tensor(scale,dtype=torch.float64)
    require(scale.shape==(model.n,) and bool(torch.isfinite(scale).all() and (scale>0).all()), 'invalid mechanical scale')
    for count,indices in groups.items():
        path = integrate(model,initial[indices],count,work=work)
        ports = model.ports(path[:,1:])
        for local,index in enumerate(indices):
            e = events[index]; target=e['targets']
            finite[index] = torch.isfinite(path[local]).all()
            motion = segmented_motion(path[local],target,model.mass,model.omega_scale,s_coi)
            terms=[]
            for key,prediction,s in (('tau_m',ports.tau_m[local],scale),('tau_e',ports.tau_e[local],model.Se)):
                mask=e['masks'][key][1:count+1]
                if mask.any(): terms.append(((prediction-target[key][1:count+1])/s).square()[mask].mean())
            port_loss = torch.stack(terms).mean() if terms else motion.new_zeros(())
            consistency = motion.new_zeros(())
            if count>=10 and len(target['theta'])>=11:
                work.encoder_events += 1
                encoded=model.initialize(target['theta'][:11],target['omega'][:11])
                consistency=((path[local,10,2*model.n:]-encoded[2*model.n:])/model.init_bounds).square().mean()
            rows[index]=torch.stack((motion,port_loss,consistency))
    path_finite.extend(finite)
    return torch.stack(rows)


def candidate_loss(components):
    return .8*components[:,0].mean()+.2*components[:,0].max()+.1*components[:,1].mean()+.01*components[:,2].mean()


def finite_tree(value):
    if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite_tree(v) for v in value)
    if isinstance(value,float): return math.isfinite(value)
    return True


def restore(saved):
    model=OSGKD.from_checkpoint(saved['model']).cpu().double()
    optimizer=p1a.old.new_optimizer(model)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    torch.set_rng_state(saved['torch_rng'].clone())
    require(p1a.tree_equal(model.checkpoint(),saved['model']) and p1a.tree_equal(optimizer.state_dict(),saved['optimizer']), 'parent restore differs')
    require(finite_tree(model.state_dict()) and finite_tree(optimizer.state_dict()), 'nonfinite parent state')
    return model,optimizer


def attempt(model,opt,batch,k,constants,work,candidate):
    for group,base in zip(opt.param_groups,(.001,.0003),strict=True): group['lr']=base*multiplier(k)
    opt.zero_grad(set_to_none=True); finite=[]
    comp=(candidate_components(model,batch,constants['S_m'],constants['s_coi'],work,finite) if candidate else
          event_objective(model,batch,800,constants['S_m'],work=work,path_finite=finite))
    loss=candidate_loss(comp) if candidate else joint_loss(comp)
    value=float(loss.detach())
    row={'attempt':1861+k,'index':k,'event_ids':[e['id'] for e in batch],'lr_multiplier':multiplier(k),
         'loss':value if math.isfinite(value) else None,'motion_event_mean':None,'ports_event_mean':None,
         'consistency_event_mean':None,'accepted':False,'gradient_norm':None,'projection':None}
    for key,value in zip(('motion_event_mean','ports_event_mean','consistency_event_mean'),comp.detach().mean(0).tolist(),strict=True):
        row[key]=value if math.isfinite(value) else None
    if not bool(torch.stack(finite).all()): return {**row,'rejection':'nonfinite_trajectory'}
    if not bool(torch.isfinite(loss)): return {**row,'rejection':'nonfinite_loss'}
    loss.backward()
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
    if not math.isfinite(norm): return {**row,'rejection':'nonfinite_gradient'}
    row['gradient_norm']=norm
    opt.step()
    require(finite_tree(model.state_dict()) and finite_tree(opt.state_dict()), 'nonfinite updated model or Adam: immediate stop')
    with torch.no_grad():
        c,beta=model.c.clone(),model.beta.clone(); model.project_parameters_()
        change=torch.cat((model.c-c,model.beta-beta))
    require(finite_tree(model.state_dict()), 'nonfinite projected model: immediate stop')
    return {**row,'accepted':True,'clipped':norm>5.,'projection':{'changed_coordinates':int((change!=0).sum()),'max_change':float(change.abs().max())}}


def validate_baseline(b, archive, summary_ref):
    require(b['status']=='completed' and b['run_id']==baseline.RUN and b['job_id']=='eval02', 'completed corrected Step1 eval02 required')
    correction=b['correction']
    require(correction['motion_label_equivalence'] is True and
            correction['coordinate']=='tau_m=Pm-Pm_ref;tau_e=te-Pm_ref', 'verified motion/label equivalence and centered ports required')
    original_ref=correction['original_summary_ref']
    require(original_ref['path']=='results/osgkd_v2_train385_baseline_20261123/eval01/summary.json' and
            original_ref['sha256']=='099a3b168a390d0d183dc84e19bdd08e4e6731d248a966e20dabac4a7aeea542', 'canonical frozen original eval01 summary required')
    original=read(original_ref)
    require(original['status']=='completed' and original['run_id']==baseline.RUN and original['job_id']=='eval01', 'completed original Step1 required')
    require(b['release']['checkpoint_ref']==original['release']['checkpoint_ref'] and
            b['result']['train']==original['result']['train'], 'correction changed parent or inherited train result')
    require(archive['run_id']==baseline.RUN and archive['execution']['status']=='COMPLETED' and
            archive['execution']['completed_jobs']['eval02']['summary']==summary_ref and
            bool(archive['evidence']['independent_review']), 'root archived corrected Step1 required')


def validate(release):
    require(set(release)=={'run_id','job_id','released','device','threads','recompute','source_hashes','input_refs','checkpoint_ref'}, 'exact release keys required')
    require(tuple(release.get(k) for k in ('run_id','job_id','released','device','threads','recompute'))==(RUN,'pair01',True,'cpu',1,False), 'fixed pair01 CPU release required')
    require(SOURCES <= release['source_hashes'].keys(), 'source coverage incomplete')
    for name,sha in release['source_hashes'].items(): require(identity(repo_path(name))['sha256']==sha, 'source differs: '+name)
    require(set(release['input_refs'])==set(INPUTS), 'exact input_refs required')
    inputs={}
    for key,path in INPUTS.items():
        require(release['input_refs'][key]['path']==path, 'noncanonical input '+key)
        inputs[key]=read(release['input_refs'][key])
    b=inputs['baseline_summary']
    validate_baseline(b,inputs['baseline_archive'],release['input_refs']['baseline_summary'])
    require(release['checkpoint_ref']==b['release']['checkpoint_ref'] and release['checkpoint_ref']['sha256']==baseline.CHECKPOINT_SHA, 'baseline parent binding differs')
    verified(release['checkpoint_ref'])
    require(inputs['conflict_profile']['status']=='completed' and inputs['decomposition_summary']['status']=='completed', 'completed diagnostic archives required')
    return inputs


def save_checkpoint(path,state,k,release,plan_ref):
    value={'attempt':1861+k,'model':state['model'].checkpoint(),'optimizer':state['opt'].state_dict(),
           'torch_rng':state['rng'],'completed_pairs':k+1,'plan_ref':plan_ref,'release':release,
           'work':state['work'].snapshot(),'rejected':state['rejected'],'consecutive':state['consecutive'],
           'provenance':{'checkpoint_ref':release['checkpoint_ref'],'recovery_only':k<199}}
    temp=path.with_suffix('.pt.tmp')
    with temp.open('xb') as stream:
        torch.save(value,stream); stream.flush(); os.fsync(stream.fileno())
    require(not path.exists(),'checkpoint already exists'); os.rename(temp,path)
    return identity(path)


def short_diagnostics(models,events,constants,inputs,directory):
    started,cpu=perf_counter(),process_time()
    profile=read(inputs['conflict_profile']['result']['rows_ref'])
    groups=profile['cohort_event_ids']; misses=groups['parent_misses']
    selected={r['event_id']:r['selected_machine'] for r in profile['signed_electrical'] if all(v['persistent_80pct'] is True for v in r['endpoints'].values())}
    require(len(set(misses))==27 and len(selected)==22 and set(selected)<=set(misses) and
            all(type(i) is int and 0<=i<10 for i in selected.values()), 'fixed short cohorts differ')
    lookup={e['id']:e for e in events}; require(set(misses)<=lookup.keys(),'short cohort missing from train')
    rows=[]; work=IntegrationWork()
    with torch.no_grad():
        for eid in misses:
            e=lookup[eid]; stop=min(25,len(e['t'])-1); chosen=selected.get(eid)
            row={'event_id':eid,'selected_machine':chosen,'models':{}}
            require(stop>=1,'short diagnostic has no future')
            for name,model in models.items():
                th,om=[torch.as_tensor(e['history'][k],dtype=torch.float64) for k in ('theta','omega')]
                initial=model.initialize(th,om); work.encoder_events+=1
                ratio=(initial[30:40]/model.init_bounds[10:20]).numpy()
                item={'a0':initial[30:40].tolist(),'a0_over_bound':ratio.tolist(),
                      'e0_normalized':initial[40:50].tolist(),'windows':{}}
                paths={}
                for label,h in (('main',.005),('fine',.0025)):
                    path=integrate(model,initial,stop,internal_step=h,work=work)
                    paths[label]=path.numpy()
                    values=None
                    if bool(torch.isfinite(path).all()):
                        try:
                            values,_,_,_=electrical.components(model,path.numpy())
                        except ValueError:
                            pass
                    for window,requested in (('0.03',3),('capped_0.25',stop)):
                        if requested>stop:
                            item['windows'][label+'_'+window]={'complete':False,'reason':'short_record'}; continue
                        mask=np.asarray(e['masks']['tau_e'][1:requested+1],bool)
                        complete=bool(mask.all()) and values is not None
                        entry={'complete':complete,'point_count':requested,'end_s':requested*.01,
                               'reason':None if complete else 'nonfinite_readout_or_missing_target'}
                        if complete:
                            d=values['Fc'][1:requested+1]+values['tau_env'][1:requested+1]-e['targets']['tau_e'][1:requested+1]
                            entry['electrical_normalized_mse']=float(np.mean((d/constants['Se'])**2))
                            entry['geometry']=electrical.decompose(d,np.asarray(constants['Se']),np.asarray(constants['mass']),chosen)
                            entry['environment_output_mean']=values['Se_e'][1:requested+1].mean(0).tolist()
                        item['windows'][label+'_'+window]=entry
                for window,requested in (('0.03',3),('capped_0.25',stop)):
                    gate=(numerical_gate(paths['main'][:requested+1],paths['fine'][:requested+1],constants['omega_scale'])
                          if requested<=stop else {'passed':False,'reason':'short_record'})
                    item['windows']['fine_'+window]['numerical_gate']=gate
                row['models'][name]=item
            rows.append(row)
    aggregates={}
    for name in models:
        aggregates[name]={}
        for window in ('0.03','capped_0.25'):
            values=[r['models'][name]['windows']['fine_'+window] for r in rows]
            aggregates[name][window]={}
            for cohort,ids in (('parent_misses',set(misses)),('persistent_22',set(selected))):
                chosen_rows=[(r,v) for r,v in zip(rows,values,strict=True) if r['event_id'] in ids]
                good=[(r,v) for r,v in chosen_rows if v['complete'] and v['numerical_gate']['passed']]
                signed=[v['geometry']['selected_total'] for r,v in good if r['event_id'] in selected]
                a0=[r['models'][name]['a0_over_bound'][selected[r['event_id']]] for r,v in good if r['event_id'] in selected]
                e0=[r['models'][name]['e0_normalized'][selected[r['event_id']]] for r,v in good if r['event_id'] in selected]
                avg=lambda x:float(np.mean(x)) if x else None
                aggregates[name][window][cohort]={'raw_denominator':len(ids),
                    'complete_events':sum(v['complete'] for r,v in chosen_rows),'cross_precision_qualified_events':len(good),
                    'event_equal_electrical_normalized_mse':avg([v['electrical_normalized_mse'] for r,v in good]),
                    'selected_complete':len(signed),'selected_signed_mean':avg(signed),
                    'selected_a0_over_bound_mean':avg(a0),'selected_e0_normalized_mean':avg(e0),
                    'selected_a0_le_negative_095_fraction':avg([x<=-.95 for x in a0]),
                    'selected_a0_ge_positive_095_fraction':avg([x>=.95 for x in a0]),
                    'selected_e0_le_zero_fraction':avg([x<=0 for x in e0]),'selected_e0_ge_zero_fraction':avg([x>=0 for x in e0]),
                    'coi_geometry':{key:avg([v['geometry'][key] for r,v in good]) for key in ('accel_common_M','accel_relative_M','accel_total_M')}}
    frozen={}
    archived=inputs['decomposition_summary']['result']['groups']
    for cohort in ('parent_misses','persistent_22'):
        frozen[cohort]={}
        for window in ('0.03','capped_0.25'):
            frozen[cohort][window]={}
            for alias,item in archived[cohort][window].items():
                means=item['event_equal_means']
                frozen[cohort][window][alias]={
                    'raw_denominator':item['denominator'],'complete_events':item['complete_events'],
                    'event_equal_electrical_normalized_mse':means['normalized_total']/10,
                    'selected_signed_mean':means.get('selected_total'),
                    'coi_geometry':{key:means[key] for key in ('accel_common_M','accel_relative_M','accel_total_M')}}
    expected_rhs=24*len(models)*sum(min(25,len(lookup[eid]['t'])-1) for eid in misses)
    require(work.snapshot()=={'rhs_forward':expected_rhs,'rhs_recompute':0,'encoder_events':27*len(models)},'short work differs')
    return {'rows_ref':baseline.atomic_write(directory/'short_rows.json',rows),'aggregates':aggregates,'work':work.snapshot(),
            'cost':{'wall_s':perf_counter()-started,'cpu_s':process_time()-cpu},
            'frozen_parent_summary_ref':identity(repo_path(INPUTS['decomposition_summary'])),
            'frozen_parent_aggregates':frozen}


def execute(path):
    started,cpu=perf_counter(),process_time()
    path=path.resolve(); require(path.is_relative_to(ROOT),'repository release required')
    release=json.loads(path.read_bytes()); inputs=validate(release)
    local,public=[ROOT/s/RUN/'pair01' for s in ('artifacts','results')]
    require(not local.exists() and not public.exists(),'pair01 already claimed; same release cannot be repeated')
    local.mkdir(parents=True,exist_ok=False); public.mkdir(parents=True,exist_ok=False)
    receipt={'run_id':RUN,'job_id':'pair01','release':release,'release_ref':identity(path),'status':'started',
             'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'command':[sys.executable,*sys.argv],
             'environment':{'torch':str(torch.__version__),'numpy':np.__version__,'python':sys.version,'device':'cpu','dtype':'float64','threads':1}}
    baseline.atomic_write(local/'started.json',receipt)
    try:
        torch.set_num_threads(1)
        from kiki.data.osgkd_train_val import load_train
        raw,constants=load_train()
        digest=baseline.data_identity(raw)
        require(digest==inputs['baseline_summary']['result']['train']['data_identity'],'baseline train digest differs')
        sequence=plan(raw); plan_ref=baseline.atomic_write(local/'batch_plan.json',sequence)
        fit=tensor_events(raw,'cpu',max_steps=800); lookup={e['id']:e for e in fit}
        saved=torch.load(verified(release['checkpoint_ref']),map_location='cpu',weights_only=True)
        require(saved['attempt']==1860 and saved['model']['config']['variant']=='C1','C1 parent1860 required')
        states={}
        for name in ('control','candidate'):
            model,opt=restore(saved); target=local/name; target.mkdir()
            require(model.n==10,'ten-machine parent required')
            for key in ('mass','damping','F','Sp','Se','omega_scale','theta_ref'):
                expected=np.asarray(constants[key])
                if key in ('mass','damping'): expected=np.diag(expected)
                require(np.array_equal(getattr(model,key).detach().numpy(),expected),'parent constant differs: '+key)
            states[name]={'model':model,'opt':opt,'rng':torch.get_rng_state().clone(),'target':target,
                          'work':IntegrationWork(),'rows':[],'rejected':0,'consecutive':0,'checkpoints':{}}
        stopped=False; manifests=[]
        for k in range(ATTEMPTS):
            for name,state in states.items():
                torch.set_rng_state(state['rng']); batch=[lookup[eid] for eid in sequence[name][k]]
                expected=8*sum(min(800,len(e['targets']['theta'])-1) for e in batch)
                before=state['work'].rhs['forward']; require(before+expected<=CAP,'training RHS cap would exceed')
                begin,pc=perf_counter(),process_time()
                row=attempt(state['model'],state['opt'],batch,k,constants,state['work'],name=='candidate')
                require(state['work'].rhs['forward']-before==expected,'actual RHS differs')
                row.update(wall_s=perf_counter()-begin,cpu_s=process_time()-pc)
                state['rng']=torch.get_rng_state().clone(); state['rows'].append(row)
                state['rejected']+=not row['accepted']; state['consecutive']=0 if row['accepted'] else state['consecutive']+1
                baseline.atomic_write(state['target']/f'log_{k+1:03d}.json',row)
                stopped |= state['rejected']>=5 or state['consecutive']>=3
            if (k+1)%20==0 or stopped or k==199:
                refs={name:save_checkpoint(s['target']/f'attempt_{1861+k}.pt',s,k,release,plan_ref) for name,s in states.items()}
                for name,s in states.items(): s['checkpoints'][str(k+1)]=refs[name]
                manifests.append(baseline.atomic_write(local/f'progress_{k+1:03d}.json',{'completed_pairs':k+1,'checkpoints':refs,'plan_ref':plan_ref,'stopped_early':stopped}))
            if k==0 or (k+1)%20==0 or stopped:
                print(json.dumps({'completed_pairs':k+1,'accepted':{name:len(s['rows'])-s['rejected'] for name,s in states.items()},'stopped_early':stopped}),flush=True)
            if stopped: break
        result={'fixed_endpoint':2060 if not stopped else None,'actual_endpoint':1861+k,'stopped_early':stopped,
                'paired_attempts':k+1,'plan_ref':plan_ref,'progress_manifest_refs':{str(i):r for i,r in enumerate(manifests)},
                'training_rhs_cap_per_group':CAP,'new_fom':0,'adoption_decision':'root_pending','treatments':{}}
        for name,s in states.items():
            require(len(s['rows'])==k+1 and s['work'].rhs['forward']<=CAP and s['work'].rhs['recompute']==0,'paired attempt accounting differs')
            result['treatments'][name]={'attempts':len(s['rows']),'accepted':len(s['rows'])-s['rejected'],'rejected':s['rejected'],
                'checkpoint':s['checkpoints'][str(k+1)],'work':s['work'].snapshot(),
                'clipped':sum(r.get('clipped',False) for r in s['rows']),
                'maximum_finite_gradient_norm':max((r['gradient_norm'] for r in s['rows'] if r['gradient_norm'] is not None),default=None),
                'nonfinite_rejections':dict(Counter(r['rejection'] for r in s['rows'] if not r['accepted'])),
                'actual_event_exposures':8*len(s['rows']),
                'attempts_ref':baseline.atomic_write(s['target']/'attempts.json',s['rows']),
                'train_cost':{'wall_s':sum(r['wall_s'] for r in s['rows']),'cpu_s':sum(r['cpu_s'] for r in s['rows'])}}
        if not stopped:
            require(baseline.data_identity(raw)==digest,'train mutated before endpoint')
            # val is deliberately imported and loaded only after both fixed endpoints are saved.
            from kiki.data.osgkd_train_val import load_val
            val=load_val(constants)
            require(baseline.data_identity(val)==inputs['baseline_summary']['result']['val']['data_identity'],'baseline val digest differs')
            for name,s in states.items():
                model=s['model'].eval().requires_grad_(False)
                result['treatments'][name]['evaluation']={split:baseline.score_split(model,events,constants,split,s['target']/split) for split,events in (('train',raw),('val',val))}
            result['short_diagnostic']=short_diagnostics({n:s['model'] for n,s in states.items()},raw,constants,inputs,local)
            result['evaluation_work']={key:sum(s['evaluation'][split]['summary']['work'][key] for s in result['treatments'].values() for split in ('train','val')) for key in ('rhs_forward','rhs_recompute','encoder_events')}
        receipt.update(status='completed',result=result,completed_utc=datetime.now(timezone.utc).isoformat(),cost={'wall_s':perf_counter()-started,'cpu_s':process_time()-cpu},
                       loss_definitions={'control':'event mean of old motion + .1 ports + .01 consistency',
                         'candidate':'segmented motion: .8 event mean + .2 max; .1 event mean ports + .01 event mean consistency',
                         'all_losses':'dimensionless normalized squared error; not RMSE'})
        ref=baseline.atomic_write(public/'summary.json',receipt,public=True)
        baseline.atomic_write(local/'completed.json',{'status':'completed','summary_ref':ref})
    except BaseException as error:
        baseline.atomic_write(local/'failed.json',{'status':'failed','error':repr(error),'wall_s':perf_counter()-started,'cpu_s':process_time()-cpu})
        raise


def self_check():
    from unittest.mock import patch
    original_ref={'path':'results/osgkd_v2_train385_baseline_20261123/eval01/summary.json','bytes':123,
                  'sha256':'099a3b168a390d0d183dc84e19bdd08e4e6731d248a966e20dabac4a7aeea542'}
    original={'status':'completed','run_id':baseline.RUN,'job_id':'eval01',
              'release':{'checkpoint_ref':{'sha256':'parent'}},'result':{'train':{'data_identity':'synthetic'}}}
    corrected=copy.deepcopy(original)
    corrected.update(job_id='eval02',correction={'original_summary_ref':original_ref,
        'motion_label_equivalence':True,'coordinate':'tau_m=Pm-Pm_ref;tau_e=te-Pm_ref'})
    summary_ref={'path':INPUTS['baseline_summary'],'bytes':456,'sha256':'b'*64}
    archive={'run_id':baseline.RUN,'execution':{'status':'COMPLETED','completed_jobs':{'eval02':{'summary':summary_ref}}},
             'evidence':{'independent_review':'synthetic'}}
    with patch(__name__+'.read',return_value=original) as reader:
        validate_baseline(corrected,archive,summary_ref)
        reader.assert_called_once_with(original_ref)
        for key,value in (('motion_label_equivalence',False),('coordinate','absolute')):
            invalid=copy.deepcopy(corrected); invalid['correction'][key]=value
            try:
                validate_baseline(invalid,archive,summary_ref)
            except ValueError:
                pass
            else:
                raise AssertionError('uncorrected baseline accepted')
        invalid=copy.deepcopy(corrected); invalid['result']['train']['data_identity']='changed'
        try:
            validate_baseline(invalid,archive,summary_ref)
        except ValueError:
            pass
        else:
            raise AssertionError('changed inherited train accepted')
    torch.set_num_threads(1); torch.manual_seed(23101)
    events=[{'id':str(i),'label':'stable' if i<342 else 'unstable'} for i in range(385)]
    assert plan(events)==plan(events)
    assert multiplier(0)==1. and multiplier(199)==.25 and all(multiplier(k)>=multiplier(k+1) for k in range(199))
    constants={k:np.ones(2) for k in ('mass','damping','Sp','Se','omega_scale','S_m')}
    constants.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]),s_coi=2.)
    model=p1a.old.new_model('C1',constants,torch.device('cpu'))
    event={'id':'synthetic','history':{'theta':np.zeros((11,2)),'omega':np.zeros((11,2))},
           't':np.arange(12)*.01,'targets':{k:np.zeros((12,2)) for k in ('theta','omega','tau_m','tau_e')},
           'masks':{k:np.ones((12,2),bool) for k in ('theta','omega','tau_m','tau_e')}}
    short=copy.deepcopy(event); short['id']='short'; short['t']=np.arange(4)*.01
    for k in short['targets']: short['targets'][k]=short['targets'][k][:4]; short['masks'][k]=np.zeros((4,2),bool) if k.startswith('tau') else short['masks'][k][:4]
    batch=tensor_events([event,short],'cpu',max_steps=800)
    work=IntegrationWork(); finite=[]
    old=event_objective(model,batch,800,constants['S_m'],work=work,path_finite=finite)
    cand=candidate_components(model,batch,constants['S_m'],2.,IntegrationWork(),[])
    assert torch.equal(old[:,1:],cand[:,1:]) and cand[1,1]==0
    for objective in (joint_loss(old),candidate_loss(cand)):
        grads=torch.autograd.grad(objective,tuple(model.parameters()),retain_graph=True,allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in grads)
    comp=torch.tensor([[1.,10.,100.],[3.,20.,200.]],dtype=torch.float64)
    assert torch.allclose(candidate_loss(comp),torch.tensor(.8*2+.2*3+.1*15+.01*150,dtype=torch.float64))
    target={k:torch.zeros((801,2),dtype=torch.float64) for k in ('theta','omega')}
    path=torch.zeros((801,10),dtype=torch.float64); path[1:101,:2]=torch.tensor([1.,-1.]);path[101:301,:2]=torch.tensor([2.,-2.]);path[301:,:2]=torch.tensor([3.,-3.])
    assert torch.allclose(segmented_motion(path,target,torch.ones(2,dtype=torch.float64),torch.ones(2),2.),torch.tensor(.4*.5+.3*2+.3*4.5,dtype=torch.float64))
    assert segmented_motion(path[:4],{k:v[:4] for k,v in target.items()},torch.ones(2,dtype=torch.float64),torch.ones(2),2.)==.5
    opt=p1a.old.new_optimizer(model)
    p1a.attempt(model,opt,batch,1860,800,constants['S_m'],IntegrationWork())
    assert opt.state_dict()['state']
    saved={'model':model.checkpoint(),'optimizer':copy.deepcopy(opt.state_dict()),'torch_rng':torch.get_rng_state().clone()}
    a,ao=restore(saved); b,bo=restore(saved)
    row=attempt(a,ao,batch,0,constants,IntegrationWork(),False)
    plain=p1a.attempt(b,bo,batch,1861,800,constants['S_m'],IntegrationWork())
    assert row['accepted']==plain['accepted'] and row['loss']==plain['J']
    assert p1a.tree_equal(a.state_dict(),b.state_dict()) and p1a.tree_equal(ao.state_dict(),bo.state_dict())
    assert work.rhs['forward']==8*(11+3) and CAP==200*8*800*8
    assert not finite_tree({'x':torch.tensor(float('nan'))})
    print(json.dumps({'self_check':'passed','scientific_payloads_opened':False,'control_gradient_update_parity':True,'segmented_math_and_short_masks':True,'scheduler_and_coverage':True,'corrected_baseline_binding':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check'); sub.add_parser('pair').add_argument('--release',required=True,type=Path)
    args=parser.parse_args(); self_check() if args.command=='self-check' else execute(args.release)
