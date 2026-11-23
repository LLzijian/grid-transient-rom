"""Frozen endpoint gradient and initial-state diagnostics; zero training."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
import numpy as np
import torch
from experiments.osgkd_v2_p0_20261123 import run as p0
from experiments.osgkd_v2_early_electrical_20261123 import run as electrical
from kiki.models.osgkd import OSGKD
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events, event_objective, joint_loss
from kiki.data.osgkd import load_fit297
RUN = 'osgkd_v2_conflict_profile_20261123'
JOB = 'probe01'
INPUTS = {'audit':'results/osgkd_fit297_20261113/port_audit_release.json',
          'early_analysis':'results/osgkd_v2_early_electrical_20261123/analysis.json',
          'p0_summary':'results/osgkd_v2_p0_20261123/probe01/summary.json'}
SOURCES = {f'experiments/{RUN}/run.py', 'experiments/osgkd_v2_p0_20261123/run.py',
 'experiments/osgkd_v2_early_electrical_20261123/run.py', 'experiments/osgkd_v2_p1a_20261123/run.py',
 'src/kiki/models/osgkd_fitting.py', 'src/kiki/models/osgkd.py', 'src/kiki/data/osgkd.py',
 'src/kiki/evaluation/osgkd.py', 'src/kiki/evaluation/gftd_analysis.py',
 'experiments/osgkd_fit297_20261113/run.py', 'experiments/evaluate_l1_latent.py'}
require = electrical.require

def verified(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','bytes','sha256'}, 'exact typed reference required')
    require(type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and
            len(ref['sha256'])==64 and all(c in '0123456789abcdef' for c in ref['sha256']), 'invalid reference types')
    return p0.verified(ref)

def read(ref):
    return json.loads(verified(ref).read_bytes())

def sample(ids):
    return sorted(ids,key=lambda x:hashlib.sha256((RUN+':'+x).encode()).hexdigest())[:4]

def blocks(model):
    result = {k:[] for k in ('initializer','mechanical','coupling','amplitude','environment')}
    for name,p in model.named_parameters():
        if name.startswith(('encoder.','initial_head.')): key='initializer'
        elif name=='mechanical_coeff': key='mechanical'
        elif name in ('c','beta'): key='coupling'
        elif name=='amplitude_coeff': key='amplitude'
        elif name in ('environment_coeff','memory_coeff'): key='environment'
        elif name=='edge_coeff':
            result['amplitude'].append((name, (...,slice(0,2))))
            result['environment'].append((name, (...,slice(2,5))))
            continue
        else: raise ValueError('unclassified parameter '+name)
        result[key].append((name, (...,)))
    return result

def gradient(loss,model,retain=False):
    params=dict(model.named_parameters())
    values=torch.autograd.grad(loss,tuple(params.values()),retain_graph=retain,allow_unused=True)
    out={name:(torch.zeros_like(p) if g is None else g.detach()) for (name,p),g in zip(params.items(),values,strict=True)}
    require(all(bool(torch.isfinite(g).all()) for g in out.values()),'nonfinite gradient')
    require(all(p.grad is None for p in params.values()),'parameter grad buffer changed')
    return out

def compare(a,b,model):
    groups=blocks(model)
    groups['all']=[(name,(...,)) for name,_ in model.named_parameters()]
    result={}
    for name,parts in groups.items():
        x=torch.cat([a[k][sl].reshape(-1) for k,sl in parts]); y=torch.cat([b[k][sl].reshape(-1) for k,sl in parts])
        nx,ny=float(x.norm()),float(y.norm()); dot=float(x@y)
        cosine=dot/(nx*ny) if nx>1e-12 and ny>1e-12 else None
        result[name]={'coordinates':x.numel(),'objective_norm':nx,'electrical_norm':ny,'inner_product':dot,
          'cosine':cosine,'electrical_over_objective_norm':ny/nx if nx>1e-12 else None,
          'weighted_0_1_electrical_over_objective_norm':.1*ny/nx if nx>1e-12 else None,
          'descriptive_gate':None if cosine is None else ('opposing' if cosine<=-.1 else 'aligned' if cosine>=.1 else 'near_orthogonal')}
    return result

def gradient_probe(model,fit,selected,missed,constants,work):
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    grads={}
    for label in ('worsened','remaining'):
        batch=[fit[x] for x in selected[label]]
        components=event_objective(model,batch,800,constants['S_m'],recompute=False,work=work)
        require(bool(torch.isfinite(components).all()),'nonfinite objective')
        grads[label]={'motion':gradient(components[:,0].mean(),model,retain=True),
                      'joint':gradient(joint_loss(components),model)}
    grads['weighted']={kind:{k:(24/267)*grads['worsened'][kind][k]+(243/267)*grads['remaining'][kind][k]
                       for k in grads['worsened'][kind]} for kind in ('motion','joint')}
    del components
    loss,paths=electrical.electrical_loss(model,[fit[x] for x in missed],work)
    aux=gradient(loss,model)
    require(all(torch.equal(v,original[k]) for k,v in model.state_dict().items()),'probe modified model')
    return {'electrical_mse':float(loss.detach()),'selected_event_ids':selected,
      'comparisons':{group:{kind:compare(g,aux,model) for kind,g in values.items()} for group,values in grads.items()}}

def initial_profile(model,events,cohorts):
    with torch.no_grad():
        theta=torch.tensor(np.stack([e['history']['theta'] for e in events]),dtype=torch.float64)
        omega=torch.tensor(np.stack([e['history']['omega'] for e in events]),dtype=torch.float64)
        state=model.initialize(theta,omega)
    require(torch.equal(state[:,:10],theta[:,-1]) and torch.equal(state[:,10:20],omega[:,-1]),'motion initial state differs')
    ratio=(state[:,30:40]/model.init_bounds[10:20]).numpy()
    require(np.isfinite(ratio).all() and np.max(abs(ratio))<=1+1e-12,'invalid amplitude initial state')
    rows=[{'event_id':e['id'],'a0':state[i,30:40].tolist(),'a0_over_bound':ratio[i].tolist()} for i,e in enumerate(events)]
    summary={}
    for group,ids in cohorts.items():
        x=ratio[[i for i,e in enumerate(events) if e['id'] in ids]]
        summary[group]={'denominator':len(ids),'absolute_a0_over_bound':p0.distribution(abs(x).ravel()),
           'events_touching_099':int((abs(x)>=.99).any(axis=1).sum()),
           'per_node_touching_099':(abs(x)>=.99).sum(axis=0).tolist(),
           'per_node_absolute_a0_over_bound':[p0.distribution(abs(x[:,j])) for j in range(10)]}
    return summary,rows

def signed_error(event,model,path,node):
    stop=min(25,len(event['t'])-1)
    mask=np.asarray(event['masks']['tau_e'][1:stop+1,node],bool)
    if not mask.any(): return {'missing_reason':'no electrical labels','mean':None,'positive_fraction':None,'persistent_80pct':None}
    with torch.no_grad(): prediction=model.ports(torch.tensor(path[1:stop+1],dtype=torch.float64)).tau_e.numpy()[:,node]
    error=(prediction-event['targets']['tau_e'][1:stop+1,node])/float(model.Se[node])
    require(np.isfinite(error[mask]).all(),'nonfinite signed electrical error')
    fraction=float((error[mask]>0).mean())
    return {'missing_reason':None,'observed_points':int(mask.sum()),'mean':float(error[mask].mean()),
            'positive_fraction':fraction,'persistent_80pct':fraction>=.8}

def execute(path):
    start,cpu=perf_counter(),process_time(); torch.set_num_threads(1)
    path=path.resolve(); require(path.is_relative_to(ROOT),'release outside repository')
    release=json.loads(path.read_bytes())
    require((release.get('run_id'),release.get('job_id'),release.get('released'))==(RUN,JOB,True),'wrong release')
    require(SOURCES<=release['source_hashes'].keys(),'missing source hashes')
    for name,digest in release['source_hashes'].items():
        require(type(digest) is str and p0.identity(p0.relative_path(name))['sha256']==digest,'source differs: '+name)
    require(set(release['input_refs'])==set(INPUTS),'input keys differ')
    for key,name in INPUTS.items(): require(release['input_refs'][key]['path']==name,'input path differs')
    analysis=read(release['input_refs']['early_analysis']); prior=read(release['input_refs']['p0_summary'])
    verified(release['input_refs']['audit']); require(prior['status']=='completed','P0 not complete')
    refs=analysis['inputs']; require(set(refs)=={'parent','control','candidate'},'endpoint keys differ')
    expected={'parent':'artifacts/osgkd_v2_p1a_20261123/pair01/8s','control':'artifacts/osgkd_v2_early_electrical_20261123/pair01/control',
              'candidate':'artifacts/osgkd_v2_early_electrical_20261123/pair01/candidate'}
    paths={}
    for name,base in expected.items():
        paths[name]={}
        require(set(refs[name])=={'checkpoint_ref','rows_ref','predictions_ref'},'endpoint ref keys differ')
        for key,suffix in (('checkpoint_ref',f'/attempt_{1700 if name=="parent" else 1732}.pt'),('rows_ref','/evaluation/rows.json'),('predictions_ref','/evaluation/predictions.npz')):
            require(refs[name][key]['path']==base+suffix,'endpoint path differs'); paths[name][key]=verified(refs[name][key])
    local=ROOT/'artifacts'/RUN/JOB; public=ROOT/'results'/RUN/JOB
    require(not local.exists() and not public.exists(),'exclusive job already exists')
    local.mkdir(parents=True,exist_ok=False)
    receipt={'run_id':RUN,'job_id':JOB,'status':'started','pid':os.getpid(),'release_ref':p0.identity(path),'release':release,
             'started_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv]}
    p0.write_json(local/'started.json',receipt)
    try:
        public.mkdir(parents=True,exist_ok=False)
        events,constants=load_fit297(release['input_refs']['audit'])
        require(len(events)==297 and Counter(e['label'] for e in events)=={'stable':267,'unstable':30},'cohort differs')
        stored_rows={name:json.loads(paths[name]['rows_ref'].read_bytes()) for name in refs}
        rows={name:p0.validate_rows(data,events) for name,data in stored_rows.items()}
        for name,data in rows.items():
            require([r['event_id'] for r in stored_rows[name]]==[e['id'] for e in events],'row order differs')
            for row in data:
                require(row['classification_8s'] is not None and row['classification_8s']['los_pair_pass'] is True,'8s LOS qualification failed')
                if row['label']=='unstable': require(row['classification_actual'] is not None and row['classification_actual']['los_pair_pass'] is True,'actual LOS qualification failed')
                if row['label']=='stable': require(row['scopes']['8']['full_horizon_truth'] and row['scopes']['8']['numerical']['passed'],'stable 8s qualification failed')
        stable=[i for i,e in enumerate(events) if e['label']=='stable']
        worsened=[events[i]['id'] for i in stable if rows['candidate'][i]['scopes']['8']['metrics']['mean_motion_error']-rows['control'][i]['scopes']['8']['metrics']['mean_motion_error']>0]
        remaining=[events[i]['id'] for i in stable if events[i]['id'] not in worsened]
        missed=[e['id'] for e,r in zip(events,rows['parent'],strict=True) if e['label']=='unstable' and r['classification_actual']['classification']['predicted_los'] is False and r['classification_actual']['true_time_record_s'] is not None]
        require((len(worsened),len(remaining),len(missed))==(24,243,27),'fixed cohorts differ')
        require(set(missed)==set(analysis['official_scores']['parent']['classification']['train_all']['actual']['missed_unstable_ids']),'frozen miss identities differ')
        cohorts={'worsened':worsened,'remaining':remaining,'parent_misses':missed}
        selected={k:sample(cohorts[k]) for k in ('worsened','remaining')}
        fit={e['id']:e for e in tensor_events(events,'cpu',max_steps=800)}
        models={}; spectra={}; details={}; initials={}; initial_rows={}; gradients={}; work=IntegrationWork()
        for name,r in refs.items():
            saved=torch.load(paths[name]['checkpoint_ref'],map_location='cpu',weights_only=True)
            require(saved['attempt']==(1700 if name=='parent' else 1732) and saved['model']['config']['variant']=='C1' and saved['provenance']['audit_ref']==release['input_refs']['audit'],'checkpoint identity differs')
            model=OSGKD.from_checkpoint(saved['model']).cpu().double().eval(); models[name]=model
            initials[name],initial_rows[name]=initial_profile(model,events,cohorts)
            if name in ('parent','control'):
                spectra[name],details[name]=p0.spectrum(model)
                gradients[name]=gradient_probe(model,fit,selected,missed,constants,work)
        require(work.rhs['forward']<=113200 and work.rhs['recompute']==0,'RHS cap exceeded')
        observations={}
        for group in ('worsened','remaining'):
            indices=[i for i,e in enumerate(events) if e['id'] in cohorts[group]]
            observations[group]={'denominator':len(indices),
              'max_theta_range':p0.distribution([np.ptp(events[i]['targets']['theta'][:801],axis=1).max() for i in indices]),
              'max_abs_omega':p0.distribution([abs(events[i]['targets']['omega'][:801]).max() for i in indices]),
              'saved_mean_motion_error_8s':{name:p0.distribution([rows[name][i]['scopes']['8']['metrics']['mean_motion_error'] for i in indices]) for name in refs}}
        signed=[]; weights=np.asarray(constants['mass']); weights=weights/weights.sum(); points=0
        with ExitStack() as stack:
            archives={name:stack.enter_context(np.load(paths[name]['predictions_ref'],allow_pickle=False)) for name,r in refs.items()}
            for i,event in enumerate(events):
                if event['id'] not in missed: continue
                stop=min(25,len(event['t'])-1); omega=event['targets']['omega'][:stop+1]
                relative=omega-(omega@weights)[:,None]
                growth=(relative[1:]-relative[0]).max(axis=0); node=int(growth.argmax())
                record={'event_id':event['id'],'selected_machine':node,'maximum_positive_relative_speed_growth':float(growth[node]),'endpoints':{}}
                for name,archive in archives.items():
                    state=archive[f'event_{i:04d}_fine'][:stop+1]; points+=len(state)
                    require(state.shape==(stop+1,50),'fine prefix shape differs')
                    record['endpoints'][name]=signed_error(event,models[name],state,node) if growth[node]>0 else {'missing_reason':'no positive relative speed growth','mean':None,'positive_fraction':None,'persistent_80pct':None}
                signed.append(record)
        signed_summary={name:{'denominator':27,'covered':sum(r['endpoints'][name]['mean'] is not None for r in signed),
           'mean_error':p0.distribution([r['endpoints'][name]['mean'] for r in signed if r['endpoints'][name]['mean'] is not None]),
           'positive_fraction':p0.distribution([r['endpoints'][name]['positive_fraction'] for r in signed if r['endpoints'][name]['positive_fraction'] is not None]),
           'persistent_80pct_events':sum(r['endpoints'][name]['persistent_80pct'] is True for r in signed)} for name in refs}
        rref=p0.write_json(local/'rows.json',{'initialization':initial_rows,'signed_electrical':signed,'cohort_event_ids':cohorts})
        sref=p0.write_json(local/'spectra.json',details)
        receipt.update(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),inputs=refs,
          result={'gradient_conflict':gradients,'spectrum':spectra,'initial_amplitude':initials,'observations':observations,
                  'signed_electrical':signed_summary,'signed_error_definition':'(predicted tau_e - observed tau_e)/Se; positive means electrical deceleration overestimated; tau_m - tau_e',
                  'rows_ref':rref,'spectra_ref':sref,'descriptive_cosine_thresholds':[-.1,.1]},
          work={**work.snapshot(),'rhs_cap':113200,'training_updates':0,'new_evaluation_integrations':0,'new_fom':0,'saved_fine_prefix_state_points':points,'eight_second_backward_calls':8,'early_electrical_backward_calls':2,'profile_initialization_events':891,'profile_initialization_calls':3,'reference_spectra':2},
          cost={'wall_s':perf_counter()-start,'cpu_s':process_time()-cpu},
          environment={'device':'cpu','dtype':'float64','threads':torch.get_num_threads(),'torch':str(torch.__version__),'numpy':np.__version__})
        ref=p0.write_json(public/'summary.json',receipt); p0.write_json(local/'completed.json',{'status':'completed','summary_ref':ref})
        return 0
    except BaseException as error:
        p0.write_json(local/'failed.json',{'status':'failed','error':f'{type(error).__name__}: {error}','wall_s':perf_counter()-start,'cpu_s':process_time()-cpu})
        raise

def self_check():
    torch.set_num_threads(1)
    constants={key:np.ones(2) for key in ('mass','damping','Sp','Se','omega_scale','S_m')}
    constants.update(F=np.eye(2),theta_ref=np.zeros(2),edge_index=np.array([[0],[1]]))
    model=electrical.old.new_model('C1',constants,torch.device('cpu'))
    partitions=blocks(model); counts={name:torch.zeros_like(p,dtype=torch.int64) for name,p in model.named_parameters()}
    for parts in partitions.values():
        for name,sl in parts: counts[name][sl]+=1
    assert all(bool((v==1).all()) for v in counts.values())
    ids=[f'event_{i}' for i in range(267)]; a,b=sample(ids[:24]),sample(ids[24:])
    assert len(set(a+b))==8 and a==sample(list(reversed(ids[:24]))) and b==sample(list(reversed(ids[24:])))
    z={name:torch.zeros_like(p) for name,p in model.named_parameters()}; ones={name:torch.ones_like(p) for name,p in model.named_parameters()}
    assert compare(z,ones,model)['all']['cosine'] is None
    assert abs(compare(ones,{k:-v for k,v in ones.items()},model)['all']['cosine']+1)<1e-12
    from types import SimpleNamespace
    toy=SimpleNamespace(Se=torch.ones(1),ports=lambda path:SimpleNamespace(tau_e=torch.tensor([[1.],[1.],[1.],[1.],[-1.]],dtype=torch.float64)))
    event={'t':np.arange(6)*.01,'targets':{'tau_e':np.zeros((6,1))},'masks':{'tau_e':np.ones((6,1),bool)}}
    signed=signed_error(event,toy,np.zeros((6,5)),0)
    assert signed['mean']==.6 and signed['positive_fraction']==.8 and signed['persistent_80pct'] is True
    event['masks']['tau_e'][:]=False
    assert signed_error(event,toy,np.zeros((6,5)),0)['mean'] is None
    print('synthetic self-check passed: exhaustive disjoint gradient blocks, deterministic eight-event sample, null/opposed cosine and positive-error persistence; no private assets read')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--release',type=Path); group.add_argument('--self-check',action='store_true'); args=parser.parse_args()
    if args.self_check: self_check()
    else: raise SystemExit(execute(args.release))


