"""Bounded, resumable physical closure discovery; historical runners are immutable.

Every scientific process enters a stage receipt. Selection reads train only.
Exports and manifests bind structure, optimizer/RNG state and actual update counts.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
PROCESS_STARTED=time.perf_counter()

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
import numpy as np
import torch

CONFIG = ROOT/'configs/closure_discovery_20261017.json'
SOURCE_HASH = 'c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(obj):
    if isinstance(obj,(bool,np.bool_)):return bool(obj)
    if isinstance(obj, dict): return {str(k):safe(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return safe(obj.tolist())
    if isinstance(obj, (float, np.floating)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (int, np.integer)): return int(obj)
    if isinstance(obj, Path): return str(obj)
    return obj


def write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(obj),indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def context():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    return cfg,out


def source():
    from shared_closure_rom_0101 import load_model
    from symbolic_closure_rom import SymbolicClosureROM
    cfg,_=context();path=ROOT/cfg['source']
    if sha(path)!=SOURCE_HASH: raise ValueError('registered source hash changed')
    model=SymbolicClosureROM.from_source(load_model(path))
    model.provenance={'source':cfg['source'],'source_sha256':SOURCE_HASH,
                      'protocol':cfg['protocol'],'protocol_sha256':sha(ROOT/cfg['protocol']),
                      'state_scope':'fixed ten-machine 40-state effective closure; research candidate'}
    identity=ROOT/cfg['output']/'implementation/identity.json'
    if identity.exists():model.provenance['implementation_identity_sha256']=sha(identity)
    return model


def optimizer(model, scale=1.):
    cfg,_=context();groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        block='encoder' if ('gru' in name or 'head' in name) else ('closure' if name in ('coefficients','_template_scalars.3','_template_scalars.4') else 'skeleton')
        groups.setdefault(block,[]).append(p)
    return torch.optim.Adam([{'params':p,'lr':cfg['rates'][b]*scale,'block':b} for b,p in groups.items()])


def train_inputs():
    from train_local_memory_0098 import read_train
    events,excluded=read_train()
    scale=float(np.sqrt(np.mean(np.concatenate([e['torque'] for e in events])**2)))
    scale=max(scale,1e-8)
    _,out=context()
    if (out/'implementation/identity.json').exists():
        rows=[{'id':e['id'],'group':e['group'],'label':e['label'],
               'shape':list(e['x'].shape),'x_sha256':hashlib.sha256(np.ascontiguousarray(e['x']).tobytes()).hexdigest(),
               'torque_sha256':hashlib.sha256(np.ascontiguousarray(e['torque']).tobytes()).hexdigest()} for e in events]
        identity={'events':rows,'excluded':excluded,'force_scale':scale,'test_read':False}
        path=out/'implementation/train_identity.json'
        if path.exists() and json.loads(path.read_text())!=safe(identity):
            raise ValueError('registered train evidence changed')
        if not path.exists():write(path,identity)
    return events,excluded,scale


def loss(model,samples,history,horizon,force_scale):
    from train_local_memory_0098 import objective
    total,terms=objective(model,samples,history,horizon,force_scale)
    return total-1e-4*terms['complexity'],terms


def horizon_for_update(step):
    """Prospective schedule with the old horizon pattern, not its 800-step cap."""
    if isinstance(step,bool) or not isinstance(step,int) or step<1:
        raise ValueError('positive integer update required')
    return 2. if step<=100 else (1.,2.,5.)[(step-101)%3]


def score(model,events,horizons=(1.,2.,5.),*,deadline=None):
    from train_local_memory_0098 import graph,autonomous,risk
    arrays=[];started=time.perf_counter()
    with torch.no_grad(),graph(model):
        for h in horizons:
            rows=[]
            for i in range(0,len(events),32):
                if deadline is not None and time.perf_counter()+10>=deadline:
                    raise TimeoutError('score incomplete at stage cap; no partial aggregate')
                rows.append(autonomous(model,events[i:i+32],h))
            v=torch.cat(rows)
            if not torch.isfinite(v).all():raise FloatingPointError('nonfinite train score; no event removed')
            arrays.append(v.detach().numpy())
    per=np.mean(arrays,axis=0)
    values=torch.as_tensor(per)
    groups=sorted({e['group'] for e in events})
    # The selection target includes the upper event tail, so uncertainty must
    # be computed for that same risk rather than for the ordinary mean MSE.
    jack=np.asarray([float(risk(values[torch.as_tensor([e['group']!=g for e in events])])) for g in groups]) if len(groups)>1 else np.zeros(1)
    group_se=float(np.sqrt((len(groups)-1)/len(groups)*np.square(jack-jack.mean()).sum())) if len(groups)>1 else 0.
    return {'risk':float(risk(values)),'group_se':group_se,'se_method':'leave-one-fault-bus-out jackknife of the nonlinear event risk',
            'events':[{'id':e['id'],'group':e['group'],'label':e['label'],'mse':float(v)} for e,v in zip(events,per)],
            'seconds':time.perf_counter()-started,'scope':'train selection; genuine available intervals','horizons':list(horizons)}


def checkpoint(path,model,opt,rng,step,trace,selection=None,coverage=None):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    model.provenance['completed_updates_in_current_fit']=step
    identity=context()[1]/'implementation/identity.json'
    if identity.exists():
        current=sha(identity);previous=model.provenance.get('implementation_identity_sha256')
        history=model.provenance.setdefault('implementation_history',[])
        for value in (previous,current):
            if value and value not in history:history.append(value)
        model.provenance['implementation_identity_sha256']=current
    model.export(path/'model.json')
    if step in (32,128,256,512,1024):
        level_path=path/f'model_at_step{step:05d}.json'
        if level_path.exists() and sha(level_path)!=sha(path/'model.json'):
            raise ValueError('immutable adaptation endpoint changed')
        if not level_path.exists():level_path.write_bytes((path/'model.json').read_bytes())
    torch.save({'optimizer':opt.state_dict(),'numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'step':step,'coverage':coverage},path/'training.pt')
    write(path/'checkpoint.json',{'step':step,'model_sha256':sha(path/'model.json'),
         'training_sha256':sha(path/'training.pt'),'trace':trace,'selection':selection,
         'test_read':False,'status':'research_only'})


def score_at_level(path,events,level,*,deadline=None):
    """Rank a saved adaptation level, regardless of later optimizer progress."""
    from symbolic_closure_rom import SymbolicClosureROM
    path=Path(path);meta=json.loads((path/'checkpoint.json').read_text())
    snapshot=path/f'model_at_step{level:05d}.json';cache=path/f'score{level}.json'
    if not snapshot.exists():
        if meta['step']!=level or sha(path/'model.json')!=meta['model_sha256']:
            raise ValueError('earlier adaptation endpoint unavailable; cannot relabel a later model')
        snapshot.write_bytes((path/'model.json').read_bytes())
    expected={'adaptation_updates':level,'model_sha256':sha(snapshot)}
    if cache.exists():
        result=json.loads(cache.read_text())
        if all(result.get(k)==v for k,v in expected.items()):return result
        # Migration is permitted only for the exact current checkpoint level.
        # Preserve its original score before adding missing identity fields.
        ids=[e['id'] for e in events]
        if meta['step']!=level or any(k in result for k in expected) or [e['id'] for e in result.get('events',[])]!=ids:
            raise ValueError('cached score adaptation identity changed')
        original=cache.with_name(f'score{level}_before_identity.json')
        if not original.exists():original.write_bytes(cache.read_bytes())
        result.update(expected);write(cache,result);return result
    model=SymbolicClosureROM.from_export(snapshot)
    result=score(model,events,deadline=deadline);result.update(expected);write(cache,result)
    return result


def bind_execution_sources(out):
    """Bind a scientific run and its resumes to the same saved implementation."""
    names=['symbolic_closure_rom','discover_closure','closure_discovery_support',
           'closure_discovery_benchmarks','closure_discovery_data',
           'closure_discovery_confirmation','closure_discovery_export',
           'closure_expression_identity',
           'train_local_memory_0098','train_joint_numerical_pair_0100',
           'train_structured_rom','local_memory_rom_0098','shared_closure_rom_0101',
           'evaluate_structured_rom','evaluate_local_memory_0098','structured_rom',
           'structure_grammar','generate_private_dataset','train_joint_campaign',
           'train_joint_latent','evaluate_long_latent','evaluate_l1_latent',
           'compare_physical_modes','kiki.data.private_dataset','kiki.models.private_pm','kiki.physics']
    import importlib.util
    paths=[Path(importlib.util.find_spec(n).origin) for n in names]
    cfg,out=context()
    paths += [CONFIG,ROOT/cfg['protocol'],out/'new_fault_manifest.json',out/'ledger/allocation.json']
    paths=[p for p in paths if p.exists()]
    import importlib.metadata
    identity={'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},
              'runtime':{n:importlib.metadata.version(n) for n in ('numpy','torch','scipy','h5py','andes')}}
    folder=Path(out)/'implementation';manifest=folder/'identity.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=identity:
            raise ValueError('scientific implementation changed; preserve results and register an explicit amendment before resume')
    else:
        for p in paths:
            dest=folder/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        write(manifest,identity)
    return sha(manifest)


def fit(model,events,force_scale,target,path,seed,*,final=False,deadline=None):
    """Resume exact optimizer/RNG state. Final refits always use a fresh path/source."""
    from train_local_memory_0098 import graph
    from train_joint_numerical_pair_0100 import next_samples
    from train_local_memory_0098 import histories_for
    from symbolic_closure_rom import SymbolicClosureROM
    path=Path(path);rng=np.random.default_rng(seed);torch.manual_seed(seed)
    coverage={'cover_order':rng.permutation(len(events)).tolist(),'cursor':0}
    opt=optimizer(model);step=0;trace=[]
    if (path/'checkpoint.json').exists():
        meta=json.loads((path/'checkpoint.json').read_text());
        if sha(path/'model.json')!=meta['model_sha256'] or sha(path/'training.pt')!=meta['training_sha256']:
            raise ValueError('resume checkpoint hash changed')
        model=SymbolicClosureROM.from_export(path/'model.json');opt=optimizer(model)
        state=torch.load(path/'training.pt',weights_only=False,map_location='cpu')
        opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
        step=state['step'];trace=meta['trace']
        coverage=state['coverage']
    started=time.perf_counter()
    model.provenance['fit_seed']=seed
    def numerical_failure(reason,attempted_step):
        path.mkdir(parents=True,exist_ok=True)
        failure_path=path/f'failure_state_step{attempted_step:05d}.pt'
        torch.save({'model_state':model.state_dict(),'optimizer':opt.state_dict(),
            'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
            'completed_updates':step,'coverage':coverage},failure_path)
        write(path/'fit_failure.json',{'reason':reason,'attempted_step':attempted_step,
            'completed_updates':step,'target':target,'trace':trace,'state_sha256':sha(failure_path),
            'status':'numerically_unresolved','not_a_physical_instability_label':True})
        raise FloatingPointError(f'{reason} at update {attempted_step}; state and cost retained')
    while step<target:
        if deadline is not None and time.perf_counter()+30>=deadline:
            checkpoint(path,model,opt,rng,step,trace,coverage=coverage)
            raise TimeoutError('stage cap; incomplete candidate preserved, not ranked')
        tick=time.perf_counter();samples,sampling_trace=next_samples(events,rng,coverage);hist=histories_for(samples)
        write(path/'current_update.json',{'attempted_update':step+1,'last_completed_in_memory':step,
            'last_recoverable_checkpoint':json.loads((path/'checkpoint.json').read_text())['step'] if (path/'checkpoint.json').exists() else 0,
            'status':'in_progress','test_read':False})
        for group in opt.param_groups:group['lr']=context()[0]['rates'][group['block']]*(.3 if final and step>=512 else 1.)
        opt.zero_grad(set_to_none=True)
        with graph(model):
            value,terms=loss(model,samples,hist,horizon_for_update(step+1),force_scale)
            if not torch.isfinite(value):numerical_failure('nonfinite training objective',step+1)
            value.backward()
        pars=[p for p in model.parameters() if p.requires_grad]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in pars):numerical_failure('nonfinite gradient',step+1)
        norm=float(torch.nn.utils.clip_grad_norm_(pars,100.))
        opt.step();step+=1
        if any(not torch.isfinite(p).all() for p in pars):numerical_failure('nonfinite updated parameter',step)
        trace.append({'step':step,'loss':float(value.detach()),'risk':float(terms['risk'].detach()),
                      'gradient_norm':norm,'clipped':norm>100.,'seconds':time.perf_counter()-tick,
                      'sample_ids':[e['id'] for e,_ in samples],'coverage_cursor':coverage['cursor']})
        if step%128==0 or step==target:
            checkpoint(path,model,opt,rng,step,trace,coverage=coverage)
            print(json.dumps({'phase':'fit','path':str(path),'step':step,'target':target,'loss':trace[-1]['loss']}),flush=True)
    checkpoint(path,model,opt,rng,step,trace,coverage=coverage)
    return model,{'seconds':time.perf_counter()-started,'updates':step,'path':str(path)}


def physical_qa(model):
    from evaluate_local_memory_0098 import energy_power_qa,jacobian_qa
    from symbolic_closure_rom import numpy_field
    grid=.03*torch.sin(torch.arange(320,dtype=torch.float64).reshape(8,40))
    with torch.no_grad():
        actual=model.field(grid).numpy()
        independent=numpy_field(model,grid.numpy())
    _,jqa=jacobian_qa(model);pqa=energy_power_qa(model)
    error=float(np.max(np.abs(actual-independent)))
    return {'power':pqa,'jacobian':jqa,'numpy_field_max_abs':error,
            'pass':pqa['pass'] and jqa['pass'] and error<1e-9}


def timing_probe(model,events,scale,path):
    from train_local_memory_0098 import graph
    from train_joint_numerical_pair_0100 import next_samples
    from train_local_memory_0098 import histories_for
    rng=np.random.default_rng(6701);opt=optimizer(model);rows=[]
    coverage={'cover_order':rng.permutation(len(events)).tolist(),'cursor':0}
    before=copy.deepcopy(model.state_dict())
    for h in (1.,2.,5.):
        samples,_=next_samples(events,rng,coverage);hist=histories_for(samples)
        t=time.perf_counter();opt.zero_grad(set_to_none=True)
        with graph(model):
            v,_=loss(model,samples,hist,h,scale);v.backward()
        rows.append({'horizon':h,'seconds':time.perf_counter()-t,'finite':bool(torch.isfinite(v))})
    if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):raise AssertionError('timing mutated source')
    s=score(model,events)
    result={'rows':rows,'max_measured_step_s':max(r['seconds'] for r in rows),
            'mean_measured_step_s':float(np.mean([r['seconds'] for r in rows])),
            'train_score_seconds':s['seconds'],'parameter_updates':False}
    write(path,result);return result


def proposal_model(parent,proposal):
    model=parent.with_terms(proposal['terms'])
    if 'direction_mask' in proposal:
        with torch.no_grad():model.template_mask[3:7]=torch.as_tensor(proposal['direction_mask'],dtype=torch.float64)
    return model


def rank_proposals(beam,mode,seed,events):
    """Use a short autonomous TRAIN gradient only to shortlist legal edits."""
    from closure_discovery_support import propose_mutations,select_mutation_shortlist
    from train_local_memory_0098 import panel_events,autonomous,graph,risk
    pool=propose_mutations([m.terms for m in beam],mode=mode,seed=seed,max_candidates=64)
    for parent,m in enumerate(beam):
        for index in range(4):
            mask=m.template_mask[3:7].detach().tolist();mask[index]=1.-mask[index]
            key=hashlib.sha256(json.dumps({'terms':m.terms,'directions':mask},sort_keys=True).encode()).hexdigest()
            pool.append({'parent':parent,'terms':m.terms,'direction_mask':mask,
                         'mutation':'add' if mask[index] else 'delete','candidate_key':key})
    panel=panel_events(events);scores={};records=[]
    for proposal in pool:
        try:
            model=proposal_model(beam[proposal['parent']],proposal)
            with graph(model):
                value=risk(autonomous(model,panel,.2))
                parameters=[model.coefficients]+[p for p in model._template_scalars[3:7] if p.requires_grad]
                gradients=torch.autograd.grad(value,parameters,allow_unused=True)
                projected=float(value.detach())-.001*sum(float(g.detach().square().sum()) for g in gradients if g is not None)
            if not np.isfinite(projected):raise FloatingPointError('nonfinite screening gradient')
            scores[proposal['candidate_key']]=projected
            records.append({'key':proposal['candidate_key'],'projected_train_score':projected,'status':'screened'})
        except (ValueError,FloatingPointError) as error:
            records.append({'key':proposal['candidate_key'],'status':'unresolved','error':repr(error)})
    return select_mutation_shortlist(pool,scores,max_total=6,seed=seed),records


def stage_a(out,deadline):
    from shared_closure_rom_0101 import load_model
    from symbolic_closure_rom import SymbolicClosureROM
    from closure_discovery_support import select_depth
    cfg,_=context();model=source();old=load_model(ROOT/cfg['source'])
    grid=.03*torch.sin(torch.arange(320,dtype=torch.float64).reshape(8,40))
    with torch.no_grad():parity=float((model.field(grid)-old.field(grid)).abs().max())
    qa=physical_qa(model);write(out/'A/parity_physics.json',{'source_sha':SOURCE_HASH,'parity':parity,**qa})
    if parity>1e-10 or not qa['pass']:raise AssertionError('source reconstruction/physics failed')
    model.export(out/'A/source_model.json')
    restored=SymbolicClosureROM.from_export(out/'A/source_model.json')
    with torch.no_grad():
        if (model.field(grid)-restored.field(grid)).abs().max()>1e-12:raise AssertionError('roundtrip changed field')
    events,excluded,scale=train_inputs()
    write(out/'A/train_identity.json',{'eligible':[e['id'] for e in events],'excluded':excluded,'force_scale':scale})
    # Confirmation conditions are locked before any search result is obtained.
    from closure_discovery_data import allowed_train_conditions,generate_conditions
    from closure_discovery_support import generate_new_fault_manifest
    manifest=generate_new_fault_manifest(allowed_train_conditions(),seed=3701)
    write(out/'new_fault_manifest.json',manifest)
    # Preflight new TRAIN FOM only. No new-val simulation or labels are opened.
    fom_folder=out/'A/fom_cost'
    if (fom_folder/'generation.json').exists():fom=json.loads((fom_folder/'generation.json').read_text())
    else:fom=generate_conditions([{**r,'split':'train'} for r in manifest['train_pool'][:2]],fom_folder,'train',min(1200.,deadline-time.perf_counter()-60))
    durations=[a['elapsed_seconds'] for row in fom['attempts'] for a in row.get('steps',{}).values() if 'elapsed_seconds' in a]
    # Fall back to the whole measured batch cost per call if schema has no step durations.
    fom_cost=float(max(durations)) if durations else float(fom['elapsed_seconds'])/max(1,2*len(fom['attempts']))
    timing=timing_probe(model,events,scale,out/'A/timing_source.json')
    from closure_discovery_benchmarks import controlled_benchmarks
    controlled=controlled_benchmarks(out/'A/controlled',deadline=deadline)
    if not controlled['implementation_pass']:raise AssertionError('controlled implementation check failed')
    measured=timing['max_measured_step_s']
    estimates={'max_measured_step_s':measured,'known_eval_cost_s':{'B':36*timing['train_score_seconds'],
               'C':14*max(251.,timing['train_score_seconds']),'D':14*max(251.,timing['train_score_seconds'])},
               'c_step_s':measured,'d_step_s':measured,'fom_cost_s':fom_cost,'evaluation_cost_s':max(251.,timing['train_score_seconds'])}
    # A representative generated expression is timed before locking depth.
    term={'channel':'port','expression':{'op':'tanh','scale':1.,'arg':{'op':'var','name':'q'}}}
    expanded=model.with_terms(model.terms+[term])
    expanded_timing=timing_probe(expanded,events,scale,out/'A/timing_generated.json')
    estimates['max_measured_step_s']=max(measured,expanded_timing['max_measured_step_s'])
    estimates['c_step_s']=estimates['d_step_s']=estimates['max_measured_step_s']
    estimates['seconds_by_horizon']={str(h):max(next(r['seconds'] for r in timing['rows'] if r['horizon']==h),
         next(r['seconds'] for r in expanded_timing['rows'] if r['horizon']==h)) for h in (1.,2.,5.)}
    choice=select_depth(estimates,stage_caps=cfg['stage_caps_seconds'])
    write(out/'A/depth_lock.json',{'timing':estimates,'choice':choice,'frozen_before_search':True,
                                 'fom_preflight_qualified':fom['qualified_count']==2,
                                 'source_sha':SOURCE_HASH,'protocol_sha':sha(ROOT/cfg['protocol'])})
    return {'stage':'A','parity':parity,'controlled':controlled,'depth_choice':choice}


def stage_b(out,deadline):
    from symbolic_closure_rom import SymbolicClosureROM,term_key
    cfg,_=context();lock=json.loads((out/'A/depth_lock.json').read_text())
    choice=lock['choice'];depth=choice.get('depth') if isinstance(choice,dict) else choice
    if not lock.get('fom_preflight_qualified'):
        raise RuntimeError('FOM cost requires a successful qualified preflight; failed imports cannot price confirmation')
    if not lock.get('full_expression_depth_priced'):
        raise RuntimeError('price legal H/port expression growth before selecting a search depth')
    if not lock.get('confirmation_execution_path_priced'):
        raise RuntimeError('price the actual batch-one three-grid confirmation path before B')
    from closure_discovery_support import ProcessLedger,select_depth
    effective_caps=ProcessLedger(out/'ledger').caps_seconds
    if choice!=safe(select_depth(lock['timing'],stage_caps=effective_caps)):
        raise RuntimeError('frozen depth choice does not match the effective stage allocation')
    if not depth:raise RuntimeError('registered minimum complete research schedule does not fit measured budget')
    if cfg['search'].get('selected_rounds',depth)!=depth:raise ValueError('registered search depth changed')
    events,_,scale=train_inputs();records=[];failures=[];searches=[]
    for mode in ('library','generated'):
        for seed in cfg['search_seeds']:
            beam=[source()]
            for round_ in range(int(depth)):
                folder=out/f'B/{mode}/{seed}/round{round_+1}';folder.mkdir(parents=True,exist_ok=True)
                proposals,screen=rank_proposals(beam,mode,seed+round_,events)
                write(folder/'screening.json',screen)
                if len(proposals)<2:raise RuntimeError('fewer than two distinct legal proposals; no invented alternative')
                fitted=[]
                for i,proposal in enumerate(proposals):
                    parent=int(proposal.get('parent',0));terms=proposal.get('terms',proposal)
                    m=proposal_model(beam[parent],proposal)
                    path=folder/f'candidate{i:02d}'
                    if (path/'candidate_failure.json').exists():
                        failures.append(json.loads((path/'candidate_failure.json').read_text()));continue
                    try:
                        actual=json.loads((path/'checkpoint.json').read_text())['step'] if (path/'checkpoint.json').exists() else 0
                        if actual>=32:m=SymbolicClosureROM.from_export(path/'model.json')
                        else:m,_=fit(m,events,scale,32,path,seed,deadline=deadline)
                        s=score_at_level(path,events,32,deadline=deadline);fitted.append((s['risk'],i,m,s))
                    except FloatingPointError as error:
                        failure={'mode':mode,'seed':seed,'path':str(path),'level':32,'status':'numerically_unresolved','error':repr(error)}
                        write(path/'candidate_failure.json',failure);failures.append(failure)
                selected=sorted(fitted,key=lambda a:(a[0],a[1]))[:2]
                final=[]
                for _,i,m,_ in selected:
                    path=folder/f'candidate{i:02d}'
                    try:
                        for level in (128,512):
                            actual=json.loads((path/'checkpoint.json').read_text())['step']
                            if actual<level:m,_=fit(m,events,scale,level,path,seed,deadline=deadline)
                        s=score_at_level(path,events,512,deadline=deadline)
                        final.append((s['risk'],i,m,s))
                    except FloatingPointError as error:
                        failure={'mode':mode,'seed':seed,'path':str(path),'level':level,'status':'numerically_unresolved','error':repr(error)}
                        write(path/'candidate_failure.json',failure);failures.append(failure)
                final.sort(key=lambda a:(a[0],a[1]));beam=[a[2] for a in final]
                for r,i,m,s in final:
                    row={'mode':mode,'seed':seed,'round':round_+1,'path':str(folder/f'candidate{i:02d}'),
                         'score':s,'terms':m.terms,'direction_mask':m.template_mask[3:7].detach().tolist(),
                         'counts':m.complexity_counts()}
                    records.append(row)
                write(out/'B/search_results.json',records)
                write(out/'B/numerical_failures.json',failures)
                if not beam:break
            searches.append({'mode':mode,'seed':seed,'completed_512_candidates':len(beam),
                             'status':'rankable' if beam else 'numerically_unresolved'})
            write(out/'B/search_attempts.json',searches)
    # Preserve the error/complexity front; all complete candidates retain histories.
    def objectives(row):
        return (row['score']['risk'],len(row['terms']),
                row['counts']['symbolic_closure']['operator_nodes'],sum(row['direction_mask']))
    fronts={}
    for mode in ('library','generated'):
        pool=[r for r in records if r['mode']==mode]
        fronts[mode]=[r for r in pool if not any(all(a<=b for a,b in zip(objectives(other),objectives(r))) and
            any(a<b for a,b in zip(objectives(other),objectives(r))) for other in pool)]
    write(out/'B/pareto_front.json',fronts)
    # Select on train only, at the same completed adaptation level.
    selected={}
    for mode in ('library','generated'):
        pool=fronts[mode]
        if not pool:raise RuntimeError(f'no numerically rankable {mode} candidate; remaining attempts preserved')
        best=min(pool,key=lambda r:r['score']['risk']);bound=best['score']['risk']+best['score']['group_se']
        near=[r for r in pool if r['score']['risk']<=bound]
        winner=min(near,key=lambda r:(len(r['terms']),r['counts']['symbolic_closure']['operator_nodes'],r['score']['risk']))
        selected[mode]=winner
    # Observable forecasts decide distinctness, not differences of latent formulas.
    from train_local_memory_0098 import graph,panel_events,histories_for
    from train_structured_rom import _point_error
    panel=panel_events(events);hist=histories_for([(e,10) for e in panel])
    main=SymbolicClosureROM.from_export(Path(selected['generated']['path'])/'model.json')
    with torch.no_grad(),graph(main):p0=main.rollout(hist,100,.01)[...,:30]
    alternatives=[]
    mainkeys={term_key(t) for t in main.terms}
    for row in records:
        if row['mode']!='generated' or {term_key(t) for t in row['terms']}==mainkeys:continue
        if row['score']['risk']>selected['generated']['score']['risk']+selected['generated']['score']['group_se']:continue
        other=SymbolicClosureROM.from_export(Path(row['path'])/'model.json')
        with torch.no_grad(),graph(other):p=other.rollout(hist,100,.01)[...,:30]
        difference=float(_point_error(p,p0,main).mean())
        if np.isfinite(difference) and difference>1e-4:alternatives.append((difference,row))
    if not alternatives:
        write(out/'B/selection.json',{'selected':selected,'alternative':None,'status':'no_distinct_near_optimal_alternative'})
        raise RuntimeError('no distinct near-optimal observable alternative; discrimination claim unresolved')
    selected['alternative']=max(alternatives,key=lambda a:a[0])[1]
    selected['alternative_disagreement']={'normalized_mse':max(alternatives,key=lambda a:a[0])[0],
        'minimum':1e-4,'horizon_s':1.,'scope':'fixed train panel, observable coordinates only'}
    write(out/'B/selection.json',selected)
    return {'stage':'B','attempted_searches':6,'rankable_searches':sum(s['status']=='rankable' for s in searches),
            'numerical_failures':len(failures),'depth':depth,'selected':selected}


def stage_c(out,deadline):
    cfg,_=context();selection=json.loads((out/'B/selection.json').read_text())
    events,_,scale=train_inputs();structures={'fixed':source().terms,
        'library':selection['library']['terms'],'main':selection['generated']['terms'],
        'alternative':selection['alternative']['terms']}
    frozen=[];failures=[]
    for name,terms in structures.items():
        for seed in cfg['refit_seeds']:
            m=source().with_terms(terms)
            if name!='fixed':
                selected=selection[{'main':'generated'}.get(name,name)]
                with torch.no_grad():m.template_mask[3:7]=torch.as_tensor(selected['direction_mask'],dtype=torch.float64)
            path=out/f'C/{name}/{seed}'
            try:
                m,_=fit(m,events,scale,1024,path,seed,final=True,deadline=deadline)
                s=score(m,events,deadline=deadline);write(path/'train_score.json',s)
                qa=physical_qa(m);write(path/'physics.json',qa)
                if not qa['pass']:raise FloatingPointError('physical implementation QA unresolved')
                frozen.append({'name':name,'seed':seed,'path':str(path),'sha256':sha(path/'model.json'),'qa':qa})
            except FloatingPointError as error:
                failure={'name':name,'seed':seed,'path':str(path),'error':repr(error),'status':'numerically_unresolved'}
                write(path/'candidate_failure.json',failure);failures.append(failure)
            write(out/'C/progress.json',{'valid_endpoints':frozen,'numerical_failures':failures,'test_read':False})
    if not any(r['name']=='main' and r['seed']==2701 for r in frozen):
        raise RuntimeError('registered main seed is unresolved; independent refits retained, no invented deletion source or confirmation')
    # Deletion choice: two strongest observable effects with all other parameters fixed;
    # the actual evidence comes from the subsequent complete re-adaptation.
    from symbolic_closure_rom import SymbolicClosureROM
    main=SymbolicClosureROM.from_export(out/'C/main/2701/model.json')
    effects=[];unscored_deletions=[]
    for i in range(len(main.terms)):
        reduced=main.with_terms(main.terms[:i]+main.terms[i+1:])
        try:effects.append((score(reduced,events,deadline=deadline)['risk'],i))
        except FloatingPointError as error:
            unscored_deletions.append({'term':main.terms[i],'index':i,'status':'numerically_unresolved','error':repr(error)})
    write(out/'C/deletion_screening.json',{'valid_risks':[{'risk':v,'index':i} for v,i in effects],
         'unresolved':unscored_deletions,'scope':'fixed-parameter proposal ordering; no necessity conclusion'})
    for _,i in sorted(effects,reverse=True)[:2]:
        terms=main.terms[:i]+main.terms[i+1:];path=out/f'C/delete{i}/2701'
        m=source().with_terms(terms)
        with torch.no_grad():m.template_mask[3:7]=main.template_mask[3:7]
        try:
            m,_=fit(m,events,scale,1024,path,2701,final=True,deadline=deadline)
            write(path/'train_score.json',score(m,events,deadline=deadline))
            frozen.append({'name':f'delete{i}','seed':2701,'deleted':main.terms[i],
                           'path':str(path),'sha256':sha(path/'model.json')})
        except FloatingPointError as error:
            failure={'name':f'delete{i}','seed':2701,'path':str(path),'error':repr(error),'status':'numerically_unresolved'}
            write(path/'candidate_failure.json',failure);failures.append(failure)
        write(out/'C/progress.json',{'valid_endpoints':frozen,'numerical_failures':failures,'test_read':False})
    if failures:raise RuntimeError('complete C comparison unresolved; all independent permitted refits were attempted and retained')
    write(out/'C/frozen.json',{'models':frozen,'source_sha':SOURCE_HASH,'test_read':False})
    return {'stage':'C','models':len(frozen)}


def stage_e(out,deadline):
    from symbolic_closure_rom import SymbolicClosureROM
    from closure_discovery_export import export_numpy_package
    freeze_path=out/'D/confirmation_freeze.json'
    if not freeze_path.exists():raise RuntimeError('final D coefficient freeze is required for E delivery')
    frozen=json.loads(freeze_path.read_text());results=[]
    events,_,_=train_inputs()
    from train_local_memory_0098 import graph,histories_for
    from train_structured_rom import _point_error
    import importlib.util
    panel=[events[i] for i in (0,len(events)//2,len(events)-1)]
    history=histories_for([(e,10) for e in panel])
    for row in frozen['models']:
        path=Path(row['path']);path=path/'model.json' if path.is_dir() else path
        if sha(path)!=row['sha256']:raise ValueError('frozen model changed')
        m=SymbolicClosureROM.from_export(path);qa=physical_qa(m)
        if time.perf_counter()+60>=deadline:raise TimeoutError('independent final validation incomplete')
        folder=out/f"E/export/{row['name']}_{row['seed']}"
        paths=export_numpy_package(m,folder)
        spec=importlib.util.spec_from_file_location('closure_numpy_runtime',paths['simulator'])
        runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
        payload=json.loads(paths['model'].read_text())
        with torch.no_grad(),graph(m):actual=m.rollout(history,800,.01).detach()
        independent=runtime.rollout(payload,history.numpy(),800,.01)
        finite=bool(torch.isfinite(actual).all() and np.isfinite(independent).all())
        error=float(_point_error(torch.as_tensor(independent),actual,m).max()) if finite else None
        independent_qa={'events':[e['id'] for e in panel],'seconds':8.,'dt':.01,
            'all_finite':finite,'max_normalized_point_error':error,
            'pass':finite and error<1e-12,
            'scope':'same saved equations and initializer, independent NumPy RK4; implementation agreement, not physical accuracy or stability'}
        write(folder/'independent_run.json',independent_qa)
        results.append({**row,'qa':qa,'independent_run':independent_qa,'counts':m.complexity_counts()})
    write(out/'E/explicit_models.json',results)
    return {'stage':'E','models':len(results),'scope':'explicit field and local implementation checks; no automatic stability certificate'}


def retry_fom_preflight(out,deadline):
    """Retain failed attempts, repair runtime, and reprice before any B result."""
    from closure_discovery_data import generate_conditions
    from closure_discovery_support import select_depth
    if (out/'B/search_results.json').exists():raise RuntimeError('cannot reprice after structural search results')
    manifest=json.loads((out/'new_fault_manifest.json').read_text())
    folder=next(out/f'A/fom_cost_retry{i}' for i in range(1,5) if not (out/f'A/fom_cost_retry{i}/generation.json').exists())
    result=generate_conditions([{**c,'split':'train'} for c in manifest['train_pool'][:2]],folder,'train',min(1200.,deadline-time.perf_counter()-60))
    if result['qualified_count']!=2:raise RuntimeError('FOM preflight not fully qualified; failure evidence retained')
    lock=json.loads((out/'A/depth_lock.json').read_text())
    old=out/'A/depth_lock_initial.json'
    if not old.exists():write(old,lock)
    # Whole batch duration is a conservative per-call planning allowance.
    lock['timing']['fom_cost_s']=float(result['elapsed_seconds'])
    lock['choice']=select_depth(lock['timing'],stage_caps=context()[0]['stage_caps_seconds']);lock['fom_preflight_qualified']=True
    lock['fom_cost_artifact']=str(folder/'generation.json')
    write(out/'A/depth_lock.json',lock)
    return {'stage':'A','repair':'qualified frozen-runtime FOM preflight','qualified':2,'depth_choice':lock['choice']}


def complete_cost_preflight(out,deadline):
    """Price every allowed growth depth with increasingly complex legal trees."""
    from closure_discovery_support import select_depth
    from closure_discovery_benchmarks import v,mul,power
    lock=json.loads((out/'A/depth_lock.json').read_text())
    if not lock.get('fom_preflight_qualified'):raise RuntimeError('qualified FOM preflight required')
    if (out/'B/search_results.json').exists():raise RuntimeError('cost lock is prospective only')
    events,_,scale=train_inputs();base=source();bydepth={};profiles=[];eval_bydepth={}
    folder=next(out/f'A/cost_preflight_{i}' for i in range(1,9) if not (out/f'A/cost_preflight_{i}').exists())
    folder.mkdir(parents=True)
    terms=[{'channel':'edge','expression':mul(power({'op':'add','args':[v('qi'),v('qj')]},2),
                                                     power({'op':'add','args':[{'op':'const','value':1},v('c')]},2))},
           {'channel':'node','expression':mul(power(v('p'),2),power(v('q'),2))},
           {'channel':'port','expression':{'op':'tanh','arg':v('q'),'scale':1.}}]
    for depth in (1,2,3):
        model=base.with_terms(base.terms+terms[:depth])
        profile=timing_probe(model,events,scale,folder/f'timing_depth{depth}.json')
        profiles.append(profile)
        bydepth[str(depth)]={str(h):max(next(r['seconds'] for r in profile['rows'] if r['horizon']==h),
                 float(lock['timing']['seconds_by_horizon'][str(h)])) for h in (1.,2.,5.)}
        evaluation=profile['train_score_seconds']
        eval_bydepth[str(depth)]={'B':48*depth*evaluation+6*depth*72*.2*bydepth[str(depth)]['1.0'],
            'C':14*max(251.,evaluation)+12*evaluation,'D':14*max(251.,evaluation)}
    lock['timing']['seconds_by_depth']=bydepth
    lock['timing']['eval_by_depth']=eval_bydepth
    lock['choice']=select_depth(lock['timing'],stage_caps=context()[0]['stage_caps_seconds']);lock['full_expression_depth_priced']=True
    previous=out/'A/depth_lock_before_full_expression_price.json'
    if not previous.exists():previous.write_bytes((out/'A/depth_lock.json').read_bytes())
    write(folder/'previous_depth_lock.json',json.loads((out/'A/depth_lock.json').read_text()))
    lock['cost_profile_artifact']=str(folder);write(out/'A/depth_lock.json',lock)
    return {'stage':'A','cost_profiles':profiles,'choice':lock['choice']}


def confirmation_cost_preflight(out,deadline):
    """Price actual batch-one, three-grid confirmation without opening new val."""
    from closure_discovery_confirmation import evaluate_events,train_disagreement_rows
    from closure_discovery_data import load_new_events,allowed_train_conditions
    from closure_discovery_benchmarks import v,mul,power
    base=source();term={'channel':'edge','expression':mul(power({'op':'add','args':[v('qi'),v('qj')]},2),
                                     power({'op':'add','args':[{'op':'const','value':1},v('c')]},2))}
    model=base.with_terms(base.terms+[term])
    events,excluded=load_new_events(out/'A/fom_cost_retry2','train')
    if len(events)!=2 or excluded:raise RuntimeError('two qualified causal train timing events required')
    folder=next(out/f'A/confirmation_cost_{i}' for i in range(1,5) if not (out/f'A/confirmation_cost_{i}').exists())
    folder.mkdir(parents=True);costs=[]
    for index,e in enumerate(events):
        start=time.perf_counter()
        result=evaluate_events(model,[e],folder/f'event{index}.json',force_finer=True,deadline=deadline)
        costs.append(time.perf_counter()-start)
    old,_,_=train_inputs();metadata={str(r['event_id']):r for r in allowed_train_conditions()}
    start=time.perf_counter();rows=train_disagreement_rows(model,model,old,metadata,deadline=deadline)
    disagreement_seconds=time.perf_counter()-start
    if len(rows)!=len(old):raise AssertionError('timing lost train events')
    profile={'max_three_grid_event_seconds':max(costs),'event_seconds':costs,
        'full_train_two_prediction_disagreement_seconds':disagreement_seconds,
        'new_val_cost_bound_seconds':14*24*max(costs),'events':len(events),
        'parameter_updates':False,'val_read':False,'scope':'implementation cost on two preregistered train conditions'}
    write(folder/'timing.json',profile)
    lock=json.loads((out/'A/depth_lock.json').read_text());write(folder/'previous_depth_lock.json',lock)
    for timings in lock['timing']['eval_by_depth'].values():
        timings['D']=max(timings['D'],profile['new_val_cost_bound_seconds']+disagreement_seconds)
    from closure_discovery_support import ProcessLedger,select_depth
    lock['choice']=select_depth(lock['timing'],stage_caps=ProcessLedger(out/'ledger').caps_seconds)
    lock['confirmation_execution_path_priced']=True;lock['confirmation_cost_artifact']=str(folder/'timing.json')
    write(out/'A/depth_lock.json',lock)
    return {'stage':'A','profile':profile,'choice':lock['choice']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['A','B','C','D','E','preflight-fom','preflight-cost','preflight-confirmation'])
    args=parser.parse_args();torch.set_num_threads(1)
    cfg,out=context()
    from closure_discovery_support import ProcessLedger
    ledger=ProcessLedger(out/'ledger')
    if {k:float(v) for k,v in cfg['stage_caps_seconds'].items()}!=ledger.caps_seconds:
        raise ValueError('configuration and prospectively registered allocation differ')
    # Ledger APIs are intentionally small; the receipt owns this scientific process.
    stage='A' if args.stage.startswith('preflight-') else args.stage;started=time.perf_counter()
    ledger.record_external(stage,'process-import-startup',started-PROCESS_STARTED)
    remaining=ledger.remaining(stage)
    if remaining<=30:raise RuntimeError('stage resource exhausted')
    if stage in ('B','C','D','E'):bind_execution_sources(out)
    write(out/'active_job.json',{'pid':os.getpid(),'stage':stage,'started_utc':time.time(),
                               'status':'RUNNING','command':sys.argv,'working_directory':str(ROOT)})
    status='failed';result=None
    try:
        with ledger.run(stage,f'closure-discovery-{stage}'):
            if args.stage=='preflight-fom':result=retry_fom_preflight(out,time.perf_counter()+remaining-15)
            elif args.stage=='preflight-cost':result=complete_cost_preflight(out,time.perf_counter()+remaining-15)
            elif args.stage=='preflight-confirmation':result=confirmation_cost_preflight(out,time.perf_counter()+remaining-15)
            elif stage=='D':
                from closure_discovery_confirmation import run_confirmation
                result=run_confirmation(out,time.perf_counter()+remaining-15)
            else:result={'A':stage_a,'B':stage_b,'C':stage_c,'E':stage_e}[stage](out,time.perf_counter()+remaining-15)
            status='complete';write(out/f'{stage}/{args.stage}_result.json' if args.stage.startswith('preflight-') else out/f'{stage}/result.json',result)
    except BaseException as error:
        write(out/f'{stage}/failure_{int(time.time())}.json',{'error':repr(error),'traceback':traceback.format_exc(),
                     'elapsed':time.perf_counter()-started,'test_read':False,'status':'unresolved'})
        raise
    finally:
        write(out/'active_job.json',{'pid':os.getpid(),'stage':stage,'status':status,'elapsed':time.perf_counter()-started})
    print(json.dumps(safe(result)),flush=True)


if __name__=='__main__':main()
