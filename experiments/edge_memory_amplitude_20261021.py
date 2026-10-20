"""Train-only three-seed edge memory amplitude study. Every scientific invocation is native-supervised."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'private_data/closure_mechanism_20261019/P2/edge_memory_amplitude01'
SOURCE_MANIFEST=ROOT/'results/spatial_extrema_20261020/frozen_evaluation_endpoints.json'
SOURCE_MANIFEST_SHA='b8c525f9b87709528c74f1b64724aace4b653e4a6948765da8471599cc1d0af4'
QUARTIC_BASE=ROOT/'private_data/closure_mechanism_20261019/P2/quartic_memory01'
SCHEDULE=QUARTIC_BASE/'prepare01/schedule.json'
PREPARE=QUARTIC_BASE/'prepare01/prepare.json'
FRESH=ROOT/'private_data/closure_mechanism_20261019/P3/fresh_design_pair01'
SEEDS=(6101,6102,6103)
VARIANTS=('E',)
RATES={'skeleton':.0009,'closure':.00009,'encoder':.000027}
FORCE=1.0068379869405584
SOURCE_ENTRIES={6101: {'seed': 6101, 'variant': 'O', 'model_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\temporal_objective01\\models\\6101_O\\checkpoints\\step000512\\model.json', 'model_sha256': '749c8f15cd72d08a253d9381766abec82d04b55c69bc04ccf50ce6fdd2bba39d', 'training_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\temporal_objective01\\models\\6101_O\\checkpoints\\step000512\\training.pt', 'training_sha256': 'bafcc320fdc8c19395f535d9dc7ab801c7f857677e4f7c3e8ca33a45d50251d7', 'checkpoint_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\temporal_objective01\\models\\6101_O\\checkpoints\\step000512\\checkpoint.json', 'checkpoint_sha256': 'de35c95ac18ac70538a7176df25d67f077509155c2bc280fd51788c1985bc428', 'imported_control': True, 'stage_step': 512}, 6102: {'seed': 6102, 'variant': 'O', 'model_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\spatial_extrema01\\models\\6102_O\\checkpoints\\step000512\\model.json', 'model_sha256': 'eecc25b1a98b470ad8333cabc1fb0a97dca1774870665d60d03a18a3feea021e', 'training_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\spatial_extrema01\\models\\6102_O\\checkpoints\\step000512\\training.pt', 'training_sha256': '85d7739e8c10a1b4042ef7769b5720a1f510bde496bb5092b7f020d9175de244', 'checkpoint_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\spatial_extrema01\\models\\6102_O\\checkpoints\\step000512\\checkpoint.json', 'checkpoint_sha256': 'f58e99562c79b829a8dc1e90a63ca2813014051011276c16881490f5e3194da9', 'imported_control': False, 'stage_step': 512}, 6103: {'seed': 6103, 'variant': 'O', 'model_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\spatial_extrema01\\models\\6103_O\\checkpoints\\step000512\\model.json', 'model_sha256': '3cdc572fe6a43420185588bd7b16cc468ddd837f1e6a9375f69d62f6ceac169f', 'training_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\spatial_extrema01\\models\\6103_O\\checkpoints\\step000512\\training.pt', 'training_sha256': 'd9afc003a7b6f4ef080658750ee673e7afc0b5e06d755635136f1268e05d31d4', 'checkpoint_path': 'E:\\03-nn_Transfer_Func\\kiki\\private_data\\closure_mechanism_20261019\\P2\\spatial_extrema01\\models\\6103_O\\checkpoints\\step000512\\checkpoint.json', 'checkpoint_sha256': '2b16cdfa08165ad337cbca25cab968cd58c6d3f76bd46a299a3161b16422723b', 'imported_control': False, 'stage_step': 512}}
SOURCE_HASHES={seed:(e["model_sha256"],e["training_sha256"],e["checkpoint_sha256"]) for seed,e in SOURCE_ENTRIES.items()}


def baseline_contract():
    manifest=ROOT/'results/quartic_memory_20261020/frozen_evaluation_endpoints.json'
    metrics=ROOT/'results/quartic_memory_20261020/fixed_train_metrics_step001024.json'
    if sha(manifest)!='a1b411dae2c00867611afe390d934b8cc718d55d3103dd81aba0b6229aa34842' or sha(metrics)!='4f53916743184e269541f42797a829bbcf938b7a8047b18978147f7d9f107820':raise ValueError('frozen N/Q evidence changed')
    schedule=checked_schedule();eq=read(PREPARE)['equilibrium_delta']
    endpoints=read(manifest)['endpoints']
    if {(e['seed'],e['variant']) for e in endpoints}!={(s,v) for s in SEEDS for v in ('N','Q')}:raise ValueError('baseline roles differ')
    for e in endpoints:
        for kind in ('model','training','checkpoint'):
            if sha(Path(e[kind+'_path']))!=e[kind+'_sha256']:raise ValueError('baseline endpoint bytes differ')
        identity=read(e['checkpoint_path'])['identity'];seed=e['seed']
        if identity['source_model_sha256']!=SOURCE_HASHES[seed][0] or identity['source_training_sha256']!=SOURCE_HASHES[seed][1] or identity['schedule_sha256']!=sha(SCHEDULE) or identity['data']!=schedule['events'] or identity['force']!=FORCE or identity['rates']!=RATES or identity['equilibrium_delta']!=eq:raise ValueError('baseline source/schedule/objective identity differs')
    for e in endpoints:
        meta=read(e['checkpoint_path']);rows=schedule['seeds'][str(e['seed'])]['rows']
        if len(meta['trace'])!=1024:raise ValueError('baseline full adaptation missing')
        for step,(t,row) in enumerate(zip(meta['trace'],rows),1):
            expected={k:x*(.3 if step>512 else 1.) for k,x in RATES.items()}
            if t['stage_step']!=step or t['sampling']!=row['sampling'] or t['horizon_s']!=row['horizon_s'] or t['learning_rates']!=expected:raise ValueError('baseline training semantics differ')
    baseline_initial_contract(endpoints,schedule)
    return {'baseline_roles':6,'schedule_sha256':sha(SCHEDULE),'baselines_refit_or_reevaluated':False}

def baseline_initial_contract(endpoints,schedule):
    # Read-only JSON producer contract; no baseline model construction or evaluation.
    for e in endpoints:
        seed=e['seed'];parent=Path(e['model_path']).parent.parent/'step000000'
        initial=read(parent/'checkpoint.json')
        if initial['step']!=0 or initial['trace']:raise ValueError('baseline Adam starting stage changed')
        if initial['identity']!=read(e['checkpoint_path'])['identity']:raise ValueError('baseline stage identity changed')
        src=read(source_folder(seed)/'model.json');src=src.get('model',src)
        zero=read(parent/'model.json');zero=zero.get('model',zero)
        if not isinstance(src.get('trainable_parameters'),list) or zero.get('trainable_parameters')!=src['trainable_parameters']:raise ValueError('baseline trainable parameter mask differs')
        for key,value in src['state_dict'].items():
            if key!='coefficients' and zero['state_dict'].get(key)!=value:raise ValueError('baseline common zero-update state differs')
        from symbolic_closure_rom import term_key
        left={term_key(t):c for t,c in zip(src['terms'],src['coefficients'])}
        right={term_key(t):c for t,c in zip(zero['terms'],zero['coefficients'])}
        if len(right)!=len(zero['terms']) or any(right.get(k)!=c for k,c in left.items()):raise ValueError('baseline common initial coefficients differ')
        if len(right)!=7+(e['variant']=='Q') or any(c!=0 for k,c in right.items() if k not in left):raise ValueError('baseline extra initial term differs')
        rows=schedule['seeds'][str(seed)]['rows']
        if len(rows)!=1024 or [sum(x['horizon_s']==h for x in rows) for h in (1,2,5)]!=[342,341,341]:raise ValueError('baseline schedule length/phases differ')
        for row in rows:
            if len(row['sampling']['sample_ids'])!=12 or len(row['long_counts'])!=12 or len(row['local_counts'])!=12:raise ValueError('baseline sample/auxiliary contract differs')
    return True

def baseline_rng_contract(device):
    import torch
    from linear_cache_cost_gate_20261019 import _equal
    entries=read(ROOT/'results/quartic_memory_20261020/frozen_evaluation_endpoints.json')['endpoints']
    for e in entries:
        old=Path(e['training_path']).parent.parent/'step000000/training.pt'
        initial=torch.load(old,map_location='cpu',weights_only=False)
        source=torch.load(source_folder(e['seed'])/'training.pt',map_location='cpu',weights_only=False)
        if initial['step']!=0 or initial['optimizer']['state']:raise ValueError('baseline must start with new empty Adam')
        groups=initial['optimizer']['param_groups']
        if {g['block'] for g in groups}!=set(RATES) or len(groups)!=3 or any(g['lr']!=RATES[g['block']] for g in groups):raise ValueError('baseline Adam group/rate policy differs')
        for name in ('torch_rng','cuda_rng','numpy_rng','coverage'):_equal(initial[name],source[name],exact=True)
    return {'baseline_initial_rng_coverage_verified':6}

def scalar_interval(values,dt):
    import numpy as np
    v=np.asarray(values,dtype=float)
    if v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() or not math.isfinite(dt) or dt<=0:raise ValueError('finite initial plus future samples required')
    n=len(v)-1;duration=n*dt;integral=float(((v[:-1]+v[1:])*.5).sum()*dt)
    positive=negative=zero=0.
    for left,right in zip(v[:-1],v[1:]):
        if left==0 and right==0:zero+=dt
        elif left>=0 and right>=0:positive+=dt
        elif left<=0 and right<=0:negative+=dt
        else:
            first=abs(left)/(abs(left)+abs(right))*dt
            if left>0:positive+=first;negative+=dt-first
            else:negative+=first;positive+=dt-first
    return {'future_points':n,'duration_seconds':duration,'time_mean':integral/duration,'integral':integral,'sample_min':float(v.min()),'sample_max':float(v.max()),'positive_time_fraction':positive/duration,'negative_time_fraction':negative/duration,'zero_time_fraction':zero/duration}

def algebraic_interval(model,states,n,qualified,dt=.01):
    import numpy as np,torch
    from edge_memory_amplitude_runtime_20261021 import extra_energy_components
    row={'available':n>=1,'future_points':n,'duration_seconds':max(n,0)*dt,'numerically_admissible':bool(qualified),'summary':None,'energy_units':'H-unit','field_norm_units':'interface Euclidean norm','initial_included_once':True}
    x=np.asarray(states,dtype=float)
    if n<1 or x.ndim!=2 or len(x)<n+1:row['available']=False;return row
    if not qualified or x.shape[1]!=40 or not np.isfinite(x[:n+1]).all():return row
    # Pure algebra on the already computed sequence, no field/encode/rollout.
    try:
        with torch.no_grad():
            parts=extra_energy_components(model,torch.as_tensor(x[:n+1],dtype=model.mass.dtype,device=model.mass.device))
            energy=parts['energy'].detach().cpu().numpy()
            norms={k:torch.linalg.vector_norm(parts[k],dim=-1).detach().cpu().numpy() for k in ('field_delta','field_q')}
        if not np.isfinite(energy).all() or not all(np.isfinite(v).all() for v in norms.values()):return row
        summary={'energy':scalar_interval(energy,dt)}
        for k,v in norms.items():
            z=scalar_interval(v,dt);z['time_mean_square']=scalar_interval(v*v,dt)['time_mean'];summary[k]=z
        row['summary']=summary
    except (ValueError,RuntimeError,OverflowError,FloatingPointError):pass
    return row

def algebraic_evidence(model,event,fine,qualified8,qualified15):
    states=fine[::2] # Producer includes t0 exactly once.
    available=min(1500,len(event['x'])-11);result={}
    for h in (1,2,5,8):
        n=h*100;eligible=event['label']=='stable' and available>=n
        result[str(h)]=algebraic_interval(model,states,n,bool(qualified8 and eligible))
        result[str(h)]['available']=eligible
    result['actual_interval15']=algebraic_interval(model,states,available,qualified15)
    return result

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def write(p,value):
    from experiments.closure_discovery_data import _write_json
    _write_json(Path(p),value)
def inside(p):
    p=Path(p).resolve()
    if not p.is_relative_to(BASE.resolve()) or p==BASE.resolve(): raise ValueError('outside fixed study root')
    return p
def source_folder(seed):
    if seed not in SEEDS:raise ValueError('unregistered source seed')
    return Path(SOURCE_ENTRIES[seed]['model_path']).parent




def source_metadata(seed):
    if sha(SOURCE_MANIFEST)!=SOURCE_MANIFEST_SHA:raise ValueError('immutable O source manifest changed')
    p=source_folder(seed);m=read(p/'checkpoint.json')
    for n,h in zip(('model.json','training.pt','checkpoint.json'),SOURCE_HASHES[seed]):
        if sha(p/n)!=h:raise ValueError('source bytes changed')
    if m['step']!=512 or m['identity']['variant']!='O' or m['identity']['seed']!=seed or m['identity']['source_global_step']!=1280:raise ValueError('O512 source identity changed')
    return m


def binding(path):
    record=read(inside(path));files=record.get('files')
    if not isinstance(files,dict) or not files: raise ValueError('nonempty explicit binding required')
    for name,h in files.items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or p.suffix.lower() in ('.h5','.hdf5'): raise ValueError('forbidden bound input')
        if sha(p)!=h: raise ValueError('bound source/data/command changed: '+name)
    return sha(path)
def load_events():
    from experiments.active_fom_pair_fit_20261020 import load_events as load
    from experiments import active_fom_pair_fit_20261020 as fit
    spec=read(FRESH/'inputs/train_combined.json')
    if spec!={'old_train':True,'new_directories':[str(FRESH/'campaign01/train')],'split':'train'}:
        if (spec.get('split')!='train' or spec.get('old_train') is not True or len(spec.get('new_directories',[]))!=1 or Path(spec['new_directories'][0]).resolve()!=(FRESH/'campaign01/train').resolve()): raise ValueError('only exact registered385 train permitted')
    previous=fit.CAMPAIGN;fit.CAMPAIGN=FRESH.resolve()
    try: events,excluded,_=load(spec)
    finally: fit.CAMPAIGN=previous
    if len(events)!=385 or sum(e['label']=='stable' for e in events)!=342 or sum(e['label']=='unstable' for e in events)!=43: raise ValueError('385/342/43 train identity changed')
    return events,excluded

def checked_schedule(path=None):
    p=SCHEDULE if path is None else Path(path).resolve()
    if p!=SCHEDULE.resolve():raise ValueError('only new fixed1024 schedule allowed')
    if sha(p)!='07fd838bd2a9d136db9ce0b789562c26c93098cf5b6bba40af8f67d674bf2b0c' or sha(PREPARE)!='c145d1e3bcb29e3a6f417fd1ccb3ccb7efb8df35249b92a0ec398050dad7fca4':raise ValueError('old frozen schedule/prepare changed')
    result=read(p)
    if result.get('updates')!=1024:raise ValueError('new1024 schedule required')
    return result



def equilibrium_delta(events):
    import numpy as np
    shared=None
    if not events:raise ValueError('nonempty equilibrium metadata required')
    for e in events:
        values=e.get('classification_meta',{}).get('equilibrium_delta')
        if values is None:raise ValueError('explicit equilibrium_delta required')
        a=np.asarray(values,dtype=float)
        if a.shape!=(10,) or not np.isfinite(a).all():raise ValueError('finite ten-machine equilibrium required')
        if shared is None:shared=a
        elif not np.array_equal(shared,a):raise ValueError('equilibrium system/machine order differs')
    return shared.tolist()

def require_prepared_equilibrium(eq):
    prepared=read(PREPARE)
    if prepared['equilibrium_delta']!=eq or prepared['schedule_sha256']!=sha(SCHEDULE):raise ValueError('prepared physical metadata changed')

def endpoint_path(seed,variant):
    if seed not in SEEDS or variant not in VARIANTS:raise ValueError('unknown endpoint role')
    return BASE/'models'/f'{seed}_{variant}'/'checkpoints/step001024/model.json'



def endpoint_role(path):
    p=Path(path).resolve()
    roles={(endpoint_path(seed,v)).resolve():(seed,v) for seed in SEEDS for v in VARIANTS}
    if p not in roles:raise ValueError('exact registered role endpoint path required')
    return roles[p]

def validate_endpoint_payload(data,variant):
    from symbolic_closure_rom import term_key
    from edge_memory_amplitude_runtime_20261021 import TERMS_KEYS
    if variant!='E' or not isinstance(data,dict):raise ValueError('E role required')
    p=data.get('model',data);terms=p.get('terms');co=p.get('coefficients');state=p.get('state_dict')
    if not isinstance(terms,list) or not isinstance(co,list) or not isinstance(state,dict):raise ValueError('missing export arrays')
    keys=[term_key(t) for t in terms]
    if len(keys)!=8 or len(set(keys))!=8 or set(keys)!=set(TERMS_KEYS):raise ValueError('exact eight E terms required')
    if len(co)!=8 or any(isinstance(c,bool) or not isinstance(c,(float,int)) or not math.isfinite(c) for c in co):raise ValueError('finite eight coefficients required')
    if state.get('coefficients')!=co:raise ValueError('coefficient copies differ')
    return p

def check_endpoint(path):
    from edge_memory_amplitude_runtime_20261021 import ERuntime
    p=Path(path).resolve();seed,v=endpoint_role(p);folder=p.parent;m=read(folder/'checkpoint.json');identity=m['identity'];schedule=checked_schedule()
    if read(folder.parent/'latest.json')['directory']!='step001024':raise ValueError('latest checkpoint must name exact final stage')
    if m['step']!=1024 or len(m['trace'])!=1024 or (identity['seed'],identity['variant'])!=(seed,v):raise ValueError('complete N/Q1024 endpoint required')
    if (identity['source_model_sha256'],identity['source_training_sha256'])!=SOURCE_HASHES[seed][:2] or identity['schedule_sha256']!=sha(SCHEDULE) or identity['data']!=schedule['events'] or identity['rates']!=RATES or identity['force']!=FORCE:raise ValueError('source/data/optimizer semantics differ')
    actual=(sha(p),sha(folder/'training.pt'),sha(folder/'checkpoint.json'))
    if actual[:2]!=(m['model_sha256'],m['training_sha256']):raise ValueError('endpoint bytes differ')
    # Validate eight-term JSON identity without constructing a model or consuming RNG.
    validate_endpoint_payload(read(p),v)
    seen=set()
    for step,(t,row) in enumerate(zip(m['trace'],schedule['seeds'][str(seed)]['rows']),1):
        if t['stage_step']!=step or t['sampling']!=row['sampling'] or t['horizon_s']!=horizon_for(v,step):raise ValueError('trace schedule differs')
        expected={k:x*(.3 if step>512 else 1.) for k,x in RATES.items()}
        if t['learning_rates']!=expected or not math.isfinite(t['loss']) or not math.isfinite(t['gradient_norm']):raise ValueError('trace optimizer/finite guard differs')
        if not finite_tree(t['components']) or not finite_tree(t['edge_memory']):raise ValueError('nonfinite component or a_E trace')
        seen.update(t['sampling']['sample_ids'])
    if seen!={e['id'] for e in schedule['events']}:raise ValueError('coverage incomplete')
    return {'seed':seed,'variant':v,'model_path':str(p),'model_sha256':actual[0],'training_path':str(folder/'training.pt'),'training_sha256':actual[1],'checkpoint_path':str(folder/'checkpoint.json'),'checkpoint_sha256':actual[2],'stage_step':1024}

def finite_tree(value):
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    return not isinstance(value,(float,int)) or math.isfinite(value)



def freeze_endpoints():
    path=BASE/'frozen_evaluation_endpoints.json'
    if path.exists():raise FileExistsError('endpoint freeze already exists')
    entries=[check_endpoint(endpoint_path(seed,v)) for seed in SEEDS for v in VARIANTS]
    record={'endpoints':entries,'endpoint_count':3,'schedule_sha256':sha(SCHEDULE),'test_read':False,'val_read':False}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
    return record

def horizon_for(variant,step):
    if variant not in VARIANTS or type(step) is not int or not 1<=step<=1024:raise ValueError('unregistered phase')
    return (1.,2.,5.)[(step-1)%3]

















def enable_runtime(model):
    from edge_memory_amplitude_runtime_20261021 import enable_runtime as enable
    return enable(model)




def save_checked(output,model,opt,rng,coverage,step,identity,trace):
    from edge_memory_amplitude_runtime_20261021 import require_finite
    from fixed_edge_pair_20261019 import save_safe as publish
    require_finite(model)
    return publish(output,model,opt,rng,coverage,step,identity,trace)

def restore_stage(checkpoints,device,identity):
    from train_closure import restore_checkpoint
    from closure_checkpoint_identity_20261019 import restore_identity_checkpoint
    from edge_memory_amplitude_runtime_20261021 import ERuntime
    folder=Path(checkpoints)/read(Path(checkpoints)/'latest.json')['directory']
    # Strict dual-copy validation precedes restoration of the trial checkpoint.
    checked=ERuntime.from_export(folder/'model.json');del checked
    result=restore_identity_checkpoint(restore_checkpoint,checkpoints,device,identity)
    if result[-1]:raise ValueError('environment continuation forbidden')
    enable_runtime(result[0]);return result


def new_stage(seed,variant,device):
    import torch,numpy as np
    from closure_training_runtime import TrainingClosureROM
    from train_closure import make_optimizer
    if variant not in VARIANTS:raise ValueError('unregistered spatial group')
    source_metadata(seed)
    state=torch.load(source_folder(seed)/'training.pt',map_location='cpu',weights_only=False)
    from edge_memory_amplitude_runtime_20261021 import build_E
    model=build_E(TrainingClosureROM.from_export(source_folder(seed)/'model.json')).to(device).double()
    opt=make_optimizer(model)
    for g in opt.param_groups:g['lr']=RATES[g['block']]
    validate_optimizer(model,opt)
    if opt.state: raise ValueError('all E8 Adam states must start empty')
    torch.set_rng_state(state['torch_rng'])
    if device.type=='cuda':
        if state['cuda_rng'] is None: raise ValueError('source CUDA RNG unavailable')
        torch.cuda.set_rng_state(state['cuda_rng'],device)
    rng=np.random.default_rng();rng.bit_generator.state=state['numpy_rng']
    return model,opt,rng,copy.deepcopy(state['coverage'])

def validate_optimizer(model,opt):
    from edge_memory_amplitude_runtime_20261021 import coefficient_indices
    if set(coefficient_indices(model))!={'a_E'}:raise ValueError('only signed a_E parameter allowed')
    groups=[g for g in opt.param_groups if any(p is model.coefficients for p in g['params'])]
    if len(groups)!=1 or groups[0]['block']!='closure' or groups[0]['lr']!=RATES['closure'] or not model.coefficients.requires_grad:raise ValueError('b must use actual closure optimizer group and rate')
    active=[p for p in model.parameters() if p.requires_grad]
    assigned=[p for g in opt.param_groups for p in g['params']]
    if len({id(p) for p in assigned})!=len(assigned) or {id(p) for p in active}!={id(p) for p in assigned}:raise ValueError('all trainable parameters must occur exactly once')
    return True

def validate_request(seed,variant,target,resume=False,external=False,start=0,extension=None):
    if seed not in SEEDS or variant not in VARIANTS or target!=1024 or resume or external or start!=0 or extension is not None:raise ValueError('only new N/Q1024 without recovery')

def apply_lrs(opt,step):
    if type(step) is not int or not 1<=step<=1024:raise ValueError('invalid optimizer phase')
    for group in opt.param_groups:group['lr']=RATES[group['block']]*(.3 if step>512 else 1.)





def train(args,check):
    validate_request(args.seed,args.variant,args.target_step,args.resume,args.external_recovery,0,args.extension_lock)
    import torch
    from closure_training_runtime import TensorEventStore,synchronize
    from active_fom_pair_fit_20261020 import event_identity
    from linear_cache_cost_gate_20261019 import load_runtime
    from edge_memory_amplitude_runtime_20261021 import one_update
    from fixed_edge_pair_20261019 import save_safe,verify_identity
    from train_joint_numerical_pair_0100 import next_samples
    load_runtime();torch.set_num_threads(1);device=torch.device('cuda',torch.cuda.current_device())
    if args.target_step!=1024:raise ValueError('only1024 updates')
    source_metadata(args.seed)
    events,_=load_events();eq_values=equilibrium_delta(events);require_prepared_equilibrium(eq_values);schedule=checked_schedule(args.schedule_path)
    eq=torch.tensor(eq_values,device=device,dtype=torch.float64)
    verify_identity(event_identity(events),schedule['events']);rows=schedule['seeds'][str(args.seed)]
    admission=load_admission(args.admission)
    if admission.get('allow1024') is not True: raise ValueError('complete1024 cost admission required')
    expected=BASE/'models'/f'{args.seed}_{args.variant}'
    if args.output!=expected.resolve():raise ValueError('exact model output required')
    identity={'seed':args.seed,'variant':args.variant,'source_global_step':1280,'source_parent_stage_step':512,'source_O_stage_step':512,'optimizer_policy':'three fresh Adam;1280 -> B512 -> O512 -> E0',
        'source_model_sha256':SOURCE_HASHES[args.seed][0],'source_training_sha256':SOURCE_HASHES[args.seed][1],
        'schedule_sha256':sha(args.schedule_path),'data':schedule['events'],'rates':RATES,'force':FORCE,
        'source_commit':args.source_commit,'equilibrium_delta':eq_values}
    checkpoints=args.output/'checkpoints';trace=[];start=0
    if args.resume:
        if not args.recovery_authorization:raise ValueError('external recovery requires root authorization')
        auth=read(inside(args.recovery_authorization))
        if auth.get('external_interruption_verified') is not True or (auth.get('seed'),auth.get('variant'))!=(args.seed,args.variant):raise ValueError('root external recovery identity mismatch')
        model,opt,rng,coverage,meta,_=restore_stage(checkpoints,device,identity);start=meta['step'];trace=meta['trace']
        progress=read(args.recovery_progress) if args.external_recovery and args.recovery_progress else None
        if args.external_recovery and (progress is None or not 0<=progress.get('stage_step',-1)-start<=32):raise ValueError('external interruption replay exceeds32 or lacks prior progress')
    else:
        if checkpoints.exists():raise FileExistsError('duplicate model queue')
        model,opt,rng,coverage=new_stage(args.seed,args.variant,device)
        checkpoints.mkdir();verify_identity(rng.bit_generator.state,rows['initial_rng']);verify_identity(coverage,rows['initial_coverage'])
    validate_request(args.seed,args.variant,args.target_step,args.resume,args.external_recovery,start,args.extension_lock)
    if len(trace)!=start or any(t['stage_step']!=i+1 for i,t in enumerate(trace)):raise ValueError('stage trace identity mismatch')
    if not args.resume:save_checked(checkpoints,model,opt,rng,coverage,0,identity,trace)
    attempt=args.worker_output or args.output
    store=TensorEventStore(events,device)
    result={'status':'RUNNING','seed':args.seed,'variant':args.variant,'source_global_step':1280,'source_parent_stage_step':512,'source_O_stage_step':512,'stage_step':start,'target_step':args.target_step,'external_recovery':args.external_recovery,'clip_count':0}
    try:
        for step in range(start+1,args.target_step+1):
            check();row=rows['rows'][step-1]
            if row['stage_step']!=step or row['horizon_s']!=horizon_for(args.variant,step):raise ValueError('schedule position/phase mismatch')
            verify_identity(rng.bit_generator.state,row['numpy_rng_before']);samples,sampling=next_samples(events,rng,coverage)
            verify_identity(sampling,row['sampling']);verify_identity(rng.bit_generator.state,row['numpy_rng_after'])
            if [min(round(row['horizon_s']*100),len(e['x'])-11) for e,_ in samples]!=row['long_counts'] or [min(20,len(e['x'])-i-1) for e,i in samples]!=row['local_counts']:raise ValueError('actual sampled interval changed')
            apply_lrs(opt,step)
            t=one_update(model,opt,store,samples,row['horizon_s'],FORCE)
            result['clip_count']+=int(t['gradient_norm']>100.)
            t['clip_count']=result['clip_count']
            trace.append({'stage_step':step,'horizon_s':row['horizon_s'],'sampling':sampling,'learning_rates':{g['block']:g['lr'] for g in opt.param_groups},**t});result['stage_step']=step
            if step%32==0:save_checked(checkpoints,model,opt,rng,coverage,step,identity,trace)
            write(attempt/'progress.json',result)
        synchronize(device);result['status']='COMPLETED';return result
    except BaseException as exc:result.update(status='UNRESOLVED',error=repr(exc));raise
    finally:write(attempt/'training_result.json',result)

def cost_admission(update_seconds,save_seconds,load_seconds,evaluation_seconds):
    if set(update_seconds)!=set(VARIANTS):raise ValueError('all actual N/Q costs required')
    per={}
    for v in VARIANTS:
        horizons=(1,2,5)
        ts=[float(update_seconds[v][str(h)]) for h in horizons]
        if any(not math.isfinite(t) or t<=0 for t in (*ts,save_seconds,load_seconds,evaluation_seconds)):raise ValueError('finite positive actual timings required')
        per[v]=1.25*(342*ts[0]+341*ts[1]+341*ts[2]+33*save_seconds+load_seconds+20.)
    total=3*sum(per.values());ev=1.25*evaluation_seconds
    return {'allow1024':total<=7320 and max(per.values())<=2440 and ev<=180,
       'allow768':False,'per_model_seconds':per,'three_models_seconds':total,'per_evaluation_seconds':ev,'three_evaluations_seconds':3*ev,'trial_updates_per_variant':12,'supervision_reserve_before_margin_per_model_seconds':20.,'extra_export_seconds':0.,'save_includes_export':True}



def load_admission(path):
    a=read(path)
    if type(a.get('allow1024')) is not bool or a.get('allow768') is not False or set(a.get('per_model_seconds',{}))!=set(VARIANTS) or a.get('trial_updates_per_variant')!=12:raise ValueError('actual E8 flat cost admission document required')
    return a



def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('cost','train','evaluate'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-output',type=Path)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--source-commit',required=True)
    for n in ('binding','schedule-path','admission','extension-lock','model'):p.add_argument('--'+n,type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--variant');p.add_argument('--target-step',type=int,default=1024)
    p.add_argument('--recovery-progress',type=Path);p.add_argument('--recovery-authorization',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--external-recovery',action='store_true');p.add_argument('--previous-seconds',type=float,default=0.)
    a=p.parse_args(argv)
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    inside(a.output)
    worker=inside(a.worker_output) if a.worker_output else a.output
    worker.mkdir(parents=True,exist_ok=False)
    from experiments.closure_p3_disagreement_20261020 import own_creation_filetime_utc
    (worker/'worker.json').write_text(json.dumps({'pid':os.getpid(),'creation_filetime_utc':own_creation_filetime_utc(),'executable':str(Path(sys.executable).resolve()),'argv':sys.argv,'cwd':str(Path.cwd())}),encoding='utf-8')
    if worker!=a.output:a.output.mkdir(parents=True,exist_ok=a.resume)
    started=time.perf_counter()
    def check():
        if time.perf_counter()-started>=a.max_seconds:raise TimeoutError('registered category deadline')
    try:
        if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('archived source commit required')
        cap={'cost':440,'train':2440,'evaluate':180}[a.mode]
        if not math.isfinite(a.max_seconds) or not math.isfinite(a.previous_seconds) or not 0<a.max_seconds<=cap-a.previous_seconds or a.previous_seconds<0:raise ValueError('cumulative mode/model cap exceeded')
        binding(a.binding);baseline_contract()
        for name in ('schedule_path','admission','extension_lock','recovery_progress','recovery_authorization'):
            value=getattr(a,name)
            if value is not None:
                if name=='schedule_path':checked_schedule(value)
                else:inside(value)
        if a.mode in ('prepare','cost') and a.output.is_relative_to((BASE/'models').resolve()):raise ValueError('QA/preparation cannot write scientific models')
        if a.resume or a.external_recovery or a.extension_lock is not None:raise ValueError('no recovery or extension')
        if a.resume and a.worker_output is None:raise ValueError('continuation needs a new worker-output directory')
        sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
        if a.mode=='train':result=train(a,check)
        elif a.mode=='cost':result=measure_cost(a,check)
        else:result=evaluate(a,check)
        binding(a.binding);write(worker/'complete.json',{'status':'COMPLETED','result':result,'elapsed_seconds':time.perf_counter()-started})
        return 0
    except BaseException as exc:
        write(worker/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started});raise

def evaluate(args,check):
    import torch
    from closure_training_runtime import TrainingClosureROM
    entry=check_endpoint(args.model);seed=entry['seed'];variant=entry['variant']
    frozen=read(BASE/'frozen_evaluation_endpoints.json')
    entries=frozen.get('endpoints',[])
    if len(entries)!=3 or {(e['seed'],e['variant']) for e in entries}!={(s,v) for s in SEEDS for v in VARIANTS}:raise ValueError('three E8 endpoint freeze required')
    for e in entries:
        actual=check_endpoint(e['model_path'])
        if any(actual[k]!=e[k] for k in ('seed','variant','model_path','model_sha256','training_sha256','checkpoint_sha256')):raise ValueError('frozen endpoint changed')
    if args.output!=(BASE/'evaluation'/f'{seed}_{variant}'/'step001024').resolve():raise ValueError('exact role evaluation path required')
    torch.set_num_threads(1);events,excluded=load_events();equilibrium_delta(events)
    from edge_memory_amplitude_runtime_20261021 import ERuntime
    model=ERuntime.from_export(Path(args.model)).cpu().double().eval()
    result=evaluate_queue(model,events,args.output/'queue',{'seed':seed,'variant':variant,'stage_step':1024,'model_sha256':entry['model_sha256'],'excluded':excluded},check)
    return {'event_count':result['event_count'],'evidence':str(args.output/'queue/evidence.json'),'evidence_sha256':sha(args.output/'queue/evidence.json'),'primary':result['primary']}



def cost_panels(events):
    ordered=sorted(events,key=lambda e:(-(len(e['x'])-11),e['id']))
    if len(ordered)<16:raise ValueError('sixteen actual long records required')
    long=ordered[:12]
    unstable=sorted((e for e in events if e['label']=='unstable'),key=lambda e:(len(e['x']),e['id']))
    if len(unstable)<12:raise ValueError('twelve short unstable events required')
    return {'long':[(e,10) for e in long],'short_unstable':[(e,10) for e in unstable[:12]]},ordered[:16]

def generic_source(seed,device):
    from symbolic_closure_rom import SymbolicClosureROM
    source_metadata(seed)
    return SymbolicClosureROM.from_export(source_folder(seed)/'model.json').to(device).double().eval()

def source_equivalence(device,check):
    import torch
    from symbolic_closure_rom import term_key
    from edge_memory_amplitude_runtime_20261021 import coefficient_indices
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    records=[]
    try:
        baseline_rng_contract(device)
        gen=torch.Generator(device='cpu');gen.manual_seed(20261020)
        h=(.02*torch.randn((3,11,30),generator=gen,dtype=torch.float64)).to(device)
        z=(.02*torch.randn((3,40),generator=gen,dtype=torch.float64)).to(device)
        for seed in SEEDS:
            source=generic_source(seed,device)
            for variant in VARIANTS:
                check();t=time.perf_counter();model,opt,source_rng,source_coverage=new_stage(seed,variant,device)
                from fixed_edge_pair_20261019 import verify_identity
                initial=checked_schedule()['seeds'][str(seed)]
                verify_identity(source_rng.bit_generator.state,initial['initial_rng']);verify_identity(source_coverage,initial['initial_coverage'])
                if device.type=='cuda':torch.cuda.synchronize(device)
                load_seconds=time.perf_counter()-t
                a,b=source.state_dict(),model.state_dict()
                for key in a:
                    if key.endswith('coefficients'):continue
                    if key not in b or not torch.equal(a[key],b[key]):raise ValueError('shared source state differs:'+key)
                left={term_key(t):c for t,c in zip(source.terms,source.coefficients)}
                right={term_key(t):c for t,c in zip(model.terms,model.coefficients)}
                for key,c in left.items():
                    if key not in right or not torch.equal(c,right[key]):raise ValueError('shared source coefficient differs')
                ix=coefficient_indices(model)['a_E']
                if any(model.coefficients[i].item()!=0. for i in coefficient_indices(model).values()):raise ValueError('E a_E must start exact0')
                flags=lambda m:{n:p.requires_grad for n,p in m.named_parameters()}
                if flags(source)!=flags(model) or opt.state:raise ValueError('source flags/new Adam differs')
                with torch.no_grad():
                    torch.testing.assert_close(source.encode(h),model.encode(h),rtol=0,atol=0)
                    torch.testing.assert_close(source.field(z),model.field(z),rtol=2e-11,atol=1e-11)
                records.append({'seed':seed,'variant':variant,'updates':0,'source_load_seconds':load_seconds,'shared_terms':len(left),'edge_memory_zero':model.coefficients[coefficient_indices(model)['a_E']].item()==0.})
        return {'comparisons':records,'count':len(records),'updates':0}
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state(cuda,device)



def cost_cases(variant):
    if variant not in VARIANTS:raise ValueError('unregistered cost group')
    hs=(1,2,5)
    first=(1,'long');second=(hs[-1],'long')
    return [first,second]+[(h,panel) for h in hs for panel in ('long','short_unstable') if (h,panel) not in (first,second)]

def _measure_cost(args,check):
    import torch,numpy as np
    from closure_training_runtime import TensorEventStore,synchronize
    from linear_cache_cost_gate_20261019 import load_runtime,_equal
    from edge_memory_amplitude_runtime_20261021 import one_update
    from fixed_edge_pair_20261019 import save_safe
    load_runtime();torch.set_num_threads(1);device=torch.device('cuda',torch.cuda.current_device())
    events,_=load_events();eq_values=equilibrium_delta(events);require_prepared_equilibrium(eq_values);eq=torch.tensor(eq_values,device=device,dtype=torch.float64);panels,cpu16=cost_panels(events);store=TensorEventStore(events,device)
    prepared=read(PREPARE)
    if prepared['panel_ids']!={name:[e['id'] for e,_ in panel] for name,panel in panels.items()} or prepared['cpu16_ids']!=[e['id'] for e in cpu16]:raise ValueError('fixed cost panels changed')
    initial_equivalence=source_equivalence(device,check)
    write(args.output/'source_equivalence.json',initial_equivalence)
    timings={};saves=[];loads=[r['source_load_seconds'] for r in initial_equivalence['comparisons']];details={};evaluation_estimate=None
    for variant in VARIANTS:
        check();torch.cuda.reset_peak_memory_stats(device);begin=time.perf_counter()
        model,opt,rng,coverage=new_stage(6101,variant,device);synchronize(device);loads.append(time.perf_counter()-begin)
        hs=(1,2,5)
        cases=cost_cases(variant)
        records=[];trace=[]
        identity={'qa_only':True,'seed':6101,'variant':variant,'source_global_step':1280,'source_parent_stage_step':512}
        folder=args.output/f'qa_checkpoint_{variant}';folder.mkdir()
        def update(case,rep):
            h,panel=case;check();synchronize(device);t=time.perf_counter()
            row=one_update(model,opt,store,panels[panel],float(h),FORCE)
            synchronize(device);seconds=time.perf_counter()-t
            records.append({'horizon_s':h,'panel':panel,'rep':rep,'seconds':seconds,'loss':row['loss'],'gradient_norm':row['gradient_norm'],'edge_memory':row['edge_memory']})
            trace.append({'stage_step':len(trace)+1,'horizon_s':h,**row})
        # Exactly12 actual updates. First update -> nonempty-Adam checkpoint;
        # second case rep1 is reference, rep2 is the restored matching update.
        update(cases[0],1)
        saved_cpu=torch.get_rng_state().clone();saved_cuda=torch.cuda.get_rng_state(device).clone();saved_numpy=copy.deepcopy(rng.bit_generator.state);saved_coverage=copy.deepcopy(coverage)
        if not opt.state:raise ValueError('QA checkpoint must contain nonempty Adam')
        t=time.perf_counter();save_checked(folder,model,opt,rng,coverage,1,identity,trace);synchronize(device);saves.append(time.perf_counter()-t)
        update(cases[1],1)
        reference=copy.deepcopy(model.state_dict());reference_opt=copy.deepcopy(opt.state_dict())
        t=time.perf_counter();model,opt,rng,coverage,metadata,_=restore_stage(folder,device,identity);synchronize(device);loads.append(time.perf_counter()-t)
        if not torch.equal(saved_cpu,torch.get_rng_state()) or not torch.equal(saved_cuda,torch.cuda.get_rng_state(device)):raise ValueError('CPU/CUDA RNG restore differs')
        _equal(saved_numpy,rng.bit_generator.state,exact=True);_equal(saved_coverage,coverage,exact=True)
        if metadata['step']!=1:raise ValueError('QA nonempty-Adam source changed')
        update(cases[1],2)
        _equal(reference,model.state_dict(),exact=False);_equal(reference_opt,opt.state_dict(),exact=False)
        update(cases[0],2)
        for case in cases[2:]:
            for rep in (1,2):update(case,rep)
        if len(records)!=12:raise ValueError('trial-update budget changed')
        t=time.perf_counter();save_checked(folder,model,opt,rng,coverage,12,identity,trace);synchronize(device);saves.append(time.perf_counter()-t)
        timings[variant]={str(h):max(x['seconds'] for x in records if x['horizon_s']==h) for h in hs}
        details[variant]={'actual_updates':12,'records':records,'restored_next_update_equal':True,'peak_cuda_bytes':torch.cuda.max_memory_allocated(device),'panel_event_ids':{k:[e['id'] for e,_ in s] for k,s in panels.items()},'maximum_actual_future_points':max(len(e['x'])-11 for e in events)}
        if variant=='E':
            check();t=time.perf_counter();evaluate_queue(model,cpu16,args.output/'qa_cpu16',{'scope':'discarded trial cost only'},check)
            elapsed=time.perf_counter()-t;evaluation_estimate=25*elapsed
            details['cpu16']={'actual_seconds':elapsed,'per385_before_margin':evaluation_estimate,'events':16,'includes':'dual-step15s scoring compression export39D'}
    admission=cost_admission(timings,max(saves),max(loads),evaluation_estimate)
    write(args.output/'admission.json',admission)
    result={'source_equivalence':initial_equivalence,'timings':details,'admission':admission,'force':FORCE,'scope':'discarded trials; no scientific endpoint'}
    write(args.output/'cost.json',result);return result


def measure_cost(args,check):
    import torch,numpy as np
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();numpy_state=np.random.get_state()
    try:return _measure_cost(args,check)
    finally:torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda);np.random.set_state(numpy_state)






def cohort_sets(events):
    path=ROOT/'results/quadratic_memory_edge_20261020/cohort_identity.json'
    if sha(path)!='0d2b3ed030cd742ae0809114008c31e3a99c35c56d0238d2d03c5473abcecbf5':raise ValueError('frozen cohort identity changed')
    groups=read(path)['cohorts'];ids={e['id'] for e in events}
    fixed={k:{e['id'] for e in groups[k]} for k in ('original366','new19')}
    if len(fixed['original366'])!=366 or len(fixed['new19'])!=19 or fixed['original366']&fixed['new19'] or not ids<=set.union(*fixed.values()):raise ValueError('fixed source cohort mismatch')
    return {k:s&ids for k,s in fixed.items()}

def actual15_qualified(real15,pair15):
    if any(real15.get(k) is not True for k in ('available','finite','numerically_admissible')):return False
    if any(pair15.get(k) is not True for k in ('finite_to_endpoint','los_pair_pass')):return False
    for row in (real15,pair15):
        for key in ('coi_normalized_rmse','common_normalized_rmse'):
            value=row.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value>.001:return False
    return True

def spatial_metrics(model,event,fine,qualified15):
    import numpy as np,torch
    from spatial_extrema_runtime_20261020 import extrema_error
    n=min(1500,len(event['x'])-11)
    row={'actual_future_points':n,'finite':False,'numerically_admissible':bool(qualified15),'combined':None,'angle_normalized_squared_error':None,'speed_normalized_squared_error':None}
    pred=fine[::2][1:n+1,:30];truth=event['x'][11:11+n,:30]
    if n>0 and np.isfinite(pred).all():
        eq=torch.tensor(equilibrium_delta([event]),dtype=torch.float64)
        with torch.no_grad():values=extrema_error(torch.as_tensor(pred),torch.as_tensor(truth),model.mass,eq)
        means=[float(x.mean()) for x in values]
        if all(math.isfinite(v) for v in means):
            row['finite']=True
            if row['numerically_admissible']:row.update(zip(('combined','angle_normalized_squared_error','speed_normalized_squared_error'),means))
    return row



def evaluate_queue(model, events, output, provenance, check, *, export_analysis=True):
    import numpy as np
    import torch
    from evaluate_structured_rom import _rollouts, _pair_row, _safe
    from summarize_train_local_memory_0098 import event_metrics, build_summary
    from fixed_edge_pair_20261019 import scientific_qualification, primary_risk, physical_metrics
    from active_fom_pair_fit_20261020 import ordered,actual_interval_predictions,interval15
    torch.set_num_threads(1)
    model = copy.deepcopy(model).cpu().double().eval(); enable_runtime(model)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    events = ordered(events); mass = model.mass.detach().numpy(); ps = model.p_scale.detach().numpy()
    records = {k: [] for k in ('records','common_frequency','normalized_p','objective_point_error','numerical')}
    extras, qualifications, prediction_files = [], {}, []
    # Small fixed batches; an inference exception is retained per event, never dropped.
    for first in range(0, len(events), 16):
        check(); block = events[first:first+16]
        histories = np.stack([e['x'][:11] for e in block])
        error = None
        try:
            fine = _rollouts(model, histories, 3000, .005)
            coarse = _rollouts(model, histories, 1500, .01)
        except Exception as exc:
            error = repr(exc); fine = np.full((len(block),3001,40),np.nan); coarse = np.full((len(block),1501,40),np.nan)
        prediction_path = output/f'predictions_{first:04d}.npz'
        np.savez_compressed(prediction_path, event_ids=np.asarray([e['id'] for e in block]),fine=fine,coarse=coarse)
        prediction_files.append({'path':str(prediction_path.resolve()),'sha256':sha(prediction_path)})
        for e,h,f,c in zip(block,histories,fine,coarse):
            metrics = event_metrics(model,e,f[:1601],c[:801])
            for k,v in metrics.items():
                records[k].extend(v)
            pair8 = _pair_row(f[:1601],c[:801],h,e,e['classification_meta'],mass)
            actual_f,actual_c,actual_n=actual_interval_predictions(e,f,c)
            pair15 = _pair_row(actual_f,actual_c,h,e,e['classification_meta'],mass)
            pair15.update(scope='actual_interval_up_to_15s',observed_future_steps=actual_n,
                          observed_future_seconds=actual_n*.01)
            real15 = interval15(model,e,f,c)
            qualified8 = scientific_qualification(pair8,metrics['numerical'])
            qualifications[e['id']] = qualified8
            spatial=spatial_metrics(model,e,f,actual15_qualified(real15,pair15))
            algebra=algebraic_evidence(model,e,f,qualified8,actual15_qualified(real15,pair15))
            extras.append({'algebraic':algebra,'spatial':spatial,'event_id':e['id'],'group':e['group'],'label':e['label'],
                'exception':error,'classification_8s':pair8,'classification_15s':pair15,
                'actual_interval15':real15,'scientific_numeric_8s':qualified8,
                'physical_1_2_5_8s':physical_metrics(f[::2],e,mass,ps)})
        write(output/'partial.json',_safe({'records':records,'extras':extras}))
    result = {'provenance':provenance,'event_count':len(events),'rows':records,'extras':extras,
              'summary':build_summary(records),'primary':primary_risk(records['objective_point_error'],qualifications),
              'test_read':False,'model_scope':'research_candidate','prediction_files':prediction_files,
              'prediction_origin':'endpoint_autonomous_inference'}
    if export_analysis:
        from train_closure import export_cpu
        from analyze_symbolic_closure import analyze_model
        export_cpu(model,output/'model.json')
        local, arrays = analyze_model(model,source_sha256=sha(output/'model.json'),source_path=str(output/'model.json'))
        np.savez_compressed(output/'local.npz',**arrays); write(output/'local.json',_safe(local))
    result['cohorts']={}
    for name,ids in cohort_sets(events).items():
        cohort_records={k:[r for r in rows if r['event_id'] in ids] for k,rows in records.items()}
        result['cohorts'][name]={'event_count':len(ids),'summary':build_summary(cohort_records),'primary':primary_risk(cohort_records['objective_point_error'],qualifications),'extras':[r for r in extras if r['event_id'] in ids]}
    write(output/'evidence.json',_safe(result))
    write(output/'complete.json',{'status':'COMPLETED','event_count':len(events),'evidence_sha256':sha(output/'evidence.json')})
    return result

if __name__=='__main__': raise SystemExit(main())
