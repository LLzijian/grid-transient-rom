"""Train-only O/F/T temporal-objective study. Every scientific invocation is native-supervised."""
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
BASE=ROOT/'private_data/closure_mechanism_20261019/P2/temporal_objective01'
FRESH=ROOT/'private_data/closure_mechanism_20261019/P3/fresh_design_pair01'
SEEDS=(6101,6102,6103)
VARIANTS=('O','F','T')
RATES={'skeleton':.0009,'closure':.00009,'encoder':.000027}
FORCE=1.0068379869405584
SOURCE_HASHES={6101: ('8bea3ed4ccc864e22e7f2f886d7ae37f0df30323532f4ac4c45580ee865bcfbb', 'f19b71f0f004d85b59899e78e95c2ec7ddb82ecc07d0ef2f5f0360af43cd05f8', 'ea76de5aa33787998aba983bcd6cc3a46a7b1c06af887e9628a166ad277d965e'), 6102: ('86220d87da064d358c1529684db76d92ebea3c26e371aef3b689475b3717575e', '607610a32cec7457511bceac6fc20858d2c7eef285d4cf014fe59bd107979ed7', '61a99ff8837f1b1078410126388eed1bf9bdacaf9e28b27e36694caa6806f7f3'), 6103: ('9434df6ca6b04f1bd066670284c651d982af71bf3860a2bbf79c7f393a3ad225', '512355eca050f55e6872682daf4fb11584d29c373a3d6cda660b4ae46b226203', '2d383b7042fcef5899a5de121aaac34d284ee99f5fddddbd4783c78d8554fd22')}

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
    if seed not in SEEDS: raise ValueError('unregistered source seed')
    return ROOT/'private_data/closure_mechanism_20261019/P2/quadratic_memory_edge01/models'/f'{seed}_B/checkpoints/step000512'


def source_metadata(seed):
    p=source_folder(seed);m=read(p/'checkpoint.json')
    for n,h in zip(('model.json','training.pt','checkpoint.json'),SOURCE_HASHES[seed]):
        if sha(p/n)!=h:raise ValueError('source bytes changed')
    if m['step']!=512 or m['identity']['variant']!='B' or m['identity']['seed']!=seed or m['identity']['source_global_step']!=1280:raise ValueError('B512 source identity changed')
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

def horizon_for(variant,step):
    if variant not in VARIANTS or type(step) is not int or not 1<=step<=512:raise ValueError('unregistered phase')
    return (1.,2.,5. if variant=='O' else 15.)[(step-1)%3]

def schedule_rows(events,rng,coverage,updates=512):
    from train_joint_numerical_pair_0100 import next_samples
    if updates!=512:raise ValueError('only512 common rows permitted')
    rows=[]
    for step in range(1,513):
        before=copy.deepcopy(rng.bit_generator.state);samples,sampling=next_samples(events,rng,coverage)
        if len(samples)!=12:raise ValueError('registered10+2 batch changed')
        hs={v:horizon_for(v,step) for v in VARIANTS}
        rows.append({'stage_step':step,'horizons':hs,'sampling':sampling,'numpy_rng_before':before,
          'numpy_rng_after':copy.deepcopy(rng.bit_generator.state),
          'long_counts':{v:[min(round(h*100),len(e['x'])-11) for e,_ in samples] for v,h in hs.items()},
          'local_counts':[min(20,len(e['x'])-i-1) for e,i in samples]})
    if {x for row in rows for x in row['sampling']['sample_ids']}!={e['id'] for e in events}:raise ValueError('common schedule lacks full event coverage')
    return rows


def prepare(output):
    import numpy as np
    import torch
    from experiments.active_fom_pair_fit_20261020 import event_identity
    events,excluded=load_events();identity=event_identity(events);result={'events':identity,'seeds':{},'force':FORCE,'excluded':excluded}
    for seed in SEEDS:
        meta=source_metadata(seed)
        if meta['identity']['data']!=identity: raise ValueError('source sorted385 data identity differs')
        state=torch.load(source_folder(seed)/'training.pt',map_location='cpu',weights_only=False)
        if state['step']!=512 or state['cuda_rng'] is None: raise ValueError('complete source RNG missing')
        rng=np.random.default_rng();rng.bit_generator.state=state['numpy_rng'];coverage=copy.deepcopy(state['coverage'])
        if sorted(coverage['cover_order'])!=list(range(385)) or type(coverage['cursor']) is not int: raise ValueError('source coverage identity differs')
        initial_rng=copy.deepcopy(rng.bit_generator.state);initial_coverage=copy.deepcopy(coverage)
        rows=schedule_rows(events,rng,coverage)
        result['seeds'][str(seed)]={'rows':rows,'initial_rng':initial_rng,'initial_coverage':initial_coverage,
            'source_model_sha256':meta['model_sha256'],'source_training_sha256':meta['training_sha256'],
            'source_metadata_sha256':sha(source_folder(seed)/'checkpoint.json')}
    write(output/'schedule.json',result)
    return {'status':'COMPLETED','schedule_sha256':sha(output/'schedule.json'),'events':385,'rows_per_seed':512}

def enable_runtime(model):
    from linear_cache_runtime_20261019 import enable_linear_cache,_DELETE7_KEYS
    from symbolic_closure_rom import term_key
    if tuple(term_key(t) for t in model.terms)!=_DELETE7_KEYS:raise ValueError('fixed delete7 terms required')
    enable_linear_cache(model);return model


def restore_stage(checkpoints,device,identity):
    from train_closure import restore_checkpoint
    from closure_checkpoint_identity_20261019 import restore_identity_checkpoint
    result=restore_identity_checkpoint(restore_checkpoint,checkpoints,device,identity)
    if result[-1]: raise ValueError('environment continuation forbidden')
    enable_runtime(result[0]);return result
def new_stage(seed,variant,device):
    import torch,numpy as np
    from closure_training_runtime import TrainingClosureROM
    from train_closure import make_optimizer
    if variant not in VARIANTS:raise ValueError('unregistered temporal group')
    source_metadata(seed)
    state=torch.load(source_folder(seed)/'training.pt',map_location='cpu',weights_only=False)
    model=enable_runtime(TrainingClosureROM.from_export(source_folder(seed)/'model.json')).to(device).double()
    opt=make_optimizer(model)
    for g in opt.param_groups:g['lr']=RATES[g['block']]
    if opt.state: raise ValueError('all nine Adam states must start empty')
    torch.set_rng_state(state['torch_rng'])
    if device.type=='cuda':
        if state['cuda_rng'] is None: raise ValueError('source CUDA RNG unavailable')
        torch.cuda.set_rng_state(state['cuda_rng'],device)
    rng=np.random.default_rng();rng.bit_generator.state=state['numpy_rng']
    return model,opt,rng,copy.deepcopy(state['coverage'])

def validate_request(seed,variant,target,resume=False,external=False,start=0,extension=None):
    if seed not in SEEDS or variant not in VARIANTS or target!=512 or extension is not None:raise ValueError('unregistered science identity/target')
    if type(start) is not int or start<0 or start%32 or start>=target:raise ValueError('unsafe or completed resume boundary')
    if resume!=external:raise ValueError('resume only for explicitly authorized external interruption')
    if not resume and start:raise ValueError('fresh stage starts0')



def train(args,check):
    if args.seed not in SEEDS or args.variant not in VARIANTS or args.target_step!=512:raise ValueError('unregistered model identity')
    import torch
    from closure_training_runtime import TensorEventStore,synchronize
    from active_fom_pair_fit_20261020 import event_identity
    from linear_cache_cost_gate_20261019 import load_runtime
    from temporal_objective_runtime_20261020 import one_update
    from fixed_edge_pair_20261019 import save_safe,verify_identity
    from train_joint_numerical_pair_0100 import next_samples
    load_runtime();torch.set_num_threads(1);device=torch.device('cuda',torch.cuda.current_device())
    if args.target_step!=512:raise ValueError('only512 temporal updates')
    source_metadata(args.seed)
    events,_=load_events();schedule=read(args.schedule_path)
    verify_identity(event_identity(events),schedule['events']);rows=schedule['seeds'][str(args.seed)]
    admission=load_admission(args.admission)
    if admission.get('allow512') is not True: raise ValueError('complete512 cost admission required')
    expected=BASE/'models'/f'{args.seed}_{args.variant}'
    if args.output!=expected.resolve():raise ValueError('exact model output required')
    identity={'seed':args.seed,'variant':args.variant,'source_global_step':1280,'source_parent_stage_step':512,'optimizer_policy':'all-nine fresh Adam at new stage0; full8 global1280 -> B512',
        'source_model_sha256':SOURCE_HASHES[args.seed][0],'source_training_sha256':SOURCE_HASHES[args.seed][1],
        'schedule_sha256':sha(args.schedule_path),'data':schedule['events'],'rates':RATES,'force':FORCE,
        'source_commit':args.source_commit}
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
    if not args.resume:save_safe(checkpoints,model,opt,rng,coverage,0,identity,trace)
    attempt=args.worker_output or args.output
    store=TensorEventStore(events,device)
    result={'status':'RUNNING','seed':args.seed,'variant':args.variant,'source_global_step':1280,'source_parent_stage_step':512,'stage_step':start,'target_step':args.target_step,'external_recovery':args.external_recovery}
    try:
        for step in range(start+1,args.target_step+1):
            check();row=rows['rows'][step-1]
            if row['stage_step']!=step or row['horizons']!={v:horizon_for(v,step) for v in VARIANTS}:raise ValueError('schedule position/phase mismatch')
            verify_identity(rng.bit_generator.state,row['numpy_rng_before']);samples,sampling=next_samples(events,rng,coverage)
            verify_identity(sampling,row['sampling']);verify_identity(rng.bit_generator.state,row['numpy_rng_after'])
            if [min(round(row['horizons'][args.variant]*100),len(e['x'])-11) for e,_ in samples]!=row['long_counts'][args.variant] or [min(20,len(e['x'])-i-1) for e,i in samples]!=row['local_counts']:raise ValueError('actual sampled interval changed')
            t=one_update(model,opt,store,samples,row['horizons'][args.variant],FORCE,tail_weighted=args.variant=='T' and step%3==0)
            trace.append({'stage_step':step,'horizon_s':row['horizons'][args.variant],'sampling':sampling,'tail_weighted':args.variant=='T' and step%3==0,**t});result['stage_step']=step
            if step%32==0:save_safe(checkpoints,model,opt,rng,coverage,step,identity,trace)
            write(attempt/'progress.json',result)
        synchronize(device);result['status']='COMPLETED';return result
    except BaseException as exc:result.update(status='UNRESOLVED',error=repr(exc));raise
    finally:write(attempt/'training_result.json',result)

def cost_admission(update_seconds,save_seconds,load_seconds,evaluation_seconds):
    if set(update_seconds)!=set(VARIANTS):raise ValueError('all actual O/F/T costs required')
    per={}
    for v in VARIANTS:
        horizons=(1,2,5) if v=='O' else (1,2,15)
        ts=[float(update_seconds[v][str(h)]) for h in horizons]
        if any(not math.isfinite(t) or t<=0 for t in (*ts,save_seconds,load_seconds,evaluation_seconds)):raise ValueError('finite positive actual timings required')
        per[v]=1.25*(171*ts[0]+171*ts[1]+170*ts[2]+17*save_seconds+load_seconds)
    total=3*sum(per.values());ev=1.25*evaluation_seconds
    return {'allow512':total<=28800 and per['O']<=2000 and per['F']<=4000 and per['T']<=4000 and ev<=400 and 9*ev<=3600,
       'allow768':False,'per_model_seconds':per,'nine_models_seconds':total,'per_evaluation_seconds':ev,'nine_evaluations_seconds':9*ev,'trial_updates_per_variant':12}



def load_admission(path):
    a=read(path)
    if type(a.get('allow512')) is not bool or a.get('allow768') is not False or set(a.get('per_model_seconds',{}))!=set(VARIANTS) or a.get('trial_updates_per_variant')!=12:raise ValueError('actual O/F/T flat cost admission document required')
    return a



def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','cost','train','evaluate'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-output',type=Path)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--source-commit',required=True)
    for n in ('binding','schedule-path','admission','extension-lock','model'):p.add_argument('--'+n,type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--variant');p.add_argument('--target-step',type=int,default=512)
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
        cap={'prepare':180,'cost':2400,'train':2000 if a.variant=='O' else 4000,'evaluate':400}[a.mode]
        if not math.isfinite(a.max_seconds) or not math.isfinite(a.previous_seconds) or not 0<a.max_seconds<=cap-a.previous_seconds or a.previous_seconds<0:raise ValueError('cumulative mode/model cap exceeded')
        binding(a.binding)
        for name in ('schedule_path','admission','extension_lock','recovery_progress','recovery_authorization'):
            value=getattr(a,name)
            if value is not None:inside(value)
        if a.mode in ('prepare','cost') and a.output.is_relative_to((BASE/'models').resolve()):raise ValueError('QA/preparation cannot write scientific models')
        if a.resume and a.worker_output is None:raise ValueError('continuation needs a new worker-output directory')
        sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
        if a.mode=='prepare':result=prepare(a.output)
        elif a.mode=='train':result=train(a,check)
        elif a.mode=='cost':result=measure_cost(a,check)
        else:result=evaluate(a,check)
        binding(a.binding);write(worker/'complete.json',{'status':'COMPLETED','result':result,'elapsed_seconds':time.perf_counter()-started})
        return 0
    except BaseException as exc:
        write(worker/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started});raise

def evaluate(args,check):
    import torch
    from closure_training_runtime import TrainingClosureROM
    p=Path(args.model).resolve()
    legal={(BASE/'models'/f'{s}_{v}'/'checkpoints'/f'step{n:06d}'/'model.json').resolve():(s,v,n) for s in SEEDS for v in VARIANTS for n in (512,)}
    if p not in legal:raise ValueError('only exact registered science endpoint may be evaluated')
    seed,variant,step=legal[p];meta=read(p.parent/'checkpoint.json')
    if (meta['identity']['seed'],meta['identity']['variant'],meta['step'])!=(seed,variant,step):raise ValueError('endpoint stage identity differs')
    if sha(p)!=meta['model_sha256'] or sha(p.parent/'training.pt')!=meta['training_sha256'] or len(meta['trace'])!=step:raise ValueError('endpoint identity incomplete')
    if args.output!=(BASE/'evaluation'/f'{seed}_{variant}'/f'step{step:06d}').resolve():raise ValueError('exclusive endpoint evaluation path required')
    torch.set_num_threads(1);events,excluded=load_events()
    model=enable_runtime(TrainingClosureROM.from_export(p)).cpu().double().eval()
    result=evaluate_queue(model,events,args.output/'queue',{'seed':seed,'variant':variant,'stage_step':step,'model_sha256':sha(p),'excluded':excluded},check)
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
    """Nine zero-update source comparisons; restore enclosing CPU/CUDA RNG."""
    import torch
    cpu_before=torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    records=[]
    try:
        # A private generator consumes none of the production source RNG stream.
        generator=torch.Generator(device='cpu');generator.manual_seed(20261020)
        histories=(.02*torch.randn((3,11,30),generator=generator,dtype=torch.float64)).to(device)
        states=(.02*torch.randn((3,40),generator=generator,dtype=torch.float64)).to(device)
        for seed in SEEDS:
            check();source=generic_source(seed,device)
            for variant in VARIANTS:
                check();model,opt,_,_=new_stage(seed,variant,device);model.eval()
                if opt.state:raise ValueError('source-equivalence object Adam is not empty')
                left=source.state_dict();right=model.state_dict()
                if left.keys()!=right.keys():raise ValueError('source state key mismatch')
                for key in left:
                    if not torch.equal(left[key],right[key]):raise ValueError('source state differs: '+key)
                flags=lambda m:{n:p.requires_grad for n,p in m.named_parameters()}
                if flags(source)!=flags(model):raise ValueError('source parameter flags differ')
                with torch.no_grad():
                    old_encoder=source.encode(histories);new_encoder=model.encode(histories)
                    old_field=source.field(states);new_field=model.field(states)
                    torch.testing.assert_close(old_encoder,new_encoder,rtol=2e-10,atol=2e-10)
                    torch.testing.assert_close(old_field,new_field,rtol=2e-10,atol=2e-10)
                records.append({'seed':seed,'variant':variant,'updates':0,'state_entries':len(left),'parameter_flags_equal':True,
                    'encoder_max_abs_difference':float((old_encoder-new_encoder).abs().max()),
                    'field_max_abs_difference':float((old_field-new_field).abs().max()),'history_shape':[3,11,30],'state_shape':[3,40]})
                del model,opt
            del source
        return {'comparisons':records,'count':len(records),'all_three_sources':True,'updates':0}
    finally:
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:torch.cuda.set_rng_state(cuda_before,device)

def cost_cases(variant):
    if variant not in VARIANTS:raise ValueError('unregistered cost group')
    hs=(1,2,5) if variant=='O' else (1,2,15)
    first=(1,'long');second=(hs[-1],'long')
    return [first,second]+[(h,panel) for h in hs for panel in ('long','short_unstable') if (h,panel) not in (first,second)]

def measure_cost(args,check):
    import torch,numpy as np
    from closure_training_runtime import TensorEventStore,synchronize
    from linear_cache_cost_gate_20261019 import load_runtime,_equal
    from temporal_objective_runtime_20261020 import one_update
    from fixed_edge_pair_20261019 import save_safe
    load_runtime();torch.set_num_threads(1);device=torch.device('cuda',torch.cuda.current_device())
    events,_=load_events();panels,cpu16=cost_panels(events);store=TensorEventStore(events,device)
    initial_equivalence=source_equivalence(device,check)
    write(args.output/'source_equivalence.json',initial_equivalence)
    timings={};saves=[];loads=[];details={};evaluation_estimate=None
    for variant in VARIANTS:
        check();torch.cuda.reset_peak_memory_stats(device);begin=time.perf_counter()
        model,opt,rng,coverage=new_stage(6101,variant,device);synchronize(device);loads.append(time.perf_counter()-begin)
        hs=(1,2,5) if variant=='O' else (1,2,15)
        cases=cost_cases(variant)
        records=[];trace=[]
        identity={'qa_only':True,'seed':6101,'variant':variant,'source_global_step':1280,'source_parent_stage_step':512}
        folder=args.output/f'qa_checkpoint_{variant}';folder.mkdir()
        def update(case,rep):
            h,panel=case;check();synchronize(device);t=time.perf_counter()
            row=one_update(model,opt,store,panels[panel],float(h),FORCE,tail_weighted=variant=='T' and h==15)
            synchronize(device);seconds=time.perf_counter()-t
            records.append({'horizon_s':h,'panel':panel,'rep':rep,'seconds':seconds,'loss':row['loss'],'gradient_norm':row['gradient_norm']})
            trace.append({'stage_step':len(trace)+1,'horizon_s':h,**row})
        # Exactly12 actual updates. First update -> nonempty-Adam checkpoint;
        # second case rep1 is reference, rep2 is the restored matching update.
        update(cases[0],1)
        saved_cpu=torch.get_rng_state().clone();saved_cuda=torch.cuda.get_rng_state(device).clone();saved_numpy=copy.deepcopy(rng.bit_generator.state);saved_coverage=copy.deepcopy(coverage)
        if not opt.state:raise ValueError('QA checkpoint must contain nonempty Adam')
        t=time.perf_counter();save_safe(folder,model,opt,rng,coverage,1,identity,trace);synchronize(device);saves.append(time.perf_counter()-t)
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
        t=time.perf_counter();save_safe(folder,model,opt,rng,coverage,12,identity,trace);synchronize(device);saves.append(time.perf_counter()-t)
        timings[variant]={str(h):max(x['seconds'] for x in records if x['horizon_s']==h) for h in hs}
        details[variant]={'actual_updates':12,'records':records,'restored_next_update_equal':True,'peak_cuda_bytes':torch.cuda.max_memory_allocated(device),'panel_event_ids':{k:[e['id'] for e,_ in s] for k,s in panels.items()},'maximum_actual_future_points':max(len(e['x'])-11 for e in events)}
        if variant=='T':
            check();t=time.perf_counter();evaluate_queue(model,cpu16,args.output/'qa_cpu16',{'scope':'discarded trial cost only'},check)
            elapsed=time.perf_counter()-t;evaluation_estimate=25*elapsed
            details['cpu16']={'actual_seconds':elapsed,'per385_before_margin':evaluation_estimate,'events':16,'includes':'dual-step15s scoring compression export39D'}
    admission=cost_admission(timings,max(saves),max(loads),evaluation_estimate)
    write(args.output/'admission.json',admission)
    result={'source_equivalence':initial_equivalence,'timings':details,'admission':admission,'force':FORCE,'scope':'discarded trials; no scientific endpoint'}
    write(args.output/'cost.json',result);return result





def cohort_sets(events):
    path=ROOT/'results/quadratic_memory_edge_20261020/cohort_identity.json'
    if sha(path)!='0d2b3ed030cd742ae0809114008c31e3a99c35c56d0238d2d03c5473abcecbf5':raise ValueError('frozen cohort identity changed')
    groups=read(path)['cohorts'];ids={e['id'] for e in events}
    fixed={k:{e['id'] for e in groups[k]} for k in ('original366','new19')}
    if len(fixed['original366'])!=366 or len(fixed['new19'])!=19 or fixed['original366']&fixed['new19'] or not ids<=set.union(*fixed.values()):raise ValueError('fixed source cohort mismatch')
    return {k:s&ids for k,s in fixed.items()}

def temporal_metrics(model,event,fine,ps):
    import numpy as np,torch
    from train_structured_rom import _point_error
    n=min(1500,len(event['x'])-11);m=max(1,(n+4)//5)
    row={'actual_future_points':n,'tail_points':m,'endpoint_group':'le5s' if n<=500 else 'gt5s','finite':False,'mean_point_error':None,'tail_point_error':None}
    pred=fine[::2][1:n+1,:30];truth=event['x'][11:11+n,:30]
    if n>0 and np.isfinite(pred).all():
        with torch.no_grad():d=_point_error(torch.as_tensor(pred),torch.as_tensor(truth),model)
        if bool(torch.isfinite(d).all()):row.update(finite=True,mean_point_error=float(d.mean()),tail_point_error=float(d[-m:].mean()))
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
            temporal=temporal_metrics(model,e,f,ps)
            extras.append({'temporal':temporal,'event_id':e['id'],'group':e['group'],'label':e['label'],
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
