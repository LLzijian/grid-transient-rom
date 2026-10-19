"""Train-only A/B/C joint study. Every scientific invocation is native-supervised."""
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
BASE=ROOT/'private_data/closure_mechanism_20261019/P2/quadratic_memory_edge01'
FRESH=ROOT/'private_data/closure_mechanism_20261019/P3/fresh_design_pair01'
SEEDS=(6101,6102,6103)
VARIANTS=('A','B','C')
RATES={'skeleton':.0009,'closure':.00009,'encoder':.000027}
FORCE=1.0068379869405584
SOURCE_HASHES={6101:('9810f6126134912d1dcaec97311decd6ca5bd6f383f67e707824d4a3ab65e74c','25961a6dc3fb43d99b31234840c6149b14ececeb9a0ab26b15c866715185f7e8'),6102:('f866d90bc90cc2391e968abe0c93531122205c3ef1166f00b78765c9c7ce984b','7b6bc037d58c3b738b88f66050b00aceabbab801aa682c9c331f47619ab29279'),6103:('6d55f970245dfa71305a24cc1c0bdfb3426c480ce27d1be4667910fb3045ccc4','7a91cd3202254d4ca554322ce838d4e7d0eea080429d8c973096a61eda39c747')}

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
    return FRESH/'models'/f'{seed}_full8/checkpoints/step001280'
def source_metadata(seed):
    p=source_folder(seed);m=read(p/'checkpoint.json')
    if m['step']!=1280 or (m['model_sha256'],m['training_sha256'])!=SOURCE_HASHES[seed]: raise ValueError('source metadata changed')
    for n,h in zip(('model.json','training.pt'),SOURCE_HASHES[seed]):
        if sha(p/n)!=h: raise ValueError('source bytes changed')
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

def schedule_rows(events,rng,coverage,updates=768):
    from train_joint_numerical_pair_0100 import next_samples
    rows=[]
    for step in range(1,updates+1):
        before=copy.deepcopy(rng.bit_generator.state);samples,sampling=next_samples(events,rng,coverage)
        h=(1.,2.,5.)[(step-1)%3]
        if len(samples)!=12: raise ValueError('registered10+2 batch changed')
        rows.append({'stage_step':step,'horizon_s':h,'sampling':sampling,'numpy_rng_before':before,
                     'numpy_rng_after':copy.deepcopy(rng.bit_generator.state),
                     'long_counts':[min(round(h*100),len(e['x'])-11) for e,_ in samples],
                     'local_counts':[min(20,len(e['x'])-i-1) for e,i in samples]})
    return rows
def prepare(output):
    import numpy as np
    import torch
    from experiments.active_fom_pair_fit_20261020 import event_identity
    events,excluded=load_events();identity=event_identity(events);result={'events':identity,'seeds':{},'force':FORCE,'excluded':excluded}
    for seed in SEEDS:
        meta=source_metadata(seed)
        if meta['identity']['events']!=identity: raise ValueError('source sorted385 data identity differs')
        state=torch.load(source_folder(seed)/'training.pt',map_location='cpu',weights_only=False)
        if state['step']!=1280 or state['cuda_rng'] is None: raise ValueError('complete source RNG missing')
        rng=np.random.default_rng();rng.bit_generator.state=state['numpy_rng'];coverage=copy.deepcopy(state['coverage'])
        if sorted(coverage['cover_order'])!=list(range(385)) or type(coverage['cursor']) is not int: raise ValueError('source coverage identity differs')
        initial_rng=copy.deepcopy(rng.bit_generator.state);initial_coverage=copy.deepcopy(coverage)
        rows=schedule_rows(events,rng,coverage)
        result['seeds'][str(seed)]={'rows':rows,'initial_rng':initial_rng,'initial_coverage':initial_coverage,
            'source_model_sha256':meta['model_sha256'],'source_training_sha256':meta['training_sha256'],
            'source_metadata_sha256':sha(source_folder(seed)/'checkpoint.json')}
    write(output/'schedule.json',result)
    return {'status':'COMPLETED','schedule_sha256':sha(output/'schedule.json'),'events':385,'rows_per_seed':768}

def enable_runtime(model):
    from quadratic_memory_runtime_20261020 import QuadraticMemoryRuntime,_check
    model.__class__=QuadraticMemoryRuntime;model._clear_runtime();_check(model)
    return model
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
    from quadratic_memory_runtime_20261020 import build_variant
    source_metadata(seed)
    state=torch.load(source_folder(seed)/'training.pt',map_location='cpu',weights_only=False)
    model=build_variant(TrainingClosureROM.from_export(source_folder(seed)/'model.json'),variant).to(device).double()
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
    if seed not in SEEDS or variant not in VARIANTS or target not in (512,768): raise ValueError('unregistered science identity/target')
    if type(start) is not int or start<0 or start%32 or start>=target: raise ValueError('unsafe or completed resume boundary')
    if external and not resume: raise ValueError('external recovery requires resume')
    if not resume and start!=0: raise ValueError('fresh run starts stage0')
    if target==768 and (not resume or start<512 or extension is None): raise ValueError('768 requires common extension lock and complete512')
    if resume and not external and start!=512: raise ValueError('normal continuation only512 to768')
def extension_trend(trace):
    if len(trace)!=512: raise ValueError('trend needs complete512')
    values=[float(t['loss']) for t in trace[384:512]]
    if not all(math.isfinite(x) for x in values): return False
    a=sum(values[:64])/64;b=sum(values[64:])/64
    return a>0 and (a-b)/a>=.03
def check_extension(path,admission):
    lock=read(inside(path))
    if admission.get('allow768') is not True or lock.get('allowed') is not True or set(lock.get('endpoints',{}))!={f'{s}_{v}' for s in SEEDS for v in VARIANTS}: raise ValueError('all-nine common extension required')
    trend=False
    for k,entry in lock['endpoints'].items():
        p=BASE/'models'/k/'checkpoints/step000512/checkpoint.json'
        if sha(p)!=entry['checkpoint_sha256']: raise ValueError('extension endpoint changed')
        meta=read(p)
        if f"{meta['identity']['seed']}_{meta['identity']['variant']}"!=k:raise ValueError('extension model identity mismatch')
        if meta['step']!=512 or len(meta['trace'])!=512 or not all(math.isfinite(float(t['loss'])) for t in meta['trace']): raise ValueError('nine finite512 endpoints required')
        if k.endswith('_C'):trend|=extension_trend(meta['trace'])
    if not trend or lock.get('all_nine_extra_cost_fits') is not True: raise ValueError('extension trend/cost gate unmet')

def train(args,check):
    if args.seed not in SEEDS or args.variant not in VARIANTS or args.target_step not in (512,768):raise ValueError('unregistered model identity')
    import torch
    from closure_training_runtime import TensorEventStore,synchronize
    from active_fom_pair_fit_20261020 import event_identity
    from linear_cache_cost_gate_20261019 import load_runtime,one_update
    from fixed_edge_pair_20261019 import save_safe,verify_identity
    from train_joint_numerical_pair_0100 import next_samples
    load_runtime();torch.set_num_threads(1);device=torch.device('cuda',torch.cuda.current_device())
    events,_=load_events();schedule=read(args.schedule_path)
    verify_identity(event_identity(events),schedule['events']);rows=schedule['seeds'][str(args.seed)]
    admission=load_admission(args.admission)
    if admission.get('allow512') is not True: raise ValueError('complete512 cost admission required')
    if args.target_step==768:check_extension(args.extension_lock,admission)
    expected=BASE/'models'/f'{args.seed}_{args.variant}'
    if args.output!=expected.resolve():raise ValueError('exact model output required')
    identity={'seed':args.seed,'variant':args.variant,'source_global_step':1280,'optimizer_policy':'all-nine fresh Adam at stage0',
        'source_model_sha256':SOURCE_HASHES[args.seed][0],'source_training_sha256':SOURCE_HASHES[args.seed][1],
        'schedule_sha256':sha(args.schedule_path),'data':schedule['events'],'rates':RATES,'force':FORCE,
        'source_commit':args.source_commit}
    checkpoints=args.output/'checkpoints';trace=[];start=0
    if args.resume:
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
    result={'status':'RUNNING','seed':args.seed,'variant':args.variant,'source_global_step':1280,'stage_step':start,'target_step':args.target_step,'external_recovery':args.external_recovery}
    try:
        for step in range(start+1,args.target_step+1):
            check();row=rows['rows'][step-1]
            if row['stage_step']!=step:raise ValueError('schedule position mismatch')
            verify_identity(rng.bit_generator.state,row['numpy_rng_before']);samples,sampling=next_samples(events,rng,coverage)
            verify_identity(sampling,row['sampling']);verify_identity(rng.bit_generator.state,row['numpy_rng_after'])
            if [min(round(row['horizon_s']*100),len(e['x'])-11) for e,_ in samples]!=row['long_counts'] or [min(20,len(e['x'])-i-1) for e,i in samples]!=row['local_counts']:raise ValueError('actual sampled interval changed')
            t=one_update(model,opt,store,samples,row['horizon_s'],FORCE)
            trace.append({'stage_step':step,'horizon_s':row['horizon_s'],'sampling':sampling,**t});result['stage_step']=step
            if step%32==0:save_safe(checkpoints,model,opt,rng,coverage,step,identity,trace)
            write(attempt/'progress.json',result)
        synchronize(device);result['status']='COMPLETED';return result
    except BaseException as exc:result.update(status='UNRESOLVED',error=repr(exc));raise
    finally:write(attempt/'training_result.json',result)

def cost_admission(update_seconds,save_seconds,load_seconds,evaluation_seconds):
    if set(update_seconds)!={'A','C'}:raise ValueError('both actual A and C costs required')
    per={h:max(update_seconds[v][str(h)] for v in ('A','C')) for h in (1,2,5)}
    if any(not math.isfinite(t) or t<=0 for t in (*per.values(),save_seconds,load_seconds,evaluation_seconds)):raise ValueError('finite positive actual timings required')
    def estimate(n):
        counts=[sum((1,2,5)[i%3]==h for i in range(n)) for h in (1,2,5)]
        model=1.25*(sum(c*per[h] for c,h in zip(counts,(1,2,5)))+(n//32+1)*save_seconds+load_seconds)
        return {'per_model_seconds':model,'nine_models_seconds':9*model}
    s512,s768=estimate(512),estimate(768)
    eval9=1.25*9*evaluation_seconds;eval18=1.25*18*evaluation_seconds
    return {'est512':s512,'est768':s768,'nine_evaluations_seconds':eval9,'eighteen_evaluations_seconds':eval18,
            'allow512':s512['per_model_seconds']<=3000 and s512['nine_models_seconds']<=27000 and eval9<=3500,
            'allow768':s768['per_model_seconds']<=3000 and s768['nine_models_seconds']<=27000 and eval18<=3500}

def load_admission(path):
    admission=read(path)
    if any(type(admission.get(k)) is not bool for k in ('allow512','allow768')) or not all(k in admission for k in ('est512','est768','nine_evaluations_seconds','eighteen_evaluations_seconds')):
        raise ValueError('actual flat cost admission document required')
    return admission

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','cost','train','evaluate'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-output',type=Path)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--source-commit',required=True)
    for n in ('binding','schedule-path','admission','extension-lock','model'):p.add_argument('--'+n,type=Path)
    p.add_argument('--seed',type=int);p.add_argument('--variant');p.add_argument('--target-step',type=int,default=512)
    p.add_argument('--recovery-progress',type=Path)
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
        if time.perf_counter()-started>=a.max_seconds-60:raise TimeoutError('registered category deadline')
    try:
        if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('archived source commit required')
        cap={'prepare':180,'cost':1200,'train':3000,'evaluate':3500}[a.mode]
        if not math.isfinite(a.max_seconds) or not math.isfinite(a.previous_seconds) or not 0<a.max_seconds<=cap-a.previous_seconds or a.previous_seconds<0:raise ValueError('cumulative mode/model cap exceeded')
        binding(a.binding)
        for name in ('schedule_path','admission','extension_lock','recovery_progress'):
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
    from quadratic_memory_runtime_20261020 import QuadraticMemoryRuntime
    p=Path(args.model).resolve()
    legal={(BASE/'models'/f'{s}_{v}'/'checkpoints'/f'step{n:06d}'/'model.json').resolve():(s,v,n) for s in SEEDS for v in VARIANTS for n in (512,768)}
    if p not in legal:raise ValueError('only exact registered science endpoint may be evaluated')
    seed,variant,step=legal[p];meta=read(p.parent/'checkpoint.json')
    if (meta['identity']['seed'],meta['identity']['variant'],meta['step'])!=(seed,variant,step):raise ValueError('endpoint stage identity differs')
    if sha(p)!=meta['model_sha256'] or sha(p.parent/'training.pt')!=meta['training_sha256'] or len(meta['trace'])!=step:raise ValueError('endpoint identity incomplete')
    if args.output!=(BASE/'evaluation'/f'{seed}_{variant}'/f'step{step:06d}').resolve():raise ValueError('exclusive endpoint evaluation path required')
    torch.set_num_threads(1);events,excluded=load_events()
    model=QuadraticMemoryRuntime.from_export(p).cpu().double().eval()
    result=evaluate_queue(model,events,args.output/'queue',{'seed':seed,'variant':variant,'stage_step':step,'model_sha256':sha(p),'excluded':excluded},check)
    return {'event_count':result['event_count'],'evidence':str(args.output/'queue/evidence.json'),'evidence_sha256':sha(args.output/'queue/evidence.json'),'primary':result['primary']}

def measure_cost(args,check):
    import torch,numpy as np
    from closure_training_runtime import TensorEventStore,synchronize
    from linear_cache_cost_gate_20261019 import load_runtime,one_update,_equal
    from fixed_edge_pair_20261019 import save_safe
    load_runtime();torch.set_num_threads(1);device=torch.device('cuda',torch.cuda.current_device())
    events,_=load_events();long=[e for e in events if len(e['x'])>=531]
    if len(long)<16:raise ValueError('actual full-horizon cost panel unavailable')
    store=TensorEventStore(events,device);samples=[(e,11) for e in long[:12]]
    timings={};saves=[];loads=[];details={}
    for variant in ('C','A'):
        check();begin=time.perf_counter();model,opt,rng,coverage=new_stage(6101,variant,device);synchronize(device);loads.append(time.perf_counter()-begin)
        recorded={str(h):[] for h in (1,2,5)};trace=[]
        for step in range(1,33):
            check();h=(1,2,5)[(step-1)%3];synchronize(device);t=time.perf_counter()
            row=one_update(model,opt,store,samples,float(h),FORCE);synchronize(device)
            recorded[str(h)].append(time.perf_counter()-t);trace.append({'stage_step':step,'horizon_s':h,**row})
        timings[variant]={h:max(v) for h,v in recorded.items()}
        identity={'qa_only':True,'seed':6101,'variant':variant,'source_global_step':1280}
        folder=args.output/f'qa_checkpoint_{variant}';folder.mkdir()
        t=time.perf_counter();save_safe(folder,model,opt,rng,coverage,32,identity,trace);synchronize(device);saves.append(time.perf_counter()-t)
        # A real next update and exact Adam/RNG restoration use the same production
        # restore_stage consumer. These updates are never scientific endpoints.
        torch_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state(device).clone()
        t=time.perf_counter();restored=restore_stage(folder,device,identity);synchronize(device);loads.append(time.perf_counter()-t)
        rm,ro,rr,rc,metadata,_=restored
        _equal(opt.state_dict(),ro.state_dict(),exact=True);_equal(rng.bit_generator.state,rr.bit_generator.state,exact=True);_equal(coverage,rc,exact=True)
        if not torch.equal(torch_before,torch.get_rng_state()) or not torch.equal(cuda_before,torch.cuda.get_rng_state(device)):raise ValueError('saved source RNG restoration differs')
        one_update(model,opt,store,samples,5.,FORCE)
        torch.set_rng_state(torch_before);torch.cuda.set_rng_state(cuda_before,device)
        one_update(rm,ro,store,samples,5.,FORCE)
        _equal(model.state_dict(),rm.state_dict(),exact=False);_equal(opt.state_dict(),ro.state_dict(),exact=False)
        details[variant]={'actual_updates':34,'safe_checkpoint_step':32,'production_restore_equal':True,'horizon_update_seconds':recorded}
        if variant=='C':
            check();t=time.perf_counter()
            result=evaluate_queue(model,long[:16],args.output/'qa_C_cpu16',{'scope':'QA cost only; trial updates discarded; no science endpoint'},check)
            elapsed=time.perf_counter()-t
            # 25 actual16-event compute/score/compress/export/spectrum packages
            # deliberately overcount export/spectrum25x before the25% margin.
            evaluation_estimate=25*elapsed
            details['C_cpu16']={'actual_seconds':elapsed,'events':16,'per385_conservative_seconds':evaluation_estimate,
                'includes':'15s fine/coarse quadratic fastpath, actual scoring, compression, export and39D analysis'}
    admission=cost_admission(timings,max(saves),max(loads),evaluation_estimate)
    write(args.output/'admission.json',admission)
    result={'timings':details,'admission':admission,'force':FORCE,'scope':'QA-only discarded updates; no scientific endpoints'}
    result['admission_path']=str(args.output/'admission.json');result['admission_sha256']=sha(args.output/'admission.json')
    write(args.output/'cost.json',result);return result



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
            extras.append({'event_id':e['id'],'group':e['group'],'label':e['label'],
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
    for name,is_new in (('original366',False),('new19',True)):
        ids={e['id'] for e in events if e['id'].startswith('newtrain_')==is_new}
        cohort_records={k:[r for r in rows if r['event_id'] in ids] for k,rows in records.items()}
        result['cohorts'][name]={'event_count':len(ids),'summary':build_summary(cohort_records),'primary':primary_risk(cohort_records['objective_point_error'],qualifications),'extras':[r for r in extras if r['event_id'] in ids]}
    write(output/'evidence.json',_safe(result))
    write(output/'complete.json',{'status':'COMPLETED','event_count':len(events),'evidence_sha256':sha(output/'evidence.json')})
    return result

if __name__=='__main__': raise SystemExit(main())
