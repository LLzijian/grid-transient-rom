"""20261019 bounded P0 audit/measurement; train only, never confirmation data."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys, time, traceback
from pathlib import Path
START=time.perf_counter()
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import numpy as np
import torch
from train_closure import write, sha, make_optimizer, environment, cohort_identity, save_checkpoint
from closure_training_runtime import TrainingClosureROM,TensorEventStore,objective,resolve_device,synchronize,all_finite
from train_local_memory_0098 import read_train,graph
SOURCE=ROOT/'private_data/task2_prediction_optimization_0102/formal_S_O/step00650/structured_numeric_export.json'
SOURCE_SHA='c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238'

def train_cohort():
    from closure_discovery_data import load_new_events
    old,old_excluded=read_train()
    new,new_excluded=load_new_events(ROOT/'private_data/closure_discovery_20261017/D/new_data/train','train')
    if (len(old),len(new))!=(346,20):
        raise ValueError(f'locked train cohort changed: {len(old)}+{len(new)}')
    events=old+new
    if len({e['id'] for e in events})!=366:raise ValueError('duplicate train ids')
    # Preserve the old train normalization; adding events does not refit scales.
    force=max(float(np.sqrt(np.mean(np.concatenate([e['torque'] for e in old])**2))),1e-8)
    return events,force,{'old_excluded':old_excluded,'new_excluded':new_excluded,
        'count':366,'old':346,'new':20,'sha256':cohort_identity(events),
        'force_scale':force,'normalization':'original 346 train','val_read':False,'test_read':False}

def source_model(device):
    from shared_closure_rom_0101 import load_model
    if sha(SOURCE)!=SOURCE_SHA:raise ValueError('source hash mismatch')
    m=TrainingClosureROM.from_source(load_model(SOURCE))
    return m.to(device)

def measure(args):
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('new output required')
    output.mkdir(parents=True,exist_ok=True)
    device=resolve_device(args.device);torch.set_num_threads(1)
    record={'status':'RUNNING','pid':os.getpid(),'argv':sys.argv,'environment':environment(device),
        'source_sha256':SOURCE_SHA,'code_sha256':sha(__file__),'test_read':False,'val_read':False,
        'max_seconds':args.max_seconds,'rows':[]}
    write(output/'measurement.json',record)
    try:
        events,force,cohort=train_cohort();write(output/'cohort.json',cohort)
        setup=time.perf_counter();store=TensorEventStore(events,device)
        synchronize(device);record['residency_seconds']=time.perf_counter()-setup
        rng=np.random.default_rng(6101)
        # 12 fixed, distributed train events, genuine intervals only.
        indices=np.linspace(0,len(events)-1,12,dtype=int)
        samples=[(events[i],10) for i in indices]
        for horizon in (1.,2.,5.):
            if time.perf_counter()-START>args.max_seconds-120:raise TimeoutError('cost probe cap')
            model=source_model(device);opt=make_optimizer(model)
            synchronize(device);tick=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with graph(model):
                value,terms=objective(model,store,samples,horizon,force)
                if not all_finite([value]):raise FloatingPointError('nonfinite loss')
                value.backward()
            parameters=[p for p in model.parameters() if p.requires_grad]
            if not all_finite([p.grad for p in parameters]):raise FloatingPointError('nonfinite gradient')
            norm=torch.nn.utils.clip_grad_norm_(parameters,100.)
            opt.step();synchronize(device)
            if not all_finite(parameters):raise FloatingPointError('nonfinite update')
            seconds=time.perf_counter()-tick
            # No timing update is adopted as a scientific candidate.
            folder=output/f'horizon{horizon:g}';folder.mkdir()
            tick=time.perf_counter()
            save_checkpoint(folder,model,opt,rng,{'cost_probe_only':True},1,
                {'source_sha256':SOURCE_SHA,'cohort_sha256':cohort['sha256'],'horizon':horizon},[])
            synchronize(device)
            row={'horizon_s':horizon,'update_seconds':seconds,
                'checkpoint_seconds':time.perf_counter()-tick,'loss':float(value.detach()),
                'gradient_norm':float(norm),'genuine_counts':[min(len(e['x'])-11,int(horizon*100)) for e,_ in samples]}
            record['rows'].append(row);write(output/'measurement.json',record)
            print(json.dumps(row),flush=True)
        med=float(np.mean([r['update_seconds'] for r in record['rows']]))
        record.update(status='COMPLETED',paired_6x1024_update_estimate_seconds=med*6*1024*1.25,
            estimate_scope='single fixed train panel, includes 25% margin; excludes full evaluation/search/FOM; not full-plan approval')
    except BaseException as error:
        record.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_process_seconds']=time.perf_counter()-START
        write(output/'measurement.json',record)
        write(output/'receipt.json',{'stage':'P0','seconds':time.perf_counter()-START,
            'status':record['status'],'pid':os.getpid(),'includes_child_cost':False,'command':sys.argv})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args()
    if a.max_seconds<=120 or not np.isfinite(a.max_seconds):p.error('finite cap >120 seconds required')
    measure(a)
