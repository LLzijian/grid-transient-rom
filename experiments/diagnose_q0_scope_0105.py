"""0105 TRAIN-only, fixed-model offline q0 diagnostic.

The default command is a no-update timing run.  Optimisation requires a pinned
JSON plan backed by that timing result.  This is an oracle diagnostic: q0 uses
future TRAIN labels and must never be used as a deployed initial state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROCESS_STARTED = time.perf_counter()
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
from train_cross_dissipation_0103 import SOURCE, EXPECTED, verify_source
from train_local_memory_0098 import (read_train, panel_events, graph,
                                     process_memory_bytes)
from train_joint_numerical_pair_0100 import MEMORY_CAP
from train_structured_rom import _point_error
from shared_closure_rom_0101 import load_model
from monitor_joint_adequacy_0099 import _model_hash
from train_shared_closure_0101 import buffer_hash

CARD = ROOT/'reports/task2_q0_scope_0105.md'
DEFAULT_OUTPUT = ROOT/'results/task2_q0_scope_0105/timing'
SOURCE_MODEL_SHA = EXPECTED['model.pt']
DT = .01
PREFIXES = (1, 2, 5)
GRID = (.01, .005)
LR = .01
CLIP = 10.
CAP_SECONDS = 1100.
TIMING_CAP_SECONDS = 150.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def snapshot():
    """Pin all code, split manifests, source artifacts, and preregistration."""
    paths = {'runner0105':Path(__file__), 'card0105':CARD,
             'model0101':ROOT/'experiments/shared_closure_rom_0101.py',
             'model0098':ROOT/'experiments/local_memory_rom_0098.py',
             'objective0078':ROOT/'experiments/train_structured_rom.py',
             'reader0098':ROOT/'experiments/train_local_memory_0098.py',
             'source_guard0103':ROOT/'experiments/train_cross_dissipation_0103.py',
             'numeric_monitor0099':ROOT/'experiments/monitor_joint_adequacy_0099.py',
             'structured_core':ROOT/'experiments/structured_rom.py',
             'source_base':ROOT/'private_data/task2_m3_0020/model.npz',
             'source_normalization':ROOT/'private_data/task2_long_0060_5s/normalization.npz'}
    for directory in ('ieee39_postfault', 'ieee39_expansion'):
        for name in ('dataset_index.csv', 'splits.json'):
            paths[f'{directory}/{name}'] = ROOT/'private_data'/directory/name
    paths.update({f'fixed_{name}':SOURCE/name for name in EXPECTED})
    return {key:{'path':str(path),'sha256':digest(path)} for key,path in paths.items()}


def snapshot_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def lock_check(locked, plan_path=None, plan_sha=None):
    if snapshot() != locked:
        raise RuntimeError('source/code/card/split hash drift')
    if plan_path is not None and digest(plan_path) != plan_sha:
        raise RuntimeError('plan hash drift')


def restore_fixed():
    meta = verify_source(SOURCE)
    model = load_model(SOURCE/'structured_numeric_export.json')
    raw = torch.load(SOURCE/'model.pt', map_location='cpu', weights_only=True)
    if (set(raw) != set(model.state_dict()) or
            any(not torch.equal(raw[k],v) for k,v in model.state_dict().items()) or
            _model_hash(model) != meta['model_state_sha256'] or
            buffer_hash(model) != meta['fixed_buffers_sha256']):
        raise ValueError('fixed model/export identity mismatch')
    if model.dim != 40 or not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('invalid fixed model')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, meta


def train_panel():
    events, excluded = read_train()
    if len(events) != 346 or len(excluded) != 23:
        raise ValueError('TRAIN population drift')
    panel = panel_events(events)
    if (len(panel) != 12 or sum(e['label']=='unstable' for e in panel) != 3 or
            len({e['id'] for e in panel}) != 12 or
            any(e['root'] not in ('ieee39_postfault','ieee39_expansion') for e in panel)):
        raise ValueError('registered 12-event TRAIN panel drift')
    return panel


def panel_fingerprint(panel):
    """Hash selected genuine payloads without scanning multi-GiB unused HDF5 rows."""
    h = hashlib.sha256()
    for e in panel:
        h.update(json.dumps({k:e[k] for k in ('id','root','group','label')},sort_keys=True).encode())
        x = np.ascontiguousarray(e['x'],dtype=np.float64)
        h.update(np.asarray(x.shape,dtype=np.int64).tobytes()); h.update(x.tobytes())
    return h.hexdigest()


def initial_q(model,panel):
    history = torch.as_tensor(np.stack([e['x'][:11] for e in panel]),dtype=torch.float64)
    with torch.no_grad():
        q = model.encode(history).detach().clone()
    if q.shape != (12,10) or not torch.isfinite(q).all():
        raise ValueError('invalid encoded q0')
    return history[:,-1].detach().clone(), q


def available_steps(panel, horizon=5):
    if horizon not in PREFIXES:
        raise ValueError('unregistered horizon')
    lengths = np.asarray([min(int(horizon/DT),len(e['x'])-11) for e in panel],dtype=int)
    if np.any(lengths <= 0):
        raise ValueError('panel event has no real future after 100ms')
    return lengths


def rollout(model,panel,physical,q,*,dt=.01,guard=None):
    """Autonomous RK4; score only original 10ms observations, never pad tails."""
    if dt not in GRID or len(panel) != len(q) or q.shape[1:] != (10,):
        raise ValueError('unregistered grid or q shape')
    if physical.shape != (len(panel),30):
        raise ValueError('physical initial state shape')
    lengths = available_steps(panel,5)
    substeps = round(DT/dt)
    state = torch.cat((physical,q),dim=-1)
    sums = state.new_zeros(len(panel)); counts = state.new_zeros(len(panel))
    out = {}
    with graph(model):
        for offset in range(1,int(lengths.max())+1):
            if guard is not None and offset%25 == 0:
                guard()
            rows = np.flatnonzero(lengths >= offset)
            ix = torch.as_tensor(rows,dtype=torch.long,device=q.device)
            advanced = state.index_select(0,ix)
            for _ in range(substeps):
                advanced = model.rk4(advanced,dt)
            if not torch.isfinite(advanced).all():
                bad = [panel[int(rows[j])]['id'] for j in range(len(rows))
                       if not torch.isfinite(advanced[j]).all()]
                raise FloatingPointError('nonfinite trajectory at step %d: %s'%(offset,bad))
            state = state.index_copy(0,ix,advanced)
            truth = torch.as_tensor(np.stack([panel[j]['x'][10+offset] for j in rows]),
                                    dtype=q.dtype,device=q.device)
            value = _point_error(advanced,truth,model)
            if not torch.isfinite(value).all():
                raise FloatingPointError('nonfinite point error at step %d'%offset)
            sums = sums.index_add(0,ix,value)
            counts = counts.index_add(0,ix,torch.ones_like(value))
            for prefix in PREFIXES:
                if offset == min(prefix*100,int(lengths.max())):
                    n = np.minimum(lengths,prefix*100)
                    if np.all(counts.detach().cpu().numpy()==n):
                        out[prefix] = sums/counts
    if set(out) != set(PREFIXES):
        raise RuntimeError('incomplete genuine prefixes')
    return out


def risk(values):
    if len(values) != 12 or not torch.isfinite(values).all():
        raise FloatingPointError('12 finite event values required for risk')
    return .8*values.mean()+.2*values.topk(math.ceil(.1*len(values))).values.mean()


def metrics(panel,values):
    result = {}
    for prefix in PREFIXES:
        v = values[prefix].detach().cpu()
        lengths = available_steps(panel,prefix)
        result[str(prefix)] = {'risk':float(risk(v)), 'mean':float(v.mean()),
            'tail_top10_mean':float(v.topk(2).values.mean()),
            'events':[{'id':e['id'],'label':e['label'],'root':e['root'],
                       'available_steps':int(n),'available_seconds':float(n*DT),
                       'error':float(error)} for e,n,error in zip(panel,lengths,v)]}
    return result


def evaluate_node(model,panel,physical,q,step,guard=None):
    grids = {}
    with torch.no_grad():
        for dt in GRID:
            grids[str(dt)] = metrics(panel,rollout(model,panel,physical,q,dt=dt,guard=guard))
    coarse = grids['0.01']['5']; fine = grids['0.005']['5']
    tolerance = max(.02,.02*abs(fine['risk']))
    difference = abs(coarse['risk']-fine['risk'])
    unresolved = []
    for a,b in zip(coarse['events'],fine['events']):
        if abs(a['error']-b['error']) > max(.02,.02*abs(b['error'])):
            unresolved.append(a['id'])
    return {'step':step,'q0':q.detach().cpu().tolist(),'grids':grids,
            'risk_grid_difference':difference,'risk_grid_tolerance':tolerance,
            'unresolved_events':unresolved,
            'numeric_qualified':difference<=tolerance and not unresolved}


def guard(deadline,reserve=5.):
    if time.perf_counter()+reserve >= deadline:
        raise TimeoutError('0105 process cap reached')
    if process_memory_bytes() > MEMORY_CAP:
        raise MemoryError('8 GiB process memory cap')


def gradient_once(model,panel,physical,q,deadline):
    probe = q.detach().clone().requires_grad_(True)
    tic = time.perf_counter()
    values = rollout(model,panel,physical,probe,dt=.01,
                     guard=lambda:guard(deadline))
    objective = risk(values[5])
    objective.backward()
    if probe.grad is None or not torch.isfinite(probe.grad).all():
        raise FloatingPointError('nonfinite/missing q0 gradient')
    norm = float(torch.nn.utils.clip_grad_norm_([probe],CLIP))
    if not np.isfinite(norm):
        raise FloatingPointError('nonfinite q0 gradient norm')
    return {'seconds':time.perf_counter()-tic,'risk':float(objective.detach()),
            'gradient_norm':norm,'q0_unchanged':torch.equal(q,probe.detach())}


def timing(output=DEFAULT_OUTPUT):
    """One real backward and complete 12-event two-grid evaluation; no step()."""
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    locked = snapshot(); deadline = PROCESS_STARTED+TIMING_CAP_SECONDS
    result = {'loop':'0105','mode':'timing','complete':False,'parameter_updates':False,
              'source_model_sha256':SOURCE_MODEL_SHA,'source_snapshot':locked,
              'source_snapshot_sha256':snapshot_digest(locked),'failure':None,
              'cap_seconds':TIMING_CAP_SECONDS}
    try:
        model,_ = restore_fixed(); before = _model_hash(model)
        panel = train_panel(); physical,q = initial_q(model,panel)
        result.update(model_state_sha256_before=before,panel_sha256=panel_fingerprint(panel),
                      panel_ids=[e['id'] for e in panel],initial_q0=q.tolist())
        result['backward'] = gradient_once(model,panel,physical,q,deadline)
        tic=time.perf_counter()
        result['node0'] = evaluate_node(model,panel,physical,q,0,
                                        guard=lambda:guard(deadline))
        result['two_grid_seconds'] = time.perf_counter()-tic
        lock_check(locked)
        result['model_state_sha256_after'] = _model_hash(model)
        if before != result['model_state_sha256_after'] or not result['backward']['q0_unchanged']:
            raise RuntimeError('timing changed fixed model or encoded q0')
        guard(deadline,reserve=0.)
        result['complete'] = True
    except Exception as exc:
        result['failure'] = {'type':type(exc).__name__,'message':str(exc)}
    result['seconds'] = time.perf_counter()-PROCESS_STARTED
    result['within_cap'] = time.perf_counter()<=deadline
    result['test_read']=False; result['validation_read']=False
    result['source_unchanged'] = snapshot()==locked
    write_json(output/'timing.json',result)
    return result


def validate_plan(plan, plan_path=None):
    """No free schedule, model, objective, grid, or budget fields."""
    mandatory = {'loop','registered','source_model_sha256','card_sha256',
                 'source_snapshot_sha256','panel_sha256','timing_evidence',
                 'timing_sha256','iterations','nodes','lr','clip_norm',
                 'cap_seconds','grids','objective','selection'}
    if set(plan) != mandatory or plan['loop']!='0105' or plan['registered'] is not True:
        raise ValueError('unregistered or incomplete 0105 plan')
    iterations = plan['iterations']
    if (type(iterations) is not int or iterations not in (30,50) or
            plan['nodes'] != [0,iterations//2,iterations] or
            plan['source_model_sha256'] != SOURCE_MODEL_SHA or
            plan['card_sha256'] != digest(CARD) or
            plan['source_snapshot_sha256'] != snapshot_digest(snapshot()) or
            plan['lr'] != LR or plan['clip_norm'] != CLIP or
            plan['cap_seconds'] != CAP_SECONDS or plan['grids'] != list(GRID) or
            plan['objective'] != '5s genuine TRAIN risk: 0.8 mean + 0.2 top10%' or
            plan['selection'] != 'lowest numeric-qualified node'):
        raise ValueError('0105 fixed plan drift')
    timing_path = Path(plan['timing_evidence'])
    if not timing_path.is_file() or digest(timing_path) != plan['timing_sha256']:
        raise ValueError('timing evidence hash drift')
    evidence = json.loads(timing_path.read_text(encoding='utf-8'))
    if (evidence.get('complete') is not True or evidence.get('within_cap') is not True or
            evidence.get('cap_seconds') != TIMING_CAP_SECONDS or
            evidence.get('parameter_updates') is not False or
            evidence.get('source_snapshot_sha256') != plan['source_snapshot_sha256'] or
            evidence.get('source_model_sha256') != SOURCE_MODEL_SHA or
            evidence.get('panel_sha256') != plan['panel_sha256'] or
            evidence.get('model_state_sha256_before') != evidence.get('model_state_sha256_after') or
            evidence.get('source_unchanged') is not True):
        raise ValueError('ineligible 0105 timing evidence')
    backward = evidence['backward']['seconds']; eval_seconds = evidence['two_grid_seconds']
    if not all(isinstance(v,(int,float)) and np.isfinite(v) and v>0
               for v in (backward,eval_seconds)):
        raise ValueError('invalid 0105 timing costs')
    estimate = 1.2*(iterations*backward + 3*eval_seconds)
    if estimate > CAP_SECONDS-30.:
        raise TimeoutError('registered 0105 plan exceeds 1100s with 20% margin')
    return evidence,estimate


def run_plan(plan_path,output):
    plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    evidence,estimate=validate_plan(plan,plan_path)
    plan_sha=digest(plan_path)
    locked=snapshot()
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    write_json(output/'plan.json',plan)
    deadline=PROCESS_STARTED+CAP_SECONDS
    torch.set_num_threads(1)
    result={'loop':'0105','mode':'offline_q0_oracle','status':'started','failure':None,
            'source_model_sha256':SOURCE_MODEL_SHA,'source_snapshot':locked,
            'plan_sha256':plan_sha,'timing_sha256':plan['timing_sha256'],
            'estimated_seconds_with_margin':estimate,'completed_iterations':0,
            'attempted_iteration':None,'nodes':{},'selected_node':None,
            'source_unchanged':None,'model_unchanged':None,
            'test_read':False,'validation_read':False,'deployment_q0':None}
    model=None; q=None; encoded=None; before=None
    try:
        lock_check(locked,plan_path,plan_sha)
        model,_=restore_fixed(); before=_model_hash(model)
        panel=train_panel(); physical,encoded=initial_q(model,panel)
        if panel_fingerprint(panel)!=plan['panel_sha256']:
            raise ValueError('TRAIN panel payload drift')
        if time.perf_counter()+estimate+30. >= deadline:
            raise TimeoutError('0105 plan no longer fits hard process cap')
        q=torch.nn.Parameter(encoded.detach().clone())
        optimizer=torch.optim.Adam([q],lr=LR)
        result['panel_ids']=[e['id'] for e in panel]
        result['panel_sha256']=plan['panel_sha256']
        result['encoded_q0']=encoded.tolist()
        for step in range(plan['iterations']+1):
            guard(deadline)
            lock_check(locked,plan_path,plan_sha)
            if step in plan['nodes']:
                try:
                    node=evaluate_node(model,panel,physical,q.detach(),step,
                                       guard=lambda:guard(deadline))
                    result['nodes'][str(step)]=node
                except (FloatingPointError,MemoryError,TimeoutError) as exc:
                    result['nodes'][str(step)]={'step':step,'numeric_qualified':False,
                                                'failure':{'type':type(exc).__name__,'message':str(exc)},
                                                'q0':q.detach().tolist()}
                    if isinstance(exc,(MemoryError,TimeoutError)):
                        raise
                write_json(output/'state.json',result)
            if step==plan['iterations']:
                break
            result['attempted_iteration']=step+1
            optimizer.zero_grad(set_to_none=True)
            tic=time.perf_counter()
            values=rollout(model,panel,physical,q,dt=.01,
                           guard=lambda:guard(deadline))
            loss=risk(values[5])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite 5s risk')
            loss.backward()
            if q.grad is None or not torch.isfinite(q.grad).all():
                raise FloatingPointError('nonfinite/missing q0 gradient')
            norm=float(torch.nn.utils.clip_grad_norm_([q],CLIP))
            if not np.isfinite(norm):
                raise FloatingPointError('nonfinite q0 gradient norm')
            old=q.detach().clone()
            optimizer.step()
            if not torch.isfinite(q).all():
                with torch.no_grad():q.copy_(old)
                raise FloatingPointError('nonfinite updated q0; last finite state restored')
            result['completed_iterations']=step+1
            row={'iteration':step+1,'risk_before_update':float(loss.detach()),
                 'gradient_norm':norm,'seconds':time.perf_counter()-tic,
                 'q0':q.detach().tolist()}
            with (output/'iterations.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            write_json(output/'state.json',result)
        qualified=[n for n in result['nodes'].values() if n.get('numeric_qualified')]
        if qualified:
            best=min(qualified,key=lambda n:(n['grids']['0.005']['5']['risk'],n['step']))
            result['selected_node']=best['step']
        result['status']=('completed' if len(result['nodes'])==3 and qualified else
                          'numeric_unresolved')
    except Exception as exc:
        result['status']='failed_state_preserved'
        result['failure']={'type':type(exc).__name__,'message':str(exc),
                           'attempted_iteration':result['attempted_iteration']}
    result['final_q0']=q.detach().tolist() if q is not None and torch.isfinite(q).all() else None
    result['model_unchanged']=bool(model is not None and before==_model_hash(model))
    result['source_unchanged']=snapshot()==locked and digest(plan_path)==plan_sha
    result['seconds']=time.perf_counter()-PROCESS_STARTED
    result['within_cap']=time.perf_counter()<=deadline
    if not result['model_unchanged'] or not result['source_unchanged'] or not result['within_cap']:
        result['status']='identity_or_cap_failure'
    write_json(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('timing','run'),default='timing')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='timing':
        if args.plan is not None:parser.error('timing takes no plan')
        result=timing(args.output or DEFAULT_OUTPUT)
    else:
        if args.plan is None or args.output is None:
            parser.error('run needs --plan and --output')
        result=run_plan(args.plan,args.output)
    print(json.dumps({'status':result.get('status'), 'complete':result.get('complete'),
                      'seconds':result.get('seconds'),'failure':result.get('failure')},allow_nan=False))
    if result.get('failure') or result.get('complete') is False or result.get('status') not in (None,'completed'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
