"""0106 TRAIN-only oracle-target pilot. Default mode measures cost without updates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
import train_cross_dissipation_0103 as source
import train_prediction_0102 as prior
from train_local_memory_0098 import graph, panel_events, process_memory_bytes
from train_joint_numerical_pair_0100 import digest, write_json, next_samples, parameter_layout, MEMORY_CAP
from monitor_joint_adequacy_0099 import build_manifest, run_monitor, _model_hash
from train_shared_closure_0101 import buffer_hash

CARD = ROOT/'reports/task2_oracle_distillation_pilot_0106.md'
TARGET = ROOT/'private_data/task2_q0_scope_0105/formal/result.json'
BASELINE = source.SOURCE.parent/'monitor_step00650/evidence.json'
START_STEP = 2817
RATES = source.RATES
TARGET_SHA256 = '645e7925fcbb6f0545e0a4a960c31a0ebb1775c69dfef7d522b3b4b34bbfef98'
PANEL_SHA256 = '7ed22834756df47bcf1d1eab24a7ce52c3ffc3992ef5cbd07dcf31e5441a9b89'
PANEL_IDS = ('expansion_0008','pilot_0174','pilot_0202','pilot_0000','expansion_0000',
             'pilot_0160','pilot_0192','pilot_0208','pilot_0240','expansion_0016',
             'pilot_0016','pilot_0272')
BASELINE_J = 13.184177688487152


def guard(deadline, reserve=15.):
    if time.perf_counter()+reserve >= deadline:
        raise TimeoutError('0106 process deadline or serialization reserve reached')
    if process_memory_bytes() > MEMORY_CAP:
        raise MemoryError('8 GiB process cap')


def snapshot():
    paths = {'runner0106':Path(__file__), 'card0106':CARD, 'target0105':TARGET,
             'baseline0102':BASELINE, 'runner0103':Path(source.__file__),
             'runner0102':Path(prior.__file__),
             'diagnostic0105':ROOT/'experiments/diagnose_q0_scope_0105.py',
             'model0101':ROOT/'experiments/shared_closure_rom_0101.py',
             'model0098':ROOT/'experiments/local_memory_rom_0098.py',
             'structured_core':ROOT/'experiments/structured_rom.py',
             'objective0098':ROOT/'experiments/train_local_memory_0098.py',
             'monitor0099':ROOT/'experiments/monitor_joint_adequacy_0099.py',
             'sampler0100':ROOT/'experiments/train_joint_numerical_pair_0100.py'}
    paths.update({'source_'+k:source.SOURCE/k for k in source.EXPECTED})
    paths.update({'dependency_'+k:v['path'] for k,v in source.snapshot().items()})
    return {k:{'path':str(v),'sha256':digest(v)} for k,v in paths.items()}


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_targets(events):
    if digest(TARGET) != TARGET_SHA256:
        raise ValueError('0105 target artifact hash drift')
    result = json.loads(TARGET.read_text(encoding='utf-8'))
    node = result.get('nodes',{}).get('50',{})
    if (result.get('status')!='completed' or result.get('failure') is not None
            or result.get('source_model_sha256')!=source.EXPECTED['model.pt']
            or result.get('selected_node')!=50 or node.get('step')!=50
            or node.get('numeric_qualified') is not True
            or result.get('panel_sha256')!=PANEL_SHA256
            or tuple(result.get('panel_ids',()))!=PANEL_IDS
            or result.get('test_read') is not False or result.get('validation_read') is not False):
        raise ValueError('0105 selected TRAIN node identity/qualification drift')
    panel=panel_events(events)
    if tuple(e['id'] for e in panel)!=PANEL_IDS:
        raise ValueError('TRAIN panel IDs/order drift')
    # Reuse the exact 0105 payload fingerprint definition on the already loaded TRAIN rows.
    from diagnose_q0_scope_0105 import panel_fingerprint
    if panel_fingerprint(panel)!=PANEL_SHA256:
        raise ValueError('TRAIN panel payload drift')
    q=torch.as_tensor(node.get('q0'),dtype=torch.float64)
    histories=torch.as_tensor(np.stack([e['x'][:11] for e in panel]),dtype=torch.float64)
    if q.shape!=(12,10) or histories.shape!=(12,11,30) or not torch.isfinite(q).all():
        raise ValueError('invalid offline target/history shape or value')
    return histories,q


def restore():
    model,rng,coverage,meta=source.restore_arm('S-C')
    if _model_hash(model)!=meta['model_state_sha256']:
        raise ValueError('0102 source model state drift')
    optimizer=source.optimizer_for(model,RATES)
    if optimizer.state:
        raise ValueError('Adam was not reset')
    return model,optimizer,rng,coverage,meta


def baseline_identity():
    row=json.loads(BASELINE.read_text(encoding='utf-8'))
    if (row.get('complete') is not True or row.get('numerical_pass') is not True
            or row.get('model_state_sha256_before')!=source.verify_source()['model_state_sha256']
            or not np.isclose(row.get('J_fine'),BASELINE_J,rtol=0,atol=1e-12)):
        raise ValueError('0102 baseline J/source identity drift')
    return {'path':str(BASELINE),'sha256':digest(BASELINE),'J_fine':BASELINE_J}


def auxiliary_weight(update,updates):
    if updates not in (45,60) or type(update) is not int or not 1<=update<=updates:
        raise ValueError('unregistered 0106 update')
    return 10. if update <= (30 if updates==60 else 22) else 0.


def gradients(model,optimizer,samples,h,scale,histories,target,weight):
    optimizer.zero_grad(set_to_none=True)
    tic=time.perf_counter()
    with graph(model):
        original,terms=prior.objective(model,samples,prior.histories_for(samples),h,scale)
        auxiliary=(model.encode(histories)-target).square().mean() if weight else original.new_zeros(())
        total=original+weight*auxiliary
        if not torch.isfinite(total) or any(not torch.isfinite(v).all() for v in terms.values()):
            raise FloatingPointError('nonfinite objective')
        total.backward()
    block_sq={}; params=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise FloatingPointError('missing/nonfinite gradient: '+name)
        block=source.parameter_block(name)
        block_sq[block]=block_sq.get(block,0.)+float(p.grad.detach().square().sum())
        params.append(p)
    norm=float(torch.nn.utils.clip_grad_norm_(params,100.))
    if not np.isfinite(norm):raise FloatingPointError('nonfinite clipped gradient norm')
    return {'original_loss':float(original.detach()),'auxiliary_mse':float(auxiliary.detach()),
            'auxiliary_weight':weight,'total_loss':float(total.detach()),
            'terms':{k:prior.summarize_term(v) for k,v in terms.items()},
            'gradient_norm':norm,'gradient_clipped':norm>100.,
            'gradient_block_l2':{k:float(np.sqrt(v)) for k,v in block_sq.items()},
            'seconds':time.perf_counter()-tic}


def update(model,optimizer,layout,completed,fixed):
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('nonfinite post-update parameter')
    if buffer_hash(model)!=fixed:raise RuntimeError('fixed model buffers changed')
    source.validate_optimizer(model,optimizer,layout,completed)
    prior.check_structure(model)
    blocks={}
    for name,p in model.named_parameters():
        if name in before:
            b=source.parameter_block(name)
            blocks[b]=blocks.get(b,0.)+float((p.detach()-before[name]).square().sum())
    return {'update_block_l2':{k:float(np.sqrt(v)) for k,v in blocks.items()}}


def timing(output,cap_seconds=120.):
    if not 0<cap_seconds<=120.:raise ValueError('QA process cap must be <=120 seconds')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+cap_seconds;torch.set_num_threads(1);guard(deadline)
    locked=snapshot();model,opt,rng,coverage,_=restore()
    baseline=baseline_identity();events,scale=prior.train_inputs();histories,target=verify_targets(events)
    initial=prior.capture_rng(rng);initial_coverage=copy.deepcopy(coverage);initial_hash=_model_hash(model)
    rows=[];monitor=None;failure=None
    try:
        for i,h in enumerate((1.,2.,5.)):
            guard(deadline,25.)
            samples,trace=next_samples(events,rng,coverage)
            row=gradients(model,opt,samples,h,scale,histories,target,10.)
            rows.append({'horizon':h,'sampling':trace,**row})
        guard(deadline,20.)
        monitor=run_monitor(model,events,build_manifest(events),output/'monitor_timing',deadline-5,
                            allow_refine=True,force_scale=scale)
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
    finally:
        opt.zero_grad(set_to_none=True);prior.restore_rng(initial,rng)
        coverage.clear();coverage.update(initial_coverage)
    unchanged=(_model_hash(model)==initial_hash and not opt.state
               and prior.rng_digest(prior.capture_rng(rng))==prior.rng_digest(initial))
    result={'loop':'0106','mode':'timing','complete':failure is None and len(rows)==3
            and monitor is not None and monitor.get('complete') is True
            and monitor.get('numerical_pass') is True and unchanged
            and snapshot()==locked and time.perf_counter()<=deadline,
            'failure':failure,'parameter_updates':False,'source_snapshot':locked,
            'source_snapshot_sha256':json_hash(locked),'source_model_sha256':source.EXPECTED['model.pt'],
            'target_sha256':TARGET_SHA256,'panel_sha256':PANEL_SHA256,'baseline':baseline,
            'model_state_sha256_before':initial_hash,'model_state_sha256_after':_model_hash(model),
            'rng_unchanged':prior.rng_digest(prior.capture_rng(rng))==prior.rng_digest(initial),
            'coverage_before':initial_coverage,'coverage_after':coverage,
            'batches':rows,'monitor_seconds':monitor.get('seconds') if monitor else None,
            'monitor_complete':monitor.get('complete') if monitor else False,
            'monitor_numerical_pass':monitor.get('numerical_pass') if monitor else False,
            'test_read':False,'validation_read':False,'seconds':time.perf_counter()-PROCESS_STARTED,
            'cap_seconds':cap_seconds}
    result['coverage_unchanged']=coverage==initial_coverage
    result['complete'] &= result['coverage_unchanged']
    write_json(output/'costs.json',result)
    return result


def projected_seconds(costs,updates):
    rows=costs['batches']
    by_h={int(r['horizon']):r['seconds'] for r in rows}
    if set(by_h)!={1,2,5} or any(not np.isfinite(v) or v<=0 for v in by_h.values()):
        raise ValueError('missing valid 1/2/5 backward timings')
    monitor=costs['monitor_seconds']
    if not isinstance(monitor,(int,float)) or not np.isfinite(monitor) or monitor<=0:
        raise ValueError('invalid monitor timing')
    return 1.2*(sum(by_h[int(prior.horizon(START_STEP+i))] for i in range(1,updates+1))+monitor)+100.


def make_plan(cost_path):
    cost_path=Path(cost_path);cost=json.loads(cost_path.read_text(encoding='utf-8'))
    if (cost.get('complete') is not True or cost.get('parameter_updates') is not False
            or cost.get('source_snapshot')!=snapshot() or cost.get('target_sha256')!=TARGET_SHA256
            or cost.get('source_model_sha256')!=source.EXPECTED['model.pt']
            or cost.get('model_state_sha256_before')!=cost.get('model_state_sha256_after')
            or cost.get('rng_unchanged') is not True or cost.get('coverage_unchanged') is not True
            or cost.get('monitor_complete') is not True
            or cost.get('monitor_numerical_pass') is not True):
        raise ValueError('timing evidence identity/invariance drift')
    updates=next((n for n in (60,45) if projected_seconds(cost,n)<700.),None)
    if updates is None:raise TimeoutError('neither registered schedule fits formal cap')
    return {'loop':'0106','registered':True,'updates':updates,
            'auxiliary_steps':30 if updates==60 else 22,
            'horizons':[prior.horizon(START_STEP+i) for i in range(1,updates+1)],
            'rates':RATES,'clip_norm':100.,'formal_cap_seconds':700.,
            'qa_cap_seconds':120.,'monitor_step':updates,
            'source_model_sha256':source.EXPECTED['model.pt'],
            'source_snapshot_sha256':json_hash(snapshot()),'card_sha256':digest(CARD),
            'target_sha256':TARGET_SHA256,'panel_sha256':PANEL_SHA256,
            'cost_evidence':str(cost_path.resolve()),'cost_evidence_sha256':digest(cost_path),
            'projected_seconds_including_margin_and_reserve':projected_seconds(cost,updates)}


def validate_plan(plan):
    expected=make_plan(plan['cost_evidence'])
    if plan!=expected:
        raise ValueError('0106 plan is not the registered timing-derived schedule')
    return plan


def checkpoint(output,name,model,opt,rng,coverage,completed,locked,plan_sha,valid=True,attempted=None):
    directory=Path(output)/name;directory.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),directory/'model.pt')
    torch.save(opt.state_dict(),directory/'optimizer.pt')
    torch.save(dict(prior.capture_rng(rng),coverage=copy.deepcopy(coverage)),directory/'rng.pt')
    meta={'format':'oracle_distill_0106','source_step':START_STEP,
          'source_model_sha256':source.EXPECTED['model.pt'],'optimizer_reset':True,
          'valid_training_state':valid,'completed_updates':completed if valid else None,
          'last_verified_updates':completed,'attempted_update':attempted,
          'step':START_STEP+completed if valid else None,'model_kind':'S23',
          'dimension':model.dim,'training_dt':.01,'config':model.config(),
          'coverage':copy.deepcopy(coverage),
          'requires_grad':{n:p.requires_grad for n,p in model.named_parameters()},
          'parameter_numel':{n:p.numel() for n,p in model.named_parameters()},
          'optimizer_parameter_names':parameter_layout(model,opt),
          'model_state_sha256':_model_hash(model),'fixed_buffers_sha256':buffer_hash(model),
          'source_snapshot':locked,'plan_sha256':plan_sha,
          'target_sha256':TARGET_SHA256,'panel_sha256':PANEL_SHA256,
          'process_seconds':time.perf_counter()-PROCESS_STARTED,
          'test_read':False,'validation_read':False}
    write_json(directory/'checkpoint.json',meta if valid else prior.safe_json(meta))
    if valid:model.export(directory/'structured_numeric_export.json')
    write_json(directory/'artifact_hashes.json',{p.name:digest(p) for p in directory.iterdir() if p.is_file()})
    return str(directory)


def run_plan(plan_path,output):
    plan_path=Path(plan_path);plan=validate_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    plan_sha=digest(plan_path);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+700.;torch.set_num_threads(1);guard(deadline)
    locked=snapshot();model,opt,rng,coverage,_=restore();baseline=baseline_identity()
    events,scale=prior.train_inputs();histories,target=verify_targets(events)
    layout=parameter_layout(model,opt);fixed=buffer_hash(model)
    if time.perf_counter()+plan['projected_seconds_including_margin_and_reserve']>=deadline:
        raise TimeoutError('formal 0106 process cannot fit measured schedule')
    write_json(output/'plan.json',plan)
    completed=0;attempted=None;failure=None;monitor=None;status='running';last=None
    try:
        for i in range(1,plan['updates']+1):
            guard(deadline,100.)
            if snapshot()!=locked or digest(plan_path)!=plan_sha:
                raise RuntimeError('source/code/card/target/plan hash drift')
            attempted=i;tic=time.perf_counter()
            samples,trace=next_samples(events,rng,coverage)
            row={'update':i,'horizon':plan['horizons'][i-1],
                 'sampling':trace,'torch_rng_before_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
            with (output/'samples.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
            row.update(gradients(model,opt,samples,row['horizon'],scale,histories,target,
                                 auxiliary_weight(i,plan['updates'])))
            row.update(update(model,opt,layout,i,fixed));completed=i
            row['seconds']=time.perf_counter()-tic
            row['torch_rng_after_sha256']=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
            with (output/'trace.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n')
        last=checkpoint(output,f'step{completed:05d}',model,opt,rng,coverage,completed,locked,plan_sha)
        state=prior.capture_rng(rng);model_hash=_model_hash(model)
        monitor=run_monitor(model,events,build_manifest(events),output/'monitor_terminal',deadline-20,
                            allow_refine=True,force_scale=scale)
        if _model_hash(model)!=model_hash or prior.rng_digest(prior.capture_rng(rng))!=prior.rng_digest(state):
            raise RuntimeError('terminal monitor changed model/RNG')
        status=('numeric_monitor_incomplete' if monitor.get('complete') is not True else
                'completed' if monitor.get('numerical_pass') is True else 'numeric_unresolved')
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'attempted_update':attempted}
        status='failed_state_preserved'
        last=checkpoint(output,'failed_state',model,opt,rng,coverage,completed,locked,plan_sha,
                        valid=False,attempted=attempted)
    result={'loop':'0106','status':status,'failure':failure,'completed_updates':completed,
            'requested_updates':plan['updates'],'terminal_monitor':monitor,
            'last_checkpoint':last,'baseline':baseline,'plan_sha256':plan_sha,
            'source_snapshot':locked,'source_unchanged':snapshot()==locked,
            'seconds':time.perf_counter()-PROCESS_STARTED,'cap_seconds':700.,
            'within_cap':time.perf_counter()<=deadline,'test_read':False,'validation_read':False,
            'checkpoint_selection':'fixed terminal only'}
    if not result['source_unchanged'] or digest(plan_path)!=plan_sha:
        result['status']='quarantined_identity_drift'
    if not result['within_cap']:result['status']='budget_exception'
    write_json(output/'evidence.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--time-no-update',action='store_true')
    mode.add_argument('--make-plan',type=Path)
    mode.add_argument('--plan',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cap-seconds',type=float,default=120.)
    a=p.parse_args()
    if a.time_no_update:result=timing(a.output,a.cap_seconds)
    elif a.make_plan:
        result=make_plan(a.make_plan)
        if a.output.exists():raise FileExistsError(a.output)
        write_json(a.output,result)
    else:result=run_plan(a.plan,a.output)
    print(json.dumps({k:result.get(k) for k in ('status','complete','seconds','failure','updates')},allow_nan=False),flush=True)
    if result.get('complete') is False or result.get('failure') or result.get('status') in ('budget_exception','quarantined_identity_drift'):
        raise SystemExit(1)


if __name__=='__main__':main()
