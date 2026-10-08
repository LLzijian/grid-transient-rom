"""Matched 0104 S-C/S-R Adam-rate adaptation; default mode never updates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_cross_dissipation_0103 as prior

ROOT = prior.ROOT
CARD = ROOT / 'reports/task2_lr_adaptation_0104.md'
START_STEP = 3117
UPDATES = 210
MONITORS = [0, 105, 210]
FALLBACK_UPDATES = 180
RATES = {'skeleton': .002, 'closure': .0002, 'encoder': .00006}
SOURCE = {a: ROOT / f'private_data/task2_cross_dissipation_0103/formal_{a.replace("-", "_")}/step00300'
          for a in ('S-C', 'S-R')}
EXPECTED = {
    'S-C': {'artifact_hashes.json': '411b6081e54013fc230ec6d6f4e01a98761f3c6a9fd17c3be101789620e80313',
            'checkpoint.json': '21db5977083dbbe24851e66e10aa18c08c14accd314150869195e0cac7359c47',
            'model.pt': 'ff393c04677edbe9a33c8a653703bbe8fc53958e2cc834e304845a41f80cbda3',
            'optimizer.pt': '3ff6fb34f36eb91d87042780f6ab81294889f52b2a2a67817476ec7ba3847497',
            'rng.pt': 'ddb7ced71630ab0e29db8dfc8e9817155387536df593e0b30a3d0ae8b825de06',
            'structured_numeric_export.json': '48860da31c5aa5d55aefaf1715f69c9fa3aa234c71a49a860f0e9037f17515c3'},
    'S-R': {'artifact_hashes.json': '347c233949de98cd087b367e80fc956862c504c70f1c354870d664a7f0bc42f2',
            'checkpoint.json': '47221ac42c255f6e6c463cda80f7d4b3e84f7a8e9500154202c79611b157a67e',
            'model.pt': '31f91d9f1c9f7ac916ef86dfcc3343301fdaa08977e2252c68322d53d9d898f6',
            'optimizer.pt': 'adc8a74b1de172ccfe2637434022bd172725debfa0947f3e7a28162057ddee63',
            'rng.pt': 'ddb7ced71630ab0e29db8dfc8e9817155387536df593e0b30a3d0ae8b825de06',
            'structured_numeric_export.json': '5daf2bed279ea56d01d3086ad50adb411209f3b1a746159cedd9f7423ecd2862'},
}


def source_snapshot():
    paths = {'runner0104': Path(__file__), 'runner0103': Path(prior.__file__),
             'card0104': CARD, 'model0101': prior.MODEL_CODE,
             'objective0098': prior.OBJECTIVE_CODE,
             'monitor0099': ROOT/'experiments/monitor_joint_adequacy_0099.py'}
    paths.update({f'source_{a}_{name}': directory/name for a, directory in SOURCE.items()
                  for name in EXPECTED[a]})
    return dict(prior.sources(), **{k: {'path': str(p), 'sha256': prior.digest(p)}
                                    for k, p in paths.items()})


def restore_source(arm):
    if arm not in SOURCE:
        raise ValueError('unregistered 0104 arm')
    directory = SOURCE[arm]
    if {name: prior.digest(directory/name) for name in EXPECTED[arm]} != EXPECTED[arm]:
        raise ValueError('0104 pinned source artifact mismatch')
    model, optimizer, rng, coverage, meta = prior.restore_checkpoint(directory, arm)
    if (meta.get('step') != START_STEP or meta.get('completed_updates') != 300
            or meta.get('optimizer_reset') is not True
            or meta.get('model_state_sha256') != prior._model_hash(model)
            or meta.get('dimension') != 40):
        raise ValueError('invalid 0104 source step/model metadata')
    if set(meta['requires_grad']) != {n for n, _ in model.named_parameters()}:
        raise ValueError('source parameter inventory changed')
    if not all(p.requires_grad for n, p in model.named_parameters()
               if n.startswith(('local_gru.', 'local_head.'))):
        raise ValueError('GRU/head must remain trainable')
    if {g['block']: g['lr'] for g in optimizer.param_groups} != prior.RATES:
        raise ValueError('unexpected source block rates')
    for group in optimizer.param_groups:
        group['lr'] = RATES[group['block']]
    layout = prior.parameter_layout(model, optimizer)
    prior.validate_optimizer(model, optimizer, layout, 300)
    if {g['block']: g['lr'] for g in optimizer.param_groups} != RATES:
        raise ValueError('rate adaptation incomplete')
    return model, optimizer, rng, coverage, meta


def optimizer_fingerprint(optimizer):
    """Stable digest of Adam moments/steps, including all parameter groups."""
    h = hashlib.sha256()
    state = optimizer.state_dict()
    for group in state['param_groups']:
        h.update(json.dumps({k:v for k,v in group.items() if k != 'params'},sort_keys=True).encode())
        h.update(json.dumps(group['params']).encode())
    for index, item in sorted(state['state'].items()):
        h.update(str(index).encode())
        for key, value in sorted(item.items()):
            h.update(key.encode())
            if torch.is_tensor(value):
                h.update(str((value.dtype, tuple(value.shape))).encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def validate_plan(plan):
    keys = {'loop','registered','arm','updates','horizons','monitor_steps','rates','clip_norm',
            'cap_seconds','monitor_seconds','cost_evidence','cost_evidence_sha256',
            'card_sha256','source_model_sha256','snapshot','snapshot_sha256','allow_refine'}
    if set(plan) != keys:
        raise ValueError('0104 plan fields incomplete or extra')
    arm = plan['arm']
    if (plan['loop'] != '0104' or plan['registered'] is not True or arm not in SOURCE
            or plan['card_sha256'] != prior.digest(CARD)
            or plan['source_model_sha256'] != EXPECTED[arm]['model.pt']
            or plan['snapshot'] != source_snapshot()
            or plan['snapshot_sha256'] != prior.prior._json_digest(plan['snapshot'])):
        raise ValueError('unregistered 0104 plan/source/code/card identity')
    n=plan['updates']
    if (type(n) is not int or n not in (UPDATES,FALLBACK_UPDATES)
            or plan['monitor_steps'] != [0,n//2,n]
            or plan['horizons'] != [prior.horizon(START_STEP+i) for i in range(1,n+1)]):
        raise ValueError('unregistered 0104 count/monitor/absolute horizons')
    if (plan['rates'] != RATES or plan['clip_norm'] != 100.
            or plan['cap_seconds'] != 2000. or plan['allow_refine'] is not True
            or not isinstance(plan['monitor_seconds'], (int,float))
            or not np.isfinite(plan['monitor_seconds']) or plan['monitor_seconds'] <= 0):
        raise ValueError('unregistered 0104 optimization/monitor/cap')
    if (not Path(plan['cost_evidence']).is_file()
            or prior.digest(plan['cost_evidence']) != plan['cost_evidence_sha256']):
        raise ValueError('missing/drifted 0104 no-update cost evidence')
    return plan


def time_no_update(arm, output, *, cap_seconds):
    if arm not in SOURCE or not 0 < cap_seconds <= 500:
        raise ValueError('timing requires registered arm and cap in (0,500]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    deadline = prior.PROCESS_STARTED + cap_seconds
    torch.set_num_threads(1)
    prior.guard(deadline)
    before = source_snapshot()
    model, optimizer, rng, coverage, _ = restore_source(arm)
    events, scale = prior.train_inputs()
    initial_rng = prior.prior.capture_rng(rng)
    initial_coverage = copy.deepcopy(coverage)
    model_hash = prior._model_hash(model)
    adam_hash = optimizer_fingerprint(optimizer)
    rows, monitor_row, failure = [], None, None
    try:
        for i, h in enumerate((1,2,5)):
            prior.guard(deadline, 30.)
            samples, trace = prior.next_samples(events, rng, coverage)
            trace.update(horizon=h, timing_batch=i+1)
            trace.update(prior.compute_gradients(model, samples, h, scale, optimizer))
            rows.append(trace)
            prior.write_json(output/'partial.json', {'complete':False, 'batches':rows})
        prior.guard(deadline, 30.)
        state = prior.prior.capture_rng(rng)
        state_hash = prior._model_hash(model)
        manifest = prior.build_manifest(events)
        try:
            monitor_row = prior.run_monitor(model, events, manifest, output/'monitor_timing',
                                            deadline-15., allow_refine=True, force_scale=scale)
            if (prior._model_hash(model) != state_hash
                    or prior.prior.rng_digest(prior.prior.capture_rng(rng)) != prior.prior.rng_digest(state)):
                raise RuntimeError('timing monitor altered model or RNG')
        finally:
            prior.prior.restore_rng(state, rng)
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        optimizer.zero_grad(set_to_none=True)
        prior.prior.restore_rng(initial_rng, rng)
        coverage = initial_coverage
    after = source_snapshot()
    unchanged = (prior._model_hash(model) == model_hash
                 and optimizer_fingerprint(optimizer) == adam_hash
                 and prior.prior.rng_digest(prior.prior.capture_rng(rng)) == prior.prior.rng_digest(initial_rng))
    result = {'loop':'0104','arm':arm,'complete':failure is None and len(rows)==3 and unchanged
              and before==after and monitor_row is not None and monitor_row.get('complete') is True
              and monitor_row.get('numerical_pass') is True
              and (arm!='S-R' or any(row.get('cross_raw_gradient') is not None for row in rows)),
              'failure':failure,'parameter_updates':False,'checkpoint_step':START_STEP,
              'batches':rows,'step_seconds':{str(row['horizon']):row['seconds'] for row in rows},
              'monitor_seconds':monitor_row.get('seconds') if monitor_row else None,
              'monitor_complete':monitor_row.get('complete') if monitor_row else False,
              'monitor_numerical_pass':monitor_row.get('numerical_pass') if monitor_row else False,
              'cross_raw_gradient_seen':any(row.get('cross_raw_gradient') is not None for row in rows),
              'source_model_sha256':EXPECTED[arm]['model.pt'],
              'model_state_sha256_before':model_hash,'model_state_sha256_after':prior._model_hash(model),
              'optimizer_sha256_before':adam_hash,'optimizer_sha256_after':optimizer_fingerprint(optimizer),
              'rng_and_model_unchanged':unchanged,'source_before':before,'source_after':after,
              'sources_unchanged':before==after,'rates':RATES,'clip_norm':100.,
              'test_read':False,'validation_read':False,
              'seconds':time.perf_counter()-prior.PROCESS_STARTED,'cap_seconds':cap_seconds}
    result['within_cap'] = time.perf_counter() <= deadline
    result['complete'] &= result['within_cap']
    prior.write_json(output/'costs.json', result)
    return result


def save_checkpoint(output, name, model, optimizer, rng, coverage, arm, updates, snapshot,
                    plan_sha, *, valid=True, attempted=None):
    directory = Path(output)/name
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), directory/'model.pt')
    torch.save(optimizer.state_dict(), directory/'optimizer.pt')
    torch.save(dict(prior.prior.capture_rng(rng), coverage=copy.deepcopy(coverage)),directory/'rng.pt')
    meta = {'format':'lr_adaptation_0104','arm':arm,'source_step':START_STEP,
            'source_model_sha256':EXPECTED[arm]['model.pt'],
            'valid_training_state':valid,'completed_updates':updates if valid else None,
            'last_verified_updates':updates,'attempted_update':attempted,
            'step':START_STEP+updates if valid else None,'dimension':model.dim,
            'coverage':copy.deepcopy(coverage),
            'requires_grad':{n:p.requires_grad for n,p in model.named_parameters()},
            'optimizer_parameter_names':prior.parameter_layout(model,optimizer),
            'model_state_sha256':prior._model_hash(model),'fixed_buffers_sha256':prior.buffer_hash(model),
            'source_snapshot':snapshot,'plan_sha256':plan_sha,
            'process_seconds':time.perf_counter()-prior.PROCESS_STARTED,
            'test_read':False,'validation_read':False}
    prior.write_json(directory/'checkpoint.json', meta if valid else prior.prior.safe_json(meta))
    if valid:
        model.export(directory/'structured_numeric_export.json')
    prior.write_json(directory/'artifact_hashes.json',
                     {p.name:prior.digest(p) for p in directory.iterdir() if p.is_file()})
    return str(directory)


def check_costs(plan, costs, before, model, optimizer):
    arm=plan['arm']
    without_card=lambda snap:{k:v for k,v in snap.items() if k!='card0104'}
    if (costs.get('complete') is not True or costs.get('parameter_updates') is not False
            or costs.get('arm') != arm or costs.get('checkpoint_step') != START_STEP
            or costs.get('source_model_sha256') != EXPECTED[arm]['model.pt']
            or costs.get('rates') != RATES or costs.get('clip_norm') != 100.
            or costs.get('model_state_sha256_before') != prior._model_hash(model)
            or costs.get('model_state_sha256_after') != prior._model_hash(model)
            or costs.get('optimizer_sha256_before') != optimizer_fingerprint(optimizer)
            or costs.get('optimizer_sha256_after') != optimizer_fingerprint(optimizer)
            or costs.get('rng_and_model_unchanged') is not True
            or costs.get('monitor_complete') is not True
            or costs.get('monitor_numerical_pass') is not True
            or arm=='S-R' and costs.get('cross_raw_gradient_seen') is not True
            or not isinstance(costs.get('monitor_seconds'),(int,float))
            or not np.isfinite(costs['monitor_seconds'])
            or plan['monitor_seconds'] < costs['monitor_seconds']
            or without_card(costs.get('source_before',{})) != without_card(before)
            or costs.get('source_after') != costs.get('source_before')):
        raise ValueError('plan lacks matching no-update timing evidence')
    values=costs.get('step_seconds',{})
    if any(not isinstance(values.get(str(h)),(int,float)) or not np.isfinite(values[str(h)])
           or values[str(h)]<=0 for h in (1,2,5)):
        raise ValueError('invalid per-horizon timing')
    estimate=1.2*(sum(values[str(int(h))] for h in plan['horizons'])+3*plan['monitor_seconds'])
    reserve=max(100.,1.5*plan['monitor_seconds']+30.)
    if time.perf_counter()+estimate+reserve >= prior.PROCESS_STARTED+plan['cap_seconds']:
        raise TimeoutError('0104 plan cannot fit 2000-second cap with 20% margin and monitor reserve')
    return estimate


def run_plan(plan_path, output):
    plan_path=Path(plan_path)
    plan=validate_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    plan_sha=prior.digest(plan_path)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    deadline=prior.PROCESS_STARTED+plan['cap_seconds']
    torch.set_num_threads(1)
    prior.guard(deadline)
    before=source_snapshot()
    model,optimizer,rng,coverage,_=restore_source(plan['arm'])
    layout=prior.parameter_layout(model,optimizer)
    fixed=prior.buffer_hash(model)
    costs=json.loads(Path(plan['cost_evidence']).read_text(encoding='utf-8'))
    estimate=check_costs(plan,costs,before,model,optimizer)
    events,scale=prior.train_inputs()
    manifest=prior.build_manifest(events)
    prior.write_json(output/'plan.json',plan)
    prior.write_json(output/'manifest.json',manifest)
    completed=0
    attempted=None
    failure=None
    monitors={}
    recent=[]
    last_monitor=plan['monitor_seconds']
    last_valid=None
    last_checkpoint=None
    status=None
    try:
        for i in range(plan['updates']+1):
            if i in plan['monitor_steps']:
                prior.guard(deadline,max(100.,1.5*last_monitor+30.))
                last_valid=save_checkpoint(output,f'step{i:05d}',model,optimizer,rng,coverage,
                                           plan['arm'],completed,before,plan_sha)
                state=prior.prior.capture_rng(rng)
                state_hash=prior._model_hash(model)
                adam_hash=optimizer_fingerprint(optimizer)
                try:
                    row=prior.run_monitor(model,events,manifest,output/f'monitor_step{i:05d}',
                                          deadline-15.,allow_refine=True,force_scale=scale)
                    if (prior._model_hash(model)!=state_hash or optimizer_fingerprint(optimizer)!=adam_hash
                            or prior.prior.rng_digest(prior.prior.capture_rng(rng))!=prior.prior.rng_digest(state)):
                        raise RuntimeError('monitor changed model, Adam or RNG')
                finally:
                    prior.prior.restore_rng(state,rng)
                monitors[str(i)]=row
                if not row.get('complete'):
                    status='partial_monitor'
                    break
                last_monitor=row['seconds']
            if i==plan['updates']:
                status='nominal_update_cap'
                break
            prior.guard(deadline,max(100.,1.5*last_monitor+30.)+1.5*max(recent[-10:],default=0.))
            if source_snapshot()!=before or prior.digest(plan_path)!=plan_sha:
                raise RuntimeError('source/code/card/plan drift')
            attempted=i+1
            tic=time.perf_counter()
            samples,trace=prior.next_samples(events,rng,coverage)
            trace.update(update=attempted,absolute_step=START_STEP+attempted,
                         horizon=plan['horizons'][i],
                         torch_rng_before_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            with (output/'samples.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(trace,allow_nan=False)+'\n')
            trace.update(prior.compute_gradients(model,samples,plan['horizons'][i],scale,optimizer))
            trace.update(prior.apply_update(model,optimizer,layout,300+attempted,fixed))
            completed=attempted
            trace.update(seconds=time.perf_counter()-tic,
                         torch_rng_after_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            recent.append(trace['seconds'])
            with (output/'trace.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(trace,allow_nan=False)+'\n')
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'attempted_update':attempted}
        status='failed_state_preserved'
        last_checkpoint=save_checkpoint(output,'failed_state',model,optimizer,rng,coverage,
                                        plan['arm'],completed,before,plan_sha,valid=False,attempted=attempted)
    if failure is None:
        last_checkpoint=save_checkpoint(output,'last_state',model,optimizer,rng,coverage,
                                        plan['arm'],completed,before,plan_sha)
    after=source_snapshot()
    result={'loop':'0104','arm':plan['arm'],'status':status,'failure':failure,
            'completed_updates':completed,'requested_updates':plan['updates'],'monitors':monitors,
            'last_valid_milestone':last_valid,'last_checkpoint':last_checkpoint,
            'plan_sha256':plan_sha,'source_before':before,'source_after':after,
            'sources_unchanged':before==after,'timing_estimate_with_margin_seconds':estimate,
            'seconds':time.perf_counter()-prior.PROCESS_STARTED,
            'cap_seconds':plan['cap_seconds'],'within_cap':time.perf_counter()<=deadline,
            'test_read':False,'validation_read':False,'checkpoint_selection':None}
    if before!=after or prior.digest(plan_path)!=plan_sha:
        result['status']='quarantined_source_or_plan_drift'
    if not result['within_cap']:
        result['status']='budget_exception'
    prior.write_json(output/'evidence.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--arm',choices=tuple(SOURCE))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cap-seconds',type=float,default=500.)
    args=parser.parse_args()
    if args.plan:
        if args.arm is not None:
            parser.error('--arm is for default no-update timing mode')
        result=run_plan(args.plan,args.output)
    else:
        if args.arm is None:
            parser.error('default no-update timing mode requires --arm')
        result=time_no_update(args.arm,args.output,cap_seconds=args.cap_seconds)
    print(json.dumps({k:result.get(k) for k in ('status','complete','seconds','failure')}),flush=True)
    if result.get('failure') or result.get('complete') is False or result.get('status') != 'nominal_update_cap' and args.plan:
        raise SystemExit(1)


if __name__=='__main__':
    main()
