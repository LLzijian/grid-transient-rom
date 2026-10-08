"""Registered 0103 S-C/S-R continuation. Defaults never update parameters."""
from __future__ import annotations

import time
PROCESS_STARTED = time.perf_counter()
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
import train_prediction_0102 as prior
from shared_closure_rom_0101 import load_model, make_pair, mapped_coefficients
from train_local_memory_0098 import sources, process_memory_bytes, horizon
from train_shared_closure_0101 import buffer_hash, validate_optimizer
from train_joint_numerical_pair_0100 import digest, write_json, parameter_layout, next_samples, MEMORY_CAP
from monitor_joint_adequacy_0099 import build_manifest, run_monitor, _model_hash

SOURCE = ROOT/'private_data/task2_prediction_optimization_0102/formal_S_O/step00650'
CARD = ROOT/'reports/task2_cross_dissipation_0103.md'
MODEL_CODE = ROOT/'experiments/shared_closure_rom_0101.py'
OBJECTIVE_CODE = ROOT/'experiments/train_local_memory_0098.py'
START_STEP = 2817
RATES = {'skeleton': .01, 'closure': .001, 'encoder': .0003}
EXPECTED = {
    'checkpoint.json': 'f7eea36b43855bd00ed370e887dde0c996f8af12edadd23def0bed710a6444d7',
    'model.pt': '96ac9fad15049d60f84b238c3158c71db7ad341e81cd402aeb073e6f9be6e639',
    'optimizer.pt': 'f509f9f473d789f20783011f2d4a5de011af08b4071ad94c9dadc38470100b5b',
    'rng.pt': 'cc015ccac0457a154e93d6df65e96847779b1b17b6a72301c80f028a65af78d3',
    'structured_numeric_export.json': 'c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238',
    'artifact_hashes.json': '832b8c2efb26c9e1da86ea1e6b988c77c1fd99398d3829f3e079a66e39449a69',
}


def guard(deadline, reserve=15.):
    if time.perf_counter()+reserve >= deadline:
        raise TimeoutError('process deadline/serialization reserve reached')
    if process_memory_bytes() > MEMORY_CAP:
        raise MemoryError('8 GiB process cap')


def verify_source(source=SOURCE):
    source = Path(source)
    if source.resolve() != SOURCE.resolve():
        raise ValueError('only registered 0102 formal_S_O/step00650 source allowed')
    if {k:digest(source/k) for k in EXPECTED} != EXPECTED:
        raise ValueError('0103 source artifact hash mismatch')
    meta = json.loads((source/'checkpoint.json').read_text(encoding='utf-8'))
    if (meta.get('format') != 'prediction_optimization_0102' or meta.get('arm') != 'S-O'
            or meta.get('step') != START_STEP or meta.get('completed_updates') != 650
            or meta.get('valid_training_state') is not True or meta.get('dimension') != 40):
        raise ValueError('invalid 0103 common source metadata')
    return meta


def activate(model, arm):
    if arm not in ('S-C','S-R'):
        raise ValueError('unregistered 0103 arm')
    if tuple(int(v) for v in model.template_mask.tolist()) != (1,1,1,1,1,1,0,1):
        raise ValueError('unexpected source S23 template mask')
    if model._template_scalars[6].requires_grad:
        raise ValueError('source cross-dissipation scalar already trainable')
    if arm == 'S-R':
        with torch.no_grad():model.template_mask[6] = 1
        model._template_scalars[6].requires_grad_(True)
    return model


def restore_arm(arm, source=SOURCE):
    meta = verify_source(source)
    source = Path(source)
    model = load_model(source/'structured_numeric_export.json')
    raw = torch.load(source/'model.pt', map_location='cpu', weights_only=True)
    if set(raw) != set(model.state_dict()) or any(not torch.equal(raw[k],v) for k,v in model.state_dict().items()):
        raise ValueError('source export/raw model mismatch')
    if (_model_hash(model) != meta['model_state_sha256']
            or buffer_hash(model) != meta['fixed_buffers_sha256']
            or {n:p.requires_grad for n,p in model.named_parameters()} != meta['requires_grad']
            or {n:p.numel() for n,p in model.named_parameters()} != meta['parameter_numel']):
        raise ValueError('source S23 parameter/buffer identity mismatch')
    coverage = copy.deepcopy(meta['coverage'])
    prior.validate_coverage(coverage,346)
    state = torch.load(source/'rng.pt', map_location='cpu', weights_only=False)
    if state.get('coverage') != coverage or 'numpy_global' not in state:
        raise ValueError('source RNG/coverage mismatch')
    rng = np.random.default_rng()
    prior.restore_rng(state,rng)
    activate(model,arm)
    if model.dim != 40 or any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('invalid restored model')
    return model,rng,coverage,meta


def parameter_block(name):
    if name.startswith(('local_gru.','local_head.')):return 'encoder'
    if name.startswith('structured.') or name in ('_q_quadratic_raw','_template_scalars.5'):return 'skeleton'
    if name == 'port_raw' or name.startswith('_template_scalars.') and name.split('.')[-1] in set('012346'):
        return 'closure'
    raise ValueError('unregistered parameter name: '+name)


def optimizer_for(model,rates):
    if rates != RATES:raise ValueError('registered 0103 block rates required')
    blocks={}
    for name,p in model.named_parameters():
        if p.requires_grad:blocks.setdefault(parameter_block(name),[]).append(p)
    optimizer=torch.optim.Adam([{'params':ps,'lr':rates[b],'block':b} for b,ps in blocks.items()])
    validate_optimizer(model,optimizer,parameter_layout(model,optimizer),0)
    return optimizer


def snapshot():
    paths={'runner0103':Path(__file__),'card0103':CARD,'model0101':MODEL_CODE,
           'objective0098':OBJECTIVE_CODE,'runner0102':ROOT/'experiments/train_prediction_0102.py',
           'monitor0099':ROOT/'experiments/monitor_joint_adequacy_0099.py',
           'runner0101':ROOT/'experiments/train_shared_closure_0101.py',
           'runner0100':ROOT/'experiments/train_joint_numerical_pair_0100.py'}
    paths.update({'source_'+k:SOURCE/k for k in EXPECTED})
    return dict(sources(),**{k:{'path':str(v),'sha256':digest(v)} for k,v in paths.items()})


def compute_gradients(model,samples,h,scale,optimizer,clip_norm=100.):
    if clip_norm != 100.:raise ValueError('registered clip norm required')
    tic=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    with prior.graph(model):
        loss,terms=prior.objective(model,samples,prior.histories_for(samples),h,scale)
        if not torch.isfinite(loss) or any(not torch.isfinite(v).all() for v in terms.values()):
            raise FloatingPointError('nonfinite objective/component')
        loss.backward()
    blocks={};params=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if p.grad is None or not torch.isfinite(p.grad).all():raise FloatingPointError('missing/nonfinite gradient: '+name)
        params.append(p); b=parameter_block(name)
        blocks[b]=blocks.get(b,0.)+float(p.grad.detach().square().sum())
    norm=float(torch.nn.utils.clip_grad_norm_(params,clip_norm))
    if not np.isfinite(norm):raise FloatingPointError('nonfinite gradient norm')
    return {'loss':float(loss.detach()),'terms':{k:prior.summarize_term(v) for k,v in terms.items()},
            'gradient_norm':norm,'gradient_clipped':norm>clip_norm,
            'gradient_block_l2':{k:float(np.sqrt(v)) for k,v in blocks.items()},
            'cross_raw_gradient':(float(model._template_scalars[6].grad) if model._template_scalars[6].grad is not None else None),
            'seconds':time.perf_counter()-tic}


def apply_update(model,optimizer,layout,completed,fixed):
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite post-update parameter')
    if buffer_hash(model)!=fixed:raise RuntimeError('fixed graph/support/unit buffers changed')
    validate_optimizer(model,optimizer,layout,completed)
    prior.check_structure(model)
    blocks={}
    for name,p in model.named_parameters():
        if name in before:
            b=parameter_block(name);blocks[b]=blocks.get(b,0.)+float((p.detach()-before[name]).square().sum())
    return {'update_block_l2':{k:float(np.sqrt(v)) for k,v in blocks.items()},
            'cross_raw_value':float(model._template_scalars[6].detach())}


def train_inputs():
    events,scale=prior.train_inputs()
    return events,scale


def identity_qa(output,*,cap_seconds):
    """Source-state structural and homogeneous-expansion checks, without TRAIN or updates."""
    if not 0<cap_seconds<=700:raise ValueError('identity QA cap must be in (0,700]')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+cap_seconds;torch.set_num_threads(1);guard(deadline)
    before=snapshot();control,rng,coverage,_=restore_arm('S-C')
    initial=prior.capture_rng(rng);control_hash=_model_hash(control)
    opened,other_rng,other_coverage,_=restore_arm('S-R')
    open_hash=_model_hash(opened);failure=None;checks={};metrics={}
    try:
        if coverage!=other_coverage or prior.rng_digest(initial)!=prior.rng_digest(prior.capture_rng(other_rng)):
            raise ValueError('arm RNG/coverage source mismatch')
        if (set(control.state_dict())!=set(opened.state_dict()) or
                any(not torch.equal(v,opened.state_dict()[k]) for k,v in control.state_dict().items() if k!='template_mask')):
            raise ValueError('S-R changed source tensors beyond template mask')
        if not torch.equal(control._template_scalars[6],opened._template_scalars[6]):
            raise ValueError('cross raw initial value was changed')
        raw=opened._template_scalars[6].detach();rho=float(torch.nn.functional.softplus(raw))
        metrics['cross_raw6']=float(raw);metrics['cross_rho']=rho
        metrics['control_mask']=control.template_mask.detach().tolist()
        metrics['opened_mask']=opened.template_mask.detach().tolist()
        checks['positive_rho']=np.isfinite(rho) and rho>0
        checks['mask_only_change']=metrics['control_mask']==[1.,1.,1.,1.,1.,1.,0.,1.] and metrics['opened_mask']==[1.]*8
        checks['trainable_only_added']=({n for n,p in opened.named_parameters() if p.requires_grad}
                                         - {n for n,p in control.named_parameters() if p.requires_grad})=={'_template_scalars.6'}
        for name,model in (('control',control),('opened',opened)):
            opt=optimizer_for(model,RATES)
            metrics[name+'_parameter_layout']=parameter_layout(model,opt)
            checks[name+'_fresh_adam']=not opt.state
        generator=torch.Generator().manual_seed(103)
        states=torch.randn((4,40),generator=generator,dtype=torch.float64)*.03
        states[0].zero_()
        metrics['state_shape']=list(states.shape)
        with torch.no_grad():
            fc=control.field(states);fr=opened.field(states)
            difference=fr-fc
            metrics['initial_rhs_delta_max_abs']=float(difference.abs().max())
            metrics['initial_rhs_delta_rms']=float(difference.square().mean().sqrt())
            checks['rhs_disturbance_measured']=bool(torch.isfinite(difference).all()) and metrics['initial_rhs_delta_max_abs']>0
            # Energy rate from the added R term is -rho/2 sum_i (dH/dpi_i+dH/dq_i)^2.
            internal=opened.interface_to_state(states)
            grad=opened.gradient(internal,create_graph=False)
            paired=(grad[:,10:20]+grad[:,30:40])
            predicted=-.5*rho*paired.square().sum(-1)
            actual=(grad*(opened.interface_to_state(fr)-opened.interface_to_state(fc))).sum(-1)
            metrics['added_energy_rate_max_abs']=float(actual.abs().max())
            metrics['power_identity_max_abs']=float((actual-predicted).abs().max())
            checks['power_identity']=bool(torch.isfinite(actual).all()) and metrics['power_identity_max_abs']<=1e-9
        guard(deadline)
        _,expanded=make_pair(opened)
        with torch.no_grad():
            ff=expanded.field(states)
            metrics['expanded_rhs_max_abs']=float((ff-fr).abs().max())
            metrics['effective_c23_complexity']=float(mapped_coefficients(opened).abs().sum())
            metrics['reported_complexity']=float(opened.complexity())
        checks['effective_coordinates']=abs(metrics['effective_c23_complexity']-metrics['reported_complexity'])<=1e-10
        checks['expanded_rhs']=metrics['expanded_rhs_max_abs']<=2e-10
        z=states[1].detach().requires_grad_(True)
        js=torch.autograd.functional.jacobian(opened.field,z)
        jf=torch.autograd.functional.jacobian(expanded.field,z)
        metrics['expanded_jacobian_max_abs']=float((js-jf).abs().max())
        checks['expanded_jacobian']=js.shape==jf.shape==(40,40) and metrics['expanded_jacobian_max_abs']<=3e-9
        for name,model in (('control',control),('opened',opened)):
            path=output/f'{name}_export.json';model.export(path)
            loaded=load_model(path)
            if name=='opened':loaded._template_scalars[6].requires_grad_(True)
            with torch.no_grad():reload_error=float((loaded.field(states)-model.field(states)).abs().max())
            metrics[name+'_reload_rhs_max_abs']=reload_error
            checks[name+'_reload']=(_model_hash(loaded)==_model_hash(model) and reload_error<=1e-12)
    except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc)}
    finally:
        prior.restore_rng(initial,rng)
        prior.restore_rng(initial,other_rng)
    after=snapshot()
    checks.update(control_unchanged=_model_hash(control)==control_hash,
                  opened_unchanged=_model_hash(opened)==open_hash,
                  rng_unchanged=prior.rng_digest(prior.capture_rng(rng))==prior.rng_digest(initial),
                  source_unchanged=before==after,within_cap=time.perf_counter()<=deadline)
    result={'loop':'0103','complete':failure is None and bool(checks) and all(checks.values()),
            'failure':failure,'checks':checks,'metrics':metrics,'source_before':before,'source_after':after,
            'parameter_updates':False,'train_read':False,'validation_read':False,'test_read':False,
            'seconds':time.perf_counter()-PROCESS_STARTED,'cap_seconds':cap_seconds}
    write_json(output/'evidence.json',result)
    return result


def save_checkpoint(output,name,model,optimizer,rng,coverage,arm,updates,source_snapshot,plan_sha,
                    *,valid=True,attempted=None):
    directory=Path(output)/name;directory.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),directory/'model.pt')
    torch.save(optimizer.state_dict(),directory/'optimizer.pt')
    torch.save(dict(prior.capture_rng(rng),coverage=copy.deepcopy(coverage)),directory/'rng.pt')
    meta={'format':'cross_dissipation_0103','arm':arm,'source_step':START_STEP,
          'source_model_sha256':EXPECTED['model.pt'],'optimizer_reset':True,
          'valid_training_state':valid,'completed_updates':updates if valid else None,
          'last_verified_updates':updates,'attempted_update':attempted,'step':START_STEP+updates if valid else None,
          'model_kind':'S23','dimension':model.dim,'training_dt':.01,'config':model.config(),
          'coverage':copy.deepcopy(coverage),'requires_grad':{n:p.requires_grad for n,p in model.named_parameters()},
          'parameter_numel':{n:p.numel() for n,p in model.named_parameters()},
          'optimizer_parameter_names':parameter_layout(model,optimizer),
          'model_state_sha256':_model_hash(model),'fixed_buffers_sha256':buffer_hash(model),
          'source_snapshot':source_snapshot,'plan_sha256':plan_sha,
          'process_seconds':time.perf_counter()-PROCESS_STARTED,
          'test_read':False,'validation_read':False}
    write_json(directory/'checkpoint.json',meta if valid else prior.safe_json(meta))
    if valid:model.export(directory/'structured_numeric_export.json')
    write_json(directory/'artifact_hashes.json',{p.name:digest(p) for p in directory.iterdir() if p.is_file()})
    return str(directory)


def restore_checkpoint(directory,arm,rates=RATES):
    directory=Path(directory);hashes=json.loads((directory/'artifact_hashes.json').read_text())
    required={'checkpoint.json','model.pt','optimizer.pt','rng.pt','structured_numeric_export.json'}
    if not required<=set(hashes) or any(Path(k).name!=k or digest(directory/k)!=v for k,v in hashes.items()):
        raise ValueError('incomplete/corrupt 0103 checkpoint')
    meta=json.loads((directory/'checkpoint.json').read_text())
    if (meta.get('format')!='cross_dissipation_0103' or meta.get('valid_training_state') is not True
            or meta.get('arm')!=arm or meta.get('source_model_sha256')!=EXPECTED['model.pt']
            or meta.get('source_snapshot')!=snapshot()):
        raise ValueError('checkpoint source/code/arm identity mismatch')
    model=load_model(directory/'structured_numeric_export.json')
    if arm=='S-R':
        if tuple(int(v) for v in model.template_mask.tolist()) != (1,1,1,1,1,1,1,1):
            raise ValueError('S-R export lacks active cross-dissipation group')
        model._template_scalars[6].requires_grad_(True)
    elif tuple(int(v) for v in model.template_mask.tolist()) != (1,1,1,1,1,1,0,1):
        raise ValueError('S-C export changed template mask')
    raw=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    if (set(raw)!=set(model.state_dict()) or any(not torch.equal(raw[k],v) for k,v in model.state_dict().items())
            or _model_hash(model)!=meta['model_state_sha256'] or buffer_hash(model)!=meta['fixed_buffers_sha256']
            or {n:p.requires_grad for n,p in model.named_parameters()}!=meta['requires_grad']):
        raise ValueError('checkpoint model/export mismatch')
    optimizer=optimizer_for(model,rates)
    if parameter_layout(model,optimizer)!=meta['optimizer_parameter_names']:
        raise ValueError('checkpoint Adam parameter layout mismatch')
    optimizer.load_state_dict(torch.load(directory/'optimizer.pt',map_location='cpu',weights_only=False))
    validate_optimizer(model,optimizer,meta['optimizer_parameter_names'],meta['completed_updates'])
    state=torch.load(directory/'rng.pt',map_location='cpu',weights_only=False)
    if state['coverage']!=meta['coverage']:raise ValueError('checkpoint RNG/coverage mismatch')
    prior.validate_coverage(state['coverage'],346)
    rng=np.random.default_rng();prior.restore_rng(state,rng)
    return model,optimizer,rng,copy.deepcopy(state['coverage']),meta


def validate_plan(plan):
    keys={'loop','registered','arm','updates','horizons','monitor_steps','rates','clip_norm',
          'cap_seconds','monitor_seconds','cost_evidence','cost_evidence_sha256','card_sha256',
          'source_model_sha256','snapshot','snapshot_sha256','allow_refine'}
    if set(plan)!=keys:raise ValueError('0103 plan fields incomplete or extra')
    n=plan['updates']
    if (plan['loop']!='0103' or plan['registered'] is not True or plan['arm'] not in ('S-C','S-R')
            or plan['card_sha256']!=digest(CARD) or plan['source_model_sha256']!=EXPECTED['model.pt']
            or plan['snapshot']!=snapshot() or plan['snapshot_sha256']!=prior._json_digest(plan['snapshot'])):
        raise ValueError('unregistered plan/source/code/card identity')
    if (type(n) is not int or n not in (600,840) or plan['monitor_steps']!=[0,n//2,n]
            or plan['horizons']!=[horizon(START_STEP+i) for i in range(1,n+1)]):
        raise ValueError('unregistered 0103 count/monitor/absolute horizon sequence')
    registered_cap = {'S-C':4700., 'S-R':4700.}[plan['arm']]
    if (plan['rates']!=RATES or plan['clip_norm']!=100. or plan['cap_seconds']!=registered_cap
            or type(plan['allow_refine']) is not bool or not np.isfinite(plan['monitor_seconds'])
            or plan['monitor_seconds']<=0):
        raise ValueError('unregistered 0103 optimization/cap/monitor policy')
    if not Path(plan['cost_evidence']).is_file() or digest(plan['cost_evidence'])!=plan['cost_evidence_sha256']:
        raise ValueError('missing/drifted no-update cost evidence')
    return plan


def time_no_update(arm,output,*,batches,cap_seconds):
    if arm not in ('S-C','S-R') or batches<3 or batches%3 or not 0<cap_seconds<=700:
        raise ValueError('timing requires arm, complete 1/2/5 cycles, QA cap <=700')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+cap_seconds;torch.set_num_threads(1);guard(deadline)
    before=snapshot();model,rng,coverage,meta=restore_arm(arm)
    optimizer=optimizer_for(model,RATES);events,scale=train_inputs()
    initial=prior.capture_rng(rng);initial_coverage=copy.deepcopy(coverage);initial_hash=_model_hash(model)
    rows=[];failure=None;monitor_row=None
    try:
        for i in range(batches):
            guard(deadline,15.+1.5*max([r['seconds'] for r in rows],default=0.))
            samples,trace=next_samples(events,rng,coverage);h=(1.,2.,5.)[i%3]
            trace.update(horizon=h,timing_batch=i+1)
            trace.update(compute_gradients(model,samples,h,scale,optimizer))
            rows.append(trace);write_json(output/'partial.json',{'complete':False,'batches':rows})
        guard(deadline,30.)
        monitor_state=prior.capture_rng(rng);monitor_hash=_model_hash(model)
        manifest=build_manifest(events)
        try:
            monitor_row=run_monitor(model,events,manifest,output/'monitor_timing',deadline-15,
                                    allow_refine=True,force_scale=scale)
            if _model_hash(model)!=monitor_hash or prior.rng_digest(prior.capture_rng(rng))!=prior.rng_digest(monitor_state):
                raise RuntimeError('timing monitor altered model or RNG')
        finally:prior.restore_rng(monitor_state,rng)
    except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc)}
    finally:optimizer.zero_grad(set_to_none=True);prior.restore_rng(initial,rng);coverage=initial_coverage
    after=snapshot()
    unchanged=(_model_hash(model)==initial_hash and not optimizer.state
               and prior.rng_digest(prior.capture_rng(rng))==prior.rng_digest(initial))
    result={'loop':'0103','arm':arm,'complete':failure is None and len(rows)==batches and unchanged and before==after
            and monitor_row is not None and monitor_row.get('complete') is True,
            'failure':failure,'parameter_updates':False,'checkpoint_step':START_STEP,'batches':rows,
            'step_seconds':{str(h):float(np.mean([r['seconds'] for r in rows if r['horizon']==h])) for h in (1,2,5)
                            if any(r['horizon']==h for r in rows)},
            'monitor_seconds':monitor_row.get('seconds') if monitor_row else None,
            'monitor_complete':monitor_row.get('complete') if monitor_row else False,
            'source_model_sha256':EXPECTED['model.pt'],'model_code_sha256':digest(MODEL_CODE),
            'objective_code_sha256':digest(OBJECTIVE_CODE),'runner_code_sha256':digest(__file__),
            'model_state_sha256_before':initial_hash,'model_state_sha256_after':_model_hash(model),
            'rng_and_model_unchanged':unchanged,'source_before':before,'source_after':after,
            'sources_unchanged':before==after,'rates':RATES,'clip_norm':100.,
            'cross_raw_gradient_seen':any(r.get('cross_raw_gradient') is not None for r in rows),
            'test_read':False,'validation_read':False,'seconds':time.perf_counter()-PROCESS_STARTED,
            'cap_seconds':cap_seconds}
    result['within_cap']=time.perf_counter()<=deadline;result['complete'] &= result['within_cap']
    write_json(output/'costs.json',result)
    return result


def run_plan(plan_path,output):
    plan_path=Path(plan_path);plan=validate_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    plan_sha=digest(plan_path);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+plan['cap_seconds'];torch.set_num_threads(1);guard(deadline)
    before=snapshot();model,rng,coverage,_=restore_arm(plan['arm'])
    optimizer=optimizer_for(model,RATES);layout=parameter_layout(model,optimizer);fixed=buffer_hash(model)
    costs=json.loads(Path(plan['cost_evidence']).read_text(encoding='utf-8'))
    if (costs.get('complete') is not True or costs.get('parameter_updates') is not False
            or costs.get('arm')!=plan['arm'] or costs.get('checkpoint_step')!=START_STEP
            or costs.get('source_model_sha256')!=EXPECTED['model.pt']
            or costs.get('model_code_sha256')!=digest(MODEL_CODE)
            or costs.get('objective_code_sha256')!=digest(OBJECTIVE_CODE)
            or costs.get('runner_code_sha256')!=digest(__file__)
            or costs.get('rates')!=RATES or costs.get('clip_norm')!=100.
            or costs.get('model_state_sha256_before')!=_model_hash(model)
            or costs.get('model_state_sha256_after')!=_model_hash(model)
            or costs.get('rng_and_model_unchanged') is not True
            or costs.get('monitor_complete') is not True
            or not isinstance(costs.get('monitor_seconds'),(int,float))
            or plan['monitor_seconds']<costs['monitor_seconds']
            # The card records the completed timing and frozen plan after timing.
            # All other source/code/data entries must remain byte-identical.
            or {k:v for k,v in costs.get('source_before',{}).items() if k!='card0103'}
               != {k:v for k,v in before.items() if k!='card0103'}
            or costs.get('source_after')!=costs.get('source_before')):
        raise ValueError('plan lacks matching no-update timing evidence')
    if plan['arm']=='S-R' and costs.get('cross_raw_gradient_seen') is not True:
        raise ValueError('cross-dissipation gradient was not available in timing')
    values=costs.get('step_seconds',{})
    if any(not np.isfinite(values.get(str(h),float('nan'))) or values[str(h)]<=0 for h in (1,2,5)):
        raise ValueError('invalid per-horizon timing')
    estimate=1.2*(sum(values[str(h)] for h in plan['horizons'])+3*plan['monitor_seconds'])
    if time.perf_counter()+estimate+max(100.,1.5*plan['monitor_seconds']+30.)>=deadline:
        raise TimeoutError('0103 complete plan cannot fit per-arm process cap with 20% margin')
    events,scale=train_inputs();manifest=build_manifest(events)
    write_json(output/'plan.json',plan);write_json(output/'manifest.json',manifest)
    completed=0;attempted=None;failure=None;monitors={};recent=[];last_monitor=plan['monitor_seconds']
    last_valid=None;last_checkpoint=None;status=None
    try:
        for i in range(plan['updates']+1):
            if i in plan['monitor_steps']:
                guard(deadline,max(100.,1.5*last_monitor+30.))
                last_valid=save_checkpoint(output,f'step{i:05d}',model,optimizer,rng,coverage,plan['arm'],completed,before,plan_sha)
                state=prior.capture_rng(rng);state_hash=_model_hash(model)
                try:
                    row=run_monitor(model,events,manifest,output/f'monitor_step{i:05d}',deadline-15,
                                    allow_refine=plan['allow_refine'],force_scale=scale)
                    if _model_hash(model)!=state_hash or prior.rng_digest(prior.capture_rng(rng))!=prior.rng_digest(state):
                        raise RuntimeError('monitor changed model or RNG')
                finally:prior.restore_rng(state,rng)
                monitors[str(i)]=row
                if not row.get('complete'):status='partial_monitor';break
                last_monitor=row['seconds']
            if i==plan['updates']:status='nominal_update_cap';break
            guard(deadline,max(100.,1.5*last_monitor+30.)+1.5*max(recent[-10:],default=0.))
            if snapshot()!=before or digest(plan_path)!=plan_sha:raise RuntimeError('source/code/card/plan drift')
            attempted=i+1;tic=time.perf_counter();samples,trace=next_samples(events,rng,coverage)
            trace.update(update=attempted,horizon=plan['horizons'][i],
                         torch_rng_before_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            with (output/'samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(trace,allow_nan=False)+'\n')
            trace.update(compute_gradients(model,samples,plan['horizons'][i],scale,optimizer))
            trace.update(apply_update(model,optimizer,layout,attempted,fixed));completed=attempted
            trace.update(seconds=time.perf_counter()-tic,
                         torch_rng_after_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            recent.append(trace['seconds'])
            with (output/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(trace,allow_nan=False)+'\n')
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'attempted_update':attempted}
        status='failed_state_preserved'
        last_checkpoint=save_checkpoint(output,'failed_state',model,optimizer,rng,coverage,plan['arm'],completed,before,plan_sha,
                                        valid=False,attempted=attempted)
    if failure is None:
        last_checkpoint=save_checkpoint(output,'last_state',model,optimizer,rng,coverage,plan['arm'],completed,before,plan_sha)
    after=snapshot()
    result={'loop':'0103','arm':plan['arm'],'status':status,'failure':failure,'completed_updates':completed,
            'requested_updates':plan['updates'],'monitors':monitors,'last_valid_milestone':last_valid,
            'last_checkpoint':last_checkpoint,'plan_sha256':plan_sha,'source_before':before,'source_after':after,
            'sources_unchanged':before==after,'seconds':time.perf_counter()-PROCESS_STARTED,
            'cap_seconds':plan['cap_seconds'],'within_cap':time.perf_counter()<=deadline,
            'test_read':False,'validation_read':False,'checkpoint_selection':None}
    if before!=after or digest(plan_path)!=plan_sha:result['status']='quarantined_source_or_plan_drift'
    if not result['within_cap']:result['status']='budget_exception'
    write_json(output/'evidence.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--restore-only',action='store_true')
    mode.add_argument('--identity-qa',action='store_true')
    mode.add_argument('--time-no-update',action='store_true')
    mode.add_argument('--plan',type=Path)
    p.add_argument('--arm',choices=('S-C','S-R'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int)
    p.add_argument('--cap-seconds',type=float)
    a=p.parse_args()
    if a.plan:result=run_plan(a.plan,a.output)
    elif a.identity_qa:
        if a.cap_seconds is None:p.error('identity QA requires cap')
        result=identity_qa(a.output,cap_seconds=a.cap_seconds)
    elif a.time_no_update:
        if a.arm is None or a.batches is None or a.cap_seconds is None:p.error('timing requires arm, batches and cap')
        result=time_no_update(a.arm,a.output,batches=a.batches,cap_seconds=a.cap_seconds)
    else:
        if a.arm is None or a.cap_seconds is None:p.error('restore requires arm and cap')
        deadline=PROCESS_STARTED+a.cap_seconds;torch.set_num_threads(1);guard(deadline)
        model,rng,coverage,meta=restore_arm(a.arm);opt=optimizer_for(model,RATES)
        result={'loop':'0103','arm':a.arm,'status':'restore_only_no_integration','parameter_updates':False,
                'source_step':START_STEP,'model_state_sha256':_model_hash(model),
                'coverage':coverage,'optimizer_parameter_names':parameter_layout(model,opt),
                'rng_sha256':prior.rng_digest(prior.capture_rng(rng)),
                'seconds':time.perf_counter()-PROCESS_STARTED,'test_read':False,'validation_read':False,'train_read':False}
        guard(deadline);a.output.mkdir(parents=True,exist_ok=False);write_json(a.output/'evidence.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','complete','seconds','failure')}),flush=True)
    if result.get('failure') or result.get('complete') is False or result.get('status') in ('budget_exception','quarantined_source_or_plan_drift'):
        raise SystemExit(1)


if __name__=='__main__':main()
