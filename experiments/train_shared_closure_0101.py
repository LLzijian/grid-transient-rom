"""Bounded 0101 S23/F23 joint fit with a common C23 objective and fresh Adam."""
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
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'),str(ROOT/'src')]
from train_local_memory_0098 import (read_train, sources, objective, NORM,
    check_structure, graph, histories_for, horizon, process_memory_bytes)
from train_joint_numerical_pair_0100 import (SOURCE, START_STEP, EXPECTED, MEMORY_CAP,
    restore as restore_ancestor, next_samples, reserve_seconds, parameter_layout,
    digest, _json_digest, write_json)
from train_joint_adequacy_0099 import severe_transition
from monitor_joint_adequacy_0099 import build_manifest, run_monitor, _model_hash

ARMS = ('S23','F23')
MILESTONES = (200,500,1000)
MODEL_CODE = ROOT/'experiments/shared_closure_rom_0101.py'
OBJECTIVE_CODE = ROOT/'experiments/train_local_memory_0098.py'


def restore_model_or_pair(arm=None, source=SOURCE, count=346):
    """Return model(s), exact ancestor Generator/coverage, and ancestor metadata.

    The old optimizer is verified during source recovery then discarded. Mapping
    the two new parameterizations must not advance the restored training RNG.
    """
    if arm is not None and arm not in ARMS:
        raise ValueError('unknown 0101 arm')
    from shared_closure_rom_0101 import make_pair
    base, old_optimizer, rng, coverage, meta = restore_ancestor(source,count)
    del old_optimizer
    ts, ns = torch.get_rng_state().clone(), np.random.get_state()
    try:
        pair = make_pair(base)
    finally:
        torch.set_rng_state(ts)
        np.random.set_state(ns)
    if isinstance(pair,tuple):
        pair = dict(zip(ARMS,pair))
    if set(pair) != set(ARMS):
        raise ValueError('make_pair must provide exactly S23 and F23')
    for key, model in pair.items():
        if model.dim != 40 or any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError(f'invalid mapped {key} source')
    meta = dict(meta, optimizer_reset=True, source_step=START_STEP,
                continuation_scope='common warm-start state; fresh Adam and C23 objective, not exact 0099 optimizer continuation')
    return (pair if arm is None else pair[arm]),rng,coverage,meta


def optimizer_for(model):
    """Registered per-coordinate rates; unknown trainable names fail closed."""
    blocks = {}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(('local_gru.','local_head.')):
            rate = 3e-4
        elif name.startswith('structured.') or name in ('_q_quadratic_raw','_template_scalars.5'):
            rate = 1e-2
        elif name in ('port_raw','closure_coefficients') or name in {f'_template_scalars.{i}' for i in range(5)}:
            rate = 1e-3
        else:
            raise ValueError(f'unregistered trainable parameter: {name}')
        blocks.setdefault(rate,[]).append(p)
    if not blocks:
        raise ValueError('no allowed trainable parameters')
    return torch.optim.Adam([{'params':values,'lr':rate} for rate,values in blocks.items()])


def validate_optimizer(model, optimizer, layout, updates):
    if parameter_layout(model,optimizer) != layout:
        raise ValueError('Adam parameter identities/order/rates drifted')
    named = {id(p):n for n,p in model.named_parameters() if p.requires_grad}
    included = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(set(included)) != len(included) or set(included) != set(named):
        raise ValueError('every allowed parameter must be optimized exactly once')
    if updates == 0:
        if optimizer.state:
            raise ValueError('0101 requires fresh Adam state')
        return
    if len(optimizer.state) != len(included):
        raise ValueError('missing Adam parameter states')
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if int(state['step']) != updates:
                raise ValueError('fresh Adam update count mismatch')
            for key in ('exp_avg','exp_avg_sq'):
                if state[key].shape != p.shape or not torch.isfinite(state[key]).all():
                    raise FloatingPointError('invalid/nonfinite Adam moment')


def buffer_hash(model):
    h = hashlib.sha256()
    for name,value in model.named_buffers():
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def snapshot(source=SOURCE):
    paths = {'runner0101':Path(__file__),'model0101':MODEL_CODE,
             'card0101':ROOT/'reports/task2_shared_closure_FS_0101.md',
             'runner0100':ROOT/'experiments/train_joint_numerical_pair_0100.py',
             'runner0099':ROOT/'experiments/train_joint_adequacy_0099.py',
             'monitor0099':ROOT/'experiments/monitor_joint_adequacy_0099.py'}
    paths.update({f'ancestor_{name}':Path(source)/name for name in EXPECTED})
    return dict(sources(),**{k:{'path':str(p),'sha256':digest(p)} for k,p in paths.items()})


def read_cost_evidence(path,arm):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (value.get('arm') != arm or value.get('training_dt') != .01
            or value.get('checkpoint_step') != START_STEP or value.get('complete') is not True
            or value.get('parameter_updates') is not False
            or value.get('source_model_sha256') != EXPECTED['model.pt']
            or value.get('model_code_sha256') != digest(MODEL_CODE)
            or value.get('objective_code_sha256') != digest(OBJECTIVE_CODE)):
        raise ValueError('wrong no-update cost evidence identity')
    if any(not np.isfinite(value['step_seconds'][str(h)]) or value['step_seconds'][str(h)] <= 0 for h in (1,2,5)):
        raise ValueError('all three measured horizon costs are required')
    if not np.isfinite(value.get('monitor_seconds',float('nan'))) or value['monitor_seconds'] <= 0:
        raise ValueError('measured historical monitor cost is required')
    return value


def estimated_cost(new_steps, costs, monitor_seconds, *, include_initial=True):
    if new_steps not in (500,1000):
        raise ValueError('only registered 500/1000 plans may be estimated')
    train = sum(float(costs[str(int(horizon(START_STEP+i)))]) for i in range(1,new_steps+1))
    monitor_count = sum(n<=new_steps for n in MILESTONES)+int(include_initial)
    return train+monitor_count*monitor_seconds+reserve_seconds(monitor_seconds,list(costs.values()))


def cached_initial_monitor(directory, arm, model, manifest):
    """Accept only a completed new-objective 0101 QA monitor, never old C7 J."""
    directory=Path(directory)
    evidence_path=directory/'evidence.json'
    context_path=directory/'context_0101.json'
    row=json.loads(evidence_path.read_text(encoding='utf-8'))
    context=json.loads(context_path.read_text(encoding='utf-8'))
    current_hash=_model_hash(model)
    current_sources=sources()
    # C23 complexity is state independent. The recorded fine-grid batch term
    # provides an additional objective identity check, beyond the outer label.
    terms=(row.get('batch_terms') or {}).get('fine') or {}
    names=terms.get('term_names') or []
    c23_matches=False
    if 'complexity' in names:
        try:
            values=np.asarray(terms['terms'],dtype=float)
            with torch.no_grad():expected_c23=float(model.complexity().detach())
            c=values[:,:,names.index('complexity')]
            c23_matches=(values.shape==(173,3,len(names)) and np.isfinite(c).all()
                         and np.allclose(c,expected_c23,atol=1e-12,rtol=1e-12))
        except (ValueError,IndexError,KeyError,TypeError):
            c23_matches=False
    checks={
        'context_loop':context.get('loop')=='0101',
        'objective':context.get('objective')=='C23' and bool(c23_matches),
        'arm':context.get('arm')==arm,
        'model_code':context.get('model_code_sha256')==digest(MODEL_CODE),
        'source_model':context.get('source_model_sha256')==EXPECTED['model.pt'],
        'source_checkpoint':context.get('source_checkpoint_sha256')==EXPECTED['checkpoint.json'],
        'context_model':context.get('model_state_sha256')==current_hash,
        'evidence_hash':context.get('monitor_evidence_sha256')==digest(evidence_path),
        'complete':row.get('complete') is True,
        'model_before':row.get('model_state_sha256_before')==current_hash,
        'model_after':row.get('model_state_sha256_after')==current_hash,
        'manifest':row.get('manifest_sha256')==manifest['sha256'],
        'monitor_code':row.get('monitor_code_sha256')==digest(ROOT/'experiments/monitor_joint_adequacy_0099.py'),
        'source_before':row.get('source_before')==current_sources,
        'source_after':row.get('source_after')==current_sources,
        'sources_unchanged':row.get('sources_unchanged') is True,
        'rng_unchanged':row.get('rng_unchanged') is True,
        'positive_finite_seconds':isinstance(row.get('seconds'),(int,float)) and
            np.isfinite(row['seconds']) and row['seconds']>0}
    if not all(checks.values()):
        raise ValueError('0101 initial-monitor cache rejected: '+','.join(k for k,v in checks.items() if not v))
    return row,{'accepted':True,'checks':checks,'directory':str(directory),
        'context_path':str(context_path),'context_sha256':digest(context_path),
        'evidence_path':str(evidence_path),'evidence_sha256':digest(evidence_path),
        'cached_monitor_seconds':row['seconds'],'additional_monitor_seconds_charged':0.,
        'charge_scope':'Initial monitor belongs to prior 0101 QA process; current process charges only cache verification.'}


def save(output,name,model,optimizer,rng,coverage,step,arm,manifest,source_snapshot,started,
         *,valid=True,attempted_step=None):
    directory = Path(output)/name
    directory.mkdir(exist_ok=False)
    torch.save(model.state_dict(),directory/'model.pt')
    torch.save(optimizer.state_dict(),directory/'optimizer.pt')
    torch.save({'numpy':rng.bit_generator.state,'torch':torch.get_rng_state(),
                'numpy_global':np.random.get_state(),'coverage':coverage},directory/'rng.pt')
    metadata = {'format':'shared_closure_FS_0101','arm':arm,'step':step if valid else None,
        'new_steps':step-START_STEP if valid else None,'last_verified_step':step,
        'attempted_step':attempted_step,'valid_training_state':valid,'ancestor_step':START_STEP,
        'optimizer_reset':True,'optimizer_updates':step-START_STEP if valid else None,
        'objective':'0098 data terms plus 1e-4 times collected effective C23 L1 norm',
        'training_dt':.01,'physical_scoring_dt':.01,'local_horizon':.2,'latent_horizon':.01,
        'dimension':model.dim,'model_kind':getattr(model,'model_kind',arm),'config':model.config(),'coverage':coverage,
        'model_state_sha256':_model_hash(model),'fixed_buffers_sha256':buffer_hash(model),
        'monitor_manifest_sha256':None if manifest is None else _json_digest(manifest),
        'requires_grad':{n:p.requires_grad for n,p in model.named_parameters()},
        'requires_grad_parameters':[n for n,p in model.named_parameters() if p.requires_grad],
        'parameter_numel':{n:p.numel() for n,p in model.named_parameters()},
        'optimizer_parameter_names':parameter_layout(model,optimizer),'source_snapshot':source_snapshot,
        'rng_scope':'source Torch CPU and NumPy Generator restored; global NumPy captured prospectively',
        'process_seconds_at_checkpoint':time.perf_counter()-started}
    write_json(directory/'checkpoint.json',metadata)
    if valid:
        model.export(directory/'structured_numeric_export.json')
    write_json(directory/'artifact_hashes.json',{p.name:digest(p) for p in directory.iterdir() if p.is_file()})
    return str(directory)


def run(arm,output,new_steps=1000,cap_seconds=9000.,qa_only=False,cost_evidence=None,
        source=SOURCE,process_start=None,initial_monitor=None):
    if arm not in ARMS or new_steps not in (500,1000) or not 0<cap_seconds<=9000:
        raise ValueError('unregistered arm, nominal steps or cap')
    output,source = Path(output),Path(source)
    if output.exists():
        raise FileExistsError(output)
    started = PROCESS_STARTED if process_start is None else process_start
    deadline = started+cap_seconds
    torch.set_num_threads(1)
    def guard():
        if time.perf_counter() >= deadline-15:
            raise TimeoutError('serialization reserve reached')
        if process_memory_bytes()>MEMORY_CAP:
            raise MemoryError('8 GiB process cap')
    guard()
    before = snapshot(source)
    model,rng,coverage,source_meta = restore_model_or_pair(arm,source)
    optimizer = optimizer_for(model)
    layout = parameter_layout(model,optimizer)
    validate_optimizer(model,optimizer,layout,0)
    fixed = buffer_hash(model)
    initial_structure = check_structure(model)
    events,manifest,force_scale = None,None,None
    if not qa_only:
        events,excluded = read_train()
        if (len(events),len(excluded),sum(e['label']=='unstable' for e in events),len({e['group'] for e in events})) != (346,23,35,24):
            raise ValueError('registered TRAIN identities changed')
        manifest = build_manifest(events)
        if source_meta['monitor_manifest_sha256'] != _json_digest(manifest):
            raise ValueError('fixed TRAIN monitor manifest changed')
        force_scale = float(np.load(NORM,allow_pickle=False)['force_scale'])
    elif initial_monitor is not None:
        # Validate the cached fixed manifest against the pinned source metadata;
        # QA-only requires neither physical TRAIN payloads nor integration.
        manifest=json.loads((Path(initial_monitor)/'manifest.json').read_text(encoding='utf-8'))
        if source_meta['monitor_manifest_sha256']!=_json_digest(manifest):
            raise ValueError('cached monitor TRAIN manifest differs from pinned source')
    cached,cache_audit=None,None
    if initial_monitor is not None:
        cached,cache_audit=cached_initial_monitor(initial_monitor,arm,model,manifest)
    output.mkdir(parents=True,exist_ok=False)
    if manifest is not None:
        write_json(output/'manifest.json',manifest)
    completed,attempted = START_STEP,None
    monitors,traces = {},[]
    stop,failure,last_path,estimates = None,None,None,None
    last_monitor = 0.
    milestone = save(output,'step0000',model,optimizer,rng,coverage,completed,arm,manifest,before,started)
    def monitor_at(n):
        nonlocal last_monitor
        guard()
        if n==0 and cached is not None:
            if before!=snapshot(source):
                raise RuntimeError('source/code drift before cached monitor reuse')
            monitors['0']=cached
            last_monitor=float(cached['seconds'])
            return cached
        if time.perf_counter()+max(300.,1.5*last_monitor+60.)>=deadline:
            raise TimeoutError('insufficient full-monitor reserve')
        ts,ns,rs = torch.get_rng_state().clone(),np.random.get_state(),copy.deepcopy(rng.bit_generator.state)
        state_hash = _model_hash(model)
        try:
            row = run_monitor(model,events,manifest,output/f'monitor_step{n:04d}',
                deadline-15,allow_refine=True,force_scale=force_scale)
            if (_model_hash(model)!=state_hash or not torch.equal(ts,torch.get_rng_state())
                    or _json_digest(rs)!=_json_digest(rng.bit_generator.state)
                    or not all(np.array_equal(a,b) for a,b in zip(ns,np.random.get_state()))):
                raise RuntimeError('monitor changed model or training RNG')
        finally:
            torch.set_rng_state(ts); np.random.set_state(ns); rng.bit_generator.state=rs
        monitors[str(n)] = row  # Frozen monitor keeps its original implementation label.
        if row.get('complete'):
            last_monitor = float(row['seconds'])
        if before!=snapshot(source):
            raise RuntimeError('source/code drift around monitor')
        return row
    try:
        costs = read_cost_evidence(cost_evidence,arm) if cost_evidence is not None else None
        if costs is not None:
            last_monitor = float(cached['seconds']) if cached is not None else float(costs['monitor_seconds'])
            write_json(output/'cost_evidence.json',{'path':str(cost_evidence),'sha256':digest(cost_evidence),'evidence':costs})
            estimates = {str(n):{'remaining_seconds':estimated_cost(n,costs['step_seconds'],last_monitor,include_initial=cached is None),
                'includes_initial_monitor':cached is None,'monitor_seconds_used':last_monitor,
                'fits_remaining_cap':time.perf_counter()+estimated_cost(n,costs['step_seconds'],last_monitor,include_initial=cached is None)<deadline} for n in (500,1000)}
        if qa_only:
            if cached is not None:
                monitors['0']=cached
            stop = 'qa_restore_structure_complete'
        else:
            if costs is None:
                raise ValueError('formal fit requires matching no-update cost evidence')
            if not estimates[str(new_steps)]['fits_remaining_cap']:
                raise TimeoutError('nominal plan cannot fit before initial monitor; root must choose common registered plan')
            initial = monitor_at(0)  # C23 is a new objective: never reuse old monitor J.
            baseline = initial.get('raw_S_fine') or {}
            if not initial.get('complete') or any(baseline.get(h) is None or not np.isfinite(baseline[h]) for h in ('1','2','5')):
                raise RuntimeError('complete finite initial severe monitor required')
            if time.perf_counter()+estimated_cost(new_steps,costs['step_seconds'],last_monitor,include_initial=False)>=deadline:
                raise TimeoutError('nominal plan cannot fit measured new initial-monitor cost')
            counters = {'1':0,'2':0,'5':0}
            model.train()
            for step in range(START_STEP+1,START_STEP+new_steps+1):
                guard()
                recent = [r['seconds'] for r in traces[-10:]] or list(costs['step_seconds'].values())
                if time.perf_counter()+reserve_seconds(last_monitor,recent)>=deadline:
                    stop='wall_cap_reserve';break
                if before!=snapshot(source):
                    raise RuntimeError('source/code drift before update')
                attempted=step
                tic=time.perf_counter()
                samples,trace=next_samples(events,rng,coverage)
                trace.update(step=step,new_step=step-START_STEP,horizon=horizon(step),training_dt=.01,
                    torch_rng_before_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
                with (output/'samples.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(trace,allow_nan=False)+'\n')
                optimizer.zero_grad(set_to_none=True)
                with graph(model):
                    loss,terms=objective(model,samples,histories_for(samples),horizon(step),force_scale)
                    if not torch.isfinite(loss) or any(not torch.isfinite(v).all() for v in terms.values()):
                        raise FloatingPointError('nonfinite objective')
                    loss.backward()
                params=[p for p in model.parameters() if p.requires_grad]
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError('missing/nonfinite allowed gradient')
                grad_norm=float(torch.nn.utils.clip_grad_norm_(params,100.))
                if not np.isfinite(grad_norm):
                    raise FloatingPointError('nonfinite gradient norm')
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite post-update parameters')
                if fixed!=buffer_hash(model):
                    raise RuntimeError('fixed graph/mask/unit buffer drift')
                validate_optimizer(model,optimizer,layout,step-START_STEP)
                check_structure(model)
                completed=step
                trace.update(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
                    gradient_norm=grad_norm,gradient_clipped=grad_norm>100.,
                    finite_gradient_parameter_tensors=len(params),
                    finite_gradient_scalar_count=sum(p.numel() for p in params),
                    seconds=time.perf_counter()-tic,elapsed_seconds=time.perf_counter()-started,
                    torch_rng_after_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
                traces.append(trace)
                with (output/'trace.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(trace,allow_nan=False)+'\n')
                if (step-START_STEP)%50==0:
                    print(json.dumps({'arm':arm,'new_steps':step-START_STEP,'seconds':time.perf_counter()-started}),flush=True)
                guard()
                if step-START_STEP in MILESTONES:
                    milestone=save(output,f'step{step:04d}',model,optimizer,rng,coverage,completed,arm,manifest,before,started)
                    row=monitor_at(step-START_STEP)
                    if not row.get('complete'):
                        stop='partial_monitor';break
                    stop=severe_transition(row,baseline,counters)
                    if stop:break
            stop=stop or 'nominal_step_cap'
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'attempted_step':attempted}
        stop='failed_state_preserved'
        save(output,'failed_state',model,optimizer,rng,coverage,completed,arm,manifest,before,started,
             valid=False,attempted_step=attempted)
    finally:
        if failure is None:
            last_path=save(output,f'last_step{completed:04d}',model,optimizer,rng,coverage,completed,arm,manifest,before,started)
        after=snapshot(source)
        evidence={'loop':'0101','arm':arm,'objective':'C23 collected effective coefficients, original data terms',
            'optimizer_reset':True,'ancestor_step':START_STEP,'requested_new_steps':new_steps,
            'verified_completed_updates':completed-START_STEP,'final_step':completed if failure is None else None,
            'status':stop,'failure':failure,'qa_only':qa_only,'monitors':monitors,
            'last_checkpoint':last_path,'last_valid_milestone':milestone,'initial_structure_qa':initial_structure,
            'initial_monitor_cache':cache_audit,
            'source_before':before,'source_after':after,'sources_unchanged':before==after,
            'nominal_cost_estimates':estimates,'phase_cap_seconds':cap_seconds,
            'seconds':time.perf_counter()-started,'within_cap':time.perf_counter()<=deadline,
            'test_read':False,'validation_read':False,'checkpoint_selection':None}
        if before!=after:evidence['status']='quarantined_source_drift'
        if not evidence['within_cap']:evidence['status']='budget_exception'
        write_json(output/'evidence.json',evidence)
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--new-steps',type=int,choices=(500,1000),default=1000)
    p.add_argument('--cap-seconds',type=float,default=9000.)
    p.add_argument('--cost-evidence',type=Path)
    p.add_argument('--initial-monitor',type=Path,help='strictly validated 0101 C23 QA monitor directory')
    p.add_argument('--qa-only',action='store_true',help='restore and structure checks only, no integration')
    result=run(**vars(p.parse_args()),process_start=PROCESS_STARTED)
    print(json.dumps({'status':result['status'],'final_step':result['final_step']}),flush=True)
    if result.get('failure') or result['status'] in ('budget_exception','quarantined_source_drift'):
        raise SystemExit(1)


if __name__=='__main__':main()
