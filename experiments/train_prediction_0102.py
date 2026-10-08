"""0102 source recovery, original autonomous objective, and explicit-plan runner.

No fit is enabled by defaults. A registered, source-pinned JSON plan is required.
The timing path performs backward/clip but never optimizer.step().
"""
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
from shared_closure_rom_0101 import load_model, make_pair
from train_local_memory_0098 import (objective, graph, histories_for, read_train,
    NORM, sources, check_structure, process_memory_bytes, horizon)
from train_shared_closure_0101 import buffer_hash, validate_optimizer
from train_joint_numerical_pair_0100 import (digest, _json_digest, write_json,
    parameter_layout, validate_coverage, next_samples, MEMORY_CAP)
from monitor_joint_adequacy_0099 import build_manifest, run_monitor, _model_hash

SOURCE = ROOT/'private_data/task2_shared_closure_FS_0101/formal_S23/step2167'
START_STEP = 2167
CARD = ROOT/'reports/task2_prediction_optimization_0102.md'
MODEL_CODE = ROOT/'experiments/shared_closure_rom_0101.py'
OBJECTIVE_CODE = ROOT/'experiments/train_local_memory_0098.py'
M_CODE = ROOT/'experiments/multiple_shooting_0102.py'
EXPECTED = {
    'checkpoint.json': '56b0361f115e8b6aebdf8f2476530443638fa32972fab88d9a38116da9e5df6a',
    'model.pt': '86433553be7859bffb7246deab7ebbe89585683f149b4c685b8141df50fd13ac',
    'optimizer.pt': 'b6eff77fc857021c8ab566a21d2385e5abc27be89e8d8700922c912b500c5926',
    'rng.pt': '8ee02b082cacb56855d6208eecd8794267fe4c23f37d8567e8d27f1d57b5dc42',
    'structured_numeric_export.json': '38fba58d4945492f9ec82d8ffcee1d4a579fb29eb667d1600f18f77e2db48adf',
}
ARMS = ('S-O', 'S-M', 'F-O', 'F-M')


def family(arm):
    if arm in ('S23', 'S-O', 'S-M'):
        return 'S23'
    if arm in ('F23', 'F-O', 'F-M'):
        return 'F23'
    raise ValueError('unknown 0102 arm')


def capture_rng(rng):
    return {'torch': torch.get_rng_state().clone(),
            'numpy_global': copy.deepcopy(np.random.get_state()),
            'numpy': copy.deepcopy(rng.bit_generator.state)}


def restore_rng(state, rng):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy_global'])
    rng.bit_generator.state = copy.deepcopy(state['numpy'])


def rng_digest(state):
    h = hashlib.sha256(state['torch'].numpy().tobytes())
    h.update(_json_digest(state['numpy']).encode())
    ns = state['numpy_global']
    h.update(ns[0].encode()); h.update(ns[1].tobytes())
    h.update(repr(ns[2:]).encode())
    return h.hexdigest()


def safe_json(value):
    if isinstance(value,dict):return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def verify_source(source=SOURCE):
    source = Path(source)
    if source.resolve() != SOURCE.resolve():
        raise ValueError('only the registered S23 step2167 source is allowed')
    if {name: digest(source/name) for name in EXPECTED} != EXPECTED:
        raise ValueError('0102 pinned source artifact hash mismatch')
    meta = json.loads((source/'checkpoint.json').read_text(encoding='utf-8'))
    if (meta.get('step') != START_STEP or meta.get('arm') != 'S23'
            or meta.get('valid_training_state') is not True or meta.get('dimension') != 40):
        raise ValueError('invalid common source metadata')
    return meta


def restore_model_or_pair(arm=None, source=SOURCE, count=346):
    """Restore exact source RNG/coverage and S field; freshly expand F, no Adam load."""
    if arm is not None:
        kind = family(arm)
    meta = verify_source(source)
    source = Path(source)
    # Use the pinned export explicitly; directory candidate precedence cannot
    # accidentally load an unregistered extra numeric_export.json.
    base = load_model(source/'structured_numeric_export.json')
    raw = torch.load(source/'model.pt', map_location='cpu', weights_only=True)
    if set(raw) != set(base.state_dict()) or any(
            not torch.equal(raw[k], v) for k, v in base.state_dict().items()):
        raise ValueError('pinned export and raw checkpoint disagree')
    if (_model_hash(base) != meta['model_state_sha256']
            or buffer_hash(base) != meta['fixed_buffers_sha256']
            or {n:p.requires_grad for n,p in base.named_parameters()} != meta['requires_grad']
            or {n:p.numel() for n,p in base.named_parameters()} != meta['parameter_numel']):
        raise ValueError('restored S parameter/fixed-buffer identity mismatch')
    coverage = copy.deepcopy(meta['coverage'])
    validate_coverage(coverage, count)
    state = torch.load(source/'rng.pt', map_location='cpu', weights_only=False)
    if state.get('coverage') != coverage or 'numpy_global' not in state:
        raise ValueError('source RNG/coverage metadata incomplete or inconsistent')
    rng = np.random.default_rng()
    try:
        s_model, f_model = make_pair(base)
    finally:
        restore_rng(state, rng)
    pair = {'S23': s_model, 'F23': f_model}
    for model in pair.values():
        if model.dim != 40 or any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError('nonfinite or wrong-dimensional source expansion')
    if _model_hash(s_model) != meta['model_state_sha256']:
        raise ValueError('S expansion modified the common source')
    return (pair if arm is None else pair[kind]), rng, coverage, dict(
        meta, source_step=START_STEP, optimizer_reset=True,
        rng_scope='0101 source Torch CPU, NumPy Generator and saved NumPy global state')


def parameter_block(name):
    if name.startswith(('local_gru.', 'local_head.')):
        return 'encoder'
    if name.startswith('structured.') or name in ('_q_quadratic_raw', '_template_scalars.5'):
        return 'skeleton'
    if name in ('port_raw', 'closure_coefficients') or name in {f'_template_scalars.{i}' for i in range(5)}:
        return 'closure'
    raise ValueError('unregistered parameter name: '+name)


def optimizer_for(model, rates):
    """Explicit block rates; no stage-specific rates are inferred here."""
    if set(rates) != {'skeleton', 'closure', 'encoder'} or any(
            not np.isfinite(v) or v <= 0 for v in rates.values()):
        raise ValueError('three positive finite registered block rates required')
    blocks = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            blocks.setdefault(parameter_block(name), []).append(p)
    opt = torch.optim.Adam([{'params':ps, 'lr':float(rates[b]), 'block':b}
                            for b,ps in blocks.items()])
    validate_optimizer(model, opt, parameter_layout(model,opt), 0)
    return opt


def snapshot(source=SOURCE):
    paths = {'runner0102':Path(__file__), 'card0102':CARD, 'model0101':MODEL_CODE,
             'shooting0102':M_CODE,
             'monitor0099':ROOT/'experiments/monitor_joint_adequacy_0099.py',
             'runner0101':ROOT/'experiments/train_shared_closure_0101.py',
             'runner0100':ROOT/'experiments/train_joint_numerical_pair_0100.py'}
    paths.update({'source_'+k:Path(source)/k for k in EXPECTED})
    return dict(sources(), **{k:{'path':str(p),'sha256':digest(p)} for k,p in paths.items()})


def guard(deadline, reserve=15.):
    if time.perf_counter()+reserve >= deadline:
        raise TimeoutError('process deadline/serialization reserve reached')
    if process_memory_bytes() > MEMORY_CAP:
        raise MemoryError('8 GiB process cap')


def summarize_term(value):
    data=value.detach().cpu().to(dtype=torch.float64)
    if data.numel()==1:return float(data)
    result={'shape':list(data.shape),'count':data.numel()}
    if data.numel():
        result.update(mean=float(data.mean()),rms=float(data.square().mean().sqrt()),
            abs_quantiles=dict(zip(('0','50','90','99','100'),torch.quantile(data.abs().reshape(-1),
                torch.tensor([0.,.5,.9,.99,1.],dtype=torch.float64)).tolist())))
        if data.ndim==2 and data.shape[1]==40:
            result['component_rms']={name:float(data[:,i:i+10].square().mean().sqrt())
                for i,name in zip((0,10,20,30),('delta','omega','p','q'))}
    return result


def compute_gradients(model, samples, horizon, force_scale, optimizer, *, clip_norm, loss_fn=None,
                      auxiliary_optimizer=None, auxiliary_parameters=(), auxiliary_clip_norm=None,
                      collect_raw_terms=False):
    """Original O loss unless an explicit M closure is supplied; does not update."""
    if not np.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError('explicit positive gradient clip required')
    tic = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    if auxiliary_optimizer is not None:
        if auxiliary_clip_norm is None or not np.isfinite(auxiliary_clip_norm) or auxiliary_clip_norm<=0:
            raise ValueError('explicit positive auxiliary clip required')
        auxiliary_optimizer.zero_grad(set_to_none=True)
    with graph(model):
        loss, terms = (objective if loss_fn is None else loss_fn)(
            model, samples, histories_for(samples), horizon, force_scale)
        if not torch.isfinite(loss) or any(not torch.isfinite(v).all() for v in terms.values()):
            raise FloatingPointError('nonfinite objective or component')
        loss.backward()
    blocks = {}
    params = []
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise FloatingPointError('missing/nonfinite gradient: '+name)
        params.append(p)
        block = blocks.setdefault(parameter_block(name), {'squared_norm':0., 'tensors':0, 'scalars':0})
        block['squared_norm'] += float(p.grad.detach().square().sum())
        block['tensors'] += 1; block['scalars'] += p.numel()
    norm = float(torch.nn.utils.clip_grad_norm_(params, clip_norm))
    if not np.isfinite(norm):
        raise FloatingPointError('nonfinite gradient norm')
    active_aux=[p for p in auxiliary_parameters if p.grad is not None]
    if any(not torch.isfinite(p.grad).all() for p in active_aux):
        raise FloatingPointError('nonfinite auxiliary gradient')
    aux_norm=(float(torch.nn.utils.clip_grad_norm_(active_aux,auxiliary_clip_norm)) if active_aux else 0.)
    if not np.isfinite(aux_norm):
        raise FloatingPointError('nonfinite auxiliary gradient norm')
    result={'loss':float(loss.detach()), 'terms':{k:summarize_term(v) for k,v in terms.items()},
            'gradient_norm':norm, 'gradient_clipped':norm>clip_norm,
            'gradient_blocks':blocks, 'aux_gradient_norm':aux_norm,
            'aux_gradient_tensors':len(active_aux),'aux_gradient_scalars':sum(p.numel() for p in active_aux),
            'aux_gradient_clipped':bool(active_aux) and aux_norm>auxiliary_clip_norm,
            'seconds':time.perf_counter()-tic}
    if collect_raw_terms:
        result['raw_terms']={k:v.detach().cpu().tolist() for k,v in terms.items() if v.numel()!=1}
    return result


def apply_update(model, optimizer, *, layout, completed_updates, fixed_buffers):
    before = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('nonfinite post-update parameters')
    if buffer_hash(model) != fixed_buffers:
        raise RuntimeError('fixed graph/support/unit buffers changed')
    validate_optimizer(model, optimizer, layout, completed_updates)
    check_structure(model)
    updates = {}
    for n,p in model.named_parameters():
        if n in before:
            b=parameter_block(n)
            updates[b]=updates.get(b,0.)+float((p.detach()-before[n]).square().sum())
    return {'update_block_l2':{k:float(np.sqrt(v)) for k,v in updates.items()}}


def save_checkpoint(output, name, model, optimizer, rng, coverage, *, arm, updates,
                    source_snapshot, started, valid=True, attempted_update=None, auxiliary=None, context=None):
    directory = Path(output)/name
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), directory/'model.pt')
    torch.save(optimizer.state_dict(), directory/'optimizer.pt')
    torch.save(dict(capture_rng(rng), coverage=copy.deepcopy(coverage)), directory/'rng.pt')
    meta = {'format':'prediction_optimization_0102', 'arm':arm, 'source_step':START_STEP,
        'source_model_sha256':EXPECTED['model.pt'], 'optimizer_reset':True,
        'valid_training_state':valid, 'completed_updates':updates if valid else None,
        'last_verified_updates':updates, 'attempted_update':attempted_update,
        'step':START_STEP+updates if valid else None, 'model_kind':family(arm),
        'dimension':model.dim, 'training_dt':.01,
        'config':model.config(), 'coverage':copy.deepcopy(coverage),
        'requires_grad':{n:p.requires_grad for n,p in model.named_parameters()},
        'parameter_numel':{n:p.numel() for n,p in model.named_parameters()},
        'optimizer_parameter_names':parameter_layout(model,optimizer),
        'model_state_sha256':_model_hash(model), 'fixed_buffers_sha256':buffer_hash(model),
        'source_snapshot':source_snapshot, 'process_seconds':time.perf_counter()-started,
        'training_auxiliary_present':auxiliary is not None, 'deployment_contains_auxiliary':False,
        'training_context':context}
    if auxiliary is not None:
        torch.save(auxiliary, directory/'training_auxiliary.pt')
    write_json(directory/'checkpoint.json', meta if valid else safe_json(meta))
    if valid:
        model.export(directory/'structured_numeric_export.json')
    write_json(directory/'artifact_hashes.json', {p.name:digest(p) for p in directory.iterdir() if p.is_file()})
    return str(directory)


def restore_checkpoint(directory, rates):
    """Exact recovery of a valid 0102 checkpoint; no automatic phase/plan choice."""
    directory=Path(directory)
    hashes=json.loads((directory/'artifact_hashes.json').read_text())
    required={'model.pt','optimizer.pt','rng.pt','checkpoint.json','structured_numeric_export.json'}
    if not required <= set(hashes) or any(Path(k).name != k or digest(directory/k)!=v for k,v in hashes.items()):
        raise ValueError('incomplete or corrupt 0102 checkpoint')
    meta=json.loads((directory/'checkpoint.json').read_text())
    if (meta.get('format')!='prediction_optimization_0102' or not meta.get('valid_training_state')
            or meta.get('source_model_sha256')!=EXPECTED['model.pt']):
        raise ValueError('checkpoint is not a valid 0102 continuation')
    saved_snapshot=meta.get('source_snapshot')
    if not isinstance(saved_snapshot,dict) or not saved_snapshot or saved_snapshot!=snapshot():
        raise ValueError('checkpoint source_snapshot does not match current source/code/protocol hashes')
    verify_source()
    model=load_model(directory/'structured_numeric_export.json')
    if _model_hash(model)!=meta['model_state_sha256'] or buffer_hash(model)!=meta['fixed_buffers_sha256']:
        raise ValueError('checkpoint model identity mismatch')
    if {n:p.requires_grad for n,p in model.named_parameters()}!=meta['requires_grad']:
        raise ValueError('allowed parameter identities changed')
    optimizer=optimizer_for(model,rates)
    if parameter_layout(model,optimizer)!=meta['optimizer_parameter_names']:
        raise ValueError('optimizer names/order/rates mismatch')
    optimizer.load_state_dict(torch.load(directory/'optimizer.pt',map_location='cpu',weights_only=False))
    validate_optimizer(model,optimizer,meta['optimizer_parameter_names'],meta['completed_updates'])
    state=torch.load(directory/'rng.pt',map_location='cpu',weights_only=False)
    if state['coverage']!=meta['coverage']:
        raise ValueError('RNG/coverage mismatch')
    validate_coverage(state['coverage'],346)
    rng=np.random.default_rng(); restore_rng(state,rng)
    auxiliary=(torch.load(directory/'training_auxiliary.pt',map_location='cpu',weights_only=False)
               if meta['training_auxiliary_present'] else None)
    return model,optimizer,rng,copy.deepcopy(state['coverage']),meta,auxiliary


def train_inputs():
    events, excluded=read_train()
    if (len(events),len(excluded),sum(e['label']=='unstable' for e in events),
            len({e['group'] for e in events}))!=(346,23,35,24):
        raise ValueError('registered TRAIN denominators changed')
    if _json_digest(build_manifest(events))!=verify_source()['monitor_manifest_sha256']:
        raise ValueError('TRAIN identities/fixed monitor manifest differ from common source')
    return events,float(np.load(NORM,allow_pickle=False)['force_scale'])


def identity_qa(output, *, cap_seconds):
    """One actual TRAIN initial state, no trajectory integration or updates."""
    if not np.isfinite(cap_seconds) or cap_seconds<=0:
        raise ValueError('positive finite QA cap required')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+cap_seconds; torch.set_num_threads(1); guard(deadline)
    before=snapshot(); pair,rng,coverage,_=restore_model_or_pair(); state=capture_rng(rng)
    hashes={k:_model_hash(v) for k,v in pair.items()}; rows={}; failure=None; checks={}
    try:
        events,_=train_inputs(); event=events[0]
        history=histories_for([(event,10)]); s,f=pair['S23'],pair['F23']
        with torch.no_grad():
            qs,qf=s.encode(history),f.encode(history)
            z=torch.cat((history[:,-1],qs),dim=-1)
            fs,ff=s.field(z),f.field(z)
        rows.update(event_id=event['id'],initial_index=10,history_shape=list(history.shape),
                    encoder_max_abs=float((qs-qf).abs().max()),rhs_max_abs=float((fs-ff).abs().max()))
        guard(deadline)
        js=torch.autograd.functional.jacobian(s.field,z[0].detach().requires_grad_(True))
        guard(deadline)
        jf=torch.autograd.functional.jacobian(f.field,z[0].detach().requires_grad_(True))
        rows['jacobian_max_abs']=float((js-jf).abs().max())
        checks.update(encoder=rows['encoder_max_abs']<=1e-13,rhs=rows['rhs_max_abs']<=2e-11,
            jacobian=rows['jacobian_max_abs']<=3e-10 and js.shape==jf.shape==(40,40),
            finite=all(bool(torch.isfinite(v).all()) for v in (qs,qf,fs,ff,js,jf)))
        exports={}
        for kind,model in pair.items():
            guard(deadline)
            path=output/(kind+'_structured_numeric_export.json'); exports[kind]=model.export(path)
            loaded=load_model(path)
            with torch.no_grad():
                dz=z.clone(); dz[:,:10]+=.17
                metrics={'reload_rhs_max_abs':float((loaded.field(z)-model.field(z)).abs().max()),
                    'reload_encoder_max_abs':float((loaded.encode(history)-model.encode(history)).abs().max()),
                    'gauge_max_abs':float((model.field(dz)-model.field(z)).abs().max()),
                    'f0_max_abs':float(model.field(torch.zeros_like(z)).abs().max()),
                    'kinematic_max_abs':float((model.field(z)[:,:10]-z[:,10:20]).abs().max())}
            rows[kind]=metrics
            checks[kind+'_roundtrip']=(_model_hash(loaded)==hashes[kind] and metrics['reload_rhs_max_abs']<=1e-12
                                       and metrics['reload_encoder_max_abs']<=1e-12)
            checks[kind+'_gauge_f0_kinematic']=all(metrics[k]<=1e-10 for k in ('gauge_max_abs','f0_max_abs','kinematic_max_abs'))
        checks['units']=exports['S23']['unit_metadata']==exports['F23']['unit_metadata']
        checks['physical_scales']=all(torch.equal(getattr(s,k),getattr(f,k)) for k in ('mass','p_scale','input_scale'))
        rows['unit_metadata']=exports['S23']['unit_metadata']
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
    finally:
        restore_rng(state,rng)
    after=snapshot()
    checks.update(model_unchanged=all(_model_hash(v)==hashes[k] for k,v in pair.items()),
        rng_unchanged=rng_digest(capture_rng(rng))==rng_digest(state),sources_unchanged=before==after,
        within_cap=time.perf_counter()<=deadline)
    result={'loop':'0102','complete':failure is None and all(checks.values()),'failure':failure,
        'checks':checks,'metrics':rows,'model_hashes':hashes,'source_before':before,'source_after':after,
        'parameter_updates':False,'trajectory_integration':False,'test_read':False,'validation_read':False,
        'seconds':time.perf_counter()-PROCESS_STARTED,'cap_seconds':cap_seconds}
    write_json(output/'evidence.json',result)
    return result


def cached_initial_monitor(directory, model, manifest):
    """Same-state C23 reuse, including the old S2167 monitor; no relabeling."""
    path=Path(directory)/'evidence.json'
    row=json.loads(path.read_text(encoding='utf-8'))
    current=_model_hash(model); source=sources()
    terms=(row.get('batch_terms') or {}).get('fine') or {}
    names=terms.get('term_names') or []
    c23=False
    try:
        values=np.asarray(terms['terms'],dtype=float)
        coeff=values[:,:,names.index('complexity')]
        with torch.no_grad(): expected=float(model.complexity().detach())
        c23=(values.shape==(173,3,len(names)) and np.isfinite(coeff).all()
             and np.allclose(coeff,expected,rtol=1e-12,atol=1e-12))
    except (KeyError,ValueError,IndexError,TypeError):
        pass
    checks={'complete':row.get('complete') is True,'model_before':row.get('model_state_sha256_before')==current,
        'model_after':row.get('model_state_sha256_after')==current,'C23':bool(c23),
        'manifest':row.get('manifest_sha256')==manifest['sha256'],
        'code':row.get('monitor_code_sha256')==digest(ROOT/'experiments/monitor_joint_adequacy_0099.py'),
        'source_before':row.get('source_before')==source,'source_after':row.get('source_after')==source,
        'sources_unchanged':row.get('sources_unchanged') is True,'rng_unchanged':row.get('rng_unchanged') is True,
        'seconds':isinstance(row.get('seconds'),(int,float)) and np.isfinite(row['seconds']) and row['seconds']>0}
    if not all(checks.values()):
        raise ValueError('initial monitor cache rejected: '+','.join(k for k,v in checks.items() if not v))
    return row,{'path':str(path),'sha256':digest(path),'checks':checks,
        'additional_monitor_seconds_charged':0.,'legacy_implementation_label_preserved':True}


def shooting_boundaries(events, segment_seconds, horizon=5.):
    stride=round(segment_seconds/.01)
    if segment_seconds not in (.5,1.,2.) or horizon!=5.:
        raise ValueError('0102 requires max 0.5/1/2s segments over real horizons up to 5s')
    result={}
    for event in events:
        end=min(10+round(horizon/.01),len(event['x'])-1)
        if end<=10 or event['id'] in result:raise ValueError('invalid/duplicate TRAIN event')
        result[event['id']]=list(range(10,end,stride))+[end]
    return result


def q_values_by_boundary(problem):
    return {(problem.events[r.event_index]['id'],r.start):problem.q_aux[r.auxiliary_index].detach().clone()
            for r in problem.segment_records if r.auxiliary_index is not None}


def prepare_shooting_cache(output, *, cap_seconds):
    """One batched source rollout per TRAIN event, recording every union boundary."""
    from multiple_shooting_0102 import train_state_scales
    if not np.isfinite(cap_seconds) or cap_seconds<=0:raise ValueError('finite positive cap required')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+cap_seconds; torch.set_num_threads(1); guard(deadline)
    before=snapshot(); model,rng,_,_=restore_model_or_pair('S23'); state=capture_rng(rng)
    model_hash=_model_hash(model); events,_=train_inputs(); manifest=build_manifest(events)
    layouts={str(s):shooting_boundaries(events,s) for s in (.5,1.,2.)}
    targets={e['id']:sorted({i for layout in layouts.values() for i in layout[e['id']][1:-1]}) for e in events}
    cache={}; completed_events=0; failure=None; scales=None
    try:
        for start in range(0,len(events),32):
            guard(deadline); batch=events[start:start+32]
            history=histories_for([(e,10) for e in batch])
            with torch.no_grad(),graph(model):
                current=torch.cat((history[:,-1],model.encode(history)),dim=-1)
                ends=[max(targets[e['id']],default=10) for e in batch]
                for index in range(11,max(ends)+1):
                    if index%20==0:guard(deadline)
                    active=[i for i,end in enumerate(ends) if index<=end]
                    ids=torch.tensor(active,dtype=torch.long)
                    values=model.rk4(current[ids],.01)
                    if not torch.isfinite(values).all():raise FloatingPointError('nonfinite source q-cache rollout')
                    current=current.index_copy(0,ids,values)
                    for i in active:
                        if index in targets[batch[i]['id']]:cache[(batch[i]['id'],index)]=current[i,30:].detach().clone()
            completed_events+=len(batch)
            write_json(output/'progress.json',{'completed_events':completed_events,'q_boundaries':len(cache),
                'seconds':time.perf_counter()-PROCESS_STARTED,'parameter_updates':False})
        reference=torch.stack(list(cache.values()))
        scales=train_state_scales(events,q_reference=reference,model=model,q_eps=1e-3)
        if scales.shape!=(40,) or not torch.isfinite(scales).all() or not (scales>0).all():
            raise FloatingPointError('invalid fixed shooting scales')
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
    finally:restore_rng(state,rng)
    payload={'q_cache':cache,'scales':scales,'layouts':layouts}
    torch.save(payload,output/'q_cache.pt')
    after=snapshot(); valid=(_model_hash(model)==model_hash and rng_digest(capture_rng(rng))==rng_digest(state))
    result={'loop':'0102','complete':failure is None and completed_events==346 and valid and before==after,
        'failure':failure,'parameter_updates':False,'source_model_sha256':EXPECTED['model.pt'],
        'source_state_sha256':model_hash,'model_code_sha256':digest(MODEL_CODE),'shooting_code_sha256':digest(M_CODE),
        'manifest_sha256':manifest['sha256'],'q_cache_sha256':digest(output/'q_cache.pt'),
        'q_boundaries':len(cache),'completed_events':completed_events,'dt':.01,'max_horizon':5.,
        'q_scale_floor':1e-3,'model_rng_unchanged':valid,'source_before':before,'source_after':after,
        'sources_unchanged':before==after,'test_read':False,'validation_read':False,
        'seconds':time.perf_counter()-PROCESS_STARTED,'cap_seconds':cap_seconds}
    result['within_cap']=time.perf_counter()<=deadline; result['complete'] &= result['within_cap']
    write_json(output/'evidence.json',result)
    return result


def load_shooting_cache(directory, events):
    directory=Path(directory); meta=json.loads((directory/'evidence.json').read_text())
    if (meta.get('complete') is not True or meta.get('parameter_updates') is not False
            or meta.get('source_model_sha256')!=EXPECTED['model.pt']
            or meta.get('model_code_sha256')!=digest(MODEL_CODE) or meta.get('shooting_code_sha256')!=digest(M_CODE)
            or meta.get('manifest_sha256')!=build_manifest(events)['sha256']
            or meta.get('q_cache_sha256')!=digest(directory/'q_cache.pt')
            or meta.get('model_rng_unchanged') is not True or meta.get('sources_unchanged') is not True):
        raise ValueError('shooting cache identity/integrity mismatch')
    payload=torch.load(directory/'q_cache.pt',map_location='cpu',weights_only=False)
    layouts={str(s):shooting_boundaries(events,s) for s in (.5,1.,2.)}
    keys={(e['id'],i) for e in events for layout in layouts.values() for i in layout[e['id']][1:-1]}
    if payload['layouts']!=layouts or set(payload['q_cache'])!=keys:
        raise ValueError('shooting cache TRAIN boundary coverage mismatch')
    if any(v.shape!=(10,) or not torch.isfinite(v).all() for v in payload['q_cache'].values()):
        raise ValueError('nonfinite/wrong-dimensional source q cache')
    scales=payload['scales']
    if scales.shape!=(40,) or not torch.isfinite(scales).all() or not (scales>0).all():
        raise ValueError('invalid fixed continuity scales')
    return payload,meta


def shooting_problem(model,events,cache,phase,previous=None):
    from multiple_shooting_0102 import MultipleShootingProblem,make_objective,nonlong_0098_objective
    if phase['aux_optimizer_policy']!='reset_all_each_shooting_phase' or phase['q_lr']!=.01 or phase['q_clip_norm']!=100.:
        raise ValueError('unregistered auxiliary optimizer policy')
    starts=dict(cache['q_cache'])
    if previous is not None:starts.update(q_values_by_boundary(previous))
    problem=MultipleShootingProblem(model,events,cache['layouts'][str(float(phase['segment_seconds']))],
        scales=cache['scales'],dt=.01,q_initial=starts)
    opt=torch.optim.Adam(problem.auxiliary_parameters,lr=phase['q_lr'])
    loss=make_objective(problem,nonlong_objective=nonlong_0098_objective,
                        trajectory_weight=1.,continuity_weight=phase['continuity_weight'],risk_weight=10.)
    return problem,opt,loss


def auxiliary_payload(problem, optimizer, phase_index):
    if problem is None:return None
    return {'phase_index':phase_index,'boundaries':problem.boundaries,'scales':problem.scales.detach().clone(),
        'event_ids':[e['id'] for e in problem.events],'q_by_boundary':q_values_by_boundary(problem),
        'auxiliary_state_dict':problem.auxiliary_state_dict(),'optimizer':optimizer.state_dict()}


def time_shooting_without_updates(arm, output, *, rates, clip_norm, batches, cap_seconds,
                                  shooting_cache, continuity_weights):
    """All three M courses, same matched sampled windows; no model/q updates."""
    if (arm not in ('S-M','F-M') or batches<3 or batches%3 or set(continuity_weights)!={'0.5','1.0','2.0'}
            or not np.isfinite(cap_seconds) or cap_seconds<=0
            or any(not np.isfinite(v) or v<=0 for v in continuity_weights.values())):
        raise ValueError('explicit M arm, full horizon cycles and three penalties required')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+cap_seconds; torch.set_num_threads(1); guard(deadline)
    before=snapshot(); model,rng,coverage,_=restore_model_or_pair(arm); initial=capture_rng(rng)
    initial_coverage=copy.deepcopy(coverage); state_hash=_model_hash(model); opt=optimizer_for(model,rates)
    events,force_scale=train_inputs(); cache,cache_meta=load_shooting_cache(shooting_cache,events)
    rows=[]; failure=None
    try:
        for segment in (.5,1.,2.):
            restore_rng(initial,rng); coverage=copy.deepcopy(initial_coverage)
            phase={'segment_seconds':segment,'aux_optimizer_policy':'reset_all_each_shooting_phase',
                'q_lr':.01,'q_clip_norm':100.,'continuity_weight':continuity_weights[str(segment)]}
            problem,qopt,loss_fn=shooting_problem(model,events,cache,phase)
            q_before=q_values_by_boundary(problem)
            for i in range(batches):
                guard(deadline,15.+1.5*max([r['seconds'] for r in rows],default=0.))
                samples,trace=next_samples(events,rng,coverage); h=(1.,2.,5.)[i%3]
                trace.update(horizon=h,segment_seconds=segment,timing_batch=i+1)
                trace.update(compute_gradients(model,samples,h,force_scale,opt,clip_norm=clip_norm,
                    loss_fn=loss_fn,auxiliary_optimizer=qopt,auxiliary_parameters=problem.auxiliary_parameters,
                    auxiliary_clip_norm=phase['q_clip_norm']))
                rows.append(trace); write_json(output/'partial.json',{'complete':False,'batches':rows})
            if qopt.state or any(not torch.equal(v,q_before[k]) for k,v in q_values_by_boundary(problem).items()):
                raise RuntimeError('no-update timing altered auxiliary q')
    except Exception as exc:failure={'type':type(exc).__name__,'message':str(exc)}
    finally:opt.zero_grad(set_to_none=True); restore_rng(initial,rng)
    after=snapshot(); unchanged=_model_hash(model)==state_hash and not opt.state and rng_digest(capture_rng(rng))==rng_digest(initial)
    result={'loop':'0102','arm':arm,'complete':failure is None and unchanged and before==after and len(rows)==3*batches,
        'failure':failure,'parameter_updates':False,'checkpoint_step':START_STEP,'training_dt':.01,
        'batches':rows,'phase_step_seconds':{str(s):{str(h):float(np.mean([r['seconds'] for r in rows if r['segment_seconds']==s and r['horizon']==h]))
            for h in (1,2,5) if any(r['segment_seconds']==s and r['horizon']==h for r in rows)} for s in (.5,1.,2.)},
        'rates':rates,'clip_norm':clip_norm,'continuity_weights':continuity_weights,
        'source_model_sha256':EXPECTED['model.pt'],'model_code_sha256':digest(MODEL_CODE),
        'objective_code_sha256':digest(OBJECTIVE_CODE),'shooting_code_sha256':digest(M_CODE),
        'model_state_sha256_before':state_hash,'model_state_sha256_after':_model_hash(model),
        'rng_and_model_unchanged':unchanged,'shooting_cache_evidence':cache_meta,
        'source_before':before,'source_after':after,'sources_unchanged':before==after,
        'test_read':False,'validation_read':False,'seconds':time.perf_counter()-PROCESS_STARTED,'cap_seconds':cap_seconds}
    result['within_cap']=time.perf_counter()<=deadline; result['complete'] &= result['within_cap']
    write_json(output/'costs.json',result)
    return result


def time_without_updates(arm, output, *, rates, clip_norm, batches, cap_seconds):
    """No-update TRAIN timing only; call only under separately authorized QA."""
    if batches<3 or batches%3 or not np.isfinite(cap_seconds) or not 0<cap_seconds or not arm.endswith('-O'):
        raise ValueError('timing requires a positive cap and full 1/2/5 cycles')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+cap_seconds
    torch.set_num_threads(1); guard(deadline)
    before=snapshot(); model,rng,coverage,meta=restore_model_or_pair(arm)
    opt=optimizer_for(model,rates); events,force_scale=train_inputs()
    initial=capture_rng(rng); initial_coverage=copy.deepcopy(coverage); state_hash=_model_hash(model)
    rows=[]; failure=None
    try:
        for i in range(batches):
            guard(deadline,15.+1.5*max([r['seconds'] for r in rows],default=0.))
            samples,trace=next_samples(events,rng,coverage); h=(1.,2.,5.)[i%3]
            trace.update(horizon=h,timing_batch=i+1,training_dt=.01)
            trace.update(compute_gradients(model,samples,h,force_scale,opt,clip_norm=clip_norm))
            rows.append(trace)
            write_json(output/'partial.json',{'complete':False,'batches':rows})
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
    finally:
        opt.zero_grad(set_to_none=True); restore_rng(initial,rng); coverage=initial_coverage
    unchanged=_model_hash(model)==state_hash and rng_digest(capture_rng(rng))==rng_digest(initial) and not opt.state
    after=snapshot()
    result={'loop':'0102','arm':arm,'training_dt':.01,'checkpoint_step':START_STEP,
        'objective':'original C23 autonomous deployment objective', 'complete':failure is None and len(rows)==batches and unchanged and before==after,
        'parameter_updates':False,'failure':failure,'batches':rows,
        'step_seconds':{str(h):float(np.mean([r['seconds'] for r in rows if r['horizon']==h])) for h in (1,2,5) if any(r['horizon']==h for r in rows)},
        'source_model_sha256':EXPECTED['model.pt'],'model_code_sha256':digest(MODEL_CODE),
        'objective_code_sha256':digest(OBJECTIVE_CODE),'model_state_sha256_before':state_hash,
        'model_state_sha256_after':_model_hash(model),'rng_and_model_unchanged':unchanged,
        'source_before':before,'source_after':after,'sources_unchanged':before==after,
        'rates':rates,'clip_norm':clip_norm,'test_read':False,'validation_read':False,
        'seconds':time.perf_counter()-PROCESS_STARTED,'cap_seconds':cap_seconds}
    result['within_cap']=time.perf_counter()<=deadline
    result['complete']=result['complete'] and result['within_cap']
    write_json(output/'costs.json',result)
    return result


def validate_plan(plan):
    required={'loop','registered','card_sha256','source_model_sha256','arm','phase','updates',
              'horizons','rates','clip_norm','cap_seconds','monitor_steps','monitor_seconds',
              'cost_evidence','allow_refine'}
    if not required <= set(plan):
        raise ValueError('explicit plan fields missing: '+','.join(sorted(required-set(plan))))
    if (plan['loop']!='0102' or plan['registered'] is not True or plan['card_sha256']!=digest(CARD)
            or plan['source_model_sha256']!=EXPECTED['model.pt'] or plan['arm'] not in ARMS
            or plan['phase'] not in ('autonomous','multiple_shooting_then_autonomous')):
        raise ValueError('unregistered plan identity or unsupported phase')
    n=plan['updates']
    if isinstance(n,bool) or not isinstance(n,int) or n!=650 or len(plan['horizons'])!=n or any(h not in (1,2,5) for h in plan['horizons']):
        raise ValueError('registered 0102 plan requires exactly 650 updates and per-update horizons')
    if plan['horizons']!=[horizon(START_STEP+i) for i in range(1,651)]:
        raise ValueError('registered horizons must continue the absolute step2168 sequence')
    ms=plan['monitor_steps']
    if ms!=[0,325,650]:
        raise ValueError('registered 0102 monitor schedule must be 0/325/650')
    if any(not np.isfinite(plan[k]) or plan[k]<=0 for k in ('clip_norm','cap_seconds','monitor_seconds')):
        raise ValueError('finite positive clip/cap/monitor estimate required')
    if (plan['cap_seconds']!=6125. or plan['monitor_seconds']!=120. or plan['clip_norm']!=100.
            or plan['rates']!={'skeleton':.01,'closure':.001,'encoder':.0003}):
        raise ValueError('registered cap/monitor cost/clip/block rates changed')
    if not isinstance(plan['allow_refine'],bool):
        raise ValueError('explicit refinement policy required')
    if plan['phase']=='autonomous' and not plan['arm'].endswith('-O'):
        raise ValueError('M cannot reset to source for its final autonomous phase')
    if plan['phase']=='multiple_shooting_then_autonomous':
        if not plan['arm'].endswith('-M') or plan.get('model_optimizer_policy')!='continuous':
            raise ValueError('M requires continuous model Adam')
        phases=plan.get('phases',[])
        if len(phases)!=4 or [p.get('kind') for p in phases]!=['shooting']*3+['autonomous']:
            raise ValueError('registered M requires three shooting phases then autonomy')
        counts=[32,32,68,n-132]
        budgets=[p['training_seconds'] for p in phases]
        if budgets!=[848.,848.,1802.,1802.]:
            raise ValueError('M training budget must retain registered 848/848/1802/1802 second caps')
        offset=0
        for i,p in enumerate(phases):
            if p['updates']!=counts[i] or p['horizons']!=plan['horizons'][offset:offset+counts[i]]:
                raise ValueError('M phase steps/horizons differ from common sequence')
            offset+=counts[i]
            if i<3 and (p.get('segment_seconds')!=(.5,1.,2.)[i]
                    or p.get('aux_optimizer_policy')!='reset_all_each_shooting_phase'
                    or p.get('q_lr')!=.01 or p.get('q_clip_norm')!=100.
                    or p.get('continuity_weight')!=(1.,10.,100.)[i]):
                raise ValueError('unregistered M segment/auxiliary policy')
        if not plan.get('shooting_cache') or not plan.get('autonomous_cost_evidence'):
            raise ValueError('M requires immutable source cache and autonomous timing evidence')
    return plan


def run_plan(plan_path, output):
    """Autonomous phase only. No selection, extension, or inherited stop rule."""
    plan_path=Path(plan_path); plan=validate_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    if plan['phase']=='multiple_shooting_then_autonomous':
        return run_m_plan(plan_path,output,plan)
    plan_sha=digest(plan_path); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+plan['cap_seconds']; torch.set_num_threads(1); guard(deadline)
    before=snapshot(); model,rng,coverage,_=restore_model_or_pair(plan['arm'])
    optimizer=optimizer_for(model,plan['rates']); layout=parameter_layout(model,optimizer); fixed=buffer_hash(model)
    costs=json.loads(Path(plan['cost_evidence']).read_text(encoding='utf-8'))
    if (not costs.get('complete') or costs.get('parameter_updates') is not False
            or costs.get('arm')!=plan['arm'] or costs.get('checkpoint_step')!=START_STEP
            or costs.get('source_model_sha256')!=EXPECTED['model.pt']
            or costs.get('model_code_sha256')!=digest(MODEL_CODE)
            or costs.get('objective_code_sha256')!=digest(OBJECTIVE_CODE)
            or costs.get('rates')!=plan['rates'] or costs.get('clip_norm')!=plan['clip_norm']
            or costs.get('model_state_sha256_before')!=_model_hash(model)
            or costs.get('model_state_sha256_after')!=_model_hash(model)
            or costs.get('rng_and_model_unchanged') is not True):
        raise ValueError('plan lacks matching no-update cost evidence')
    if any(not np.isfinite(costs['step_seconds'][str(h)]) or costs['step_seconds'][str(h)]<=0 for h in (1,2,5)):
        raise ValueError('invalid measured horizon costs')
    estimate=1.2*(sum(costs['step_seconds'][str(int(h))] for h in plan['horizons'])+len(plan['monitor_steps'])*plan['monitor_seconds'])
    if time.perf_counter()+estimate+max(300.,1.5*plan['monitor_seconds']+60.)>=deadline:
        raise TimeoutError('registered complete plan cannot fit process cap')
    events,force_scale=train_inputs(); manifest=build_manifest(events)
    cached,cache_audit=(cached_initial_monitor(plan['initial_monitor'],model,manifest)
                        if plan.get('initial_monitor') else (None,None))
    write_json(output/'plan.json',plan); write_json(output/'manifest.json',manifest)
    completed=0; attempted=None; failure=None; monitors={}; recent=[]; last_monitor=plan['monitor_seconds']
    last_valid=None; final_path=None; stop=None
    try:
        for i in range(plan['updates']+1):
            if i in plan['monitor_steps']:
                guard(deadline,max(300.,1.5*last_monitor+60.))
                last_valid=save_checkpoint(output,f'step{i:05d}',model,optimizer,rng,coverage,
                    arm=plan['arm'],updates=completed,source_snapshot=before,started=PROCESS_STARTED)
                if i==0 and cached is not None:
                    row=cached
                else:
                    state=capture_rng(rng); state_hash=_model_hash(model)
                    try:
                        row=run_monitor(model,events,manifest,output/f'monitor_step{i:05d}',deadline-15,
                                        allow_refine=plan['allow_refine'],force_scale=force_scale)
                        if _model_hash(model)!=state_hash or rng_digest(capture_rng(rng))!=rng_digest(state):
                            raise RuntimeError('monitor changed model or RNG')
                    finally:
                        restore_rng(state,rng)
                monitors[str(i)]=row
                if not row.get('complete'):
                    stop='partial_monitor'; break
                last_monitor=row['seconds']
            if i==plan['updates']:
                stop='nominal_update_cap'; break
            guard(deadline,max(300.,1.5*last_monitor+60.)+1.5*max(recent[-10:],default=0.))
            if snapshot()!=before or digest(plan_path)!=plan_sha:
                raise RuntimeError('source/code/plan drift')
            attempted=i+1; tic=time.perf_counter(); samples,trace=next_samples(events,rng,coverage)
            trace.update(update=attempted,horizon=plan['horizons'][i],torch_rng_before_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            with (output/'samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(trace,allow_nan=False)+'\n')
            trace.update(compute_gradients(model,samples,plan['horizons'][i],force_scale,optimizer,clip_norm=plan['clip_norm']))
            trace.update(apply_update(model,optimizer,layout=layout,completed_updates=attempted,fixed_buffers=fixed))
            completed=attempted; trace.update(seconds=time.perf_counter()-tic,torch_rng_after_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
            recent.append(trace['seconds'])
            with (output/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(trace,allow_nan=False)+'\n')
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'attempted_update':attempted}
        stop='failed_state_preserved'
        save_checkpoint(output,'failed_state',model,optimizer,rng,coverage,arm=plan['arm'],updates=completed,
            source_snapshot=before,started=PROCESS_STARTED,valid=False,attempted_update=attempted)
    if failure is None:
        final_path=save_checkpoint(output,'last_state',model,optimizer,rng,coverage,arm=plan['arm'],updates=completed,
            source_snapshot=before,started=PROCESS_STARTED)
    after=snapshot(); seconds=time.perf_counter()-PROCESS_STARTED
    result={'loop':'0102','arm':plan['arm'],'phase':plan['phase'],'status':stop,'failure':failure,
        'completed_updates':completed,'requested_updates':plan['updates'],'monitors':monitors,'initial_monitor_cache':cache_audit,
        'last_valid_milestone':last_valid,'last_checkpoint':final_path,'plan_sha256':plan_sha,
        'source_before':before,'source_after':after,'sources_unchanged':before==after,
        'seconds':seconds,'cap_seconds':plan['cap_seconds'],'within_cap':time.perf_counter()<=deadline,
        'test_read':False,'validation_read':False,'checkpoint_selection':None}
    if before!=after or digest(plan_path)!=plan_sha:result['status']='quarantined_source_or_plan_drift'
    if not result['within_cap']:result['status']='budget_exception'
    write_json(output/'evidence.json',result)
    return result


def run_m_plan(plan_path, output, plan):
    """Registered M course with continuous model Adam and separate q Adam."""
    plan_sha=digest(plan_path); output=Path(output); output.mkdir(parents=True,exist_ok=False)
    deadline=PROCESS_STARTED+plan['cap_seconds']; torch.set_num_threads(1); guard(deadline)
    before=snapshot(); model,rng,coverage,_=restore_model_or_pair(plan['arm'])
    optimizer=optimizer_for(model,plan['rates']); layout=parameter_layout(model,optimizer); fixed=buffer_hash(model)
    state_hash=_model_hash(model)
    costs=json.loads(Path(plan['cost_evidence']).read_text()); ordinary=json.loads(Path(plan['autonomous_cost_evidence']).read_text())
    for value,expected_arm in ((costs,plan['arm']),(ordinary,plan['arm'][0]+'-O')):
        if (value.get('complete') is not True or value.get('parameter_updates') is not False
                or value.get('arm')!=expected_arm or value.get('checkpoint_step')!=START_STEP
                or value.get('source_model_sha256')!=EXPECTED['model.pt']
                or value.get('model_code_sha256')!=digest(MODEL_CODE)
                or value.get('objective_code_sha256')!=digest(OBJECTIVE_CODE)
                or value.get('rates')!=plan['rates'] or value.get('clip_norm')!=plan['clip_norm']
                or value.get('model_state_sha256_before')!=state_hash or value.get('model_state_sha256_after')!=state_hash
                or value.get('rng_and_model_unchanged') is not True):
            raise ValueError('M requires matching source-state no-update timing')
    if costs.get('shooting_code_sha256')!=digest(M_CODE):raise ValueError('M timing code drift')
    phase_costs=[]; estimates=[]
    for phase in plan['phases']:
        if phase['kind']=='shooting':
            key=str(float(phase['segment_seconds'])); measured=costs['phase_step_seconds'][key]
            if costs['continuity_weights'][key]!=phase['continuity_weight']:raise ValueError('M penalty timing mismatch')
        else:measured=ordinary['step_seconds']
        if any(not np.isfinite(measured[str(h)]) or measured[str(h)]<=0 for h in (1,2,5)):
            raise ValueError('all measured horizon costs required')
        estimate=1.2*sum(measured[str(int(h))] for h in phase['horizons'])
        if estimate>=phase['training_seconds']:raise TimeoutError('nominal M phase cannot fit with 20 percent margin')
        phase_costs.append(measured); estimates.append(estimate)
    reserve=max(300.,1.5*plan['monitor_seconds']+60.)
    if sum(p['training_seconds'] for p in plan['phases'])+1.2*len(plan['monitor_steps'])*plan['monitor_seconds']+reserve>=plan['cap_seconds']:
        raise TimeoutError('M phase caps/monitors/reserve exceed registered total process cap')
    events,force_scale=train_inputs(); manifest=build_manifest(events)
    cache,cache_meta=load_shooting_cache(plan['shooting_cache'],events)
    cached,cache_audit=(cached_initial_monitor(plan['initial_monitor'],model,manifest) if plan.get('initial_monitor') else (None,None))
    write_json(output/'plan.json',plan); write_json(output/'manifest.json',manifest)
    completed=0; attempted=None; failure=None; problem=qopt=loss_fn=None; phase_index=-1
    phase_seconds=[0.]*4; phase_updates=[0]*4; monitors={}; recent=[]; last_monitor=plan['monitor_seconds']
    last_valid=last_path=None; status=None
    def save_here(name,valid=True):
        return save_checkpoint(output,name,model,optimizer,rng,coverage,arm=plan['arm'],updates=completed,
            source_snapshot=before,started=PROCESS_STARTED,valid=valid,attempted_update=attempted,
            auxiliary=auxiliary_payload(problem,qopt,phase_index),
            context={'phase_index':phase_index,'phase_seconds':phase_seconds,'phase_updates':phase_updates,
                     'plan_sha256':plan_sha,'model_optimizer_policy':'continuous'})
    def monitor_at(n):
        nonlocal last_monitor,last_valid
        guard(deadline,max(300.,1.5*last_monitor+60.)); last_valid=save_here(f'step{n:05d}')
        if n==0 and cached is not None:row=cached
        else:
            state=capture_rng(rng); old_hash=_model_hash(model)
            try:
                row=run_monitor(model,events,manifest,output/f'monitor_step{n:05d}',deadline-15,
                    allow_refine=plan['allow_refine'],force_scale=force_scale)
                if _model_hash(model)!=old_hash or rng_digest(capture_rng(rng))!=rng_digest(state):
                    raise RuntimeError('monitor changed model/RNG')
            finally:restore_rng(state,rng)
        monitors[str(n)]=row
        if not row.get('complete'):raise RuntimeError('partial autonomous monitor; M strategy incomplete')
        last_monitor=row['seconds']
    try:
        monitor_at(0)
        for phase_index,phase in enumerate(plan['phases']):
            tic=time.perf_counter()
            if phase['kind']=='shooting':problem,qopt,loss_fn=shooting_problem(model,events,cache,phase,problem)
            else:problem=qopt=loss_fn=None  # No reset variables survive the final autonomous phase.
            phase_seconds[phase_index]+=time.perf_counter()-tic
            last_valid=save_here(f'phase{phase_index}_start')
            for local_step,h in enumerate(phase['horizons']):
                measured=phase_costs[phase_index][str(int(h))]
                remaining=phase['training_seconds']-phase_seconds[phase_index]
                if remaining<=1.2*max(measured,max(recent[-10:],default=0.)):
                    raise TimeoutError('M phase cap reserve reached; no borrowing from final autonomy')
                guard(deadline,max(300.,1.5*last_monitor+60.)+1.5*max(recent[-10:],default=measured))
                if snapshot()!=before or digest(plan_path)!=plan_sha:raise RuntimeError('source/code/plan drift')
                tic=time.perf_counter(); attempted=completed+1; samples,trace=next_samples(events,rng,coverage)
                trace.update(update=attempted,phase_index=phase_index,phase_kind=phase['kind'],horizon=h,
                    torch_rng_before_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
                with (output/'samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(trace,allow_nan=False)+'\n')
                boundary_audit=(problem is not None and (attempted in plan['monitor_steps'] or local_step+1==phase['updates']))
                trace.update(compute_gradients(model,samples,h,force_scale,optimizer,clip_norm=plan['clip_norm'],loss_fn=loss_fn,
                    auxiliary_optimizer=qopt,auxiliary_parameters=() if problem is None else problem.auxiliary_parameters,
                    auxiliary_clip_norm=None if problem is None else phase['q_clip_norm'],collect_raw_terms=boundary_audit))
                if boundary_audit:
                    write_json(output/f'residuals_before_update{attempted:05d}.json',{
                        'scope':'registered node sampled TRAIN windows, before this parameter update; not full cohort',
                        'model_state_sha256':_model_hash(model),'update':attempted,'phase_index':phase_index,
                        'sample_ids':trace['sample_ids'],'horizon':h,'event_index_to_id':[e['id'] for e in events],
                        'raw_terms':trace.pop('raw_terms')})
                trace.update(apply_update(model,optimizer,layout=layout,completed_updates=attempted,fixed_buffers=fixed))
                if qopt is not None:
                    qopt.step()
                    if any(not torch.isfinite(p).all() for p in problem.auxiliary_parameters):
                        raise FloatingPointError('nonfinite post-update q auxiliary')
                    if any(not torch.isfinite(v).all() for state in qopt.state.values() for v in state.values() if torch.is_tensor(v)):
                        raise FloatingPointError('nonfinite q Adam state')
                completed=attempted; phase_updates[phase_index]+=1
                seconds=time.perf_counter()-tic; phase_seconds[phase_index]+=seconds; recent.append(seconds)
                trace.update(seconds=seconds,phase_training_seconds=phase_seconds[phase_index],
                    torch_rng_after_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
                with (output/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(trace,allow_nan=False)+'\n')
                if phase_seconds[phase_index]>phase['training_seconds']:raise TimeoutError('M phase hard cap exceeded')
                if completed in plan['monitor_steps']:monitor_at(completed)
            last_valid=save_here(f'phase{phase_index}_end')
        status='nominal_update_cap'
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'attempted_update':attempted}; status='failed_state_preserved'
        save_here('failed_state',valid=False)
    if failure is None:last_path=save_here('last_state')
    after=snapshot(); seconds=time.perf_counter()-PROCESS_STARTED
    result={'loop':'0102','arm':plan['arm'],'status':status,'failure':failure,'phase':plan['phase'],
        'completed_updates':completed,'requested_updates':plan['updates'],'phase_seconds':phase_seconds,'phase_updates':phase_updates,
        'strategy_complete':failure is None and completed==plan['updates'] and problem is None,
        'model_optimizer_continuous':True,'monitors':monitors,'initial_monitor_cache':cache_audit,
        'last_checkpoint':last_path,'last_valid_milestone':last_valid,'plan_sha256':plan_sha,
        'shooting_cache_evidence':cache_meta,'phase_cost_estimates_with_margin':estimates,
        'source_before':before,'source_after':after,'sources_unchanged':before==after,'seconds':seconds,
        'cap_seconds':plan['cap_seconds'],'within_cap':time.perf_counter()<=deadline,
        'test_read':False,'validation_read':False,'checkpoint_selection':None}
    if before!=after or digest(plan_path)!=plan_sha:result['status']='quarantined_source_or_plan_drift';result['strategy_complete']=False
    if not result['within_cap']:result['status']='budget_exception';result['strategy_complete']=False
    write_json(output/'evidence.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--qa-only',action='store_true')
    mode.add_argument('--time-no-update',action='store_true')
    mode.add_argument('--identity-qa',action='store_true')
    mode.add_argument('--prepare-shooting-cache',action='store_true')
    mode.add_argument('--plan',type=Path)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rates',type=json.loads,help='explicit JSON skeleton/closure/encoder rates')
    p.add_argument('--clip-norm',type=float)
    p.add_argument('--batches',type=int)
    p.add_argument('--cap-seconds',type=float)
    p.add_argument('--shooting-cache',type=Path)
    p.add_argument('--continuity-weights',type=json.loads,help='explicit JSON 0.5/1.0/2.0 penalties')
    a=p.parse_args()
    if a.plan:
        result=run_plan(a.plan,a.output)
    elif a.identity_qa:
        if a.cap_seconds is None:p.error('identity QA requires cap-seconds')
        result=identity_qa(a.output,cap_seconds=a.cap_seconds)
    elif a.prepare_shooting_cache:
        if a.cap_seconds is None:p.error('source q-cache QA requires cap-seconds')
        result=prepare_shooting_cache(a.output,cap_seconds=a.cap_seconds)
    else:
        if a.arm is None or a.rates is None or a.cap_seconds is None or a.cap_seconds<=0:
            p.error('QA requires explicit arm, rates and cap-seconds')
        if a.time_no_update:
            if a.clip_norm is None or a.batches is None:p.error('timing requires clip-norm and batches')
            if a.arm.endswith('-M'):
                if a.shooting_cache is None or a.continuity_weights is None:p.error('M timing requires shooting-cache and continuity-weights')
                result=time_shooting_without_updates(a.arm,a.output,rates=a.rates,clip_norm=a.clip_norm,
                    batches=a.batches,cap_seconds=a.cap_seconds,shooting_cache=a.shooting_cache,
                    continuity_weights=a.continuity_weights)
            else:
                result=time_without_updates(a.arm,a.output,rates=a.rates,clip_norm=a.clip_norm,batches=a.batches,cap_seconds=a.cap_seconds)
        else:
            torch.set_num_threads(1); guard(PROCESS_STARTED+a.cap_seconds)
            model,rng,coverage,meta=restore_model_or_pair(a.arm); opt=optimizer_for(model,a.rates)
            result={'loop':'0102','arm':a.arm,'status':'restore_only_no_integration','parameter_updates':False,
                'source_step':meta['source_step'],'model_state_sha256':_model_hash(model),
                'coverage':coverage,'optimizer_parameter_names':parameter_layout(model,opt),
                'rng_sha256':rng_digest(capture_rng(rng)),'seconds':time.perf_counter()-PROCESS_STARTED,
                'test_read':False,'validation_read':False,'train_read':False}
            guard(PROCESS_STARTED+a.cap_seconds)
            a.output.mkdir(parents=True,exist_ok=False); write_json(a.output/'evidence.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','complete','seconds','failure')}),flush=True)
    if result.get('failure') or result.get('complete') is False or result.get('status') in ('budget_exception','quarantined_source_or_plan_drift'):
        raise SystemExit(1)


if __name__=='__main__':
    main()
