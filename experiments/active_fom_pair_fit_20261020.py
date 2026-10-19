"""Finite fixed-structure continuation and one-way new-condition evaluation.

No discovery, checkpoint selection, historical val access or test access.
"""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
SEEDS = (6101, 6102, 6103)
RATES = {'skeleton': .003, 'closure': .0003, 'encoder': .00009}
CAMPAIGN = (ROOT / 'private_data/closure_mechanism_20261019/P3/active_fom_pair01').resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    from fixed_edge_pair_20261019 import write as writer
    writer(path, obj)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ordered(events):
    events = sorted(events, key=lambda e: e['id'])
    if not events or len({e['id'] for e in events}) != len(events):
        raise ValueError('nonempty unique event identities required')
    if any(len(e['x']) < 12 or e['label'] not in ('stable', 'unstable') for e in events):
        raise ValueError('genuine eligible histories/futures and physical labels required')
    return events


def event_identity(events):
    import numpy as np
    rows = []
    for e in ordered(events):
        row = {k: e[k] for k in ('id', 'group', 'label')}
        for k in ('x', 'torque'):
            a = np.ascontiguousarray(e[k])
            row[k] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                      'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
        rows.append(row)
    return rows


def freeze_schedules(events, output):
    import numpy as np
    from train_joint_numerical_pair_0100 import next_samples
    events = ordered(events)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    result = {'events': event_identity(events), 'seeds': {}}
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 1000)
        coverage = {'cover_order': rng.permutation(len(events)).tolist(), 'cursor': 0}
        initial_rng, initial_coverage = copy.deepcopy(rng.bit_generator.state), copy.deepcopy(coverage)
        rows = []
        for relative in range(1, 257):
            before = copy.deepcopy(rng.bit_generator.state)
            samples, sampling = next_samples(events, rng, coverage)
            h = (1., 2., 5.)[(relative-1) % 3]
            rows.append({'relative_step': relative, 'global_step': 1024+relative,
                         'horizon_s': h, 'sampling': sampling, 'numpy_rng_before': before,
                         'numpy_rng_after': copy.deepcopy(rng.bit_generator.state),
                         'long_counts': [min(round(h*100), len(e['x'])-11) for e, _ in samples],
                         'local_counts': [min(20, len(e['x'])-i-1) for e, i in samples]})
        included = sorted({eid for r in rows for eid in r['sampling']['sample_ids']})
        if included != [e['id'] for e in events]:
            raise ValueError('256-update frozen schedule fails complete event coverage')
        result['seeds'][str(seed)] = {'rows': rows, 'sampler_seed': seed+1000,
            'initial_rng': initial_rng, 'initial_coverage': initial_coverage,
            'included_ids': included, 'final_coverage': coverage,
            'final_rng': rng.bit_generator.state}
    write(output/'schedule.json', result)
    return result


def validate_resume_step(global_step):
    if global_step not in range(1024, 1281, 32):
        raise ValueError('only immutable registered 32-step boundaries can resume')
    return global_step - 1024


def validate_trace(trace, relative):
    if not isinstance(trace,list) or len(trace)!=relative:
        raise ValueError('new-stage raw trace length differs from checkpoint position')
    for i,row in enumerate(trace,1):
        if row.get('relative_step')!=i or row.get('step')!=1024+i or row.get('horizon_s')!=(1.,2.,5.)[(i-1)%3]:
            raise ValueError('new-stage trace relative/global/horizon sequence changed')


def adapt_one(sourceoutput, events, newoutput, seed, structure, max_seconds,
              sourceidentity, newstageidentity, recovery=False, schedule_path=None,
              previous_seconds=0.):
    from linear_cache_cost_gate_20261019 import load_runtime, restore_cached, one_update
    from fixed_edge_pair_20261019 import save_safe, verify_identity
    from closure_training_runtime import TensorEventStore, synchronize
    from train_joint_numerical_pair_0100 import next_samples
    load_runtime()
    import torch
    import numpy as np
    if seed not in SEEDS or structure not in ('full8', 'delete7'):
        raise ValueError('unregistered model')
    if not math.isfinite(max_seconds) or not math.isfinite(previous_seconds) or previous_seconds < 0 or not 0 < max_seconds <= 3000-previous_seconds:
        raise ValueError('all-attempt model envelope exceeds 3000 seconds')
    started = time.perf_counter()
    events = ordered(events); output = Path(newoutput)
    output.mkdir(parents=True, exist_ok=recovery)
    attempt = output / ('recovery' if recovery else 'initial'); attempt.mkdir(exist_ok=False)
    worker_identity(attempt)
    worker_identity(output)
    if sourceidentity.get('seed') != seed or sourceidentity.get('structure') != structure:
        raise ValueError('source seed/structure identity differs')
    schedule_path = Path(schedule_path) if schedule_path else output.parent/'schedules/schedule.json'
    schedules = read(schedule_path)
    verify_identity(event_identity(events), schedules['events'])
    schedule = schedules['seeds'][str(seed)]
    identity = {'stage': newstageidentity, 'source_identity': sourceidentity, 'seed': seed,
                'structure': structure, 'schedule_sha256': sha(schedule_path),
                'events': schedules['events'], 'rates': RATES, 'source_step': 1024}
    checkpoint = output/'checkpoints'; checkpoint.mkdir(exist_ok=recovery)
    device = torch.device('cuda', torch.cuda.current_device())
    if recovery:
        model, opt, rng, coverage, meta, changed = restore_cached(checkpoint, device, identity)
        if changed:
            raise ValueError('recovery environment continuation is forbidden')
        relative = validate_resume_step(meta['step'])
        trace = meta['trace']
        validate_trace(trace,relative)
    else:
        model, opt, old_rng, old_coverage, meta, changed = restore_cached(sourceoutput, device, sourceidentity)
        if changed:
            raise ValueError('source environment continuation is forbidden')
        if meta['step'] != 1024:
            raise ValueError('source must be original complete 1024 endpoint')
        for field in ('model_sha256','training_sha256'):
            expected=newstageidentity.get('source_'+field)
            if expected is not None and expected!=meta[field]:
                raise ValueError('registered P2 source bytes changed')
        write(attempt/'source_lineage.json', {'checkpoint': str(Path(sourceoutput).resolve()),
              'model_sha256': meta['model_sha256'], 'training_sha256': meta['training_sha256'],
              'old_numpy_rng': old_rng.bit_generator.state, 'old_coverage': old_coverage})
        rng = np.random.default_rng(); rng.bit_generator.state = schedule['initial_rng']
        coverage = copy.deepcopy(schedule['initial_coverage']); relative = 0; trace = []
        save_safe(checkpoint, model, opt, rng, coverage, 1024, identity, trace)
    if changed:
        raise ValueError('environment continuation is forbidden')
    store = TensorEventStore(events, device)
    force = float(newstageidentity['force_scale'])
    state = {'status': 'RUNNING', 'last_completed_relative': relative,
             'last_safe_relative': relative, 'seed': seed, 'structure': structure}
    try:
        for rel in range(relative+1, 257):
            if time.perf_counter()-started >= max_seconds-60:
                raise TimeoutError('model envelope: stop before next update')
            row = schedule['rows'][rel-1]
            if row['relative_step'] != rel or row['global_step'] != 1024+rel:
                raise ValueError('relative/global schedule mismatch')
            verify_identity(rng.bit_generator.state, row['numpy_rng_before'])
            samples, sampling = next_samples(events, rng, coverage)
            verify_identity(sampling, row['sampling'])
            verify_identity(rng.bit_generator.state, row['numpy_rng_after'])
            if [min(round(row['horizon_s']*100), len(e['x'])-11) for e, _ in samples] != row['long_counts'] or [min(20,len(e['x'])-i-1) for e,i in samples] != row['local_counts']:
                raise ValueError('genuine sampled endpoints changed')
            for g in opt.param_groups:
                g['lr'] = RATES[g['block']]
            result = one_update(model, opt, store, samples, row['horizon_s'], force)
            trace.append({'relative_step': rel, 'step': 1024+rel, 'sampling': sampling,
                          'horizon_s': row['horizon_s'], **result})
            state['last_completed_relative'] = rel
            if rel % 32 == 0:
                save_safe(checkpoint, model, opt, rng, coverage, 1024+rel, identity, trace)
                state['last_safe_relative'] = rel
            write(attempt/'progress.json', state)
        synchronize(device)
        state['status'] = 'COMPLETED'
    except BaseException as exc:
        state.update(status='UNRESOLVED', error=repr(exc))
        raise
    finally:
        state['elapsed_seconds'] = time.perf_counter()-started
        state['all_attempt_seconds'] = previous_seconds+state['elapsed_seconds']
        write(attempt/'result.json', state)
    return state


def worker_identity(output):
    from closure_p3_disagreement_20261020 import own_creation_filetime_utc
    write(Path(output)/'worker.json', {'pid': os.getpid(),
          'creation_filetime_utc': own_creation_filetime_utc(), 'argv': sys.argv})


def weighted_risk(values, weights=None):
    """80% weighted mean + 20% top-10%-weight mean, fractional boundary."""
    import numpy as np
    a = np.asarray(values, dtype=float)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    if a.ndim != 1 or w.shape != a.shape or len(a)==0 or not np.isfinite(a).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('finite nonempty values and positive aligned weights required')
    w = w/w.sum(); remaining = .1; tail = 0.
    for i in np.argsort(-a, kind='stable'):
        take = min(float(w[i]), remaining); tail += take*float(a[i]); remaining -= take
        if remaining <= 1e-15:
            break
    return float(.8*(a*w).sum()+.2*tail/.1)


def interval15(model, event, fine, coarse):
    import numpy as np
    import torch
    from evaluate_structured_rom import SCALES
    from train_structured_rom import _point_error
    mass = model.mass.detach().cpu().numpy(); pscale = model.p_scale.detach().cpu().numpy()
    n = min(1500, len(event['x'])-11)
    row = {'n_time_points': n, 'last_record_time_s': .1+n*.01,
           'available': n > 0, 'finite': False, 'numerically_admissible': False}
    if n <= 0:
        return row
    f, c = fine[::2][:n+1], coarse[:n+1]
    finite = bool(np.isfinite(fine[:2*n+1]).all() and np.isfinite(c).all())
    if not finite:
        return row
    with np.errstate(over='ignore', invalid='ignore'):
        delta = f-c; de = delta[:,:10]; we = delta[:,10:20]
        dc = de-(de*mass).sum(1,keepdims=True)/mass.sum()
        wc = we-(we*mass).sum(1,keepdims=True)/mass.sum()
        cn = float(np.sqrt(.5*np.mean((dc/SCALES[0])**2+(wc/SCALES[1])**2)))
        cm = float(np.sqrt(np.mean(((we*mass).sum(1)/mass.sum()/SCALES[1])**2)))
        pred, truth = f[1:,:30], event['x'][11:11+n,:30]
        common = ((pred[:,10:20]-truth[:,10:20])*mass).sum(1)/mass.sum()
        power = pred[:,20:30]-truth[:,20:30]
        with torch.no_grad():
            point = float(_point_error(torch.as_tensor(pred), torch.as_tensor(truth), model).mean())
        metrics = {'coi_normalized_rmse': cn, 'common_normalized_rmse': cm,
                   'mean_point_error': point, 'frequency_rmse_rad_s': float(np.sqrt(np.mean(common**2))),
                   'power_rmse_system_pu': float(np.sqrt(np.mean(power**2))),
                   'power_rmse_MW': float(100*np.sqrt(np.mean(power**2))),
                   'power_normalized_rmse': float(np.sqrt(np.mean((power/pscale)**2)))}
    row.update(metrics); row['finite'] = bool(np.isfinite(list(metrics.values())).all())
    row['numerically_admissible'] = bool(row['finite'] and cn <= .001 and cm <= .001)
    return row


def actual_interval_predictions(event, fine, coarse):
    """Preserve every internal substep through the genuine observed endpoint."""
    n=min(1500,max(0,len(event['x'])-11))
    return fine[:2*n+1],coarse[:n+1],n


def evaluate_queue(model, events, output, provenance, check, *, export_analysis=True,
                   frozen_predictions=None, frozen_model_key=None):
    import numpy as np
    import torch
    from evaluate_structured_rom import _rollouts, _pair_row, _safe
    from summarize_train_local_memory_0098 import event_metrics, build_summary
    from fixed_edge_pair_20261019 import scientific_qualification, primary_risk, physical_metrics
    from linear_cache_runtime_20261019 import enable_linear_cache
    torch.set_num_threads(1)
    model = copy.deepcopy(model).cpu().double().eval(); enable_linear_cache(model)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    worker_identity(output)
    events = ordered(events); mass = model.mass.detach().numpy(); ps = model.p_scale.detach().numpy()
    records = {k: [] for k in ('records','common_frequency','normalized_p','objective_point_error','numerical')}
    extras, qualifications, prediction_files = [], {}, []
    frozen = None
    if frozen_predictions is not None:
        if frozen_model_key not in {f'{s}_{k}' for s in SEEDS for k in ('full8','delete7')}:
            raise ValueError('frozen prediction source model key required')
        frozen = {r['id']:r for r in frozen_predictions if r['model']==frozen_model_key}
        if len(frozen)!=24:
            raise ValueError('all 24 pre-FOM prediction attempts must be retained')
    # Small fixed batches; an inference exception is retained per event, never dropped.
    for first in range(0, len(events), 16):
        check(); block = events[first:first+16]
        histories = np.stack([e['x'][:11] for e in block])
        error = None
        try:
            if frozen is None:
                fine = _rollouts(model, histories, 3000, .005)
                coarse = _rollouts(model, histories, 1500, .01)
            else:
                fine=[]; coarse=[]
                for e,h in zip(block,histories):
                    eid=e['id'].removeprefix('newtrain_')
                    row=frozen[eid]
                    if row['path'] is None:
                        f=np.full((3001,40),np.nan); c=np.full((1501,40),np.nan)
                    else:
                        if sha(row['path'])!=row['sha256']:
                            raise ValueError('pre-FOM frozen prediction hash changed')
                        with np.load(row['path'],allow_pickle=False) as data:
                            f=data['fine'].copy(); c=data['coarse'].copy()
                            # Acquisition/alignment tolerance is checked by the train
                            # loader qualification. These remain the original early
                            # predictions; never replace them using later histories.
                            if data['history'].shape!=(11,30):
                                raise ValueError('frozen causal history shape changed')
                    if f.shape!=(3001,40) or c.shape!=(1501,40):
                        raise ValueError('full15s original prediction shape changed')
                    fine.append(f); coarse.append(c)
                fine=np.stack(fine); coarse=np.stack(coarse)
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
              'prediction_origin':'before_full_FOM' if frozen is not None else 'endpoint_autonomous_inference'}
    if export_analysis:
        from train_closure import export_cpu
        from analyze_symbolic_closure import analyze_model
        export_cpu(model,output/'model.json')
        local, arrays = analyze_model(model,source_sha256=sha(output/'model.json'),source_path=str(output/'model.json'))
        np.savez_compressed(output/'local.npz',**arrays); write(output/'local.json',_safe(local))
    write(output/'evidence.json',_safe(result))
    write(output/'complete.json',{'status':'COMPLETED','event_count':len(events),'evidence_sha256':sha(output/'evidence.json')})
    return result


def freeze_val(models, val_conditions, condition_manifest_path, output):
    from closure_discovery_data import _attempt_record
    original=read(condition_manifest_path)
    if isinstance(original,dict):
        original=original.get('val_conditions',original.get('conditions',original.get('val')))
    normalized=[_attempt_record(r,'val') for r in val_conditions]
    expected_conditions=[_attempt_record(r,'val') for r in original]
    if normalized!=expected_conditions:
        raise ValueError('val conditions differ from prelocked condition file')
    val_conditions=normalized
    expected = {(s,k) for s in SEEDS for k in ('full8','delete7')}
    if len(models)!=6 or {(int(r['seed']),r['structure']) for r in models} != expected:
        raise ValueError('all six distinct complete endpoints required')
    rows=[]
    for r in models:
        p=Path(r['path']).resolve()
        if p.name!='model.json' or sha(p)!=r['sha256']:
            raise ValueError('final model hash/path mismatch')
        meta=read(p.parent/'checkpoint.json')
        if meta['step']!=1280 or meta['model_sha256']!=r['sha256'] or meta['identity']['seed']!=r['seed'] or meta['identity']['structure']!=r['structure']:
            raise ValueError('endpoint must be complete relative256/global1280')
        rows.append({**r,'path':str(p)})
    if len(val_conditions)!=24 or len({r['event_id'] for r in val_conditions})!=24:
        raise ValueError('exactly 24 frozen val attempts required')
    payload={'models':rows,'conditions':val_conditions,'conditions_sha256':sha(condition_manifest_path),
        'primary_seed':6101,'contrast':'delete7-full8','test_read':False,
        'primary':'design-weighted stable complete5s risk: .8 mean+.2 fractional upper10% mass',
        'weights':'one/24 per designed attempt; condition on actually qualified stable events',
        'inference':'point estimates and paired seed directions; no significance or bootstrap'}
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    write(output,payload)
    from closure_discovery_data import _freeze_snapshot
    _freeze_snapshot(output)
    return payload


def evaluate_frozen_prediction_queue(model_for_metric_scales,events,frozen_inventory,
                                     modelkey,output,provenance,check,**kwargs):
    """Offline score only: this path never calls a model's rollout."""
    rows=frozen_inventory['frozen_predictions'] if isinstance(frozen_inventory,dict) else frozen_inventory
    return evaluate_queue(model_for_metric_scales,events,output,provenance,check,
        frozen_predictions=rows,frozen_model_key=modelkey,**kwargs)


def paired_summary(evidence, attempted_count=24):
    import numpy as np
    pairs=[]
    for seed in SEEDS:
        models={k:evidence[f'{seed}_{k}'] for k in ('full8','delete7')}
        scores={}; ids={}; counts={}
        for structure,e in models.items():
            q8={x['event_id']:x['scientific_numeric_8s'] for x in e['extras']}
            selected=[r for r in e['rows']['objective_point_error'] if r['horizon']=='5' and r['label']=='stable' and r['available']]
            good=[r for r in selected if r['finite'] and r['numerically_admissible'] and q8.get(r['event_id'],False) and r.get('mean_point_error') is not None]
            unresolved=[r['event_id'] for r in selected if r not in good]
            values=[r['mean_point_error'] for r in good]
            scores[structure]={'design_weighted_conditional_stable5s':weighted_risk(values) if values and not unresolved else None,
                'legacy_ceil_stable5s':float(.8*np.mean(values)+.2*np.mean(np.sort(values)[-max(1,math.ceil(.1*len(values))):])) if values and not unresolved else None,
                'stable_eligible':len(selected),'stable_qualified':len(good),'unresolved_ids':unresolved}
            ids[structure]={r['event_id']:r['mean_point_error'] for r in good}
            counts[structure]=e['event_count']
        full=scores['full8']['design_weighted_conditional_stable5s']; delete=scores['delete7']['design_weighted_conditional_stable5s']
        delta=delete-full if full is not None and delete is not None and ids['full8'].keys()==ids['delete7'].keys() else None
        pairs.append({'seed':seed,'models':scores,'delete7_minus_full8':delta,
            'paired_events':[{'event_id':i,'full8':ids['full8'][i],'delete7':ids['delete7'][i]} for i in sorted(ids['full8'].keys()&ids['delete7'].keys())],
            'whole_24_resolved':all(counts[k]==attempted_count and not scores[k]['unresolved_ids'] and all(x.get('exception') is None and x['actual_interval15']['numerically_admissible'] and x['classification_15s']['los_pair_pass'] and x['classification_15s']['finite_to_endpoint'] and x['classification_15s']['coi_normalized_rmse']<=.001 and x['classification_15s']['common_normalized_rmse']<=.001 for x in models[k]['extras']) for k in counts)})
    deltas=[p['delete7_minus_full8'] for p in pairs]
    return {'attempted_count':attempted_count,'pairs':pairs,'primary_seed':6101,
        'all_three_same_direction':all(d is not None and d<0 for d in deltas) or all(d is not None and d>0 for d in deltas),
        'delete7_improves_all_three':all(d is not None and d<0 for d in deltas),
        'delete7_worsens_all_three':all(d is not None and d>0 for d in deltas),
        'scope':'conditional fixed design; no formal improvement claim or model adoption'}


def load_events(spec):
    from closure_discovery_data import load_new_events
    from fixed_edge_pair_20261019 import metadata
    split=spec['split']
    if split not in ('train','val') or (split=='val' and spec.get('old_train')):
        raise ValueError('only original train and current-stage explicit train/val permitted')
    if split=='val':
        freeze_path=Path(spec['freeze_path']).resolve()
        if not freeze_path.is_relative_to(CAMPAIGN):
            raise ValueError('new val requires this campaign freeze manifest')
        frozen=read(freeze_path)
        if len(frozen.get('models',[]))!=6 or {(r['seed'],r['structure']) for r in frozen['models']} != {(s,k) for s in SEEDS for k in ('full8','delete7')}:
            raise ValueError('new val requires all six registered final endpoints')
    events=[]; excluded=[]; force=None
    if spec.get('old_train'):
        from closure_mechanism_study import train_cohort
        events,force,_=train_cohort(); metadata(events)
        if len(events)!=366:
            raise ValueError('original eligible train denominator changed')
    for directory in spec.get('new_directories',[]):
        path=Path(directory).resolve()
        if not path.is_relative_to(CAMPAIGN) or split not in path.parts:
            raise ValueError('new data must be explicitly inside this campaign split directory')
        added, rejected=load_new_events(path,split,freeze_path=spec.get('freeze_path'))
        events+=added; excluded+=rejected
    return ordered(events),excluded,force


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['schedule','adapt','evaluate','freeze-val','summary'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--events-spec',type=Path)
    p.add_argument('--max-seconds',type=float,required=True); p.add_argument('--source-output',type=Path)
    p.add_argument('--source-identity',type=Path); p.add_argument('--stage-identity',type=Path)
    p.add_argument('--schedule-path',type=Path); p.add_argument('--model',type=Path)
    p.add_argument('--frozen-predictions',type=Path); p.add_argument('--frozen-model-key')
    p.add_argument('--seed',type=int); p.add_argument('--structure'); p.add_argument('--recovery',action='store_true')
    p.add_argument('--previous-seconds',type=float,default=0.); p.add_argument('--manifest',type=Path)
    args=p.parse_args(); started=time.perf_counter()
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0:
        raise ValueError('finite positive envelope required')
    if args.mode=='schedule':
        events,_,_=load_events(read(args.events_spec)); freeze_schedules(events,args.output); worker_identity(args.output)
    elif args.mode=='adapt':
        events,_,_=load_events(read(args.events_spec))
        adapt_one(args.source_output,events,args.output,args.seed,args.structure,args.max_seconds,
            read(args.source_identity),read(args.stage_identity),args.recovery,args.schedule_path,args.previous_seconds)
    elif args.mode=='evaluate':
        from linear_cache_cost_gate_20261019 import load_runtime
        load_runtime()
        from closure_training_runtime import TrainingClosureROM
        events,excluded,_=load_events(read(args.events_spec))
        model=TrainingClosureROM.from_export(args.model)
        def check():
            if time.perf_counter()-started>args.max_seconds-60:
                raise TimeoutError('evaluation envelope')
        spec=read(args.events_spec)
        frozen_path=args.frozen_predictions or spec.get('frozen_source_inventory')
        frozen=read(frozen_path) if frozen_path else None
        if isinstance(frozen,dict):
            frozen=frozen['frozen_predictions']
        result=evaluate_queue(model,events,args.output,{'model_path':str(args.model),'model_sha256':sha(args.model),'excluded':excluded},check,
            frozen_predictions=frozen,frozen_model_key=args.frozen_model_key or spec.get('modelkey'))
        worker_identity(args.output)
    elif args.mode=='freeze-val':
        args.output.mkdir(parents=True,exist_ok=False); worker_identity(args.output)
        m=read(args.manifest); freeze_val(m['models'],m['conditions'],m['condition_manifest_path'],args.output/'frozen.json')
    else:
        args.output.mkdir(parents=True,exist_ok=False); worker_identity(args.output)
        m=read(args.manifest); write(args.output/'summary.json',paired_summary({k:read(v) for k,v in m['evidence'].items()},m.get('attempted_count',24)))


if __name__=='__main__':
    main()
