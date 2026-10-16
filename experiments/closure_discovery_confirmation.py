"""One-way new-train discrimination and frozen new-val confirmation."""
from pathlib import Path
import json
import time
import numpy as np
import torch


def evaluate_events(model,events,output):
    """Generic cohort adapter around the unchanged historical metric/grid helpers."""
    from discover_closure import write
    from evaluate_structured_rom import _rollouts,_pair_row,_scored_numerics,_metric_safe,_needs_finer
    from train_structured_rom import _point_error
    from train_local_memory_0098 import risk
    output=Path(output);rows=[];mass=model.mass.detach().numpy()
    for e in events:
        hist=e['x'][:11];batch=hist[None]
        # Preserve substeps for numerical checks. No true trajectory is extended.
        fine=_rollouts(model,batch,1600,.005)[0]
        coarse=_rollouts(model,batch,800,.01)[0]
        pair=_pair_row(fine,coarse,hist,e,e['classification_meta'],mass)
        scored=_scored_numerics(fine,coarse,e,mass)
        ref=fine[::2];selected_pair=pair
        if _needs_finer([pair]+[r for r in scored if r['available']]):
            finer=_rollouts(model,batch,3200,.0025)[0]
            selected_pair=_pair_row(finer,fine,hist,e,e['classification_meta'],mass,stride=2,output_dt=.005)
            scored=_scored_numerics(finer,fine,e,mass,stride=4,coarse_stride=2)
            # Primary prediction remains .005; finer is qualification only.
        numerical=not _needs_finer([selected_pair]+[r for r in scored if r['available']])
        metrics=[_metric_safe(ref,e,h,mass) for h in (1,2,5,8,'prefix')]
        actual=min(500,len(e['x'])-11)
        truth=torch.as_tensor(e['x'][11:11+actual],dtype=torch.float64)
        pred=torch.as_tensor(ref[1:1+actual],dtype=torch.float64)
        mse=float(_point_error(pred,truth,model).mean()) if actual>0 and np.isfinite(pred.numpy()).all() else None
        rows.append({'id':e['id'],'group':e['group'],'label':e['label'],'numerical_pass':numerical,
                     'actual_seconds':actual*.01,'complete_5s':actual==500,'mse':mse,
                     'metrics':metrics,'pair':pair,'selected_pair':selected_pair,'scored_numerics':scored})
    primary=[r for r in rows if r['label']=='stable' and r['complete_5s']]
    qualified=bool(primary) and all(r['numerical_pass'] and r['mse'] is not None for r in primary)
    summary={'records':rows,'event_count':len(events),'primary_complete_stable_count':len(primary),
             'primary_risk':float(risk(torch.as_tensor([r['mse'] for r in primary]))) if qualified else None,
             'all_numerical_pass':all(r['numerical_pass'] for r in rows),'test_read':False,
             'scope':'new frozen val under same simulator/system, not frozen original test or safety guarantee'}
    classified=[r for r in rows if r['numerical_pass']]
    summary['classification_counts']={
        'true_unstable':sum(r['label']=='unstable' for r in rows),
        'missed_unstable':sum(r['label']=='unstable' and not r['selected_pair']['fine_classification']['predicted_los'] for r in classified),
        'false_los':sum(r['label']=='stable' and r['selected_pair']['fine_classification']['predicted_los'] for r in classified),
        'numerically_unresolved':len(rows)-len(classified)}
    write(output,summary);return summary


def paired_primary(main,control):
    """Cluster bootstrap of the registered nonlinear risk, not point/time samples."""
    from train_local_memory_0098 import risk
    left={r['id']:r for r in main['records'] if r['label']=='stable' and r['complete_5s']}
    right={r['id']:r for r in control['records'] if r['label']=='stable' and r['complete_5s']}
    if set(left)!=set(right) or not left:return {'status':'unresolved','reason':'primary cohort mismatch/empty'}
    ids=sorted(left)
    if any(not left[k]['numerical_pass'] or not right[k]['numerical_pass'] or left[k]['mse'] is None or right[k]['mse'] is None for k in ids):
        return {'status':'unresolved','reason':'numerically unresolved primary event retained'}
    groups=sorted({left[k]['group'] for k in ids});rng=np.random.default_rng(5701);diff=[]
    def delta(chosen):
        return float(risk(torch.as_tensor([left[k]['mse'] for k in chosen]))-risk(torch.as_tensor([right[k]['mse'] for k in chosen])))
    for _ in range(2000):
        sampled=rng.choice(groups,len(groups),replace=True)
        chosen=[k for g in sampled for k in ids if left[k]['group']==g]
        diff.append(delta(chosen))
    low,high=np.quantile(diff,[.025,.975])
    return {'status':'complete','difference':delta(ids),'ci95':[float(low),float(high)],
            'supports_primary_improvement':bool(high<0),'events':len(ids),'groups':len(groups),'resamples':2000}


def run_confirmation(out,deadline):
    from discover_closure import write,sha,fit,train_inputs,score,source
    from symbolic_closure_rom import SymbolicClosureROM
    from closure_discovery_data import generate_conditions,load_new_events,select_new_train_conditions,allowed_train_conditions
    from train_local_memory_0098 import graph,histories_for
    out=Path(out);frozen=json.loads((out/'C/frozen.json').read_text());manifest=json.loads((out/'new_fault_manifest.json').read_text())
    main=SymbolicClosureROM.from_export(out/'C/main/2701/model.json')
    alternative=SymbolicClosureROM.from_export(out/'C/alternative/2701/model.json')
    old,_,scale=train_inputs();metadata={str(r['event_id']):r for r in allowed_train_conditions()}
    discrepancy=[]
    for e in old:
        hist=histories_for([(e,10)])
        with torch.no_grad(),graph(main):a=main.rollout(hist,100,.01)[...,:30]
        with torch.no_grad(),graph(alternative):b=alternative.rollout(hist,100,.01)[...,:30]
        discrepancy.append({**metadata[e['id']],'score':float((a-b).square().mean())})
    conditions=select_new_train_conditions(manifest['train_pool'],discrepancy,count=24)
    write(out/'D/selected_new_train.json',conditions)
    generated=generate_conditions(conditions,out/'D/new_data/train','train',max(1.,deadline-time.perf_counter()-60))
    new,excluded=load_new_events(out/'D/new_data/train','train')
    if not new:raise RuntimeError('no qualified new train; no replacement sampled')
    combined=old+new;endpoints=[]
    for row in frozen['models']:
        path=Path(row['path'])/'model.json'
        if sha(path)!=row['sha256']:raise ValueError('C source model changed')
        m=SymbolicClosureROM.from_export(path)
        dest=out/'D/models'/row['name']/str(row['seed'])
        m,_=fit(m,combined,scale,256,dest,row['seed'],deadline=deadline)
        endpoints.append({**row,'path':str(dest/'model.json'),'sha256':sha(dest/'model.json')})
    freeze=out/'D/confirmation_freeze.json'
    write(freeze,{'models':endpoints,'primary':'stable complete5s normalized event risk',
        'main_name':'main','main_seed':2701,'control_name':'fixed','control_seed':2701,
        'bootstrap_seed':5701,'bootstrap_resamples':2000,'test_read':False,
        'conditions':manifest['val24'],'conditions_sha256':sha(out/'new_fault_manifest.json')})
    # The val generator refuses to run without this complete immutable freeze.
    generated_val=generate_conditions(manifest['val24'],out/'D/new_data/val','val',max(1.,deadline-time.perf_counter()-60),freeze_path=freeze)
    val,excluded_val=load_new_events(out/'D/new_data/val','val',freeze_path=freeze)
    evaluations={}
    for row in endpoints:
        if deadline-time.perf_counter()<60:raise TimeoutError('confirmation incomplete; val must not be reused for selection')
        m=SymbolicClosureROM.from_export(row['path'])
        key=f"{row['name']}_{row['seed']}"
        evaluations[key]=evaluate_events(m,val,out/f'D/evaluation/{key}.json')
    primary=paired_primary(evaluations['main_2701'],evaluations['fixed_2701'])
    if excluded_val:
        primary={'status':'unresolved','reason':'new val has unavailable/numerically unqualified events; no aggregate improvement claim',
                 'excluded':excluded_val,'qualified_cohort_diagnostic':primary}
    write(out/'D/primary_comparison.json',primary)
    return {'stage':'D','new_train':len(new),'new_train_excluded':excluded,'new_val':len(val),
            'new_val_excluded':excluded_val,'frozen_sha256':sha(freeze),'primary':primary,
            'status':'frozen_confirmation_complete','test_read':False}
