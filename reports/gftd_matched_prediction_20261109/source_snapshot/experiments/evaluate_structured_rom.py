"""Fixed-endpoint, validation-only evaluator for the Loop 0078 structure screen.

The public ``evaluate`` function is deliberately model-object based: loading a
0078 checkpoint is owned by the trainer, while this module owns no model
family assumptions beyond the documented rollout/history interface.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'experiments')); sys.path.insert(0,str(ROOT/'src'))
from evaluate_long_latent import _metric_row, _groups, aggregate, DT, SCALES
from evaluate_l1_latent import common_frequency_metrics, restore_physical_path, classify_prediction

HORIZONS=(1,2,5,8)

def _safe(v):
    if isinstance(v,dict): return {str(k):_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_safe(x) for x in v]
    if isinstance(v,np.ndarray): return _safe(v.tolist())
    if isinstance(v,(np.floating,float)): return float(v) if np.isfinite(v) else None
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.bool_,bool)): return bool(v)
    return v

def _rollouts(model, histories, steps, dt):
    """One batched inference call; structure trainers use torch histories."""
    import torch
    h=torch.as_tensor(np.asarray(histories),dtype=torch.float64)
    with torch.no_grad():
        if hasattr(model,'begin_graph_cache'): model.begin_graph_cache()
        try: path=model.rollout(h,int(steps),float(dt))
        finally:
            if hasattr(model,'end_graph_cache'): model.end_graph_cache()
    path=path.detach().cpu().numpy() if hasattr(path,'detach') else np.asarray(path,dtype=np.float64)
    if path.ndim!=3 or path.shape[0]!=len(h) or path.shape[1]!=steps+1: raise ValueError('rollout must return [batch,steps+1,state]')
    return path

def _truth(event):
    x=np.asarray(event['x'],dtype=np.float64)
    if x.ndim!=2 or x.shape[1] < 30 or len(x)<11: raise ValueError('event requires canonical x with causal 11-point history')
    return x[10:]

def _classification(path, history, meta, mass, dt=.01):
    sync=np.r_[np.asarray(history)[:10,:20],path[:,:20]]
    raw=restore_physical_path(sync,np.asarray(meta['equilibrium_delta']),float(meta.get('omega_s',2*np.pi*60)))
    return classify_prediction(raw,np.asarray(meta.get('inertia',mass)),true_event_time_s=meta.get('event_time_s'),config={'sample_interval_s':dt,'los_hold_samples':max(1,round(.05/dt))})

def _pair_row(fine, coarse, history, event, meta, mass, *, stride=2, output_dt=.01):
    """Compare .005/.01 only through the relevant observed/LOS interval."""
    fine10=fine[::stride]; assert len(fine10)==len(coarse)
    cf,cc=_classification(fine10,history,meta,mass,output_dt),_classification(coarse,history,meta,mass,output_dt)
    if cf['category']=='latency_uncovered' or cc['category']=='latency_uncovered': limit=len(coarse)-1; los_ok=True
    elif cf['predicted_los'] and cc['predicted_los']:
        limit=max(cf['trigger_index'],cc['trigger_index'])-10; los_ok=abs(cf['trigger_index']-cc['trigger_index'])*output_dt <= .02
    elif not cf['predicted_los'] and not cc['predicted_los']:
        limit=len(coarse)-1; los_ok=True
    else: limit=len(coarse)-1; los_ok=False
    limit=max(0,min(limit,len(coarse)-1)); a,b=fine10[:limit+1],coarse[:limit+1]
    # A reported LOS is numerically admissible only if every fine substep up
    # to the comparison endpoint is finite, not merely the decimated values.
    finite=bool(np.isfinite(fine[:stride*limit+1]).all() and np.isfinite(b).all())
    if finite:
        diff=a-b; de=diff[:,:10]; we=diff[:,10:20]; coi=(de-(de*mass).sum(1,keepdims=True)/mass.sum())
        coi_norm=float(np.sqrt(.5*np.mean((coi/SCALES[0])**2+(we-(we*mass).sum(1,keepdims=True)/mass.sum())**2/SCALES[1]**2)))
        common=float(np.sqrt(np.mean(((we*mass).sum(1)/mass.sum()/SCALES[1])**2)))
    else: coi_norm=common=float('inf')
    return {'event_id':event['id'],'group':event['group'],'label':event['label'],'finite_to_endpoint':finite,'full_fine_finite':bool(np.isfinite(fine).all()),'full_coarse_finite':bool(np.isfinite(coarse).all()),'comparison_last_step':int(limit),'coi_normalized_rmse':coi_norm,'common_normalized_rmse':common,'los_pair_pass':bool(los_ok),'fine_classification':cf,'coarse_classification':cc}

def _needs_finer(rows):
    return any((not r['finite_to_endpoint']) or (not r['los_pair_pass']) or r['coi_normalized_rmse']>.001 or r['common_normalized_rmse']>.001 for r in rows)

def _scored_numerics(fine,coarse,event,mass,stride=2,coarse_stride=1):
    # Both output paths are on .01s; retain each original substep grid for QA.
    fine_full, coarse_full = fine, coarse
    out=[]; fine=fine[::stride]; coarse=coarse[::coarse_stride]; available=len(event['x'])-11
    if fine.shape != coarse.shape: raise ValueError('numerical output grids differ')
    for h in (*HORIZONS,'prefix'):
        steps=min(800,available) if h=='prefix' else round(float(h)/DT)
        if steps<=0 or (h!='prefix' and available<steps): out.append({'event_id':event['id'],'horizon':str(h),'available':False}); continue
        a,b=fine[:steps+1],coarse[:steps+1]
        finite=bool(np.isfinite(fine_full[:stride*steps+1]).all() and np.isfinite(coarse_full[:coarse_stride*steps+1]).all())
        if finite:
            de=a[:,:10]-b[:,:10]; we=a[:,10:20]-b[:,10:20]; coi=de-(de*mass).sum(1,keepdims=True)/mass.sum(); wcoi=we-(we*mass).sum(1,keepdims=True)/mass.sum(); common=(we*mass).sum(1)/mass.sum()
            cn=float(np.sqrt(.5*np.mean((coi/SCALES[0])**2+(wcoi/SCALES[1])**2))); cm=float(np.sqrt(np.mean((common/SCALES[1])**2)))
        else: cn=cm=float('inf')
        out.append({'event_id':event['id'],'horizon':str(h),'available':True,'finite_to_endpoint':finite,'los_pair_pass':True,'coi_normalized_rmse':cn,'common_normalized_rmse':cm})
    return out

def _metric_safe(pred,event,horizon,mass):
    requested=len(event['x'])-11 if horizon=='prefix' else round(float(horizon)/DT)
    available=len(event['x'])-11
    if requested <= 0 or (horizon!='prefix' and available < requested):
        return {'event_id':event['id'],'group':event['group'],'label':event['label'],'horizon':str(horizon),'scope':'available_prefix' if horizon=='prefix' else 'complete','available':False,'finite':None}
    steps=min(requested,len(pred)-1)
    if not np.isfinite(pred[:steps+1]).all():
        return {'event_id':event['id'],'group':event['group'],'label':event['label'],'horizon':str(horizon),'scope':'available_prefix' if horizon=='prefix' else 'complete','available':True,'finite':False,'normalized_sse':None,'rmse':None}
    row=_metric_row(pred,event,horizon,mass); row.update(available=True,finite=True); return row

def evaluate(model, events, starts, *, output=None, metadata=None, run_finer=True):
    """Evaluate explicit validation events; it neither reads datasets nor TEST."""
    events=list(events); mass=np.asarray(model.mass,dtype=np.float64); metadata=dict(metadata or {})
    if len(events)!=107: raise ValueError('0078 requires exactly 107 explicit eligible validation events')
    if set(e['id'] for e in events)!=set(starts): raise ValueError('starts must cover exactly the evaluated IDs')
    histories=np.stack([np.asarray(starts[e['id']]['history'],dtype=np.float64) for e in events])
    fine=_rollouts(model,histories,1600,.005); coarse=_rollouts(model,histories,800,.01); rows=[]; common=[]; numerical=[]; scored_numerical=[]; classifications=[]
    for i,e in enumerate(events):
        hist=histories[i]; truth=_truth(e); f=fine[i]; c=coarse[i]; f10=f[::2]
        for h in (*HORIZONS,'prefix'): rows.append(_metric_safe(f10,e,h,mass))
        for cr in common_frequency_metrics(f10,truth,mass):
            steps=round(float(cr['horizon'])/DT) if cr['horizon']!='prefix' else min(len(truth)-1,800)
            cr.update(event_id=e['id'],group=e['group'],label=e['label'],finite=bool(np.isfinite(f10[:steps+1]).all()))
            if not cr['finite']: cr.update(sse=None,rmse=None)
            common.append(cr)
        nr=_pair_row(f,c,hist,e,starts[e['id']],mass); numerical.append(nr); scored_numerical.extend(_scored_numerics(f,c,e,mass)); classifications.append({'event_id':e['id'],'fine':nr['fine_classification'],'coarse':nr['coarse_classification']})
    trigger=_needs_finer(numerical+[r for r in scored_numerical if r['available']]); finer=None; finer_rows=[]; finer_scored=[]
    if trigger and run_finer:
        finer=_rollouts(model,histories,3200,.0025)
        for i,e in enumerate(events):
            q=finer[i]; hist=histories[i]
            # .0025 has two substeps per .005, evaluated through the same LOS-aware rule.
            fr=_pair_row(q, fine[i,::2], hist, e, starts[e['id']], mass, stride=4, output_dt=.01)
            # The compared .005 path also has unobserved substeps; retain its
            # endpoint finiteness rather than treating decimation as evidence.
            fr['old_fine_substeps_finite_to_endpoint']=bool(np.isfinite(fine[i,:2*fr['comparison_last_step']+1]).all())
            fr['finite_to_endpoint']=bool(fr['finite_to_endpoint'] and fr['old_fine_substeps_finite_to_endpoint'])
            finer_rows.append(fr)
            finer_scored.extend(_scored_numerics(q,fine[i],e,mass,stride=4,coarse_stride=2))
    complete={str(h):[r for r in rows if r['horizon']==str(h)] for h in HORIZONS}; prefix=[r for r in rows if r['horizon']=='prefix']
    valid=lambda rs:[r for r in rs if r.get('available') and r.get('finite')]
    def strict_summary(rs):
        eligible = [r for r in rs if r.get('available')]
        return aggregate(eligible) if eligible and all(r.get('finite') for r in eligible) else None
    official={h:strict_summary(v) for h,v in complete.items()}
    finer_pass=(finer is not None and not _needs_finer(finer_rows+[r for r in finer_scored if r['available']]))
    selected_numeric = finer_rows if finer is not None else numerical
    selected_scored = finer_scored if finer is not None else scored_numerical
    model_numeric_pass = not _needs_finer(selected_numeric + [r for r in selected_scored if r['available']])
    gated=[]
    for e,nr,comparison in zip(events,numerical,selected_numeric,strict=True):
        # The .005s forecast is fixed as the reporting reference. Refinement
        # diagnoses its accuracy; it never substitutes a selected trajectory.
        c=dict(nr['fine_classification']); ok=not _needs_finer([comparison])
        if not ok:
            c.update(category='numerical_unresolved',predicted_los=False,window_settled=False,
                     unreliable_trigger_index=c.get('trigger_index'),trigger_index=None)
        hit=c.get('trigger_index') if ok else None
        meta=starts[e['id']]; delay=float(meta.get('record_start_delay_s') or 0.)
        record_time=None if hit is None else float(hit)*DT
        true_time=meta.get('event_time_s')
        gated.append({'event_id':e['id'],'label':e['label'],'classification':c,'numerically_admissible':bool(ok),
                      'predicted_los_time_record_s':record_time,
                      'predicted_los_time_since_clearing_s':None if record_time is None else record_time+delay,
                      'true_event_time_record_s':true_time,
                      'trigger_time_error_s':None if record_time is None or true_time is None else record_time-float(true_time)})
    eligible=[x for x in gated if x['classification']['category']!='latency_uncovered']
    l1={'eligible':len(eligible),'detected_unstable':sum(x['label']=='unstable' and x['classification'].get('predicted_los') for x in eligible),'false_los':sum(x['label']!='unstable' and x['classification'].get('predicted_los') for x in eligible),'false_window_settled':sum(x['label']=='unstable' and x['classification'].get('window_settled') for x in eligible),'numerical_unresolved':sum(x['classification']['category']=='numerical_unresolved' for x in eligible)}
    l1['unstable_eligible_denominator']=sum(e['label']=='unstable' for e in events)
    l1['raw_validation_denominator']=117; l1['latency_uncovered_count']=10
    l1['raw_unstable_denominator']=22
    evidence={'loop':'0078','test_read':False,'split_guard':{'explicit_validation_ids':len(events),'expected_eligible_validation':107,'raw_validation_denominator':117,'latency_excluded_denominator':10,'test_enumerated':False},'dt':{'fine':.005,'coarse':.01,'finer':.0025 if finer is not None else None},'forecast_reference':'fixed .005s; finer is convergence diagnostic only','horizons':list(HORIZONS),'complete_counts':{h:{'all_rows':len(v),'eligible_cohort':sum(bool(r.get('available')) for r in v),'finite_scored':len(valid(v)),'unavailable_or_nonfinite':len(v)-len(valid(v))} for h,v in complete.items()},'summary':official,'finite_subcohort_diagnostic':{h:aggregate(valid(v)) if valid(v) else None for h,v in complete.items()},'finite_group_diagnostics':{h:_groups(valid(complete[h])) for h in complete},'available_prefix':strict_summary(prefix),'unstable_available_prefix':strict_summary([r for r in prefix if r['label']=='unstable']),'records':rows,'common_frequency':common,'numerical_initial':numerical,'numerical_scored_initial':scored_numerical,'numerical_trigger':trigger,'finer_executed':finer is not None,'numerical_finer':finer_rows,'numerical_scored_finer':finer_scored,'finer_numeric_gate_pass':finer_pass if finer is not None else None,'model_numerical_gate_pass':model_numeric_pass,'l1_numerically_gated_counts':l1,'classifications':gated,'metadata':metadata}
    if output is not None:
        output=Path(output); output.mkdir(parents=True,exist_ok=False); arrays={'event_ids':np.array([e['id'] for e in events]),'fine_predictions':np.asarray(fine),'coarse_predictions':np.asarray(coarse)}
        if finer is not None: arrays['finer_predictions']=np.asarray(finer)
        for e in events: arrays[f"{e['id']}_truth"]=_truth(e)
        np.savez_compressed(output/'trajectories.npz',**arrays); (output/'per_event_rows.json').write_text(json.dumps(_safe(rows),allow_nan=False)); (output/'evidence.json').write_text(json.dumps(_safe(evidence),indent=2,allow_nan=False))
    return evidence

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    # All event access is explicit development-only and starts at CLI execution.
    from train_structured_rom import load_model
    from train_joint_latent import read_development, BASE, NETWORK
    from kiki.models.private_pm import MechanicalROM
    from evaluate_l1_latent import _read_validation_starts
    base=MechanicalROM.load(BASE); network=dict(np.load(NETWORK,allow_pickle=False))
    network['observed_buses']=np.unique(np.r_[network['load_buses'],network['generator_buses']])
    network['load_positions']=np.searchsorted(network['observed_buses'],network['load_buses'])
    development,excluded,identities=read_development(base,network)
    starts,latency=_read_validation_starts(base.pm_star)
    if len(starts)!=107 or len(latency)!=10: raise ValueError('unexpected validation coverage')
    if sum(e['label']=='unstable' for e in development['validation'])!=12:
        raise ValueError('unexpected eligible unstable denominator')
    def hashes():
        paths={'model':a.checkpoint/'model.pt','normalization':a.checkpoint/'normalization.npz',
               'base':Path(BASE),'network':Path(NETWORK),'evaluation_code':Path(__file__),
               'training_code':ROOT/'experiments/train_structured_rom.py',
               'structure_code':ROOT/'experiments/structured_rom.py'}
        export=a.checkpoint/'structured_numeric_export.json'
        if export.exists(): paths['numeric_export']=export
        return {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
    before=hashes(); model=load_model(a.checkpoint)
    meta={'source_hashes_before':before,'identities':identities,'excluded':excluded,
          'latency_rows':[{key:value for key,value in row.items()
                           if key not in ('inertia','equilibrium_delta')} for row in latency]}
    result=evaluate(model,development['validation'],starts,output=a.output,metadata=meta)
    after=hashes()
    if before!=after: raise RuntimeError('source changed during evaluation')
    result['metadata']['source_hashes_after']=after
    (a.output/'evidence.json').write_text(json.dumps(_safe(result),indent=2,allow_nan=False))
if __name__=='__main__': main()
