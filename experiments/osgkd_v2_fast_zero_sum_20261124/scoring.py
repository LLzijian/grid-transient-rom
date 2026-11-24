"""Augmented-state scoring; the old 50 coordinates retain their original gate."""
from __future__ import annotations
from collections import defaultdict
from time import perf_counter, process_time
import numpy as np
import torch
from experiments.osgkd_v2_train385_baseline_20261123 import run as baseline
from kiki.evaluation.osgkd import score_event, summarize, numerical_gate
from kiki.evaluation.osgkd_validation import missing_row
from kiki.models.osgkd_fitting import IntegrationWork


def integrate(model, state, steps, *, internal_step=.005, work=None):
    """Identical RK4 arithmetic, counting actual leading batch dimensions."""
    if type(steps) is not int or steps < 0 or internal_step not in (.005, .0025):
        raise ValueError('registered output count and mesh required')
    work = IntegrationWork() if work is None else work
    batch = int(np.prod(state.shape[:-1]))
    def field(x):
        work.rhs[work.phase] += batch
        return model.rhs(None, x)
    outputs = [state]
    for _ in range(steps):
        for _ in range(round(.01/internal_step)):
            h=internal_step
            k1=field(state); k2=field(state+h/2*k1)
            k3=field(state+h/2*k2); k4=field(state+h*k3)
            state=state+h/6*(k1+2*k2+2*k3+k4)
        outputs.append(state)
    return torch.stack(outputs, dim=-2)


def fast_gate(main, fine):
    finite=bool(np.isfinite(main).all() and np.isfinite(fine).all())
    rms=maximum=None
    if finite:
        with np.errstate(over='ignore',invalid='ignore'):
            d=main-fine; rms=float(np.sqrt(np.mean(d*d))); maximum=float(np.max(np.abs(d)))
        if not np.isfinite([rms,maximum]).all(): rms=maximum=None
    return dict(finite=finite,rms=rms,maximum=maximum,
                passed=finite and rms is not None and rms<=1e-4 and maximum<=1e-3)


def augmented_row(event, main, fine, ports, constants):
    row=score_event(event,main[:,:50],fine[:,:50],ports,constants)
    if main.shape[-1]==50: return row
    available=len(event['t'])-1
    for scope,item in row['scopes'].items():
        count=min(available,1500 if scope=='actual' else int(scope)*100)
        gate=fast_gate(main[:count+1,50:],fine[:count+1,50:])
        item['fast_numerical']=gate
        item['numerical']['passed'] &= gate['passed']
        if not item['numerical']['passed']: item['metrics']=None
    for scope,key in (('8s','classification_8s'),('actual','classification_actual')):
        item=row[key]
        if item is None: continue
        count=800 if scope=='8s' else min(available,1500)
        gate=fast_gate(main[:count+1,50:],fine[:count+1,50:])
        item['fast_numerical']=gate
        if not gate['passed']:
            from experiments.evaluate_l1_latent import apply_numerical_gate
            item['los_pair_pass']=False; item['numerical']['passed']=False
            item['classification']=apply_numerical_gate(item['fine_classification'],False)
            item['predicted_time_record_s']=None; item['predicted_time_since_clearing_s']=None
            item['timing_comparable']=False; item['trigger_time_error_s']=None
    return row


@torch.no_grad()
def score_split(model, events, constants, split, directory, *, work=None):
    baseline.require(split in ('train','val'),'invalid scoring split')
    directory.mkdir(parents=True,exist_ok=False)
    started,cpu=perf_counter(),process_time(); work=IntegrationWork() if work is None else work; rows={}; groups=defaultdict(list)
    for event in events:
        if event['history'] is None: rows[event['id']]=missing_row(event); continue
        count=max(800,min(1500,len(event['t'])-1)) if event['label']=='unstable' else 800
        groups[count].append(event)
    for count,cohort in groups.items():
        for start in range(0,len(cohort),16):
            batch=cohort[start:start+16]
            th,om=[torch.as_tensor(np.stack([e['history'][key] for e in batch]),dtype=torch.float64) for key in ('theta','omega')]
            initial=model.initialize(th,om); work.encoder_events+=len(batch)
            baseline.require(initial.shape==(len(batch),6*model.n if hasattr(model,'fast_gamma') else 5*model.n),'state shape differs')
            for actual,wanted in ((initial[:,:10],th[:,-1]),(initial[:,10:20],om[:,-1])):
                baseline.require(torch.equal(actual.contiguous().view(torch.int64),wanted.contiguous().view(torch.int64)),'observed endpoints changed')
            main=integrate(model,initial,count,work=work).numpy()
            fine=integrate(model,initial,count,internal_step=.0025,work=work)
            ports=model.ports(fine); output={key:getattr(ports,key).numpy() for key in ('tau_m','tau_e')}; fine=fine.numpy()
            for i,event in enumerate(batch): rows[event['id']]=augmented_row(event,main[i],fine[i],{k:v[i] for k,v in output.items()},constants)
    rows=[rows[e['id']] for e in events]
    mapped=summarize(rows,{**constants,'fit_ids':[],'check_ids':[]})
    metrics=mapped['splits']['train_all']
    for name,metric in metrics.items():
        if name.startswith('stable_'):
            scope=name.split('_')[-1][:-1]
            if any(not r['scopes'][scope]['full_horizon_truth'] or r['scopes'][scope]['metrics'] is None for r in rows if r['label']=='stable'): metric['official']=None
    summary=dict(metric=mapped['metric'],splits={split:metrics},classification={split:mapped['classification']['train_all']},
                 coverage=dict(denominator=len(events),input_eligible=sum(e['history'] is not None for e in events),
                               missing_history_ids=[e['id'] for e in events if e['history'] is None]),
                 work=work.snapshot(),cost=dict(wall_s=perf_counter()-started,cpu_s=process_time()-cpu))
    expected=sum(24*count*len(cohort) for count,cohort in groups.items())
    baseline.require(work.rhs['forward']==expected and work.rhs['recompute']==0,'evaluation RHS accounting differs')
    rows_ref=baseline.atomic_write(directory/'rows.json',rows)
    metrics_ref=baseline.atomic_write(directory/'metrics.json',summary)
    completed_ref=baseline.atomic_write(directory/'completed.json',dict(status='completed',rows_ref=rows_ref,metrics_ref=metrics_ref))
    return dict(data_identity=baseline.data_identity(events),summary=baseline.public_summary(summary),
                timing=baseline.timing_summary(rows,events),rows_ref=rows_ref,metrics_ref=metrics_ref,completed_ref=completed_ref)
