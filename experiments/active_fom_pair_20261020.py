"""Finite acquisition stages for the preregistered 2026-10-20 fixed pair study.

No historical mixed-split HDF5 hashing, training, or historical val access occurs here.
Stage artifacts are immutable: incomplete runs require root resolution, not retry.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / 'experiments', ROOT / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from experiments import closure_p3_disagreement_20261020 as d
from experiments import closure_discovery_data as data

BUSES = (1,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24,27,28,29,30,31,32,33)
STRATA = ('short', 'medium', 'long')
DURATIONS = (.075,.125,.175,.225,.275,.35,.5)
EXPECTED_COUNTS = ((0,9,6),(4,1,6),(6,9,0),(6,8,4),(4,9,6),(6,8,4),
                   (6,9,6),(6,9,5),(6,9,6),(6,7,6),(5,9,3),(4,9,5),
                   (5,9,6),(6,9,6),(5,9,5),(5,8,6),(6,8,6),(6,9,6),
                   (6,9,6),(6,9,6),(6,8,6),(6,9,6),(6,8,6),(6,9,6))
OLD = ROOT / 'private_data/closure_discovery_20261017'
OLD_VAL = OLD / 'new_fault_manifest.json'
OLD_TRAIN = [OLD / 'D/new_data/train/generation.json'] + [OLD / ('A/'+s+'/generation.json')
             for s in ('fom_cost','fom_cost_retry1','fom_cost_retry2')]
EXTRA_SOURCES = ('experiments/active_fom_pair_20261020.py',
 'experiments/closure_short_history_20261019.py',
 'experiments/active_fom_pair_fit_20261020.py', 'tests/test_active_fom_pair_acquire_20261020.py',
 'tests/test_active_fom_pair_fit_20261020.py', 'reports/active_fom_pair_20261020_protocol.md',
 'reports/astra6_active_fom_20261020.md', 'reports/astra6_active_fom_20261020_design_amendment.md',
 'reports/active_fom_20261020_independent_precheck.md',
 'private_data/closure_mechanism_20261019/P3/active_fom_pair01/launch_stage.py',
 'private_data/closure_mechanism_20261019/P3/active_fom_pair01/qa_once.py')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def key(row):
    return int(row.get('bus',row.get('bus_id'))), round(float(row['duration_s']),12), round(float(row['xf']),12)

def stratum(row):
    t = float(row['duration_s'])
    return 0 if t < .15 else 1 if t <= .30 else 2

def draw_assignment(allowed, quotas, rng):
    """Exact uniform feasible assignment; integer DP, also exact marginals."""
    allowed = tuple(tuple(v) for v in allowed)
    @lru_cache(None)
    def count(i, q):
        if min(q) < 0 or sum(q) != len(allowed)-i:
            return 0
        if i == len(allowed):
            return int(q == (0,0,0))
        return sum(count(i+1,tuple(v-(j==s) for j,v in enumerate(q))) for s in allowed[i])
    total = count(0,tuple(quotas))
    if not 0 < total < 2**63:
        raise ValueError('no complete feasible design or unsupported integer count')
    q, assignment, trace = tuple(quotas), [], []
    for i, options in enumerate(allowed):
        weights = [count(i+1,tuple(v-(j==s) for j,v in enumerate(q))) for s in options]
        r = int(rng.integers(0,sum(weights)))
        cumulative = 0
        for s,w in zip(options,weights):
            cumulative += w
            if r < cumulative:
                assignment.append(s)
                trace.append({'bus_index':i,'remaining':list(q),'draw':r,'total':sum(weights),
                              'options':list(options),'completion_counts':weights,'selected':s})
                q = tuple(v-(j==s) for j,v in enumerate(q))
                break
    # Forward prefix counts combine with the suffix DP, without enumerating assignments.
    forward = {tuple(quotas):1}
    marginal = []
    for i,options in enumerate(allowed):
        numerators = [0,0,0]
        next_forward = {}
        for q,n_prefix in forward.items():
            for s in options:
                nq = tuple(v-(j==s) for j,v in enumerate(q))
                completions = count(i+1,nq)
                numerators[s] += n_prefix*completions
                if completions:
                    next_forward[nq] = next_forward.get(nq,0)+n_prefix
        marginal.append(numerators)
        forward = next_forward
    return assignment, {'total_count':total,'path':trace,'marginal_numerators':marginal,
                        'marginal_denominator':total}

def make_design(pool):
    pool = sorted(pool,key=lambda r:(key(r),str(r['line']),float(r['location'])))
    cells = [[ [r for r in pool if int(r['bus'])==b and stratum(r)==s] for s in range(3)] for b in BUSES]
    counts = [[len(c) for c in row] for row in cells]
    if counts != [list(v) for v in EXPECTED_COUNTS] or len(pool)!=456:
        raise ValueError('registered 456-condition pool/count table differs: '+repr(counts))
    rng = np.random.Generator(np.random.PCG64(7201))
    assignment, val_dp = draw_assignment([[s for s,c in enumerate(row) if c] for row in cells],(8,8,8),rng)
    if val_dp['total_count'] != 4252621230:
        raise ValueError('val exact count differs')
    val=[]
    for i,s in enumerate(assignment):
        choices=cells[i][s]; r=dict(choices[int(rng.integers(0,len(choices)))])
        r.update(design_role='val',stratum=STRATA[s],cell_count=len(choices),
                 probability_numerator=val_dp['marginal_numerators'][i][s],
                 probability_denominator=val_dp['total_count']*len(choices))
        val.append(r)
    reserved={key(r) for r in val}
    remaining=[[[r for r in c if key(r) not in reserved] for c in row] for row in cells]
    rng=np.random.Generator(np.random.PCG64(7202))
    omission,candidate_dp=draw_assignment([[s for s in range(3) if all(row[j] for j in range(3) if j!=s)]
                                          for row in remaining],(8,8,8),rng)
    if candidate_dp['total_count'] not in (1097450640,399072960):
        raise ValueError('candidate exact count differs')
    candidates=[]
    for i,omit in enumerate(omission):
        for s in range(3):
            if s==omit: continue
            choices=remaining[i][s]; r=dict(choices[int(rng.integers(0,len(choices)))])
            r.update(design_role='candidate',stratum=STRATA[s],cell_count=len(choices),
                     probability_numerator=candidate_dp['total_count']-candidate_dp['marginal_numerators'][i][s],
                     probability_denominator=candidate_dp['total_count']*len(choices))
            candidates.append(r)
    rng=np.random.Generator(np.random.PCG64(7203)); random_ids=[]
    for s in STRATA:
        options=[r['id'] for r in candidates if r['stratum']==s]
        random_ids.extend(options[int(i)] for i in rng.choice(16,size=4,replace=False))
    return {'schema':1,'algorithm':'integer_DP_uniform_feasible_assignment', 'numpy':np.__version__,
            'seeds':[7201,7202,7203],'pool_counts':counts,'pool_size':len(pool),
            'val_dp':val_dp,'candidate_dp':candidate_dp,'val_conditions':val,
            'candidates':candidates,'random_ids':random_ids,'remaining_counts':[[len(c) for c in row] for row in remaining]}

def source_snapshot():
    snap=d.frozen_inputs()
    for name in EXTRA_SOURCES:
        snap['source'][name]=d.sha(ROOT/name)
    for path in [OLD_VAL,*OLD_TRAIN]:
        snap['data'][path.relative_to(ROOT).as_posix()]=d.sha(path)
    if any(Path(p).name=='dataset.h5' for p in snap['data']):
        raise ValueError('mixed HDF5 hashing forbidden')
    return snap

def verify_sources(snap):
    base={k:v for k,v in snap.items() if k!='design_artifacts'}
    if source_snapshot()!=base:
        raise ValueError('source/model/allowed data changed')
    for p,h in snap.get('design_artifacts',{}).items():
        if d.sha(p)!=h: raise ValueError('frozen design changed')

def prepare_conditions(campaign):
    campaign=Path(campaign); campaign.mkdir(parents=True,exist_ok=True)
    if any((campaign/p).exists() for p in ('design.json','val_conditions.json','source_manifest.json')):
        raise FileExistsError('prepare artifacts already exist')
    snap=source_snapshot()
    if d.sha(OLD_VAL)!='adb8d1829811bb9b7834a4405a2c46007bcc7f0282dceb7d8aecc5af4361c9df':
        raise ValueError('old preregistered condition identity changed')
    allowed=data.allowed_train_conditions()
    if tuple(sorted({int(r['bus']) for r in allowed}))!=BUSES:
        raise ValueError('legal train buses changed')
    historical=[*allowed,*read(OLD_VAL)['val_conditions']]
    for p in OLD_TRAIN:
        record=read(p)
        historical.extend(record['conditions'])
        historical.extend(a['condition'] for a in record['attempts'])
    for row in historical:
        data._resolve_line(row,allowed)  # Verify every historical endpoint, including failed attempts.
    excluded={key(r) for r in historical}
    pool=[]
    for b in BUSES:
        for duration in DURATIONS:
            for xf in (.001,math.sqrt(.001*.01),.01):
                r=data._resolve_line({'bus':b,'duration_s':duration,'xf':xf},allowed)
                if key(r) not in excluded: pool.append(r)
    design=make_design(pool)
    design['condition_sources']={p.relative_to(ROOT).as_posix():d.sha(p) for p in [OLD_VAL,*OLD_TRAIN]}
    design['old_val_read_scope']='val_conditions only; no generation, outcomes, trajectories or scores'
    d.write(campaign/'val_conditions.json',{'val_conditions':design['val_conditions']})
    d.write(campaign/'design.json',design)
    snap['design_artifacts']={str(campaign/p):d.sha(campaign/p) for p in ('design.json','val_conditions.json')}
    d.write(campaign/'source_manifest.json',snap)
    return {'status':'COMPLETED','design_sha256':d.sha(campaign/'design.json'),
            'val_conditions_sha256':d.sha(campaign/'val_conditions.json'),'pool_size':456}

def load_short_result(folder,message):
    if message.get('raw_sha256') and d.sha(Path(folder)/message['raw_file'])!=message['raw_sha256']:
        raise ValueError('short raw identity changed')
    return data._load_result(Path(folder),{'steps':{'short':{'raw_file':message['raw_file'],
                            'metadata':message['metadata']}}},'short')

def history_pair(a,b, *, full=False):
    """Qualification of real common 11 samples, never using future endpoints."""
    data._assert_result_system_compatible(a,data._load_reference_fingerprint())
    data._assert_result_system_compatible(b,data._load_reference_fingerprint())
    if any(len(r[0]['t'])<11 for r in (a,b)):
        raise ValueError('100ms history uncovered')
    def times(r):
        m=r[1]
        start=m.get('record_start_absolute_s')
        if start is None: start=float(m['clearing_time_s'])+float(m['record_start_delay_s'])
        return float(start)+np.asarray(r[0]['t'][:11])
    if not np.allclose(times(a),times(b),rtol=0,atol=1e-8):
        raise ValueError('absolute history sample times differ')
    if not np.allclose(np.diff(times(a)),.01,rtol=0,atol=1e-8):
        raise ValueError('history grid differs')
    ma,mb=a[1],b[1]
    if full:
        # Unknown short horizon imposes no future stable label. Early LOS must agree.
        if ma['label']=='unstable' and mb['label']!='unstable':
            raise ValueError('short physical instability conflicts with full run')
    elif ma['label']!=mb['label'] or ma['label'] not in ('unknown','unstable'):
        raise ValueError('short pair classification differs')
    de=np.asarray(a[0]['delta'][:11])-np.asarray(b[0]['delta'][:11])
    we=(np.asarray(a[0]['omega'][:11])-np.asarray(b[0]['omega'][:11]))*data.OMEGA_S
    mass=np.asarray(a[3]); de-=np.sum(de*mass,axis=1,keepdims=True)/mass.sum()
    we-=np.sum(we*mass,axis=1,keepdims=True)/mass.sum()
    rms=float(np.sqrt(.5*np.mean((de/data.SCALES[0])**2+(we/data.SCALES[1])**2)))
    if not math.isfinite(rms) or rms>.001:
        raise ValueError('short history scaled RMS failed: '+str(rms))
    if not full and not (ma.get('history_eligible') is True and mb.get('history_eligible') is True):
        raise ValueError('short history uninitializable')
    return {'pass':True,'samples':11,'scaled_rms':rms,'absolute_times':times(a).tolist()}

def short_acquire(campaign,output,deadline):
    from experiments.closure_short_history_20261019 import run_short_history
    rows=[]
    for condition in read(campaign/'design.json')['candidates']:
        row={'condition':condition,'steps':{},'qualification':{'pass':False,'reason':'not_run_budget'}}
        if time.perf_counter()+60+30 < deadline:
            started=time.perf_counter()
            for step in (.005,.0025):
                name=f'{step:g}'; folder=campaign/'short'/condition['id']/name
                remaining=30-(time.perf_counter()-started)-2
                if remaining<=2:
                    row['steps'][name]={'status':'not_run_pair_budget'}; continue
                try:
                    row['steps'][name]=run_short_history(condition,folder,max_seconds=min(13.,remaining),
                                   step=step,stop_rule='first_record_plus_100ms')
                    row['steps'][name]['folder']=str(folder)
                    if row['steps'][name].get('raw_file'):
                        raw=folder/row['steps'][name]['raw_file']
                        row['steps'][name]['raw_sha256']=d.sha(raw)
                except Exception as exc:
                    row['steps'][name]={'status':'failed','error':repr(exc),'folder':str(folder)}
            row['elapsed_seconds']=time.perf_counter()-started
            try:
                if any(v.get('status')!='complete' for v in row['steps'].values()) or len(row['steps'])!=2:
                    raise ValueError('short simulation failed or timed out')
                a,b=[load_short_result(row['steps'][s]['folder'],row['steps'][s]) for s in ('0.005','0.0025')]
                row['qualification']=history_pair(a,b)
            except Exception as exc:
                row['qualification']={'pass':False,'reason':repr(exc)}
        rows.append(row); d.write(output/'partial.json',{'rows':rows})
    result={'status':'COMPLETED','attempts':rows,'denominator':48,
            'qualified_count':sum(r['qualification']['pass'] for r in rows)}
    d.write(campaign/'short/final.json',result)
    return result

def canonical_history(result,base):
    a,m,eq,mass,*_=result
    x=np.column_stack((np.asarray(a['delta'][:11])-np.asarray(eq['delta']),
                       (np.asarray(a['omega'][:11])-1)*data.OMEGA_S,
                       np.asarray(a['Pm'][:11])-np.asarray(base.pm_star)))
    if x.shape!=(11,30) or not np.isfinite(x).all(): raise ValueError('invalid causal history')
    meta={'equilibrium_delta':np.asarray(eq['delta']), 'omega_s':data.OMEGA_S,
          'inertia':np.asarray(mass),'event_time_s':None}
    return x,meta

def choose_active(rows,random_ids):
    chosen=[]
    for s in STRATA:
        eligible=[r for r in rows if r['condition']['stratum']==s and r['condition']['id'] not in random_ids
                  and r.get('qualified') is True]
        eligible.sort(key=lambda r:(-min(r[t]['total']['D'] for t in ('fine','coarse')),
                                   -min(r[t]['total']['B'] for t in ('fine','coarse')),key(r['condition'])))
        if len(eligible)<4: return None
        chosen.extend(r['condition']['id'] for r in eligible[:4])
    return chosen

def full_prediction_numerical(fine,coarse,history,meta,mass,id_):
    from evaluate_structured_rom import _pair_row
    if fine.shape!=(3001,40) or coarse.shape!=(1501,40):
        raise ValueError('15s prediction shape changed')
    e={'id':id_,'group':'new_condition','label':'unknown'}
    row=_pair_row(fine,coarse,history,e,meta,mass)
    if not np.isfinite(fine).all() or not np.isfinite(coarse).all():
        return {'pass':False,'reason':'nonfinite_full_15s'}
    de=fine[::2,:10]-coarse[:,:10]; we=fine[::2,10:20]-coarse[:,10:20]
    common=(we*mass).sum(1)/mass.sum()
    de-=(de*mass).sum(1,keepdims=True)/mass.sum()
    we-=(we*mass).sum(1,keepdims=True)/mass.sum()
    cn=float(np.sqrt(.5*np.mean((de/d.S_DELTA)**2+(we/d.S_OMEGA)**2)))
    cm=float(np.sqrt(np.mean((common/d.S_OMEGA)**2)))
    return {'pass':bool(row['los_pair_pass'] and cn<=.001 and cm<=.001),
            'full_15s_coi_normalized_rmse':cn,'full_15s_common_normalized_rmse':cm,
            'los_pair_pass':row['los_pair_pass']}

def score_and_freeze(campaign,output,deadline):
    import torch
    from closure_training_runtime import TrainingClosureROM
    from linear_cache_runtime_20261019 import enable_linear_cache
    from train_local_memory_0098 import graph
    torch.set_num_threads(1)
    models={k:enable_linear_cache(TrainingClosureROM.from_export(d.P2/k/'checkpoints/step001024/model.json').cpu()).double().eval() for k in d.KEYS}
    base=data._load_base()
    physical=[]
    for k in d.KEYS:
        payload=read(d.P2/k/'checkpoints/step001024/model.json')
        physical.append({'input_scale':payload['config']['input_scale'],
            **{n:payload['physical'][n] for n in ('mass','D','delta_star','phi','graph_coupling',
                    'input_scale_omega','p_scale','neighbor_A','port_s_weight','port_c_weight')}})
    if any(p!=physical[0] for p in physical): raise ValueError('six fixed physical scales differ')
    mass=models[d.KEYS[0]].mass.detach().numpy()
    for m in models.values():
        if not np.array_equal(m.mass.detach().numpy(),mass): raise ValueError('model masses differ')
    design=read(campaign/'design.json'); short=read(campaign/'short/final.json')
    histories={}; metas={}; rows=[]
    # Batch only genuine causal histories; future truth is unavailable by construction.
    for attempt in short['attempts']:
        c=attempt['condition']; row={'condition':c,'qualified':False,'reason':'history_uncovered'}
        if attempt['qualification']['pass']:
            msg=attempt['steps']['0.005']; result=load_short_result(msg['folder'],msg)
            histories[c['id']],metas[c['id']]=canonical_history(result,base)
        rows.append(row)
    def predict(ids,seconds):
        if time.perf_counter()+60>=deadline: raise TimeoutError('score category deadline')
        h=torch.as_tensor(np.stack([histories[i] for i in ids]),dtype=torch.float64)
        preds={}
        for k,m in models.items():
            if time.perf_counter()+60>=deadline: raise TimeoutError('score category deadline')
            f=None
            try:
                with torch.no_grad(),graph(m):
                    f=m.rollout(h,round(seconds/.005),.005).detach().numpy()
                    c=m.rollout(h,round(seconds/.01),.01).detach().numpy()
                preds[k]=(f,c)
            except Exception as exc:
                preds[k]={'error':repr(exc),'fine':f}
        return preds
    valid_ids=list(histories)
    for first in range(0,len(valid_ids),24):
        ids=valid_ids[first:first+24]; preds=predict(ids,5.)
        path=output/f'D_{first:03d}.npz'
        arrays={k+'_'+s:a for k,p in preds.items() if isinstance(p,tuple) for s,a in zip(('fine','coarse'),p)}
        arrays.update({k+'_fine':p['fine'] for k,p in preds.items() if isinstance(p,dict) and p.get('fine') is not None})
        np.savez_compressed(path,event_ids=np.asarray(ids),**arrays)
        d.write(path.with_suffix('.json'),{'errors':{k:p['error'] for k,p in preds.items() if not isinstance(p,tuple)}})
        for index,id_ in enumerate(ids):
            row=next(r for r in rows if r['condition']['id']==id_)
            row.update(prediction_file=str(path),prediction_sha256=d.sha(path),prediction_index=index)
            try:
                e={'id':id_,'group':str(row['condition']['bus']),'label':'unknown'}
                row['numerical']={k:d.numerical(p[0][index],p[1][index],histories[id_],e,metas[id_],mass,500) for k,p in preds.items()}
                for s,j,stride in (('fine',0,2),('coarse',1,1)):
                    row[s]=d.disagreement({k:p[j][index,stride::stride] for k,p in preds.items()},mass)
                row['qualified']=all(row[s]['total']['D']>=1 and row[s]['total']['B']>=2.5e-5 for s in ('fine','coarse'))
                row['reason']=None if row['qualified'] else 'D_or_B_below_gate'
            except Exception as exc: row['reason']=repr(exc)
        d.write(output/'partial.json',{'rows':rows})
    active=choose_active(rows,design['random_ids'])
    result={'status':'NO_COMPLETE_ACTIVE_DESIGN' if active is None else 'COMPLETED','rows':rows,
            'random_ids':design['random_ids'],'active_ids':active,'denominator':48,'frozen_predictions':[]}
    if active is not None:
        chosen=set(design['random_ids']+active)
        result['selected_conditions']=[c for c in design['candidates'] if c['id'] in chosen]
        # Save all 24 x six identities, including uninitializable placeholders and failed paths.
        available=[c['id'] for c in result['selected_conditions'] if c['id'] in histories]
        for first in range(0,len(available),12):
            ids=available[first:first+12]; preds=predict(ids,15.)
            for index,id_ in enumerate(ids):
                for k,p in preds.items():
                    if not isinstance(p,tuple):
                        partial_path=None
                        if p.get('fine') is not None:
                            folder=campaign/'frozen_predictions'; folder.mkdir(exist_ok=True)
                            partial_path=folder/f'{id_}_{k}_partial.npz'
                            np.savez_compressed(partial_path,history=histories[id_],fine=p['fine'][index],
                                equilibrium_delta=metas[id_]['equilibrium_delta'],inertia=mass)
                        result['frozen_predictions'].append({'id':id_,'model':k,'status':'prediction_failed',
                            'error':p['error'],'path':str(partial_path) if partial_path else None,
                            'sha256':d.sha(partial_path) if partial_path else None,
                            'missing_arrays':['coarse'] if partial_path else ['fine','coarse'],'horizon_s':15})
                        continue
                    folder=campaign/'frozen_predictions'; folder.mkdir(exist_ok=True)
                    path=folder/f'{id_}_{k}.npz'; f,c=p[0][index],p[1][index]
                    np.savez_compressed(path,history=histories[id_],fine=f,coarse=c,
                                        equilibrium_delta=metas[id_]['equilibrium_delta'],inertia=mass)
                    finite=np.isfinite(f).all() and np.isfinite(c).all()
                    bad=np.flatnonzero(~np.isfinite(f).all(axis=1))
                    bad_c=np.flatnonzero(~np.isfinite(c).all(axis=1))
                    try: qualification=full_prediction_numerical(f,c,histories[id_],metas[id_],mass,id_)
                    except Exception as exc: qualification={'pass':False,'reason':repr(exc)}
                    result['frozen_predictions'].append({'id':id_,'model':k,'status':'finite_unqualified' if finite else 'nonfinite',
                         'first_nonfinite_fine_index':int(bad[0]) if len(bad) else None,
                         'first_nonfinite_coarse_index':int(bad_c[0]) if len(bad_c) else None,
                         'numerical_qualification':qualification,
                         'path':str(path),'sha256':d.sha(path),'horizon_s':15})
        for c in result['selected_conditions']:
            if c['id'] not in histories:
                result['frozen_predictions'].extend({'id':c['id'],'model':k,'status':'uncovered','path':None,'horizon_s':15} for k in d.KEYS)
        if len(result['frozen_predictions'])!=144: raise ValueError('incomplete 24x6 frozen inventory')
    d.write(campaign/'score/final.json',result)
    return result

def verify_val_freeze(campaign,freeze_path):
    if freeze_path is None: raise ValueError('val requires actual six-endpoint freeze')
    campaign=Path(campaign).resolve(); freeze_path=Path(freeze_path).resolve()
    if not freeze_path.is_relative_to(campaign): raise ValueError('freeze must belong to current campaign')
    snap=data._freeze_snapshot(freeze_path)
    manifest=read(freeze_path)
    expected={(s,k) for s in (6101,6102,6103) for k in ('full8','delete7')}
    rows=manifest['models']
    if not isinstance(rows,list) or len(rows)!=6 or {(int(r['seed']),r['structure']) for r in rows}!=expected:
        raise ValueError('six distinct preregistered endpoints required')
    conditions=manifest.get('conditions',[])
    locked=read(campaign/'design.json')['val_conditions']
    physical=lambda r:(key(r),str(r['line']),float(r['location']))
    if len(conditions)!=24 or sorted(map(physical,conditions))!=sorted(map(physical,locked)):
        raise ValueError('confirmation conditions differ from locked design')
    if manifest.get('conditions_sha256')!=d.sha(campaign/'val_conditions.json'):
        raise ValueError('condition file hash mismatch')
    for row in rows:
        p=Path(row['path']).resolve()
        if not p.is_relative_to(campaign) or p.name!='model.json':
            raise ValueError('confirmation endpoint outside current campaign')
        meta=read(p.parent/'checkpoint.json')
        identity=meta['identity']
        if (meta['step']!=1280 or meta['model_sha256']!=row['sha256'] or
                identity['seed']!=row['seed'] or identity['structure']!=row['structure']):
            raise ValueError('incomplete or mismatched global1280 endpoint')
        if d.sha(p.parent/'training.pt')!=meta['training_sha256']:
            raise ValueError('confirmation checkpoint training identity mismatch')
    return snap

def generate_full(campaign,conditions,split,deadline,freeze_path=None):
    if split not in ('train','val'): raise ValueError('only current train/val allowed')
    freeze=verify_val_freeze(campaign,freeze_path) if split=='val' else None
    target=campaign/split; target.mkdir(exist_ok=False)
    config=data._safe_config(); reference=data._load_reference_fingerprint()
    data._assert_config_compatible(config,reference)
    conditions=[data._attempt_record(c,split) for c in conditions]
    generation={'schema_version':2,'split':split,'conditions':conditions,'attempts':[],
                'freeze':freeze,'status':'RUNNING','test_read':False,
                'system_fingerprint':{'sha256':reference['fingerprint_sha256'],
                    **{n:reference[n] for n in ('sources','andes_version','case','case_sha256','omega_s','postfault_signature')},
                    'comparison_atol':data.SYSTEM_ATOL}}
    data._write_json(target/'generation.json',generation)
    started=time.perf_counter()
    for condition in conditions:
        attempt={'condition':condition,'steps':{},'pair_qualification':None}
        results={}; valid={}; pair_start=time.perf_counter()
        enough=pair_start+120+60<deadline
        for step in (.005,.0025):
            sk=f'{step:g}'; step_config=dict(config,integration_step_s=step)
            remaining=min(55.,120-(time.perf_counter()-pair_start)-5,deadline-time.perf_counter()-60)
            if not enough or remaining<=0:
                attempt['steps'][sk]={'status':'not_run_budget_exhausted'}; valid[sk]=False; continue
            raw_path=None
            try:
                if split=='val' and verify_val_freeze(campaign,freeze_path)!=freeze:
                    raise ValueError('val freeze invalidated before simulation')
                result,diagnostics,raw_path=data._run_bounded_simulation(condition,step_config,target,
                    condition['event_id'],step,remaining,reference)
                if split=='val' and verify_val_freeze(campaign,freeze_path)!=freeze:
                    raise ValueError('val freeze invalidated after simulation')
                results[sk]=result
                data._validate_raw(result,step_config)
                data._assert_result_system_compatible(result,reference)
                attestation=diagnostics['system_attestation']
                data._assert_attestation_compatible(attestation,step_config,reference)
                valid[sk]=True
                attempt['steps'][sk]={'status':'completed','raw_file':raw_path,
                    'raw_sha256':d.sha(target/raw_path),'metadata':dict(result[1]),
                    'diagnostics':diagnostics,'system_attestation':attestation,'raw_validation':{'pass':True}}
            except Exception as exc:
                valid[sk]=False
                attempt['steps'][sk]={'status':'simulation_timeout' if isinstance(exc,TimeoutError) else 'simulation_failed',
                    'error':repr(exc),'diagnostics':getattr(exc,'diagnostics',{}),
                    'raw_file':raw_path or getattr(exc,'raw_file',None)}
        attempt['elapsed_seconds']=time.perf_counter()-pair_start
        attempt['pair_qualification']=(data._pair_qualification(results,valid,config) if len(results)==2
                                       else {'pass':False,'reason':'raw_simulation_or_budget_failure'})
        generation['attempts'].append(attempt)
        generation['elapsed_seconds']=time.perf_counter()-started
        data._write_json(target/'generation.json',generation)
    generation['status']='complete'
    if split=='val':
        try:
            if verify_val_freeze(campaign,freeze_path)!=freeze: raise ValueError('freeze changed')
        except Exception as exc:
            generation['status']='freeze_invalidated'; generation['freeze_error']=repr(exc)
            for attempt in generation['attempts']:
                attempt['pair_qualification']={'pass':False,'reason':'freeze_invalidated'}
    generation['qualified_count']=sum(bool(a['pair_qualification'].get('pass')) for a in generation['attempts'])
    data._write_json(target/'generation.json',generation)
    return generation

def train_acquire(campaign,output,deadline):
    frozen=read(campaign/'score/final.json')
    if frozen['status']!='COMPLETED' or len(frozen['frozen_predictions'])!=144:
        raise ValueError('full prediction freeze prerequisite failed')
    for row in frozen['frozen_predictions']:
        if row['path'] is not None and d.sha(row['path'])!=row['sha256']:
            raise ValueError('frozen prediction changed')
    generation=generate_full(campaign,frozen['selected_conditions'],'train',deadline)
    target=campaign/'train'
    short={r['condition']['id']:r for r in read(campaign/'short/final.json')['attempts']}
    for attempt in generation['attempts']:
        id_=attempt['condition']['id']; attempt['original_pair_qualification']=dict(attempt['pair_qualification'])
        try:
            s=short[id_]
            if not s['qualification']['pass']: raise ValueError('short history unavailable for full match')
            matches={}
            for step in ('0.005','0.0025'):
                msg=s['steps'][step]
                matches[step]=history_pair(load_short_result(msg['folder'],msg),data._load_result(target,attempt,step),full=True)
            attempt['short_full_alignment']={'pass':True,'steps':matches}
        except Exception as exc:
            attempt['short_full_alignment']={'pass':False,'reason':repr(exc)}
            attempt['pair_qualification']={'pass':False,'reason':'short_full_history_unresolved'}
    generation['qualified_count']=sum(bool(a['pair_qualification'].get('pass')) for a in generation['attempts'])
    d.write(target/'generation.json',generation)
    return {'status':'COMPLETED','generation':str(target/'generation.json'),
            'generation_sha256':d.sha(target/'generation.json'),'attempts':len(generation['attempts']),
            'qualified_count':generation['qualified_count']}

def val_acquire(campaign,output,deadline,freeze_path):
    verify_val_freeze(campaign,freeze_path)
    conditions=read(campaign/'design.json')['val_conditions']
    generation=generate_full(campaign,conditions,'val',deadline,freeze_path)
    target=campaign/'val'
    return {'status':'COMPLETED' if generation['status']=='complete' else generation['status'],
            'generation':str(target/'generation.json'),'generation_sha256':d.sha(target/'generation.json'),
            'attempts':len(generation['attempts']),'qualified_count':generation['qualified_count']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','short','score','train','val'),required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--freeze-manifest',type=Path)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--max-seconds',type=float,required=True)
    args=parser.parse_args(); started=time.perf_counter()
    caps={'prepare':1000,'short':1500,'score':750,'train':3000,'val':3000}
    if not math.isfinite(args.max_seconds) or not 0<args.max_seconds<=caps[args.stage]: raise ValueError('stage cap invalid')
    args.campaign=args.campaign.resolve(); args.output=args.output.resolve()
    private=(ROOT/'private_data').resolve()
    if not args.campaign.is_relative_to(private) or not args.output.is_relative_to(private): raise ValueError('private output required')
    identity={'stage':args.stage,'campaign':str(args.campaign),'source_commit':args.source_commit,
              'source_manifest_sha256':d.sha(args.source_manifest) if args.source_manifest else None,
              'freeze_manifest_sha256':d.sha(args.freeze_manifest) if args.freeze_manifest else None}
    if args.output.exists():
        complete=read(args.output/'complete.json')
        if complete['identity']!=identity or d.sha(args.output/'result.json')!=complete['result_sha256']:
            raise ValueError('existing stage identity mismatch')
        for p,h in complete.get('artifacts',{}).items():
            if d.sha(p)!=h: raise ValueError('completed stage artifact changed')
        verify_sources(read(args.campaign/'source_manifest.json'))
        if args.stage=='val': verify_val_freeze(args.campaign,args.freeze_manifest)
        return 0
    args.output.mkdir(parents=True,exist_ok=False)
    d.write(args.output/'worker_identity.json',{'pid':os.getpid(),'creation_filetime_utc':d.own_creation_filetime_utc(),
              'source_commit':args.source_commit,'argv':sys.argv,'started_unix':time.time()})
    d.write(args.output/'identity.json',identity)
    try:
        if args.stage=='prepare': result=prepare_conditions(args.campaign)
        else:
            if args.source_manifest is None: raise ValueError('frozen sources required')
            snap=read(args.source_manifest); verify_sources(snap)
            for required in ('design.json','val_conditions.json'):
                # Root records these explicit hashes in each stage launcher too.
                if not (args.campaign/required).is_file(): raise ValueError('missing frozen design')
            if args.stage=='val':
                result=val_acquire(args.campaign,args.output,started+args.max_seconds,args.freeze_manifest)
            else:
                result=globals()[{'short':'short_acquire','score':'score_and_freeze','train':'train_acquire'}[args.stage]](args.campaign,args.output,started+args.max_seconds)
        verify_sources(read(args.campaign/'source_manifest.json'))
        result['elapsed_seconds']=time.perf_counter()-started
        d.write(args.output/'result.json',result)
        artifacts={}
        if args.stage=='short':
            artifacts[str(args.campaign/'short/final.json')]=d.sha(args.campaign/'short/final.json')
            for row in result['attempts']:
                for step in row['steps'].values():
                    if step.get('raw_file'):
                        p=Path(step['folder'])/step['raw_file']; artifacts[str(p)]=d.sha(p)
        if args.stage=='score':
            artifacts[str(args.campaign/'score/final.json')]=d.sha(args.campaign/'score/final.json')
            for row in result.get('frozen_predictions',[]):
                if row.get('path'): artifacts[row['path']]=row['sha256']
            for row in result.get('rows',[]):
                if row.get('prediction_file'): artifacts[row['prediction_file']]=row['prediction_sha256']
        if args.stage in ('train','val'):
            p=args.campaign/args.stage/'generation.json'; artifacts[str(p)]=d.sha(p)
            for row in read(p)['attempts']:
                for step in row['steps'].values():
                    if step.get('raw_file'):
                        path=args.campaign/args.stage/step['raw_file']; artifacts[str(path)]=d.sha(path)
            if args.stage=='val':
                frozen=verify_val_freeze(args.campaign,args.freeze_manifest)
                artifacts[str(args.freeze_manifest.resolve())]=d.sha(args.freeze_manifest)
                for row in frozen['models']:
                    modelpath=Path(row['path'])
                    for name in ('model.json','checkpoint.json','training.pt'):
                        path=modelpath.parent/name; artifacts[str(path)]=d.sha(path)
        d.write(args.output/'complete.json',{'identity':identity,'result_sha256':d.sha(args.output/'result.json'),
                                           'artifacts':artifacts})
        return 0
    except BaseException as exc:
        d.write(args.output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started})
        raise

if __name__=='__main__':
    raise SystemExit(main())
