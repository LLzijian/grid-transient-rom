"""Frozen H/A estimates and offline truth-driven diagnostics; independent E/Q releases.

No training, autonomous rollout, model export, or source qualification entry point.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time
import traceback
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import p_form_discrimination_20261031 as old
import p_history_transfer_20261031_source as source

RUN='p_history_transfer_20261031'
PRIVATE=ROOT/'private_data'/RUN
PUBLIC=ROOT/'results'/RUN
PRIMARY=('stable','unstable')
DENOM={'all':297,'stable':267,'unstable':30}
KINDS=('beta0','H','A')
sha,read,write_once=source.sha,source.read,old.write_once
ADOPTION_SHA='3750a1ca973a6154b86ce616f3aa7a5ff3e2b5c563d1aefd43c77a3eb5a11553'
S_MANIFEST_SHA='523524a4d9dfbd2baa663005f1c71f938d79c2c51900795aa852312de8313981'
ADDENDUM=ROOT/'reports'/('p_history_transfer_20261031_numerical_reference_addendum.md')
ADDENDUM_SHA='ce8f3534b57fd3f67e44a1bbfc49f51edb726e5a97bfff3f6ae2584884e58e43'

def source_closure():
    return {**old.source_closure(),'p_form_discrimination_20261031.py':sha(ROOT/'experiments/p_form_discrimination_20261031.py'),
      Path(source.__file__).name:sha(Path(source.__file__)),Path(__file__).name:sha(Path(__file__))}

def runtime():
    import scipy
    return {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
      'device':'CPU','dtype':'float64','threads':1,'thread_environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}

def release_expected(stage):
    return {'source_sha256':sha(Path(__file__)), 'tests_sha256':sha(ROOT/'tests'/('test_'+RUN+'.py')),
      'source_closure_sha256':old.digest(source_closure()),
      'protocol_sha256':source.FROZEN['protocol'][1],'authorization_sha256':source.FROZEN['authorization'][1],
      'source_bindings_sha256':source.FROZEN['bindings'][1], 'source_adoption_sha256':ADOPTION_SHA,
      'source_manifest_sha256':S_MANIFEST_SHA,'source_summary_sha256':S_MANIFEST_SHA,
      'numerical_reference_addendum_sha256':ADDENDUM_SHA,
      'maximum_main_parameter_estimates':20 if stage=='estimate' else 0,
      'same_objective_independent_qr_checks_upper_bound':20 if stage=='estimate' else 0,
      'fixed_coefficient_numerical_check_slots_upper_bound':30 if stage=='analyze' else 0,
      'fixed_truth_driven_analysis_suites_upper_bound':1 if stage=='analyze' else 0,
      'optimizer_upper_bound':0,'new_autonomous_rom_rollouts_upper_bound':0,'model_writes_upper_bound':0}

def verify_release(stage,release,expected):
    if release.get('released') is not True or release.get('allowed_stage')!=stage:
        raise PermissionError('root stage release required')
    for key,value in expected.items():
        if release.get(key)!=value or (type(value) is int and type(release.get(key)) is not int):
            raise PermissionError('stage release identity/limit differs: '+key)
    if stage=='estimate' and release.get('scalar_filter_calls_upper_bound')!=0:
        raise PermissionError('E cannot filter')
    if stage=='analyze' and release.get('fixed_coefficient_analysis_only') is not True:
        raise PermissionError('Q must be fixed-coefficient only')

def checked_source(stage):
    expected=release_expected(stage); release=read(PUBLIC/(stage+'_execution_release.json'))
    verify_release(stage,release,expected)
    for key,(name,h) in source.FROZEN.items():
        if sha(ROOT/name)!=h:raise ValueError('frozen '+key+' changed')
    if sha(ADDENDUM)!=ADDENDUM_SHA:raise ValueError('numerical reference addendum changed')
    if sha(PUBLIC/'source_adoption.json')!=ADOPTION_SHA or sha(PRIVATE/'manifest.json')!=S_MANIFEST_SHA or sha(PUBLIC/'source_summary.json')!=S_MANIFEST_SHA:
        raise ValueError('adopted S identity differs')
    adopt=read(PUBLIC/'source_adoption.json'); manifest=read(PRIVATE/'manifest.json')
    if any(adopt.get(k) is not True for k in ('source_qualified','time_qualified','coverage_qualified')) or manifest['status']!='verified':
        raise ValueError('S not qualified')
    if adopt.get('deployment_index_unchanged')!=10 or adopt.get('resampling_performed') is not False:
        raise ValueError('deployment indexing conflict')
    complete=read(PRIVATE/'source_complete.json')
    if complete.get('manifest_sha256')!=S_MANIFEST_SHA:raise ValueError('S completion differs')
    for filename,key in (('members.json','private_members_sha256'),('prefixes.npz','prefixes_sha256')):
        if sha(PRIVATE/filename)!=manifest[key]:raise ValueError('S artifact differs')
    return manifest,{'sha256':sha(PUBLIC/(stage+'_execution_release.json')),'expected':expected,'source_closure':source_closure()}

def checked_inputs(manifest):
    bindings=read(ROOT/source.FROZEN['bindings'][0]); entry=bindings['fit_cache']
    path=source.bound_path(ROOT,entry)
    if sha(path)!=entry['sha256']:raise ValueError('adopted fit cache changed')
    members=read(PRIVATE/'members.json'); frozen={r['id']:r for r in members}
    events=[]
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata_json']))
        for i,m in enumerate(metadata):
            x=data[f'event_{i:04d}_x'].copy(); row=frozen.get(m['id'])
            if row is None or m['label']!=row['label'] or len(x)!=row['coverage']['length'] or source.array_hash(source.member_prefix(x))!=row['prefix_sha256']:
                raise ValueError('S member identity differs')
            events.append({'id':m['id'],'label':m['label'],'x':x})
    if len(events)!=297 or len({e['id'] for e in events})!=297 or {e['id'] for e in events}!=set(frozen):raise ValueError('297 identity differs')
    if any(sum(e['label']==g for e in events)!=DENOM[g] for g in PRIMARY):raise ValueError('267/30 identity differs')
    constants=read(PRIVATE/'parent_constants.json')
    parent_entry=bindings['parent_p_form']['manifest.json']
    if sha(ROOT/parent_entry['path'])!=parent_entry['sha256']:raise ValueError('parent constants source differs')
    parent=read(ROOT/parent_entry['path']); params=source.parent_constants(parent)
    if constants['parent_manifest_sha256']!=parent_entry['sha256'] or old.digest(constants['parameters'])!=old.digest(params):raise ValueError('S constants differ')
    return sorted(events,key=lambda e:e['id']),np.asarray(params['beta0']),np.asarray(params['p_scale'])

def future_steps(event):
    return 800 if event['label']=='stable' else min(1500,len(event['x'])-11)

def truth_slice(event,machine,scope):
    x=np.asarray(event['x']); n=future_steps(event)
    if scope=='H':start,end=0,8
    elif scope=='history':start,end=0,10
    elif scope=='history80':start,end=0,8
    elif scope=='future20':start,end=10,12
    elif scope=='future':start,end=10,10+n
    elif scope in ('A','A80'):start,end=0,10+n
    else:raise ValueError('unknown frozen scope')
    if (scope in ('A','A80','future','future20') and n<2) or end>=len(x) or x.ndim!=2 or x.shape[1]!=30:return None
    return x[start:end+1][:,[10+machine,20+machine]].copy()

def collection(events,machine,group,scope,steps,scale):
    rows=[]
    for event in events:
        if group!='all' and event['label']!=group:continue
        truth=truth_slice(event,machine,scope)
        row=old.window_design(truth,steps,scale,common80=scope=='A80') if truth is not None else {'available':False,'reason':'incomplete_registered_interval','windows':0,'X_S':None,'X_T':None,'y':None}
        rows.append({'event_id':event['id'],**row})
    return rows

def target(events,machine,kind,scale):
    rows=collection(events,machine,'all',kind,4,scale)
    t,meta=old.weighted_target(rows,297)
    if t is not None and not all(np.isfinite(t[k]).all() for k in ('A','b','X_S','X_T','weights','scales','column_norms')):
        return None,{**meta,'status':'unknown','reason':'nonfinite_derived_target'}
    meta['index_anchor']=0
    meta['event_window_coverage']=[{'id':e['id'],**source.windows(10 if kind=='H' else 10+future_steps(e),4)} for e in events]
    return t,meta

def verify_slot(folder):
    receipt=read(folder/'complete.json')
    for name in ('parameters.json','qr.json'):
        if sha(folder/name)!=receipt[name+'_sha256']:raise ValueError('frozen slot artifact changed')
    return read(folder/'parameters.json'),read(folder/'qr.json'),receipt

def estimate_slot(folder,kind,machine,t,coverage,input_identity):
    folder=Path(folder)
    if (folder/'complete.json').exists():
        values=verify_slot(folder)
        if values[2]['input_identity']!=input_identity or values[2]['kind']!=kind or values[2]['machine']!=machine:
            raise ValueError('completed slot cannot be rebound to a different target')
        return values
    if folder.exists():raise RuntimeError('unknown started estimate slot; no replay')
    folder.mkdir(parents=True,exist_ok=False)
    write_once(folder/'started.json',{'kind':kind,'machine':machine,'input_identity':input_identity,'maximum_main_SVD_calls':1,'maximum_QR_calls':1})
    start=time.perf_counter()
    if t is None:
        p={'status':'unknown','reason':'target_incomplete_or_nonfinite','rank':None,'beta40':None,'gamma':None,'parameter_estimates':0,'independent_identifiability':False,'signs_legal':False,'svd_numerical_checks_passed':False}
    else:
        try:p=old.primary_svd(t)
        except (np.linalg.LinAlgError,FloatingPointError,ValueError) as exc:
            p={'status':'unknown','reason':repr(exc),'rank':None,'beta40':None,'gamma':None,'parameter_estimates':1,'independent_identifiability':False,'signs_legal':False,'svd_numerical_checks_passed':False}
    # Parameter result is sealed BEFORE the independent QR. Rank-deficient minimum
    # norm coefficients are retained privately as projection references only.
    p.update(kind=kind,machine=machine,coverage=coverage,input_identity=input_identity,svd_wall_seconds=time.perf_counter()-start)
    write_once(folder/'parameters.json',p)
    qstart=time.perf_counter()
    if t is None or p['rank'] is None:q={'status':'unknown','qr_checks':0,'reason':'no_solved_target'}
    else:
        try:q=old.qr_check(t,p)
        except (np.linalg.LinAlgError,FloatingPointError,ValueError) as exc:q={'status':'unknown','qr_checks':1,'reason':repr(exc)}
    q['wall_seconds']=time.perf_counter()-qstart;write_once(folder/'qr.json',q)
    receipt={'kind':kind,'machine':machine,'parameters.json_sha256':sha(folder/'parameters.json'),'qr.json_sha256':sha(folder/'qr.json'),
      'input_identity':input_identity,'numeric_optimality_passed':p.get('svd_numerical_checks_passed') is True and q['status']=='verified',
      'wall_seconds':time.perf_counter()-start}
    write_once(folder/'complete.json',receipt)
    return old.safe(p),old.safe(q),receipt

def coefficient(primary,receipt):
    if primary.get('rank')!=2 or receipt.get('numeric_optimality_passed') is not True:return None
    beta=np.asarray(primary.get('beta40'),dtype=np.float64)
    return beta if beta.shape==(2,) and np.isfinite(beta).all() else None

def estimate():
    manifest,release=checked_source('estimate')
    if (PRIVATE/'estimate_complete.json').exists():raise RuntimeError('E already completed; no repeat')
    if (PRIVATE/'estimate_started.json').exists():raise RuntimeError('unknown E execution; reconcile before any replay')
    write_once(PRIVATE/'estimate_started.json',{'status':'STARTED','pid':os.getpid(),'release':release})
    start=time.perf_counter()
    try:
        rstart=time.perf_counter();events,beta0,scales=checked_inputs(manifest);read_seconds=time.perf_counter()-rstart
        slots=[]
        for kind in ('H','A'):
            for machine in range(10):
                target_start=time.perf_counter()
                t,meta=target(events,machine,kind,scales[machine])
                target_seconds=time.perf_counter()-target_start
                identities={'target_sha256':old.digest(t) if t is not None else None,'scope':kind,'phase':0,'steps':4,'scale_sha256':source.array_hash(np.asarray(scales[machine]))}
                p,q,receipt=estimate_slot(PRIVATE/'estimates'/kind/f'machine_{machine:02d}',kind,machine,t,meta,identities)
                slots.append({'kind':kind,'machine':machine,'parameters_sha256':receipt['parameters.json_sha256'],'receipt_sha256':sha(PRIVATE/'estimates'/kind/f'machine_{machine:02d}'/'complete.json'),
                  'rank':p['rank'],'condition':p.get('condition'),'eta':p.get('eta'),'independent_identifiability':p['independent_identifiability'],'signs_legal':p['signs_legal'],'numeric_optimality_passed':receipt['numeric_optimality_passed'],'main_SVD_calls':p['parameter_estimates'],'QR_checks':q['qr_checks'],
                  'target_construction_seconds':target_seconds,'SVD_seconds':p['svd_wall_seconds'],'QR_seconds':q['wall_seconds'],'slot_wall_seconds':receipt['wall_seconds']})
        result={'status':'COMPLETED','stage':'estimate','slots':slots,'denominators':DENOM,'source_manifest_sha256':S_MANIFEST_SHA,'release':release,'runtime':runtime(),
          'read_seconds':read_seconds,'wall_seconds':time.perf_counter()-start,'cpu_seconds':None,'optimizer_calls':0,'scalar_filters':0,'new_ROM_rollouts':0,'model_writes':0}
        write_once(PRIVATE/'estimate_manifest.json',result);write_once(PUBLIC/'estimate_summary.json',result)
        write_once(PRIVATE/'estimate_complete.json',{'status':'COMPLETED','manifest_sha256':sha(PRIVATE/'estimate_manifest.json')})
        return result
    except BaseException as exc:
        write_once(PRIVATE/'estimate_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-start,'automatic_reestimate':False});raise

def response_paths(truth,beta,scale):
    if truth is None or beta is None or not np.isfinite(truth[:,0]).all() or not np.isfinite(truth[0,1]):return None,None
    try:
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            # Only the first true p supplies a state. Later truth p is scoring only.
            omega=np.asarray(truth)[:,0,None];p0=[truth[0,1]]
            l=old.filters.linear_filter(omega,p0,[beta[0]],[beta[1]],.01)[:,0]
            r=old.filters.right_endpoint_filter(omega,p0,[beta[0]],[beta[1]],.01)[:,0]
        if not np.isfinite(l).all() or not np.isfinite(r).all():return None,None
        return l,r
    except (ValueError,FloatingPointError,OverflowError):return None,None

def response_stats(events,machine,group,scope,steps,beta,scale,check,*,persistent=False,cache=None):
    left=[];right=[];private=[]
    for event in events:
        if group!='all' and event['label']!=group:continue
        key=(event['id'],scope,steps,persistent)
        if cache is not None and key in cache:
            l,r,detail=cache[key];left.append(l);right.append(r);private.extend(detail);continue
        truth=truth_slice(event,machine,scope)
        detail=[]
        if check.get('status')!='PASS' or truth is None or beta is None:l,r=None,None
        elif persistent:
            pl,pr=response_paths(truth,beta,scale)
            l=(pl[1:]-truth[1:,1])/scale if pl is not None else None
            r=(pr[1:]-truth[1:,1])/scale if pr is not None else None
            detail.append({'id':event['id'],'linear_path':pl,'right_path':pr,'duration_s':(len(truth)-1)*.01})
        else:l,r=old.local_response(truth,steps,beta,scale)
        left.append(l);right.append(r)
        detail.append({'id':event['id'],'error_linear':l,'error_right':r});private.extend(detail)
        if cache is not None:cache[key]=(l,r,detail)
    error=check.get('normalized_error')
    stats=old.equal_event_stats(left,right,DENOM[group],check_error=error if error is not None else np.nan,kind='response')
    return stats,private

def weak_cell(events,machine,group,scope,steps,beta,scale):
    stats,left,right=old.weak_stats(collection(events,machine,group,scope,steps,scale),beta,DENOM[group])
    return stats,{'left':left,'right':right}

def truth_value(cell,key):return cell.get('status')=='computed' and cell.get(key) is True

def branch_decision(qual,cells):
    def cell(kind,scope,group,steps=None):return cells[kind][scope][group] if steps is None else cells[kind][scope][group][str(steps)]
    def compatible(c):return truth_value(c,'compatible')
    def resolved(c):return truth_value(c,'discernible')
    def remaining(kind,scope,group,steps=None):
        return old.weak_pair(cell('beta0',scope,group,steps),cell(kind,scope,group,steps)).get('remaining') is True
    def history(kind):
        return all(all(compatible(cell(kind,scope,g,s)) for scope in ('history_weak','history_local') for s in (2,4,8))
          and compatible(cell(kind,'history_persistent',g)) and compatible(cell(kind,'history_full20',g)) for g in PRIMARY)
    def future(kind):return all(all(compatible(cell(kind,scope,g)) for scope in ('future20_weak','future20_local','future_persistent')) for g in PRIMARY)
    def fail(kind,g):return all(resolved(cell(kind,scope,g)) for scope in ('future20_weak','future20_local','future_persistent'))
    def alltarget(kind):return all(compatible(cell(kind,'A40',g)) and all(compatible(cell(kind,'A80',g,s)) for s in (2,4,8)) for g in ('all',*PRIMARY))
    def admitted(k):return qual[k]['numeric_optimality'] and qual[k]['identifiable']
    old_resolved=any(all(resolved(cell('beta0','history_weak',g,s)) for s in (2,4,8)) or resolved(cell('beta0','future20_weak',g)) for g in PRIMARY)
    hgood={k:history(k) for k in ('H','A')};fgood={k:future(k) for k in ('H','A')}
    allgood=alltarget('A')
    c=all(admitted(k) and qual[k]['signs_legal'] for k in ('H','A')) and all(hgood.values()) and all(fgood.values()) and allgood and old_resolved
    transfer=all(admitted(k) and qual[k]['signs_legal'] for k in ('H','A')) and hgood['H'] and any(fail('H',g) for g in PRIMARY) and allgood and hgood['A'] and fgood['A']
    supporting=[]
    for g in PRIMARY:
        common=all(remaining('A','A80',g,s) for s in (2,4,8))
        hist=all(remaining('A','history_weak',g,s) and resolved(cell('A','history_local',g,s)) for s in (2,4,8)) and resolved(cell('A','history_persistent',g))
        fut=remaining('A','future20_weak',g) and resolved(cell('A','future20_local',g)) and resolved(cell('A','future_persistent',g))
        if common and (hist or fut):supporting.append({'group':g,'history_location':hist,'future_location':fut})
    f=admitted('A') and remaining('A','A40','all') and bool(supporting)
    return {'C_H':bool(c),'transfer_failure_full_target_compatible':bool(transfer),'F_A':bool(f),'F_A_supporting_groups':supporting if f else [],
      'H_unidentifiable_A_identifiable':not qual['H']['identifiable'] and qual['A']['identifiable'],
      'HistoryGood':hgood,'FutureGood':fgood,'AllTargetGood':allgood,'beta0_discriminable':old_resolved,
      'accumulation_or_mixed_groups':{k:[g for g in PRIMARY if resolved(cell(k,'future_persistent',g)) and not fail(k,g)] for k in ('H','A')},
      'A_unconstrained_reference_only':not qual['A']['signs_legal'],'status':'C_H' if c else 'transfer_failure' if transfer else 'F_A' if f else 'mixed_or_unresolved',
      'model_adopted':False,'physical_law_falsified':False,'majority_vote_used':False}

def analyze_machine(events,machine,coefficients,scale,checks,qual):
    cells={k:{} for k in KINDS};private={k:{} for k in KINDS}
    weak_seconds=0.;filter_seconds=0.
    weak_scopes={'history_weak':('history80',(2,4,8)),'history_full20':('history',(2,)),
      'A40':('A',(4,)),'A80':('A80',(2,4,8)),'future20_weak':('future20',(2,))}
    for kind in KINDS:
        weak_start=time.perf_counter()
        beta=coefficients[kind]
        # Analytic qualification gates all coefficient-dependent response/weak
        # interpretation, including nonfinite derived statistics.
        usable=beta if checks[kind].get('status')=='PASS' else None
        for name,(scope,steps) in weak_scopes.items():
            cells[kind][name]={};private[kind][name]={}
            for group in ('all',*PRIMARY):
                records={s:weak_cell(events,machine,group,scope,s,usable,scale) for s in steps}
                multi=len(steps)>1
                cells[kind][name][group]={str(s):r[0] for s,r in records.items()} if multi else next(iter(records.values()))[0]
                private[kind][name][group]={str(s):r[1] for s,r in records.items()}
        weak_seconds+=time.perf_counter()-weak_start
        filter_start=time.perf_counter();response_cache={}
        for name,scope,steps,persistent in (('history_local','history80',(2,4,8),False),('history_full_local','history',(2,4,8),False),
          ('history_persistent','history',(None,),True),('future20_local','future20',(2,),False),('future_persistent','future',(None,),True)):
            cells[kind][name]={};private[kind][name]={}
            for group in ('all',*PRIMARY):
                records={s:response_stats(events,machine,group,scope,s,usable,scale,checks[kind],persistent=persistent,cache=response_cache) for s in steps}
                multi=len(steps)>1
                cells[kind][name][group]={str(s):r[0] for s,r in records.items()} if multi else next(iter(records.values()))[0]
                private[kind][name][group]={str(s):r[1] for s,r in records.items()}
        filter_seconds+=time.perf_counter()-filter_start
    differences={kind:{scope:{group:{'linear':old.relative_change(cells['beta0'][scope][group],cells[kind][scope][group],'error_L'),
      'right':old.relative_change(cells['beta0'][scope][group],cells[kind][scope][group],'error_R')} for group in ('all',*PRIMARY)} for scope in ('history_persistent','future20_local','future_persistent')} for kind in ('H','A')}
    return {'machine':machine,'qualification':qual,'coefficient_checks':checks,'cells':cells,'response_changes_vs_beta0':differences,'decision':branch_decision(qual,cells),
      'cost':{'weak_seconds':weak_seconds,'filter_seconds':filter_seconds}},private

def combined_rms(machines):
    """Descriptive ten-machine RMS; no pooled threshold or branch decision."""
    if len(machines)!=10:return {'status':'unknown','reason':'ten machines required'}
    def rms(values):
        if not all(old.finite_number(v) for v in values):return None
        top=max(abs(v) for v in values)
        result=top*float(np.sqrt(np.mean((np.asarray(values)/top)**2))) if top else 0.
        return result if np.isfinite(result) else None
    def walk(nodes):
        first=nodes[0]
        if isinstance(first,dict) and 'status' in first:
            valid=all(n.get('status')=='computed' for n in nodes)
            return {'status':'computed' if valid else 'unknown','machine_denominator':10,
              'event_denominator':first.get('denominator'),
              **{k:rms([n.get(k) for n in nodes]) if valid else None for k in ('error_L','error_R')}}
        return {k:walk([n[k] for n in nodes]) for k in first}
    return walk([r['cells'] for r in machines])

def analyze():
    manifest,release=checked_source('analyze')
    if (PRIVATE/'analyze_complete.json').exists() or (PRIVATE/'analyze_started.json').exists():raise RuntimeError('Q complete or unknown started; no replay')
    complete=read(PRIVATE/'estimate_complete.json')
    if sha(PRIVATE/'estimate_manifest.json')!=complete['manifest_sha256']:raise ValueError('E completion differs')
    e_manifest=read(PRIVATE/'estimate_manifest.json')
    if e_manifest['release']['expected']!=release_expected('estimate') or e_manifest['release']['source_closure']!=source_closure():
        raise ValueError('E source closure or identities differ')
    e_slots={(r['kind'],r['machine']):r for r in e_manifest['slots']}
    if set(e_slots)!={(k,i) for k in ('H','A') for i in range(10)}:raise ValueError('fixed20 slot set differs')
    estimates={};receipts={}
    for kind in ('H','A'):
        estimates[kind]=[];receipts[kind]=[]
        for machine in range(10):
            p,q,r=verify_slot(PRIVATE/'estimates'/kind/f'machine_{machine:02d}')
            slot=e_slots[(kind,machine)]
            if slot['receipt_sha256']!=sha(PRIVATE/'estimates'/kind/f'machine_{machine:02d}'/'complete.json') or slot['parameters_sha256']!=r['parameters.json_sha256']:
                raise ValueError('slot not bound to completed E manifest')
            estimates[kind].append(p);receipts[kind].append(r)
    write_once(PRIVATE/'analyze_started.json',{'status':'STARTED','pid':os.getpid(),'release':release,'estimate_complete_sha256':sha(PRIVATE/'estimate_complete.json'),'new_parameter_estimates':0})
    start=time.perf_counter()
    try:
        rstart=time.perf_counter();events,beta0,scales=checked_inputs(manifest);read_seconds=time.perf_counter()-rstart
        fixed=old.filters.artificial_qualification()
        if fixed.get('passed') is not True and fixed.get('status')!='PASS':
            # Exact original helper schema is preserved in the receipt; no
            # assumption that a completed function equals qualification.
            if fixed.get('all_passed') is not True:raise ValueError('fixed artificial analytic qualification not passed')
        machines=[];check_seconds=0.;analysis_seconds=0.;check_cache={};check_slots=[]
        for machine in range(10):
            coeff={'beta0':beta0[machine],**{k:coefficient(estimates[k][machine],receipts[k][machine]) for k in ('H','A')}}
            cstart=time.perf_counter();checks={}
            for kind,b in coeff.items():
                slot=f'{kind}/machine_{machine:02d}'
                if b is None:
                    checks[kind]={'status':'unknown','normalized_error':None,'reason':'no_unique_numerically_qualified_frozen_coefficient','slot':slot,'reference_slot':None}
                    check_slots.append({'slot':slot,'valid_frozen_coefficient':False,'unique_calculation':False,'reference_slot':None})
                else:
                    key=source.array_hash(np.asarray([b[0],b[1],scales[machine]],dtype=np.float64))
                    new=key not in check_cache
                    if new:check_cache[key]=(old.coefficient_check(*b,scales[machine]),slot)
                    result,reference=check_cache[key];checks[kind]={**result,'slot':slot,'reference_slot':reference}
                    check_slots.append({'slot':slot,'valid_frozen_coefficient':True,'unique_calculation':new,'reference_slot':reference,'coefficient_and_scale_sha256':key})
            check_seconds+=time.perf_counter()-cstart
            qual={k:{'numeric_optimality':receipts[k][machine]['numeric_optimality_passed'],'identifiable':estimates[k][machine]['independent_identifiability'],
              'signs_legal':estimates[k][machine]['signs_legal'],'rank':estimates[k][machine]['rank'],
              'condition':estimates[k][machine].get('condition'),'eta':estimates[k][machine].get('eta')} for k in ('H','A')}
            astart=time.perf_counter();row,private=analyze_machine(events,machine,coeff,scales[machine],checks,qual);analysis_seconds+=time.perf_counter()-astart
            write_once(PRIVATE/f'analysis_machine_{machine:02d}.json',private)
            row['private_details_sha256']=sha(PRIVATE/f'analysis_machine_{machine:02d}.json');machines.append(row)
        result={'status':'COMPLETED','stage':'analyze','machines':machines,'combined_ten_machine_RMS':combined_rms(machines),'denominators':DENOM,'release':release,'runtime':runtime(),'fixed_artificial_qualification':fixed,
          'coefficient_numerical_slots':check_slots,'coefficient_numerical_slot_count':30,'valid_coefficient_slots':sum(r['valid_frozen_coefficient'] for r in check_slots),'unique_coefficient_checks':len(check_cache),
          'oracle_usage':'true historical p / future omega are offline only; never deployment or autonomous predictions',
          'cost':{'read_seconds':read_seconds,'analytic_check_seconds':check_seconds,'weak_seconds':sum(r['cost']['weak_seconds'] for r in machines),'filter_seconds':sum(r['cost']['filter_seconds'] for r in machines),'weak_and_filter_seconds':analysis_seconds,'wall_seconds':time.perf_counter()-start,'cpu_seconds':None,'LLM_seconds':None,'inclusion':'weak and filter are within weak_and_filter, and read/check/weak_and_filter within total wall; do not add to enclosing times'},
          'new_parameter_estimates':0,'new_QR_checks':0,'optimizer_calls':0,'new_ROM_rollouts':0,'model_writes':0,'model_adopted':False}
        write_once(PRIVATE/'analysis.json',result);write_once(PUBLIC/'analysis_summary.json',result)
        write_once(PRIVATE/'analyze_complete.json',{'status':'COMPLETED','analysis_sha256':sha(PRIVATE/'analysis.json')})
        return result
    except BaseException as exc:
        write_once(PRIVATE/'analyze_failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-start,'automatic_retry':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['estimate','analyze']);args=p.parse_args()
    estimate() if args.stage=='estimate' else analyze()
