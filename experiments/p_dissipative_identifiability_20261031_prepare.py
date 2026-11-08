"""Prepare only: frozen source proofs, 16 operator cells, then immutable raw targets.

No estimator, rank check, residual, sensitivity, coefficient or model implementation.
Scientific libraries and weight qualification are invoked only inside released P.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
ROOT=Path(__file__).resolve().parents[1]
RUN='p_dissipative_identifiability_20261031'
PRIVATE=ROOT/'private_data'/RUN
PUBLIC=ROOT/'results'/RUN
FROZEN={
 'protocol':('reports/'+RUN+'_protocol.md','7c7fb0041c374ee5cd479cb5389a40b1a2a4eb0abe0eeb49bc8f896a19bb8b05'),
 'architecture':('reports/p_dissipative_closure_20261031_architecture_brief.md','e2731f82f46b72958c8eb79de069688b0fd4194400a86b2991aaa6d4c702e707'),
 'sensitivity':('reports/'+RUN+'_sensitivity_addendum.md','d52486189219de73b89f0a6e11d53f471888c2de879bf9fb4557377dc04413b2'),
 'source_bindings':('results/'+RUN+'/source_bindings.json','c5cf41b6694de55f1c6b2adba9b88f1e2bfb16edc9859a08c866935bb691712d'),
 'authorization':('results/'+RUN+'/authorization.json','496eeb96c636b72bc960a7af207adb04d71fa9af7e0b094cccf57ffb8b304323')}
DENOM={'all':297,'stable':267,'unstable':30}

class PreparationUnknown(Exception):pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(safe(value),stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')

def release_expected():
    return {'source_sha256':sha(Path(__file__)),'tests_sha256':sha(ROOT/'tests'/('test_'+RUN+'_prepare.py')),
      **{key+'_sha256':h for key,(path,h) in FROZEN.items()},
      'real_prepare_upper_bound':1,'operator_qualification_cells_upper_bound':16,
      'parameter_estimates_upper_bound':0,'qr_checks_upper_bound':0,'optimizer_upper_bound':0,
      'new_autonomous_rom_rollouts_upper_bound':0,'new_fom_upper_bound':0,'model_writes_upper_bound':0}

def verify_release(release,expected):
    if release.get('released') is not True or release.get('allowed_stage')!='prepare':raise PermissionError('root prepare release required')
    for key,value in expected.items():
        if release.get(key)!=value or (type(value) is int and type(release.get(key)) is not int):raise PermissionError('identity or P limit differs: '+key)

def checked_release():
    expected=release_expected()
    for key,(path,h) in FROZEN.items():
        if sha(ROOT/path)!=h:raise ValueError('frozen '+key+' changed')
    path=PUBLIC/'prepare_execution_release.json';release=read(path);verify_release(release,expected)
    review=(ROOT/release['review_snapshot_path']).resolve();review.relative_to(ROOT.resolve())
    if review.suffix.lower() not in ('.md','.json') or sha(review)!=release['review_snapshot_sha256']:raise ValueError('frozen review changed')
    return {'release_sha256':sha(path),'review_snapshot_sha256':release['review_snapshot_sha256'],'identities':expected}

def claim_private(directory,release):
    Path(directory).mkdir(parents=True,exist_ok=False)
    write_once(Path(directory)/'prepare_started.json',{'status':'STARTED','pid':os.getpid(),'release':release})

def bound_paths(bindings,root=ROOT):
    if len(bindings['files'])!=15:raise ValueError('exact15 finite binding set required')
    root=Path(root).resolve();paths={}
    for key,entry in bindings['files'].items():
        p=(root/entry['path']).resolve();p.relative_to(root)
        if p.suffix.lower() in ('.h5','.hdf5','.pt','.pth'):raise ValueError('forbidden source payload')
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('bound input identity differs: '+key)
        paths[key]=p
    if len(set(paths.values()))!=15:raise ValueError('duplicate bound file')
    return paths

def source_proofs(paths,bindings):
    manifest=read(paths['prior_data_manifest']);adopt=read(paths['prior_source_adoption']);complete=read(paths['prior_data_complete'])
    if manifest.get('status')!='verified' or complete.get('manifest_sha256')!=bindings['files']['prior_data_manifest']['sha256']:
        raise PreparationUnknown('inherited S completion is not verified')
    if any(adopt.get(k) is not True for k in ('source_qualified','time_qualified','coverage_qualified')) or adopt.get('denominators')!=DENOM:
        raise PreparationUnknown('inherited source/time/coverage adoption unresolved')
    if manifest.get('private_members_sha256')!=bindings['files']['prior_members']['sha256'] or manifest.get('prefixes_sha256')!=bindings['files']['prior_prefixes']['sha256']:
        raise PreparationUnknown('S members or prefix identity differs')
    if adopt.get('source_summary_sha256')!=bindings['files']['prior_source_summary']['sha256'] or adopt.get('source_execution_receipt_sha256')!=bindings['files']['prior_source_receipt']['sha256'] or adopt.get('source_evidence_review_snapshot_sha256')!=bindings['files']['prior_source_review']['sha256']:
        raise PreparationUnknown('S adoption evidence binding differs')
    if adopt.get('source_source_sha256')!=bindings['files']['prefix_parser_spec']['sha256']:
        raise PreparationUnknown('S producer parser identity differs')
    for key in ('fit_cache','producer_prepare'):
        entry=bindings['files'][key]
        if manifest.get('bound_inputs',{}).get(entry['path'])!=entry['sha256']:
            raise PreparationUnknown('new bound input is not the adopted S input')
    if adopt.get('deployment_index_unchanged')!=10 or adopt.get('resampling_performed') is not False:
        raise PreparationUnknown('inherited index contract differs')
    return {'status':'verified','source':'inherited adopted exported-record S; no HDF or generating model recheck',
      'manifest_sha256':bindings['files']['prior_data_manifest']['sha256'],'adoption_sha256':bindings['files']['prior_source_adoption']['sha256'],
      'record_start_delay_s':adopt.get('record_start_delay_s'),'deployment_since_clearing_s':adopt.get('deployment_since_clearing_s'),
      'historical_generator_execution_identity':'unknown','physical_unit_certification':'unknown','deployment_index':10,'resampling':False}

def multiply(a,b):
    out=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out

def derivative(a):return [F(i)*a[i] for i in range(1,len(a))]

def integral(a,lo,hi):return sum((value*(hi**(i+1)-lo**(i+1))/F(i+1) for i,value in enumerate(a)),F(0))

def rational_weights(steps,representation):
    length=F(steps,100);phi=[F(0),F(0),1/length**2,-2/length**3,1/length**4]
    kernels=[phi,derivative(phi),derivative(derivative(phi))];out=[[F(0)]*(steps+1) for _ in kernels]
    stride=2 if representation=='Q' else 1
    if representation not in ('Q','L') or steps not in (4,8):raise ValueError('only frozen operators allowed')
    for start in range(0,steps,stride):
        nodes=[F(i,100) for i in range(start,start+stride+1)]
        for offset,node in enumerate(nodes):
            basis=[F(1)]
            for other in nodes:
                if other!=node:basis=[v/(node-other) for v in multiply(basis,[-other,F(1)])]
            for order,kernel in enumerate(kernels):
                # Exact rational normalization precedes the one binary64 cast.
                out[order][start+offset]+=integral(multiply(basis,kernel),nodes[0],nodes[-1])/(length/F(30))
    return out

def qualification():
    import numpy as np
    import mpmath as mp
    operators={};rows=[]
    pairs=((0,0),(1,1),(2,1),(-1,2))
    def analytic_moment(degree,order,length):
        if degree<order:return F(0)
        remaining=degree-order
        moment=F(60)*length**remaining/F((remaining+3)*(remaining+4)*(remaining+5))
        factorial=math.prod(range(degree-order+1,degree+1))
        return (-1)**order*factorial*moment
    def reference(degree,order,steps,rep):
        if degree<0:return 0.,'identically zero input'
        length=F(steps,100)
        if rep=='Q' or degree<=1:return float(analytic_moment(degree,order,length)),'independent analytic moment identity'
        # Independent representation of the linear interpolant of t^2:
        # t^2+(t-lo)(hi-t). No generated Lagrange weights enter this reference.
        with mp.workdps(100):
            L=mp.mpf(steps)/100
            def kernel(t):
                if order==0:return (t/L)**2*(1-t/L)**2
                if order==1:return 2*t/L**2-6*t*t/L**3+4*t**3/L**4
                return 2/L**2-12*t/L**3+12*t*t/L**4
            value=mp.mpf(0)
            for j in range(steps):
                lo=mp.mpf(j)/100;hi=mp.mpf(j+1)/100
                value+=mp.quad(lambda t:kernel(t)*(t*t+(t-lo)*(hi-t)),[lo,hi])
            return float(value/(L/30)),'independent MP100 integral of piecewise linear t^2 interpolant'
    for steps in (4,8):
        for rep in ('Q','L'):
            rational=rational_weights(steps,rep);weights=np.asarray([[float(v) for v in row] for row in rational],dtype=np.float64)
            operators[f'{rep}{steps}']={'weights':weights,'rational':[[str(v) for v in row] for row in rational]}
            times=np.asarray([float(F(j,100)) for j in range(steps+1)],dtype=np.float64)
            for pair_index,(pd,wd) in enumerate(pairs):
                actions=[]
                for field,degree in (('p',pd),('omega',wd)):
                    values=np.zeros_like(times) if degree<0 else times**degree
                    for order in range(3):
                        actual=float(weights[order]@values);expected,method=reference(degree,order,steps,rep)
                        passed=bool(np.isfinite([actual,expected]).all() and np.isclose(actual,expected,atol=1e-12,rtol=1e-12))
                        actions.append({'field':field,'derivative_order':order,'actual':actual,'reference':expected,'reference_method':method,
                          'absolute_error':abs(actual-expected) if np.isfinite([actual,expected]).all() else None,'passed':passed})
                rows.append({'pair_index':pair_index,'steps':steps,'representation':rep,'status':'PASS' if all(a['passed'] for a in actions) else 'unknown','actions':actions})
    return operators,{'status':'PASS' if len(rows)==16 and all(r['status']=='PASS' for r in rows) else 'unknown','cells':rows,'cell_count':len(rows),'atol':1e-12,'rtol':1e-12,'MP_decimal_digits':100}

def array_hash(a):
    import numpy as np
    a=np.ascontiguousarray(a);h=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def p_scale_only(path):
    # Do not decode the old beta0 or candidate parameter records.
    text=Path(path).read_text(encoding='utf-8-sig');match=re.search(r'"p_scale"\s*:\s*(\[)',text)
    if match is None:raise PreparationUnknown('frozen p_scale not found')
    return json.JSONDecoder().raw_decode(text[match.start(1):])[0]

def read_events(paths):
    import numpy as np
    members=read(paths['prior_members']);frozen={m['id']:m for m in members}
    producer=read(paths['producer_prepare']);ids=producer['fit_ids']
    if len(frozen)!=297 or len(ids)!=297 or len(set(ids))!=297 or set(ids)!=set(frozen):raise PreparationUnknown('fixed297 member set differs')
    scale=np.asarray(p_scale_only(paths['prior_constants']),dtype=np.float64)
    if scale.shape!=(10,) or not np.isfinite(scale).all() or np.any(scale<=0):raise PreparationUnknown('p_scale invalid')
    events=[]
    with np.load(paths['fit_cache'],allow_pickle=False) as data:
        meta=json.loads(str(data['metadata_json']))
        if [m['id'] for m in meta]!=ids:raise PreparationUnknown('producer cache member order differs')
        for i,m in enumerate(meta):
            prior=frozen[m['id']];x=data[f'event_{i:04d}_x'];label=m['label']
            if label!=prior['label'] or x.ndim!=2 or x.shape[1]!=30 or x.dtype!=np.dtype('float64') or len(x)!=prior['coverage']['length'] or len(x)<13:
                raise PreparationUnknown('member shape/label/length differs')
            if array_hash(x[:13,10:30])!=prior['prefix_sha256']:raise PreparationUnknown('member prefix hash differs')
            if prior['time']['status']!='verified' or prior['coverage']['status']!='verified':raise PreparationUnknown('inherited member qualification unresolved')
            n=800 if label=='stable' else min(1500,len(x)-11)
            if n<2 or len(x)<11+n:raise PreparationUnknown('registered member interval missing')
            values=x[:11+n,10:30].copy()
            if not np.isfinite(values).all():raise PreparationUnknown('nonfinite member within registered analysis interval')
            events.append({'id':m['id'],'label':label,'n_future':n,'values':values})
    if len(events)!=297 or any(sum(e['label']==g for e in events)!=DENOM[g] for g in ('stable','unstable')):raise PreparationUnknown('267/30 labels differ')
    return sorted(events,key=lambda e:e['id']),scale

def construct_targets(events,scale,operators):
    import numpy as np
    blocks={key:{'X':[],'y':[]} for key in ('Q8','L8','Q4','L4')};row_records=[];offsets={8:0,4:0};raw={}
    for number,event in enumerate(events):
        values=event['values'];last=(len(values)-1)//8*8
        if last<8:raise PreparationUnknown('event has no complete80 window')
        raw[f'event_{number:04d}_omega_p']=values
        record={'id':event['id'],'label':event['label'],'n_future':event['n_future'],'last_original_index':len(values)-1,
          'last_scored_index':last,'unscored_tail_steps':len(values)-1-last,'relative_analysis_range_s':[-.1,event['n_future']*.01],
          'relative_last_scored_time_s':(last-10)*.01,'windows':{}}
        for steps in (8,4):
            starts=np.arange(0,last,steps);indices=starts[:,None]+np.arange(steps+1)[None,:];windows=values[indices]
            classes=['history' if start+steps<=10 else 'future' if start>=10 else 'cross_t0' for start in starts]
            count=len(starts);counts={c:classes.count(c) for c in ('history','cross_t0','future')}
            record['windows'][str(steps)]={'offset':offsets[steps],'count':count,'phase':0,'classes':classes,'class_counts':counts,
              'row_weight_all':1/(297*count),'row_weight_group':1/(DENOM[event['label']]*count),
              'class_weight_all':{c:v/(297*count) for c,v in counts.items()},'class_weight_group':{c:v/(DENOM[event['label']]*count) for c,v in counts.items()}}
            offsets[steps]+=count
            for rep in ('Q','L'):
                key=f'{rep}{steps}';w=operators[key]['weights']
                omega=[np.einsum('j,njk->nk',w[d],windows[:,:,:10]) for d in range(2)]
                p=[np.einsum('j,njk->nk',w[d],windows[:,:,10:]) for d in range(3)]
                X=np.stack([-p[1],p[0],-omega[1],omega[0]],axis=2)/scale[None,:,None];y=-p[2]/scale[None,:]
                if not np.isfinite(X).all() or not np.isfinite(y).all():raise PreparationUnknown('nonfinite raw target')
                blocks[key]['X'].append(X);blocks[key]['y'].append(y)
        row_records.append(record)
    arrays={}
    for key,block in blocks.items():
        arrays[key+'_X']=np.concatenate(block['X']);arrays[key+'_y']=np.concatenate(block['y'])
    for steps in (8,4):
        weights=np.concatenate([np.full(r['windows'][str(steps)]['count'],r['windows'][str(steps)]['row_weight_all']) for r in row_records])
        arrays[f'weights_all_{steps}']=weights
        for group in ('stable','unstable'):
            arrays[f'weights_{group}_{steps}']=np.concatenate([np.full(r['windows'][str(steps)]['count'],r['windows'][str(steps)]['row_weight_group'] if r['label']==group else 0.) for r in row_records])
    with np.errstate(over='ignore',invalid='ignore'):
        D=np.sqrt(np.sum(arrays['Q8_X']**2*arrays['weights_all_8'][:,None,None],axis=0))
    if not np.isfinite(D).all():raise PreparationUnknown('nonfinite mainQ column scales')
    arrays['D_main_Q80']=D;arrays['p_scale']=scale
    return arrays,row_records,raw

def public_coverage(rows):
    out={}
    for group in ('all','stable','unstable'):
        selected=[r for r in rows if group=='all' or r['label']==group];windows={}
        for steps in ('8','4'):
            windows[steps]={'total':sum(r['windows'][steps]['count'] for r in selected),
              'class_counts':{c:sum(r['windows'][steps]['class_counts'][c] for r in selected) for c in ('history','cross_t0','future')},
              'events_with_class':{c:sum(r['windows'][steps]['class_counts'][c]>0 for r in selected) for c in ('history','cross_t0','future')},
              'class_group_weight_sums':{c:sum(r['windows'][steps]['class_weight_all' if group=='all' else 'class_weight_group'][c] for r in selected) for c in ('history','cross_t0','future')}}
        out[group]={'denominator':DENOM[group],'recorded_events':len(selected),'windows':windows,
          'unscored_tail_step_counts':{str(i):sum(r['unscored_tail_steps']==i for r in selected) for i in range(8)},
          'actual_future_seconds_range':[min(r['n_future']*.01 for r in selected),max(r['n_future']*.01 for r in selected)] if selected else None}
    return out

def prepare():
    release=checked_release();claim_private(PRIVATE,release);start=time.perf_counter();w_done=False;real_decoded=False
    try:
        bindings=read(ROOT/FROZEN['source_bindings'][0]);paths=bound_paths(bindings);proof=source_proofs(paths,bindings)
        wstart=time.perf_counter();operators,w=qualification();w_seconds=time.perf_counter()-wstart;w_done=True
        write_once(PRIVATE/'operator_qualification.json',w)
        write_once(PRIVATE/'operators.json',{key:{'rational':value['rational'],'float64_weights':value['weights'].tolist()} for key,value in operators.items()})
        if w['status']!='PASS':raise PreparationUnknown('W did not pass; real arrays not decoded')
        rstart=time.perf_counter();real_decoded=True;events,scale=read_events(paths);read_seconds=time.perf_counter()-rstart
        tstart=time.perf_counter();arrays,rows,raw=construct_targets(events,scale,operators);target_seconds=time.perf_counter()-tstart
        import numpy as np
        for filename,values in (('targets.npz',arrays),('analysis_records.npz',raw)):
            with (PRIVATE/filename).open('xb') as stream:np.savez_compressed(stream,**values)
        write_once(PRIVATE/'rows.json',rows)
        target_identity={key:{'shape':list(value.shape),'dtype':str(value.dtype),'sha256':array_hash(value)} for key,value in arrays.items()}
        manifest={'status':'verified','stage':'prepare','source_proof':proof,'release':release,'bindings':bindings,'operator_sha256':sha(PRIVATE/'operators.json'),
          'qualification_sha256':sha(PRIVATE/'operator_qualification.json'),'targets_sha256':sha(PRIVATE/'targets.npz'),'raw_analysis_records_sha256':sha(PRIVATE/'analysis_records.npz'),
          'rows_sha256':sha(PRIVATE/'rows.json'),'target_array_identities':target_identity,'theta_column_order':['c1','c0','b1','b0'],
          'scope':'complete80 blocks from original index0;40 windows share identical80 coverage','denominators':DENOM,
          'main_column_scale':'main Q80 weighted column L2; exact zeros retained, no epsilon repair',
          'source_sha256':sha(Path(__file__)),'scientific_work':{'operator_cells':16,'real_prepare_attempts':1,'parameter_estimates':0,'QR':0,'optimizer':0,'ROM':0,'FOM':0,'model_writes':0},
          'cost':{'W_seconds':w_seconds,'read_seconds':read_seconds,'target_seconds':target_seconds,'wall_seconds':time.perf_counter()-start,'cpu_seconds':None,'LLM_seconds':None},
          'runtime':{'python':sys.version,'numpy':np.__version__,'device':'CPU','dtype':'float64','threads':1}}
        write_once(PRIVATE/'manifest.json',manifest)
        public={'status':'verified','stage':'prepare','denominators':DENOM,'coverage':public_coverage(rows),
          'W_status':w['status'],'W_cell_count':16,'W_max_absolute_error':max(a['absolute_error'] for row in w['cells'] for a in row['actions']),
          'source_qualification':'inherited adopted S; actual record first100ms, no reinterpolation','source_proof':proof,
          'manifest_sha256':sha(PRIVATE/'manifest.json'),'targets_sha256':manifest['targets_sha256'],'operator_sha256':manifest['operator_sha256'],
          'qualification_sha256':manifest['qualification_sha256'],'rows_sha256':manifest['rows_sha256'],'release':release,
          'scientific_work':manifest['scientific_work'],'cost':manifest['cost'],'dependent_stages_released':False}
        write_once(PUBLIC/'prepare_summary.json',public)
        write_once(PRIVATE/'prepare_complete.json',{'status':'COMPLETED','qualification':'verified','manifest_sha256':sha(PRIVATE/'manifest.json'),'summary_sha256':sha(PUBLIC/'prepare_summary.json')})
        return public
    except PreparationUnknown as exc:
        detail={'status':'unknown','stage':'prepare','reason':str(exc),'denominators':DENOM,'W_completed':w_done,
          'W_status':w['status'] if w_done else None,'W_cell_count':w['cell_count'] if w_done else 0,
          'qualification_sha256':sha(PRIVATE/'operator_qualification.json') if w_done else None,
          'real_arrays_decode_started':real_decoded,'dependent_stages_released':False,'release':release,'wall_seconds':time.perf_counter()-start}
        write_once(PRIVATE/'prepare_unknown.json',detail);write_once(PUBLIC/'prepare_summary.json',detail)
        write_once(PRIVATE/'prepare_complete.json',{'status':'COMPLETED','qualification':'unknown','unknown_sha256':sha(PRIVATE/'prepare_unknown.json')});return detail
    except BaseException as exc:
        write_once(PRIVATE/'prepare_failure.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'W_completed':w_done,'real_arrays_decode_started':real_decoded,'wall_seconds':time.perf_counter()-start,'automatic_retry':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare']);p.parse_args();prepare()
