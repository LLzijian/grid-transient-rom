"""Frozen-target E/Q only. Main estimates all seal before independent QR starts.

No W generation, target construction, filters, initialization, model or FOM.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1];RUN='p_dissipative_identifiability_20261031'
PRIVATE=ROOT/'private_data'/RUN;PUBLIC=ROOT/'results'/RUN
CONTRACT={
 'protocol':('reports/'+RUN+'_protocol.md','7c7fb0041c374ee5cd479cb5389a40b1a2a4eb0abe0eeb49bc8f896a19bb8b05'),
 'architecture':('reports/p_dissipative_closure_20261031_architecture_brief.md','e2731f82f46b72958c8eb79de069688b0fd4194400a86b2991aaa6d4c702e707'),
 'sensitivity':('reports/'+RUN+'_sensitivity_addendum.md','d52486189219de73b89f0a6e11d53f471888c2de879bf9fb4557377dc04413b2'),
 'source_bindings':('results/'+RUN+'/source_bindings.json','c5cf41b6694de55f1c6b2adba9b88f1e2bfb16edc9859a08c866935bb691712d'),
 'authorization':('results/'+RUN+'/authorization.json','496eeb96c636b72bc960a7af207adb04d71fa9af7e0b094cccf57ffb8b304323'),
 'prepare_adoption':('results/'+RUN+'/prepare_adoption.json','a9f4104a378d2873b4f3fa58b1b7780b1606160db62e429aac0df4988313a2d9'),
 'prepare_review':('results/'+RUN+'/prepare_evidence_review_at_adoption.md','95cd09907683c179716ffee7ac1ec4e43625e478a4819bdf5552fc64c68f35e5'),
 'prepare_summary':('results/'+RUN+'/prepare_summary.json','7f1e2a5a5ed89522c6d659b9b878685700cd251674e57f5085e29e5c4fe6d452'),
 'prepare_manifest':('private_data/'+RUN+'/manifest.json','9c6ec019075af9097a3528d237cbf53585a7f850627f646933cf0892ad94623e'),
 'targets':('private_data/'+RUN+'/targets.npz','60f54a2a0eca797319dec5d2f17177ae27d5c6fc1dc906743a1d9d00d1dc08d1'),
 'rows':('private_data/'+RUN+'/rows.json','7dcae10b585d4134616d406e6ce80a22561a9b43e4b37369274830be7d68a859')}
DENOM={'all':297,'stable':267,'unstable':30}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if hasattr(v,'tolist'):return safe(v.tolist())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(safe(value),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def digest(value):return hashlib.sha256(json.dumps(safe(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_hash(a):
    import numpy as np
    a=np.ascontiguousarray(a);h=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def release_expected(stage):
    if stage not in ('estimate','analyze'):raise ValueError('unknown E/Q stage')
    expected={'source_sha256':sha(Path(__file__)),'tests_sha256':sha(ROOT/'tests'/('test_'+RUN+'.py')),
      **{key+'_sha256':h for key,(path,h) in CONTRACT.items()},
      'main_parameter_estimates_upper_bound':10 if stage=='estimate' else 0,'same_target_qr_checks_upper_bound':10 if stage=='estimate' else 0,
      'fixed_analysis_suites_upper_bound':1 if stage=='analyze' else 0,'operator_qualification_cells_upper_bound':0,
      'optimizer_upper_bound':0,'new_p_filters_upper_bound':0,'initializers_upper_bound':0,
      'new_autonomous_rom_rollouts_upper_bound':0,'new_fom_upper_bound':0,'model_writes_upper_bound':0}
    if stage=='analyze':
        for key,name in (('estimate_manifest','manifest.json'),('estimate_complete','complete.json'),('all_main_seal','all_main_estimates_sealed.json')):
            p=PRIVATE/'estimate'/name;expected[key+'_sha256']=sha(p) if p.is_file() else None
    return expected
def verify_release(stage,release,expected):
    if release.get('released') is not True or release.get('allowed_stage')!=stage:raise PermissionError('root stage release required')
    for key,value in expected.items():
        if release.get(key)!=value or (type(value) is int and type(release.get(key)) is not int):raise PermissionError('identity/limit differs: '+key)
def checked_release(stage):
    expected=release_expected(stage)
    if stage=='analyze' and any(expected[k] is None for k in ('estimate_manifest_sha256','estimate_complete_sha256','all_main_seal_sha256')):raise PermissionError('completed immutable E identities required before Q release')
    for key,(path,h) in CONTRACT.items():
        if sha(ROOT/path)!=h:raise ValueError('frozen '+key+' changed')
    path=PUBLIC/(stage+'_execution_release.json');r=read(path);verify_release(stage,r,expected)
    review=(ROOT/r['review_snapshot_path']).resolve();review.relative_to(ROOT.resolve())
    if review.suffix.lower() not in ('.json','.md') or sha(review)!=r['review_snapshot_sha256']:raise ValueError('review snapshot differs')
    adoption=read(ROOT/CONTRACT['prepare_adoption'][0]);p=read(ROOT/CONTRACT['prepare_manifest'][0])
    if any(adoption.get(k) is not True for k in ('source_qualified','operator_qualified','target_construction_qualified')) or p.get('status')!='verified' or p.get('denominators')!=DENOM:
        raise ValueError('adopted P qualification unresolved')
    if p['targets_sha256']!=CONTRACT['targets'][1] or p['rows_sha256']!=CONTRACT['rows'][1]:raise ValueError('P artifact bindings differ')
    return {'release_sha256':sha(path),'review_snapshot_sha256':r['review_snapshot_sha256'],'identities':expected},p

def native_identity():
    result={'pid':os.getpid(),'creation_FILETIME_100ns':None}
    if os.name=='nt':
        try:
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)];kernel.GetProcessTimes.restype=wintypes.BOOL
            creation,exit_time,kernel_time,user_time=[wintypes.FILETIME() for _ in range(4)]
            if kernel.GetProcessTimes(kernel.GetCurrentProcess(),*[ctypes.byref(v) for v in (creation,exit_time,kernel_time,user_time)]):
                result['creation_FILETIME_100ns']=(creation.dwHighDateTime<<32)|creation.dwLowDateTime
        except (OSError,AttributeError,TypeError):pass
    return result
def claim_stage(folder,stage,release):
    Path(folder).mkdir(parents=True,exist_ok=False)
    write_once(Path(folder)/'started.json',{'status':'STARTED','stage':stage,'native_interpreter':native_identity(),'release':release})
def runtime():
    import numpy as np
    import scipy
    return {'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'device':'CPU','dtype':'float64','threads':1,
      'thread_environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}

def load_frozen(p):
    import numpy as np
    with np.load(ROOT/CONTRACT['targets'][0],allow_pickle=False) as data:
        if set(data.files)!=set(p['target_array_identities']):raise ValueError('frozen target member set differs')
        arrays={key:data[key].copy() for key in data.files}
    for key,a in arrays.items():
        spec=p['target_array_identities'][key]
        if a.dtype!=np.dtype('float64') or list(a.shape)!=spec['shape'] or array_hash(a)!=spec['sha256'] or not np.isfinite(a).all():raise ValueError('frozen target member differs')
    rows=read(ROOT/CONTRACT['rows'][0])
    ids=[r['id'] for r in rows]
    if len(rows)!=297 or len(set(ids))!=297 or ids!=sorted(ids) or any(sum(r['label']==g for r in rows)!=DENOM[g] for g in ('stable','unstable')):raise ValueError('frozen297 rows differ')
    for steps in (8,4):
        offset=0
        for row in rows:
            item=row['windows'][str(steps)];count=item['count']
            if count<=0 or item['offset']!=offset or item['phase']!=0:raise ValueError('frozen row offsets differ')
            if not np.array_equal(arrays[f'weights_all_{steps}'][offset:offset+count],np.full(count,1/(297*count))):raise ValueError('frozen event weights differ')
            offset+=count
        if arrays[f'Q{steps}_X'].shape!=(offset,10,4) or arrays[f'Q{steps}_y'].shape!=(offset,10):raise ValueError('frozen target shape differs')
    return arrays,rows

def target_for_machine(arrays,machine):
    import numpy as np
    D=arrays['D_main_Q80'][machine];w=arrays['weights_all_8'];X=arrays['Q8_X'][:,machine,:];y=arrays['Q8_y'][:,machine]
    if np.any(D<=0):return None,{'reason':'zero_column','D':D}
    A=X*np.sqrt(w[:,None])/D;v=y*np.sqrt(w)
    if not np.isfinite(A).all() or not np.isfinite(v).all():return None,{'reason':'nonfinite_scaled_target','D':D}
    return {'A':A,'v':v,'D':D},None

def main_estimate(target,machine):
    import numpy as np
    A,v,D=target['A'],target['v'],target['D'];u,s,vt=np.linalg.svd(A,full_matrices=False)
    threshold=max(A.shape)*np.finfo(np.float64).eps*s[0];keep=s>threshold;rank=int(np.sum(keep))
    gamma=vt[keep].T@((u[:,keep].T@v)/s[keep]);fitted=A@gamma;r=v-fitted;rN=A.T@r
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):theta=gamma/D
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        rhsnorm=float(np.linalg.norm(v));normalnorm=float(np.linalg.norm(rN));normalproduct=float(s[0]*rhsnorm)
        denominator=max(1.,normalproduct);normal=float(np.float64(normalnorm)/denominator)
    rebuild=(u*s)@vt;project=u[:,keep]@(u[:,keep].T@v)
    checks=bool(all(np.isfinite(x).all() for x in (s,gamma,theta,fitted,r,rN,rebuild,project)) and np.isfinite([rhsnorm,normalnorm,normalproduct,denominator,normal]).all()
      and normal<=1e-12 and np.allclose(A,rebuild,atol=1e-12,rtol=1e-12) and np.allclose(fitted,project,atol=1e-12,rtol=1e-12))
    return {'machine':machine,'status':'estimated','rank':rank,'threshold':threshold,'singular_values':s,'gamma':gamma,
      'theta':theta if rank==4 else None,'minimum_norm_projection_theta':theta if rank<4 else None,
      'fitted':fitted,'r':r,'r_N':rN,'D':D,'normalized_normal_equation_residual':normal,
      'normal_equation_scalars':{'rhs_norm':rhsnorm,'r_N_norm':normalnorm,'sigma_max_times_rhs_norm':normalproduct,'denominator':denominator},
      'reconstruction_maxabs':float(np.max(np.abs(A-rebuild))),'svd_checks_passed':checks,'unique':rank==4,
      'condition':float(s[0]/s[-1]) if rank==4 and s[-1]>0 and np.isfinite(s[0]/s[-1]) else None,
      'target_sha256':digest(target),'main_SVD_calls':1}

def same_target_qr(target,primary):
    import numpy as np
    from scipy.linalg import qr,solve_triangular
    if target is None or primary.get('rank') is None:return {'status':'unknown','reason':'no_estimated_target','QR_calls':0,'numeric_passed':False}
    A,v,D=target['A'],target['v'],target['D'];q,r,pivot=qr(A,mode='economic',pivoting=True)
    rank=primary['rank'];gamma=np.zeros(4)
    if rank:gamma[pivot[:rank]]=solve_triangular(r[:rank,:rank],(q.T@v)[:rank])
    fit=A@gamma;finite=bool(np.isfinite(gamma).all() and np.isfinite(fit).all())
    vector_ok=finite and np.allclose(fit,np.asarray(primary['fitted']),atol=1e-12,rtol=1e-12)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):theta=gamma/D
    theta_ok=bool(np.isfinite(theta).all() and np.allclose(theta,np.asarray(primary['theta'],dtype=np.float64),atol=1e-12,rtol=1e-12)) if rank==4 and finite else None
    passed=bool(primary['svd_checks_passed'] and vector_ok and (theta_ok is True or rank<4))
    return {'status':'verified' if passed else 'unknown','QR_calls':1,'numeric_passed':passed,'theta_equal_1e12':theta_ok,
      'fitted_equal_1e12':bool(vector_ok),'gamma_reference':gamma,'fit_reference':fit,'same_target_sha256':digest(target),'parameter_candidate_created':False}

def estimate():
    release,p=checked_release('estimate');folder=PRIVATE/'estimate';claim_stage(folder,'estimate',release)
    start=time.perf_counter();cpu=time.process_time()
    try:
        arrays,rows=load_frozen(p);primaries=[];targets=[]
        for machine in range(10):
            slot=folder/f'machine_{machine:02d}';slot.mkdir(exist_ok=False)
            write_once(slot/'main_started.json',{'machine':machine,'targets_sha256':CONTRACT['targets'][1],'maximum_main_SVD_calls':1})
            before=time.perf_counter();target,problem=target_for_machine(arrays,machine);targets.append(target)
            if target is None:primary={'machine':machine,'status':'unidentifiable','reason':problem['reason'],'rank':None,'theta':None,'gamma':None,'unique':False,'svd_checks_passed':False,'main_SVD_calls':0,'D':problem['D']}
            else:
                try:primary=main_estimate(target,machine)
                except (np_error(),FloatingPointError,ValueError) as exc:primary={'machine':machine,'status':'unknown','reason':repr(exc),'rank':None,'theta':None,'gamma':None,'unique':False,'svd_checks_passed':False,'main_SVD_calls':1}
            primary['wall_seconds']=time.perf_counter()-before
            write_once(slot/'parameters.json',primary);primaries.append(read(slot/'parameters.json'))
        # All ten immutable main artifacts must exist before the first QR call.
        seal={'parameters_sha256':[sha(folder/f'machine_{i:02d}'/'parameters.json') for i in range(10)],'targets_sha256':CONTRACT['targets'][1],'release':release}
        write_once(folder/'all_main_estimates_sealed.json',seal)
        summaries=[]
        for machine,(target,primary) in enumerate(zip(targets,primaries)):
            slot=folder/f'machine_{machine:02d}'
            if sha(slot/'parameters.json')!=seal['parameters_sha256'][machine]:raise ValueError('sealed main estimate changed')
            write_once(slot/'QR_started.json',{'main_seal_sha256':sha(folder/'all_main_estimates_sealed.json'),'same_target_only':True})
            before=time.perf_counter()
            try:check=same_target_qr(target,primary)
            except (np_error(),FloatingPointError,ValueError) as exc:check={'status':'unknown','reason':repr(exc),'QR_calls':1,'numeric_passed':False}
            check['wall_seconds']=time.perf_counter()-before;write_once(slot/'QR.json',check)
            receipt={'machine':machine,'parameters_sha256':sha(slot/'parameters.json'),'QR_sha256':sha(slot/'QR.json'),
              'numeric_passed':check['numeric_passed'],'unique':primary['unique'],'main_SVD_calls':primary['main_SVD_calls'],'QR_calls':check['QR_calls']}
            write_once(slot/'complete.json',receipt);summaries.append({**receipt,'receipt_sha256':sha(slot/'complete.json')})
        result={'status':'COMPLETED','stage':'estimate','slots':summaries,'main_seal_sha256':sha(folder/'all_main_estimates_sealed.json'),
          'release':release,'denominators':DENOM,'runtime':runtime(),'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,
          'optimizer_calls':0,'new_filters':0,'ROM':0,'FOM':0,'model_writes':0}
        write_once(folder/'manifest.json',result);write_once(PUBLIC/'estimate_summary.json',result)
        write_once(folder/'complete.json',{'status':'COMPLETED','manifest_sha256':sha(folder/'manifest.json')});return result
    except BaseException as exc:
        write_once(folder/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'automatic_retry':False});raise

def np_error():
    import numpy as np
    return np.linalg.LinAlgError

def rounded(value,direction):
    if not math.isfinite(value):return None
    result=math.nextafter(value,math.inf if direction>0 else -math.inf)
    return result if math.isfinite(result) else None
def valid_interval(x):return x is not None and len(x)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in x) and x[0]<=x[1]
def interval_binary(a,b,operation):
    if not valid_interval(a) or not valid_interval(b) or (operation=='div' and b[0]<=0<=b[1]):return None
    try:
        if operation=='sub':values=[a[0]-b[1],a[1]-b[0]]
        elif operation=='mul':values=[x*y for x in a for y in b]
        elif operation=='div':values=[x/y for x in a for y in b]
        else:raise ValueError('unsupported interval operation')
        lo=rounded(min(values),-1);hi=rounded(max(values),1)
        return [lo,hi] if lo is not None and hi is not None else None
    except (OverflowError,ZeroDivisionError):return None
def theta_box(theta,radius,D):
    if not math.isfinite(radius) or radius<0 or len(theta)!=4 or len(D)!=4:return None
    boxes=[];widths=[]
    for center,scale in zip(theta,D):
        if not math.isfinite(center) or not math.isfinite(scale) or scale<=0:return None
        raw=radius/scale;low=rounded(raw,-1);high=rounded(raw,1)
        if low is None or high is None:return None
        lo=rounded(center-high,-1);hi=rounded(center+high,1)
        if lo is None or hi is None:return None
        widths.append([low,high]);boxes.append([lo,hi])
    return {'theta':boxes,'halfwidths':widths}
def map_box(box):
    if box is None or not isinstance(box,dict) or len(box.get('theta',[]))!=4 or not all(valid_interval(v) for v in box['theta']):return {'status':'unknown','reason':'invalid_initial_box','R':None,'J':None}
    c1,c0,g,b0=box['theta']
    if g[0]<=0<=g[1]:return {'status':'unknown','reason':'g_box_touches_zero','R':None,'J':None}
    c=interval_binary(b0,g,'div');a=interval_binary(c1,c,'sub');product=interval_binary(a,c,'mul');k=interval_binary(c0,product,'sub')
    if not all(valid_interval(i) for i in (g,c,a,c0,k)):return {'status':'unknown','reason':'nonfinite_or_invalid_propagation','R':None,'J':None}
    common=g[0]>0 and a[0]>=0 and c[0]>=0
    r=bool(common and c0[0]>=0 and k[1]<0);j=bool(common and k[0]>0)
    return {'status':'classified','R':r,'J':j,'classification':'R' if r else 'J' if j else 'unresolved',
      'g':g,'c':c,'a':a,'c0':c0,'k':k,'k_touches_or_crosses_zero':k[0]<=0<=k[1],
      'whole_box_violates_g_positive':g[1]<=0,'scope':'outward propagation of already computed scalar radius; not a verified-SVD certificate'}

def sensitivity_progress(state):
    """Snapshot already computed values; never evaluate a radius or a matrix."""
    fields=('D','sigma_min','delta_A','delta_v','gamma_norm','r_norm','r_N_norm','v_norm','t',
      'radius_term1','radius_term2','radius','r','r_N','A_L','v_L')
    saved=safe({**{key:None for key in fields},**state})
    def ratio(numerator,denominator,floor=None):
        n=saved[numerator];d=saved[denominator]
        if not isinstance(n,(int,float)) or not isinstance(d,(int,float)) or n<0 or d<0:return None
        if floor is None and d<=0:return None
        value=n/(max(d,floor) if floor is not None else d)
        return value if math.isfinite(value) else None
    return {'intermediate':saved,'relative':{
      'delta_A_over_sigma_min':ratio('delta_A','sigma_min'),
      'delta_v_over_v':ratio('delta_v','v_norm',1e-30),
      'radius_over_gamma':ratio('radius','gamma_norm',1e-30)}}

def sensitivity_unknown(reason,stage,state):
    return {'status':'unknown','reason':reason,'failure_stage':stage,**sensitivity_progress(state),
      'theta_box':None,'mapping':{'status':'unknown','R':None,'J':None}}

def sensitivity(target,arrays,machine,primary,qualified):
    import numpy as np
    # Sealed E quantities already exist even when Q's numerical premise fails.
    singular=primary.get('singular_values') or []
    normal_scalars=primary.get('normal_equation_scalars',{})
    state={'D':primary.get('D'),'sigma_min':singular[-1] if singular else None,
      'r':primary.get('r'),'r_N':primary.get('r_N'),'v_norm':normal_scalars.get('rhs_norm'),
      'r_N_norm':normal_scalars.get('r_N_norm'),'residual_source':'sealed E until Q recomputation'}
    unknown=lambda reason,stage:sensitivity_unknown(reason,stage,state)
    if target is None or not qualified or not primary.get('unique'):return unknown('original_rank_or_numeric_gate_not_passed','original_qualification')
    A,v,D=target['A'],target['v'],target['D'];gamma=np.asarray(primary['gamma']);theta=np.asarray(primary['theta']);weights=arrays['weights_all_8']
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        AL=arrays['L8_X'][:,machine,:]*np.sqrt(weights[:,None])/D;vL=arrays['L8_y'][:,machine]*np.sqrt(weights)
    state.update(D=D,A_L=AL,v_L=vL)
    if not np.isfinite(AL).all() or not np.isfinite(vL).all():return unknown('nonfinite_scaled_L_representation','scaled_L')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        r=v-A@gamma;rN=A.T@r
        state.update(r=r,r_N=rN,residual_source='Q recomputation with sealed gamma')
        deltav=float(np.linalg.norm(vL-v));smin=float(primary['singular_values'][-1])
        gn=float(np.linalg.norm(gamma));rn=float(np.linalg.norm(r));normal=float(np.linalg.norm(rN));vn=float(np.linalg.norm(v))
        state.update(delta_v=deltav,sigma_min=smin,gamma_norm=gn,r_norm=rn,r_N_norm=normal,v_norm=vn)
        try:deltaA=float(np.linalg.norm(AL-A,2))
        except np.linalg.LinAlgError:return unknown('diagnostic_spectral_norm_unresolved','spectral_norm')
        t=smin-deltaA
        state.update(delta_A=deltaA,t=t)
    scalars=[deltaA,deltav,smin,gn,rn,normal,vn,t]
    if not np.isfinite(scalars).all() or min(deltaA,deltav,gn,rn,normal,vn)<0 or t<=0 or np.any(D<=0):return unknown('nonfinite_or_nonpositive_sensitivity_premise','radius_premise')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        term1=(deltav+deltaA*gn)/np.float64(t);term2=(deltaA*rn+normal)/(np.float64(t)*np.float64(t));radius=term1+term2
    state.update(radius_term1=term1,radius_term2=term2,radius=radius)
    if not np.isfinite([term1,term2,radius]).all() or min(term1,term2,radius)<0:return unknown('nonfinite_or_negative_radius','radius')
    box=theta_box(theta.tolist(),radius,D.tolist());mapping=map_box(box)
    progress=sensitivity_progress(state)
    if any(value is None for value in progress['relative'].values()):
        result=unknown('nonfinite_relative_sensitivity','relative_summary');result['theta_box']=box;return result
    return {'status':'computed' if mapping['status']=='classified' else 'unknown','reason':mapping.get('reason'),
      'failure_stage':None if mapping['status']=='classified' else 'box_mapping',
      **progress,'mapping':mapping,'theta_box':box}

def fixed_stats(arrays,rows,machine,theta,steps,group):
    import numpy as np
    expected=DENOM[group];left=[];right=[];details=[]
    def unknown(reason):return {'status':'unknown','reason':reason,'denominator':expected,'available_events':len(left),'Q_Q':None,'Q_L':None,'nu':None,'compatible':None,'resolved':None,'classification':'unknown'},details
    selected=[r for r in rows if group=='all' or r['label']==group]
    if theta is None or len(selected)!=expected:return unknown('no_unique_qualified_theta_or_incomplete_group')
    for row in selected:
        item=row['windows'][str(steps)];start=item['offset'];end=start+item['count']
        with np.errstate(over='ignore',invalid='ignore'):
            q=arrays[f'Q{steps}_y'][start:end,machine]-arrays[f'Q{steps}_X'][start:end,machine,:]@theta
            l=arrays[f'L{steps}_y'][start:end,machine]-arrays[f'L{steps}_X'][start:end,machine,:]@theta
        if not len(q) or not np.isfinite(q).all() or not np.isfinite(l).all():return unknown('missing_or_nonfinite_event_residual')
        left.append(q);right.append(l);details.append({'id':row['id'],'Q_residual':q,'L_residual':l})
    with np.errstate(over='ignore',invalid='ignore'):
        Q=float(np.sqrt(np.mean([np.mean(v*v) for v in left])));L=float(np.sqrt(np.mean([np.mean(v*v) for v in right])))
        difference=float(np.sqrt(np.mean([np.mean((a-b)**2) for a,b in zip(left,right)])));nu=max(1e-10,difference);threshold=10*nu
    if not np.isfinite([Q,L,difference,nu,threshold]).all():return unknown('nonfinite_derived_statistics')
    compatible=max(Q,L)<=threshold;resolved=min(Q,L)>threshold
    return {'status':'computed','denominator':expected,'available_events':expected,'Q_Q':Q,'Q_L':L,'nu':nu,
      'representation_difference_RMS':difference,'compatible':bool(compatible),'resolved':bool(resolved),
      'classification':'compatible' if compatible else 'resolved' if resolved else 'mixed','units':'normalized p second-variation rate 1/s^2'},details

def analyze():
    release,p=checked_release('analyze');E=PRIVATE/'estimate';ecomplete=read(E/'complete.json')
    if (PRIVATE/'analyze').exists():raise FileExistsError('Q already started or completed; no replay')
    if sha(E/'manifest.json')!=ecomplete['manifest_sha256']:raise ValueError('E completion differs')
    em=read(E/'manifest.json')
    if em['release']['identities']!=release_expected('estimate'):raise ValueError('E frozen implementation differs')
    if sha(E/'all_main_estimates_sealed.json')!=em['main_seal_sha256'] or len(em['slots'])!=10:raise ValueError('all-main seal differs')
    primaries=[];qualifications=[]
    for machine,receipt in enumerate(em['slots']):
        slot=E/f'machine_{machine:02d}'
        if receipt['machine']!=machine or sha(slot/'complete.json')!=receipt['receipt_sha256'] or sha(slot/'parameters.json')!=receipt['parameters_sha256'] or sha(slot/'QR.json')!=receipt['QR_sha256']:raise ValueError('E slot artifact differs')
        primaries.append(read(slot/'parameters.json'));qualifications.append(receipt['numeric_passed'])
    folder=PRIVATE/'analyze';claim_stage(folder,'analyze',release);start=time.perf_counter();cpu=time.process_time()
    try:
        arrays,rows=load_frozen(p);public=[]
        for machine,primary in enumerate(primaries):
            target,problem=target_for_machine(arrays,machine)
            if target is not None and primary.get('target_sha256') is not None and digest(target)!=primary['target_sha256']:raise ValueError('frozen E/Q target differs')
            qualified=qualifications[machine] and primary.get('unique') is True
            theta=None if not qualified else __import__('numpy').asarray(primary['theta'])
            sense=sensitivity(target,arrays,machine,primary,qualified);stats={};details={}
            for group in ('all','stable','unstable'):
                stats[group]={};details[group]={}
                for steps in (8,4):stats[group][str(steps)],details[group][str(steps)]=fixed_stats(arrays,rows,machine,theta,steps,group)
            private={'machine':machine,'sensitivity':sense,'fixed_residuals':details,'statistics':stats,'parameters_sha256':em['slots'][machine]['parameters_sha256']}
            write_once(folder/f'machine_{machine:02d}.json',private)
            mapping=sense['mapping'];valid=sense['status']=='computed' and mapping['status']=='classified'
            compatible=all(stats[g][s].get('compatible') is True for g in ('stable','unstable') for s in ('8','4'))
            resolved=any(stats[g][s].get('resolved') is True for g in ('stable','unstable') for s in ('8','4'))
            r=bool(mapping.get('R')) if valid else None;j=bool(mapping.get('J')) if valid else None
            public.append({'machine':machine,'sensitivity_status':sense['status'],'reason':sense.get('reason'),
              'relative_sensitivity':sense['relative'],'R_robust':r,'J_robust':j,
              'classification':'R_weak_compatible' if r and compatible else 'R_with_remaining_residual' if r and resolved else 'R_other_weak_status' if r else 'J_robust' if j else 'unresolved',
              'k_touches_or_crosses_zero':mapping.get('k_touches_or_crosses_zero') if valid else None,
              'whole_box_violates_g_positive':mapping.get('whole_box_violates_g_positive') if valid else None,
              'fixed_residuals':stats,'private_evidence_sha256':sha(folder/f'machine_{machine:02d}.json')})
        # Q is strictly read-only with respect to every sealed E slot.
        for machine,receipt in enumerate(em['slots']):
            slot=E/f'machine_{machine:02d}'
            if sha(slot/'parameters.json')!=receipt['parameters_sha256'] or sha(slot/'QR.json')!=receipt['QR_sha256'] or sha(slot/'complete.json')!=receipt['receipt_sha256']:
                raise ValueError('E slot changed during Q')
        if sha(E/'manifest.json')!=release['identities']['estimate_manifest_sha256'] or sha(E/'complete.json')!=release['identities']['estimate_complete_sha256'] or sha(E/'all_main_estimates_sealed.json')!=release['identities']['all_main_seal_sha256']:
            raise ValueError('E manifest or main seal changed during Q')
        result={'status':'COMPLETED','stage':'analyze','machines':public,'denominators':DENOM,'release':release,
          'runtime':runtime(),'E_slots_unchanged':True,'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'new_parameter_estimates':0,'new_QR_checks':0,
          'new_filters':0,'initializers':0,'optimizer':0,'ROM':0,'FOM':0,'model_writes':0,'model_adopted':False,
          'scope':'consumed train, frozen complete80-block Q/L representations; operational sensitivity, no verified floating-point/noise/statistical/continuous-truth certificate'}
        write_once(folder/'manifest.json',result);write_once(PUBLIC/'analysis_summary.json',result)
        write_once(folder/'complete.json',{'status':'COMPLETED','manifest_sha256':sha(folder/'manifest.json')});return result
    except BaseException as exc:
        write_once(folder/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'automatic_retry':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['estimate','analyze']);args=p.parse_args();estimate() if args.stage=='estimate' else analyze()
