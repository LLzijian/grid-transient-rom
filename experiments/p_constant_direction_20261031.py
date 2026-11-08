"""Sealed scalar E and fixed-constant Q on adopted selected-residual snapshots only."""
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
ROOT=Path(__file__).resolve().parents[1];RUN='p_constant_direction_20261031'
PRIVATE=ROOT/'private_data'/RUN;PUBLIC=ROOT/'results'/RUN
DENOM={'all':297,'stable':267,'unstable':30};PHASES=('history','cross_t0','future')
FROZEN={
 'scientist':('reports/'+RUN+'_scientist.md','a25715fe056629ab63204905125ae2a1827c03386c29b001b403ddb54fca4be1'),
 'protocol':('reports/'+RUN+'_protocol.md','c7c69017ae1608f06469e9fa3cda6a4bb7427acdf0abc095aeed180be8c39892'),
 'source_bindings':('results/'+RUN+'/source_bindings.json','2fe73e3b9c16200fbf3637182a8e53d737ab6225a8df62b83dc10026fefe578e'),
 'authorization':('results/'+RUN+'/authorization.json','f8881aa127a7b1cc6c172274a44e890ed592926950ecd236aa2287eb57c7f12c'),
 'source_adoption':('results/'+RUN+'/source_adoption.json','7b4784a3bc86fd9881a8b0fb6952d09ed35a2ed41bcc949885208ec0a780a1b8'),
 'source_review':('results/'+RUN+'/source_evidence_review_at_adoption.md','5f8b69142fcd699b722bff8f74d6360ba162cfb0baa05eb36e446128576d1fb7'),
 'source_manifest':('private_data/'+RUN+'/manifest.json','4eceb511d4048e72b100bb3d45e8b75f49622f69d7d81d87d041f6511202b45f'),
 'source_summary':('results/'+RUN+'/source_summary.json','e8236e3299ff30964315556324e8e54d9a43f793a6fd6447c0f9b0af781652b1'),
 'residual_snapshot':('private_data/'+RUN+'/frozen_residuals.npz','5d565747d5b95c6ae19711c6522fada23f11940bfcd0b1a53cf9a6123b05307f'),
 'rows':('private_data/'+RUN+'/rows_frozen.json','7dcae10b585d4134616d406e6ce80a22561a9b43e4b37369274830be7d68a859'),
 'parent_statistics':('private_data/'+RUN+'/parent_statistics.json','0e5e751121a63fa4aa59bdae65cd32ff24753ec39545a5327a30d97fee85821b')}
class NumericUnknown(Exception):pass
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [safe(x) for x in v]
    if hasattr(v,'tolist'):return safe(v.tolist())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(safe(value),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def digest(v):return hashlib.sha256(json.dumps(safe(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def close(a,b):return finite(a) and finite(b) and abs(a-b)<=1e-12+1e-12*abs(b)
def array_hash(a):
    import numpy as np
    a=np.ascontiguousarray(a);h=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def release_expected(stage):
    if stage not in ('estimate','analyze'):raise ValueError('unknown stage')
    out={'source_sha256':sha(Path(__file__)),'tests_sha256':sha(ROOT/'tests'/('test_'+RUN+'.py')),
      **{k+'_sha256':h for k,(path,h) in FROZEN.items()},'eta_estimates_upper_bound':10 if stage=='estimate' else 0,
      'MP100_checks_upper_bound':10 if stage=='estimate' else 0,'overall_cells_upper_bound':60 if stage=='analyze' else 0,
      'stage_cells_upper_bound':120 if stage=='analyze' else 0,
      **{k+'_upper_bound':0 for k in ('four_coefficient_fits','QR','new_residual_Xtheta_minus_y','integrals','sensitivity_bounds','filters','initializers','optimizer','ROM','FOM','model_writes')}}
    if stage=='analyze':
        for key,name in (('estimate_manifest','manifest.json'),('estimate_complete','complete.json'),('all_eta_sealed','all_eta_sealed.json')):
            p=PRIVATE/'estimate'/name;out[key+'_sha256']=sha(p) if p.is_file() else None
    return out
def verify_release(stage,r,expected):
    if r.get('released') is not True or r.get('allowed_stage')!=stage:raise PermissionError('root E/Q release required')
    for k,v in expected.items():
        if r.get(k)!=v or (type(v) is int and type(r.get(k)) is not int):raise PermissionError('stage identity/limit differs: '+k)
def verify_inputs():
    for key,(path,h) in FROZEN.items():
        if sha(ROOT/path)!=h:raise ValueError('frozen '+key+' differs')
def checked_release(stage):
    expected=release_expected(stage)
    if stage=='analyze' and any(expected[k] is None for k in ('estimate_manifest_sha256','estimate_complete_sha256','all_eta_sealed_sha256')):raise PermissionError('completed E required before Q release')
    if stage=='analyze':
        e=read(PRIVATE/'estimate'/'manifest.json')
        if e.get('qualification')!='verified' or len(e.get('slots',[]))!=10 or any(s.get('numeric_passed') is not True for s in e['slots']):raise PermissionError('all ten E numerical checks required before Q release')
    verify_inputs();path=PUBLIC/(stage+'_execution_release.json');r=read(path);verify_release(stage,r,expected)
    review=(ROOT/r['review_snapshot_path']).resolve();review.relative_to(ROOT.resolve())
    if review.suffix.lower() not in ('.json','.md') or sha(review)!=r['review_snapshot_sha256']:raise ValueError('review snapshot differs')
    adoption=read(ROOT/FROZEN['source_adoption'][0]);manifest=read(ROOT/FROZEN['source_manifest'][0])
    if adoption.get('source_alignment_qualified') is not True or adoption.get('parent60_and120_closures_qualified') is not True or manifest.get('status')!='verified' or manifest.get('denominators')!=DENOM:raise ValueError('adopted source not qualified')
    if manifest['residual_snapshot_sha256']!=FROZEN['residual_snapshot'][1] or manifest['rows_sha256']!=FROZEN['rows'][1] or manifest['parent_statistics_sha256']!=FROZEN['parent_statistics'][1]:raise ValueError('adopted artifact chain differs')
    return {'release_sha256':sha(path),'review_snapshot_sha256':r['review_snapshot_sha256'],'identities':expected},manifest
def native_identity():
    result={'pid':os.getpid(),'creation_FILETIME_100ns':None}
    if os.name=='nt':
        try:
            import ctypes
            from ctypes import wintypes
            k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE
            k.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)];k.GetProcessTimes.restype=wintypes.BOOL
            values=[wintypes.FILETIME() for _ in range(4)]
            if k.GetProcessTimes(k.GetCurrentProcess(),*[ctypes.byref(x) for x in values]):result['creation_FILETIME_100ns']=(values[0].dwHighDateTime<<32)|values[0].dwLowDateTime
        except (OSError,AttributeError,TypeError):pass
    return result
def claim_stage(path,stage,release):
    Path(path).mkdir(parents=True,exist_ok=False)
    write_once(Path(path)/'started.json',{'status':'STARTED','stage':stage,'native_interpreter':native_identity(),'python_executable':sys.executable,'argv':list(sys.argv),'release':release})
def runtime():
    return {'python':sys.version,'numpy':getattr(sys.modules.get('numpy'),'__version__',None),
      'mpmath':getattr(sys.modules.get('mpmath'),'__version__',None),'device':'CPU','dtype':'float64','threads':1,
      'thread_environment':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}}

def load_frozen(manifest):
    import numpy as np
    with np.load(ROOT/FROZEN['residual_snapshot'][0],allow_pickle=False) as data:
        if set(data.files)!={'Q8','L8','Q4','L4'}:raise ValueError('snapshot field set differs')
        arrays={k:data[k].copy() for k in data.files}
    for k,a in arrays.items():
        spec=manifest['snapshots'][k]
        if a.dtype!=np.dtype('float64') or list(a.shape)!=spec['shape'] or array_hash(a)!=spec['sha256'] or not np.isfinite(a).all():raise NumericUnknown('snapshot dtype/shape/hash/finite mismatch')
    rows=read(ROOT/FROZEN['rows'][0]);ids=[r['id'] for r in rows]
    if len(rows)!=297 or len(set(ids))!=297 or ids!=sorted(ids) or any(sum(r['label']==g for r in rows)!=DENOM[g] for g in ('stable','unstable')):raise NumericUnknown('frozen297 ids/groups/order mismatch')
    for s in (8,4):
        offset=0
        for r in rows:
            item=r['windows'][str(s)];m=item['count'];last=r['last_scored_index']
            classes=['history' if j+s<=10 else 'future' if j>=10 else 'cross_t0' for j in range(0,last,s)]
            if type(m) is not int or m<=0 or item['offset']!=offset or item['phase']!=0 or item['classes']!=classes or m!=last//s or not close(item['row_weight_all'],1/(297*m)) or not close(item['row_weight_group'],1/(DENOM[r['label']]*m)):raise NumericUnknown('frozen offsets/phase/classes/weights mismatch')
            offset+=m
        if any(arrays[rep+str(s)].shape!=(offset,10) for rep in ('Q','L')):raise NumericUnknown('snapshot row extent mismatch')
    parent=read(ROOT/FROZEN['parent_statistics'][0]);overall={(r['machine'],r['steps'],r['group']):r for r in parent['overall60']};stages={(r['machine'],r['steps'],r['group'],r['phase']):r for r in parent['stage120']}
    if len(parent['overall60'])!=60 or set(overall)!={(i,s,g) for i in range(10) for s in (8,4) for g in DENOM}:raise NumericUnknown('parent overall statistic keys mismatch')
    if len(parent['stage120'])!=120 or set(stages)!={(i,s,g,p) for i in range(10) for s in (8,4) for g in ('stable','unstable') for p in PHASES}:raise NumericUnknown('parent stage statistic keys mismatch')
    return arrays,rows,overall,stages
def event_vectors(array,rows,machine):
    out=[]
    for row in rows:
        item=row['windows']['8'];out.append([float(v) for v in array[item['offset']:item['offset']+item['count'],machine]])
    return out
def weighted_mean(events,N):
    if len(events)!=N or any(not values or any(not finite(v) for v in values) for values in events):raise NumericUnknown('invalid fixed mean inputs')
    result=math.fsum(math.fsum(values)/len(values) for values in events)/N
    if not finite(result):raise NumericUnknown('nonfinite main mean')
    return result
def binary_projection_diagnostics(events,eta,N):
    if not finite(eta):raise NumericUnknown('nonfinite sealed constant')
    corrected=[[v-eta for v in values] for values in events]
    if any(not finite(v) for values in corrected for v in values):raise NumericUnknown('nonfinite corrected residual')
    mean=weighted_mean(corrected,N)
    J0=math.fsum(math.fsum(v*v for v in values)/len(values) for values in events)/N
    J=math.fsum(math.fsum(v*v for v in values)/len(values) for values in corrected)/N
    identity=J+eta*eta
    if not all(finite(v) and v>=0 for v in (J0,J,identity)):raise NumericUnknown('nonfinite direct square diagnostic')
    return {'corrected_mean':mean,'J0':J0,'J_eta_direct':J,'J_eta_plus_eta_squared':identity,'identity_passed':close(J0,identity),'corrected_mean_zero':close(mean,0.)}
def seal_all_eta(folder):
    folder=Path(folder);paths=[folder/f'machine_{i:02d}'/'eta.json' for i in range(10)]
    if not all(p.is_file() for p in paths):raise RuntimeError('all ten main eta files must precede MP checks')
    seal={'eta_sha256':[sha(p) for p in paths]};write_once(folder/'all_eta_sealed.json',seal);return seal
def verify_eta_seal(folder,seal):
    if len(seal['eta_sha256'])!=10 or any(sha(Path(folder)/f'machine_{i:02d}'/'eta.json')!=h for i,h in enumerate(seal['eta_sha256'])):raise ValueError('sealed main constant changed')
def mp_check(events,eta,N,parent_J0,progress=None):
    import mpmath as mp
    binary=binary_projection_diagnostics(events,eta,N)
    if progress is not None:progress['MP100_checks_started']+=1
    with mp.workdps(100):
        def exact(v):
            n,d=float(v).as_integer_ratio();return mp.mpf(n)/mp.mpf(d)
        values=[[exact(v) for v in row] for row in events];eta64=exact(eta)
        average=lambda rows:mp.fsum(mp.fsum(row)/len(row) for row in rows)/N
        eta_ref=average(values);mean_ref=average([[v-eta64 for v in row] for row in values])
        J0=average([[v*v for v in row] for row in values]);Jmain=average([[(v-eta64)**2 for v in row] for row in values]);Joptimal=average([[(v-eta_ref)**2 for v in row] for row in values])
        identity_error=J0-Joptimal-eta_ref**2
        references={k:mp.nstr(v,105) for k,v in {'eta':eta_ref,'corrected_mean_main_eta':mean_ref,'J0':J0,'J_main_eta':Jmain,'J_optimal_eta':Joptimal,'identity_error':identity_error}.items()}
        rounded={k:float(v) for k,v in {'eta':eta_ref,'corrected_mean':mean_ref,'J0':J0,'J':Jmain,'Joptimal':Joptimal,'identity_error':identity_error}.items()}
    passed=all(finite(v) for v in rounded.values()) and binary['identity_passed'] and binary['corrected_mean_zero'] and close(eta,rounded['eta']) and close(rounded['corrected_mean'],0.) and close(binary['corrected_mean'],rounded['corrected_mean']) and close(binary['J_eta_direct'],rounded['J']) and close(binary['J_eta_direct'],rounded['Joptimal']) and close(binary['J0'],rounded['J0']) and close(binary['J0'],parent_J0) and close(rounded['identity_error'],0.)
    return {'status':'verified' if passed else 'unknown','numeric_passed':bool(passed),'MP_checks':1,'MP_decimal_digits':100,'binary':binary,'MP_references':references,'main_eta_replaced':False,'atol':1e-12,'rtol':1e-12}

def estimate():
    release,manifest=checked_release('estimate');folder=PRIVATE/'estimate';claim_stage(folder,'estimate',release);start=time.perf_counter();cpu=time.process_time()
    counts={'eta_attempts':0,'eta_files_sealed':0,'MP_attempts':0,'MP100_checks_started':0,'MP_checks_completed':0,'qualified_slots':0};slots=[]
    try:
        arrays,rows,overall,stages=load_frozen(manifest)
        for i in range(10):
            directory=folder/f'machine_{i:02d}';directory.mkdir(exist_ok=False);write_once(directory/'main_started.json',{'machine':i,'input_snapshot_sha256':FROZEN['residual_snapshot'][1]});counts['eta_attempts']+=1
            try:
                vectors=event_vectors(arrays['Q8'],rows,i);eta=weighted_mean(vectors,297);primary={'status':'estimated','eta':eta,'main_estimates':1}
            except (NumericUnknown,OverflowError,ValueError) as exc:primary={'status':'unknown','eta':None,'main_estimates':1,'reason':repr(exc)}
            write_once(directory/'eta.json',{'machine':i,**primary,'rows_sha256':FROZEN['rows'][1],'Q80_array_sha256':manifest['snapshots']['Q8']['sha256'],'mean_order':'event math.fsum/m, then math.fsum/297'});counts['eta_files_sealed']+=1
        seal=seal_all_eta(folder)
        for i in range(10):
            verify_eta_seal(folder,seal);directory=folder/f'machine_{i:02d}';primary=read(directory/'eta.json')
            if primary['status']!='estimated':check={'status':'unknown','numeric_passed':False,'MP_checks':0,'reason':'no_finite_main_eta'}
            else:
                write_once(directory/'MP_started.json',{'all_eta_sealed_sha256':sha(folder/'all_eta_sealed.json'),'maximum_MP_checks':1});counts['MP_attempts']+=1
                before=counts['MP100_checks_started']
                try:check=mp_check(event_vectors(arrays['Q8'],rows,i),primary['eta'],297,overall[(i,8,'all')]['moment']['C_Q'],counts);counts['MP_checks_completed']+=1
                except (NumericUnknown,OverflowError,ValueError,ZeroDivisionError) as exc:check={'status':'unknown','numeric_passed':False,'MP_checks':counts['MP100_checks_started']-before,'reason':repr(exc)}
            write_once(directory/'MP.json',check);counts['qualified_slots']+=int(check['numeric_passed'])
            receipt={'machine':i,'eta_sha256':sha(directory/'eta.json'),'MP_sha256':sha(directory/'MP.json'),'numeric_passed':check['numeric_passed'],'main_estimates':1,'MP_checks':check['MP_checks']};write_once(directory/'complete.json',receipt)
            slots.append({**receipt,'receipt_sha256':sha(directory/'complete.json')})
        verify_eta_seal(folder,seal);verify_inputs()
        result={'status':'COMPLETED','qualification':'verified' if counts['qualified_slots']==10 else 'unknown','stage':'estimate','counts':counts,'slots':slots,
          'all_eta_sealed_sha256':sha(folder/'all_eta_sealed.json'),'release':release,'denominators':DENOM,'eta_is_new_parameter_estimate':True,'main_eta_replaced':False,
          'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'runtime':runtime(),
          'wrapper_native_wall_seconds':None,'tool_wall_seconds':None,'Q_automatically_released':False,'model_adopted':False}
        write_once(folder/'manifest.json',result);write_once(PUBLIC/'estimate_summary.json',result);write_once(folder/'complete.json',{'status':'COMPLETED','manifest_sha256':sha(folder/'manifest.json')});return result
    except BaseException as exc:
        write_once(folder/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'counts':counts,'slots':slots,'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'automatic_retry':False});raise

def corrected_condition(Cq,Cl,mu,nu,overall=False):
    if not all(finite(v) and v>=0 for v in (Cq,Cl,mu)):raise NumericUnknown('invalid corrected moment')
    if mu==0:
        if Cq!=0 or Cl!=0 or nu is not None:raise NumericUnknown('invalid empty corrected moment')
        return {'R_Q':None,'R_L':None,'nu':None,'compatible':None,'resolved':None,'classification':'unknown','reason':'no_window'}
    if not finite(nu) or nu<1e-10:raise NumericUnknown('invalid frozen nu')
    q=math.sqrt(Cq if overall else Cq/mu);l=math.sqrt(Cl if overall else Cl/mu);threshold=10*nu
    if not all(finite(v) for v in (q,l,threshold)):raise NumericUnknown('nonfinite corrected conditional statistic')
    good=max(q,l)<=threshold;bad=min(q,l)>threshold
    return {'R_Q':q,'R_L':l,'nu':nu,'compatible':good,'resolved':bad,'classification':'compatible' if good else 'resolved' if bad else 'intermediate','reason':None}
def corrected_C(arrays,rows,machine,steps,group,phase=None):
    qparts=[];lparts=[];available=0;count=0;details=[]
    for row in rows:
        if row['label']!=group:continue
        item=row['windows'][str(steps)];start=item['offset'];m=item['count'];indices=list(range(m)) if phase is None else [j for j,p in enumerate(item['classes']) if p==phase]
        q=[float(arrays['Q'+str(steps)][start+j,machine]) for j in indices];l=[float(arrays['L'+str(steps)][start+j,machine]) for j in indices]
        if any(not finite(v) for v in q+l):raise NumericUnknown('nonfinite corrected residual event')
        weight=1/(DENOM[group]*m);Cq=math.fsum(v*v*weight for v in q);Cl=math.fsum(v*v*weight for v in l)
        if not finite(Cq) or not finite(Cl):raise NumericUnknown('nonfinite direct corrected square contribution')
        qparts.append(Cq);lparts.append(Cl);available+=bool(indices);count+=len(indices);details.append({'id':row['id'],'m_e_all':m,'original_weight':weight,'windows':len(indices),'C_Q_corrected':Cq,'C_L_corrected':Cl})
    out={'C_Q':math.fsum(qparts),'C_L':math.fsum(lparts),'N_available':available,'windows':count}
    if not finite(out['C_Q']) or not finite(out['C_L']):raise NumericUnknown('nonfinite corrected square sum')
    return out,details
def table_row(key,new,parent,is_overall):
    moment=parent['moment'];old=parent['parent_statistics'];nu=old['nu'];condition=corrected_condition(new['C_Q'],new['C_L'],moment['mu'],nu,is_overall)
    if new['N_available']!=moment['N_available'] or new['windows']!=moment['windows']:raise NumericUnknown('frozen covered event/window count changed')
    return {**key,'N_original':moment['N_original'],'N_available':moment['N_available'],'windows':moment['windows'],'mu':moment['mu'],
      'C_Q_before':moment['C_Q'],'C_L_before':moment['C_L'],'C_diff_unchanged':moment['C_diff'],'C_Q_corrected':new['C_Q'],'C_L_corrected':new['C_L'],
      'R_Q_before':old['Q_Q'] if is_overall else old['R_Q'],'R_L_before':old['Q_L'] if is_overall else old['R_L'],
      'before_classification':old['classification'],'before_compatible':old['compatible'],'before_resolved':old['resolved'],**condition,
      'nu_source':'frozen S parent statistics, never recomputed after correction','units':{'C':'s^-4','R_nu':'s^-2','mu':'dimensionless'}}
def verify_E(folder,manifest):
    if len(manifest['slots'])!=10:raise ValueError('ten E slots required')
    for i,slot in enumerate(manifest['slots']):
        directory=Path(folder)/f'machine_{i:02d}'
        if slot['machine']!=i or sha(directory/'eta.json')!=slot['eta_sha256'] or sha(directory/'MP.json')!=slot['MP_sha256'] or sha(directory/'complete.json')!=slot['receipt_sha256']:raise ValueError('sealed E slot changed')
    if sha(Path(folder)/'all_eta_sealed.json')!=manifest['all_eta_sealed_sha256']:raise ValueError('all-eta seal changed')
def analyze():
    release,manifest=checked_release('analyze');folder=PRIVATE/'analyze';claim_stage(folder,'analyze',release);start=time.perf_counter();cpu=time.process_time()
    counts={'corrected_matrices_attempted':0,'corrected_matrices_completed':0,'difference_check_attempts':0,'difference_checks_completed':0,'overall_attempts':0,'overall_completed':0,'stage_attempts':0,'stage_completed':0};overall_table=[];stage_table=[];private_details=[];checks=[]
    try:
        E=PRIVATE/'estimate';complete=read(E/'complete.json');em=read(E/'manifest.json')
        if sha(E/'manifest.json')!=complete['manifest_sha256'] or em['qualification']!='verified' or em['release']['identities']!=release_expected('estimate') or any(s['numeric_passed'] is not True for s in em['slots']):raise PermissionError('all finite E checks must pass before Q')
        verify_E(E,em);etas=[read(E/f'machine_{i:02d}'/'eta.json')['eta'] for i in range(10)]
        if not all(finite(v) for v in etas):raise NumericUnknown('nonfinite sealed eta')
        arrays,rows,overall,stages=load_frozen(manifest);import numpy as np
        corrected={}
        for key,a in arrays.items():
            counts['corrected_matrices_attempted']+=1
            with np.errstate(over='ignore',invalid='ignore'):new=a-np.asarray(etas)[None,:]
            if not np.isfinite(new).all():raise NumericUnknown('nonfinite constant-corrected snapshot')
            corrected[key]=new;counts['corrected_matrices_completed']+=1
        for s in (8,4):
            for i in range(10):
                counts['difference_check_attempts']+=1
                with np.errstate(over='ignore',invalid='ignore'):
                    before=arrays[f'Q{s}'][:,i]-arrays[f'L{s}'][:,i];after=corrected[f'Q{s}'][:,i]-corrected[f'L{s}'][:,i]
                if not np.isfinite(before).all() or not np.isfinite(after).all() or not np.allclose(after,before,atol=1e-12,rtol=1e-12):raise NumericUnknown('finite pointwise representation difference check failed; original nu retained')
                checks.append({'machine':i,'steps':s,'status':'verified','max_abs_difference':float(np.max(np.abs(after-before)))});counts['difference_checks_completed']+=1
        for i in range(10):
            for s in (8,4):
                group_values={}
                for g in ('stable','unstable'):
                    counts['overall_attempts']+=1;new,detail=corrected_C(corrected,rows,i,s,g);group_values[g]=new
                    overall_table.append(table_row({'machine':i,'steps':s,'group':g},new,overall[(i,s,g)],True));private_details.append({'machine':i,'steps':s,'group':g,'phase':'overall','events':detail});counts['overall_completed']+=1
                counts['overall_attempts']+=1;new={k:267/297*group_values['stable'][k]+30/297*group_values['unstable'][k] for k in ('C_Q','C_L')}
                new.update(N_available=297,windows=group_values['stable']['windows']+group_values['unstable']['windows']);overall_table.append(table_row({'machine':i,'steps':s,'group':'all'},new,overall[(i,s,'all')],True));counts['overall_completed']+=1
                for g in ('stable','unstable'):
                    for p in PHASES:
                        counts['stage_attempts']+=1;new,detail=corrected_C(corrected,rows,i,s,g,p)
                        stage_table.append(table_row({'machine':i,'steps':s,'group':g,'phase':p},new,stages[(i,s,g,p)],False));private_details.append({'machine':i,'steps':s,'group':g,'phase':p,'events':detail});counts['stage_completed']+=1
        pairs=[]
        for i in range(10):
            for g in ('stable','unstable'):
                future=[r for r in stage_table if r['machine']==i and r['group']==g and r['phase']=='future'];classes={str(r['steps']):r['classification'] for r in future}
                label='unknown' if any(c=='unknown' for c in classes.values()) else 'both_future_resolved' if all(c=='resolved' for c in classes.values()) else 'both_future_compatible' if all(c=='compatible' for c in classes.values()) else 'mixed_future_scales'
                pairs.append({'machine':i,'group':g,'before_future40_80':{str(r['steps']):r['before_classification'] for r in future},'future40_80':classes,'classification':label,'any_covered_future_resolved':any(r['resolved'] is True for r in future)})
        with (folder/'corrected_residuals.npz').open('xb') as f:np.savez_compressed(f,**corrected)
        write_once(folder/'event_square_contributions.json',private_details);write_once(folder/'difference_checks.json',checks)
        verify_E(E,em);verify_inputs()
        for key,name in (('estimate_manifest','manifest.json'),('estimate_complete','complete.json'),('all_eta_sealed','all_eta_sealed.json')):
            if sha(E/name)!=release['identities'][key+'_sha256']:raise ValueError('E aggregate identities changed during Q')
        result={'status':'COMPLETED','stage':'analyze','counts':counts,'overall60':overall_table,'stage120':stage_table,'future_pairs':pairs,'denominators':DENOM,
          'release':release,'difference_checks_sha256':sha(folder/'difference_checks.json'),'corrected_snapshot_sha256':sha(folder/'corrected_residuals.npz'),
          'private_event_contributions_sha256':sha(folder/'event_square_contributions.json'),'E_unchanged':True,'S_inputs_unchanged':True,'new_eta_estimates':0,'new_MP_checks':0,
          'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'runtime':runtime(),
          'wrapper_native_wall_seconds':None,'tool_wall_seconds':None,'model_adopted':False,
          'scope':'fixed theta shared constant direction only; main loss decrease is algebraic, no general affine-family/real-drive/initialization/autonomous claim'}
        write_once(folder/'manifest.json',result);write_once(PUBLIC/'analysis_summary.json',result);write_once(folder/'complete.json',{'status':'COMPLETED','manifest_sha256':sha(folder/'manifest.json')});return result
    except BaseException as exc:
        write_once(folder/'partial_tables.json',{'overall':overall_table,'stages':stage_table,'event_contributions':private_details,'difference_checks':checks})
        write_once(folder/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'counts':counts,'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'automatic_retry':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['estimate','analyze']);args=p.parse_args();estimate() if args.stage=='estimate' else analyze()
