"""One frozen residual-only aggregate. No parameters, targets, or parent execution.

Q containers are hashed whole, but decoded only at the frozen root-machine and
fixed_residuals.all anchors. No generic fallback parser is provided.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1];RUN='p_residual_stage_20261031'
PRIVATE=ROOT/'private_data'/RUN;PUBLIC=ROOT/'results'/RUN
DENOM={'all':297,'stable':267,'unstable':30};PHASES=('history','cross_t0','future')
FROZEN={
 'scientist':('reports/'+RUN+'_scientist.md','e7ad256d71b4f003c49abc3b3a5cf7821e432b22d7e3d43b9f520fb56608abb7'),
 'protocol':('reports/'+RUN+'_protocol.md','a4d8f01853fe51083d96f254dbbd26529d0be4eaaa9147f0b36508d0fd186fd5'),
 'source_bindings':('results/'+RUN+'/source_bindings.json','6d473d299a08652bed36ae623c42b4cefc44e2535c6a3a5bc35c5844dc8027ba'),
 'authorization':('results/'+RUN+'/authorization.json','22e08f0e0dfc7f6d93d31af4065daf3343dfcdfbfd494aa3fe739e935c35ee6d')}

class QualificationUnknown(Exception):pass
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def release_expected():
    return {'source_sha256':sha(Path(__file__)),'tests_sha256':sha(ROOT/'tests'/('test_'+RUN+'.py')),
      **{k+'_sha256':h for k,(path,h) in FROZEN.items()},'aggregate_jobs_upper_bound':1,
      'bound_assets_upper_bound':24,'stage_moment_cells_upper_bound':120,'parent_closure_cells_upper_bound':60,
      **{k+'_upper_bound':0 for k in ('parameter_estimates','QR_checks','new_residual_evaluations','integrals','sensitivity_bounds','filters','initializers','optimizer','ROM','FOM','model_writes')}}
def verify_release(release,expected):
    if release.get('released') is not True or release.get('allowed_stage')!='aggregate':raise PermissionError('root aggregate release required')
    for k,v in expected.items():
        if release.get(k)!=v or (type(v) is int and type(release.get(k)) is not int):raise PermissionError('release identity/limit differs: '+k)
def checked_release():
    expected=release_expected()
    for key,(path,h) in FROZEN.items():
        if sha(ROOT/path)!=h:raise ValueError('frozen '+key+' changed')
    path=PUBLIC/'aggregate_execution_release.json';r=read(path);verify_release(r,expected)
    review=(ROOT/r['review_snapshot_path']).resolve();review.relative_to(ROOT.resolve())
    if review.suffix.lower() not in ('.md','.json') or sha(review)!=r['review_snapshot_sha256']:raise ValueError('review snapshot differs')
    return {'release_sha256':sha(path),'review_snapshot_sha256':r['review_snapshot_sha256'],'identities':expected}
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
def claim_private(path,release):
    Path(path).mkdir(parents=True,exist_ok=False)
    write_once(Path(path)/'started.json',{'status':'STARTED','native_interpreter':native_identity(),
      'python_executable':sys.executable,'argv':list(sys.argv),'release':release})

def verify_assets(bindings,root,counts):
    if len(bindings['files'])!=24:raise QualificationUnknown('exact24 source asset set required')
    paths={};root=Path(root).resolve()
    for key,e in bindings['files'].items():
        path=(root/e['path']).resolve();path.relative_to(root)
        if path.suffix.lower() not in ('.json','.md','.py'):raise QualificationUnknown('forbidden source asset type')
        if path.stat().st_size!=e['bytes'] or sha(path)!=e['sha256']:raise QualificationUnknown('source asset identity mismatch')
        paths[key]=path;counts['source_files_verified']+=1;counts['source_bytes_verified']+=e['bytes']
    if len(set(paths.values()))!=24:raise QualificationUnknown('duplicate source asset')
    return paths
def unique_object(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise QualificationUnknown('duplicate selected JSON key')
        out[key]=value
    return out
def reject_constant(value):raise QualificationUnknown('nonfinite JSON token in selected subtree')

def selected_residuals(data,decoder=None):
    """Only decode root machine and fixed_residuals.all, never the full JSON."""
    if not isinstance(data,bytes):raise QualificationUnknown('frozen JSON envelope differs')
    if data.startswith(b'{\r\n'):eol=b'\r\n'
    elif data.startswith(b'{\n'):eol=b'\n'
    else:raise QualificationUnknown('frozen JSON envelope differs')
    if not data.endswith(b'}'+eol):raise QualificationUnknown('frozen JSON envelope differs')
    # Accept either producer text-mode convention without normalizing any source
    # bytes. Mixed or broken line endings cannot establish the frozen boundaries.
    if (eol==b'\n' and b'\r' in data) or (eol==b'\r\n' and (re.search(rb'(?<!\r)\n',data) or re.search(rb'\r(?!\n)',data))):
        raise QualificationUnknown('inconsistent producer line endings')
    root_keys=list(re.finditer(rb'(?m)^  "([^"\\]+)": ',data))
    names=[m.group(1) for m in root_keys]
    if names!=[b'machine',b'sensitivity',b'fixed_residuals',b'statistics',b'parameters_sha256']:
        raise QualificationUnknown('nonunique or changed root anchors')
    decoder=decoder or json.JSONDecoder(object_pairs_hook=unique_object,parse_constant=reject_constant)
    machine_fragment=data[root_keys[0].end():root_keys[1].start()].rstrip()
    if not re.fullmatch(rb'[0-9]+,',machine_fragment):raise QualificationUnknown('root machine scalar boundary differs')
    token=machine_fragment[:-1].decode('ascii');machine,end=decoder.raw_decode(token)
    if end!=len(token) or type(machine) is not int or not 0<=machine<10:raise QualificationUnknown('invalid root machine scalar')
    fixed=data[root_keys[2].end():root_keys[3].start()]
    if not fixed.startswith(b'{'+eol) or not fixed.rstrip().endswith(eol+b'  },'):raise QualificationUnknown('fixed residual object boundary differs')
    group_keys=list(re.finditer(rb'(?m)^    "([^"\\]+)": ',fixed))
    if [m.group(1) for m in group_keys]!=[b'all',b'stable',b'unstable']:raise QualificationUnknown('nonunique or changed group anchors')
    allowed=fixed[group_keys[0].end():group_keys[1].start()].rstrip()
    if not allowed.startswith(b'{'+eol) or not allowed.endswith(eol+b'    },'):raise QualificationUnknown('selected all object boundary differs')
    text=allowed[:-1].decode('utf-8');value,end=decoder.raw_decode(text)
    if text[end:].strip() or not isinstance(value,dict) or set(value)!={'8','4'}:raise QualificationUnknown('selected residual scope differs')
    for records in value.values():
        if not isinstance(records,list):raise QualificationUnknown('selected residual records are not lists')
        for record in records:
            if not isinstance(record,dict) or set(record)!={'id','Q_residual','L_residual'}:raise QualificationUnknown('unexpected selected record field')
    return machine,value,len(token.encode('ascii'))+len(allowed)-1

def controls(paths,bindings,counts):
    def control(key):
        value=read(paths[key]);counts['control_files_decoded']+=1;return value
    P=control('parent_P_manifest');Q=control('parent_Q_manifest');summary=control('parent_P_summary');E=control('parent_E_manifest')
    table=control('parent_evidence_digest')
    if P.get('status')!='verified' or Q.get('status')!='COMPLETED' or Q.get('denominators')!=DENOM or summary.get('denominators')!=DENOM:raise QualificationUnknown('parent qualification control mismatch')
    if P['rows_sha256']!=bindings['files']['rows']['sha256'] or summary['manifest_sha256']!=bindings['files']['parent_P_manifest']['sha256']:raise QualificationUnknown('parent row binding mismatch')
    if len(Q['machines'])!=10 or len(E['slots'])!=10 or table.get('core_artifacts_count')!=128:raise QualificationUnknown('parent fixed machine/core set differs')
    indexed={item['path']:item for item in table['artifacts']}
    if len(indexed)!=len(table['artifacts']):raise QualificationUnknown('duplicate parent core path')
    for machine,row in enumerate(Q['machines']):
        key=f'residual_machine_{machine:02d}';entry=bindings['files'][key]
        if row['machine']!=machine or row['private_evidence_sha256']!=entry['sha256'] or E['slots'][machine]['machine']!=machine or E['slots'][machine]['numeric_passed'] is not True or E['slots'][machine]['unique'] is not True:
            raise QualificationUnknown('parent machine eligibility or residual binding mismatch')
        if entry['path'] not in indexed or any(indexed[entry['path']][k]!=entry[k] for k in ('bytes','sha256')):raise QualificationUnknown('parent128 residual identity mismatch')
    entry=bindings['files']['rows']
    if entry['path'] not in indexed or any(indexed[entry['path']][k]!=entry[k] for k in ('bytes','sha256')):raise QualificationUnknown('parent128 row identity mismatch')
    return Q,summary

def number(value):return type(value) in (int,float) and math.isfinite(value)
def close(a,b):return number(a) and number(b) and abs(a-b)<=1e-12+1e-12*abs(b)
def validate_rows(rows,bindings,Psummary):
    keys={'id','label','n_future','last_original_index','last_scored_index','unscored_tail_steps','relative_analysis_range_s','relative_last_scored_time_s','windows'}
    wkeys={'offset','count','phase','classes','class_counts','row_weight_all','row_weight_group','class_weight_all','class_weight_group'}
    if not isinstance(rows,list) or len(rows)!=297 or any(set(r)!=keys for r in rows):raise QualificationUnknown('row schema/count mismatch')
    ids=[r['id'] for r in rows]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=297 or ids!=sorted(ids):raise QualificationUnknown('row ids/original order mismatch')
    if any(sum(r['label']==g for r in rows)!=DENOM[g] for g in ('stable','unstable')):raise QualificationUnknown('original group denominator mismatch')
    offsets={8:0,4:0}
    for row in rows:
        n=row['n_future'];last=row['last_original_index'];scored=row['last_scored_index']
        if any(type(v) is not int for v in (n,last,scored,row['unscored_tail_steps'])) or n<2 or n>1500 or (row['label']=='stable' and n!=800) or last!=10+n or scored!=last//8*8 or scored<8 or row['unscored_tail_steps']!=last-scored:
            raise QualificationUnknown('row registered extent mismatch')
        if row['relative_analysis_range_s']!=[-.1,n*.01] or not close(row['relative_last_scored_time_s'],(scored-10)*.01) or set(row['windows'])!={'8','4'}:raise QualificationUnknown('row relative time/index mismatch')
        for steps in (8,4):
            w=row['windows'][str(steps)];count=scored//steps
            classes=['history' if j+steps<=10 else 'future' if j>=10 else 'cross_t0' for j in range(0,scored,steps)]
            counts={c:classes.count(c) for c in PHASES}
            if set(w)!=wkeys or type(w['count']) is not int or type(w['offset']) is not int or w['count']!=count or w['offset']!=offsets[steps] or type(w['phase']) is not int or w['phase']!=0 or w['classes']!=classes or w['class_counts']!=counts or any(type(v) is not int for v in w['class_counts'].values()):
                raise QualificationUnknown('row offset/phase/window class mismatch')
            for group,key in (('all','all'),(row['label'],'group')):
                if not close(w['row_weight_'+key],1/(DENOM[group]*count)) or set(w['class_weight_'+key])!=set(PHASES) or any(not close(w['class_weight_'+key][c],counts[c]/(DENOM[group]*count)) for c in PHASES):raise QualificationUnknown('row original weight mismatch')
            offsets[steps]+=count
    for steps in (8,4):
        if offsets[steps]!=bindings['expected_rows_per_representation_per_machine'][str(steps)]:raise QualificationUnknown('total original window count mismatch')
        for group in ('all','stable','unstable'):
            selected=[r for r in rows if group=='all' or r['label']==group];reference=Psummary['coverage'][group]['windows'][str(steps)]
            for phase in PHASES:
                count=sum(r['windows'][str(steps)]['class_counts'][phase] for r in selected)
                available=sum(r['windows'][str(steps)]['class_counts'][phase]>0 for r in selected)
                if count!=reference['class_counts'][phase] or available!=reference['events_with_class'][phase]:raise QualificationUnknown('parent phase coverage mismatch')
                if phase=='future' and available!=bindings['expected_pure_future_events'][str(steps)][group]:raise QualificationUnknown('pure future coverage mismatch')
    return ids
def align_residuals(machine,value,expected_machine,rows):
    if machine!=expected_machine:raise QualificationUnknown('machine alignment mismatch')
    for steps in ('8','4'):
        records=value[steps]
        if [r['id'] for r in records]!=[r['id'] for r in rows] or len({r['id'] for r in records})!=297:raise QualificationUnknown('residual event keys/order mismatch')
        for record,row in zip(records,rows):
            count=row['windows'][steps]['count']
            for field in ('Q_residual','L_residual'):
                values=record[field]
                if not isinstance(values,list) or len(values)!=count or any(not number(v) for v in values):raise QualificationUnknown('residual dimensions/finite values mismatch')

def moments(records,rows,group,steps,phase=None):
    """Direct path uses event means; stage path retains original per-window w."""
    N=DENOM[group];mass=[];qparts=[];lparts=[];dparts=[];available=0;windows=0;details=[]
    for record,row in zip(records,rows):
        if row['label']!=group:continue
        q=record['Q_residual'];l=record['L_residual'];m=row['windows'][str(steps)]['count'];weight=1/(N*m)
        indices=list(range(m)) if phase is None else [i for i,c in enumerate(row['windows'][str(steps)]['classes']) if c==phase]
        if indices:available+=1
        windows+=len(indices)
        if phase is None:
            mu=1/N;Cq=math.fsum(v*v for v in q)/m/N;Cl=math.fsum(v*v for v in l)/m/N;Cd=math.fsum((a-b)**2 for a,b in zip(q,l))/m/N
        else:
            mu=len(indices)*weight;Cq=math.fsum(weight*q[i]*q[i] for i in indices);Cl=math.fsum(weight*l[i]*l[i] for i in indices);Cd=math.fsum(weight*(q[i]-l[i])**2 for i in indices)
        if not all(number(v) and v>=0 for v in (mu,Cq,Cl,Cd)):raise QualificationUnknown('nonfinite derived moment')
        mass.append(mu);qparts.append(Cq);lparts.append(Cl);dparts.append(Cd)
        details.append({'id':row['id'],'m_e_all':m,'original_window_weight':weight,'selected_windows':len(indices),'mu':mu,'C_Q':Cq,'C_L':Cl,'C_diff':Cd})
    out={'mu':math.fsum(mass),'C_Q':math.fsum(qparts),'C_L':math.fsum(lparts),'C_diff':math.fsum(dparts),
      'N_original':N,'N_available':available,'windows':windows}
    if not all(number(out[k]) for k in ('mu','C_Q','C_L','C_diff')):raise QualificationUnknown('nonfinite total moment')
    return out,details
def mix_groups(stable,unstable):
    return {**{k:267/297*stable[k]+30/297*unstable[k] for k in ('mu','C_Q','C_L','C_diff')},
      'N_original':297,'N_available':stable['N_available']+unstable['N_available'],'windows':stable['windows']+unstable['windows']}
def conditional(moment):
    if moment['mu']==0:
        if moment['windows']!=0 or moment['N_available']!=0 or any(moment[k]!=0 for k in ('C_Q','C_L','C_diff')):raise QualificationUnknown('invalid empty-set moment')
        return {'status':'unknown','reason':'no_window','R_Q':None,'R_L':None,'nu':None,'compatible':None,'resolved':None,'classification':'unknown'}
    q=math.sqrt(moment['C_Q']/moment['mu']);l=math.sqrt(moment['C_L']/moment['mu']);difference=math.sqrt(moment['C_diff']/moment['mu']);nu=max(1e-10,difference);threshold=10*nu
    if not all(number(v) for v in (q,l,difference,nu,threshold)):raise QualificationUnknown('nonfinite conditional statistic')
    compatible=max(q,l)<=threshold;resolved=min(q,l)>threshold
    return {'status':'computed','reason':None,'R_Q':q,'R_L':l,'nu':nu,'representation_difference_RMS':difference,'compatible':compatible,'resolved':resolved,
      'classification':'compatible' if compatible else 'resolved' if resolved else 'intermediate'}
def parent_closure(moment,parent):
    # Parent whole RMS is sqrt(C); do not condition on a rounded sum(mu).
    q=math.sqrt(moment['C_Q']);l=math.sqrt(moment['C_L']);nu=max(1e-10,math.sqrt(moment['C_diff']))
    if not all(number(v) for v in (q,l,nu)):raise QualificationUnknown('nonfinite overall reconstruction')
    compatible=max(q,l)<=10*nu;resolved=min(q,l)>10*nu
    if parent.get('status')!='computed' or parent.get('denominator')!=moment['N_original'] or parent.get('available_events')!=moment['N_original'] or not all(close(a,parent.get(k)) for a,k in ((q,'Q_Q'),(l,'Q_L'),(nu,'nu'))) or parent.get('compatible') is not compatible or parent.get('resolved') is not resolved:
        raise QualificationUnknown('parent whole statistic or exact boolean closure failed')
    return {'status':'verified','Q_Q':q,'Q_L':l,'nu':nu,'compatible':compatible,'resolved':resolved,'parent_statistics_sha256':digest(parent)}
def partition_closure(parts,direct):
    for k in ('mu','C_Q','C_L','C_diff'):
        total=math.fsum(p[k] for p in parts)
        if not close(total,1. if k=='mu' else direct[k]):raise QualificationUnknown('stage partition identity failed')
    return {'status':'verified','stage_sum_sha256':digest({k:math.fsum(p[k] for p in parts) for k in ('mu','C_Q','C_L','C_diff')})}
def pair_classification(rows):
    future=[r for r in rows if r['phase']=='future'];early=[r for r in rows if r['phase']!='future']
    classes={str(r['steps']):r['classification'] for r in future}
    if any(c=='unknown' for c in classes.values()):status='unknown'
    elif all(c=='resolved' for c in classes.values()):status='both_future_resolved'
    elif all(c=='compatible' for c in classes.values()):status='both_future_compatible_early_resolved' if any(r['resolved'] is True for r in early) else 'both_future_compatible'
    else:status='mixed_future_scales'
    return {'future_40_80':classes,'classification':status,'any_future_resolved':any(r['resolved'] is True for r in future),
      'scope':'aggregate conditional RMS on covered fixed windows; not pointwise/eventwise persistence or initialization causality'}

def aggregate():
    release=checked_release();claim_private(PRIVATE,release);start=time.perf_counter();cpu=time.process_time()
    counts={k:0 for k in ('source_files_verified','source_bytes_verified','source_files_reverified','control_files_decoded','label_files_decoded','residual_decode_attempts','residual_containers_decoded','residual_container_bytes_read','selected_bytes_decoded','machines_aligned','direct_moment_attempts','direct_moments_computed','parent_closures_attempted','parent_closures_verified','stage_moment_attempts','stage_moments_computed','partition_closures_verified','partition_parent_closures_verified','stage_classification_attempts','stage_classifications_computed','stage_classifications_finalized')}
    direct={};closures=[];staged={};contributions=[];phase='source';paths={};bindings=None
    try:
        bindings=read(ROOT/FROZEN['source_bindings'][0]);paths=verify_assets(bindings,ROOT,counts);parent,Psummary=controls(paths,bindings,counts)
        rows=read(paths['rows']);counts['label_files_decoded']+=1;validate_rows(rows,bindings,Psummary);payloads=[]
        for machine in range(10):
            counts['residual_decode_attempts']+=1
            raw=paths[f'residual_machine_{machine:02d}'].read_bytes();counts['residual_container_bytes_read']+=len(raw)
            actual,selected,size=selected_residuals(raw);del raw
            counts['residual_containers_decoded']+=1;counts['selected_bytes_decoded']+=size
            align_residuals(actual,selected,machine,rows);payloads.append(selected);counts['machines_aligned']+=1
        write_once(PRIVATE/'alignment.json',{'status':'verified','row_ids_in_frozen_order':[r['id'] for r in rows],
          'input_hashes':{k:e['sha256'] for k,e in bindings['files'].items()},'window_order_evidence':'bound parent producer uses contiguous rows offset/count slices; payload carries no independent time index'})
        phase='parent_closure'
        for machine,payload in enumerate(payloads):
            for steps in (8,4):
                for group in ('stable','unstable'):
                    counts['direct_moment_attempts']+=1
                    value,_=moments(payload[str(steps)],rows,group,steps);direct[(machine,steps,group)]=value;counts['direct_moments_computed']+=1
                counts['direct_moment_attempts']+=1;direct[(machine,steps,'all')]=mix_groups(direct[(machine,steps,'stable')],direct[(machine,steps,'unstable')]);counts['direct_moments_computed']+=1
                for group in ('stable','unstable','all'):
                    counts['parent_closures_attempted']+=1
                    check=parent_closure(direct[(machine,steps,group)],parent['machines'][machine]['fixed_residuals'][group][str(steps)])
                    closures.append({'machine':machine,'steps':steps,'group':group,**check});counts['parent_closures_verified']+=1
        write_once(PRIVATE/'parent_closures.json',closures)
        phase='stage_moments'
        for machine,payload in enumerate(payloads):
            for group in ('stable','unstable'):
                for steps in (8,4):
                    for stage in PHASES:
                        counts['stage_moment_attempts']+=1
                        value,event_values=moments(payload[str(steps)],rows,group,steps,stage)
                        staged[(machine,steps,group,stage)]=value;counts['stage_moments_computed']+=1
                        contributions.append({'machine':machine,'steps':steps,'group':group,'phase':stage,'events':event_values})
        phase='partition_closure';partitions=[]
        for machine in range(10):
            for steps in (8,4):
                sums={}
                for group in ('stable','unstable'):
                    parts=[staged[(machine,steps,group,s)] for s in PHASES]
                    check=partition_closure(parts,direct[(machine,steps,group)]);counts['partition_closures_verified']+=1
                    sums[group]={k:math.fsum(p[k] for p in parts) for k in ('mu','C_Q','C_L','C_diff')}
                    parent_closure({**direct[(machine,steps,group)],**sums[group]},parent['machines'][machine]['fixed_residuals'][group][str(steps)])
                    counts['partition_parent_closures_verified']+=1
                    partitions.append({'machine':machine,'steps':steps,'group':group,**check})
                mixed={k:267/297*sums['stable'][k]+30/297*sums['unstable'][k] for k in ('mu','C_Q','C_L','C_diff')}
                if not close(mixed['mu'],1.) or any(not close(mixed[k],direct[(machine,steps,'all')][k]) for k in ('C_Q','C_L','C_diff')):raise QualificationUnknown('all297 stage-mixture closure failed')
                parent_closure({**direct[(machine,steps,'all')],**mixed},parent['machines'][machine]['fixed_residuals']['all'][str(steps)])
                counts['partition_parent_closures_verified']+=1
                counts['partition_closures_verified']+=1;partitions.append({'machine':machine,'steps':steps,'group':'all','status':'verified'})
        # No conditional stage output is published until all preceding gates pass.
        phase='classify';output=[]
        for (machine,steps,group,stage),moment in staged.items():
            counts['stage_classification_attempts']+=1
            stats=conditional(moment);overall=direct[(machine,steps,group)]
            output.append({'machine':machine,'steps':steps,'group':group,'phase':stage,**moment,**stats,
              'square_contribution_fraction_Q':moment['C_Q']/overall['C_Q'] if overall['C_Q']>0 else None,
              'square_contribution_fraction_L':moment['C_L']/overall['C_L'] if overall['C_L']>0 else None,
              'square_contribution_fraction_diff':moment['C_diff']/overall['C_diff'] if overall['C_diff']>0 else None,
              'units':{'C_Q_C_L_C_diff':'s^-4','R_Q_R_L_nu':'s^-2','mu_contribution_fractions':'dimensionless'}})
            counts['stage_classifications_computed']+=1
        pairs=[{'machine':machine,'group':group,**pair_classification([r for r in output if r['machine']==machine and r['group']==group])} for machine in range(10) for group in ('stable','unstable')]
        write_once(PRIVATE/'event_moment_contributions.json',contributions);write_once(PRIVATE/'partition_closures.json',partitions)
        phase='final_source_recheck'
        for key,e in bindings['files'].items():
            if paths[key].stat().st_size!=e['bytes'] or sha(paths[key])!=e['sha256']:raise QualificationUnknown('parent asset changed during aggregation')
            counts['source_files_reverified']+=1
        counts['stage_classifications_finalized']=len(output)
        result={'status':'COMPLETED','qualification':'verified','denominators':DENOM,'counts':counts,'stage_rows':output,'future_pairs':pairs,
          'parent_closures':[{'machine':r['machine'],'steps':r['steps'],'group':r['group'],'status':r['status'],'closure_sha256':digest(r)} for r in closures],
          'parent_closures_sha256':sha(PRIVATE/'parent_closures.json'),'partition_closures_sha256':sha(PRIVATE/'partition_closures.json'),
          'event_contributions_sha256':sha(PRIVATE/'event_moment_contributions.json'),'alignment_sha256':sha(PRIVATE/'alignment.json'),
          'all24_parent_assets_unchanged':True,'release':release,'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,
          'new_parameter_estimates':0,'new_QR':0,'new_residual_evaluations':0,'new_filters':0,'optimizer':0,'ROM':0,'FOM':0,'model_writes':0,
          'confirmation':'consumed train exploratory localization; no independent confirmation','model_adopted':False}
        result['scope']='original complete blocks only; unscored tails and eventwise future continuation remain unknown'
        result['runtime']={'python':sys.version,'device':'CPU','float_precision':'binary64','threads':1}
        result['wrapper_native_wall_seconds']=None;result['tool_wall_seconds']=None;result['project_CPU_seconds']=None
        write_once(PRIVATE/'manifest.json',result);write_once(PUBLIC/'aggregate_summary.json',result)
        write_once(PRIVATE/'complete.json',{'status':'COMPLETED','manifest_sha256':sha(PRIVATE/'manifest.json')});return result
    except BaseException as exc:
        # Preserve completed work/counters; never report an executed stage as zero.
        write_once(PRIVATE/'partial_moments.json',{'direct':[{'key':list(k),'moment':v} for k,v in direct.items()],
          'stage_moments':[{'key':list(k),'moment':v} for k,v in staged.items()],'parent_closures':closures,'event_contributions':contributions})
        unchanged=None
        if phase=='final_source_recheck':
            unchanged=False if isinstance(exc,QualificationUnknown) else None
        elif len(paths)==24:
            try:
                matches=[]
                for k,e in bindings['files'].items():
                    matches.append(paths[k].stat().st_size==e['bytes'] and sha(paths[k])==e['sha256'])
                    counts['source_files_reverified']+=1
                unchanged=all(matches)
            except (OSError,ValueError):unchanged=None
        detail={'status':'unknown','failure_stage':phase,'counts':counts,'denominators':DENOM,
          'reason':str(exc) if isinstance(exc,QualificationUnknown) else 'technical error; details retained privately',
          'parent_assets_unchanged_at_failure':unchanged,
          'stage_rows_published':False,'release':release,'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'automatic_retry':False}
        write_once(PRIVATE/'failure.json',{**detail,'error':repr(exc),'traceback':traceback.format_exc()});write_once(PUBLIC/'aggregate_unknown.json',detail)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['aggregate']);p.parse_args();aggregate()
