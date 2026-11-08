"""M/S only: conditional proof references, bounded static excerpts, old-stat closures.

No eta, signed mean, MP, fitting, target, scale, model, or E/Q implementation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1];RUN='p_constant_direction_20261031'
PRIVATE=ROOT/'private_data'/RUN;PUBLIC=ROOT/'results'/RUN
DENOM={'all':297,'stable':267,'unstable':30};PHASES=('history','cross_t0','future')
FROZEN={
 'scientist':('reports/'+RUN+'_scientist.md','a25715fe056629ab63204905125ae2a1827c03386c29b001b403ddb54fca4be1'),
 'protocol':('reports/'+RUN+'_protocol.md','c7c69017ae1608f06469e9fa3cda6a4bb7427acdf0abc095aeed180be8c39892'),
 'source_bindings':('results/'+RUN+'/source_bindings.json','2fe73e3b9c16200fbf3637182a8e53d737ab6225a8df62b83dc10026fefe578e'),
 'authorization':('results/'+RUN+'/authorization.json','f8881aa127a7b1cc6c172274a44e890ed592926950ecd236aa2287eb57c7f12c'),
 'design_review':('results/'+RUN+'/design_review_at_adoption.md','6baa675ab1078fe3e62f1cd824ec1a77bcc2c60bd2734aedceab55a6d09901ef')}
HELPERS=('selected_residuals','validate_rows','align_residuals','moments','conditional','parent_closure','mix_groups','close','digest')
class SourceUnknown(Exception):pass
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def release_expected():
    return {'source_sha256':sha(Path(__file__)),'tests_sha256':sha(ROOT/'tests'/('test_'+RUN+'_source.py')),
      **{k+'_sha256':h for k,(path,h) in FROZEN.items()},'normal_MS_jobs_upper_bound':1,'bound_assets_upper_bound':28,
      'static_content_files_upper_bound':8,'parent_overall_closure_cells_upper_bound':60,'parent_stage_closure_cells_upper_bound':120,
      **{k+'_upper_bound':0 for k in ('eta_estimates','signed_mean_evaluations','MP_checks','fits','QR','new_residual_evaluations','integrals','sensitivity_bounds','filters','initializers','optimizer','ROM','FOM','model_writes')}}
def verify_release(release,expected):
    if release.get('released') is not True or release.get('allowed_stage')!='source':raise PermissionError('root M/S release required')
    for k,v in expected.items():
        if release.get(k)!=v or (type(v) is int and type(release.get(k)) is not int):raise PermissionError('source release identity/limit differs: '+k)
def checked_release():
    expected=release_expected()
    for key,(path,h) in FROZEN.items():
        if sha(ROOT/path)!=h:raise ValueError('frozen '+key+' differs')
    path=PUBLIC/'source_execution_release.json';r=read(path);verify_release(r,expected)
    review=(ROOT/r['review_snapshot_path']).resolve();review.relative_to(ROOT.resolve())
    if review.suffix.lower() not in ('.json','.md') or sha(review)!=r['review_snapshot_sha256']:raise ValueError('frozen implementation review differs')
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
            if k.GetProcessTimes(k.GetCurrentProcess(),*[ctypes.byref(v) for v in values]):result['creation_FILETIME_100ns']=(values[0].dwHighDateTime<<32)|values[0].dwLowDateTime
        except (OSError,AttributeError,TypeError):pass
    return result
def claim_private(path,release):
    Path(path).mkdir(parents=True,exist_ok=False)
    write_once(Path(path)/'started.json',{'status':'STARTED','native_interpreter':native_identity(),'python_executable':sys.executable,'argv':list(sys.argv),'release':release})
def verify_assets(bindings,root,counts):
    root=Path(root).resolve();paths={}
    if len(bindings['files'])!=28 or sum(e['kind']=='static_content' for e in bindings['files'].values())!=8:raise SourceUnknown('exact28/8 source set required')
    for key,e in bindings['files'].items():
        p=(root/e['path']).resolve();p.relative_to(root)
        if p.suffix.lower() not in ('.json','.py','.md') or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise SourceUnknown('bound source identity mismatch')
        paths[key]=p;counts['assets_verified']+=1;counts['source_bytes_hashed']+=e['bytes']
    if len(set(paths.values()))!=28:raise SourceUnknown('duplicate source path')
    return paths
def excerpt(text,entry):
    lines=text.splitlines();needles=entry['needles'];matched=[i for i,line in enumerate(lines) if any(n in line for n in needles)]
    selected=matched[:entry['max_matching_lines']];context=entry['context_lines_each_side'];indices=set()
    for i in selected:indices.update(range(max(0,i-context),min(len(lines),i+context+1)))
    kept=sorted(indices)[:entry['max_excerpt_lines']]
    hits={n:sum(n in line for line in lines) for n in needles}
    return {'status':'matched_static_declaration' if all(hits.values()) else 'unknown','needle_matches':hits,
      'matching_lines_total':len(matched),'matching_lines_selected':len(selected),'excerpt_truncated':len(indices)>len(kept) or len(matched)>len(selected),
      'lines':[{'line':i+1,'text':lines[i]} for i in kept]}
def load_helpers(path,expected_sha):
    if sha(path)!=expected_sha:raise SourceUnknown('bound pure-helper source changed')
    spec=importlib.util.spec_from_file_location('_constant_direction_frozen_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if any(not callable(getattr(module,key,None)) for key in HELPERS):raise SourceUnknown('helper whitelist incomplete')
    return SimpleNamespace(**{key:getattr(module,key) for key in HELPERS})
def math_reference():
    return {'status':'adopted_independently_reviewed_conditional_proof','scientist_sha256':FROZEN['scientist'][1],
      'design_review_sha256':FROZEN['design_review'][1],
      'eliminated_drive':"u_p' + c*u_p - d*u_q",'constant_drive':'h_phys=c*u_p-d*u_q',
      'common_equilibrium':'h_phys=c*g*omega_star+(a*c-d*d*S)*p_star',
      'consistent_zero_reference_implies_h0':True,'h0_implies_individual_drives_zero':False,
      'h0_implies_fixed_theta_residual_mean_zero':False,'normalized_sign':'h_norm=-eta',
      'physical_sign_formal_only':'h_phys=-p_scale*eta; p_scale not read, no numeric physical h',
      'added_power':'S*p*u_p+q*u_q','boundaries':'d=0 retains first-order initial condition; PSD with c=0 implies d=0; c0=0 retained without division',
      'scope':'reference to reviewed conditional algebra; not a program proof of FOM or train equilibrium'}
def control_inputs(paths,bindings,counts):
    def control(key):
        value=read(paths[key]);counts['controls_decoded']+=1;return value
    P=control('parent_P_manifest');E=control('parent_E_manifest');Q=control('parent_Q_manifest');Psum=control('parent_P_summary');stage=control('stage_summary');oldbindings=control('stage_source_bindings')
    if P.get('status')!='verified' or Q.get('status')!='COMPLETED' or stage.get('qualification')!='verified' or Q.get('denominators')!=DENOM or stage.get('denominators')!=DENOM:raise SourceUnknown('old qualification controls unresolved')
    if P['rows_sha256']!=bindings['files']['rows']['sha256'] or Psum['manifest_sha256']!=bindings['files']['parent_P_manifest']['sha256']:raise SourceUnknown('row/P identity chain differs')
    if len(Q['machines'])!=10 or len(E['slots'])!=10 or len(stage['stage_rows'])!=120 or len(stage['parent_closures'])!=60:raise SourceUnknown('old fixed statistics set differs')
    for i,q in enumerate(Q['machines']):
        key=f'residual_machine_{i:02d}';e=bindings['files'][key]
        if q['machine']!=i or q['private_evidence_sha256']!=e['sha256'] or E['slots'][i]['machine']!=i or E['slots'][i]['numeric_passed'] is not True or E['slots'][i]['unique'] is not True:raise SourceUnknown('old machine eligibility/hash differs')
        if any(oldbindings['files'][key][k]!=e[k] for k in ('path','bytes','sha256')):raise SourceUnknown('old stage residual input differs')
    if any(oldbindings['files']['rows'][k]!=bindings['files']['rows'][k] for k in ('path','bytes','sha256')):raise SourceUnknown('old stage row input differs')
    references={(r['machine'],r['steps'],r['group'],r['phase']):r for r in stage['stage_rows']}
    required={(i,s,g,p) for i in range(10) for s in (8,4) for g in ('stable','unstable') for p in PHASES}
    if set(references)!=required:raise SourceUnknown('old stage statistic keys differ')
    closure_keys=[(r['machine'],r['steps'],r['group']) for r in stage['parent_closures']]
    if len(set(closure_keys))!=60 or set(closure_keys)!={(i,s,g) for i in range(10) for s in (8,4) for g in ('all','stable','unstable')} or any(r.get('status')!='verified' for r in stage['parent_closures']):raise SourceUnknown('old overall closure keys/status differ')
    return Q,Psum,stage,references
def compare_stage(h,moment,stats,reference):
    for key in ('mu','C_Q','C_L','C_diff'):
        if not h.close(moment[key],reference[key]):raise SourceUnknown('old stage moment closure failed')
    for key in ('N_original','N_available','windows'):
        if type(reference[key]) is not int or moment[key]!=reference[key]:raise SourceUnknown('old stage count closure failed')
    for key in ('R_Q','R_L','nu'):
        if not (stats[key] is None and reference[key] is None) and not h.close(stats[key],reference[key]):raise SourceUnknown('old stage RMS/nu closure failed')
    if 'representation_difference_RMS' in reference:
        if not h.close(stats.get('representation_difference_RMS'),reference['representation_difference_RMS']):raise SourceUnknown('old stage representation RMS closure failed')
    for key in ('status','reason','classification','compatible','resolved'):
        if type(stats[key]) is not type(reference[key]) or stats[key]!=reference[key]:raise SourceUnknown('old stage exact gate closure failed')
    return {'status':'verified','moment':moment,'statistics':stats,'parent_row_sha256':h.digest(reference)}
def array_hash(a):
    import numpy as np
    a=np.ascontiguousarray(a);v=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode());v.update(a.tobytes());return v.hexdigest()

def source():
    release=checked_release();claim_private(PRIVATE,release);start=time.perf_counter();cpu=time.process_time();phase='identity';paths={};bindings=None
    counts={k:0 for k in ('assets_verified','source_bytes_hashed','static_files_read','controls_decoded','rows_decoded','residual_decode_attempts','residual_containers_decoded','residual_bytes_read','selected_bytes_decoded','machines_aligned','overall_moments_attempted','overall_moments_computed','overall_attempts','overall_closures_verified','stage_attempts','stage_moments_computed','stage_closures_verified','assets_reverified')}
    excerpts={};overall=[];stages=[]
    try:
        bindings=read(ROOT/FROZEN['source_bindings'][0]);paths=verify_assets(bindings,ROOT,counts)
        for key,e in bindings['files'].items():
            if e['kind']=='static_content':
                excerpts[key]={'source_sha256':e['sha256'],**excerpt(paths[key].read_text(encoding='utf-8-sig'),e)};counts['static_files_read']+=1
        write_once(PRIVATE/'M_static_evidence.json',{'mathematical_reference':math_reference(),'excerpts':excerpts,
          'levels':{'conditional_algebra':'reviewed proof reference','current_source_declarations':'per-needle excerpt evidence','historical_execution':'unknown','effective_time_conversion':'unknown','actual_reference_limiter_state':'unknown','physical_unit_certification':'unknown'},'static_unknown_does_not_block_residual_diagnostic':True})
        h=load_helpers(paths['static_selected_reader'],bindings['files']['static_selected_reader']['sha256'])
        Q,Psum,stage,reference=control_inputs(paths,bindings,counts)
        phase='alignment';rows=read(paths['rows']);counts['rows_decoded']+=1;h.validate_rows(rows,bindings,Psum);payloads=[]
        for machine in range(10):
            counts['residual_decode_attempts']+=1;raw=paths[f'residual_machine_{machine:02d}'].read_bytes();counts['residual_bytes_read']+=len(raw)
            actual,selected,size=h.selected_residuals(raw);del raw
            counts['residual_containers_decoded']+=1;counts['selected_bytes_decoded']+=size;h.align_residuals(actual,selected,machine,rows)
            payloads.append(selected);counts['machines_aligned']+=1
        phase='old_overall_closure';direct={}
        old_closure_hashes={(r['machine'],r['steps'],r['group']):r['closure_sha256'] for r in stage['parent_closures']}
        for machine,payload in enumerate(payloads):
            for steps in (8,4):
                for group in ('stable','unstable'):
                    counts['overall_moments_attempted']+=1;direct[(machine,steps,group)]=h.moments(payload[str(steps)],rows,group,steps)[0];counts['overall_moments_computed']+=1
                counts['overall_moments_attempted']+=1;direct[(machine,steps,'all')]=h.mix_groups(direct[(machine,steps,'stable')],direct[(machine,steps,'unstable')]);counts['overall_moments_computed']+=1
                for group in ('stable','unstable','all'):
                    counts['overall_attempts']+=1;moment=direct[(machine,steps,group)];check=h.parent_closure(moment,Q['machines'][machine]['fixed_residuals'][group][str(steps)])
                    recalculated=h.digest({'machine':machine,'steps':steps,'group':group,**check});old_hash=old_closure_hashes[(machine,steps,group)]
                    overall.append({'machine':machine,'steps':steps,'group':group,'moment':moment,'closure':check,'parent_statistics':Q['machines'][machine]['fixed_residuals'][group][str(steps)],
                      'parent_stage_closure_sha256':old_hash,'recomputed_closure_sha256':recalculated,'closure_hash_agrees':recalculated==old_hash});counts['overall_closures_verified']+=1
        phase='old_stage_closure'
        for machine,payload in enumerate(payloads):
            for steps in (8,4):
                for group in ('stable','unstable'):
                    parts=[]
                    for p in PHASES:
                        counts['stage_attempts']+=1;moment=h.moments(payload[str(steps)],rows,group,steps,p)[0];counts['stage_moments_computed']+=1;stats=h.conditional(moment)
                        check=compare_stage(h,moment,stats,reference[(machine,steps,group,p)])
                        for suffix,field in (('Q','C_Q'),('L','C_L'),('diff','C_diff')):
                            key='square_contribution_fraction_'+suffix
                            if key in reference[(machine,steps,group,p)]:
                                total=direct[(machine,steps,group)][field];fraction=moment[field]/total if total>0 else None;old=reference[(machine,steps,group,p)][key]
                                if not (fraction is None and old is None) and not h.close(fraction,old):raise SourceUnknown('old stage contribution fraction closure failed')
                        stages.append({'machine':machine,'steps':steps,'group':group,'phase':p,**check,'parent_statistics':reference[(machine,steps,group,p)]});parts.append(moment);counts['stage_closures_verified']+=1
                    for key in ('mu','C_Q','C_L','C_diff'):
                        if not h.close(math.fsum(m[key] for m in parts),1. if key=='mu' else direct[(machine,steps,group)][key]):raise SourceUnknown('stage partition differs from original overall')
        phase='snapshot';import numpy as np
        arrays={f'{rep}{s}':np.column_stack([np.concatenate([np.asarray(r[rep+'_residual'],dtype=np.float64) for r in payload[str(s)]]) for payload in payloads]) for s in (8,4) for rep in ('Q','L')}
        identities={key:{'dtype':str(a.dtype),'shape':list(a.shape),'sha256':array_hash(a)} for key,a in arrays.items()}
        for key,a in arrays.items():
            steps=key[-1]
            if a.dtype!=np.dtype('float64') or a.shape!=(bindings['expected_rows_per_representation_per_machine'][steps],10) or not np.isfinite(a).all():raise SourceUnknown('selected residual snapshot shape/finite qualification failed')
        with (PRIVATE/'frozen_residuals.npz').open('xb') as f:np.savez_compressed(f,**arrays)
        with (PRIVATE/'rows_frozen.json').open('xb') as f:f.write(paths['rows'].read_bytes())
        write_once(PRIVATE/'parent_statistics.json',{'overall60':overall,'stage120':stages,'signed_means_evaluated':0,'eta_estimates':0})
        phase='final_source_recheck'
        for key,e in bindings['files'].items():
            if paths[key].stat().st_size!=e['bytes'] or sha(paths[key])!=e['sha256']:raise SourceUnknown('source asset changed during M/S')
            counts['assets_reverified']+=1
        manifest={'status':'verified','stage':'source','release':release,'counts':counts,'denominators':DENOM,'input_bindings':bindings,
          'snapshots':identities,'residual_snapshot_sha256':sha(PRIVATE/'frozen_residuals.npz'),'rows_sha256':sha(PRIVATE/'rows_frozen.json'),
          'parent_statistics_sha256':sha(PRIVATE/'parent_statistics.json'),'M_evidence_sha256':sha(PRIVATE/'M_static_evidence.json'),
          'snapshot_order':'rows original lexicographic event order; contiguous original windows within each event; columns machine0..9',
          'all28_assets_unchanged':True,'eta_estimates':0,'signed_means':0,'E_Q_released':False,
          'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'runtime':{'python':sys.version,'numpy':np.__version__,'device':'CPU','dtype':'float64','threads':1}}
        write_once(PRIVATE/'manifest.json',manifest)
        public={'status':'verified','stage':'source','counts':counts,'denominators':DENOM,'coverage':Psum['coverage'],
          'source_layers':{'mathematics':'independently reviewed conditional proof reference; no FOM proof','static_excerpts':excerpts,
          'historical_execution':'unknown','effective_time_reference_limiter_state':'unknown','physical_unit_certification':'unknown'},
          'mathematical_scope':math_reference(),'manifest_sha256':sha(PRIVATE/'manifest.json'),'snapshots':identities,
          'residual_snapshot_sha256':manifest['residual_snapshot_sha256'],'parent_statistics_sha256':manifest['parent_statistics_sha256'],
          'all28_assets_unchanged':True,'E_Q_released':False,'eta_estimates':0,'signed_means':0,'release':release,
          'wall_seconds':manifest['wall_seconds'],'interpreter_CPU_seconds':manifest['interpreter_CPU_seconds'],'wrapper_native_wall_seconds':None,'tool_wall_seconds':None}
        write_once(PUBLIC/'source_summary.json',public);write_once(PRIVATE/'complete.json',{'status':'COMPLETED','manifest_sha256':sha(PRIVATE/'manifest.json'),'summary_sha256':sha(PUBLIC/'source_summary.json')});return public
    except BaseException as exc:
        write_once(PRIVATE/'partial_closures.json',{'overall':overall,'stages':stages,'excerpts':excerpts})
        unchanged=None
        if phase=='final_source_recheck':unchanged=False if isinstance(exc,SourceUnknown) else None
        elif len(paths)==28:
            try:
                matches=[]
                for key,e in bindings['files'].items():
                    matches.append(paths[key].stat().st_size==e['bytes'] and sha(paths[key])==e['sha256']);counts['assets_reverified']+=1
                unchanged=all(matches)
            except (OSError,ValueError):unchanged=None
        detail={'status':'unknown','failure_stage':phase,'counts':counts,'denominators':DENOM,'reason':'M/S qualification or technical failure; full details private',
          'source_assets_unchanged_at_failure':unchanged,
          'E_Q_released':False,'eta_estimates':0,'signed_means':0,'wall_seconds':time.perf_counter()-start,'interpreter_CPU_seconds':time.process_time()-cpu,'release':release,'automatic_retry':False}
        write_once(PRIVATE/'failure.json',{**detail,'error':repr(exc),'traceback':traceback.format_exc()});write_once(PUBLIC/'source_unknown.json',detail);raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['source']);p.parse_args();source()
