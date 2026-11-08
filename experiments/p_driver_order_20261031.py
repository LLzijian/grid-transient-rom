"""Bound S0 static extraction and six frozen mathematical cases, separately released.

Bootstrap imports only the standard library. No pointer traversal or S1 exists.
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
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1]
RUN='p_driver_order_20261031'
PRIVATE=ROOT/'private_data'/RUN
PUBLIC=ROOT/'results'/RUN
FROZEN={
 'protocol':('reports/'+RUN+'_protocol.md','dbcb0c7210bba1dedd40c373c51f180fd0f09e28f98a6f8b928fb704fb8e1e7f'),
 'mathematics':('reports/'+RUN+'_mathematics.md','a4ee61d22952e1aea4cfe3f2692877bda5f37a1101de779ba16207edfede79e1'),
 'architecture_brief':('reports/p_minimal_closure_20261031_architecture_brief.md','8fa22d9a54d69c23cb4a592fbe40a1e4f22396ae38cf6f7926d401e801d156e2'),
 'source_bindings':('results/'+RUN+'/source_bindings.json','b5f728ed82e0e0de4a6b3f3289925fd22bef89f13ac0cbd7d9cf7c6feeea74ac'),
 'authorization':('results/'+RUN+'/authorization.json','0056b5e2c527df902ee97c6acf0bfe49dde19d60600996f0cc41bc18c98a0909')}
CASES=((1.,0.,1.),(1.,.5,1.),(0.,.5,1.),(1.,.5,0.),(2.,.5,1.),(1.,1.e-6,1.))
IDENTITY_KEY=re.compile(r'python|executable|environment|version|case|fingerprint|attestation',re.I)
SOURCE_LINE=re.compile(r'andes.*(?:version|__version__)|(?:version|__version__).*andes|\bcase\b|get_case|GENROU|\btm\b|controller|governor|fingerprint|attestation|TurbineGov|remove.*(?:gov|control)|retain.*(?:gov|control)',re.I)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(json_safe(value),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value

def release_expected(stage):
    if stage not in ('bootstrap','mathematics'):raise ValueError('unknown stage')
    return {'source_sha256':sha(Path(__file__)),'tests_sha256':sha(ROOT/'tests'/('test_'+RUN+'.py')),
      **{k+'_sha256':h for k,(path,h) in FROZEN.items()},
      'static_content_files_upper_bound':11 if stage=='bootstrap' else 0,
      'matrix_cases_upper_bound':6 if stage=='mathematics' else 0,
      'scipy_expm_calls_upper_bound':6 if stage=='mathematics' else 0,
      'mp100_expm_calls_upper_bound':6 if stage=='mathematics' else 0,
      'directory_metadata_enumerations_upper_bound':0,'real_trajectory_reads_upper_bound':0,
      'cache_array_reads_upper_bound':0,'parameter_estimates_upper_bound':0,'optimizer_upper_bound':0,
      'new_autonomous_rom_rollouts_upper_bound':0,'new_fom_upper_bound':0,'model_writes_upper_bound':0}

def verify_release(stage,release,expected):
    if release.get('released') is not True or release.get('allowed_stage')!=stage:raise PermissionError('independent root release required')
    for key,value in expected.items():
        if release.get(key)!=value or (type(value) is int and type(release.get(key)) is not int):raise PermissionError('frozen identity or limit differs: '+key)

def checked_release(stage):
    expected=release_expected(stage)
    for key,(path,h) in FROZEN.items():
        if sha(ROOT/path)!=h:raise ValueError('frozen '+key+' changed')
    path=PUBLIC/(stage+'_execution_release.json')
    release=read(path);verify_release(stage,release,expected)
    # The root-selected frozen review is a control input: hash only, do not read
    # a mutable review draft or follow references from the snapshot.
    review=(ROOT/release['review_snapshot_path']).resolve();review.relative_to(ROOT.resolve())
    if review.suffix.lower() not in ('.json','.md') or review.stat().st_size>1024*1024 or sha(review)!=release['review_snapshot_sha256']:
        raise ValueError('frozen review snapshot identity differs')
    return {'release_sha256':sha(path),'identities':expected,'review_snapshot_sha256':release['review_snapshot_sha256']}

def claim_stage(directory,stage,release):
    directory=Path(directory)
    # A directory from any prior attempt is never reused, even if completed.
    directory.mkdir(parents=True,exist_ok=False)
    write_once(directory/'started.json',{'status':'STARTED','stage':stage,'pid':os.getpid(),'release':release})

def bound_file(root,entry):
    root=Path(root).resolve();path=(root/entry['path']).resolve();path.relative_to(root)
    if path.suffix.lower() not in ('.json','.py','.md'):raise ValueError('only frozen static text files allowed')
    size=path.stat().st_size
    if size>1024*1024 or size!=entry['bytes']:raise ValueError('bound static size differs or exceeds 1 MiB')
    if sha(path)!=entry['sha256']:raise ValueError('bound static SHA differs')
    return path

def scalar_identities(value,prefix=()):
    """Match scalar leaf keys, never ancestor objects or list contents."""
    result=[]
    if not isinstance(value,dict):return result
    for key,item in value.items():
        path=prefix+(str(key),)
        if isinstance(item,dict):result.extend(scalar_identities(item,path))
        elif isinstance(item,(str,int,float,bool)) or item is None:
            if IDENTITY_KEY.search(str(key)):
                if isinstance(item,float) and not math.isfinite(item):continue
                result.append({'key_path':list(path),'value':item})
        # Lists are never expanded, even under a matching parent key.
    return result

def source_lines(text):
    return [{'line':i,'text':line} for i,line in enumerate(text.splitlines(),1) if SOURCE_LINE.search(line)]

def bootstrap(directory):
    bindings=read(ROOT/FROZEN['source_bindings'][0]);entries=bindings['bootstrap_files']
    if len(entries)!=11 or len({e['path'] for e in entries})!=11:raise ValueError('exact11 bootstrap list required')
    # Validate every finite input before any content extraction. Do not follow any
    # paths inside the records, including preflight_inventory/job_reconciliation.
    paths=[bound_file(ROOT,e) for e in entries]
    extracts=[];config_snapshot=None
    for entry,path in zip(entries,paths):
        text=path.read_text(encoding='utf-8-sig');item={'source':entry,'identities':[],'source_lines':[]}
        if path.suffix.lower()=='.json':
            content=json.loads(text);item['identities']=scalar_identities(content)
            if entry['path']=='configs/private_dataset.json':config_snapshot=content
        elif path.suffix.lower()=='.py':item['source_lines']=source_lines(text)
        else:item['source_lines']=source_lines(text) # Existing statements only; no metric recomputation.
        extracts.append(item)
    detail={'stage':'bootstrap','extracts':extracts,'static_config_snapshot':config_snapshot,
      'pointer_contents_followed':False,'ANDES_imported':False,'trajectory_or_cache_loaded':False}
    write_once(directory/'extraction.json',detail)
    return {'status':'COMPLETED','stage':'bootstrap','source_binding':'unknown',
      'files_verified':11,'identity_scalar_count':sum(len(r['identities']) for r in extracts),
      'source_line_count':sum(len(r['source_lines']) for r in extracts),'extraction_sha256':sha(directory/'extraction.json'),
      'bound_input_hashes':[e['sha256'] for e in entries],
      'identity_levels':{'L1_static_readable':'verified','L2_package_case_config_match':'unknown','L3_historical_execution':'unknown','L4_train_activation':'unknown'},
      'gaps':['package/case content identity not established by S0; exact extracted pointers require a separately frozen S1','historical generating-source identity unknown','actual train operating intervals not read'],
      'S1_implemented':False,'next_stage_automatically_released':False,'actual_directory_metadata_enumerations':0,
      'actual_static_content_files':11,'actual_real_trajectory_reads':0,'actual_parameter_estimates':0}

def mathematics(directory):
    # These imports and every scientific operation are confined to the released M
    # entry point. Engineering tests never call this function.
    import numpy as np
    import scipy
    from scipy.linalg import expm
    import mpmath as mp
    counts={'scipy_expm_calls':0,'mp100_expm_calls':0};rows=[]
    def finite(a):return bool(np.isfinite(a).all())
    def compare(left,right):
        good=finite(left) and finite(right)
        with np.errstate(over='ignore',invalid='ignore'):
            error=float(np.max(np.abs(left-right))) if good else None
        good=good and error is not None and np.isfinite(error)
        return {'status':'PASS' if good and np.allclose(left,right,atol=1e-12,rtol=1e-12) else 'unknown',
          'max_abs_error':error if good else None}
    def exact(value):
        numerator,denominator=float(value).as_integer_ratio()
        return mp.mpf(numerator)/mp.mpf(denominator)
    for index,(a,b,c) in enumerate(CASES):
        folder=directory/f'case_{index:02d}';folder.mkdir(exist_ok=False)
        write_once(folder/'started.json',{'case_index':index,'parameters':[a,b,c],'maximum_scipy_expm_calls':1,'maximum_mp100_expm_calls':1})
        start=time.perf_counter()
        A=np.array([[0,1,0,0,0],[-1,-1,1,-1,0],[0,-1,-a,0,b],[0,1,0,-1,0],[0,0,-b,0,-c]],dtype=np.float64)
        h=np.float64(.01);Cp=np.eye(5)[2:3];Cw=np.eye(5)[1:2];Ch=np.eye(5)[:2];Cout=np.eye(5)[:3]
        counts['scipy_expm_calls']+=1;E=expm(-h*A)
        powers=[np.eye(5)]
        for k in range(10):powers.append(powers[-1]@E)
        with mp.workdps(100):
            amp=mp.matrix([[exact(v) for v in row] for row in A]);hmp=exact(h)
            counts['mp100_expm_calls']+=1;eref=mp.expm(-hmp*amp)
            mp_powers=[mp.eye(5)]
            for k in range(10):mp_powers.append(mp_powers[-1]*eref)
            # Form each power in MP100 first, then round once to binary64.
            references=[np.array([[float(power[i,j]) for j in range(5)] for i in range(5)],dtype=np.float64) for power in mp_powers]
        power_checks=[compare(left,right) for left,right in zip(powers,references)]
        Oh=np.vstack([Ch@power for power in powers]+[Cp]);Oh_ref=np.vstack([Ch@power for power in references]+[Cp])
        ohcheck=compare(Oh,Oh_ref)
        if not finite(Oh):
            singular=None;threshold=None;rank=None;condition=None;null_rows=[]
        else:
            _,sv,vt=np.linalg.svd(Oh,full_matrices=False)
            threshold=float(max(Oh.shape)*np.finfo(np.float64).eps*sv[0]);rank=int(np.sum(sv>threshold))
            singular=sv.tolist();condition=float(sv[0]/sv[-1]) if sv[-1]>0 and np.isfinite(sv[0]/sv[-1]) else None
            null_rows=[v for value,v in zip(sv,vt) if value<=threshold]
        apowers=[np.eye(5)]
        for k in range(4):apowers.append(apowers[-1]@A)
        Nout=np.vstack([Cout@power for power in apowers])
        energy=compare(A.T+A,-2*np.diag([0,1,a,1,c]))
        elimination=compare(Cp@A@A+(a+c)*Cp@A+(a*c+b*b)*Cp+Cw@A+c*Cw,np.zeros((1,5)))
        R=np.vstack([np.eye(5)[0:1],Cw,Cp,Cw@A,Cw@A@A])
        determinant=compare(np.asarray([np.linalg.det(R)]),np.asarray([-b]))
        nested=None;eqcheck=None;kernel=None
        if b==0:
            eq=np.eye(5)[:,4]
            nested={'A_parent_q_column':compare(A[:4,4],np.zeros(4)),
              'powers_parent_q_columns':[compare(power[:4,4],np.zeros(4)) for power in powers],
              'parent_block':A[:4,:4].tolist(),'parent_expm_calls':0}
            eqcheck=compare(Cp@A+Cp+Cw,np.zeros((1,5)))
            kernel={'Oh_e_q':compare(Oh@eq,np.zeros(23)),'Nout_e_q':compare(Nout@eq,np.zeros(15))}
        null_details=[]
        for vector in null_rows:
            left=Oh@vector;right=Nout@vector
            null_details.append({'vector':vector.tolist(),'Oh_residual_L2':float(np.linalg.norm(left)) if finite(left) else None,
              'Nout_residual_L2':float(np.linalg.norm(right)) if finite(right) else None,'interpretation':'numerical threshold direction only; not exact ambiguity'})
        expected=4 if b==0 else 5
        comparisons=all(r['status']=='PASS' for r in [*power_checks,ohcheck,energy,elimination,determinant])
        if b==0:comparisons=comparisons and eqcheck['status']=='PASS' and all(r['status']=='PASS' for r in kernel.values()) and nested['A_parent_q_column']['status']=='PASS' and all(r['status']=='PASS' for r in nested['powers_parent_q_columns'])
        detail={'case_index':index,'parameters':[a,b,c],'A':A.tolist(),'h':float(h),'E':E.tolist(),
          'powers':[p.tolist() for p in powers],'MP100_power_references_rounded_binary64':[p.tolist() for p in references],
          'Oh':Oh.tolist(),'Oh_reference':Oh_ref.tolist(),'Nout':Nout.tolist(),'threshold_directions':null_details,
          'source_input_binary64_hex':{'h':float(h).hex(),'A':[[float(v).hex() for v in row] for row in A]}}
        write_once(folder/'matrices.json',detail)
        row={'case_index':index,'parameters':[a,b,c],'numeric_comparisons':'PASS' if comparisons else 'unknown',
          'power_comparisons':power_checks,'Oh_comparison':ohcheck,'singular_values':singular,'rank_threshold':threshold,'numerical_rank':rank,
          'theoretical_rank':expected,'rank_status':'agrees_with_theory' if rank==expected else 'numerical_rank_disagreement',
          'Oh_original_coordinate_condition':condition,'condition_status':'finite' if condition is not None else ('singular' if rank is not None and rank<5 else 'unknown'),
          'energy_identity':energy,'elimination_identity':elimination,'determinant_identity':determinant,
          'b0_nested':nested,'b0_consistent_initial_value_identity':eqcheck,'b0_output_kernel':kernel,
          'prediction_uniqueness':'mathematics attachment exact theorem for this fixed case; numerical observations do not replace proof',
          'nonfinite_status':'verified_finite' if finite(A) and all(finite(p) for p in powers) and finite(Oh) and finite(Nout) and singular is not None and all(np.isfinite(singular)) and threshold is not None and np.isfinite(threshold) else 'unknown',
          'matrix_artifact_sha256':sha(folder/'matrices.json'),'wall_seconds':time.perf_counter()-start,
          'scipy_expm_calls':1,'mp100_expm_calls':1,'MP_precision_decimal_digits':100}
        write_once(folder/'complete.json',row);rows.append(row)
    return {'status':'COMPLETED','stage':'mathematics','cases':rows,'counts':counts,
      'numerical_comparison_status':'PASS' if all(r['numeric_comparisons']=='PASS' and r['nonfinite_status']=='verified_finite' for r in rows) else 'unknown',
      'rank_agreement_status':'agrees_with_theory' if all(r['rank_status']=='agrees_with_theory' for r in rows) else 'numerical_rank_disagreement',
      'atol':1e-12,'rtol':1e-12,'precision':'binary64 plus fixed MP100 reference',
      'runtime':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__,'device':'CPU','threads':1},
      'actual_parameter_estimates':0,'actual_real_trajectory_reads':0,'actual_future_trajectories':0,
      'theory_reference_sha256':FROZEN['mathematics'][1],'scope':'six fixed artificial systems only; no actual source mechanism or prediction improvement established'}

def run(stage):
    release=checked_release(stage);directory=PRIVATE/stage
    claim_stage(directory,stage,release);start=time.perf_counter()
    try:
        result=bootstrap(directory) if stage=='bootstrap' else mathematics(directory)
        result.update(release=release,wall_seconds=time.perf_counter()-start,cpu_seconds=None,LLM_seconds=None,
          source_sha256=sha(Path(__file__)),optimizer_calls=0,new_ROM_rollouts=0,new_FOM=0,model_writes=0)
        write_once(directory/'result.json',result)
        # Public release identities contain hashes/limits only. No PID, native
        # environment paths, per-event identifiers or source excerpts are public.
        write_once(PUBLIC/(stage+'_summary.json'),result)
        write_once(directory/'complete.json',{'status':'COMPLETED','result_sha256':sha(directory/'result.json')})
        return result
    except BaseException as exc:
        write_once(directory/'failure.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-start,'automatic_retry':False});raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['bootstrap','mathematics']);args=parser.parse_args();run(args.stage)
