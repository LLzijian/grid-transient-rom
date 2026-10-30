"""Bound, read-only S qualification. No estimation or response implementation.

Only this module and NumPy are executed; production loaders are inspected as text.
Missing production attestations remain unknown, never inferred from waveforms.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import time
import traceback
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = 'p_history_transfer_20261031'
FROZEN = {
 'scientist': ('reports/'+RUN+'_scientist.md', 'a51c2292179967ad4ae05c69f2c68500aac51225b57a40cd14ad5d48ff8c560c'),
 'protocol': ('reports/'+RUN+'_protocol.md', '725a5ae143a46db70305f27266e7f7dd16b36cf614560dfd56b05114a129c276'),
 'bindings': ('results/'+RUN+'/source_bindings.json', 'f7fc5c061d82af7d1e26ef0443b85fb0bc1b0c28ea8c93a8b463a35100c250dc'),
 'authorization': ('results/'+RUN+'/authorization.json', 'db4a6ec8dd205c524245aaf7a6b7789000e260a178ccef0e369d1e494bf9b821')}
ZERO_LIMITS = ('maximum_main_parameter_estimates', 'same_objective_independent_qr_checks_upper_bound',
 'scalar_filter_calls_upper_bound', 'optimizer_upper_bound', 'new_autonomous_rom_rollouts_upper_bound',
 'model_writes_upper_bound')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_once(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def array_hash(a):
    a = np.ascontiguousarray(a)
    h=hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def parent_constants(parent):
    params=parent['parameters']
    b=np.asarray(params['beta0'],dtype=np.float64); scale=np.asarray(params['p_scale'],dtype=np.float64)
    if b.shape!=(10,2) or scale.shape!=(10,) or not np.isfinite(b).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('parent constants invalid')
    expected={k:array_hash(v) for k,v in {'g':b[:,0],'a':b[:,1],'p_scale':scale}.items()}
    proofs=parent['parent_qualification']
    if set(proofs)!={'A96_analytic','A16_analytic'} or any(p.get('all_original_qualification_passed') is not True or p.get('parameter_hashes')!=expected for p in proofs.values()):
        raise ValueError('parent parameter hashes or original qualification differ')
    return params

def bound_path(root, entry):
    path = (Path(root)/entry['path']).resolve()
    path.relative_to(Path(root).resolve())
    if path.suffix.lower() in ('.h5', '.hdf5'): raise ValueError('raw HDF forbidden')
    return path

def entries(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value: yield value
        else:
            for item in value.values(): yield from entries(item)

def verify_bindings(bindings, root):
    # Exactly the finite bound set. No resolution of links inside those records.
    checked = {}
    for entry in entries(bindings):
        path = bound_path(root, entry)
        if entry['path'] in checked: continue
        actual = sha(path)
        if actual != entry['sha256']: raise ValueError('binding hash mismatch: '+entry['path'])
        if 'bytes' in entry and path.stat().st_size != entry['bytes']: raise ValueError('binding size mismatch')
        checked[entry['path']] = actual
    return checked

def verify_release(release, hashes):
    if release.get('released') is not True or release.get('allowed_stage') != 'source':
        raise PermissionError('source stage is not released')
    if any(release.get(k) != v for k,v in hashes.items()): raise PermissionError('release identity differs')
    if any(type(release.get(k)) is not int or release[k] != 0 for k in ZERO_LIMITS):
        raise PermissionError('source stage must have zero computation limits')

def claim_private(path):
    # Even completed runs cannot be replayed, and unknown started jobs cannot be retried.
    Path(path).mkdir(parents=True, exist_ok=False)

def metadata_rows(text, ids):
    wanted = set(ids); selected = {}
    fields = ('event_id','clearing_time_s','record_start_delay_s','valid_length','trajectory_end_s')
    for row in csv.DictReader(io.StringIO(text)):
        if row.get('event_id') in wanted:
            eid = row['event_id']
            if eid in selected: raise ValueError('duplicate bound metadata id')
            selected[eid] = {k: row.get(k) for k in fields}
    return selected

def finite_number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) and not isinstance(value, bool) else None
    except (TypeError, ValueError): return None

def time_evidence(meta, index_row, *, verified_grid=False, verified_postclear=False, bound_nondefault_decoder=False):
    # Cache fallback delay is deliberately ignored; the original index takes priority.
    original = index_row or {}
    delay = finite_number(original.get('record_start_delay_s'))
    clearing = finite_number(original.get('clearing_time_s'))
    delay_level = 'original_metadata' if delay is not None else 'unknown; cache fallback is not measurement'
    cached_delay = finite_number(meta.get('record_start_delay_s'))
    if index_row is None and bound_nondefault_decoder and cached_delay is not None and cached_delay != 0:
        delay = cached_delay
        delay_level = 'bound decoder nondefault cached value; original key not reread'
        clearing = finite_number(meta.get('clearing_time_s'))
    time_present = 't' in meta
    t = meta.get('t')
    direct_grid = False
    if time_present:
        try:
            t = np.asarray(t, dtype=np.float64)
            direct_grid = bool(t.ndim == 1 and len(t)>=13 and np.isfinite(t).all()
                               and abs(t[0])<=1e-10
                               and np.allclose(np.diff(t), .01, atol=1e-9, rtol=0))
        except (TypeError, ValueError):
            direct_grid = False
    grid = direct_grid if time_present else verified_grid
    postclear = verified_postclear and (delay is None or delay >= 0)
    return {'status': 'verified' if grid and postclear else 'unknown',
      'grid_evidence': ('cache_time' if direct_grid else 'explicit_time_conflict') if time_present else ('bound_producer_check' if verified_grid else 'unknown'),
      'source_contract_conflict': bool(time_present and not direct_grid),
      'postclear_domain': 'verified' if postclear else 'unknown',
      'clear_time_s': clearing, 'record_start_delay_s': delay,
      'record_start_absolute_s': clearing+delay if clearing is not None and delay is not None else None,
      'deployment_since_clearing_s': delay+.1 if delay is not None else None,
      'history_description': 'existing postclear record first 100 ms' if postclear else 'unknown',
      'delay_evidence': delay_level}

def windows(last, steps):
    rows = [(start,start+steps) for start in range(0,max(0,last-steps+1),steps)]
    counts = {k:0 for k in ('history','cross_t0','future')}
    for start,end in rows:
        counts['history' if end<=10 else ('future' if start>=10 else 'cross_t0')] += 1
    m = len(rows)
    return {'windows':m,'phase':0,'last_included_index':m*steps,'tail_steps':last-m*steps,
      'counts':counts,'class_row_weight_sums':{k:v/(297*m) if m else None for k,v in counts.items()}}

def coverage(length, label):
    available = length-11
    n = 800 if label=='stable' else min(1500,available)
    valid = length>=13 and available>=n and n>=2
    return {'status':'verified' if valid else 'unknown','length':length,'n_future':n,
      'future_seconds':n*.01 if valid else None,'relative_record_range_s':[-.1,n*.01] if valid else None,
      'history11':length>=11,'future20':length>=13,
      'H40':windows(10,4),'H80':windows(10,8),'H20':windows(10,2),
      'future20_indices':[10,12], 'A':{str(s):windows(10+n,s) for s in (2,4,8)} if valid else None}

def member_prefix(x):
    if x.ndim!=2 or x.shape[1]!=30 or x.dtype!=np.dtype('float64') or len(x)<13:
        raise ValueError('member shape, float64 or prefix length differs')
    prefix = np.array(x[:13,10:30], copy=True)
    if not np.isfinite(prefix).all(): raise ValueError('nonfinite source prefix')
    return prefix

def source_proof(bindings, producer, texts, m0=None):
    code = bindings['producer_code']
    matched = len(code)==24 and producer.get('code')=={k:v['sha256'] for k,v in code.items()}
    clauses = {
      'export': ('experiments/generate_private_dataset.py', ('("Pm", "tm")','ss.GENROU','post_sample_start')),
      'old_adapter': ('src/kiki/data/private_dataset.py', ('result["Pm_raw"] = np.asarray(raw["Pm"])',)),
      'old_coordinate': ('experiments/train_local_memory_0098.py', ("can['Pm_raw']-base.pm_star","np.column_stack","np.allclose(np.diff(can['t']), .01")),
      'new_coordinate': ('experiments/closure_discovery_data.py', ('pm - np.asarray(base_model.pm_star','"t": t','system_attestation_complete')),
      'save': ('experiments/r_memory_study_20261025.py', ('np.asarray(event["x"], dtype=np.float64)','_save_event_cache(base / "fit_events.npz", fit)'))}
    checks = {k:all(token in texts.get(path,'') for token in tokens) for k,(path,tokens) in clauses.items()}
    # The frozen M0 attests the already exported source declaration, not a new
    # raw HDF inspection. Current generator text is corroborating semantics only.
    m0 = m0 or {}
    historical = (m0.get('asset_identity')=='verified_frozen_model_cache'
      and m0.get('coordinate_alignment')=='matched_source_mapping_derivation_and_fingerprint'
      and m0.get('equilibrium_reference_alignment')=='matched_at_1e12'
      and m0.get('fit_source_metadata_covered')==297 and m0.get('fit_source_metadata_denominator')==297
      and 'Pm=GENROU.tm' in m0.get('raw_units','')
      and bool(m0.get('metadata_fragment',{}).get('state_mapping'))
      and bool(m0.get('historical_metadata_fingerprint')))
    lineage = matched and all(checks.values()) and historical
    return {'status':'verified' if lineage else 'unknown','producer_code_matches':matched,
      'source_clauses':checks,'historical_export_declaration_verified':historical,
      'evidence_levels':['frozen M0 exported-source metadata/mapping/fingerprint','production prepare 24 code hashes to exact cache','bound adapter applies same coordinate at every sample','current generator corroboration only'],
      'raw_HDF_events_rechecked':False,'generator_historical_execution_identity':'unknown',
      'reason':None if lineage else 'historical source declaration or actual producer code binding incomplete'}

def run(root=ROOT):
    root = Path(root).resolve(); public = root/'results'/RUN; private = root/'private_data'/RUN
    frozen = {}
    for key,(name,expected) in FROZEN.items():
        if sha(root/name)!=expected: raise ValueError('frozen '+key+' changed')
        frozen[key] = expected
    hashes = {'source_sha256':sha(Path(__file__)), 'tests_sha256':sha(root/'tests'/('test_'+RUN+'_source.py')),
      'protocol_sha256':frozen['protocol'],'authorization_sha256':frozen['authorization'],
      'source_bindings_sha256':frozen['bindings']}
    release_path = public/'source_execution_release.json'
    verify_release(read(release_path),hashes)
    claim_private(private)
    start = time.perf_counter()
    write_once(private/'source_started.json',{'status':'STARTED','pid':os.getpid(),'release_sha256':sha(release_path),'hashes':hashes})
    try:
        bindings = read(root/FROZEN['bindings'][0]); checked = verify_bindings(bindings,root)
        producer = read(bound_path(root,bindings['producer_prepare']))
        parent = read(bound_path(root,bindings['parent_p_form']['manifest.json']))
        proof = read(bound_path(root,bindings['parent_p_form']['prepare_complete.json']))
        if proof.get('manifest_sha256')!=bindings['parent_p_form']['manifest.json']['sha256']: raise ValueError('parent completion differs')
        cache_entry = bindings['fit_cache']
        if producer.get('fit_cache_sha256')!=cache_entry['sha256']: raise ValueError('producer cache differs')
        ids = producer['fit_ids']
        if len(ids)!=297 or len(set(ids))!=297 or set(ids)!=set(parent['fit_ids']): raise ValueError('frozen cohort differs')
        texts = {e['path']:bound_path(root,e).read_text(encoding='utf-8-sig') for e in [*bindings['producer_code'].values(),bindings['generator_source']]}
        m0=read(bound_path(root,bindings['prior_p_port']['manifest.json']))['M0_source_metadata']
        lineage = source_proof(bindings,producer,texts,m0)
        indexes = {family:metadata_rows(bound_path(root,e).read_text(encoding='utf-8-sig'),ids) for family,e in bindings['original_metadata_indexes'].items()}
        rows=[]; prefixes={}; labels={}
        with np.load(bound_path(root,cache_entry),allow_pickle=False) as data:
            meta=json.loads(str(data['metadata_json']))
            if len(meta)!=297 or [m['id'] for m in meta]!=ids: raise ValueError('cache order or identity differs')
            for i,m in enumerate(meta):
                eid=m['id']; label=m['label']; labels[eid]=label
                x=data[f'event_{i:04d}_x']; prefix=member_prefix(x)
                prefixes[f'event_{i:04d}_omega_p']=prefix
                family=m.get('root'); old=family in indexes
                index=indexes.get(family,{}).get(eid)
                cls=m.get('classification_meta',{})
                family_known=old or (family=='closure_discovery' and cls.get('system_attestation_complete') is True and cls.get('system_fingerprint_sha256')==m0.get('historical_metadata_fingerprint'))
                # Exact original metadata is required for old-source linkage. New
                # cached delay can never authenticate itself because of `or 0.0`.
                timing=time_evidence(m,index,verified_grid=lineage['status']=='verified' and family_known,
                  verified_postclear=lineage['status']=='verified' and family_known and (not old or index is not None),
                  bound_nondefault_decoder=family=='closure_discovery' and lineage['status']=='verified')
                length_checks={
                  'cache_time_matches_length': 't' not in m or (isinstance(m['t'],(list,tuple,np.ndarray)) and len(m['t'])==len(x)),
                  'original_length_matches': not old or (index is not None and finite_number(index.get('valid_length'))==len(x)),
                  'endpoint_matches': finite_number((index or m).get('trajectory_end_s')) is not None and abs(float((index or m)['trajectory_end_s'])-(len(x)-1)*.01)<=1e-8}
                timing['length_checks']=length_checks
                if not all(length_checks.values()): timing['status']='unknown'
                c=coverage(len(x),label)
                rows.append({'id':eid,'family':family,'label':label,'coverage':c,'time':timing,'prefix_sha256':array_hash(prefix),
                  'history_p_sha256':array_hash(prefix[:11,10:]),'p0_sha256':array_hash(prefix[10,10:]),'independent_snapshot':'not_available'})
                del x
        if labels!=parent['event_labels'] or list(labels.values()).count('stable')!=267 or list(labels.values()).count('unstable')!=30: raise ValueError('267/30 cohort differs')
        params=parent_constants(parent)
        # Reuse only the already qualified, hash-bound parent manifest; no model load.
        write_once(private/'parent_constants.json',{'parameters':params,'parent_qualification':parent['parent_qualification'],'parent_manifest_sha256':bindings['parent_p_form']['manifest.json']['sha256']})
        with (private/'prefixes.npz').open('xb') as f: np.savez_compressed(f,**prefixes)
        write_once(private/'members.json',rows)
        qualified=lineage['status']=='verified' and all(r['coverage']['status']=='verified' and r['time']['status']=='verified' for r in rows)
        result={'status':'verified' if qualified else 'unknown','stage':'source','denominators':{'all':297,'stable':267,'unstable':30},
          'source_lineage':lineage,'coverage_verified':sum(r['coverage']['status']=='verified' for r in rows),
          'time_verified':sum(r['time']['status']=='verified' for r in rows),'delay_known':sum(r['time']['record_start_delay_s'] is not None for r in rows),
          'independent_snapshot':'not_available','dependencies_released':False,'reason':None if qualified else 'source/time/coverage unresolved; dependent stages stopped',
          'actual_future_range_seconds':{label:[min(r['coverage']['future_seconds'] for r in rows if r['label']==label and r['coverage']['future_seconds'] is not None),max(r['coverage']['future_seconds'] for r in rows if r['label']==label and r['coverage']['future_seconds'] is not None)] if any(r['label']==label and r['coverage']['future_seconds'] is not None for r in rows) else None for label in ('stable','unstable')},
          'hashes':hashes,'bound_inputs':checked,'private_members_sha256':sha(private/'members.json'),'prefixes_sha256':sha(private/'prefixes.npz'),
          'source_closure':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
          'environment':{'python':platform.python_version(),'numpy':np.__version__,'device':'CPU','dtype':'float64','threads':1,'thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}},
          'work':{k:0 for k in ZERO_LIMITS},'wall_seconds':time.perf_counter()-start,'cpu_seconds':None}
        write_once(private/'manifest.json',result); write_once(public/'source_summary.json',result)
        write_once(private/'source_complete.json',{'status':'COMPLETED','qualification':result['status'],'manifest_sha256':sha(private/'manifest.json')})
        return result
    except BaseException as exc:
        write_once(private/'source_failure.json',{'status':'FAILED','error':repr(exc),'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-start,'automatic_retry':False})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['source']); parser.parse_args()
    run()
