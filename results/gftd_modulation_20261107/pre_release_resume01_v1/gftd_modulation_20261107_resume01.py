"""Single released technical continuation of the frozen Run2 I/O failure."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import gftd_modulation_20261107 as original

ROOT=original.ROOT
OUTPUT=original.OUTPUT
AUDIT_SHA256='faf652029db757a21ee04e61a8f397e749460c819ea568db79b444e067f9078e'
SOURCE_INDEX=36
TARGET_INDEX=38
NEW_SOURCES=('experiments/gftd_modulation_20261107_resume01.py','tests/test_gftd_modulation_resume01.py')


def atomic_json(path,value,*,exclusive=False):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+'.tmp')
    def serial(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        raise TypeError(type(item).__name__)
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,default=serial,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def difference(new,old):
    if isinstance(new,dict):
        if set(new)!=set(old):raise ValueError('counter schema changed')
        return {key:difference(new[key],old[key]) for key in new}
    if not isinstance(new,(int,float)) or not isinstance(old,(int,float)) or new<old:
        raise ValueError('consumed counters must not decrease')
    return new-old


def sequence(fits):
    return [dict(batch_id=fit['batch_id'],mode=fit['mode'],input_id=input_id,fit=fit)
            for fit in fits['records'] for input_id in fit['packet_ids']]


def row_key(row):return row['batch_id'],row['mode'],row['input_id']


def four_keys(packet):
    if set(packet)!={'theta_history','omega_history','history_times','p0','sigma'}:
        raise ValueError('only frozen legal observation packets are allowed')
    return {key:packet[key] for key in ('theta_history','omega_history','history_times','p0')}


def same_four_keys(left,right):
    for key in ('theta_history','omega_history','history_times','p0'):
        a,b=(np.asarray(value[key],dtype='<f8') for value in (left,right))
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or a.tobytes()!=b.tobytes():
            raise ValueError('recovery requires exact four-key float64 identity including signed zero')


def recover_lost(partial,failed,fits,observed,audit):
    rows=partial['rows'];ordered=sequence(fits)
    if len(rows)!=38 or len(ordered)!=48:raise ValueError('expected 38 durable outputs and 48 planned pairs')
    if any(row_key(row)!=row_key(ordered[i]) for i,row in enumerate(rows)):
        raise ValueError('durable rows are not the original prefix')
    if failed['status']!='FAILED' or failed['counts']['initializations_attempted']!=39 or failed['counts']['initializations_completed']!=39:
        raise ValueError('unexpected failed action state')
    if partial['counts']['initializations_attempted']!=38 or partial['counts']['initializations_completed']!=38:
        raise ValueError('unexpected durable prefix counters')
    if audit.get('identity_all_conditions_hold') is not True or not audit.get('durable_source_outputs_float64_identical'):
        raise ValueError('independent identity audit did not pass')
    source,target=rows[SOURCE_INDEX],ordered[TARGET_INDEX]
    if source['status']!='completed' or source['diagnostics']['final_numerical'].get('passed') is not True:
        raise ValueError('source must be completed and numerically qualified')
    left,right=ordered[SOURCE_INDEX]['fit'],target['fit']
    if left['model_sha256']!=right['model_sha256'] or not left['model_sha256']:
        raise ValueError('recovery requires the identical frozen model')
    same_four_keys(four_keys(observed[source['input_id']]),four_keys(observed[target['input_id']]))
    # Bind the audit entries to the actual frozen inputs/model, not only its boolean verdict.
    by_index={item['index_zero_based']:item for item in audit['identity_records']}
    for index in (SOURCE_INDEX,TARGET_INDEX):
        entry=by_index[index]
        if entry['model_sha256']!=ordered[index]['fit']['model_sha256']:
            raise ValueError('audit model identity changed')
        same_four_keys(entry['deployment_four_keys'],four_keys(observed[ordered[index]['input_id']]))
    consumed=difference(failed['counts'],partial['counts'])
    if consumed!=audit['exact_consumed_counter_difference']:
        raise ValueError('lost-cost audit changed')
    if consumed['initializations_attempted']!=1 or consumed['initializations_completed']!=1:
        raise ValueError('only one lost completed initialization is recoverable')
    for key in ('fit','generation','qualification'):
        if any(consumed[key].values()):raise ValueError('unrelated work cannot be attributed to the lost pair')
    for key in ('state','future_main','future_fine'):
        value=np.asarray(source[key],dtype=np.float64)
        if not np.isfinite(value).all():raise ValueError('recovery source is nonfinite')
    certification={key:copy.deepcopy(value) for key,value in source['diagnostics']['final_numerical'].items()
                   if key not in ('counts','wall_seconds')}
    recovered=dict(batch_id=target['batch_id'],mode=target['mode'],input_id=target['input_id'],status='completed',
        state=copy.deepcopy(source['state']),future_main=copy.deepcopy(source['future_main']),future_fine=copy.deepcopy(source['future_fine']),
        artifact_recovery='artifact_recovered_by_identical_model_and_input',
        diagnostics=dict(status='completed',stage1=None,stage2=None,stage1_numerical=None,
            final_numerical=certification,counts=copy.deepcopy(consumed['initialization']),wall_seconds=None,
            optimizer_trace=None,returned_point=None,original_diagnostics_status='unknown',
            counts_origin='failed_receipt_minus_durable_partial',numerical_certification_source_index=SOURCE_INDEX),
        recovery_provenance=dict(source_index=SOURCE_INDEX,target_index=TARGET_INDEX,
            model_sha256=left['model_sha256'],consumed_original_counts=consumed,
            optimizer_replayed=False,cost_added_again=False,original_optimizer_trace=None,original_elapsed_seconds=None))
    return recovered,consumed


def load_context(release):
    if release.get('released') is not True:raise PermissionError('root repair release required')
    if release.get('source_row_index')!=SOURCE_INDEX or release.get('target_row_index')!=TARGET_INDEX:
        raise ValueError('release must bind source 37 to target 39')
    manifest_path=OUTPUT/'pre_repair01/manifest.json'
    if original.digest(manifest_path)!=release['frozen_manifest_sha256']:raise ValueError('original snapshot changed')
    manifest=original.read_json(manifest_path)
    for path,sha in manifest['files'].items():
        if original.digest(ROOT/path)!=sha:raise ValueError('original evidence changed: '+path)
    audit_path=OUTPUT/'io_failure_identity_audit.json'
    if original.digest(audit_path)!=AUDIT_SHA256 or release['identity_audit_sha256']!=AUDIT_SHA256:
        raise ValueError('identity audit changed')
    if not set(NEW_SOURCES)<=set(release['source_sha256']):raise ValueError('repair source manifest incomplete')
    for path,sha in release['source_sha256'].items():
        if original.digest(ROOT/path)!=sha:raise ValueError('repair source changed: '+path)
    required_fixed={(OUTPUT/name).relative_to(ROOT).as_posix() for name in
                    ('fits.json','observations.json','observation_plan.json','isolated_truth.json')}
    if not required_fixed<=set(release['fixed_sha256']):raise ValueError('repair release omits frozen inputs or isolated truth hash')
    for path,sha in release['fixed_sha256'].items():
        if original.digest(ROOT/path)!=sha:raise ValueError('repair fixed input changed: '+path)
    old_release=original.read_json(OUTPUT/'execution_release.json')
    if len(old_release['source_sha256'])!=42:raise ValueError('original 42-source manifest changed')
    if original.digest(ROOT/original.PROTOCOL)!=original.PROTOCOL_SHA256:raise ValueError('protocol changed')
    for path,sha in old_release['source_sha256'].items():
        if original.digest(ROOT/path)!=sha:raise ValueError('original source changed: '+path)
    frozen=original.read_json(OUTPUT/'fits_frozen.json')
    original.verify_frozen(OUTPUT,frozen)
    for name,file in (('observations','observations.json'),('plan','observation_plan.json')):
        if original.digest(OUTPUT/file)!=frozen['input_sha256'][name]:raise ValueError('frozen legal inputs changed')
    partial=original.read_json(OUTPUT/'initializations.partial.json')
    failed=original.read_json(OUTPUT/'execution_receipt.json')
    fits=original.read_json(OUTPUT/'fits.json')
    observed=original.read_json(OUTPUT/'observations.json')
    recovered,consumed=recover_lost(partial,failed,fits,observed,original.read_json(audit_path))
    return dict(partial=partial,failed=failed,fits=fits,observed=observed,frozen=frozen,
                recovered=recovered,consumed=consumed,original_hashes=manifest['files'])


def continue_initializations(context,counts):
    rows=copy.deepcopy(context['partial']['rows'])
    rows.append(copy.deepcopy(context['recovered']))
    events=OUTPUT/'resume01_events'
    atomic_json(events/'pair_039.json',dict(index=38,row=rows[-1],counts_after=copy.deepcopy(counts)),exclusive=True)
    for index,entry in enumerate(sequence(context['fits'])):
        if index<39:continue
        path=events/f'pair_{index+1:03d}.json'
        if path.exists() or path.with_name(path.name+'.tmp').exists():raise FileExistsError(path)
        fit=entry['fit']
        model_path=OUTPUT/fit['model_path']
        if original.digest(model_path)!=fit['model_sha256']:raise ValueError('frozen model changed before loading')
        model=original.GFTDKuramotoModel.load(model_path)
        original.compare_saved_model(model,original.read_json(model_path))
        np.testing.assert_array_equal(model.functional.rhs(np.linspace(-.01,.01,20)),fit['rhs_probe'])
        row=dict(batch_id=entry['batch_id'],mode=entry['mode'],input_id=entry['input_id'],state=None,
                 status='blocked_by_fit',diagnostics=None,future_main=None,future_fine=None)
        counts['initializations_attempted']+=1
        try:
            state=model.initialize(four_keys(context['observed'][entry['input_id']]))
            row.update(state=state,status='completed',diagnostics=model.last_initialization)
            for step,name in ((.01,'future_main'),(.005,'future_fine')):
                row[name]=original.integrate(model.functional,state,step,1.,counts['evaluation'])[::int(round(.01/step))]
            counts['initializations_completed']+=1
        except original.engine.InitializationFailure as error:
            row.update(status=error.diagnostics.get('status','failed'),diagnostics=error.diagnostics,error=str(error))
        original.add_counts(counts['initialization'],row['diagnostics'].get('counts',{}))
        rows.append(row)
        atomic_json(path,dict(index=index,row=row,counts_after=copy.deepcopy(counts)),exclusive=True)
    if len(rows)!=48:raise AssertionError('original 48-pair denominator changed')
    original.verify_frozen(OUTPUT,context['frozen'])
    atomic_json(OUTPUT/'initializations.json',dict(rows=rows,counts=copy.deepcopy(counts['initialization']),
        repair='resume01',recovered_ordinal=39,original_failed_receipt_sha256=original.digest(OUTPUT/'execution_receipt.json')),
        exclusive=True)
    sha=original.digest(OUTPUT/'initializations.json')
    atomic_json(OUTPUT/'initializations_frozen.json',dict(initializations_sha256=sha,
        fits_sha256=context['frozen']['fits_sha256'],repair='resume01'),exclusive=True)
    return sha


def main():
    release_path=OUTPUT/'repair_release01.json'
    release=original.read_json(release_path)
    context=load_context(release)
    release_sha=original.digest(release_path)
    atomic_json(OUTPUT/'started01.json',dict(action_id=release['action_id'],parent_action_id=context['failed']['action_id'],
        repair_release_sha256=release_sha,status='RUNNING'),exclusive=True)
    counts=copy.deepcopy(context['failed']['counts'])
    receipt=dict(action_id=release['action_id'],parent_action_id=context['failed']['action_id'],status='FAILED',
        counts=counts,original_lost_consumed_counts=context['consumed'],original_lost_optimizer_trace=None,
        original_lost_elapsed_seconds=None,io_root_cause='unknown',real_data_reads=False,
        repair_release_sha256=release_sha,remaining_optimization_pairs=9,artifact_recovered_pairs=1)
    begin=time.perf_counter()
    try:
        initialized_sha=continue_initializations(context,counts)
        accounting=original.validate_counts(counts,context['fits'])
        atomic_json(OUTPUT/'accounting.json',accounting,exclusive=True)
        summary=original.score_frozen(OUTPUT,context['frozen'],initialized_sha,OUTPUT/'isolated_truth.json',
                                     original.read_json(OUTPUT/'observation_plan.json'))
        atomic_json(OUTPUT/'summary.json',summary,exclusive=True)
        receipt.update(status='COMPLETED',initializations_sha256=initialized_sha,
                       summary_sha256=original.digest(OUTPUT/'summary.json'))
    except Exception as error:
        receipt['error']=repr(error)
        raise
    finally:
        receipt.update(wall_seconds=time.perf_counter()-begin,
            original_evidence_unchanged=all(original.digest(ROOT/path)==sha for path,sha in context['original_hashes'].items()),
            repair_source_unchanged=all(original.digest(ROOT/path)==sha for path,sha in release['source_sha256'].items()),
            repair_release_unchanged=original.digest(release_path)==release_sha,
            original_receipt_preserved=True)
        if receipt['status']=='COMPLETED' and not all(receipt[key] for key in
            ('original_evidence_unchanged','repair_source_unchanged','repair_release_unchanged')):
            receipt['status']='INVALID_FROZEN_INPUT_CHANGED'
        atomic_json(OUTPUT/'resume_execution_receipt.json',receipt,exclusive=True)
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
