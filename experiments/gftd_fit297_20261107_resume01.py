"""IO-only continuation of the exact41 durable Run3 initialization results."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import uuid
import numpy as np
import gftd_fit297_20261107 as original
from kiki.models.gftd_kuramoto import GFTDKuramotoModel
from kiki.models import gftd_fitting as engine, gftd_fit297 as fit_engine
ROOT=original.ROOT
PARENT_OUTPUT=original.OUTPUT
PARENT_PRIVATE=original.PRIVATE
OUTPUT=PARENT_OUTPUT/'resume01'
PRIVATE=PARENT_PRIVATE/'resume01'
EXPECTED_LEDGER=dict(information_svd=416,conditional_projections=315,memory_svd=594)
EXPECTED_INIT=dict(optimization_tasks=49,actual_residual_calls=9347,rk4_steps_attempted=396020,rk4_steps_completed=396020)
REPLACE_ATTEMPTS=100
REPLACE_DELAY=.05
REVIEW_PATH='reports/gftd_fit297_20261107_resume01_design_review.md'
REVIEW_SHA256='3f16b5a11a5f7d44110fc2c472a51538a2f03278023efbb94c2ec64e8e21d5a3'
ADOPTION_PATH='results/gftd_fit297_20261107/resume01_implementation_adoption.json'
ADOPTION_SHA256='505044e2a6d86b4048f70afa6c1bad19276d76edf36a2d20787355515f2d6c5a'
HOST_PATH='results/gftd_fit297_20261107/execution_host_receipt.json'
HOST_SHA256='78ab5d6594c29ebe143f7bdef9eb02fc13b6bbc8285a8b5c75c645aa87f78c6a' 


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    with temp.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(temp,path)
            break
        except PermissionError:
            if attempt+1==REPLACE_ATTEMPTS:raise
            time.sleep(REPLACE_DELAY)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=original.digest(path),bytes=len(data))


def write_json(path,value):
    return atomic_bytes(path,(json.dumps(original.safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))


def write_npz(path,**arrays):
    import io
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    return atomic_bytes(path,stream.getvalue())


def verify_release(release):
    if release.get('released') is not True:raise PermissionError('exact root resume01 release required')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',release.get('action_id','')):raise ValueError('explicit action_id required')
    if release.get('parent_job_ended') is not True:raise ValueError('parent termination must be reconciled')
    if release.get('environment')!=original.environment():raise ValueError('runtime environment changed')
    for field,name,sha in (('design_review_sha256',REVIEW_PATH,REVIEW_SHA256),('implementation_adoption_sha256',ADOPTION_PATH,ADOPTION_SHA256),('parent_host_receipt_sha256',HOST_PATH,HOST_SHA256)):
        if release.get(field)!=sha or original.digest(ROOT/name)!=sha:raise ValueError('repair authorization binding changed: '+field)
    parent=original.read_json(PARENT_OUTPUT/'execution_release.json')
    if parent.get('released') is not True:raise ValueError('parent was not released')
    if release['action_id']==parent['action_id']:raise ValueError('repair action must have a distinct identity')
    required=[PARENT_OUTPUT/name for name in ('execution_release.json','started.json','execution.partial.json','execution.partial.json.tmp')]
    required += [PARENT_PRIVATE/name for name in ('inputs.json','preparation.json','constants.json','static_model.json','modulated_model.json','static_fit_details.json','modulated_fit_details.json')]
    required += sorted((PARENT_PRIVATE/'initializations').glob('*'))
    assets=release.get('parent_artifact_sha256',{})
    if not {p.relative_to(ROOT).as_posix() for p in required if p.is_file()}<=set(assets):raise ValueError('release omits parent artifacts')
    inherited_assets=dict(parent['asset_sha256'])
    inherited_assets.update({original.PROTOCOL:parent['protocol_sha256'],original.REVIEW:parent['design_review_sha256'],
        (PARENT_OUTPUT/'design_adoption.json').relative_to(ROOT).as_posix():parent['design_adoption_sha256'],
        (PARENT_OUTPUT/'parent_archive_verified.json').relative_to(ROOT).as_posix():parent['parent_archive_sha256']})
    if any(assets.get(name)!=sha for name,sha in inherited_assets.items()):raise ValueError('original scientific asset binding omitted or changed')
    sources=release.get('source_sha256',{})
    for name,sha in parent['source_sha256'].items():
        if sources.get(name)!=sha:raise ValueError('parent source binding changed: '+name)
    for name in ('experiments/gftd_fit297_20261107_resume01.py','tests/test_gftd_fit297_resume01.py'):
        if name not in sources:raise ValueError('repair source binding missing')
    for mapping in (assets,sources):
        for name,sha in mapping.items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha):raise ValueError('invalid binding')
            if Path(name).name!='fit_events.npz' and original.digest(path)!=sha:raise ValueError('bound bytes changed: '+name)
    if release.get('parent_release_sha256')!=original.digest(PARENT_OUTPUT/'execution_release.json'):raise ValueError('parent release mismatch')
    return parent


def restore(release):
    """Stored outputs only: no fitting, reconstruction, integration or SVD."""
    latest=original.read_json(PARENT_OUTPUT/'execution.partial.json.tmp')
    inputs=original.read_json(PARENT_PRIVATE/'inputs.json')
    preparation=original.read_json(PARENT_PRIVATE/'preparation.json')
    constants=original.read_json(PARENT_PRIVATE/'constants.json')
    ids=inputs['receipt']['legal_ids'];packets=inputs['packets']
    if len(ids)!=297 or ids!=sorted(set(ids)) or len(packets)!=297 or set(ids)!=set(preparation['fit_ids']):raise ValueError('ordered fit297 identity changed')
    if inputs['receipt']['invalid']:raise ValueError('repair requires original297 legal packets')
    for packet in packets:
        if set(packet)!={'theta_history','omega_history','history_times','p0','sigma'} or packet['sigma']!=0.:raise ValueError('original causal packet changed')
        original.core.validate_observation({('p_end' if k=='p0' else k):v for k,v in packet.items()},10)
    models={mode:GFTDKuramotoModel.load(PARENT_PRIVATE/(mode+'_model.json')) for mode in original.MODES}
    keys={m.inference_config['preparation_key'] for m in models.values()}
    if len(keys)!=1:raise ValueError('preparation identity differs')
    for mode,model in models.items():
        if model.fit_state.get('preparation_key')!=model.inference_config['preparation_key']:raise ValueError('checkpoint preparation keys differ')
        if model.fit_state['counts']!=latest['counts']['fit'][mode]:raise ValueError('fit counts changed')
        if model.fit_state['adaptation_indices']!=inputs['receipt']['adaptation_indices']:raise ValueError('adaptation changed')
    states={mode:[None]*297 for mode in original.MODES};done={};summed={};success=0
    files=sorted((PARENT_PRIVATE/'initializations').glob('*.json'))
    if [p.name for p in files]!=[f'static_{i:03d}.json' for i in range(41)]:raise ValueError('exact41 boundary required')
    for i,path in enumerate(files):
        row=original.read_json(path)
        if row['event_id']!=ids[i]:raise ValueError('result identity changed')
        diag=row['diagnostics']
        for k,v in diag.get('counts',{}).items():summed[k]=summed.get(k,0)+v
        artifact=row['state']
        if artifact is not None:
            state_path=(ROOT/artifact['path']).resolve()
            if not state_path.is_relative_to(PARENT_PRIVATE.resolve()) or original.digest(state_path)!=artifact['sha256']:raise ValueError('stored state changed')
            with np.load(state_path,allow_pickle=False) as data:
                if set(data.files)!={'state'} or data['state'].dtype!=np.float64:raise ValueError('unexpected state keys or precision')
                state=np.asarray(data['state'],dtype=np.float64).copy()
            if diag['status']!='completed' or state.shape!=(50,) or not np.isfinite(state).all():raise ValueError('invalid stored state')
            if not np.array_equal(state[20:30],np.asarray(packets[i]['p0'])):raise ValueError('p0 anchor changed')
            states['static'][i]=state;success+=1
        elif diag['status']=='completed':raise ValueError('completed result missing state')
        done[('static',i)]=row
    if success!=8 or latest['counts']['initializations_attempted']!=41 or latest['counts']['initializations_completed']!=8:raise ValueError('boundary mismatch')
    if summed!=latest['counts']['initialization'] or any(summed.get(k)!=v for k,v in EXPECTED_INIT.items()):raise ValueError('all41 costs must reconcile')
    ledger=models['modulated'].fit_state['analysis_budget']
    for k,v in EXPECTED_LEDGER.items():
        if ledger[k]['actual']!=v or sum(m.fit_state['counts'].get(k,0) for m in models.values())!=v:raise ValueError('information ledger mismatch')
    if any(latest['counts'][k] for k in ('future_pairs_attempted','scoring_attempted')) or latest['counts']['coverage']:raise ValueError('pre-future boundary changed')
    if fit_engine._CACHE or fit_engine._PREPARATION_FAILURES or fit_engine._LEDGERS:raise ValueError('fresh process required')
    key=next(iter(keys));fit_engine._LEDGERS[key]=EXPECTED_LEDGER.copy()
    if latest['counts']['fits_attempted']!=2 or latest['counts']['fits_completed']!=2:raise ValueError('both fits must already be complete')
    remaining=[(mode,i) for mode in original.MODES for i in range(297) if (mode,i) not in done]
    if len(remaining)!=553:raise ValueError('remaining work changed')
    return dict(latest=latest,inputs=inputs,preparation=preparation,constants=constants,models=models,states=states,
        done=done,remaining=remaining,packets=packets,ids=ids,preparation_key=key)


def run(release_path=None):
    release_path=Path(release_path or OUTPUT/'execution_release.json')
    release=original.read_json(release_path);parent_release=verify_release(release)
    OUTPUT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    claim=dict(action_id=release['action_id'],parent_action_id=parent_release['action_id'],release_sha256=original.digest(release_path),status='RUNNING')
    with (OUTPUT/'started.json').open('x',encoding='utf-8') as stream:
        json.dump(claim,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    receipt=dict(claim,denominators=original.l1.DENOMINATORS.copy(),current={'stage':'restore'},artifacts=[],costs={},analysis={},
        counts=dict(fit={},initialization={},coverage={},rk4_attempted=0,rk4_completed=0),status='RUNNING')
    begin=time.perf_counter();primary_error=None
    def persist():write_json(OUTPUT/'execution.partial.json',receipt)
    def at(stage,mode=None,event_id=None):receipt['current']=dict(stage=stage,mode=mode,event_id=event_id);persist()
    try:
        saved=restore(release)
        models,states=saved['models'],saved['states'];packets=saved['packets'];ids=saved['ids']
        input_receipt=saved['inputs']['receipt'];scales=saved['constants']['scales'];template=models['static']
        receipt.update(counts=copy.deepcopy(saved['latest']['counts']),costs=copy.deepcopy(saved['latest']['costs']),
            artifacts=copy.deepcopy(saved['latest']['artifacts']),restored_initializations=41,remaining_initializations=553,
            inherited_receipt_sha256=original.digest(PARENT_OUTPUT/'execution.partial.json.tmp'),input_identity=copy.deepcopy(saved['latest'].get('input_identity')))
        counts=receipt['counts'];failures={}
        inherited_init=copy.deepcopy(counts['initialization'])
        for (mode,index),row in saved['done'].items():
            if row['state'] is None:failures[(mode,ids[index])]=dict(stage='initialization',status=row['diagnostics']['status'],restored=True)
        persist()
        for mode,index in saved['remaining']:
            model=models[mode];eid=ids[index];packet=packets[index]
            at('initialize',mode,eid);counts['initializations_attempted']+=1;persist();tick=time.perf_counter();diag=None;state=None
            try:
                state=model.initialize({k:v for k,v in packet.items() if k!='sigma'})
                diag=model.last_initialization or {};counts['initializations_completed']+=1
            except engine.InitializationFailure as error:
                diag=error.diagnostics;failures[(mode,eid)]=dict(stage='initialization',type=type(error).__name__,message=str(error))
            finally:
                if diag is not None:
                    for k,v in diag.get('counts',{}).items():
                        if isinstance(v,(int,float)) and not isinstance(v,bool):counts['initialization'][k]=counts['initialization'].get(k,0)+v
                receipt['costs'][mode]['initialization_seconds']+=time.perf_counter()-tick
            new_tasks=counts['initialization'].get('optimization_tasks',0)-inherited_init.get('optimization_tasks',0)
            new_calls=counts['initialization'].get('actual_residual_calls',0)-inherited_init.get('actual_residual_calls',0)
            receipt['remaining_work_budget']=dict(new_tasks=new_tasks,new_task_cap=1106,new_residual_calls=new_calls,new_residual_cap=995400)
            if new_tasks>1106 or new_calls>995400:raise RuntimeError('remaining original opportunities exceeded')
            states[mode][index]=state
            artifact=None if state is None else write_npz(PRIVATE/'initializations'/f'{mode}_{index:03d}.npz',state=state)
            detail=write_json(PRIVATE/'initializations'/f'{mode}_{index:03d}.json',dict(event_id=eid,diagnostics=diag,state=artifact))
            receipt['artifacts'].append(detail);persist()
        receipt['artifacts'].append(write_json(PRIVATE/'all_initializations_frozen.json',dict(
            models={mode:original.digest(PARENT_PRIVATE/(mode+'_model.json')) for mode in models},counts=counts,
            legal_ids=ids,success={mode:[s is not None for s in states[mode]] for mode in states},restored_initializations=41)))
        validated=[original.core.validate_observation({('p_end' if k=='p0' else k):v for k,v in p.items()},10) for p in packets]
        for mode,model in models.items():
            at('coverage_information',mode)
            coverage=fit_engine.coverage_information(model.functional,validated,states[mode],model.inference_config)
            counts['coverage'][mode]=coverage.get('counts',{})
            if coverage.get('status')!='unknown' or coverage.get('reason')!='matching preparation cache unavailable' or any(counts['coverage'][mode].values()):raise ValueError('lost cache must not be rebuilt or consumed')
            coverage['interruption_provenance']=dict(parent_action_id=parent_release['action_id'],cache_durable=False,cache_rebuilt=False,
                scope_impact='deployment coverage information unknown; original fit information retained')
            receipt.setdefault('coverage_information',{})[mode]=dict(status=coverage['status'],reason=coverage['reason'],interruption_provenance=coverage['interruption_provenance'])
            receipt['artifacts'].append(write_json(PRIVATE/(mode+'_coverage_information.json'),coverage))
        import closure_probe_20261030 as loader
        at('load_frozen_train_for_scoring')
        for name,sha in release['parent_artifact_sha256'].items():
            if Path(name).name=='fit_events.npz' and original.digest(ROOT/name)!=sha:raise ValueError('parent fit cache bytes changed')
        preparation,events=loader.load_frozen_events()
        if preparation!=saved['preparation'] or {str(e['id']) for e in events}!=set(ids):raise ValueError('scoring identity changed')
        by_id={str(e['id']):e for e in events}
        records={mode:[] for mode in original.MODES}
        for mode in original.MODES:
            model=models.get(mode)
            for index,eid in enumerate(input_receipt['ordered_all_ids']):
                event=by_id[eid]
                if model is None or eid not in input_receipt['legal_ids']:
                    records[mode].append(original.l1.failure_event(event,'fit' if model is None else 'input',RuntimeError(str(failures.get(mode,'illegal input')))))
                    continue
                pos=input_receipt['legal_ids'].index(eid);state=states[mode][pos]
                if state is None:
                    records[mode].append(original.l1.failure_event(event,'initialization',RuntimeError(str(failures[(mode,eid)]))))
                    continue
                at('future_pair',mode,eid);counts['future_pairs_attempted']+=1;persist();tick=time.perf_counter()
                coarse,ce=original.integrate(model.functional,state,.01,counts)
                fine,fe=original.integrate(model.functional,state,.005,counts)
                receipt['costs'][mode]['integration_seconds']+=time.perf_counter()-tick
                counts['future_pairs_completed' if ce is None and fe is None else 'future_pairs_failed']+=1
                arrays=write_npz(PRIVATE/'predictions'/f'{mode}_{index:03d}.npz',coarse=coarse,fine=fine)
                receipt['artifacts'].append(arrays);counts['future_pairs_persisted']+=1
                at('score',mode,eid);counts['scoring_attempted']+=1;persist()
                record=original.l1.score_event(event,fine,coarse,delta_star=model.delta_star,mass=model.mass,p_scale=scales['p_scale'])
                record['integration_errors']=[e for e in (ce,fe) if e is not None]
                record['trajectory']=arrays
                records[mode].append(record);counts['scoring_completed']+=1
                receipt['artifacts'].append(write_json(OUTPUT/'events'/f'{mode}_{index:03d}.json',record));persist()
        for mode in original.MODES:
            for index,record in enumerate(records[mode]):
                if record.get('failure') is not None:
                    receipt['artifacts'].append(write_json(OUTPUT/'events'/f'{mode}_{index:03d}.json',record))
        paired=original.l1.paired_summary(records['static'],records['modulated'])
        receipt['artifacts'].append(write_json(OUTPUT/'l1_summary.json',paired))
        at('model_analysis')
        import gftd_fit297_analysis_20261107 as analysis
        for mode,model in models.items():
            certificate=analysis.l2_certificate(model.functional,model.delta_star)
            receipt['analysis'][mode]=dict(certificate_status=certificate.get('status'),
                l2_radius_candidates_actual=len(certificate.get('candidates',[])),l2_point_calls_actual=0)
            points=[]
            for state in states[mode]:
                if state is None:
                    points.append(None)
                else:
                    point=analysis.l2_point(certificate,model.functional,state)
                    receipt['analysis'][mode]['l2_point_calls_actual']+=1
                    points.append(point)
            modes=analysis.rom_modes(model.functional,model.delta_star,certificate)
            receipt['analysis'][mode].update(point_count=len(points),mode_status=modes.get('status'),rom_counts=modes.get('counters'))
            receipt['artifacts'].append(write_json(PRIVATE/(mode+'_l2_l3.json'),dict(certificate=certificate,points=points,modes=modes)))
        if models:
            reference=analysis.kron_reference(PARENT_PRIVATE/'reference/ieee39_full.xlsx')
            receipt['analysis']['kron_counts']={k:reference.get(k) for k in ('svd_count','solve_count')}
            receipt['artifacts'].append(write_json(PRIVATE/'kron_reference.json',reference))
            identity=parent_release.get('fom_expected_identity',dict(equilibrium_identity_verified=False))
            fom=analysis.fom_modes(ROOT/'private_data/task2_fullorder_descriptor_0035/descriptor.npz',
                ROOT/'private_data/task2_fullorder_descriptor_0035/evidence.json',ROOT/'private_data/task2_verified_modes_0035b/evidence.json',
                template.mass,expected_identity=identity)
            receipt['analysis']['fom_qz_actual']=fom.get('qz_count',fom.get('checks',{}).get('qz_count'))
            receipt['artifacts'].append(write_json(PRIVATE/'fom_modes.json',fom))
            for mode,model in models.items():
                content=original.read_json(PRIVATE/(mode+'_l2_l3.json'))
                matched=analysis.match_modes(fom,content['modes'])
                receipt['artifacts'].append(write_json(PRIVATE/(mode+'_modal_match.json'),matched))
            for mode,model in models.items():
                correspondence=analysis.graph_correspondence(reference,model.functional,model.delta_star)
                receipt['artifacts'].append(write_json(PRIVATE/(mode+'_graph_correspondence.json'),correspondence))
            key=next(iter(models.values())).fit_state.get('preparation_key')
            receipt['analysis_budget']=fit_engine.analysis_budget_snapshot(key)
        checks=original.model_analysis_budget_checks(receipt['analysis'],models)
        receipt['analysis_budget_checks']=checks
        if any(c['within_limit'] is False for c in checks): raise RuntimeError('registered model analysis budget exceeded')
        receipt['execution_budget']=original.execution_budget(counts)
        if not receipt['execution_budget']['within_limits']: raise RuntimeError('registered execution budget exceeded')
        receipt['status']='COMPLETED'
    except BaseException as error:
        primary_error=error
        receipt.update(status='INTERRUPTED_RUNTIME',error=dict(type=type(error).__name__,message=str(error)),traceback=traceback.format_exc())
        raise
    finally:
        receipt['execution_budget']=original.execution_budget(receipt['counts'])
        receipt['resume_wall_seconds']=time.perf_counter()-begin
        receipt['original_runner_wall_seconds']=None
        receipt['parent_host_seconds']=release.get('parent_host_seconds')
        receipt['combined_host_seconds']=None
        receipt['source_unchanged']=all(original.digest(ROOT/p)==sha for p,sha in release['source_sha256'].items())
        receipt['parent_artifacts_unchanged']=all(original.digest(ROOT/p)==sha for p,sha in release['parent_artifact_sha256'].items())
        receipt['release_unchanged']=original.digest(release_path)==claim['release_sha256']
        receipt['evidence_binding_valid']=all(receipt[k] for k in ('source_unchanged','parent_artifacts_unchanged','release_unchanged'))
        if not receipt['evidence_binding_valid']:receipt['status']='INTERRUPTED_SOURCE_CHANGED'
        for cost in receipt['costs'].values():
            cost['deployment_total_seconds']=cost['initialization_seconds']+cost['integration_seconds']
            cost['same_precision_conditions']=dict(device='cpu',dtype='float64',future_steps=[.01,.005],duration_s=15.,batch=1)
        receipt['FOM_speedup']='not_compared'
        try:
            persist();write_json(OUTPUT/'execution.json',receipt)
        except Exception as error:
            receipt['final_persistence_error']=dict(type=type(error).__name__,message=str(error))
            receipt['status_before_final_persistence_failure']=receipt['status']
            receipt['status']='INTERRUPTED_RUNTIME'
            emergency=OUTPUT/('emergency-'+uuid.uuid4().hex+'.json')
            try:
                with emergency.open('xb') as stream:
                    stream.write((json.dumps(original.safe(receipt),ensure_ascii=False,indent=2)+'\n').encode('utf-8'));stream.flush();os.fsync(stream.fileno())
            except Exception as emergency_error:
                if primary_error is not None:primary_error.add_note('emergency receipt failed: '+repr(emergency_error))
                else:error.add_note('emergency receipt failed: '+repr(emergency_error))
            if primary_error is None:raise
            primary_error.add_note('final persistence failed; emergency receipt: '+str(emergency)+'; '+repr(error))
    return receipt


if __name__=='__main__':run()
