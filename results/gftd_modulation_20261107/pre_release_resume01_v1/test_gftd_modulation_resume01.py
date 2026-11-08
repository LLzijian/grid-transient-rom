"""Mock technical recovery only; no original action is resumed by these tests."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import gftd_modulation_20261107_resume01 as run


def ledger(n):
    return dict(fits_attempted=16,fits_completed=16,initializations_attempted=n,initializations_completed=n,
        fit={},generation={},qualification={},initialization={'optimization_tasks':2*n,'actual_residual_calls':21*n},
        evaluation={'integrations_attempted':2*n,'integrations_completed':2*n})


def fixture_data(tmp_path):
    model_path=tmp_path/'model.json';model_path.write_text('{}')
    model_sha=run.original.digest(model_path)
    observed={}
    fits={'records':[]}
    rows=[]
    for fit_index in range(16):
        keys=[]
        for event in range(3):
            index=3*fit_index+event
            key=f'obs_{index:03d}';keys.append(key)
            observed[key]=dict(theta_history=np.zeros((11,4)),omega_history=np.zeros((11,4)),
                history_times=np.linspace(-.1,0.,11),p0=np.zeros(4),sigma=0.)
            rows.append(dict(batch_id=fit_index//2,mode=('static','modulated')[fit_index%2],input_id=key,
                status='completed',state=np.zeros(20).tolist(),future_main=np.zeros((101,20)).tolist(),
                future_fine=np.zeros((101,20)).tolist(),diagnostics={'final_numerical':{'passed':True,'wall_seconds':.3,'counts':{'unused_source_cost':7}},
                    'stage1':{'private_trace':'do not copy'},'stage2':{},'counts':{'optimization_tasks':2,'actual_residual_calls':21},'wall_seconds':99.}))
        fits['records'].append(dict(batch_id=fit_index//2,mode=('static','modulated')[fit_index%2],packet_ids=keys,
            model_path='model.json',model_sha256=model_sha,rhs_probe=np.zeros(20).tolist()))
    partial={'rows':rows[:38],'counts':ledger(38)}
    failed={'status':'FAILED','counts':ledger(39),'action_id':'original'}
    run.atomic_json(tmp_path/'execution_receipt.json',failed)
    audit=dict(identity_all_conditions_hold=True,durable_source_outputs_float64_identical=True,
        exact_consumed_counter_difference=run.difference(failed['counts'],partial['counts']),
        identity_records=[dict(index_zero_based=index,model_sha256=model_sha,
            deployment_four_keys=run.four_keys(observed[rows[index]['input_id']])) for index in (36,37,38)])
    recovered,consumed=run.recover_lost(partial,failed,fits,observed,audit)
    return dict(partial=partial,failed=failed,fits=fits,observed=observed,audit=audit,recovered=recovered,
                consumed=consumed,frozen={'fits_sha256':'frozen'},original_hashes={})


@pytest.fixture
def mocks(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    context=fixture_data(tmp_path)
    calls=[]
    class Model:
        functional=SimpleNamespace(rhs=lambda state:np.zeros(20))
        def initialize(self,packet):
            assert set(packet)=={'theta_history','omega_history','history_times','p0'}
            calls.append('initialize')
            self.last_initialization={'status':'completed','counts':{'optimization_tasks':2,'actual_residual_calls':1},'final_numerical':{'passed':True}}
            return np.zeros(20)
    monkeypatch.setattr(run.original,'GFTDKuramotoModel',SimpleNamespace(load=lambda path:Model()))
    monkeypatch.setattr(run.original,'compare_saved_model',lambda *args:None)
    monkeypatch.setattr(run.original,'verify_frozen',lambda *args:None)
    def integrate(f,state,step,duration,counts):
        counts['integrations_attempted']+=1;counts['integrations_completed']+=1
        return np.zeros((int(round(1/step))+1,20))
    monkeypatch.setattr(run.original,'integrate',integrate)
    return context,calls,Model


def test_recovered_row_keeps_lost_trace_unknown_and_cost_not_double_counted(mocks,tmp_path):
    context,calls,_=mocks
    recovered=context['recovered']
    assert recovered['diagnostics']['stage1'] is None
    assert recovered['diagnostics']['stage2'] is None
    assert recovered['diagnostics']['wall_seconds'] is None
    assert recovered['diagnostics']['counts']['actual_residual_calls']==21
    assert 'wall_seconds' not in recovered['diagnostics']['final_numerical']
    assert 'counts' not in recovered['diagnostics']['final_numerical']
    counts=copy.deepcopy(context['failed']['counts'])
    sha=run.continue_initializations(context,counts)
    assert len(calls)==9
    assert counts['initializations_attempted']==counts['initializations_completed']==48
    assert counts['initialization']['actual_residual_calls']==39*21+9
    rows=run.original.read_json(tmp_path/'initializations.json')['rows']
    assert len(rows)==48 and rows[38]['artifact_recovery']=='artifact_recovered_by_identical_model_and_input'
    assert len(list((tmp_path/'resume01_events').glob('*.json')))==10
    assert run.original.read_json(tmp_path/'initializations_frozen.json')['initializations_sha256']==sha
    event=run.original.read_json(tmp_path/'resume01_events/pair_039.json')
    assert event['counts_after']==context['failed']['counts']


@pytest.mark.parametrize('bad',['signed_zero','model','numerical','audit','cost','order'])
def test_identity_or_cost_mismatch_stops_without_replay(mocks,bad):
    context,calls,_=mocks
    if bad=='signed_zero':context['observed']['obs_038']['theta_history'][0,0]=-0.
    if bad=='model':context['fits']['records'][12]['model_sha256']='wrong'
    if bad=='numerical':context['partial']['rows'][36]['diagnostics']['final_numerical']['passed']=False
    if bad=='audit':context['audit']['identity_all_conditions_hold']=False
    if bad=='cost':context['failed']['counts']['initialization']['actual_residual_calls']+=1
    if bad=='order':context['partial']['rows'][2]['input_id']='wrong'
    with pytest.raises(ValueError):run.recover_lost(context['partial'],context['failed'],context['fits'],context['observed'],context['audit'])
    assert calls==[]


def test_event_io_error_never_retries_optimization(mocks,monkeypatch,tmp_path):
    context,calls,_=mocks
    real=run.atomic_json
    def fail(path,*args,**kwargs):
        if Path(path).name=='pair_040.json':raise OSError(22,'mock write failed')
        return real(path,*args,**kwargs)
    monkeypatch.setattr(run,'atomic_json',fail)
    counts=copy.deepcopy(context['failed']['counts'])
    with pytest.raises(OSError):run.continue_initializations(context,counts)
    assert calls==['initialize']
    assert counts['initializations_attempted']==40
    assert (tmp_path/'resume01_events/pair_039.json').exists()
    assert not (tmp_path/'initializations.json').exists()


def test_typed_failure_retains_all_48_rows(mocks,monkeypatch,tmp_path):
    context,calls,Model=mocks
    def fail(self,packet):
        calls.append('failure')
        raise run.original.engine.InitializationFailure('blocked',{'status':'blocked_by_stage1','counts':{'optimization_tasks':1,'actual_residual_calls':2}})
    monkeypatch.setattr(Model,'initialize',fail)
    counts=copy.deepcopy(context['failed']['counts'])
    run.continue_initializations(context,counts)
    rows=run.original.read_json(tmp_path/'initializations.json')['rows']
    assert len(rows)==48 and len(calls)==9
    assert sum(row['status']=='blocked_by_stage1' for row in rows)==9
    assert counts['initializations_attempted']==48 and counts['initializations_completed']==39


def test_atomic_exclusive_does_not_overwrite_or_replay(tmp_path):
    path=tmp_path/'event.json'
    run.atomic_json(path,{'first':True},exclusive=True)
    with pytest.raises(FileExistsError):run.atomic_json(path,{'second':True},exclusive=True)
    assert json.loads(path.read_text())=={'first':True}


def test_main_freezes_all_outputs_before_scoring_and_preserves_receipt(mocks,monkeypatch,tmp_path):
    context,calls,_=mocks
    release={'released':True,'action_id':'repair-only','source_sha256':{}}
    run.atomic_json(tmp_path/'repair_release01.json',release)
    run.atomic_json(tmp_path/'execution_receipt.json',{'original':'do not overwrite'})
    old=(tmp_path/'execution_receipt.json').read_bytes()
    monkeypatch.setattr(run,'load_context',lambda release:context)
    monkeypatch.setattr(run.original,'validate_counts',lambda counts,fits:{'pairs':counts['initializations_attempted']})
    run.atomic_json(tmp_path/'observation_plan.json',[])
    def score(output,frozen,sha,*args):
        assert len(calls)==9
        assert run.original.digest(output/'initializations.json')==sha
        assert (output/'initializations_frozen.json').exists()
        return {'prediction_denominator':48}
    monkeypatch.setattr(run.original,'score_frozen',score)
    run.main()
    assert (tmp_path/'execution_receipt.json').read_bytes()==old
    receipt=run.original.read_json(tmp_path/'resume_execution_receipt.json')
    assert receipt['status']=='COMPLETED' and receipt['counts']['initializations_attempted']==48
    with pytest.raises(FileExistsError):run.main()
    assert len(calls)==9


def test_release_gates_validate_frozen_hashes_before_any_continuation(mocks,monkeypatch,tmp_path):
    context,calls,_=mocks
    monkeypatch.setattr(run,'ROOT',tmp_path)
    sources={}
    for index in range(42):
        path=f'sources/source_{index}.txt'
        target=tmp_path/path;target.parent.mkdir(exist_ok=True);target.write_text('frozen')
        sources[path]=run.original.digest(target)
    old_release={'source_sha256':sources}
    run.atomic_json(tmp_path/'execution_release.json',old_release)
    run.atomic_json(tmp_path/'fits.json',context['fits'])
    run.atomic_json(tmp_path/'observations.json',context['observed'])
    run.atomic_json(tmp_path/'observation_plan.json',[])
    run.atomic_json(tmp_path/'isolated_truth.json',{'not_read_before_scoring':True})
    run.atomic_json(tmp_path/'initializations.partial.json',context['partial'])
    frozen=dict(context['frozen'],input_sha256=dict(observations=run.original.digest(tmp_path/'observations.json'),
                                                   plan=run.original.digest(tmp_path/'observation_plan.json')))
    run.atomic_json(tmp_path/'fits_frozen.json',frozen)
    run.atomic_json(tmp_path/'io_failure_identity_audit.json',context['audit'])
    monkeypatch.setattr(run,'AUDIT_SHA256',run.original.digest(tmp_path/'io_failure_identity_audit.json'))
    protocol=tmp_path/run.original.PROTOCOL;protocol.parent.mkdir(exist_ok=True);protocol.write_text('protocol')
    monkeypatch.setattr(run.original,'PROTOCOL_SHA256',run.original.digest(protocol))
    manifest={'files':sources}
    run.atomic_json(tmp_path/'pre_repair01/manifest.json',manifest)
    new_sources={}
    for path in run.NEW_SOURCES:
        target=tmp_path/path;target.parent.mkdir(exist_ok=True);target.write_text('repair candidate')
        new_sources[path]=run.original.digest(target)
    fixed={name:run.original.digest(tmp_path/name) for name in ('fits.json','observations.json','observation_plan.json','isolated_truth.json')}
    release=dict(released=True,source_row_index=36,target_row_index=38,
        frozen_manifest_sha256=run.original.digest(tmp_path/'pre_repair01/manifest.json'),
        identity_audit_sha256=run.AUDIT_SHA256,source_sha256=new_sources,fixed_sha256=fixed)
    loaded=run.load_context(release)
    assert loaded['recovered']['input_id']=='obs_038' and calls==[]
    bad=copy.deepcopy(release);bad['released']=False
    with pytest.raises(PermissionError):run.load_context(bad)
    bad=copy.deepcopy(release);del bad['fixed_sha256']['isolated_truth.json']
    with pytest.raises(ValueError,match='omits'):run.load_context(bad)
    (tmp_path/'sources/source_0.txt').write_text('tampered')
    with pytest.raises(ValueError,match='evidence changed'):run.load_context(release)
    assert calls==[]
