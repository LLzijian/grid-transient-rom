"""Mock Run2 scheduling only: no registered generation, seed or optimizer is called."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import gftd_modulation_20261107 as run


def legal_inputs():
    observed,plan = {},[]
    for family in run.FAMILIES:
        for sigma in run.SIGMAS:
            for event in range(3):
                key=f'obs_{len(plan):03d}'
                observed[key]=dict(theta_history=np.full((11,4),len(plan)*1e-4),omega_history=np.zeros((11,4)),
                    history_times=np.linspace(-.1,0.,11),p0=np.zeros(4),sigma=sigma)
                plan.append(dict(input_id=key,family=family,sigma=sigma,event=event,trajectory=0))
    return observed,plan


def counts():
    return dict(fits_attempted=0,fits_completed=0,initializations_attempted=0,initializations_completed=0,
        fit={},initialization={},evaluation={},qualification={},generation=dict(generated_history_grids=24,
        generated_future_grids=24,observation_packages=24))


@pytest.fixture
def fake_models(monkeypatch):
    trace=[]
    cache={}
    monkeypatch.setattr(run.engine,'_BATCH_CACHE',cache)
    monkeypatch.setattr(run.engine,'validate_config',lambda *args: None)
    class Model(run.GFTDKuramotoModel):
        def fit(self,train,val,context):
            assert not val and len(train)==3
            assert all(set(item)=={'theta_history','omega_history','history_times','p0','sigma'} for item in train)
            key=str(float(train[0]['theta_history'][0][0]))
            hit=key in cache
            cache[key]=True
            trace.append(('fit',self.fit_mode,key))
            self.initialization_mode='causal_two_stage'
            self.inference_config=dict(sigma=float(train[0]['sigma']))
            self.fit_state=dict(status='not_identified',counts=dict(algebra_terms_actual=0,
                algebra_terms_planned=0 if hit else 287,algebra_terms_skipped=0 if hit else 287,nnls_attempts=0,reconstructions=0 if hit else 6),
                preparation_key=key,preparation_cache_hit=hit,shared_preparation_counts={'reconstructions':6},
                initial_static={'raw_k':None,'unused_algebra_items':{'count':157}},initial_information={},static_enumeration_references={
                    '0.0025':{'enumeration':[],'unused_algebra_items':{'count':65}},'0.00125':{'enumeration':[],'unused_algebra_items':{'count':65}}},
                support=None,modulation_support=None,k=self.functional.k_edge.tolist(),beta=[0.]*6)
            return SimpleNamespace(status='not_identified',best_validation_loss=None,val_case_ids=())
        def initialize(self,packet):
            assert len([item for item in trace if item[0]=='fit'])==16
            assert set(packet)=={'theta_history','omega_history','history_times','p0'}
            trace.append(('initialize',self.fit_mode))
            self.last_initialization=dict(status='completed',counts={'actual_residual_calls':2},
                stage1={'status':'converged'},stage2={'status':'converged'},final_numerical={'passed':True})
            return self.functional.pack_state(np.asarray(packet['theta_history'])[-1],np.asarray(packet['omega_history'])[-1],
                        packet['p0'],np.zeros(4),np.zeros(4))
    monkeypatch.setattr(run,'GFTDKuramotoModel',Model)
    def integration(f,state,step,duration,ledger,**kwargs):
        ledger['integrations_attempted']=ledger.get('integrations_attempted',0)+1
        ledger['integrations_completed']=ledger.get('integrations_completed',0)+1
        return np.tile(state,(int(round(duration/step))+1,1))
    monkeypatch.setattr(run,'integrate',integration)
    return trace


def test_two_methods_freeze_all_before_exactly_48_loaded_initializations(fake_models,tmp_path):
    observed,plan=legal_inputs()
    ledger=counts()
    frozen=run.fit_B(observed,plan,tmp_path,ledger)
    assert len(fake_models)==16
    assert len(frozen['models'])==16 and len(frozen['preparations'])==8
    fits=run.read_json(tmp_path/'fits.json')
    assert sum(row['preparation_cache_hit'] for row in fits['records'])==8
    assert all(row['fit_state']['support'] is None for row in fits['records'])
    sha=run.initialize_frozen(tmp_path,frozen,observed,ledger)
    assert sha==run.digest(tmp_path/'initializations.json')
    assert len(fake_models)==64
    assert ledger['initializations_attempted']==ledger['initializations_completed']==48
    assert len(run.read_json(tmp_path/'initializations.json')['rows'])==48
    accounting=run.validate_counts(ledger,fits)
    assert accounting['prepared_batches']==8 and accounting['static_grid_references']==16
    assert ledger['fit']['reconstructions']==48
    assert ledger['initialization']['actual_residual_calls']==96


def test_frozen_model_tampering_prevents_initialization(fake_models,tmp_path):
    observed,plan=legal_inputs()
    frozen=run.fit_B(observed,plan,tmp_path,counts())
    (tmp_path/next(iter(frozen['models']))).write_text('{}')
    with pytest.raises(ValueError,match='frozen model'):
        run.initialize_frozen(tmp_path,frozen,observed,counts())
    assert len(fake_models)==16


def test_failed_fit_keeps_three_uncovered_predictions(fake_models,monkeypatch,tmp_path):
    cls=run.GFTDKuramotoModel
    original=cls.fit
    attempts=0
    def fit(self,*args):
        nonlocal attempts
        attempts+=1
        if attempts==1:
            raise run.engine.FitFailure('no candidate',{'counts':{}})
        return original(self,*args)
    monkeypatch.setattr(cls,'fit',fit)
    # Freeze assertions in the stub count this deliberately failed call too.
    fake_models.append(('fit','failed','0'))
    observed,plan=legal_inputs()
    ledger=counts()
    frozen=run.fit_B(observed,plan,tmp_path,ledger)
    sha=run.initialize_frozen(tmp_path,frozen,observed,ledger)
    rows=run.read_json(tmp_path/'initializations.json')['rows']
    assert len(rows)==48 and sum(row['status']=='blocked_by_fit' for row in rows)==3
    assert ledger['fits_attempted']==16 and ledger['fits_completed']==15
    assert ledger['initializations_attempted']==45 and sha


def test_typed_stage_failure_is_not_retried(fake_models,monkeypatch,tmp_path):
    observed,plan=legal_inputs()
    ledger=counts()
    frozen=run.fit_B(observed,plan,tmp_path,ledger)
    calls=0
    def blocked(self,packet):
        nonlocal calls
        calls+=1
        raise run.engine.InitializationFailure('blocked',{'status':'blocked_by_stage1','counts':{'actual_residual_calls':3}})
    monkeypatch.setattr(run.GFTDKuramotoModel,'initialize',blocked)
    run.initialize_frozen(tmp_path,frozen,observed,ledger)
    rows=run.read_json(tmp_path/'initializations.json')['rows']
    assert calls==48 and len(rows)==48
    assert all(row['status']=='blocked_by_stage1' for row in rows)
    assert ledger['initialization']['actual_residual_calls']==144


def release_fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    output=tmp_path/'results'
    monkeypatch.setattr(run,'OUTPUT',output)
    sources={}
    for path in run.REQUIRED_SOURCES:
        target=tmp_path/path
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('fixture')
        sources[path]=run.digest(target)
    protocol=tmp_path/run.PROTOCOL
    protocol.parent.mkdir(parents=True,exist_ok=True)
    protocol.write_text('protocol fixture')
    monkeypatch.setattr(run,'PROTOCOL_SHA256',run.digest(protocol))
    run.write_json(output/'execution_release.json',dict(released=True,action_id='mock-only',
        source_sha256=sources,protocol_sha256=run.PROTOCOL_SHA256,design_review_sha256=run.DESIGN_REVIEW_SHA256))
    monkeypatch.setattr(run.engine,'_BATCH_CACHE',{})
    monkeypatch.setattr(run.np,'__version__','2.5.2')
    monkeypatch.setattr(run.scipy,'__version__','1.18.0')
    return output


def test_A_failure_stops_B_and_exclusive_marker_blocks_replay(monkeypatch,tmp_path):
    output=release_fixture(monkeypatch,tmp_path)
    calls=[]
    def generate(directory,ledger):
        calls.append('generate')
        paths=[directory/name for name in ('isolated_truth.json','observations.json','observation_plan.json')]
        for path in paths: run.write_json(path,[])
        return tuple(paths)
    monkeypatch.setattr(run,'generate',generate)
    monkeypatch.setattr(run,'qualify_A',lambda *args: {'passed':False})
    monkeypatch.setattr(run,'fit_B',lambda *args: pytest.fail('B must remain stopped'))
    run.main()
    receipt=run.read_json(output/'execution_receipt.json')
    assert receipt['status']=='STOPPED_A_FAILED_B_NOT_RUN'
    assert receipt['source_unchanged'] and receipt['release_unchanged']
    with pytest.raises(FileExistsError): run.main()
    assert calls==['generate']


def test_release_hash_mismatch_precedes_started_or_generation(monkeypatch,tmp_path):
    output=release_fixture(monkeypatch,tmp_path)
    release=run.read_json(output/'execution_release.json')
    release['source_sha256'][run.REQUIRED_SOURCES[0]]='wrong'
    run.write_json(output/'execution_release.json',release)
    monkeypatch.setattr(run,'generate',lambda *args: pytest.fail('generation forbidden'))
    with pytest.raises(ValueError,match='source changed'): run.main()
    assert not (output/'started.json').exists()


def test_support_null_and_static_beta_are_not_false_success():
    assert run.support_score(None,[0])==dict(exact_support=None,false_positive=None,false_negative=None)
    assert run.support_score([],[])['exact_support'] is True


def test_registered_batch_boundary_and_budget_ceilings():
    observed,plan=legal_inputs()
    assert len(run.batches(observed,plan))==8
    with pytest.raises(ValueError): run.batches(observed,plan[:-1])
    broken=copy.deepcopy(plan)
    broken[1]['event']=0
    with pytest.raises(ValueError,match='E1/E2/E3'): run.batches(observed,broken)
    broken=copy.deepcopy(observed)
    broken['obs_000']['sigma']=1e-4
    with pytest.raises(ValueError,match='sigma'): run.batches(broken,plan)
    ledger=counts()
    ledger['fit']['actual_residual_calls']=76801
    with pytest.raises(AssertionError,match='budgets'):
        run.validate_counts(ledger,{'records':[]})


def test_mock_main_orders_freeze_initialization_then_truth_scoring(fake_models,monkeypatch,tmp_path):
    output=release_fixture(monkeypatch,tmp_path)
    observed,plan=legal_inputs()
    def generated(directory,ledger):
        ledger.update(generated_history_grids=24,generated_future_grids=24,observation_packages=24)
        paths=(directory/'isolated_truth.json',directory/'observations.json',directory/'observation_plan.json')
        for path,content in zip(paths,([{'isolated':'scoring only'}],observed,plan)): run.write_json(path,content)
        return paths
    monkeypatch.setattr(run,'generate',generated)
    monkeypatch.setattr(run,'qualify_A',lambda *args: {'passed':True})
    def score(directory,frozen,sha,truth_path,assignments):
        assert len(fake_models)==64
        assert run.digest(directory/'initializations.json')==sha
        assert run.read_json(truth_path)==[{'isolated':'scoring only'}]
        assert len(run.read_json(directory/'initializations.json')['rows'])==48
        return {'prediction_denominator':48}
    monkeypatch.setattr(run,'score_frozen',score)
    run.main()
    receipt=run.read_json(output/'execution_receipt.json')
    assert receipt['status']=='COMPLETED'
    assert receipt['counts']['fits_attempted']==16
    assert receipt['counts']['initializations_attempted']==48
    assert receipt['source_unchanged'] and receipt['real_data_reads'] is False
    accounting=run.read_json(output/'accounting.json')
    assert accounting['algebra_breakdown']['initial_static']['algebra_terms_planned']==1256
    assert accounting['algebra_breakdown']['static_enumeration']['algebra_terms_planned']==1040
