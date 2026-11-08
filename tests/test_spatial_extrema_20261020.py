"""Synthetic-only role, source, shared schedule and production consumer tests."""
from pathlib import Path
import sys,copy,json
import numpy as np,pytest,torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import spatial_extrema_20261020 as r

def test_five_new_training_roles_and_shared_phase_lock():
    for seed,v in ((6101,'S'),(6102,'O'),(6102,'S'),(6103,'O'),(6103,'S')):
        r.validate_request(seed,v,512)
        assert [sum(r.horizon_for(v,i)==h for i in range(1,513)) for h in (1,2,5)]==[171,171,170]
    for seed,v,target,kwargs in [(6101,'O',512,{}),(6102,'F',512,{}),(6102,'O',768,{}),(6102,'S',512,{'resume':True,'external':True,'start':32})]:
        with pytest.raises(ValueError):r.validate_request(seed,v,target,**kwargs)

def test_imported_role_exact_path_shared_by_freeze_and_consumer(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'BASE',tmp_path/'new');monkeypatch.setattr(r,'TEMPORAL',tmp_path/'old')
    assert r.endpoint_role(tmp_path/'old/models/6101_O/checkpoints/step000512/model.json')==(6101,'O')
    assert r.endpoint_role(r.endpoint_path(6102,'S'))==(6102,'S')
    for p in (tmp_path/'new/models/6101_O/checkpoints/step000512/model.json',tmp_path/'old/models/6101_F/checkpoints/step000512/model.json'):
        with pytest.raises(ValueError):r.endpoint_role(p)

def test_equilibrium_metadata_explicit_ten_common_machine_order():
    a={'classification_meta':{'equilibrium_delta':list(range(10))}};b=copy.deepcopy(a)
    assert r.equilibrium_delta([a,b])==list(range(10))
    for bad in ({},{'classification_meta':{'equilibrium_delta':[0]*9}},{'classification_meta':{'equilibrium_delta':[float('nan')]*10}}):
        with pytest.raises(ValueError):r.equilibrium_delta([bad])
    b['classification_meta']['equilibrium_delta'].reverse()
    with pytest.raises(ValueError):r.equilibrium_delta([a,b])

def test_cost_output_consumed_by_train_and_full_five_plus_six(tmp_path):
    timing={v:{str(h):1. for h in (1,2,5)} for v in r.VARIANTS}
    a=r.cost_admission(timing,1.,1.,100.)
    path=tmp_path/'admission.json';path.write_text(json.dumps(a))
    assert r.load_admission(path)==a and a['allow512'] and not a['allow768']
    assert a['five_models_seconds']==2*a['per_model_seconds']['O']+3*a['per_model_seconds']['S']
    timing['S']['5']=100.;assert not r.cost_admission(timing,1.,1.,100.)['allow512']
    assert not r.cost_admission({v:{str(h):1. for h in (1,2,5)} for v in r.VARIANTS},1.,1.,300.)['allow512']

def test_fixed_schedule_path_hash_not_reallocated(tmp_path,monkeypatch):
    path=tmp_path/'original.json';path.write_text('{"seeds":{}}')
    monkeypatch.setattr(r,'SCHEDULE',path);monkeypatch.setattr(r,'SCHEDULE_SHA',r.sha(path))
    assert r.checked_schedule()=={'seeds':{}}
    other=tmp_path/'copied.json';other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):r.checked_schedule(other)
    path.write_text('{}')
    with pytest.raises(ValueError):r.checked_schedule()

def test_freeze_and_evaluate_accept_same_six_roles_without_moving_old_control(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(r,'BASE',tmp_path/'new');monkeypatch.setattr(r,'TEMPORAL',tmp_path/'old')
    entries={}
    for seed in r.SEEDS:
        for v in r.VARIANTS:
            p=r.endpoint_path(seed,v);entries[p.resolve()]={'seed':seed,'variant':v,'model_path':str(p.resolve()),'model_sha256':str(seed)+v,'training_sha256':'adam','checkpoint_sha256':'meta'}
    monkeypatch.setattr(r,'check_endpoint',lambda path:copy.deepcopy(entries[Path(path).resolve()]))
    frozen=r.freeze_endpoints();assert len(frozen['endpoints'])==6
    assert frozen['endpoints'][0]['model_path'].startswith(str(tmp_path/'old'))
    with pytest.raises(FileExistsError):r.freeze_endpoints()
    from test_spatial_extrema_runtime_20261020 import model_fixture
    import closure_training_runtime
    model=model_fixture(torch.device('cpu'))
    monkeypatch.setattr(closure_training_runtime.TrainingClosureROM,'from_export',lambda path:copy.deepcopy(model))
    monkeypatch.setattr(r,'load_events',lambda:([{'id':'synthetic'}],[]));monkeypatch.setattr(r,'equilibrium_delta',lambda events:[0.]*10)
    seen=[]
    def queue(m,e,output,provenance,check):
        seen.append(provenance);output.mkdir(parents=True);(output/'evidence.json').write_text('{}');return {'event_count':385,'primary':None}
    monkeypatch.setattr(r,'evaluate_queue',queue)
    args=SimpleNamespace(model=r.endpoint_path(6101,'O'),output=(r.BASE/'evaluation/6101_O/step000512').resolve())
    result=r.evaluate(args,lambda:None)
    assert result['event_count']==385 and seen[0]['variant']=='O'
    frozen['endpoints'][0]['model_sha256']='changed';(r.BASE/'frozen_evaluation_endpoints.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError,match='frozen endpoint changed'):r.evaluate(args,lambda:None)

def test_spatial_same_prediction_diagnostic_is_null_when_15s_unqualified():
    from test_spatial_extrema_runtime_20261020 import model_fixture
    model=model_fixture(torch.device('cpu'));e={'x':np.zeros((12,30)),'classification_meta':{'equilibrium_delta':list(np.linspace(-.2,.3,10))}}
    pred=np.zeros((3001,40));pred[2,0]=2.
    no=r.spatial_metrics(model,e,pred,False)
    yes=r.spatial_metrics(model,e,pred,True)
    assert no['finite'] and no['combined'] is None and yes['combined']>0 and yes['actual_future_points']==1

def test_longest_restore_cost_witness_does_not_add_trials():
    for v in r.VARIANTS:
        cases=r.cost_cases(v);assert cases[:2]==[(1,'long'),(5,'long')]
        assert len(cases)==len(set(cases))==6
        assert len([cases[0],cases[1],cases[1],cases[0]]+[c for c in cases[2:] for _ in (1,2)])==12


def test_actual15_full_pair_join_keeps_unqualified_explanation_null():
    from test_spatial_extrema_runtime_20261020 import model_fixture
    model=model_fixture(torch.device('cpu'))
    event={'x':np.zeros((12,30)),'classification_meta':{'equilibrium_delta':list(np.linspace(-.2,.3,10))}}
    pred=np.zeros((3001,40));pred[2,0]=2.
    real={'available':True,'finite':True,'numerically_admissible':True,'coi_normalized_rmse':.0001,'common_normalized_rmse':.0001}
    pair={'finite_to_endpoint':True,'los_pair_pass':True,'coi_normalized_rmse':.0001,'common_normalized_rmse':.0001}
    assert r.actual15_qualified(real,pair)
    cases=[]
    for key in ('available','finite','numerically_admissible'):
        bad=dict(real);bad[key]=False;cases.append((bad,pair))
    for key in ('finite_to_endpoint','los_pair_pass'):
        bad=dict(pair);bad[key]=False;cases.append((real,bad))
    for row_index in (0,1):
        for key in ('coi_normalized_rmse','common_normalized_rmse'):
            for value in (.00101,float('nan'),None):
                rows=[dict(real),dict(pair)];rows[row_index][key]=value;cases.append(tuple(rows))
    for bad_real,bad_pair in cases:
        qualified=r.actual15_qualified(bad_real,bad_pair)
        assert not qualified
        diagnostic=r.spatial_metrics(model,event,pred,qualified)
        assert diagnostic['finite'] and not diagnostic['numerically_admissible']
        assert diagnostic['actual_future_points']==1
        assert diagnostic['combined'] is diagnostic['angle_normalized_squared_error'] is diagnostic['speed_normalized_squared_error'] is None
