"""Synthetic N/Q1024 production schedule, checkpoint and role contracts."""
import sys,copy,json
from pathlib import Path
import numpy as np,pytest,torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import quartic_memory_20261020 as r
import quartic_memory_runtime_20261020 as rt

def test_only_six_new1024_roles_and_full1024_schedule():
    for seed in r.SEEDS:
        for v in r.VARIANTS:
            r.validate_request(seed,v,1024)
            assert [sum(r.horizon_for(v,i)==h for i in range(1,1025)) for h in (1,2,5)]==[342,341,341]
    for target in (512,768):
        with pytest.raises(ValueError):r.validate_request(6101,'Q',target)
    with pytest.raises(ValueError):r.validate_request(6101,'Q',1024,True,True,32)
    with pytest.raises(ValueError):r.endpoint_role(r.BASE/'models/6101_Q/checkpoints/step000512/model.json')

def test_new_schedule_inherits_source_rng_coverage_once():
    events=[{'id':f'synthetic{i}','label':'stable' if i<30 else 'unstable','group':'g','x':np.zeros((35+i,30)),'torque':np.zeros((35+i,10))} for i in range(40)]
    rng=np.random.default_rng(3101);coverage={'cover_order':rng.permutation(40).tolist(),'cursor':17}
    other=copy.deepcopy(rng);cc=copy.deepcopy(coverage)
    rows=r.schedule_rows(events,rng,coverage);assert rows==r.schedule_rows(events,other,cc)
    assert len(rows)==1024 and all(len(row['sampling']['sample_ids'])==12 for row in rows)
    assert {eid for row in rows for eid in row['sampling']['sample_ids']}=={e['id'] for e in events}

def test_513_lr_decay_preserves_nonempty_adam_and_parameters():
    from types import SimpleNamespace
    p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));opt=torch.optim.Adam([{'params':[p],'block':'closure','lr':r.RATES['closure']}])
    p.grad=torch.tensor(.2,dtype=torch.float64);opt.step();state=copy.deepcopy(opt.state_dict()['state']);identity=id(opt)
    r.apply_lrs(opt,512);assert opt.param_groups[0]['lr']==r.RATES['closure']
    r.apply_lrs(opt,513);assert opt.param_groups[0]['lr']==.3*r.RATES['closure'] and id(opt)==identity
    for key,value in state[0].items():torch.testing.assert_close(value,opt.state_dict()['state'][0][key],rtol=0,atol=0)

def test_flat_cost_admission_whole_six_and_no512_fallback(tmp_path):
    timing={v:{str(h):1. for h in (1,2,5)} for v in r.VARIANTS};a=r.cost_admission(timing,1.,1.,100.)
    path=tmp_path/'admission.json';path.write_text(json.dumps(a));assert r.load_admission(path)==a and a['allow1024']
    assert a['six_models_seconds']==3*sum(a['per_model_seconds'].values())
    timing['Q']['5']=100.;assert not r.cost_admission(timing,1.,1.,100.)['allow1024']
    path.write_text(json.dumps({'allow512':True}))
    with pytest.raises(ValueError):r.load_admission(path)

def test_projected_checkpoint_restore_preserves_q_adam_rng_and_next_update(tmp_path):
    from test_quartic_memory_runtime_20261020 import source,inputs
    from train_closure import make_optimizer
    from linear_cache_cost_gate_20261019 import load_runtime,_equal
    load_runtime();torch.set_num_threads(1);dev=torch.device('cpu');model=rt.build_variant(source(dev),'Q');opt=make_optimizer(model)
    _,store,samples=inputs(dev);rng=np.random.default_rng(5);coverage={'synthetic':True}
    row=rt.one_update(model,opt,store,samples,1.,r.FORCE);folder=tmp_path/'ckpt';folder.mkdir();identity={'qa_only':True}
    r.save_checked(folder,model,opt,rng,coverage,1,identity,[{'stage_step':1,**row}])
    rt.one_update(model,opt,store,samples,5.,r.FORCE)
    restored,ro,rr,rc,meta,_=r.restore_stage(folder,dev,identity);rt.one_update(restored,ro,store,samples,5.,r.FORCE)
    _equal(model.state_dict(),restored.state_dict(),exact=False);_equal(opt.state_dict(),ro.state_dict(),exact=False)
    assert rr.bit_generator.state==rng.bit_generator.state and rc==coverage and meta['step']==1

def test_full_actual15_join_and_qualified_only_same_prediction_diagnostic():
    good={'available':True,'finite':True,'numerically_admissible':True,'coi_normalized_rmse':.0001,'common_normalized_rmse':.0001}
    pair={'finite_to_endpoint':True,'los_pair_pass':True,'coi_normalized_rmse':.0001,'common_normalized_rmse':.0001}
    assert r.actual15_qualified(good,pair)
    pair['los_pair_pass']=False;assert not r.actual15_qualified(good,pair)

def test_shared_role_freeze_contract_has_no_old_source_as_new_endpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'BASE',tmp_path/'new')
    for seed in r.SEEDS:
        for v in r.VARIANTS:assert r.endpoint_role(r.endpoint_path(seed,v))==(seed,v)
    for e in r.SOURCE_ENTRIES.values():
        with pytest.raises(ValueError):r.endpoint_role(e['model_path'])
    entries={r.endpoint_path(seed,v).resolve():{'seed':seed,'variant':v,'model_path':str(r.endpoint_path(seed,v))} for seed in r.SEEDS for v in r.VARIANTS}
    monkeypatch.setattr(r,'check_endpoint',lambda p:entries[Path(p).resolve()]);monkeypatch.setattr(r,'sha',lambda p:'synthetic')
    assert len(r.freeze_endpoints()['endpoints'])==6
    with pytest.raises(FileExistsError):r.freeze_endpoints()
