"""Synthetic production contracts; no private event or checkpoint access."""
import copy,json,sys
from pathlib import Path
import numpy as np,pytest,torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import cubic_memory_port_20261021 as r
import cubic_memory_port_runtime_20261021 as rt

def test_only_three_new_D1024_roles_and_fixed_phases():
    for seed in r.SEEDS:
        r.validate_request(seed,'D',1024)
        assert [sum(r.horizon_for('D',i)==h for i in range(1,1025)) for h in (1,2,5)]==[342,341,341]
    for variant in ('N','Q','S','d'):
        with pytest.raises(ValueError):r.validate_request(6101,variant,1024)
    for target in (512,768,1025):
        with pytest.raises(ValueError):r.validate_request(6101,'D',target)
    with pytest.raises(ValueError):r.validate_request(6101,'D',1024,True)

def test_D_payload_keys_order_dualcopy_projection_and_signed_b():
    from symbolic_closure_rom import term_key
    values=[0.]*9;values[list(rt.TERMS_KEYS).index(rt.B_KEY)]=-2.
    p={'terms':copy.deepcopy(rt.TERMS),'coefficients':values,'state_dict':{'coefficients':values.copy()}}
    assert r.validate_endpoint_payload(p,'D')==p
    for key in ('terms','coefficients'):p[key].reverse()
    p['state_dict']['coefficients']=p['coefficients'].copy()
    assert r.validate_endpoint_payload(p,'D')==p
    for kind in ('duplicate','missing','nan','copy','a_negative'):
        bad=copy.deepcopy(p)
        if kind=='duplicate':bad['terms'][1]=bad['terms'][0]
        if kind=='missing':bad['terms'].pop()
        if kind=='nan':bad['coefficients'][0]=float('nan')
        if kind=='copy':bad['state_dict']['coefficients'][0]+=1.
        if kind=='a_negative':
            idx=[term_key(t) for t in bad['terms']].index(rt.A_KEY);bad['coefficients'][idx]=-1.;bad['state_dict']['coefficients']=bad['coefficients'].copy()
        with pytest.raises(ValueError):r.validate_endpoint_payload(bad,'D')

def test_time_fraction_linear_crossing_and_exact_zero():
    x=r.scalar_interval([1.,-3.],.01)
    assert x['positive_time_fraction']==pytest.approx(.25) and x['negative_time_fraction']==pytest.approx(.75)
    assert x['integral']==pytest.approx(-.01) and x['duration_seconds']==.01
    z=r.scalar_interval([0.,0.,0.],.01);assert z['zero_time_fraction']==1.
    isolated=r.scalar_interval([-1.,0.,1.],.01);assert isolated['zero_time_fraction']==0.
    assert isolated['positive_time_fraction']==isolated['negative_time_fraction']==.5

def test_n1_initial_is_one_interval_finite_fail_and_truncation():
    x=np.ones((2,40));result=r.algebraic_interval(x,1,.2,-.5,True)
    assert result['duration_seconds']==.01 and result['summary']['port_power']['time_mean']==-5.
    assert result['summary']['quartic_energy']['integral']==pytest.approx(.02)
    assert r.algebraic_interval(x,2,.2,-.5,True)['summary'] is None
    assert r.algebraic_interval(x,1,.2,-.5,False)['summary'] is None
    x[1,30]=float('inf');assert r.algebraic_interval(x,1,.2,-.5,True)['summary'] is None

def test_actual_producer_uses_existing_t0_once_no_rollout(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(rt,'coefficient_indices',lambda model:{'a':0,'b':1})
    model=SimpleNamespace(coefficients=torch.tensor([0.,1.],dtype=torch.float64))
    fine=np.ones((3,40));fine[0,10:20]=1.;fine[2,10:20]=-3.
    event={'x':np.zeros((12,30)),'label':'unstable'}
    out=r.algebraic_evidence(model,event,fine,True,True)
    actual=out['actual_interval15'];assert actual['future_points']==1
    assert actual['summary']['port_power']['positive_time_fraction']==pytest.approx(.25)
    assert not out['1']['available'] and out['1']['summary'] is None
    assert r.algebraic_evidence(model,event,fine,True,False)['actual_interval15']['summary'] is None

def test_513_learning_rate_keeps_nonempty_adam():
    p=torch.nn.Parameter(torch.tensor(1.,dtype=torch.float64));opt=torch.optim.Adam([{'params':[p],'block':'closure','lr':.1}])
    p.grad=torch.tensor(.2,dtype=torch.float64);opt.step();before=copy.deepcopy(opt.state_dict())
    r.apply_lrs(opt,513);assert opt.param_groups[0]['lr']==r.RATES['closure']*.3
    for k,v in before['state'][0].items():torch.testing.assert_close(v,opt.state_dict()['state'][0][k],rtol=0,atol=0)

def test_whole_three_cost_not_old_six_budget(tmp_path):
    a=r.cost_admission({'D':{'1':1.,'2':1.,'5':1.}},.1,.1,100.)
    assert a['allow1024'] and a['three_models_seconds']==3*a['per_model_seconds']['D']
    p=tmp_path/'a.json';p.write_text(json.dumps(a));assert r.load_admission(p)==a
    assert not r.cost_admission({'D':{'1':1.,'2':1.,'5':20.}},.1,.1,100.)['allow1024']
    assert not r.cost_admission({'D':{'1':1.,'2':1.,'5':1.}},.1,.1,145.)['allow1024']

def test_actual15_full_flags_not_finite_only():
    real={'available':True,'finite':True,'numerically_admissible':True,'coi_normalized_rmse':0.,'common_normalized_rmse':0.}
    pair={'finite_to_endpoint':True,'los_pair_pass':True,'coi_normalized_rmse':0.,'common_normalized_rmse':0.}
    assert r.actual15_qualified(real,pair)
    pair['los_pair_pass']=False;assert not r.actual15_qualified(real,pair)

def test_shared_freeze_consumer_only_new_D_roles(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'BASE',tmp_path)
    for seed in r.SEEDS:assert r.endpoint_role(r.endpoint_path(seed,'D'))==(seed,'D')
    with pytest.raises(ValueError):r.endpoint_role(tmp_path/'models/6101_Q/checkpoints/step001024/model.json')
    monkeypatch.setattr(r,'check_endpoint',lambda p:{'model_path':str(p)})
    monkeypatch.setattr(r,'sha',lambda p:'synthetic')
    assert r.freeze_endpoints()['endpoint_count']==3
    with pytest.raises(FileExistsError):r.freeze_endpoints()

def test_nonempty_D_adam_restore_and_next_update(tmp_path):
    from test_quartic_memory_runtime_20261020 import source,inputs
    from train_closure import make_optimizer
    from linear_cache_cost_gate_20261019 import load_runtime,_equal
    load_runtime();torch.set_num_threads(1);dev=torch.device('cpu')
    model=rt.build_D(source(dev));opt=make_optimizer(model);_,store,samples=inputs(dev)
    rng=np.random.default_rng(5);coverage={'synthetic':True};identity={'qa_only':True}
    row=rt.one_update(model,opt,store,samples,1.,r.FORCE);folder=tmp_path/'ckpt';folder.mkdir()
    r.save_checked(folder,model,opt,rng,coverage,1,identity,[{'stage_step':1,**row}])
    rt.one_update(model,opt,store,samples,5.,r.FORCE)
    restored,ro,rr,rc,meta,_=r.restore_stage(folder,dev,identity);rt.one_update(restored,ro,store,samples,5.,r.FORCE)
    _equal(model.state_dict(),restored.state_dict(),exact=False);_equal(opt.state_dict(),ro.state_dict(),exact=False)
    assert rr.bit_generator.state==rng.bit_generator.state and rc==coverage and meta['step']==1
