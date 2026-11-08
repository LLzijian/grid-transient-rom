"""Synthetic production contracts; no private event or checkpoint access."""
import copy,json,sys
from pathlib import Path
import numpy as np,pytest,torch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'experiments'),str(Path(__file__).resolve().parents[1]/'src')]
import cubic_port_only_20261021 as r
import cubic_port_only_runtime_20261021 as rt

def test_only_three_new_P1024_roles_and_fixed_phases():
    for seed in r.SEEDS:
        r.validate_request(seed,'P',1024)
        assert [sum(r.horizon_for('P',i)==h for i in range(1,1025)) for h in (1,2,5)]==[342,341,341]
    for variant in ('N','Q','D','S','p'):
        with pytest.raises(ValueError):r.validate_request(6101,variant,1024)
    for target in (512,768,1025):
        with pytest.raises(ValueError):r.validate_request(6101,'P',target)
    with pytest.raises(ValueError):r.validate_request(6101,'P',1024,True)

def test_P_payload_keys_order_dualcopy_and_signed_b():
    from symbolic_closure_rom import term_key
    values=[0.]*8;values[list(rt.TERMS_KEYS).index(rt.B_KEY)]=-2.
    p={'terms':copy.deepcopy(rt.TERMS),'coefficients':values,'state_dict':{'coefficients':values.copy()}}
    assert r.validate_endpoint_payload(p,'P')==p
    for key in ('terms','coefficients'):p[key].reverse()
    p['state_dict']['coefficients']=p['coefficients'].copy()
    assert r.validate_endpoint_payload(p,'P')==p
    for kind in ('duplicate','missing','nan','copy'):
        bad=copy.deepcopy(p)
        if kind=='duplicate':bad['terms'][1]=bad['terms'][0]
        if kind=='missing':bad['terms'].pop()
        if kind=='nan':bad['coefficients'][0]=float('nan')
        if kind=='copy':bad['state_dict']['coefficients'][0]+=1.
        with pytest.raises(ValueError):r.validate_endpoint_payload(bad,'P')

def test_time_fraction_linear_crossing_and_exact_zero():
    x=r.scalar_interval([1.,-3.],.01)
    assert x['positive_time_fraction']==pytest.approx(.25) and x['negative_time_fraction']==pytest.approx(.75)
    assert x['integral']==pytest.approx(-.01) and x['duration_seconds']==.01
    z=r.scalar_interval([0.,0.,0.],.01);assert z['zero_time_fraction']==1.
    isolated=r.scalar_interval([-1.,0.,1.],.01);assert isolated['zero_time_fraction']==0.
    assert isolated['positive_time_fraction']==isolated['negative_time_fraction']==.5

def test_n1_initial_is_one_interval_finite_fail_and_truncation():
    x=np.ones((2,40));result=r.algebraic_interval(x,1,-.5,True)
    assert result['duration_seconds']==.01 and result['summary']['port_power']['time_mean']==-5.
    assert r.algebraic_interval(x,2,-.5,True)['summary'] is None
    assert r.algebraic_interval(x,1,-.5,False)['summary'] is None
    x[1,30]=float('inf');assert r.algebraic_interval(x,1,-.5,True)['summary'] is None

def test_actual_producer_uses_existing_t0_once_no_rollout(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(rt,'coefficient_indices',lambda model:{'b':0})
    model=SimpleNamespace(coefficients=torch.tensor([1.],dtype=torch.float64))
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
    a=r.cost_admission({'P':{'1':1.,'2':1.,'5':1.}},.1,.1,100.)
    assert a['allow1024'] and a['three_models_seconds']==3*a['per_model_seconds']['P']
    p=tmp_path/'a.json';p.write_text(json.dumps(a));assert r.load_admission(p)==a
    assert not r.cost_admission({'P':{'1':1.,'2':1.,'5':20.}},.1,.1,100.)['allow1024']
    assert not r.cost_admission({'P':{'1':1.,'2':1.,'5':1.}},.1,.1,145.)['allow1024']

def test_actual15_full_flags_not_finite_only():
    real={'available':True,'finite':True,'numerically_admissible':True,'coi_normalized_rmse':0.,'common_normalized_rmse':0.}
    pair={'finite_to_endpoint':True,'los_pair_pass':True,'coi_normalized_rmse':0.,'common_normalized_rmse':0.}
    assert r.actual15_qualified(real,pair)
    pair['los_pair_pass']=False;assert not r.actual15_qualified(real,pair)

def test_shared_freeze_consumer_only_new_P_roles(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'BASE',tmp_path)
    for seed in r.SEEDS:assert r.endpoint_role(r.endpoint_path(seed,'P'))==(seed,'P')
    with pytest.raises(ValueError):r.endpoint_role(tmp_path/'models/6101_Q/checkpoints/step001024/model.json')
    monkeypatch.setattr(r,'check_endpoint',lambda p:{'model_path':str(p)})
    monkeypatch.setattr(r,'sha',lambda p:'synthetic')
    assert r.freeze_endpoints()['endpoint_count']==3
    with pytest.raises(FileExistsError):r.freeze_endpoints()

def test_nonempty_P_adam_restore_and_next_update(tmp_path):
    from test_quartic_memory_runtime_20261020 import source,inputs
    from train_closure import make_optimizer
    from linear_cache_cost_gate_20261019 import load_runtime,_equal
    load_runtime();torch.set_num_threads(1);dev=torch.device('cpu')
    model=rt.build_P(source(dev));opt=make_optimizer(model);_,store,samples=inputs(dev)
    rng=np.random.default_rng(5);coverage={'synthetic':True};identity={'qa_only':True}
    row=rt.one_update(model,opt,store,samples,1.,r.FORCE);folder=tmp_path/'ckpt';folder.mkdir()
    r.save_checked(folder,model,opt,rng,coverage,1,identity,[{'stage_step':1,**row}])
    rt.one_update(model,opt,store,samples,5.,r.FORCE)
    restored,ro,rr,rc,meta,_=r.restore_stage(folder,dev,identity);rt.one_update(restored,ro,store,samples,5.,r.FORCE)
    _equal(model.state_dict(),restored.state_dict(),exact=False);_equal(opt.state_dict(),ro.state_dict(),exact=False)
    assert rr.bit_generator.state==rng.bit_generator.state and rc==coverage and meta['step']==1

def test_baseline_initial_common_state_and_sampling_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'source_folder',lambda seed:tmp_path/'source')
    old=tmp_path/'models/6101_N/checkpoints/step001024/model.json'
    initial=old.parent.parent/'step000000'
    initial.mkdir(parents=True);old.parent.mkdir()
    (tmp_path/'source').mkdir()
    term={'channel':'node','expression':{'op':'var','name':'q'}}
    # Seven unique synthetic ASTs; no private source or actual baseline access.
    terms=[{'channel':'node','expression':{'op':'var','name':str(i)}} for i in range(7)]
    model={'terms':terms,'coefficients':[.1]*7,'state_dict':{'coefficients':[.1]*7,'mass':[1.]*10},'trainable_parameters':['coefficients']}
    ident={'synthetic':True}
    for path,value in [(initial/'model.json',model),(tmp_path/'source/model.json',model),(initial/'checkpoint.json',{'step':0,'trace':[],'identity':ident}),(old.parent/'checkpoint.json',{'identity':ident})]:
        path.write_text(json.dumps(value),encoding='utf-8')
    rows=[{'horizon_s':r.horizon_for('P',i),'sampling':{'sample_ids':list(range(12))},'long_counts':[100]*12,'local_counts':[20]*12} for i in range(1,1025)]
    schedule={'seeds':{'6101':{'rows':rows}}}
    entries=[{'seed':6101,'variant':'N','model_path':str(old),'checkpoint_path':str(old.parent/'checkpoint.json')}]
    assert r.baseline_initial_contract(entries,schedule)
    rows[0]['local_counts'].pop()
    with pytest.raises(ValueError):r.baseline_initial_contract(entries,schedule)

def test_scalar_summary_never_silently_uses_finite_subset():
    for values in ([0.,float('nan')],[float('inf'),1.]):
        with pytest.raises(ValueError):r.scalar_interval(values,.01)
    assert r.algebraic_interval(np.ones((2,40)),1,float('nan'),True)['summary'] is None

def test_actual_b_optimizer_group_and_all_parameter_ownership():
    from test_quartic_memory_runtime_20261020 import source
    from train_closure import make_optimizer
    model=rt.build_P(source('cpu'));opt=make_optimizer(model)
    for g in opt.param_groups:g['lr']=r.RATES[g['block']]
    assert r.validate_optimizer(model,opt)
    group=next(g for g in opt.param_groups if any(p is model.coefficients for p in g['params']))
    group['lr']*=2
    with pytest.raises(ValueError):r.validate_optimizer(model,opt)

def test_source_binding_and_nontrain_spec_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'BASE',tmp_path)
    monkeypatch.setattr(r,'ROOT',tmp_path)
    asset=tmp_path/'source.txt';asset.write_text('a',encoding='utf-8')
    manifest=tmp_path/'binding.json';manifest.write_text(json.dumps({'files':{str(asset):r.sha(asset)}}),encoding='utf-8')
    r.binding(manifest);asset.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError):r.binding(manifest)
    monkeypatch.setattr(r,'read',lambda path:{'split':'val','old_train':True,'new_directories':[]})
    with pytest.raises(ValueError):r.load_events()
