"""Finite D-native envelope production checks; never launches a worker."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
MODULE=Path(__file__).resolve().parents[1]/'private_data/closure_mechanism_20261019/P2/cubic_memory_port01/launch_stage.py'
def setup(tmp_path,monkeypatch,category='cost'):
    sp=importlib.util.spec_from_file_location('D_envelope',MODULE);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    monkeypatch.setattr(m,'BASE',tmp_path.resolve());prior=tmp_path/'prior.json';prior.write_text('[]')
    a=SimpleNamespace(category=category,cap=m.CAPS[category],prior=prior)
    return m,a,{'arguments':[]}
def test_actual_caps_and_3600_boundary(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch)
    assert m.execution_contract()
    assert m.native_limits('train',3400.,3300.)['timeout_seconds']==3370.
    assert m.native_limits('evaluate',240.,180.)['timeout_seconds']==210.
    for category,cap,selfcap in [('train',3400.,3340.),('evaluate',240.,140.),('train',3601.,3300.)]:
        with pytest.raises(ValueError):m.native_limits(category,cap,selfcap)
    with pytest.raises(ValueError):m.ps_limits(3600,3601)
def test_only_D_current_stage_priors_and_single_attempt(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch,'train');args=['--seed','6101','--variant','D','--target-step','1024'];c['arguments']=args
    assert m.prior_cost_gate(a,c)['prior_native_seconds']['train']==0.
    folder=tmp_path/'native_previous';folder.mkdir();p=folder/'execution.json'
    rec={'category':'train','status':'COMPLETED','charged_seconds':1.,'worker_command':{'arguments':args},'inner':{'cleanup_verified':True,'fixture_identity_after':'GONE','native_receipt_complete':True},'outer':{'native_receipt_complete':True}}
    p.write_text(json.dumps(rec));(folder/'launch.json').write_text('{}');a.prior.write_text(json.dumps([str(p)]))
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
    rec['inner']['cleanup_verified']=False;p.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='ownership'):m.prior_cost_gate(a,c)
def test_no_Q_role_or_recovery(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch,'train');c['arguments']=['--seed','6101','--variant','Q','--target-step','1024']
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
    c['arguments']=['--seed','6101','--variant','D','--target-step','1024','--resume']
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
def test_static_cost_and_whole_budget(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch)
    (tmp_path/'implementation_cost.json').write_text(json.dumps({'qa_seconds':100.}))
    assert m.prior_cost_gate(a,c)['prior_native_seconds']['qa']==100.
    (tmp_path/'implementation_cost.json').write_text(json.dumps({'qa_seconds':100.01}))
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
def test_unreconciled_or_foreign_prior_rejected(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch)
    (tmp_path/'native_unclosed').mkdir();(tmp_path/'native_unclosed/launch.json').write_text('{}')
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
def test_qa_is_one_attempt(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch,'qa')
    folder=tmp_path/'native_previous';folder.mkdir();p=folder/'execution.json'
    p.write_text(json.dumps({'category':'qa','status':'FAILED','charged_seconds':1.,'inner':{'cleanup_verified':True,'fixture_identity_after':'GONE','native_receipt_complete':True},'outer':{'native_receipt_complete':True}}));(folder/'launch.json').write_text('{}');a.prior.write_text(json.dumps([str(p)]))
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
