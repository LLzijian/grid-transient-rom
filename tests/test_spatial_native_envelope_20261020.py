"""Synthetic native budget and ownership checks; no worker is launched."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/"private_data/closure_mechanism_20261019/P2/spatial_extrema01/launch_stage.py"

def envelope(tmp_path,monkeypatch,category="cost",args=None):
    spec=importlib.util.spec_from_file_location("spatial_native_test",MODULE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,"BASE",tmp_path.resolve())
    prior=tmp_path/"prior.json";prior.write_text("[]")
    a=SimpleNamespace(prior=prior,category=category,cap=1200.)
    command={"arguments":args or ["-m","experiments.spatial_extrema_20261020","cost"]}
    return m,a,command

def record(m,a,category="qa",args=None,cleanup=True):
    folder=m.BASE/"native_previous";folder.mkdir()
    receipt={"category":category,"status":"COMPLETED","charged_seconds":20.,
      "worker_command":{"arguments":args or []},
      "inner":{"cleanup_verified":cleanup,"fixture_identity_after":"GONE","native_receipt_complete":True},
      "outer":{"native_receipt_complete":True}}
    path=folder/"execution.json";path.write_text(json.dumps(receipt))
    (folder/"launch.json").write_text("{}")
    a.prior.write_text(json.dumps([str(path)]))
    return path

def test_exact_prior_and_complete_owned_cleanup(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch);record(m,a)
    assert m.prior_cost_gate(a,c)["prior_native_seconds"]["qa"]==20.
    a.prior.write_text("[]")
    with pytest.raises(ValueError,match="prior receipts incomplete"):m.prior_cost_gate(a,c)

def test_unresolved_cleanup_blocks_independent_launch(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch);record(m,a,cleanup=False)
    with pytest.raises(ValueError,match="ownership/cleanup unresolved"):m.prior_cost_gate(a,c)

def test_same_train_endpoint_cannot_run_twice(tmp_path,monkeypatch):
    args=["--seed","6101","--variant","S","--target-step","512"]
    m,a,c=envelope(tmp_path,monkeypatch,"train",args);a.cap=1800.;record(m,a,"train",args)
    with pytest.raises(ValueError,match="endpoint already attempted"):m.prior_cost_gate(a,c)

def test_same_evaluation_cannot_run_twice(tmp_path,monkeypatch):
    args=["--model","endpoint512/model.json"]
    m,a,c=envelope(tmp_path,monkeypatch,"evaluate",args);a.cap=180.;record(m,a,"evaluate",args)
    with pytest.raises(ValueError,match="evaluation already attempted"):m.prior_cost_gate(a,c)

def test_category_and_nested_recovery_caps(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch);a.cap=4201.
    with pytest.raises(ValueError,match="cumulative category"):m.prior_cost_gate(a,c)
    a.category="train";a.cap=1201.;c["arguments"]=["--seed","6101","--variant","S","--target-step","512","--external-recovery"]
    with pytest.raises(ValueError,match="no scientific recovery"):m.prior_cost_gate(a,c)

def test_exclusive_native_claim_preserves_other_owner(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch)
    lock=tmp_path/"native_active.lock";lock.write_text('{"pid":123}')
    monkeypatch.setattr(m,"main_inner",lambda:0)
    with pytest.raises(FileExistsError):m.main()
    assert json.loads(lock.read_text())=={"pid":123}


def test_explicit_preparation_cost_counts_against_engineering_and_stage_caps(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch)
    (tmp_path/'implementation_cost.json').write_text(json.dumps({'qa_seconds':3100.}))
    a.cap=1200.
    with pytest.raises(ValueError,match='cumulative category'):m.prior_cost_gate(a,c)
    a.cap=1000.
    result=m.prior_cost_gate(a,c)
    assert result['prior_native_seconds']['qa']==3100. and result['preparation_qa_seconds']==3100.

def test_production_entry_requires_preparation_cost_binding(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch)
    a.command=m.ROOT/'synthetic_command.json';a.prior=m.ROOT/'synthetic_prior.json';a.cap=1500.;c['arguments']+=['--max-seconds','1440']
    with pytest.raises(ValueError,match='required source/entry/command/protocol binding absent'):
        m.command_gate(a,c,{'files':{}})


def test_real_ps_3600_boundary_and_all_actual_commands(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch)
    assert m.execution_contract() and m.ps_limits(3570,3600)
    for timeout,maximum in ((3570,3601),(3600,3600),(0,3600)):
        with pytest.raises(ValueError):m.ps_limits(timeout,maximum)
    assert m.native_limits('train',2200,2100)=={'max_seconds':2200,'timeout_seconds':2170,'worker_seconds':2100}
    for mode,cap,worker in [('train',3850,3790),('train',3601,3500),('evaluate',400,370),('cost',4200,4140)]:
        with pytest.raises(ValueError):m.native_limits(mode,cap,worker)

def test_imported_old_O_cannot_train_or_enter_prior_receipts(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch,'train',['--seed','6101','--variant','O','--target-step','512']);a.cap=2200
    with pytest.raises(ValueError,match='read-only'):m.prior_cost_gate(a,c)
    old=tmp_path.parent/'old_execution.json';old.write_text('{}');a.prior.write_text(json.dumps([str(old)]))
    with pytest.raises(ValueError,match='nonlocal'):m.prior_cost_gate(a,c)
