"""Synthetic native budget and ownership checks; no worker is launched."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/"private_data/closure_mechanism_20261019/P2/quartic_memory01/launch_stage.py"

def envelope(tmp_path,monkeypatch,category="cost",args=None):
    spec=importlib.util.spec_from_file_location("quartic_native_test",MODULE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,"BASE",tmp_path.resolve())
    prior=tmp_path/"prior.json";prior.write_text("[]")
    a=SimpleNamespace(prior=prior,category=category,cap=1200.)
    command={"arguments":args or ["-m","experiments.quartic_memory_20261020","cost"]}
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
    args=["--seed","6101","--variant","Q","--target-step","1024"]
    m,a,c=envelope(tmp_path,monkeypatch,"train",args);a.cap=1800.;record(m,a,"train",args)
    with pytest.raises(ValueError,match="endpoint already attempted"):m.prior_cost_gate(a,c)

def test_same_evaluation_cannot_run_twice(tmp_path,monkeypatch):
    args=["--model","endpoint512/model.json"]
    m,a,c=envelope(tmp_path,monkeypatch,"evaluate",args);a.cap=180.;record(m,a,"evaluate",args)
    with pytest.raises(ValueError,match="evaluation already attempted"):m.prior_cost_gate(a,c)

def test_category_and_nested_recovery_caps(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch);a.cap=4201.
    with pytest.raises(ValueError,match="cumulative category"):m.prior_cost_gate(a,c)
    a.category="train";a.cap=1201.;c["arguments"]=["--seed","6101","--variant","Q","--target-step","1024","--external-recovery"]
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
    a.command=m.ROOT/'synthetic_command.json';a.prior=m.ROOT/'synthetic_prior.json';a.cap=1800.;c['arguments']+=['--max-seconds','1740']
    with pytest.raises(ValueError,match='required source/entry/command/protocol binding absent'):
        m.command_gate(a,c,{'files':{}})





def test_actual_native_boundary_and_complete_six_command_budget(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch)
    assert m.execution_contract() and m.ps_limits(3570,3600)
    assert m.native_limits('train',3500,3400)=={'max_seconds':3500,'timeout_seconds':3470,'worker_seconds':3400}
    for cap,worker in ((3850,3790),(3601,3500),(3500,3470)):
        with pytest.raises(ValueError):m.native_limits('train',cap,worker)
