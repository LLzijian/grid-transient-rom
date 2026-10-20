"""Finite P-native envelope production checks; never launches a worker."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import pytest
MODULE=Path(__file__).resolve().parents[1]/'private_data/closure_mechanism_20261019/P2/edge_memory_amplitude01/launch_stage.py'
def setup(tmp_path,monkeypatch,category='cost'):
    sp=importlib.util.spec_from_file_location('D_envelope',MODULE);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    monkeypatch.setattr(m,'BASE',tmp_path.resolve());prior=tmp_path/'prior.json';prior.write_text('[]')
    a=SimpleNamespace(category=category,cap=m.CAPS[category],prior=prior)
    return m,a,{'arguments':[]}
def test_actual_caps_and_3600_boundary(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch)
    assert m.execution_contract()
    assert m.native_limits('train',2500.,2440.)['timeout_seconds']==2470.
    assert m.native_limits('evaluate',240.,180.)['timeout_seconds']==210.
    for category,cap,selfcap in [('train',2500.,2400.),('evaluate',240.,140.),('train',3601.,2440.)]:
        with pytest.raises(ValueError):m.native_limits(category,cap,selfcap)
    with pytest.raises(ValueError):m.ps_limits(3600,3601)
def test_only_E_current_stage_priors_and_single_attempt(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch,'train');args=['--seed','6101','--variant','E','--target-step','1024'];c['arguments']=args
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
    c['arguments']=['--seed','6101','--variant','E','--target-step','1024','--resume']
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
def test_static_cost_and_whole_budget(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch)
    (tmp_path/'implementation_cost.json').write_text(json.dumps({'qa_seconds':100.}))
    assert m.prior_cost_gate(a,c)['prior_native_seconds']['qa']==100.
    (tmp_path/'implementation_cost.json').write_text(json.dumps({'qa_seconds':180.01}))
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

def test_actual_command_consumer_float_caps_entry_and_worker(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch,'train')
    root=tmp_path/'root';base=root/'campaign';base.mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'BASE',base);monkeypatch.setattr(m,'__file__',str(base/'launch_stage.py'))
    ps=root/'experiments/native_process_receipt_20261020.ps1';ps.parent.mkdir()
    ps.write_text('$MaxSeconds -gt 3600',encoding='utf-8')
    a.command=base/'command.json';a.prior=base/'prior.json';a.worker=base/'models/6101_E/worker.json';a.source_commit='a'*40
    args=['-m','experiments.edge_memory_amplitude_20261021','train','--output',str(base/'models/6101_E'),'--max-seconds','2440.0','--source-commit',a.source_commit,'--seed','6101','--variant','E','--target-step','1024']
    names=['campaign/implementation_cost.json','campaign/launch_stage.py','campaign/command.json','campaign/prior.json','experiments/native_process_receipt_20261020.ps1','experiments/edge_memory_amplitude_runtime_20261021.py','experiments/edge_memory_amplitude_20261021.py','tests/test_edge_memory_amplitude_runtime_20261021.py','tests/test_edge_memory_amplitude_20261021.py','tests/test_edge_memory_amplitude_native_20261021.py','reports/edge_memory_amplitude_20261021_protocol.md','reports/astra6_after_port_only_20261021.md']
    binding={'files':dict.fromkeys(names,'synthetic')};c={'arguments':args}
    m.command_gate(a,c,binding)
    args[args.index('--variant')+1]='D'
    with pytest.raises(ValueError):m.command_gate(a,c,binding)
    args[args.index('--variant')+1]='E';args[args.index('--max-seconds')+1]='2441.0'
    with pytest.raises(ValueError):m.command_gate(a,c,binding)

def test_qa_ps_explicit_and_authorized_only_after_failure(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch,'qa');a.source_commit='a'*40
    assert m.native_limits('qa',200.,160.)['timeout_seconds']==180.
    folder=tmp_path/'native_qa01';folder.mkdir();p=folder/'execution.json'
    rec={'category':'qa','status':'COMPLETED','charged_seconds':1.,'inner':{'cleanup_verified':True,'fixture_identity_after':'GONE','native_receipt_complete':True},'outer':{'native_receipt_complete':True}}
    p.write_text(json.dumps(rec));(folder/'launch.json').write_text('{}');a.prior.write_text(json.dumps([str(p)]))
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
    rec['status']='UNRESOLVED_EXECUTION';p.write_text(json.dumps(rec))
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
    auth={'authorized':True,'pure_engineering_fault':True,'source_commit':a.source_commit,'qa01_execution_sha256':m.sha(p)}
    (tmp_path/'qa02_authorization.json').write_text(json.dumps(auth))
    assert m.prior_cost_gate(a,c)['prior_native_seconds']['qa']==1.

def test_actual_prior_consumer_accepts_E_and_rejects_P(tmp_path,monkeypatch):
    m,a,c=setup(tmp_path,monkeypatch,'train')
    c['arguments']=['--seed','6101','--variant','E','--target-step','1024']
    assert m.prior_cost_gate(a,c)['prior_native_seconds']['train']==0.
    c['arguments'][3]='P'
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)

def test_actual_frozen_table_producer_skips_old_large_leaf(tmp_path,monkeypatch):
    path=MODULE.with_name('make_invocation.py')
    sp=importlib.util.spec_from_file_location('E_generator',path);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g)
    root=tmp_path/'root';base=root/'campaign';base.mkdir(parents=True)
    old=root/'old/bindings/qa01.json';old.parent.mkdir(parents=True)
    leaf=root/'frozen_train_leaf.bin';leaf.write_bytes(b'synthetic immutable data')
    table={'files':{'frozen_train_leaf.bin':g.sha(leaf)}};old.write_text(json.dumps(table))
    receipt=old.parents[1]/'native_qa01/execution.json';receipt.parent.mkdir();receipt.write_text(json.dumps({'source_before':table['files'],'source_after':table['files']}))
    monkeypatch.setattr(g,'ROOT',root);monkeypatch.setattr(g,'BASE',base);monkeypatch.setattr(g,'BASE_TABLE',old);monkeypatch.setattr(g,'EXPECTED_BASE_COUNT',1);monkeypatch.setattr(g,'__file__',str(base/'make_invocation.py'))
    names=['campaign/make_invocation.py','campaign/launch_stage.py','campaign/qa_driver.py','campaign/implementation_cost.json','experiments/edge_memory_amplitude_20261021.py','experiments/edge_memory_amplitude_runtime_20261021.py','tests/test_edge_memory_amplitude_20261021.py','tests/test_edge_memory_amplitude_runtime_20261021.py','tests/test_edge_memory_amplitude_native_20261021.py','reports/edge_memory_amplitude_20261021_protocol.md','reports/edge_memory_amplitude_20261021_plan_review.md','reports/astra6_after_port_only_20261021.md']
    for n in names:
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('')
    realsha=g.sha
    def checked_sha(p):
        assert Path(p).resolve()!=leaf.resolve(),'operator must reuse this frozen hash'
        return realsha(p)
    monkeypatch.setattr(g,'sha',checked_sha)
    a=SimpleNamespace(name='qa01',category='qa',source_commit='a'*40,cap=200.,output=base/'qa01',extra=[])
    g.generate(a)
    binding=json.loads((base/'bindings/qa01.json').read_text())
    assert binding['files']['frozen_train_leaf.bin']==table['files']['frozen_train_leaf.bin']
    assert binding['base_table_sha256']==realsha(old)
    actual_args=json.loads((base/'commands/qa01.json').read_text())['arguments']
    assert actual_args[actual_args.index('--max-seconds')+1]=='160.0'
    assert (base/'generation/qa01.json').exists() and not (base/'native_qa01').exists()
    with pytest.raises(FileExistsError):g.generate(a)
