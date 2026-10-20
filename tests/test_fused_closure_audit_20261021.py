"""Synthetic metadata and native producer guards; no scientific data access."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import fused_closure_audit_20261021 as audit

def records(factor=.8):
    return [{'structure':s,'horizon':h,'block':block,'path':p,'rep':rep,'seconds':(h+(10 if rep==0 else 0))*(factor if p=='new' else 1.)} for s in ('N','D') for h in (1,2,5) for block in ('AB','BA') for p in ('old','new') for rep in (0,1,2)]
def test_whole72_and_four_repeats_preserve_slow_first():
    rows=records();result=audit.audit_ratios(rows)
    assert result['speed_gate_pass'] and result['sum_median_ratios']['N']==pytest.approx(.8)
    assert result['medians']['D']['5']['new']==4.
    with pytest.raises(ValueError):audit.audit_ratios(rows[:-1])
    rows[0]['seconds']=float('nan')
    with pytest.raises(ValueError):audit.audit_ratios(rows)
def test_each_horizon_gate_and_both_structures_gate():
    assert not audit.audit_ratios(records(.95))['speed_gate_pass']
    rows=records(.8)
    for row in rows:
        if row['structure']=='D' and row['horizon']==1 and row['path']=='new':row['seconds']=1.06
    assert not audit.audit_ratios(rows)['speed_gate_pass']
def envelope(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('fused_native',audit.BASE/'launch_stage.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    monkeypatch.setattr(m,'BASE',tmp_path);prior=tmp_path/'prior.json';prior.write_text('[]')
    return m,SimpleNamespace(prior=prior,category='prepare',cap=100.,source_commit='a'*40),{'arguments':['--output',str(tmp_path/'prepare01')]}
def test_native_actual_limits_and_foreign_data_refusal(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch);assert m.execution_contract()
    assert m.native_limits('prepare',100.,40.)['timeout_seconds']==70.
    assert m.native_limits('benchmark',1800.,1740.)['timeout_seconds']==1770.
    for cap in (3601.,3850.):
        with pytest.raises(ValueError):m.ps_limits(cap-30,cap)
    with pytest.raises(ValueError):m.verify_binding({'files':{'private_data/foreign/model.json':'x'}})
    with pytest.raises(ValueError):m.verify_binding({'files':{'mixed.h5':'x'}})
def test_current_stage_budget_and_unclosed_queue(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch)
    (tmp_path/'implementation_cost.json').write_text(json.dumps({'static_seconds':99.}))
    assert m.prior_cost_gate(a,c)['static_seconds']==99.
    (tmp_path/'implementation_cost.json').write_text(json.dumps({'static_seconds':100.1}))
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
    (tmp_path/'implementation_cost.json').write_text(json.dumps({'static_seconds':1.}))
    folder=tmp_path/'native_pending';folder.mkdir();(folder/'launch.json').write_text('{}')
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
def test_QA02_and_benchmark_require_real_authorization_and_QA(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch);a.category='qa';a.cap=400.;c['arguments']=['--output',str(tmp_path/'qa02')]
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
    a.category='benchmark';a.cap=1800.;c['arguments']=['--output',str(tmp_path/'benchmark01')]
    with pytest.raises(ValueError):m.prior_cost_gate(a,c)
def test_exclusive_claim_never_overwrites_another_owner(tmp_path,monkeypatch):
    m,a,c=envelope(tmp_path,monkeypatch);lock=tmp_path/'native_active.lock';lock.write_text('{"pid":123}')
    with pytest.raises(FileExistsError):m.main()
    assert json.loads(lock.read_text())['pid']==123

def test_actual_prepare_producer_frozen_assets_and_mutation_refusal(monkeypatch):
    # Actual root-generated prepare01 is mandatory. Never regenerate it in QA.
    manifest=audit.verify_fixture();models,events,rng,other=audit.load_fixture('cpu')
    assert manifest==other and len(events)==12 and all(e['x'].shape==(531,30) for e in events)
    assert audit.data_identity(events,manifest['specification'])==manifest['data_sha256']
    assert set(models)=={'N','D'} and set(rng)=={'cpu','cuda','numpy'}
    original=audit.sha
    monkeypatch.setattr(audit,'sha',lambda path:'changed' if Path(path).name=='synthetic.npz' else original(path))
    with pytest.raises(ValueError,match='asset changed'):audit.verify_fixture()
def test_copied_generator_source_and_identity_rule_are_exact():
    import ast
    old=(ROOT/'tests/test_linear_cache_runtime_20261019.py').read_text(encoding='utf-8');new=Path(audit.__file__).read_text(encoding='utf-8')
    def body(source):return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_synthetic_events')
    a=body(old);b=body(new);b.body=b.body[1:] # only local NumPy import added
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def test_actual_old_N_benchmark_consumer_initialization_and_complete_update():
    import ast,math,torch
    from closure_training_runtime import TensorEventStore
    from train_closure import make_optimizer
    from linear_cache_cost_gate_20261019 import load_runtime,one_update
    load_runtime()
    models,events,rng,manifest=audit.load_fixture('cpu')
    model=models['N'];opt=make_optimizer(model);assert not opt.state
    audit.restore_rng(rng,'cpu')
    row=one_update(model,opt,TensorEventStore(events,torch.device('cpu')),[(e,10) for e in events],.01,audit.FORCE)
    assert math.isfinite(row['loss']) and math.isfinite(row['gradient_norm'])
    assert all(math.isfinite(v) for v in row['components'].values()) and opt.state
    assert getattr(model,'_runtime_cache',None) is None and getattr(model,'_linear_cache',None) is None
    source=Path(audit.__file__).read_text(encoding='utf-8');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='benchmark')
    initializer=[n.lineno for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load_old_runtime']
    timers=[n.lineno for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='perf_counter']
    assert len(initializer)==1 and initializer[0]<min(timers)
