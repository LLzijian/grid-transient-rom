"""Pure finite-study QA: mathematical fixtures, zero optimizer calls/cache data."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import force_exchange_study_20261028 as study
from force_exchange_model_20261028 import ForceExchangeROM
from test_force_graph_model_20261028 import fixture,history

torch.set_num_threads(1)


def events(): return [{"id":str(i),"label":"unstable" if i<30 else "stable","group":str(i%5)} for i in range(297)]


def p(risk=1.,miss=3):
    return dict(all_8s_rows_qualified=True,all_required_rows_qualified=True,
      horizon_risks={h:dict(value=1.,numerical_score_perturbation=.001) for h in ("5","8")},
      unstable_interval_risk=risk,unstable_interval_numerical_score_perturbation=.001,
      unstable_interval_unresolved_ids=[],classification_8s=dict(missed_resolved_unstable=miss,false_los_resolved_stable=0),
      common_frequency_rmse_8s=1.,normalized_p_rmse_8s=1.,parameter_count=48,qualified_inference_seconds_per_event=.1)


def trace(n=96): return [dict(gradient_norm=1.,block_updates={"a_bar":dict(relative_l2=.001)}) for _ in range(n)]


@pytest.mark.parametrize("seed",study.SEEDS)
def test_frozen_schedule_pairing_coverage_and_rng(seed):
    ev=events();a=study.make_schedule(ev,seed);b=study.make_schedule(ev,seed)
    assert a==b and len(a["rows"])==192
    assert {ev[i]["id"] for row in a["rows"][:75] for i in row["coverage_indices"]}=={e["id"] for e in ev}
    for i,row in enumerate(a["rows"],1):
        assert row["step"]==i and row["horizon_s"]==(5. if i%2 else 8.)
        assert len(row["sample_ids"])==12 and len(row["random_indices"])==8
        assert len(set(row["coverage_indices"]))==4 and row["endpoints"]==[10]*12
    with pytest.raises(ValueError): study.make_schedule(ev,28101)


def test_extension_only_key_events_and_original_safety_boundaries():
    before=p();after=p(.98,3)
    assert study.extension_decision(before,after,trace())["extend"]
    stable_only=p();stable_only["horizon_risks"]["8"]["value"]=.5
    assert not study.extension_decision(before,stable_only,trace())["extend"]
    after=p(1.,2);assert study.extension_decision(before,after,trace())["extend"]
    after["classification_8s"]["false_los_resolved_stable"]=1
    assert not study.extension_decision(before,after,trace())["extend"]
    tr=trace();[r.update(gradient_norm=101.) for r in tr[-12:]]
    assert not study.optimization(tr)["optimization_unresolved"]
    tr[-13]["gradient_norm"]=101.;assert study.optimization(tr)["optimization_unresolved"]
    tr=trace();[r["block_updates"]["a_bar"].update(relative_l2=.251) for r in tr[-12:]]
    assert study.optimization(tr)["optimization_unresolved"]
    missing=p(.9,2);missing["unstable_interval_numerical_score_perturbation"]=None
    assert not study.extension_decision(before,missing,trace())["extend"]


def test_common96_retention_A_B_nonworsening_and_no_192_rescue():
    F,C0,C1=p(1.,3),p(.99,3),p(.99,2)
    assert study.retention(F,C0,C1)["A"] and study.retention(F,C0,C1)["second_seed"]
    C1=p(.93,3);assert study.retention(F,C0,C1)["B"] and not study.retention(F,C0,C1)["A"]
    C1["normalized_p_rmse_8s"]=1.051;assert not study.retention(F,C0,C1)["second_seed"]
    assert not study.retention(F,C0,p(.9,1),optimization_unresolved=True)["second_seed"]
    C1=p(.9,1);C1["all_required_rows_qualified"]=False
    assert not study.retention(F,C0,C1)["second_seed"]
    assert study.retention(F,C0,p())["comparison"]=="common96"


def test_unstable_perturbation_ceil_risk_true_durations_unresolved_not_deleted():
    records={k:[] for k in ("records","common_frequency","normalized_p","objective_point_error","numerical")}
    extra=[];ev=[]
    for i,v in enumerate((1.,2.,6.)):
        eid=str(i);ev.append(dict(id=eid,label="unstable",x=np.zeros((22+i,30))))
        records["objective_point_error"].append(dict(event_id=eid,label="unstable",horizon="prefix",available=True,finite=True,
                                                     numerically_admissible=True,mean_point_error=v))
        extra.append(dict(event_id=eid,label="unstable",scientific_numeric_8s=True,coarse_point_errors={"prefix":v+.5},
            classification_8s={"fine_classification":dict(predicted_los=True)}))
    out=study.profile(records,extra,ev,1.,48)
    expected=.8*3.+.2*6.
    assert out["unstable_interval_risk"]==pytest.approx(expected)
    assert out["unstable_interval_numerical_score_perturbation"]==pytest.approx(.5)
    assert out["unstable_interval_steps"]=={"0":11,"1":12,"2":13}
    assert out["J_actual_intervals_up_to8_fine"]==pytest.approx(10*expected)
    extra[1]["scientific_numeric_8s"]=False
    out=study.profile(records,extra,ev,1.,48)
    assert out["unstable_interval_risk"] is None and out["unstable_interval_numerical_score_perturbation"] is None
    assert out["unstable_interval_eligible"]==3 and out["unstable_interval_unresolved_ids"]==["1"]


def test_real_source_JSON_identity_and_trainable_groups_without_old_Adam_load():
    identity=study.source_identity()
    assert identity["files"]==study.SOURCE_SHA
    F,c0,c1=[study.build_model(v) for v in ("F","C0","C1")]
    assert type(F) is study.ForceGraphROM and not any(p.requires_grad for p in F.parameters())
    assert c0.gamma.item()==c1.gamma.item()==0.
    assert sum(p.numel() for p in c0.parameters() if p.requires_grad)==47
    assert sum(p.numel() for p in c1.parameters() if p.requires_grad)==48
    for m in (c0,c1):
        opt=study.make_optimizer(m)
        assert not opt.state and all(g["lr"]==.03 for g in opt.param_groups)
        assert {g["block"] for g in opt.param_groups}=={n for n,p in m.named_parameters() if p.requires_grad}
    assert torch.equal(c0.initialize(history()),c1.initialize(history()))


@pytest.mark.parametrize("train_gamma",(False,True))
def test_zero_update_checkpoint_preserves_subclass_gamma_layout_and_all_RNG(train_gamma):
    model=ForceExchangeROM(fixture(),"M",train_gamma=train_gamma)
    with torch.no_grad(): model.raw_gamma.fill_(-.4)
    opt=study.make_optimizer(model);rng=np.random.default_rng(11);cov=dict(cover_order=[0,1],cursor=0)
    identity=dict(fixture_only=True,test="zero-update serialization; not real recovery qualification")
    with tempfile.TemporaryDirectory(prefix="exchange-study-zero-") as folder:
        snap=study.old._cpu_snapshot(dict(model=model.state_dict(),Adam=opt.state_dict(),RNG=study.fg._rng_state(rng,cov),layout=study.old._parameter_layout(model,opt)))
        path=study.save_checkpoint(Path(folder),model,opt,rng,cov,0,identity,[])
        restored,other,rr,cc,meta=study.restore_checkpoint(Path(folder),"cpu",identity)
        assert type(restored) is ForceExchangeROM and restored.raw_gamma.requires_grad==train_gamma
        study.old._exact_nested(snap,study.old._cpu_snapshot(dict(model=restored.state_dict(),Adam=other.state_dict(),RNG=study.fg._rng_state(rr,cc),layout=study.old._parameter_layout(restored,other))))
        assert meta["step"]==0 and not other.state
        with (path/"model.json").open("a") as stream: stream.write(" ")
        with pytest.raises(ValueError,match="asset changed"): study.restore_checkpoint(Path(folder),"cpu",identity)


def test_ledger_pairing_limits_and_reserves_with_explicit_synthetic_records_only():
    # These are schema fixtures in a disposable directory, never science counts.
    with tempfile.TemporaryDirectory(prefix="exchange-ledger-fixture-") as folder:
        base=Path(folder);path=base/"optimizer_calls.jsonl"
        assert study.call_counts(base)["attempts"]==0
        rows=[dict(event="optimizer_step_attempt",category="qa",call_id="fixture",run_id="fixture",step=1)]
        path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        assert len(study.call_counts(base)["pending"])==1
        with pytest.raises(ValueError,match="unreconciled"): study._journal(base,"optimizer_step_attempt","qa",call_id="x",run_id="fixture",step=2)
        rows.append({**rows[0],"event":"optimizer_step_completed"});path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        assert study.call_counts(base)["attempts"]==study.call_counts(base)["completed"]==1
        with pytest.raises(ValueError,match="reserved"): study._journal(base,"optimizer_step_attempt","repair",call_id="x",run_id="fixture",step=2)
        path.write_text("\n".join(json.dumps(dict(event=event,category="qa",call_id=str(i),run_id="fixture",step=i))
            for i in range(72) for event in ("optimizer_step_attempt","optimizer_step_completed"))+"\n")
        with pytest.raises(ValueError,match="bound exhausted"): study._journal(base,"optimizer_step_attempt","qa",call_id="x",run_id="fixture",step=73)
        assert sum(study.LIMITS.values())==976


def test_phase_device_and_admission_guards_are_not_CPU_fallbacks():
    with tempfile.TemporaryDirectory(prefix="exchange-guard-") as folder:
        with pytest.raises(ValueError,match="CUDA"): study.qualify(folder,"cpu")
        with pytest.raises(ValueError): study.qa(folder,"cpu")
        with pytest.raises(ValueError): study.study(folder,"cpu")


def cross_fixture():
    model=ForceExchangeROM(fixture(),"M")
    t=np.arange(3001)*.005
    fine=np.zeros((3001,40),dtype=np.float64)
    fine[:,:10]=(.05*np.sin(20*t))[:,None]
    fine[:,10:20]=np.cos(20*t)[:,None]
    ht=np.arange(-10,1)*.01
    h=np.zeros((11,30));h[:,:10]=(.05*np.sin(20*ht))[:,None];h[:,10:20]=np.cos(20*ht)[:,None]
    event=dict(id="cross_fixture",group="synthetic",label="stable",x=np.r_[h,fine[::2][1:,:30]],
        classification_meta=dict(equilibrium_delta=np.zeros(10).tolist(),omega_s=2*np.pi*60,
                                 inertia=model.mass.tolist()))
    return model,event,fine


def test_cross_model_same005_dynamic_path_exact_alignment_and_actual15_pass():
    model,event,fine=cross_fixture()
    row=study._cross_event(model,event,fine,fine.copy())
    assert row["qualified"] and row["actual_steps"]==1500
    for scope in ("8s","actual15"):
        assert row["intervals"][scope]["qualified"]
        assert row["intervals"][scope]["F_to_C0_pair"]["coi_normalized_rmse"]==0.
    short=copy.deepcopy(event);short["x"]=short["x"][:134]
    actual=study._cross_event(model,short,fine,fine.copy())
    assert actual["qualified"] and actual["actual_steps"]==123
    assert actual["intervals"]["actual15"]["actual_output_steps"]==123


@pytest.mark.parametrize("fault",("wrong_time","physical_offset","odd_fine_nan"))
def test_cross_model_rejects_misalignment_physics_and_unobserved_substep_nan(fault):
    model,event,fine=cross_fixture();changed=fine.copy()
    if fault=="wrong_time": changed=np.roll(changed,1,axis=0)
    elif fault=="physical_offset": changed[:,10:20]+=.1
    else: changed[1,0]=np.nan
    row=study._cross_event(model,event,fine,changed)
    assert not row["qualified"]
    if fault=="odd_fine_nan":
        assert not row["intervals"]["8s"]["full_C0_fine_finite"]
        assert row["intervals"]["8s"]["F_to_C0_pair"]["full_coarse_finite"]


def test_cross_model_requires_all297_ids_once_and_missing_cache_is_unresolved():
    ids=[str(i) for i in range(297)]
    study._exact_event_ids(ids,ids)
    for changed in (ids[:-1],ids[:-1]+["wrong"],ids[:-1]+[ids[0]]):
        with pytest.raises(ValueError): study._exact_event_ids(changed,ids)
    model,event,_=cross_fixture()
    with tempfile.TemporaryDirectory(prefix="exchange-cross-missing-") as folder:
        path=Path(folder)/"cross.json"
        result=study.cross_model_qualification(model,[event],{}, {},dict(fit_ids=ids,source={"fixture":True}),path)
        assert not result["qualified"] and result["status"]=="UNRESOLVED" and result["record_count"]==0
        assert result["predictions_recomputed"] is False


class Reports:
    def __init__(self): self.rows=[]
    def pytest_runtest_logreport(self,report):
        if report.when=="call" or report.failed:
            self.rows.append(dict(test=report.nodeid,outcome=report.outcome,error=str(report.longrepr) if report.failed else None))


if __name__=="__main__":
    tick=time.time();reports=Reports()
    preserved={str(p.relative_to(ROOT)):study.sha(p) for p in (ROOT/"experiments/force_exchange_model_20261028.py",
      ROOT/"tests/test_force_exchange_model_20261028.py",ROOT/"results/force_exchange_20261028/model_pre_freeze_qa.json",
      ROOT/"results/force_exchange_20261028/matched_control_cpu_qualification.json")}
    code=pytest.main([__file__,"-q","-p","no:cacheprovider"],plugins=[reports])
    unchanged=all(study.sha(ROOT/p)==digest for p,digest in preserved.items())
    result=dict(status="PASS_PURE_IMPLEMENTATION_QA" if code==0 and unchanged else "FAILED_PURE_IMPLEMENTATION_QA",
      exit_code=code,tests=reports.rows,wall_seconds=time.time()-tick,optimizer_calls=0,real_event_arrays_read=False,
      source_model_JSON_read=True,old_training_pt_loaded=False,cuda_executed=False,training_admission=False,
      old_legacy_drop_in_qualification="FAILED_NOT_ADOPTED",preserved=preserved,preserved_unchanged=unchanged,
      files={p:study.sha(ROOT/p) for p in ("experiments/force_exchange_study_20261028.py","tests/test_force_exchange_study_20261028.py","reports/force_exchange_20261028_protocol.md")},
      unverified=["real fit qualification","CPU/CUDA parity","8s backward cost","real72 Adam/RNG recovery",
                  "formal training and conditional seed","385/noise real reports","native ownership/recovery"])
    out=ROOT/"results/force_exchange_20261028/study_pre_freeze_qa.json"
    if out.exists():
        previous=json.loads(out.read_text());result["previous_attempts"]=previous.get("previous_attempts",[])+[{k:v for k,v in previous.items() if k!="previous_attempts"}]
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    raise SystemExit(code if unchanged else 1)
