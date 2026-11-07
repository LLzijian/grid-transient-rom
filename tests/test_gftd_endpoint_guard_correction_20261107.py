"""Pure-data qualification of the append-only guard correction."""
import copy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import gftd_endpoint_guard_correction_20261107 as run


def evidence():
    return {name:True for name in run.EVIDENCE_KEYS}


def observation():
    return dict(theta_history=[[0.]*3 for _ in range(11)],omega_history=[[0.]*3 for _ in range(11)],
                history_times=[-.1+.01*i for i in range(11)],p_end=[0.]*3,sigma=1e-4)


def parent_task():
    return dict(best=dict(u=[0.]*3,state=[0.]*15,observation_loss=math.nextafter(1.,0.)),
                status="converged",nonfinite_evaluations=0,history_step=.0025,optimizer_residual_divisor=1e-4)


def child_task(group="anchor",false=True):
    u=[0.]*(3 if group=="anchor" else 8)
    return dict(best=dict(u=u.copy(),state=[0.]*15,observation_loss=1.),initial=u.copy(),feasible_parent=u.copy(),
                status="converged",nonfinite_evaluations=0,history_step=.0025,optimizer_residual_divisor=1e-4,
                objective_guard_passed=True,frozen_parent_objective_guard=not false,
                parent_same_step_loss=1.,initial_same_step_loss=1.)


def reference():
    return dict(endpoint_qualification=dict(passed=True),reference_numerical=dict(passed=True))


def numerical():
    return dict(passed=True,roundtrip_scaled_max=[0.,0.],history_main_fine_rms=0.,
                future_main_fine_rms=0.,future_main_fine_peak=0.)


def metrics(value):
    return dict(joint_rms=value,peak=value,common_omega_rms=value,theta_rms=value,omega_rms=value,p_rms=value)


def group_score(value,classification="pass"):
    prediction=value<=1e-3
    return dict(classification=classification,metrics=dict(future_main=metrics(value),future_fine=metrics(value)),
                scored=dict(main=dict(p_constraint_error=[0.]*3,future_main=metrics(value),future_fine=metrics(value),
                                     future_main_pass=prediction,future_fine_pass=prediction),fine=None),
                refit_metrics=None)


def fake_dataset():
    child,parent,obs,refs,rows={},{},{},{},[]
    for index in range(132):
        key=f"input_{index:03d}"
        noiseless=index%33==0
        sigma=0. if noiseless else 1e-4 if index%33%2==0 else 1e-6
        row=dict(input_id=key,case=index//33,sigma=sigma,parent=group_score(.002 if index==72 else .0005),
                 anchor=group_score(.002 if index==72 else .0005),smooth=group_score(.0004))
        if index==72:
            row["parent"]["classification"]=row["anchor"]["classification"]="prediction_fail"
        child[key]={}
        parent[key]=dict(main=parent_task())
        obs[key]=observation()
        refs[key]=reference()
        for group in ("anchor","smooth"):
            if noiseless:
                child[key][group]=dict(main=None,status="reused_parent")
                continue
            affected=(key,group) in run.EXPECTED_IMPACT
            child[key][group]=dict(main=child_task(group,affected),fine=None,numerical=numerical(),numerical_error=None)
            if affected:
                row[group]["classification"]="numerically_unresolved"
        rows.append(row)
    summary=dict(rows=rows,comparisons=dict(unchanged=[1.,2.]),worst_errors=dict(unchanged=3.),
                 noiseless_equilibrium_calibration=True)
    run.aggregate(summary)
    assert summary["worth_independent_confirmation"] is False
    return dict(records=child),dict(records=parent),summary,obs,refs


def test_binary64_uses_bits_shape_finiteness_without_tolerance():
    assert not run.same_bits(0.,-0.)
    assert not run.same_bits([0.],[0.,0.])
    assert not run.same_bits(1.,math.nextafter(1.,math.inf))
    assert not run.same_bits(float("nan"),float("nan"))
    assert not run.same_bits([True],[1.])


@pytest.mark.parametrize("group",["anchor","smooth"])
def test_identity_certificate_never_uses_prediction_or_ulp_threshold(group):
    task=child_task(group)
    old=parent_task()
    old["best"]["observation_loss"]=.5
    result=run.identity_certificate(task,old,observation(),observation(),group,evidence())
    assert result["same_point_certificate"]
    assert result["arithmetic_facts"]["ulp_is_admission_condition"] is False
    assert result["original_frozen_parent_objective_guard"] is False
    assert result["effective_guard"] is True


@pytest.mark.parametrize("change",["state","u","dt","scale","loss","same_guard","nonfinite","obs","source"])
def test_one_ulp_or_other_identity_mismatch_stays_unresolved(change):
    task=child_task()
    obs=observation()
    ev=evidence()
    if change=="state":
        task["best"]["state"][0]=math.nextafter(0.,math.inf)
    elif change=="u":
        task["best"]["u"][0]=math.nextafter(0.,math.inf)
    elif change=="dt":
        task["history_step"]=math.nextafter(.0025,math.inf)
    elif change=="scale":
        task["optimizer_residual_divisor"]=math.nextafter(1e-4,math.inf)
    elif change=="loss":
        task["initial_same_step_loss"]=math.nextafter(1.,math.inf)
    elif change=="same_guard":
        task["objective_guard_passed"]=False
    elif change=="nonfinite":
        task["best"]["state"][0]=float("nan")
    elif change=="obs":
        obs["theta_history"][0][0]=math.nextafter(0.,math.inf)
    else:
        ev["common_sources"]=False
    certificate=run.identity_certificate(task,parent_task(),obs,observation(),"anchor",ev)
    assert not certificate["same_point_certificate"]
    assert certificate["failed_identity_conditions"]


def test_same_point_prediction_failure_gets_failure_not_success():
    task=child_task()
    cert=run.identity_certificate(task,parent_task(),observation(),observation(),"anchor",evidence())
    assert cert["same_point_certificate"]
    record=dict(main=task,fine=None,numerical=numerical(),numerical_error=None)
    result=run.numerical_qualification(record,group_score(.002),reference())
    assert result==dict(passed=True,classification="prediction_fail")
    record["numerical"]["roundtrip_scaled_max"][0]=1e-6
    assert run.numerical_qualification(record,group_score(.0001),reference())["classification"]=="numerically_unresolved"


def test_parent_budget_limited_does_not_certify_rescaled_success_arithmetic_path():
    parent=parent_task()
    parent["status"]="budget_limited"
    result=run.identity_certificate(child_task(),parent,observation(),observation(),"anchor",evidence())
    assert result["same_point_certificate"] is False
    assert "parent_success_rescaled_path" in result["failed_identity_conditions"]


def test_scans_all_256_and_changes_only_expected_classification_and_aggregates(monkeypatch):
    child,parent,summary,obs,refs=fake_dataset()
    before=copy.deepcopy((child,parent,summary,obs))
    result=run.audit(child,parent,summary,obs,copy.deepcopy(obs),evidence(),references=refs)
    assert result["scan_count"]==256
    assert {(change["input_id"],change["group"]) for change in result["changes"]}==run.EXPECTED_IMPACT
    assert result["original_worth_independent_confirmation"] is False
    assert result["summary"]["worth_independent_confirmation"] is True
    assert result["summary"]["comparisons"]==summary["comparisons"]
    assert result["summary"]["worst_errors"]==summary["worst_errors"]
    assert result["summary"]["rows"][72]["anchor"]["classification"]=="prediction_fail"
    assert result["classification_counts"]["anchor"]["pass"]==131
    assert result["paired_prediction_outcomes"]["anchor"]["counts"]["repaired"]==1
    assert (child,parent,summary,obs)==before


def test_additional_impact_pauses_appendix_publication():
    child,parent,summary,obs,refs=fake_dataset()
    child["records"]["input_044"]["anchor"]["main"]["frozen_parent_objective_guard"]=False
    summary["rows"][44]["anchor"]["classification"]="numerically_unresolved"
    with pytest.raises(ValueError,match="requires independent review"):
        run.audit(child,parent,summary,obs,copy.deepcopy(obs),evidence(),references=refs)


def test_extra_false_guard_identity_pauses_even_if_other_numerical_failure_remains():
    child,parent,summary,obs,refs=fake_dataset()
    extra=child["records"]["input_044"]["anchor"]
    extra["main"]["frozen_parent_objective_guard"]=False
    extra["numerical"]["roundtrip_scaled_max"]=[1e-6,0.]
    summary["rows"][44]["anchor"]["classification"]="numerically_unresolved"
    with pytest.raises(ValueError,match="identity impact"):
        run.audit(child,parent,summary,obs,copy.deepcopy(obs),evidence(),references=refs)


def test_shared_model_factory_signature_requires_same_fixed_argument():
    source="def fit_packages(x):\n    f=functional(.5)\n    return f\n"
    assert run.functional_call_signature(source)==run.functional_call_signature(source)
    assert run.functional_call_signature(source)!=run.functional_call_signature(source.replace(".5",".4"))
