"""Data-free accounting and interface checks with mocked numerical optimization."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
import gftd_endpoint_denoising_20261107 as run


def observation(sigma=1e-4):
    return dict(theta_history=np.zeros((11,3)),omega_history=np.zeros((11,3)),
                p_end=np.array([.1,-.2,.3]),history_times=np.linspace(-.1,0,11),sigma=sigma)


def result_stub(fun,u,**kwargs):
    value=fun(u)
    return SimpleNamespace(x=u.copy(),fun=value,success=True,status=1,message="mock",grad=np.ones(len(u)),
                           optimality=1.,nfev=1,njev=0)


def test_fixed_gram_schmidt_basis_and_endpoint_gauge_hard_p():
    f=run.functional(.5)
    Q=run.angle_basis(f.mass)
    np.testing.assert_allclose(Q.T@Q,np.eye(2),atol=1e-15)
    np.testing.assert_allclose(f.coi_weights@Q,0.,atol=1e-15)
    for column in Q.T:
        assert column[np.flatnonzero(np.abs(column)>1e-14)[0]]>0
    obs=run.validate_observation(observation())
    old=run.endpoint(obs,f,np.array([.2,-.1,.3]),False)
    same=run.endpoint(obs,f,np.r_[np.zeros(5),[.2,-.1,.3]],True)
    np.testing.assert_array_equal(old,same)
    np.testing.assert_array_equal(f.rhs(old),f.rhs(same))
    corrected=run.endpoint(obs,f,np.array([4.,-3.,1.,2.,3.,.2,-.1,.3]),True)
    np.testing.assert_allclose(f.coi_weights@(corrected[:3]-same[:3]),0.,atol=1e-18)
    np.testing.assert_array_equal(corrected[6:9],obs["p_end"])
    np.testing.assert_array_equal(corrected[9:12],np.zeros(3))


@pytest.mark.parametrize("extra",["case","seed","future","true_state","history_p"])
def test_white_list_excludes_truth_before_optimizer(monkeypatch,extra):
    obs=observation()
    obs[extra]=123
    monkeypatch.setattr(run,"least_squares",lambda *args,**kwargs:pytest.fail("illegal observation reached optimizer"))
    with pytest.raises(ValueError):
        run.infer_endpoint(obs,run.functional(.5),np.zeros(3),smooth=True)


def test_mixed_bounds_and_absolute_jacobian():
    low,high=run.bounds(True)
    np.testing.assert_array_equal(high,[5.,5.,5.,5.,5.,2.,2.,2.])
    evaluated=[]
    def residual(u):
        assert np.all(u>=low) and np.all(u<=high)
        evaluated.append(u.copy())
        return u**2
    u=np.array([-5.,5.,0.,1.,-2.,-2.,2.,.1])
    actual=run.jacobian(residual,u,True)
    np.testing.assert_allclose(actual,np.diag(2*u),atol=1e-9,rtol=1e-9)
    assert len(evaluated)>16


def test_1200_actual_calls_include_feasible_start_initial_and_jacobian(monkeypatch):
    monkeypatch.setattr(run,"backward_history",lambda f,state,step:np.tile(state,(11,1)))
    def solver(fun,u,**kwargs):
        while True:
            kwargs["jac"](u)
    monkeypatch.setattr(run,"least_squares",solver)
    task=run.infer_endpoint(observation(),run.functional(.5),np.zeros(3),smooth=True)
    assert task["actual_residual_calls"]==1200
    assert task["status"]=="budget_limited" and task["objective_guard_passed"]
    assert task["projected_gradient_kkt"] is None
    assert task["jacobian"] is not None


def test_selected_best_cannot_be_overwritten_and_kkt_marks_gradient_point(monkeypatch):
    monkeypatch.setattr(run,"backward_history",lambda f,state,step:np.tile(state,(11,1)))
    monkeypatch.setattr(run,"history_residual",lambda f,states,obs:np.full(66,states[-1,12]))
    def worse_solver(fun,u,**kwargs):
        bad=u.copy()
        bad[-1]=1.
        result=result_stub(fun,bad,**kwargs)
        return result
    monkeypatch.setattr(run,"least_squares",worse_solver)
    task=run.infer_endpoint(observation(),run.functional(.5),np.zeros(3),smooth=True)
    assert task["best"]["observation_loss"]==0.
    assert task["objective_guard_passed"]
    assert task["gradient_evaluated_at"][-1]==1.
    assert task["projected_gradient_kkt"] is None
    assert task["actual_residual_calls"]==3


def test_fine_guard_recomputes_same_step_parent_and_initial_within_budget(monkeypatch):
    calls=[]
    def backward(f,state,step):
        calls.append(step)
        return np.tile(state,(11,1))
    monkeypatch.setattr(run,"backward_history",backward)
    monkeypatch.setattr(run,"history_residual",lambda f,states,obs:np.full(66,.00125))
    monkeypatch.setattr(run,"least_squares",result_stub)
    task=run.infer_endpoint(observation(),run.functional(.5),np.zeros(3),smooth=False,
                            initial=np.array([.1,0.,0.]),step=.00125)
    assert calls==[.00125]*3
    assert task["actual_residual_calls"]==3
    assert task["objective_guard_passed"]
    np.testing.assert_allclose(task["parent_same_step_loss"],66*.00125**2/2)


def fake_task(obs,f,parent_u,*,smooth,initial=None,step=.0025):
    u=np.r_[np.zeros(5),parent_u] if smooth else np.asarray(parent_u)
    return dict(status="converged",best=dict(u=u,state=run.endpoint(run.validate_observation(obs),f,u,smooth),
        observation_loss=0.,history_rms=0.),actual_residual_calls=1,history_step=step,
        parent_same_step_loss=0.,objective_guard_passed=True)


def test_noiseless_reused_two_groups_planned_checkpoint_and_truth_isolation(monkeypatch,tmp_path):
    calls=[]
    def solver(*args,**kwargs):
        calls.append(kwargs)
        return fake_task(*args,**kwargs)
    monkeypatch.setattr(run,"infer_endpoint",solver)
    monkeypatch.setattr(run,"evaluate_numerical",lambda *args:(dict(passed=True,history_main_fine_rms=0.),None))
    parent=dict(records={key:dict(main=dict(best=dict(u=np.zeros(3),observation_loss=0.))) for key in ("zero","noise")})
    hidden=tmp_path/"isolated_truth.json"
    hidden.write_text('{"future":[1e99]}')
    packets={"zero":observation(0.),"noise":observation()}
    first=run.fit_packages(packets,parent,tmp_path)
    assert len(calls)==2
    assert all(first["records"]["zero"][group]["status"]=="reused_parent" for group in run.GROUPS)
    hidden.write_text('{"future":[-1e99]}')
    second=run.fit_packages(packets,parent)
    for group in run.GROUPS:
        np.testing.assert_array_equal(first["records"]["noise"][group]["main"]["best"]["state"],
                                      second["records"]["noise"][group]["main"]["best"]["state"])
    checkpoint=json.loads((tmp_path/"fits.partial.json").read_text())
    assert set(checkpoint["records"]["zero"])==set(run.GROUPS)


def metric(value=.0005):
    return dict(joint_rms=value,peak=value,theta_rms=value,omega_rms=value,common_omega_rms=value,p_rms=value)


def adoption_rows():
    rows=[]
    for i in range(132):
        row=dict(input_id=f"input_{i:03d}",case=i//33,sigma=0. if i%33==0 else 1e-4 if i%2==0 else 1e-6)
        for group in ("parent",)+run.GROUPS:
            row[group]=dict(classification="pass",metrics=dict(future_main=metric(.0004 if group=="smooth" else .0005),
                           future_fine=metric(.0004 if group=="smooth" else .0005)))
        if i==72:
            row["parent"]["classification"]="prediction_fail"
        rows.append(row)
    # Use exactly the protocol's 64 high-noise inputs.
    for i,row in enumerate(rows):
        within=i%33
        row["sigma"]=0. if within==0 else 1e-4 if within%2==0 else 1e-6
    return rows


def test_complete_adoption_criteria_and_main_fine_mean_consistency():
    rows=adoption_rows()
    assert run.adoption_conditions(rows,"future_main")["meets_all"]
    bad=copy.deepcopy(rows)
    bad[1]["smooth"]["classification"]="prediction_fail"
    report=run.adoption_conditions(bad,"future_main")
    assert not report["conditions"]["no_parent_regressions"]
    assert not report["conditions"]["no_anchor_regressions"]
    assert not report["meets_all"]
    for row in rows:
        if row["sigma"]==1e-4:
            row["smooth"]["metrics"]["future_fine"]=metric(.0006)
    assert run.adoption_conditions(rows,"future_main")["meets_all"]
    assert not run.adoption_conditions(rows,"future_fine")["meets_all"]
    assert not run.adoption_conditions(rows[:-1],"future_main")["meets_all"]


def test_score_real_json_with_actual_parent_metrics_noiseless_exact_reuse(monkeypatch,tmp_path):
    import gftd_noise_confirmation_20261107 as scoring
    monkeypatch.setattr(scoring,"sampled",lambda f,state,duration,step:np.tile(state,(101,1)))
    monkeypatch.setattr(scoring,"backward_history",lambda f,state,step:np.tile(state,(11,1)))
    obs=observation(0.)
    paths={key:tmp_path/(key+".json") for key in ("fits","observations","parent_fits","parent_summary","truth","plan")}
    parent_row=dict(input_id="input_000",case=0,rep=None,sigma=0.,classification="pass",
                    main=dict(future_main=metric(0.),future_fine=metric(0.)),unique_parent_field=[1,2,3])
    noisy=observation()
    task=fake_task(noisy,run.functional(.5),np.zeros(3),smooth=False)
    task["frozen_parent_objective_guard"]=True
    state=task["best"]["state"]
    run.write_json(paths["observations"],{"input_000":obs,"input_001":noisy})
    run.write_json(paths["parent_fits"],{})
    noisy_parent=dict(parent_row,input_id="input_001",case=0,rep=0,sigma=1e-4)
    run.write_json(paths["parent_summary"],dict(rows=[parent_row,noisy_parent]))
    run.write_json(paths["truth"],{"input_001":dict(history=np.tile(state,(11,1)),future=np.tile(state,(101,1)),
                    reference_numerical=dict(passed=True),endpoint_qualification=dict(passed=True))})
    run.write_json(paths["plan"],[dict(input_id="input_000",case=0,rep=None,sigma=0.),
                                dict(input_id="input_001",case=0,rep=0,sigma=1e-4)])
    fits=dict(input_sha256={"observations":run.digest(paths["observations"]),"parent_fits":run.digest(paths["parent_fits"])},
        records={"input_000":{group:dict(status="reused_parent") for group in run.GROUPS},
                 "input_001":{group:dict(status="completed",main=task,fine=None,numerical=dict(passed=True),
                                         fine_numerical=None) for group in run.GROUPS}})
    run.write_json(paths["fits"],fits)
    report=run.score_frozen(paths["fits"],run.digest(paths["fits"]),paths["observations"],paths["parent_fits"],
                            paths["parent_summary"],paths["truth"],paths["plan"])
    assert report["noiseless_equilibrium_calibration"]
    for group in run.GROUPS:
        assert report["rows"][0][group]["reused_parent_score"]==parent_row
        scored=report["rows"][1][group]
        assert scored["classification"]=="pass"
        np.testing.assert_array_equal(scored["scored"]["main"]["p_constraint_error"],np.zeros(3))
        np.testing.assert_array_equal(scored["scored"]["main"]["endpoint_estimation_scaled_error"],np.zeros(6))
        assert "endpoint_observation_scaled_noise" not in scored["scored"]["main"]
        assert scored["metrics"]["future_fine"]["common_omega_rms"]==0.
    assert not report["worth_independent_confirmation"]


def test_fine_cannot_cover_original_nonhistory_failure():
    task=dict(status="converged",best=dict(u=np.zeros(3)),objective_guard_passed=True,
              frozen_parent_objective_guard=True)
    bad=dict(passed=False,history_main_fine_rms=1e-6,roundtrip_scaled_max=[1e-6,0.],
             future_main_fine_rms=0.,future_main_fine_peak=0.)
    record=dict(main=task,fine=task,numerical=bad,fine_numerical=dict(passed=True))
    scored={name:dict(p_constraint_error=np.zeros(3),future_main_pass=True,future_fine_pass=True)
            for name in ("main","fine")}
    truth=dict(reference_numerical=dict(passed=True),endpoint_qualification=dict(passed=True))
    assert run.classified_prediction(record,scored,truth)=="numerically_unresolved"


def test_hash_mismatch_rejected_before_truth_read(tmp_path):
    path=tmp_path/"fits.json"
    path.write_text("{}")
    with pytest.raises(ValueError,match="fits changed"):
        run.score_frozen(path,"wrong",path,path,path,path,path)
