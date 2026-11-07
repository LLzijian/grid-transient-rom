"""Interface and optimizer accounting checks with the numerical optimizer mocked."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import gftd_initialization_20261107 as run


def observation():
    return dict(theta_history=np.zeros((11,3)), omega_history=np.zeros((11,3)),
                history_times=np.linspace(-.1,0,11), p_end=np.array([.01,-.02,.03]), sigma=0.)


def solver_stub(fun, x, **kwargs):
    residual = fun(x)
    return SimpleNamespace(success=True, status=1, message="mock optimizer", fun=residual,
                           x=np.array(x), grad=np.zeros(len(x)), optimality=0., nfev=1, njev=0)


@pytest.mark.parametrize("extra", ["future", "case", "true_a", "generator", "initial_state"])
def test_optimizer_rejects_non_observations_before_solver(monkeypatch, extra):
    obs = observation()
    obs[extra] = 0
    monkeypatch.setattr(run, "least_squares", lambda *args, **kwargs: pytest.fail("solver received illegal data"))
    with pytest.raises(ValueError, match="only theta"):
        run.infer_endpoint(obs, run.functional(.5), amplitude_free=False, initial=np.zeros(3))


def test_endpoint_keeps_actual_measured_p_and_same_nested_model():
    obs = run.validate_observation(observation())
    r = np.array([.3,-.2,.1])
    simple = run.endpoint(obs,r,False)
    joint = run.endpoint(obs,np.r_[np.zeros(3),r],True)
    np.testing.assert_array_equal(simple,joint)
    np.testing.assert_array_equal(simple[6:9],obs["p_end"])
    f = run.functional(.5)
    np.testing.assert_array_equal(run.backward_history(f,simple,.0025),run.backward_history(f,joint,.0025))


def test_absolute_bounded_jacobian_interior_and_second_order_boundary():
    points = []
    def residual(x):
        assert np.all(np.abs(x)<=2)
        points.append(x.copy())
        return np.array([x[0]**2+2*x[1],x[1]**2])
    for x in (np.array([0.,.3]),np.array([-2.,2.]),np.array([-2.+1e-6,2.-1e-6])):
        actual = run.bounded_jacobian(residual,x)
        expected = np.array([[2*x[0],2.],[0.,2*x[1]]])
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-9)
    assert len(points)>0


def test_actual_budget_includes_jacobian_and_preserves_best(monkeypatch):
    monkeypatch.setattr(run,"backward_history",lambda f,state,step: np.tile(state,(11,1)))
    def exhausting(fun,x,**kwargs):
        while True:
            kwargs["jac"](x)
    monkeypatch.setattr(run,"least_squares",exhausting)
    result=run.infer_endpoint(observation(),run.functional(.5),amplitude_free=True,initial=np.zeros(6))
    assert result["status"]=="budget_limited"
    assert result["actual_residual_calls"]==600
    assert result["best"] is not None
    assert result["projected_gradient_kkt"] is None


def test_noise_objective_scaling_and_raw_report(monkeypatch):
    monkeypatch.setattr(run,"backward_history",lambda f,state,step: np.tile(state,(11,1)))
    monkeypatch.setattr(run,"history_residual",lambda f,predicted,obs: np.full(66,.2))
    seen=[]
    def solver(fun,x,**kwargs):
        result=solver_stub(fun,x,**kwargs)
        seen.append(result.fun.copy())
        return result
    monkeypatch.setattr(run,"least_squares",solver)
    for sigma in (0.,1e-6,1e-4):
        obs=observation()
        obs["sigma"]=sigma
        result=run.infer_endpoint(obs,run.functional(.5),amplitude_free=False,initial=np.zeros(3))
        np.testing.assert_allclose(seen[-1],.2/max(sigma,1e-8))
        np.testing.assert_allclose(result["best"]["history_rms"],.2)


def test_isolated_truth_changes_cannot_change_fit(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"least_squares",solver_stub)
    monkeypatch.setattr(run,"backward_history",lambda f,state,step: np.tile(state,(11,1)))
    path=tmp_path/"isolated_truth.json"
    obs=observation()
    path.write_text(json.dumps({"secret": [1000]}))
    a=run.infer_endpoint(obs,run.functional(.5),amplitude_free=True,initial=np.zeros(6))
    path.write_text(json.dumps({"secret": [-1e99],"future": [999]}))
    b=run.infer_endpoint(obs,run.functional(.5),amplitude_free=True,initial=np.zeros(6))
    for key in ("u","state"):
        np.testing.assert_array_equal(a["best"][key],b["best"][key])
    assert a["best"]["observation_loss"]==b["best"]["observation_loss"]
    assert a["actual_residual_calls"]==b["actual_residual_calls"]


def test_failure_does_not_discard_other_inputs_and_guard_uses_no_extra_integration(monkeypatch,tmp_path):
    calls=[]
    def fake_fit(obs,f,**kwargs):
        calls.append(kwargs)
        u=np.asarray(kwargs["initial"])
        state=run.endpoint(run.validate_observation(obs),u,kwargs["amplitude_free"])
        return dict(status="failed" if len(calls)==1 else "converged",best=dict(
            u=u,state=state,observation_loss=0. if len(u)==3 else 1.,history_rms=0.))
    monkeypatch.setattr(run,"infer_endpoint",fake_fit)
    monkeypatch.setattr(run,"numerical_check",lambda *args: dict(history_main_fine_rms=0.,passed=True))
    results=run.fit_observations({"first":observation(),"second":observation()},output=tmp_path)
    assert len(calls)==6
    assert results["records"]["first"]["simple"]["selected_index"] is None
    joint=results["records"]["second"]["joint"]
    assert joint["selected_index"]==0
    assert joint["both_starts_worse_than_simple_feasible"]
    assert joint["simple_feasible_observation_loss"]==0.
    saved=json.loads((tmp_path/"fits.partial.json").read_text())
    assert set(saved["records"])=={"first","second"}
    assert not (tmp_path/"fits.partial.tmp.json").exists()


def test_scoring_serialized_candidates_preserves_two_start_comparison(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"sampled",lambda f,state,duration,step: np.tile(state,(101,1)))
    monkeypatch.setattr(run,"backward_history",lambda f,state,step: np.tile(state,(11,1)))
    state=np.zeros(15)
    task=dict(best=dict(state=state,history_rms=0.),history_step=.0025)
    base=dict(candidates=[task],selected_index=0,numerical=dict(passed=True),fine_refit=None)
    joint=dict(base,candidates=[task,task],simple_feasible_observation_loss=0.,
               both_starts_worse_than_simple_feasible=False)
    fits=tmp_path/"fits.json"
    obs=tmp_path/"observations.json"
    truth=tmp_path/"truth.json"
    run.write_json(fits,dict(records={"input":dict(simple=base,joint=joint)}))
    measured=observation()
    measured["p_end"]=np.zeros(3)
    run.write_json(obs,{"input":measured})
    run.write_json(truth,{"input":dict(case=0,sigma=0.,history=np.zeros((11,15)),future=np.zeros((101,15)),
                                      endpoint_qualification=dict(passed=True),reference_numerical=dict(passed=True))})
    report=run.score_frozen_fits(fits,run.digest(fits),obs,truth)
    row=report["scores"][0]
    assert row["classification"]=="both_pass"
    assert row["equilibrium_negative_calibration"]
    assert row["groups"]["joint"]["joint_start_future_difference"]["joint_rms"]==0.


def test_truth_scoring_requires_final_fit_hash(tmp_path):
    fits=tmp_path/"fits.json"
    fits.write_text("{}")
    with pytest.raises(ValueError,match="changed"):
        run.score_frozen_fits(fits,"wrong",tmp_path/"absent_obs",tmp_path/"absent_truth")
