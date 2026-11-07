"""Analytic and mock-only preparation; no registered trajectories or seeds run."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
import gftd_coupling_memory_20261107 as run


@pytest.mark.parametrize("intervals,simpson_integral",[(40,-1/192000),(80,-1/3072000)])
def test_boole_constant_speed_high_order_derivative_independent_counterexample(intervals,simpson_integral):
    t=np.linspace(0.,.1,intervals+1)
    tau=t/.1
    # Independently expanded derivative of v3, whose endpoint difference is zero.
    derivative=(-100*tau**4+200*tau**3-126*tau**2+26*tau-1)/.1
    weights=run.boole_weights(t)
    np.testing.assert_allclose(weights@derivative,0.,atol=2e-14,rtol=0)
    np.testing.assert_allclose(run.test_functions(t)[1][3],derivative,atol=5e-13,rtol=1e-13)
    simpson=np.ones(intervals+1)
    simpson[1:-1:2]=4
    simpson[2:-1:2]=2
    simpson*=.1/intervals/3
    np.testing.assert_allclose(simpson@derivative,simpson_integral,atol=2e-14,rtol=0)
    assert abs(.05/.1*(simpson@derivative))>1e-7
    for dv in run.test_functions(t)[1]:
        assert abs(.05/.1*(weights@dv))<1e-12


@pytest.mark.parametrize("step",run.STEPS)
@pytest.mark.parametrize("speed",[0.,.031])
def test_backward_p_and_forward_zero_memory_match_constant_forcing_analytic_solution(step,speed):
    t=np.linspace(0.,.1,int(round(.1/step))+1)
    omega=np.full((len(t),3),speed)
    p_end=np.array([.013,-.017,.019])
    f=run.functional(0.)
    p,rf=run.reconstruct_ports(t,omega,p_end,f)
    expected_p=np.exp(.3*(.1-t))[:,None]*p_end+np.expm1(.3*(.1-t))[:,None]*speed/.3
    expected_rf=.7/.3*(-np.expm1(-.3*t))[:,None]*np.full((1,3),speed)
    np.testing.assert_allclose(p,expected_p,atol=2e-14,rtol=1e-12)
    np.testing.assert_allclose(rf,expected_rf,atol=2e-14,rtol=1e-12)
    np.testing.assert_array_equal(p[-1],p_end)
    np.testing.assert_array_equal(rf[0],np.zeros(3))
    if speed>0:
        assert np.all(rf[-1]>0.)


def packet():
    return dict(theta_history=np.zeros((11,3)),omega_history=np.zeros((11,3)),history_times=np.linspace(0.,.1,11),p_end=np.zeros(3))


@pytest.mark.parametrize("illegal",["c","actual_k","true_r","excitation","E0","future","generator"])
def test_estimator_whitelist_rejects_generation_information(illegal):
    obs=packet()
    obs[illegal]=0
    with pytest.raises(ValueError,match="only theta"):
        run.build_system(obs,run.STEPS[0])


def algebra_system(c):
    Z=np.vstack([np.eye(3)*scale for scale in (1.,.3,.1,.2)])
    A=np.arange(1.,13.)/100
    u=np.array([.2,-.1,.4])
    b=A*c+Z@u
    return A,Z,b,u


@pytest.mark.parametrize("c",[-.7,0.,1.7,3.1])
def test_complete_multifunction_global_linear_solution_without_clipping(c):
    A,Z,b,u=algebra_system(c)
    result=run.solve_linear(A,Z,b)
    assert result["identified"]
    np.testing.assert_allclose(result["c_hat"],c,atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(result["rho_hat"],.1*u,atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(result["parameter_residual"],0.,atol=1e-13)
    np.testing.assert_allclose(result["normal_residual"],0.,atol=1e-13)
    assert result["profile_curvature"]>0.


def test_single_function_cannot_identify_shared_c_even_across_events():
    A,Z,b,_=algebra_system(.83)
    systems=[dict(A=A,Z=Z,b=b,p_term=np.zeros(12),rf_term=np.zeros(12)) for _ in range(3)]
    a,z,target=run.combine(systems,"single_test")
    assert z.shape==(9,9)
    result=run.solve_linear(a,z,target)
    assert result["c_hat"] is None
    assert result["residual_column_ratio"]<=1e-10
    assert all(not d["identifiable"] for d in result["threshold_diagnostics"])


def test_flat_zero_column_is_not_a_zero_parameter_estimate():
    A,Z,b,_=algebra_system(0.)
    result=run.solve_linear(np.zeros_like(A),Z,np.zeros_like(b))
    assert result["c_hat"] is None
    assert result["profile_curvature"]==0.
    assert all(row["loss"]==0. for row in result["profile"])
    # A nonzero observation-induced column must be judged by matrices, without an E0 label.
    perturbed=run.solve_linear(A,Z,b+A*2.)
    assert perturbed["identified"] and perturbed["c_hat"] is not None


def test_omitted_port_dictionary_signs():
    A,Z,b,_=algebra_system(.83)
    p=np.linspace(.1,.4,12)
    rf=np.linspace(-.03,.07,12)
    system=dict(A=A,Z=Z,b=b,p_term=p,rf_term=rf)
    _,_,omitp=run.combine([system],"omit_p")
    _,omitr_columns,omitr=run.combine([system],"omit_r")
    np.testing.assert_array_equal(omitp,b-p)
    np.testing.assert_array_equal(omitr,b+rf)
    assert omitr_columns.shape==(12,0)


def test_build_zero_observations_returns_natural_flat_negative_calibration():
    system=run.build_system(packet(),.0025)
    result=run.solve_linear(system["A"],system["Z"],system["b"])
    assert result["c_hat"] is None
    assert result["profile_curvature"]==0.


def test_registered_seed_and_schedule_are_mocked_not_consumed(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"initial_conditions",lambda:[(0.,"mock",np.zeros(15)) for _ in range(9)])
    monkeypatch.setattr(run,"rk4",lambda f,state,step:state.copy())
    seeds=[]
    monkeypatch.setattr(run.np.random,"PCG64",lambda seed:seed)
    def generator(seed):
        seeds.append(seed)
        return SimpleNamespace(standard_normal=lambda shape:np.full(shape,seed/1e6))
    monkeypatch.setattr(run.np.random,"Generator",generator)
    counts=dict(generation_attempts=0,generated_trajectories=0,observation_packages=0)
    _,observed_path,plan_path=run.generate(tmp_path,counts)
    observed=json.loads(observed_path.read_text())
    plan=json.loads(plan_path.read_text())
    assert seeds==list(range(140000,140009))
    assert counts==dict(generation_attempts=18,generated_trajectories=18,observation_packages=27)
    for row in plan:
        packet=observed[row["input_id"]]
        expected=np.full((11,6),.1*row["sigma"]*row["seed"]/1e6)
        np.testing.assert_allclose(np.column_stack((packet["theta_history"],packet["omega_history"])),expected,rtol=1e-15)


def test_A_failure_never_executes_B_and_durably_records_counts(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"ROOT",tmp_path)
    monkeypatch.setattr(run,"OUTPUT",tmp_path)
    monkeypatch.setattr(run,"digest",lambda path:run.PROTOCOL_SHA256)
    run.write_json(tmp_path/"execution_release.json",dict(released=True,source_sha256={},action_id="mock"))
    def generate(output,counts):
        path=tmp_path/"mock_truth.json"
        run.write_json(path,[])
        return path,tmp_path/"absent_obs.json",tmp_path/"absent_plan.json"
    monkeypatch.setattr(run,"generate",generate)
    monkeypatch.setattr(run,"qualify_A",lambda *args:dict(passed=False,reason="analytic mock failure"))
    monkeypatch.setattr(run,"fit_B",lambda *args:pytest.fail("B ran after failed A"))
    run.main()
    receipt=json.loads((tmp_path/"execution_receipt.json").read_text())
    assert receipt["status"]=="STOPPED_A_FAILED_B_NOT_RUN"
    assert receipt["actual_counts"]["linear_problems"]==0
    assert (tmp_path/"qualification_A.json").exists()


@pytest.mark.parametrize("fine_error,expected",[(0.,True),(1e-5,False)])
def test_A_reconstruction_gate_uses_fine_grid_while_reporting_both(monkeypatch,fine_error,expected):
    event=dict(index=0,c=1.,excitation="E1",initial=np.zeros(15),grids={})
    for step in run.STEPS:
        t=np.linspace(0.,.1,int(round(.1/step))+1)
        event["grids"][str(step)]=dict(t=t,states=np.zeros((len(t),15)))
    def terms(*args):
        # Both nonzero ports cancel exactly, satisfying the algebra/weak gates.
        return dict(A=np.zeros(12),inertial=np.zeros(12),damping=np.zeros(12),
                    p=np.full(12,.01),r=np.full(12,.01))
    monkeypatch.setattr(run,"weak_terms",terms)
    def reconstruct(t,omega,p_end,f):
        error=1e-5 if len(t)==41 else fine_error
        return np.full_like(omega,error),np.full_like(omega,error)
    monkeypatch.setattr(run,"reconstruct_ports",reconstruct)
    counts=dict(A_grid_attempts=0,A_grid_checks=0)
    qualification=run.qualify_A([event],counts)
    assert qualification["passed"] is expected
    records=qualification["records"][0]["grids"]
    assert records[0]["p_reconstruction_max"]>1e-7
    assert records[0]["r_reconstruction_max"]>1e-7
    assert records[0]["passed"] is True
    assert records[1]["passed"] is expected


def test_score_requires_frozen_estimate_hash_before_truth_read(tmp_path):
    path=tmp_path/"estimates.json"
    path.write_text("{}")
    with pytest.raises(ValueError,match="estimates changed"):
        run.score_frozen(path,"wrong",tmp_path/"missingtruth",tmp_path/"missingplan")
