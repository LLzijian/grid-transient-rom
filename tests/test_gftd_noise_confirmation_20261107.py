"""Mock-optimizer qualification for the fixed independent-direction confirmation."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
import gftd_noise_confirmation_20261107 as run


def all_rows(classification="pass"):
    return [dict(case=case, sigma=sigma, rep=rep, classification=classification)
            for case in range(4) for sigma in (1e-6,1e-4) for rep in range(16)]


def test_exact_statistics_at_zero_and_sixteen_successes():
    failed=run.statistical_summary(all_rows("prediction_fail"))
    passed=run.statistical_summary(all_rows())
    assert failed["simultaneous_bound_available"]
    assert passed["simultaneous_bound_available"]
    for layer in failed["layers"]:
        np.testing.assert_allclose(layer["cp95"],[0.,1-.025**(1/16)])
        assert layer["simultaneous_failure_upper95"]==1.
    for layer in passed["layers"]:
        np.testing.assert_allclose(layer["cp95"],[.025**(1/16),1.])
        np.testing.assert_allclose(layer["simultaneous_failure_upper95"],1-(.05/8)**(1/16))


@pytest.mark.parametrize("incomplete",["missing","planned","not_run"])
def test_incomplete_layers_never_gain_failure_counts_or_intervals(incomplete):
    rows=all_rows()
    if incomplete=="missing":
        rows.pop()
    else:
        rows[-1]["classification"]=incomplete
    report=run.statistical_summary(rows)
    last=report["layers"][-1]
    assert not last["complete"]
    assert last["cp95"] is None
    assert last["counts"]["prediction_fail"]==0
    assert last["actual_results"]==15
    assert not report["simultaneous_bound_available"]
    assert all(layer["simultaneous_failure_upper95"] is None for layer in report["layers"])
    assert all(layer["cp95"] is not None for layer in report["layers"][:-1])


def test_unresolved_is_a_separate_count_and_not_success():
    rows=all_rows()
    rows[0]["classification"]="numerically_unresolved"
    rows[1]["classification"]="optimization_no_output"
    report=run.statistical_summary(rows)
    layer=report["layers"][0]
    assert layer["complete"] and layer["successes"]==14
    assert layer["counts"]["prediction_fail"]==0
    assert layer["counts"]["numerically_unresolved"]==1
    assert layer["counts"]["optimization_no_output"]==1
    assert layer["cp95"] is not None


def test_duplicate_directions_rejected():
    rows=all_rows()
    rows[1]["rep"]=0
    with pytest.raises(ValueError,match="duplicate"):
        run.statistical_summary(rows)


def test_new_pcg64_seed_table_and_paired_amplitudes(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"sampled",lambda f,state,duration,step: np.zeros((int(round(duration/.01))+1,15)))
    monkeypatch.setattr(run,"numerical_check",lambda *args: dict(passed=True))
    obs_path,truth_path,plan_path=run.generate_packages(tmp_path)
    obs=json.loads(obs_path.read_text())
    truth=json.loads(truth_path.read_text())
    plan=json.loads(plan_path.read_text())
    assert len(obs)==len(truth)==len(plan)==132
    directions={}
    for entry in plan:
        key=entry["input_id"]
        packet=obs[key]
        assert set(packet)=={"theta_history","omega_history","history_times","p_end","sigma"}
        if entry["rep"] is None:
            continue
        case,rep=entry["case"],entry["rep"]
        assert entry["seed"]==120000+100*case+rep
        expected=np.random.Generator(np.random.PCG64(entry["seed"])).standard_normal((11,6))
        np.testing.assert_array_equal(truth[key]["noise"],expected)
        actual=np.column_stack((packet["theta_history"],packet["omega_history"]))
        np.testing.assert_array_equal(actual,.1*entry["sigma"]*expected)
        directions.setdefault((case,rep),[]).append(expected)
    assert len(directions)==64
    for pair in directions.values():
        np.testing.assert_array_equal(pair[0],pair[1])


def fake_task(obs, amplitude_free, initial, step):
    assert amplitude_free is False
    np.testing.assert_array_equal(initial,np.zeros(3))
    state=np.r_[np.asarray(obs["theta_history"])[-1],np.asarray(obs["omega_history"])[-1],obs["p_end"],np.zeros(6)]
    return dict(status="converged",actual_residual_calls=1,history_step=step,
                best=dict(state=state,u=np.zeros(3),history_rms=0.))


def observation():
    return dict(theta_history=np.zeros((11,3)),omega_history=np.zeros((11,3)),
                history_times=np.linspace(-.1,0,11),p_end=np.array([.1,.2,.3]),sigma=1e-6)


def test_simple_only_calls_whitelist_checkpoint_and_truth_isolation(monkeypatch,tmp_path):
    def infer(obs,f,*,amplitude_free,initial,step=.0025):
        run.validate_observation(obs)
        return fake_task(obs,amplitude_free,initial,step)
    monkeypatch.setattr(run,"infer_endpoint",infer)
    monkeypatch.setattr(run,"numerical_check",lambda *args: dict(passed=True,history_main_fine_rms=0.))
    hidden=tmp_path/"isolated_truth.json"
    hidden.write_text('{"future": [1e99]}')
    first=run.fit_packages({"input_000":observation()},output=tmp_path)
    hidden.write_text('{"future": [-1e99], "secret_a": [1000]}')
    second=run.fit_packages({"input_000":observation()})
    a=first["records"]["input_000"]["main"]["best"]["state"]
    b=second["records"]["input_000"]["main"]["best"]["state"]
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[6:9],observation()["p_end"])
    assert first["performed_fine_refits"]==0
    partial=json.loads((tmp_path/"fits.partial.json").read_text())
    assert partial["records"]["input_000"]["status"]=="completed"


def test_fine_refits_only_four_largest_history_discrepancies(monkeypatch):
    calls=[]
    def infer(obs,f,*,amplitude_free,initial,step=.0025):
        calls.append(step)
        return fake_task(obs,amplitude_free,initial,step)
    monkeypatch.setattr(run,"infer_endpoint",infer)
    def numerical(f,state):
        return dict(passed=False,history_main_fine_rms=state[0],roundtrip_scaled_max=[0.,0.],
                    future_main_fine_rms=0.,future_main_fine_peak=0.)
    monkeypatch.setattr(run,"numerical_check",numerical)
    packets={}
    for i in range(6):
        obs=observation()
        obs["theta_history"][-1,0]=(i+1)*1e-6
        packets[f"input_{i:03d}"]=obs
    fit=run.fit_packages(packets)
    assert calls.count(.0025)==6 and calls.count(.00125)==4
    assert fit["requested_fine_refits"]==6
    assert fit["records"]["input_000"]["fine"] is None
    assert fit["records"]["input_001"]["fine"] is None


def test_fine_refits_cannot_repair_original_roundtrip_or_future_failure(monkeypatch):
    calls=[]
    def infer(obs,f,*,amplitude_free,initial,step=.0025):
        calls.append(step)
        return fake_task(obs,amplitude_free,initial,step)
    monkeypatch.setattr(run,"infer_endpoint",infer)
    def numerical(f,state):
        index=int(round(state[0]))
        check=dict(passed=False,history_main_fine_rms=1e-6,roundtrip_scaled_max=[0.,0.],
                   future_main_fine_rms=0.,future_main_fine_peak=0.)
        if index==0:
            check["roundtrip_scaled_max"]=[1e-6,0.]
        elif index==1:
            check["future_main_fine_rms"]=1e-4
        elif index==2:
            check["future_main_fine_peak"]=1e-3
        return check
    monkeypatch.setattr(run,"numerical_check",numerical)
    packets={}
    for index in range(4):
        obs=observation()
        obs["theta_history"][-1,0]=index
        packets[f"input_{index:03d}"]=obs
    result=run.fit_packages(packets)
    assert calls.count(.0025)==4 and calls.count(.00125)==1
    assert result["requested_fine_refits"]==1
    assert all(result["records"][f"input_{index:03d}"]["fine"] is None for index in range(3))
    assert result["records"]["input_003"]["fine"] is not None


def test_fits_hash_required_before_any_truth_file_read(tmp_path):
    fits=tmp_path/"fits.json"
    fits.write_text("{}")
    with pytest.raises(ValueError,match="hash"):
        run.score_frozen(fits,"wrong",tmp_path/"no_obs",tmp_path/"no_truth",tmp_path/"no_plan",tmp_path/"no_parent")


@pytest.mark.parametrize("original_failure",[None,"history","roundtrip","future_rms","future_peak"])
def test_complete_score_path_hard_p_dual_step_calibration_and_noiseless_regression(monkeypatch,tmp_path,original_failure):
    monkeypatch.setattr(run,"sampled",lambda f,state,duration,step: np.tile(state,(101,1)))
    monkeypatch.setattr(run,"backward_history",lambda f,state,step: np.tile(state,(11,1)))
    obs=observation()
    obs["sigma"]=0.
    task=fake_task(obs,False,np.zeros(3),.0025)
    state=task["best"]["state"]
    obs_path=tmp_path/"observations.json"
    truth_path=tmp_path/"truth.json"
    fits_path=tmp_path/"fits.json"
    plan_path=tmp_path/"plan.json"
    parent_path=tmp_path/"parent.json"
    run.write_json(obs_path,{"input_000":obs})
    run.write_json(truth_path,{"input_000":dict(case=0,sigma=0.,rep=None,history=np.tile(state,(11,1)),
        future=np.tile(state,(101,1)),endpoint_qualification=dict(passed=True),reference_numerical=dict(passed=True))})
    run.write_json(plan_path,[dict(input_id="input_000",case=0,sigma=0.,rep=None)])
    numerical=dict(passed=original_failure is None,history_main_fine_rms=0. if original_failure is None else 1e-6,
                   roundtrip_scaled_max=[0.,0.],future_main_fine_rms=0.,future_main_fine_peak=0.)
    if original_failure=="roundtrip":
        numerical["roundtrip_scaled_max"]=[0.,1e-6]
    elif original_failure=="future_rms":
        numerical["future_main_fine_rms"]=1e-4
    elif original_failure=="future_peak":
        numerical["future_main_fine_peak"]=1e-3
    run.write_json(fits_path,dict(observations_sha256=run.digest(obs_path),records={"input_000":dict(
        status="completed",main=task,fine=task if original_failure else None,numerical=numerical,
        fine_numerical=dict(passed=True) if original_failure else None)}))
    metrics=dict(joint_rms=0.,theta_rms=0.,omega_rms=0.,common_omega_rms=0.,peak=0.,p_rms=0.)
    run.write_json(parent_path,dict(scores=[dict(case=0,sigma=0.,groups=dict(simple=dict(
        selected_index=0,future_pass=True,candidates=[dict(future_fine=metrics)])))]))
    report=run.score_frozen(fits_path,run.digest(fits_path),obs_path,truth_path,plan_path,parent_path)
    row=report["rows"][0]
    assert row["classification"]==("pass" if original_failure in (None,"history") else "numerically_unresolved")
    assert row["equilibrium_negative_calibration"]
    np.testing.assert_array_equal(row["main"]["p_constraint_error"],np.zeros(3))
    assert row["parent_noiseless_regression"]["future_metric_differences"]==metrics
    assert row["main"]["future_fine"]["common_omega_rms"]==0.
    assert not report["statistics"]["simultaneous_bound_available"]
    obs_path.write_text("{}")
    with pytest.raises(ValueError,match="observations changed"):
        run.score_frozen(fits_path,run.digest(fits_path),obs_path,truth_path,plan_path,parent_path)
