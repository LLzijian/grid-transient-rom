"""Pure-mock preparation of stage dependencies, final-point rules and statistics."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
import gftd_endpoint_confirmation_20261107 as run


def observation(sigma=1e-4):
    return dict(theta_history=np.zeros((11,3)),omega_history=np.zeros((11,3)),
                p_end=np.zeros(3),history_times=np.linspace(-.1,0,11),sigma=sigma)


def numerical(history=0.,passed=True):
    return dict(passed=passed,roundtrip_scaled_max=[0.,0.],history_main_fine_rms=history,
                future_main_fine_rms=0.,future_main_fine_peak=0.)


def simple_task(obs,status="converged"):
    u=np.array([.1,.2,.3])
    return dict(status=status,actual_residual_calls=7,history_step=.0025,
                best=dict(u=u,state=run.hard_endpoint(run.validate_observation(obs),u,False),history_rms=0.))


def smooth_task(obs,u,step=.0025):
    state=run.hard_endpoint(run.validate_observation(obs),u,False)
    return dict(status="converged",actual_residual_calls=8,history_step=step,objective_guard_passed=True,
                best=dict(u=np.r_[np.zeros(5),u],state=state,history_rms=0.))


def test_stage1_numerical_failure_blocks_stage2_no_stage1_fine(monkeypatch,tmp_path):
    simple_calls=[]
    smooth_calls=[]
    def first(obs,f,*,amplitude_free,initial):
        assert amplitude_free is False
        np.testing.assert_array_equal(initial,np.zeros(3))
        simple_calls.append(obs)
        return simple_task(obs,"budget_limited")
    def second(obs,f,u,*,smooth,initial=None,step=.0025):
        assert smooth is True
        np.testing.assert_array_equal(u,[.1,.2,.3])
        smooth_calls.append(step)
        return smooth_task(obs,u,step)
    monkeypatch.setattr(run,"infer_simple",first)
    monkeypatch.setattr(run,"infer_smooth",second)
    def qualify(f,task):
        if len(simple_calls)==1:
            return numerical(history=1e-6,passed=False),None
        return numerical(),None
    monkeypatch.setattr(run,"evaluate_numerical",qualify)
    result=run.fit_packages({"a":observation(),"b":observation(),"z":observation(0.)},tmp_path)
    assert len(simple_calls)==2 and smooth_calls==[.0025]
    assert result["records"]["a"]["status"]=="blocked_by_stage1"
    assert result["records"]["a"]["stage2"] is None
    assert result["records"]["b"]["stage1"]["status"]=="budget_limited"
    assert result["records"]["b"]["status"]=="completed"
    assert result["records"]["z"]["status"]=="reused_noiseless"
    assert result["performed_fine_refits"]==0
    partial=json.loads((tmp_path/"fits.partial.json").read_text())
    assert set(partial["records"])=={"a","b","z"}


def test_stage2_only_four_fines_do_not_replace_or_rerun_upstream(monkeypatch):
    first_calls=[]
    second_calls=[]
    def first(obs,f,**kwargs):
        first_calls.append(1)
        return simple_task(obs)
    def second(obs,f,u,*,smooth,initial=None,step=.0025):
        second_calls.append(step)
        if step==.00125:
            np.testing.assert_array_equal(initial,np.r_[np.zeros(5),[.1,.2,.3]])
        return smooth_task(obs,u,step)
    monkeypatch.setattr(run,"infer_simple",first)
    monkeypatch.setattr(run,"infer_smooth",second)
    monkeypatch.setattr(run,"evaluate_numerical",lambda f,task:(
        numerical(history=1e-6,passed=False) if len(task["best"]["u"])==8 and task["history_step"]==.0025
        else numerical(),None))
    result=run.fit_packages({f"input_{i:03d}":observation() for i in range(6)})
    assert len(first_calls)==6
    assert second_calls.count(.0025)==6 and second_calls.count(.00125)==4
    assert result["requested_fine_refits"]==6
    assert result["records"]["input_000"]["final_point"]=="fine"
    assert result["records"]["input_005"]["final_point"]=="main"
    assert all(row["stage1"]["history_step"]==.0025 for row in result["records"].values())


def scored(passed):
    return dict(p_constraint_error=[0.,0.,0.],future_main_pass=passed,future_fine_pass=passed)


@pytest.mark.parametrize("fine_qualified,main_prediction,fine_prediction,expected",[
    (True,True,True,"pass"),(False,True,True,"numerically_unresolved"),
    (True,False,True,"numerically_unresolved"),(True,False,False,"prediction_fail")])
def test_final_fine_must_certify_itself_and_match_main_prediction(fine_qualified,main_prediction,fine_prediction,expected):
    obs=observation()
    one=simple_task(obs)
    two=smooth_task(obs,one["best"]["u"])
    fine=smooth_task(obs,one["best"]["u"],.00125)
    record=dict(status="completed",stage1=one,stage2=two,fine=fine,stage2_numerical=numerical(history=1e-6,passed=False),
                fine_numerical=numerical(passed=fine_qualified),final_point="fine")
    reference=dict(reference_numerical=dict(passed=True),endpoint_qualification=dict(passed=True))
    result=run.complete_class(record,"prediction_fail",dict(main=scored(main_prediction),fine=scored(fine_prediction)),reference)
    assert result==expected
    record["stage2_numerical"]["future_main_fine_peak"]=1e-3
    assert run.complete_class(record,"pass",dict(main=scored(True),fine=scored(True)),reference)=="numerically_unresolved"


def test_failed_final_fine_never_falls_back_to_a_qualified_main():
    obs=observation()
    one=simple_task(obs)
    main=smooth_task(obs,one["best"]["u"])
    record=dict(status="completed",stage1=one,stage2=main,fine=dict(status="failed",best=None),
        stage2_numerical=numerical(history=1e-6,passed=False),fine_numerical=None,final_point="fine")
    reference=dict(reference_numerical=dict(passed=True),endpoint_qualification=dict(passed=True))
    assert run.complete_class(record,"pass",dict(main=scored(True),fine=None),reference)=="numerically_unresolved"


def test_blocked_stage1_is_a_complete_terminal_in_main_statistical_family():
    rows=[dict(case=case,sigma=sigma,rep=rep,stage1_classification="numerically_unresolved",
                complete_classification="blocked_by_stage1")
          for case in range(4) for sigma in (1e-6,1e-4) for rep in range(16)]
    result=run.statistics(rows,"complete_classification")
    assert result["simultaneous_bound_available"]
    assert result["family"]=="complete_process_eight_strata_primary"
    for layer in result["layers"]:
        assert layer["complete"] and layer["counts"]["blocked_by_stage1"]==16
        assert layer["successes"]==0 and layer["cp95"][0]==0.
    rows[-1]["complete_classification"]="planned"
    incomplete=run.statistics(rows,"complete_classification")
    assert incomplete["layers"][-1]["cp95"] is None
    assert all(layer["simultaneous_failure_upper95"] is None for layer in incomplete["layers"])
    assert run.statistics(rows,"stage1_classification")["family"]=="stage1_eight_strata_separate_secondary"


def test_all_success_bounds_are_eight_layers_per_separate_family():
    rows=[dict(case=case,sigma=sigma,rep=rep,stage1_classification="pass",complete_classification="pass")
          for case in range(4) for sigma in (1e-6,1e-4) for rep in range(16)]
    process=run.statistics(rows,"complete_classification")
    stage1=run.statistics(rows,"stage1_classification")
    for layer in process["layers"]:
        np.testing.assert_allclose(layer["cp95"],[.025**(1/16),1.])
        np.testing.assert_allclose(layer["simultaneous_failure_upper95"],1-(.05/8)**(1/16))
    assert process["family"]!=stage1["family"]
    assert not process["joint_sixteen_strata_coverage_claim"]
    rows[-1]["stage1_classification"]="not_run"
    assert not run.statistics(rows,"stage1_classification")["simultaneous_bound_available"]
    assert run.statistics(rows,"complete_classification")["simultaneous_bound_available"]


def test_mock_noise_generation_pairing_and_exact_noiseless_reference(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"ROOT",tmp_path)
    truth,obs,rows={},{},[]
    for case in range(4):
        key=run.SOURCE_IDS[str(case)]
        obs[key]=observation(0.)
        truth[key]=dict(history=np.zeros((11,15)),reference_numerical=dict(passed=True),endpoint_qualification=dict(passed=True))
        rows.append(dict(input_id=key,case=case,sigma=0.,classification="pass",main=dict(
            future_main=dict(joint_rms=0.),future_fine=dict(joint_rms=0.))))
    run.write_json(tmp_path/"truth.json",truth)
    run.write_json(tmp_path/"old_obs.json",obs)
    run.write_json(tmp_path/"old_summary.json",dict(rows=rows))
    manifest=dict(case_to_source_input=run.SOURCE_IDS.copy())
    for name,path in (("truth","truth.json"),("noiseless_observations","old_obs.json"),("noiseless_summary","old_summary.json")):
        manifest[name]=dict(path=path,sha256=run.digest(tmp_path/path))
    seeds=[]
    monkeypatch.setattr(run.np.random,"PCG64",lambda seed:seed)
    def generator(seed):
        seeds.append(seed)
        return SimpleNamespace(standard_normal=lambda shape:np.full(shape,seed/1e6))
    monkeypatch.setattr(run.np.random,"Generator",generator)
    out=tmp_path/"generated"
    out.mkdir()
    obs_path,plan_path=run.generate_observations(manifest,out)
    actual=json.loads(obs_path.read_text())
    plan=json.loads(plan_path.read_text())
    assert len(actual)==len(plan)==132 and len(seeds)==64
    assert seeds==[130000+100*case+rep for case in range(4) for rep in range(16)]
    for entry in plan:
        packet=actual[entry["input_id"]]
        if entry["sigma"]==0.:
            assert packet==json.loads((tmp_path/"old_obs.json").read_text())[entry["reference_input_id"]]
        else:
            expected=np.full((11,6),.1*entry["sigma"]*entry["seed"]/1e6)
            np.testing.assert_allclose(np.column_stack((packet["theta_history"],packet["omega_history"])),expected,rtol=1e-15)
    assert not (out/"truth.json").exists()
    assert not (out/"isolated_truth.json").exists()


def test_future_metadata_never_reaches_stage1(monkeypatch):
    obs=observation()
    obs["future"]=[]
    monkeypatch.setattr(run,"infer_simple",lambda *args,**kwargs:pytest.fail("illegal data reached stage1"))
    with pytest.raises(ValueError):
        run.fit_packages({"input":obs})


def test_fit_results_independent_of_isolated_truth_contents(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"infer_simple",lambda obs,f,**kwargs:simple_task(obs))
    monkeypatch.setattr(run,"infer_smooth",lambda obs,f,u,**kwargs:smooth_task(obs,u))
    monkeypatch.setattr(run,"evaluate_numerical",lambda *args:(numerical(),None))
    hidden=tmp_path/"truth.json"
    hidden.write_text('{"true_r":[999],"future":[1e99]}')
    first=run.fit_packages({"input":observation()})
    hidden.write_text('{"true_r":[-999],"future":[-1e99]}')
    second=run.fit_packages({"input":observation()})
    for stage in ("stage1","stage2"):
        np.testing.assert_array_equal(first["records"]["input"][stage]["best"]["state"],
                                      second["records"]["input"][stage]["best"]["state"])


def test_fit_hash_protects_truth_reading(tmp_path):
    path=tmp_path/"fits.json"
    path.write_text("{}")
    with pytest.raises(ValueError,match="fits changed"):
        run.score_frozen(path,"wrong",path,path,{})


def test_actual_json_scoring_uses_final_fine_not_old_main(monkeypatch,tmp_path):
    monkeypatch.setattr(run,"ROOT",tmp_path)
    import gftd_noise_confirmation_20261107 as scoring
    monkeypatch.setattr(scoring,"sampled",lambda f,state,duration,step:np.tile(state,(101,1)))
    monkeypatch.setattr(scoring,"backward_history",lambda f,state,step:np.tile(state,(11,1)))
    obs=observation()
    one=simple_task(obs)
    main=smooth_task(obs,one["best"]["u"])
    fine=smooth_task(obs,one["best"]["u"],.00125)
    # This final point changes omega while staying within both prediction gates.
    fine["best"]["state"][3:6]=1e-6
    fine["best"]["u"][2:5]=.1
    state=one["best"]["state"]
    truth=dict(history=np.tile(state,(11,1)),future=np.tile(state,(101,1)),
               reference_numerical=dict(passed=True),endpoint_qualification=dict(passed=True))
    paths={name:tmp_path/(name+".json") for name in ("truth","old_obs","old_summary","obs","plan","fits")}
    run.write_json(paths["truth"],{run.SOURCE_IDS["0"]:truth})
    run.write_json(paths["old_obs"],{})
    run.write_json(paths["old_summary"],dict(rows=[]))
    manifest=dict(case_to_source_input=run.SOURCE_IDS.copy())
    for key,name in (("truth","truth"),("noiseless_observations","old_obs"),("noiseless_summary","old_summary")):
        manifest[key]=dict(path=paths[name].name,sha256=run.digest(paths[name]))
    run.write_json(paths["obs"],{"input_001":obs})
    run.write_json(paths["plan"],[dict(input_id="input_001",case=0,rep=0,sigma=1e-4,reference_input_id=run.SOURCE_IDS["0"])])
    row=dict(status="completed",stage1=one,stage1_numerical=numerical(),stage2=main,
             stage2_numerical=numerical(history=1e-6,passed=False),fine=fine,fine_numerical=numerical(),final_point="fine")
    run.write_json(paths["fits"],dict(records={"input_001":row},observations_sha256=run.digest(paths["obs"]),
                                     plan_sha256=run.digest(paths["plan"])))
    report=run.score_frozen(paths["fits"],run.digest(paths["fits"]),paths["obs"],paths["plan"],manifest)
    scored_row=report["rows"][0]
    assert scored_row["complete_classification"]=="pass"
    assert scored_row["scores"]["stage2"]["future_fine"]["joint_rms"]==0.
    assert scored_row["scores"]["fine"]["future_fine"]["joint_rms"]>0.
    assert report["comparisons"][0]["classification"]=="worsened"
    np.testing.assert_array_equal(scored_row["scores"]["fine"]["p_constraint_error"],np.zeros(3))
    assert scored_row["scores"]["fine"]["future_fine"]["common_omega_rms"]>0.
    assert report["worst_errors"]["complete"]["worst_rms"]==scored_row["scores"]["fine"]["future_fine"]["joint_rms"]
