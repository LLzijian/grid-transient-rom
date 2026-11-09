"""Fixed DP8 API checks on artificial two-node observations."""

from collections import Counter
import copy
import json

import numpy as np
import pytest

from kiki.models import GFTDKuramotoModel, gftd_fit297, gftd_fitting
from kiki.numerics import dop853_step
from test_gftd_fit297_engine import functional, request


def synthetic_candidate():
    field = functional(2)
    model = GFTDKuramotoModel(n_nodes=2, mass=field.mass, damping=field.damping)
    model.functional, model.gates = field, field.gates
    model.initialization_mode = "causal_two_stage"
    model.fit_state = {"synthetic_fit": True}
    parent = gftd_fit297.make_config(field, model.fit_mode, request(field, 2), 2, "synthetic")
    model.inference_config = gftd_fitting.make_fixed_dp8_config(field, parent)

    state = field.pack_state([0.03, -0.01], [0.02, 0.01], [0.01, 0.02], [0., 0.], [0.01, -0.02])
    history = [state]
    for _ in range(40):
        history.append(dop853_step(field.rhs, history[-1], -.0025))
    history = np.asarray(history[::-4])
    inputs = dict(theta_history=history[:, :2], omega_history=history[:, 2:4],
                  history_times=np.linspace(-.1, 0., 11), p0=state[4:6])
    return model, parent, inputs


def test_candidate_roundtrip_uses_fixed_dp8_from_history_through_rollout(tmp_path):
    model, parent, inputs = synthetic_candidate()
    checkpoint = tmp_path / "fixed_dp8.json"
    trajectories = []
    for step in (.01, .005):
        # np.linspace produces slightly different per-interval floating steps.
        trajectory = model.rollout_state(inputs, np.linspace(0., 6 * step, 7))
        expected = [trajectory[0]]
        for _ in range(6):
            expected.append(dop853_step(model.functional.rhs, expected[-1], step))
        np.testing.assert_array_equal(trajectory, expected)
        np.testing.assert_array_equal(trajectory[0, 4:6], inputs["p0"])
        trajectories.append(trajectory)

        diag = model.last_initialization
        assert diag["method"] == "fixed_dp8" and diag["status"] == "completed"
        for name, limit in (("stage1", 600), ("stage2", 1200)):
            counts = diag[name]["counts"]
            assert 0 < counts["actual_residual_calls"] <= limit
            assert counts["fixed_dp8_steps_completed"] == counts["actual_residual_calls"] * 40
            assert counts["actual_rhs_calls"] == counts["fixed_dp8_steps_completed"] * 12
        assert diag["stage1"]["initial"] == [0., 0.]
        assert diag["stage2"]["initial"] == diag["stage2"]["feasible"]
        assert diag["stage2"]["feasible"] == [0., 0.] + diag["stage1"]["best"]["u"]
        stage1 = diag["stage1"]["best"]
        np.testing.assert_array_equal(stage1["state"][-2:], np.array(parent["r_scale"]) * stage1["u"])
        assert diag["stage1_numerical"]["counts"]["fixed_dp8_steps_completed"] == 540
        assert diag["final_numerical"]["p_endpoint_error"] == 0.
        assert diag["counts"]["rk4_steps_attempted"] == 0

        if step == .01:
            model.save(checkpoint)
            restored = GFTDKuramotoModel.load(checkpoint)
            assert restored.inference_config == model.inference_config
            assert restored.inference_config["fit297_config"] == parent
            assert restored.fit_state == model.fit_state
            assert restored.last_initialization == model.last_initialization
            np.testing.assert_array_equal(restored.functional.rhs(trajectory[0]),
                                          model.functional.rhs(trajectory[0]))
            model = restored
    np.testing.assert_array_equal(trajectories[0][0], trajectories[1][0])


def test_all_six_gate_legs_use_registered_signed_steps():
    model, _, inputs = synthetic_candidate()
    state = model.functional.pack_state([0., 0.], [0., 0.], inputs["p0"], [0., 0.], [0., 0.])
    steps = Counter()

    def tracked_step(rhs, state, step):
        steps[step] += 1
        return dop853_step(rhs, state, step)

    check = gftd_fitting.numerical_check(model.functional, state, {"p_end": inputs["p0"]}, stepper=tracked_step)
    assert check["passed"]
    assert steps == {-.0025: 40, .0025: 40, -.00125: 80, .00125: 80, .01: 100, .005: 200}
    assert check["counts"]["actual_rhs_calls"] == 12 * sum(steps.values())


def test_failed_fifth_rhs_stops_at_stage1_and_retains_actual_cost():
    model, _, inputs = synthetic_candidate()
    rhs = model.functional.rhs
    calls = 0

    def interrupted_rhs(state):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise FloatingPointError("synthetic fifth-stage failure")
        return rhs(state)

    model.functional.rhs = interrupted_rhs
    with pytest.raises(gftd_fitting.InitializationFailure, match="blocked_by_stage1") as failure:
        model.initialize(inputs)
    diag = failure.value.diagnostics
    assert diag["stage2"] is None and diag["stage1_numerical"] is None
    assert diag["counts"]["fixed_dp8_steps_attempted"] == 1
    assert diag["counts"]["fixed_dp8_steps_completed"] == 0
    assert diag["counts"]["actual_rhs_calls"] == calls == 5
    assert diag["counts"]["actual_residual_calls"] == 1


@pytest.mark.parametrize("key", ["format", "method", "protocol_sha256", "coefficients_sha256", "fit297_config", "extra"])
def test_checkpoint_rejects_changed_fixed_dp8_identity(tmp_path, key):
    model, _, inputs = synthetic_candidate()
    checkpoint = tmp_path / "fixed_dp8.json"
    model.save(checkpoint)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    config = payload["inference_config"]
    if key == "fit297_config":
        config[key]["r_scale"][0] += .1
    else:
        config[key] = "tampered"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    model.inference_config = config
    with pytest.raises(gftd_fitting.InitializationFailure):
        GFTDKuramotoModel.load(checkpoint)
    with pytest.raises(gftd_fitting.InitializationFailure):
        model.save(checkpoint)
    with pytest.raises(gftd_fitting.InitializationFailure):
        model.initialize(inputs)
    assert model.last_initialization["status"] == "invalid_configuration"


def test_candidate_rejects_changed_model_dispatch_and_illegal_inputs():
    model, _, inputs = synthetic_candidate()
    for invalid_times in ([0., .0025], [0., .005, .015], [0., .005, .010000000002]):
        with pytest.raises(ValueError, match="uniform"):
            model.rollout_state(inputs, invalid_times)
    with pytest.raises(ValueError, match="exogenous"):
        model.rollout_state(inputs, [0., .01], {"p_future": [0.]})
    with pytest.raises(ValueError, match="exactly"):
        model.initialize(dict(inputs, event_id="synthetic"))
    for field, value in (("fit_mode", "static"), ("initialization_mode", "legacy_static")):
        changed = copy.copy(model)
        setattr(changed, field, value)
        with pytest.raises(gftd_fitting.InitializationFailure, match="dispatch"):
            changed.initialize(inputs)
    model.functional.k_edge[0] += .01
    with pytest.raises(gftd_fitting.InitializationFailure, match="fingerprint"):
        model.initialize(inputs)
