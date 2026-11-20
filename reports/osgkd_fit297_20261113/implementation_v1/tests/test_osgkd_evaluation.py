import numpy as np

from kiki.evaluation.osgkd import aggregate_metrics, classification, motion_metrics, risk, score_event


def constants():
    return {"mass": np.ones(10), "omega_scale": np.ones(10), "s_coi": 1.,
            "omega_s": 2*np.pi*60, "S_m": np.ones(10), "Se": np.ones(10)}


def event(points=801):
    targets = {key: np.zeros((points, 10)) for key in ("theta", "omega", "tau_m", "tau_e")}
    return {"id": "synthetic", "group": "synthetic_group", "label": "stable",
            "history": {"theta": np.zeros((11, 10)), "omega": np.zeros((11, 10))},
            "t": np.arange(points)*.01, "targets": targets,
            "masks": {key: np.ones_like(x, dtype=bool) for key, x in targets.items()},
            "metadata": {"true_event_time_record_s": None, "record_start_delay_s": .0001,
                         "clearing_time_s": 1.1}}


def test_motion_risk_retains_common_speed_but_ignores_global_angle():
    truth = np.zeros((20, 20))
    prediction = np.column_stack((np.full((20, 10), 5.), np.full((20, 10), 2.)))
    result = motion_metrics(prediction, truth, np.full(10, .1), np.ones(10), 1.)
    assert result["mean_motion_error"] == 4.
    assert result["common_speed_rmse_rad_s"] == 2.
    assert np.isclose(risk([1., 2., 3.]), .8*2+.2*3)


def test_late_los_is_not_a_comparable_true_record_detection_and_numerics_dominate():
    item = event(11)
    item["label"] = "unstable"
    item["metadata"]["true_event_time_record_s"] = .15
    path = np.zeros((801, 50))
    path[9:, 0] = np.pi + .1
    path[9:, 10] = .03 * constants()["omega_s"]
    result = classification(item, path, path.copy(), constants(), 800)
    assert result["classification"]["predicted_los"]
    assert result["predicted_time_record_s"] == .23
    assert not result["timing_comparable"]
    coarse = path.copy()
    coarse[:, 40] += 1
    failed = classification(item, coarse, path, constants(), 800)
    assert failed["classification"]["category"] == "numerical_unresolved"
    assert not failed["classification"]["predicted_los"]


def test_one_numerical_failure_keeps_original_risk_denominator():
    item = event()
    path = np.zeros((801, 50))
    ports = {key: np.zeros((801, 10)) for key in ("tau_m", "tau_e")}
    good = score_event(item, path, path, ports, constants())
    bad_path = path.copy()
    bad_path[10:, 30] = 1
    bad = score_event({**item, "id": "second"}, bad_path, path, ports, constants())
    result = aggregate_metrics([good, bad], "1")
    assert result["denominator"] == 2 and result["covered"] == 1
    assert result["official"] is None and result["unresolved_ids"] == ["second"]
    assert result["covered_only_diagnostic"]["motion_risk_v1"] == 0
