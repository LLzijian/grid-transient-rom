from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


TEST_FILE = Path(__file__).resolve()
STAGING = next((parent for parent in TEST_FILE.parents
                if parent.name == "repair04_staging"
                and (parent / "experiments" / "closure_discovery_confirmation.py").is_file()), None)
ROOT = STAGING.parents[2] if STAGING is not None else TEST_FILE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
if STAGING is not None:
    sys.path.insert(0, str(STAGING / "experiments"))
confirmation = importlib.import_module("closure_discovery_confirmation")
EXPECTED_MODULE = (STAGING / "experiments" / "closure_discovery_confirmation.py"
                   if STAGING is not None else ROOT / "experiments" / "closure_discovery_confirmation.py")
assert Path(confirmation.__file__).resolve() == EXPECTED_MODULE.resolve()


def _event(future_points: int, *, event_id: str = "synthetic"):
    x = np.zeros((11 + future_points, 30), dtype=np.float64)
    return {"id": event_id, "group": "g", "label": "unstable", "x": x,
            "torque": np.zeros((len(x), 10)), "classification_meta": {}}


def test_components_use_weighted_common_frequency_machine_scales_and_first_future_point():
    event = _event(100)
    mass = np.arange(1.0, 11.0)
    p_scale = np.linspace(0.5, 1.4, 10)
    pred = np.zeros((101, 30), dtype=np.float64)
    # Give x[10] a sentinel: it must never enter the score.
    event["x"][10, 10:20] = 1000.0
    event["x"][10, 20:30] = -1000.0
    omega_error = np.arange(1.0, 11.0)
    normalized_power_error = np.arange(1.0, 11.0)
    event["x"][11:111, 10:20] = 3.0
    pred[1:101, 10:20] = 3.0 + omega_error
    event["x"][11:111, 20:30] = 2.0
    pred[1:101, 20:30] = 2.0 + p_scale * normalized_power_error

    result = confirmation._component_metrics(pred, event, mass, p_scale)["1"]
    common_error = float((omega_error * mass).sum() / mass.sum())
    assert result["available"] is True
    assert result["n_time_points"] == 100
    freq = result["frequency"]
    assert freq["finite"] is True
    assert freq["n_scalar_points"] == 100
    assert freq["physical_sse"] == pytest.approx(100 * common_error**2)
    assert freq["rmse_rad_s"] == pytest.approx(abs(common_error))
    assert freq["normalized_mse"] == pytest.approx(common_error**2 / 1.0033667249602924**2)
    power = result["mechanical_power"]
    expected_power_normalized = float(np.mean(np.square(normalized_power_error)))
    assert power["finite"] is True
    assert power["n_scalar_points"] == 1000
    assert power["physical_sse"] == pytest.approx(100 * np.square(p_scale * normalized_power_error).sum())
    assert power["rmse_pu"] == pytest.approx(np.sqrt(power["physical_sse"] / 1000))
    assert power["normalized_mse"] == pytest.approx(expected_power_normalized)


def test_full_and_short_intervals_and_prefix_boundary_are_explicit():
    mass = np.ones(10)
    p_scale = np.ones(10)
    full = _event(800)
    full_metrics = confirmation._component_metrics(np.zeros((801, 30)), full, mass, p_scale)
    assert all(full_metrics[str(h)]["available"] for h in (1, 2, 5, 8))
    assert full_metrics["8"]["n_time_points"] == 800
    assert full_metrics["prefix"]["available"] is True
    assert full_metrics["prefix"]["n_time_points"] == 800

    short = _event(5)
    short_metrics = confirmation._component_metrics(np.zeros((6, 30)), short, mass, p_scale)
    assert all(short_metrics[str(h)]["available"] is False for h in (1, 2, 5, 8))
    assert short_metrics["prefix"]["available"] is True
    assert short_metrics["prefix"]["n_time_points"] == 5
    assert short_metrics["1"]["frequency"]["physical_sse"] is None


def test_nonfinite_component_is_retained_and_marked_without_dropping_other_component():
    event = _event(100)
    pred = np.zeros((101, 30), dtype=np.float64)
    pred[1, 13] = np.nan
    result = confirmation._component_metrics(pred, event, np.ones(10), np.linspace(0.6, 1.5, 10))["1"]
    assert result["available"] is True
    assert result["frequency"]["finite"] is False
    assert result["frequency"]["n_scalar_points"] == 100
    assert result["frequency"]["physical_sse"] is None
    assert result["mechanical_power"]["finite"] is True
    assert result["mechanical_power"]["n_scalar_points"] == 1000


def test_finite_errors_with_overflowing_statistics_are_marked_nonfinite():
    event = _event(100)
    pred = np.zeros((101, 30), dtype=np.float64)
    pred[1:101, 10:20] = 1e200
    pred[1:101, 20:30] = 1e200
    result = confirmation._component_metrics(pred, event, np.ones(10), np.ones(10))["1"]
    for name in ("frequency", "mechanical_power"):
        component = result[name]
        assert component["finite"] is False
        for field in ("physical_sse", "rmse_rad_s" if name == "frequency" else "rmse_pu", "normalized_mse"):
            assert component[field] is None


def test_evaluate_events_keeps_the_existing_primary_mse_formula_and_adds_components(monkeypatch, tmp_path):
    evaluator = importlib.import_module("evaluate_structured_rom")
    fine = np.zeros((1, 1601, 30), dtype=np.float64)
    fine[:, :, 10:20] = np.arange(10, dtype=np.float64)[None, None, :]
    fine[:, :, 20:30] = np.linspace(0.1, 1.0, 10)[None, None, :]
    coarse = np.zeros((1, 801, 30), dtype=np.float64)
    monkeypatch.setattr(evaluator, "_rollouts", lambda model, batch, steps, dt: fine if steps == 1600 else coarse)
    monkeypatch.setattr(evaluator, "_pair_row", lambda *args, **kwargs: {
        "fine_classification": {"predicted_los": False}, "coarse_classification": {"predicted_los": False}})
    monkeypatch.setattr(evaluator, "_scored_numerics", lambda *args, **kwargs: [])
    monkeypatch.setattr(evaluator, "_metric_safe", lambda pred, event, horizon, mass: {
        "horizon": str(horizon), "available": True, "finite": True})
    monkeypatch.setattr(evaluator, "_needs_finer", lambda rows: False)

    event = _event(500)
    event["x"][11:, 10:20] = 0.25
    event["x"][11:, 20:30] = -0.1
    model = type("SyntheticModel", (), {})()
    model.mass = torch.arange(1, 11, dtype=torch.float64)
    model.p_scale = torch.linspace(0.5, 1.4, 10, dtype=torch.float64)
    ref = fine[0, ::2]
    trainer = importlib.import_module("train_structured_rom")
    expected = float(trainer._point_error(torch.as_tensor(ref[1:501]), torch.as_tensor(event["x"][11:511]), model).mean())

    result = confirmation.evaluate_events(model, [event], tmp_path / "synthetic.json")
    row = result["records"][0]
    assert row["mse"] == pytest.approx(expected)
    assert set(row["components"]) == {"1", "2", "5", "8", "prefix"}
    assert row["components"]["prefix"]["n_time_points"] == 500
    saved = json.loads((tmp_path / "synthetic.json").read_text(encoding="utf-8"))
    assert saved["records"][0]["mse"] == pytest.approx(expected)
    assert "components" in saved["records"][0]


def test_cache_missing_components_is_rejected_and_not_rewritten(tmp_path):
    event = _event(5)
    identity = {"model_sha256": "synthetic-model", "cohort": confirmation._cohort_identity([event])}
    old_row = {"id": event["id"], "group": event["group"], "label": event["label"],
               "numerical_pass": False, "complete_5s": False, "mse": None, "metrics": [],
               "pair": {}, "selected_pair": {}, "scored_numerics": []}
    old_result = {"_identity": identity, "records": [old_row], "event_count": 1,
                  "primary_risk": None, "primary_complete_stable_count": 0,
                  "all_numerical_pass": False, "classification_counts": {},
                  "scope": "synthetic", "test_read": False}
    cache = tmp_path / "old-cache.json"
    cache.write_text(json.dumps(old_result), encoding="utf-8")
    before = cache.read_bytes()
    with pytest.raises(confirmation.RecoveryUnresolved, match="incomplete"):
        confirmation.evaluate_or_reuse(None, [event], cache, "synthetic-model")
    assert cache.read_bytes() == before

