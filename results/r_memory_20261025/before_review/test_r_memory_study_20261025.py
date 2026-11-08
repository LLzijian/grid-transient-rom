"""Pure study-contract checks: no real dataset or optimizer updates."""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import r_memory_study_20261025 as study


def events():
    result = []
    for group in range(20):
        for index in range(3):
            result.append({"id": f"event_{group:02d}_{index}", "group": str(group),
                "label": "unstable" if index == 0 else "stable",
                "x": np.zeros((19 + index, 30)), "torque": np.zeros((19 + index, 10))})
    return result


def test_whole_group_partition_is_fixed_and_source_exposure_is_not_erased():
    source = events()
    fit, held = study.partition_events(source)
    assert {e["id"] for e in fit} | {e["id"] for e in held} == {e["id"] for e in source}
    assert not {e["group"] for e in fit} & {e["group"] for e in held}
    assert study.partition_events(list(reversed(source)))[0][0]["id"] == fit[0]["id"]
    for event in held:
        integer = int.from_bytes(hashlib.sha256(("r_memory_20261025|" + event["group"]).encode()).digest(), "big")
        assert integer % 5 == 0
    with pytest.raises(ValueError, match="empty or full"):
        study.partition_events([source[0]])


def test_calibration_panel_retains_both_outcomes_and_group_round_robin():
    fit, _ = study.partition_events(events())
    panel = study.panel_events(fit)
    assert len(panel) == len({e["id"] for e in panel}) == 12
    assert sum(e["label"] == "unstable" for e in panel) == 3
    assert sum(e["label"] == "stable" for e in panel) == 9
    assert len({e["group"] for e in panel if e["label"] == "unstable"}) == 3


def test_frozen_schedule_pairs_models_and_uses_real_ragged_counts():
    fit, _ = study.partition_events(events())
    first = study.make_schedule(fit, 25101)
    second = study.make_schedule(fit, 25101)
    assert first == second
    rows = first["rows"]
    assert len(rows) == 2048
    assert [rows[i - 1]["horizon_s"] for i in (1, 256, 257, 512, 513, 514, 2048)] == [1., 1., 2., 2., 5., 8., 8.]
    by_id = {e["id"]: e for e in fit}
    for row in rows:
        assert len(row["sample_ids"]) == 12
        assert row["long_counts"] == [len(by_id[eid]["x"]) - 11 for eid in row["sample_ids"]]
        assert all(10 <= endpoint < len(by_id[eid]["x"]) - 1
            for eid, endpoint in zip(row["sample_ids"], row["endpoints"]))
    with pytest.raises(ValueError):
        study.horizon_for(2049)


def test_calibration_joint_choice_is_arithmetic_and_requires_both_valid():
    scores = {"A": {"1": {"qualified": True, "ratio": .7}, "3": {"qualified": True, "ratio": .8}},
              "B": {"1": {"qualified": True, "ratio": .1}, "3": {"qualified": True, "ratio": 1.}},
              "C": {"1": {"qualified": True, "ratio": 3.}, "3": {"qualified": True, "ratio": 1.}}}
    assert study.select_multipliers(scores) == {"A": 1, "B": 3, "C": 3}
    scores["B"]["3"]["qualified"] = False
    assert study.select_multipliers(scores)["B"] == 1
    scores["C"]["1"]["qualified"] = False
    assert study.select_multipliers(scores)["C"] is None
    scores["A"]["3"]["ratio"] = .7
    assert study.select_multipliers(scores)["A"] == 1


def test_noise_is_event_paired_and_preserves_all_p_and_future_truth():
    event = events()[0]
    event["x"] = np.arange(19 * 30, dtype=float).reshape(19, 30) * .001
    prior = event["x"].copy()
    first = study.noise_history(event, np.arange(1, 11), np.ones(10) * 2.)
    second = study.noise_history(event, np.arange(1, 11), np.ones(10) * 2.)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 20:30], event["x"][:11, 20:30])
    np.testing.assert_array_equal(event["x"], prior)
    assert np.any(first[-1, :20] != event["x"][10, :20])
    different = study.noise_history({**event, "id": "different_event"}, np.arange(1, 11), np.ones(10) * 2.)
    assert not np.array_equal(first[:, :20], different[:, :20])


def test_latent_normalization_distinguishes_old_q_and_physical_r():
    class Model:
        mass = torch.ones(10, dtype=torch.float64)
        rstar = torch.arange(1., 11., dtype=torch.float64)
    model = Model()
    torch.testing.assert_close(study.latent_scale(model, "A"), torch.ones(10, dtype=torch.float64) * .1)
    torch.testing.assert_close(study.latent_scale(model, "B"), model.rstar * .1)
    model.rstar = torch.zeros(10)
    with pytest.raises(ValueError, match="positive physical rstar"):
        study.latent_scale(model, "C")


def test_fit_only_cache_does_not_materialize_no_update_payload(tmp_path, monkeypatch):
    fit, held = study.partition_events(events())
    study._save_event_cache(tmp_path / "fit_events.npz", fit)
    study._save_event_cache(tmp_path / "no_update_events.npz", held)
    study.write(tmp_path / "schedule.json", {})
    metadata = {"source_sha256": study.SOURCE_SHA, "code": {"fixture": "pure"},
        "schedule_sha256": study.sha(tmp_path / "schedule.json"),
        "fit_cache_sha256": study.sha(tmp_path / "fit_events.npz"),
        "no_update_cache_sha256": study.sha(tmp_path / "no_update_events.npz"),
        "fit_ids": [e["id"] for e in fit], "no_update_ids": [e["id"] for e in held]}
    study.write(tmp_path / "prepare.json", metadata)
    (tmp_path / "no_update_events.npz").write_bytes(b"deliberately unreadable held payload")
    monkeypatch.setattr(study, "require_source", lambda: None)
    monkeypatch.setattr(study, "code_identity", lambda: {"fixture": "pure"})
    _, restored_fit = study.checked_preparation(tmp_path, fit_only=True)
    assert [e["id"] for e in restored_fit] == metadata["fit_ids"]
    with pytest.raises(ValueError, match="cache identity"):
        study.checked_preparation(tmp_path, fit_only=False)


def test_eight_second_gate_controls_primary_five_but_fifteen_failure_does_not(tmp_path, monkeypatch):
    import summarize_train_local_memory_0098
    monkeypatch.setattr(summarize_train_local_memory_0098, "build_summary", lambda _: {})
    rows = [{"event_id": "e", "group": "g", "label": "stable", "horizon": h,
             "available": True, "finite": True, "numerically_admissible": True,
             "mean_point_error": value} for h, value in (("5", 1.), ("8", 1.5))]
    records = {key: [] for key in ("records", "common_frequency", "normalized_p", "numerical")}
    records["objective_point_error"] = rows
    pair = {"finite_to_endpoint": True, "los_pair_pass": True,
        "coi_normalized_rmse": 0., "common_normalized_rmse": 0.,
        "fine_classification": {"predicted_los": False, "window_settled": True, "trigger_index": None}}
    extra = {"event_id": "e", "label": "stable", "scientific_numeric_8s": False,
        "classification_8s": copy.deepcopy(pair), "classification_15s": copy.deepcopy(pair),
        "coarse_point_errors": {"5": .9, "8": 1.4},
        "actual_interval15": {"available": True, "finite": False, "numerically_admissible": False}}
    blocked = study.group_summary(records, [extra], {"e"})
    assert blocked["primary"]["value"] is None
    assert blocked["horizon_risks"]["5"]["value"] is None
    extra["scientific_numeric_8s"] = True
    retained = study.group_summary(records, [extra], {"e"})
    assert retained["primary"]["value"] == 1.
    assert retained["horizon_risks"]["8"]["value"] == pytest.approx(1.5)
    assert retained["horizon_risks"]["5"]["numerical_score_perturbation"] == pytest.approx(.1)
    assert retained["all_8s_rows_qualified"]
    assert not retained["all_actual15s_rows_qualified"]
