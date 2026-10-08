from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import experiments.check_train_boundary as checker
from experiments.check_train_boundary import _read_root, compare_pair, load_selection


def _event(*, label: str = "stable", t: np.ndarray | None = None) -> dict:
    if t is None:
        t = np.arange(0.0, 15.0 + 0.01 / 2, 0.01)
    delta = np.column_stack((0.01 * np.sin(t), -0.01 * np.sin(t)))
    omega = np.column_stack((0.02 * np.cos(t), -0.02 * np.cos(t)))
    return {
        "attrs": {
            "label": label,
            "postfault_signature": "sig",
            "record_start_delay_s": 0.0,
            "trajectory_end_s": float(t[-1]),
            "event_time_s": 1.0 if label == "unstable" else None,
        },
        "arrays": {"t": t, "delta": delta, "omega": omega},
        "row": {},
    }


def test_compare_pair_qualified_on_shared_grid_and_zero_error():
    event = _event()
    result = compare_pair(event, _event(), mass=np.array([1.0, 2.0]), omega_s=2.0 * np.pi * 60.0)
    assert result["qualified"] is True
    assert result["status"] == "QUALIFIED"
    assert result["shared_points"] == 1501
    assert result["combined_rmse_normalized"] == 0.0
    assert result["common_frequency_rmse_normalized"] == 0.0


def test_compare_pair_rejects_label_mismatch_and_missing_event():
    result = compare_pair(_event(), _event(label="unstable"), mass=np.ones(2), omega_s=1.0)
    assert result["qualified"] is False
    assert result["checks"]["label_agreement"] is False
    missing = compare_pair(_event(), None, mass=np.ones(2), omega_s=1.0)
    assert missing["status"] == "UNKNOWN"
    assert missing["missing_fine"] is True


def test_compare_pair_rejects_nonshared_grid_without_extrapolation():
    coarse = _event(t=np.array([0.0, 0.01, 0.02]))
    fine = _event(t=np.array([0.0, 0.01, 0.03]))
    result = compare_pair(coarse, fine, mass=np.ones(2), omega_s=1.0)
    assert result["checks"]["shared_grid"] is False
    assert result["qualified"] is False


def test_compare_pair_nonzero_common_frequency_metric_and_los_tolerance():
    coarse = _event(label="unstable", t=np.arange(0.0, 1.21, 0.01))
    fine = _event(label="unstable", t=np.arange(0.0, 1.21, 0.01))
    fine["attrs"]["event_time_s"] = 1.02
    fine["arrays"]["omega"] = fine["arrays"]["omega"] + 1e-5
    result = compare_pair(coarse, fine, mass=np.ones(2), omega_s=2.0 * np.pi * 60.0)
    assert result["checks"]["los_agreement"] is True
    assert result["common_frequency_rmse_normalized"] > 0.001
    assert result["qualified"] is False
    fine["attrs"]["event_time_s"] = 1.03
    rejected = compare_pair(coarse, fine, mass=np.ones(2), omega_s=1.0)
    assert rejected["checks"]["los_agreement"] is False


def test_compare_pair_nonzero_zero_mean_coi_delta_has_expected_combined_metric():
    coarse = _event()
    fine = _event()
    error = 1e-4
    fine["arrays"]["delta"] = fine["arrays"]["delta"] + np.array([error, -error])
    result = compare_pair(coarse, fine, mass=np.ones(2), omega_s=1.0)
    assert result["delta_rmse_normalized"] == pytest.approx(error / checker.S_DELTA)
    assert result["combined_rmse_normalized"] == pytest.approx(error / checker.S_DELTA / np.sqrt(2.0))


def test_compare_pair_latency_and_nonfinite_export_fail_closed():
    short = _event(label="unstable", t=np.array([0.0, 0.1]))
    short["attrs"]["event_time_s"] = 0.05
    result = compare_pair(short, _event(label="unstable", t=np.array([0.0, 0.1])), mass=np.ones(2), omega_s=1.0)
    assert result["checks"]["future_coverage"] is False
    assert result["qualified"] is False
    bad = _event()
    bad["arrays"]["delta"][3, 0] = np.nan
    result = compare_pair(bad, _event(), mass=np.ones(2), omega_s=1.0)
    assert result["checks"]["finite_all_exported"] is False
    assert result["qualified"] is False


def test_load_selection_requires_exact_unique_boundary_ids(tmp_path: Path):
    rows = [
        {
            "line": "L1",
            "location": 0.5,
            "bus": i + 1,
            "fault_type": "3ph",
            "xf": 0.1,
            "duration_s": 0.1 + i * 0.001,
        }
        for i in range(20)
    ]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selected_events": rows}), encoding="utf-8")
    assert len(load_selection(path)) == 20
    path.write_text(json.dumps({"selected_events": rows[:19]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 20"):
        load_selection(path)


def test_read_root_preserves_all_failed_attempts_without_index(tmp_path: Path):
    root = tmp_path / "empty_boundary"
    root.mkdir()
    (root / "splits.json").write_text(json.dumps({"train": [], "validation": [], "test": []}), encoding="utf-8")
    with h5py.File(root / "dataset.h5", "w") as file:
        file.attrs["attempted"] = 20
        file.attrs["config"] = json.dumps({"integration_step_s": 0.005})
        file.attrs["discard_counts"] = json.dumps({"numerical_TDS_failure": 20})
        file.create_group("events")
    loaded = _read_root(root, [{} for _ in range(20)])
    assert loaded["events"] == {}
    assert loaded["rejected"] == {"numerical_TDS_failure": 20}
    assert loaded["hashes"]["dataset_index.csv"] is None


def test_read_root_rejects_nontrain_split_before_opening_h5(tmp_path: Path, monkeypatch):
    root = tmp_path / "mixed_boundary"
    root.mkdir()
    (root / "dataset_index.csv").write_text("event_id\nboundary_0000\n", encoding="utf-8")
    (root / "splits.json").write_text(json.dumps({"train": [], "validation": ["boundary_0000"], "test": []}), encoding="utf-8")
    (root / "dataset.h5").write_bytes(b"must not be opened")

    def fail_open(*args, **kwargs):
        raise AssertionError("H5 opened before split guard")

    monkeypatch.setattr(checker.h5py, "File", fail_open)
    with pytest.raises(ValueError, match="all-TRAIN"):
        _read_root(root, [{} for _ in range(20)])
