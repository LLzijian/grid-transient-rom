import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import closure_discovery_data as data


def _metadata_roots(tmp_path, monkeypatch):
    roots = []
    for name, event_prefix in (("ieee39_postfault", "pilot"), ("ieee39_expansion", "expansion")):
        root = tmp_path / name
        root.mkdir()
        roots.append(root)
        rows = [
            {"event_id": f"{event_prefix}_train", "fault_bus": "1", "fault_line": "Line_1",
             "fault_location": "0.0", "fault_duration_s": "0.1", "fault_xf": "0.001"},
            {"event_id": f"{event_prefix}_val", "fault_bus": "2", "fault_line": "Line_2",
             "fault_location": "1.0", "fault_duration_s": "0.2", "fault_xf": "0.01"},
            {"event_id": f"{event_prefix}_test", "fault_bus": "3", "fault_line": "Line_3",
             "fault_location": "0.0", "fault_duration_s": "0.3", "fault_xf": "0.001"},
        ]
        with (root / "dataset_index.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (root / "splits.json").write_text(json.dumps({
            "train": [f"{event_prefix}_train"],
            "validation": [f"{event_prefix}_val"],
            "test": [f"{event_prefix}_test"],
        }), encoding="utf-8")
    monkeypatch.setattr(data, "DATASET_ROOTS", tuple(roots))
    reference = _reference_fingerprint()
    monkeypatch.setattr(data, "_load_reference_fingerprint", lambda: reference)
    return roots


def _condition():
    return {"bus_id": "1", "duration_s": 0.125, "xf": 0.00141421356237,
            "split": "train"}


def _base():
    return SimpleNamespace(pm_star=np.full(10, 0.5, dtype=np.float64),
                           mass=np.ones(10, dtype=np.float64))


def _reference_fingerprint():
    equilibrium = {"delta": np.linspace(0.1, 0.2, 10),
                   "omega": np.ones(10), "Pm": np.full(10, 0.5),
                   "Pe": np.full(10, 0.5), "full_state": np.zeros(1),
                   "algebraic_state": np.zeros(1)}
    reference = {
        "source": "fixture", "sources": ["ieee39_postfault", "ieee39_expansion"],
        "andes_version": "2.0.0", "case": "ieee39/ieee39_full.xlsx",
        "case_sha256": "a" * 64, "omega_s": 2 * np.pi * 60,
        "mva_base": 100.0, "postfault_signature": "f" * 64,
        "raw_units": "delta rad; omega pu; powers on system base",
        "equilibrium": equilibrium, "inertia": np.arange(1.0, 11.0),
        "state_mapping": [[i, "x", "GENROU", str(i)] for i in range(40)],
        "algebraic_names": [],
        "config_checks": {"andes_version": "2.0.0", "case": "ieee39/ieee39_full.xlsx",
                           "disable_pss": False, "fault_type": "temporary_bus_three_phase_ground",
                           "postfault_horizon_s": 15.0, "sample_interval_s": 0.01},
    }
    reference["fingerprint_sha256"] = data._reference_fingerprint_sha256(reference)
    return reference


def _result(config, *, label="stable", offset=0.0):
    t = np.arange(0.0, 15.0 + 1e-9, config["sample_interval_s"])
    eq_delta = np.linspace(0.1, 0.2, 10)
    delta = eq_delta[None, :] + offset + np.zeros((len(t), 10))
    omega = np.ones((len(t), 10))
    pm = np.full((len(t), 10), 0.5)
    te = np.full((len(t), 10), 0.8)
    arrays = {"t": t, "delta": delta, "omega": omega, "Pm": pm, "Pe": pm,
              "te": te, "full_state": t[:, None], "algebraic_state": t[:, None],
              "bus_v": np.ones((len(t), 1)), "bus_a": np.zeros((len(t), 1)),
              "terminal_v": np.ones((len(t), 1))}
    metadata = {"label": label, "event_time_s": "", "clearing_time_s": 1.125,
                "record_start_delay_s": 0.0001, "trajectory_end_s": float(t[-1]),
                "valid_length": len(t), "postfault_signature": "f" * 64}
    equilibrium = {"delta": eq_delta, "omega": np.ones(10), "Pm": np.full(10, 0.5),
                   "Pe": np.full(10, 0.5), "full_state": np.zeros(1),
                   "algebraic_state": np.zeros(1)}
    return arrays, metadata, equilibrium, np.arange(1.0, 11.0), [[i, "x", "GENROU", str(i)] for i in range(40)], []


def test_allowed_conditions_reads_only_explicit_train_index(tmp_path, monkeypatch):
    roots = _metadata_roots(tmp_path, monkeypatch)
    rows = data.allowed_train_conditions()
    assert len(rows) == 2
    assert {row["event_id"] for row in rows} == {"pilot_train", "expansion_train"}
    assert all(set(("bus", "line", "location", "duration_s", "xf", "event_id", "root")) <= row.keys()
               for row in rows)
    assert not (roots[0] / "dataset.h5").exists()


def test_new_train_selection_uses_nearest_scored_old_train_event(tmp_path, monkeypatch):
    _metadata_roots(tmp_path, monkeypatch)
    picked = data.select_new_train_conditions(
        [_condition()], {"pilot_train": 7.0, "expansion_train": 2.0}, count=1)
    assert len(picked) == 1
    assert picked[0]["nearest_train_event_id"] == "pilot_train"
    assert picked[0]["selection_score"] == 7.0


def test_formal_new_train_selection_balances_duration_before_scores(tmp_path,monkeypatch):
    _metadata_roots(tmp_path,monkeypatch)
    pool=[{**_condition(),'duration_s':duration,'xf':.001+.00001*i} for duration in (.125,.225,.35) for i in range(10)]
    selected=data.select_new_train_conditions(pool,{'pilot_train':7.,'expansion_train':2.},count=24)
    assert {s:sum(r['stratum']==s for r in selected) for s in ('short','medium','long')}=={'short':8,'medium':8,'long':8}


def test_generation_keeps_failed_attempt_and_runs_both_steps(tmp_path, monkeypatch):
    _metadata_roots(tmp_path, monkeypatch)
    calls = []

    def simulator(condition, config, diagnostics=None):
        calls.append(config["integration_step_s"])
        assert config["postfault_horizon_s"] == 15.0
        assert config["disable_pss"] is False
        if config["integration_step_s"] == 0.005:
            diagnostics["attempted"] = True
            raise ValueError("synthetic numerical failure")
        return _result(config)

    artifact = tmp_path / "run"
    manifest = data.generate_conditions([_condition()], artifact, "train", 60.0,
                                        simulator=simulator)
    assert calls == [0.005, 0.0025]
    attempt = manifest["attempts"][0]
    assert attempt["steps"]["0.005"]["status"] == "simulation_failed"
    assert attempt["steps"]["0.005"]["diagnostics"]["attempted"] is True
    assert attempt["steps"]["0.005"]["diagnostics"]["system_attestation"]["source"] == "injected_simulator"
    assert attempt["steps"]["0.0025"]["status"] == "completed"
    assert attempt["pair_qualification"]["pass"] is False
    events, excluded = data.load_new_events(artifact, "train", base=_base())
    assert events == []
    assert excluded[0]["reason"] == "raw_simulation_or_validation_failure"


def test_qualified_event_preserves_canonical_units_and_actual_time(tmp_path, monkeypatch):
    _metadata_roots(tmp_path, monkeypatch)
    calls = []

    def simulator(condition, config, diagnostics=None):
        calls.append(config["integration_step_s"])
        return _result(config)

    artifact = tmp_path / "run"
    manifest = data.generate_conditions([_condition()], artifact, "train", 60.0,
                                        simulator=simulator)
    assert calls == [0.005, 0.0025]
    assert manifest["attempts"][0]["pair_qualification"]["pass"] is True
    events, excluded = data.load_new_events(artifact, "train", base=_base())
    assert excluded == []
    event = events[0]
    assert event["x"].shape == (1501, 30)
    assert event["torque"].shape == (1501, 10)
    assert np.allclose(event["x"][:, :10], 0.0)
    assert np.allclose(event["x"][:, 10:20], 0.0)
    assert np.allclose(event["x"][:, 20:30], 0.0)
    assert np.allclose(event["torque"], 0.3)
    assert event["absolute_t"][0] == pytest.approx(1.1251)
    assert event["classification_meta"]["omega_s"] == pytest.approx(2 * np.pi * 60)
    assert event["classification_meta"]["system_attestation_complete"] is True
    assert event["new_split"] == "train"


def test_events_without_100ms_coverage_are_counted_as_latency_excluded(tmp_path, monkeypatch):
    _metadata_roots(tmp_path, monkeypatch)

    def simulator(condition, config, diagnostics=None):
        arrays, metadata, equilibrium, inertia, mapping, names = _result(config, label="unstable")
        arrays = {key: value[:11] if np.asarray(value).ndim and len(value) == 1501 else value
                  for key, value in arrays.items()}
        metadata.update({"event_time_s": 0.1, "trajectory_end_s": 0.1, "valid_length": 11})
        return arrays, metadata, equilibrium, inertia, mapping, names

    artifact = tmp_path / "latency"
    data.generate_conditions([_condition()], artifact, "train", 60.0, simulator=simulator)
    events, excluded = data.load_new_events(artifact, "train", base=_base())
    assert events == []
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "latency_uncovered"
    assert excluded[0]["trajectory_end_s"] == pytest.approx(0.1)


@pytest.mark.parametrize("mismatch", ["equilibrium", "case_hash"])
def test_system_fingerprint_mismatch_is_retained_as_data_failure(tmp_path, monkeypatch, mismatch):
    _metadata_roots(tmp_path, monkeypatch)

    def simulator(condition, config, diagnostics=None):
        result = _result(config)
        if mismatch == "equilibrium":
            result[2]["delta"] = result[2]["delta"].copy()
            result[2]["delta"][0] += 2e-9
        else:
            diagnostics["system_attestation"]["case_sha256"] = "b" * 64
        return result

    artifact = tmp_path / f"fingerprint_{mismatch}"
    manifest = data.generate_conditions([_condition()], artifact, "train", 60.0,
                                        simulator=simulator)
    attempt = manifest["attempts"][0]
    assert attempt["pair_qualification"]["pass"] is False
    failed = [row for row in attempt["steps"].values()
              if row["status"] == "raw_validation_failed"]
    assert len(failed) == 2
    expected_text = "equilibrium mismatch" if mismatch == "equilibrium" else "case-file SHA256"
    assert all(expected_text in row["raw_validation"]["error"] for row in failed)
    assert all(row["metadata"]["label"] == "stable" for row in attempt["steps"].values()
               if "metadata" in row)
    events, excluded = data.load_new_events(artifact, "train", base=_base())
    assert events == []
    assert excluded[0]["reason"] == "raw_simulation_or_validation_failure"


def test_val_freeze_is_checked_before_generation_and_again_at_load(tmp_path, monkeypatch):
    _metadata_roots(tmp_path, monkeypatch)
    model_path = tmp_path / "model.json"
    model_path.write_text('{"fixed":true}', encoding="utf-8")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    freeze_path = tmp_path / "frozen.json"
    freeze_path.write_text(json.dumps({"main_name": "main", "models": [
        {"name": "main", "path": str(model_path), "sha256": model_hash},
    ]}), encoding="utf-8")
    calls = []

    def simulator(condition, config, diagnostics=None):
        calls.append(config["integration_step_s"])
        return _result(config)

    wrong_freeze = tmp_path / "wrong.json"
    wrong_freeze.write_text(json.dumps({"models": [
        {"path": str(model_path), "sha256": "0" * 64},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.generate_conditions([_condition()], tmp_path / "rejected", "val", 60.0,
                                 freeze_path=wrong_freeze, simulator=simulator)
    assert calls == []

    artifact = tmp_path / "val_run"
    val_condition = {**_condition(), "split": "val"}
    data.generate_conditions([val_condition], artifact, "val", 60.0,
                             freeze_path=freeze_path, simulator=simulator)
    events, excluded = data.load_new_events(artifact, "val", freeze_path, base=_base())
    assert len(events) == 1 and excluded == []
    model_path.write_text('{"fixed":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load_new_events(artifact, "val", freeze_path, base=_base())


def test_budget_prevents_any_new_simulator_call_and_test_is_forbidden(tmp_path, monkeypatch):
    _metadata_roots(tmp_path, monkeypatch)
    calls = []

    def simulator(condition, config, diagnostics=None):
        calls.append(config["integration_step_s"])
        return _result(config)

    result = data.generate_conditions([_condition()], tmp_path / "budget", "train", 0.0,
                                      simulator=simulator)
    assert calls == []
    assert result["status"] == "budget_exhausted"
    assert all(step["status"] == "not_run_budget_exhausted"
               for step in result["attempts"][0]["steps"].values())
    with pytest.raises(ValueError, match="train.*val"):
        data.generate_conditions([], tmp_path / "test", "test", 1.0, simulator=simulator)
    with pytest.raises(ValueError, match="train.*val"):
        data.load_new_events(tmp_path / "budget", "test")
