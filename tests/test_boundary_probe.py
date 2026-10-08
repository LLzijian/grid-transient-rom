from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments import probe_boundary_numerics as probe

ROOT = Path(__file__).resolve().parents[1]


def test_preregistered_cases_and_arms_are_exact():
    assert probe.EVENT_IDS == ("boundary_0000", "boundary_0002", "boundary_0015")
    assert probe.ARMS == (
        ("repeat_0025_tol1e-7", 0.0025, 1e-7),
        ("refine_00125_tol1e-7", 0.00125, 1e-7),
        ("refine_000625_tol1e-7", 0.000625, 1e-7),
        ("tol_0025_tol1e-9", 0.0025, 1e-9),
    )


def test_source_loader_rejects_non_train_split(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "splits.json").write_text(json.dumps({"train": list(probe.EVENT_IDS),
                                                        "validation": ["v"], "test": []}))
    with pytest.raises(ValueError, match="non-TRAIN"):
        probe._load_source(source)


def test_source_loader_requires_selected_ids_in_train(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "splits.json").write_text(json.dumps({"train": [probe.EVENT_IDS[0]],
                                                        "validation": [], "test": []}))
    with pytest.raises(ValueError, match="not all TRAIN"):
        probe._load_source(source)


def test_generator_has_observational_diagnostics_without_return_contract_change():
    source = (ROOT / "experiments" / "generate_private_dataset.py").read_text(encoding="utf-8")
    assert "def simulate(event, config, solver_workdir=None, diagnostics=None):" in source
    assert "return arrays, metadata, equilibrium, inertia, mapping, ss.dae.y_name" in source
    assert '"initial_f"' in source and '"event_brackets"' in source


def _stable_case(*, omega_offset=0.0, event_time=""):
    t = np.arange(0.0, 15.01, 0.01)
    zeros = np.zeros((len(t), 2))
    omega = zeros + omega_offset
    attrs = {"label": "stable", "postfault_signature": "sig",
             "record_start_delay_s": 0.0, "clearing_time_s": 1.5,
             "trajectory_end_s": 15.0, "event_time_s": event_time}
    return {"attrs": attrs, "arrays": {"t": t, "delta": zeros.copy(),
            "omega": omega, "Pm": zeros.copy(), "Pe": zeros.copy(),
            "Qe": zeros.copy(), "te": zeros.copy()},
            "provenance": {}}


def test_repeat_gate_requires_strict_normalized_metrics_and_identity():
    base = _stable_case()
    assert probe._repeat_gate(base, _stable_case(), mass=np.ones(2), omega_s=1.0)["qualified"]
    common_error = _stable_case(omega_offset=5e-8)
    rejected = probe._repeat_gate(base, common_error, mass=np.ones(2), omega_s=1.0)
    assert not rejected["qualified"]
    changed_stop = _stable_case(event_time=0.1)
    assert not probe._repeat_gate(base, changed_stop, mass=np.ones(2), omega_s=1.0)["qualified"]
    changed_grid = _stable_case()
    changed_grid["arrays"]["t"] = changed_grid["arrays"]["t"].copy()
    changed_grid["arrays"]["t"][1] += 1e-7
    assert not probe._repeat_gate(base, changed_grid, mass=np.ones(2), omega_s=1.0)["qualified"]
    unknown = probe._repeat_gate(None, base, mass=np.ones(2), omega_s=1.0)
    assert unknown["status"] == "UNKNOWN" and not unknown["qualified"]


def test_run_preserves_all_12_attempts_and_marks_failed_tolerance_unknown(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("dataset_index.csv", "splits.json"):
        (source / name).write_text("synthetic", encoding="utf-8")
    case_file = tmp_path / "case.xlsx"
    case_file.write_bytes(b"case")
    h5_path = source / "dataset.h5"
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset("inertia", data=np.ones(2))
    config = {"case": str(case_file), "integration_step_s": 0.0025,
              "newton_tolerance": 1e-7}
    events = {event_id: {"bus": i} for i, event_id in enumerate(probe.EVENT_IDS)}
    old = {event_id: _stable_case() for event_id in probe.EVENT_IDS}
    for row in old.values():
        row["provenance"] = {"case_sha256": __import__("hashlib").sha256(case_file.read_bytes()).hexdigest(),
                              "postfault_signature": "sig", "state_mapping": [[0, "x", "M", "1"]],
                              "equilibrium_full_state": np.zeros(1), "inertia": np.ones(2)}
    monkeypatch.setattr(probe, "_load_source", lambda _: (config, events, old))

    def fake_simulate(event, arm_config, solver_workdir=None, diagnostics=None):
        if arm_config["newton_tolerance"] == 1e-9 and event["bus"] == 2:
            raise ValueError("synthetic_failure")
        diagnostics["label"] = "stable"
        arr = _stable_case()["arrays"]
        metadata = {"postfault_signature": "sig", "label": "stable",
                    "record_start_delay_s": 0.0, "clearing_time_s": 1.5,
                    "trajectory_end_s": 15.0, "event_time_s": ""}
        equilibrium = {"full_state": np.zeros(1)}
        return arr, metadata, equilibrium, np.ones(2), [[0, "x", "M", "1"]], ["y"]

    monkeypatch.setattr(probe, "simulate", fake_simulate)
    monkeypatch.setattr(probe.andes, "get_case", lambda _: str(case_file))
    output = tmp_path / "output"
    report = probe.run(source, output)
    assert report["attempts"] == 12
    assert len(list(output.glob("*.json"))) == 13  # 12 arms plus report
    case = next(row for row in report["cases"] if row["event_id"] == "boundary_0015")
    assert sum(item["status"] for item in case["arms"]) == 3
    tol = next(row for row in report["comparisons"]
               if row["event_id"] == "boundary_0015" and row["comparison"] == "repeat_vs_tight_tolerance")
    assert tol["status"] == "UNKNOWN"
