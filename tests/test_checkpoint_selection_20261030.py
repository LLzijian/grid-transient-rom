"""Synthetic audit tests. No private payloads, optimizer or real model rollouts."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import checkpoint_selection_20261030 as audit


def test_import_and_registry_are_inert(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("import may not read any payload")
    monkeypatch.setattr(Path, "read_text", forbidden)
    spec = importlib.util.spec_from_file_location("audit_inert", audit.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.LABELS) == 11
    assert module.model_spec("step000016")["export"].name == "model.json"
    with pytest.raises(ValueError):
        module.model_spec("step000001")


def test_exclusive_json_never_overwrites_and_rejects_nan(tmp_path):
    path = tmp_path / "out.json"
    audit.write_once(path, {"a": 1})
    with pytest.raises(FileExistsError):
        audit.write_once(path, {"a": 2})
    assert audit.read(path) == {"a": 1}
    with pytest.raises(ValueError):
        audit.write_once(tmp_path / "invalid.json", {"a": float("nan")})


def trow(truth=0., fine=.2, coarse=.21, qualified=True):
    return {"truth": {"time_s": truth}, "fine": {"time_s": fine}, "coarse": {"time_s": coarse}, "numerically_qualified": qualified}


def test_common_T_uses_only_joint_qualified_confirmations():
    profiles = {"R_N": {"timing_actual15": {"ok": trow(), "bad": trow(qualified=False), "none": trow()}},
                "step000016": {"timing_actual15": {"ok": trow(fine=.1, coarse=.13), "bad": trow(), "none": trow(truth=None)}}}
    result = audit.timing_common(profiles)
    assert result["common_event_ids"] == ["ok"]
    assert result["metrics"]["R_N"]["E"] == pytest.approx(.2)
    assert result["metrics"]["step000016"]["u"] == pytest.approx(.03)
    assert audit.timing_common({})["status"] == "UNRESOLVED"


def test_classification_projection_keeps_numerical_and_uncovered_identity(monkeypatch):
    def extra(eid, label, category, qualified, los=False):
        return {"event_id": eid, "label": label, "scientific_numeric_8s": qualified,
                "classification_8s": {"fine_classification": {"category": category, "predicted_los": los, "trigger_index": None}}}
    extras = [extra("miss", "unstable", "window_no_los_unsettled", True),
              extra("false", "stable", "predicted_los", True, True),
              extra("numeric", "unstable", "window_no_los_unsettled", False),
              extra("uncovered", "unstable", "latency_uncovered", False)]
    original = {"event_count": 4, "missed_resolved_unstable": 1, "false_los_resolved_stable": 1}
    result = audit.classification_detail(extras, {"classification_8s": original}, "classification_8s")
    assert result["denominator"] == 4
    assert result["missed_ids"] == ["miss"]
    assert result["false_los_ids"] == ["false"]
    assert result["numerical_unresolved_ids"] == ["numeric", "uncovered"]
    assert result["latency_uncovered_ids"] == ["uncovered"]
    assert result["window_no_los_unsettled_ids"] == ["miss", "numeric"]


@pytest.fixture
def synthetic_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "private_data/fixture/predictions.npz"
    path.parent.mkdir(parents=True)
    history = np.zeros((11, 30))
    np.savez(path, event_ids=np.array(["e"]), histories=history[None], fine=np.zeros((1, 3001, 40)), coarse=np.zeros((1, 1501, 40)))
    events = [{"id": "e", "label": "stable", "x": history.copy()}]
    evidence = {"event_count": 1, "val_read": False, "test_read": False,
                "history_dt_main": .0025, "history_dt_reference": .00125,
                "prediction_files": [{"path": str(path), "sha256": audit.sha(path)}], "rows": {"numerical": []}, "extras": [{"event_id": "e"}]}
    return events, evidence, path


def test_predictions_require_exact_ids_histories_and_paired_shapes(synthetic_predictions):
    events, evidence, _ = synthetic_predictions
    audit.validate_predictions(evidence, events)
    events[0]["x"][0, 0] = 1
    with pytest.raises(ValueError, match="history"):
        audit.validate_predictions(evidence, events)


def test_prediction_hash_change_is_rejected(synthetic_predictions):
    events, evidence, path = synthetic_predictions
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        audit.validate_predictions(evidence, events)


def test_duplicate_prediction_events_are_rejected(synthetic_predictions):
    events, evidence, _ = synthetic_predictions
    evidence["prediction_files"] *= 2
    with pytest.raises(ValueError, match="duplicated"):
        audit.validate_predictions(evidence, events)


def test_complete_receipt_binds_manifest_summary_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    folder = tmp_path / "private_data/cp"
    audit.write_once(folder / "summary.json", {"status": "COMPLETED"})
    artifact = folder / "per_event.json"
    audit.write_once(artifact, {"e": 1})
    audit.write_once(folder / "complete.json", {"manifest_sha256": "frozen", "summary_sha256": audit.sha(folder / "summary.json"), "artifacts": [audit.identity(artifact)]})
    assert audit.verify_complete(folder, "frozen")["status"] == "COMPLETED"
    with pytest.raises(ValueError, match="receipt"):
        audit.verify_complete(folder, "different")
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="artifact"):
        audit.verify_complete(folder, "frozen")


def test_complete_checkpoint_never_loads_model_or_replays(tmp_path, monkeypatch):
    import closure_probe_20261030 as probe
    monkeypatch.setattr(audit, "PRIVATE", tmp_path)
    folder = tmp_path / "step000016"
    audit.write_once(folder / "complete.json", {})
    expected = {"checkpoint": "step000016"}
    monkeypatch.setattr(audit, "verify_complete", lambda *a: expected)
    monkeypatch.setattr(probe, "load_model", lambda *a: pytest.fail("completed checkpoint must not rerun"))
    assert audit.evaluate_one("step000016", [], {}, "x") == expected


def test_unknown_attempt_is_never_replayed_even_retry_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PRIVATE", tmp_path)
    audit.write_once(tmp_path / "step000016/attempt001/started.json", {"pid": 42})
    with pytest.raises(RuntimeError, match="reconcile"):
        audit.evaluate_one("step000016", [], {}, "x", retry_failed=True)


def test_terminal_failure_needs_explicit_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PRIVATE", tmp_path)
    audit.write_once(tmp_path / "step000016/attempt001/failure.json", {"status": "FAILED"})
    with pytest.raises(RuntimeError, match="retry-failed"):
        audit.evaluate_one("step000016", [], {}, "x")
