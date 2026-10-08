"""Isolated multiblock regression for the 0101 TRAIN metadata repair candidate."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


STAGE = Path(__file__).resolve().parents[1] / "experiments"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_shared_closure_0101",
    STAGE / "evaluate_shared_closure_0101.py",
)
assert SPEC and SPEC.loader
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def _metric_rows(event):
    horizons = ("1", "2", "5", "8", "prefix")
    shared = {"event_id": event["id"], "group": event["group"],
              "label": event["label"], "available": True, "finite": True,
              "numerically_admissible": True}
    return {
        "records": [dict(shared, horizon=h, rmse=0.1, sse=1.0, n=100) for h in horizons],
        "common_frequency": [dict(shared, horizon=h, rmse=0.2, sse=2.0, n=100) for h in horizons],
        "normalized_p": [dict(shared, horizon=h, rmse=0.3, sse=3.0, n=100) for h in horizons],
        "objective_point_error": [dict(shared, horizon=h, mean_point_error=0.4) for h in horizons],
        "numerical": [dict(shared, horizon=h) for h in horizons],
    }


def test_event_metadata_keeps_all_ids_in_order_across_rollout_blocks(tmp_path, monkeypatch):
    fake_rollouts = types.ModuleType("evaluate_structured_rom")
    fake_rollouts._rollouts = lambda model, histories, steps, dt: np.zeros(
        (len(histories), steps + 1, 40), dtype=np.float64
    )
    fake_metrics = types.ModuleType("summarize_train_local_memory_0098")
    fake_metrics.HORIZONS = ("1", "2", "5", "8", "prefix")
    fake_metrics.event_metrics = lambda model, event, fine, coarse: _metric_rows(event)
    fake_metrics.build_summary = lambda records: {"train_record_count": len(records["records"])}
    monkeypatch.setitem(sys.modules, "evaluate_structured_rom", fake_rollouts)
    monkeypatch.setitem(sys.modules, "summarize_train_local_memory_0098", fake_metrics)
    monkeypatch.setattr(candidate, "_check_deadline", lambda *args: None)
    monkeypatch.setattr(candidate, "_memory_bytes", lambda: 0)

    events = []
    for i in range(40):
        events.append({
            "id": f"train_{i:03d}", "group": f"group_{i % 24:02d}",
            "label": "unstable" if i < 35 else "stable",
            "x": np.zeros((12, 30), dtype=np.float64),
        })
    result = candidate._train_summary(object(), events, tmp_path / "train", float("inf"))

    expected_ids = [event["id"] for event in events]
    rows = result["event_metadata"]
    assert len(result["batch_traces"]) == 2
    assert [row["event_id"] for row in rows] == expected_ids
    assert all(row["truth_length"] == 12 for row in rows)
    assert all(row.get("canonical_train_truth_sha256") for row in rows)
    assert len(result["unstable_prefix_event_rows"]["records"]) == 35
    assert result["summary"]["train_record_count"] == 40 * 5
    persisted = json.loads((tmp_path / "train" / "evidence.json").read_text())
    assert [row["event_id"] for row in persisted["event_metadata"]] == expected_ids
