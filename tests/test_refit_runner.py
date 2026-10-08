from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.run_refit_v2 import (
    H100_POINTS,
    atomic_write_json,
    h100_event_report,
    h100_report,
    json_safe,
    normalize_json_file,
    sha256_ids,
)


class _FakeModel:
    def __init__(self, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()

    def rollout(self, initial, time, exogenous):
        n = len(time)
        return np.zeros((n, 2)), np.zeros((n, 2))


def _sample(n: int, event_id: str = "e1") -> dict:
    time = np.arange(n, dtype=float) * 0.01
    zeros = np.zeros((n, 2), dtype=float)
    return {
        "event_id": event_id,
        "model_inputs": {
            "static": {"delta0": np.zeros(2), "omega0": np.zeros(2)},
            "exogenous": {"t": time, "Pm": zeros},
        },
        "training_context": {
            "targets": {"t": time, "delta": zeros, "omega": zeros},
            "label": "stable",
            "claim_type": "causal_zoh_rollout",
            "metadata": {
                "source_dataset": "synthetic",
                "fault_line": "1-2",
            },
        },
    }


def test_h100_report_does_not_shorten_short_events() -> None:
    short = h100_event_report(_FakeModel(), _sample(H100_POINTS - 1, "short"))
    assert short["status"] == "excluded_short"
    assert short["available_samples"] == H100_POINTS - 1
    assert "trajectory_metrics" not in short


def test_h100_report_counts_eligible_failures_and_successes() -> None:
    samples = [_sample(H100_POINTS, "ok"), _sample(H100_POINTS, "bad"), _sample(20, "short")]
    class EventFailModel(_FakeModel):
        def rollout(self, initial, time, exogenous):
            if len(time) == H100_POINTS and exogenous["Pm"].shape[1] == 2:
                # The second sample is made distinguishable by its zero-valued
                # data only through the call order in this tiny fake model.
                self.calls = getattr(self, "calls", 0) + 1
                if self.calls == 2:
                    raise RuntimeError("synthetic rollout failure")
            return super().rollout(initial, time, exogenous)

    report = h100_report(EventFailModel(), samples)
    assert report["aggregate"]["n_declared"] == 3
    assert report["aggregate"]["n_eligible"] == 2
    assert report["aggregate"]["n_success"] == 1
    assert report["aggregate"]["n_failures"] == 1
    assert report["aggregate"]["failure_rate"] == 0.5
    assert len(report["events"]) == 2
    assert len(report["excluded_short_events"]) == 1
    assert report["events"][0]["n_samples"] == H100_POINTS
    assert report["events"][0]["endpoint_seconds"] == 1.0
    assert report["events"][1]["status"] == "failure"


def test_json_safe_and_atomic_write_are_strict() -> None:
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        path = Path(directory) / "summary.json"
        atomic_write_json(path, {"finite": np.float64(1.5), "nan": float("nan")})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == {"finite": 1.5, "nan": None}
    assert json_safe(np.array([1, 2])) == [1, 2]


def test_model_sidecar_normalization_removes_nonfinite_json() -> None:
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        path = Path(directory) / "model.json"
        path.write_text('{"nan": NaN, "pos_inf": Infinity}', encoding="utf-8")
        normalize_json_file(path)
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "nan": None,
            "pos_inf": None,
        }


def test_id_hash_is_order_sensitive_and_deterministic() -> None:
    assert sha256_ids(["a", "b"]) == sha256_ids(["a", "b"])
    assert sha256_ids(["a", "b"]) != sha256_ids(["b", "a"])
