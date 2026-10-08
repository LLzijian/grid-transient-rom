from __future__ import annotations

import numpy as np

from kiki.models.base import RunContext
from kiki.models.prunable_energy_model import (
    PrunableEnergyModel,
    _eligible_horizon_stops,
)


class _ReplayModel:
    def __init__(self) -> None:
        self.maximum_rollout_points = 0

    def rollout(self, initial_state, time_grid, exogenous):
        self.maximum_rollout_points = max(self.maximum_rollout_points, len(time_grid))
        t = np.asarray(time_grid, dtype=float)
        delta = np.column_stack((np.sin(t), -np.sin(t)))
        omega = np.column_stack((np.cos(t), -np.cos(t)))
        return delta, omega


def _sample(event_id: str, stop: float, group: str) -> dict:
    t = np.arange(0.0, stop + 0.5, 1.0)
    delta = np.column_stack((np.sin(t), -np.sin(t)))
    omega = np.column_stack((np.cos(t), -np.cos(t)))
    return {
        "event_id": event_id,
        "model_inputs": {
            "static": {
                "delta0": delta[[0]],
                "omega0": omega[[0]],
                "mass": np.ones(2),
                "incidence": np.asarray([[1.0], [-1.0]]),
            },
            "exogenous": {"t": t, "Pm": np.zeros((len(t), 2))},
        },
        "training_context": {
            "targets": {"t": t, "delta": delta, "omega": omega},
            "claim_type": "causal_zoh_rollout",
            "metadata": {"source_dataset": "source", "fault_line": group},
        },
    }


def test_horizon_stops_require_real_coverage() -> None:
    assert _eligible_horizon_stops(np.arange(0.0, 2.1, 0.1)) == {1.0: 11}
    assert _eligible_horizon_stops(np.arange(0.0, 10.1, 0.1)) == {
        1.0: 11,
        5.0: 51,
        10.0: 101,
    }


def test_source_degradation_uses_true_physical_horizons() -> None:
    short = _sample("short", 2.0, "L1")
    long = _sample("long", 10.0, "L2")
    reference = _ReplayModel()
    candidate = _ReplayModel()
    report = PrunableEnergyModel.source_stratified_degradation(
        reference,
        candidate,
        [short, long],
        RunContext(horizon_steps=2, max_cases=2),
        bootstrap_draws=4,
    )
    assessed = {
        row["horizon_seconds"]: row["events"]
        for row in report["rows"]
        if row["status"] == "assessed"
    }
    assert assessed == {1.0: 2, 5.0: 1, 10.0: 1}
    assert reference.maximum_rollout_points == 11
    assert candidate.maximum_rollout_points == 11


def test_validation_summary_does_not_relabel_one_second_as_ten() -> None:
    sample = _sample("one-second", 1.0, "L1")
    model = PrunableEnergyModel(active_blocks=())
    model.validation_scales = {"edge_angle": 1.0, "omega": 1.0}
    replay = _ReplayModel()
    model.rollout = replay.rollout  # type: ignore[method-assign]
    summary = model.validation_summary([sample], RunContext(horizon_steps=1, max_cases=1))
    assert np.isfinite(summary["horizon_standardized_loss"]["1s"])
    assert np.isinf(summary["horizon_standardized_loss"]["5s"])
    assert np.isinf(summary["horizon_standardized_loss"]["10s"])
