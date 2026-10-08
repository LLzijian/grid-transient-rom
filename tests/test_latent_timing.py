"""Synthetic contracts for Loop0067 timing; never open private TRAIN/TEST data."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("latent_timing", ROOT / "experiments/benchmark_latent_timing.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def test_matched_interval_caps_at_eight_seconds_and_true_prefix():
    event = {"x": np.zeros((1002, 30))}
    matched = timing.matched_interval(event)
    assert matched["steps"] == 800
    assert matched["post100_start_s"] == pytest.approx(.1)
    assert matched["end_s"] == pytest.approx(8.1)

    short = timing.matched_interval({"x": np.zeros((50, 30))})
    assert short["steps"] == 39
    assert short["duration_s"] == pytest.approx(.39)
    assert short["end_s"] == pytest.approx(.49)


def test_interpolation_requires_actual_endpoint_and_allows_one_step_overshoot():
    times = np.array([0., .005, .015])
    values = np.column_stack((times, 2 * times))
    aligned, boundary = timing.interpolate_solver(times, values, np.array([.005, .01]), .005)
    np.testing.assert_allclose(aligned, [[.005, .01], [.01, .02]])
    assert boundary["within_one_step"]
    with pytest.raises(ValueError, match="coverage"):
        timing.interpolate_solver(times, values, np.array([.005, .016]), .005)
    with pytest.raises(ValueError, match="coverage"):
        timing.interpolate_solver(times, values, np.array([-.001, .005]), .005)
    with pytest.raises(ValueError, match="overshoots"):
        timing.interpolate_solver(times, values, np.array([.005, .009]), .001)


def test_coi_error_can_hide_common_frequency_but_explicit_frequency_metric_catches_it():
    reference = np.zeros((3, 20))
    prediction = reference.copy()
    prediction[:, :10] += 0.7
    prediction[:, 10:20] += 2.0
    metrics = timing.normalized_coi_rmse(prediction, reference, np.ones(10))
    assert metrics["delta"] == pytest.approx(0.)
    assert metrics["omega"] == pytest.approx(0.)
    assert timing.common_frequency_rmse(prediction, reference, np.ones(10)) == pytest.approx(2.)


def test_los_rule_requires_five_consecutive_exact_samples():
    config = {"los_angle_rad": 1., "los_relative_speed_pu": .02, "los_hold_samples": 5}
    inertia = np.ones(3)
    delta = np.zeros((7, 3))
    omega = np.ones((7, 3))
    delta[2:, 0] = 2.
    omega[2:, 0] = 1.1
    assert timing.los_index(delta[:4], omega[:4], inertia, config) is None
    assert timing.los_index(delta, omega, inertia, config) == 6


def test_event_selection_is_stable_and_uses_only_explicit_train_ids(tmp_path):
    roots = []
    for index, events in enumerate((
            [("zstable", "stable"), ("bunstable", "unstable")],
            [("astable", "stable"), ("cunstable", "unstable")])):
        root = tmp_path / f"pool{index}"
        root.mkdir()
        (root / "splits.json").write_text(json.dumps({
            "train": [event_id for event_id, _ in events], "validation": [], "test": ["never-read"]
        }), encoding="utf-8")
        with (root / "dataset_index.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["event_id", "label", "trajectory_end_s",
                                                        "solver_success", "valid_length"])
            writer.writeheader()
            for event_id, label in events:
                writer.writerow({"event_id": event_id, "label": label, "trajectory_end_s": "1.0",
                                 "solver_success": "True", "valid_length": "100"})
        roots.append(root)
    selected = timing.select_events(tuple(reversed(roots)))
    assert selected["stable"]["event_id"] == "astable"
    assert selected["unstable"]["event_id"] == "bunstable"


def test_rom_timing_performs_one_encode_and_autonomous_steps():
    class Dummy(torch.nn.Module):
        def encode(self, history):
            assert tuple(history.shape) == (1, 11, 30)
            return torch.ones((1, 1), dtype=history.dtype)

        def rk4(self, state, dt):
            return state + dt

    values, record = timing.rom_rollout_timed(Dummy(), np.zeros((11, 30)), 4, .005)
    assert values.shape == (5, 31)
    assert record["gru_seconds"] >= 0.
    assert record["integration_seconds"] >= 0.
    assert record["end_to_end_seconds"] == pytest.approx(record["gru_seconds"] + record["integration_seconds"])


def test_median_summary_is_not_minimum():
    rows = [{"seconds": 4.}, {"seconds": 1.}, {"seconds": 3.}]
    assert timing.median_repetitions(rows, ("seconds",))["seconds"] == 3.
