"""Seconds-only synthetic/math qualification; never accesses physical events."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from kiki.data.osgkd_synthetic import (
    make_system, initial_state, rhs, forces, generate_event, motion_packet,
    recovery_metrics, support_metrics,
)


@pytest.mark.parametrize("n", [8, 16, 32])
def test_graph_and_two_event_mesh(n):
    system = make_system(n, seed=31)
    assert len(system["edges"]) == 3*n//2
    assert len(system["candidate_edges"]) == n*(n-1)//2
    reached = {0}
    for _ in range(n):
        for i, j in system["edges"]:
            if i in reached or j in reached:
                reached.update((i, j))
    assert len(reached) == n
    for seed, excitation in [(12, "distributed"), (13, "local")]:
        event, truth = generate_event(system, seed, excitation)
        _, fine = generate_event(system, seed, excitation, internal_dt=.00125)
        assert np.max(np.abs(truth["state"]-fine["state"])) < 1e-9
        assert event["history"]["theta"].shape == (11, n)
        assert np.allclose(np.diff(event["history"]["t"]), .01)
        internal, load = forces(system, truth["initial_state"])
        assert abs(internal.sum()) < 1e-12
        assert abs(load.sum()) > 1e-3
    assert np.array_equal(rhs(system, np.zeros((5, n))), np.zeros((5, n)))


def test_noise_truth_and_input_separation():
    system = make_system()
    event, truth = generate_event(system, seed=7)
    noisy, noisy_truth = generate_event(system, seed=7, noise_std=.01)
    assert np.array_equal(truth["state"], noisy_truth["state"])
    assert not np.array_equal(event["history"]["theta"], noisy["history"]["theta"])
    packet = motion_packet(noisy)
    assert set(packet) == {"history", "constants"}
    assert set(packet["constants"]) == {"candidate_edges", "mass", "damping",
                                       "reference", "scales", "angle_gauge"}
    packet["history"]["theta"][:] = 100
    assert not np.array_equal(packet["history"]["theta"], noisy["history"]["theta"])


def test_conditions_controls_and_scoring():
    for condition in ("static", "modulated", "open"):
        system = make_system(condition=condition)
        state = initial_state(system, excitation="equilibrium")
        assert not np.any(rhs(system, state))
        assert np.any(system["beta"]) == (condition != "static")
        assert np.any(system["environment_gain"]) == (condition == "open")
        mod = system["edges"][system["beta"] != 0]
        metrics = recovery_metrics(system["edges"][:, ::-1], mod, system)
        assert metrics["edges"]["f1"] == 1
        assert metrics["beta"]["f1"] == 1
        assert metrics["modulation_accuracy"] == 1
    assert support_metrics([(1, 0), (0, 1)], [(0, 1)])["f1"] == 1
    with pytest.raises(ValueError):
        recovery_metrics([], [(0, 1)], system)
    for mode, block in (("mechanical", 2), ("amplitude", 3), ("environment", 4)):
        state = initial_state(system, seed=2, excitation=mode)
        assert np.all(state[block] > 0)
        assert np.count_nonzero(state) == system["n"]
    # Hidden modulation cannot be identified without relative-angle excitation.
    amplitude_only = initial_state(system, excitation="amplitude")
    assert np.array_equal(rhs(system, amplitude_only)[:3], np.zeros((3, 8)))


def test_historical_escape_example():
    path = Path(__file__).resolve().parents[1]/"experiments/osgkd_v2_layer1/escape_example.py"
    spec = importlib.util.spec_from_file_location("escape_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run_checks()["status"] == "passed"
