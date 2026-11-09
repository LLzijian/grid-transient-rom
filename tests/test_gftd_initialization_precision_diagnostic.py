from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import DOP853, Radau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.gftd_initialization_precision_20261109 import run as diagnostic


class Linear:
    mass = np.ones(2)

    def split_state(self, state):
        if np.shape(state) != (10,) or not np.isfinite(state).all():
            raise ValueError("invalid state")
        return np.split(state, 5)

    def rhs(self, state):
        return -2 * state


def test_selects_uniform_positions_and_retains_every_exception():
    rows = [dict(mode="static", group="future_rms_peak", index=i) for i in range(11)]
    rows += [dict(mode="modulated", group="exp", index=i) for i in range(14)]
    rows += [dict(mode="modulated", group="stage2", index=i) for i in range(14, 19)]
    result = diagnostic.select_states(rows)
    assert result[:4] == [("static", i, "selected") for i in (0, 2, 5, 8)]
    assert result[4:18] == [("modulated", i, "selected") for i in range(14)]
    assert result[18:] == [("modulated", i, stage) for i in range(14, 19) for stage in ("selected", "stage1")]


def test_frozen_overflow_phase_uses_per_check_counts():
    assert diagnostic.failed_integration(dict(integrations_attempted=5, integrations_completed=4,
                                               rk4_steps_attempted=298, rk4_steps_completed=297)) == {
                                                   "leg": "future_main", "step": 58,
                                               }


def test_rk4_negative_time_and_midstage_failure_preserve_cost():
    state = np.arange(1., 11.)
    f = Linear()
    complete = diagnostic.integrate(f, state, -.1, step=.0025)
    assert complete.record["actual_rhs_calls"] == 160
    np.testing.assert_allclose(complete.path[-1], state * np.exp(.2), rtol=1e-11)

    def rhs(y):
        if y[0] < .99:
            raise FloatingPointError("forced intermediate failure")
        return -2 * y

    f.rhs = rhs
    failed = diagnostic.integrate(f, state, .1, step=.01)
    assert not failed.record["completed"]
    assert failed.record["actual_rhs_calls"] == 4
    assert failed.record["failure"]["step"] == 1
    assert failed.record["failure"]["substage"] == 4
    assert failed.record["failure"]["time"] == .01
    np.testing.assert_array_equal(failed.last_state, state)
    assert np.isnan(failed.path[1:]).all()


def test_adaptive_includes_hidden_radau_fd_calls_and_honors_limit():
    state = np.ones(10)
    complete = diagnostic.integrate(Linear(), state, .1, solver=Radau, rtol=1e-9,
                                    atol=1e-11, max_step=.005, rhs_limit=20000)
    assert complete.record["completed"]
    assert complete.record["actual_rhs_calls"] > complete.record["solver_nfev"]
    np.testing.assert_allclose(complete.path[-1], np.exp(-.2), rtol=1e-10)
    failed = diagnostic.integrate(Linear(), state, .1, solver=DOP853, rtol=1e-11,
                                  atol=1e-13, max_step=.005, rhs_limit=8)
    assert failed.record["actual_rhs_calls"] == 8
    assert not failed.record["completed"]
    assert "RhsLimit" in failed.record["error"]


def test_agreement_keeps_unobserved_difference_and_rejects_partial_solution():
    f = Linear()
    a = np.zeros((11, 10))
    b = a.copy()
    b[1:, 6] = 1.
    one = diagnostic.Integration(a, a[-1], None, {"completed": True})
    two = diagnostic.Integration(b, b[-1], None, {"completed": True})
    three = diagnostic.Integration(a, a[-1], None, {"completed": False})
    difference = diagnostic.path_delta(f, one, two)
    assert difference["observed"]["peak"] == 0.
    assert difference["state"]["peak"] == 10.
    assert difference["components"]["a"]["peak"] == 1.
    assert not diagnostic.reference_agreement(f, dict(a=one, b=two, c=three), True)["qualified"]


def test_missing_reverse_integral_cannot_pass_gates():
    path = np.zeros((11, 10))
    failed = diagnostic.Integration(path, path[-1], None, {"completed": False})
    record = diagnostic.rk4_gates(Linear(), path[-1], [failed] * 2, [None] * 2, [failed] * 2)
    assert not record["passed"]
    assert record["values"]["roundtrip_main_scaled_max"] is None


def test_claim_does_not_replay_or_replace_existing_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    diagnostic.claim("prepare", "first", "sha")
    diagnostic.claim("diagnostic", "second", "sha")
    with pytest.raises(FileExistsError):
        diagnostic.claim("diagnostic", "repeat", "other_sha")
    assert b"second" in (tmp_path / diagnostic.OUTPUT / "started.json").read_bytes()


def test_second_preparation_preserves_original_failure_and_freezes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "recover", lambda release: ({}, [], {}))
    diagnostic.claim("prepare", "first", "sha")
    previous_path = diagnostic.OUTPUT / "prepare_execution.json"
    previous = dict(phase="prepare", status="INTERRUPTED_RUNTIME", action_id="first", completed=[])
    previous_hash = diagnostic.write_json(previous_path, previous)["sha256"]
    release = dict(released=True, phase="prepare02", action_id="second", previous_prepare_sha256=previous_hash)
    for key, paths in (("source_sha256", diagnostic.SOURCE_PATHS), ("asset_sha256", diagnostic.PARENT_RECORDS)):
        release[key] = {Path(p).as_posix(): diagnostic.write_bytes(Path(p), b"fixture")["sha256"] for p in paths}
    release["protocol_sha256"] = diagnostic.write_bytes(diagnostic.PROTOCOL, b"fixture")["sha256"]
    path = tmp_path / "release.json"
    path.write_bytes(diagnostic.json_bytes(release))
    result = diagnostic.run(path, "prepare")
    assert result["status"] == "COMPLETED"
    assert result["phase"] == "prepare02"
    assert (tmp_path / diagnostic.OUTPUT / "prepare02_started.json").exists()
    assert (tmp_path / diagnostic.OUTPUT / "prepare02_execution.json").exists()
    assert diagnostic.read_verified(previous_path, previous_hash) == diagnostic.json_bytes(previous)
    with pytest.raises(ValueError, match="no frozen selection"):
        diagnostic.run(path, "prepare")
