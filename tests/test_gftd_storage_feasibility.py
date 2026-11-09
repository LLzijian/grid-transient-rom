"""Engineering checks use mock callbacks only, outside the scientific RHS ledger."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.gftd_storage_feasibility_20261109 import run


class MockField:
    def __init__(self):
        self.callbacks = 0

    def split_state(self, value):
        if not np.isfinite(value).all():
            raise ValueError("nonfinite mock state")

    def rhs(self, value):
        self.callbacks += 1
        return np.zeros_like(value)


class MockSolver:
    nfev, njev, nlu = 0, 0, 0

    def __init__(self, callback, t0, state, end, *, rtol, atol, max_step):
        assert (t0, end, rtol, atol, max_step) == (0., 1., 1e-11, 1e-13, .005)
        self.callback, self.t, self.y = callback, t0, state
        self.steps, self.status = 0, "running"
        for _ in range(2):
            callback(t0, state)

    def step(self):
        self.steps += 1
        self.t = self.steps / 200.
        self.y = np.full_like(self.y, self.t)
        self.callback(self.t, self.y)
        if self.steps == 200:
            self.status = "finished"

    def dense_output(self):
        for _ in range(3):
            self.callback(self.t, self.y)
        return lambda t: np.full_like(self.y, t)


def test_mock_complete_at_exact_cap_and_separate_costs(monkeypatch, capsys):
    monkeypatch.setitem(run.REFERENCES, "dop853_adaptive", (MockSolver, 1e-11, 1e-13))
    field = MockField()
    record, arrays = run.integrate_reference(field, np.zeros(10), "dop853_adaptive", rhs_limit=502)
    assert record["completed"] and record["rhs_version"] == run.VERSION
    assert record["actual_rhs_calls"] == field.callbacks == 502
    assert record["rhs_calls_by_operation"] == {"solver_setup": 2, "solver_step": 200, "dense_output": 300}
    assert record["last_accepted_time"] == record["last_output_time"] == 1.
    assert len(arrays["accepted_times"]) == 201
    np.testing.assert_array_equal(arrays["grid_states"][:, 0], run.GRID)
    print(f"mock callbacks: {field.callbacks}, real model RHS: 0")
    assert "mock callbacks: 502" in capsys.readouterr().out


def test_mock_dense_cap_preserves_accepted_mesh_and_missing_grid(monkeypatch):
    monkeypatch.setitem(run.REFERENCES, "dop853_adaptive", (MockSolver, 1e-11, 1e-13))
    field = MockField()
    record, arrays = run.integrate_reference(field, np.zeros(10), "dop853_adaptive", rhs_limit=5)
    assert record["status"] == "rhs_limit" and record["actual_rhs_calls"] == field.callbacks == 5
    assert record["rhs_calls_by_operation"] == {"solver_setup": 2, "solver_step": 2, "dense_output": 1}
    assert record["failure"] == {"operation": "dense_output", "time": .01, "evaluated": False, "step": 2}
    assert record["last_accepted_time"] == .01 and record["last_output_time"] == 0.
    np.testing.assert_array_equal(arrays["accepted_times"], [0., .005, .01])
    np.testing.assert_array_equal(arrays["failure_state"], np.full(10, .01))
    assert np.isnan(arrays["grid_states"][1:]).all()


def test_mock_solver_failure_saves_attempt_without_marking_it_accepted(monkeypatch):
    class FailedStep(MockSolver):
        def step(self):
            self.callback(.002, np.full_like(self.y, .5))
            self.status = "failed"
            return "mock solver failed"

    monkeypatch.setitem(run.REFERENCES, "dop853_adaptive", (FailedStep, 1e-11, 1e-13))
    field = MockField()
    record, arrays = run.integrate_reference(field, np.zeros(10), "dop853_adaptive")
    assert record["status"] == "solver_failure" and record["actual_rhs_calls"] == field.callbacks == 3
    assert record["failure"]["time"] is None and record["last_accepted_time"] == 0.
    assert record["last_attempted_callback"] == {"operation": "solver_step", "time": .002, "evaluated": True, "step": 1}
    np.testing.assert_array_equal(arrays["last_attempted_state"], np.full(10, .5))
    np.testing.assert_array_equal(arrays["accepted_times"], [0.])
    assert "failure_state" not in arrays


def test_hidden_disagreement_cuts_reference_prefix_permanently():
    base = np.zeros((len(run.GRID), 10))
    bad = base.copy()
    bad[8, 6] = .01
    truncated = base.copy()
    truncated[20:] = np.nan
    assert run.reference_prefix(np.ones(2), {"a": base, "b": bad, "c": truncated}) == (20, 8)


def test_matched_ranges_stop_at_shorter_kernel_prefix(monkeypatch):
    base = np.zeros((len(run.GRID), 10))
    truncated = base.copy()
    truncated[93:] = np.nan
    states = {stage: {method: base for method in run.REFERENCES} for stage in run.STATE_HASHES}
    original = dict(states, stage2={method: truncated for method in run.REFERENCES})
    completion = dict(completed=True, status="completed", last_accepted_time=1., last_output_time=1., error=None)
    jobs = {stage: {method: completion for method in run.REFERENCES} for stage in run.STATE_HASHES}
    growth = {"stage1": None, "stage2": {"window": {"start": .71, "end": .91}}}
    monkeypatch.setattr(run, "path_ranges", lambda f, path, bounded, counts: len(path))
    field = type("Field", (), {"mass": np.ones(2)})()
    summary = run.summarize(field, jobs, original, jobs, states, growth)
    assert summary["stages"]["stage1"]["T_pair"] == 1.
    stage2 = summary["stages"]["stage2"]
    assert stage2["T_pair"] == .92
    assert stage2["within_model"]["candidate"]["qualified_full_second"]
    assert stage2["inherited_original_growth"] == growth["stage2"]
    assert stage2["matched_window"] == {version: dict.fromkeys(run.REFERENCES, 93)
                                         for version in ("original", "candidate")}


def test_unregistered_version_rejected_before_access_to_bound_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    release = tmp_path / "release.json"
    release.write_text(json.dumps({
        "version": "original_strict", "action_id": "fake", "prepare_sha256": "",
        "qualification_sha256": "", "adoption": {}, "implementation_review": {},
    }))
    with pytest.raises(ValueError, match="candidate version"):
        run.release_inputs(release)


def test_witness_identities_match_public_frozen_receipts():
    execution = json.loads(run.read_bound(*run.OLD_EXECUTION))
    for case, (index, digest) in run.WITNESSES.items():
        summary = json.loads(run.read_ref(execution["summaries"][case], run.OLD_RESULTS / "summaries" / f"{case}.json"))
        assert summary["reference"]["agreement_end"] == index / 100.
        assert summary["arrays"]["sha256"] == digest


def test_invalid_amplitude_shape_and_nonfinite_constant_rejected():
    constants = dict(mass=[1., 1.], damping=[.2, .2], k_edge=[1.], eta_edge=[0.], gamma_amp=[.5, .5],
                     lambda_amp=[1., 1.], lambda_mem=[1., 1.], tau_amp=[1., 1.], tau_mem_sync=[1., 1.],
                     j_omega_mem=[1., 1.], gates=run.StructuralGates.overcomplete(1, 2))
    with pytest.raises(ValueError, match="tau_mem_coi must be finite"):
        run.BoundedAmplitudeKuramoto(**constants, tau_mem_coi=np.nan)
    field = run.BoundedAmplitudeKuramoto(**constants, tau_mem_coi=1.)
    with pytest.raises(ValueError, match="node-aligned amplitude"):
        field._kernel(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="node-aligned amplitude"):
        field._kernel(np.array([np.nan, 0.]))


def test_claim_is_exclusive_and_cached_input_cannot_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    claim = tmp_path / "claim.json"
    run.save(claim, b"first", exclusive=True)
    with pytest.raises(FileExistsError):
        run.save(claim, b"second", exclusive=True)
    assert claim.read_bytes() == b"first"
    with pytest.raises(ValueError, match="escapes"):
        run.read_bound(tmp_path / ".." / "forbidden", "")


@pytest.mark.parametrize(("phase", "claimed", "returned", "total"), (
    ("solver", 1, 0, None),
    ("array", 1, 1, 17),
    ("summary", 6, 6, 102),
))
def test_interrupted_execution_preserves_job_identity_and_known_costs(tmp_path, monkeypatch, phase, claimed, returned, total):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    states = dict.fromkeys(run.STATE_HASHES, np.zeros(10))
    release = dict(action_id="mock", prepare_sha256="prepare", qualification_sha256="qualification")
    prepared = dict(source_sha256={}, runtime={})
    qualification = dict(counts={"rhs": 0})
    parent = object()
    monkeypatch.setattr(run, "release_inputs", lambda path: (
        release, "mock-release", prepared, qualification, parent, states, {}, {}, {}))
    monkeypatch.setattr(run.BoundedAmplitudeKuramoto, "from_functional", lambda field: field)

    def fail(*args, **kwargs):
        raise OSError(f"mock {phase} interruption")

    monkeypatch.setattr(run, "integrate_reference", fail if phase == "solver" else lambda *args: (
        dict(actual_rhs_calls=17, cpu_seconds=.01, wall_seconds=.02, status="completed"),
        dict(grid_states=np.zeros((len(run.GRID), 10)))))
    if phase == "array":
        monkeypatch.setattr(run.np, "savez_compressed", fail)
    if phase == "summary":
        monkeypatch.setattr(run, "summarize", fail)

    with pytest.raises(OSError, match=f"mock {phase} interruption"):
        run.execute(Path("mock.json"))
    receipt = json.loads((tmp_path / run.OUTPUT / "execution.json").read_bytes())
    assert receipt["status"] == "INTERRUPTED_RUNTIME" and receipt["summary"] is None
    assert receipt["claimed_job_count"] == claimed and receipt["returned_integration_jobs"] == returned
    assert receipt["known_integration_rhs_calls"] == 17 * returned
    assert receipt["actual_integration_rhs_calls"] == total
    assert len(receipt["unknown_jobs"]) == (phase == "solver")
    assert len(receipt["jobs_without_arrays"]) == (phase == "array")
    for stage, refs in receipt["jobs"].items():
        for method, ref in refs.items():
            job = json.loads(run.read_ref(ref, run.OUTPUT / "jobs" / stage / f"{method}.json"))
            assert job["actual_rhs_calls"] == 17
            assert (job["arrays"] is None) == (phase == "array")
