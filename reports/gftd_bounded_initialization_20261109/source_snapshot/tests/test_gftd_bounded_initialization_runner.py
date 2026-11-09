"""Exercise receipt and trajectory handling using preset arrays and mocked solvers."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.gftd_bounded_initialization_20261109 import run


def mock_initializer(monkeypatch, outcome):
    simple, smooth = np.arange(50, dtype=np.float64), np.arange(50, dtype=np.float64)
    smooth[-1] += 1.
    functional = object()
    received = []

    def mock_integrate(f, state, step, duration, counts, *, backward, dense, stepper):
        assert f is functional
        assert stepper is run.fitting.dop853_step
        steps = round(duration / step)
        counts["integrations_attempted"] += 1
        counts["integrations_completed"] += 1
        counts["fixed_dp8_steps_attempted"] += steps
        counts["fixed_dp8_steps_completed"] += steps
        counts["actual_rhs_calls"] += 12 * steps
        values = np.tile(state, (steps + 1 if dense else round(duration / .01) + 1, 1))
        received.append((duration, step, backward, dense, values))
        return values

    monkeypatch.setattr(run.fitting, "_integrate", mock_integrate)

    class Model:
        def initialize(self, packet):
            assert set(packet) == {"p0"}
            counter = lambda: run.fitting._counts(run.fitting.dop853_step)

            def integrate(state, step, duration, counts, *, backward=False, dense=False):
                return run.fitting._integrate(functional, state, step, duration, counts, backward=backward,
                                              dense=dense, stepper=run.fitting.dop853_step)

            def stage(state):
                counts = counter()
                counts["actual_residual_calls"] = 1
                integrate(state, .0025, .1, counts, backward=True, dense=True)
                return dict(best=dict(state=state.tolist(), history_mse=1.), status="converged", reason="mock",
                            actual_residual_calls=1, same_step_feasible_sse=None, projected_gradient_kkt=None,
                            observation_projected_gradient_kkt=None, active_bounds=[], counts=counts)

            def gates(state, passed):
                counts = counter()
                for step in (.0025, .00125):
                    integrate(state, step, .1, counts, backward=True)
                    integrate(state, step, .1, counts)
                for step in (.01, .005):
                    integrate(state, step, 1., counts)
                return dict(passed=passed, p_endpoint_error=0., counts=counts)

            stage1, stage1_gates = stage(simple), gates(simple, True)
            stage2 = stage(smooth)
            same = outcome == "reuse"
            state = simple if same else smooth
            final_gates = stage1_gates if same else gates(smooth, outcome == "smooth")
            counts = counter()
            for part in (stage1, stage1_gates, stage2, *(() if same else (final_gates,))):
                run.fitting._add(counts, part["counts"])
            self.last_initialization = dict(
                method="fixed_dp8", counts=counts, stage1=stage1, stage2=stage2, stage1_numerical=stage1_gates,
                final_numerical=final_gates, returned_point="stage1" if same else "stage2",
                final_check_reused_stage1=same, status="numerically_unresolved" if outcome == "failure" else "completed")
            if outcome == "failure":
                raise run.fitting.InitializationFailure("mock final gates", self.last_initialization)
            return state

    model = Model()
    model.functional = functional
    return model, {"p0": simple[20:30]}, mock_integrate, received


@pytest.mark.parametrize("outcome,gate_count", [("reuse", 1), ("smooth", 2), ("failure", 2)])
def test_capture_gate_arrays_from_the_original_call(monkeypatch, outcome, gate_count):
    model, packet, original, received = mock_initializer(monkeypatch, outcome)
    record, arrays, diagnostic = {}, {}, {}
    state, paths = run.initialize(model, packet, record, arrays, diagnostic)
    assert run.fitting._integrate is original
    assert len(received) == 2 + 6 * gate_count
    assert len(record["dp8_capture_endpoints"]) == 2 * gate_count
    assert record["initialization_counts"]["actual_residual_calls"] == 2
    assert record["initialization_counts"]["actual_rhs_calls"] == 960 + 6480 * gate_count
    for captured, source in zip(arrays.values(), [v[-1] for v in received if v[0] == 1.]):
        assert captured.tobytes() == source.tobytes()
        assert captured is not source
    if outcome == "failure":
        assert state is paths is record["formal_point"] is None
        assert record["stage1_gates"]["passed"] is True and record["six_gates_passed"] is False
    else:
        assert record["formal_point"]["p0_exact"]
        for method, path in paths.items():
            assert path is arrays[f"dp8_gate{gate_count}_{method}"]
            assert path[0].tobytes() == state.tobytes()


class MockRadau:
    def __init__(self, rhs, t0, state, end, *, rtol, atol, max_step):
        assert (t0, end, rtol, atol, max_step) == (0., 1., 1e-11, 1e-13, .005)
        self.rhs, self.t, self.state, self.y = rhs, t0, state, state
        self.status = "running"
        for _ in range(2):
            rhs(t0, state)

    def step(self):
        for _ in range(2):
            self.rhs(self.t, self.state)
        self.t = run.GRID[35] if self.t == 0. else 1.
        if self.t == 1.:
            self.status = "finished"

    def dense_output(self):
        self.rhs(self.t, self.state)
        return lambda _: self.state.copy()


@pytest.mark.parametrize("limit,by_operation,expected_rows", [
    (1, (1, 0, 0), 1),
    (4, (2, 2, 0), 1),
    (5, (2, 2, 1), 36),
    (8, (2, 4, 2), 101),
])
def test_radau_actual_callbacks_and_continuous_prefix(monkeypatch, limit, by_operation, expected_rows):
    monkeypatch.setattr(run, "Radau", MockRadau)
    rhs_calls = []

    def rhs(state):
        rhs_calls.append(state.shape)
        return np.zeros_like(state)

    state = np.zeros(50)
    record = {}
    path = run.radau_reference(SimpleNamespace(rhs=rhs), state, record, rhs_limit=limit)
    assert record["actual_rhs_calls"] == len(rhs_calls) == limit
    assert tuple(record["rhs_calls_by_operation"].values()) == by_operation
    assert record["output_rows"] == expected_rows
    assert record["status"] == ("completed" if limit == 8 else "rhs_limit")
    reference = run.reference_comparison(np.ones(10), np.zeros((101, 50)), path, record["status"] == "completed")
    assert reference["common_rows"] == expected_rows
    assert reference["common_through_s"] == run.GRID[expected_rows - 1]
    assert reference["observation_rms"] == reference["observation_peak"] == reference["full_state_max_relative"] == 0.
    assert reference["confirmed"] is (limit == 8)


def test_reference_checks_unobserved_state_separately():
    fine, main = np.zeros((101, 50)), np.zeros((101, 50))
    main[-1, -1] = 2e-5
    assert run.reference_comparison(np.ones(10), fine, fine, True)["confirmed"]
    main_result = run.reference_comparison(np.ones(10), main, fine, True)
    assert main_result["observation_rms"] == 0.
    assert main_result["full_state_max_relative"] > 1e-5
    assert not main_result["confirmed"]


def test_archive_failure_keeps_completed_work_and_blocks_replay(monkeypatch, tmp_path):
    model, packet, _, received = mock_initializer(monkeypatch, "reuse")
    model.functional = SimpleNamespace(mass=np.ones(10), rhs=np.zeros_like)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "MODEL_IN_WORKER", model)

    def initialize(model, packet, record, arrays, diagnostic):
        received.append("initialize")
        state = np.zeros(50)
        arrays["formal_state"] = state
        record["initialization_counts"] = dict(actual_residual_calls=3, actual_rhs_calls=50)
        path = np.zeros((101, 50))
        return state, {"main": path, "fine": path}

    def fail_radau_archive(path, arrays):
        if path.name == "radau.npz":
            raise OSError("mock archive write failed")
        return {"mock_archive": True}

    monkeypatch.setattr(run, "initialize", initialize)
    monkeypatch.setattr(run, "Radau", MockRadau)
    monkeypatch.setattr(run, "write_arrays", fail_radau_archive)
    row = dict(index=0, event_id="mock", label="stable")
    action = dict(run=run.RUN, phase="pilot", action_id="mock", release_sha256="release")
    prepared = dict(model={"sha256": "model"}, config_sha256="config")
    with pytest.raises(OSError, match="mock archive"):
        run.run_job(row, packet, "packet", action, prepared)
    job, _ = run.job_paths("pilot", 0)
    persisted = json.loads((tmp_path / job).read_bytes())
    assert persisted["status"] == "interrupted" and persisted["reference_arrays"] is None
    assert persisted["reference"]["actual_rhs_calls"] == 8
    assert persisted["initialization_counts"]["actual_rhs_calls"] == 50
    with pytest.raises(FileExistsError):
        run.run_job(row, packet, "packet", action, prepared)
    assert received == ["initialize"]


def test_full_reuses_pilot_result_and_artifact_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "PILOT", (0,))
    rows = [dict(index=0, event_id="mock", label="stable")]
    packets = [(None, "input")]
    action = dict(run=run.RUN, phase="pilot", action_id="mock-pilot", release_sha256="release")
    job, private = run.job_paths("pilot", 0)
    claim = run.save_json(job.parent.parent / "claims" / job.name, action, exclusive=True)
    diagnostics = run.save(private / "diagnostics.json.gz", b"original diagnostics")
    dp8 = run.save(private / "initialization.npz", b"original DP8")
    radau = run.save(private / "radau.npz", b"original Radau")
    result = dict(action, index=0, event_id="mock", claim=claim, input_sha256="input", status="completed",
                  formal_point={"p0_exact": True}, diagnostics=diagnostics,
                  initialization_arrays=dp8, reference_arrays=radau)
    ref = run.save_json(job, result)
    original = (tmp_path / job).read_bytes()
    bindings = dict(prepare_sha256="prepare", qualification_sha256="qualification")
    pilot = dict(action=action, bindings=bindings, status="completed", eligible_for_full=True,
                 jobs={"0": ref}, unknown_jobs=[], partial_jobs={})
    receipt = run.save_json(run.RESULTS / "pilot/execution.json", pilot)
    release = dict(action_id="mock-full", pilot_execution_sha256=receipt["sha256"])
    eligible = run.eligible_pilot(release, bindings)
    assert run.reuse_pilot(eligible, rows, packets) == {0: (ref, result)}
    assert (tmp_path / job).read_bytes() == original
    (tmp_path / private / "radau.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="frozen bytes changed"):
        run.reuse_pilot(eligible, rows, packets)
