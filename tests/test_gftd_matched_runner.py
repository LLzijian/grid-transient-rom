"""Check the registered runner boundaries with mock calls and preset arrays."""
from collections import Counter
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.gftd_matched_prediction_20261109 import run
from kiki.models import gftd_fit297


@pytest.mark.parametrize("mode,kernel,version", [
    ("static", "exp_v1", "gftd_kuramoto_5n_v2"),
    ("modulated", "bounded_exp_tanh_v1", "gftd_kuramoto_5n_v3"),
])
def test_public_fit_dispatch_and_observation_whitelist(monkeypatch, tmp_path, mode, kernel, version):
    model = run.GFTDKuramotoModel(n_nodes=3, fit_mode=mode, modulation_kernel=kernel)
    packet = dict(theta_history=np.zeros((11, 3)), omega_history=np.zeros((11, 3)),
                  history_times=np.linspace(-.1, 0., 11), p0=np.ones(3), sigma=0.)
    req = {"fixture": 1}
    parent = model.functional
    scale_request = dict(format=gftd_fit297.FIT_FORMAT, protocol_sha256=gftd_fit297.PROTOCOL_SHA256,
                         adaptation_indices=[0, 1], input_scale_delta=[.2, .3, .4],
                         input_scale_omega=[.9, 1.1, 1.3], rstar=[.23, .33, .53], k_ref=1.,
                         source_sha256={"fixture": "a" * 64})
    expected_inner = gftd_fit297.make_config(parent, mode, scale_request, 2, "fixed-fit-dispatch-fixture")
    expected_config = run.fitting.make_fixed_dp8_config(parent, expected_inner)
    fit_state = dict(format=run.FIT_FORMAT, kernel=kernel, strategy=dict(proxy_start_calls=1200))
    calls = []

    def fit_mock(f, packets, *, mode, config):
        calls.append(config)
        assert f is not parent and f.modulation_kernel == kernel and mode == model.fit_mode
        assert len(packets) == 1
        assert set(packets[0]) == {"theta_history", "omega_history", "history_times", "p_end", "sigma"}
        assert np.array_equal(packets[0]["p_end"], packet["p0"])
        return dict(functional=deepcopy(f), config=expected_config, fit_state=fit_state,
                    private_details={}, status="completed")

    monkeypatch.setattr(run.matched, "fit_shared", fit_mock)
    context = run.RunContext(gftd_matched_prediction_v1=req)
    with pytest.raises(ValueError):
        model.fit([packet], [], run.RunContext(gftd_fit297_v1={}, gftd_matched_prediction_v1=req))
    with pytest.raises(ValueError):
        model.fit([dict(packet, future=np.zeros(3))], [], context)
    with pytest.raises(ValueError):
        model.fit([packet], [packet], context)
    assert calls == []
    result = model.fit([packet], [], context)
    assert result.status == "completed" and calls == [req]
    path = tmp_path / "model.json"
    model.save(path)
    raw = path.read_bytes()
    payload = json.loads(raw)
    assert payload["format"] == version
    assert payload["functional"].get("modulation_kernel") == (None if mode == "static" else kernel)
    restored = run.GFTDKuramotoModel.load(path)
    assert restored.functional.modulation_kernel == kernel and restored._uses_fixed_dp8
    assert restored.inference_config == expected_config and restored.fit_state == fit_state
    assert restored.inference_config["fit297_config"]["proxy_start_calls"] == 200
    assert restored.fit_state["strategy"]["proxy_start_calls"] == 1200
    assert (run.fitting._hash(run.fitting._physics(restored.functional))
            == run.fitting._hash(run.fitting._physics(parent)) == expected_inner["model_sha256"])
    copy_path = tmp_path / "restored.json"
    restored.save(copy_path)
    assert run.sha(copy_path.read_bytes()) == run.sha(raw)


def test_qualification_guard_counts_without_profile_and_latches_loss():
    from experiments.gftd_matched_prediction_20261109.qualify import QualificationMeter

    active = True
    meter = QualificationMeter(Counter(rhs=127), lambda: active)
    calls = []

    def mock_call(name):
        calls.append(name)
        meter.observe_model(name)

    rhs = meter.guard(mock_call, "rhs")
    energy = meter.guard(mock_call, "hamiltonian")
    forbidden = meter.guard(mock_call, "solver", blocked=True)
    rhs("rhs")
    assert meter.counts == {"rhs": 1}
    with pytest.raises(RuntimeError, match="budget exhausted"):
        rhs("rhs")
    with pytest.raises(RuntimeError, match="blocks"):
        forbidden("solver")
    energy("hamiltonian")
    active = False
    with pytest.raises(RuntimeError, match="meter unavailable"):
        energy("hamiltonian")
    active = True
    with pytest.raises(RuntimeError, match="meter unavailable"):
        energy("hamiltonian")
    assert meter.cost_unknown and calls == ["rhs", "hamiltonian"]
    assert meter.counts == meter.observed == {"rhs": 1, "hamiltonian": 1}

    unguarded = QualificationMeter(Counter(), lambda: True)
    unguarded.observe_model("rhs")
    with pytest.raises(RuntimeError, match="meter unavailable"):
        unguarded.guard(mock_call, "hamiltonian")("hamiltonian")
    assert unguarded.cost_unknown and not unguarded.counts


def test_deployment_captures_all_six_original_legs(monkeypatch):
    state = np.arange(50, dtype=np.float64)
    field = object()
    packet = dict(theta_history=np.tile(state[:10], (11, 1)), omega_history=np.tile(state[10:20], (11, 1)),
                  p0=state[20:30])
    actual = []

    def integrate(f, endpoint, step, duration, counts, *, backward=False, dense=False, stepper=None):
        assert f is field and stepper is run.fitting.dop853_step and not dense
        steps = round(duration / step)
        counts["actual_rhs_calls"] += 12 * steps
        for key in ("fixed_dp8_steps_attempted", "fixed_dp8_steps_completed"):
            counts[key] += steps
        output = np.tile(endpoint, (round(duration / .01) + 1, 1))
        actual.append(output)
        return output

    class Model:
        functional = field

        def initialize(self, packet):
            counts = run.fitting._counts(run.fitting.dop853_step)
            gate_counts = counts.copy()
            options = {"stepper": run.fitting.dop853_step}
            for step in (.0025, .00125):
                path = run.fitting._integrate(field, state, step, .1, gate_counts, backward=True, **options)
                run.fitting._integrate(field, path[0], step, .1, gate_counts, **options)
            for step in (.01, .005):
                run.fitting._integrate(field, state, step, 1., gate_counts, **options)
            gates = dict(counts=gate_counts, passed=True, p_endpoint_error=0.)
            stage = dict(counts=counts.copy(), status="converged", reason="mock", actual_residual_calls=0,
                         same_step_feasible_sse=0., projected_gradient_kkt=None,
                         observation_projected_gradient_kkt=None, active_bounds=[],
                         best=dict(state=state.tolist(), history_mse=0.))
            run.fitting._add(counts, gate_counts)
            self.last_initialization = dict(method="fixed_dp8", status="completed", returned_point="stage1",
                                            stage1=stage, stage2=stage, stage1_numerical=gates,
                                            final_numerical=gates, final_check_reused_stage1=True, counts=counts)
            return state.copy()

    monkeypatch.setattr(run.fitting, "_integrate", integrate)
    record, arrays = {}, {}
    result, paths = run.initialize(Model(), packet, record, arrays, {})
    assert run.fitting._integrate is integrate
    assert len(actual) == len(record["dp8_capture_endpoints"]) == 6
    assert record["initialization_rhs_by_operation"] == {"historical_residual": 0, "six_gates": 6480}
    for raw, copy in zip(actual, list(arrays.values())[:6], strict=True):
        assert raw is not copy and raw.tobytes() == copy.tobytes()
    assert set(paths) == {"main", "fine"}
    assert result.tobytes() == state.tobytes()


def test_unfinished_job_cost_stays_partial_and_claim_blocks_replay(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    action = dict(phase="fit", action_id="fixture")
    path, _, record = run.job_record(action, "static")
    _, first = run.reconcile(action, ["static", "modulated"])
    assert first["unknown_jobs"] == ["static"] and first["unclaimed_jobs"] == ["modulated"]
    record["known_costs"] = {"actual_rhs_calls": 17}
    run.save_json(path, record)
    _, partial = run.reconcile(action, ["static", "modulated"])
    assert not partial["jobs"] and set(partial["partial_jobs"]) == {"static"}
    assert partial["known_costs"]["actual_rhs_calls"] == 17 and partial["actual_costs"] is None
    with pytest.raises(FileExistsError):
        run.job_record(action, "static")


@pytest.mark.parametrize("cost_flag", [{}, {"cost_unknown": True}])
def test_fit_failure_without_closed_cost_cannot_freeze(monkeypatch, tmp_path, cost_flag):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    counts = run.matched._counts()
    counts.update(actual_rhs_calls=12, actual_residual_calls=1, proxy_rhs_calls=12)

    def mock_fit(*args):
        diagnostics = dict(status="execution_unresolved", counts=counts, **cost_flag)
        raise run.fitting.FitFailure("fixture", diagnostics) from FloatingPointError("fixture")

    model = SimpleNamespace(fit=mock_fit, inference_config={"fit297_config": {}})
    monkeypatch.setattr(run, "load_parent", lambda mode: model)
    monkeypatch.setattr(run, "fixed_identity", lambda model: "fixture")
    action = dict(phase="fit", action_id="fixture")
    with pytest.raises(run.fitting.FitFailure):
        run.fit_job(action, "static", dict(adaptation_slots=[], source_sha256={}), [])
    path, _ = run.job_paths(action, "static")
    job = json.loads((tmp_path / path).read_bytes())
    assert job["status"] == "interrupted" and job["cost_unknown"] and job["details"] is not None
    assert job["known_costs"]["actual_rhs_calls"] == 12
    _, phase = run.reconcile(action, ["static"])
    assert phase["jobs"] == {} and "static" in phase["partial_jobs"] and phase["actual_costs"] is None


def test_interrupted_l3_without_counters_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    model = SimpleNamespace(functional=object(), delta_star=np.zeros(10))
    monkeypatch.setattr(run, "final_model", lambda *args: model)
    monkeypatch.setattr(run.analysis, "l2_certificate", lambda *args: dict(candidates=[{}]))

    def mock_l3(*args):
        raise RuntimeError("fixture L3 interruption")

    monkeypatch.setattr(run.analysis, "rom_modes", mock_l3)
    action = dict(phase="deploy", action_id="fixture")
    with pytest.raises(RuntimeError, match="fixture L3 interruption"):
        run.analysis_job(action, "static", {"final_model": {"sha256": "fixture"}}, [], {},
                         SimpleNamespace(calls=0), {}, {})
    path, _ = run.job_paths(action, "analysis_static")
    job = json.loads((tmp_path / path).read_bytes())
    assert job["status"] == "interrupted" and job["cost_unknown"] and job["l3_started"]
    assert job["l3_counts"] is None and "l3_rhs_calls" not in job["known_costs"]
    assert job["known_costs"]["l2_radii"] == 1


def test_scoring_cannot_reach_payload_when_selection_is_unfrozen(monkeypatch):
    release = dict(phase="score", workers=1, future_sha256=run.FUTURE[1], action_id="fixture")
    monkeypatch.setattr(run, "release_record", lambda path: (release, "fixture"))
    monkeypatch.setattr(run, "bound_preparation", lambda release: {})
    monkeypatch.setattr(run, "qualification_binding", lambda *args: {})
    monkeypatch.setattr(run, "public_rows", lambda: [])
    monkeypatch.setattr(run, "frozen_fits", lambda *args: ({}, {}))

    def unfrozen(*args):
        raise PermissionError("missing complete selection")

    def inaccessible(*args):
        pytest.fail("no scoring inputs can be reached before the freeze")

    monkeypatch.setattr(run, "frozen_deployment", unfrozen)
    monkeypatch.setattr(run, "load_packets", inaccessible)
    monkeypatch.setattr(run, "score_stage", inaccessible)
    with pytest.raises(PermissionError, match="missing complete selection"):
        run.execute("fixture")


def test_counted_forecast_preserves_failure_prefix_without_model(monkeypatch):
    state = np.zeros(50)
    attempted = []

    def stepper(rhs, value, step):
        attempted.append(step)
        if len(attempted) == 2:
            rhs(value)
            raise FloatingPointError("preset numerical failure")
        for _ in range(12):
            rhs(value)
        return value.copy()

    monkeypatch.setattr(run, "dop853_step", stepper)
    receipt = {}
    array = run.prediction(SimpleNamespace(rhs=np.zeros_like), state, .005, receipt)
    assert array.shape == (3001, 50) and np.isfinite(array[:2]).all() and np.isnan(array[2:]).all()
    assert receipt["first_failed_step"] == 2 and receipt["actual_rhs_calls"] == 13
    assert receipt["steps_completed"] == 1


def test_radau_cap_keeps_original_common_prefix(monkeypatch):
    class Solver:
        def __init__(self, rhs, t0, state, end, *, rtol, atol, max_step):
            assert (t0, end, rtol, atol, max_step) == (0., 1., 1e-11, 1e-13, .005)
            self.rhs, self.t, self.y, self.status = rhs, t0, state, "running"
            rhs(t0, state)
            rhs(t0, state)

        def step(self):
            self.rhs(self.t, self.y)
            self.t = run.GRID[1]

        def dense_output(self):
            self.rhs(self.t, self.y)
            return lambda time: self.y.copy()

    monkeypatch.setattr(run, "Radau", Solver)
    receipt = {}
    state = np.zeros(50)
    radau = run.radau_reference(SimpleNamespace(rhs=np.zeros_like), state, receipt, rhs_limit=4)
    assert receipt["status"] == "rhs_limit"
    assert receipt["rhs_calls_by_operation"] == {"solver_setup": 2, "solver_step": 1, "dense_output": 1}
    fine = np.zeros((101, 50))
    fine[1, -1] = 2e-5
    comparison = run.reference_comparison(np.ones(10), fine, radau, False)
    assert comparison["common_rows"] == 2 and comparison["common_through_s"] == .01
    assert comparison["observation_rms"] == 0. and comparison["full_state_max_relative"] > 1e-5
    assert not comparison["confirmed"]


def test_frozen_l1_units_los_full_grid_and_denominators():
    l1 = run.scoring_module(None)
    mass = scale = np.ones(10)
    delta = np.zeros(10)
    fine, main = np.zeros((3001, 50)), np.zeros((1501, 50))
    event = dict(id="fixture_000", group="fixture", label="stable", x=np.zeros((1511, 30)),
                 classification_meta=dict(equilibrium_delta=delta, inertia=mass, event_time_s=2.))
    record = l1.score_event(event, fine, main, delta_star=delta, mass=mass, p_scale=scale)
    assert record["scopes"]["5"]["fine"]["common_omega_rmse"] == 0.
    changed = np.zeros((1, 30))
    changed[0, 10:20], changed[0, 20] = 2., 3.
    metrics = l1.error_metrics(changed, np.zeros_like(changed), mass, scale)
    assert metrics["common_omega_rmse"] == pytest.approx(2., rel=0., abs=1e-14)
    assert metrics["normalized_p_rmse"] == pytest.approx(3. / np.sqrt(10))
    broken = fine.copy()
    broken[1, -1] = np.nan
    record_nan = l1.score_event(event, broken, main, delta_star=delta, mass=mass, p_scale=scale)
    assert not record_nan["scopes"]["1"]["numerically_admissible"]
    assert run.finite_json(record_nan["pair8"])["coi_normalized_rmse"] is None
    with pytest.raises(ValueError, match="shape"):
        l1.score_event(event, fine[:-1], main, delta_star=delta, mass=mass, p_scale=scale)
    fine[2:, 0], fine[2:, 10] = 4., 20.
    divergent = l1.score_event(event, fine, main, delta_star=delta, mass=mass, p_scale=scale)
    assert divergent["pair8"]["fine_classification"]["predicted_los"]
    assert not divergent["pair8"]["coarse_classification"]["predicted_los"]
    assert not divergent["pair8"]["los_pair_pass"]
    cohort = [record]
    for index in range(1, 297):
        label = "stable" if index < 267 else "unstable"
        other = dict(id=f"fixture_{index:03d}", group="fixture", label=label)
        cohort.append(l1.failure_event(other, "initialization", RuntimeError("fixture")))
    summary = l1.summarize(cohort)
    assert summary["scopes"]["stable_5"]["denominator"] == 267
    assert summary["scopes"]["stable_5"]["covered"] == 1
    assert summary["scopes"]["stable_5"]["official_fine"] is None
    assert summary["scopes"]["unstable_actual15"]["denominator"] == 30
    assert summary["classifications"]["8s"]["raw_denominator"] == 297
