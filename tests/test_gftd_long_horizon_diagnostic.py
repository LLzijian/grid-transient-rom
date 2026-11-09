import gzip
import io
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from experiments.gftd_long_horizon_diagnostic_20261109 import run
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates


def test_verified_input_rejects_escape_and_changed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    relative = Path("frozen/asset.json")
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(b"{}")
    ref = dict(path=relative.as_posix(), sha256=run.sha(b"{}"), bytes=2)
    manifest = {ref["path"]: dict(ref, local_only=True, non_upload=True)}
    assert run.read_ref(ref, relative, manifest) == b"{}"
    with pytest.raises(ValueError, match="reference"):
        run.read_ref(dict(ref, path="../asset.json"), relative, manifest)
    with pytest.raises(ValueError, match="workspace path"):
        run.checked_bytes(Path("../asset.json"), ref["sha256"])
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="changed"):
        run.read_ref(ref, relative, manifest)


def test_exclusive_claim_preserves_first_action(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    path = Path("results/started.json")
    digest = run.claim(path, {"action_id": "first"})
    with pytest.raises(FileExistsError):
        run.claim(path, {"action_id": "duplicate"})
    assert run.sha((tmp_path / path).read_bytes()) == digest


def test_stage_diagnostics_read_best_state_and_reject_changed_anchor(monkeypatch):
    stage1 = np.arange(50, dtype=np.float64)
    stage2 = stage1.copy()
    stage2[30:] += .25
    data = dict(status="numerically_unresolved", returned_point="stage2", method="fixed_dp8",
                stage1_numerical=dict(passed=True),
                final_numerical=dict(passed=False, error="underflow encountered in exp"),
                stage1=dict(best=dict(state=stage1.tolist())), stage2=dict(best=dict(state=stage2.tolist())))
    monkeypatch.setattr(run, "STAGE_STATE_HASHES",
                        {key: run.sha(value.tobytes()) for key, value in (("stage1", stage1), ("stage2", stage2))})
    monkeypatch.setattr(run, "STAGE_P0_HASH", run.sha(stage1[20:30].tobytes()))
    raw = gzip.compress(run.json_bytes(data))
    assert run.stage_states(raw)["stage2"].tobytes() == stage2.tobytes()
    monkeypatch.setattr(run, "STAGE_P0_HASH", run.sha(b"wrong anchor"))
    with pytest.raises(ValueError, match="anchor"):
        run.stage_states(raw)


def test_cached_fine_retains_last_accepted_state_between_grid_rows():
    state = np.zeros(50, dtype=np.float64)
    path = np.full((1501, 50), np.nan)
    path[0] = state
    off_grid = np.ones_like(state)
    payload = io.BytesIO()
    np.savez(payload, times=np.linspace(0., 15., 1501), initial_state=state, main_path=path, fine_path=path,
             main_last_state=state, fine_last_state=off_grid, main_failure_state=off_grid, fine_failure_state=off_grid)
    main = dict(step=.01, target_time=15., accepted_steps=0, last_accepted_time=0., completed=False,
                actual_rhs_calls=3, interrupted_step_rhs_calls=3)
    fine = dict(main, step=.005, accepted_steps=1, last_accepted_time=.005, actual_rhs_calls=15)
    integrals = dict(main=main, fine=fine)
    cache = run.load_cached(payload.getvalue(), state, integrals)
    assert np.isnan(cache["fine_path"][1:]).all()
    np.testing.assert_array_equal(cache["fine_last_state"], off_grid)
    with pytest.raises(ValueError, match="different initial state"):
        run.load_cached(payload.getvalue(), off_grid, integrals)


class LinearField:
    def split_state(self, state):
        if not np.isfinite(state).all():
            raise ValueError("nonfinite state")

    def rhs(self, state):
        return -state


def test_fixed_call_limit_exact_completion_and_terminal_substage():
    state = np.ones(2)
    field = LinearField()
    completed, arrays = run.integrate(field, state, 2, "main", rhs_limit=24)
    assert completed["completed"] and completed["actual_rhs_calls"] == 24
    assert np.allclose(arrays["grid_states"][-1], state * np.exp(-.02))
    failed, arrays = run.integrate(field, state, 2, "main", rhs_limit=17)
    assert failed["status"] == "rhs_limit" and failed["actual_rhs_calls"] == 17
    assert failed["last_accepted_time"] == .01 and failed["failure"]["substage"] == 6
    assert failed["interrupted_step_rhs_calls"] == 5
    assert arrays["terminal_trial_rhs_evaluated"].tolist() == [True] * 5 + [False]
    assert np.isnan(arrays["grid_states"][-1]).all()


def test_actual_reference_calls_include_jacobian_and_dense_output():
    field = LinearField()
    for method in ("radau", "dop853_adaptive"):
        record, arrays = run.integrate(field, np.ones(2), 2, method)
        assert record["completed"] and record["last_accepted_time"] == record["last_output_time"] == .02
        assert len(arrays["accepted_times"]) == record["accepted_steps"] + 1
        assert record["internal_rejected_steps"] is None
        assert record["actual_rhs_calls"] == sum(record["rhs_calls_by_operation"].values())
        if method == "radau":
            assert record["actual_rhs_calls"] > record["solver_nfev"]
        else:
            assert record["rhs_calls_by_operation"]["dense_output"] == 3 * record["dense_output_attempts"]


def test_dense_limit_preserves_accepted_state_and_missing_grid(monkeypatch):
    class DenseCallsRhs:
        nfev = 0
        njev = 0
        nlu = 0

        def __init__(self, rhs, t0, y0, target, **options):
            self.rhs, self.t, self.y = rhs, t0, y0
            self.status = "running"
            self.rhs(t0, y0)

        def step(self):
            self.t = .01
            self.y = self.y / 2
            self.rhs(self.t, self.y)
            self.status = "finished"

        def dense_output(self):
            self.rhs(.005, self.y)

    monkeypatch.setitem(run.REFERENCES, "radau", (DenseCallsRhs, 1e-9, 1e-11))
    record, arrays = run.integrate(LinearField(), np.ones(2), 1, "radau", rhs_limit=2)
    assert record["status"] == "rhs_limit" and record["failure"]["operation"] == "dense_output"
    assert record["last_accepted_time"] == .01 and record["last_output_time"] == 0.
    assert arrays["accepted_states"][-1].tolist() == [.5, .5]
    assert np.isnan(arrays["grid_states"][-1]).all()
    assert arrays["terminal_trial_operations"].tolist() == ["solver_step", "dense_output"]


def test_reference_prefix_never_resumes_after_disagreement():
    initial = np.zeros((20, 10))
    altered = initial.copy()
    altered[4, 2:4] = 9e-7
    third = initial.copy()
    third[15:] = np.nan
    paths = dict(first=initial, second=altered, third=third)
    assert run.reference_prefix(np.ones(2), paths) == (15, 4)
    assert run.agreed(run.differences(np.ones(2), initial, altered)) is False


def artificial_functional(n=2):
    n_edges = n * (n - 1) // 2
    modulation = np.r_[1., np.zeros(n_edges - 1)]
    return GeneratingFunctionalKuramoto(
        mass=np.ones(n), damping=np.full(n, .2), k_edge=np.full(n_edges, .5), eta_edge=np.zeros(n_edges),
        gamma_amp=np.ones(n), lambda_amp=np.ones(n), lambda_mem=np.ones(n), tau_amp=np.full(n, 5.),
        tau_mem_sync=np.ones(n), tau_mem_coi=1., j_omega_mem=np.ones(n),
        gates=StructuralGates(s_edge=np.ones(n_edges), s_mod=modulation, s_mem=np.ones(n)),
    )


def test_inactive_overflow_and_active_subnormal_underflow_are_separate():
    field = artificial_functional(3)
    state = np.zeros(15)
    state[9:12] = [-360., -360., 1200.]
    info, arrays = run.inspect_state(field, state, {})
    assert run.exponent_labels(info["exponents"]) == ["active_exp_underflow", "inactive_zero_exp_out_of_range"]
    for name in ("endpoint_sum", "incidence_sum"):
        assert arrays[name][0] == -720.
        assert info["exponents"][name]["active"]["underflow"] == [0]
        assert info["exponents"][name]["inactive"]["overflow"] == [1, 2]


def test_growth_window_and_power_sampling_use_registered_inputs():
    field = artificial_functional()
    path = np.zeros((11, 10))
    path[:, 6:8] = np.linspace(0., 3., 11)[:, None]
    growth = run.continuous_growth(field, {name: path for name in run.REFERENCES}, len(path))
    assert growth["window"]["start"] == 0. and growth["window"]["end"] == .05
    assert growth["matching_windows"] > 1
    budget, counts = [0], {}
    sample, arrays = run.inspect_state(field, path[3], counts, sample_budget=budget)
    assert counts["rhs"] == budget[0] == 1 and sample["power"]["absolute_residual"] < 1e-12
    assert arrays["rhs"].shape == path[3].shape
    with pytest.raises(RuntimeError, match="single-point RHS limit"):
        run.inspect_state(field, path[3], {}, sample_budget=[500])


def test_frozen_plan_has_only_registered_integrations():
    cases = {f"{name}_{stage}" if stage else name: dict(target_samples=target)
             for (name, stage), target in zip(run.CASES, run.TARGET_SAMPLES, strict=True)}
    references, fixed = run.phase_plan(dict(cases=cases))
    assert len(references) == 10 and sum(map(len, fixed.values())) == 16
    assert set(fixed["modulated_268_stage2"]) == set(run.FIXED_STEPS)
    assert all(name not in fixed for name in ("static_000", "modulated_296", "modulated_048"))


def test_artificial_worker_receipts_reopen_and_summarize(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    for name in run.THREADS:
        monkeypatch.setenv(name, "1")
    field = artificial_functional()
    state = np.zeros(10)
    state[2] = .01
    name, result = run.run_case_jobs("synthetic", "fixture", field, state, 2, tuple(run.REFERENCES))
    assert name == "fixture"
    assert all(ref["path"].endswith(f"/{method}.json") for method, (_, ref) in result.items())
    jobs = {method: record for method, (record, _) in result.items()}
    sample_budget = [0]
    summary, arrays = run.summarize_case(field, run.FrozenCase(state, None, None), 2, jobs, sample_budget)
    assert summary["reference"]["completed"] and summary["reference"]["agreement_end"] == .02
    assert summary["inactive_zero_skip_candidate"] is False and sample_budget[0] == 9
    assert "sample_radau_2_amplitude_gradient" in arrays
    run.json_bytes(summary)
