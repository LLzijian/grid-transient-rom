"""Algebra and mocked orchestration only; no Run2 trajectories or scientific fits."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from kiki.models import gftd_discovery as core
from kiki.models import gftd_fitting as engine
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates


def functional(n=4):
    e = n*(n-1)//2
    return GeneratingFunctionalKuramoto(mass=np.linspace(1., 2., n), damping=np.full(n, .2),
        k_edge=np.full(e, .2), eta_edge=np.zeros(e), gamma_amp=np.full(n, .5), lambda_amp=np.ones(n),
        lambda_mem=np.full(n, 1/.7), tau_amp=np.full(n, 5.), tau_mem_sync=np.full(n, 1/(.7*.3)),
        tau_mem_coi=1., j_omega_mem=np.full(n, .7), storage_p=np.ones(n), gain_p=np.ones(n),
        rate_p=np.full(n, .3), gates=StructuralGates.static_baseline_limit(np.ones(e), n))


def packet(n=4, sigma=1e-4):
    return dict(theta_history=np.zeros((11, n)), omega_history=np.zeros((11, n)),
                p_end=np.linspace(.01, .02, n), history_times=np.linspace(-.1, 0., 11), sigma=sigma)


@pytest.fixture(autouse=True)
def isolated_cache():
    engine._BATCH_CACHE.clear()
    yield
    engine._BATCH_CACHE.clear()


def test_block_projection_equals_global_with_common_cutoff_and_rms_scale():
    systems = []
    for scale in (1., 1e-12):
        A = np.arange(24, dtype=float).reshape(8, 3)/10
        Z = np.zeros((8, 2))
        Z[:2] = np.diag([scale, scale*.3])
        systems.append(dict(A=A, Z=Z, b=np.linspace(-.3, .4, 8)))
    A, Z, b = core.combine_systems(systems)
    global_projection = core.project_memory(A, Z, b)
    block_projection = engine.project_systems(systems)
    np.testing.assert_allclose(block_projection["X"], global_projection["X"], atol=3e-16, rtol=0.)
    np.testing.assert_allclose(block_projection["y"], global_projection["y"], atol=3e-17, rtol=0.)
    assert sum(item["memory_rank"] for item in block_projection["blocks"]) == global_projection["memory_rank"]
    assert block_projection["memory_ranks"] == global_projection["memory_ranks"]


def test_static_backward_selection_counts_once_debiases_and_retains_zero_graph():
    X = np.eye(6)*.01
    k = np.array([.7, 0., 0., .9, 0., 1.1])
    counts = engine._counts()
    result = engine.backward_select(X, X@k, 0., np.ones(6, dtype=bool), np.zeros(6, dtype=bool), counts)
    assert result["status"] == "qualified"
    np.testing.assert_allclose(result["raw_k"], k, atol=2e-16)
    assert np.flatnonzero(result["edge_mask"]).tolist() == [0, 3, 5]
    assert counts["algebra_terms_planned"] == 157
    assert counts["algebra_terms_actual"]+counts["algebra_terms_skipped"] == 157
    assert counts["nnls_attempts"] == counts["nnls_completed"] == counts["nnls_qualified"]
    zero = engine.backward_select(X, np.zeros(6), 0., np.ones(6, dtype=bool), np.zeros(6, dtype=bool), engine._counts())
    assert not zero["edge_mask"].any() and zero["raw_k"].tolist() == [0.]*6
    assert zero["debias"]["actual_nnls_calls"] == 0


def test_hierarchical_solver_uses_uv_mapping_and_deterministic_ties():
    X = np.eye(4)*.01
    counts = engine._counts()
    result = engine.backward_select(X, X@np.array([.3, .8, .2, 0.]), 0., np.ones(2, dtype=bool), np.ones(2, dtype=bool), counts)
    np.testing.assert_allclose(result["raw_k"], [.5, .8])
    np.testing.assert_allclose(result["raw_beta"], [.2, 0.])
    assert result["mod_mask"].tolist() == [True, False]
    tied_X = np.ones((2, 2))*.01
    tied = engine.backward_select(tied_X, np.full(2, .01), 0., np.ones(2, dtype=bool), np.zeros(2, dtype=bool), engine._counts())
    assert tied["edge_mask"].tolist() == [False, True]


def test_unresolved_comparison_rejects_whole_proposal_and_counts_failure(monkeypatch):
    real = core.solve_nnls
    calls = []
    def solve(X, y):
        calls.append(X.shape[1])
        if len(calls) == 3:
            return dict(status="solver_unresolved", coefficients=None, loss=None,
                        actual_algebra_terms=1, actual_nnls_calls=1, nnls_completed=0, nnls_qualified=0)
        return real(X, y)
    monkeypatch.setattr(core, "solve_nnls", solve)
    counts = engine._counts()
    result = engine.backward_select(np.eye(3), np.array([1., 0., 0.]), 0., np.ones(3, dtype=bool), np.zeros(3, dtype=bool), counts)
    assert result["status"] == "solver_unresolved" and result["raw_k"] is None
    assert len(calls) == counts["nnls_attempts"] == 3
    assert counts["algebra_terms_actual"]+counts["algebra_terms_skipped"] == counts["algebra_terms_planned"]


@pytest.mark.parametrize("n,sigma,mode,role,dimension", [
    (4, 0., "static", "smooth", 4), (4, 0., "modulated", "smooth", 8),
    (4, 1e-4, "static", "smooth", 11), (4, 1e-4, "modulated", "smooth", 15),
    (10, 1e-4, "modulated", "smooth", 39), (10, 1e-4, "modulated", "simple", 10)])
def test_endpoint_layout_has_no_equal_bounds_and_preserves_p(n, sigma, mode, role, dimension):
    f, obs = functional(n), packet(n, sigma)
    _, low, high = engine._layout(n, sigma, mode, role)
    assert len(low) == dimension and np.all(low < high)
    end = engine._endpoint(obs, f, np.zeros(dimension), mode, role)
    np.testing.assert_array_equal(end[2*n:3*n], obs["p_end"])
    changed = copy.deepcopy(obs)
    changed["theta_history"] += 1.7
    shifted = engine._endpoint(changed, f, np.zeros(dimension), mode, role)
    np.testing.assert_allclose(shifted[:n]-end[:n], 1.7)


def test_jacobian_boundary_stencils_stay_inside_box():
    points = []
    def residual(u):
        points.append(u.copy())
        assert np.all(np.abs(u) <= 2.)
        return u**3
    u = np.array([-2., .3, 2.])
    J = engine._jacobian(residual, u, np.full(3, -2.), np.full(3, 2.))
    np.testing.assert_allclose(J, np.diag(3*u*u), atol=3e-9, rtol=0.)


def test_budget_counts_jacobian_evaluations_and_retains_probe_best(monkeypatch):
    f, obs = functional(2), packet(2, 0.)
    def integrate(f, state, step, duration, counts, **kwargs):
        history = np.tile(state, (41, 1))
        history[:, 0] = state[-2]-.01
        return history
    monkeypatch.setattr(engine, "_integrate", integrate)
    def solver(fun, x, jac, **kwargs):
        jac(x)
        pytest.fail("budget should interrupt the next Jac column")
    monkeypatch.setattr(engine, "least_squares", solver)
    result = engine.infer_endpoint(f, obs, mode="static", role="proxy", call_limit=3)
    assert result["status"] == "budget_limited" and result["actual_residual_calls"] == 3
    assert result["best"]["u"][0] == 2.5e-5
    assert result["projected_gradient_kkt"] is None


def test_mock_optimizer_success_path_records_actual_calls_and_same_point_gradient(monkeypatch):
    f, obs = functional(2), packet(2, 0.)
    monkeypatch.setattr(engine, "_integrate", lambda f, state, *args, **kwargs: np.tile(state, (41, 1)))
    def solver(fun, x, jac, **kwargs):
        residual, J = fun(x), jac(x)
        return SimpleNamespace(success=True, status=1, message="mock stationarity", x=x.copy(),
                               grad=J.T@residual, nfev=1, njev=1)
    monkeypatch.setattr(engine, "least_squares", solver)
    task = engine.infer_endpoint(f, obs, mode="static", role="simple", call_limit=600)
    assert task["status"] == "converged" and task["actual_residual_calls"] == 6
    assert task["projected_gradient_kkt"] == 0.
    np.testing.assert_array_equal(task["gradient_evaluated_at"], task["best"]["u"])
    json.dumps(engine._plain(task), allow_nan=False)


def test_config_binds_all_parameters_and_rejects_incomplete_or_changed_config():
    f = functional()
    config = engine._config(f, "modulated", 0.)
    assert engine.validate_config(f, json.loads(json.dumps(config))) == config
    for wrong in (None, {}, dict(config, simple_calls=601)):
        with pytest.raises(engine.InitializationFailure) as caught:
            engine.validate_config(f, wrong)
        assert caught.value.diagnostics["status"] == "invalid_configuration"
    f.rate_p[0] += .1
    with pytest.raises(engine.InitializationFailure):
        engine.validate_config(f, config)


def local_task(f, obs, *, role, sse=1., status="budget_limited", state=None):
    dimension = len(engine._layout(f.n_nodes, float(obs["sigma"]), "modulated", role)[1])
    state = engine._endpoint(obs, f, np.zeros(dimension), "modulated", role) if state is None else state
    return dict(status=status, best=dict(state=state, u=np.zeros(dimension), observation_sse=sse),
                same_step_feasible_sse=1., counts=engine._counts(), projected_gradient_kkt=None)


def test_stage1_failure_is_typed_and_never_runs_stage2(monkeypatch):
    f, obs, calls = functional(), packet(), []
    def infer(*args, **kwargs):
        calls.append(kwargs["role"])
        return dict(status="failed", best=None, counts=engine._counts())
    monkeypatch.setattr(engine, "infer_endpoint", infer)
    with pytest.raises(engine.InitializationFailure) as caught:
        engine.initialize_complete(f, obs, engine._config(f, "modulated", 1e-4))
    assert caught.value.diagnostics["status"] == "blocked_by_stage1" and calls == ["simple"]


def test_no_improvement_returns_checked_stage1_without_cross_point_certification(monkeypatch):
    f, obs, checks = functional(), packet(), []
    def infer(f, packet, **kwargs):
        return local_task(f, packet, role=kwargs["role"])
    def check(f, state, packet):
        checks.append(state.copy())
        return dict(passed=True, counts=engine._counts())
    monkeypatch.setattr(engine, "infer_endpoint", infer)
    monkeypatch.setattr(engine, "numerical_check", check)
    state, diagnostics = engine.initialize_complete(f, obs, engine._config(f, "modulated", 1e-4))
    assert len(checks) == 1 and diagnostics["returned_point"] == "stage1" and diagnostics["no_improvement"]
    np.testing.assert_array_equal(state, checks[0])
    json.dumps(diagnostics, allow_nan=False)


def test_improved_stage2_must_pass_its_own_numerical_check(monkeypatch):
    f, obs, checks = functional(), packet(), []
    def infer(f, packet, **kwargs):
        task = local_task(f, packet, role=kwargs["role"], sse=1. if kwargs["role"] == "simple" else .5)
        if kwargs["role"] == "smooth":
            task["best"]["state"][-1] = .001
        return task
    def check(*args):
        checks.append(1)
        return dict(passed=len(checks) == 1, counts=engine._counts())
    monkeypatch.setattr(engine, "infer_endpoint", infer)
    monkeypatch.setattr(engine, "numerical_check", check)
    with pytest.raises(engine.InitializationFailure) as caught:
        engine.initialize_complete(f, obs, engine._config(f, "modulated", 1e-4))
    assert len(checks) == 2 and caught.value.diagnostics["status"] == "numerically_unresolved"


def test_numerical_check_compares_vectors_same_endpoint_and_all_5n_roundtrip(monkeypatch):
    f, obs, requests = functional(2), packet(2), []
    state = f.pack_state(np.zeros(2), np.zeros(2), obs["p_end"], np.zeros(2), np.zeros(2))
    def integrate(f, endpoint, step, duration, counts, **kwargs):
        requests.append((endpoint.copy(), step, duration, kwargs.get("backward", False)))
        states = np.tile(endpoint, (int(round(duration/.01))+1, 1))
        if duration == .1 and kwargs.get("backward"):
            states[:, 0] += 1e-5 if step == .0025 else -1e-5
            states[-1] = endpoint
        return states
    monkeypatch.setattr(engine, "_integrate", integrate)
    result = engine.numerical_check(f, state, obs)
    assert not result["passed"] and result["history_main_fine_rms"] > 1e-7
    for endpoint, _, duration, _ in requests[-2:]:
        assert duration == 1.
        np.testing.assert_array_equal(endpoint, state)
    assert result["p_endpoint_error"] == 0.


def test_initial_preparation_is_copy_safe_cached_and_counts_references_once(monkeypatch):
    f, obs = functional(), packet()
    prepared_calls = []
    def prepare(packet, f, step):
        prepared_calls.append(step)
        A = np.zeros((16, 6))
        A[4:10] = np.eye(6)*.01
        Z = np.zeros((16, 4))
        Z[:4] = np.eye(4)
        return dict(A_static=A, Z=Z, b=A@np.array([.7, 0., 0., .9, 0., 1.1]))
    monkeypatch.setattr(core, "prepare_weak_history", prepare)
    first, hit, key = engine._prepare_batch(f, [obs, obs])
    assert not hit and len(prepared_calls) == 4
    assert first["counts"]["reconstructions"] == 4
    assert first["counts"]["algebra_terms_planned"] == 157+2*65
    first["grids"][.0025][0]["A_static"][:] = 99.
    second, hit, same_key = engine._prepare_batch(f, [obs, obs])
    assert hit and same_key == key and np.max(second["grids"][.0025][0]["A_static"]) == .01
    f.k_edge[:] = 9.
    assert engine._prepare_batch(f, [obs, obs])[1]
    changed = copy.deepcopy(obs)
    changed["sigma"] = 0.
    assert not engine._prepare_batch(f, [changed, changed])[1]
    f.mass[0] += .1
    assert not engine._prepare_batch(f, [obs, obs])[1]


def test_failed_preparation_retains_partial_actual_cost_without_cache(monkeypatch):
    f, calls = functional(), []
    def prepare(*args):
        calls.append(1)
        if len(calls) == 2:
            raise FloatingPointError("mock partial reconstruction failure")
        return {}
    monkeypatch.setattr(core, "prepare_weak_history", prepare)
    with pytest.raises(engine.FitFailure) as caught:
        engine._prepare_batch(f, [packet(), packet()])
    counts = caught.value.diagnostics["counts"]
    assert counts["reconstruction_attempts"] == 2 and counts["reconstructions"] == 1
    assert not engine._BATCH_CACHE


def test_fit_failure_is_typed_and_does_not_mutate_source():
    f = functional()
    before = engine._hash(engine._physics(f))
    invalid = dict(packet(), true_r=np.zeros(4))
    with pytest.raises(engine.FitFailure):
        engine.fit_shared(f, [invalid], mode="modulated")
    assert engine._hash(engine._physics(f)) == before


def test_proxy_event_mean_and_two_initial_starts_for_non_three_batch(monkeypatch):
    f, calls = functional(), []
    obs = [packet(), packet()]
    obs[1]["theta_history"][:] = 2.
    def infer(f, packet, **kwargs):
        calls.append(kwargs)
        task = local_task(f, packet, role="proxy")
        state = task["best"]["state"]
        task["best"].update(history_mse=1.+float(packet["theta_history"][0, 0]),
                             dense_history=np.tile(state, (41, 1)))
        return task
    monkeypatch.setattr(engine, "infer_endpoint", infer)
    result = engine._proxy_candidate(f, obs, "modulated", np.zeros(6, dtype=bool), np.zeros(6, dtype=bool))
    assert len(calls) == 4 and all(item["call_limit"] == 200 for item in calls)
    assert result["objective"] == 2.
    np.testing.assert_array_equal(calls[1]["initial"][7:11], [.2, -.2, .2, -.2])


def test_shared_fit_uses_incumbent_after_rejection_and_serializes_full_path(monkeypatch):
    f, packets = functional(), [packet(), packet()]
    original = engine._hash(engine._physics(f))
    selected = dict(status="qualified", threshold_inconsistent=False, raw_k=np.full(6, .7),
                    edge_mask=np.ones(6, dtype=bool))
    preparation = dict(counts=engine._counts(), initial_static=selected,
                       initial_information=dict(columns_qualified=[True]*6), references={})
    monkeypatch.setattr(engine, "_prepare_batch", lambda *args: (preparation, False, "mock-key"))
    losses = iter([10., 12., 5., 8.])
    def proxy(f, packets, mode, edges, mods):
        counts = engine._counts()
        counts["optimization_tasks"] = 2*len(packets)
        counts["actual_residual_calls"] = 2*len(packets)
        return dict(status="completed", counts=counts, objective=next(losses), events=[])
    monkeypatch.setattr(engine, "_proxy_candidate", proxy)
    monkeypatch.setattr(engine, "_systems_for_candidate", lambda *args: [dict(A=np.eye(6)*.01,
        A_static=np.eye(6)*.01, Z=np.empty((6, 0)), b=np.ones(6)*.01)])
    masks = []
    def backward(X, y, sigma, edges, mods, counts):
        index = len(masks)
        masks.append(edges.copy())
        proposed = edges.copy()
        proposed[index] = False
        k = np.where(proposed, .4+.1*index, 0.)
        return dict(status="qualified", raw_k=k, raw_beta=np.zeros(6), edge_mask=proposed,
                    mod_mask=np.zeros(6, dtype=bool), edge_threshold_inconsistent=False,
                    mod_threshold_inconsistent=False, debias=dict(coefficients=k))
    monkeypatch.setattr(engine, "backward_select", backward)
    monkeypatch.setattr(engine, "mechanism_information", lambda *args: dict(k_qualified=True, beta_qualified=False))
    result = engine.fit_shared(f, packets, mode="static")
    assert result["fit_state"]["accepted_proposal_indices"] == [1]
    np.testing.assert_array_equal(masks[0], masks[1])
    assert not masks[2][1] and masks[2][0]
    np.testing.assert_allclose(result["functional"].k_edge, [ .5, 0., .5, .5, .5, .5])
    assert result["fit_state"]["from_data"] and not result["fit_state"]["prior_only"]
    assert result["fit_state"]["counts"]["optimization_tasks"] == 16
    assert engine._hash(engine._physics(f)) == original
    engine.validate_config(result["functional"], result["config"])
    json.dumps(result["fit_state"], allow_nan=False)


def test_unaccepted_exploration_prior_cannot_become_a_discovery(monkeypatch):
    f = functional()
    preparation = dict(counts=engine._counts(), initial_static=dict(status="insufficient_information"),
                       initial_information=dict(columns_qualified=[False]*6), references={})
    monkeypatch.setattr(engine, "_prepare_batch", lambda *args: (preparation, True, "mock-key"))
    monkeypatch.setattr(engine, "_proxy_candidate", lambda *args: dict(status="completed", counts=engine._counts(), objective=1., events=[]))
    monkeypatch.setattr(engine, "_systems_for_candidate", lambda *args: [dict(A=np.eye(12), Z=np.empty((12, 0)), b=np.zeros(12))])
    monkeypatch.setattr(engine, "backward_select", lambda *args: dict(status="insufficient_information"))
    monkeypatch.setattr(engine, "mechanism_information", lambda *args: dict(k_qualified=True, beta_qualified=True))
    result = engine.fit_shared(f, [packet()], mode="modulated")
    state = result["fit_state"]
    assert state["prior_only"] and not state["from_data"] and state["accepted_proposal_indices"] == []
    assert state["support"] is None and state["modulation_support"] is None
    assert state["null_reasons"] == dict(k=["prior_only"], beta=["prior_only"])
    np.testing.assert_array_equal(result["functional"].k_edge, np.full(6, .2))


def test_zero_amplitude_explanation_keeps_k_information_but_not_beta(monkeypatch):
    f = functional()
    static = np.zeros((16, 6))
    static[:6] = np.eye(6)*.01
    system = dict(A=np.hstack((static, static)), A_static=static,
                  explanation=np.hstack((static, np.zeros_like(static))), Z=np.empty((16, 0)), b=np.zeros(16))
    monkeypatch.setattr(engine, "_systems_for_candidate", lambda *args: [system])
    info = engine.mechanism_information({}, f, {}, "modulated", engine._counts())
    assert info["k_qualified"] and not info["beta_qualified"]
    assert info["rank_signatures_agree"]
    assert info["grids"]["0.0025"]["shared"]["column_norms"][6:].tolist() == [0.]*6


def test_debias_threshold_failure_is_reported_without_reselection(monkeypatch):
    calls = []
    def solve(X, y):
        calls.append(X.shape[1])
        coefficients = np.zeros(0) if not X.shape[1] else np.array([.04 if len(calls) == 3 else .06])
        return dict(status="qualified", coefficients=coefficients, loss=1. if not X.shape[1] else 0.,
                    actual_algebra_terms=1, actual_nnls_calls=int(bool(X.shape[1])),
                    nnls_completed=int(bool(X.shape[1])), nnls_qualified=int(bool(X.shape[1])))
    monkeypatch.setattr(core, "solve_nnls", solve)
    result = engine.backward_select(np.ones((2, 1)), np.ones(2), 0., np.ones(1, dtype=bool),
                                    np.zeros(1, dtype=bool), engine._counts())
    assert calls == [1, 0, 1]
    assert result["edge_mask"].tolist() == [True] and result["raw_k"].tolist() == [.04]
    assert result["edge_threshold_inconsistent"] and not result["mod_threshold_inconsistent"]


def test_nonfinite_proposals_are_recorded_rejected_and_do_not_destroy_incumbent(monkeypatch):
    f = functional()
    preparation = dict(counts=engine._counts(), initial_static=dict(status="qualified", threshold_inconsistent=False,
        raw_k=np.full(6, .7), edge_mask=np.ones(6, dtype=bool)), initial_information=dict(columns_qualified=[True]*6), references={})
    monkeypatch.setattr(engine, "_prepare_batch", lambda *args: (preparation, True, "mock-key"))
    monkeypatch.setattr(engine, "_proxy_candidate", lambda *args: dict(status="completed", counts=engine._counts(), objective=1., events=[]))
    def nonfinite(*args):
        raise FloatingPointError("mock candidate exponential overflow")
    monkeypatch.setattr(engine, "_systems_for_candidate", nonfinite)
    monkeypatch.setattr(engine, "mechanism_information", lambda *args: dict(k_qualified=True, beta_qualified=False))
    result = engine.fit_shared(f, [packet()], mode="static")
    state = result["fit_state"]
    assert len(state["trace"]) == 3 and all(row["reason"] == "parameter_nonfinite" for row in state["trace"])
    assert state["accepted_proposal_indices"] == [] and state["counts"]["algebra_terms_skipped"] == 471
    np.testing.assert_array_equal(result["functional"].k_edge, np.full(6, .7))
