"""Fixed mock qualification for matched shared fitting, without optimizer or ODE."""

from types import SimpleNamespace

import numpy as np
import pytest

from kiki.models import gftd_matched_prediction as fit


def slots():
    return [dict(global_index=index, local_slot=slot, event_id=f"fixture_{index:03d}")
            for slot, index in enumerate(fit.ADAPTATION_INDICES)]


def request_for(mode):
    kernel, parent_sha = fit.PARENTS[mode]
    return dict(format=fit.FORMAT, run_id=fit.RUN_ID, protocol_sha256=fit.PROTOCOL_SHA256,
                parent_model_sha256=parent_sha, kernel=kernel, mode=mode, training_count=297,
                adaptation_slots=slots(), parent_fit297_config=dict(mode=mode, training_count=297),
                source_sha256={"fixture": "a" * 64}, input_sha256=fit.INPUT_SHA256,
                event_order_sha256=fit.EVENT_ORDER_SHA256)


@pytest.mark.parametrize("mode", ["static", "modulated"])
def test_strict_registered_request(monkeypatch, mode):
    request = request_for(mode)
    functional = SimpleNamespace(n_nodes=10, n_edges=45, modulation_kernel=request["kernel"],
                                 k_edge=np.ones(45), gates=SimpleNamespace(s_edge=np.ones(45), s_mod=np.zeros(45)))
    monkeypatch.setattr(fit.fit297, "validate_config", lambda f, config: config)
    validated = fit.validate_request(functional, request, mode, 297)
    assert validated == request and validated is not request
    for changed in (dict(extra=True), dict(parent_model_sha256="a" * 64),
                    dict(training_count=297.), dict(adaptation_slots=list(reversed(slots())))):
        with pytest.raises(ValueError):
            fit.validate_request(functional, dict(request, **changed), mode, 297)


@pytest.mark.parametrize("mode", ["static", "modulated"])
@pytest.mark.parametrize("failure", [None, "gate", "missing", "zero_debiased"])
def test_unique_proxy_start_and_original_gate(monkeypatch, mode, failure):
    events = slots()
    packets = [dict(slot) for slot in events]
    gates = SimpleNamespace(s_edge=np.array([1., 0., 1.]), s_mod=np.zeros(3))
    f = SimpleNamespace(n_nodes=3, k_edge=np.array([1., 0., 1.]), gates=gates)
    starts, checks = [], []

    def infer(f, packet, *, mode, role, call_limit, initial, config, stepper):
        assert role == "proxy" and call_limit == 1200 and stepper is fit.old.dop853_step
        slot = packet["local_slot"]
        second = len(starts) % 2
        alternate = np.array([.2, -.2, .2, 0., 0., 0.] if mode == "modulated" else [.1, -.1, .1])
        np.testing.assert_array_equal(initial, alternate if second else np.zeros_like(alternate))
        starts.append(slot)
        # Failed first start retains a lower finite MSE at slot zero.
        failed = (slot == 0 and not second) or (slot == 1 and failure == "missing")
        mse = .5 if slot == 0 and not second else 1.
        best = dict(u=initial, state=np.full(15, second), history_mse=mse, residual=np.zeros(66),
                    dense_history=np.zeros((41, 15)))
        return dict(status="failed" if failed else "budget_limited", best=best,
                    counts=dict(optimization_tasks=1, actual_residual_calls=1, actual_rhs_calls=480))

    def check(f, state, packet, *, stepper, capture):
        assert stepper is fit.old.dop853_step
        slot = packet["local_slot"]
        checks.append((slot, int(state[0])))
        capture["fixture"] = state.copy()
        return dict(passed=failure != "gate" or slot != 1, counts=dict(actual_rhs_calls=6480))

    monkeypatch.setattr(fit.old, "infer_endpoint", infer)
    monkeypatch.setattr(fit.old, "numerical_check", check)
    counts, record = fit._counts(), {}
    edge_mask = np.array([True, failure == "zero_debiased", True])
    mod_mask = np.array([False, mode == "modulated" and failure == "zero_debiased", False])
    fit._candidate(f, packets, events, mode, {"model_sha256": "fixture"}, counts, record,
                   edge_mask=edge_mask, mod_mask=mod_mask)
    assert starts == [slot for slot in range(24) for _ in range(2)]
    assert counts["actual_residual_calls"] == 48 and counts["proxy_rhs_calls"] == 48 * 480
    assert record["events"][0]["selected"] == 1
    if failure == "missing":
        assert record["status"] == "no_output" and record["objective"] is None and not checks
    elif failure == "gate":
        assert record["status"] == "numerical_failure"
        assert checks == [(0, 1)] + [(slot, 0) for slot in range(1, 24)]
        assert record["events"][1]["qualification_status"] == "failed"
        assert record["events"][2]["qualification_status"] == "passed"
    else:
        assert record["status"] == "completed"
        assert checks == [(0, 1)] + [(slot, 0) for slot in range(1, 24)]
        penalty = 1e-12 * (edge_mask.sum() + mod_mask.sum())
        assert record["objective"] == pytest.approx(1 + penalty, abs=1e-15, rel=0)
        if failure == "zero_debiased":
            np.testing.assert_array_equal(record["edge_mask"], [True, True, True])
            np.testing.assert_array_equal(record["mod_mask"], [False, mode == "modulated", False])
            assert fit.old._improves(1 + 2e-12, 1 + 2.5e-12)
            assert not fit.old._improves(record["objective"], 1 + 2.5e-12)
    assert counts["qualification_suites"] == len(checks)
    assert counts["qualification_rhs_calls"] == len(checks) * 6480


@pytest.mark.parametrize("case, expected", [
    ("retention", [0, 2]), ("tie", []), ("solver_failure", []),
    ("gate_failure", []), ("missing", []), ("initial_failure", None), ("technical_failure", None),
])
def test_three_proposals_reopen_complete_fixture_library(monkeypatch, case, expected):
    mode = "modulated"
    request = request_for(mode)
    parent = request["parent_fit297_config"]
    parent.update(fit_request={"k_ref": 1.}, preparation_key="parent-key")
    packets = [dict(sigma=0., index=index) for index in range(297)]
    f = SimpleNamespace(n_nodes=3, n_edges=3, k_edge=np.ones(3),
                        gates=SimpleNamespace(s_edge=np.array([1., 0., 1.]), s_mod=np.zeros(3)))
    objectives = iter([1., .9, 1.1, .8] if case != "tie" else [1.] * 4)
    proposal_calls, prepared_keys = [], []
    monkeypatch.setattr(fit, "validate_request", lambda f, config, mode, count: config)
    monkeypatch.setattr(fit.core, "validate_observation", lambda packet, n: packet)
    monkeypatch.setattr(fit.old, "make_fixed_dp8_config", lambda f, config: config)
    monkeypatch.setattr(fit.fit297, "make_config", lambda f, mode, req, count, key: parent)
    monkeypatch.setattr(fit.fit297, "project_cached", lambda systems, factors: dict(X="fixture", y="fixture"))
    monkeypatch.setattr(fit, "_system", lambda f, cache, dense, mode: cache)
    monkeypatch.setattr(fit, "_diagnose", lambda *args: [])
    monkeypatch.setattr(fit, "_audit", lambda *args: dict(columns_qualified=[True] * 6))

    def prepare(f, subset, counts, ledger):
        prepared_keys.extend(packet["index"] for packet in subset)
        return {.0025: (prepared_keys, [None] * 24)}

    def candidate(f, subset, slots, mode, config, counts, record, *, edge_mask, mod_mask):
        assert [packet["index"] for packet in subset] == list(fit.ADAPTATION_INDICES)
        assert [slot["local_slot"] for slot in slots] == list(range(24))
        if not proposal_calls:
            np.testing.assert_array_equal(f.gates.s_edge, [1., 0., 1.])
            np.testing.assert_array_equal(edge_mask, [True, False, True])
            np.testing.assert_array_equal(mod_mask, [False, False, False])
        else:
            assert f.k_edge[1] == 0.
            np.testing.assert_array_equal(edge_mask, [True, True, True])
            np.testing.assert_array_equal(mod_mask, [True, True, True])
        status = "completed"
        if case == "initial_failure" or (case == "gate_failure" and proposal_calls):
            status = "numerical_failure"
        elif case == "missing" and proposal_calls:
            status = "no_output"
        record.update(status=status, objective=next(objectives),
                      events=[dict(best={"dense_history": None})] * 24)
        counts["optimization_tasks"] += 48
        if case == "technical_failure":
            raise RuntimeError("fixture exception before authoritative callback count")

    def propose(X, y, sigma, edge_mask, mod_mask, counts, *, k_ref):
        assert edge_mask.tolist() == mod_mask.tolist() == [True] * 3
        proposal_calls.append(True)
        return dict(status="solver_unresolved" if case == "solver_failure" else "qualified",
                    edge_mask=edge_mask, mod_mask=mod_mask, raw_k=np.array([1., 0., 1.]),
                    raw_beta=np.array([.5, 0., .5]), threshold_inconsistent=False)

    def parameters(f, k, beta):
        return SimpleNamespace(k_edge=k, gates=SimpleNamespace(s_edge=np.ones(3), s_mod=beta))

    monkeypatch.setattr(fit, "_prepare", prepare)
    monkeypatch.setattr(fit, "_candidate", candidate)
    monkeypatch.setattr(fit.old, "backward_select", propose)
    monkeypatch.setattr(fit.old, "_parameters", parameters)
    if expected is None:
        with pytest.raises(fit.old.FitFailure) as error:
            fit.fit_shared(f, packets, mode=mode, config=request)
        assert error.value.diagnostics["status"] == (
            "execution_unresolved" if case == "technical_failure" else "fit_failed")
        assert error.value.diagnostics["cost_unknown"] is (case == "technical_failure")
        assert error.value.diagnostics["counts"]["optimization_tasks"] == 48
        assert not proposal_calls
    else:
        result = fit.fit_shared(f, packets, mode=mode, config=request)
        assert result["fit_state"]["accepted_proposal_indices"] == expected
        assert result["fit_state"]["counts"]["optimization_tasks"] == (48 if case == "solver_failure" else 192)
        assert len(proposal_calls) == 3
    assert prepared_keys == list(fit.ADAPTATION_INDICES)
    np.testing.assert_array_equal(f.gates.s_edge, [1., 0., 1.])


def test_prepare_caches_preserve_noncontiguous_slots_per_kernel(monkeypatch):
    packets = slots()

    def prepare(packet, functional, step, *, on_port_rhs):
        for _ in range(2 * 4 * round(.1 / step)):
            on_port_rhs()
        return dict(packet, kernel=functional.modulation_kernel, step=step)

    def factors(cache, ledger, counts):
        ledger["memory_svd"] += 1
        counts["memory_svd"] += 1
        return dict(cache)

    monkeypatch.setattr(fit.core, "prepare_weak_history", prepare)
    monkeypatch.setattr(fit.fit297, "_factors", factors)
    prepared = {}
    for mode, (kernel, _) in fit.PARENTS.items():
        counts, ledger = fit._counts(), {"memory_svd": 0}
        prepared[mode] = fit._prepare(SimpleNamespace(modulation_kernel=kernel), packets, counts, ledger)
        assert counts["port_rhs_calls"] == 23040
        assert counts["reconstructions"] == counts["memory_svd"] == ledger["memory_svd"] == 48
    for step in (.0025, .00125):
        for mode, (kernel, _) in fit.PARENTS.items():
            caches, cached_factors = prepared[mode][step]
            assert caches == cached_factors
            assert [row["global_index"] for row in caches] == list(fit.ADAPTATION_INDICES)
            assert [row["local_slot"] for row in caches] == list(range(24))
            assert [row["event_id"] for row in caches] == [row["event_id"] for row in packets]
            assert {row["kernel"] for row in caches} == {kernel}
            assert {row["step"] for row in caches} == {step}


@pytest.mark.parametrize("u, kind, calls", [
    ((0., 0.), "polynomial", 8), ((-2., 2.), "polynomial", 12), ((0., 0.), "constant", 8),
    ((0., 0.), "nonfinite_column", 8),
])
def test_fixed_adequacy_stencils_and_projection(u, kind, calls):
    u = np.array(u)
    visited = []

    def evaluate(point):
        if kind == "constant":
            return np.array([0., 0., 1.])
        if kind == "nonfinite_column" and point[0] != 0:
            return np.array([np.nan, 0., 1.])
        return np.r_[point, 1.]

    def residual(point):
        assert np.all(np.abs(point) <= 2)
        visited.append(point)
        return evaluate(point)

    out = fit._adequacy(residual, u, evaluate(u), np.full(2, -2.), np.full(2, 2.))
    assert len(visited) == calls
    if kind == "nonfinite_column":
        assert out["d_g"] is None
        assert all(row["status"] == "unknown" for row in out["steps"])
        assert all(row["status"] == "not_run" for row in out["trials"])
    elif u[0] == -2.:
        assert out["locally_consistent"] is True
        assert out["steps"][1]["eta"] == pytest.approx(2/(3*np.sqrt(2)))
        assert all(row["improved"] is True for row in out["trials"])
    else:
        assert out["small_relative_gradient"] is True
        assert all(row["status"] == "zero_direction" for row in out["trials"])
