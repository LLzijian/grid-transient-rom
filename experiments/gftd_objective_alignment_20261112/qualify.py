"""Released six-group algebra/mock qualification for the isolated runner."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import run


np = run.np
DENY = ("fit_shared", "_prepare", "_audit", "_diagnose")
INFER_WITH_MOCKS = run.fitting.infer_endpoint


class MockBudget:
    def __init__(self, previous):
        self.previous = previous
        self.calls = Counter()

    def wrap(self, name, callback):
        def counted(*args, **kwargs):
            self.calls[name] += 1
            run.require(self.total <= 1000 and self.total + self.previous <= 3000, "mock limit reached")
            return callback(*args, **kwargs)
        return counted

    @property
    def total(self):
        return sum(self.calls.values())


def rejects(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("expected rejection")


def slots():
    return [dict(local_slot=i, global_index=index, event_id=f"fixture_{index:03d}")
            for i, index in enumerate(run.SLOTS)]


def fixture_parameters(parent, c, beta):
    return SimpleNamespace(k_edge=np.asarray(c), modulation_kernel=parent.modulation_kernel,
                           gates=SimpleNamespace(s_edge=(c > 0).astype(float), s_mem=parent.gates.s_mem,
                                                 s_mod=np.divide(beta, c, out=np.zeros_like(c), where=c > 0),
                                                 s_coi=0.0))


def algebra(budget):
    parent = SimpleNamespace(modulation_kernel=run.MODES["modulated"],
                             gates=SimpleNamespace(s_mem=np.ones(3), s_coi=0.))
    endpoints = [dict(c=[2., 1., 0.], beta=[0., 1., 0.]), dict(c=[0., 2., 3.], beta=[0., 0., 3.])]

    def physics(f, *, include_edges=True):
        return dict(kernel=f.modulation_kernel, s_mem=f.gates.s_mem, s_coi=f.gates.s_coi)

    with patch.object(run.fitting, "_parameters", budget.wrap("construct_fixture", fixture_parameters)), \
         patch.object(run.fitting, "_physics", budget.wrap("fixture_non_edge", physics)):
        for t in run.POINTS.values():
            point, target, actual = run.interior(parent, "modulated", endpoints, t)
            assert all(np.array_equal(a, b) for a, b in zip(target, actual, strict=True))
            assert tuple(np.count_nonzero(v) for v in actual) == (3, 2)
            assert point.modulation_kernel == parent.modulation_kernel and point.gates.s_coi == 0
            data = run.fitting._plain(dict(c=actual[0], beta=actual[1], **physics(point)))
            assert json.loads(run.evidence.json_bytes(data)) == data
        rejects(lambda: run.interior(parent, "modulated", endpoints, 0.))
        rejects(lambda: run.interior(parent, "modulated", endpoints, 1.))
        rejects(lambda: run.interior(parent, "modulated", [endpoints[0], dict(c=[0., 2., 3.], beta=[0., 0., 4.])], .5))
    point.gates.s_mod[0] = 2.
    rejects(lambda: run.effective(point, "modulated"))
    x = np.array([1., 0.])
    support = (x > 0,)
    run.mapped_support((x,), (np.nextafter(x, [2., 0.]),), support)
    for bad in ([np.nan, 0.], [1.+32*np.finfo(float).eps, 0.], [0., 0.], [1., np.finfo(float).tiny]):
        rejects(lambda bad=bad: run.mapped_support((x,), (np.array(bad),), support))


def identities(_budget):
    config = dict(model_sha256="parent", scale="frozen")
    run.same_config(dict(model_sha256="new", scale="frozen"), config, "new")
    rejects(lambda: run.same_config(dict(model_sha256="new", scale="moved"), config, "new"))
    for key in ("protocol_sha256", "design_adoption_sha256", "endpoint_digest_sha256",
                "source_sha256", "runtime"):
        release = dict(run=run.RUN, phase="prepare", released=True, action_id="mock",
                       protocol_sha256=run.PROTOCOL_SHA, design_adoption_sha256=run.ADOPTION_SHA,
                       endpoint_digest_sha256=run.ENDPOINT_SHA, source_sha256={"core": "old", "runner": "new"},
                       runtime={"python": "frozen"})
        run.release_identity("prepare", release, release["source_sha256"], release["runtime"])
        rejects(lambda: run.release_identity("prepare", {**release, key: "wrong"},
                                             release["source_sha256"], release["runtime"]))
    frozen_slots = slots()
    run.verify_slots(frozen_slots, frozen_slots)
    for changed in (frozen_slots[:-1], frozen_slots[::-1], [{**frozen_slots[0], "event_id": "other"}, *frozen_slots[1:]]):
        rejects(lambda changed=changed: run.verify_slots(frozen_slots, changed))
    run.check_masks((np.array([1., 0.]),), ([True, False],))
    rejects(lambda: run.check_masks((np.array([1., 0.]),), ([False, True],)))


def shared_candidate(budget):
    frozen_slots = slots()
    packets = [dict(slot=item["local_slot"]) for item in frozen_slots]
    functional = SimpleNamespace(n_nodes=10)
    config = dict(model_sha256="fixture", history_theta_scale="theta", history_omega_scale="omega", r_scale="r")
    scenarios = {
        "static": {"fail": False, "gate_fail": True},
        "modulated": {"fail": True, "gate_fail": False},
    }
    for mode, scenario in scenarios.items():
        calls = Counter()

        def infer(f, packet, *, mode, role, call_limit, initial, config, stepper):
            assert f is functional and mode in scenarios and role == "proxy" and call_limit == 1200
            assert config is frozen_config and stepper is run.fitting.dop853_step
            index = packet["slot"]
            start = calls[index]
            calls[index] += 1
            alt = np.tile([.2, -.2] if mode == "modulated" else [.1, -.1], 5)
            if mode == "modulated":
                alt = np.r_[alt, np.zeros(10)]
            assert np.array_equal(initial, alt if start else np.zeros_like(alt))
            status = "failed" if scenario["fail"] and index == 23 else ("budget_limited" if start else "converged")
            loss = 1. if index == 0 or start else 2.
            best = dict(u=initial, state=np.zeros(50), history_mse=loss, residual=np.ones(220),
                        dense_history=np.zeros((41, 50)))
            return dict(status=status, best=best, counts=dict(optimization_tasks=1, actual_rhs_calls=480,
                                                            actual_residual_calls=1), actual_residual_calls=1)

        def gate(f, state, packet, *, stepper, capture):
            assert f is functional and stepper is run.fitting.dop853_step and not capture
            return dict(passed=not (scenario["gate_fail"] and packet["slot"] == 1),
                        counts=dict(actual_rhs_calls=6480))

        frozen_config = config
        counts, record = run.matched._counts(), {}
        mask = np.array([True, False, True])
        with patch.object(run.fitting, "infer_endpoint", budget.wrap("mock_infer", infer)), \
             patch.object(run.fitting, "numerical_check", budget.wrap("mock_six_gates", gate)):
            run.matched._candidate(functional, packets, frozen_slots, mode, frozen_config, counts, record,
                                   edge_mask=mask, mod_mask=np.zeros(3, dtype=bool))
        run.verify_slots(frozen_slots, record["events"])
        assert set(calls.values()) == {2}
        assert counts["optimization_tasks"] == 48 and np.array_equal(record["edge_mask"], mask)
        assert record["events"][0]["selected"] == 0 and record["events"][1]["selected"] == 1
        assert counts["qualification_suites"] == (24 if mode == "static" else 0)
        assert record["status"] == ("numerical_failure" if mode == "static" else "no_output")
        assert record["events"][1]["best"]["history_mse"] == 1.
    parent = .5
    better = [dict(point=name, status="completed", objective=value) for name, value in
              zip(run.POINTS, (.4, .3, .2), strict=True)]
    assert run.choose(parent, better)["point"] == "eighth"
    assert run.choose(parent, [{**x, "objective": .4} for x in better])["point"] == "half"
    assert run.choose(parent, [{**x, "objective": parent-1e-15} for x in better])["point"] == "parent"
    assert run.choose(parent, [{**x, "status": "numerical_failure"} for x in better])["point"] == "parent"


def residual_budget(budget):
    functional = SimpleNamespace(n_nodes=10)
    packet = {"sigma": 0.}
    reached = []

    def optimizer(residual, initial, *, jac, **kwargs):
        assert kwargs["method"] == "trf" and kwargs["max_nfev"] == 80
        try:
            jac(initial)
        except run.fitting._BudgetReached:
            reached.append(True)
            raise
        raise AssertionError("Jacobian did not exhaust the finite residual budget")

    def integrate(_f, _state, _step, _duration, counts, **kwargs):
        counts["actual_rhs_calls"] += 480
        return np.zeros((41, 50))

    with patch.object(run.core, "validate_observation", budget.wrap("mock_observation", lambda p, _n: p)), \
         patch.object(run.fitting, "_endpoint", budget.wrap("mock_endpoint", lambda *_: np.zeros(50))), \
         patch.object(run.fitting, "_integrate", budget.wrap("mock_history", integrate)), \
         patch.object(run.core, "history_residual", budget.wrap("mock_residual", lambda *_: np.ones(220))), \
         patch.object(run.fitting, "least_squares", budget.wrap("mock_optimizer", optimizer)):
        result = INFER_WITH_MOCKS(functional, packet, mode="static", role="proxy", call_limit=3,
                                  initial=np.zeros(10), stepper=run.fitting.dop853_step)
    assert reached == [True] and result["status"] == "budget_limited"
    assert result["actual_residual_calls"] == 3 and result["counts"]["actual_rhs_calls"] == 1440


def witnesses(budget):
    parent = {"u": [0.] * 10, "mse": 5.}
    conf, functional, packet = {}, object(), object()

    def endpoint(p, f, u, mode, role, config):
        assert p is packet and f is functional and u is parent["u"] and config is conf
        assert (mode, role) == ("static", "proxy")
        return np.zeros(50)

    def integrate(f, state, step, duration, counts, **kwargs):
        assert f is functional and step == .0025 and duration == .1
        assert kwargs == dict(backward=True, dense=True, stepper=run.fitting.dop853_step)
        counts["actual_rhs_calls"] += 480
        return np.zeros((41, 50))

    def residual(f, history, p, config):
        assert f is functional and p is packet and config is conf and history.shape == (11, 50)
        return np.full(220, 2.)

    with patch.object(run.fit297, "endpoint", budget.wrap("witness_endpoint", endpoint)), \
         patch.object(run.fitting, "_integrate", budget.wrap("witness_history", integrate)), \
         patch.object(run.fit297, "history_residual", budget.wrap("witness_residual", residual)):
        row = run.witness(functional, packet, "static", conf, parent, 6.)
        missing = run.witness(functional, packet, "static", conf, parent, None)
        with patch.object(run.fit297, "history_residual",
                          budget.wrap("nonfinite_witness", lambda *_: np.full(220, np.nan))):
            nonfinite = run.witness(functional, packet, "static", conf, parent, 6.)
    assert row == dict(mse=4., minus_parent=-1., minus_candidate=-2., rhs_calls=480)
    assert missing["minus_candidate"] is None and nonfinite["mse"] is None
    assert run.summarize_witness([row]*24)["minus_candidate"] == dict(finite=24, mean=-2.)
    assert run.summarize_witness([row]*23+[nonfinite])["minus_parent"] == dict(finite=23, mean=None)
    assert run.choose(1., [dict(point="half", status="completed", objective=2., witness=row)])["point"] == "parent"


def io_failures(budget):
    files = {}

    def write(path, value, *, exclusive=False):
        assert exclusive
        if path in files:
            raise FileExistsError(path)
        files[path] = value
        return dict(path=path.as_posix())

    def reject_mapping(*_args):
        raise ValueError("fixture construction failure")

    cache = dict(mode="static", slots=slots(), parent_points=[None]*24, config={}, endpoints=[])
    with patch.object(run.evidence, "save_json", budget.wrap("mock_receipt", write)), \
         patch.object(run.evidence, "save", budget.wrap("mock_artifact", write)), \
         patch.object(run, "interior", budget.wrap("mock_bad_mapping", reject_mapping)):
        _, failed = run.candidate_job({}, cache, [None]*24, None, "half", .5)
        rejects(lambda: run.candidate_job({}, cache, [None]*24, None, "half", .5))
    assert failed["status"] == "interrupted" and failed["known_rhs_including_witness"] == 0
    assert failed["cost_unknown"] is False and len(files) == 3

    files.clear()
    functional = object()
    actual = np.array([1.]), np.array([0.])

    def load(ref, path):
        assert ref["path"] == path.as_posix()
        return files[path]

    def interrupted(_f, _packets, _slots, _mode, _inner, counts, record, **_masks):
        counts["optimization_tasks"] = 1
        counts["actual_residual_calls"] = 1
        counts["proxy_rhs_calls"] = counts["actual_rhs_calls"] = 480
        record["events"] = []
        raise RuntimeError("fixture interrupted after one recorded callback")

    cache.update(config=dict(fit_request={}, preparation_key="fixture"), delta_star=[], endpoints=[])
    with ExitStack() as stack:
        for module, name, callback in (
            (run.evidence, "save_json", write), (run.evidence, "save", write),
            (run, "bound_json", load),
            (run, "interior", lambda *_: (functional, actual, actual)),
            (run, "restore_physics", lambda _: functional),
            (run.fitting, "_physics", lambda _: {"mock": True}),
            (run, "effective", lambda *_: actual),
            (run.fit297, "make_config", lambda *_: {**cache["config"], "model_sha256": run.fitting._hash({"mock": True})}),
            (run.matched, "_candidate", interrupted),
        ):
            stack.enter_context(patch.object(module, name, budget.wrap("failure_" + name, callback)))
        _, failed = run.candidate_job({}, cache, [None]*24, None, "half", .5)
    assert failed["status"] == "interrupted" and failed["cost_unknown"] is True
    assert failed["known_rhs_including_witness"] == 480 and failed["known_counts"]["optimization_tasks"] == 1

    costs = {k: v//6 for k, v in run.LIMITS.items()}
    costs["actual_rhs_calls"] = costs["proxy_rhs_calls"] + costs["qualification_rhs_calls"]
    run.check_costs(costs)
    assert costs["proxy_rhs_calls"] == 57600*480 and costs["qualification_rhs_calls"] == 24*6480
    assert sum(run.LIMITS[k] for k in ("proxy_rhs_calls", "qualification_rhs_calls", "witness_rhs_calls")) == 166890240
    rejects(lambda: run.check_costs({**costs, "witness_rhs_calls": costs["witness_rhs_calls"]+1}))


GROUPS = (algebra, identities, shared_candidate, residual_budget, witnesses, io_failures)


def qualify(budget, outcomes, violations):
    def forbidden(*_args, **_kwargs):
        violations.append("forbidden_science_callback")
        raise AssertionError("unmocked science or data access")

    with ExitStack() as stack:
        for module, names in (
            (run, ("restore_physics", "legal_packets", "scientific", "prepare")),
            (run.matched, DENY),
            (run.evidence, ("load_packets", "score_stage", "deploy_stage", "final_model")),
            (run.fitting, ("_parameters", "_integrate", "least_squares", "infer_endpoint", "numerical_check",
                           "dop853_step", "backward_select")),
            (run.fit297, ("endpoint", "history_residual")),
            (run.GeneratingFunctionalKuramoto, ("rhs", "gradients", "hamiltonian")),
            (np.linalg, ("svd",)),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        for group in GROUPS:
            group(budget)
            outcomes.append(group.__name__)


def main():
    run.require(__debug__, "qualification assertions must be enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    raw = run.evidence.local_path(args.release).read_bytes()
    release = json.loads(raw)
    run.frozen_context("qualify", release)
    attempt = release["attempt"]
    previous = release["previous"]
    run.require(type(attempt) is int and 1 <= attempt <= 3 and len(previous) == attempt-1,
                "qualification attempt or predecessor count differs")
    cumulative = 0
    for i, ref in enumerate(previous, 1):
        old = run.bound_json(ref, run.RESULTS / f"qualification_{i}.json")
        run.require(old["run"] == run.RUN and old["attempt"] == i and old["status"] == "failed"
                    and old["protocol_sha256"] == run.PROTOCOL_SHA and 0 <= old["mock_callbacks"] <= 1000,
                    "previous qualification identity differs")
        cumulative += old["mock_callbacks"]
        run.require(old["cumulative_mock_callbacks"] == cumulative, "previous qualification ledger differs")
    identity = dict(run=run.RUN, attempt=attempt, action_id=release["action_id"], release_sha256=run.evidence.sha(raw),
                    source_sha256=release["source_sha256"], runtime=release["runtime"],
                    protocol_sha256=run.PROTOCOL_SHA, previous=previous)
    claim = run.claim(run.RESULTS / f"qualification_claim_{attempt}.json", identity)
    budget = MockBudget(cumulative)
    outcomes, violations = [], []
    receipt = dict(**identity, claim=claim, status="failed")
    try:
        qualify(budget, outcomes, violations)
        run.require(len(outcomes) == 6 and not violations, "qualification failed")
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(groups=outcomes, callbacks=dict(budget.calls), mock_callbacks=budget.total,
                       cumulative_mock_callbacks=budget.total+cumulative, real_science_calls=0,
                       forbidden_attempts=violations)
        run.evidence.save_json(run.RESULTS / f"qualification_{attempt}.json", receipt, exclusive=True)


if __name__ == "__main__":
    main()
