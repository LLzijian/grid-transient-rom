"""Released algebra and mock qualification, with one bounded 2x2 SciPy problem."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import run
np = run.np


class MockBudget:
    def __init__(self, previous):
        self.previous = previous
        self.calls = Counter()

    @property
    def total(self):
        return sum(self.calls.values())

    def wrap(self, name, callback):
        def invoke(*args, **kwargs):
            run.require(self.total < 1000 and self.previous+self.total < 3000, "mock callback limit reached")
            self.calls[name] += 1
            return callback(*args, **kwargs)
        return invoke


def rejects(callback):
    try:
        callback()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("expected rejection")


def algebra():
    x = np.zeros(45)
    beta = np.r_[1., np.zeros(44)]
    u = np.zeros((24, 20))
    u[0, :2] = -2., 2.
    lower, upper = run.bounds(x, u, run.K_REF)
    assert len(lower) == len(upper) == 525
    assert (lower[0], upper[0], lower[45], upper[46]) == (0., 1., 0., 0.)
    assert run.feasible(x, beta, u)
    assert not run.feasible(x, -beta, u)
    assert not run.feasible(x, beta, u+1e-5)
    xx, bb, uu = run.shifted(x, beta, u[0], beta, 65, run.H)
    assert np.array_equal(xx, x) and np.array_equal(uu, u[0]) and bb[0]+xx[0] == 1.+run.H
    y = np.array([1., 0.])
    run.prior.mapped_support((y,), (np.nextafter(y, [2., 0.]),), (y > 0,))
    for bad in ([0., 0.], [1.+32*run.EPS, 0.], [1., run.TINY], [np.nan, 0.]):
        rejects(lambda bad=bad: run.prior.mapped_support((y,), (np.asarray(bad),), (y > 0,)))
    A, b = np.zeros((5280, 525)), np.zeros(5280)
    row = np.ones(220)
    run.assemble_column(A, b, 0, 45, row)
    run.assemble_column(A, b, 4, 45, 2*row)
    run.assemble_column(A, b, 4, 3, 3*row)
    run.assemble_column(A, b, 4, 65, 4*row)
    assert A[0, 45] == 1 and A[880, 125] == 2 and A[880, 3] == 3 and b[880] == 4
    assert A[0, 125] == A[880, 45] == b[0] == 0


def stencils(budget):
    r = budget.wrap("polynomial", lambda t: np.full(220, 1.+2*t+3*t*t))
    for can, sign in ((lambda _: True, 0), (lambda x: x >= 0, 1), (lambda x: x <= 0, -1)):
        assert run.template(can) == sign
        for step in (run.H, run.H/2):
            t = (step, -step) if sign == 0 else (sign*step, sign*2*step)
            column, indicator = run.difference(r(0), r(t[0]), r(t[1]), step, sign)
            assert np.allclose(column, 2, atol=1e-9, rtol=0) and indicator > 0
    rejects(lambda: run.template(lambda _: False))
    ones = np.ones(220)
    assert run.column_gate((ones, 0), (ones, 0))["passed"]
    assert not run.column_gate((ones, 0), (3*ones, 0))["passed"]
    assert not run.column_gate((ones, .01), (ones, .01))["passed"]
    assert not run.column_gate((ones, 0), (np.full(220, np.nan), 0))["passed"]
    for huge in (1e200, np.finfo(float).max):
        # Both residual columns are finite. The norm or their difference is not.
        first, second = np.full(220, huge), np.full(220, -huge)
        assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
        verdict = run.column_gate((first, 0), (second, 0))
        assert not verdict["finite"] and not verdict["passed"]
    assert run.check_reference(np.array([ones]), [([ones, ones], 0.)])[0]["passed"]
    assert not run.check_reference(np.array([ones]), [([ones+1e-9, ones], 0.)])[0]["passed"]


def maps(budget):
    ledger = {"reference": [dict(maps=23, rhs=23*cap, cost_unknown=False) for cap in (480, 960)]}
    calls = []

    def mapped(_f, _c, _p, _u, step):
        calls.append(step)
        return (np.zeros(220), 480) if step == .0025 else (None, 700)

    calculator = run.Maps(ledger, budget.wrap("history_pair", mapped))
    _, gap = calculator.pair(None, None, None, np.zeros(20), "reference")
    assert calls == [.0025, .00125] and gap is None
    assert ledger["reference"][1] == dict(maps=24, rhs=23*960+700, cost_unknown=False)
    rejects(lambda: calculator.pair(None, None, None, np.zeros(20), "reference"))
    ledger["reference"][0] = dict(maps=0, rhs=0, cost_unknown=False)
    calculator.callback = budget.wrap("invalid_rhs", lambda *_: (None, 481))
    rejects(lambda: calculator.pair(None, None, None, np.zeros(20), "reference"))
    assert ledger["reference"][0]["cost_unknown"]


def solver(budget, modules, technical):
    lower, upper = np.zeros(2), np.ones(2)
    A, b = np.eye(2), np.array([-2., -.25])
    technical["problems"] = 0
    technical["lsmr"] = []
    v = run.solve(A, b, lower, upper, modules, technical, outer=10, inner=20)
    assert run.feasible_vector(v, lower, upper) and np.allclose(v, [1., .25], atol=1e-7, rtol=0)
    assert {row["alias"] for row in technical["lsmr"]} == {"initial", "trf"}
    assert all(row["itn"] <= 20 for row in technical["lsmr"])
    rejects(lambda: run.solve(A, b, lower, upper, modules, technical, outer=10, inner=20))

    def lsmr(*_args, **_kwargs):
        return np.zeros(2), 7, 20, 1., 1., 1., 1., 1.

    first = SimpleNamespace(lsmr=budget.wrap("mock_initial_lsmr", lsmr))
    trf = SimpleNamespace(lsmr=budget.wrap("mock_trf_lsmr", lsmr))

    def mock_solver(_A, _b, **kwargs):
        assert kwargs["method"] == "trf" and kwargs["lsq_solver"] == "lsmr"
        first.lsmr(maxiter=20)
        for _ in range(10):
            trf.lsmr(maxiter=20)
        trf.lsmr(maxiter=20)
        raise AssertionError("twelfth internal solve accepted")

    first.lsq_linear = budget.wrap("mock_bounded", mock_solver)
    guard = dict(problems=0, lsmr=[])
    rejects(lambda: run.solve(A, b, lower, upper, (first, trf), guard, outer=10, inner=20))
    assert len(guard["lsmr"]) == 11
    first.lsq_linear = budget.wrap("mock_at_limit", lambda *_args, **_kw: SimpleNamespace(
        x=np.array([.5, .25]), nit=10, status=0, message="limit", cost=1., optimality=.2))
    guard = dict(problems=0, lsmr=[])
    assert run.solve(A, b, lower, upper, (first, trf), guard, outer=10, inner=20) is not None
    assert guard["result"]["status"] == 0 and guard["result"]["feasible"]
    first.lsq_linear = budget.wrap("mock_infeasible", lambda *_args, **_kw: SimpleNamespace(
        x=np.array([1.01, .25]), nit=1, status=-1, message="failed", cost=1., optimality=.2))
    assert run.solve(A, b, lower, upper, (first, trf), dict(problems=0, lsmr=[]), outer=10, inner=20) is None


def verification(budget):
    reference = np.zeros((2, 24, 220))
    control = np.full_like(reference, 2e-4)
    compensation = np.full_like(reference, 2e-8)
    for alpha in run.ALPHAS:
        prediction = np.full(5280, 2e-4/alpha/np.sqrt(5280))
        result = run.verification_metrics(reference, control, compensation, alpha, prediction, np.zeros(5280))
        assert result["passed"] and np.isclose(result["B"][1], 2e-4)
        low = run.verification_metrics(reference, reference, reference, alpha, np.zeros(5280), np.zeros(5280))
        assert not low["passed"] and not low["gates"]["signal"]
        shifted = compensation.copy()
        shifted[0, 0] += 1e-5
        assert not run.verification_metrics(reference, control, shifted, alpha, prediction, np.zeros(5280))["gates"]["mesh"]
        assert not run.verification_metrics(reference, control, compensation, alpha, -prediction, np.zeros(5280))["gates"]["linear"]

    for huge in (1e200, np.finfo(float).max):
        high = np.full_like(reference, huge)
        low = -high if huge > 1e300 else reference
        rejected = run.verification_metrics(low, high, low, run.ALPHAS[0], np.zeros(5280), np.zeros(5280))
        assert np.all(np.isfinite(high)) and np.all(np.isfinite(low))
        assert not rejected["passed"] and not rejected["gates"]["finite"]
        assert rejected["D"] == [0., 0.] and not rejected["linear"][0]["passed"]
    huge_prediction = np.full(5280, 1e200)
    rejected = run.verification_metrics(reference, control, compensation, run.ALPHAS[0],
                                        huge_prediction, np.zeros(5280))
    assert not rejected["passed"] and not rejected["linear"][0]["finite"]

    vector = np.r_[np.zeros(45), np.tile(np.r_[1., np.zeros(19)], 24)]
    x, beta, u = np.ones(45), np.zeros(45), np.zeros((24, 20))
    visited = []
    def pair(_f, _c, packet, endpoint, phase):
        assert phase == "verification"
        visited.append((packet, endpoint[0]))
        return [np.zeros(220)]*2, 0.

    private, public = dict(verification=[]), dict(amplitudes=[])
    with patch.object(run, "mapped", budget.wrap("verification_mapping", lambda *_: (None, None, {}))):
        run.verify_amplitudes(None, None, list(range(24)), x, beta, u, np.ones(45), reference,
                              SimpleNamespace(pair=budget.wrap("verification_pair", pair)),
                              np.zeros((5280, 525)), np.zeros(5280), vector, private, public)
    assert visited == [(j, amount) for alpha in run.ALPHAS for amount in (0., alpha) for j in range(24)]
    assert len(private["verification"]) == len(public["amplitudes"]) == 2


def identity_and_io(budget):
    slots = [dict(local_slot=i, global_index=j, event_id=str(j)) for i, j in enumerate(run.prior.SLOTS)]
    run.prior.verify_slots(slots, slots)
    rejects(lambda: run.prior.verify_slots(slots, slots[::-1]))
    path = run.prior.PRIVATE/"prepare/packets.json"
    ref = dict(path=path.as_posix(), sha256=run.CACHE_SHAS["packets"])
    assert run.cache_ref(ref, path, "packets") is ref
    rejects(lambda: run.cache_ref({**ref, "sha256": "changed"}, path, "packets"))
    files = {}

    def save(path, value, *, exclusive=False):
        assert exclusive
        if path in files:
            raise FileExistsError(path)
        files[path] = value
        return dict(path=path.as_posix())

    def fail_binding(*_args):
        raise ValueError("fixture identity differs")

    with patch.object(run.io, "save_json", budget.wrap("receipt", save)), \
         patch.object(run, "parent_bindings", budget.wrap("bad_binding", fail_binding)):
        rejects(lambda: run.prepare({"runtime": {}}, {}))
        rejects(lambda: run.prepare({"runtime": {}}, {}))
    assert files[run.RESULTS/"prepare.json"]["status"] == "interrupted"
    assert files[run.RESULTS/"prepare.json"]["private_cache_opened"] is False


def qualify(budget, modules, technical, groups, violations):
    def forbidden(*_args, **_kwargs):
        violations.append("forbidden_science_or_data")
        raise AssertionError("unmocked science or data access")

    with ExitStack() as stack:
        for module, names in (
            (run, ("history", "mapped", "science", "bound_json", "write_private")),
            (run.prior, ("legal_packets", "restore_physics", "candidate_job")),
            (run.io, ("load_packets", "score_stage", "deploy_stage", "final_model")),
            (run.prior.matched, ("fit_shared", "_prepare", "_audit", "_diagnose", "_candidate")),
            (run.engine, ("_parameters", "_integrate", "dop853_step", "infer_endpoint", "least_squares",
                          "numerical_check", "backward_select", "initialize_complete")),
            (run.fit, ("endpoint", "history_residual", "initialize_complete")),
            (run.prior.GeneratingFunctionalKuramoto, ("rhs", "gradients", "hamiltonian")),
            (np.linalg, ("svd", "lstsq")),
        ):
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden))
        for name, group in (("algebra", algebra), ("stencils", lambda: stencils(budget)),
                            ("maps", lambda: maps(budget)), ("solver", lambda: solver(budget, modules, technical)),
                            ("verification", lambda: verification(budget)),
                            ("identity_and_io", lambda: identity_and_io(budget))):
            group()
            groups.append(name)


def main():
    run.require(__debug__, "qualification assertions must be enabled")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    raw = run.io.local_path(args.release).read_bytes()
    release = json.loads(raw)
    _, modules = run.frozen_context("qualify", release)
    attempt, previous = release["attempt"], release["previous"]
    run.require(type(attempt) is int and 1 <= attempt <= 3 and len(previous) == attempt-1,
                "qualification attempt differs")
    mock_total = problems = calls = iterations = 0
    for index, ref in enumerate(previous, 1):
        old = run.bound_json(ref, run.RESULTS/f"qualification_{index}.json")
        run.require(old["run"] == run.RUN and old["attempt"] == index and old["status"] == "failed"
                    and old["protocol_sha256"] == run.PROTOCOL and old["real_science_calls"] == 0
                    and not old["cost_unknown"] and 0 <= old["mock_callbacks"] <= 1000
                    and 0 <= old["linear_problems"] <= 1 and 0 <= old["lsmr_calls"] <= 11
                    and 0 <= old["lsmr_iterations"] <= 220, "previous qualification ledger differs")
        mock_total += old["mock_callbacks"]
        problems += old["linear_problems"]
        calls += old["lsmr_calls"]
        iterations += old["lsmr_iterations"]
    run.require(mock_total+1000 <= 3000 and problems+1 <= 3 and calls+11 <= 33
                and iterations+220 <= 660, "qualification balance exhausted")
    identity = {key: release[key] for key in ("run", "phase", "action_id", "protocol_sha256",
                                             "design_adoption_sha256", "source_sha256", "runtime",
                                             "scipy_source_sha256", "attempt", "previous")}
    identity["release_sha256"] = run.io.sha(raw)
    claim = run.prior.claim(run.RESULTS/f"qualification_claim_{attempt}.json", identity)
    budget = MockBudget(mock_total)
    technical, groups, violations = dict(problems=0, lsmr=[]), [], []
    receipt = dict(**identity, claim=claim, status="failed")
    try:
        qualify(budget, modules, technical, groups, violations)
        run.require(len(groups) == 6 and not violations, "qualification failed")
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt.update(groups=groups, mock_callbacks=budget.total, callbacks=dict(budget.calls),
                       cumulative_mock_callbacks=mock_total+budget.total, real_science_calls=0,
                       forbidden_attempts=violations, technical_solver=technical,
                       linear_problems=technical["problems"], lsmr_calls=len(technical["lsmr"]),
                       lsmr_iterations=sum(x.get("itn", 0) for x in technical["lsmr"]),
                       cost_unknown=any(x["status"] != "returned" for x in technical["lsmr"]))
        run.io.save_json(run.RESULTS/f"qualification_{attempt}.json",
                         run.prior.matched._json_value(receipt), exclusive=True)


if __name__ == "__main__":
    main()
