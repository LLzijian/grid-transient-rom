import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.gftd_long_horizon_diagnostic_20261109 import run, zero_skip
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates


def functional(modulation, stiffness):
    return GeneratingFunctionalKuramoto(
        mass=np.array([1., 1.3, 1.7]), damping=np.full(3, .2), k_edge=np.asarray(stiffness), eta_edge=np.zeros(3),
        gamma_amp=np.ones(3), lambda_amp=np.ones(3), lambda_mem=np.ones(3), tau_amp=np.full(3, 5.),
        tau_mem_sync=np.ones(3), tau_mem_coi=1., j_omega_mem=np.ones(3),
        gates=StructuralGates(s_edge=np.ones(3), s_mod=np.asarray(modulation), s_mem=np.ones(3)),
    )


def test_four_artificial_equivalence_fixtures(monkeypatch):
    normal = np.array([.2, -.3, .1, .02, -.01, .03, .04, .05, .06, .15, .25, 0., -.02, .1, -.3])
    inactive_overflow = normal.copy()
    inactive_overflow[11] = 800.
    fixtures = {
        "all_closed": (functional([0., 0., 0.], [.5, .6, .7]), normal),
        "partially_closed": (functional([.5, 0., 1.], [.5, .6, 0.]), normal),
        "inactive_overflow": (functional([.5, 0., 1.], [.5, .6, 0.]), inactive_overflow),
        "normal_active": (functional([.5, .75, 1.], [.5, .6, np.finfo(np.float64).tiny / 4]), normal),
    }
    receipts, results, budget = {}, {}, [0]
    original_exp = np.exp
    exp_contexts = []

    def audited_exp(*args, **kwargs):
        exp_contexts.append((np.geterr()["over"], np.geterr()["under"]))
        return original_exp(*args, **kwargs)

    monkeypatch.setattr(zero_skip.np, "exp", audited_exp)
    for name, (original, state) in fixtures.items():
        candidate = zero_skip.InactiveZeroSkip.from_functional(original)
        before = budget[0]
        record, arrays = zero_skip.point_comparison(original, candidate, state, budget)
        record["actual_rhs_calls"] = budget[0] - before
        receipts[name], results[name] = record, arrays

    assert budget[0] == 8 and all(record["actual_rhs_calls"] == 2 for record in receipts.values())
    assert all(record["passed"] for record in receipts.values())
    assert receipts["inactive_overflow"]["status"] == "zero_term_unblocked"
    assert set(receipts["inactive_overflow"]["original_errors"]) == {"couplings", "hamiltonian", "gradients", "rhs"}
    assert receipts["inactive_overflow"]["labels"] == ["inactive_zero_exp_out_of_range"]
    assert exp_contexts and all(over == under == "raise" for over, under in exp_contexts)
    assert all(record["status"] == "equivalent" for name, record in receipts.items() if name != "inactive_overflow")

    np.testing.assert_array_equal(results["all_closed"]["candidate_couplings"], [.5, .6, .7])
    np.testing.assert_array_equal(results["all_closed"]["candidate_gradients"][3], normal[9:12])
    base, large = results["partially_closed"], results["inactive_overflow"]
    np.testing.assert_array_equal(base["candidate_couplings"], large["candidate_couplings"])
    assert large["candidate_couplings"][1] == .6 and large["candidate_couplings"][2] == 0.
    assert large["candidate_hamiltonian"] == pytest.approx(base["candidate_hamiltonian"] + .5 * 800.**2)
    expected_gradient, expected_rhs = base["candidate_gradients"].copy(), base["candidate_rhs"].copy()
    expected_gradient[3, 2], expected_rhs[11] = 800., -800. / 5
    np.testing.assert_array_equal(large["candidate_gradients"], expected_gradient)
    np.testing.assert_array_equal(large["candidate_rhs"], expected_rhs)
    active = zero_skip.InactiveZeroSkip.from_functional(fixtures["normal_active"][0])
    assert active._modulated.all()
    for exponent, kind in ((800., "overflow"), (-720., "underflow")):
        active_extreme = normal.copy()
        active_extreme[9] = exponent
        for operation in (active.hamiltonian, active.gradients):
            with pytest.raises(FloatingPointError, match=f"{kind} encountered in exp"):
                operation(active_extreme)

    print(json.dumps(dict(rhs_version=zero_skip.VERSION, actual_rhs_calls=budget[0], fixtures=receipts,
                          active_range_guard_algebra_calls=4), sort_keys=True))


def test_branch_worker_restores_original_paths_after_exception(monkeypatch):
    original = run.OUTPUT, run.PRIVATE, run.RHS_VERSION

    def interrupted():
        assert run.OUTPUT == original[0] / zero_skip.VERSION
        assert run.PRIVATE == original[1] / zero_skip.VERSION
        assert run.RHS_VERSION == zero_skip.VERSION
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(run, "run_case_jobs", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        zero_skip.branch_worker()
    assert (run.OUTPUT, run.PRIVATE, run.RHS_VERSION) == original


def test_conditional_integrations_stay_within_registered_caps():
    assert len(zero_skip.BRANCH_TARGETS) * (len(run.FIXED_STEPS) + len(run.REFERENCES)) == 28
    assert sum(zero_skip.BRANCH_TARGETS.values()) * 240 * len(run.REFERENCES) == 1614240
    assert sum(12 * samples * round(.01 / step) for samples in zero_skip.BRANCH_TARGETS.values()
               for step in run.FIXED_STEPS.values()) == 403560
