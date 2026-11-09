from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.gftd_initialization_precision_20261109 import fixed_state
from kiki.physics.kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates


def artificial_functional():
    n = 2
    return GeneratingFunctionalKuramoto(
        mass=np.ones(n), damping=np.full(n, .2), k_edge=np.array([.5]), eta_edge=np.zeros(1),
        gamma_amp=np.full(n, .5), lambda_amp=np.ones(n), lambda_mem=np.ones(n), tau_amp=np.full(n, 5.),
        tau_mem_sync=np.ones(n), tau_mem_coi=1., j_omega_mem=np.ones(n), gates=StructuralGates.overcomplete(1, n),
    )


def test_fixed_state_uses_all_original_gates_and_twelve_rhs_per_step(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the fixed-state pilot must never fall back to RK4")

    monkeypatch.setattr(fixed_state.diagnostic.core, "rk4", forbidden)
    f = artificial_functional()
    state = np.array([.1, -.1, .01, -.01, .02, -.02, .01, -.01, 0., 0.])
    result, arrays = fixed_state.qualify(f, state)
    assert result["gates"]["passed"]
    assert set(result["gates"]["values"]) == fixed_state.diagnostic.GATE_LIMITS.keys()
    assert result["gates"]["values"]["p_endpoint_error"] == 0.
    assert result["costs"]["integrations_completed"] == 6
    assert result["costs"]["actual_rhs_calls"] == 12 * (40 + 80 + 40 + 80 + 100 + 200)
    for name, step in zip(result["integrals"], (-.0025, -.00125, .0025, .00125, .01, .005), strict=True):
        receipt = result["integrals"][name]
        assert receipt["step"] == step and receipt["interrupted_step_rhs_calls"] == 0
        assert arrays[f"{name}_path"].shape[1] == len(state)
        if not name.startswith("return"):
            np.testing.assert_array_equal(arrays[f"{name}_path"][0], state)


def test_interrupted_stage_keeps_exact_cost_and_no_fabricated_endpoint(monkeypatch):
    f = artificial_functional()
    calls = 0
    rhs = f.rhs

    def fail_on_seventeenth(state):
        nonlocal calls
        calls += 1
        if calls == 17:
            raise FloatingPointError("forced stage failure")
        return rhs(state)

    monkeypatch.setattr(f, "rhs", fail_on_seventeenth)
    state = np.ones(10) * .01
    result = fixed_state.integrate(f, state, 1., .01)
    assert not result.record["completed"]
    assert result.record["accepted_steps"] == 1 and result.record["attempted_steps"] == 2
    assert result.record["actual_rhs_calls"] == 17 and result.record["interrupted_step_rhs_calls"] == 5
    assert result.record["failure"]["substage"] == 5
    assert result.record["last_accepted_time"] == .01
    assert np.isnan(result.path[2:]).all()
    assert result.failure_state.shape == state.shape


def test_pilot_requires_a_recovered_failure_and_keeps_original_controls():
    control = dict(state="control", role="selected", old_passed=True, dp8_passed=True)
    stage1 = dict(state="stage1", role="stage1", old_passed=False, dp8_passed=True)
    improved = dict(state="improved", role="selected", old_passed=False, dp8_passed=True)
    assert not fixed_state.decision([control, stage1])["eligible_for_initializer_pilot"]
    passed = fixed_state.decision([control, stage1, improved])
    assert passed["eligible_for_initializer_pilot"]
    assert passed["newly_qualified"] == ["improved"]
    assert not fixed_state.decision([dict(control, dp8_passed=False), improved])["eligible_for_initializer_pilot"]


def test_release_requires_completed_diagnostic_with_verified_trigger(monkeypatch):
    diagnostic = fixed_state.diagnostic
    selected = [dict(mode="static", index=i, role="selected", state_sha256="frozen",
                     original_state_gates={"passed": False}) for i in range(49)]
    cohort = {"selection": selected}
    parent = dict(action_id="diagnostic", selection_sha256="selection")
    execution = dict(status="COMPLETED", phase="diagnostic", action_id="diagnostic", release_sha256="parent",
                     selected_count=49, completed=[])
    for item in selected:
        name = fixed_state.state_name(item)
        execution["completed"].append(dict(state=name, evidence=dict(
            path=(diagnostic.OUTPUT / "states" / f"{name}.json").as_posix(), sha256=name)))
    trigger = dict(selected[0], future={"discretization_evidence": True})
    records = {"parent": parent, "selection": cohort, "execution": execution, "adoption": {},
               "static_000_selected": trigger, "source": {}}
    monkeypatch.setattr(diagnostic, "read_verified", lambda path, expected: diagnostic.json_bytes(records[expected]))
    monkeypatch.setattr(diagnostic, "verify_release", lambda parent, phase: None)
    release = dict(released=True, phase="fixed_state", action_id="pilot", diagnostic_release_sha256="parent",
                   diagnostic_execution_sha256="execution", selection_sha256="selection", design_adoption_sha256="adoption",
                   trigger={"state": "static_000_selected", "leg": "future"},
                   source_sha256=dict.fromkeys(fixed_state.SOURCE_PATHS, "source"))
    assert fixed_state.verify(release) == (parent, diagnostic.json_bytes(cohort))
    trigger["future"]["discretization_evidence"] = False
    with pytest.raises(ValueError, match="old failure"):
        fixed_state.verify(release)
    execution["status"] = "RUNNING"
    with pytest.raises(ValueError, match="complete frozen"):
        fixed_state.verify(release)


def test_fixed_state_claim_prevents_reopening_completed_action(tmp_path, monkeypatch):
    diagnostic = fixed_state.diagnostic
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "recover", lambda parent: ({}, [], {}))
    monkeypatch.setattr(fixed_state, "verify", lambda release: (None, diagnostic.json_bytes({"selection": [], "rows": []})))
    release_path = tmp_path / "release.json"
    release_path.write_bytes(diagnostic.json_bytes(dict(action_id="pilot", selection_sha256="fixture")))
    assert fixed_state.run(release_path)["status"] == "COMPLETED"
    receipt = tmp_path / fixed_state.OUTPUT / "execution.json"
    before = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        fixed_state.run(release_path)
    assert receipt.read_bytes() == before
