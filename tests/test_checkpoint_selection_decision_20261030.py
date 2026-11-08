"""Synthetic frozen-rule tests: no private events, model inference or optimizer."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import checkpoint_selection_decision_20261030 as d


def fixture():
    summaries, details = {}, {}
    for label in d.LABELS:
        reference = label in ("R_N", "A96")
        p = {"qualified": True, "horizon_risks": {"5": {"value": .83 if label == "R_N" else .75}, "8": {"value": .59 if label == "R_N" else .55}},
             "common_frequency_rmse_8s": .3, "normalized_p_rmse_8s": .4,
             "unstable_actual15_risk": 3.5 if label == "R_N" else 2.4 if label == "A96" else 2.2,
             "unstable_actual15_numerical_score_perturbation": .0001,
             "classification_8s": {"missed_resolved_unstable": 3 if reference else 2, "false_los_resolved_stable": 0},
             "classification_actual15s": {"missed_resolved_unstable": 21 if reference else 19, "false_los_resolved_stable": 0}, "timing_actual15": {}}
        detail = {"classification": {scope: {"rows": []} for scope in ("8s", "actual15")}}
        for i in range(297):
            eid = f"private-event-{i:03d}"
            unstable = i < 30
            for scope in ("8s", "actual15"):
                count = (27 if reference else 28) if scope == "8s" else (9 if reference else 11)
                detail["classification"][scope]["rows"].append({"event_id": eid, "label": "unstable" if unstable else "stable", "numerically_admissible": True, "predicted_los": unstable and i < count})
            p["timing_actual15"][eid] = {"numerically_qualified": True, "truth": {"time_s": 1. if i < 2 else None}, "fine": {"time_s": 1.1 if i < 2 else None}, "coarse": {"time_s": 1.11 if i < 2 else None}}
        summaries[label] = {"decision_profile": p}
        details[label] = detail
    return summaries, details


def run(summaries, details, target_fn=None):
    return d.decide(summaries, details, valid_fn=lambda p: p["qualified"], target_fn=target_fn or (lambda *a: {"passed": True}))


def test_full_frozen_rule_selects_earliest_and_never_adopts():
    s, e = fixture()
    result = run(s, e)
    assert result["suggested_adaptation_source"] == "step000000"
    assert result["pareto_front"] == list(d.CANDIDATES)
    assert result["T_anchor"]["count"] == 2
    assert result["model_adopted"] is False and result["future_training_executed"] is False
    assert result["candidates"]["step000016"]["source_optimization_health"]["status"] == "insufficient_history"
    assert result["candidates"]["step000096"]["source_optimization_health"]["status"] == "unknown"
    assert result["candidates"]["step000096"]["current_optimizer_health"] == "not_applicable"


def test_same_counts_cannot_swap_error_identity():
    s, e = fixture()
    rows = e["step000016"]["classification"]["actual15"]["rows"]
    rows[0]["predicted_los"] = False
    rows[20]["predicted_los"] = True
    result = run(s, e)["candidates"]["step000016"]
    assert result["gates"]["actual15_misses"]
    assert not result["gates"]["protected_detection_actual15"]
    assert result["identity_changes"]["actual15"]["A96"]["new_missed_ids"] == ["private-event-000"]
    assert "private-event-020" in result["identity_changes"]["actual15"]["A96"]["repaired_missed_ids"]


def test_union_protects_each_reference_detection():
    s, e = fixture()
    e["R_N"]["classification"]["actual15"]["rows"][15]["predicted_los"] = True
    result = run(s, e)
    assert "private-event-015" in result["protected_detection_sets"]["actual15"]["event_ids"]
    assert not result["candidates"]["step000016"]["gates"]["protected_detection_actual15"]


def test_T_cannot_shrink_to_easy_candidate_intersection():
    s, e = fixture()
    s["step000016"]["decision_profile"]["timing_actual15"]["private-event-001"]["fine"]["time_s"] = None
    result = run(s, e)
    row = result["candidates"]["step000016"]
    assert result["T_anchor"]["count"] == 2
    assert len(row["timing"]["common_event_ids"]) == 1
    assert not row["gates"]["T_anchor_coverage"] and not row["gates"]["T_E_vs_R_N"]
    assert row["timing"]["fixed_anchor_metrics"]["E"] is None


def test_empty_anchor_is_unresolved_and_no_fallback_argmin():
    s, e = fixture()
    for ref in ("A96", "R_N"):
        for row in s[ref]["decision_profile"]["timing_actual15"].values():
            row["truth"]["time_s"] = None
    result = run(s, e)
    assert result["T_anchor"]["count"] == 0
    assert result["suggested_adaptation_source"] is None
    assert result["scientific_status"] == "unresolved"
    assert all(not row["gates"]["T_anchor_nonempty_and_refs_valid"] for row in result["candidates"].values())


def test_reference_truth_mismatch_invalidates_all_candidates():
    s, e = fixture()
    s["R_N"]["decision_profile"]["timing_actual15"]["private-event-000"]["truth"]["time_s"] = 1.01
    result = run(s, e)
    assert result["T_anchor"]["reference_truth_mismatch_ids"] == ["private-event-000"]
    assert result["suggested_adaptation_source"] is None


def test_candidate_truth_mismatch_fails_its_own_gate():
    s, e = fixture()
    s["step000016"]["decision_profile"]["timing_actual15"]["private-event-000"]["truth"]["time_s"] = 1.01
    result = run(s, e)
    assert not result["candidates"]["step000016"]["gates"]["T_truth_equal"]
    assert not result["candidates"]["step000016"]["gates"]["T_E_vs_A96"]


def test_missing_any_candidate_or_reference_refuses_before_target():
    s, e = fixture()
    del s["step000128"]
    with pytest.raises(ValueError, match="all eleven"):
        run(s, e, target_fn=lambda *a: pytest.fail("must not select before all results"))


def test_pareto_uses_each_original_component_then_earliest_step():
    vectors = {"step000000": [3, 3, 3, 3, 3, 3, 3, 3], "step000016": [1, 2, 2, 2, 2, 2, 2, 2],
               "step000032": [2, 1, 2, 2, 2, 2, 2, 2], "step000048": [1, 2, 2, 2, 2, 2, 2, 2]}
    front, selected = d.pareto_earliest(vectors)
    assert front == ["step000016", "step000032", "step000048"]
    assert selected == "step000016"
    assert d.pareto_earliest({}) == ([], None)


def test_original_efficiency_target_is_called_for_every_candidate(monkeypatch):
    import state_gated_memory_study_20261028 as sgm
    from memory_transport_study_20261029 import efficiency_target
    monkeypatch.setattr(sgm, "valid", lambda p: p["qualified"])
    calls = []
    def original(*args):
        calls.append(args[0])
        return efficiency_target(*args)
    s, e = fixture()
    result = run(s, e, target_fn=original)
    assert len(calls) == 9
    assert result["candidates"]["step000016"]["gates"]["original_efficiency_target"]


def test_timing_error_gate_and_numerical_failure_remain_hard():
    s, e = fixture()
    p = s["step000016"]["decision_profile"]
    for eid in ("private-event-000", "private-event-001"):
        p["timing_actual15"][eid]["fine"]["time_s"] = 5.
        p["timing_actual15"][eid]["coarse"]["time_s"] = 5.
    s["step000032"]["decision_profile"]["qualified"] = False
    result = run(s, e)
    assert not result["candidates"]["step000016"]["gates"]["T_E_vs_R_N"]
    assert not result["candidates"]["step000032"]["gates"]["numerical_qualification"]


def test_public_output_has_no_event_identity():
    s, e = fixture()
    result = run(s, e)
    public = d.public_copy(result)
    assert "private-event-" not in json.dumps(public)
    assert public["T_anchor"]["count"] == 2
    assert public["private_decision_sha256"] == d.digest(result)


def test_frozen_anchors_are_exactly_reconstructed_and_cannot_be_edited():
    s, e = fixture()
    original = run(s, e)
    anchors = {"detection_union": {scope: cell["event_ids"] for scope, cell in original["protected_detection_sets"].items()},
               "T_anchor": original["T_anchor"]["event_ids"], "truth_times": {eid: 1. for eid in original["T_anchor"]["event_ids"]}, "extra_field": "allowed"}
    result = d.decide(s, e, anchors=anchors, valid_fn=lambda p: p["qualified"], target_fn=lambda *a: {"passed": True})
    assert result["T_anchor"] == original["T_anchor"]
    anchors["T_anchor"] = anchors["T_anchor"][:1]
    with pytest.raises(ValueError, match="frozen anchors"):
        d.decide(s, e, anchors=anchors, valid_fn=lambda p: p["qualified"], target_fn=lambda *a: {"passed": True})
