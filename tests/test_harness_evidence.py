"""Scientific-claim boundary tests for the offline record assessor.

Fixtures are synthetic declarations, not research evidence. These tests exercise
missing, contradictory and malformed evidence as well as qualified record paths.
"""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("harness_evidence_for_test", ROOT / "harness_runtime/evidence.py")
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)
REF = "harness/examples/synthetic_evidence.md"


def fixture(name):
    return json.loads((ROOT / "harness/examples" / f"run_{name}.json").read_text(encoding="utf-8"))


def minimal():
    return {"schema_version": 2, "run_id": "unit_fixture", "purpose": "probe", "scope": "one registered local probe"}


def promotion_record():
    record = fixture("performance_equivalence")
    inf = record["inference"]
    for key in ("critical_events_reference", "verification_reference", "acceptance_reference", "independent_review_reference"):
        inf[key] = REF
    inf["independent_review"] = True
    inf["review_record"] = {
        "reference": REF, "scope": record["scope"], "reviewer_id": "reviewer_B",
        "evidence_author_ids": ["author_A"], "reviewed_references": [REF],
        "verdict": "supports", "supports": ["promote_incumbent"],
    }
    record["proposal"]["model_decision"] = "promote_incumbent"
    return record


class EvidenceTests(unittest.TestCase):
    def assert_denied(self, record, claim="no_practical_gain_within_scope"):
        result = evidence.assess_record(record)
        self.assertNotIn(claim, result["allowed_claims"], result)
        return result

    def test_short_observation_is_valid_but_missing_evidence_is_unknown(self):
        record = minimal()
        result = evidence.assess_record(record)
        self.assertTrue(result["schema_valid"])
        self.assertEqual(result["valid"], result["schema_valid"])
        self.assertEqual(set(result["normalized_evidence"].values()), {"unknown"})
        self.assertEqual(set(result["allowed_claims"]), {"observation_only", "unresolved"})
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["confirmation_need"], "diagnose_first")
        self.assertIsNone(result["actual_use_eligible"])

    def test_malformed_identity_or_enumeration_is_schema_error(self):
        records = [None, [], {}, {**minimal(), "schema_version": True}, {**minimal(), "scope": ""},
                   {**minimal(), "purpose": "TRAIN_FOREVER"}, {**minimal(), "evidence": []}]
        for key in evidence.EVIDENCE_ENUMS:
            records.append({**minimal(), "evidence": {key: "unrecognized"}})
        records.extend([{**minimal(), "evidence": {"validity": []}},
                        {**minimal(), "proposal": {"science_decision": "proved_physics"}}])
        for record in records:
            with self.subTest(record=record):
                result = evidence.assess_record(record)
                self.assertFalse(result["schema_valid"])
                self.assertEqual(result["allowed_claims"], [])
                self.assertFalse(result["promotion_eligible"])
                self.assertEqual(result["confirmation_need"], "diagnose_first")

    def test_legacy_health_never_implies_adequacy_or_null(self):
        for flag, health in ((False, "healthy"), (True, "unknown"), (None, "unknown")):
            result = evidence.normalize_legacy_optimization({"optimization_unresolved": flag, "status": "NOT_RETAINED", "lambda": 0})
            self.assertTrue(result["schema_valid"])
            self.assertEqual(result["optimizer_health"], health)
            self.assertEqual(result["optimization_adequacy"], "unknown")
            self.assertEqual(result["science_decision"], "unresolved")
            self.assertEqual(result["legacy_outcomes"]["status"], "NOT_RETAINED")

    def test_legacy_falsey_values_are_not_booleans(self):
        for value in (0, 1, "false", [], {}):
            with self.subTest(value=value):
                result = evidence.normalize_legacy_optimization({"optimization_unresolved": value})
                self.assertFalse(result["schema_valid"])
                self.assertEqual(result["optimizer_health"], "unknown")
        self.assertFalse(evidence.normalize_legacy_optimization(None)["schema_valid"])

    def test_single_negative_is_retained_without_null(self):
        result = evidence.assess_record(fixture("single_negative"))
        self.assertTrue(result["schema_valid"])
        self.assertTrue(result["proposed_claim_eligible"])
        self.assertNotIn("no_practical_gain_within_scope", result["allowed_claims"])
        self.assertNotIn("mechanism_null_within_scope", result["allowed_science_decisions"])
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["recommended_next"], "adapt")

    def test_one_matched_gain_is_budget_specific_without_adequate_optimization(self):
        result = evidence.assess_record(fixture("single_matched_gain"))
        self.assertTrue(result["proposed_claim_eligible"])
        self.assertEqual(result["science_decision_basis"]["supported_within_scope"], ["budget_specific_gain"])
        self.assertNotIn("mechanism_supported_within_scope", result["allowed_claims"])
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["recommended_next"], "confirm")

    def test_gain_needs_matched_and_indexed_comparison_effect_budget_contract(self):
        for missing in ("comparison_reference", "effect_reference", "budget_reference", "evaluation_contract_reference"):
            record = fixture("single_matched_gain")
            record["inference"].pop(missing)
            with self.subTest(missing=missing):
                self.assert_denied(record, "budget_specific_gain")
        record = fixture("single_matched_gain")
        record["evidence"]["comparison"] = "exploratory"
        self.assert_denied(record, "budget_specific_gain")
        record = fixture("single_matched_gain")
        record["evidence"]["references"] = []
        self.assert_denied(record, "budget_specific_gain")

    def test_two_or_many_seed_ids_and_booleans_are_not_null_evidence(self):
        for count in (2, 30):
            record = fixture("two_seeds_not_null")
            record["inference"]["confirmation_record"]["units"] = [{"unit_id": str(i), "reference": REF} for i in range(count)]
            result = self.assert_denied(record)
            self.assertTrue(result["schema_valid"])
            self.assertFalse(result["proposed_science_decision_eligible"])

    def test_complete_performance_equivalence_is_not_mechanism_null(self):
        record = fixture("performance_equivalence")
        record["proposal"]["science_decision"] = "mechanism_null_within_scope"
        result = evidence.assess_record(record)
        self.assertIn("no_practical_gain_within_scope", result["allowed_claims"])
        self.assertFalse(result["proposed_science_decision_eligible"])
        self.assertNotIn("mechanism_null_within_scope", result["allowed_science_decisions"])
        self.assertFalse(result["promotion_eligible"])

    def test_equivalence_boundary_is_inclusive_but_not_rounded_inward(self):
        record = fixture("performance_equivalence")
        record["inference"]["effect_interval"] = [-0.05, 0.05]
        self.assertIn("no_practical_gain_within_scope", evidence.assess_record(record)["allowed_claims"])
        record["inference"]["effect_interval"][1] = math.nextafter(0.05, math.inf)
        self.assert_denied(record)

    def test_interval_containing_zero_is_not_itself_equivalence(self):
        record = fixture("performance_equivalence")
        record["inference"]["effect_interval"] = [-0.2, 0.2]
        self.assert_denied(record)
        record["inference"]["effect_interval"] = [0.01, 0.02]
        self.assertIn("no_practical_gain_within_scope", evidence.assess_record(record)["allowed_claims"])

    def test_invalid_margins_intervals_and_truthy_strings_fail_schema(self):
        cases = [("practical_margin", v) for v in (0, -1, True, math.nan, math.inf, 10**400)]
        cases += [("effect_interval", v) for v in ([1, -1], [0], [0, math.nan], [False, 0], [0, math.inf], "[-.01,.01]")]
        cases += [(flag, "true") for flag in evidence.FLAGS]
        for key, value in cases:
            record = fixture("performance_equivalence")
            record["inference"][key] = value
            with self.subTest(key=key, value=str(value)[:30]):
                result = evidence.assess_record(record)
                self.assertFalse(result["schema_valid"])
                self.assertEqual(result["allowed_claims"], [])

    def test_equivalence_requires_referenced_uncertainty_selection_and_adequacy(self):
        for missing in ("uncertainty_reference", "selection_accounting_reference", "margin_registration_reference", "optimization_adequacy_reference", "optimization_scope", "effect_metric", "probe_not_applicable_reference"):
            record = fixture("performance_equivalence")
            record["inference"].pop(missing)
            with self.subTest(missing=missing):
                self.assert_denied(record)
        record = fixture("performance_equivalence")
        record["inference"]["optimization_scope"] = "different setting"
        self.assert_denied(record)
        record["inference"]["optimization_scope"] = record["scope"]
        record["evidence"]["optimization_adequacy"] = "unknown"
        self.assert_denied(record)

    def test_known_unhealthy_optimization_blocks_strong_empirical_claims(self):
        for name in ("performance_equivalence", "functional_prediction_equivalence"):
            record = fixture(name)
            record["evidence"]["optimizer_health"] = "unhealthy"
            result = self.assert_denied(record)
            self.assertNotIn("mechanism_null_within_scope", result["allowed_science_decisions"])
            self.assertEqual(result["recommended_next"], "repair")
            self.assertEqual(result["confirmation_need"], "diagnose_first")
            self.assertIn("OPTIMIZATION_EVIDENCE_CONFLICT", {issue["code"] for issue in result["issues"]})
        record = promotion_record()
        record["evidence"]["optimizer_health"] = "unhealthy"
        self.assertFalse(evidence.assess_record(record)["promotion_eligible"])

    def test_confirmation_requires_frozen_protocol_exposure_and_assessment(self):
        for key in ("protocol_reference", "data_exposure_reference", "independence_reference", "assessment_reference"):
            record = fixture("performance_equivalence")
            record["inference"]["confirmation_record"].pop(key)
            with self.subTest(key=key):
                self.assert_denied(record)
        for exposure in ("development_seen", "mixed", "unknown"):
            record = fixture("performance_equivalence")
            record["evidence"]["confirmation"] = "new_conditions"
            record["inference"]["confirmation_record"].update(kind="new_conditions", data_exposure=exposure)
            with self.subTest(exposure=exposure):
                self.assert_denied(record)
        record["inference"]["confirmation_record"]["data_exposure"] = "unseen_for_selection"
        self.assertIn("no_practical_gain_within_scope", evidence.assess_record(record)["allowed_claims"])

    def test_confirmation_kind_scope_and_unique_units_cannot_be_mixed(self):
        record = fixture("performance_equivalence")
        record["inference"]["confirmation_record"]["kind"] = "new_conditions"
        self.assert_denied(record)
        record = fixture("performance_equivalence")
        record["inference"]["confirmation_record"]["scope"] = "other scope"
        self.assert_denied(record)
        record = fixture("performance_equivalence")
        units = record["inference"]["confirmation_record"]["units"]
        units[1]["unit_id"] = units[0]["unit_id"]
        self.assertFalse(evidence.assess_record(record)["schema_valid"])

    def test_scoped_functional_equivalence_has_a_limited_declared_basis(self):
        result = evidence.assess_record(fixture("functional_prediction_equivalence"))
        self.assertTrue(result["proposed_claim_eligible"])
        self.assertTrue(result["proposed_science_decision_eligible"])
        self.assertEqual(result["science_decision_basis"]["mechanism_null_within_scope"], ["practical_equivalence_of_registered_functional_prediction_only"])
        self.assertFalse(result["actual_use_eligible"])

    def test_functional_null_requires_specific_prediction_and_matched_probe(self):
        cases = [("claim_target", "algorithm_performance"), ("registered_prediction_reference", None)]
        for key, value in cases:
            record = fixture("functional_prediction_equivalence")
            record["inference"][key] = value
            self.assertNotIn("mechanism_null_within_scope", evidence.assess_record(record)["allowed_science_decisions"])
        for key, value in (("scope", "other"), ("comparison", "exploratory"), ("protocol_reference", None), ("prediction_reference", None), ("results_reference", None)):
            record = fixture("functional_prediction_equivalence")
            record["inference"]["probe_record"][key] = value
            with self.subTest(key=key):
                self.assertNotIn("mechanism_null_within_scope", evidence.assess_record(record)["allowed_science_decisions"])

    def test_independent_review_needs_identity_scope_coverage_and_explicit_verdict(self):
        cases = (("reviewer_id", "synthetic_author_A"), ("scope", "other scope"), ("verdict", "concerns"),
                 ("supports", []), ("reviewed_references", []), ("evidence_author_ids", []))
        for key, value in cases:
            record = fixture("functional_prediction_equivalence")
            record["inference"]["review_record"][key] = value
            with self.subTest(key=key):
                self.assertNotIn("mechanism_null_within_scope", evidence.assess_record(record)["allowed_science_decisions"])

    def test_mechanism_support_requires_alternative_explanations_and_own_review(self):
        record = fixture("functional_prediction_equivalence")
        record["evidence"]["effect"] = "gain"
        record["inference"]["effect_interval"] = [0.1, 0.2]
        record["inference"]["alternative_explanations_reference"] = REF
        record["inference"]["review_record"]["supports"] = ["mechanism_supported_within_scope"]
        result = evidence.assess_record(record)
        self.assertIn("mechanism_supported_within_scope", result["allowed_claims"])
        record["inference"].pop("alternative_explanations_reference")
        self.assert_denied(record, "mechanism_supported_within_scope")

    def test_opposite_sign_effect_interval_blocks_effect_based_decisions(self):
        record = fixture("functional_prediction_equivalence")
        record["evidence"]["effect"] = "gain"
        inf = record["inference"]
        inf["effect_interval"] = [0.1, 0.2]
        for key in ("alternative_explanations_reference", "critical_events_reference",
                    "verification_reference", "acceptance_reference", "frontier_comparison_reference"):
            inf[key] = REF
        inf["review_record"]["supports"] = ["mechanism_supported_within_scope", "promote_incumbent"]
        record["proposal"].update(scientific_claim="mechanism_supported_within_scope",
                                  model_decision="promote_incumbent", science_decision="supported_within_scope")
        baseline = evidence.assess_record(record)
        self.assertIn("mechanism_supported_within_scope", baseline["allowed_claims"])
        self.assertTrue(baseline["promotion_eligible"])
        self.assertIn("add_frontier", baseline["allowed_model_decisions"])

        for interval in ([-0.2, -0.1], [-0.02, -0.01]):
            with self.subTest(interval=interval):
                inf["effect_interval"] = interval
                result = evidence.assess_record(record)
                self.assertTrue(result["schema_valid"])
                self.assertEqual(set(result["allowed_claims"]), {"observation_only", "unresolved"})
                self.assertNotIn("supported_within_scope", result["allowed_science_decisions"])
                self.assertNotIn("mechanism_null_within_scope", result["allowed_science_decisions"])
                self.assertNotIn("add_frontier", result["allowed_model_decisions"])
                self.assertFalse(result["promotion_eligible"])
                self.assertFalse(result["proposed_claim_eligible"])
                self.assertFalse(result["proposed_science_decision_eligible"])
                self.assertFalse(result["proposed_model_decision_eligible"])
                self.assertEqual(result["recommended_next"], "repair")
                self.assertEqual(result["confirmation_need"], "diagnose_first")
                self.assertIn("EFFECT_INTERVAL_CONFLICT", {issue["code"] for issue in result["issues"]})

    def test_opposite_sign_loss_interval_cannot_be_promoted_as_equivalence(self):
        record = promotion_record()
        record["evidence"]["effect"] = "loss"
        record["inference"]["effect_interval"] = [0.01, 0.02]
        result = self.assert_denied(record)
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["confirmation_need"], "diagnose_first")

    def test_zero_crossing_interval_does_not_add_significance_gate_to_observation(self):
        for interval in ([-0.2, 0.1], [-0.2, 0.0], [0.1, 0.2]):
            with self.subTest(interval=interval):
                record = fixture("single_matched_gain")
                record["inference"]["effect_interval"] = interval
                result = evidence.assess_record(record)
                self.assertTrue(result["proposed_claim_eligible"])
                self.assertNotIn("EFFECT_INTERVAL_CONFLICT", {issue["code"] for issue in result["issues"]})

    def test_effect_conflict_cannot_be_hidden_by_missing_or_different_target(self):
        for target in (None, "unknown", "mathematical_statement", "algorithm_performance", "registered_functional_prediction"):
            with self.subTest(target=target):
                record = promotion_record()
                record["evidence"]["effect"] = "gain"
                record["inference"]["effect_interval"] = [-0.2, -0.1]
                record["inference"]["frontier_comparison_reference"] = REF
                if target is None:
                    record["inference"].pop("claim_target")
                else:
                    record["inference"]["claim_target"] = target
                result = evidence.assess_record(record)
                self.assertTrue(result["schema_valid"])
                self.assertIn("EFFECT_INTERVAL_CONFLICT", {issue["code"] for issue in result["issues"]})
                self.assertFalse(result["promotion_eligible"])
                self.assertNotIn("add_frontier", result["allowed_model_decisions"])
                self.assertEqual(result["confirmation_need"], "diagnose_first")

    def test_effect_based_model_decisions_require_an_empirical_target(self):
        for effect in ("gain", "tradeoff"):
            for target in (None, "unknown", "mathematical_statement"):
                with self.subTest(effect=effect, target=target):
                    record = promotion_record()
                    record["evidence"]["effect"] = effect
                    record["inference"]["effect_interval"] = [0.1, 0.2]
                    record["inference"]["frontier_comparison_reference"] = REF
                    if target is None:
                        record["inference"].pop("claim_target")
                    else:
                        record["inference"]["claim_target"] = target
                    result = evidence.assess_record(record)
                    self.assertTrue(result["schema_valid"])
                    self.assertEqual(set(result["allowed_claims"]), {"observation_only", "unresolved"})
                    self.assertNotIn("add_frontier", result["allowed_model_decisions"])
                    self.assertFalse(result["promotion_eligible"])

    def test_unknown_health_cannot_be_supplied_by_supported_adequacy(self):
        for health in (None, "unknown", "not_applicable", "unhealthy"):
            for name in ("performance_equivalence", "functional_prediction_equivalence"):
                with self.subTest(health=health, fixture=name):
                    record = fixture(name)
                    if health is None:
                        record["evidence"].pop("optimizer_health")
                    else:
                        record["evidence"]["optimizer_health"] = health
                    result = self.assert_denied(record)
                    self.assertTrue(result["schema_valid"])
                    self.assertNotIn("mechanism_null_within_scope", result["allowed_science_decisions"])
                    self.assertEqual(result["confirmation_need"], "diagnose_first")
                    self.assertEqual(result["recommended_next"], "repair" if health == "unhealthy" else "discriminate")
                    if name == "functional_prediction_equivalence":
                        record["evidence"]["effect"] = "gain"
                        record["inference"]["effect_interval"] = [0.1, 0.2]
                        record["inference"]["alternative_explanations_reference"] = REF
                        record["inference"]["review_record"]["supports"] = ["mechanism_supported_within_scope"]
                        self.assert_denied(record, "mechanism_supported_within_scope")
            record = promotion_record()
            if health is None:
                record["evidence"].pop("optimizer_health")
            else:
                record["evidence"]["optimizer_health"] = health
            self.assertFalse(evidence.assess_record(record)["promotion_eligible"])

    def test_health_nonapplicability_requires_indexed_reference(self):
        record = promotion_record()
        record["evidence"]["optimizer_health"] = "not_applicable"
        for reference in (None, "reports/not_indexed.md", REF):
            with self.subTest(reference=reference):
                record["inference"]["optimizer_health_not_applicable_reference"] = reference
                result = evidence.assess_record(record)
                self.assertEqual(result["promotion_eligible"], reference == REF)
                self.assertEqual("no_practical_gain_within_scope" in result["allowed_claims"], reference == REF)

    def test_unknown_health_preserves_budget_observation_and_mathematical_proof(self):
        for health in (None, "unknown", "not_applicable"):
            for name in ("single_matched_gain", "reviewed_counterexample"):
                with self.subTest(health=health, fixture=name):
                    record = fixture(name)
                    if health is None:
                        record["evidence"].pop("optimizer_health")
                    else:
                        record["evidence"]["optimizer_health"] = health
                    result = evidence.assess_record(record)
                    self.assertTrue(result["proposed_claim_eligible"])
                    self.assertEqual(result["confirmation_need"], "not_required_for_this_claim")

    def test_mathematical_proof_does_not_waive_simultaneous_promotion_requirements(self):
        record = fixture("reviewed_counterexample")
        record["proposal"]["model_decision"] = "promote_incumbent"
        result = evidence.assess_record(record)
        self.assertTrue(result["proposed_claim_eligible"])
        self.assertTrue(result["proposed_science_decision_eligible"])
        self.assertFalse(result["promotion_eligible"])
        self.assertFalse(result["proposed_model_decision_eligible"])
        self.assertEqual(result["confirmation_need"], "diagnose_first")
        self.assertEqual(result["recommended_next"], "discriminate")

    def test_confirmation_advice_covers_all_proposed_strong_claims(self):
        record = promotion_record()
        record["evidence"]["confirmation"] = "new_conditions"
        confirmation = record["inference"]["confirmation_record"]
        confirmation.update(kind="new_conditions", data_exposure="unseen_for_selection")
        confirmation["units"] = confirmation["units"][:1]
        result = self.assert_denied(record)
        self.assertFalse(result["promotion_eligible"])
        self.assertEqual(result["confirmation_need"], "needed")
        self.assertEqual(result["recommended_next"], "confirm")
        # Gain can meet the promotion record gate while the simultaneous
        # equivalence request still lacks its own repetition requirement.
        record["evidence"]["effect"] = "gain"
        record["inference"]["effect_interval"] = [0.01, 0.02]
        result = self.assert_denied(record)
        self.assertTrue(result["promotion_eligible"])
        self.assertEqual(result["confirmation_need"], "needed")
        # Repetition alone does not replace the other evidence prerequisites.
        confirmation["units"].append({"unit_id": "another_unit", "reference": REF})
        record["inference"].pop("uncertainty_reference")
        result = self.assert_denied(record)
        self.assertEqual(result["confirmation_need"], "diagnose_first")
        self.assertEqual(result["recommended_next"], "discriminate")

    def test_deleting_target_health_or_supporting_records_never_adds_eligibility(self):
        records = [promotion_record(), fixture("functional_prediction_equivalence")]
        conflicting = promotion_record()
        conflicting["evidence"]["effect"] = "gain"
        conflicting["inference"]["effect_interval"] = [-0.2, -0.1]
        records.append(conflicting)
        paths = [("inference", "claim_target"), ("evidence", "optimizer_health"),
                 ("inference", "optimization_adequacy_reference"), ("inference", "uncertainty_reference"),
                 ("inference", "comparison_reference"), ("inference", "confirmation_record"),
                 ("inference", "review_record")]
        for record in records:
            baseline = evidence.assess_record(record)
            for group, field in paths:
                with self.subTest(run=record["run_id"], removed=(group, field)):
                    variant = deepcopy(record)
                    variant[group].pop(field, None)
                    result = evidence.assess_record(variant)
                    self.assertTrue(result["schema_valid"])
                    for output in ("allowed_claims", "allowed_science_decisions", "allowed_model_decisions"):
                        self.assertLessEqual(set(result[output]), set(baseline[output]))
                    self.assertLessEqual(result["promotion_eligible"], baseline["promotion_eligible"])

    def test_exact_reviewed_counterexample_does_not_require_optimization_or_seeds(self):
        record = fixture("reviewed_counterexample")
        result = evidence.assess_record(record)
        self.assertTrue(result["proposed_claim_eligible"])
        self.assertTrue(result["proposed_science_decision_eligible"])
        self.assertEqual(result["confirmation_need"], "not_required_for_this_claim")
        self.assertFalse(result["promotion_eligible"])
        record["evidence"]["optimizer_health"] = "unhealthy"
        self.assertIn("falsified_by_proof", evidence.assess_record(record)["allowed_claims"])

    def test_boolean_review_and_proof_reference_alone_are_not_mathematical_evidence(self):
        record = fixture("reviewed_counterexample")
        record["inference"].pop("review_record")
        self.assert_denied(record, "falsified_by_proof")
        record = fixture("reviewed_counterexample")
        record["inference"].pop("proof_record")
        self.assert_denied(record, "falsified_by_proof")
        for key, value in (("scope", "different theorem"), ("target_statement", ""), ("assumptions_reference", None)):
            record = fixture("reviewed_counterexample")
            record["inference"]["proof_record"][key] = value
            self.assert_denied(record, "falsified_by_proof")
        record = fixture("reviewed_counterexample")
        record["inference"]["claim_target"] = "algorithm_performance"
        self.assert_denied(record, "falsified_by_proof")

    def test_promotion_formal_gate_requires_separate_record_and_never_actual_use(self):
        record = promotion_record()
        result = evidence.assess_record(record)
        self.assertTrue(result["promotion_eligible"], result)
        self.assertTrue(result["proposed_model_decision_eligible"])
        self.assertEqual(result["assessment_mode"], "synthetic_fixture")
        self.assertFalse(result["actual_use_eligible"])
        self.assertEqual(result["actual_use_status"], "excluded_synthetic_fixture")
        for missing in ("verification_reference", "critical_events_reference", "acceptance_reference", "independent_review_reference"):
            variant = deepcopy(record)
            variant["inference"].pop(missing)
            with self.subTest(missing=missing):
                self.assertFalse(evidence.assess_record(variant)["promotion_eligible"])

    def test_critical_event_regression_cannot_be_overridden_by_review(self):
        for value in ("regressed", "unknown", "not_applicable"):
            record = promotion_record()
            record["evidence"]["critical_events"] = value
            result = evidence.assess_record(record)
            self.assertFalse(result["promotion_eligible"])
            if value == "regressed":
                self.assertEqual(result["confirmation_need"], "diagnose_first")

    def test_unknown_validity_is_diagnosed_before_confirmation_advice(self):
        for status in ("unknown", "invalid"):
            record = fixture("single_matched_gain")
            record["evidence"]["validity"] = status
            result = evidence.assess_record(record)
            self.assertFalse(result["proposed_claim_eligible"])
            self.assertEqual(result["confirmation_need"], "diagnose_first")

    def test_references_are_safe_canonical_paths_and_missing_is_not_success(self):
        for bad in ("/tmp/a.md", "../a.md", "a/../b.md", "./a.md", "a//b.md", "a\\b.md", "https://example.org/proof", "a\n.md", "a/*.md", " a.md"):
            record = minimal()
            record["evidence"] = {"references": [bad]}
            with self.subTest(bad=bad):
                self.assertFalse(evidence.assess_record(record)["schema_valid"])
        record = fixture("performance_equivalence")
        record["inference"]["uncertainty_reference"] = "reports/not_indexed.md"
        result = self.assert_denied(record)
        self.assertTrue(result["schema_valid"])
        self.assertIn("UNINDEXED_REFERENCE", {issue["code"] for issue in result["issues"]})

    def test_input_is_not_mutated_and_assessor_performs_no_file_io(self):
        record = fixture("functional_prediction_equivalence")
        before = deepcopy(record)
        with patch("builtins.open", side_effect=AssertionError("assessor must not read files")), patch.object(Path, "read_text", side_effect=AssertionError("assessor must not read files")):
            result = evidence.assess_record(record)
        self.assertEqual(record, before)
        self.assertTrue(result["schema_valid"])
        json.dumps(result, allow_nan=False)

    def test_all_public_fixtures_are_synthetic_valid_json_and_expected_outcomes(self):
        paths = sorted((ROOT / "harness/examples").glob("run_*.json"))
        self.assertEqual(len(paths), 6)
        expected_keys = {
            "budget_specific_gain_eligible": ("allowed_claims", "budget_specific_gain"),
            "no_practical_gain_eligible": ("allowed_claims", "no_practical_gain_within_scope"),
            "mechanism_null_eligible": ("allowed_science_decisions", "mechanism_null_within_scope"),
            "falsified_by_proof_eligible": ("allowed_claims", "falsified_by_proof"),
        }
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(path=path.name):
                result = evidence.assess_record(record)
                self.assertIs(record["synthetic"], True)
                self.assertTrue(result["schema_valid"], result["issues"])
                self.assertFalse(result["actual_use_eligible"])
                for key, wanted in record["fixture_expected"].items():
                    if key in expected_keys:
                        field, member = expected_keys[key]
                        self.assertEqual(member in result[field], wanted)
                    elif key in {"promotion_eligible", "proposed_claim_eligible", "proposed_science_decision_eligible"}:
                        self.assertEqual(result[key], wanted)
                    elif key == "empirical_replication_required":
                        self.assertFalse(wanted)
                        self.assertEqual(result["confirmation_need"], "not_required_for_this_claim")
                    elif key == "physical_mechanism_absence_established":
                        self.assertFalse(wanted)
                        self.assertEqual(result["science_decision_basis"]["mechanism_null_within_scope"], ["practical_equivalence_of_registered_functional_prediction_only"])
                    else:
                        self.fail("Unknown fixture expectation: " + key)
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
