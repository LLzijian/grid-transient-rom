"""Offline record-eligibility checks, not statistical or scientific certification.

No files are opened and no model, process, Git command or network is invoked.
References must be canonical repository-relative paths indexed in
``evidence.references``; a caller may separately check that they exist. Several
references may point to one reviewable report: this is not a file-count gate.

Short observation records need only identity, purpose, scope and their available
evidence. Strong claims conditionally require the records illustrated in
``harness/examples/run_*.json``. Missing evidence means unknown, never success.
All numerical effects use delta = candidate benefit minus comparator benefit:
positive delta is better. An interval contained in [-margin, margin] supports
practical equivalence of that measured quantity, not absence of a mechanism.
"""
from __future__ import annotations

import math
from pathlib import PurePosixPath


PURPOSES = {"probe", "adaptation", "discrimination", "confirmation"}
EVIDENCE_ENUMS = {
    "validity": {"valid", "invalid", "unknown"},
    "optimizer_health": {"healthy", "unhealthy", "unknown", "not_applicable"},
    "optimization_adequacy": {"supported", "insufficient", "unknown", "not_applicable"},
    "mechanism_probe": {"passed", "failed", "unidentifiable", "unknown", "not_applicable"},
    "comparison": {"matched", "exploratory", "none", "unknown"},
    "confirmation": {"none", "seed_replication", "new_conditions", "unknown"},
    "effect": {"gain", "tradeoff", "no_clear_gain", "equivalence", "loss", "unknown"},
    "critical_events": {"pass", "regressed", "unknown", "not_applicable"},
}
CLAIMS = {
    "observation_only", "budget_specific_gain", "tradeoff",
    "no_practical_gain_within_scope", "mechanism_supported_within_scope",
    "falsified_by_proof", "unresolved",
}
MODEL_DECISIONS = {"promote_incumbent", "add_frontier", "retain_evidence", "ineligible"}
SCIENCE_DECISIONS = {
    "continue", "supported_within_scope", "mechanism_null_within_scope",
    "falsified_within_scope", "unresolved", "abandon",
}
NEXT_ACTIONS = {"repair", "adapt", "discriminate", "confirm", "continue_registered", "defer", "synthesize"}
TARGETS = {"algorithm_performance", "registered_functional_prediction", "mathematical_statement", "unknown"}
EXPOSURES = {"unseen_for_selection", "development_seen", "mixed", "unknown"}
FLAGS = {
    "margin_preregistered", "uncertainty_justified", "selection_accounted_for",
    "proof_reviewed", "independent_review",
}
REFERENCE_FIELDS = {
    "comparison_reference", "effect_reference", "budget_reference",
    "optimization_adequacy_reference", "probe_reference", "probe_not_applicable_reference",
    "margin_registration_reference", "uncertainty_reference", "selection_accounting_reference",
    "registered_prediction_reference", "alternative_explanations_reference",
    "evaluation_contract_reference", "critical_events_reference", "verification_reference",
    "acceptance_reference", "frontier_comparison_reference",
    "proof_reference", "independent_review_reference",
}
NOTICE = (
    "Record requirements only: this evaluator does not read evidence, verify its truth, "
    "certify statistics or mathematics, or authorize a scientific conclusion or promotion."
)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _safe_reference(value):
    if not _text(value) or value != value.strip() or any(ord(c) < 32 for c in value):
        return False
    if "\\" in value or ":" in value or any(c in value for c in "*?[]"):
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and str(path) == value and value != "."


class _Check:
    def __init__(self):
        self.issues = []
        self.references = set()

    def issue(self, code, path, message, severity="warning"):
        item = {"code": code, "severity": severity, "path": path, "message": message}
        if item not in self.issues:
            self.issues.append(item)

    def obj(self, value, path):
        if value is None:
            return {}
        if not isinstance(value, dict):
            self.issue("SCHEMA_TYPE", path, "Expected an object.", "error")
            return {}
        return value

    def enum(self, value, choices, path, default=None):
        if value is None:
            return default
        if not isinstance(value, str) or value not in choices:
            self.issue("SCHEMA_ENUM", path, "Unrecognized enumeration value.", "error")
            return default
        return value

    def reference(self, value, path):
        if value is None:
            return False
        if not _safe_reference(value):
            self.issue("SCHEMA_REFERENCE", path, "Expected a canonical safe repository-relative file reference.", "error")
            return False
        if value not in self.references:
            self.issue("UNINDEXED_REFERENCE", path, "Reference is not indexed in evidence.references; it cannot satisfy a gate.")
            return False
        return True

    def reference_list(self, value, path):
        if value is None:
            return []
        if not isinstance(value, list):
            self.issue("SCHEMA_TYPE", path, "Expected a list of file references.", "error")
            return []
        good = []
        for i, item in enumerate(value):
            if self.reference(item, f"{path}[{i}]"):
                good.append(item)
        return good


def assess_record(record: dict) -> dict:
    """Return JSON-compatible eligibility and advisory results without side effects.

    ``valid`` is retained for the package API and is exactly ``schema_valid``.
    A denied proposed claim does not make a well-formed record schema-invalid.
    ``promotion_eligible`` means record requirements met, never actual promotion.
    """
    check = _Check()
    r = check.obj(record, "record")
    if not isinstance(record, dict):
        r = {}
    if type(r.get("schema_version")) is not int or r.get("schema_version") != 2:
        check.issue("SCHEMA_VERSION", "schema_version", "Expected integer schema_version=2.", "error")
    for key in ("run_id", "scope"):
        if not _text(r.get(key)):
            check.issue("SCHEMA_REQUIRED", key, "A nonempty string is required.", "error")
    purpose = check.enum(r.get("purpose"), PURPOSES, "purpose")
    if purpose is None and r.get("purpose") is None:
        check.issue("SCHEMA_REQUIRED", "purpose", "A registered purpose is required.", "error")
    if "synthetic" in r and type(r["synthetic"]) is not bool:
        check.issue("SCHEMA_TYPE", "synthetic", "Expected a boolean.", "error")
    e = check.obj(r.get("evidence"), "evidence")
    inf = check.obj(r.get("inference"), "inference")
    proposal = check.obj(r.get("proposal"), "proposal")
    observed = {key: check.enum(e.get(key), values, "evidence." + key, "unknown")
                for key, values in EVIDENCE_ENUMS.items()}
    missing = [key for key in EVIDENCE_ENUMS if e.get(key) is None]
    if missing:
        check.issue("MISSING_EVIDENCE", "evidence", "Missing fields are unknown: " + ", ".join(missing))
    references = e.get("references", [])
    if not isinstance(references, list):
        check.issue("SCHEMA_TYPE", "evidence.references", "Expected a list of references.", "error")
        references = []
    for i, reference in enumerate(references):
        if not _safe_reference(reference):
            check.issue("SCHEMA_REFERENCE", f"evidence.references[{i}]", "Unsafe or noncanonical reference.", "error")
        else:
            check.references.add(reference)
    for key in FLAGS:
        if key in inf and type(inf[key]) is not bool:
            check.issue("SCHEMA_TYPE", "inference." + key, "Expected a boolean; truthy strings are not evidence.", "error")
    refs = {key: check.reference(inf.get(key), "inference." + key) for key in REFERENCE_FIELDS}
    target = check.enum(inf.get("claim_target"), TARGETS, "inference.claim_target", "unknown")
    direction = check.enum(inf.get("effect_direction"), {"candidate_gain_positive"},
                           "inference.effect_direction", "candidate_gain_positive")
    margin = inf.get("practical_margin")
    if margin is not None and (not _number(margin) or margin <= 0):
        check.issue("SCHEMA_MARGIN", "inference.practical_margin", "Margin must be a finite positive number, not a boolean.", "error")
    interval = inf.get("effect_interval")
    interval_valid = (isinstance(interval, list) and len(interval) == 2
                      and all(_number(x) for x in interval) and interval[0] <= interval[1])
    if interval is not None and not interval_valid:
        check.issue("SCHEMA_INTERVAL", "inference.effect_interval", "Expected [finite lower, finite upper] in increasing order.", "error")
    contained = bool(interval_valid and _number(margin) and margin > 0
                     and -margin <= interval[0] <= interval[1] <= margin)
    # A missing or zero-crossing interval does not prevent a budget-specific
    # observation. A wholly opposite-sign interval contradicts its own summary;
    # keep the record, but require reconciliation before any effect-based gate.
    effect_consistent = not (interval_valid
        and target in {"algorithm_performance", "registered_functional_prediction"}
        and ((observed["effect"] == "gain" and interval[1] < 0)
             or (observed["effect"] == "loss" and interval[0] > 0)))
    if not effect_consistent:
        check.issue("EFFECT_INTERVAL_CONFLICT", "inference.effect_interval",
                    "The effect summary and interval have opposite signs under candidate_gain_positive; reconcile the record before effect-based claims or model decisions.")
    for key in ("optimization_scope", "effect_metric"):
        if key in inf and not _text(inf[key]):
            check.issue("SCHEMA_TYPE", "inference." + key, "Expected a nonempty string.", "error")

    # Confirmation records describe exposure and a scoped assessment, not a
    # universal seed-count rule. Two units establish repetition only.
    confirmation = check.obj(inf.get("confirmation_record"), "inference.confirmation_record")
    ck = check.enum(confirmation.get("kind"), {"seed_replication", "new_conditions", "unknown"},
                    "inference.confirmation_record.kind", "unknown")
    exposure = check.enum(confirmation.get("data_exposure"), EXPOSURES,
                          "inference.confirmation_record.data_exposure", "unknown")
    confirmation_refs = {key: check.reference(confirmation.get(key), "inference.confirmation_record." + key)
                         for key in ("protocol_reference", "data_exposure_reference", "independence_reference", "assessment_reference")}
    units = confirmation.get("units", [])
    if not isinstance(units, list):
        check.issue("SCHEMA_TYPE", "inference.confirmation_record.units", "Expected independent-unit records.", "error")
        units = []
    unit_ids = set()
    unit_refs_good = bool(units)
    for i, unit in enumerate(units):
        unit = check.obj(unit, f"inference.confirmation_record.units[{i}]")
        uid = unit.get("unit_id")
        if not _text(uid):
            check.issue("SCHEMA_REQUIRED", f"inference.confirmation_record.units[{i}].unit_id", "A unit identity is required.", "error")
            unit_refs_good = False
        elif uid in unit_ids:
            check.issue("SCHEMA_DUPLICATE_UNIT", "inference.confirmation_record.units", "Repeated IDs are not independent replications.", "error")
            unit_refs_good = False
        else:
            unit_ids.add(uid)
        unit_refs_good &= check.reference(unit.get("reference"), f"inference.confirmation_record.units[{i}].reference")
    confirm_scope = bool(confirmation.get("scope") == r.get("scope") and _text(r.get("scope")))
    exposure_ok = exposure == "unseen_for_selection" if ck == "new_conditions" else exposure in {"unseen_for_selection", "development_seen"}
    confirm_ok = bool(ck in {"seed_replication", "new_conditions"} and ck == observed["confirmation"]
                      and confirm_scope and all(confirmation_refs.values()) and unit_refs_good
                      and exposure_ok and (len(unit_ids) >= 2 if ck == "seed_replication" else len(unit_ids) >= 1))
    repeated = confirm_ok and len(unit_ids) >= 2
    if observed["confirmation"] == "seed_replication":
        check.issue("SEEDS_ARE_NOT_NEW_CONDITIONS", "evidence.confirmation", "Seed replication alone establishes neither adequate confirmation nor new-condition generalization.")
    if observed["confirmation"] == "new_conditions" and exposure != "unseen_for_selection":
        check.issue("EXPOSURE_UNRESOLVED", "inference.confirmation_record", "New conditions are not independent confirmation when exposure for selection is mixed, seen or unknown.")

    review = check.obj(inf.get("review_record"), "inference.review_record")
    verdict = check.enum(review.get("verdict"), {"supports", "concerns", "unknown"}, "inference.review_record.verdict", "unknown")
    reviewed = set(check.reference_list(review.get("reviewed_references"), "inference.review_record.reviewed_references"))
    supports = review.get("supports", [])
    if not isinstance(supports, list):
        check.issue("SCHEMA_TYPE", "inference.review_record.supports", "Expected a list of explicitly reviewed claim/decision names.", "error")
        supports = []
    for i, value in enumerate(supports):
        check.enum(value, CLAIMS | MODEL_DECISIONS | SCIENCE_DECISIONS, f"inference.review_record.supports[{i}]")
    authors = review.get("evidence_author_ids", [])
    if not isinstance(authors, list) or not all(_text(x) for x in authors):
        check.issue("SCHEMA_TYPE", "inference.review_record.evidence_author_ids", "Expected a list of nonempty author identities.", "error")
        authors = []
    reviewer = review.get("reviewer_id")
    if reviewer is not None and not _text(reviewer):
        check.issue("SCHEMA_TYPE", "inference.review_record.reviewer_id", "Expected a nonempty identity.", "error")
    review_ref_good = check.reference(review.get("reference"), "inference.review_record.reference")
    review_ok = bool(inf.get("independent_review") is True and refs["independent_review_reference"]
                     and review_ref_good and review.get("reference") == inf.get("independent_review_reference")
                     and review.get("scope") == r.get("scope") and _text(reviewer) and authors
                     and reviewer not in authors and verdict == "supports" and reviewed)
    if _text(reviewer) and reviewer in authors:
        check.issue("REVIEWER_IS_AUTHOR", "inference.review_record", "A declared evidence author cannot satisfy the independent-review requirement.")

    def reviewed_for(name, required):
        return bool(review_ok and name in supports and all(_text(x) and x in reviewed for x in required))

    probe = check.obj(inf.get("probe_record"), "inference.probe_record")
    probe_comparison = check.enum(probe.get("comparison"), {"matched", "exploratory", "none", "unknown"},
                                  "inference.probe_record.comparison", "unknown")
    probe_refs = {key: check.reference(probe.get(key), "inference.probe_record." + key)
                  for key in ("prediction_reference", "protocol_reference", "results_reference")}
    functional_probe = bool(target == "registered_functional_prediction" and refs["registered_prediction_reference"]
                            and observed["mechanism_probe"] == "passed" and probe_comparison == "matched"
                            and probe.get("scope") == r.get("scope") and all(probe_refs.values())
                            and probe.get("prediction_reference") == inf.get("registered_prediction_reference"))
    probe_ok = ((observed["mechanism_probe"] == "passed" and (refs["probe_reference"] or functional_probe))
                or (observed["mechanism_probe"] == "not_applicable" and refs["probe_not_applicable_reference"]))

    proof = check.obj(inf.get("proof_record"), "inference.proof_record")
    for name, nested in (("confirmation_record", confirmation), ("review_record", review),
                         ("probe_record", probe), ("proof_record", proof)):
        if "scope" in nested and not _text(nested["scope"]):
            check.issue("SCHEMA_TYPE", "inference." + name + ".scope", "Expected a nonempty scope string.", "error")
    if "target_statement" in proof and not _text(proof["target_statement"]):
        check.issue("SCHEMA_TYPE", "inference.proof_record.target_statement", "Expected the precise nonempty statement being challenged.", "error")
    proof_kind = check.enum(proof.get("kind"), {"theorem", "exact_counterexample", "unknown"},
                            "inference.proof_record.kind", "unknown")
    proof_refs = {key: check.reference(proof.get(key), "inference.proof_record." + key)
                  for key in ("claim_reference", "assumptions_reference", "proof_reference")}
    mathematical_proof = bool(target == "mathematical_statement" and observed["validity"] == "valid"
                              and inf.get("proof_reviewed") is True and refs["proof_reference"]
                              and proof_kind in {"theorem", "exact_counterexample"} and all(proof_refs.values())
                              and _text(proof.get("target_statement")) and proof.get("scope") == r.get("scope")
                              and proof.get("proof_reference") == inf.get("proof_reference")
                              and reviewed_for("falsified_by_proof", [proof.get("proof_reference"), proof.get("claim_reference"), proof.get("assumptions_reference")]))

    validity = observed["validity"] == "valid"
    empirical_health = observed["optimizer_health"] != "unhealthy"
    comparison = bool(observed["comparison"] == "matched" and refs["comparison_reference"]
                      and refs["effect_reference"] and refs["evaluation_contract_reference"])
    adequacy = bool(observed["optimization_adequacy"] == "supported" and refs["optimization_adequacy_reference"]
                    and inf.get("optimization_scope") == r.get("scope"))
    selection = inf.get("selection_accounted_for") is True and refs["selection_accounting_reference"]
    common_equivalence = {
        "valid evidence": validity,
        "effect summary consistent with supplied interval": effect_consistent,
        "no unresolved known optimizer pathology": empirical_health,
        "reported effect summary": observed["effect"] != "unknown",
        "matched comparison with effect/evaluation references": comparison,
        "scoped optimization adequacy with reference": adequacy,
        "passed probe or justified nonapplicability": probe_ok,
        "scoped repetition with frozen protocol and exposure records": repeated,
        "preregistered positive practical margin with reference": bool(_number(margin) and margin > 0 and inf.get("margin_preregistered") is True and refs["margin_registration_reference"]),
        "justified uncertainty interval with reference": bool(interval_valid and inf.get("uncertainty_justified") is True and refs["uncertainty_reference"] and _text(inf.get("effect_metric"))),
        "whole interval contained in the equivalence margin": contained,
        "adaptive selection addressed with reference": selection,
        "performance or registered functional target": target in {"algorithm_performance", "registered_functional_prediction"},
        "effect direction is candidate gain positive": direction == "candidate_gain_positive",
    }
    equivalence = all(common_equivalence.values())
    budget_gain = bool(validity and effect_consistent and comparison and refs["budget_reference"] and observed["effect"] == "gain"
                       and target in {"algorithm_performance", "registered_functional_prediction"})
    tradeoff = bool(validity and effect_consistent and comparison and observed["effect"] == "tradeoff")
    mechanism_supported = bool(validity and effect_consistent and empirical_health and comparison and adequacy and confirm_ok and functional_probe
                               and observed["effect"] in {"gain", "tradeoff"} and selection
                               and refs["alternative_explanations_reference"]
                               and reviewed_for("mechanism_supported_within_scope", [inf.get("registered_prediction_reference"),
                                   probe.get("results_reference"), inf.get("alternative_explanations_reference"), inf.get("effect_reference")]))
    functional_null = bool(equivalence and functional_probe
                           and reviewed_for("mechanism_null_within_scope", [inf.get("registered_prediction_reference"),
                               probe.get("results_reference"), inf.get("uncertainty_reference"), inf.get("comparison_reference")]))
    promotion_checks = {
        "valid evidence": validity,
        "effect summary consistent with supplied interval": effect_consistent,
        "no unresolved known optimizer pathology": empirical_health,
        "matched referenced comparison": comparison,
        "documented gain, tradeoff or practical equivalence": observed["effect"] in {"gain", "tradeoff"} or equivalence,
        "no critical-event regression with reference": observed["critical_events"] == "pass" and refs["critical_events_reference"],
        "verified-result reference": refs["verification_reference"],
        "scoped confirmation and known exposure": confirm_ok,
        "selection accounted for with reference": bool(selection),
        "acceptance contract reference": refs["acceptance_reference"],
        "independent scoped review explicitly recommends promotion": reviewed_for("promote_incumbent", [inf.get("verification_reference"),
            inf.get("critical_events_reference"), inf.get("evaluation_contract_reference"), confirmation.get("assessment_reference"), inf.get("acceptance_reference")]),
    }
    promotion = all(promotion_checks.values())
    allowed = {"observation_only", "unresolved"}
    for name, eligible in (("budget_specific_gain", budget_gain), ("tradeoff", tradeoff),
                           ("no_practical_gain_within_scope", equivalence),
                           ("mechanism_supported_within_scope", mechanism_supported),
                           ("falsified_by_proof", mathematical_proof)):
        if eligible:
            allowed.add(name)
    allowed_science = {"continue", "unresolved", "abandon"}
    if budget_gain or tradeoff or mechanism_supported:
        allowed_science.add("supported_within_scope")
    if functional_null:
        allowed_science.add("mechanism_null_within_scope")
    if mathematical_proof:
        allowed_science.add("falsified_within_scope")
    allowed_models = {"retain_evidence", "ineligible"}
    if validity and effect_consistent and comparison and observed["effect"] in {"gain", "tradeoff"} and refs["frontier_comparison_reference"]:
        allowed_models.add("add_frontier")
    if promotion:
        allowed_models.add("promote_incumbent")

    pclaim = check.enum(proposal.get("scientific_claim"), CLAIMS, "proposal.scientific_claim")
    pscience = check.enum(proposal.get("science_decision"), SCIENCE_DECISIONS, "proposal.science_decision")
    pmodel = check.enum(proposal.get("model_decision"), MODEL_DECISIONS, "proposal.model_decision")
    check.enum(proposal.get("next_action"), NEXT_ACTIONS, "proposal.next_action")
    for name, value, eligible_set in (("scientific_claim", pclaim, allowed), ("science_decision", pscience, allowed_science), ("model_decision", pmodel, allowed_models)):
        if value is not None and value not in eligible_set:
            check.issue("PROPOSAL_NOT_ELIGIBLE", "proposal." + name, "This proposal exceeds the record's supported eligibility; retain the observation and resolve missing evidence.")
    if pclaim == "no_practical_gain_within_scope" and not equivalence:
        check.issue("EQUIVALENCE_REQUIREMENTS_MISSING", "inference", "; ".join(key for key, passed in common_equivalence.items() if not passed))
    if pmodel == "promote_incumbent" and not promotion:
        check.issue("PROMOTION_REQUIREMENTS_MISSING", "proposal.model_decision", "; ".join(key for key, passed in promotion_checks.items() if not passed))
    if pscience == "mechanism_null_within_scope" and not functional_null:
        check.issue("PERFORMANCE_EQUIVALENCE_IS_NOT_MECHANISM_NULL", "proposal.science_decision", "A registered functional prediction, matched probe and scoped independent review are additionally required; this never asserts that a physical mechanism is absent.")
    if observed["optimizer_health"] == "healthy" and observed["optimization_adequacy"] != "supported":
        check.issue("HEALTH_IS_NOT_ADEQUACY", "evidence", "Healthy optimization diagnostics do not establish adequate optimization.")
    if not empirical_health and observed["optimization_adequacy"] == "supported":
        check.issue("OPTIMIZATION_EVIDENCE_CONFLICT", "evidence", "Known unhealthy optimization conflicts with an unqualified adequacy assertion; repair or resolve that conflict before strong empirical claims or promotion.")
    if observed["effect"] in {"no_clear_gain", "loss", "equivalence"} and not equivalence:
        check.issue("NEGATIVE_OBSERVATION_NOT_NULL", "evidence.effect", "Keep the negative observation; lack of a clear gain or a narrow budget is not a null-mechanism conclusion.")
    check.issue("RECORD_CHECK_NOT_CERTIFICATION", "record", NOTICE)
    if r.get("synthetic") is True:
        check.issue("SYNTHETIC_FIXTURE", "synthetic", "This is a logic fixture, not actual scientific or promotion evidence.")
    schema_valid = not any(x["severity"] == "error" for x in check.issues)
    if not schema_valid:
        allowed, allowed_science, allowed_models = set(), set(), set()
        promotion = False

    if not schema_valid or observed["validity"] == "invalid" or not effect_consistent:
        recommended = "repair"
    elif mathematical_proof:
        recommended = "synthesize"
    elif observed["validity"] != "valid" or observed["mechanism_probe"] in {"failed", "unidentifiable"}:
        recommended = "discriminate"
    elif observed["optimizer_health"] == "unhealthy":
        recommended = "repair"
    elif equivalence:
        recommended = "synthesize"
    elif observed["effect"] in {"no_clear_gain", "loss", "equivalence"} and not adequacy:
        recommended = "adapt"
    elif observed["effect"] in {"gain", "tradeoff"} and not confirm_ok:
        recommended = "confirm"
    else:
        recommended = "continue_registered"
    strong_request = pclaim in {"no_practical_gain_within_scope", "mechanism_supported_within_scope"} or pmodel == "promote_incumbent" or pscience == "mechanism_null_within_scope"
    if not schema_valid or observed["validity"] != "valid" or not effect_consistent:
        confirmation_need = "diagnose_first"
    elif mathematical_proof:
        confirmation_need = "not_required_for_this_claim"
    elif not empirical_health or observed["mechanism_probe"] in {"failed", "unidentifiable"} or observed["critical_events"] == "regressed":
        confirmation_need = "diagnose_first"
    elif (pclaim in {"no_practical_gain_within_scope", "mechanism_supported_within_scope"} or pscience == "mechanism_null_within_scope") and not adequacy:
        confirmation_need = "diagnose_first"
    elif pclaim in {"observation_only", "budget_specific_gain", "tradeoff", "unresolved"} and not strong_request:
        confirmation_need = "not_required_for_this_claim"
    elif strong_request:
        confirmation_need = "not_required_for_this_claim" if confirm_ok else "needed"
    else:
        confirmation_need = "undetermined"
    science_basis = {}
    if "supported_within_scope" in allowed_science:
        science_basis["supported_within_scope"] = [name for name, passed in (
            ("budget_specific_gain", budget_gain), ("tradeoff", tradeoff),
            ("mechanism_supported_within_scope", mechanism_supported)) if passed]
    if "mechanism_null_within_scope" in allowed_science:
        science_basis["mechanism_null_within_scope"] = ["practical_equivalence_of_registered_functional_prediction_only"]
    if "falsified_within_scope" in allowed_science:
        science_basis["falsified_within_scope"] = ["reviewed_exact_mathematical_statement_only"]
    return {
        "valid": schema_valid, "schema_valid": schema_valid, "issues": check.issues,
        "allowed_claims": sorted(allowed), "allowed_science_decisions": sorted(allowed_science),
        "allowed_model_decisions": sorted(allowed_models), "recommended_next": recommended,
        "promotion_eligible": bool(promotion), "confirmation_need": confirmation_need,
        "proposed_claim_eligible": None if pclaim is None else pclaim in allowed,
        "proposed_science_decision_eligible": None if pscience is None else pscience in allowed_science,
        "proposed_model_decision_eligible": None if pmodel is None else pmodel in allowed_models,
        "normalized_evidence": observed, "effect_convention": "candidate_gain_positive",
        "science_decision_basis": science_basis,
        "assessment_mode": "synthetic_fixture" if r.get("synthetic") is True else "record_only",
        "actual_use_eligible": False if r.get("synthetic") is True else None,
        "actual_use_status": "excluded_synthetic_fixture" if r.get("synthetic") is True else "not_assessed_or_authorized",
        "record_only_notice": NOTICE,
    }


def normalize_legacy_optimization(payload: dict) -> dict:
    """Interpret the old flag without inventing optimization or science evidence.

    False means the legacy health heuristic did not fire. True records unresolved
    optimization, not a diagnosed pathology. Neither value establishes adequacy.
    Boundary parameters and NOT_RETAINED are preserved without scientific inference.
    """
    issues = []
    if not isinstance(payload, dict):
        payload = {}
        issues.append({"code": "SCHEMA_TYPE", "severity": "error", "path": "payload", "message": "Expected an object."})
    flag = payload.get("optimization_unresolved")
    if flag is not None and type(flag) is not bool:
        issues.append({"code": "SCHEMA_TYPE", "severity": "error", "path": "optimization_unresolved", "message": "Expected a boolean; strings and integers are not booleans."})
        flag = None
    retained = {key: payload[key] for key in ("status", "locality", "retention") if isinstance(payload.get(key), str)}
    issues.append({"code": "LEGACY_SCOPE_ONLY", "severity": "warning", "path": "payload", "message": "Legacy health/retention flags do not establish adequate optimization, mechanism null, or scientific falsification."})
    valid = not any(x["severity"] == "error" for x in issues)
    return {"valid": valid, "schema_valid": valid,
            "optimizer_health": "healthy" if flag is False else "unknown",
            "optimization_adequacy": "unknown", "legacy_optimization_unresolved": flag,
            "legacy_outcomes": retained, "scientific_claim": "unresolved",
            "science_decision": "unresolved", "issues": issues}
