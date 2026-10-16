import json
import sys
import time
from pathlib import Path

import pytest

from experiments.closure_discovery_support import (
    FAULT_DURATIONS_SECONDS,
    FAULT_XF_VALUES,
    FIXED_LIBRARY_TERMS,
    InsufficientBudget,
    PHASE_CAPS_SECONDS,
    ProcessLedger,
    build_new_fault_manifest,
    estimate_stage_costs,
    fault_condition_key,
    fixed_library,
    generate_mutation_pool,
    nondominated_front,
    paired_group_bootstrap_ci,
    select_depth,
    select_discrepant_train_conditions,
    select_mutation_shortlist,
    simplest_within_one_se,
    term_key,
)


def test_fixed_library_has_registered_nine_terms_and_is_fresh():
    library = fixed_library()
    assert len(library) == 9 == len(FIXED_LIBRARY_TERMS)
    assert [term["channel"] for term in library].count("node") == 1
    assert [term["channel"] for term in library].count("edge") == 2
    assert [term["channel"] for term in library].count("port") == 6
    library[0]["expression"]["op"] = "mutated"
    assert fixed_library()[0]["expression"]["op"] == "mul"


def _timings(max_step=0.01, eval_b=20.0):
    return {
        "max_measured_step_s": max_step,
        "known_eval_cost_s": {"B": eval_b, "C": 10.0, "D": 30.0},
        "fom_cost_s": 0.02,
    }


def test_depth_estimate_uses_phase_specific_costs_and_explicit_six_runs():
    costs = estimate_stage_costs(2, _timings())
    assert costs["B"] == pytest.approx(1.25 * (6912 * 2 * 0.01 + 20))
    assert costs["C"] == pytest.approx(1.25 * (14 * 1024 * 0.01 + 10))
    assert costs["D"] == pytest.approx(1.25 * (14 * 256 * 0.01 + 96 * 0.02 + 30))
    assert select_depth(_timings())["depth"] == 3
    assert select_depth(_timings(max_step=3.2, eval_b=10000))["depth"] == 2
    depth_one = _timings(max_step=3.2, eval_b=40_000)
    depth_one["known_eval_cost_s"]["C"] = 0
    assert select_depth(depth_one)["depth"] == 1
    impossible = _timings(max_step=0, eval_b=PHASE_CAPS_SECONDS["B"])
    assert select_depth(impossible)["depth"] is None


def test_depth_cost_rejects_shared_scalar_eval_that_could_double_count():
    with pytest.raises(TypeError):
        estimate_stage_costs(1, {"max_measured_step_s": 0.1, "fom_cost_s": 0, "known_eval_cost_s": 0})


def test_process_ledger_persists_failed_nested_and_external_attempts_once(tmp_path):
    ledger = ProcessLedger(tmp_path / "ledger")
    with ledger.run("B", "outer") as outer_id:
        with ledger.run("B", "inner"):
            time.sleep(0.002)
    ledger.record_external("C", "external", 2.5, status="failed", receipt_id="external-1")

    rows = [json.loads(line) for line in (tmp_path / "ledger" / "receipts.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    outer = next(row for row in rows if row["receipt_id"] == outer_id)
    inner = next(row for row in rows if row["parent_receipt_id"] == outer_id)
    assert outer["counted"] is True
    assert inner["counted"] is False
    assert rows[2]["status"] == "failed"
    assert ledger.spent("C") == pytest.approx(2.5)
    assert ledger.spent() == pytest.approx(outer["elapsed_seconds"] + 2.5)
    state = json.loads((tmp_path / "ledger" / "state.json").read_text())
    assert state["receipt_count"] == 3
    assert state["stage_spent_seconds"]["B"] == pytest.approx(outer["elapsed_seconds"])


def test_process_ledger_refuses_stage_transfer_and_preserves_overrun(tmp_path):
    ledger = ProcessLedger(tmp_path / "ledger")
    ledger.record_external("B", "prior", PHASE_CAPS_SECONDS["B"])
    with pytest.raises(InsufficientBudget) as exc:
        ledger.require_budget("B", 1.0)
    assert exc.value.stage == "B"
    assert ledger.remaining("C") == PHASE_CAPS_SECONDS["C"]
    with pytest.raises(ValueError):
        ledger.record_external("reserve", "forbidden", 1.0)
    ledger.record_external("reserve", "repair:numeric", 1.0)
    assert ledger.spent("reserve") == pytest.approx(1.0)
    ledger.record_external("D", "overrun", PHASE_CAPS_SECONDS["D"] + 3.0)
    assert ledger.spent("D") == PHASE_CAPS_SECONDS["D"] + 3.0
    rows = [json.loads(line) for line in (tmp_path / "ledger" / "receipts.jsonl").read_text().splitlines()]
    assert rows[-1]["over_budget_after_receipt"] is True


def test_mutation_pool_constructs_real_ast_edits_and_shortlist_is_balanced():
    experiments_dir = str(Path(__file__).resolve().parents[1] / "experiments")
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    from symbolic_closure_rom import term_key as model_term_key, validate_expression

    parent = fixed_library()
    pool = generate_mutation_pool(parent, pool_size=32, seed=11)
    assert len(pool) == 32
    kinds = {row["mutation"] for row in pool}
    assert kinds == {"add", "delete", "replace"}
    assert all(len(row["terms"]) in (8, 9, 10) for row in pool)
    baseline_keys={term_key(t) for t in parent}
    assert any(any(term_key(t) not in baseline_keys for t in row['terms']) for row in pool)
    assert any("tanh" in json.dumps(row["terms"]) for row in pool)
    scores = {row["candidate_key"]: i for i, row in enumerate(reversed(pool))}
    chosen = select_mutation_shortlist(pool, scores, max_total=6, seed=9)
    assert len(chosen) == 6
    assert {row["mutation"] for row in chosen} == kinds
    assert {row["mutation"]: sum(r["mutation"] == row["mutation"] for r in chosen) for row in chosen} == {
        "add": 2,
        "delete": 2,
        "replace": 2,
    }
    assert term_key(parent[0]) == term_key(FIXED_LIBRARY_TERMS[0])
    assert all(term_key(term) == model_term_key(term) for term in parent)
    assert all(
        term_key(term) == model_term_key(term) and validate_expression(term["expression"], term["channel"])
        for proposal in pool
        for term in proposal["terms"]
    )
    assert generate_mutation_pool(parent, pool_size=32, seed=11) == pool


def test_runner_mutation_api_handles_multiple_beam_parents_and_modes():
    from experiments.closure_discovery_support import propose_mutations

    fixed = fixed_library()
    beams = [fixed[:3], fixed[3:6]]
    library_pool = propose_mutations(beams, mode="library", seed=7, max_candidates=64)
    assert library_pool
    assert {candidate["parent"] for candidate in library_pool} == {0, 1}
    fixed_keys = {term_key(term) for term in fixed}
    assert all(term_key(term) in fixed_keys for row in library_pool for term in row["terms"])
    assert all(row["mode"] == "library" for row in library_pool)

    generated_pool = propose_mutations(beams, mode="generated", seed=7, max_candidates=64)
    assert len(generated_pool) == 64
    assert {row["parent"] for row in generated_pool} == {0, 1}
    assert all(row["mode"] == "generated" for row in generated_pool)
    assert all(len([r for r in generated_pool if r["parent"] == parent]) <= 32 for parent in (0, 1))


def test_pareto_one_se_and_event_group_bootstrap():
    front = nondominated_front(
        [
            {"name": "a", "error": 1.0, "complexity": 5},
            {"name": "b", "error": 1.2, "complexity": 3},
            {"name": "c", "error": 1.4, "complexity": 6},
        ]
    )
    assert {row["name"] for row in front} == {"a", "b"}
    selected = simplest_within_one_se(
        [
            {"name": "best", "error": 1.0, "se": 0.1, "complexity": 8},
            {"name": "simple", "error": 1.08, "se": 0.3, "complexity": 2},
            {"name": "too_worse", "error": 1.2, "se": 0.01, "complexity": 1},
        ]
    )
    assert selected["name"] == "simple"
    first = paired_group_bootstrap_ci({"e1": [1.0, 3.0], "e2": -1.0, "e3": 2.0}, seed=27)
    second = paired_group_bootstrap_ci({"e1": [1.0, 3.0], "e2": -1.0, "e3": 2.0}, seed=27)
    assert first == second
    assert first["event_count"] == 3
    assert first["mean_difference"] == pytest.approx((2 - 1 + 2) / 3)
    assert len(first["ci95"]) == 2


def test_fault_manifest_uses_only_allowed_bus_metadata_and_keeps_frozen_val():
    allowed = [{"bus_id": f"train-{i}"} for i in range(8)]
    manifest = build_new_fault_manifest(allowed, existing_pairs=[] , seed=4)
    val = manifest["val_conditions"]
    assert len(val) == 24
    assert [sum(row["stratum"] == name for row in val) for name in ("short", "medium", "long")] == [8, 8, 8]
    assert all(row["bus_id"] in {r["bus_id"] for r in allowed} for row in val)
    assert all(row["duration_s"] in FAULT_DURATIONS_SECONDS and row["xf"] in FAULT_XF_VALUES for row in val)
    val_keys = {(row["bus_id"], row["duration_s"], row["xf"]) for row in val}
    pool = manifest["train_candidate_pool"]
    assert len(pool) >= 24
    assert val_keys.isdisjoint((row["bus_id"], row["duration_s"], row["xf"]) for row in pool)
    scores = {fault_condition_key(row): float(i) for i, row in enumerate(pool)}
    selected = select_discrepant_train_conditions(pool, scores, count=24)
    assert len(selected) == 24
    assert all(row["discrepancy_score"] in scores.values() for row in selected)

    frozen = json.loads(json.dumps(val))
    rebuilt = build_new_fault_manifest(allowed, [], frozen_val_conditions=frozen, seed=4)
    assert rebuilt["val_conditions"] == frozen
    assert build_new_fault_manifest(allowed, [], seed=4)["val_conditions"] == val


def test_fault_manifest_excludes_existing_pairs_and_fails_when_not_enough_allowed_buses():
    allowed = ["a", "b", "c", "d"]
    manifest = build_new_fault_manifest(allowed, [("a", 0.075, 0.001), (0.125, 0.01)], seed=0)
    assert all((row["bus_id"], row["duration_s"], row["xf"]) != ("a", 0.075, 0.001) for row in manifest["val_conditions"])
    assert all((row["duration_s"], row["xf"]) != (0.125, 0.01) for row in manifest["val_conditions"])
    with pytest.raises(ValueError):
        build_new_fault_manifest(["only-one"], [], seed=0)
