"""Offline, deterministic, CPU-only tests for the read-only closure probe.

No frozen prediction file and no model checkpoint is loaded here: the heavy P1/P2 work is
exercised by the CLI, not by the unit tests.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]

import closure_probe_20261030 as probe  # noqa: E402
from train_local_memory_0098 import risk  # noqa: E402
from train_structured_rom import _point_error  # noqa: E402


def _scales():
    generator = torch.Generator().manual_seed(20261030)
    mass = torch.rand(10, dtype=torch.float64, generator=generator) + 0.2
    p_scale = torch.rand(10, dtype=torch.float64, generator=generator) + 0.3
    return SimpleNamespace(mass=mass, p_scale=p_scale)


# --------------------------------------------------------------------- P1 terms


@pytest.mark.parametrize("shape", [(13, 30), (5, 40), (1, 30), (2, 30)])
def test_four_terms_sum_to_registered_point_error(shape):
    generator = torch.Generator().manual_seed(1 + shape[0] * shape[1])
    pred = torch.randn(*shape, dtype=torch.float64, generator=generator)
    truth = torch.randn(*shape, dtype=torch.float64, generator=generator)
    model = _scales()
    total = _point_error(pred, truth, model)
    terms = probe.point_error_terms(pred, truth, model)
    assert terms.shape == shape[:-1] + (probe.N_TERMS,)
    torch.testing.assert_close(terms.sum(-1), total, atol=1e-12, rtol=0)


def test_term_order_is_registered():
    assert probe.TERM_NAMES == ("delta_coi", "omega_coi", "common_omega", "p_normalized")


# ------------------------------------------------------- fine/coarse alignment


def test_fine_and_coarse_indexing():
    fine = np.arange(1601 * 40, dtype=np.float64).reshape(1601, 40)
    coarse = np.arange(801 * 40, dtype=np.float64).reshape(801, 40)
    steps = 500
    np.testing.assert_array_equal(probe.fine_slice(fine, steps), fine[::2][1 : steps + 1, :30])
    np.testing.assert_array_equal(probe.fine_slice(fine, steps), fine[2 : 2 * steps + 1 : 2, :30])
    np.testing.assert_array_equal(probe.coarse_slice(coarse, steps), coarse[1 : steps + 1, :30])
    event = {"id": "e", "label": "stable", "x": np.arange(1611 * 30, dtype=np.float64).reshape(1611, 30)}
    np.testing.assert_array_equal(probe.truth_slice(event, steps), event["x"][11 : 11 + steps, :30])


def test_scope_steps_registered():
    long_event = {"id": "s", "label": "stable", "x": np.zeros((1611, 30))}
    assert probe.scope_steps(long_event, "5") == 500
    assert probe.scope_steps(long_event, "8") == 800
    assert probe.scope_steps(long_event, "prefix") == 1500
    short_event = {"id": "u", "label": "unstable", "x": np.zeros((47, 30))}
    assert probe.scope_steps(short_event, "prefix") == 36
    assert probe.scope_steps(short_event, "5") is None
    with pytest.raises(ValueError):
        probe.scope_steps(short_event, "2")


# ------------------------------------------------------------ denominators


def _event(event_id, label, length):
    return {"id": event_id, "label": label, "x": np.zeros((length, 30), dtype=np.float64)}


def test_stable_unstable_cohort_selection():
    events = [
        _event("s_complete8", "stable", 1611),
        _event("s_exact8", "stable", 811),
        _event("s_five_only", "stable", 611),
        _event("u_long", "unstable", 1611),
        _event("u_short", "unstable", 20),
    ]
    assert [event["id"] for event in probe.cohort(events, "5")] == ["s_complete8", "s_exact8", "s_five_only"]
    assert [event["id"] for event in probe.cohort(events, "8")] == ["s_complete8", "s_exact8"]
    assert [event["id"] for event in probe.cohort(events, "prefix")] == ["u_long", "u_short"]
    assert probe.scope_steps(_event("u_short", "unstable", 20), "prefix") == 9
    assert probe.scope_steps(_event("u_long", "unstable", 1611), "prefix") == 1500


def test_registered_denominators_are_frozen_constants():
    assert (probe.EXPECTED_EVENTS, probe.EXPECTED_STABLE, probe.EXPECTED_UNSTABLE) == (297, 267, 30)
    assert len(probe.PUBLIC_VALUES) == 12
    assert probe.ACCEPTANCE_ATOL == 1e-9


# --------------------------------------------------------------- risk aggregation


def test_risk_matches_hand_calculation_and_source():
    values = np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    k = max(1, int(np.ceil(0.1 * len(values))))
    hand = 0.8 * values.mean() + 0.2 * np.sort(values)[-k:].mean()
    assert probe.risk_aggregate(values) == pytest.approx(float(hand), rel=1e-15, abs=0.0)
    assert probe.risk_aggregate(values) == float(risk(torch.tensor(values, dtype=torch.float64)))


def test_risk_tail_uses_ceil_ten_percent():
    values = np.arange(1.0, 21.0)  # n=20 -> ceil(2.0)=2
    hand = 0.8 * values.mean() + 0.2 * np.array([20.0, 19.0]).mean()
    assert probe.risk_aggregate(values) == pytest.approx(float(hand), rel=1e-15, abs=0.0)


def test_pooled_rmse_matches_hand_calculation():
    rows = [
        {"common_omega_sse": 4.0, "common_omega_n": 2, "normalized_p_sse": 9.0, "normalized_p_n": 3},
        {"common_omega_sse": 6.0, "common_omega_n": 3, "normalized_p_sse": 3.0, "normalized_p_n": 1},
    ]
    out = probe.pooled_8s_rmse(rows)
    assert out["common_omega_rmse_8s"] == pytest.approx(math.sqrt(10.0 / 5.0), rel=1e-15)
    assert out["normalized_p_rmse_8s"] == pytest.approx(math.sqrt(12.0 / 4.0), rel=1e-15)
    assert out["common_omega_n"] == 5 and out["normalized_p_n"] == 4


def test_rmse_gate_constants():
    assert len(probe.RMSE_GATES) == 4
    assert probe.RMSE_GATES[("A", "common_omega_rmse_8s")] == 0.32140212114056005
    assert probe.RMSE_GATES[("R_N", "normalized_p_rmse_8s")] == 0.4426070157702325
    assert probe.TERM_SUM_ATOL == 1e-12


# ------------------------------------------------------------------ P2 diagnostics


def _synthetic_system(n=4, ridge=1e-4, seed=11, zero_design=False):
    generator = torch.Generator().manual_seed(seed)
    if zero_design:
        A = torch.zeros(n, 50, 10, dtype=torch.float64)
        F = torch.eye(10, dtype=torch.float64).expand(n, 10, 10).clone()
    else:
        A = torch.randn(n, 50, 10, dtype=torch.float64, generator=generator)
        F = torch.eye(10, dtype=torch.float64).expand(n, 10, 10).clone()
        F = F + 0.05 * torch.randn(n, 10, 10, dtype=torch.float64, generator=generator)
    beta = torch.randn(n, 50, dtype=torch.float64, generator=generator)
    q = torch.randn(n, 10, dtype=torch.float64, generator=generator)
    rstar = torch.rand(10, dtype=torch.float64, generator=generator) + 0.2
    design = torch.cat((A, math.sqrt(ridge) * F), dim=-2)
    target = torch.cat((beta, -math.sqrt(ridge) * q), dim=-1)
    r0 = rstar[None, :] * probe.qr_solve(design, target)
    return {"A": A, "F": F, "bscaled": beta, "q": q, "r0": r0}, rstar


def test_history_system_diagnostics_shapes_and_finiteness():
    system, rstar = _synthetic_system()
    out = probe.analyze_history_system(
        system,
        rstar,
        ridge=1e-4,
        event_ids=["e0", "e1", "e2", "e3"],
        labels=["stable", "stable", "unstable", "unstable"],
    )
    assert out["design"]["shape"] == [4, 60, 10]
    assert out["spectrum"]["sigma_min"]["count"] == 4
    assert out["spectrum"]["sigma_max"]["count"] == 4
    assert out["spectrum"]["condition"]["count"] == 4
    for entry in (out["spectrum"]["sigma_min"], out["spectrum"]["condition"], out["ridge_contribution"]["stats"]):
        assert all(math.isfinite(entry[key]) for key in ("min", "median", "p90", "max"))
    assert out["spectrum"]["condition"]["min"] >= 1.0
    assert 1 <= out["spectrum"]["effective_rank"]["min"] <= 10
    assert set(out["scan"]) == {"0.0001", "1e-06", "1e-08"}
    for entry in out["scan"].values():
        assert entry["relative_change"]["count"] == 4
        assert entry["relative_change"]["max"] >= 0.0
    assert out["counts"]["events"] == 4
    assert out["counts"]["ridge_dominated"] == 0
    assert out["grouped"]["stable"]["events"] == 2
    assert out["grouped"]["unstable"]["events"] == 2
    assert out["r0"]["reconstruction_passed"] is True
    assert out["r0"]["reconstruction_max_abs_diff"] <= 1e-12
    assert len(out["per_event"]) == 4
    assert all(math.isfinite(row["r0_norm"]) for row in out["per_event"])


def test_ridge_only_system_is_ridge_dominated_and_full_rank():
    system, rstar = _synthetic_system(n=2, zero_design=True)
    out = probe.analyze_history_system(system, rstar, ridge=1e-4, scans=(1e-4, 1e-6, 1e-8))
    assert out["spectrum"]["sigma_min"]["median"] == pytest.approx(1e-2, rel=1e-12)
    assert out["spectrum"]["sigma_max"]["median"] == pytest.approx(1e-2, rel=1e-12)
    assert out["spectrum"]["effective_rank"]["max"] == 10
    assert out["counts"]["effective_rank_lt_full"] == 0
    assert out["ridge_contribution"]["stats"]["median"] == pytest.approx(1.0, abs=1e-12)
    assert out["counts"]["ridge_dominated"] == 2


def test_compare_initializers_relative_change_and_grouping():
    generator = torch.Generator().manual_seed(17)
    r0 = torch.randn(4, 10, dtype=torch.float64, generator=generator)
    rstar = torch.rand(10, dtype=torch.float64, generator=generator) + 0.5
    scaled = r0 * 1.01
    systems = {
        "A128": {"r0": r0, "s": r0 / rstar[None, :], "rstar": rstar},
        "T": {"r0": scaled, "s": scaled / rstar[None, :], "rstar": rstar.clone()},
    }
    out = probe.compare_initializers(systems, labels=["stable", "unstable", "stable", "unstable"])
    assert out["baseline"] == "A128"
    assert out["rstar_identical_across_initializers"] is True
    comparison = out["comparisons"]["T"]
    assert comparison["r0_relative_change"]["median"] == pytest.approx(0.01, rel=1e-12)
    assert comparison["coefficient_relative_change"]["median"] == pytest.approx(0.01, rel=1e-12)
    expected_drift = float(np.median(np.abs((0.01 * r0.sum(-1) / 10.0).numpy())))
    assert comparison["abs_delta_sum_r0_over_10"]["median"] == pytest.approx(expected_drift, rel=1e-12)
    assert comparison["grouped"]["stable"]["events"] == 2
    assert comparison["grouped"]["unstable"]["events"] == 2
    assert out["evidence_class"].startswith("diagnostic")


# --------------------------------------------------------------------- public


def _minimal_p1():
    return {
        "schema": probe.P1_SCHEMA,
        "status": "COMPLETED",
        "source": {
            "checkpoints": {},
            "prediction_manifest_sha256": "0" * 64,
            "prediction_files": {"A": [{}, {}]},
            "scales": {},
        },
        "acceptance": {"passed": True, "rows": []},
        "models": {
            "A": {
                "scopes": {"5": {"risk_total": 1.0}},
                "per_event": {"5": {"e0": {"total": 1.0}}},
                "unresolved": {"5": []},
            }
        },
        "gaps": {},
        "denominators": {},
        "per_event": {"A": {"5": {"e0": {"total": 1.0}}}},
    }


def _minimal_p2():
    return {
        "schema": probe.P2_SCHEMA,
        "status": "COMPLETED",
        "model": {"checkpoint": "frozen"},
        "diagnostics": {
            "design": {},
            "spectrum": {},
            "ridge_contribution": {},
            "scan": {},
            "counts": {},
            "grouped": {},
            "r0": {},
            "interpretation_scope": "spectral evidence only",
        },
        "per_event": [{"event_id": "e0"}],
    }


def test_public_payload_strips_event_level_fields():
    payload = probe.public_payload(_minimal_p1(), _minimal_p2())
    text = json.dumps(payload)
    assert "per_event" not in text
    assert "e0" not in text
    assert '"histories":' not in text
    probe._assert_public(payload)


def test_public_refuses_failed_gate():
    p1 = _minimal_p1()
    p1["acceptance"]["passed"] = False
    with pytest.raises(ValueError):
        probe.public_payload(p1, _minimal_p2())


def test_public_validator_rejects_long_arrays():
    with pytest.raises(ValueError):
        probe._assert_public({"ok": list(range(65))})


# ------------------------------------------------------------------ CLI / read-only


def test_cli_parses_registered_commands():
    parser = probe.build_parser()
    args = parser.parse_args(["decompose", "--output", "private_data/closure_probe_20261030/p1.json"])
    assert args.command == "decompose" and args.output == Path("private_data/closure_probe_20261030/p1.json")
    args = parser.parse_args(["init-conditioning", "--output", "private_data/closure_probe_20261030/p2.json"])
    assert args.command == "init-conditioning"
    args = parser.parse_args(
        [
            "publish",
            "--p1",
            "private_data/closure_probe_20261030/p1.json",
            "--p2",
            "private_data/closure_probe_20261030/p2.json",
            "--output",
            "results/closure_probe_20261030/public_probe.json",
        ]
    )
    assert args.command == "publish"
    assert args.p1.name == "p1.json" and args.p2.name == "p2.json"


def test_output_guard_confines_writes():
    expected = (ROOT / "private_data/closure_probe_20261030/p1.json").resolve()
    assert probe.guard_output("private_data/closure_probe_20261030/p1.json", probe.PRIVATE_ROOT) == expected
    with pytest.raises(ValueError):
        probe.guard_output("experiments/closure_probe_out.json", probe.PRIVATE_ROOT)
    with pytest.raises(ValueError):
        probe.guard_output("private_data/other_place/p1.json", probe.PRIVATE_ROOT)
    with pytest.raises(ValueError):
        probe.guard_output("results/closure_probe_20261030/public_probe.json", probe.PRIVATE_ROOT)
    with pytest.raises(ValueError):
        probe.guard_output("private_data/closure_probe_20261030/p1.txt", probe.PRIVATE_ROOT)


def test_publish_input_guard_rejects_foreign_inputs():
    with pytest.raises(ValueError):
        probe.guard_input("private_data/state_gated_memory_20261028/study01/prepare.json", probe.PRIVATE_ROOT)
    with pytest.raises(ValueError):
        probe.guard_input("results/closure_probe_20261030/public_probe.json", probe.PRIVATE_ROOT)


def test_main_rejects_outside_output_before_doing_work():
    with pytest.raises(ValueError):
        probe.main(["decompose", "--output", "experiments/closure_probe_out.json"])


def test_data_sources_are_fit_only():
    assert probe.PRIVATE_ROOT == (ROOT / "private_data/closure_probe_20261030")
    assert probe.RESULTS_ROOT == (ROOT / "results/closure_probe_20261030")
    for path in probe.DATA_SOURCE_PATHS:
        parts = {part.lower() for part in Path(path).parts}
        assert "val" not in parts
        assert "test" not in parts
    assert probe.SGM_STUDY.name == "study01"
    assert "fit297" in probe.MODEL_SPECS["R_N"]["predictions"].name
