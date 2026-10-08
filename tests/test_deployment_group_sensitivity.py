"""Synthetic contract tests for Loop0066 group-sensitivity evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
import audit_deployment_group_sensitivity as audit  # noqa: E402

from train_long_latent import (  # noqa: E402
    assert_inactive_xi_unchanged,
    sample_batch,
    select_training_events,
)
from audit_deployment_group_sensitivity import (  # noqa: E402
    aggregate,
    common,
    midpoint_samples,
    safe_aggregate,
    score,
    frozen_check,
    registered_classification,
    scored_rows,
)


def _events():
    x20 = np.arange(20 * 33, dtype=float).reshape(20, 33)
    x25 = np.arange(25 * 33, dtype=float).reshape(25, 33)
    x18 = np.arange(18 * 33, dtype=float).reshape(18, 33)
    return [
        {"id": "a", "group": "31", "label": "stable", "x": x20},
        {"id": "b", "group": "31", "label": "unstable", "x": x25},
        {"id": "c", "group": "7", "label": "stable", "x": x18},
    ]


def test_training_filter_preserves_objects_and_excludes_only_requested_group():
    events = _events()
    fit, excluded = select_training_events(events, "31")
    assert [e["id"] for e in fit] == ["c"]
    assert [e["id"] for e in excluded] == ["a", "b"]
    assert [e["id"] for e in events] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        select_training_events(events, "missing")
    with pytest.raises(ValueError):
        select_training_events([events[0]], "31")


def test_sampler_and_midpoint_use_genuine_histories_without_future_leak():
    events = _events()
    samples, histories = sample_batch(events, np.random.default_rng(3), count=100)
    assert histories.shape == (100, 11, 33)
    assert all(10 <= endpoint <= len(event["x"]) - 2 for event, endpoint in samples)
    assert all(np.array_equal(history, event["x"][endpoint - 10:endpoint + 1])
               for (event, endpoint), history in zip(samples, histories, strict=True))
    panel = midpoint_samples(events)
    assert [endpoint for _, endpoint in panel] == [14, 16, 13]
    for event, endpoint in panel:
        assert endpoint + 1 < len(event["x"])


def test_prefix_caps_all_short_events_and_excludes_initial_output():
    mass = np.ones(10)
    truth = np.zeros((6, 33)); pred = np.zeros_like(truth); pred[0, :20] = 999.0
    result = score(pred, truth, mass, 5)
    assert result["n"] == 50
    assert result["rmse"] == pytest.approx(0.0)
    short = score(pred[:3], truth[:3], mass, 5)
    assert short["n"] == 20
    assert common(pred, truth, mass, 5)["n"] == 5


def test_event_mean_pooled_rmse_and_cvar10_are_distinct():
    rows = [
        {"event_id": "a", "rmse": 1.0, "normalized_sse": 10.0, "n": 10},
        {"event_id": "b", "rmse": 3.0, "normalized_sse": 90.0, "n": 10},
    ]
    summary = aggregate(rows)
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["pooled"] == pytest.approx(np.sqrt(5.0))
    assert summary["cvar10"] == pytest.approx(3.0)


def test_disagreement_uses_same_prefix_score_and_no_initial_state():
    mass = np.ones(10)
    truth = np.zeros((5, 33)); A = np.zeros_like(truth); B = np.zeros_like(truth)
    A[0, :20] = 1000.0; B[0, :20] = -1000.0
    B[1:, 10] = 1.0
    disagreement = score(B - A, np.zeros_like(truth), mass, 5)
    assert disagreement["n"] == 40
    expected = np.sqrt((4 * .9) / (4 * 10 * 1.0033667249602924**2) / 2)
    assert disagreement["rmse"] == pytest.approx(expected)


def test_inactive_xi_entries_are_unchanged():
    class Model:
        support = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        xi = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    source = {"xi": torch.tensor([[2.0, 3.0], [4.0, 5.0]])}
    assert_inactive_xi_unchanged(Model(), source)
    source["xi"][0, 1] = 9.0
    with pytest.raises(AssertionError):
        assert_inactive_xi_unchanged(Model(), source)


def test_primary_capped_five_is_distinct_from_eight_second_prefix_and_complete():
    mass = np.ones(10)
    truth = np.zeros((801, 33))
    pred = np.zeros_like(truth)
    pred[501:, 10] = 2.0
    # The registered primary is capped at five seconds, so post-5 s error
    # cannot affect it; an 8 s prefix must expose the same error.
    five = score(pred, truth, mass, 5)
    eight = score(pred, truth, mass, 8)
    assert five["rmse"] == pytest.approx(0.0)
    assert eight["rmse"] > 0.0
    assert score(pred, truth, mass, 8, complete=True)["n"] == 8000
    short = np.zeros((500, 33))
    assert score(short, short, mass, 5, complete=True) is None  # exact endpoint is not complete


def test_empty_cohort_is_explicitly_unavailable():
    assert safe_aggregate([]) == {"n_events": 0, "mean": None, "pooled": None, "cvar10": None}
    assert safe_aggregate([])["n_events"] == 0


def test_unequal_event_counts_use_event_mean_and_n_weighted_pooling():
    rows = [
        {"event_id": "short", "rmse": 1.0, "normalized_sse": 1.0, "n": 1},
        {"event_id": "long", "rmse": 3.0, "normalized_sse": 27.0, "n": 3},
    ]
    summary = aggregate(rows)
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["pooled"] == pytest.approx(np.sqrt(7.0))
    assert summary["cvar10"] == pytest.approx(3.0)


def test_classification_precedence_is_numerical_then_optimization_then_ratios():
    ratios = {"mean": 1.0, "pooled": 1.0, "cvar10": 1.0}
    assert registered_classification(False, False, ratios, 0.0) == "numerical_unresolved"
    assert registered_classification(True, False, ratios, 0.0) == "optimization_unresolved"
    assert registered_classification(True, True, ratios, 0.0) == "supported"
    assert registered_classification(True, True, {"mean": 1.11, "pooled": 1.0, "cvar10": 1.0}, 0.0) == "rejected"


def test_scored_rows_contains_complete_only_when_available_and_both_prefixes():
    events = [
        {"id": "long-a", "group": "31", "label": "stable", "x": np.zeros((120, 33))},
        {"id": "long-b", "group": "31", "label": "unstable", "x": np.zeros((120, 33))},
    ]
    truths = [e["x"] for e in events]
    predictions = {"A": truths, "B": truths}
    rows, commonrows = scored_rows(predictions, events, truths, np.ones(10))
    assert len([r for r in rows if r["scope"] == "complete" and r["horizon"] == "1"]) == 4
    assert len([r for r in rows if r["scope"] == "complete" and r["horizon"] == "8"]) == 0
    assert len([r for r in rows if r["scope"] == "capped5_prefix"]) == 4
    assert len([r for r in rows if r["scope"] == "capped8_prefix"]) == 4
    assert len(commonrows) == 8


def test_run_stub_builds_all_arms_selects_refinement_and_persists_schema(monkeypatch, tmp_path):
    events = []
    for i in range(346):
        events.append({"id": f"e{i}", "group": "31" if i < 13 else "7",
                       "label": "unstable" if i % 2 else "stable",
                       "x": np.zeros((120, 30))})

    encode_calls = []
    class Model:
        def __init__(self, name): self.name = name
        def encode(self, histories):
            encode_calls.append(self.name)
            return torch.zeros((len(histories), 3), dtype=torch.float64)

    state = {"xi": torch.zeros((4, 3)), "support": torch.ones((4, 3), dtype=torch.bool)}
    monkeypatch.setattr(audit, "sha", lambda path: audit.EXPECTED_SOURCE)
    monkeypatch.setattr(audit.MechanicalROM, "load", lambda path: object())
    monkeypatch.setattr(audit, "_construct", lambda directory, base, network, role: Model(str(directory)))
    monkeypatch.setattr(audit, "frozen_check", lambda source, arm: (state, state))
    monkeypatch.setattr(audit, "read_development", lambda base, network, full_train: ({"train": events}, [], []))
    monkeypatch.setattr(audit, "panel_loss", lambda model, es: {
        "total": 0.0 if "A" in model.name or "B" in model.name else 1.0,
        "finite": True, "event_count": len(es), "event_ids": [e["id"] for e in es],
        "endpoints": [14 for _ in es]})
    monkeypatch.setattr(audit.stageb, "params", lambda source, matrices: {
        "mass": np.ones(10), "support": np.ones((4, 3), dtype=bool)})
    monkeypatch.setattr(audit, "analytic_jacobian", lambda p: np.eye(33))
    monkeypatch.setattr(audit, "gauge_maps", lambda: (np.c_[np.eye(32), np.zeros(32)],
                                                        np.r_[np.eye(32), np.zeros((1, 32))]))
    monkeypatch.setattr(audit, "live_equivalence", lambda model, p, initial: {
        "field_max_abs": 0.0, "one_rk4_max_abs": 0.0})
    original_np_load = audit.np.load
    monkeypatch.setattr(audit.np, "load", lambda path, allow_pickle=False: {
        "load_buses": np.array([0]), "generator_buses": np.array([1])})
    def fake_scope_values(*args):
        h = args[-1]
        companion = float(np.asarray(args[1]).flat[0])
        return {
        "available_prefix": {"max": 0.002 if h == 5 and companion == pytest.approx(.005) else 0.0},
        "complete": {"max": 0.0}}
    monkeypatch.setattr(audit.stageb, "scope_values", fake_scope_values)
    original_rollout = audit.core.decoded_rollout
    def fake_rollout(initial, horizon, p, dt):
        prediction = np.zeros((len(initial), 801, 33))
        prediction[:, :, 0] = dt
        return prediction
    monkeypatch.setattr(audit.core, "decoded_rollout", fake_rollout)
    try:
        report = audit.run(Path("synthetic-S"), Path("synthetic-A"), Path("synthetic-B"), tmp_path / "out")
    finally:
        monkeypatch.setattr(audit.core, "decoded_rollout", original_rollout)
    assert set(report["selected_grid"]) == {"S", "A", "B"}
    assert all(grid == ".00125" for grid in report["selected_grid"].values())
    assert encode_calls == ["synthetic-S", "synthetic-A", "synthetic-B"]
    assert report["train_counts"] == {"all": 346, "complement31": 333, "group31": 13}
    assert report["adequacy"]["numerical_valid"] is True
    # A one-generator angle error survives COI projection; a common shift
    # would be zero at both grids and would not test actual score selection.
    expected = np.sqrt(.5 * .9 / 10) * .00125 / .47675314588683515
    assert report["summary"]["S"]["capped5_prefix"]["pooled"] == pytest.approx(expected)
    npz_path = tmp_path / "out" / "group31_predictions.npz"
    assert npz_path.exists()
    audit.np.load = original_np_load
    saved = np.load(npz_path, allow_pickle=False)
    assert {"S_fine", "S_coarse", "S_initial", "S_Xi", "e0_truth"}.issubset(saved.files)
    assert {"J32_S", "J32_A", "J32_B"}.issubset(saved.files)
    assert np.all(saved["S_fine"][:, :, 0] == .0025)
    assert np.all(saved["S_finer"][:, :, 0] == .00125)
    assert np.all(saved["S_coarse"][:, :, 0] == .005)
    assert (tmp_path / "out" / "evidence.json").exists()


def test_frozen_check_requires_inactive_xi_and_entire_normalization_to_match(tmp_path):
    source = tmp_path / "source"; arm = tmp_path / "arm"
    source.mkdir(); arm.mkdir()
    support = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    base = {"support": support, "xi": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "frozen": torch.tensor([7.0])}
    changed_active = {**base, "xi": torch.tensor([[9.0, 2.0], [3.0, 4.0]])}
    torch.save(base, source / "model.pt"); torch.save(changed_active, arm / "model.pt")
    np.savez(source / "normalization.npz", mean=np.array([1.0, 2.0]), scale=np.array([3.0, 4.0]))
    np.savez(arm / "normalization.npz", mean=np.array([1.0, 2.0]), scale=np.array([3.0, 4.0]))
    frozen_check(source, arm)
    bad_inactive = {**base, "xi": torch.tensor([[9.0, 8.0], [3.0, 4.0]])}
    torch.save(bad_inactive, arm / "model.pt")
    with pytest.raises(AssertionError, match="inactive Xi"):
        frozen_check(source, arm)
    torch.save(changed_active, arm / "model.pt")
    np.savez(arm / "normalization.npz", mean=np.array([1.0, 2.0]), scale=np.array([3.0, 5.0]))
    with pytest.raises(AssertionError, match="normalization"):
        frozen_check(source, arm)
