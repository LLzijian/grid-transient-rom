"""Independent synthetic checks for the frozen KIKI rule port; no source I/O.

Expected LOS endpoints are derived with a scalar hold counter, and expected
RMS values with explicit weighted sums, rather than reusing implementation
helpers. Only constructed arrays are used; no KIKI or TEST payload is opened.
"""
import numpy as np
import pytest

from task4.classification import (
    DECISION_HORIZON_S, DELTA_SCALE, OMEGA_SCALE,
    apply_numerical_gate, classify_forecast, classify_prediction,
    numerical_pair, restore_physical_path, summarize_classifications,
    trigger_index, trigger_times, window_settled, with_observed,
)


OMEGA_S = 2 * np.pi * 60


def canonical(length=801, machines=2):
    return np.zeros((length, machines, 3))


def raw_calm(length=811, machines=2):
    raw = canonical(length, machines)
    raw[..., 1] = 1
    return raw


def add_los(raw, start, stop=None):
    raw[start:stop, 0, 0] = -2
    raw[start:stop, 1, 0] = 2
    raw[start:stop, 0, 1] = 0.97
    raw[start:stop, 1, 1] = 1.03
    return raw


def from_raw(raw):
    out = raw.copy()
    out[..., 1] = (out[..., 1] - 1) * OMEGA_S
    return out


def pair(first, second, **kwargs):
    n = first.shape[1]
    return numerical_pair(first, second, canonical(11, n), np.ones(n),
                          np.zeros(n), OMEGA_S, **kwargs)


@pytest.mark.parametrize("machines", [2, 3, 10, 17])
def test_restore_units_and_individual_equilibrium(machines):
    x = canonical(12, machines)
    x[..., 0] = 0.2
    x[..., 1] = 0.015 * OMEGA_S
    x[..., 2] = 3.0
    equilibrium = np.linspace(-1, 1, machines)
    raw = restore_physical_path(x, equilibrium, OMEGA_S)
    np.testing.assert_allclose(raw[..., 0], np.broadcast_to(equilibrium + 0.2, (12, machines)))
    np.testing.assert_allclose(raw[..., 1], 1.015)
    np.testing.assert_allclose(raw[..., 2], 3)
    assert np.all(x[..., 0] == 0.2)  # The input is not changed.


def test_equilibrium_restoration_changes_physical_los_decision():
    history = canonical(11)
    forecast = canonical()
    forecast[:, :, 1] = np.array([-0.03, 0.03]) * OMEGA_S
    restored = classify_forecast(history, forecast, np.ones(2), np.array([-2.0, 2.0]),
                                 OMEGA_S, numerical_passed=True)
    unshifted = classify_forecast(history, forecast, np.ones(2), np.zeros(2),
                                  OMEGA_S, numerical_passed=True)
    assert restored["trigger_index"] == 14
    assert restored["category"] == "predicted_los"
    assert unshifted["category"] == "window_no_los_unsettled"


def test_los_matches_independent_scalar_counter_with_unequal_inertia():
    mass = np.array([1.0, 2.0, 4.0])
    raw = raw_calm(70, 3)
    raw[4:8, :, 0] = [-2, 2, 0]
    raw[4:8, :, 1] = [0.94, 1.03, 1.0]  # Four is insufficient.
    raw[11:22, :, 0] = [-2, 2, 0]
    raw[11:13, :, 1] = [0.97, 1.0, 1.0]
    raw[13:22, :, 1] = [0.94, 1.03, 1.0]
    expected, run = None, 0
    for t in range(len(raw)):
        angle = max(raw[t, :, 0]) - min(raw[t, :, 0])
        mean = sum(raw[t, i, 1] * mass[i] for i in range(3)) / sum(mass)
        speed = max(abs(raw[t, i, 1] - mean) for i in range(3))
        run = run + 1 if angle > np.pi and speed > 0.02 else 0
        if run == 5:
            expected = t
            break
    assert expected is not None
    assert trigger_index(raw, mass) == expected


def test_los_is_strict_and_with_consecutive_endpoint():
    raw = raw_calm(30)
    raw[:5, :, 0] = [0, np.pi]
    raw[:5, :, 1] = [0.97, 1.03]  # Angle exactly at threshold.
    raw[5:10, :, 0] = [-2, 2]
    raw[5:10, :, 1] = [-0.02, 0.02]  # Relative speed exactly at threshold.
    add_los(raw, 11, 15)  # Only four samples.
    add_los(raw, 17, 22)
    assert trigger_index(raw, np.ones(2)) == 21
    assert trigger_index(raw[:21], np.ones(2)) is None


def test_common_speed_does_not_count_as_relative_speed():
    raw = raw_calm(811, 3)
    raw[..., 0] = [-2, 0, 2]
    raw[..., 1] = 100.0
    assert trigger_index(raw, [1, 3, 8]) is None


def test_observed_hold_counter_and_forecast_initial_state_are_preserved():
    history = raw_calm(11)
    add_los(history, 8)
    forecast = raw_calm(801)
    add_los(forecast, 0)
    history[10] = np.nan  # Replaced by the forecast decision state, not duplicated.
    full = with_observed(from_raw(history), from_raw(forecast))
    assert full.shape == (811, 2, 3)
    assert np.isfinite(full).all()
    result = classify_forecast(from_raw(history), from_raw(forecast), [1, 1], [0, 0],
                               OMEGA_S, numerical_passed=True)
    assert result["trigger_index"] == 12


def test_tail_is_last_100_samples_and_is_not_los_complement():
    raw = raw_calm(811)
    raw[-101, :, 0] = [-2, 2]
    assert window_settled(raw, [1, 1])
    raw[-100, :, 0] = [-2, 2]
    assert not window_settled(raw, [1, 1])
    assert trigger_index(raw, [1, 1]) is None
    assert not window_settled(raw_calm(99), [1, 1])
    result = classify_prediction(raw, [1, 1], numerical_passed=True)
    assert result["category"] == "window_no_los_unsettled"


def test_default_requires_complete_eight_seconds_after_decision():
    assert DECISION_HORIZON_S == 8.0
    result = classify_prediction(raw_calm(810), [1, 1], numerical_passed=True)
    assert result["category"] == "numerical_unresolved"
    assert result["unresolved_reason"] == "incomplete_prediction"
    assert result["required_samples"] == 811
    complete = classify_prediction(raw_calm(811), [1, 1], numerical_passed=True)
    assert complete["category"] == "window_settled"


def test_late_los_is_seen_even_when_true_event_has_a_short_prefix():
    raw = add_los(raw_calm(), 700)
    result = classify_prediction(raw, [1, 1], true_event_time_s=0.25, numerical_passed=True)
    assert result["trigger_index"] == 704
    # Additional samples beyond the registered window cannot change its category.
    too_late = add_los(raw_calm(900), 850)
    assert classify_prediction(too_late, [1, 1])["category"] == "window_settled"


@pytest.mark.parametrize("time", [0.0, 0.09, 0.10])
def test_latency_uncovered_cannot_become_a_numerical_failure(time):
    result = classify_prediction(raw_calm(), [1, 1], true_event_time_s=time, numerical_passed=False)
    assert result["category"] == "latency_uncovered"
    assert not result["predicted_los"]


@pytest.mark.parametrize("failed_channel", [0, 1, 2])
def test_physical_nonfinite_before_los_is_unresolved(failed_channel):
    raw = add_los(raw_calm(), 20)
    raw[19, 0, failed_channel] = np.nan
    result = classify_prediction(raw, [1, 1], numerical_passed=True)
    assert result["category"] == "numerical_unresolved"
    assert result["first_nonfinite_index"] == 19
    assert trigger_index(raw, [1, 1]) is None


@pytest.mark.parametrize("rank", [1, 2, 3, 7])
def test_arbitrary_latent_rank_cannot_hide_nonfinite_states(rank):
    raw = add_los(raw_calm(), 20)
    latent = np.zeros((len(raw), 2, rank))
    latent[21, 1, -1] = np.inf
    mask = np.isfinite(latent).all(axis=(1, 2))
    result = classify_prediction(raw, [1, 1], latent_finite=mask, numerical_passed=True)
    assert result["category"] == "numerical_unresolved"
    assert result["first_nonfinite_index"] == 21
    assert result["latent_finiteness_checked"]


def test_nonfinite_after_established_hit_retains_hit_until_gate():
    raw = add_los(raw_calm(), 20)
    mask = np.ones(len(raw), dtype=bool)
    mask[25] = False
    result = classify_prediction(raw, [1, 1], latent_finite=mask)
    assert result["trigger_index"] == 24
    gated = apply_numerical_gate(result, False)
    assert gated["category"] == "numerical_unresolved"
    assert gated["unreliable_trigger_index"] == 24
    assert gated["trigger_index"] is None
    assert result["category"] == "predicted_los"  # No in-place mutation.


def test_gate_and_latent_mask_require_explicit_booleans():
    with pytest.raises(ValueError, match="explicit boolean"):
        apply_numerical_gate({"category": "window_settled"}, "false")
    with pytest.raises(ValueError, match="boolean"):
        classify_prediction(raw_calm(), [1, 1], latent_finite=np.ones(811))


def test_trigger_times_use_record_start_delay_and_decision_origin():
    times = trigger_times(124, clearing_time_s=2.35, record_start_delay_s=0.005)
    assert times["predicted_los_time_record_s"] == pytest.approx(1.24)
    assert times["predicted_los_time_decision_s"] == pytest.approx(1.14)
    assert times["predicted_los_time_absolute_s"] == pytest.approx(3.595)
    assert set(trigger_times(None, clearing_time_s=2).values()) == {None}


def test_pair_rms_uses_independent_weighted_formula_and_no_observation_padding():
    first, second = canonical(), canonical()
    mass = np.array([1.0, 3.0])
    first[:, :, 0] = [0.002, -0.001]
    first[:, :, 1] = [0.0003, 0.0009]
    row, passed = numerical_pair(first, second, canonical(11), mass, [0, 0], OMEGA_S)
    angle_mean = (0.002 - 0.003) / 4
    speed_mean = (0.0003 + 3 * 0.0009) / 4
    angle_terms = [((value - angle_mean) / DELTA_SCALE) ** 2 for value in (0.002, -0.001)]
    speed_terms = [((value - speed_mean) / OMEGA_SCALE) ** 2 for value in (0.0003, 0.0009)]
    expected = np.sqrt((sum(angle_terms) + sum(speed_terms)) / 4)
    assert row["normalized_coi_discrepancy"] == pytest.approx(expected, rel=1e-13)
    assert row["comparison_endpoint_index"] == 810
    assert passed is False


def test_pair_common_motion_and_finite_p_are_not_relative_rms():
    first, second = canonical(), canonical()
    first[..., 0] = 0.3
    first[..., 1] = 0.9
    first[..., 2] = 20
    row, passed = pair(first, second)
    assert passed
    assert row["normalized_coi_discrepancy"] == pytest.approx(0, abs=1e-14)
    first[500, 0, 2] = np.inf
    assert not pair(first, second)[1]


def test_pair_nonhit_requires_full_window_and_checks_latent_at_late_time():
    assert not pair(canonical(100), canonical(100))[1]
    mask = np.ones(801, dtype=bool)
    mask[800] = False
    row, passed = pair(canonical(), canonical(), first_latent_finite=mask)
    assert not passed
    assert not row["finite_to_comparison_endpoint"]
    assert row["comparison_endpoint_index"] == 810


def test_pair_two_hits_stop_at_later_hit_and_allow_posthit_failure():
    first = from_raw(add_los(raw_calm(801), 0))
    second = first.copy()
    first[5:, :, :] = np.nan
    second[6:, :, :] = np.nan
    row, passed = pair(first, second)
    assert passed
    assert row["fine_hit_index"] == row["coarse_hit_index"] == 14
    assert row["comparison_endpoint_index"] == 14
    assert row["finite_to_comparison_endpoint"]
    assert not row["finite_all_cached_state"]


@pytest.mark.parametrize("delay, expected", [(2, True), (3, False)])
def test_pair_hit_time_agreement_has_frozen_twenty_ms_threshold(delay, expected):
    first = canonical()
    first[:, :, 0] = [-2, 2]
    first[:, :, 1] = [-0.02000001 * OMEGA_S, 0.02000001 * OMEGA_S]
    second = first.copy()
    second[:delay, :, 1] = [-0.01999999 * OMEGA_S, 0.01999999 * OMEGA_S]
    row, passed = pair(first, second)
    assert row["normalized_coi_discrepancy"] < 1e-3
    assert row["hit_time_difference_s"] == pytest.approx(delay * 0.01)
    assert row["comparison_endpoint_index"] == 14 + delay
    assert passed is expected


def test_pair_disagreeing_hits_cannot_pass_tiny_relative_rms():
    first = canonical()
    first[:, :, 0] = [-2, 2]
    first[:, :, 1] = [-0.02000001 * OMEGA_S, 0.02000001 * OMEGA_S]
    second = first.copy()
    second[:, :, 1] = [-0.01999999 * OMEGA_S, 0.01999999 * OMEGA_S]
    row, passed = pair(first, second)
    assert row["normalized_coi_discrepancy"] < 1e-3
    assert not passed and not row["same_hit_or_nonhit"]
    assert row["comparison_endpoint_index"] == 810


@pytest.mark.parametrize("latent_failure", [False, True])
def test_pair_audits_unsampled_fine_substeps(latent_failure):
    fine = canonical()
    substeps = canonical(1601)
    mask = np.ones(1601, dtype=bool)
    if latent_failure:
        mask[1] = False
    else:
        substeps[1, 0, 2] = np.nan
    row, passed = pair(fine, fine, first_substeps=substeps, first_substeps_latent_finite=mask)
    assert not passed
    assert not row["finite_to_comparison_endpoint"]


def test_summary_keeps_all_denominators_and_false_stable_cases():
    categories = ["predicted_los", "predicted_los", "window_settled",
                  "numerical_unresolved", "window_no_los_unsettled"]
    labels = ["unstable", "stable", "unstable", "unstable", "stable"]
    rows = [{"id": str(i), "group": "A", "label": label, "category": category,
             "predicted_los": category == "predicted_los", "trigger_index": 30 if i < 2 else None}
            for i, (category, label) in enumerate(zip(categories, labels))]
    rows[0].update(trigger_times(30, clearing_time_s=2.0))
    latency = [{"id": "early", "group": "B", "label": "unstable", "reason": "latency_uncovered"}]
    summary = summarize_classifications(rows, latency)
    assert summary["event_count_raw"] == 6
    assert summary["eligible_event_count"] == 5
    assert summary["latency_uncovered_count"] == 1
    assert summary["unstable_total"] == 4
    assert summary["unstable_eligible_denominator"] == 3
    assert summary["latency_unstable_count"] == 1
    assert summary["missed_true_unstable"] == 2
    assert summary["missed_true_unstable_raw"] == 3
    assert summary["false_window_settled"] == summary["false_los"] == 1
    assert summary["undetermined_count"] == summary["numerical_unresolved_count"] == 1
    assert summary["detection_rate_eligible"] == pytest.approx(1 / 3)
    assert summary["detection_rate_raw"] == pytest.approx(1 / 4)
    assert summary["trigger_times"][0]["predicted_los_time_record_s"] == 0.3
    assert sum(summary["categories"].values()) == 6
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_classifications(rows, [dict(latency[0], id="0")])


def test_summary_zero_denominators_are_none_and_unknown_labels_rejected():
    assert summarize_classifications([])["detection_rate_raw"] is None
    with pytest.raises(ValueError, match="original"):
        summarize_classifications([{"category": "numerical_unresolved", "label": "unknown"}])
