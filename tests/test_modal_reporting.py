from __future__ import annotations

import pytest

from kiki.evaluation.modal import summarize_event_modal_matches


def _match(error: float, damping_error: float, mac: float) -> dict[str, float]:
    return {
        "relative_frequency_error": error,
        "absolute_damping_ratio_error": damping_error,
        "mac": mac,
    }


def test_modal_summary_uses_all_assignments_not_per_event_best() -> None:
    cases = [
        {
            "event_id": "a",
            "matches": [_match(0.01, 0.02, 0.9), _match(0.49, 0.20, 0.3)],
        },
        {"event_id": "b", "matches": [_match(0.20, 0.10, 0.5)]},
    ]
    summary = summarize_event_modal_matches(cases)
    # Event a contributes median 0.25, not its attractive 0.01 best pair.
    assert summary["median_event_relative_frequency_error"] == pytest.approx(0.225)
    assert summary["n_assigned_pairs"] == 3
    assert summary["best_match_selection_used"] is False


def test_modal_summary_reports_assignment_coverage() -> None:
    summary = summarize_event_modal_matches(
        [{"event_id": "a", "matches": []}, {"event_id": "b", "matches": []}]
    )
    assert summary["n_events_with_assignments"] == 0
    assert summary["event_assignment_coverage"] == 0.0
    assert summary["median_event_relative_frequency_error"] is None
