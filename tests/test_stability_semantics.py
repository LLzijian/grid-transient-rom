from __future__ import annotations

import numpy as np
import pytest

from kiki.evaluation.stability import (
    grouped_missed_instability_audit,
    one_sided_binomial_upper_bound,
)
from kiki.models.prunable_energy_model import PrunableEnergyModel


class _CandidateOnlyModel:
    def critical_energy_level(self, static, *, pm_clear=None, angle_limit=np.pi / 2):
        return {
            "certified": False,
            "certificate_status": "refused_no_verified_global_escape_energy_lower_bound",
            "critical_energy_lower_bound": None,
            "critical_energy_candidate": 5.0,
            "candidate_search_completed": True,
            "equilibrium_delta": [0.0, 0.0],
        }

    def initial_hidden(self, omega, static):
        return {}

    def storage(self, delta, omega, static, hidden, *, pm_clear, reference_delta):
        return 3.0


def test_positive_candidate_margin_is_not_a_stability_certificate() -> None:
    static = {
        "delta0": np.zeros(2),
        "omega0": np.zeros(2),
        "Pm_clear": np.zeros(2),
        "Pm_equilibrium": np.zeros(2),
    }
    result = PrunableEnergyModel.stability_margin(_CandidateOnlyModel(), static)
    assert result["screening_margin"] == pytest.approx(2.0)
    assert result["screening_predicted_stable"] is True
    assert result["certified"] is False
    assert result["margin"] is None
    assert result["predicted_stable"] is False
    assert result["verdict"] == "refused"


def test_zero_failures_still_has_positive_exact_upper_bound() -> None:
    upper = one_sided_binomial_upper_bound(0, 38, confidence=0.95)
    assert upper == pytest.approx(1.0 - 0.05 ** (1.0 / 38.0))
    assert upper > 0.0


def test_missed_instability_uses_declared_independence_groups() -> None:
    rows = [
        {
            "true_stable": False,
            "source": "A",
            "fault_line": "L1",
            "eligible": True,
            "called_stable": False,
        },
        {
            "true_stable": False,
            "source": "A",
            "fault_line": "L1",
            "eligible": True,
            "called_stable": True,
        },
        {
            "true_stable": False,
            "source": "B",
            "fault_line": "L1",
            "eligible": False,
            "called_stable": True,
        },
    ]
    audit = grouped_missed_instability_audit(
        rows,
        predicted_stable_key="called_stable",
        eligible_key="eligible",
    )
    assert audit["missed_unstable_events"] == 1
    assert audit["n_independent_unstable_groups"] == 2
    assert audit["missed_unstable_groups"] == 1
