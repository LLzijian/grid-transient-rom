import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from summarize_train_local_memory_0098 import event_metrics, summarize_rows, scalar_summary, build_summary


class Scales:
    mass = torch.ones(10, dtype=torch.float64)
    p_scale = torch.ones(10, dtype=torch.float64)


def sample(length=1501, label="stable"):
    event = {"id": "synthetic", "label": label, "group": "g", "x": np.zeros((length, 30))}
    return event, np.zeros((1601, 41)), np.zeros((801, 41))


def test_true_prefix_and_horizon_denominators_never_pad_short_truth():
    event, fine, coarse = sample(14, "unstable")
    result = event_metrics(Scales(), event, fine, coarse)
    assert not any(r["available"] for r in result["records"][:4])
    prefix = result["records"][-1]
    assert prefix["n"] == 30
    assert prefix["numerically_admissible"] and prefix["rmse"] == 0
    assert result["common_frequency"][-1]["n"] == 3
    assert result["normalized_p"][-1]["n"] == 30


def test_long_true_prefix_is_capped_at_eight_seconds():
    event, fine, coarse = sample()
    result = event_metrics(Scales(), event, fine, coarse)
    assert result["records"][-1]["n"] == 8000
    assert result["common_frequency"][-1]["n"] == 800


def test_hidden_latent_substep_failure_quarantines_only_affected_intervals():
    event, fine, coarse = sample()
    fine[401, 40] = np.nan
    result = event_metrics(Scales(), event, fine, coarse)
    assert result["records"][0]["numerically_admissible"]
    assert result["records"][1]["numerically_admissible"]
    assert result["records"][2]["finite"]
    assert not result["records"][2]["numerically_admissible"]
    score = summarize_rows([result["records"][2]])
    assert score["official"] is None
    assert score["finite_subcohort_diagnostic"]["mean"] == 0
    assert score["unresolved_event_ids"] == [event["id"]]


def test_common_frequency_numerical_mismatch_cannot_hide_in_coi_metric():
    event, fine, coarse = sample()
    coarse[:, 10:20] = .01
    result = event_metrics(Scales(), event, fine, coarse)
    assert not result["records"][0]["numerically_admissible"]
    assert result["numerical"][0]["coi_normalized_rmse"] < 1e-14
    assert result["numerical"][0]["common_normalized_rmse"] > .001


def test_physical_nonfinite_is_retained_without_finite_official_filtering():
    event, fine, coarse = sample()
    fine[10, 0] = np.nan
    result = event_metrics(Scales(), event, fine, coarse)
    row = result["records"][0]
    assert row["available"] and not row["finite"]
    score = summarize_rows([row])
    assert score["official"] is None and score["finite_subcohort_diagnostic"] is None


def test_p_and_objective_use_checkpoint_scales_without_adding_fake_events():
    event, fine, coarse = sample()
    fine[:, 20:30] = 2.
    coarse[:, 20:30] = 2.
    model = Scales()
    model.p_scale = torch.full((10,), 2., dtype=torch.float64)
    result = event_metrics(model, event, fine, coarse)
    assert result["normalized_p"][0]["rmse"] == 1.
    assert result["objective_point_error"][0]["mean_point_error"] == 1.
    summary = build_summary(result)
    assert summary["normalized_p"]["1"]["official"]["n_events"] == 1
    assert summary["records"]["1"]["official"]["mean"] == 0


def test_all_unstable_prefix_rows_retained_in_summary():
    event, fine, coarse = sample(30, "unstable")
    result = event_metrics(Scales(), event, fine, coarse)
    summary = build_summary(result)
    assert summary["records"]["unstable_prefix"]["all_rows"] == 1
    assert summary["records"]["unstable_prefix"]["official"]["n_events"] == 1


def test_finite_state_metric_overflow_is_unresolved_not_null_fields_in_official():
    event, fine, coarse = sample()
    fine[:, 20:30] = 1e200
    coarse[:, 20:30] = 1e200
    with np.errstate(over="ignore", invalid="ignore"):
        result = event_metrics(Scales(), event, fine, coarse)
    summary = build_summary(result)
    assert summary["normalized_p"]["1"]["official"] is None
    assert summary["objective_point_error"]["1"]["official"] is None
