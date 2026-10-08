from __future__ import annotations

import numpy as np
from scipy.stats import beta


def classify_predicted_trajectory(
    delta: np.ndarray,
    omega: np.ndarray,
    *,
    angle_range_threshold: float = np.pi,
    speed_threshold_rad_s: float = 0.05 * 376.99111843077515,
) -> str:
    delta = np.asarray(delta); omega = np.asarray(omega)
    angle_range = np.max(delta, axis=1) - np.min(delta, axis=1)
    if np.any(angle_range >= angle_range_threshold) or np.any(np.abs(omega) >= speed_threshold_rad_s):
        return "unstable"
    return "stable"


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float | int]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("classification requires a nonempty aligned population")
    tp = sum(a == "unstable" and b == "unstable" for a, b in zip(y_true, y_pred))
    fn = sum(a == "unstable" and b == "stable" for a, b in zip(y_true, y_pred))
    fp = sum(a == "stable" and b == "unstable" for a, b in zip(y_true, y_pred))
    tn = sum(a == "stable" and b == "stable" for a, b in zip(y_true, y_pred))
    fsr = fn / (tp + fn) if tp + fn else float("nan")
    fur = fp / (tn + fp) if tn + fp else float("nan")
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "FSR": fsr, "FUR": fur,
        "Accuracy": (tp + tn) / len(y_true),
        "BalancedAccuracy": float(np.nanmean([tpr, tnr])),
    }


def one_sided_binomial_upper_bound(
    failures: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Exact Clopper--Pearson upper bound for a binomial failure probability."""

    if isinstance(failures, bool) or isinstance(trials, bool):
        raise TypeError("failures and trials must be integer counts")
    if int(failures) != failures or int(trials) != trials:
        raise TypeError("failures and trials must be integer counts")
    failures = int(failures)
    trials = int(trials)
    if trials <= 0 or failures < 0 or failures > trials:
        raise ValueError("require 0 <= failures <= trials and trials > 0")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if failures == trials:
        return 1.0
    return float(beta.ppf(float(confidence), failures + 1, trials - failures))


def grouped_missed_instability_audit(
    rows: list[dict],
    *,
    predicted_stable_key: str,
    eligible_key: str,
    group_keys: tuple[str, ...] = ("source", "fault_line"),
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Audit dangerous stable calls at event and independent-group levels.

    Only truly unstable rows contribute. A group is counted as failed if any
    eligible event in that group is called stable. Treating the disturbance
    group, rather than every correlated trajectory, as the binomial unit avoids
    pseudoreplication. The exact upper confidence bound still assumes the
    declared groups are independent and identically distributed.
    """

    unstable = [row for row in rows if not bool(row["true_stable"])]
    if not unstable:
        raise ValueError("missed-instability audit requires unstable cases")
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in unstable:
        try:
            group = tuple(str(row[key]) for key in group_keys)
        except KeyError as exc:
            raise KeyError(f"missing independence-group field: {exc.args[0]}") from exc
        grouped.setdefault(group, []).append(row)

    def missed(row: dict) -> bool:
        return bool(row[eligible_key]) and bool(row[predicted_stable_key])

    missed_events = sum(missed(row) for row in unstable)
    missed_groups = sum(any(missed(row) for row in members) for members in grouped.values())
    group_trials = len(grouped)
    return {
        "n_unstable_events": len(unstable),
        "missed_unstable_events": int(missed_events),
        "event_missed_instability_rate": missed_events / len(unstable),
        "n_independent_unstable_groups": group_trials,
        "missed_unstable_groups": int(missed_groups),
        "group_missed_instability_rate": missed_groups / group_trials,
        "group_missed_instability_upper": one_sided_binomial_upper_bound(
            missed_groups,
            group_trials,
            confidence=confidence,
        ),
        "confidence": float(confidence),
        "interval": "one-sided exact Clopper-Pearson",
        "independence_unit": "+".join(group_keys),
        "group_failure_definition": "any eligible unstable event called stable",
    }
