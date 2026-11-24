"""Eight-second autonomous LOS from the existing full-window dual-grid score rows."""

import math
from collections import Counter

import numpy as np


DECISION_S = 0.1
HORIZON_S = 8.0
TIME_TOL_S = 1e-12


def autonomous_outcome(row):
    """Count only a qualified first trigger after the observation period."""
    if row['actual_future_points'] is None:
        return 'input_uncovered'
    item = row['classification_8s']
    if item is None or not item['los_pair_pass']:
        return 'numerical_unresolved'
    if item['classification']['category'] == 'latency_uncovered':
        return 'latency_uncovered'
    if not item['classification']['predicted_los']:
        return 'no_trigger'
    predicted = item['predicted_time_record_s']
    if predicted is None or not math.isfinite(predicted) or not 0 <= predicted <= DECISION_S + HORIZON_S + TIME_TOL_S:
        raise ValueError('invalid trigger time in qualified 8 s score row')
    return 'history_trigger' if predicted <= DECISION_S + TIME_TOL_S else 'autonomous_trigger'


def summarize_autonomous_8s(rows):
    """Keep raw and input-eligible denominators and report latency separately."""
    rows = list(rows)
    if len({row['event_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate scored event')
    stable, unstable = Counter(), Counter()
    lags = []
    actual_hits = within = tail = missing_reference = 0
    for row in rows:
        outcome = autonomous_outcome(row)
        label = row['label']
        if label == 'stable':
            stable[outcome] += 1
            continue
        if label != 'unstable':
            raise ValueError('invalid scored label')
        unstable[outcome] += 1
        actual = row['classification_actual']
        actual_hits += bool(actual and actual['los_pair_pass'] and actual['truth_supported'] and
                            actual['classification']['predicted_los'])
        if outcome == 'autonomous_trigger':
            item = row['classification_8s']
            within += bool(item['truth_supported'])
            tail += not item['truth_supported']
            true = item['true_time_record_s']
            if true is not None and math.isfinite(true):
                lags.append(item['predicted_time_record_s'] - true)
            else:
                missing_reference += 1

    def population(counts):
        keys = ('autonomous_trigger', 'history_trigger', 'no_trigger', 'numerical_unresolved',
                'latency_uncovered', 'input_uncovered')
        raw = sum(counts.values())
        eligible = raw - counts['input_uncovered']
        hits = counts['autonomous_trigger']
        return {**{key: counts[key] for key in keys}, 'raw_denominator': raw, 'input_eligible_denominator': eligible,
                'rate_raw': hits / raw if raw else None, 'rate_input_eligible': hits / eligible if eligible else None}

    q1, median, q3 = (float(x) for x in np.quantile(lags, (.25, .5, .75), method='linear')) if lags else (None, None, None)
    return {'stable_false_alarm_8s': population(stable), 'unstable_recall_8s': population(unstable),
            'unstable_actual_interval_hits': actual_hits, 'autonomous_within_record': within,
            'autonomous_after_record': tail, 'trigger_lag_s': {'count': len(lags), 'reference_missing': missing_reference,
                'q1': q1, 'median': median, 'q3': q3, 'iqr': q3 - q1 if lags else None}}
