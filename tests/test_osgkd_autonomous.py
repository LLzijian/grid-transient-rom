import pytest

from kiki.evaluation.osgkd_autonomous import summarize_autonomous_8s


def _row(name, *, label='unstable', prediction=None, truth=.2, available=True, qualified=True, within=False):
    item = {'los_pair_pass': qualified, 'truth_supported': within,
            'classification': {'category': 'predicted_los' if prediction is not None else 'window_settled',
                               'predicted_los': prediction is not None},
            'predicted_time_record_s': prediction, 'true_time_record_s': truth}
    return {'event_id': name, 'label': label, 'actual_future_points': 10 if available else None,
            'classification_8s': item if available else None,
            'classification_actual': item if available and within else None}


def test_autonomous_recall_does_not_truncate_and_retains_coverage():
    rows = [_row('tail', prediction=.45), _row('within', prediction=.17, within=True),
            _row('history', prediction=.1), _row('last_tick', prediction=8.1),
            _row('failed_mesh', prediction=.4, qualified=False), _row('short_history', available=False),
            _row('stable_hit', label='stable', prediction=1.), _row('stable_quiet', label='stable')]
    summary = summarize_autonomous_8s(rows)
    assert summary['unstable_recall_8s']['autonomous_trigger'] == 3
    assert summary['unstable_recall_8s']['rate_input_eligible'] == pytest.approx(3 / 5)
    assert summary['unstable_recall_8s']['rate_raw'] == pytest.approx(3 / 6)
    assert summary['unstable_recall_8s']['history_trigger'] == 1
    assert summary['unstable_recall_8s']['numerical_unresolved'] == 1
    assert summary['unstable_actual_interval_hits'] == 1
    assert summary['autonomous_after_record'] == 2
    assert summary['stable_false_alarm_8s']['rate_raw'] == .5
    lag = summary['trigger_lag_s']
    assert lag['count'] == 3 and lag['median'] == pytest.approx(.25)
    assert lag['iqr'] == pytest.approx(lag['q3'] - lag['q1'])
