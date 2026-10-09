"""Synthetic frozen-report comparisons; no event, cache or model execution."""
import copy
import importlib.util
import math

import pytest

from task4.paths import ROOT


@pytest.fixture
def comparison_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('comparison_test_module', ROOT / 'scripts/compare_endpoints.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_report(endpoint, errors=(1., 2.), failed=()):
    delta_scale, omega_scale = .47675314588683515, 1.0033667249602924
    report = {
        'endpoint': endpoint, 'test_payload_read': False, 'history_ms': 100, 'evidence_role': 'development',
        'initializations_per_event': 1, 'forecast_steps': 800, 'decision_horizon_s': 8.,
        'fine_dt': .0025, 'coarse_dt': .005, 'tolerance': .001,
        'classification_rms_tolerance': .001, 'classification_hit_time_tolerance_s': .02,
        'raw_event_count': 3, 'eligible_event_count': 2, 'latency_uncovered_count': 1,
        'comparison_context': {
            'saved_training_scales': [.2, .3, .4],
            'metric_scales_delta_omega': [delta_scale, omega_scale], 'mass': [1., 2.],
            'source_physics': {'mass': [1., 2.], 'omega_s': 2 * math.pi * 60},
            'truth_time_origin': 'postfault t=0; initializer ends at t=0.1 s',
            'error_time_origin': 'forecast elapsed after t=0.1 s', 'history_samples': 11,
        },
        'source_facts': {'split': 'validation', 'raw_count': 3, 'eligible_count': 2,
                         'unstable_count': 1, 'test_payload_read': False, 'schema': 'synthetic_report',
                         'excluded': [{'id': 'early', 'group': 'unknown', 'label': 'unstable', 'reason': 'latency_uncovered'}]},
        'rows': [], 'summary': {}, 'classification_numerical': [], 'integration_failures': [],
        # This nonexistent cache must never be followed by this comparison.
        'cache': {'directory': '/does/not/exist/poisoned_event_payload'},
    }
    for horizon, steps in (('1.0', 100), ('2.0', 200), ('5.0', 500), ('8.0', 800), ('prefix', 800)):
        for index, (eid, label, group) in enumerate((('a', 'stable', '10'), ('b', 'unstable', '20'))):
            r, n = errors[index], 2 * steps
            metric = {
                'n': n, 'rmse': r, 'normalized_sse': r * r * n,
                'delta_rmse_rad': delta_scale * r, 'delta_sse_rad2': (delta_scale * r) ** 2 * n,
                'omega_rmse_rad_s': omega_scale * r, 'omega_sse_rad_s2': (omega_scale * r) ** 2 * n,
                'common_rmse_rad_s': .1 * r, 'common_sse_rad_s2': (.1 * r) ** 2 * steps, 'common_n': steps,
                'p_rmse_pu': .4 * r, 'p_sse_pu2': (.4 * r) ** 2 * n,
                'p_normalized_rmse': r, 'p_normalized_sse': r * r * n,
            }
            report['rows'].append({
                'id': eid, 'label': label, 'group': group, 'horizon': horizon,
                'scope': 'available_prefix' if horizon == 'prefix' else 'complete',
                'steps': steps, 'true_available_steps': 800, 'scorable': True,
                'qualified': eid not in failed, 'finite_fine': eid not in failed,
                'finite_coarse': eid not in failed, 'metrics': metric,
                'numerical_difference': dict.fromkeys(('delta', 'omega', 'common', 'p'), 0.),
                'failure': 'nonfinite_prediction' if eid in failed else None,
            })
    for horizon in ('1.0', '2.0', '5.0', '8.0', 'prefix', 'severe_prefix'):
        severeness = horizon == 'severe_prefix'
        values = [errors[1]] if severeness else list(errors)
        unavailable = ('b' in failed) if severeness else bool(failed)
        report['summary'][horizon] = {
            'mean': None if unavailable else sum(values) / len(values),
            'pooled': None if unavailable else math.sqrt(sum(value * value for value in values) / len(values)),
            'tail': None if unavailable else max(values),
            'count': len(values), 'truth_scorable_denominator': len(values),
            'numerically_qualified_count': (0 if 'b' in failed else 1) if severeness else 2 - len(failed),
            'raw_denominator': 2 if severeness else 3,
        }
    classified = []
    for eid, label, group in (('a', 'stable', '10'), ('b', 'unstable', '20')):
        category = 'numerical_unresolved' if eid in failed else ('predicted_los' if label == 'unstable' else 'window_settled')
        classified.append({'id': eid, 'label': label, 'group': group, 'category': category,
                           'predicted_los': category == 'predicted_los', 'numerical_gate_passed': eid not in failed,
                           'trigger_index': 50 if category == 'predicted_los' else None})
        report['classification_numerical'].append({
            'id': eid, 'screen_pass': eid not in failed, 'finite_to_comparison_endpoint': eid not in failed,
            'same_hit_or_nonhit': eid not in failed, 'normalized_coi_discrepancy': 0. if eid not in failed else None,
            'hit_time_difference_s': 0. if label == 'unstable' and eid not in failed else None,
        })
        report['integration_failures'].append({'id': eid, 'fine': {'first_bad_substep': 400} if eid in failed else None,
                                               'coarse': {'first_bad_substep': 200} if eid in failed else None})
    classified.append({'id': 'early', 'label': 'unstable', 'group': 'unknown',
                       'category': 'latency_uncovered', 'predicted_los': False, 'trigger_index': None})
    report['classification'] = {'rows': classified}
    return report


def test_old_resolution_is_rejected_by_registered_endpoint_policy(comparison_script):
    report = frozen_report('P-N')
    report.update(fine_dt=.005, coarse_dt=.01)
    result = comparison_script.compare_reports({'M3': frozen_report('M3'), 'P-N': report})
    assert result['audits']['P-N']['comparison_to_M3']['status'] == 'incomparable'
    assert not result['automatic_model_acceptance']


@pytest.mark.parametrize('marker', [dict(source_identity_unchanged=False),
                                  dict(evidence_validity='invalid_source_changed')])
def test_changed_frozen_source_is_rejected(comparison_script, marker):
    candidate = frozen_report('P-N')
    candidate.update(marker)
    result = comparison_script.compare_reports({'M3': frozen_report('M3'), 'P-N': candidate})
    audit = result['audits']['P-N']['comparison_to_M3']
    assert audit['status'] == 'incomparable'
    assert 'invalid_frozen_source_identity' in audit['mismatches']
    assert not result['automatic_model_acceptance']


def test_missing_endpoints_are_explicit_and_all_population_differences_are_descriptive(comparison_script):
    result = comparison_script.compare_reports({'M3': frozen_report('M3'), 'P-N': frozen_report('P-N', (.5, 1.))})
    assert result['missing_endpoints'] == ['P-S', 'A', 'U']
    assert not result['all_registered_endpoints_present']
    assert result['audits']['P-N']['comparison_to_M3']['status'] == 'verified_declared_report_contract'
    row = next(row for row in result['contrasts_to_M3'] if row['horizon'] == '1')
    assert row['components']['relative']['statistics']['pooled']['candidate_minus_M3'] == pytest.approx(-.5 * math.sqrt(2.5))
    assert row['components']['common_rad_s']['descriptive_direction'] == 'lower_error'
    assert not row['overall_improvement_claim']
    assert not result['automatic_model_acceptance'] and not result['candidate_selection_performed']
    assert not result['event_payload_read'] and not result['trajectory_cache_read']
    text = comparison_script.markdown_report(result)
    assert '缺失端点：P-S, A, U' in text and '错误 window_settled' in text


def test_failed_event_cannot_be_removed_to_claim_overall_improvement(comparison_script):
    result = comparison_script.compare_reports({'M3': frozen_report('M3'),
                                               'P-N': frozen_report('P-N', (.01, .01), failed=('b',))})
    capability = next(row for row in result['capability_table'] if row['endpoint'] == 'P-N' and row['horizon'] == '8')
    assert capability['truth_scorable_denominator'] == 2
    assert capability['numerically_qualified_count'] == 1
    assert capability['components']['relative']['pooled'] is None
    assert capability['numerically_unresolved_ids'] == ['b']
    contrast = next(row for row in result['contrasts_to_M3'] if row['horizon'] == '8')
    assert contrast['components']['relative']['statistics']['pooled']['candidate_minus_M3'] is None
    group_a = next(row for row in result['fault_group_contrasts_to_M3'] if row['group'] == '10' and row['horizon'] == '8')
    group_b = next(row for row in result['fault_group_contrasts_to_M3'] if row['group'] == '20' and row['horizon'] == '8')
    assert group_a['components']['relative']['descriptive_direction'] == 'lower_error'
    assert group_b['candidate']['numerically_unresolved_ids'] == ['b']
    assert group_b['full_8s_los_count_changes_candidate_minus_M3']['missed_true_unstable'] == 1
    assert group_b['full_8s_los_count_changes_candidate_minus_M3']['numerical_unresolved'] == 1
    failures = result['failure_registry']['P-N']
    assert failures['accuracy_unresolved_ids'] == failures['integration_failure_ids'] == ['b']
    assert failures['missed_true_unstable_ids'] == ['b']
    assert failures['classification_category_ids']['latency_uncovered'] == ['early']
    los = next(row for row in result['classification_table'] if row['endpoint'] == 'P-N')
    assert (los['raw_denominator'], los['covered_denominator'], los['latency_uncovered_count']) == (3, 2, 1)


@pytest.mark.parametrize('corruption', ['group', 'scale', 'interval', 'gate', 'forged_qualified_subset'])
def test_inconsistent_report_contract_suppresses_improvement(comparison_script, corruption):
    baseline = frozen_report('M3')
    candidate = frozen_report('P-N', (.1, .2), failed=('b',) if corruption == 'forged_qualified_subset' else ())
    if corruption == 'group':
        for row in candidate['rows'] + candidate['classification']['rows']:
            if row['id'] == 'a':
                row['group'] = '99'
    elif corruption == 'scale':
        candidate['comparison_context']['saved_training_scales'][0] = .21
    elif corruption == 'interval':
        candidate['rows'][0]['true_available_steps'] = 700
    elif corruption == 'gate':
        candidate['tolerance'] = .1
    else:
        candidate['summary']['8.0'].update(mean=.1, pooled=.1, tail=.1)
    result = comparison_script.compare_reports({'M3': baseline, 'P-N': candidate})
    audit = result['audits']['P-N']['comparison_to_M3']
    assert audit['status'] == 'incomparable' and audit['mismatches']
    assert all(row['components']['relative']['statistics']['pooled']['candidate_minus_M3'] is None
               for row in result['contrasts_to_M3'])


def test_missing_scale_declarations_are_unverified_not_silently_equal(comparison_script):
    baseline, candidate = frozen_report('M3'), frozen_report('P-N', (.1, .2))
    baseline.pop('comparison_context')
    candidate.pop('comparison_context')
    before = copy.deepcopy(candidate)
    result = comparison_script.compare_reports({'M3': baseline, 'P-N': candidate})
    assert candidate == before  # Frozen input reports are not edited.
    assert result['audits']['P-N']['comparison_to_M3']['status'] == 'unverified'
    assert all(row['components']['relative']['statistics']['pooled']['candidate_minus_M3'] is None
               for row in result['contrasts_to_M3'])


def test_eligible_unstable_cannot_be_moved_into_latency_denominator(comparison_script):
    baseline, candidate = frozen_report('M3'), frozen_report('P-N', (.1, .2))
    unstable = next(row for row in candidate['classification']['rows'] if row['id'] == 'b')
    unstable.update(category='latency_uncovered', predicted_los=False, trigger_index=None)
    # Make all saved classification totals internally consistent with the
    # forged labels: only the independent source-exclusion set exposes it.
    candidate['classification'].update(event_count_raw=3, eligible_event_count=1,
                                       latency_uncovered_count=2, unstable_total=2,
                                       unstable_eligible_denominator=0, raw_unstable_detected=0,
                                       missed_true_unstable=0, false_window_settled=0,
                                       false_los=0, numerical_unresolved_count=0)
    result = comparison_script.compare_reports({'M3': baseline, 'P-N': candidate})
    audit = result['audits']['P-N']['comparison_to_M3']
    assert audit['status'] == 'incomparable'
    assert 'classification_latency_IDs_mismatch_source_exclusions' in audit['mismatches']
    assert 'eligible_event_reclassified_as_latency_uncovered:b' in audit['mismatches']
    assert all(row['components']['relative']['statistics']['pooled']['candidate_minus_M3'] is None
               for row in result['contrasts_to_M3'])


@pytest.mark.parametrize('forged_counts', [(2000, 1000), (200.5, 100.25)])
def test_pooled_weights_must_equal_integer_future_sample_counts(comparison_script, forged_counts):
    baseline, candidate = frozen_report('M3'), frozen_report('P-N', (.1, .2))
    row = next(row for row in candidate['rows'] if row['id'] == 'a' and row['horizon'] == '1.0')
    metric = row['metrics']
    old_n, old_common_n = metric['n'], metric['common_n']
    metric['n'], metric['common_n'] = forged_counts
    # Preserve every SSE=RMSE^2*n identity and normalization relation while
    # upweighting the easier event. Those identities alone are insufficient.
    for name in ('normalized_sse', 'delta_sse_rad2', 'omega_sse_rad_s2', 'p_sse_pu2', 'p_normalized_sse'):
        metric[name] *= metric['n'] / old_n
    metric['common_sse_rad_s2'] *= metric['common_n'] / old_common_n
    other = next(row['metrics'] for row in candidate['rows'] if row['id'] == 'b' and row['horizon'] == '1.0')
    candidate['summary']['1.0']['pooled'] = math.sqrt((metric['normalized_sse'] + other['normalized_sse'])
                                                   / (metric['n'] + other['n']))
    result = comparison_script.compare_reports({'M3': baseline, 'P-N': candidate})
    audit = result['audits']['P-N']['comparison_to_M3']
    assert audit['status'] == 'incomparable'
    assert 'metric_n_interval_machine_count_mismatch:a:1' in audit['mismatches']
    assert 'metric_common_n_interval_mismatch:a:1' in audit['mismatches']
    if type(forged_counts[0]) is float:
        assert 'metric_count_not_positive_integer:a:1:n' in audit['mismatches']
        assert 'metric_count_not_positive_integer:a:1:common_n' in audit['mismatches']
    assert all(row['components']['relative']['statistics']['pooled']['candidate_minus_M3'] is None
               for row in result['contrasts_to_M3'])
