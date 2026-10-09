"""Compare explicitly supplied frozen endpoint reports; no model/data loading.

This checks declared report provenance and recomputes descriptive aggregates
from all scorable event rows. It never reads the trajectory cache, opens an
event payload, executes a model, chooses a candidate, or accepts a model.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from task4.paths import project_path, write_json


ENDPOINTS = ('M3', 'P-N', 'P-S', 'A', 'U')
HORIZONS = ('1', '2', '5', '8', 'prefix', 'severe_prefix')
CATEGORIES = ('predicted_los', 'window_settled', 'window_no_los_unsettled',
              'numerical_unresolved', 'latency_uncovered')
CONTEXT_KEYS = ('saved_training_scales', 'metric_scales_delta_omega', 'mass', 'source_physics',
                'truth_time_origin', 'error_time_origin', 'history_samples')
FROZEN_POLICY = {
    'history_ms': 100, 'initializations_per_event': 1, 'forecast_steps': 800,
    'decision_horizon_s': 8.0, 'fine_dt': .0025, 'coarse_dt': .005,
    'tolerance': .001, 'classification_rms_tolerance': .001,
    'classification_hit_time_tolerance_s': .02,
}
COMPONENTS = {
    'relative': ('rmse', 'normalized_sse', 'n'),
    'delta_rad': ('delta_rmse_rad', 'delta_sse_rad2', 'n'),
    'omega_rad_s': ('omega_rmse_rad_s', 'omega_sse_rad_s2', 'n'),
    'common_rad_s': ('common_rmse_rad_s', 'common_sse_rad_s2', 'common_n'),
    'p_pu': ('p_rmse_pu', 'p_sse_pu2', 'n'),
    'p_normalized': ('p_normalized_rmse', 'p_normalized_sse', 'n'),
}


def same(left, right):
    """Strict structure, with only rounding-size tolerance for numeric data."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def identity(row):
    return row.get('id', row.get('event_id'))


def horizon_key(value):
    if value in ('prefix', 'severe_prefix'):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else str(number)


def source_summary(report, horizon):
    summaries = report.get('summary', {})
    return next((value for name, value in summaries.items() if horizon_key(name) == horizon), None)


def normalized_facts(facts):
    result = dict(facts)
    if isinstance(result.get('excluded'), list):
        result['excluded'] = sorted(result['excluded'], key=lambda row: str(identity(row)))
    return result


def aggregate_component(rows, component):
    rmse_key, sse_key, count_key = COMPONENTS[component]
    result = {'mean': None, 'pooled': None, 'tail': None, 'status': 'no_scorable_events'}
    if not rows:
        return result
    if any(row.get('qualified') is not True for row in rows):
        return {**result, 'status': 'numerically_unresolved'}
    values, sums, counts = [], [], []
    invalid = []
    for row in rows:
        metric = row.get('metrics') or {}
        if (not finite_nonnegative(metric.get(rmse_key)) or not finite_nonnegative(metric.get(sse_key))
                or type(metric.get(count_key)) not in (int, float) or metric[count_key] <= 0
                or not math.isfinite(metric[count_key])):
            invalid.append(identity(row))
            continue
        if not same(metric[sse_key], metric[rmse_key] ** 2 * metric[count_key]):
            invalid.append(identity(row))
            continue
        values.append(metric[rmse_key])
        sums.append(metric[sse_key])
        counts.append(metric[count_key])
    if invalid:
        return {**result, 'status': 'missing_or_inconsistent_metric_fields', 'invalid_metric_ids': invalid}
    tail_n = max(1, math.ceil(.1 * len(values)))
    return {'mean': sum(values) / len(values), 'pooled': math.sqrt(sum(sums) / sum(counts)),
            'tail': sum(sorted(values)[-tail_n:]) / tail_n, 'status': 'complete_scorable_population'}


def accuracy_bundle(record, horizon, group=None):
    key = 'prefix' if horizon == 'severe_prefix' else horizon
    all_ids = [eid for eid, metadata in record['eligible'].items()
               if (horizon != 'severe_prefix' or metadata['label'] == 'unstable')
               and (group is None or metadata['group'] == group)]
    selected = [record['accuracy'][(eid, key)] for eid in all_ids if (eid, key) in record['accuracy']]
    scorable = [row for row in selected if row.get('scorable') is True]
    missing = [eid for eid in all_ids if (eid, key) not in record['accuracy']]
    unresolved = [identity(row) for row in scorable if row.get('qualified') is not True]
    raw = [metadata for metadata in record['raw'].values()
           if (horizon != 'severe_prefix' or metadata['label'] == 'unstable')
           and (group is None or metadata['group'] == group)]
    components = {name: aggregate_component(scorable, name) for name in COMPONENTS}
    if missing:
        components = {name: {'mean': None, 'pooled': None, 'tail': None,
                             'status': 'missing_event_rows'} for name in COMPONENTS}
    return {
        'horizon': horizon, 'group': group, 'raw_denominator': len(raw),
        'initialization_eligible_denominator': len(all_ids),
        'truth_scorable_denominator': len(scorable),
        'numerically_qualified_count': sum(row.get('qualified') is True for row in scorable),
        'numerically_unresolved_count': len(unresolved),
        'truth_unavailable_ids': [identity(row) for row in selected if row.get('scorable') is not True],
        'numerically_unresolved_ids': sorted(unresolved), 'missing_event_row_ids': sorted(missing),
        'components': components,
        'population_status': 'missing_event_rows' if missing else ('numerically_unresolved' if unresolved else 'complete_declared_scorable_population'),
        'uses_qualified_subset_mean': False,
    }


def los_bundle(record, group=None):
    rows = [row for row in record['classification'].values()
            if group is None or str(row.get('group', 'unknown')) == group]
    covered = [row for row in rows if row.get('category') != 'latency_uncovered']
    unstable = [row for row in rows if row.get('label') == 'unstable']
    covered_unstable = [row for row in covered if row.get('label') == 'unstable']
    detected = [identity(row) for row in covered_unstable if row.get('category') == 'predicted_los']
    missed = [identity(row) for row in covered_unstable if row.get('category') != 'predicted_los']
    false_settled = [identity(row) for row in covered_unstable if row.get('category') == 'window_settled']
    false_los = [identity(row) for row in covered if row.get('label') == 'stable' and row.get('category') == 'predicted_los']
    categories = {name: [identity(row) for row in rows if row.get('category') == name] for name in CATEGORIES}
    return {
        'raw_denominator': len(rows), 'covered_denominator': len(covered),
        'latency_uncovered_count': len(categories['latency_uncovered']),
        'raw_unstable_denominator': len(unstable), 'covered_unstable_denominator': len(covered_unstable),
        'detected_true_unstable': len(detected), 'missed_true_unstable': len(missed),
        'false_window_settled': len(false_settled), 'false_los': len(false_los),
        'window_no_los_unsettled': len(categories['window_no_los_unsettled']),
        'numerical_unresolved': len(categories['numerical_unresolved']),
        'detection_rate_raw': len(detected) / len(unstable) if unstable else None,
        'detection_rate_covered': len(detected) / len(covered_unstable) if covered_unstable else None,
        'categories': {name: len(ids) for name, ids in categories.items()},
        'category_ids': categories, 'missed_true_unstable_ids': missed,
        'false_window_settled_ids': false_settled, 'false_los_ids': false_los,
        'group': group,
    }


def index_report(endpoint, report):
    errors, unverified = [], []
    if (report.get('source_identity_unchanged') is False
            or str(report.get('evidence_validity', '')).startswith('invalid')):
        errors.append('invalid_frozen_source_identity')
    accuracy, eligible, classified, true_lengths = {}, {}, {}, {}
    if report.get('endpoint') != endpoint:
        errors.append('endpoint_identity_mismatch')
    if report.get('test_payload_read') is not False:
        errors.append('test_payload_status_not_explicitly_false')
    for key, expected in FROZEN_POLICY.items():
        if key not in report:
            unverified.append('missing_policy:' + key)
        elif not same(report[key], expected):
            errors.append('frozen_policy_mismatch:' + key)
    context = report.get('comparison_context') or {}
    for key in CONTEXT_KEYS:
        if key not in context or context[key] is None:
            unverified.append('missing_context:' + key)
    for key, length in (('saved_training_scales', 3), ('metric_scales_delta_omega', 2)):
        values = context.get(key)
        if values is not None and (not isinstance(values, list) or len(values) != length
                                   or any(not finite_nonnegative(value) or value == 0 for value in values)):
            errors.append('invalid_context:' + key)
    if 'mass' in context and (not isinstance(context['mass'], list) or not context['mass']
                              or any(not finite_nonnegative(value) or value == 0 for value in context['mass'])):
        errors.append('invalid_context:mass')
    if context.get('history_samples', 11) != 11:
        errors.append('invalid_context:history_samples')
    for key, expected in (('truth_time_origin', 'postfault t=0; initializer ends at t=0.1 s'),
                          ('error_time_origin', 'forecast elapsed after t=0.1 s')):
        if key in context and context[key] != expected:
            errors.append('time_origin_mismatch:' + key)
    source_physics = context.get('source_physics')
    if source_physics is not None:
        if not isinstance(source_physics, dict):
            errors.append('invalid_context:source_physics')
        elif 'mass' not in source_physics:
            unverified.append('missing_context:source_physics.mass')
        elif 'mass' in context and not same(source_physics['mass'], context['mass']):
            errors.append('source_physics_mass_mismatch')
    facts = report.get('source_facts') or {}
    if facts.get('split') not in ('validation', 'train', 'synthetic'):
        unverified.append('missing_or_unknown_source_split')
    if facts.get('test_payload_read') is not False:
        errors.append('source_test_payload_status_not_explicitly_false')
    if facts.get('split') == 'validation':
        if 'evidence_role' not in report:
            unverified.append('missing_development_evidence_role')
        elif report['evidence_role'] != 'development':
            errors.append('validation_evidence_role_mismatch')
    for row in report.get('rows', []):
        eid, key = identity(row), horizon_key(row.get('horizon'))
        if not isinstance(eid, str) or key not in HORIZONS[:-1]:
            errors.append('invalid_event_id_or_horizon:' + str(eid))
            continue
        metadata = {'label': row.get('label'), 'group': str(row.get('group', 'unknown'))}
        if metadata['label'] not in ('stable', 'unstable'):
            errors.append('invalid_original_label:' + eid)
        if eid in eligible and eligible[eid] != metadata:
            errors.append('inconsistent_label_or_group:' + eid)
        eligible.setdefault(eid, metadata)
        if (eid, key) in accuracy:
            errors.append('duplicate_accuracy_row:' + eid + ':' + key)
            continue
        accuracy[(eid, key)] = dict(row)
        if type(row.get('scorable')) is not bool or type(row.get('qualified')) is not bool:
            errors.append('nonboolean_scoring_gate:' + eid + ':' + key)
        if row.get('qualified') and not row.get('scorable'):
            errors.append('qualified_without_truth:' + eid + ':' + key)
        if any(type(row.get(name)) is not int or row[name] < 0 for name in ('steps', 'true_available_steps')):
            errors.append('invalid_truth_interval:' + eid + ':' + key)
        else:
            expected_steps = min(800, row['true_available_steps']) if key == 'prefix' else int(key) * 100
            if row['steps'] != expected_steps or row.get('scorable') != (row['steps'] >= 1 and row['true_available_steps'] >= row['steps']):
                errors.append('truth_interval_or_censor_gate_mismatch:' + eid + ':' + key)
            if eid in true_lengths and true_lengths[eid] != row['true_available_steps']:
                errors.append('inconsistent_true_available_steps:' + eid)
            true_lengths.setdefault(eid, row['true_available_steps'])
        if row.get('scope') != ('available_prefix' if key == 'prefix' else 'complete'):
            errors.append('scoring_scope_mismatch:' + eid + ':' + key)
        # evaluate.metric uses pred[1:] - truth[1:]: rows.steps is the
        # number of future samples, excluding the shared initial state.
        # Counts are determined by that interval, never free pooling weights.
        metric_counts = row.get('metrics')
        if isinstance(metric_counts, dict):
            for count_name in ('n', 'common_n'):
                if type(metric_counts.get(count_name)) is not int or metric_counts[count_name] <= 0:
                    errors.append('metric_count_not_positive_integer:' + eid + ':' + key + ':' + count_name)
            if type(row.get('steps')) is int and row['steps'] >= 1:
                if metric_counts.get('common_n') != row['steps']:
                    errors.append('metric_common_n_interval_mismatch:' + eid + ':' + key)
                masses = context.get('mass')
                if isinstance(masses, list) and masses and metric_counts.get('n') != row['steps'] * len(masses):
                    errors.append('metric_n_interval_machine_count_mismatch:' + eid + ':' + key)
        if row.get('qualified') and (row.get('finite_fine') is not True or row.get('finite_coarse') is not True):
            errors.append('qualified_without_finite_pair:' + eid + ':' + key)
        if row.get('qualified'):
            differences = row.get('numerical_difference') or {}
            if (set(differences) != {'delta', 'omega', 'common', 'p'}
                    or any(not finite_nonnegative(value) or value > .001 for value in differences.values())):
                errors.append('qualified_without_frozen_component_gate:' + eid + ':' + key)
            metric = row.get('metrics') or {}
            scales = context.get('saved_training_scales')
            relative_scales = context.get('metric_scales_delta_omega')
            if (isinstance(scales, list) and len(scales) == 3 and finite_nonnegative(scales[2]) and scales[2] > 0
                    and all(finite_nonnegative(metric.get(key)) for key in ('p_sse_pu2', 'p_normalized_sse'))):
                if not same(metric['p_normalized_sse'], metric['p_sse_pu2'] / scales[2] ** 2):
                    errors.append('declared_p_scale_metric_mismatch:' + eid + ':' + key)
            if (isinstance(relative_scales, list) and len(relative_scales) == 2
                    and all(finite_nonnegative(value) and value > 0 for value in relative_scales)
                    and all(finite_nonnegative(metric.get(key)) for key in ('delta_sse_rad2', 'omega_sse_rad_s2', 'normalized_sse'))):
                expected_sse = .5 * (metric['delta_sse_rad2'] / relative_scales[0] ** 2
                                     + metric['omega_sse_rad_s2'] / relative_scales[1] ** 2)
                if not same(metric['normalized_sse'], expected_sse):
                    errors.append('declared_relative_scales_metric_mismatch:' + eid + ':' + key)
    excluded = {}
    for row in facts.get('excluded', []):
        eid = identity(row)
        if not isinstance(eid, str) or eid in excluded or eid in eligible or row.get('reason') != 'latency_uncovered':
            errors.append('invalid_or_duplicate_latency_exclusion:' + str(eid))
            continue
        excluded[eid] = {'label': row.get('label'), 'group': str(row.get('group', 'unknown'))}
        if row.get('label') not in ('stable', 'unstable'):
            errors.append('invalid_latency_original_label:' + eid)
    raw = {**eligible, **excluded}
    for row in (report.get('classification') or {}).get('rows', []):
        eid = identity(row)
        if not isinstance(eid, str) or eid in classified:
            errors.append('invalid_or_duplicate_classification_row:' + str(eid))
            continue
        classified[eid] = dict(row)
        metadata = {'label': row.get('label'), 'group': str(row.get('group', 'unknown'))}
        if eid not in raw or raw[eid] != metadata:
            errors.append('classification_population_mismatch:' + eid)
        if row.get('category') not in CATEGORIES:
            errors.append('unknown_classification:' + eid)
        if row.get('category') in CATEGORIES[:3] and row.get('numerical_gate_passed') is not True:
            errors.append('physical_classification_without_numerical_gate:' + eid)
        if bool(row.get('predicted_los')) != (row.get('category') == 'predicted_los'):
            errors.append('classification_flag_mismatch:' + eid)
    if set(classified) != set(raw):
        errors.append('classification_raw_ID_set_incomplete')
    classified_latency = {eid for eid, row in classified.items() if row.get('category') == 'latency_uncovered'}
    if classified_latency != set(excluded):
        errors.append('classification_latency_IDs_mismatch_source_exclusions')
    for eid in sorted(classified_latency & set(eligible)):
        errors.append('eligible_event_reclassified_as_latency_uncovered:' + eid)
    numerical = report.get('classification_numerical')
    if not isinstance(numerical, list):
        unverified.append('missing_classification_numerical_rows')
    else:
        seen_numerical = set()
        for row in numerical:
            eid = identity(row)
            if eid not in eligible or eid in seen_numerical:
                errors.append('invalid_or_duplicate_los_numerical_row:' + str(eid))
            seen_numerical.add(eid)
            passed = row.get('screen_pass')
            if type(passed) is not bool:
                errors.append('nonboolean_los_numerical_gate:' + str(eid))
            if passed:
                discrepancy, hit_difference = row.get('normalized_coi_discrepancy'), row.get('hit_time_difference_s')
                if (row.get('finite_to_comparison_endpoint') is not True or row.get('same_hit_or_nonhit') is not True
                        or not finite_nonnegative(discrepancy) or discrepancy > .001
                        or (hit_difference is not None and (not finite_nonnegative(hit_difference) or hit_difference > .02 + 1e-12))):
                    errors.append('los_numerical_gate_values_inconsistent:' + str(eid))
            category = classified.get(eid, {})
            if category.get('category') in CATEGORIES[:3] and passed is not True:
                errors.append('physical_classification_despite_failed_pair:' + str(eid))
        if seen_numerical != set(eligible):
            errors.append('los_numerical_eligible_ID_set_incomplete')
    if not eligible:
        errors.append('empty_initialization_eligible_population')
    for eid in eligible:
        if any((eid, h) not in accuracy for h in HORIZONS[:-1]):
            errors.append('incomplete_horizon_rows:' + eid)
    for source, key, expected in ((report, 'raw_event_count', len(raw)), (report, 'eligible_event_count', len(eligible)),
                                  (report, 'latency_uncovered_count', len(excluded)),
                                  (facts, 'raw_count', len(raw)), (facts, 'eligible_count', len(eligible))):
        if key not in source:
            unverified.append('missing_denominator:' + key)
        elif source[key] != expected:
            errors.append('denominator_mismatch:' + key)
    record = {'report': report, 'endpoint': endpoint, 'accuracy': accuracy, 'eligible': eligible,
              'excluded': excluded, 'raw': raw, 'classification': classified,
              'errors': errors, 'unverified': unverified, 'context': context, 'facts': facts}
    for horizon in HORIZONS:
        bundle = accuracy_bundle(record, horizon)
        reported = source_summary(report, horizon)
        if reported is None:
            unverified.append('missing_saved_summary:' + horizon)
        else:
            for name in ('mean', 'pooled', 'tail'):
                if not same(reported.get(name), bundle['components']['relative'][name]):
                    errors.append('saved_summary_differs_from_full_population:' + horizon + ':' + name)
            expected_counts = {'count': bundle['truth_scorable_denominator'],
                               'truth_scorable_denominator': bundle['truth_scorable_denominator'],
                               'numerically_qualified_count': bundle['numerically_qualified_count'],
                               'raw_denominator': bundle['raw_denominator']}
            for name, value in expected_counts.items():
                if name in reported and reported[name] != value:
                    errors.append('saved_summary_denominator_mismatch:' + horizon + ':' + name)
        for name, value in bundle['components'].items():
            if value['status'] == 'missing_or_inconsistent_metric_fields':
                errors.append('metric_fields_inconsistent:' + horizon + ':' + name)
    observed_los = los_bundle(record)
    saved_los = report.get('classification') or {}
    fields = {'event_count_raw': 'raw_denominator', 'eligible_event_count': 'covered_denominator',
              'latency_uncovered_count': 'latency_uncovered_count', 'unstable_total': 'raw_unstable_denominator',
              'unstable_eligible_denominator': 'covered_unstable_denominator', 'raw_unstable_detected': 'detected_true_unstable',
              'missed_true_unstable': 'missed_true_unstable', 'false_window_settled': 'false_window_settled',
              'false_los': 'false_los', 'numerical_unresolved_count': 'numerical_unresolved'}
    for saved, computed in fields.items():
        if saved in saved_los and saved_los[saved] != observed_los[computed]:
            errors.append('saved_los_count_mismatch:' + saved)
    if 'categories' in saved_los and saved_los['categories'] != observed_los['categories']:
        errors.append('saved_los_category_counts_mismatch')
    record['errors'] = sorted(set(errors))
    record['unverified'] = sorted(set(unverified))
    return record


def compare_contract(baseline, candidate):
    mismatches = list(baseline['errors']) + list(candidate['errors'])
    unverified = list(baseline['unverified']) + list(candidate['unverified'])
    if baseline['facts'].get('split') != candidate['facts'].get('split'):
        mismatches.append('split_mismatch')
    if baseline['raw'] != candidate['raw']:
        mismatches.append('raw_ID_label_group_mismatch')
    if baseline['eligible'] != candidate['eligible']:
        mismatches.append('eligible_ID_label_group_mismatch')
    if not same(normalized_facts(baseline['facts']), normalized_facts(candidate['facts'])):
        mismatches.append('source_facts_mismatch')
    for key in CONTEXT_KEYS:
        if key in baseline['context'] and key in candidate['context']:
            if not same(baseline['context'][key], candidate['context'][key]):
                mismatches.append('context_mismatch:' + key)
    for key in FROZEN_POLICY:
        if key in baseline['report'] and key in candidate['report']:
            if not same(baseline['report'][key], candidate['report'][key]):
                mismatches.append('policy_mismatch:' + key)
    intervals = lambda record: {key: (row.get('scope'), row.get('steps'), row.get('true_available_steps'), row.get('scorable'))
                                for key, row in record['accuracy'].items()}
    if intervals(baseline) != intervals(candidate):
        mismatches.append('genuine_truth_interval_mismatch')
    return {'status': 'incomparable' if mismatches else ('unverified' if unverified else 'verified_declared_report_contract'),
            'mismatches': sorted(set(mismatches)), 'unverified': sorted(set(unverified)),
            'actual_truth_values_compared': False, 'model_or_cache_executed_or_loaded': False,
            'scope': 'declared frozen report consistency; not independent trajectory recomputation'}


def differences(baseline, candidate, contract):
    result = {'status': contract['status'], 'components': {}, 'overall_improvement_claim': False}
    for name in COMPONENTS:
        left, right = baseline['components'][name], candidate['components'][name]
        comparable = (contract['status'] == 'verified_declared_report_contract'
                      and left['status'] == right['status'] == 'complete_scorable_population')
        values = {}
        for statistic in ('mean', 'pooled', 'tail'):
            a, b = left[statistic], right[statistic]
            delta = b - a if comparable else None
            values[statistic] = {'baseline': a, 'candidate': b, 'candidate_minus_M3': delta,
                                 'relative_change': delta / abs(a) if delta is not None and a != 0 else None}
        if comparable:
            delta = values['pooled']['candidate_minus_M3']
            direction = 'lower_error' if delta < 0 else ('higher_error' if delta > 0 else 'equal_error')
        else:
            direction = 'not_assessable_from_complete_comparable_population'
        result['components'][name] = {'descriptive_direction': direction, 'statistics': values,
                                     'baseline_status': left['status'], 'candidate_status': right['status']}
    if contract['status'] == 'verified_declared_report_contract':
        result['status'] = ('descriptive_difference_only' if result['components']['relative']['statistics']['pooled']['candidate_minus_M3'] is not None
                            else 'numerically_unresolved_or_no_complete_scorable_population')
    return result


def failure_registry(record):
    los = los_bundle(record)
    failures = [{key: row.get(key) for key in ('id', 'group', 'label', 'horizon', 'scorable', 'qualified',
                                              'failure', 'finite_fine', 'finite_coarse', 'numerical_difference', 'metrics')}
                for row in record['accuracy'].values() if row.get('qualified') is not True]
    integration = [row for row in record['report'].get('integration_failures', [])
                   if row.get('fine') is not None or row.get('coarse') is not None]
    return {
        'accuracy_unresolved_ids': sorted({identity(row) for row in record['accuracy'].values()
                                          if row.get('scorable') is True and row.get('qualified') is not True}),
        'truth_unavailable_ids': sorted({identity(row) for row in record['accuracy'].values() if row.get('scorable') is not True}),
        'integration_failure_ids': sorted({identity(row) for row in integration}),
        'integration_failures': integration, 'accuracy_failure_or_unavailable_rows': failures,
        'classification_category_ids': los['category_ids'],
        'missed_true_unstable_ids': los['missed_true_unstable_ids'],
        'false_window_settled_ids': los['false_window_settled_ids'], 'false_los_ids': los['false_los_ids'],
        'classification_rows_with_triggers': list(record['classification'].values()),
    }


def compare_reports(reports):
    """Pure report-to-report comparison; suitable for synthetic contract QA."""
    if not reports or set(reports) - set(ENDPOINTS):
        raise ValueError('Provide at least one recognized frozen endpoint report')
    records = {endpoint: index_report(endpoint, reports[endpoint]) for endpoint in ENDPOINTS if endpoint in reports}
    missing = [endpoint for endpoint in ENDPOINTS if endpoint not in records]
    groups = sorted({metadata['group'] for record in records.values() for metadata in record['raw'].values()})
    capabilities, contrasts, group_contrasts, los_rows = [], [], [], []
    audits = {}
    for endpoint, record in records.items():
        audits[endpoint] = {'internal_errors': record['errors'], 'unverified_fields': record['unverified']}
        los_rows.append({'endpoint': endpoint, **los_bundle(record)})
        for horizon in HORIZONS:
            bundle = accuracy_bundle(record, horizon)
            capabilities.append({'endpoint': endpoint, **bundle})
            if endpoint != 'M3' and 'M3' in records:
                contract = compare_contract(records['M3'], record)
                audits[endpoint]['comparison_to_M3'] = contract
                contrasts.append({'endpoint': endpoint, 'horizon': horizon,
                                  **differences(accuracy_bundle(records['M3'], horizon), bundle, contract)})
        if endpoint != 'M3' and 'M3' in records:
            contract = audits[endpoint]['comparison_to_M3']
            for group in groups:
                for horizon in HORIZONS:
                    baseline = accuracy_bundle(records['M3'], horizon, group)
                    candidate = accuracy_bundle(record, horizon, group)
                    comparison = differences(baseline, candidate, contract)
                    baseline_los, candidate_los = los_bundle(records['M3'], group), los_bundle(record, group)
                    changes = {name: candidate_los[name] - baseline_los[name]
                               if contract['status'] == 'verified_declared_report_contract' else None
                               for name in ('false_window_settled', 'missed_true_unstable', 'false_los',
                                            'numerical_unresolved', 'window_no_los_unsettled', 'detected_true_unstable')}
                    group_contrasts.append({
                        'endpoint': endpoint, 'group': group, 'group_metadata_missing': group == 'unknown',
                        'horizon': horizon, 'baseline': baseline, 'candidate': candidate,
                        'baseline_los': baseline_los, 'candidate_los': candidate_los,
                        'full_8s_los_count_changes_candidate_minus_M3': changes,
                        **comparison,
                    })
    return {
        'status': 'descriptive_frozen_endpoint_comparison_no_acceptance',
        'baseline': 'M3', 'provided_endpoints': list(records), 'missing_endpoints': missing,
        'all_registered_endpoints_present': not missing,
        'baseline_present': 'M3' in records,
        'audits': audits, 'capability_table': capabilities, 'contrasts_to_M3': contrasts,
        'fault_group_contrasts_to_M3': group_contrasts, 'classification_table': los_rows,
        'failure_registry': {name: failure_registry(record) for name, record in records.items()},
        'source_splits': {name: record['facts'].get('split') for name, record in records.items()},
        'evidence_role': ('frozen_node_development_evidence'
                          if all(record['facts'].get('split') == 'validation' for record in records.values())
                          else 'descriptive_report_evidence_from_declared_splits'),
        'automatic_model_acceptance': False, 'overall_pass_assigned': False, 'candidate_selection_performed': False,
        'qualified_subset_used_for_overall_improvement': False,
        'model_executed': False, 'event_payload_read': False, 'trajectory_cache_read': False,
        'limitations': [
            'A missing endpoint is missing evidence, never a completed control.',
            'Every aggregate uses all declared scorable events; any unresolved member makes that aggregate unavailable.',
            'Pair differences are suppressed for mismatched or unverified report contracts.',
            'Report context and genuine interval declarations are checked; actual truth values and model execution are not repeated.',
            'Separate physical, classification and numerical harms remain visible; no scalar score confers acceptance.',
            'No same-accuracy speed conclusion or original modal-match conclusion is calculated here.',
            'Different optimization paths or budgets are not treated as an isolated causal-factor experiment.',
        ],
    }


def number(value):
    return '—' if value is None else f'{value:.6g}'


def escape(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown_report(result):
    missing = ', '.join(result['missing_endpoints']) or '无'
    evidence = ('这些结果是固定节点的开发证据。' if result['evidence_role'] == 'frozen_node_development_evidence'
                else '这些结果按报告声明的来源分割作描述性比较。')
    text = ['# 冻结端点能力对比', '',
            evidence + '本文不接受模型、不判总体通过，也不选择新候选。', '',
            f"已提供：{', '.join(result['provided_endpoints'])}。缺失端点：{missing}。主参照为 M3。", '',
            '## 对比口径核查', '', '| 端点 | 来源分割 | 内部错误 | 未核验字段 | 与 M3 的口径 |', '|---|---|---:|---:|---|']
    problem_notes = []
    for name, audit in result['audits'].items():
        status = audit.get('comparison_to_M3', {}).get('status', 'baseline' if name == 'M3' else 'M3 missing')
        text.append(f"| {name} | {escape(result['source_splits'][name])} | {len(audit['internal_errors'])} | {len(audit['unverified_fields'])} | {escape(status)} |")
        problems = audit['internal_errors'] + audit['unverified_fields']
        problems += audit.get('comparison_to_M3', {}).get('mismatches', [])
        if problems:
            problem_notes.append(f"{name}：" + '; '.join(escape(value) for value in sorted(set(problems))))
    for note in problem_notes:
        text.extend(['', note])
    text.extend(['', '## 轨迹能力', '',
                 '误差 mean/pooled/tail 均使用全部真实可评分事件。任一成员数值未决时，该总体不显示聚合误差。', '',
                 '| 端点 | 时域 | 原始/可初始化/真值可评分/数值合格 | 相对运动 mean | pooled | tail | 共同频率 pooled (rad/s) | p pooled (pu) |',
                 '|---|---|---|---:|---:|---:|---:|---:|'])
    for row in result['capability_table']:
        components = row['components']
        rel = components['relative']
        denominators = '/'.join(str(row[key]) for key in ('raw_denominator', 'initialization_eligible_denominator', 'truth_scorable_denominator', 'numerically_qualified_count'))
        text.append(f"| {row['endpoint']} | {row['horizon']} | {denominators} | {number(rel['mean'])} | {number(rel['pooled'])} | {number(rel['tail'])} | {number(components['common_rad_s']['pooled'])} | {number(components['p_pu']['pooled'])} |")
    text.extend(['', '各物理量的完整 mean/pooled/tail、失败与真值缺失 ID 均保存在 comparison.json。', '',
                 '## 相对 M3 的描述性差值', '',
                 '差值=候选−M3；负值表示该项误差较低。它不抵消错误判稳、漏检或其他物理量损害。', '',
                 '| 端点 | 时域 | 状态 | 相对运动 pooled 差值 | common pooled 差值 | p pooled 差值 |',
                 '|---|---|---|---:|---:|---:|'])
    for row in result['contrasts_to_M3']:
        delta = lambda name: number(row['components'][name]['statistics']['pooled']['candidate_minus_M3'])
        text.append(f"| {row['endpoint']} | {row['horizon']} | {escape(row['status'])} | {delta('relative')} | {delta('common_rad_s')} | {delta('p_pu')} |")
    text.extend(['', '## 故障组改善与损害', '',
                 '逐组使用该组全部可评分事件；unknown 表示来源缺少故障组信息。', '',
                 '| 端点 | 故障组 | 时域 | 相对 pooled 差值 | common pooled 差值 | p pooled 差值 | 完整8s计数Δ：误判稳/漏检/假LOS/数值未决 | 该组候选数值未决 ID |',
                 '|---|---|---|---:|---:|---:|---|---|'])
    for row in result['fault_group_contrasts_to_M3']:
        delta = lambda name: number(row['components'][name]['statistics']['pooled']['candidate_minus_M3'])
        ids = ', '.join(row['candidate']['numerically_unresolved_ids']) or '—'
        changes = row['full_8s_los_count_changes_candidate_minus_M3']
        harms = '/'.join(number(changes[name]) for name in ('false_window_settled', 'missed_true_unstable', 'false_los', 'numerical_unresolved'))
        text.append(f"| {row['endpoint']} | {escape(row['group'])} | {row['horizon']} | {delta('relative')} | {delta('common_rad_s')} | {delta('p_pu')} | {harms} | {escape(ids)} |")
    text.extend(['', '## 完整 LOS 与时延口径', '',
                 '| 端点 | 原始/可覆盖/时延不可覆盖 | 原始失稳/可覆盖失稳 | 检出 | 漏检 | 错误 window_settled | 假 LOS | 未定 | 数值未决 | 检出率 原始/可覆盖 |',
                 '|---|---|---|---:|---:|---:|---:|---:|---:|---|'])
    for row in result['classification_table']:
        text.append(f"| {row['endpoint']} | {row['raw_denominator']}/{row['covered_denominator']}/{row['latency_uncovered_count']} | {row['raw_unstable_denominator']}/{row['covered_unstable_denominator']} | {row['detected_true_unstable']} | {row['missed_true_unstable']} | {row['false_window_settled']} | {row['false_los']} | {row['window_no_los_unsettled']} | {row['numerical_unresolved']} | {number(row['detection_rate_raw'])}/{number(row['detection_rate_covered'])} |")
    text.extend(['', '## 失败事件清单', ''])
    for endpoint, failures in result['failure_registry'].items():
        text.append(f'### {endpoint}')
        text.append('')
        for label, ids in (
            ('轨迹数值未决', failures['accuracy_unresolved_ids']),
            ('积分失败（包含晚期失败）', failures['integration_failure_ids']),
            ('真值不足的时域', failures['truth_unavailable_ids']),
            ('可覆盖真实失稳漏检', failures['missed_true_unstable_ids']),
            ('错误 window_settled', failures['false_window_settled_ids']),
            ('假 LOS', failures['false_los_ids']),
            ('分类数值未决', failures['classification_category_ids']['numerical_unresolved']),
            ('窗口未定', failures['classification_category_ids']['window_no_los_unsettled']),
            ('时延不可覆盖', failures['classification_category_ids']['latency_uncovered']),
        ):
            text.append(f"- {label}：{escape(', '.join(ids)) if ids else '无'}")
        text.append('')
    text.extend(['完整触发时刻、逐事件失败原因及每故障组 LOS 变化见 comparison.json。', '',
                 '训练路径或优化预算的差异须另行交代，表中差值不作为单因素因果结论。', '',
                 '未加载轨迹缓存、未读事件载荷、未重跑模型；未验证真实轨迹数值本身。'])
    return '\n'.join(text) + '\n'


def compare_files(paths, output):
    from task4.training import check_authorized
    check_authorized(False)
    if not paths:
        raise ValueError('At least one explicit report path is required')
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    reports, sources = {}, {}
    for endpoint, value in paths.items():
        path = project_path(value)
        content = path.read_bytes()
        reports[endpoint] = json.loads(content.decode('utf-8-sig'))
        sources[endpoint] = {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest()}
    result = compare_reports(reports)
    result['input_reports'] = sources
    result['wall_seconds'] = time.monotonic() - start
    write_json(output / 'comparison.json', result)
    (output / 'capability_table.md').write_text(markdown_report(result), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m3')
    parser.add_argument('--p-n', dest='p_n')
    parser.add_argument('--p-s', dest='p_s')
    parser.add_argument('--a')
    parser.add_argument('--u')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = {name: value for name, value in (('M3', args.m3), ('P-N', args.p_n), ('P-S', args.p_s), ('A', args.a), ('U', args.u)) if value is not None}
    result = compare_files(paths, args.output)
    print(json.dumps({'provided_endpoints': result['provided_endpoints'],
                       'missing_endpoints': result['missing_endpoints'],
                       'automatic_model_acceptance': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
