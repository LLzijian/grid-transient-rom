"""Bounded independent arithmetic audit of explicitly supplied frozen caches.

No model execution, source HDF access, event search, candidate selection, or
acceptance occurs. The panel is fixed from metadata before reading predictions:
the first event per (group, original label), in frozen manifest order, at most
twelve. This audits that panel and the report's declared population bookkeeping;
it does not revalidate a complete cohort or independently reread source splits.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from task4.paths import project_path, write_json


# Intentionally independent constants: never import evaluation/classification.
DELTA_SCALE = 0.47675314588683515
OMEGA_SCALE = 1.0033667249602924
DT = .01
HORIZONS = ('1', '2', '5', '8', 'prefix')
PANEL_LIMIT = 12
FORMAL_DT = {'fine_dt': .0025, 'coarse_dt': .005}
LATENT_PER_MACHINE = {'M3': 0, 'P-N': 2, 'P-S': 2, 'A': 2, 'U': 2}


def required_state_width(report, machines):
    """A report cannot redefine the registered state rank by omitting q."""
    endpoint = report.get('endpoint')
    if endpoint not in LATENT_PER_MACHINE:
        raise ValueError('Unknown registered endpoint')
    latent = LATENT_PER_MACHINE[endpoint]
    # The current evaluator does not publish rank fields. If a report adds an
    # explicit per-machine rank declaration, it must agree with the protocol.
    # Generic "rank" is not a contract field (it can describe a modal matrix).
    for record in (report, report.get('comparison_context') or {}, report.get('model_identity') or {}):
        for key in ('latent_per_machine', 'latent_rank_per_machine'):
            if key in record and (type(record[key]) is not int or record[key] != latent):
                raise ValueError('Registered per-machine latent rank mismatch')
    return (3 + latent) * machines


def deadline_check(deadline):
    if time.time() >= deadline - 5:
        raise TimeoutError('Managed save deadline reached')


def sha256(path, deadline):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            deadline_check(deadline)
            value.update(block)
    return value.hexdigest()


def strict_json(content):
    def invalid(value):
        raise ValueError('Nonfinite JSON input: ' + value)
    return json.loads(content.decode('utf-8-sig'), parse_constant=invalid)


def same(left, right):
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return isinstance(left, (bool, np.bool_)) and isinstance(right, (bool, np.bool_)) and bool(left) == bool(right)
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-11)
    return left == right


def comparison(actual, reported):
    delta = None
    if (type(actual) in (int, float) and type(reported) in (int, float)
            and math.isfinite(actual) and math.isfinite(reported)):
        with np.errstate(over='ignore'):
            difference = float(np.float64(actual) - np.float64(reported))
        delta = difference if math.isfinite(difference) else None
    return {'recomputed': actual, 'reported': reported, 'difference': delta, 'matches': bool(same(actual, reported))}


def horizon_key(value):
    if value == 'prefix':
        return value
    number = float(value)
    if number not in (1., 2., 5., 8.):
        raise ValueError('Unexpected frozen horizon')
    return str(int(number))


def weighted_relative(values, mass):
    # Explicit scalar denominator and einsum form, independent of production COI.
    centre = np.einsum('tm,m->t', values, mass) / float(sum(mass))
    return values - centre[:, None], centre


def independent_metrics(predicted, truth, mass, p_scale):
    """Score future samples only; return overflow evidence without NaN JSON."""
    if predicted.shape != truth.shape or predicted.ndim != 3 or len(truth) < 2:
        raise ValueError('Equal [time>=2,machine,3] intervals required')
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        return None, 'nonfinite_physical_interval'
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        errors = np.subtract(predicted[1:], truth[1:])
        angle, _ = weighted_relative(errors[:, :, 0], mass)
        speed, common = weighted_relative(errors[:, :, 1], mass)
        power = errors[:, :, 2]
        n, common_n = int(angle.size), int(common.size)
        sums = {
            'delta_sse_rad2': float(np.einsum('tm,tm->', angle, angle)),
            'omega_sse_rad_s2': float(np.einsum('tm,tm->', speed, speed)),
            'common_sse_rad_s2': float(np.dot(common, common)),
            'p_sse_pu2': float(np.einsum('tm,tm->', power, power)),
        }
        sums['normalized_sse'] = .5 * (sums['delta_sse_rad2'] / DELTA_SCALE ** 2 + sums['omega_sse_rad_s2'] / OMEGA_SCALE ** 2)
        sums['p_normalized_sse'] = sums['p_sse_pu2'] / p_scale ** 2
        metrics = {'n': n, 'common_n': common_n, **sums}
        for name, sse, count in (
            ('rmse', 'normalized_sse', n), ('delta_rmse_rad', 'delta_sse_rad2', n),
            ('omega_rmse_rad_s', 'omega_sse_rad_s2', n), ('common_rmse_rad_s', 'common_sse_rad_s2', common_n),
            ('p_rmse_pu', 'p_sse_pu2', n), ('p_normalized_rmse', 'p_normalized_sse', n),
        ):
            metrics[name] = float(np.sqrt(metrics[sse] / count))
    return (metrics, None) if all(math.isfinite(value) for value in metrics.values()) else (None, 'nonfinite_metric_statistic')


def independent_difference(first, second, mass, p_scale):
    if first.shape != second.shape or len(first) < 2 or not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        error = np.subtract(first[1:], second[1:])
        angle, _ = weighted_relative(error[:, :, 0], mass)
        speed, common = weighted_relative(error[:, :, 1], mass)
        result = {'delta': float(np.max(np.abs(angle)) / DELTA_SCALE),
                  'omega': float(np.max(np.abs(speed)) / OMEGA_SCALE),
                  'common': float(np.max(np.abs(common)) / OMEGA_SCALE),
                  'p': float(np.max(np.abs(error[:, :, 2])) / p_scale)}
    return result if all(math.isfinite(value) for value in result.values()) else None


def raw_signals(history, predicted, finite, mass, equilibrium, omega_s):
    """Only original raw hit/settled signals; no classification decision library."""
    path = np.concatenate((history[:10], predicted), axis=0)
    valid = np.concatenate((np.ones(10, bool), finite)) & np.isfinite(path).all(axis=(1, 2))
    bad = np.flatnonzero(~valid)
    stop = int(bad[0]) if len(bad) else len(path)
    with np.errstate(over='ignore', invalid='ignore'):
        angles = path[:, :, 0] + equilibrium
        speed_pu = path[:, :, 1] / omega_s + 1.
        relative, _ = weighted_relative(speed_pu, mass)
        angle_spread = np.max(angles, axis=1) - np.min(angles, axis=1)
        speed_max = np.max(np.abs(relative), axis=1)
    hit, run = None, 0
    for index in range(stop):
        run = run + 1 if angle_spread[index] > math.pi and speed_max[index] > .02 else 0
        if run == 5:
            hit = index
            break
    settled = bool(len(path) >= 100 and valid[-100:].all()
                   and np.all(angle_spread[-100:] < math.pi) and np.all(speed_max[-100:] < .005))
    return {'hit_index': hit, 'tail_settled_signal': settled, 'first_invalid_output_index': stop if stop < len(path) else None}


def select_panel(metadata):
    selected, cells = [], set()
    for row in metadata:
        cell = (str(row['group']), row['label'])
        if cell not in cells and len(selected) < PANEL_LIMIT:
            selected.append(row['id'])
            cells.add(cell)
    return selected


def declared_contract(report, metadata):
    """Check identities and whole-report denominators without trajectory values."""
    if (report.get('source_identity_unchanged') is False
            or str(report.get('evidence_validity', '')).startswith('invalid')):
        raise ValueError('Invalid frozen source identity')
    facts = report.get('source_facts') or {}
    if facts.get('split') not in ('train', 'validation') or report.get('test_payload_read') is not False or facts.get('test_payload_read') is not False:
        raise PermissionError('Only explicitly TEST-free TRAIN/validation frozen reports are allowed')
    context = report.get('comparison_context') or {}
    expected = {'history_ms': 100, 'initializations_per_event': 1, 'forecast_steps': 800,
                'decision_horizon_s': 8., 'tolerance': .001, **FORMAL_DT}
    if any(not same(report.get(key), value) for key, value in expected.items()):
        raise ValueError('Frozen single-initialization policy mismatch')
    if (context.get('history_samples') != 11
            or context.get('truth_time_origin') != 'postfault t=0; initializer ends at t=0.1 s'
            or context.get('error_time_origin') != 'forecast elapsed after t=0.1 s'
            or not same(context.get('metric_scales_delta_omega'), [DELTA_SCALE, OMEGA_SCALE])):
        raise ValueError('Frozen scales or time origins mismatch')
    mass = np.asarray(context.get('mass'), dtype=float)
    scales = np.asarray(context.get('saved_training_scales'), dtype=float)
    if mass.ndim != 1 or not len(mass) or not np.isfinite(mass).all() or np.any(mass <= 0) or scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Finite positive mass and TRAIN scales required')
    required_state_width(report, len(mass))
    physics = context.get('source_physics') or {}
    if physics and not same(physics.get('mass'), mass.tolist()):
        raise ValueError('Authoritative physics mass mismatch')
    ids = [row['id'] for row in metadata]
    if not ids or len(ids) != len(set(ids)) or any(row.get('label') not in ('stable', 'unstable') or 'group' not in row for row in metadata):
        raise ValueError('Duplicate IDs or missing original metadata')
    excluded = facts.get('excluded', [])
    if any(row.get('reason') != 'latency_uncovered' for row in excluded):
        raise ValueError('Unknown population exclusion')
    excluded_ids = [row['id'] for row in excluded]
    if len(excluded_ids) != len(set(excluded_ids)) or set(ids) & set(excluded_ids):
        raise ValueError('Invalid latency exclusion identities')
    if (facts.get('eligible_count') != len(ids) or facts.get('raw_count') != len(ids) + len(excluded)
            or report.get('eligible_event_count') != len(ids) or report.get('raw_event_count') != len(ids) + len(excluded)):
        raise ValueError('Frozen population denominator mismatch')
    by_id = {row['id']: row for row in metadata}
    indexed = {}
    for row in report.get('rows', []):
        key = (row['id'], horizon_key(row['horizon']))
        if key in indexed or row['id'] not in by_id:
            raise ValueError('Duplicate or undeclared scoring row')
        meta = by_id[row['id']]
        if str(row.get('group')) != str(meta['group']) or row.get('label') != meta['label']:
            raise ValueError('Original group/label mismatch')
        if type(row.get('steps')) is not int or type(row.get('true_available_steps')) is not int or row['true_available_steps'] < 1:
            raise ValueError('Invalid true interval declaration')
        count = min(800, row['true_available_steps']) if key[1] == 'prefix' else int(key[1]) * 100
        if row['steps'] != count or type(row.get('scorable')) is not bool or row['scorable'] != (row['true_available_steps'] >= count and count > 0) or type(row.get('qualified')) is not bool or (row['qualified'] and not row['scorable']):
            raise ValueError('Invalid scorable interval gate')
        indexed[key] = row
    if set(indexed) != {(eid, horizon) for eid in ids for horizon in HORIZONS}:
        raise ValueError('Missing frozen event/horizon rows')
    summaries = {}
    for horizon in (*HORIZONS, 'severe_prefix'):
        rows = [indexed[(eid, 'prefix' if horizon == 'severe_prefix' else horizon)] for eid in ids
                if horizon != 'severe_prefix' or by_id[eid]['label'] == 'unstable']
        scorable = [row for row in rows if row['scorable']]
        saved = next((value for name, value in report.get('summary', {}).items() if name == horizon or (horizon != 'severe_prefix' and horizon_key(name) == horizon)), None)
        checks = {'count': len(scorable), 'truth_scorable_denominator': len(scorable),
                  'numerically_qualified_count': sum(row['qualified'] for row in scorable),
                  'truth_unavailable_count': len(rows) - len(scorable)}
        if saved is None or any(saved.get(key) != value for key, value in checks.items()):
            raise ValueError('Whole-report denominator mismatch: ' + horizon)
        unavailable = [row['id'] for row in rows if not row['scorable']]
        if saved.get('truth_unavailable_ids') != unavailable:
            raise ValueError('Truth-unavailable identity mismatch: ' + horizon)
        summaries[horizon] = checks
    failures = {row['id']: row for row in report.get('integration_failures', [])}
    if len(failures) != len(report.get('integration_failures', [])) or set(failures) != set(ids):
        raise ValueError('Missing or duplicate integration failure rows')
    return indexed, mass, float(scales[2]), summaries


def verify_event(report, rows, cache, local, metadata, mass, p_scale):
    eid, errors = metadata['id'], []
    machines = len(mass)
    width = required_state_width(report, machines)
    initial_states = np.asarray(cache['initial_state'], dtype=float)
    if initial_states.shape != (len(cache['event_ids']), width):
        raise ValueError('Registered initial-state width mismatch')
    truth = np.asarray(cache[f'truth_{local:03d}'], dtype=float)
    times = np.asarray(cache[f'truth_time_{local:03d}'], dtype=float)
    history = np.asarray(cache['history'][local], dtype=float)
    if truth.ndim != 3 or truth.shape[1:] != (machines, 3) or len(truth) < 12 or not np.isfinite(truth).all() or times.shape != (len(truth),) or not np.allclose(times, np.arange(len(truth)) * DT, rtol=0, atol=1e-9):
        raise ValueError('Invalid genuine truth/time grid: ' + eid)
    if history.shape != (11, machines, 3) or not np.array_equal(history, truth[:11]):
        raise ValueError('History/truth alignment mismatch: ' + eid)
    physical, subfinite, sampled_finite = {}, {}, {}
    audits = {}
    for backend in ('fine', 'coarse'):
        saved_sub = float(cache[backend + '_substeps_per_output'])
        sub = int(saved_sub)
        if sub < 1 or saved_sub != sub or not same(DT / sub, DT / max(1, math.ceil(DT / report[backend + '_dt'] - 1e-10))):
            raise ValueError('Internal integration grid mismatch')
        full = np.asarray(cache[backend + '_substep_states'][local], dtype=float)
        view = np.asarray(cache[backend + '_substep_physical'][local], dtype=float)
        saved_mask = np.asarray(cache[backend + '_substep_state_finite'][local])
        sampled = np.asarray(cache[backend + '_states'][local], dtype=float)
        physical[backend] = np.asarray(cache[backend + '_physical'][local], dtype=float)
        mask = np.asarray(cache[backend + '_state_finite'][local])
        if (full.shape != (800 * sub + 1, width) or sampled.shape != (801, width)
                or physical[backend].shape != (801, machines, 3) or view.shape != (800 * sub + 1, machines, 3)
                or saved_mask.shape != (len(full),) or saved_mask.dtype.kind != 'b'
                or mask.shape != (801,) or mask.dtype.kind != 'b'):
            raise ValueError('Invalid full-state/substep cache layout')
        actual = np.isfinite(full).all(axis=1) & np.isfinite(view).all(axis=(1, 2))
        first_bad = np.flatnonzero(~actual)
        first_bad = int(first_bad[0]) if len(first_bad) else None
        physical_from_state = np.stack((full[:, :machines], full[:, machines:2 * machines] / mass, full[:, 2 * machines:3 * machines]), axis=-1)
        comparisons = {
            'substep_mask_equals_all_state_and_physical_finite': bool(np.array_equal(saved_mask, actual)),
            'sampled_state_matches_substeps': bool(np.array_equal(sampled, full[::sub], equal_nan=True)),
            'sampled_physical_matches_substeps': bool(np.array_equal(physical[backend], view[::sub], equal_nan=True)),
            'sampled_mask_matches_substeps': bool(np.array_equal(mask, saved_mask[::sub])),
            'physical_readout_matches_state_units': bool(np.allclose(view, physical_from_state, rtol=1e-12, atol=1e-12, equal_nan=True)),
            'initial_state_matches_cache': bool(np.array_equal(full[0], initial_states[local], equal_nan=True)),
            'no_finite_resurrection_after_first_failure': first_bad is None or not bool(actual[first_bad:].any()),
        }
        failure = metadata.get(backend + '_failure')
        saved_failure = next(row[backend] for row in report['integration_failures'] if row['id'] == eid)
        comparisons['failure_metadata_matches_report'] = failure == saved_failure
        comparisons['failure_substep_matches_actual_finiteness'] = (failure is None and first_bad is None) or (first_bad is not None and isinstance(failure, dict) and failure.get('first_bad_substep') == first_bad and failure.get('first_missing_output_index') == math.ceil(first_bad / sub))
        if actual[0]:
            comparisons['initial_physical_matches_history_100ms'] = bool(np.allclose(view[0], history[-1], rtol=1e-12, atol=1e-12))
        errors.extend(backend + ':' + key for key, passed in comparisons.items() if not passed)
        audits[backend] = {**comparisons, 'first_nonfinite_substep': first_bad, 'substeps_per_output': sub,
                          'registered_full_state_width': width, 'registered_latent_per_machine': LATENT_PER_MACHINE[report['endpoint']]}
        subfinite[backend] = actual
        sampled_finite[backend] = actual[::sub]
    verified_rows = []
    available = len(truth) - 11
    for horizon in HORIZONS:
        original = rows[(eid, horizon)]
        count = min(800, available) if horizon == 'prefix' else int(horizon) * 100
        scorable = available >= count and count > 0
        expected = {'steps': count, 'true_available_steps': available, 'scorable': scorable,
                    'scope': 'available_prefix' if horizon == 'prefix' else 'complete'}
        metrics = difference = None
        failure = 'truth_interval_unavailable' if not scorable else None
        qualified = False
        if scorable:
            f, c = physical['fine'][:count + 1], physical['coarse'][:count + 1]
            fine_finite = bool(subfinite['fine'][:count * audits['fine']['substeps_per_output'] + 1].all())
            coarse_finite = bool(subfinite['coarse'][:count * audits['coarse']['substeps_per_output'] + 1].all())
            expected.update(finite_fine=fine_finite, finite_coarse=coarse_finite)
            metrics, statistic_status = independent_metrics(f, truth[10:11 + count], mass, p_scale)
            difference = independent_difference(f, c, mass, p_scale)
            qualified = bool(fine_finite and coarse_finite and difference is not None and max(difference.values()) <= .001 and metrics is not None)
            if statistic_status == 'nonfinite_metric_statistic':
                failure = statistic_status
            elif not qualified:
                failure = 'nonfinite_prediction' if not (fine_finite and coarse_finite) else 'step_discrepancy'
        expected.update(qualified=qualified, failure=failure)
        fields = {key: comparison(value, original.get(key)) for key, value in expected.items()}
        if metrics is None:
            fields['metrics'] = comparison(None, original.get('metrics'))
        else:
            for key, value in metrics.items():
                fields['metrics.' + key] = comparison(value, (original.get('metrics') or {}).get(key))
        if difference is None:
            fields['numerical_difference'] = comparison(None, original.get('numerical_difference'))
        else:
            for key, value in difference.items():
                fields['numerical_difference.' + key] = comparison(value, (original.get('numerical_difference') or {}).get(key))
        match = all(field['matches'] for field in fields.values())
        verified_rows.append({'id': eid, 'horizon': horizon, 'matches': match, 'fields': fields})
        if not match:
            errors.append('row:' + horizon)
    signals = {'status': 'unverified_missing_authoritative_physics'}
    source_physics = report['comparison_context'].get('source_physics') or {}
    if 'delta_star' in source_physics and 'omega_s' in source_physics:
        equilibrium = np.asarray(source_physics['delta_star'], float)
        omega_s = float(source_physics['omega_s'])
        if equilibrium.shape != (machines,) or not np.isfinite(equilibrium).all() or not math.isfinite(omega_s) or omega_s <= 0:
            raise ValueError('Invalid authoritative LOS restoration')
        if not np.array_equal(cache['delta_star'], equilibrium) or not same(float(cache['omega_s']), omega_s):
            raise ValueError('Cache/report LOS restoration mismatch')
        values = {name: raw_signals(history, physical[name], sampled_finite[name], mass, equilibrium, omega_s) for name in ('fine', 'coarse')}
        numeric = next((row for row in report.get('classification_numerical', []) if row['id'] == eid), {})
        classified = next((row for row in report.get('classification', {}).get('rows', []) if row['id'] == eid), {})
        checks = {name + '_hit': comparison(values[name]['hit_index'], numeric.get(name + '_hit_index')) for name in ('fine', 'coarse')}
        if classified.get('numerical_gate_passed') is True:
            checks['gated_trigger'] = comparison(values['fine']['hit_index'], classified.get('trigger_index'))
            if values['fine']['hit_index'] is None and values['fine']['first_invalid_output_index'] is None:
                checks['settled_signal'] = comparison(values['fine']['tail_settled_signal'], classified.get('window_settled'))
        elif classified.get('numerical_gate_passed') is False:
            checks['unreliable_raw_trigger'] = comparison(values['fine']['hit_index'], classified.get('unreliable_trigger_index'))
        signals = {'status': 'raw_signals_only', 'values': values, 'checks': checks,
                   'classification_category_verified': False, 'original_los_rms_gate_recomputed': False}
        errors.extend('raw_signal:' + key for key, value in checks.items() if not value['matches'])
    return {'id': eid, 'group': metadata['group'], 'label': metadata['label'], 'rows': verified_rows,
            'finite_audits': audits, 'raw_classification_signals': signals, 'matches': not errors, 'mismatches': errors}


def verify_files(evaluations, output):
    from task4.training import check_authorized
    check_authorized(False)  # Before opening any report or cache.
    deadline = float(os.environ['TASK4_SAVE_DEADLINE_UNIX'])
    if not math.isfinite(deadline):
        raise ValueError('Finite managed deadline required')
    deadline_check(deadline)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'incomplete', 'reports': [], 'errors': [], 'generation_hash_checks': [], 'automatic_model_acceptance': False,
              'candidate_selection_performed': False, 'models_executed': False, 'source_event_payloads_read': False,
              'whole_cohort_trajectory_revalidation': False, 'source_split_independently_reread': False,
              'stage_derivatives_independently_reexecuted': False,
              'selection_rule': 'manifest order; first per (group, original label); at most 12, before prediction reads',
              'evidence_role': 'bounded independent cache arithmetic verification'}
    opened = {}
    write_json(output / 'verification.json', result)
    try:
        paths = [project_path(value) for value in evaluations]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError('Distinct explicit evaluation JSON paths required')
        for path in paths:
            deadline_check(deadline)
            if path.suffix.lower() != '.json':
                raise ValueError('Evaluation JSON path required')
            content = path.read_bytes()
            opened[path] = hashlib.sha256(content).hexdigest()
            report = strict_json(content)
            facts = report.get('source_facts') or {}
            if facts.get('split') not in ('train', 'validation') or facts.get('test_payload_read') is not False or report.get('test_payload_read') is not False:
                raise PermissionError('TEST or undeclared split cache provenance refused')
            manifests = report.get('cache', {}).get('batches', [])
            if not manifests or report.get('cache', {}).get('generation_hash_algorithm') != 'SHA256 after each cache/metadata file is written':
                raise ValueError('Generation-hashed frozen cache manifest required')
            metadata, mapping = [], {}
            for manifest in manifests:
                deadline_check(deadline)
                meta_path, cache_path = project_path(manifest['metadata_path']), project_path(manifest['path'])
                if meta_path.suffix.lower() != '.json' or cache_path.suffix.lower() != '.npz' or meta_path == path or meta_path in opened:
                    raise ValueError('Invalid/duplicate explicit cache paths')
                digest = sha256(meta_path, deadline)
                opened[meta_path] = digest
                result['generation_hash_checks'].append({'path': str(meta_path), 'generation_sha256': manifest.get('metadata_sha256'),
                                                        'before_sha256': digest, 'matches': digest == manifest.get('metadata_sha256')})
                if digest != manifest.get('metadata_sha256'):
                    raise ValueError('Metadata generation SHA mismatch')
                batch_meta = strict_json(meta_path.read_bytes())
                if not isinstance(batch_meta, list) or [row['id'] for row in batch_meta] != manifest.get('event_ids'):
                    raise ValueError('Manifest/metadata event order mismatch')
                for local, row in enumerate(batch_meta):
                    if row['id'] in mapping or (row.get('metadata') or {}).get('split', facts['split']) not in ('train', 'validation'):
                        raise PermissionError('Duplicate/TEST event metadata refused')
                    metadata.append(row)
                    mapping[row['id']] = (manifest, local, row)
            rows, mass, p_scale, denominators = declared_contract(report, metadata)
            panel = select_panel(metadata)  # No NPZ/prediction value has been opened.
            record = {'evaluation': str(path), 'evaluation_sha256': opened[path], 'endpoint': report.get('endpoint'),
                      'split': facts['split'], 'selected_event_ids': panel, 'declared_population_denominators': denominators,
                      'events': [], 'status': 'incomplete', 'sample_selection_complete_before_predictions': True}
            result['reports'].append(record)
            selected = set(panel)
            for manifest in manifests:
                chosen = [eid for eid in manifest['event_ids'] if eid in selected]
                if not chosen:
                    continue
                cache_path = project_path(manifest['path'])
                if cache_path in opened:
                    raise ValueError('Duplicate cache path')
                digest = sha256(cache_path, deadline)
                opened[cache_path] = digest
                result['generation_hash_checks'].append({'path': str(cache_path), 'generation_sha256': manifest.get('sha256'),
                                                        'before_sha256': digest, 'matches': digest == manifest.get('sha256')})
                if digest != manifest.get('sha256'):
                    raise ValueError('Cache generation SHA mismatch')
                with np.load(cache_path, allow_pickle=False) as cache:
                    if cache['event_ids'].tolist() != manifest['event_ids'] or not np.array_equal(cache['mass'], mass) or not same(float(cache['p_scale']), p_scale) or not same(float(cache['sample_interval_s']), DT) or not same(float(cache['decision_horizon_s']), 8.):
                        raise ValueError('Cache identity/scales/time contract mismatch')
                    for eid in chosen:
                        deadline_check(deadline)
                        _, local, meta = mapping[eid]
                        record['events'].append(verify_event(report, rows, cache, local, meta, mass, p_scale))
            record['events'].sort(key=lambda row: panel.index(row['id']))
            record['status'] = 'verified_selected_panel' if len(record['events']) == len(panel) and all(row['matches'] for row in record['events']) else 'mismatch'
            write_json(output / 'verification.json', result)
        result['status'] = 'verified_selected_panels' if all(row['status'] == 'verified_selected_panel' for row in result['reports']) else 'mismatch'
    except Exception as exc:
        result['status'] = 'incomplete_deadline' if isinstance(exc, TimeoutError) else 'failed'
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        integrity = []
        for path, before in opened.items():
            try:
                # Final integrity remains necessary at a save deadline. Hashing
                # is bounded to files already opened; the outer runner owns cap.
                after = sha256(path, float('inf'))
                integrity.append({'path': str(path), 'before_sha256': before, 'end_sha256': after, 'unchanged': before == after})
            except OSError as exc:
                integrity.append({'path': str(path), 'before_sha256': before, 'end_sha256': None, 'unchanged': False, 'error': str(exc)})
        result['input_integrity'] = integrity
        if not all(row['unchanged'] for row in integrity):
            result['status'] = 'failed'
            result['errors'].append('Input changed or disappeared during audit')
        write_json(output / 'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify_files(args.evaluation, args.output)
    print(json.dumps({'status': result['status'], 'errors': result['errors'],
                      'panels': [row['selected_event_ids'] for row in result['reports']],
                      'whole_cohort_trajectory_revalidation': False}, ensure_ascii=False, allow_nan=False))
    if result['status'] != 'verified_selected_panels':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
