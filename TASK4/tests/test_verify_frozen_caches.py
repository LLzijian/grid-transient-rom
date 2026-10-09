"""Constructed cache arithmetic only; no real model, source data, or cache."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import pytest

from task4.paths import ROOT


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('cache_verifier_test', ROOT / 'scripts/verify_frozen_caches.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_metrics(count):
    # These two coordinate errors have zero weighted COI for mass=(1,3).
    # Relative angle/speed MSE=.05; common speed error=.4; power MSE=.1.
    ds, ws = .47675314588683515, 1.0033667249602924
    n = 2 * count
    normalized = .5 * (.05 / ds ** 2 + .05 / ws ** 2)
    return {'n': n, 'common_n': count, 'normalized_sse': normalized * n,
            'rmse': math.sqrt(normalized), 'delta_sse_rad2': .05 * n,
            'omega_sse_rad_s2': .05 * n, 'common_sse_rad_s2': .16 * count,
            'p_sse_pu2': .1 * n, 'p_normalized_sse': .1 / .5 ** 2 * n,
            'delta_rmse_rad': math.sqrt(.05), 'omega_rmse_rad_s': math.sqrt(.05),
            'common_rmse_rad_s': .4, 'p_rmse_pu': math.sqrt(.1),
            'p_normalized_rmse': math.sqrt(.1) / .5}


@pytest.mark.parametrize('marker', [dict(source_identity_unchanged=False),
                                  dict(evidence_validity='invalid_source_changed')])
def test_invalid_source_marker_is_rejected_before_cache_scoring(verifier, marker):
    with pytest.raises(ValueError, match='Invalid frozen source identity'):
        verifier.declared_contract(marker, [])


def constructed_files(directory, *, late_failure=False, short=False, endpoint='P-N'):
    """Cache saved arrays are made algebraically, with no ODE execution."""
    directory.mkdir()
    mass = np.array([1., 3.])
    times = np.arange(811) * .01
    truth = np.zeros((811, 2, 3))
    truth[:, :, 0] = times[:, None] * np.array([.1, -.2])
    truth[:, :, 1] = times[:, None] * .01
    predicted = truth[10:].copy()
    predicted[1:, :, 0] += [.3, -.1]
    predicted[1:, :, 1] += [.7, .3]
    predicted[1:, :, 2] += [.2, -.4]
    arrays = {'event_ids': np.array(['first', 'second']), 'history': np.stack((truth[:11], truth[:11])),
              'mass': mass, 'p_scale': np.asarray(.5), 'sample_interval_s': np.asarray(.01),
              'decision_horizon_s': np.asarray(8.), 'delta_star': np.zeros(2), 'omega_s': np.asarray(100.)}
    latent_width = 0 if endpoint == 'M3' else 2 * len(mass)
    initial = np.concatenate((predicted[0, :, 0], predicted[0, :, 1] * mass,
                              predicted[0, :, 2], np.zeros(latent_width)))
    arrays['initial_state'] = np.stack((initial, initial))
    metadata = [{'id': 'first', 'group': 'g1', 'label': 'stable', 'source': 'constructed', 'metadata': {}},
                {'id': 'second', 'group': 'g2', 'label': 'unstable', 'source': 'constructed', 'metadata': {}}]
    for backend, sub in (('fine', 4), ('coarse', 2)):
        physical = np.stack((predicted, predicted))
        internals = np.concatenate((np.repeat(physical[:, :-1], sub, axis=1), physical[:, -1:]), axis=1)
        states = np.concatenate((internals[:, :, :, 0], internals[:, :, :, 1] * mass,
                                 internals[:, :, :, 2], np.zeros((2, len(internals[0]), latent_width))), axis=-1)
        mask = np.ones((2, 800 * sub + 1), bool)
        for index in range(2):
            failure = None
            if late_failure and index == 1:
                bad = 200 * sub + 1
                states[index, bad:] = np.nan
                internals[index, bad:] = np.nan
                mask[index, bad:] = False
                failure = {'first_bad_substep': bad, 'first_missing_output_index': 201,
                           'stage': 'k2_field', 'reason': 'constructed_latent_failure'}
            metadata[index][backend + '_failure'] = failure
        arrays.update({backend + '_substep_states': states, backend + '_substep_physical': internals,
                       backend + '_substep_state_finite': mask, backend + '_states': states[:, ::sub],
                       backend + '_physical': internals[:, ::sub], backend + '_state_finite': mask[:, ::sub],
                       backend + '_substeps_per_output': np.asarray(sub)})
    report = {'endpoint': endpoint, 'history_ms': 100, 'initializations_per_event': 1, 'forecast_steps': 800,
              'decision_horizon_s': 8., 'tolerance': .001, 'fine_dt': .0025, 'coarse_dt': .005,
              'test_payload_read': False, 'eligible_event_count': 2, 'raw_event_count': 3,
              'source_facts': {'split': 'validation', 'test_payload_read': False, 'raw_count': 3,
                               'eligible_count': 2, 'excluded': [{'id': 'latency', 'label': 'unstable', 'reason': 'latency_uncovered'}]},
              'comparison_context': {'history_samples': 11, 'mass': mass.tolist(), 'saved_training_scales': [.2, .4, .5],
                                     'metric_scales_delta_omega': [.47675314588683515, 1.0033667249602924],
                                     'truth_time_origin': 'postfault t=0; initializer ends at t=0.1 s',
                                     'error_time_origin': 'forecast elapsed after t=0.1 s',
                                     'source_physics': {'mass': mass.tolist(), 'delta_star': [0., 0.], 'omega_s': 100.}},
              'rows': [], 'summary': {}, 'classification': {'rows': []}, 'classification_numerical': [],
              'integration_failures': []}
    for index, meta in enumerate(metadata):
        length = 51 if short and index == 1 else 811
        arrays[f'truth_{index:03d}'] = truth[:length]
        arrays[f'truth_time_{index:03d}'] = times[:length]
        available = length - 11
        for horizon in ('1.0', '2.0', '5.0', '8.0', 'prefix'):
            count = min(800, available) if horizon == 'prefix' else int(float(horizon) * 100)
            scorable = available >= count
            finite = not (late_failure and index == 1 and count > 200)
            row = {'id': meta['id'], 'group': meta['group'], 'label': meta['label'], 'horizon': horizon,
                   'steps': count, 'true_available_steps': available, 'scorable': scorable,
                   'scope': 'available_prefix' if horizon == 'prefix' else 'complete', 'qualified': scorable and finite,
                   'metrics': reference_metrics(count) if scorable and finite else None,
                   'numerical_difference': dict.fromkeys(('delta', 'omega', 'common', 'p'), 0.) if scorable and finite else None}
            if scorable:
                row.update(finite_fine=finite, finite_coarse=finite)
            if not scorable or not finite:
                row['failure'] = 'truth_interval_unavailable' if not scorable else 'nonfinite_prediction'
            report['rows'].append(row)
        report['integration_failures'].append({'id': meta['id'], 'fine': meta['fine_failure'], 'coarse': meta['coarse_failure']})
        failed = late_failure and index == 1
        report['classification_numerical'].append({'id': meta['id'], 'fine_hit_index': None, 'coarse_hit_index': None})
        report['classification']['rows'].append({'id': meta['id'], 'numerical_gate_passed': not failed,
                                                'trigger_index': None, 'window_settled': not failed})
    for horizon in ('1.0', '2.0', '5.0', '8.0', 'prefix', 'severe_prefix'):
        rows = [row for row in report['rows'] if row['horizon'] == ('prefix' if horizon == 'severe_prefix' else horizon)
                and (horizon != 'severe_prefix' or row['label'] == 'unstable')]
        scorable = [row for row in rows if row['scorable']]
        report['summary'][horizon] = {'count': len(scorable), 'truth_scorable_denominator': len(scorable),
                                     'numerically_qualified_count': sum(row['qualified'] for row in scorable),
                                     'truth_unavailable_count': len(rows) - len(scorable),
                                     'truth_unavailable_ids': [row['id'] for row in rows if not row['scorable']]}
    cache_path, metadata_path = directory / 'batch.npz', directory / 'batch.json'
    np.savez_compressed(cache_path, **arrays)
    write_json(metadata_path, metadata)
    report['cache'] = {'generation_hash_algorithm': 'SHA256 after each cache/metadata file is written',
                       'batches': [{'path': str(cache_path), 'metadata_path': str(metadata_path), 'event_ids': ['first', 'second'],
                                    'sha256': digest(cache_path), 'metadata_sha256': digest(metadata_path)}]}
    report_path = directory / 'evaluation.json'
    write_json(report_path, report)
    return report_path, report, arrays, metadata


def allow_constructed(monkeypatch, verifier):
    monkeypatch.setattr('task4.training.check_authorized', lambda synthetic: None)
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', str(time.time() + 60))
    monkeypatch.setattr(verifier, 'project_path', lambda value: Path(value))
    monkeypatch.setattr(verifier, 'write_json', write_json)


def test_independent_scales_common_power_and_initial_sample_exclusion(verifier):
    truth = np.zeros((4, 2, 3))
    predicted = truth.copy()
    predicted[0] = 1000.  # The initializer's shared sample never enters SSE.
    predicted[1:, :, 0] = [.3, -.1]
    predicted[1:, :, 1] = [.7, .3]
    predicted[1:, :, 2] = [.2, -.4]
    metric, status = verifier.independent_metrics(predicted, truth, np.array([1., 3.]), .5)
    assert status is None
    assert metric == pytest.approx(reference_metrics(3))
    difference = verifier.independent_difference(predicted, truth, np.array([1., 3.]), .5)
    assert difference == pytest.approx({'delta': .3 / .47675314588683515, 'omega': .3 / 1.0033667249602924,
                                       'common': .4 / 1.0033667249602924, 'p': .8})


def test_fixed_manifest_selection_is_independent_of_accuracy_and_failure(verifier):
    rows = [{'id': 'hard', 'group': '0', 'label': 'unstable', 'failure': 'late'},
            {'id': 'easy', 'group': '0', 'label': 'unstable'}]
    rows += [{'id': str(index), 'group': str(index), 'label': 'stable'} for index in range(20)]
    panel = verifier.select_panel(rows)
    assert panel[0] == 'hard' and 'easy' not in panel and len(panel) == 12


def test_frozen_panel_recalculates_future_alignment_and_units(tmp_path, monkeypatch, verifier):
    allow_constructed(monkeypatch, verifier)
    path, _, _, _ = constructed_files(tmp_path / 'source')
    result = verifier.verify_files([path], tmp_path / 'audit')
    assert result['status'] == 'verified_selected_panels'
    assert result['reports'][0]['selected_event_ids'] == ['first', 'second']
    assert not result['whole_cohort_trajectory_revalidation'] and not result['automatic_model_acceptance']
    assert all(row['unchanged'] for row in result['input_integrity'])
    assert all(event['finite_audits']['fine']['physical_readout_matches_state_units'] for event in result['reports'][0]['events'])


def test_late_full_state_failure_preserves_early_scores_and_short_truth_denominators(tmp_path, monkeypatch, verifier):
    allow_constructed(monkeypatch, verifier)
    path, _, _, _ = constructed_files(tmp_path / 'late', late_failure=True)
    result = verifier.verify_files([path], tmp_path / 'late_audit')
    assert result['status'] == 'verified_selected_panels'
    event = result['reports'][0]['events'][1]
    assert event['finite_audits']['fine']['first_nonfinite_substep'] == 801
    early, late = event['rows'][0], event['rows'][2]
    assert early['fields']['qualified']['recomputed'] is True
    assert late['fields']['qualified']['recomputed'] is False
    path, _, _, _ = constructed_files(tmp_path / 'short', short=True)
    result = verifier.verify_files([path], tmp_path / 'short_audit')
    assert result['status'] == 'verified_selected_panels'
    rows = result['reports'][0]['events'][1]['rows']
    assert rows[0]['fields']['scorable']['recomputed'] is False
    assert rows[-1]['fields']['steps']['recomputed'] == 40
    assert rows[-1]['fields']['metrics.n']['recomputed'] == 80


def test_latent_nan_is_seen_even_when_physical_cache_stays_finite(tmp_path, verifier):
    _, report, arrays, metadata = constructed_files(tmp_path / 'latent')
    # Forge an output/substep latent NaN while leaving all physical states,
    # reported masks, and reported scores finite. The cache audit must fail.
    arrays['fine_substep_states'][0, 801:, -1] = np.nan
    rows, mass, power, _ = verifier.declared_contract(report, metadata)
    event = verifier.verify_event(report, rows, arrays, 0, metadata[0], mass, power)
    assert not event['matches']
    assert 'fine:substep_mask_equals_all_state_and_physical_finite' in event['mismatches']
    assert event['rows'][0]['fields']['qualified']['recomputed'] is True
    assert event['rows'][2]['fields']['qualified']['recomputed'] is False


@pytest.mark.parametrize('endpoint,width', [('M3', 6), ('P-N', 10), ('P-S', 10), ('A', 10), ('U', 10)])
def test_registered_endpoint_has_exact_full_state_width(tmp_path, verifier, endpoint, width):
    _, report, arrays, metadata = constructed_files(tmp_path / endpoint, endpoint=endpoint)
    rows, mass, power, _ = verifier.declared_contract(report, metadata)
    event = verifier.verify_event(report, rows, arrays, 0, metadata[0], mass, power)
    assert event['matches']
    assert arrays['initial_state'].shape == (2, width)
    assert event['finite_audits']['fine']['registered_full_state_width'] == width
    assert event['finite_audits']['fine']['registered_latent_per_machine'] == (0 if endpoint == 'M3' else 2)


def test_removed_q_is_refused_even_if_every_remaining_state_and_mask_is_finite(tmp_path, verifier):
    _, report, arrays, metadata = constructed_files(tmp_path / 'removed')
    for backend in ('fine', 'coarse'):
        arrays[backend + '_substep_states'] = arrays[backend + '_substep_states'][:, :, :6]
        arrays[backend + '_states'] = arrays[backend + '_states'][:, :, :6]
    arrays['initial_state'] = arrays['initial_state'][:, :6]
    rows, mass, power, _ = verifier.declared_contract(report, metadata)
    with pytest.raises(ValueError, match='initial-state width'):
        verifier.verify_event(report, rows, arrays, 0, metadata[0], mass, power)


@pytest.mark.parametrize('field', ['fine_substep_states', 'coarse_substep_states', 'fine_states', 'coarse_states', 'initial_state'])
def test_sampled_full_and_initial_widths_cannot_disagree(tmp_path, verifier, field):
    _, report, arrays, metadata = constructed_files(tmp_path / field)
    arrays[field] = arrays[field][..., :-1]
    rows, mass, power, _ = verifier.declared_contract(report, metadata)
    with pytest.raises(ValueError, match='width|cache layout'):
        verifier.verify_event(report, rows, arrays, 0, metadata[0], mass, power)


def test_unknown_endpoint_and_explicit_wrong_per_machine_rank_are_refused(tmp_path, verifier):
    _, report, _, metadata = constructed_files(tmp_path / 'unknown')
    report['endpoint'] = 'renamed-neural-model'
    with pytest.raises(ValueError, match='Unknown registered endpoint'):
        verifier.declared_contract(report, metadata)
    report['endpoint'] = 'P-N'
    report['comparison_context']['latent_per_machine'] = 1
    with pytest.raises(ValueError, match='latent rank'):
        verifier.declared_contract(report, metadata)


def test_legacy_resolution_refused_before_prediction_decode(tmp_path, monkeypatch, verifier):
    allow_constructed(monkeypatch, verifier)
    path, report, _, _ = constructed_files(tmp_path / 'old_resolution')
    report.update(fine_dt=.005, coarse_dt=.01)
    write_json(path, report)
    monkeypatch.setattr(verifier.np, 'load', lambda *args, **kwargs: pytest.fail('Old resolution must not decode prediction arrays'))
    result = verifier.verify_files([path], tmp_path / 'old_resolution_audit')
    assert result['status'] == 'failed' and 'policy mismatch' in result['errors'][0]


def test_generation_sha_rejects_tamper_before_npz_decode(tmp_path, monkeypatch, verifier):
    allow_constructed(monkeypatch, verifier)
    path, report, _, _ = constructed_files(tmp_path / 'source')
    cache_path = Path(report['cache']['batches'][0]['path'])
    with cache_path.open('ab') as stream:
        stream.write(b'tampered')
    monkeypatch.setattr(verifier.np, 'load', lambda *args, **kwargs: pytest.fail('Tampered cache must not decode'))
    result = verifier.verify_files([path], tmp_path / 'audit')
    assert result['status'] == 'failed' and 'generation SHA mismatch' in result['errors'][0]


def test_authorization_precedes_all_path_reads(monkeypatch, verifier):
    def deny(_):
        raise PermissionError('not authorized')
    monkeypatch.setattr('task4.training.check_authorized', deny)
    monkeypatch.setattr(verifier, 'project_path', lambda value: pytest.fail('No path access before authorization'))
    with pytest.raises(PermissionError, match='not authorized'):
        verifier.verify_files(['poisoned.json'], 'output')


def test_test_provenance_refused_before_metadata_or_npz(tmp_path, monkeypatch, verifier):
    allow_constructed(monkeypatch, verifier)
    path, report, _, _ = constructed_files(tmp_path / 'source')
    report['source_facts']['split'] = 'test'
    report['cache']['batches'][0]['metadata_path'] = 'forbidden-do-not-open.json'
    write_json(path, report)
    monkeypatch.setattr(verifier.np, 'load', lambda *args, **kwargs: pytest.fail('TEST cache must not decode'))
    result = verifier.verify_files([path], tmp_path / 'audit')
    assert result['status'] == 'failed' and 'TEST' in result['errors'][0]


def test_time_shift_and_wrong_scales_fail_closed(tmp_path, verifier):
    _, report, arrays, metadata = constructed_files(tmp_path / 'source')
    rows, mass, power, _ = verifier.declared_contract(report, metadata)
    arrays['truth_time_000'] += .01
    with pytest.raises(ValueError, match='truth/time'):
        verifier.verify_event(report, rows, arrays, 0, metadata[0], mass, power)
    report['comparison_context']['metric_scales_delta_omega'] = [.2, .4]
    with pytest.raises(ValueError, match='scales'):
        verifier.declared_contract(report, metadata)


def test_finite_input_statistic_overflow_is_json_safe(verifier):
    prediction = np.full((3, 2, 3), 1e308)
    prediction[:, 1, :2] = -1e308
    metric, status = verifier.independent_metrics(prediction, np.zeros_like(prediction), np.array([1., 3.]), .5)
    assert metric is None and status == 'nonfinite_metric_statistic'
    assert verifier.independent_difference(prediction, np.zeros_like(prediction), np.array([1., 3.]), .5) is None
    json.dumps({'metric': metric, 'status': status}, allow_nan=False)


def test_original_strict_five_sample_los_and_tail_signal(verifier):
    history, predicted = np.zeros((11, 2, 3)), np.zeros((801, 2, 3))
    predicted[4:9, 0, 0] = math.pi + .01
    predicted[4:9, 0, 1] = 3.  # omega_s=100 and equal mass => .015pu, below .02
    signals = verifier.raw_signals(history, predicted, np.ones(801, bool), np.ones(2), np.zeros(2), 100.)
    assert signals['hit_index'] is None
    predicted[4:9, 0, 1] = 5.
    signals = verifier.raw_signals(history, predicted, np.ones(801, bool), np.ones(2), np.zeros(2), 100.)
    assert signals['hit_index'] == 18 and signals['tail_settled_signal']
    predicted[4:9, 0, 0] = math.pi  # Strict exceedance, not >=.
    assert verifier.raw_signals(history, predicted, np.ones(801, bool), np.ones(2), np.zeros(2), 100.)['hit_index'] is None
