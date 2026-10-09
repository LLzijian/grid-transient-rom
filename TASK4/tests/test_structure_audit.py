"""Synthetic formula, provenance and failure-retention checks; no real payloads."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task4.data import Physics
from task4.dynamics import PowerGraph, UnstructuredGraph
from task4.paths import ROOT
from task4.symbolic import PolynomialFunction


@pytest.fixture
def audit_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('structure_audit_test_module', ROOT / 'scripts/audit_structure.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_model(n=3, symbolic=False):
    incidence = np.zeros((n, n))
    for edge in range(n):
        incidence[edge, edge] = 1
        incidence[(edge + 1) % n, edge] = -1
    physics = Physics(np.linspace(.8, 1.2, n), incidence, np.linspace(-.1, .1, n),
                      np.ones(n), np.linspace(.3, .6, n), np.linspace(.1, .3, n),
                      np.linspace(.5, .7, n), np.linspace(.2, .4, n), 2*np.pi*60)
    torch.manual_seed(97)
    model = PowerGraph(physics, [.2, .3, .1], rank=2, hidden=8, strength=.15)
    if symbolic:
        rng = np.random.default_rng(78)
        for name, module in list(model.function_modules().items()):
            powers = np.zeros((10, module.inputs), dtype=np.int64)
            for term in range(1, 10):
                for _ in range(1 + term % 3):
                    powers[term, rng.integers(module.inputs)] += 1
            coefficients = rng.normal(0, .01, (10, module.outputs))
            coefficients[::3, ::2] = 0.
            setattr(model, name, PolynomialFunction(powers, coefficients))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, physics


@pytest.mark.parametrize('n', [3, 5])
@pytest.mark.parametrize('symbolic', [False, True])
def test_independent_autograd_and_global_matrices_match_full_field(audit_script, n, symbolic):
    model, _ = synthetic_model(n, symbolic)
    state = np.random.default_rng(777).normal(0, .08, model.dim)
    result = audit_script.decompose_state(model, state)
    assert result['all_finite'] and result['numerically_verified']
    np.testing.assert_allclose(result['gradient_autograd'], result['gradient_analytic'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result['manual_field'], result['field'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result['J'] + result['J'].T, 0, atol=1e-15)
    np.testing.assert_allclose(result['R'] - result['R'].T, 0, atol=1e-15)
    assert np.linalg.eigvalsh(result['R']).min() >= -1e-14
    stats = result['statistics']
    assert abs(stats['power_identity_residual']) < 1e-12
    assert abs(stats['J_power_residual']) < 1e-12
    assert abs(stats['canonical_power_residual']) < 1e-12
    assert stats['dissipation'] >= 0
    assert set(result['components']) == set(audit_script.COMPONENTS)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_zero_learned_functions_recover_known_fixed_motion_p_and_governor(audit_script):
    model, physics = synthetic_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    state = np.linspace(-.13, .27, model.dim)
    n = model.n
    delta, pi, p, q = state[:n], state[n:2*n], state[2*n:3*n], state[3*n:].reshape(n, 2)
    omega = pi / physics.mass
    angle = delta @ physics.incidence
    phase = physics.delta_star @ physics.incidence
    electric = ((np.sin(phase + angle) - np.sin(phase)) * physics.coupling) @ physics.incidence.T
    dq = -.25*q
    dq[:, 0] -= model.strength*omega
    expected = np.concatenate((omega, -electric - physics.damping*omega + p + model.strength*q[:, 0],
                               -physics.rate*p - physics.gain*omega, dq.reshape(-1)))
    result = audit_script.decompose_state(model, state)
    np.testing.assert_allclose(result['field'], expected, atol=1e-14, rtol=1e-13)
    np.testing.assert_array_equal(result['components']['kinematics'][:n], omega)
    np.testing.assert_array_equal(result['components']['fixed_p'][n:2*n], p)
    np.testing.assert_allclose(result['components']['learned_port_pi'], 0, atol=1e-15)
    np.testing.assert_allclose(result['components']['learned_port_p'], 0, atol=1e-15)
    stats = result['statistics']
    assert abs(stats['supply']) < 1e-14
    assert stats['internal_supply_cancellation'] == pytest.approx(1.)
    assert stats['Hdot_complete_field'] == pytest.approx(-stats['dissipation'], abs=1e-13)


def test_independent_path_detects_corrupted_analytic_gradient(audit_script, monkeypatch):
    model, _ = synthetic_model()
    original = model.energy_gradient

    def incorrect_gradient(x, origins=None):
        return original(x, origins) + .007

    monkeypatch.setattr(model, 'energy_gradient', incorrect_gradient)
    facts, result = audit_script.inspect_state(model, np.full(model.dim, .05), True)
    assert facts['status'] == 'independent_formula_mismatch'
    assert not facts['numerically_verified']
    assert result['statistics']['gradient_maxabs_difference'] > .006


def test_nonfinite_preselected_cache_point_is_not_evaluated_or_replaced(audit_script, monkeypatch):
    model, _ = synthetic_model()

    def forbidden(*args, **kwargs):
        raise AssertionError('No substitute or field calculation for an invalid cached state')

    monkeypatch.setattr(audit_script, 'decompose_state', forbidden)
    for state, flag in [(np.full(model.dim, np.nan), False), (np.ones(model.dim), False),
                        (np.full(model.dim, np.inf), True)]:
        facts, result = audit_script.inspect_state(model, state, flag)
        assert result is None
        assert facts['status'] == 'cached_nonfinite_or_invalid'
        assert not facts['numerically_verified']


def test_finite_1e308_state_overflow_is_failed_and_strict_json_preserves_reason(audit_script):
    model, _ = synthetic_model(symbolic=True)
    state = np.full(model.dim, 1e308)
    facts, result = audit_script.inspect_state(model, state, True)
    assert facts['state_all_finite']
    assert facts['status'] in ('nonfinite_audit', 'audit_exception')
    assert not facts['numerically_verified']
    serialized = audit_script.report_for_json({'rows': [facts]})
    json.dumps(serialized, allow_nan=False)
    if result is not None:
        assert not result['all_finite']
        assert facts['nonfinite_arithmetic_fields']
        assert serialized['nonfinite_json_fields']
        assert all(row['reason'] == 'nonfinite_arithmetic' for row in serialized['nonfinite_json_fields'])
        assert any(row['value'] is None for row in serialized['nonfinite_json_fields'])


def synthetic_report(ids=('short-severe', 'full-stable'), steps=800):
    return {'forecast_steps': steps, 'decision_horizon_s': steps*.01,
            'eligible_event_count': len(ids),
            'cache': {'batches': [{'event_ids': list(ids), 'path': 'explicit.npz', 'metadata_path': 'explicit.json'}]},
            'rows': [{'id': event, 'group': f'g{index}', 'label': 'unstable' if index == 0 else 'stable',
                      'horizon': 'prefix', 'true_available_steps': 7 if index == 0 else steps}
                     for index, event in enumerate(ids)]}


def test_plan_samples_whole_forecast_after_severe_truth_prefix(audit_script):
    report = synthetic_report()
    rows = audit_script.fixed_plan(report)
    assert len(rows) == 16
    severe = [row for row in rows if row['id'] == 'short-severe']
    np.testing.assert_array_equal([row['output_index'] for row in severe], np.linspace(0, 800, 8, dtype=int))
    assert severe[-1]['output_index'] == 800
    assert severe[-1]['forecast_elapsed_s'] == 8.
    assert severe[-1]['postfault_time_s'] == 8.1
    assert severe[-1]['within_true_prefix_metadata'] is False
    # Changing truth length cannot change the fixed sampling indices.
    report['rows'][0]['true_available_steps'] = 1
    altered = audit_script.fixed_plan(report)
    assert [row['output_index'] for row in altered] == [row['output_index'] for row in rows]


def test_total_cap_reduces_equal_counts_without_dropping_events(audit_script):
    report = synthetic_report(tuple(f'e{i}' for i in range(346)))
    rows = audit_script.fixed_plan(report)
    assert len(rows) == 692
    for event in report['cache']['batches'][0]['event_ids']:
        assert [row['output_index'] for row in rows if row['id'] == event] == [0, 800]
    with pytest.raises(ValueError, match='every event'):
        audit_script.fixed_plan(synthetic_report(tuple(f'e{i}' for i in range(513))))


def test_cache_read_whitelist_never_materializes_history_truth_or_other_grid(audit_script):
    model, physics = synthetic_model()
    report = synthetic_report(steps=8)
    report['comparison_context'] = {'source_physics': physics.as_dict()}
    declared = report['cache']['batches'][0]
    metadata = [{'id': event} for event in declared['event_ids']]
    values = {'event_ids': np.asarray(declared['event_ids']), 'mass': physics.mass,
              'p_scale': np.asarray(.1), 'delta_star': physics.delta_star, 'omega_s': np.asarray(physics.omega_s),
              'decision_horizon_s': np.asarray(.08), 'sample_interval_s': np.asarray(.01),
              'fine_states': np.zeros((2, 9, model.dim)), 'fine_state_finite': np.ones((2, 9), dtype=bool)}
    accesses = []

    class WhitelistedArchive:
        def __getitem__(self, key):
            assert key in values, f'Unauthorized archive member: {key}'
            accesses.append(key)
            return values[key]

    states, flags = audit_script.read_prediction_batch(WhitelistedArchive(), metadata, declared, report, model)
    assert states.shape == (2, 9, model.dim) and flags.all()
    assert set(accesses) == set(values)
    values['event_ids'] = values['event_ids'][::-1]
    with pytest.raises(ValueError, match='order mismatch'):
        audit_script.read_prediction_batch(WhitelistedArchive(), metadata, declared, report, model)


def make_arrays(module, model, states, valid):
    arrays = {name: np.zeros_like(states) for name in
              ('gradient_autograd', 'gradient_analytic', 'field', 'manual_field') + module.COMPONENTS}
    arrays['states'] = states
    arrays['cache_valid'] = np.asarray(valid)
    return arrays


def test_actual_rms_normalization_reports_q_and_failed_coverage(audit_script):
    model, _ = synthetic_model()
    states = np.ones((3, model.dim))
    states[:2, :model.n] = 2.
    states[:2, model.n:2*model.n] = 3*model.mass.numpy()
    states[:2, 2*model.n:3*model.n] = 4.
    states[:2, 3*model.n:] = 5.
    states[2] = np.nan
    arrays = make_arrays(audit_script, model, states, [True, True, False])
    arrays['fixed_p'][:2, model.n:2*model.n] = 6*model.mass.numpy()
    summary = audit_script.summarize([{'status': 'verified'}, {'status': 'verified'},
                                     {'status': 'cached_nonfinite_or_invalid'}], arrays, model)
    assert summary['actual_state_rms'] == pytest.approx({'delta': 2., 'omega': 3., 'p': 4., 'q': 5.})
    assert summary['normalized_component_derivative_rms_per_s']['fixed_p']['omega'] == pytest.approx(2.)
    assert summary['q_amplitude']['rms'] == 5.
    assert summary['q_amplitude']['maxabs'] == 5.
    assert summary['normalization_state_count'] == summary['contribution_state_count'] == 2
    assert not summary['all_preselected_states_verified']
    assert summary['status_counts']['cached_nonfinite_or_invalid'] == 1


def test_rms_and_aggregate_ratio_overflow_do_not_break_strict_json(audit_script):
    assert audit_script.rms([1e308, -1e308]) == 1e308
    model, _ = synthetic_model()
    states = np.zeros((1, model.dim))
    arrays = make_arrays(audit_script, model, states, [True])
    arrays['field'][:, :model.n] = 1e308
    summary = audit_script.summarize([{'status': 'verified'}], arrays, model)
    assert not summary['all_preselected_states_verified']
    assert summary['nonfinite_arithmetic_fields']
    serialized = audit_script.report_for_json({'summary': summary})
    assert serialized['summary']['normalized_component_derivative_rms_per_s']['field']['delta'] is None
    assert serialized['nonfinite_json_fields'][0]['reason'] == 'nonfinite_arithmetic'
    json.dumps(serialized, allow_nan=False)


def test_authorization_precedes_any_input_or_output_access(audit_script, monkeypatch):
    calls = []

    def denied(synthetic):
        calls.append(synthetic)
        raise PermissionError('managed authorization absent')

    def forbidden(*args, **kwargs):
        raise AssertionError('Unauthorized input read/output creation')

    monkeypatch.setattr(audit_script, 'check_authorized', denied)
    monkeypatch.setattr(audit_script, 'project_path', forbidden)
    monkeypatch.setattr(audit_script, 'read_json', forbidden)
    monkeypatch.setattr(audit_script, 'load_checkpoint', forbidden)
    with pytest.raises(PermissionError, match='authorization absent'):
        audit_script.audit('unused.pt', 'unused.json', 'unused-output')
    assert calls == [False]


def test_no_unmanaged_infinite_deadline(audit_script, monkeypatch):
    monkeypatch.setattr(audit_script, 'check_authorized', lambda synthetic: None)
    monkeypatch.delenv('TASK4_SAVE_DEADLINE_UNIX', raising=False)
    with pytest.raises(PermissionError, match='finite managed save deadline'):
        audit_script.audit('unused.pt', 'unused.json', 'unused-output')


def test_validate_hash_mode_physics_and_reject_test_provenance(audit_script):
    model, physics = synthetic_model()
    checkpoint = {'config': {'arm': 'P'}, 'kind': 'T4-N', 'synthetic': True,
                  'scales': model.scales.tolist(), 'physics': physics.as_dict()}
    report = {'endpoint': 'P-N', 'model_identity': {'checkpoint_sha256': 'abc', 'kind': 'T4-N'},
              'source_facts': {'split': 'synthetic'}, 'history_ms': 100, 'initializations_per_event': 1,
              'comparison_context': {'saved_training_scales': model.scales.tolist(), 'mass': physics.mass.tolist(),
                                     'source_physics': physics.as_dict()}}
    audit_script.validate_endpoint(model, checkpoint, report, 'abc')
    with pytest.raises(ValueError, match='hash/kind'):
        audit_script.validate_endpoint(model, checkpoint, report, 'wrong')
    report['source_facts']['split'] = 'test'
    with pytest.raises(PermissionError, match='split must'):
        audit_script.validate_endpoint(model, checkpoint, report, 'abc')
    unstructured = UnstructuredGraph(physics, [.2, .3, .1], rank=2, hidden=8)
    with pytest.raises(ValueError, match='U/M3'):
        audit_script.validate_endpoint(unstructured, checkpoint, report, 'abc')


@pytest.mark.parametrize('interrupt', [False, True])
def test_explicit_synthetic_cache_pipeline_hashes_and_saves_failures_or_deadline(
        audit_script, monkeypatch, tmp_path, interrupt):
    model, physics = synthetic_model()
    checkpoint = tmp_path / 'frozen.pt'
    checkpoint.write_bytes(b'synthetic mock checkpoint; load function is replaced')
    saved = {'config': {'arm': 'P'}, 'kind': 'T4-N', 'synthetic': True, 'step': 5,
             'scales': model.scales.tolist(), 'physics': physics.as_dict()}
    report = synthetic_report(steps=8)
    report.update(endpoint='P-N', source_facts={'split': 'synthetic'}, history_ms=100,
                  initializations_per_event=1, raw_event_count=2, latency_uncovered_count=0,
                  evidence_role='engineering_or_train',
                  model_identity={'checkpoint_sha256': audit_script.sha256(checkpoint), 'kind': 'T4-N'},
                  comparison_context={'saved_training_scales': model.scales.tolist(), 'mass': physics.mass.tolist(),
                                      'source_physics': physics.as_dict()})
    prediction_path, metadata_path = tmp_path / 'cache.npz', tmp_path / 'cache.json'
    declared = report['cache']['batches'][0]
    declared.update(path=str(prediction_path), metadata_path=str(metadata_path))
    states = np.random.default_rng(319).normal(0, .03, (2, 9, model.dim))
    states[0, -1] = np.nan
    flags = np.ones((2, 9), dtype=bool)
    flags[0, -1] = False
    np.savez(prediction_path, event_ids=np.asarray(declared['event_ids']), mass=physics.mass,
             p_scale=np.asarray(.1), delta_star=physics.delta_star, omega_s=np.asarray(physics.omega_s),
             decision_horizon_s=np.asarray(.08), sample_interval_s=np.asarray(.01),
             fine_states=states, fine_state_finite=flags,
             # Accessing either member with allow_pickle=False would fail.
             history=np.asarray({'do not read': True}, dtype=object),
             truth_000=np.asarray({'do not read': True}, dtype=object))
    metadata = [{'id': row['id'], 'group': row['group'], 'label': row['label'],
                 'fine_failure': {'reason': 'synthetic_nonfinite'} if index == 0 else None}
                for index, row in enumerate(report['rows'])]
    metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
    declared.update(sha256=audit_script.sha256(prediction_path),
                    metadata_sha256=audit_script.sha256(metadata_path))
    evaluation_path = tmp_path / 'evaluation.json'
    evaluation_path.write_text(json.dumps(report), encoding='utf-8')
    output = tmp_path / 'audit'
    authorized = []
    monkeypatch.setattr(audit_script, 'check_authorized', lambda synthetic: authorized.append(synthetic))
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', '1000000000000')
    monkeypatch.setattr(audit_script, 'project_path', lambda path: Path(path))
    monkeypatch.setattr(audit_script, 'write_json',
                        lambda path, data: Path(path).write_text(json.dumps(data, allow_nan=False), encoding='utf-8'))
    monkeypatch.setattr(audit_script, 'load_checkpoint', lambda *args, **kwargs: (model, None, None, saved))
    processed = [0]
    original_inspect = audit_script.inspect_state

    def inspect(*args, **kwargs):
        result = original_inspect(*args, **kwargs)
        processed[0] += 1
        return result

    def deadline(*args, **kwargs):
        if interrupt and processed[0] == 1:
            raise TimeoutError('synthetic deadline after one selected state')

    monkeypatch.setattr(audit_script, 'inspect_state', inspect)
    monkeypatch.setattr(audit_script, 'check_deadline', deadline)
    result = audit_script.audit(str(checkpoint), str(evaluation_path), str(output), points_per_event=3)
    persisted = json.loads((output / 'structure_audit.json').read_text(encoding='utf-8'))
    assert authorized == [False]
    assert len(result['rows']) == len(persisted['rows']) == 6
    assert all(row['output_index'] in (0, 4, 8) for row in persisted['rows'])
    assert len(persisted['inputs']) == 4
    assert all(len(item['sha256']) == 64 for item in persisted['inputs'])
    with np.load(output / 'selected_states_and_contributions.npz', allow_pickle=False) as artifacts:
        np.testing.assert_array_equal(artifacts['output_indices'], [0, 4, 8, 0, 4, 8])
        assert np.isnan(artifacts['states'][2]).all()
    assert not persisted['numerically_verified']
    if interrupt:
        assert not persisted['complete']
        assert persisted['stop_reason'] == 'save_deadline'
        assert persisted['summary']['status_counts']['not_processed'] == 5
    else:
        assert persisted['complete']
        assert persisted['cache_generation_anchor_status'] == 'fully_verified'
        assert persisted['status'] == 'audit_complete_with_failures'
        assert persisted['rows'][2]['status'] == 'cached_nonfinite_or_invalid'
        assert persisted['summary']['status_counts']['verified'] == 5
        assert all(item['unchanged'] for item in persisted['inputs'])
    with pytest.raises(FileExistsError):
        # The existing directory is never overwritten, even on an incomplete audit.
        processed[0] = 0
        audit_script.audit(str(checkpoint), str(evaluation_path), str(output))


def test_generation_anchor_detects_prior_tampering_and_marks_legacy(audit_script, tmp_path):
    prediction, metadata = tmp_path/'prediction.npz', tmp_path/'metadata.json'
    prediction.write_bytes(b'original synthetic prediction')
    metadata.write_bytes(b'original synthetic metadata')
    declared = {'sha256': audit_script.sha256(prediction),
                'metadata_sha256': audit_script.sha256(metadata)}
    assert audit_script.verify_generation_hashes(
        declared, audit_script.sha256(prediction), audit_script.sha256(metadata))
    prediction.write_bytes(b'tampered before audit; stable throughout audit')
    with pytest.raises(ValueError, match='changed since evaluation generation'):
        audit_script.verify_generation_hashes(
            declared, audit_script.sha256(prediction), audit_script.sha256(metadata))
    assert not audit_script.verify_generation_hashes({}, 'legacy_prediction', 'legacy_metadata')
    with pytest.raises(ValueError, match='Missing or malformed'):
        audit_script.verify_generation_hashes({'sha256': declared['sha256']}, declared['sha256'], 'x')
