"""Synthetic cached-center membership/provenance checks; no real data/model."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pytest

from task4.paths import ROOT


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('initializer_membership_test', ROOT/'scripts/check_region_initializers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_boundary_common_angle_and_full_state_membership(checker):
    p = [[Fraction(i == j) for j in range(9)] for i in range(9)]
    scales = [Fraction(1)]*9
    zero = np.zeros(10)
    zero[:2] = 1000  # exact common angle disappears, nothing else is removed
    assert checker.membership(zero, 2, 2, scales, p, Fraction(1))['status'] == 'inside'
    for index in (0, 2, 4, 8):  # relative delta, pi/common omega, p, q
        value = zero.copy()
        value[index] += 1
        row = checker.membership(value, 2, 2, scales, p, Fraction(1))
        assert row['status'] == 'inside' and row['V_exact'] == '1/1'
        value[index] = (1000 if index == 0 else 0)+2
        assert checker.membership(value, 2, 2, scales, p, Fraction(1))['status'] == 'outside'
    bad = zero.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match='Finite'):
        checker.membership(bad, 2, 2, scales, p, Fraction(1))


def fake_frozen_files(checker, directory):
    """Only synthetic JSON/NPZ; poison members must remain undecoded."""
    import mpmath
    from certify_local_region import backend_sources
    paths = {name: directory/(name+'.json') for name in ('certificate', 'evaluation', 'symbolic')}
    checkpoint = str(directory/'must_not_open.pt')
    model_hash = 'a'*64
    functions = {name: {'type': 'polynomial'} for name in
                 ('node_energy', 'edge_energy', 'node_exchange', 'node_dissipation',
                  'edge_exchange', 'edge_dissipation', 'port')}
    symbolic = {'format': 'task4-polynomial-powergraph', 'version': 1, 'n': 2, 'rank': 2,
                'scales': [1., 1., 1.], 'physics': {'mass': [1., 1.]}, 'functions': functions,
                'metadata': {'checkpoint': checkpoint, 'checkpoint_sha256': model_hash,
                             'step': 400, 'final_steps_complete': True}}
    paths['symbolic'].write_text(json.dumps(symbolic))
    symbolic_hash = checker.sha256(paths['symbolic'])
    exact_p = [['1/1' if i == j else '0/1' for j in range(9)] for i in range(9)]
    backend = backend_sources()
    certificate = {'complete': True, 'status': 'certified', 'L2_region_certificate': True,
                   'source_unchanged': True, 'source_mutation': False, 'interval_backend_unchanged': True,
                   'uncomputed_radii': [], 'source_model_path': str(paths['symbolic']),
                   'interval_backend_sources_before': backend, 'interval_backend_sources_after': backend,
                   'mpmath_version': mpmath.__version__,
                   'coordinates': {'full_dimension': 10, 'quotient_dimension': 9,
                                   'full_order': checker.FULL_ORDER, 'quotient_order': checker.QUOTIENT_ORDER,
                                   'lift': 'last delta=0; all pi/p/q retained',
                                   'projection_lift_left_inverse': 'exact by algebra',
                                   'xi_definition': 'xi=y/s; each stored s is a fixed exact binary64 dyadic',
                                   'pi_scale_construction': 'mass*saved TRAIN omega scale rounded once to binary64 and then fixed exactly',
                                   'scale_units': ['delta: rad', 'pi: mass unit * rad/s', 'p: system pu', 'q: latent unit'], 'time_unit': 's',
                                   'scale': [1.]*9, 'scale_exact_rationals': ['1/1']*9},
                   'quadratic_candidate': {'P_exact_rationals': exact_p, 'R_exact_rationals': exact_p,
                                           'P_exact_binary64_coefficients': np.eye(9).tolist()},
                   'validated_P': {'strict_positive': True, 'P_eigenvalue_lower_exact': '1/9'},
                   'radii_scaled_box': list(checker.RADII), 'attempts': []}
    for key in ('source_model_sha256', 'source_sha256_after_load', 'source_sha256_after_validation', 'source_sha256_final'):
        certificate[key] = symbolic_hash
    candidate = certificate['quadratic_candidate']
    import hashlib
    candidate['fixed_candidate_parameter_sha256'] = hashlib.sha256(
        json.dumps({'P': exact_p, 'R': exact_p}, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()
    for index, radius in enumerate(checker.RADII):
        row = {'complete': True, 'certified': index <= 4, 'radius_scaled_box': radius,
               'radius_exact_rational': checker.text(Fraction.from_float(radius))}
        if row['certified']:
            row.update(Q_gershgorin_lower_exact='1/1',
                       region={'V': 'xi.T P xi (no factor one half)',
                               'rho_exact_rational': checker.text(Fraction.from_float(radius)**2/18),
                               'm_P_exact_lower': '1/9', 'm_Q_exact_lower_per_s': '1/1'})
        certificate['attempts'].append(row)
    metadata = [{'id': eid, 'label': label, 'group': group} for eid, label, group in
                (('inside', 'stable', 'a'), ('outside', 'unstable', 'b'), ('invalid', 'stable', 'b'))]
    cache_dir = directory/'cache'
    cache_dir.mkdir()
    cache = cache_dir/'batch_0000.npz'
    initial = np.zeros((3, 10))
    initial[1, 2] = 1
    initial[2, -1] = np.nan
    np.savez(cache, event_ids=np.array([row['id'] for row in metadata]), initial_state=initial,
             history=np.array([{'forbidden': True}], dtype=object),
             fine_states=np.array([{'forbidden': True}], dtype=object),
             truth_000=np.array([{'forbidden': True}], dtype=object))
    meta_path = cache.with_suffix('.json')
    meta_path.write_text(json.dumps(metadata))
    excluded = {'id': 'early', 'label': 'unstable', 'group': 'a', 'reason': 'latency_uncovered'}
    evaluation = {'endpoint': 'P-S', 'source_identity_unchanged': True, 'test_payload_read': False,
                  'history_ms': 100, 'initializations_per_event': 1, 'encoder_calls_per_event': 1,
                  'forecast_steps': 800, 'decision_horizon_s': 8., 'fine_dt': .0025, 'coarse_dt': .005,
                  'raw_event_count': 4, 'eligible_event_count': 3, 'latency_uncovered_count': 1,
                  'rows': [{**row, 'horizon': horizon} for horizon in checker.HORIZONS for row in metadata],
                  'source_facts': {'split': 'validation', 'test_payload_read': False,
                                   'raw_count': 4, 'eligible_count': 3, 'unstable_count': 1, 'excluded': [excluded]},
                  'classification': {'rows': [{**row, 'category': 'numerical_unresolved'} for row in metadata]
                                             + [{**excluded, 'category': 'latency_uncovered'}]},
                  'model_identity': {'checkpoint': checkpoint, 'checkpoint_sha256': model_hash, 'kind': 'T4-S', 'step': 400},
                  'comparison_context': {'saved_training_scales': symbolic['scales'], 'mass': symbolic['physics']['mass'],
                                         'history_samples': 11, 'truth_time_origin': 'postfault t=0; initializer ends at t=0.1 s'},
                  'cache': {'directory': str(cache_dir), 'generation_hash_algorithm': 'SHA256 after each cache/metadata file is written',
                            'batches': [{'path': str(cache), 'metadata_path': str(meta_path),
                                         'sha256': checker.sha256(cache), 'metadata_sha256': checker.sha256(meta_path),
                                         'event_ids': [row['id'] for row in metadata]}]}}
    paths['certificate'].write_text(json.dumps(certificate))
    paths['evaluation'].write_text(json.dumps(evaluation))
    return paths, certificate, evaluation, symbolic


def install_temp_io(checker, monkeypatch):
    monkeypatch.setattr(checker, 'check_authorized', lambda: None)
    monkeypatch.setattr(checker, 'project_path', lambda path: Path(path).resolve())
    monkeypatch.setattr(checker, 'write_json', lambda path, data: Path(path).write_text(json.dumps(data, allow_nan=False)))
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', str(time.time()+180))


def test_complete_cache_center_check_decodes_only_two_arrays(checker, monkeypatch, tmp_path):
    pytest.importorskip('mpmath')
    files, _, _, _ = fake_frozen_files(checker, tmp_path)
    install_temp_io(checker, monkeypatch)
    accessed = []
    original = np.load
    class Guard:
        def __init__(self, archive):
            self.archive = archive
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            accessed.append(key)
            assert key in ('event_ids', 'initial_state')
            return self.archive[key]
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: Guard(original(*args, **kwargs)))
    result = checker.check(**files, output=tmp_path/'result')
    assert accessed == ['event_ids', 'initial_state']
    assert result['summary']['inside'] == result['summary']['outside'] == result['summary']['invalid'] == 1
    assert result['population']['raw'] == 4 and result['population']['eligible'] == 3
    assert result['selected_region']['radius_scaled_box'] == .003
    assert result['input_identity_unchanged'] and result['cache_identity_unchanged_at_end']
    assert not result['initializer_uncertainty_covered'] and not result['FOMguarantee']
    assert not (tmp_path/'must_not_open.pt').exists()
    assert len(result['rows']) == 3
    with pytest.raises(FileExistsError):
        checker.check(**files, output=tmp_path/'result')


def test_bad_cache_generation_hash_prevents_any_npz_decode_and_keeps_population(checker, monkeypatch, tmp_path):
    pytest.importorskip('mpmath')
    files, _, ev, _ = fake_frozen_files(checker, tmp_path)
    ev['cache']['batches'][0]['sha256'] = 'b'*64
    files['evaluation'].write_text(json.dumps(ev))
    install_temp_io(checker, monkeypatch)
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('decode before generation identity check'))
    result = checker.check(**files, output=tmp_path/'result')
    assert result['summary']['invalid'] == 3 and len(result['rows']) == 3
    assert result['status'] == 'inconclusive'


@pytest.mark.parametrize('change', ['missing_scale', 'wrong_checkpoint', 'incomplete_cert', 'wrong_ids', 'wrong_label'])
def test_provenance_or_coordinate_mismatch_fails_before_decode(checker, monkeypatch, tmp_path, change):
    pytest.importorskip('mpmath')
    files, cert, ev, symbolic = fake_frozen_files(checker, tmp_path)
    if change == 'missing_scale':
        del cert['coordinates']['scale_exact_rationals']
    elif change == 'incomplete_cert':
        cert['complete'] = False
    elif change == 'wrong_checkpoint':
        ev['model_identity']['checkpoint_sha256'] = 'b'*64
    elif change == 'wrong_ids':
        ev['cache']['batches'][0]['event_ids'][0] = 'unknown'
    else:
        ev['rows'][0]['label'] = 'unstable'
    files['certificate'].write_text(json.dumps(cert))
    files['evaluation'].write_text(json.dumps(ev))
    install_temp_io(checker, monkeypatch)
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('decode before provenance check'))
    with pytest.raises((ValueError, KeyError)):
        checker.check(**files, output=tmp_path/'result')
    stored = json.loads((tmp_path/'result'/'membership.json').read_text())
    assert stored['status'] == 'inconclusive' and not stored['complete']


def test_authorization_and_deadline_precede_artifact_io(checker, monkeypatch, tmp_path):
    monkeypatch.setattr(checker, 'read_json', lambda path: pytest.fail('artifact read before gate'))
    def denied():
        raise PermissionError('gate')
    monkeypatch.setattr(checker, 'check_authorized', denied)
    with pytest.raises(PermissionError):
        checker.check('missing', 'missing', 'missing', tmp_path/'output')
    assert not (tmp_path/'output').exists()
    monkeypatch.setattr(checker, 'check_authorized', lambda: None)
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', '0')
    with pytest.raises(TimeoutError):
        checker.check('missing', 'missing', 'missing', tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_source_change_after_membership_invalidates_every_center(checker, monkeypatch, tmp_path):
    pytest.importorskip('mpmath')
    files, _, _, symbolic = fake_frozen_files(checker, tmp_path)
    install_temp_io(checker, monkeypatch)
    original = checker.membership
    def changing_source(*args):
        result = original(*args)
        symbolic['changed_after_identity_check'] = True
        files['symbolic'].write_text(json.dumps(symbolic))
        return result
    monkeypatch.setattr(checker, 'membership', changing_source)
    result = checker.check(**files, output=tmp_path/'result')
    assert not result['input_identity_unchanged']
    assert not result['complete'] and result['status'] == 'inconclusive'
    assert result['summary']['invalid'] == 3 and result['summary']['inside'] == 0


def test_earlier_cache_changed_while_processing_later_batch_is_revoked_at_end(checker, monkeypatch, tmp_path):
    """B0 passes its own after-hash; only the final sweep can detect this edit."""
    pytest.importorskip('mpmath')
    files, _, evaluation, _ = fake_frozen_files(checker, tmp_path)
    directory = Path(evaluation['cache']['directory'])
    early, later = directory/'batch_0000.npz', directory/'batch_0001.npz'
    original_labels = {row['id']: {'id': row['id'], 'label': row['label'], 'group': row['group']}
                       for row in evaluation['rows']}
    early_ids, later_ids = ['inside'], ['outside', 'invalid']
    np.savez(early, event_ids=np.array(early_ids), initial_state=np.zeros((1, 10)))
    later_states = np.zeros((2, 10))
    later_states[0, 2], later_states[1, -1] = 1., np.nan
    np.savez(later, event_ids=np.array(later_ids), initial_state=later_states)
    batches = []
    for path, ids in ((early, early_ids), (later, later_ids)):
        metadata = path.with_suffix('.json')
        metadata.write_text(json.dumps([original_labels[eid] for eid in ids]))
        batches.append({'path': str(path), 'metadata_path': str(metadata), 'event_ids': ids,
                        'sha256': checker.sha256(path), 'metadata_sha256': checker.sha256(metadata)})
    # Both batches and their distinct legal IDs are frozen before checker I/O.
    evaluation['cache']['batches'] = batches
    files['evaluation'].write_text(json.dumps(evaluation))
    install_temp_io(checker, monkeypatch)
    original_membership, calls = checker.membership, []
    def edit_earlier_cache_on_later_membership(*args):
        calls.append(None)
        result = original_membership(*args)
        if len(calls) == 2:
            assert result['status'] == 'outside'  # First center of later batch.
            with early.open('ab') as stream:
                stream.write(b'changed-while-later-batch-is-being-processed')
        return result
    monkeypatch.setattr(checker, 'membership', edit_earlier_cache_on_later_membership)
    result = checker.check(**files, output=tmp_path/'result')
    assert len(calls) == 3 and result['status'] == 'inconclusive' and not result['complete']
    assert result['input_identity_unchanged'] and not result['cache_identity_unchanged_at_end']
    audits = result['cache_audits']
    assert audits[0]['sha256_before'] == audits[0]['sha256_after'] == batches[0]['sha256']
    assert audits[0]['sha256_final'] != batches[0]['sha256']
    assert not audits[0]['identity_unchanged_at_end'] and audits[1]['identity_unchanged_at_end']
    rows = {row['id']: row for row in result['rows']}
    assert set(rows) == {'inside', 'outside', 'invalid'} and len(result['rows']) == 3
    assert rows['inside']['status'] == 'invalid' and rows['inside']['reason'] == 'cache_changed_or_unavailable_at_end'
    assert rows['outside']['status'] == 'outside' and rows['invalid']['status'] == 'invalid'
    assert result['summary']['eligible_centers'] == 3 and result['summary']['inside'] == 0
    assert result['summary']['outside'] == 1 and result['summary']['invalid'] == 2
