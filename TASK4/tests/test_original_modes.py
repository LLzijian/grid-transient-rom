"""Independent linear modes and synthetic cache contracts; no real assets."""
import copy
import importlib.util
import json
import time

import numpy as np
import pytest
import torch

from task4.paths import ROOT


@pytest.fixture
def modes_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('original_modes_test_module', ROOT/'scripts/match_original_modes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LinearMomentumModel:
    n, rank, dim = 10, 2, 50
    mass = torch.linspace(1., 2., 10, dtype=torch.float64)
    laplacian = torch.eye(10, dtype=torch.float64)-torch.ones((10, 10), dtype=torch.float64)/10

    def field(self, state, create_graph=True):
        delta, pi, p, q = state[..., :10], state[..., 10:20], state[..., 20:30], state[..., 30:]
        omega = pi/self.mass
        # Written directly in momentum units. The interface must cancel both M factors.
        return torch.cat((omega, -self.mass*(delta@self.laplacian.T+.2*omega), -2*p, -3*q), dim=-1)


def test_pi_conversion_original_qa_and_known_full_reduced_poles(modes_script):
    model = LinearMomentumModel()
    matrix, qa = modes_script.jacobian_qa(model, time.time()+120)
    expected = np.zeros((50, 50))
    expected[:10, 10:20] = np.eye(10)
    expected[10:20, :10] = -np.eye(10)+np.ones((10, 10))/10
    expected[10:20, 10:20] = -.2*np.eye(10)
    expected[20:30, 20:30] = -2*np.eye(10)
    expected[30:, 30:] = -3*np.eye(20)
    np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-15)
    assert qa['pass'] and qa['fd_columns_completed'] == {'1e-05': 50, '1e-06': 50}
    embedding = modes_script.common_angle_embedding(50)
    assert embedding.shape == (50, 49)
    np.testing.assert_allclose(embedding.T@embedding, np.eye(49), atol=1e-14)
    values, _, residuals = modes_script.ordinary_modes(embedding.T@matrix@embedding)
    oscillatory = values[values.imag > 1e-6]
    assert len(oscillatory) == 9
    np.testing.assert_allclose(oscillatory.real, -.1, atol=1e-14)
    np.testing.assert_allclose(oscillatory.imag, np.sqrt(.99), atol=1e-14)
    assert np.count_nonzero(np.isclose(values, -.2)) == 1  # Full common frequency retained.
    assert np.count_nonzero(np.isclose(values, -2)) == 10
    assert np.count_nonzero(np.isclose(values, -3)) == 20
    assert max(residuals) <= 1e-8


def test_physical_output_coi_and_descriptor_omega_scale(modes_script):
    mass = np.arange(1., 11.)
    vector = np.zeros(50)
    vector[:10], vector[10:20] = 3., 2.
    output = modes_script.physical_output(vector, mass)
    np.testing.assert_allclose(output[:10], 0, atol=1e-14)
    np.testing.assert_allclose(output[10:], 2/modes_script.S_OMEGA, atol=1e-14)
    descriptor = vector.copy()
    descriptor[10:20] /= modes_script.OMEGA_S
    np.testing.assert_allclose(modes_script.physical_output(descriptor, mass, omega_scale=modes_script.OMEGA_S), output)


def target(index, pole):
    return {'index': index, 'real': float(pole.real), 'imag': float(pole.imag),
            'frequency_hz': float(pole.imag/(2*np.pi)), 'damping_ratio': float(-pole.real/abs(pole))}


def test_original_assignment_uses_dummies_and_never_reuses_candidate(modes_script):
    pole = -.2+1j
    values = np.array([pole, pole.conjugate(), -.4+100j, -.3+2j])
    vectors = np.zeros((50, 4), dtype=complex)
    vectors[10, 0], vectors[11, 1], vectors[12, 2], vectors[30, 3] = 1, 1, 1, 1
    output = np.r_[np.zeros(10), np.eye(10)[0]/modes_script.S_OMEGA]
    targets = [target(62, pole), target(64, pole)]
    result = modes_script.assignment(targets, [output, output], values, vectors, np.ones(10), None)
    assert result['pass'] and result['assignment_executed']
    assert len(result['matched']) == 1 and len(result['unmatched_fom']) == 1
    assert len(result['eligible_pairs']) == 2 and len(result['all_pair_metrics']) == 4
    assert result['matched'][0]['rom_index'] == 0
    assert result['excluded_null_output_rom'] == [3]
    assert result['excluded_nonpositive_imaginary_rom'] == [1]
    assert result['unmatched_rom'] == [2]
    cost = np.asarray(result['cost_matrix'])
    np.testing.assert_array_equal(cost[:, 2:], np.ones_like(cost[:, 2:]))
    assert any(row['rom_index'] is None and row['cost'] == 1 for row in result['selected_assignment'])
    assert modes_script.THRESHOLDS == {'frequency_relative': .15, 'damping_absolute': .05, 'mac': .8}


def synthetic_cache(module):
    mass = np.linspace(1., 2., 10)
    values = -.1+1j*np.arange(1, 14)
    vectors = np.eye(24, 13, dtype=complex)
    # Independent linear readout matrix; the implementation maps one column at a time.
    output_matrix = np.zeros((20, 24))
    output_matrix[:10, :10] = (np.eye(10)-np.outer(np.ones(10), mass/mass.sum()))/module.S_DELTA
    output_matrix[10:, 10:20] = module.OMEGA_S*np.eye(10)/module.S_OMEGA
    evidence = {'targets': [target(index, value) for index, value in zip(module.TARGET_IDS, values)],
                'integrity_pass': True, 'hashes_unchanged': True, 'thresholds': dict(module.THRESHOLDS)}
    data = {'target_ids': np.asarray(module.TARGET_IDS), 'target_recomputed_indices': np.arange(13),
            'descriptor_eigenvalues': values, 'descriptor_right_vectors': vectors,
            'mass': mass, 'target_outputs': output_matrix@vectors}
    return evidence, data, mass


@pytest.mark.parametrize('change', ['identity', 'threshold', 'integrity', 'mass', 'mapping', 'output'])
def test_original_cached_contract_rejects_source_changes(modes_script, change):
    evidence, data, mass = synthetic_cache(modes_script)
    targets, outputs = modes_script.validate_cached_targets(evidence, data, mass)
    assert len(targets) == len(outputs) == 13
    evidence, data = copy.deepcopy(evidence), copy.deepcopy(data)
    if change == 'identity':
        evidence['targets'][0]['index'] = 63
    elif change == 'threshold':
        evidence['thresholds']['mac'] = .79
    elif change == 'integrity':
        evidence['hashes_unchanged'] = False
    elif change == 'mass':
        data['mass'][0] += .1
    elif change == 'mapping':
        data['target_recomputed_indices'][0] = 1
    else:
        data['target_outputs'][0, 0] += .1
    with pytest.raises(ValueError):
        modes_script.validate_cached_targets(evidence, data, mass)


def test_nonfinite_qa_and_mac_overflow_leave_json_safe_failure(modes_script):
    class BadModel(LinearMomentumModel):
        def field(self, state, create_graph=True):
            return torch.full_like(state, float('nan'))

    qa, arrays = {}, {}
    with pytest.raises(FloatingPointError, match='origin field'):
        modes_script.jacobian_qa(BadModel(), time.time()+120, qa=qa, arrays=arrays)
    assert qa['pass'] is False and 'origin_field' in arrays
    pole = -.2+1j
    vectors = np.zeros((50, 1), dtype=complex)
    vectors[10, 0] = 1
    result = modes_script.assignment([target(62, pole)], [np.full(20, 1e308)],
                                     np.asarray([pole]), vectors, np.ones(10), None)
    assert not result['pass'] and not result['assignment_executed']
    assert len(result['failed_pairs']) == 1
    assert result['all_pair_metrics'][0]['mac'] is None
    json.dumps(result, allow_nan=False)
    issues = []
    clean = modes_script.json_safe({'qa': qa, 'failed_numeric': float('inf')}, issues)
    assert clean['failed_numeric'] is None and issues[0]['reason']
    json.dumps(clean, allow_nan=False)


def test_authorization_and_deadline_precede_asset_reads(modes_script, monkeypatch, tmp_path):
    import task4.training

    monkeypatch.setattr(modes_script, 'sha256', lambda path: pytest.fail('asset read before gate'))
    monkeypatch.setattr(task4.training, 'load_checkpoint', lambda *a, **k: pytest.fail('checkpoint read before gate'))

    def deny(synthetic):
        raise PermissionError('synthetic rejection fixture')

    monkeypatch.setattr(task4.training, 'check_authorized', deny)
    with pytest.raises(PermissionError):
        modes_script.run('unused.pt', tmp_path/'output')
    monkeypatch.setattr(task4.training, 'check_authorized', lambda synthetic: None)
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', str(time.time()-1))
    with pytest.raises(TimeoutError):
        modes_script.run('unused.pt', tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_fixed_cache_respects_config_and_environment_source_root(modes_script, monkeypatch, tmp_path):
    monkeypatch.delenv('TASK4_KIKI_ROOT', raising=False)
    configured = tmp_path/'configured_kiki'
    overridden = tmp_path/'override_kiki'
    config = {'kiki_root': str(configured)}
    assert modes_script.fixed_cache(config) == configured/modes_script.CACHE_RELATIVE
    monkeypatch.setenv('TASK4_KIKI_ROOT', str(overridden))
    assert modes_script.fixed_cache(config) == overridden/modes_script.CACHE_RELATIVE
