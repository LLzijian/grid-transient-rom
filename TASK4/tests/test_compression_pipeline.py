"""Small synthetic pipeline checks; never reads real TRAIN/validation/TEST."""
import importlib.util
import time

import numpy as np
import pytest
import torch

from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder
from task4.paths import ROOT, write_json
from task4.symbolic import PolynomialFunction
from task4.synthetic import fixture


@pytest.fixture
def compression_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('compression_test_module', ROOT/'scripts/compress_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(steps=31, events=3):
    data, physics = fixture(events=events, steps=steps)
    torch.manual_seed(611)
    model = PowerGraph(physics, [0.2, 0.3, 0.1], rank=2, hidden=32)
    return data, physics, model, HistoryEncoder(model, hidden=32)


def test_feature_dimensions_and_fixed_physical_mu_origins(compression_script):
    _, _, model, _ = make_model()
    states = torch.randn(4, model.dim, dtype=torch.float64)*0.1
    for name in compression_script.FUNCTION_NAMES:
        features, origins = compression_script.function_features(model, states, name)
        assert features.shape[-1] == model.function_modules()[name].inputs
        rows = 2*len(model.edges) if name.startswith('edge_') else model.n
        assert len(features) == 4*rows and len(origins) == rows
        physical_columns = 8 if name.startswith('edge_') else 4
        torch.testing.assert_close(origins[:, :-physical_columns],
                                   torch.zeros_like(origins[:, :-physical_columns]), rtol=0, atol=0)
        assert torch.count_nonzero(origins[:, -physical_columns:]) > 0
        selected, facts = compression_script.select_features(features, origins, 16, seed=91)
        again, _ = compression_script.select_features(features, origins, 16, seed=91)
        assert len(selected) <= 16 and facts['origin_rows'] == rows
        torch.testing.assert_close(selected[:rows], origins, rtol=0, atol=0)
        torch.testing.assert_close(selected, again, rtol=0, atol=0)


def test_teacher_perturbations_are_bounded_by_declared_scales(compression_script):
    _, _, model, _ = make_model()
    values = torch.randn(7, model.dim, dtype=torch.float64)*0.1
    original = values.clone()
    perturbed, scales = compression_script.perturb_states(values, model, seed=18)
    displacement = (perturbed-values).numpy()/scales
    assert np.max(np.abs(displacement)) <= 0.02+1e-12
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    with pytest.raises(ValueError):
        compression_script.perturb_states(values, model, fraction=0.0201)


@pytest.mark.parametrize('actual,reference,reason', [
    (1e308, -1e308, 'Difference overflow'),
    (1e200, 0., 'square overflow'),
    (1e200, 1e200, 'square overflow'),
])
def test_probe_diagnostic_arithmetic_overflow_is_json_safe(compression_script, actual, reference, reason):
    import json

    class ConstantProbe:
        def field(self, probes, create_graph=False):
            return torch.full_like(probes, actual)

        def energy_gradient(self, probes):
            return torch.full_like(probes, actual)

    probes = torch.zeros((2, 4), dtype=torch.float64)
    teacher = torch.full_like(probes, reference)
    result = compression_script.field_gap(ConstantProbe(), probes, teacher, teacher)
    assert not result['diagnostics_finite']
    for name in ('field', 'energy_gradient'):
        assert result[name+'_inputs_finite'] and not result[name+'_finite']
        assert result[name+'_rms_difference'] is None
        assert result[name+'_teacher_rms'] is None
        assert result[name+'_relative_rms_difference'] is None
        assert reason.lower() in result[name+'_reason'].lower()
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_perturbation_overflow_keeps_incomplete_manifest_and_last_checkpoint(compression_script, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    class LargeLatent:
        n, dim = 1, 4
        mass = torch.ones(1, dtype=torch.float64)
        scales = torch.ones(3, dtype=torch.float64)

        def initial(self, history, encoder):
            states = torch.zeros((len(history), 4), dtype=torch.float64)
            states[:, 3] = 1e308
            return states

        def split(self, states):
            return states[:, :1], states[:, 1:2], states[:, 2:3], states[:, 3:].reshape(-1, 1, 1)

    event = SimpleNamespace(id='synthetic', group='a', label='stable', y=np.zeros((12, 1, 3)))
    monkeypatch.setattr(compression_script, 'write_json', lambda path, value: write_json(path, value, root=tmp_path))
    checkpoint = tmp_path/'last.pt'
    checkpoint.write_bytes(b'prior-synthetic-checkpoint')
    with pytest.raises(FloatingPointError, match='latent perturbation scale'):
        compression_script.collect_teacher_states(LargeLatent(), None, [event], tmp_path, time.time()+120, seed=5)
    manifest = json.loads((tmp_path/'teacher_states_manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] is False and 'latent perturbation scale' in manifest['stop_reason']
    assert manifest['state_count'] == 1 and 'perturbation_count' not in manifest
    assert checkpoint.read_bytes() == b'prior-synthetic-checkpoint'
    with np.load(tmp_path/'teacher_states.npz', allow_pickle=False) as cache:
        assert cache['states'].shape == (1, 4) and np.isfinite(cache['states']).all()
    json.dumps(manifest, allow_nan=False)


def test_unexpected_nested_nonfinite_diagnostics_are_saved_before_failure(compression_script, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(compression_script, 'write_json', lambda path, value: write_json(path, value, root=tmp_path))
    report = {'complete': True, 'functions': [{'train': np.float64(np.inf), 'check': [float('nan')]}]}
    with pytest.raises(FloatingPointError, match='Nonfinite diagnostics saved'):
        compression_script.write_diagnostics(tmp_path/'partial.json', report)
    saved = json.loads((tmp_path/'partial.json').read_text(encoding='utf-8'))
    assert saved['complete'] is False and saved['diagnostics_finite'] is False
    assert saved['functions'][0] == {'train': None, 'check': [None]}
    assert {entry['path'] for entry in saved['nonfinite_diagnostics']} == {
        '$.functions[0].train', '$.functions[0].check[0]'}
    assert all(entry['reason'] for entry in saved['nonfinite_diagnostics'])
    json.dumps(saved, allow_nan=False)


def test_autonomous_samples_stop_at_each_actual_prefix(compression_script, tmp_path, monkeypatch):
    events, _, model, encoder = make_model(steps=66)
    calls = []
    def cheap_rollout(teacher, z0, steps, **kwargs):
        calls.append(steps)
        return torch.stack([z0+0.001*j for j in range(steps+1)], dim=1)
    monkeypatch.setattr(compression_script, 'rollout', cheap_rollout)
    monkeypatch.setattr(compression_script, 'write_json', lambda path, value: write_json(path, value, root=tmp_path))
    states, manifest = compression_script.collect_teacher_states(model, encoder, events, tmp_path,
                                                                  time.time()+120, seed=33)
    assert manifest['complete'] and calls == [50, 50, 50]
    assert manifest['all_train_source_ids'] == [event.id for event in events]
    encoded = [row for row in manifest['sources'] if row['kind'] == 'TRAIN_window_encoded']
    autonomous = [row for row in manifest['sources'] if row['kind'] == 'teacher_autonomous']
    perturbed = [row for row in manifest['sources'] if row['kind'] == 'teacher_perturbed']
    assert len(encoded) == 24 and len(autonomous) == 6 and len(perturbed) == 30
    assert len(states) == 60
    assert {row['horizon_seconds'] for row in autonomous} == {0.25, 0.5}
    assert all(row['sample_index'] <= row['truth_prefix_end_index'] for row in autonomous)
    assert not manifest['generated_fom_data'] and not manifest['test_payload_read']
    with np.load(tmp_path/'teacher_states.npz', allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache['states'], states.numpy())


def test_degree_selection_uses_same_support_limit_and_both_inner_errors(compression_script, monkeypatch):
    calls, retained = [], []
    def sparse_fit(teacher, x, *, max_terms, degree, ridge, gradient_weight):
        calls.append((max_terms, degree, gradient_weight, x.clone()))
        student = PolynomialFunction([[0, 0], [1, 0]], [[degree], [0.1]])
        # Cubic has smaller value error but loses after including gradients.
        return student, {'inner_check_samples': 2, 'inner_check_value_mse': {2: 0.2, 3: 0.1}[degree],
                         'inner_check_gradient_mse': {2: 0.1, 3: 2.0}[degree], 'degree': degree}
    monkeypatch.setattr(compression_script, 'fit_sparse_hermite', sparse_fit)
    inputs = torch.ones((10, 2), dtype=torch.float64)
    student, facts = compression_script.fit_candidate_libraries(None, inputs, time.time()+120,
                                                                on_candidate=lambda detail, model: retained.append(detail['degree']))
    assert facts['selected_degree'] == 2 and retained == [2, 3]
    assert [call[:3] for call in calls] == [(16, 2, 0.1), (16, 3, 0.1)]
    torch.testing.assert_close(calls[0][3], calls[1][3], rtol=0, atol=0)
    assert max(student.export()['nonzero_per_output']) <= 16


@pytest.mark.parametrize('invalid_metric', ['train_value_mse', 'train_gradient_mse'])
def test_finite_inner_score_cannot_accept_nonfinite_train_diagnostics(compression_script, monkeypatch, invalid_metric):
    import json

    retained = []

    def sparse_fit(teacher, x, *, max_terms, degree, ridge, gradient_weight):
        student = PolynomialFunction([[0, 0]], [[.1]])
        detail = {'degree': degree, 'inner_check_samples': 2, 'inner_check_value_mse': .01,
                  'inner_check_gradient_mse': .02, 'train_value_mse': .01, 'train_gradient_mse': .02}
        if degree == 2:
            detail[invalid_metric] = float('inf')
        return student, detail

    monkeypatch.setattr(compression_script, 'fit_sparse_hermite', sparse_fit)
    _, facts = compression_script.fit_candidate_libraries(
        None, torch.ones((10, 2), dtype=torch.float64), time.time()+120,
        on_candidate=lambda detail, candidate: retained.append((detail, candidate)))
    assert facts['selected_degree'] == 3
    failed = facts['candidates'][0]
    assert failed['status'] == 'failed' and failed[invalid_metric] is None
    assert failed['diagnostics_finite'] is False and invalid_metric in failed['reason']
    assert failed['nonfinite_diagnostics'][0]['path'] == '$.'+invalid_metric
    assert retained[0][1] is None and retained[1][0]['status'] == 'fitted'
    json.dumps(facts, allow_nan=False)


def test_teacher_checks_reject_wrong_source_ids_and_mode(compression_script):
    events, physics, model, _ = make_model()
    obj = {'kind': 'T4-N', 'synthetic': False, 'config': {'synthetic': False, 'arm': 'P',
            'hidden': 32, 'latent_per_machine': 2}, 'event_ids': [event.id for event in events],
           'scales': model.scales.tolist(), 'physics': physics.as_dict()}
    compression_script.validate_teacher(model, obj, events, physics, model.scales.numpy())
    obj['event_ids'] = list(reversed(obj['event_ids']))
    with pytest.raises(ValueError, match='TRAIN ID'):
        compression_script.validate_teacher(model, obj, events, physics, model.scales.numpy())
    obj['synthetic'] = True
    with pytest.raises(ValueError, match='nonsynthetic'):
        compression_script.validate_teacher(model, obj, events, physics, model.scales.numpy())


def test_adaptation_keeps_truth_losses_and_disables_shooting(compression_script):
    base = {'latent_weight': 0.1, 'torque_weight': 0.2, 'complexity_weight': 1e-5,
            'anchor_weight': 0.2, 'shooting_fraction': 0.45}
    cfg = compression_script.adaptation_config(base, ROOT/'teacher.pt', 16, 'node_energy', 'cpu', 70)
    assert cfg['steps'] == 16 and cfg['horizons'] == [0.25] and cfg['batch_size'] == 2
    assert cfg['lr'] == 1e-4 and cfg['shooting_fraction'] == 0 and cfg['anchor_weight'] == 0
    assert cfg['latent_weight'] == 0.1 and cfg['torque_weight'] == 0.2
    assert cfg['teacher_field_weight'] == cfg['teacher_gradient_weight'] == 0.1
    assert not cfg['final_single_initialization_5s_finetune_completed']


def test_unauthorized_pipeline_cannot_open_checkpoint_or_real_data(compression_script, tmp_path, monkeypatch):
    def deny(synthetic):
        assert synthetic is False
        raise PermissionError('synthetic authorization rejection fixture')
    monkeypatch.setattr(compression_script, 'check_authorized', deny)
    monkeypatch.setattr(compression_script, 'load_checkpoint', lambda *a, **k: pytest.fail('checkpoint read'))
    monkeypatch.setattr(compression_script, 'load_events', lambda *a, **k: pytest.fail('real payload read'))
    with pytest.raises(PermissionError):
        compression_script.compress('unused.pt', tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()
