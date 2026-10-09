"""Cheap constructed paths only; no frozen endpoints or event payloads."""
import importlib.util
import time

import numpy as np
import pytest
import torch

from task4.paths import ROOT


@pytest.fixture
def benchmark_script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('benchmark_endpoint_test_module', ROOT / 'scripts/benchmark_endpoints.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NumpyClock:
    n = 1
    dim = 4

    def __init__(self, fail=False):
        self.initializations = 0
        self.fields = 0
        self.fail = fail

    def initial(self, history):
        self.initializations += 1
        return np.zeros((len(history), 4), dtype=np.float64)

    def field(self, z):
        self.fields += 1
        result = np.ones_like(z)
        # Only latent derivative fails, while every physical coordinate is finite.
        if self.fail and z[0, 0] >= .011:
            result[0, 3] = np.nan
        return result

    def physical(self, z):
        return z[:, :3, None].swapaxes(-1, -2)


def test_grid_and_exactly_one_initializer(benchmark_script):
    module = benchmark_script
    grid, substeps = module.output_grid()
    assert len(grid) == 501 and grid[0] == 0 and grid[-1] == 5
    assert substeps == 4
    model = NumpyClock()
    values, states, report = module.deploy(model, np.zeros((2, 11, 1, 3)), backend='numpy',
                                   deadline=time.time() + 30, horizon=.03)
    assert model.initializations == report['initial_calls'] == 1
    assert model.fields == 3 * 4 * 4
    assert report['valid_runtime'] and values.shape == (2, 4, 1, 3)
    assert states.shape == (2, 4, 4)
    np.testing.assert_allclose(values[0, :, 0, 0], [0, .01, .02, .03], atol=1e-15)
    assert report['retained_fine_substep_states'] is False


def test_latent_failure_invalidates_batch_keeps_prefix_and_nan_future(benchmark_script):
    values, states, report = benchmark_script.deploy(NumpyClock(fail=True), np.zeros((2, 11, 1, 3)),
                                             backend='numpy', deadline=time.time() + 30, horizon=.03)
    assert not report['valid_runtime'] and not report['complete']
    assert report['first_bad']['batch_indices'] == [0]
    assert report['first_bad']['stage'] == 'k2_derivative'
    assert report['output_samples_completed'] == 2
    assert np.isfinite(values[:, :2]).all()
    assert np.isnan(values[:, 2:]).all()
    assert np.isnan(states[:, 2:]).all() and not report['all_sampled_full_states_finite']
    assert not benchmark_script.parity(values, values, [1, 1, 1])['passed']


def test_torch_no_grad_single_initial_and_deadline(benchmark_script):
    class TorchClock:
        n = 1
        dim = 4

        def __init__(self):
            self.calls = 0
            self.origins = 0

        def initial(self, history, encoder):
            assert not torch.is_grad_enabled()
            self.calls += 1
            return torch.zeros((len(history), 4), dtype=torch.float64)

        def origin_terms(self, z):
            self.origins += 1
            return (torch.zeros_like(z),)

        def field(self, z, create_graph, origins):
            assert not create_graph and not torch.is_grad_enabled() and len(origins) == 1
            return torch.ones_like(z)

        def physical(self, z):
            return z[:, None, :3]

    model = TorchClock()
    values, states, report = benchmark_script.deploy(model, np.zeros((1, 11, 1, 3)), backend='torch',
                                             reuse_origins=True, deadline=time.time() + 30, horizon=.01)
    assert report['valid_runtime'] and model.calls == model.origins == 1
    np.testing.assert_allclose(values[0, -1], .01, atol=1e-15)
    values, states, report = benchmark_script.deploy(model, np.zeros((1, 11, 1, 3)), backend='torch',
                                             deadline=time.time() - 1, horizon=.01)
    assert report['status'] == 'deadline' and report['initial_calls'] == 0
    assert not report['valid_runtime'] and np.isnan(values).all()


def test_panel_first_pair_order_uses_train_metadata_only(benchmark_script):
    rows = {name: {'fault_bus': group, 'label': label, 'trajectory_end_s': end}
            for name, group, label, end in [('late', 'a', 'unstable', '.09'),
                                           ('first', 'a', 'stable', '1'),
                                           ('repeat', 'a', 'stable', '2'),
                                           ('second', 'a', 'unstable', '1'),
                                           ('third', 'b', 'stable', '1')]}
    infos = [('source', 'never-opened.h5', {'train': list(rows), 'validation': ['dev'], 'test': ['test']}, rows)]
    selected, facts = benchmark_script.select_panel(infos, limit=2)
    assert [entry['id'] for entry, _ in selected] == ['first', 'second']
    assert len(facts['eligible']) == 4 and facts['latency_uncovered'][0]['id'] == 'late'


def test_full_state_normalization_includes_momentum_and_inferred_q(benchmark_script):
    reference = np.zeros((1, 2, 8))
    candidate = reference.copy()
    candidate[0, 1] = [.2, .2, .6, .9, .07, .07, .5, -.25]
    result = benchmark_script.state_parity(candidate, reference, [2, 3], [.2, .3, .07], 1)
    np.testing.assert_allclose(result['scale_vector'], [.2, .2, .6, .9, .07, .07, 1, 1])
    np.testing.assert_allclose(result['max_normalized_difference'], 1)
    assert result['latent_truth_claim'] is False


@pytest.mark.parametrize('reference_value,scale,reason', [
    (-1e308, 1., 'difference overflow'),
    (0., .1, 'normalization overflow'),
])
def test_finite_physical_inputs_overflow_fails_with_json_safe_evidence(
        benchmark_script, reference_value, scale, reason):
    import json

    candidate = np.full((1, 2, 1, 3), 1e308)
    reference = np.full_like(candidate, reference_value)
    assert np.isfinite(candidate).all() and np.isfinite(reference).all()
    result = benchmark_script.parity(candidate, reference, [scale] * 3)
    assert not result['passed'] and not result['valid_runtime']
    assert result['max_normalized_difference'] is None
    assert result['per_channel_max_normalized_difference'] is None
    assert reason in result['reason']
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_finite_opposite_large_q_difference_invalidates_runtime_and_is_json_safe(benchmark_script):
    import json

    candidate = np.zeros((1, 2, 4))
    reference = candidate.copy()
    candidate[0, 1, 3], reference[0, 1, 3] = 1e308, -1e308
    physical = candidate[:, :, None, :3]
    assert benchmark_script.parity(physical, physical, [1, 1, 1])['passed']
    assert np.isfinite(candidate).all() and np.isfinite(reference).all()
    result = benchmark_script.state_parity(candidate, reference, [2], [.2, .3, .07], 1)
    assert not result['passed'] and not result['valid_runtime'] and not result['complete_finite']
    assert result['max_normalized_difference'] is None
    assert result['per_state_max_normalized_difference'] is None
    assert 'difference overflow' in result['reason']
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_finite_mass_scale_product_overflow_is_json_safe(benchmark_script):
    import json

    state = np.zeros((1, 2, 4))
    result = benchmark_script.state_parity(state, state, [1e308], [1, 1e308, 1], 1)
    assert not result['passed'] and not result['valid_runtime']
    assert result['scale_vector'] is None and result['max_normalized_difference'] is None
    assert 'scale product overflow' in result['reason']
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_loader_reads_only_11_point_slices_and_converts_units(benchmark_script, monkeypatch, tmp_path):
    import csv
    import json
    import h5py

    source = tmp_path / 'synthetic_source'
    dataset = source / 'dataset'
    dataset.mkdir(parents=True)
    physics = {'mass': [2., 3.], 'incidence': [[1.], [-1.]], 'delta_star': [.1, -.2],
               'pm_star': [.4, .5], 'coupling': [2.], 'damping': [.2, .3],
               'rate': [1., 1.], 'gain': [.1, .2], 'omega_s': 100.}
    np.savez(source / 'baseline.npz', **{k: np.asarray(v) for k, v in physics.items() if k != 'omega_s'})
    (dataset / 'splits.json').write_text(json.dumps({'train': ['train'], 'validation': ['dev'], 'test': ['test']}))
    with (dataset / 'dataset_index.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['event_id', 'fault_bus', 'label', 'trajectory_end_s'])
        writer.writeheader()
        for event_id, group in [('train', 'a'), ('dev', 'b'), ('test', 'c')]:
            writer.writerow({'event_id': event_id, 'fault_bus': group, 'label': 'stable', 'trajectory_end_s': '.14'})
    real_file = h5py.File
    with real_file(dataset / 'dataset.h5', 'w') as handle:
        handle.attrs['omega_s'] = 100.
        handle['inertia'] = [200., 300.]
        handle['equilibrium/delta'] = [1.1, .8]  # Constant common offset from model delta_star.
        handle['equilibrium/Pm'] = [.4, .5]
        handle['events/train/t'] = np.arange(15) * .01
        for name, values in [('delta', [1.12, .77]), ('omega', [1.01, .98]), ('Pm', [.5, .3])]:
            data = np.tile(values, (15, 1))
            data[11:] = np.nan  # Reading any future prefix would be observable and invalid.
            handle['events/train/' + name] = data
    accessed = []

    class DatasetGuard:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, selection):
            assert selection == slice(None, 11)
            accessed.append((self.value.name, selection))
            return self.value[selection]

    class GroupGuard:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, key):
            if self.value.name == '/events':
                assert key == 'train'
                return GroupGuard(self.value[key])
            return DatasetGuard(self.value[key])

    class FileGuard:
        def __init__(self, path, mode):
            assert mode == 'r'
            self.value = real_file(path, mode)
            self.attrs = self.value.attrs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.value.close()

        def __getitem__(self, key):
            return GroupGuard(self.value[key]) if key == 'events' else self.value[key]

    monkeypatch.setattr(benchmark_script.h5py, 'File', FileGuard)
    config = {'kiki_root': str(source), 'baseline': 'baseline.npz', 'datasets': ['dataset'],
              'sample_dt': .01, 'history_ms': 100}
    histories, facts = benchmark_script.read_panel(config, physics, (['train'], ['train']), time.time() + 30)
    assert len(accessed) == 4 and histories.shape == (1, 11, 2, 3)
    np.testing.assert_allclose(histories[0, 0], [[.02, 1., .1], [-.03, -2., -.2]], atol=1e-13)
    assert facts['both_checkpoint_train_ids_exact_match_before_payload']
    assert not facts['future_train_payload_read'] and not facts['test_payload_read']
    accessed.clear()
    with pytest.raises(ValueError, match='before payload access'):
        benchmark_script.read_panel(config, physics, (['other'], ['train']), time.time() + 30)
    assert accessed == []
