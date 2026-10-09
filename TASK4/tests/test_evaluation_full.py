"""Full-window evaluation regression tests using constructed ODEs only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import task4.evaluation as evaluation
from task4.data import Event
from task4.classification import numerical_pair
from task4.synthetic import fixture


class ClockFailureODE(torch.nn.Module):
    """Physical states stay constant; a hidden clock can fail after 2 s."""
    def __init__(self, raise_on_failure=False):
        super().__init__()
        self.n = 2
        self.register_buffer('mass', torch.ones(2, dtype=torch.float64))
        self.register_buffer('scales', torch.ones(3, dtype=torch.float64))
        self.initial_calls = 0
        self.histories = []
        self.field_batch_sizes = []
        self.raise_on_failure = raise_on_failure

    def initial(self, history, encoder):
        self.initial_calls += 1
        self.histories.append(history.detach().cpu().numpy().copy())
        marker = history[:, -1, 0, 2:3]
        return torch.cat((history[:, -1].flatten(1), torch.zeros_like(marker), marker), -1)

    def physical(self, state):
        return state[..., :6].reshape(*state.shape[:-1], 2, 3)

    def field(self, state, create_graph=True):
        assert create_graph is False
        assert not torch.is_grad_enabled()
        self.field_batch_sizes.append(len(state))
        bad = (state[:, 6] > 2.01) & (state[:, 7] > .5)
        if self.raise_on_failure and bool(bad.any()):
            raise FloatingPointError('constructed single-event failure')
        derivative = torch.zeros_like(state)
        derivative[:, 6] = 1.0
        # The nonfinite derivative is exclusively latent: a physical-only
        # integrator check would miss this constructed failure.
        derivative[bad, 7] = float('nan')
        return derivative


def event(identity, length=811, *, fail=False, label='stable'):
    y = np.zeros((length, 2, 3))
    if fail:
        y[:, 0, 2] = 1.0
    return Event(identity, identity, label, np.arange(length) * .01, y, None, 'constructed', {})


def allow_test_outputs(monkeypatch):
    # The production writer still refuses paths outside TASK4. Tests use
    # their isolated temporary directory without relaxing production code.
    monkeypatch.setattr(evaluation, 'project_path', lambda value: Path(value))
    monkeypatch.setattr(evaluation, 'write_json',
                        lambda path, obj: Path(path).write_text(json.dumps(obj, allow_nan=False), encoding='utf-8'))


def test_late_failure_preserves_early_scores_full_forecast_and_denominators(tmp_path, monkeypatch):
    allow_test_outputs(monkeypatch)
    events = [event('late_failure', fail=True, label='unstable'),
              event('finite_neighbor'), event('short_severe', length=51, label='unstable')]
    truth_before = [item.y.copy() for item in events]
    model = ClockFailureODE()
    physics = SimpleNamespace(mass=np.ones(2), delta_star=np.zeros(2), omega_s=2 * np.pi * 60)
    facts = {'split': 'validation', 'raw_count': 4, 'eligible_count': 3,
             'excluded': [{'id': 'latency', 'label': 'unstable', 'reason': 'latency_uncovered'}],
             'test_payload_read': False}
    result = evaluation.evaluate(model, torch.nn.Identity(), events, tmp_path / 'evidence.json',
                                 physics=physics, facts=facts)
    assert model.initial_calls == 1
    assert model.histories[0].shape == (3, 11, 2, 3)
    assert result['initializations_per_event'] == 1
    assert result['forecast_steps'] == 800
    early = next(row for row in result['rows'] if row['id'] == 'late_failure' and row['horizon'] == '1.0')
    late = next(row for row in result['rows'] if row['id'] == 'late_failure' and row['horizon'] == '5.0')
    assert early['qualified'] and early['metrics']['rmse'] == 0
    assert not late['qualified'] and late['failure'] == 'nonfinite_prediction'
    assert result['summary']['1.0']['truth_scorable_denominator'] == 2
    assert result['summary']['1.0']['numerically_qualified_count'] == 2
    assert result['summary']['1.0']['truth_unavailable_ids'] == ['short_severe']
    assert result['summary']['5.0']['truth_scorable_denominator'] == 2
    assert result['summary']['5.0']['count'] == 2
    assert result['summary']['5.0']['pooled'] is None
    assert result['summary']['5.0']['unresolved_ids'] == ['late_failure']
    assert result['summary']['1.0']['physical']['p_pu']['pooled'] == 0
    assert len(result['severe_prefix_rows']) == 2
    classification = result['classification']
    assert classification['event_count_raw'] == 4
    assert classification['eligible_event_count'] == 3
    assert classification['unstable_total'] == 3
    assert classification['unstable_eligible_denominator'] == 2
    assert classification['latency_uncovered_count'] == 1
    assert classification['group_counts']['unknown']['latency_uncovered'] == 1
    assert classification['false_window_settled'] == 1  # Short severe truth still has an 8-s prediction.
    assert classification['numerical_unresolved_count'] == 1
    assert classification['missed_true_unstable'] == 2
    assert any(size == 2 for size in model.field_batch_sizes)
    cache_path = result['cache']['batches'][0]['path']
    declared = result['cache']['batches'][0]
    assert declared['sha256'] == hashlib.sha256(Path(cache_path).read_bytes()).hexdigest()
    assert declared['metadata_sha256'] == hashlib.sha256(Path(declared['metadata_path']).read_bytes()).hexdigest()
    with np.load(cache_path, allow_pickle=False) as cache:
        assert cache['fine_physical'].shape == (3, 801, 2, 3)
        assert cache['fine_substep_states'].shape == (3, 1601, 8)
        assert np.isfinite(cache['fine_physical'][0, :101]).all()
        assert np.isnan(cache['fine_physical'][0, 500:]).all()
        assert np.isfinite(cache['fine_physical'][1]).all()
        assert np.isfinite(cache['fine_physical'][2]).all()
        assert cache['truth_002'].shape == (51, 2, 3)
        for i, expected in enumerate(truth_before):
            np.testing.assert_array_equal(cache[f'truth_{i:03d}'], expected)
            np.testing.assert_array_equal(events[i].y, expected)
    failure = result['integration_failures'][0]['fine']
    assert failure['time_decision_s'] > 2
    assert failure['stage'].endswith('_field')
    assert result['integration_failures'][1]['fine'] is None
    with pytest.raises(FileExistsError):
        evaluation.evaluate(model, torch.nn.Identity(), events, tmp_path / 'evidence.json',
                            physics=physics, facts=facts)


def test_exception_in_one_event_does_not_erase_other_events_or_earlier_states():
    model = ClockFailureODE(raise_on_failure=True)
    hist = torch.zeros((2, 11, 2, 3), dtype=torch.float64)
    hist[0, :, 0, 2] = 1
    initial = model.initial(hist, None)
    result = evaluation.forecast_batch(model, initial, steps=220, max_step=.01)
    assert result['state_finite'][0, 100]
    assert not result['state_finite'][0, -1]
    assert result['state_finite'][1].all()
    assert result['failures'][0]['reason'].startswith('FloatingPointError')
    assert result['failures'][1] is None
    assert np.isnan(result['states'][0, -1]).all()


def test_absolute_times_are_not_guessed_without_metadata():
    item = event('missing_metadata')
    timing = evaluation._timing(item, 120)
    assert timing['predicted_los_time_record_s'] == pytest.approx(1.2)
    assert timing['predicted_los_time_decision_s'] == pytest.approx(1.1)
    assert timing['predicted_los_time_absolute_s'] is None
    item.metadata = {'clearing_time_s': '2.35', 'record_start_delay_s': '0.005', 'event_time_s': '1.18'}
    timing = evaluation._timing(item, 120)
    assert timing['predicted_los_time_absolute_s'] == pytest.approx(3.555)
    assert timing['trigger_time_error_s'] == pytest.approx(.02)


def test_common_and_power_step_differences_do_not_redefine_los_gate():
    first, second = np.zeros((801, 2, 3)), np.zeros((801, 2, 3))
    first[:, :, 1] = 0.01
    first[:, :, 2] = 0.02
    difference = evaluation.numerical_difference(first, second, np.ones(2), .1)
    assert difference['common'] > .001
    assert difference['p'] == pytest.approx(.2)
    pair, passed = numerical_pair(first, second, np.zeros((11, 2, 3)), np.ones(2), np.zeros(2), 2 * np.pi * 60)
    assert passed and pair['normalized_coi_discrepancy'] == pytest.approx(0)


def test_fixed_m3_matches_direct_physical_equations_and_has_no_encoder():
    events, physics = fixture(n=3, events=1, steps=12)
    model = evaluation.FrozenM3(physics, [1, 1, 1])
    history = torch.as_tensor(events[0].y[:11][None], dtype=torch.float64)
    state = model.initial(history)
    y = events[0].y[10]
    np.testing.assert_allclose(model.physical(state).numpy()[0], y, atol=1e-14)
    actual = model.field(state).numpy()[0]
    edge_phase = physics.delta_star @ physics.incidence
    angle_difference = y[:, 0] @ physics.incidence
    electric = ((np.sin(edge_phase + angle_difference) - np.sin(edge_phase)) * physics.coupling) @ physics.incidence.T
    expected = np.concatenate((y[:, 1], y[:, 2] - electric - physics.damping * y[:, 1],
                               -physics.rate * y[:, 2] - physics.gain * y[:, 1]))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert state.shape[-1] == 3 * physics.n


def test_declared_population_mismatch_fails_before_any_initialization(tmp_path, monkeypatch):
    allow_test_outputs(monkeypatch)
    model = ClockFailureODE()
    with pytest.raises(ValueError, match='denominators'):
        evaluation.evaluate(model, torch.nn.Identity(), [event('one')], tmp_path / 'bad.json',
                            facts={'raw_count': 9, 'eligible_count': 1, 'excluded': []})
    assert model.initial_calls == 0
