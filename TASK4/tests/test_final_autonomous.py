"""Final D uses only one 100 ms initialization and real-prefix rollout losses."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from task4.dynamics import PowerGraph
from task4.encoder import HistoryEncoder
from task4.objectives import batch_objective, one_event_loss, tail_risk
from task4.synthetic import fixture


class ForbiddenAuxiliary:
    def __call__(self, *args, **kwargs):
        raise AssertionError('Final D called an auxiliary module')

    def get(self, *args, **kwargs):
        raise AssertionError('Final D queried anchors or shooting states')

    def integers(self, *args, **kwargs):
        raise AssertionError('Final D sampled a future/sliding history window')

    def add_param_group(self, *args, **kwargs):
        raise AssertionError('Final D created latent shooting parameters')


class RecordingEncoder(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.histories = []

    def forward(self, histories):
        self.histories.append(histories.detach().cpu().clone())
        return self.inner(histories)


def setup():
    events, physics = fixture(events=3, steps=19)
    # Unequal real prefix lengths force two autonomous batch groups; no padding
    # can become a target or an encoder input for the short event.
    short = events[1]
    events[1] = replace(short, t=short.t[:14], y=short.y[:14], torque=short.torque[:14])
    torch.manual_seed(781)
    model = PowerGraph(physics, [0.2, 0.3, 0.1], rank=2, hidden=8)
    encoder = RecordingEncoder(HistoryEncoder(model, hidden=8))
    cfg = {'autonomous_only': True, 'integration_dt': 0.01, 'arm': 'P',
           'anchor_weight': 0., 'latent_weight': 0., 'torque_weight': 0.,
           'teacher_field_weight': 0., 'teacher_gradient_weight': 0.,
           'complexity_weight': 2e-5, 'continuity_weight': 1.}
    return model, encoder, events, cfg


def final_loss(model, encoder, events, cfg, **kwargs):
    forbidden = ForbiddenAuxiliary()
    return batch_objective(model, encoder, forbidden, events, 0.05, cfg,
                           forbidden, forbidden, forbidden, forbidden, **kwargs)


def test_final_d_initializes_each_event_once_and_never_reads_auxiliaries():
    model, encoder, events, cfg = setup()
    loss, terms = final_loss(model, encoder, events, cfg)
    assert torch.isfinite(loss)
    assert len(encoder.histories) == 2
    observed = torch.cat(encoder.histories)
    expected = [torch.as_tensor(event.y[:11]) for event in events]
    assert len(observed) == len(events)
    for history in expected:
        assert sum(torch.equal(row, history) for row in observed) == 1
    for name in ('continuity', 'anchor', 'latent', 'torque', 'teacher_field', 'teacher_gradient'):
        assert terms[name].item() == 0
    torch.testing.assert_close(loss, terms['rollout']+cfg['complexity_weight']*terms['regularization'],
                               rtol=0, atol=0)


def test_final_d_matches_individual_autonomous_losses_and_parameter_gradients():
    model, encoder, events, cfg = setup()
    independent_model = copy.deepcopy(model)
    independent_encoder = copy.deepcopy(encoder.inner)
    loss, terms = final_loss(model, encoder, events, cfg)
    per_event = [one_event_loss(independent_model, independent_encoder, event,
                               0.05, cfg['integration_dt'], training=True)[0] for event in events]
    expected_main = tail_risk(torch.stack(per_event))
    expected_complexity = sum(parameter.square().mean() for parameter in independent_model.parameters()
                              if parameter.requires_grad)
    expected = expected_main+cfg['complexity_weight']*expected_complexity
    torch.testing.assert_close(terms['rollout'], expected_main, rtol=2e-10, atol=2e-13)
    torch.testing.assert_close(loss, expected, rtol=2e-10, atol=2e-13)
    actual_parameters = list(model.parameters())+list(encoder.parameters())
    expected_parameters = list(independent_model.parameters())+list(independent_encoder.parameters())
    actual_gradients = torch.autograd.grad(loss, actual_parameters, allow_unused=True)
    expected_gradients = torch.autograd.grad(expected, expected_parameters, allow_unused=True)
    for parameter, actual, reference in zip(actual_parameters, actual_gradients, expected_gradients):
        if actual is None:
            actual = torch.zeros_like(parameter)
        if reference is None:
            reference = torch.zeros_like(parameter)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=3e-9, atol=2e-12)
    assert sum(float(g.abs().sum()) for g in actual_gradients[-len(list(encoder.parameters())):]
               if g is not None) > 0


@pytest.mark.parametrize('weight', ['anchor_weight', 'latent_weight', 'torque_weight',
                                    'teacher_field_weight', 'teacher_gradient_weight'])
def test_final_d_rejects_active_auxiliary_weights_before_initialization(weight):
    model, encoder, events, cfg = setup()
    cfg[weight] = 0.1
    with pytest.raises(ValueError, match='zero auxiliary weights'):
        final_loss(model, encoder, events, cfg)
    assert not encoder.histories


@pytest.mark.parametrize('kwargs,match', [({'segment_seconds': 0.5}, 'segment_seconds=None'),
                                        ({'teacher': ForbiddenAuxiliary()}, 'teacher=None')])
def test_final_d_rejects_teacher_and_even_overwide_segments(kwargs, match):
    model, encoder, events, cfg = setup()
    with pytest.raises(ValueError, match=match):
        final_loss(model, encoder, events, cfg, **kwargs)
    assert not encoder.histories


def test_future_truth_changes_loss_but_cannot_change_initialization():
    model, encoder, events, cfg = setup()
    original_loss, _ = final_loss(model, encoder, events, cfg)
    original_histories = [history.clone() for history in encoder.histories]
    changed = []
    for event in events:
        y = event.y.copy()
        # Alter every physical channel after initialization, including common
        # frequency and p. These are loss targets, never new GRU inputs.
        y[11:] += np.array([0.08, 0.15, 0.04])
        changed.append(replace(event, y=y))
    encoder.histories.clear()
    changed_loss, _ = final_loss(model, encoder, changed, cfg)
    assert len(encoder.histories) == len(original_histories)
    for original, actual in zip(original_histories, encoder.histories):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert not torch.isclose(original_loss, changed_loss, rtol=1e-5, atol=1e-8)
