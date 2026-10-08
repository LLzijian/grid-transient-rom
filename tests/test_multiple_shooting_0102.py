import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))
from multiple_shooting_0102 import MultipleShootingProblem, make_objective, train_state_scales
from train_local_memory_0098 import histories_for
from train_structured_rom import _event_rollout_errors


class DummyROM(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        self.mass = torch.ones(10, dtype=torch.float64)
        self.p_scale = torch.ones(10, dtype=torch.float64)
        self.encode_calls = 0
        self.rk4_calls = 0

    def encode(self, history):
        self.encode_calls += 1
        return self.theta * history[:, -1, :10]

    def rk4(self, state, dt):
        self.rk4_calls += 1
        return state + float(dt) * self.theta * torch.ones_like(state)


def _events():
    t = torch.arange(31, dtype=torch.float64)[:, None]
    x = t.expand(31, 30) * .002
    return [{"id": "train-a", "x": x}]


def _events_length(length, event_id="train-long"):
    t = torch.arange(length, dtype=torch.float64)[:, None]
    x = t.expand(length, 30) * .002
    return [{"id": event_id, "label": "unstable", "x": x}]


def _align_aux_to_continuous_rollout(model, problem):
    """Make segmented dummy dynamics identical so weighting can be checked."""
    event = problem.events[0]
    x = event["x"]
    with torch.no_grad():
        state = torch.cat((x[10], model.encode(x[0:11].unsqueeze(0))[0]))
        for record in problem.segment_records:
            if record.auxiliary_index is not None:
                target = record.start
                # state currently sits at the previous record endpoint.
                previous = problem.segment_records[problem.segment_records.index(record) - 1]
                for _ in range(previous.end, target):
                    state = model.rk4(state, problem.dt)
                problem.q_aux[record.auxiliary_index].copy_(state[30:])
            for _ in range(record.start, record.end):
                state = model.rk4(state, problem.dt)


def test_aux_q_source_initialization_and_first_segment_only_uses_gru():
    model = DummyROM()
    problem = MultipleShootingProblem(model, _events(), {"train-a": [10, 15, 20]})
    assert problem.auxiliary_count == 1
    assert problem.scales.shape == (40,)
    assert torch.all(problem.scales[30:] >= 1e-3)
    model.encode_calls = model.rk4_calls = 0
    problem.loss(model)
    assert model.encode_calls == 1
    assert model.rk4_calls == 10


def test_full40_continuity_backpropagates_to_gru_and_combines_autonomous():
    model = DummyROM()
    problem = MultipleShootingProblem(model, _events(), {"train-a": [10, 15, 20]})

    def nonlong(m, samples, history, horizon, force_scale):
        del samples, history, horizon, force_scale
        value = m.theta.square()
        return value, {"local": value}

    objective = make_objective(problem, nonlong_objective=nonlong, continuity_weight=10.0)
    samples = [(_events()[0], 10)]
    loss, terms = objective(model, samples, None, 1.0, 1.0)
    grad = torch.autograd.grad(loss, model.theta)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad) and grad.abs() > 0
    assert "local" in terms and "shooting_continuity" in terms
    assert terms["shooting_continuity"].ndim == 0
    residuals = terms["shooting_continuity_residuals"]
    assert residuals.shape[1] == 40
    assert torch.allclose(terms["shooting_continuity"], residuals.square().sum(-1).mean())


def test_scales_are_train_only_and_boundaries_cannot_invent_tails():
    events = _events()
    scales = train_state_scales(events)
    altered = [{"id": "ignored-dev", "x": events[0]["x"] * 100.0}]
    assert torch.equal(scales, train_state_scales(events))
    assert not torch.equal(scales, train_state_scales(altered))
    with pytest.raises(ValueError):
        MultipleShootingProblem(DummyROM(), events, {"train-a": [10, 20, 31]})


def test_one_segment_matches_original_point_error_and_duplicate_sampling_is_preserved():
    model = DummyROM()
    events = _events()
    problem = MultipleShootingProblem(model, events, {"train-a": [10, 20]})
    samples = [(events[0], 10), (events[0], 10)]
    histories = histories_for(samples)
    reference = _event_rollout_errors(model, samples[:1], histories[:1], .1)[0]
    _, terms = problem.loss(model, samples=samples, horizon=.1)
    assert terms["shooting_event_values"].shape == (2,)
    assert torch.allclose(terms["shooting_event_values"], reference.expand(2), atol=1e-12, rtol=0)

    # Splitting the same ten real future points at an unequal boundary must
    # retain the pointwise mean, rather than averaging the two segment means.
    unequal = MultipleShootingProblem(model, events, {"train-a": [10, 11, 20]})
    _align_aux_to_continuous_rollout(model, unequal)
    _, unequal_terms = unequal.loss(model, [(events[0], 10)], horizon=.1)
    assert torch.allclose(unequal_terms["shooting_event_values"][0], reference, atol=1e-12, rtol=0)


def test_horizon_cuts_inside_long_segment_and_respects_genuine_short_tail():
    model = DummyROM()
    long_events = _events_length(601)
    long_problem = MultipleShootingProblem(model, long_events, {"train-long": [10, 210, 410, 600]})
    _align_aux_to_continuous_rollout(model, long_problem)
    long_sample = [(long_events[0], 10)]
    long_history = histories_for(long_sample)
    # One second ends inside the first nominal two-second segment.
    _, one_terms = long_problem.loss(model, long_sample, horizon=1.0)
    one_reference = _event_rollout_errors(model, long_sample, long_history, 1.0)[0]
    assert torch.allclose(one_terms["shooting_event_values"][0], one_reference, atol=1e-12, rtol=0)
    # Five seconds reaches the real observed cutoff without inventing a tail.
    _, five_terms = long_problem.loss(model, long_sample, horizon=5.0)
    five_reference = _event_rollout_errors(model, long_sample, long_history, 5.0)[0]
    assert torch.allclose(five_terms["shooting_event_values"][0], five_reference, atol=1e-12, rtol=0)

    short_events = _events_length(25, "train-short")
    short_problem = MultipleShootingProblem(model, short_events, {"train-short": [10, 24]})
    short_sample = [(short_events[0], 10)]
    short_history = histories_for(short_sample)
    _, short_terms = short_problem.loss(model, short_sample, horizon=5.0)
    short_reference = _event_rollout_errors(model, short_sample, short_history, 5.0)[0]
    assert torch.allclose(short_terms["shooting_event_values"][0], short_reference, atol=1e-12, rtol=0)
    assert short_terms["shooting_continuity_residuals"].shape == (0, 40)
