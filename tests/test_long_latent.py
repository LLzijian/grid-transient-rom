import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).parents[1] / 'experiments'))
from train_long_latent import campaign_loss, freeze_xi_only, sample_batch


class LinearFixture(nn.Module):
    """Small autonomous ROM whose physical state advances one unit per second."""
    def __init__(self):
        super().__init__()
        self.xi = nn.Parameter(torch.ones(4, 1, dtype=torch.float64))
        self.other = nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.register_buffer('support', torch.ones_like(self.xi))
        self.register_buffer('metric_scale', torch.ones(2, dtype=torch.float64))
        self.register_buffer('force_scale', torch.tensor(1., dtype=torch.float64))
        self.register_buffer('load_scale', torch.tensor(1., dtype=torch.float64))
        self.register_buffer('z_scale', torch.ones(20, dtype=torch.float64))
        self.initial_encode_calls = 0

    def encode(self, history):
        # Only the separate long initial batch starts at t=0 in this fixture.
        self.initial_encode_calls += int(torch.all(history[:, 0, 0] == 0).item())
        return torch.zeros((len(history), 1), dtype=torch.float64)

    def coi(self, x):
        return x-x.mean(dim=-1, keepdim=True)

    def rk4(self, state, dt):
        velocity = torch.zeros_like(state)
        velocity[:, 0] = 1.
        return state + dt * velocity + 0 * self.xi.sum()

    def algebraic_coordinate(self, state):
        return torch.zeros((len(state), 1), dtype=torch.float64)

    def memory(self, q):
        return torch.zeros((len(q), 10), dtype=torch.float64)

    def electrical_output(self, state, q):
        return torch.zeros((len(state), 20), dtype=torch.float64), torch.zeros((len(state), 1), dtype=torch.float64)


def _event(name, length=1000):
    x = np.zeros((length, 30))
    x[:, 0] = np.arange(length) * .01
    return {'id': name, 'x': x, 'force': np.zeros((length, 10)),
            'z': np.zeros((length, 20)), 'load': np.zeros(length)}


def test_long_forecast_starts_from_initial_history_not_local_endpoint():
    model = LinearFixture()
    first, second = _event('first'), _event('second')
    samples = [(first, 300), (second, 700)]
    histories = torch.as_tensor(np.stack([e['x'][i-10:i+1] for e, i in samples]), dtype=torch.float64)
    total, terms = campaign_loss(model, samples, histories, .08)
    assert float(terms['long_mean'].detach()) < 1e-25
    assert model.initial_encode_calls == 1
    total.backward()
    assert torch.isfinite(model.xi.grad).all()


def test_short_prefixes_have_equal_event_weight_and_odd_final_observation():
    model = LinearFixture()
    short, long = _event('short', 14), _event('long', 1000)
    # The fixture predicts the unmodified unit-slope path.  These nonzero
    # errors make the event denominator and odd true-final observation exact.
    short['x'][11:, 0] += 1.
    short['x'][13, 0] += 2.
    long['x'][11:, 0] += 2.
    samples = [(short, 10), (long, 10)]
    histories = torch.as_tensor(np.stack([e['x'][i-10:i+1] for e, i in samples]), dtype=torch.float64)
    _, terms = campaign_loss(model, samples, histories, .2)

    assert np.isclose(float(terms['long_mean'].detach()), .1725)
    assert np.isclose(float(terms['long_tail'].detach()), .18)
    assert np.isclose(float(terms['risk_long'].detach()), .174)


def test_sampler_reaches_full_train_beyond_old_7_and_8_second_caps():
    long = _event('long', 1000)
    _, sampled_histories = sample_batch([long], np.random.default_rng(60), count=4096)
    endpoints = (sampled_histories[:, -1, 0].numpy() / .01).astype(int)
    assert endpoints.min() >= 10 and endpoints.max() <= 998
    assert np.any(endpoints > 700) and np.any(endpoints > 810)


def test_xi_only_freeze_keeps_other_weights_and_buffers_bitwise_equal():
    model = LinearFixture()
    before = freeze_xi_only(model)
    assert model.xi.requires_grad and not model.other.requires_grad
    optimizer = torch.optim.Adam([model.xi], lr=.0003)
    optimizer.zero_grad()
    (model.xi.square().mean()).backward()
    optimizer.step()
    for name, value in model.state_dict().items():
        if name != 'xi':
            assert torch.equal(value, before[name]), name
