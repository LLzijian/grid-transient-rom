"""No project data: deterministic monitor must not average minibatch tails."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_train_structured_rom import screen


def test_monitor_uses_global_top35_and_every_event(monkeypatch):
    class Dummy(torch.nn.Module):
        pass
    events = [{'id': str(i), 'x': np.zeros((12, 30)), 'error': float(i)}
              for i in range(346)]
    def errors(model, samples, histories, horizon):
        assert all(endpoint == 10 for _, endpoint in samples)
        return torch.tensor([e['error'] * horizon for e, _ in samples], dtype=torch.float64)
    monkeypatch.setattr(screen, '_event_rollout_errors', errors)
    model = Dummy().train()
    result = screen._train_monitor(model, events)
    expected = .8 * np.arange(346).mean() + .2 * np.arange(346)[-35:].mean()
    assert result['average_risk'] == pytest.approx(expected * 8 / 3)
    assert len(result['per_event']) == 346 * 3
    assert all(row['n_events'] == 346 for row in result['horizons'].values())
    assert model.training
