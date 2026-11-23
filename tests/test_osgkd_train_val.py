from types import SimpleNamespace

import numpy as np
import torch

from kiki.evaluation import osgkd_validation as validation


def test_val_keeps_missing_history_in_original_denominator(monkeypatch):
    constants = dict(mass=np.ones(10), omega_scale=np.ones(10), s_coi=1.,
                     omega_s=2*np.pi*60, S_m=np.ones(10), Se=np.ones(10))
    targets = {key: np.zeros((801, 10)) for key in ("theta", "omega", "tau_m", "tau_e")}
    event = dict(id="synthetic", group="synthetic", label="stable",
                 history={key: np.zeros((11, 10)) for key in ("theta", "omega")},
                 t=np.arange(801)*.01, targets=targets,
                 masks={key: np.ones_like(value, dtype=bool) for key, value in targets.items()},
                 metadata=dict(true_event_time_record_s=None, record_start_delay_s=None))
    missing = dict(id="short", group="synthetic", label="unstable", history=None,
                   missing_reason="fewer_than_11_original_history_points")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mass = torch.ones(10, dtype=torch.float64)

        def initialize(self, theta, omega):
            assert not torch.is_grad_enabled()
            assert len(theta) == 1
            return theta[:, 0]

        def ports(self, path):
            return SimpleNamespace(tau_m=path[..., :10], tau_e=path[..., :10])

    calls = []

    def integrate(model, initial, steps, *, internal_step, work):
        calls.append(internal_step)
        return torch.zeros((len(initial), steps+1, 50), dtype=torch.float64)

    monkeypatch.setattr(validation, "integrate", integrate)
    model = Model()
    result = validation.evaluate_val(model, [event, missing], constants)
    metric = result["summary"]["splits"]["val"]["all_8s"]
    assert metric["denominator"] == 2 and metric["covered"] == 1
    assert metric["official"] is None
    assert metric["covered_only_diagnostic"]["motion_risk_v1"] == 0.
    assert result["summary"]["coverage"]["missing_history_ids"] == ["short"]
    assert result["summary"]["classification"]["val"]["8s"]["uncovered_ids"] == ["short"]
    assert calls == [.005, .0025] and model.training
