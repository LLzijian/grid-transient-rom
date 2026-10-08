"""Numerical-grid adapter for the preregistered 0100 paired training check.

The wrapped LocalMemoryROM remains the sole model.  The adapter changes only the
physical 10 ms transition used by the frozen 0098 objective: one raw .01 RK4
step for the coarse arm, or two raw .005 RK4 steps for the fine arm.  The
objective's targets, sampling, tails, local windows and latent interval remain
10 ms quantities.
"""
from __future__ import annotations

import math
from typing import Any

from train_local_memory_0098 import objective as _objective_0098


PHYSICAL_DT = 0.01
GRID_DTS = (0.01, 0.005)


def _validate_grid_dt(dt: float) -> float:
    value = float(dt)
    if not any(math.isclose(value, candidate, rel_tol=0.0, abs_tol=1e-12) for candidate in GRID_DTS):
        raise ValueError("0100 grid dt must be exactly .01 or .005 seconds")
    return 0.01 if math.isclose(value, 0.01, rel_tol=0.0, abs_tol=1e-12) else 0.005


class IntegrationView:
    """Read-through view whose only changed method is the physical RK4 step.

    This is intentionally not an ``nn.Module`` wrapper: parameters, buffers,
    state dictionaries and methods are forwarded to the original model, so the
    caller's optimizer and parameter identities remain those of the raw model.
    """

    def __init__(self, raw_model: Any, grid_dt: float = PHYSICAL_DT):
        if isinstance(raw_model, IntegrationView):
            raw_model = raw_model.raw_model
        object.__setattr__(self, "raw_model", raw_model)
        object.__setattr__(self, "grid_dt", _validate_grid_dt(grid_dt))

    def __getattr__(self, name: str) -> Any:
        # Called only after normal attributes (raw_model/grid_dt/rk4) fail.
        return getattr(object.__getattribute__(self, "raw_model"), name)

    def __dir__(self) -> list[str]:
        return sorted(set(object.__dir__(self)) | set(dir(object.__getattribute__(self, "raw_model"))))

    def rk4(self, state, physical_dt: float = PHYSICAL_DT):
        """Advance one physical 10 ms interval on the selected raw grid."""
        physical = float(physical_dt)
        if not math.isclose(physical, PHYSICAL_DT, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("0100 IntegrationView.rk4 accepts one physical .01-second interval")
        raw = object.__getattribute__(self, "raw_model")
        grid_dt = object.__getattribute__(self, "grid_dt")
        substeps = 1 if math.isclose(grid_dt, 0.01, rel_tol=0.0, abs_tol=1e-12) else 2
        advanced = state
        for _ in range(substeps):
            advanced = raw.rk4(advanced, grid_dt)
        return advanced


def objective_at_grid(model: Any, samples, history, horizon: float, force_scale: float, dt: float):
    """Run the unchanged 0098 objective through the selected integration view."""
    grid_dt = _validate_grid_dt(dt)
    view = IntegrationView(model, grid_dt)
    return _objective_0098(view, samples, history, horizon, force_scale)


__all__ = ["GRID_DTS", "PHYSICAL_DT", "IntegrationView", "objective_at_grid"]
