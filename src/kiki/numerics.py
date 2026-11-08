"""Shared NumPy numerical kernels for new experiments.

The RK4 expression follows audit_latent_order_rollout.rk4. Frozen historical
runners retain their original source identity and do not import this module.
"""
from collections.abc import Callable

import numpy as np


def rk4_step(rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray, step: float) -> np.ndarray:
    """Advance a finite float64 state with four autonomous field evaluations.

    State may contain leading batch dimensions. The caller defines physical
    coordinates, projection, time step and any numerical-qualification protocol.
    A negative step supports backward integration with the same expression.
    """
    if np.iscomplexobj(state):
        raise ValueError("RK4 state must be real")
    y = np.asarray(state, dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(step) or step == 0:
        raise ValueError("finite state and finite nonzero RK4 step required")

    def field(value):
        result = rhs(value)
        if np.iscomplexobj(result):
            raise ValueError("RK4 derivative must be real")
        result = np.asarray(result, dtype=np.float64)
        if result.shape != y.shape:
            raise ValueError("RK4 derivative must preserve state shape")
        if not np.isfinite(result).all():
            raise FloatingPointError("nonfinite RK4 derivative")
        return result

    k1 = field(y)
    k2 = field(y + step * k1 / 2)
    k3 = field(y + step * k2 / 2)
    k4 = field(y + step * k3)
    result = y + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite RK4 state")
    return result
