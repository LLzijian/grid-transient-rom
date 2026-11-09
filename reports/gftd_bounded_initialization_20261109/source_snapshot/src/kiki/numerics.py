"""Shared NumPy numerical kernels for new experiments.

The RK4 expression follows audit_latent_order_rollout.rk4. Frozen historical
runners retain their original source identity and do not import this module.
"""
from collections.abc import Callable

import numpy as np
from scipy.integrate import DOP853


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


def dop853_step(rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray, step: float) -> np.ndarray:
    """Advance a float64 state by the fixed DOP853 eighth-order formula.

    A completed step uses 12 autonomous field evaluations. State may have
    leading batch dimensions, and a negative step integrates backward.
    """
    if np.iscomplexobj(state):
        raise ValueError("DOP853 state must be real")
    if np.ndim(step) != 0 or np.iscomplexobj(step):
        raise ValueError("DOP853 step must be a real scalar")
    y = np.asarray(state, dtype=np.float64)
    step = float(step)
    if not np.isfinite(y).all() or not np.isfinite(step) or step == 0:
        raise ValueError("finite state and finite nonzero DOP853 step required")

    stages = np.empty((DOP853.n_stages, y.size), dtype=np.float64)
    stage_state = y
    for index, coefficients in enumerate(DOP853.A):
        if index:
            increment = np.dot(stages[:index].T, coefficients[:index]).reshape(y.shape)
            stage_state = y + step * increment
            if not np.isfinite(stage_state).all():
                raise FloatingPointError("nonfinite DOP853 stage state")
        derivative = rhs(stage_state)
        if np.iscomplexobj(derivative):
            raise ValueError("DOP853 derivative must be real")
        derivative = np.asarray(derivative, dtype=np.float64)
        if derivative.shape != y.shape:
            raise ValueError("DOP853 derivative must preserve state shape")
        if not np.isfinite(derivative).all():
            raise FloatingPointError("nonfinite DOP853 derivative")
        stages[index] = derivative.ravel()

    result = y + step * np.dot(stages.T, DOP853.B).reshape(y.shape)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite DOP853 state")
    return result
