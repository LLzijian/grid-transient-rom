"""First-order equilibrium-preserving mechanical closure, no hidden state."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear

from kiki.models.private_rom import SwingROM


@dataclass
class MechanicalROM(SwingROM):
    rate: np.ndarray
    gain: np.ndarray

    def field(self, state, pm_unused=None):
        n = len(self.mass)
        pm = state[..., 2*n:3*n]
        swing = super().field(state[..., :2*n], pm)
        derivative = -self.rate*(pm-self.pm_star)-self.gain*state[..., n:2*n]
        return np.concatenate((swing, derivative), axis=-1)

    def energy(self, state):
        """Local storage modulo angle gauge; not a global positive/basin claim."""
        if np.any(self.gain <= 0):
            raise ValueError("positive governor feedback required for this storage")
        n = len(self.mass)
        delta, omega, p = state[..., :n], state[..., n:2*n], state[..., 2*n:3*n]-self.pm_star
        shift, equilibrium = delta@self.incidence, self.delta_star@self.incidence
        potential = np.sum(self.coupling*(np.cos(equilibrium)-np.cos(equilibrium+shift)-np.sin(equilibrium)*shift), axis=-1)
        return .5*np.sum(self.mass*omega**2+p**2/self.gain, axis=-1)+potential

    def energy_gradient(self, state):
        if np.any(self.gain <= 0):
            raise ValueError("positive governor feedback required for this storage")
        n = len(self.mass)
        eta = (state[..., :n]+self.delta_star)@self.incidence
        force = ((np.sin(eta)-np.sin(self.delta_star@self.incidence))*self.coupling)@self.incidence.T
        return np.concatenate((force, self.mass*state[..., n:2*n], (state[..., 2*n:3*n]-self.pm_star)/self.gain), axis=-1)

    def dissipation(self, state):
        if np.any(self.gain <= 0):
            raise ValueError("positive governor feedback required for this storage")
        n = len(self.mass)
        return -np.sum(self.damping*state[..., n:2*n]**2+self.rate/self.gain*(state[..., 2*n:3*n]-self.pm_star)**2, axis=-1)


def fit_mechanical(events, backbone, ridge=1e-6):
    designs, responses = [], []
    for event in events:
        t = event["t"]
        p = event["Pm_raw"]-event["Pm_star"]
        integrals = cumulative_trapezoid(np.stack((p, event["omega_sync"]), axis=-1), t, axis=0, initial=0)
        windows = []
        for width in (1, 5, 20):
            starts = np.arange(0, len(t)-width, width)
            if len(starts) > 40:
                starts = starts[np.linspace(0, len(starts)-1, 40, dtype=int)]
            windows.extend((int(a), int(a+width)) for a in starts)
        weight = 1/np.sqrt(len(events)*len(windows))
        for a, b in windows:
            designs.append(-weight*(integrals[b]-integrals[a]))
            responses.append(weight*(p[b]-p[a]))
    design, response = np.array(designs), np.array(responses)
    parameters = []
    for machine in range(len(backbone.mass)):
        x, y = design[:, machine], response[:, machine]
        scale = np.maximum(np.linalg.norm(x, axis=0), 1e-10)
        fit = lsq_linear(np.vstack((x/scale, np.sqrt(ridge)*np.eye(2))),
                         np.r_[y, 0., 0.], bounds=(np.array([.01, 0])*scale, np.array([50., np.inf])*scale),
                         tol=1e-12, max_iter=1000)
        if not fit.success:
            raise RuntimeError(fit.message)
        parameters.append(fit.x/scale)
    parameters = np.array(parameters)
    return MechanicalROM(**vars(backbone), rate=parameters[:, 0], gain=parameters[:, 1])


def forced_pm(model, event):
    """Diagnostic only: true future speed forcing, never a deployable rollout."""
    pm = event["Pm_raw"][0].copy()
    output = [pm.copy()]
    for i, dt in enumerate(np.diff(event["t"])):
        # Exact integration of a stable linear ODE with linearly interpolated speed.
        a = model.rate
        decay = np.exp(-a*dt)
        integral0 = -np.expm1(-a*dt)/a
        integral1 = dt/a-integral0/a
        slope = (event["omega_sync"][i+1]-event["omega_sync"][i])/dt
        pm = model.pm_star + decay*(pm-model.pm_star)-model.gain*(integral0*event["omega_sync"][i]+integral1*slope)
        output.append(pm.copy())
    return np.array(output)
