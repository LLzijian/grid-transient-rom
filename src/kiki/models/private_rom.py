"""Small explicit swing baseline for the private restored-system experiment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import nnls


@dataclass
class SwingROM:
    mass: np.ndarray
    incidence: np.ndarray
    delta_star: np.ndarray
    pm_star: np.ndarray
    coupling: np.ndarray
    damping: np.ndarray

    def field(self, state, pm):
        n = len(self.mass)
        delta, omega = state[..., :n], state[..., n:2*n]
        eta = (delta+self.delta_star) @ self.incidence
        electrical = ((np.sin(eta)-np.sin(self.delta_star@self.incidence))*self.coupling)@self.incidence.T
        acceleration = (pm-self.pm_star-electrical-self.damping*omega)/self.mass
        return np.concatenate((omega,acceleration),axis=-1)

    def rollout(self, initial, pm_initial, time, max_step=.005):
        state = np.asarray(initial,dtype=float).copy()
        output = [state.copy()]
        for interval in np.diff(time):
            if interval <= 0:
                raise ValueError("time must increase")
            steps=max(1,int(np.ceil(interval/max_step-1e-9)))
            dt=interval/steps
            for _ in range(steps):
                a=self.field(state,pm_initial)
                b=self.field(state+.5*dt*a,pm_initial)
                c=self.field(state+.5*dt*b,pm_initial)
                d=self.field(state+dt*c,pm_initial)
                state=state+dt*(a+2*b+2*c+d)/6
                if not np.isfinite(state).all():
                    raise FloatingPointError("nonfinite explicit ROM rollout")
            output.append(state.copy())
        return np.stack(output,axis=-2)

    def save(self,path):
        np.savez_compressed(path,**vars(self))

    @classmethod
    def load(cls,path):
        with np.load(path) as data:
            return cls(**{key:data[key] for key in cls.__dataclass_fields__})


def fit_swing(events, *, ridge=1e-6):
    """One event-balanced nonnegative weak fit, held decision-time Pm only."""
    first=events[0]
    mass,incidence,eq,pmstar=(first[k] for k in ("mass","incidence","equilibrium_delta","Pm_star"))
    n,edges=incidence.shape
    designs,responses=[],[]
    for event in events:
        t,d,w=event["t"],event["delta_sync"],event["omega_sync"]
        sine=np.sin((d+eq)@incidence)-np.sin(eq@incidence)
        sint=cumulative_trapezoid(sine,t,axis=0,initial=0)
        wint=cumulative_trapezoid(w,t,axis=0,initial=0)
        windows=[]
        for width in (1,5,20):
            starts=np.arange(0,len(t)-width,width)
            if len(starts)>40:
                starts=starts[np.linspace(0,len(starts)-1,40,dtype=int)]
            windows.extend((int(a),int(a+width)) for a in starts)
        if not windows:
            continue
        weight=1/np.sqrt(len(events)*len(windows)*n)
        for a,b in windows:
            design=np.zeros((n,edges+n))
            design[:,:edges]=incidence*(sint[b]-sint[a])[None,:]
            design[:,edges:]=np.diag(wint[b]-wint[a])
            rhs=(t[b]-t[a])*(event["Pm_raw"][0]-pmstar)-mass*(w[b]-w[a])
            designs.append(weight*design); responses.append(weight*rhs)
    design,response=np.vstack(designs),np.concatenate(responses)
    # Whiten columns before the one fixed regularization; no validation fitting.
    scale=np.maximum(np.sqrt(np.sum(design**2,axis=0)),1e-10)
    normalized=design/scale
    parameter,_=nnls(np.vstack((normalized,np.sqrt(ridge)*np.eye(edges+n))),
                     np.concatenate((response,np.zeros(edges+n))),maxiter=5000)
    parameter=parameter/scale
    model=SwingROM(mass,incidence,eq,pmstar,parameter[:edges],parameter[edges:])
    return model,{"weighted_force_integral_rmse":float(np.sqrt(np.mean((design@parameter-response)**2))),
                  "n_scalar_equations":len(response),"nonzero_parameters":int(np.count_nonzero(parameter))}
