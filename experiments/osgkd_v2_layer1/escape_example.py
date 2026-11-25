"""Historical two-machine GFTD storage-family counterexample, not IEEE39.

Run: ../.venv/Scripts/python.exe experiments/osgkd_v2_layer1/escape_example.py
No model, data store, or training code is imported.
"""
from __future__ import annotations

import json
import math
import numpy as np

A = math.pi/2


def unbounded_rhs(x):
    return A*np.exp(x)-x


def bounded_rhs(x):
    h = np.tanh(x)
    return A*np.exp(h)*(1-h*h)-x


def threshold_time(threshold, step):
    """Composite Simpson integral of inverse velocity, without trajectory blowup."""
    intervals = int(math.ceil(threshold/step))
    intervals += intervals % 2
    x = np.linspace(0., threshold, intervals+1)
    f = 1/unbounded_rhs(x)
    return float(threshold/intervals/3*(f[0]+f[-1]+4*f[1:-1:2].sum()+2*f[2:-1:2].sum()))


def bounded_trajectory(end, step):
    count = math.ceil(end/step)
    dt = end/count
    x = 0.
    max_violation = 0.
    for index in range(count):
        k1 = bounded_rhs(x)
        k2 = bounded_rhs(x+dt*k1/2)
        k3 = bounded_rhs(x+dt*k2/2)
        k4 = bounded_rhs(x+dt*k3)
        x += dt*(k1+2*k2+2*k3+k4)/6
        t = (index+1)*dt
        bound = A*math.e*(-math.expm1(-t))
        max_violation = max(max_violation, x-bound, -x)
    return dict(end=end, step=dt, final=x, analytic_upper=A*math.e*(1-math.exp(-end)),
                maximum_bound_violation=max_violation)


def run_checks():
    thresholds = []
    for r in (4., 8., 12., 16.):
        coarse, fine = (threshold_time(r, h) for h in (.0025, .00125))
        tail = math.exp(-r)/(A-1)
        thresholds.append(dict(threshold=r, coarse_time=coarse, fine_time=fine,
                               mesh_gap=abs(coarse-fine), analytic_tail_upper=tail,
                               upper_time_estimate=fine+tail))
    upper = thresholds[-1]["upper_time_estimate"]
    # Also pass the coarser analytic escape upper bound 1/(pi/2-1).
    end = max(2*upper, 1/(A-1)+.5)
    bounded = [bounded_trajectory(end, h) for h in (.0025, .00125)]
    gap = abs(bounded[0]["final"]-bounded[1]["final"])
    assert all(r["mesh_gap"] < 1e-9 for r in thresholds)
    assert all(r["maximum_bound_violation"] < 1e-10 for r in bounded)
    assert gap < 1e-8
    assert all(thresholds[i+1]["fine_time"] > thresholds[i]["fine_time"]
               for i in range(len(thresholds)-1))
    return dict(scope="historical GFTD two-machine storage-family invariant subsystem; not IEEE39",
                equation="x_dot=(pi/2)K_prime(x)-x; x(0)=0",
                unbounded_kernel="exp(x)", bounded_kernel="exp(tanh(x))",
                inverse_integral=thresholds, analytic_escape_upper=1/(A-1),
                quadrature_note="mesh convergence is numerical, not interval-certified quadrature",
                bounded=bounded, bounded_mesh_gap=gap, status="passed")


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
