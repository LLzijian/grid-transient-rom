"""Standalone NumPy RK4. Only the first 100 ms history initializes the ODE."""
import argparse
from pathlib import Path
import numpy as np
from runtime import SymbolicPowerGraph

def simulate(model, history, seconds=5., dt=.005):
    if seconds <= 0 or not 0 < dt <= .01:
        raise ValueError("positive duration and 0 < dt <= .01 required")
    h = np.asarray(history, dtype=np.float64)
    if h.ndim == 3: h = h[None]
    x = model.initial(h)
    count = int(np.ceil(seconds/dt)); step = seconds/count
    states = [x.copy()]
    for _ in range(count):
        a = model.field(x)
        b = model.field(x+step*a/2)
        c = model.field(x+step*b/2)
        d = model.field(x+step*c)
        x = x+step*(a+2*b+2*c+d)/6
        if not all(np.isfinite(v).all() for v in (a,b,c,d,x)):
            raise FloatingPointError("Nonfinite integration; no fabricated tail returned")
        states.append(x.copy())
    return .1+np.arange(count+1)*step, np.stack(states, axis=1)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(Path(__file__).with_name("model.json")))
    p.add_argument("--history", required=True, help="NPY [11,N,3] or [B,11,N,3], physical delta/omega/p")
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=5.)
    p.add_argument("--dt", type=float, default=.005)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    model = SymbolicPowerGraph.from_json(a.model)
    times, states = simulate(model, np.load(a.history, allow_pickle=False), a.seconds, a.dt)
    np.savez_compressed(output, time=times, state=states, physical=model.physical(states))
