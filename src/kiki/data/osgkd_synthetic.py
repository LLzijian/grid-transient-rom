"""Independent NumPy oscillator truth; scoring data never enters motion_packet.

Fixed zero-angle reference, dimensionless scales, RK4 at .0025 s (qualification
reference .00125 s), observations every .01 s. No learned model is imported.
"""
from __future__ import annotations

import numpy as np

DT = .01


def canonical_edges(edges):
    """Unordered edges under known machine identities, with no latent matching."""
    result = set()
    for i, j in edges:
        if i == j:
            raise ValueError("self edges are invalid")
        result.add(tuple(sorted((int(i), int(j)))))
    return result


def support_metrics(predicted, truth):
    predicted, truth = canonical_edges(predicted), canonical_edges(truth)
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else float(not truth)
    recall = tp / len(truth) if truth else 1.
    return dict(precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                true_positive=tp, predicted_count=len(predicted), truth_count=len(truth))


def recovery_metrics(predicted_edges, predicted_modulated, truth):
    """Report c and beta support separately; modulation accuracy on candidates."""
    pred_mod = canonical_edges(predicted_modulated)
    candidates = canonical_edges(truth["candidate_edges"])
    pred_edges = canonical_edges(predicted_edges)
    if not pred_mod <= pred_edges or not pred_edges <= candidates:
        raise ValueError("predicted supports must be nested inside candidates")
    true_mod = canonical_edges(truth["edges"][truth["beta"] != 0])
    return dict(edges=support_metrics(pred_edges, truth["edges"]),
                beta=support_metrics(pred_mod, true_mod),
                modulation_accuracy=1-len(pred_mod ^ true_mod)/len(candidates))


def make_system(n=8, seed=0, condition="open"):
    """Connected random relabelled ring plus n//2 non-ring edges (1.5n edges)."""
    if n not in (8, 16, 32) or condition not in ("static", "modulated", "open"):
        raise ValueError("n must be 8/16/32 and condition static/modulated/open")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    ring = canonical_edges(zip(permutation, np.roll(permutation, -1)))
    candidates = np.array([(i, j) for i in range(n) for j in range(i+1, n)])
    extra = sorted(canonical_edges(candidates)-ring)
    selected = rng.choice(len(extra), n//2, replace=False)
    edges = np.array(sorted(ring | {extra[i] for i in selected}))
    c = rng.uniform(.7, 1.3, len(edges))
    beta = np.zeros(len(edges))
    if condition != "static":
        active = rng.choice(len(edges), len(edges)//3, replace=False)
        beta[active] = .3*c[active]
    return dict(n=n, seed=seed, condition=condition, edges=edges, c=c, beta=beta,
                candidate_edges=candidates, mass=rng.uniform(.8, 1.2, n),
                damping=rng.uniform(.15, .3, n), reference=np.zeros(n),
                environment_gain=np.full(n, .4 if condition == "open" else 0.))


def forces(system, state):
    """Internal pair exchange sums to zero; environment load need not."""
    theta, omega, p, a, e = np.asarray(state).reshape(5, system["n"])
    i, j = system["edges"].T
    k = system["c"] + system["beta"]*np.expm1(np.tanh(a[i]+a[j]))
    pair = k*np.sin(theta[i]-theta[j])
    internal = np.zeros(system["n"])
    np.add.at(internal, i, pair)
    np.add.at(internal, j, -pair)
    load = system["environment_gain"]*e
    return internal, load


def rhs(system, state):
    theta, omega, p, a, e = np.asarray(state).reshape(5, system["n"])
    internal, load = forces(system, state)
    return np.stack((omega, (p-internal-load-system["damping"]*omega)/system["mass"],
                     -.8*p-.2*omega, -.6*a+.3*omega,
                     -.4*e+.2*omega))


def initial_state(system, seed=0, excitation="distributed"):
    """Event excitation is an unobserved initial disturbance, never a feature.

    distributed: heterogeneous nodes/latent phases; local: one machine;
    common: identical node excitation; equilibrium: exact mechanism-off control.
    mechanical/amplitude/environment isolate their .8/.6/.4 relaxation rates.
    Temporal contrasts arise from independent p/a/e relaxation, without forcing.
    """
    rng = np.random.default_rng(seed)
    n = system["n"]
    if excitation not in ("distributed", "local", "common", "equilibrium",
                           "mechanical", "amplitude", "environment"):
        raise ValueError("unknown excitation")
    state = np.zeros((5, n))
    if excitation == "distributed":
        state = rng.normal(size=(5, n))*np.array([.2, .12, .2, .4, .3])[:, None]
        state[4] += .4
    elif excitation == "local":
        state[:, seed % n] = [.4, .2, .3, .6, .7]
    elif excitation == "common":
        state[:] = np.array([.2, .1, .2, .4, .5])[:, None]
    elif excitation in ("mechanical", "amplitude", "environment"):
        block = {"mechanical": 2, "amplitude": 3, "environment": 4}[excitation]
        state[block] = rng.uniform(.2, .6, n)
    return state


def simulate(system, state, duration=.12, internal_dt=.0025):
    if not np.isfinite(duration) or duration < .1:
        raise ValueError("at least 100 ms history required")
    if internal_dt not in (.0025, .00125):
        raise ValueError("use preregistered RK4 grids .0025/.00125")
    samples = int(round(duration/DT))
    if not np.isclose(samples*DT, duration, atol=1e-12, rtol=0):
        raise ValueError("duration must align with .01 s sampling")
    state = np.array(state, dtype=float, copy=True)
    if state.shape != (5, system["n"]) or not np.isfinite(state).all():
        raise ValueError("finite 5 by n initial state required")
    values = [state.copy()]
    for _ in range(samples):
        for _ in range(round(DT/internal_dt)):
            k1 = rhs(system, state)
            k2 = rhs(system, state+.5*internal_dt*k1)
            k3 = rhs(system, state+.5*internal_dt*k2)
            k4 = rhs(system, state+internal_dt*k3)
            state += internal_dt/6*(k1+2*k2+2*k3+k4)
        values.append(state.copy())
    return np.arange(samples+1)*DT, np.stack(values)


def generate_event(system, seed=0, excitation="distributed", duration=.12,
                   internal_dt=.0025, noise_std=0.):
    """Return (event, truth), physically separate legal observations and truth.

    Observation noise uses its own RNG; changing it preserves exact truth.
    event.targets are supervision only; motion_packet explicitly whitelists input.
    """
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")
    initial = initial_state(system, seed, excitation)
    t, state = simulate(system, initial, duration, internal_dt)
    theta, omega = state[:, 0].copy(), state[:, 1].copy()
    gauge = float(theta[0] @ (system["mass"]/system["mass"].sum()))
    theta -= gauge
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1907]))
    noisy_theta = theta+rng.normal(0, noise_std, theta.shape)
    noisy_omega = omega+rng.normal(0, noise_std, omega.shape)
    # The gauge is a single constant for history and future, never framewise COI.
    public = dict(candidate_edges=system["candidate_edges"].copy(),
                  mass=system["mass"].copy(), damping=system["damping"].copy(),
                  reference=system["reference"].copy(),
                  scales=dict(theta_scale=1., omega_scale=1., torque_scale=1.),
                  angle_gauge="single_constant_initial_inertia_mean")
    future = state[10:]
    targets = dict(theta=theta[10:].copy(), omega=omega[10:].copy(),
                   tau_m=future[:, 2].copy(), tau_e=np.stack([sum(forces(system, s)) for s in future]))
    event = dict(id=f"synthetic_{system['n']}_{system['seed']}_{system['condition']}_{excitation}_{seed}",
                 history=dict(t=t[:11]-.1, theta=noisy_theta[:11].copy(),
                              omega=noisy_omega[:11].copy()),
                 constants=public, t=t[10:]-.1,
                 targets=targets, masks={key: np.ones_like(value, dtype=bool) for key, value in targets.items()})
    truth = dict(**system, initial_state=initial, state=state, t=t,
                 excitation=excitation, event_seed=seed, noise_std=noise_std,
                 internal_dt=internal_dt, angular_offset=gauge)
    return event, truth


def motion_packet(event):
    """Deployment whitelist: no targets, hidden states, labels or truth graph."""
    import copy
    return copy.deepcopy({key: event[key] for key in ("history", "constants")})


def model_constants(packet):
    """Use only the declared public constants to construct the candidate graph."""
    public = packet['constants']
    n = len(public['mass'])
    return dict(mass=public['mass'].copy(), damping=public['damping'].copy(), F=np.eye(n),
                theta_ref=public['reference'].copy(), edge_index=public['candidate_edges'].T.copy(),
                omega_scale=np.full(n, public['scales']['omega_scale']),
                Sp=np.full(n, public['scales']['torque_scale']), Se=np.full(n, public['scales']['torque_scale']),
                S_m=np.full(n, public['scales']['torque_scale']), s_coi=public['scales']['omega_scale'])


def smoke():
    """Two .12 s events per N on both frozen meshes; no persistent dataset."""
    rows = []
    for n in (8, 16, 32):
        system = make_system(n, seed=31)
        for seed, excitation in ((12, "distributed"), (13, "local")):
            _, coarse = generate_event(system, seed, excitation)
            _, fine = generate_event(system, seed, excitation, internal_dt=.00125)
            error = float(np.max(np.abs(coarse["state"]-fine["state"])))
            if error > 1e-9:
                raise ArithmeticError("synthetic two-grid qualification failed")
            rows.append(dict(n=n, seed=seed, excitation=excitation,
                             duration=.12, max_state_mesh_difference=error))
    return dict(status="passed", sample_dt=DT, internal_steps=[.0025, .00125], rows=rows)


if __name__ == "__main__":
    import json
    print(json.dumps(smoke(), indent=2))
