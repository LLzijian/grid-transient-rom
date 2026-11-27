"""Independent lab-frame pendulum truth and shared known-gravity wrappers.

Absolute gravity breaks common-angle U(1) symmetry; node permutations remain
equivariant. The unchanged graph encoder still removes its common-angle offset,
but initialization retains the absolute observed theta/omega endpoints.
"""

from copy import deepcopy

import numpy as np
import torch

from kiki.data import osgkd_synthetic as base_truth
from kiki.models.osgkd import OSGKDConfig
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_baselines import GraphNeuralODE, ClosedPassiveKuramoto


EXCITATIONS = ('distributed', 'local', 'common', 'mixed_hidden')
NOISE_LEVELS = (0., .01)
SYSTEM_SEEDS = (31, 41, 51)
FORECAST_S = 4.


def make_system(n=8, seed=31):
    if n not in (8, 16):
        raise ValueError('pendulum family supports N=8 or 16')
    system = base_truth.make_system(n=n, seed=seed, condition='open')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 981]))
    system['damping'] = system['mass']*rng.uniform(.1, .3, n)
    system['gravity'] = 9.81*system['mass']
    system['reference'] = np.zeros(n)
    system['condition'] = 'pendulum_open'
    system['units'] = dict(mass='kg m^2', theta='rad', omega='rad/s', torque='N m', time='s')
    system['hidden_initial_bounds'] = dict(p=.3, a=.6, e=.4)
    system['relaxation'] = dict(p_decay=.8, p_omega=-.2, a_decay=.6, a_omega=.3, e_decay=.4, e_omega=.2)
    return system


def forces(system, state):
    """Unknown internal and environment torque; known gravity is separate."""
    theta, _, _, a, e = np.asarray(state).reshape(5, system['n'])
    left, right = system['edges'].T
    kernel = system['c']+system['beta']*np.expm1(np.tanh(a[left]+a[right]))
    edge = kernel*np.sin(theta[left]-theta[right])
    internal = np.zeros(system['n'])
    np.add.at(internal, left, edge)
    np.add.at(internal, right, -edge)
    return internal, system['environment_gain']*e


def rhs(system, state):
    """Independent NumPy mechanics, without calling any learned-model RHS."""
    theta, omega, p, a, e = np.asarray(state).reshape(5, system['n'])
    internal, environment = forces(system, state)
    r = system['relaxation']
    return np.stack((omega,
            (p-internal-environment-system['gravity']*np.sin(theta)-system['damping']*omega)/system['mass'],
            -r['p_decay']*p+r['p_omega']*omega,
            -r['a_decay']*a+r['a_omega']*omega,
            -r['e_decay']*e+r['e_omega']*omega))


def initial_state(system, seed, excitation):
    if excitation not in EXCITATIONS:
        raise ValueError('unknown pendulum excitation')
    rng = np.random.default_rng(seed)
    n = system['n']
    state = np.zeros((5, n))
    if excitation == 'local':
        index = seed % n
        state[0, index], state[1, index] = rng.uniform(-.6, .6), rng.uniform(-.5, .5)
    elif excitation == 'common':
        state[0].fill(rng.uniform(-.6, .6))
        state[1].fill(rng.uniform(-.5, .5))
    else:
        state[0], state[1] = rng.uniform(-.6, .6, n), rng.uniform(-.5, .5, n)
        if excitation == 'mixed_hidden':
            for row, name in enumerate(('p', 'a', 'e'), 2):
                bound = system['hidden_initial_bounds'][name]
                state[row] = rng.uniform(-bound, bound, n)
    return state


def simulate(system, state, duration, internal_dt=.0025):
    if not np.isfinite(duration) or duration < .1 or internal_dt not in (.0025, .00125):
        raise ValueError('100ms or longer duration and frozen RK4 mesh required')
    samples = int(round(duration/.01))
    if not np.isclose(samples*.01, duration, atol=1e-12, rtol=0):
        raise ValueError('duration must align with .01s outputs')
    state = np.array(state, dtype=float, copy=True)
    if state.shape != (5, system['n']) or not np.isfinite(state).all():
        raise ValueError('finite 5 by N initial state required')
    output = [state.copy()]
    for _ in range(samples):
        for _ in range(round(.01/internal_dt)):
            k1 = rhs(system, state)
            k2 = rhs(system, state+.5*internal_dt*k1)
            k3 = rhs(system, state+.5*internal_dt*k2)
            k4 = rhs(system, state+internal_dt*k3)
            state += internal_dt/6*(k1+2*k2+2*k3+k4)
        output.append(state.copy())
    return np.arange(samples+1)*.01, np.stack(output)


def motion_packet(event):
    return deepcopy({key: event[key] for key in ('history', 'constants')})


def model_constants(packet):
    result = base_truth.model_constants(packet)
    result['gravity'] = packet['constants']['gravity'].copy()
    return result


def generate_episode(system, episode_seed, excitation, noise_levels=NOISE_LEVELS,
                     forecast_s=FORECAST_S, internal_dt=.0025):
    if not np.isfinite(forecast_s) or forecast_s <= 0:
        raise ValueError('positive forecast duration required')
    levels = tuple(float(value) for value in noise_levels)
    if (not levels or any(not np.isfinite(value) or value < 0 for value in levels) or
            len({f'{value:.2f}' for value in levels}) != len(levels)):
        raise ValueError('distinct finite nonnegative two-decimal noise levels required')
    initial = initial_state(system, episode_seed, excitation)
    times, states = simulate(system, initial, .1+forecast_s, internal_dt)
    public = dict(candidate_edges=system['candidate_edges'].copy(), mass=system['mass'].copy(),
                  damping=system['damping'].copy(), reference=np.zeros(system['n']),
                  gravity=system['gravity'].copy(), scales=dict(theta_scale=1., omega_scale=1., torque_scale=1.),
                  angle_gauge='absolute_lab_vertical_down; no common-angle removal', units=deepcopy(system['units']))
    future = states[10:]
    targets = dict(theta=future[:, 0].copy(), omega=future[:, 1].copy(), tau_m=future[:, 2].copy(),
                   tau_e=np.stack([sum(forces(system, state)) for state in future]))
    clean = dict(id=f"pendulum_{system['n']}_{system['seed']}_{excitation}_{episode_seed}",
                 history=dict(t=times[:11]-.1, theta=states[:11, 0].copy(), omega=states[:11, 1].copy()),
                 constants=public, t=times[10:]-.1, targets=targets,
                 masks={key: np.ones_like(value, dtype=bool) for key, value in targets.items()})
    rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 981031]))
    direction = {key: rng.standard_normal(clean['history'][key].shape) for key in ('theta', 'omega')}
    observed = {}
    for level in levels:
        event = deepcopy(clean)
        for key in direction:
            event['history'][key] += level*direction[key]
        observed[f'{level:.2f}'] = event
    private = dict(system=deepcopy(system), initial_state=initial, state=states, t=times,
                   excitation=excitation, event_seed=episode_seed, noise_direction=direction,
                   noise_levels=levels, internal_dt=internal_dt)
    return observed, private


def make_dataset(n=8, system_seed=31, train_count=64, val_count=16,
                 noise_levels=NOISE_LEVELS, forecast_s=FORECAST_S, internal_dt=.0025,
                 train_seed=1000, val_seed=2000):
    for count in (train_count, val_count):
        if type(count) is not int or count <= 0 or count % 4:
            raise ValueError('positive episode counts divisible by four required')
    if set(range(train_seed, train_seed+train_count)) & set(range(val_seed, val_seed+val_count)):
        raise ValueError('train and val seeds overlap')
    system = make_system(n, system_seed)
    result = dict(train={}, val={}, constants=None,
                  contract=dict(history_points=11, history_s=.1, forecast_s=forecast_s,
                                output_step=.01, truth_internal_step=internal_dt,
                                angle_reference='absolute_lab_vertical_down',
                                noise_scope='history_only', noise_levels=list(noise_levels),
                                port_convention='tau_e=internal+environment; known gravity excluded'))
    private = dict(system=deepcopy(system), train=[], val=[])
    for split, count, first in (('train', train_count, train_seed), ('val', val_count, val_seed)):
        for index in range(count):
            events, truth = generate_episode(system, first+index, EXCITATIONS[index//(count//4)],
                                              noise_levels, forecast_s, internal_dt)
            for key, event in events.items():
                result[split].setdefault(key, []).append(event)
                if result['constants'] is None:
                    result['constants'] = model_constants(motion_packet(event))
            private[split].append(truth)
    return result, private


def select_subset16(public_pack):
    """Frozen 4-per-class subset of the 64 ordered train events, at every noise."""
    indices = [16*group+offset for group in range(4) for offset in range(4)]
    if any(len(events) != 64 for events in public_pack['train'].values()):
        raise ValueError('the ordered 64-event train package is required')
    result = deepcopy(public_pack)
    result['train'] = {key: [events[index] for index in indices] for key, events in result['train'].items()}
    result['contract']['train_subset_indices'] = indices
    return result


class GravityMixin:
    """Fixed known conservative drift, shared by all three candidate families."""

    def __init__(self, config, *, gravity, **constants):
        super().__init__(config, **constants)
        value = torch.as_tensor(gravity, dtype=torch.float64).detach().clone()
        if value.shape != (self.n,) or not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError('finite nonnegative gravity coefficients required')
        if self.theta_ref.ne(0).any() or not torch.equal(self.mass, self.mass.diagonal().diag()):
            raise ValueError('lab-zero reference and diagonal inertia required')
        self.register_buffer('gravity', value)

    def rhs(self, t, state):
        original = super().rhs(t, state)
        theta = state[..., :self.n]
        drift = -self.gravity*theta.sin()/self.mass.diagonal()
        return torch.cat((original[..., :self.n], original[..., self.n:2*self.n]+drift,
                          original[..., 2*self.n:]), -1)

    forward = rhs

    def gravity_potential(self, state):
        return (self.gravity*(1-state[..., :self.n].cos())).sum(-1)

    def storage_power(self, state):
        # NeuralODE deliberately still raises: it has no declared storage law.
        original = super().storage_power(state)
        return original._replace(H=original.H+self.gravity_potential(state))

    def checkpoint(self):
        return {**super().checkpoint(), 'gravity_wrapper': type(self).__name__, 'gravity_version': 1}

    @classmethod
    def from_checkpoint(cls, saved):
        if (saved.get('gravity_wrapper'), saved.get('gravity_version'), saved.get('contract'), saved.get('version')) != (
                cls.__name__, 1, cls.contract, cls.checkpoint_version):
            raise ValueError('pendulum gravity checkpoint identity differs')
        state = saved['state_dict']
        names = ('mass', 'damping', 'F', 'Sp', 'Se', 'omega_scale', 'theta_ref', 'edge_index', 'gravity')
        model = cls(OSGKDConfig(**saved['config']), **{key: state[key] for key in names})
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError('nonfinite pendulum checkpoint')
        return model


class PendulumOSGKDGraph(GravityMixin, OSGKDGraph):
    pass


class PendulumGraphNeuralODE(GravityMixin, GraphNeuralODE):
    pass


class PendulumClosedPassive(GravityMixin, ClosedPassiveKuramoto):
    pass


def closed_conservative_check():
    """Independent two-bob undamped example: d(T+Uc+Vg)/dt=0 exactly."""
    inertia = np.array([.8, 1.2])
    gravity = 9.81*inertia
    theta = np.array([.4, -.3])
    omega = np.array([.2, -.1])
    coupling = .9*np.sin(theta[0]-theta[1])*np.array([1., -1.])
    acceleration = (-coupling-gravity*np.sin(theta))/inertia
    dkinetic = float((inertia*omega*acceleration).sum())
    dcoupling = float(coupling@omega)
    dgravity = float((gravity*np.sin(theta)*omega).sum())
    assert dgravity > 0 and -dgravity < 0
    assert abs(dkinetic+dcoupling+dgravity) < 1e-14
    return dict(scope='independent closed two-bob mechanics; D=p=e=beta=0',
                gravity_potential_derivative_forward=dgravity,
                gravity_potential_derivative_reversed=-dgravity,
                gravity_kinetic_power=-dgravity, total_energy_chain_residual=dkinetic+dcoupling+dgravity)


def smoke():
    """A single .11s artificial trajectory plus algebraic wrapper checks."""
    system = make_system()
    events, private = generate_episode(system, 11, 'mixed_hidden', forecast_s=.01)
    clean, noisy = events['0.00'], events['0.01']
    assert clean['history']['theta'].shape == (11, 8)
    assert clean['targets']['theta'].shape == (2, 8)
    assert np.array_equal(clean['history']['theta'], private['state'][:11, 0])
    assert all(np.array_equal(clean['targets'][key], noisy['targets'][key]) for key in clean['targets'])
    constants = model_constants(motion_packet(clean))
    model_inputs = {key: value for key, value in constants.items() if key not in ('S_m', 's_coi')}
    config = OSGKDConfig('C1', hidden_size=20)
    theta = torch.as_tensor(clean['history']['theta'])
    omega = torch.as_tensor(clean['history']['omega'])
    checks = []
    for cls in (PendulumOSGKDGraph, PendulumGraphNeuralODE, PendulumClosedPassive):
        model = cls(config, **model_inputs)
        state = model.initialize(theta, omega)
        torch.testing.assert_close(state[:8], theta[-1], rtol=0, atol=0)
        torch.testing.assert_close(state[8:16], omega[-1], rtol=0, atol=0)
        torch.testing.assert_close(model(state.new_zeros(()), state), model.rhs(None, state))
        clone = cls.from_checkpoint(model.checkpoint())
        torch.testing.assert_close(clone.rhs(None, state), model.rhs(None, state))
        if cls is not PendulumGraphNeuralODE:
            z = state.detach().requires_grad_()
            power = model.storage_power(z)
            gradient = torch.autograd.grad(power.H, z)[0]
            torch.testing.assert_close((gradient*model.rhs(None, z)).sum(), power.total_power, atol=1e-12, rtol=1e-11)
        checks.append(cls.__name__)
    return dict(status='passed', artificial_truth_duration_s=.11, wrappers=checks,
                closed_conservative=closed_conservative_check(), formal_dataset_generated=False)
