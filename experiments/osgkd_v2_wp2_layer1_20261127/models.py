"""Layer 1 model assembly, train-only structure selection and size mapping."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from kiki.models.osgkd import OSGKDConfig, Ports
from kiki.models.osgkd_baselines import ClosedPassiveKuramoto, GraphNeuralODE
from kiki.models.osgkd_graph import OSGKDGraph

from .synthetic import full_offset_lift


CONSTANTS = ('mass', 'damping', 'F', 'Sp', 'Se', 'omega_scale', 'theta_ref', 'edge_index')
NODE_COEFFICIENTS = ('mechanical_coeff', 'amplitude_coeff', 'environment_coeff', 'memory_coeff')
METHODS = ('osgkd', 'static_c0', 'neural_ode', 'closed_passive', 'weak_static')
FUNCTION_THRESHOLD = .01


class WeakStatic(ClosedPassiveKuramoto):
    """Weak-fit static exchange and a jointly trained mechanical port."""

    def __init__(self, config, **constants):
        super().__init__(config, **constants)
        self.c.requires_grad_(False)
        self.mechanical_coeff = nn.Parameter(torch.zeros((2, self.n), dtype=torch.float64))

    def initialize(self, theta_history, omega_history):
        initial = OSGKDGraph.initialize(self, theta_history, omega_history)
        return torch.cat((initial[..., :3 * self.n], torch.zeros_like(initial[..., 3 * self.n:])), -1)

    def ports(self, state):
        parent = super().ports(state)
        return Ports(state[..., 2 * self.n:3 * self.n], parent.Fc, parent.tau_env)

    def rhs(self, t, state):
        _, omega, p, a, e = self._split(state)
        ports = self.ports(state)
        acceleration = (p - ports.tau_e - omega @ self.damping.T) @ self.inverse_mass.T
        dp = self.mechanical_coeff[0] * p + self.mechanical_coeff[1] * omega
        return torch.cat((omega, acceleration, dp, -a, -e), -1)

    forward = rhs

    @property
    def effective_state_dimension(self):
        return 3 * self.n

    def storage_power(self, state):
        parent = super().storage_power(state)
        omega = self._split(state)[1]
        return parent._replace(mechanical_power=(omega * self.ports(state).tau_m).sum(-1))


def make_model(method, constants, seed):
    torch.manual_seed(seed)
    cls = {'osgkd': OSGKDGraph, 'static_c0': OSGKDGraph, 'neural_ode': GraphNeuralODE,
           'closed_passive': ClosedPassiveKuramoto, 'weak_static': WeakStatic}[method]
    model = cls(OSGKDConfig('C1', hidden_size=20, a_init_bound=2.),
                **{name: constants[name] for name in CONSTANTS})
    with torch.no_grad():
        for name in ('mechanical_coeff', 'amplitude_coeff', 'memory_coeff'):
            if hasattr(model, name):
                getattr(model, name)[0].fill_(-1.)
    if method == 'static_c0':
        with torch.no_grad():
            model.beta.zero_()
        model.beta.requires_grad_(False)
    if method == 'closed_passive':
        model.encoder.requires_grad_(False)
    return model


def weak_force_fit(episodes, model, steps, iterations=512):
    """Fit observed total electric torque with a static reciprocal dictionary.

    Four equally spaced smooth test functions integrate the public signals
    without differentiating noisy motion or subtracting a true hidden port.
    """
    if steps < 8 or steps % 4:
        raise ValueError('four windows of at least two steps required for weak-form fitting')
    left, right = model.edge_index.detach().numpy()
    incidence = model.incidence.detach().numpy()
    weights = np.sin(np.pi * np.linspace(0, 1, steps // 4 + 1)) ** 2
    weights[[0, -1]] = 0.
    weights /= weights.sum()
    blocks, targets = [], []
    for event in episodes:
        obs = event['targets']
        phase = obs['theta'][:steps + 1, left] - obs['theta'][:steps + 1, right]
        phase = np.sin(phase)
        for start in range(0, steps, steps // 4):
            window = slice(start, start + len(weights))
            blocks.append(incidence * (weights @ phase[window]))
            targets.append(weights @ obs['tau_e'][window])
    design = np.concatenate(blocks)
    target = np.concatenate(targets)
    gram = (design.T @ design) / len(target)
    rhs = (design.T @ target) / len(target)
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    # Projected FISTA for nonnegative coefficients with a fixed L1 penalty.
    coeff = np.zeros_like(rhs)
    momentum = coeff.copy()
    rate = 1.
    if lipschitz > 0:
        for _ in range(iterations):
            updated = np.maximum(0., momentum - (gram @ momentum - rhs + 1e-4) / lipschitz)
            next_rate = (1 + np.sqrt(1 + 4 * rate ** 2)) / 2
            momentum = updated + (rate - 1) / next_rate * (updated - coeff)
            coeff, rate = updated, next_rate
    with torch.no_grad():
        model.c.copy_(torch.from_numpy(coeff))
    projected = np.where(coeff > 0, gram @ coeff - rhs + 1e-4,
                         np.minimum(gram @ coeff - rhs + 1e-4, 0.))
    return {'iterations': iterations if lipschitz > 0 else 0, 'rows': len(target),
            'l1': 1e-4, 'active_coefficients': int(np.count_nonzero(coeff)),
            'projected_gradient_l2': float(np.linalg.norm(projected)),
            'weak_torque_rmse': float(np.sqrt(np.mean((design @ coeff - target) ** 2)))}


def edge_features(model, theta, amplitude):
    left, right = model.edge_index
    sine = (theta[..., left] - theta[..., right]).sin()
    return sine - model.sin_ref, (amplitude[..., left] + amplitude[..., right]).tanh().expm1() * sine


@dataclass(frozen=True)
class EdgeSupport:
    base: torch.Tensor
    modulation: torch.Tensor

    def masks(self, model):
        result = {'c': self.base}
        if hasattr(model, 'beta'):
            result['beta'] = self.modulation
            result['edge_coeff'] = self.base[:, None, None].expand_as(model.edge_coeff)
        return result


@torch.no_grad()
def project(model, masks):
    model.project_parameters_()
    for name, mask in masks.items():
        getattr(model, name).masked_fill_(~mask, 0.)


def mask_gradients(model, masks):
    for name, mask in masks.items():
        parameter = getattr(model, name)
        if parameter.grad is not None:
            parameter.grad.masked_fill_(~mask, 0.)


def threshold_support(rms, threshold=FUNCTION_THRESHOLD):
    modulation = rms[1] >= threshold
    base = (rms[0] >= threshold) | modulation
    return EdgeSupport(base=base, modulation=modulation)


@torch.no_grad()
def lifted_model(source, constants, seed):
    """Copy the encoder and lift the open ODE using public node identities."""
    target = make_model('osgkd', constants, seed)
    if target.n % source.n:
        raise ValueError('integer number of source node copies required')
    lift = full_offset_lift(target.n // source.n, source_n=source.n)
    nodes = torch.as_tensor(lift['node_source'])
    edges = torch.as_tensor(lift['edge_source'])
    weight = torch.as_tensor(lift['edge_weight'])
    if not (torch.equal(source.edge_index.T, torch.as_tensor(lift['source_edges'])) and
            torch.equal(target.edge_index.T, torch.as_tensor(lift['target_edges']))):
        raise ValueError('candidate edge ordering differs from the registered lift')
    mapped = edges.clamp_min(0)
    # At an edge whose roles reverse order, swap endpoint coefficients and
    # change the sign of the two sine dictionary columns.
    first, second = target.edge_index
    reversed_roles = nodes[first] > nodes[second]
    target.encoder.load_state_dict(source.encoder.state_dict())
    for name in NODE_COEFFICIENTS:
        getattr(target, name).copy_(getattr(source, name)[:, nodes])
    target.c.copy_(source.c[mapped] * weight)
    target.beta.copy_(source.beta[mapped] * weight)
    terms = source.edge_coeff[mapped].clone()
    signs = terms.new_tensor([-1, 1, -1, 1, 1])
    terms[reversed_roles] = terms[reversed_roles].flip((1,)) * signs
    target.edge_coeff.copy_(terms * weight[:, None, None])
    return target
