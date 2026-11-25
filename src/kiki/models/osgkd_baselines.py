"""Matched graph-initialized baselines under motion_only_v1.

The neural baseline's Ports decomposition is an interface convention only:
Fc is zero and tau_env carries its entire learned electrical readout.
The closed baseline has effective state dimension 2N; its shared encoder is
retained for interface/capacity accounting but cannot influence predictions.
"""

import torch
from torch import nn

from .osgkd import Ports, StoragePower
from .osgkd_graph import OSGKDGraph


class _Baseline(OSGKDGraph):
    def __init__(self, config, **constants):
        super().__init__(config, **constants)
        for name in ('c', 'beta', 'mechanical_coeff', 'amplitude_coeff',
                     'environment_coeff', 'memory_coeff', 'edge_coeff'):
            delattr(self, name)

    def _split(self, state):
        if state.ndim == 0 or state.shape[-1] != 5 * self.n:
            raise ValueError('state must have shape (..., 5N)')
        return state.split(self.n, dim=-1)


class GraphNeuralODE(_Baseline):
    """Dense shared message ODE, without reciprocity or passivity constraints."""
    checkpoint_version = 6

    def __init__(self, config, **constants):
        super().__init__(config, **constants)
        self.edge_mlp = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 6)).double()
        self.node_mlp = nn.Sequential(nn.Linear(12, 28), nn.Tanh(), nn.Linear(28, 5)).double()

    def _readout(self, state):
        theta, omega, p, a, e = self._split(state)
        phase = theta.unsqueeze(-1) - theta.unsqueeze(-2)
        reference = self.theta_ref.unsqueeze(-1) - self.theta_ref.unsqueeze(-2)
        differences = [v.unsqueeze(-1) - v.unsqueeze(-2) for v in (omega, p, a, e)]
        edge = torch.stack((phase.sin() - reference.sin(),
                            phase.cos() - reference.cos(), *differences), dim=-1)
        if self.n == 1:
            messages = state.new_zeros((*state.shape[:-1], 1, 6))
        else:
            mask = ~torch.eye(self.n, dtype=torch.bool, device=state.device)
            messages = (self.edge_mlp(edge) * mask[..., None]).sum(-2) / (self.n - 1)
        mass = self.mass.diagonal()
        features = torch.stack((omega, p, a, e, mass.log().expand_as(omega),
                                (mass / mass.sum()).expand_as(omega)), dim=-1)
        return self.node_mlp(torch.cat((features, messages), dim=-1))

    def _outputs(self, state):
        self._split(state)
        reference = torch.cat((self.theta_ref, self.theta_ref.new_zeros(4 * self.n)))
        return (self._readout(state) - self._readout(reference)).unbind(-1)

    def ports(self, state):
        tau_m, tau_e, _, _, _ = self._outputs(state)
        return Ports(tau_m, torch.zeros_like(tau_e), tau_e)

    def rhs(self, t, state):
        omega = self._split(state)[1]
        tau_m, tau_e, dp, da, de = self._outputs(state)
        acceleration = (tau_m - tau_e - omega @ self.damping.T) @ self.inverse_mass.T
        return torch.cat((omega, acceleration, dp, da, de), dim=-1)

    forward = rhs

    @torch.no_grad()
    def project_parameters_(self):
        pass

    def storage_power(self, state):
        raise NotImplementedError('GraphNeuralODE has no declared storage law')


class ClosedPassiveKuramoto(_Baseline):
    """Nonnegative static cosine potential with frozen PSD damping.

    U=sum c*(1-cos(theta_i-theta_j-phi_ref)); latent coordinates are
    inactive and initialized to zero. The retained 1928-parameter encoder
    has zero effective initialization capacity for this 2N dynamical system.
    """
    checkpoint_version = 7

    def __init__(self, config, **constants):
        super().__init__(config, **constants)
        self.c = nn.Parameter(torch.full((self.edge_index.shape[1],), config.c_init,
                                         dtype=torch.float64))

    @property
    def effective_state_dimension(self):
        return 2 * self.n

    def initialize(self, theta_history, omega_history):
        shared = super().initialize(theta_history, omega_history)
        return torch.cat((shared[..., :2 * self.n], torch.zeros_like(shared[..., 2 * self.n:])), -1)

    def ports(self, state):
        theta = self._split(state)[0]
        left, right = self.edge_index
        delta = theta[..., left] - theta[..., right] - self.phi_ref
        coupling = (self.c.clamp_min(0) * delta.sin()) @ self.incidence.T
        zero = torch.zeros_like(theta)
        return Ports(zero, coupling, zero)

    def rhs(self, t, state):
        omega = self._split(state)[1]
        acceleration = (-self.ports(state).tau_e - omega @ self.damping.T) @ self.inverse_mass.T
        # Keep the embedded dummy slots out of the leading reference spectrum.
        return torch.cat((omega, acceleration, -state[..., 2 * self.n:]), -1)

    forward = rhs

    def storage_power(self, state):
        theta, omega, _, _, _ = self._split(state)
        left, right = self.edge_index
        delta = theta[..., left] - theta[..., right] - self.phi_ref
        potential = (self.c.clamp_min(0) * (1 - delta.cos())).sum(-1)
        kinetic = .5 * (omega * (omega @ self.mass)).sum(-1)
        zero = torch.zeros_like(potential)
        damping = -(omega * (omega @ self.damping)).sum(-1)
        return StoragePower(kinetic + potential, potential, zero, zero, damping, zero)

    @torch.no_grad()
    def project_parameters_(self):
        self.c.clamp_(min=0)
