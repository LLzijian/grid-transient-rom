"""Reference-anchored high-order electrical potential, nested in OSGKD."""

import torch
from torch import nn

from .osgkd import OSGKD, Ports


class OSGKDAnchoredPotential(OSGKD):
    checkpoint_version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = nn.Parameter(torch.zeros(self.c.numel(), 2, dtype=torch.float64))
        self.register_buffer("harmonics", torch.tensor((2., 3.), dtype=torch.float64), persistent=False)

    @classmethod
    def from_parent_checkpoint(cls, payload):
        """Validate a parent snapshot and embed it with exactly zero new coefficients."""
        parent = OSGKD.from_checkpoint(payload)
        nested = parent.checkpoint()
        nested["version"] = cls.checkpoint_version
        nested["state_dict"]["rho"] = parent.c.new_zeros(parent.c.numel(), 2)
        return cls.from_checkpoint(nested)

    def _correction(self, phase, *, potential=False):
        d = phase - self.phi_ref
        m = self.harmonics
        gamma = .5 * self.rho.tanh()
        md = d.unsqueeze(-1) * m
        if potential:
            basis = 1 - md.cos() - m.square() * (1 - d.cos().unsqueeze(-1))
            return (gamma * basis).sum(-1)
        force = m * md.sin() - m.square() * d.sin().unsqueeze(-1)
        return (gamma * force).sum(-1)

    def _parts(self, state):
        parts = super()._parts(state)
        force = self._correction(parts.phase)
        ports = parts.ports
        return parts._replace(ports=Ports(
            ports.tau_m, ports.Fc + (parts.kernel * force) @ self.incidence.T, ports.tau_env))

    def storage_power(self, state):
        base = super().storage_power(state)
        parts = self._parts(state)
        potential = self._correction(parts.phase, potential=True)
        delta_u = (parts.kernel * potential).sum(-1)
        left, right = self.edge_index
        activation = (parts.a[..., left] + parts.a[..., right]).tanh()
        dk_da = parts.beta * activation.exp() * (1 - activation.square())
        grad_a = (dk_da * potential) @ self.unsigned_incidence.T
        a_dot = self._amplitude_derivative(parts, parts.omega / self.omega_scale)
        return base._replace(H=base.H + delta_u, Uc=base.Uc + delta_u,
                             amplitude_power=base.amplitude_power + (grad_a * a_dot).sum(-1))
