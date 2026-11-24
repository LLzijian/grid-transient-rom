"""Size-independent graph initialization for the unchanged C1 dynamics."""

import torch
from torch import nn

from .osgkd import OSGKD, OSGKDConfig


class _GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 4)).double()
        self.gru = nn.GRU(8, 20, batch_first=True, dtype=torch.float64)
        self.readout = nn.Linear(20, 3, bias=False, dtype=torch.float64)

    def forward(self, theta, omega, theta_ref, omega_scale, mass):
        n = theta.shape[-1]
        offset = (theta[..., -1, :] - theta_ref).mean(-1, keepdim=True)
        centered = theta - offset.unsqueeze(-2) - theta_ref
        diagonal = mass.diagonal()
        local = torch.stack((centered, omega / omega_scale,
                             diagonal.log().expand_as(theta),
                             (diagonal / diagonal.sum()).expand_as(theta)), dim=-1)
        if n == 1:
            messages = theta.new_zeros((*theta.shape, 4))
        else:
            phase = theta.unsqueeze(-1) - theta.unsqueeze(-2)
            reference = theta_ref.unsqueeze(-1) - theta_ref.unsqueeze(-2)
            velocity = omega.unsqueeze(-1) - omega.unsqueeze(-2)
            features = torch.stack((phase.sin() - reference.sin(),
                                    phase.cos() - reference.cos(), velocity), dim=-1)
            edge = self.edge_mlp(features)
            mask = ~torch.eye(n, dtype=torch.bool, device=theta.device)
            messages = (edge * mask[..., None]).sum(-2) / (n - 1)
        features = torch.cat((local, messages), dim=-1).transpose(-3, -2)
        _, hidden = self.gru(features.reshape(-1, theta.shape[-2], 8))
        return self.readout(hidden[0]).reshape(*theta.shape[:-2], n, 3)


class OSGKDGraph(OSGKD):
    checkpoint_version = 4

    def __init__(self, config: OSGKDConfig, **constants):
        if config.variant != "C1" or config.hidden_size != 20:
            raise ValueError("OSGKDGraph requires C1 with hidden_size=20")
        super().__init__(config, **constants)
        self.encoder = _GraphEncoder()
        self.initial_head = nn.Identity()

    def initialize(self, theta_history, omega_history):
        shape = theta_history.shape
        if shape != omega_history.shape or shape[-2:] != (self.history_points, self.n):
            raise ValueError("theta_history and omega_history must have the same (..., 11, N) shape")
        if any(value.dtype != torch.float64 or value.device != self.mass.device
               for value in (theta_history, omega_history)):
            raise ValueError("motion histories must use float64 on the model device")
        if not (torch.isfinite(theta_history).all() & torch.isfinite(omega_history).all()):
            raise ValueError("motion histories must be finite")
        args = (self.theta_ref, self.omega_scale, self.mass)
        raw = self.encoder(theta_history, omega_history, *args)
        reference = self.encoder(self.theta_ref.expand_as(theta_history),
                                 torch.zeros_like(omega_history), *args)
        p, a, e = (raw - reference).unbind(-1)
        latent = (self.config.p_init_bound * p.tanh(),
                  self.config.a_init_bound * (a / 2).tanh(),
                  self.config.e_init_bound * e.tanh())
        return torch.cat((theta_history[..., -1, :], omega_history[..., -1, :], *latent), -1)
