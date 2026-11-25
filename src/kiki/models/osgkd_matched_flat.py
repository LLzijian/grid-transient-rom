"""Flat motion initializer matched to the graph initializer's reference and bounds."""

import torch
from torch import nn

from .osgkd import OSGKD, OSGKDConfig


class OSGKDMatchedFlat(OSGKD):
    checkpoint_version = 5

    def __init__(self, config: OSGKDConfig, **constants):
        if config.variant != "C1" or config.hidden_size != 32:
            raise ValueError("OSGKDMatchedFlat requires C1 with hidden_size=32")
        super().__init__(config, **constants)
        self.initial_head = nn.Linear(32, 3 * self.n, bias=False, dtype=torch.float64)

    def initialize(self, theta_history, omega_history):
        shape = theta_history.shape
        if shape != omega_history.shape or shape[-2:] != (self.history_points, self.n):
            raise ValueError("theta_history and omega_history must have the same (..., 11, N) shape")
        if any(value.dtype != torch.float64 or value.device != self.mass.device
               for value in (theta_history, omega_history)):
            raise ValueError("motion histories must use float64 on the model device")
        if not (torch.isfinite(theta_history).all() & torch.isfinite(omega_history).all()):
            raise ValueError("motion histories must be finite")
        offset = (theta_history[..., -1, :] - self.theta_ref).mean(-1, keepdim=True)
        centered = theta_history - offset.unsqueeze(-2) - self.theta_ref
        features = torch.cat((centered, omega_history / self.omega_scale), -1)
        features = features.reshape(-1, self.history_points, 2 * self.n)
        _, hidden = self.encoder(features)
        _, reference = self.encoder(torch.zeros_like(features))
        logits = self.initial_head(hidden[0]) - self.initial_head(reference[0])
        p, a, e = logits.reshape(*shape[:-2], 3 * self.n).split(self.n, -1)
        latent = (self.config.p_init_bound * p.tanh(),
                  self.config.a_init_bound * (a / 2).tanh(),
                  self.config.e_init_bound * e.tanh())
        return torch.cat((theta_history[..., -1, :], omega_history[..., -1, :], *latent), -1)
