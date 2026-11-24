"""Versioned zero-sum fast electrical memory extension of OSGKD."""

from math import ceil, isfinite, log
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .osgkd import OSGKD, OSGKDConfig


class FastPorts(NamedTuple):
    tau_m: Tensor
    Fc: Tensor
    tau_env: Tensor
    F_fast: Tensor

    @property
    def tau_e(self):
        return self.Fc + self.tau_env + self.F_fast


class FastStoragePower(NamedTuple):
    H: Tensor
    Uc: Tensor
    mechanical_power: Tensor
    environment_power: Tensor
    damping_power: Tensor
    amplitude_power: Tensor
    fast_power: Tensor

    @property
    def total_power(self):
        return (self.mechanical_power + self.environment_power + self.damping_power
                + self.amplitude_power + self.fast_power)


class OSGKDFastZeroSum(OSGKD):
    checkpoint_version = 3

    def __init__(self, config: OSGKDConfig, *, fast_seed=24103, **constants):
        super().__init__(config, **constants)
        # The new head must neither consume nor change the shared RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(fast_seed)
            self.fast_head = nn.Linear(config.hidden_size, self.n, dtype=torch.float64)
            nn.init.normal_(self.fast_head.weight, std=0.01)
            nn.init.zeros_(self.fast_head.bias)
        self.fast_gamma = nn.Parameter(torch.zeros(self.n, dtype=torch.float64))
        self.fast_rate_raw = nn.Parameter(torch.full((self.n,), log(1 / 8), dtype=torch.float64))
        self.fast_kappa = nn.Parameter(torch.full((self.n,), 0.1, dtype=torch.float64))
        self.fast_edge_coeff = nn.Parameter(torch.zeros(self.c.numel(), 2, 2, dtype=torch.float64))

    @property
    def fast_rate(self):
        return 5 + 45 * self.fast_rate_raw.sigmoid()

    def fast_parameters(self):
        return iter((self.fast_gamma, self.fast_rate_raw, self.fast_kappa,
                     self.fast_edge_coeff, *self.fast_head.parameters()))

    def _split_fast(self, state):
        if state.ndim == 0 or state.shape[-1] != 6 * self.n:
            raise ValueError("state must have shape (..., 6N)")
        return state[..., :5 * self.n], state[..., 5 * self.n:]

    def initialize(self, theta_history, omega_history):
        shape = theta_history.shape
        if shape != omega_history.shape or shape[-2:] != (self.history_points, self.n):
            raise ValueError("theta_history and omega_history must have the same (..., 11, N) shape")
        if any(value.dtype != torch.float64 or value.device != self.mass.device
               for value in (theta_history, omega_history)):
            raise ValueError("motion histories must use float64 on the model device")
        if not (torch.isfinite(theta_history).all() & torch.isfinite(omega_history).all()):
            raise ValueError("motion histories must be finite")
        endpoint, omega_endpoint = theta_history[..., -1, :], omega_history[..., -1, :]
        offset = (endpoint - self.theta_ref).mean(dim=-1, keepdim=True)
        theta = theta_history - offset.unsqueeze(-2) - self.theta_ref
        features = torch.cat((theta, omega_history / self.omega_scale), dim=-1)
        _, last = self.encoder(features.reshape(-1, self.history_points, 2 * self.n))
        latent = (self.initial_head(last[0]).tanh() * self.init_bounds).reshape(*shape[:-2], 3 * self.n)
        old = torch.cat((endpoint, omega_endpoint, latent), dim=-1)
        fast = self.fast_head(last[0]).tanh().reshape(*old.shape[:-1], self.n)
        return torch.cat((old, fast), dim=-1)

    def fast_force(self, f):
        q = self.Se * self.fast_gamma * f
        return q - q.mean(dim=-1, keepdim=True)

    def ports(self, state):
        old, f = self._split_fast(state)
        ports = super().ports(old)
        return FastPorts(*ports, self.fast_force(f))

    def rhs(self, t, state):
        old, f = self._split_fast(state)
        derivative = super().rhs(t, old)
        theta, omega = old[..., :self.n], old[..., self.n:2 * self.n]
        force = self.fast_force(f)
        # Separate correction preserves the original acceleration's bit pattern at zero gate.
        acceleration = derivative[..., self.n:2 * self.n] - force @ self.inverse_mass.T
        derivative = torch.cat((derivative[..., :self.n], acceleration,
                                derivative[..., 2 * self.n:]), dim=-1)
        weights = self.mass.sum(dim=0)
        coi = (omega * weights).sum(dim=-1, keepdim=True) / weights.sum()
        left, right = self.edge_index
        phase = theta[..., left] - theta[..., right]
        features = torch.stack((phase.sin() - self.sin_ref, phase.cos() - self.cos_ref), dim=-1)
        terms = (features.unsqueeze(-2) * self.fast_edge_coeff).sum(dim=-1).flatten(-2)
        incident = f.new_zeros(f.shape).index_add(-1, self.edge_nodes, terms)
        f_dot = -self.fast_rate * f + self.fast_kappa * (omega - coi) / self.omega_scale + incident
        return torch.cat((derivative, f_dot), dim=-1)

    forward = rhs

    def storage_power(self, state):
        old, f = self._split_fast(state)
        base = super().storage_power(old)
        omega = old[..., self.n:2 * self.n]
        return FastStoragePower(*base, -(omega * self.fast_force(f)).sum(dim=-1))

    def rollout(self, initial_state, times, *, max_step=0.01):
        self._split_fast(initial_state)
        if initial_state.dtype != torch.float64 or initial_state.device != self.mass.device:
            raise ValueError("initial_state must use float64 on the model device")
        if not isfinite(max_step) or max_step <= 0:
            raise ValueError("max_step must be finite and positive")
        times = torch.as_tensor(times, dtype=torch.float64, device="cpu")
        if times.ndim != 1 or times.numel() < 1 or not torch.isfinite(times).all():
            raise ValueError("times must be a finite nonempty vector")
        intervals = times.diff().tolist()
        if any(interval <= 0 or not isfinite(interval) for interval in intervals):
            raise ValueError("times must be strictly increasing with finite intervals")
        state, trajectory = initial_state, [initial_state]
        for interval in intervals:
            substeps = max(1, ceil(interval / max_step - 1e-12))
            for _ in range(substeps):
                state = self.rk4_step(state, interval / substeps)
            trajectory.append(state)
        return torch.stack(trajectory, dim=-2)

    @classmethod
    def from_parent_checkpoint(cls, payload, fast_seed=24103):
        parent = OSGKD.from_checkpoint(payload)
        names = ("mass", "damping", "F", "Sp", "Se", "omega_scale", "theta_ref", "edge_index")
        model = cls(parent.config, fast_seed=fast_seed,
                    **{name: getattr(parent, name) for name in names})
        model.load_state_dict(parent.state_dict(), strict=False)
        return model
