"""Three-port OSGKD dynamics under the motion_only_v1 deployment contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import NamedTuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OSGKDConfig:
    variant: str
    hidden_size: int = 32
    p_init_bound: float = 2.0
    a_init_bound: float = 1.0
    e_init_bound: float = 2.0
    c_init: float = 0.1
    beta_fraction_init: float = 0.1

    def __post_init__(self):
        if self.variant not in ("C0", "C1"):
            raise ValueError("variant must be C0 or C1")
        if type(self.hidden_size) is not int or self.hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        for name in ("p_init_bound", "a_init_bound", "e_init_bound"):
            if not isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.c_init) or self.c_init < 0:
            raise ValueError("c_init must be finite and nonnegative")
        if not 0 <= self.beta_fraction_init <= 1:
            raise ValueError("beta_fraction_init must be between zero and one")


class Ports(NamedTuple):
    tau_m: Tensor
    Fc: Tensor
    tau_env: Tensor

    @property
    def tau_e(self) -> Tensor:
        return self.Fc + self.tau_env


class StoragePower(NamedTuple):
    H: Tensor
    Uc: Tensor
    mechanical_power: Tensor
    environment_power: Tensor
    damping_power: Tensor
    amplitude_power: Tensor

    @property
    def total_power(self) -> Tensor:
        return self.mechanical_power + self.environment_power + self.damping_power + self.amplitude_power


class _Parts(NamedTuple):
    omega: Tensor
    p: Tensor
    a: Tensor
    e: Tensor
    phase: Tensor
    cos_phase: Tensor
    tanh_a: Tensor
    a_edges: Tensor
    e_edges: Tensor
    kernel: Tensor
    c: Tensor
    beta: Tensor
    ports: Ports


def _frozen(name, value, shape=None) -> Tensor:
    value = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    if (shape is not None and value.shape != shape) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return value


def _symmetric(name, value, n) -> Tensor:
    value = _frozen(name, value)
    if value.shape == (n,):
        value = value.diag()
    if value.shape != (n, n) or not torch.equal(value, value.T):
        raise ValueError(f"{name} must be a length-{n} diagonal or symmetric ({n}, {n}) matrix")
    return value


class OSGKD(nn.Module):
    contract = "motion_only_v1"
    checkpoint_version = 1
    history_points = 11

    def __init__(self, config: OSGKDConfig, *, mass, damping, F, Sp, Se, omega_scale,
                 theta_ref, edge_index):
        super().__init__()
        self.config = config
        theta_ref = _frozen("theta_ref", theta_ref)
        if theta_ref.ndim != 1 or theta_ref.numel() < 1:
            raise ValueError("theta_ref must be a nonempty node vector")
        self.n = n = theta_ref.numel()
        mass = _symmetric("mass", mass, n)
        damping = _symmetric("damping", damping, n)
        if torch.linalg.cholesky_ex(mass).info != 0:
            raise ValueError("mass must be positive definite")
        if torch.linalg.eigvalsh(damping).min() < 0:
            raise ValueError("damping must be positive semidefinite")

        edge_index = torch.as_tensor(edge_index, device="cpu").detach().clone()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype not in (
            torch.int16, torch.int32, torch.int64
        ):
            raise ValueError("edge_index must be an integer tensor of shape (2, E)")
        edge_index = edge_index.long()
        left, right = edge_index
        edges = left.numel()
        if ((left < 0) | (right >= n) | (left >= right)).any():
            raise ValueError("each edge must satisfy 0 <= left < right < N")
        if (left * n + right).unique().numel() != edges:
            raise ValueError("edge_index contains duplicate edges")

        for name, value in (("Sp", Sp), ("Se", Se), ("omega_scale", omega_scale)):
            value = _frozen(name, value, (n,))
            if (value <= 0).any():
                raise ValueError(f"{name} must be positive")
            self.register_buffer(name, value)
        self.register_buffer("F", _frozen("F", F, (n, n)))
        self.register_buffer("mass", mass)
        self.register_buffer("damping", damping)
        self.register_buffer("theta_ref", theta_ref)
        self.register_buffer("edge_index", edge_index)

        # Derived quantities are reconstructed from the frozen checkpoint inputs.
        incidence = torch.zeros(n, edges, dtype=torch.float64)
        incidence[left, torch.arange(edges)] = 1
        incidence[right, torch.arange(edges)] = -1
        phi_ref = theta_ref[left] - theta_ref[right]
        self.register_buffer("incidence", incidence, persistent=False)
        self.register_buffer("unsigned_incidence", incidence.abs(), persistent=False)
        self.register_buffer("edge_nodes", edge_index.T.flatten(), persistent=False)
        self.register_buffer("phi_ref", phi_ref, persistent=False)
        self.register_buffer("sin_ref", phi_ref.sin(), persistent=False)
        self.register_buffer("cos_ref", phi_ref.cos(), persistent=False)
        self.register_buffer("inverse_mass", torch.cholesky_inverse(torch.linalg.cholesky(mass)),
                             persistent=False)
        self.register_buffer("mechanical_output", self.F * self.Sp, persistent=False)
        bounds = (config.p_init_bound, config.a_init_bound, config.e_init_bound)
        self.register_buffer("init_bounds", torch.tensor(bounds, dtype=torch.float64).repeat_interleave(n),
                             persistent=False)

        self.c = nn.Parameter(torch.full((edges,), config.c_init, dtype=torch.float64))
        beta = torch.full_like(self.c, config.c_init * config.beta_fraction_init)
        if config.variant == "C0":
            self.register_buffer("beta", torch.zeros_like(beta), persistent=False)
        else:
            self.beta = nn.Parameter(beta)
        self.mechanical_coeff = nn.Parameter(torch.zeros(4, n, dtype=torch.float64))
        self.amplitude_coeff = nn.Parameter(torch.zeros(3, n, dtype=torch.float64))
        self.environment_coeff = nn.Parameter(torch.zeros(2, n, dtype=torch.float64))
        self.memory_coeff = nn.Parameter(torch.zeros(5, n, dtype=torch.float64))
        # Each endpoint has independent coefficients for its own incident-edge terms.
        self.edge_coeff = nn.Parameter(torch.zeros(edges, 2, 5, dtype=torch.float64))
        self.encoder = nn.GRU(2 * n, config.hidden_size, batch_first=True, dtype=torch.float64)
        self.initial_head = nn.Linear(config.hidden_size, 3 * n, dtype=torch.float64)

    def initialize(self, theta_history: Tensor, omega_history: Tensor) -> Tensor:
        """Encode only motion histories shaped (..., 11, N)."""
        shape = theta_history.shape
        if shape != omega_history.shape or shape[-2:] != (self.history_points, self.n):
            raise ValueError("theta_history and omega_history must have the same (..., 11, N) shape")
        if any(value.dtype != torch.float64 or value.device != self.mass.device
               for value in (theta_history, omega_history)):
            raise ValueError("motion histories must use float64 on the model device")
        if not (torch.isfinite(theta_history).all() & torch.isfinite(omega_history).all()):
            raise ValueError("motion histories must be finite")
        theta_endpoint, omega_endpoint = theta_history[..., -1, :], omega_history[..., -1, :]
        theta_offset = (theta_endpoint - self.theta_ref).mean(dim=-1, keepdim=True)
        theta_encoder = theta_history - theta_offset.unsqueeze(-2) - self.theta_ref
        features = torch.cat((theta_encoder, omega_history / self.omega_scale), dim=-1)
        _, last = self.encoder(features.reshape(-1, self.history_points, 2 * self.n))
        latent = (self.initial_head(last[0]).tanh() * self.init_bounds).reshape(*shape[:-2], 3 * self.n)
        return torch.cat((theta_endpoint, omega_endpoint, latent), dim=-1)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        """Project trainable edge strengths onto 0 <= beta <= c after an update."""
        self.c.clamp_(min=0)
        if self.config.variant == "C1":
            self.beta.copy_(torch.minimum(self.beta.clamp_min(0), self.c))

    def _edge_kernel(self, a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c = self.c.clamp_min(0)
        if self.config.variant == "C0":
            return c.expand(*a.shape[:-1], c.numel()), c, self.beta
        left, right = self.edge_index
        activation = (a[..., left] + a[..., right]).tanh()
        beta = torch.minimum(self.beta.clamp_min(0), c)
        return c + beta * activation.expm1(), c, beta

    def edge_kernel(self, a: Tensor) -> Tensor:
        """Return positive or zero edge strengths in edge_index order."""
        return self._edge_kernel(a)[0]

    def _parts(self, state: Tensor) -> _Parts:
        if state.ndim == 0 or state.shape[-1] != 5 * self.n:
            raise ValueError("state must have shape (..., 5N)")
        theta, omega, p, a, e = state.split(self.n, dim=-1)
        left, right = self.edge_index
        phase = theta[..., left] - theta[..., right]
        sin_phase, cos_phase = phase.sin(), phase.cos()
        sin_delta, cos_delta = sin_phase - self.sin_ref, cos_phase - self.cos_ref
        tanh_a = a.tanh()
        # Compute the three incident-edge dictionaries together.
        edge_features = torch.stack((sin_delta, cos_delta, sin_delta, cos_delta, cos_delta), dim=-1)
        edge_terms = edge_features.unsqueeze(-2) * self.edge_coeff
        edge_terms = torch.stack(
            (edge_terms[..., 0] + edge_terms[..., 1],
             edge_terms[..., 2] + edge_terms[..., 3], edge_terms[..., 4]), dim=-1
        ).flatten(-3, -2)
        node_terms = state.new_zeros((*state.shape[:-1], self.n, 3)).index_add(
            -2, self.edge_nodes, edge_terms
        )
        a_edges, e_edges, env_edges = node_terms.unbind(dim=-1)
        kernel, c, beta = self._edge_kernel(a)
        coupling = (kernel * sin_phase - c * self.sin_ref) @ self.incidence.T
        environment = self.Se * e + self.environment_coeff[0] * tanh_a
        environment = environment + self.environment_coeff[1] * tanh_a.square() + env_edges
        ports = Ports(p @ self.mechanical_output.T, coupling, environment)
        return _Parts(omega, p, a, e, phase, cos_phase, tanh_a, a_edges, e_edges, kernel, c, beta, ports)

    def _amplitude_derivative(self, parts: _Parts, normalized_omega: Tensor) -> Tensor:
        features = torch.stack((parts.a, normalized_omega, parts.tanh_a), dim=-2)
        return (features * self.amplitude_coeff).sum(dim=-2) + parts.a_edges

    def ports(self, state: Tensor) -> Ports:
        return self._parts(state).ports

    def rhs(self, t, state: Tensor) -> Tensor:
        """Autonomous right-hand side in (..., theta | omega | p | a | e) order."""
        parts = self._parts(state)
        omega, p, a, e, tanh_a, ports = parts.omega, parts.p, parts.a, parts.e, parts.tanh_a, parts.ports
        normalized_omega = omega / self.omega_scale
        p_features = torch.stack((p, omega, p.tanh(), normalized_omega.tanh()), dim=-2)
        e_features = torch.stack((e, a, normalized_omega, e.tanh(), tanh_a), dim=-2)
        p_dot = (p_features * self.mechanical_coeff).sum(dim=-2)
        a_dot = self._amplitude_derivative(parts, normalized_omega)
        e_dot = (e_features * self.memory_coeff).sum(dim=-2) + parts.e_edges
        acceleration = (ports.tau_m - ports.tau_e - omega @ self.damping.T) @ self.inverse_mass.T
        return torch.cat((omega, acceleration, p_dot, a_dot, e_dot), dim=-1)

    forward = rhs

    def storage_power(self, state: Tensor) -> StoragePower:
        parts = self._parts(state)
        omega, a, cos_phase, ports = parts.omega, parts.a, parts.cos_phase, parts.ports
        left, right = self.edge_index
        centered_phase = parts.phase - self.phi_ref
        amplitude = a[..., left] + a[..., right]
        v = parts.kernel * (1 - cos_phase)
        v_ref = parts.c * (1 - self.cos_ref)
        grad_a_ref = parts.beta * (1 - self.cos_ref)
        potential = (v - v_ref - parts.c * self.sin_ref * centered_phase - grad_a_ref * amplitude).sum(dim=-1)
        activation = amplitude.tanh()
        dk_da = parts.beta * activation.exp() * (1 - activation.square())
        grad_a = (dk_da * (1 - cos_phase) - grad_a_ref) @ self.unsigned_incidence.T
        a_dot = self._amplitude_derivative(parts, omega / self.omega_scale)
        kinetic = 0.5 * (omega * (omega @ self.mass)).sum(dim=-1)
        return StoragePower(
            kinetic + potential,
            potential,
            (omega * ports.tau_m).sum(dim=-1),
            -(omega * ports.tau_env).sum(dim=-1),
            -(omega * (omega @ self.damping)).sum(dim=-1),
            (grad_a * a_dot).sum(dim=-1),
        )

    def rk4_step(self, state: Tensor, step: float) -> Tensor:
        k1 = self.rhs(None, state)
        k2 = self.rhs(None, state + (step / 2) * k1)
        k3 = self.rhs(None, state + (step / 2) * k2)
        k4 = self.rhs(None, state + step * k3)
        return state + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, initial_state: Tensor, times, *, max_step: float = 0.01) -> Tensor:
        """Integrate autonomously; nonfinite outcomes remain visible to the caller."""
        if initial_state.ndim == 0 or initial_state.shape[-1] != 5 * self.n:
            raise ValueError("initial_state must have shape (..., 5N)")
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
        state = initial_state
        trajectory = [state]
        for interval in intervals:
            # Decimal sample grids can exceed max_step by machine roundoff.
            substeps = max(1, ceil(interval / max_step - 1e-12))
            for _ in range(substeps):
                state = self.rk4_step(state, interval / substeps)
            trajectory.append(state)
        return torch.stack(trajectory, dim=-2)

    def checkpoint(self) -> dict:
        """Export a detached CPU snapshot with JSON-safe configuration."""
        return {
            "contract": self.contract,
            "version": self.checkpoint_version,
            "config": asdict(self.config),
            "state_dict": {name: value.detach().cpu().clone() for name, value in self.state_dict().items()},
        }

    @classmethod
    def from_checkpoint(cls, payload: dict) -> OSGKD:
        if payload.get("contract") != cls.contract or payload.get("version") != cls.checkpoint_version:
            raise ValueError("unsupported OSGKD checkpoint contract or version")
        state = payload["state_dict"]
        static_names = ("mass", "damping", "F", "Sp", "Se", "omega_scale", "theta_ref", "edge_index")
        model = cls(OSGKDConfig(**payload["config"]), **{name: state[name] for name in static_names})
        model.load_state_dict(state)
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError("checkpoint parameters must be finite")
        return model
