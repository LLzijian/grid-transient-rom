from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .graph import complete_incidence


EXP_MODULATION_KERNEL = "exp_v1"
BOUNDED_MODULATION_KERNEL = "bounded_exp_tanh_v1"


def amplitude_modulation_factor(exponent: np.ndarray, kernel: str) -> tuple[np.ndarray, np.ndarray | float]:
    """Return K(h) and the multiplier K'(h)/K(h) for the specified kernel."""
    if kernel == EXP_MODULATION_KERNEL:
        return np.exp(exponent), 1.0
    if kernel == BOUNDED_MODULATION_KERNEL:
        bounded = np.tanh(exponent)
        return np.exp(bounded), 1.0 - bounded**2
    raise ValueError("unknown amplitude modulation kernel")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def coi_sync_projectors(mass: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inertia-weighted COI and differential-synchronization projectors.

    For state/velocity vectors ``v``, ``P_coi @ v = bar_v * 1`` where
    ``bar_v = (1^T M v) / (1^T M 1)``, and ``P_sync = I - P_coi``.
    They satisfy ``P_coi^2 = P_coi``, ``P_sync^2 = P_sync``,
    ``P_coi @ P_sync = 0``, and ``P_coi + P_sync = I``.
    """
    m = np.asarray(mass, dtype=np.float64).reshape(-1)
    if m.ndim != 1 or m.size < 2 or np.any(m <= 0.0):
        raise ValueError("mass must be a 1D array of at least 2 positive entries")
    total_m = float(np.sum(m))
    ones = np.ones_like(m)
    p_coi = np.outer(ones, m) / total_m
    p_sync = np.eye(m.size, dtype=np.float64) - p_coi
    coi_weights = m / total_m
    return p_coi, p_sync, coi_weights


def bregman_kuramoto_potential(phi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Bregman-centered Kuramoto edge potential psi(phi; eta).

    ``psi(phi; eta) = cos(eta) - cos(phi) - sin(eta) * (phi - eta)``.
    Vanishes to first order at the reference phase difference ``phi = eta``
    and has local curvature ``cos(eta) > 0`` for ``|eta| < pi/2``.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    eta_arr = np.asarray(eta, dtype=np.float64)
    return np.cos(eta_arr) - np.cos(phi_arr) - np.sin(eta_arr) * (phi_arr - eta_arr)


def bregman_kuramoto_force(phi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Derivative d/d(phi) psi(phi; eta) = sin(phi) - sin(eta)."""
    phi_arr = np.asarray(phi, dtype=np.float64)
    eta_arr = np.asarray(eta, dtype=np.float64)
    return np.sin(phi_arr) - np.sin(eta_arr)


@dataclass(frozen=True)
class StructuralGates:
    """Top-down structural gates S = (S_edge, S_mod, S_mem, s_coi).

    Enforces the hierarchical structural constraint:
        0 <= s_mod[e] <= s_edge[e] <= 1
    so that a nonexistent edge (s_edge[e] == 0) cannot carry latent amplitude
    modulation.
    """

    s_edge: np.ndarray
    s_mod: np.ndarray
    s_mem: np.ndarray
    s_coi: float = 0.0

    def __post_init__(self) -> None:
        edge = np.asarray(self.s_edge, dtype=np.float64).reshape(-1)
        mod = np.asarray(self.s_mod, dtype=np.float64).reshape(-1)
        mem = np.asarray(self.s_mem, dtype=np.float64).reshape(-1)
        if not all(np.isfinite(value).all() for value in (edge, mod, mem)) or not np.isfinite(self.s_coi):
            raise ValueError("structural gates must be finite")
        if edge.shape != mod.shape:
            raise ValueError("s_edge and s_mod must have identical shapes")
        if np.any(edge < 0.0) or np.any(edge > 1.0):
            raise ValueError("s_edge entries must lie in [0, 1]")
        if np.any(mod < 0.0) or np.any(mod > 1.0):
            raise ValueError("s_mod entries must lie in [0, 1]")
        # Enforce hierarchical gate dominance: effective modulation is bounded by s_edge
        effective_mod = np.minimum(edge, mod)
        object.__setattr__(self, "s_edge", edge)
        object.__setattr__(self, "s_mod", effective_mod)
        object.__setattr__(self, "s_mem", np.clip(mem, 0.0, 1.0))
        object.__setattr__(self, "s_coi", float(np.clip(self.s_coi, 0.0, 1.0)))

    @classmethod
    def overcomplete(cls, n_edges: int, n_nodes: int) -> StructuralGates:
        return cls(
            s_edge=np.ones(n_edges, dtype=np.float64),
            s_mod=np.ones(n_edges, dtype=np.float64),
            s_mem=np.ones(n_nodes, dtype=np.float64),
            s_coi=0.0,
        )

    @classmethod
    def static_baseline_limit(cls, edge_mask: np.ndarray, n_nodes: int) -> StructuralGates:
        """Exact-Zero embedding limit reducing to static topology without latent modulation."""
        mask = np.asarray(edge_mask, dtype=np.float64).reshape(-1)
        return cls(
            s_edge=np.clip(mask, 0.0, 1.0),
            s_mod=np.zeros_like(mask),
            s_mem=np.ones(n_nodes, dtype=np.float64),
            s_coi=0.0,
        )


def latent_coupling_modulation(
    z_amp: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    gamma: np.ndarray,
    gates: StructuralGates,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive latent-amplitude coupling kernel Phi_e(z_i^amp, z_j^amp; S_e^mod).

    ``Phi_e = 1 + S_e^mod * (exp(gamma_i * z_i^amp + gamma_j * z_j^amp) - 1) > 0``.
    Returns ``(phi_mod, dphi_dzi, dphi_dzj)`` of shape ``(n_edges,)``.
    When ``S_e^mod == 0`` or ``gamma == 0``, ``Phi_e == 1`` and both partials vanish.
    """
    z_a = np.asarray(z_amp, dtype=np.float64).reshape(-1)
    gam = np.asarray(gamma, dtype=np.float64).reshape(-1)
    if z_a.shape != gam.shape:
        raise ValueError("z_amp and gamma must have the same node dimension")
    idx_i = np.array([e[0] for e in edges], dtype=np.intp)
    idx_j = np.array([e[1] for e in edges], dtype=np.intp)
    if not np.all(np.isfinite(z_a)) or not np.all(np.isfinite(gam)):
        raise ValueError("amplitude states and gamma must be finite")
    # Clipping the exponent changes H without preserving its declared derivatives.
    with np.errstate(over="raise", under="raise", invalid="raise"):
        exponent = gam[idx_i] * z_a[idx_i] + gam[idx_j] * z_a[idx_j]
        exp_term, _ = amplitude_modulation_factor(exponent, EXP_MODULATION_KERNEL)
    s_mod = gates.s_mod
    phi_mod = (1.0 - s_mod) + s_mod * exp_term
    dphi_dzi = s_mod * gam[idx_i] * exp_term
    dphi_dzj = s_mod * gam[idx_j] * exp_term
    return phi_mod, dphi_dzi, dphi_dzj


@dataclass
class GeneratingFunctionalKuramoto:
    """5N autonomous state (absolute theta, omega, p, amplitude, force memory).

    Memory dissipation is diagonal. COI projections are report-only.
    Defaults for the p port are declared artificial constants, not IEEE39 fits.
    """

    mass: np.ndarray
    damping: np.ndarray
    k_edge: np.ndarray
    eta_edge: np.ndarray
    gamma_amp: np.ndarray
    lambda_amp: np.ndarray
    lambda_mem: np.ndarray
    tau_amp: np.ndarray
    tau_mem_sync: np.ndarray
    tau_mem_coi: float
    j_omega_mem: np.ndarray
    gates: StructuralGates
    gate_weight_amp: np.ndarray | None = None
    gate_bias_amp: np.ndarray | None = None
    gate_weight_mem: np.ndarray | None = None
    gate_bias_mem: np.ndarray | None = None
    mem_laplacian_weights: np.ndarray | None = None
    storage_p: np.ndarray | None = None
    gain_p: np.ndarray | None = None
    rate_p: np.ndarray | None = None
    modulation_kernel: str = EXP_MODULATION_KERNEL

    def __post_init__(self) -> None:
        if not isinstance(self.modulation_kernel, str) or self.modulation_kernel not in (
            EXP_MODULATION_KERNEL, BOUNDED_MODULATION_KERNEL,
        ):
            raise ValueError("unknown amplitude modulation kernel")
        if self.modulation_kernel == BOUNDED_MODULATION_KERNEL and not np.isfinite(self.tau_mem_coi):
            raise ValueError("tau_mem_coi must be finite")
        self.mass = np.asarray(self.mass, dtype=np.float64).reshape(-1).copy()
        self.n_nodes = self.mass.size
        self.incidence, self.edges = complete_incidence(self.n_nodes)
        self.n_edges = len(self.edges)
        node_names = (
            "mass", "damping", "gamma_amp", "lambda_amp", "lambda_mem",
            "tau_amp", "tau_mem_sync", "j_omega_mem", "storage_p", "gain_p", "rate_p",
        )
        for name in node_names:
            value = getattr(self, name)
            if value is None:
                value = np.ones(self.n_nodes)
            value = np.asarray(value, dtype=np.float64).reshape(self.n_nodes).copy()
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in ("k_edge", "eta_edge"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(self.n_edges).copy()
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in ("mass", "lambda_amp", "lambda_mem", "tau_amp", "tau_mem_sync", "storage_p"):
            if np.any(getattr(self, name) <= 0):
                raise ValueError(f"{name} must be positive")
        for name in ("damping", "k_edge", "rate_p"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} must be nonnegative")
        if self.gates.s_edge.shape != (self.n_edges,) or self.gates.s_mem.shape != (self.n_nodes,):
            raise ValueError("structural gate dimensions must match the functional")
        if self.gates.s_coi != 0:
            raise ValueError("COI dynamical gate is retired in the 5N format")
        if self.mem_laplacian_weights is not None:
            raise ValueError("memory Laplacian requires a separately qualified force operator")
        for name in ("gate_weight_amp", "gate_weight_mem", "gate_bias_amp", "gate_bias_mem"):
            value = getattr(self, name)
            if value is not None:
                shape = (self.n_nodes, 3) if "weight" in name else (self.n_nodes,)
                value = np.asarray(value, dtype=np.float64).reshape(shape).copy()
                if not np.isfinite(value).all():
                    raise ValueError(f"{name} must be finite")
                setattr(self, name, value)
        self.p_coi, self.p_sync, self.coi_weights = coi_sync_projectors(self.mass)

    def split_state(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
        arr = np.asarray(state, dtype=np.float64)
        if arr.shape != (5 * self.n_nodes,) or not np.isfinite(arr).all():
            raise ValueError(f"finite state of shape ({5 * self.n_nodes},) required")
        return tuple(np.split(arr, 5))

    def pack_state(self, theta, omega, p, z_amp, z_mem) -> np.ndarray:
        state = np.concatenate([
            np.asarray(value, dtype=np.float64).reshape(self.n_nodes)
            for value in (theta, omega, p, z_amp, z_mem)
        ])
        self.split_state(state)
        return state

    def _bounded_edge_modulation(self, z_amp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = np.asarray(z_amp, dtype=np.float64)
        if a.shape != (self.n_nodes,) or not np.isfinite(a).all():
            raise ValueError("finite node-aligned amplitude required")
        exponent = np.abs(self.incidence).T @ (self.gamma_amp * a)
        factor, slope_multiplier = amplitude_modulation_factor(exponent, self.modulation_kernel)
        s_mod = self.gates.s_mod
        return (1.0 - s_mod) + s_mod * factor, s_mod * factor * slope_multiplier

    def effective_edge_couplings(self, z_amp: np.ndarray) -> np.ndarray:
        if self.modulation_kernel == EXP_MODULATION_KERNEL:
            phi_mod, _, _ = latent_coupling_modulation(z_amp, self.edges, self.gamma_amp, self.gates)
        else:
            phi_mod, _ = self._bounded_edge_modulation(z_amp)
        return self.gates.s_edge * self.k_edge * phi_mod

    def hamiltonian(self, state: np.ndarray) -> float:
        theta, omega, p, z_amp, z_mem = self.split_state(state)
        psi = bregman_kuramoto_potential(self.incidence.T @ theta, self.eta_edge)
        return float(
            .5 * np.sum(self.mass * omega**2)
            + .5 * np.sum(self.storage_p * p**2)
            + np.sum(self.effective_edge_couplings(z_amp) * psi)
            + .5 * np.sum(self.lambda_amp * z_amp**2)
            + .5 * np.sum(self.lambda_mem * z_mem**2)
        )

    def gradients(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
        theta, omega, p, z_amp, z_mem = self.split_state(state)
        phi = self.incidence.T @ theta
        psi = bregman_kuramoto_potential(phi, self.eta_edge)
        legacy = self.modulation_kernel == EXP_MODULATION_KERNEL
        if legacy:
            mod, _, _ = latent_coupling_modulation(z_amp, self.edges, self.gamma_amp, self.gates)
        else:
            mod, slope = self._bounded_edge_modulation(z_amp)
        active_k = self.gates.s_edge * self.k_edge
        absolute_incidence = np.abs(self.incidence)
        if legacy:
            with np.errstate(over="raise", under="raise", invalid="raise"):
                exp_h, _ = amplitude_modulation_factor(
                    absolute_incidence.T @ (self.gamma_amp * z_amp), EXP_MODULATION_KERNEL,
                )
            grad_amp = self.lambda_amp * z_amp + self.gamma_amp * (
                absolute_incidence @ (active_k * self.gates.s_mod * exp_h * psi))
        else:
            grad_amp = self.lambda_amp * z_amp + self.gamma_amp * (absolute_incidence @ (active_k * slope * psi))
        return (
            self.incidence @ (active_k * mod * bregman_kuramoto_force(phi, self.eta_edge)),
            self.mass * omega, self.storage_p * p, grad_amp, self.lambda_mem * z_mem,
        )

    def dissipative_operators(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta, omega, _, z_amp, z_mem = self.split_state(state)
        if self.gate_weight_amp is None and self.gate_weight_mem is None:
            return 1. / self.tau_amp, np.diag(1. / self.tau_mem_sync)
        omega_diff = omega - self.coi_weights @ omega
        node_sin_sq = np.abs(self.incidence) @ np.sin(.5 * (self.incidence.T @ theta))**2
        gate_a = np.ones(self.n_nodes)
        gate_m = np.ones(self.n_nodes)
        if self.gate_weight_amp is not None:
            bias = 0 if self.gate_bias_amp is None else self.gate_bias_amp
            features = np.stack([omega_diff, node_sin_sq, z_amp], axis=-1)
            gate_a = _sigmoid(np.sum(self.gate_weight_amp * features, axis=-1) + bias)
        if self.gate_weight_mem is not None:
            bias = 0 if self.gate_bias_mem is None else self.gate_bias_mem
            features = np.stack([omega_diff, node_sin_sq, z_mem], axis=-1)
            gate_m = _sigmoid(np.sum(self.gate_weight_mem * features, axis=-1) + bias)
        return gate_a / self.tau_amp, np.diag(gate_m / self.tau_mem_sync)

    def rhs(self, state: np.ndarray) -> np.ndarray:
        gt, gw, gp, ga, gr = self.gradients(state)
        ra, rm = self.dissipative_operators(state)
        omega = gw / self.mass
        j = self.gates.s_mem * self.j_omega_mem
        return self.pack_state(
            omega, (-gt - self.damping * omega + self.gain_p * gp - j * gr) / self.mass,
            -self.gain_p * omega - self.rate_p * gp, -ra * ga, j * omega - rm @ gr,
        )

    def power_balance_audit(self, state: np.ndarray) -> dict[str, float | bool]:
        gt, gw, gp, ga, gr = self.gradients(state)
        ra, rm = self.dissipative_operators(state)
        power = float(self.pack_state(gt, gw, gp, ga, gr) @ self.rhs(state))
        mech = float(np.sum(self.damping * (gw / self.mass)**2))
        drive = float(np.sum(self.rate_p * gp**2))
        amp = float(np.sum(ra * ga**2))
        mem = float(gr @ rm @ gr)
        diss = mech + drive + amp + mem
        residual = abs(power + diss)
        tolerance = 1e-11 * max(1., abs(power), diss)
        return dict(dH_dt_chain=power, expected_dH_dt=-diss, diss_mech=mech,
                    diss_drive=drive, diss_amp=amp, diss_mem=mem,
                    power_leakage_abs=residual,
                    strictly_dissipative=bool(np.isfinite(power) and power <= tolerance and residual <= tolerance))

    def canonical_gauge_transform(self, state: np.ndarray):
        """Diagonal latent rescaling, preserving the same observable dynamics."""
        from dataclasses import fields
        theta, omega, p, amp, mem = self.split_state(state)
        sa, sm = np.sqrt(self.lambda_amp), np.sqrt(self.lambda_mem)
        kwargs = {field.name: getattr(self, field.name) for field in fields(self)}
        kwargs.update(gamma_amp=self.gamma_amp / sa, lambda_amp=np.ones(self.n_nodes),
                      lambda_mem=np.ones(self.n_nodes), tau_amp=self.tau_amp / self.lambda_amp,
                      tau_mem_sync=self.tau_mem_sync / self.lambda_mem,
                      j_omega_mem=self.j_omega_mem * sm)
        for name, scale in (("gate_weight_amp", sa), ("gate_weight_mem", sm)):
            weight = kwargs[name]
            if weight is not None:
                weight = weight.copy()
                weight[:, 2] /= scale
                kwargs[name] = weight
        transformed = type(self)(**kwargs)
        return transformed, transformed.pack_state(theta, omega, p, sa * amp, sm * mem)
