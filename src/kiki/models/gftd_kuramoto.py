from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..numerics import dop853_step
from ..physics.graph import complete_incidence
from ..physics.kuramoto_functional import (
    BOUNDED_MODULATION_KERNEL,
    EXP_MODULATION_KERNEL,
    GeneratingFunctionalKuramoto,
    StructuralGates,
    bregman_kuramoto_force,
    coi_sync_projectors,
)
from .base import DynamicsModel, FitResult, RunContext
from .gftd_fitting import FIXED_DP8_FORMAT


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class SO2InvariantHistoryEncoder:
    """Causal Graph-GRU history encoder I_phi(H_0) invariant under theta -> theta + c*1.

    Uses only relative edge phase differences ``phi_e(t_k) = theta_i(t_k) - theta_j(t_k)``,
    COI frequency ``omega_coi(t_k)``, and differential speeds ``omega(t_k) - omega_coi(t_k)``
    over the 100 ms causal observation window ``t_0 ... t_K``.
    """

    n_nodes: int
    mass: np.ndarray
    w_update_amp: np.ndarray
    w_reset_amp: np.ndarray
    w_cand_amp: np.ndarray
    w_update_mem: np.ndarray
    w_reset_mem: np.ndarray
    w_cand_mem: np.ndarray

    def __init__(self, n_nodes: int, mass: np.ndarray, seed: int = 324) -> None:
        self.n_nodes = int(n_nodes)
        self.mass = np.asarray(mass, dtype=np.float64).reshape(self.n_nodes)
        self.incidence, self.edges = complete_incidence(self.n_nodes)
        _, _, self.coi_weights = coi_sync_projectors(self.mass)
        rng = np.random.default_rng(seed)
        # 4 node-wise SO(2)-invariant features + 1 recurrent state -> 5 weights per gate
        scale = 0.25
        self.w_update_amp = rng.normal(0.0, scale, size=(self.n_nodes, 5))
        self.w_reset_amp = rng.normal(0.0, scale, size=(self.n_nodes, 5))
        self.w_cand_amp = rng.normal(0.0, scale, size=(self.n_nodes, 5))
        self.w_update_mem = rng.normal(0.0, scale, size=(self.n_nodes, 5))
        self.w_reset_mem = rng.normal(0.0, scale, size=(self.n_nodes, 5))
        self.w_cand_mem = rng.normal(0.0, scale, size=(self.n_nodes, 5))

    def invariant_node_features(
        self, theta_step: np.ndarray, omega_step: np.ndarray
    ) -> np.ndarray:
        """Compute 4 node-wise features strictly invariant to global angle shifts."""
        theta_arr = np.asarray(theta_step, dtype=np.float64).reshape(self.n_nodes)
        omega_arr = np.asarray(omega_step, dtype=np.float64).reshape(self.n_nodes)
        phi = self.incidence.T @ theta_arr
        omega_coi = float(np.sum(self.coi_weights * omega_arr))
        omega_diff = omega_arr - omega_coi
        abs_inc = np.abs(self.incidence)
        degree = np.maximum(np.sum(abs_inc, axis=1), 1.0)
        mean_sin_sq = (abs_inc @ np.square(np.sin(0.5 * phi))) / degree
        net_sin = (self.incidence @ np.sin(phi)) / degree
        coi_vec = np.full(self.n_nodes, omega_coi, dtype=np.float64)
        return np.stack([coi_vec, omega_diff, mean_sin_sq, net_sin], axis=-1)

    def encode(
        self,
        theta_history: np.ndarray,
        omega_history: np.ndarray,
        p0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Causally encode history ``(K, N)`` into initial latent states ``(z_amp0, z_mem0)``."""
        theta_hist = np.asarray(theta_history, dtype=np.float64)
        omega_hist = np.asarray(omega_history, dtype=np.float64)
        if theta_hist.ndim != 2 or theta_hist.shape != omega_hist.shape:
            raise ValueError("theta_history and omega_history must have matching 2D shape (K, N)")
        if theta_hist.shape != (11, self.n_nodes):
            raise ValueError("encoder requires exactly eleven node-aligned history points")
        if not np.isfinite(theta_hist).all() or not np.isfinite(omega_hist).all():
            raise ValueError("history must be finite")
        if p0 is not None and not np.isfinite(p0).all():
            raise ValueError("p0 must be finite")

        z_amp = np.zeros(self.n_nodes, dtype=np.float64)
        z_mem = (
            np.zeros(self.n_nodes, dtype=np.float64)
            if p0 is None
            else 0.1 * np.asarray(p0, dtype=np.float64).reshape(self.n_nodes)
        )

        for step in range(theta_hist.shape[0]):
            feat = self.invariant_node_features(theta_hist[step], omega_hist[step])
            inp_a = np.column_stack([feat, z_amp])
            u_a = _sigmoid(np.sum(self.w_update_amp * inp_a, axis=-1))
            r_a = _sigmoid(np.sum(self.w_reset_amp * inp_a, axis=-1))
            cand_inp_a = np.column_stack([feat, r_a * z_amp])
            h_a = np.tanh(np.sum(self.w_cand_amp * cand_inp_a, axis=-1))
            z_amp = (1.0 - u_a) * z_amp + u_a * h_a

            inp_m = np.column_stack([feat, z_mem])
            u_m = _sigmoid(np.sum(self.w_update_mem * inp_m, axis=-1))
            r_m = _sigmoid(np.sum(self.w_reset_mem * inp_m, axis=-1))
            cand_inp_m = np.column_stack([feat, r_m * z_mem])
            h_m = np.tanh(np.sum(self.w_cand_mem * cand_inp_m, axis=-1))
            z_mem = (1.0 - u_m) * z_mem + u_m * h_m

        return z_amp, z_mem


@dataclass(frozen=True)
class WeakDiscoveryResult:
    """Result of top-down weak-form variational structure discovery."""

    s_edge: np.ndarray
    s_mod: np.ndarray
    k_edge: np.ndarray
    mod_coeff: np.ndarray
    weak_residual_norm: float
    gram_condition_number: float


class WeakFormVariationalProjector:
    """Conditional linear weak dictionary, assuming supplied latent proxies and zero omitted ports.

    Uses compactly supported C^1 polynomial test functions
        phi_m(tau) = (1 - tau^2)^2   on tau in [-1, 1]
    to integrate the second-order momentum balance by parts:
        - int_{t_a}^{t_b} M_i * omega_i(t) * d/dt phi_m(t) dt
        - int_{t_a}^{t_b} D_i * omega_i(t) * phi_m(t) dt
        = int_{t_a}^{t_b} F_i(theta(t), z(t)) * phi_m(t) dt
    avoiding numerical differentiation of noisy frequency observations.
    """

    def __init__(self, n_nodes: int, mass: np.ndarray, damping: np.ndarray) -> None:
        self.n_nodes = int(n_nodes)
        self.mass = np.asarray(mass, dtype=np.float64).reshape(self.n_nodes)
        self.damping = np.asarray(damping, dtype=np.float64).reshape(self.n_nodes)
        self.incidence, self.edges = complete_incidence(self.n_nodes)
        self.n_edges = len(self.edges)

    @staticmethod
    def test_function_weights(t_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return quadrature weights for phi(t) and dphi/dt on a uniform/nonuniform grid."""
        t = np.asarray(t_window, dtype=np.float64).reshape(-1)
        if t.size < 3:
            raise ValueError("Weak-form window requires at least 3 time samples")
        t_mid = 0.5 * (t[0] + t[-1])
        half_len = 0.5 * (t[-1] - t[0])
        if half_len <= 0.0:
            raise ValueError("Time window must have positive duration")
        tau = (t - t_mid) / half_len
        phi = np.square(1.0 - np.square(tau))
        dphi_dt = (-4.0 * tau * (1.0 - np.square(tau))) / half_len

        # Use 4th-order composite Simpson quadrature on uniform odd-length grids,
        # falling back to trapezoidal weights on irregular grids.
        dt_steps = np.diff(t)
        if t.size >= 3 and (t.size % 2 == 1) and np.allclose(dt_steps, dt_steps[0], rtol=1e-12, atol=1e-14):
            h = float(dt_steps[0])
            quad = np.empty_like(t)
            quad[0] = 1.0
            quad[-1] = 1.0
            quad[1:-1:2] = 4.0
            quad[2:-1:2] = 2.0
            quad *= h / 3.0
        else:
            quad = np.zeros_like(t)
            quad[0] = 0.5 * (t[1] - t[0])
            quad[-1] = 0.5 * (t[-1] - t[-2])
            quad[1:-1] = 0.5 * (t[2:] - t[:-2])
        return quad * phi, quad * dphi_dt

    def momentum_lhs_integral(
        self, t_window: np.ndarray, omega_window: np.ndarray
    ) -> np.ndarray:
        """Compute weak-form momentum LHS ``int (M domega/dt + D omega) phi(t) dt`` by parts."""
        w_phi, w_dphi = self.test_function_weights(t_window)
        omega_w = np.asarray(omega_window, dtype=np.float64)
        # Since phi(t_a) = phi(t_b) = 0, int M domega/dt phi dt = - int M omega dphi/dt dt
        inertial_term = -self.mass * (w_dphi @ omega_w)
        damping_term = self.damping * (w_phi @ omega_w)
        return inertial_term + damping_term

    def build_weak_dictionary(
        self,
        t_grid: np.ndarray,
        theta_traj: np.ndarray,
        omega_traj: np.ndarray,
        z_amp_proxy: np.ndarray,
        eta_edge: np.ndarray | None = None,
        window_size: int = 11,
        stride: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble weak-form design matrix ``[Psi_static, Psi_mod]`` and target ``b_weak``."""
        t = np.asarray(t_grid, dtype=np.float64).reshape(-1)
        theta = np.asarray(theta_traj, dtype=np.float64)
        omega = np.asarray(omega_traj, dtype=np.float64)
        z_amp = np.asarray(z_amp_proxy, dtype=np.float64)
        eta = (
            np.zeros(self.n_edges, dtype=np.float64)
            if eta_edge is None
            else np.asarray(eta_edge, dtype=np.float64).reshape(self.n_edges)
        )

        idx_i = np.array([e[0] for e in self.edges], dtype=np.intp)
        idx_j = np.array([e[1] for e in self.edges], dtype=np.intp)
        rows_static: list[np.ndarray] = []
        rows_mod: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for start in range(0, t.size - window_size + 1, stride):
            end = start + window_size
            t_win = t[start:end]
            w_phi, _ = self.test_function_weights(t_win)
            lhs = self.momentum_lhs_integral(t_win, omega[start:end])

            phi_win = theta[start:end] @ self.incidence
            force_win = bregman_kuramoto_force(phi_win, eta[None, :])
            mod_factor_win = z_amp[start:end, idx_i] + z_amp[start:end, idx_j]

            int_static_edge = w_phi @ force_win
            int_mod_edge = w_phi @ (mod_factor_win * force_win)

            # Force in M domega/dt is -B @ edge_force
            psi_static_win = -self.incidence * int_static_edge[None, :]
            psi_mod_win = -self.incidence * int_mod_edge[None, :]

            rows_static.append(psi_static_win)
            rows_mod.append(psi_mod_win)
            targets.append(lhs)

        if not rows_static:
            raise ValueError("Trajectory shorter than window_size")
        design_static = np.vstack(rows_static)
        design_mod = np.vstack(rows_mod)
        design = np.hstack([design_static, design_mod])
        b_vec = np.concatenate(targets)
        return design, b_vec

    def discover_hierarchical_structure(
        self,
        design: np.ndarray,
        b_vec: np.ndarray,
        edge_threshold: float = 0.25,
        mod_threshold: float = 0.15,
        ridge: float = 1e-6,
    ) -> WeakDiscoveryResult:
        """Top-down two-stage hierarchical collapse enforcing 0 <= S_mod <= S_edge <= 1."""
        m = self.n_edges
        if design.shape[1] != 2 * m:
            raise ValueError(f"Expected {2 * m} dictionary columns, got {design.shape[1]}")

        gram = design.T @ design + ridge * np.eye(2 * m, dtype=np.float64)
        rhs = design.T @ b_vec
        coeff = np.linalg.solve(gram, rhs)

        k_raw = np.maximum(coeff[:m], 0.0)
        mod_raw = coeff[m:]

        # Stage B1: top-down topology gate collapse S_edge
        s_edge = (k_raw >= edge_threshold).astype(np.float64)
        if not np.any(s_edge > 0.0) and np.max(k_raw) > 0.0:
            s_edge[int(np.argmax(k_raw))] = 1.0

        # Stage B2: hierarchical modulation gate collapse S_mod <= S_edge
        rel_mod = np.abs(mod_raw) / np.maximum(k_raw, 1e-8)
        s_mod = ((s_edge > 0.0) & (rel_mod >= mod_threshold)).astype(np.float64)

        # Stage C: debiased refit on surviving support
        active_cols = np.concatenate([s_edge > 0.0, s_mod > 0.0])
        debiased = np.zeros(2 * m, dtype=np.float64)
        if np.any(active_cols):
            sub_design = design[:, active_cols]
            sub_gram = sub_design.T @ sub_design + ridge * np.eye(
                int(np.sum(active_cols)), dtype=np.float64
            )
            debiased[active_cols] = np.linalg.solve(sub_gram, sub_design.T @ b_vec)
            svals = np.linalg.svd(sub_design, compute_uv=False)
            cond = float(svals[0] / max(svals[-1], 1e-12)) if svals.size > 0 else 1.0
        else:
            cond = 1.0

        k_final = np.maximum(debiased[:m], 0.0) * s_edge
        mod_final = debiased[m:] * s_mod
        residual_norm = float(
            np.linalg.norm(design @ np.concatenate([k_final, mod_final]) - b_vec)
            / max(np.linalg.norm(b_vec), 1e-12)
        )
        return WeakDiscoveryResult(
            s_edge=s_edge,
            s_mod=s_mod,
            k_edge=k_final,
            mod_coeff=mod_final,
            weak_residual_norm=residual_norm,
            gram_condition_number=cond,
        )


class GFTDKuramotoModel(DynamicsModel):
    """Top-Down Generating-Functional Discovery model for partially observed Kuramoto networks."""

    method_name = "gftd_kuramoto_v2"

    def __init__(
        self,
        n_nodes: int = 10,
        mass: np.ndarray | None = None,
        damping: np.ndarray | None = None,
        seed: int = 324,
        fit_mode: str = "modulated",
        *,
        modulation_kernel: str = EXP_MODULATION_KERNEL,
    ) -> None:
        if fit_mode not in {"static", "modulated"}:
            raise ValueError("fit_mode must be static or modulated")
        self.fit_mode = fit_mode
        self.initialization_mode = "legacy_static"
        self.inference_config = None
        self.fit_state = None
        self.fit_private_details = None
        self.last_initialization = None
        self.n_nodes = int(n_nodes)
        self.mass = (
            np.ones(self.n_nodes, dtype=np.float64)
            if mass is None
            else np.asarray(mass, dtype=np.float64).reshape(self.n_nodes)
        )
        self.damping = (
            np.full(self.n_nodes, 0.2, dtype=np.float64)
            if damping is None
            else np.asarray(damping, dtype=np.float64).reshape(self.n_nodes)
        )
        self.seed = int(seed)
        self.incidence, self.edges = complete_incidence(self.n_nodes)
        self.n_edges = len(self.edges)

        self.encoder = SO2InvariantHistoryEncoder(self.n_nodes, self.mass, seed=self.seed)
        self.gates = StructuralGates.static_baseline_limit(np.ones(self.n_edges), self.n_nodes)
        self.delta_star = np.zeros(self.n_nodes)
        self.parent_initializer = None
        self.functional = GeneratingFunctionalKuramoto(
            mass=self.mass,
            damping=self.damping,
            k_edge=np.ones(self.n_edges, dtype=np.float64),
            eta_edge=np.zeros(self.n_edges, dtype=np.float64),
            gamma_amp=np.full(self.n_nodes, 0.3, dtype=np.float64),
            lambda_amp=np.ones(self.n_nodes, dtype=np.float64),
            lambda_mem=np.ones(self.n_nodes, dtype=np.float64),
            tau_amp=np.full(self.n_nodes, 0.5, dtype=np.float64),
            tau_mem_sync=np.full(self.n_nodes, 0.5, dtype=np.float64),
            tau_mem_coi=0.4,
            j_omega_mem=np.full(self.n_nodes, 0.2, dtype=np.float64),
            gates=self.gates,
            modulation_kernel=modulation_kernel,
        )

    def fit(
        self, train: list[dict[str, Any]], val: list[dict[str, Any]], run_context: RunContext
    ) -> FitResult:
        if val:
            raise ValueError("val is closed for GFTD fitting")
        if not train:
            raise ValueError("fit requires a nonempty observation batch")
        from . import gftd_fitting as engine
        from .gftd_discovery import validate_observation
        packets = []
        for sample in train:
            if set(sample) != {"theta_history", "omega_history", "history_times", "p0", "sigma"}:
                raise ValueError("fit accepts exactly history, p0 and sigma")
            packet = {("p_end" if key == "p0" else key): copy.deepcopy(value)
                      for key, value in sample.items()}
            packets.append(validate_observation(packet, self.n_nodes))
        if any(packet["sigma"] != packets[0]["sigma"] for packet in packets):
            raise ValueError("all observations must have the same sigma")
        if run_context.gftd_fit297_v1 is not None and run_context.gftd_matched_prediction_v1 is not None:
            raise ValueError("GFTD fit requests are mutually exclusive")
        if run_context.gftd_matched_prediction_v1 is not None:
            from . import gftd_matched_prediction
            result = gftd_matched_prediction.fit_shared(copy.deepcopy(self.functional), packets,
                mode=self.fit_mode, config=copy.deepcopy(run_context.gftd_matched_prediction_v1))
        elif run_context.gftd_fit297_v1 is None:
            result = engine.fit_shared(copy.deepcopy(self.functional), packets, mode=self.fit_mode)
        else:
            from . import gftd_fit297
            result = gftd_fit297.fit_shared(copy.deepcopy(self.functional), packets,
                mode=self.fit_mode, config=copy.deepcopy(run_context.gftd_fit297_v1))
        functional = result["functional"]
        functional.__post_init__()
        if functional.n_nodes != self.n_nodes:
            raise engine.FitFailure("fitted node count changed", {"status": "invalid_output"})
        config = copy.deepcopy(result["config"])
        engine.validate_config(functional, config)
        fit_state = copy.deepcopy(result["fit_state"])
        json.dumps(dict(config=config, fit_state=fit_state), allow_nan=False)
        status = result["status"]
        if status not in {"completed", "not_identified"}:
            raise engine.FitFailure("no legal fitted candidate", fit_state)
        fit_result = FitResult(self.method_name, status, (), (), None, copy.deepcopy(fit_state))
        self.functional, self.gates = functional, functional.gates
        self.mass, self.damping = functional.mass.copy(), functional.damping.copy()
        self.inference_config, self.fit_state = config, fit_state
        self.fit_private_details = copy.deepcopy(result.get("private_details"))
        self.initialization_mode = "causal_two_stage"
        self.parent_initializer = None
        self.last_initialization = None
        return fit_result

    @classmethod
    def from_parent(cls, parent):
        """Embed an RMemoryROM and its frozen initializer in absolute theta coordinates."""
        import copy
        if type(parent).__name__ != "RMemoryROM" or type(parent).__module__ != "r_memory_model_20261025":
            raise ValueError("only the qualified RMemoryROM with Q=lambda I can be embedded")
        def array(value):
            return value.detach().cpu().numpy().copy()
        constants = tuple(array(value) for value in parent._constants()[:-1])
        k, damping, gain, rate, storage, kappa, lam = constants
        model = cls(n_nodes=len(parent.mass), mass=array(parent.mass), damping=damping)
        model.delta_star = array(parent.delta_star)
        model.parent_initializer = copy.deepcopy(parent).cpu().eval()
        model.functional.k_edge[:] = 0.
        mask = np.zeros(model.n_edges)
        for row, weight in zip(array(parent.base_D), k):
            i, j = np.flatnonzero(row)
            if not (np.array_equal(row[[i, j]], [1., -1.])
                    or np.array_equal(row[[i, j]], [-1., 1.])):
                raise ValueError("parent incidence must have one +1 and one -1")
            edge = model.edges.index((int(i), int(j)))
            mask[edge] = 1.
            model.functional.k_edge[edge] += weight
        model.gates = StructuralGates.static_baseline_limit(mask, model.n_nodes)
        model.functional.gates = model.gates
        model.functional.eta_edge = model.incidence.T @ model.delta_star
        model.functional.storage_p = storage
        model.functional.gain_p = gain
        model.functional.rate_p = rate
        model.functional.j_omega_mem = np.full(model.n_nodes, float(kappa))
        model.functional.lambda_mem = np.full(model.n_nodes, 1. / float(kappa))
        model.functional.tau_mem_sync = np.full(model.n_nodes, 1. / (float(kappa) * float(lam)))
        model.functional.__post_init__()
        return model

    @property
    def _uses_fixed_dp8(self):
        config = self.inference_config
        return isinstance(config, dict) and config.get("format") == FIXED_DP8_FORMAT

    def _validate_inference_config(self):
        if self.initialization_mode != "causal_two_stage" and not self._uses_fixed_dp8:
            return None
        from . import gftd_fitting as engine
        config = self.inference_config
        engine.validate_config(self.functional, config)
        if self._uses_fixed_dp8:
            parent = config["fit297_config"]
            if (self.initialization_mode != "causal_two_stage" or self.fit_mode != parent["mode"]
                    or self.parent_initializer is not None):
                raise engine.InitializationFailure("fixed DP8 model dispatch changed",
                                                   {"status": "invalid_configuration"})
            return parent
        return config

    def initialize(self, initial_state: dict[str, np.ndarray]) -> np.ndarray:
        allowed = {"theta_history", "omega_history", "history_times", "p0"}
        if set(initial_state) != allowed:
            raise ValueError(f"initialization requires exactly {sorted(allowed)}")
        theta = np.asarray(initial_state["theta_history"], dtype=np.float64)
        omega = np.asarray(initial_state["omega_history"], dtype=np.float64)
        times = np.asarray(initial_state["history_times"], dtype=np.float64)
        p0 = np.asarray(initial_state["p0"], dtype=np.float64)
        if theta.shape != (11, self.n_nodes) or omega.shape != theta.shape:
            raise ValueError("exactly 11 node-aligned history points required")
        if times.shape != (11,) or not np.allclose(times, np.linspace(-.1, 0., 11), atol=1e-12, rtol=0):
            raise ValueError("history_times must cover -0.1 to 0 seconds at 0.01 seconds")
        if p0.shape != (self.n_nodes,) or not all(np.isfinite(v).all() for v in (theta, omega, times, p0)):
            raise ValueError("history and p0 must be finite and node-aligned")
        if self.initialization_mode == "causal_two_stage" or self._uses_fixed_dp8:
            from . import gftd_fitting as engine
            try:
                config = self._validate_inference_config()
                packet = dict(theta_history=theta, omega_history=omega, history_times=times,
                              p_end=p0, sigma=config["sigma"])
                state, diagnostics = engine.initialize_complete(self.functional, packet, self.inference_config)
                state = np.asarray(state, dtype=np.float64)
                if state.shape != (5*self.n_nodes,) or not np.isfinite(state).all():
                    raise engine.InitializationFailure("invalid initialized state", {"status": "invalid_output"})
                if not np.array_equal(state[2*self.n_nodes:3*self.n_nodes], p0):
                    raise engine.InitializationFailure("p_end hard constraint changed", {"status": "invalid_output"})
            except engine.InitializationFailure as error:
                self.last_initialization = copy.deepcopy(error.diagnostics)
                raise
            self.last_initialization = copy.deepcopy(diagnostics)
            return state
        if np.any(self.functional.gates.s_mod != 0):
            raise NotImplementedError("nonzero-modulation causal latent initialization is not implemented")
        amp = np.zeros(self.n_nodes)
        if self.parent_initializer is None:
            _, mem = self.encoder.encode(theta, omega, p0)
        else:
            import torch
            history = np.zeros((11, 3 * self.n_nodes))
            history[:, :self.n_nodes] = theta - self.delta_star
            history[:, self.n_nodes:2*self.n_nodes] = omega
            history[-1, 2*self.n_nodes:] = p0
            with torch.no_grad():
                mem = self.parent_initializer.encode(history).detach().cpu().numpy()
        return self.functional.pack_state(theta[-1], omega[-1], p0, amp, mem)

    def rollout_state(self, initial_state, time_grid, exogenous=None):
        if exogenous:
            raise ValueError("autonomous rollout accepts no exogenous inputs")
        t = np.asarray(time_grid, dtype=np.float64)
        if t.ndim != 1 or not t.size or not np.isfinite(t).all() or t[0] != 0.:
            raise ValueError("time_grid must be finite, nonempty, one-dimensional and start at zero")
        steps = np.diff(t)
        if np.any(steps <= 0) or np.any(steps > .01 + 1e-12):
            raise ValueError("future steps must be positive and at most 0.01 seconds")
        fixed_step = None
        if self._uses_fixed_dp8:
            for step in (.01, .005):
                if np.allclose(t, np.arange(t.size) * step, atol=1e-12, rtol=0):
                    fixed_step = step
                    break
            else:
                raise ValueError("fixed DP8 requires a uniform 0.01 or 0.005 second future grid")
        state = self.initialize(initial_state)
        trajectory = np.empty((t.size, state.size))
        trajectory[0] = state
        if fixed_step is not None:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for index in range(1, t.size):
                    state = dop853_step(self.functional.rhs, state, fixed_step)
                    trajectory[index] = state
            return trajectory
        for index, dt in enumerate(steps):
            k1 = self.functional.rhs(state)
            k2 = self.functional.rhs(state + .5 * dt * k1)
            k3 = self.functional.rhs(state + .5 * dt * k2)
            k4 = self.functional.rhs(state + dt * k3)
            state = state + dt / 6. * (k1 + 2*k2 + 2*k3 + k4)
            self.functional.split_state(state)
            trajectory[index+1] = state
        return trajectory

    def rollout(self, initial_state, time_grid, exogenous):
        states = self.rollout_state(initial_state, time_grid, exogenous)
        return states[:, :self.n_nodes], states[:, self.n_nodes:2*self.n_nodes]

    def physics_audit(self):
        state = self.functional.pack_state(*np.random.default_rng(self.seed).uniform(
            -.1, .1, (5, self.n_nodes)))
        return dict(self.functional.power_balance_audit(state), method=self.method_name,
                    state_dimension=5*self.n_nodes, coi_dynamics_decoupled=False,
                    partial_observation_fit_implemented=True,
                    initialization_mode=self.initialization_mode)

    def save(self, path: str | Path) -> None:
        if type(self.functional) is not GeneratingFunctionalKuramoto:
            raise ValueError("functional subclass cannot be represented by the versioned checkpoint")
        kernel = self.functional.modulation_kernel
        if kernel not in (EXP_MODULATION_KERNEL, BOUNDED_MODULATION_KERNEL):
            raise ValueError("unknown amplitude modulation kernel")
        self._validate_inference_config()
        from dataclasses import fields
        def serial(value):
            return value.tolist() if isinstance(value, np.ndarray) else value
        functional = {
            field.name: serial(getattr(self.functional, field.name))
            for field in fields(self.functional) if field.name != "gates"
        }
        if kernel == EXP_MODULATION_KERNEL:
            functional.pop("modulation_kernel")
        checkpoint_format = "gftd_kuramoto_5n_v2" if kernel == EXP_MODULATION_KERNEL else "gftd_kuramoto_5n_v3"
        parent = self.parent_initializer
        payload = dict(format=checkpoint_format, seed=self.seed, functional=functional,
                       initialization_mode=self.initialization_mode, fit_mode=self.fit_mode,
                       inference_config=self.inference_config, fit_state=self.fit_state,
                       last_initialization=self.last_initialization,
                       delta_star=self.delta_star.tolist(),
                       gates={field.name: serial(getattr(self.functional.gates, field.name))
                              for field in fields(self.functional.gates)},
                       encoder={name: value.tolist() for name, value in vars(self.encoder).items()
                                if name.startswith("w_")}, parent_initializer=None)
        if parent is not None:
            payload["parent_initializer"] = dict(
                config=parent.config(), mode=parent.mode, seed=parent.seed,
                source_identity=parent.source_identity,
                state_dict={key: value.detach().cpu().tolist() for key, value in parent.state_dict().items()})
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checkpoint_format = payload.get("format")
        if checkpoint_format not in {"gftd_kuramoto_5n_v1", "gftd_kuramoto_5n_v2", "gftd_kuramoto_5n_v3"}:
            raise ValueError("only the explicit 5N format is supported; old 4N checkpoints are incompatible")
        kwargs = payload["functional"]
        if checkpoint_format == "gftd_kuramoto_5n_v3":
            if kwargs.get("modulation_kernel") != BOUNDED_MODULATION_KERNEL:
                raise ValueError("v3 checkpoint requires the explicit bounded modulation kernel")
        elif "modulation_kernel" in kwargs:
            raise ValueError("legacy checkpoint cannot carry a modulation kernel field")
        gates = StructuralGates(**payload["gates"])
        functional = GeneratingFunctionalKuramoto(**kwargs, gates=gates)
        model = cls(n_nodes=functional.n_nodes, mass=functional.mass, damping=functional.damping,
                    seed=payload["seed"])
        model.functional, model.gates = functional, gates
        model.delta_star = np.asarray(payload["delta_star"], dtype=np.float64).reshape(model.n_nodes)
        if not np.isfinite(model.delta_star).all():
            raise ValueError("delta_star must be finite")
        expected_encoder = {name for name in vars(model.encoder) if name.startswith("w_")}
        if set(payload["encoder"]) != expected_encoder:
            raise ValueError("checkpoint must include all encoder parameters")
        for name, value in payload["encoder"].items():
            if name not in vars(model.encoder) or not name.startswith("w_"):
                raise ValueError("unknown encoder parameter")
            array = np.asarray(value, dtype=np.float64)
            if array.shape != getattr(model.encoder, name).shape or not np.isfinite(array).all():
                raise ValueError("invalid encoder parameter")
            setattr(model.encoder, name, array)
        parent = payload["parent_initializer"]
        if parent is not None:
            import torch
            from r_memory_model_20261025 import RMemoryROM
            initializer = RMemoryROM(parent["config"], mode=parent["mode"], seed=parent["seed"],
                                     source_identity=parent["source_identity"])
            template = initializer.state_dict()
            initializer.load_state_dict({
                key: torch.as_tensor(value, dtype=template[key].dtype)
                for key, value in parent["state_dict"].items()}, strict=True)
            model.parent_initializer = initializer.eval()
        if checkpoint_format != "gftd_kuramoto_5n_v1":
            mode = payload["initialization_mode"]
            if mode not in {"legacy_static", "causal_two_stage"}:
                raise ValueError("unknown initialization dispatch")
            if payload["fit_mode"] not in {"static", "modulated"}:
                raise ValueError("unknown fit mode")
            model.initialization_mode, model.fit_mode = mode, payload["fit_mode"]
            model.inference_config = payload["inference_config"]
            model.fit_state = payload["fit_state"]
            model.last_initialization = payload["last_initialization"]
            model._validate_inference_config()
        else:
            config = payload.get("inference_config")
            if isinstance(config, dict) and config.get("format") == FIXED_DP8_FORMAT:
                raise ValueError("fixed DP8 requires the causal checkpoint format")
        return model
