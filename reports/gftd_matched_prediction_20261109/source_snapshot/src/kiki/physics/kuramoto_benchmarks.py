from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .kuramoto_functional import GeneratingFunctionalKuramoto, StructuralGates


@dataclass(frozen=True)
class KuramotoBenchmarkDataset:
    """Synthetic Stage-I partially observed generalized Kuramoto benchmark."""

    n_nodes: int
    edges: tuple[tuple[int, int], ...]
    true_edge_mask: np.ndarray
    true_mod_mask: np.ndarray
    true_k_edge: np.ndarray
    true_gamma_amp: np.ndarray
    time_grid: np.ndarray
    history_steps: int
    cases: tuple[dict[str, Any], ...]


def _rk4_step(func: GeneratingFunctionalKuramoto, state: np.ndarray, dt: float) -> np.ndarray:
    k1 = func.rhs(state)
    k2 = func.rhs(state + 0.5 * dt * k1)
    k3 = func.rhs(state + 0.5 * dt * k2)
    k4 = func.rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_partially_observed_kuramoto_benchmark(
    n_nodes: int = 6,
    n_cases: int = 12,
    dt: float = 0.01,
    history_steps: int = 11,
    future_steps: int = 80,
    seed: int = 20261106,
) -> KuramotoBenchmarkDataset:
    """Generate same-family artificial trajectories for mathematical checks only.

    Only ``(theta, omega)`` are exposed as observables in ``cases[k]["observed"]``,
    while the ground-truth sparse graph ``true_edge_mask``, latent-modulated subset
    ``true_mod_mask``, and latent trajectories ``(z_amp, z_mem)`` are stored under
    ``cases[k]["ground_truth_latent"]`` exclusively for post-discovery evaluation.
    """
    rng = np.random.default_rng(seed)
    mass = np.linspace(1.0, 1.5, n_nodes, dtype=np.float64)
    damping = np.full(n_nodes, 0.25, dtype=np.float64)

    n_edges = n_nodes * (n_nodes - 1) // 2
    edges = tuple((i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes))

    # Construct a sparse true ring + chord topology with a subset of latent-modulated edges
    true_edge_mask = np.zeros(n_edges, dtype=np.float64)
    true_mod_mask = np.zeros(n_edges, dtype=np.float64)
    true_k_edge = np.zeros(n_edges, dtype=np.float64)

    for e_idx, (i, j) in enumerate(edges):
        if j == i + 1 or (i == 0 and j == n_nodes - 1):
            true_edge_mask[e_idx] = 1.0
            true_k_edge[e_idx] = 2.0 + 0.4 * ((i + j) % 3)
            if i in (0, 2):
                true_mod_mask[e_idx] = 1.0
        elif i == 0 and j == n_nodes // 2:
            true_edge_mask[e_idx] = 1.0
            true_mod_mask[e_idx] = 1.0
            true_k_edge[e_idx] = 2.6

    gates = StructuralGates(
        s_edge=true_edge_mask,
        s_mod=true_mod_mask,
        s_mem=np.ones(n_nodes, dtype=np.float64),
        s_coi=0.0,
    )
    gamma_amp = np.full(n_nodes, 0.65, dtype=np.float64)
    model = GeneratingFunctionalKuramoto(
        mass=mass,
        damping=damping,
        k_edge=true_k_edge,
        eta_edge=np.zeros(n_edges, dtype=np.float64),
        gamma_amp=gamma_amp,
        lambda_amp=np.ones(n_nodes, dtype=np.float64),
        lambda_mem=np.ones(n_nodes, dtype=np.float64),
        tau_amp=np.full(n_nodes, 0.4, dtype=np.float64),
        tau_mem_sync=np.full(n_nodes, 0.5, dtype=np.float64),
        tau_mem_coi=0.35,
        j_omega_mem=np.full(n_nodes, 0.3, dtype=np.float64),
        gates=gates,
    )

    total_steps = history_steps + future_steps
    time_grid = (np.arange(total_steps, dtype=np.float64) - (history_steps - 1)) * dt
    cases: list[dict[str, Any]] = []

    for case_idx in range(n_cases):
        theta0 = rng.uniform(-0.85, 0.85, size=n_nodes)
        theta0 -= np.sum(mass * theta0) / np.sum(mass)
        omega0 = rng.uniform(-0.7, 0.7, size=n_nodes)
        # Alternate amplitude excitation levels so latent modulation is identifiable (Theorem 2)
        amp_scale = -0.65 if (case_idx % 2 == 1) else 0.55
        z_amp0 = amp_scale * rng.uniform(0.6, 1.2, size=n_nodes)
        z_mem0 = rng.uniform(-0.3, 0.3, size=n_nodes)

        p0 = rng.uniform(-0.2, 0.2, size=n_nodes)
        state = model.pack_state(theta0, omega0, p0, z_amp0, z_mem0)
        traj = np.zeros((total_steps, 5 * n_nodes), dtype=np.float64)
        energies = np.zeros(total_steps, dtype=np.float64)
        for step in range(total_steps):
            traj[step] = state
            energies[step] = model.hamiltonian(state)
            if step + 1 < total_steps:
                state = _rk4_step(model, state, dt)

        theta_traj = traj[:, :n_nodes]
        omega_traj = traj[:, n_nodes : 2 * n_nodes]
        p_traj = traj[:, 2 * n_nodes : 3 * n_nodes]
        z_amp_traj = traj[:, 3 * n_nodes : 4 * n_nodes]
        z_mem_traj = traj[:, 4 * n_nodes : 5 * n_nodes]

        cases.append(
            {
                "case_id": f"kuramoto_case_{case_idx:03d}",
                "observed": {
                    "t": time_grid.copy(),
                    "history_times": time_grid[:history_steps].copy(),
                    "p0": p_traj[history_steps - 1].copy(),
                    "theta_history": theta_traj[:history_steps].copy(),
                    "omega_history": omega_traj[:history_steps].copy(),
                    "theta_full": theta_traj.copy(),
                    "omega_full": omega_traj.copy(),
                    "mass": mass.copy(),
                    "damping": damping.copy(),
                },
                "ground_truth_latent": {
                    "p_full": p_traj.copy(),
                    "z_amp_full": z_amp_traj.copy(),
                    "z_mem_full": z_mem_traj.copy(),
                    "hamiltonian": energies.copy(),
                },
            }
        )

    return KuramotoBenchmarkDataset(
        n_nodes=n_nodes,
        edges=edges,
        true_edge_mask=true_edge_mask,
        true_mod_mask=true_mod_mask,
        true_k_edge=true_k_edge,
        true_gamma_amp=gamma_amp,
        time_grid=time_grid,
        history_steps=history_steps,
        cases=tuple(cases),
    )
