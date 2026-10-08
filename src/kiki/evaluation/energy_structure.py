"""Local numerical structure audits for energy-structured models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from kiki.physics.coordinates import coi_basis


@dataclass(frozen=True)
class EnergyStructureAudit:
    storage_center_error: float
    storage_gradient_norm: float
    angle_speed_projected_hessian_min: float
    local_angle_speed_curvature_positive: bool
    max_local_supply_identity_error: float
    local_supply_identity_pass: bool
    structure_audit_passed: bool
    claim_boundary: str = (
        "local numerical audit only; no hidden-state curvature, global decrease, "
        "region-of-attraction, or escape-energy certification"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gradient(fun, x, epsilon: float = 1e-6) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(len(x)):
        direction = np.zeros_like(x)
        direction[i] = epsilon
        out[i] = (fun(x + direction) - fun(x - direction)) / (2 * epsilon)
    return out


def _hessian(fun, x, epsilon: float = 2e-5) -> np.ndarray:
    hessian = np.empty((len(x), len(x)))
    for i in range(len(x)):
        direction = np.zeros_like(x)
        direction[i] = epsilon
        hessian[:, i] = (_gradient(fun, x + direction) - _gradient(fun, x - direction)) / (2 * epsilon)
    return 0.5 * (hessian + hessian.T)


def evaluate_energy_structure(
    model,
    samples: list[dict],
    *,
    identity_tolerance: float = 1e-6,
    hessian_tolerance: float = 1e-9,
) -> EnergyStructureAudit:
    """Audit local storage geometry and the supply identity of a frozen model.

    Uses the post-clear equilibrium as the storage anchor (ZOH Pm_clear), and
    checks centering, finite-difference stationarity, angle/speed curvature on
    the COI quotient, and local supply-identity residuals. Passing does not
    certify the hidden-state curvature, global decrease, a region of
    attraction, or a critical escape-energy lower bound.
    """
    if not samples:
        raise ValueError("energy-structure audit requires at least one sample")
    sample = samples[0]
    static = sample["model_inputs"]["static"]
    mass = np.asarray(static["mass"], dtype=float)
    pm_clear = np.asarray(static["Pm_clear"], dtype=float).reshape(-1)
    n = len(mass)
    incidence = np.asarray(static["incidence"], dtype=float)
    n_edges = incidence.shape[1]

    eq_delta, _eq_audit = model.solve_postclear_equilibrium(static)
    eq_delta = np.asarray(eq_delta, dtype=float)

    # Centering: storage(eq) ~ 0 (model.storage defaults to pm_clear/reference=eq)
    h_eq = model.storage(eq_delta, np.zeros(n), static, pm_clear=pm_clear, reference_delta=eq_delta)
    center_error = float(abs(h_eq))

    def scalar_h(x: np.ndarray) -> float:
        delta, omega, hidden = x[:n], x[n : 2 * n], x[2 * n :]
        return float(
            model.storage(delta, omega, static, hidden, pm_clear=pm_clear, reference_delta=eq_delta)
        )

    hidden_shapes = model._hidden_shapes(n, n_edges)
    zero_hidden = np.concatenate(
        [np.zeros(int(np.prod(shapes))) for shapes in hidden_shapes.values()]
    )
    x_eq = np.concatenate((eq_delta, np.zeros(n), zero_hidden))
    grad = _gradient(scalar_h, x_eq)
    grad_norm = float(np.linalg.norm(grad))

    # Curvature on the COI-projected angle/speed subspace only.
    basis = coi_basis(mass)
    # central-difference Hessian over full state is expensive; restrict to a
    # projected quadratic form via directional sampling on the leading
    # (angle-speed) subspace.
    # Use the angle/speed block only; hidden curvature is outside this audit.
    hs = _hessian(scalar_h, x_eq)
    block = hs[: (2 * n), : (2 * n)]
    P = np.zeros((2 * n, 2 * (n - 1)))
    P[:n, : n - 1] = basis
    P[n:, n - 1 :] = basis
    proj_hessian = P.T @ block @ P
    min_projected = float(np.min(np.linalg.eigvalsh(proj_hessian)))
    curvature_positive = bool(min_projected > hessian_tolerance)

    # Numeric supply identity along up to 8 validation states. The local identity
    # dH/dt = supply - dissipation is checked by finite differences at one state
    # per event (see supply_identity_error). A separate global monotone
    # "dissipation-adjusted storage" audit is intentionally NOT asserted here:
    # it depends on choosing a ZOH post-clear energy reference and would be a
    # semantic claim, not a numerical one. Global monotonicity is not asserted.
    identity_errors = []
    for candidate in samples[:8]:
        static_i = candidate["model_inputs"]["static"]
        identity_errors.append(model.supply_identity_error(static_i))
    max_identity = float(max(identity_errors))
    identity_pass = bool(max_identity <= identity_tolerance)
    structure_audit_passed = bool(
        abs(center_error) <= identity_tolerance
        and grad_norm <= 10 * identity_tolerance
        and curvature_positive
        and identity_pass
    )
    return EnergyStructureAudit(
        storage_center_error=center_error,
        storage_gradient_norm=grad_norm,
        angle_speed_projected_hessian_min=min_projected,
        local_angle_speed_curvature_positive=curvature_positive,
        max_local_supply_identity_error=max_identity,
        local_supply_identity_pass=identity_pass,
        structure_audit_passed=structure_audit_passed,
    )
