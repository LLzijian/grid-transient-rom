"""Exact local spectral penalty, removing only the common-angle gauge."""

import torch


def reference_geometry(model):
    """Return the fixed equilibrium, state scales, and common-angle quotient basis."""
    n = model.n
    x0 = torch.cat((model.theta_ref, model.theta_ref.new_zeros(4 * n)))
    scale = torch.cat((torch.ones_like(model.theta_ref), model.omega_scale, model.init_bounds))
    Q = x0.new_zeros((5 * n, 5 * n - 1))
    for j in range(n - 1):
        denominator = ((j + 1) * (j + 2)) ** .5
        Q[:j + 1, j] = 1 / denominator
        Q[j + 1, j] = -(j + 1) / denominator
    Q[n:, n - 1:] = torch.eye(4 * n, dtype=x0.dtype, device=x0.device)
    return x0, scale, Q


def projected_jacobian(model, geometry, *, create_graph=False):
    x0, scale, Q = geometry
    jacobian = torch.autograd.functional.jacobian(
        lambda x: model.rhs(None, x), x0, create_graph=create_graph, vectorize=False)
    normalized = jacobian * scale.unsqueeze(0) / scale.unsqueeze(1)
    return Q.T @ normalized @ Q


def spectral_penalty(model, geometry):
    """Exact spectral abscissa in s^-1; retain parameter derivatives for training."""
    alpha = torch.linalg.eigvals(projected_jacobian(model, geometry, create_graph=True)).real.amax()
    return .05 * (alpha + .05).clamp_min(0).square(), alpha
