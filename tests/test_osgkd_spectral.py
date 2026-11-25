"""Synthetic Sprint2A gates; no event data or saved weights are used."""

import pytest
import torch

from kiki.models.osgkd import OSGKDConfig
from kiki.models.osgkd_matched_flat import OSGKDMatchedFlat
from kiki.models.osgkd_spectral import reference_geometry, projected_jacobian, spectral_penalty


def model():
    torch.manual_seed(42)
    n = 3
    m = OSGKDMatchedFlat(OSGKDConfig('C1', hidden_size=32, p_init_bound=4,
                                    a_init_bound=2, e_init_bound=4),
                          mass=torch.tensor([1., 1.3, 1.7]), damping=torch.ones(n) * .3,
                          F=torch.eye(n), Sp=torch.ones(n), Se=torch.ones(n),
                          omega_scale=torch.tensor([.8, 1., 1.2]),
                          theta_ref=torch.tensor([.1, -.2, .3]),
                          edge_index=torch.triu_indices(n, n, 1))
    with torch.no_grad():
        m.mechanical_coeff[0].copy_(torch.tensor([.4, -.3, -.4]))
        m.amplitude_coeff[0].fill_(-.6)
        m.memory_coeff[0].fill_(-.7)
    return m


def test_flat_equilibrium_shift_bounds_and_checkpoint():
    m = model()
    theta = m.theta_ref.expand(2, 11, m.n).clone()
    omega = torch.zeros_like(theta)
    initial = m.initialize(theta, omega)
    assert torch.equal(initial[:, 2*m.n:], torch.zeros(2, 3*m.n, dtype=torch.float64))
    assert torch.equal(m.rhs(None, initial), torch.zeros_like(initial))
    theta += torch.randn_like(theta) * .5
    omega += torch.randn_like(omega)
    initial = m.initialize(theta, omega)
    torch.testing.assert_close(m.initialize(theta + 2, omega)[:, 2*m.n:], initial[:, 2*m.n:],
                               rtol=0, atol=1e-14)
    assert torch.equal(initial[:, :m.n], theta[:, -1])
    assert torch.equal(initial[:, m.n:2*m.n], omega[:, -1])
    assert (initial[:, 2*m.n:].abs() <= m.init_bounds).all()
    assert m.initial_head.bias is None
    with torch.no_grad():
        m.initial_head.weight.mul_(1e4)
    wide = m.initialize(theta, omega)[:, 2*m.n:]
    assert (wide.abs().amax(0) > .99 * m.init_bounds).all()
    restored = OSGKDMatchedFlat.from_checkpoint(m.checkpoint())
    assert m.checkpoint()['version'] == 5
    assert torch.equal(restored.initialize(theta, omega), m.initialize(theta, omega))


@pytest.mark.parametrize('step', [1e-4, 2e-5])
def test_projected_matrix_and_alpha_against_full_vector_field_fd(step):
    m = model()
    x0, scale, Q = geometry = reference_geometry(m)
    torch.testing.assert_close(Q.T @ Q, torch.eye(5*m.n-1, dtype=torch.float64))
    gauge = x0.new_zeros(5*m.n)
    gauge[:m.n] = 1
    torch.testing.assert_close(Q.T @ gauge, torch.zeros(5*m.n-1, dtype=torch.float64))
    common_speed = x0.new_zeros(5*m.n)
    common_speed[m.n:2*m.n] = 1
    torch.testing.assert_close(Q @ Q.T @ common_speed, common_speed)
    columns = []
    for direction in torch.eye(5*m.n, dtype=torch.float64):
        delta = step * scale * direction
        columns.append((m.rhs(None, x0 + delta) - m.rhs(None, x0 - delta)) / (2*step) / scale)
    fd = Q.T @ torch.stack(columns, 1) @ Q
    ad = projected_jacobian(m, geometry)
    torch.testing.assert_close(ad, fd, rtol=1e-3, atol=1e-7)
    torch.testing.assert_close(torch.linalg.eigvals(ad).real.amax(),
                               torch.linalg.eigvals(fd).real.amax(), rtol=1e-3, atol=1e-7)


@pytest.mark.parametrize('step', [1e-4, 2e-5])
@pytest.mark.parametrize('coefficient,index', [('mechanical_coeff', (0, 0)), ('memory_coeff', (0, 1))])
def test_active_hinge_parameter_derivative(step, coefficient, index):
    m = model()
    parameter = getattr(m, coefficient)
    if coefficient == 'memory_coeff':
        with torch.no_grad():
            parameter[index] = .55
    geometry = reference_geometry(m)
    loss, alpha = spectral_penalty(m, geometry)
    assert alpha.item() > .1
    gradient, = torch.autograd.grad(loss, parameter)
    # Each case has a unique dominant unstable mode in a separate parameter block.
    assert gradient[index].abs() > .01
    original = parameter[index].item()
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            parameter[index] = original + sign * step
        values.append(spectral_penalty(m, geometry)[0].item())
    with torch.no_grad():
        parameter[index] = original
    fd = (values[0] - values[1]) / (2*step)
    assert fd == pytest.approx(gradient[index].item(), rel=1e-3, abs=1e-7)
