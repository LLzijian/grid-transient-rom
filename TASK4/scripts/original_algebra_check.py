"""Synthetic algebra checks for the TASK4 design, not a power-grid experiment.

Run with: python checks/check_power_structure.py --output checks/algebra_check.json
No user/project data, training, symbolic search, or file reads are performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

DT = torch.float64
N = 3
DIM = 5 * N
MASS = torch.tensor([1.0, 1.4, 0.8], dtype=DT)
D = torch.tensor([[1., -1., 0.], [0., 1., -1.], [1., 0., -1.]], dtype=DT)
PHI = torch.tensor([.12, -.08, .04], dtype=DT)
K = torch.tensor([1.1, .8, .6], dtype=DT)


def split(z: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if z.shape != (DIM,):
        raise ValueError(f"Expected a ({DIM},) state, got {tuple(z.shape)}")
    return tuple(z[k * N:(k + 1) * N] for k in range(5))


def energy(z: torch.Tensor) -> torch.Tensor:
    delta, pi, p, q1, q2 = split(z)
    eta = D @ delta
    ss = torch.sin(PHI + eta) - torch.sin(PHI)
    cc = torch.cos(PHI + eta) - torch.cos(PHI)
    base = ((PHI.cos() - (PHI + eta).cos() - PHI.sin() * eta) * K).sum()
    local = (.5 * p.square() + .6 * q1.square() + .4 * q2.square()
             + .03 * p * q1 + .01 * q2.pow(4) + .02 * p.square() * q1.square()).sum()
    edge = (.025 * (D @ q1) * ss + .018 * (D.abs() @ q2) * cc).sum()
    return .5 * (pi.square() / MASS).sum() + base + local + edge


def operators(z: torch.Tensor) -> tuple[torch.Tensor, ...]:
    delta, pi, p, q1, q2 = split(z)
    omega = pi / MASS
    J = torch.zeros((DIM, DIM), dtype=DT)
    R = torch.zeros_like(J)
    shaft = torch.zeros(DIM, dtype=DT)
    port = torch.zeros(DIM, dtype=DT)
    J[:N, N:2*N] = torch.eye(N, dtype=DT)
    J[N:2*N, :N] = -torch.eye(N, dtype=DT)
    shaft[N:2*N] = p

    def exchange(a: int, b: int, strength: torch.Tensor) -> None:
        J[a, b] = J[a, b] + strength
        J[b, a] = J[b, a] - strength

    for i in range(N):
        ip, iq1, iq2 = 2*N+i, 3*N+i, 4*N+i
        # Smooth, state-dependent but pointwise antisymmetric exchanges.
        exchange(N+i, iq1, .09 + .025 * torch.tanh(p[i] + q2[i]))
        exchange(ip, iq2, .07 + .02 * torch.tanh(q1[i]))
        exchange(iq1, iq2, .15 + .04 * torch.tanh(omega[i]))
        for index, value in [(N+i, .3), (ip, .23), (iq1, .16), (iq2, .19)]:
            R[index, index] += value**2
        direction = torch.zeros(DIM, dtype=DT)
        direction[N+i], direction[iq2] = 1., -.4
        direction /= torch.linalg.vector_norm(direction)
        ell = .12 + .025 * torch.tanh(q1[i] + p[i])
        R += ell.square() * torch.outer(direction, direction)
        # Two fixed sparse local port directions, no angle rows.
        port[ip] += -.15 * omega[i] - .05 * p[i]
        u2 = .015 * q2[i] + .01 * (omega[(i+1) % N] - omega[i])
        b2 = torch.zeros(DIM, dtype=DT)
        b2[N+i], b2[iq1] = 1., .2
        b2 /= torch.linalg.vector_norm(b2)
        port += b2 * u2
    return J, R, shaft, port


def field_and_terms(z: torch.Tensor) -> tuple[torch.Tensor, ...]:
    z = z.detach().clone().requires_grad_(True)
    H = energy(z)
    e = torch.autograd.grad(H, z)[0]
    J, R, shaft, port = operators(z)
    f = (J - R) @ e + shaft + port
    return H.detach(), e.detach(), f.detach(), J.detach(), R.detach(), shaft.detach(), port.detach()


def run() -> dict:
    rng = np.random.default_rng(4101)
    maxima = dict(kinematic_error=0., skew_error=0., power_identity_error=0.,
                  angle_shift_energy_error=0., angle_shift_field_error=0.,
                  energy_gradient_fd_error=0.)
    min_r = float("inf")
    for _ in range(40):
        z = torch.tensor(rng.normal(0., .3, DIM), dtype=DT)
        H, e, f, J, R, shaft, port = field_and_terms(z)
        maxima['kinematic_error'] = max(maxima['kinematic_error'], float((f[:N] - z[N:2*N]/MASS).abs().max()))
        maxima['skew_error'] = max(maxima['skew_error'], float((J + J.T).abs().max()))
        actual = e @ f
        predicted = -(e @ R @ e) + e @ shaft + e @ port
        maxima['power_identity_error'] = max(maxima['power_identity_error'], abs(float(actual - predicted)))
        min_r = min(min_r, float(torch.linalg.eigvalsh(R).min()))
        zs = z.clone()
        zs[:N] += .731
        Hs, _, fs, *_ = field_and_terms(zs)
        maxima['angle_shift_energy_error'] = max(maxima['angle_shift_energy_error'], abs(float(H-Hs)))
        maxima['angle_shift_field_error'] = max(maxima['angle_shift_field_error'], float((f-fs).abs().max()))
        h = 1e-6
        grad = []
        for j in range(DIM):
            zp, zm = z.clone(), z.clone()
            zp[j] += h
            zm[j] -= h
            grad.append(float((energy(zp)-energy(zm))/(2*h)))
        fd_error = np.max(np.abs(np.asarray(grad)-e.numpy()))
        maxima['energy_gradient_fd_error'] = max(maxima['energy_gradient_fd_error'], float(fd_error))
    origin_field = float(field_and_terms(torch.zeros(DIM, dtype=DT))[2].abs().max())
    powered = torch.zeros(DIM, dtype=DT)
    powered[N:2*N] = MASS * .5
    powered[2*N:3*N] = 1.
    _, ep, fp, *_ = field_and_terms(powered)
    positive_hdot = float(ep @ fp)
    if max(v for k, v in maxima.items() if k != 'energy_gradient_fd_error') > 1e-11:
        raise AssertionError(f"Algebra check failed: {maxima}")
    if maxima['energy_gradient_fd_error'] > 2e-8 or min_r < -1e-12 or origin_field > 1e-12:
        raise AssertionError("Gradient, PSD, or equilibrium check failed")
    if positive_hdot <= 0:
        raise AssertionError("Powered test did not demonstrate allowed positive Hdot")
    return {
        'scope': 'Synthetic 3-node, 2-memory-per-node algebra only; not KIKI data or trained models.',
        'dtype': 'float64', 'random_states': 40, 'passed': True,
        'max_errors': maxima, 'min_R_eigenvalue': min_r,
        'origin_field_max_abs': origin_field, 'powered_state_Hdot': positive_hdot,
        'not_tested': ['GRU training', 'predictive accuracy', 'symbolic compression',
                       'Poisson/Jacobi property', 'identifiability', 'region of attraction',
                       'complete TASK4 implementation']
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('algebra_check.json'))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
