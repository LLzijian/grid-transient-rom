"""Exact non-identifiability companions with explicit restricted scope."""

import numpy as np

from kiki.data import osgkd_synthetic as truth_model
from experiments.osgkd_v2_wp2_layer1_20261127.synthetic import make_system, NOISE_LEVELS


def svd_summary(design):
    design = np.asarray(design, dtype=float)
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = max(design.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    rank = int(np.count_nonzero(singular > tolerance))
    return dict(shape=list(design.shape), singular_values=singular.tolist(), rank=rank,
                nullity=design.shape[1]-rank, tolerance=float(tolerance))


def recovery_curve(design, parameters, noise_levels=NOISE_LEVELS, seed=27031):
    """Minimum-norm LS companion curve, never an OSGKD training experiment.

    One fixed standard Gaussian observation direction is reused at all levels.
    The companion's observations are only its eleven 100ms history frames.
    Parameter error is against one representative of an equivalence class.
    """
    design, parameters = np.asarray(design, dtype=float), np.asarray(parameters, dtype=float)
    clean = design @ parameters
    direction = np.random.default_rng(seed).standard_normal(clean.shape)
    rows = []
    for level in noise_levels:
        if not np.isfinite(level) or level < 0:
            raise ValueError('finite nonnegative observation noise required')
        observed = clean + level*direction
        fitted = np.linalg.lstsq(design, observed, rcond=None)[0]
        rows.append(dict(noise_std=float(level), parameter_relative_l2=float(np.linalg.norm(fitted-parameters)/max(np.linalg.norm(parameters), 1e-30)),
                         clean_observation_rmse=float(np.sqrt(np.mean((design@fitted-clean)**2))),
                         noisy_residual_rmse=float(np.sqrt(np.mean((design@fitted-observed)**2))),
                         fitted_parameter_l2=float(np.linalg.norm(fitted))))
    return rows


def candidate_parameters(system):
    lookup = {tuple(edge): index for index, edge in enumerate(system['candidate_edges'].tolist())}
    c = np.zeros(len(lookup))
    beta = c.copy()
    for index, edge in enumerate(system['edges'].tolist()):
        c[lookup[tuple(edge)]] = system['c'][index]
        beta[lookup[tuple(edge)]] = system['beta'][index]
    return np.concatenate((c, beta))


def synchronization(system=None):
    """Exact common-angle equilibrium: c/beta unrestricted by motion here."""
    system = make_system() if system is None else system
    n = system['n']
    state = np.zeros((5, n))
    state[0].fill(.2)
    derivative = truth_model.rhs(system, state)
    parameters = candidate_parameters(system)
    design = np.zeros((11*n, len(parameters)))
    return dict(scope='exact common-angle zero-speed zero-p/a/e equilibrium of this NumPy ODE only',
                observed_rhs_abs_max=float(np.abs(derivative[:2]).max()),
                full_rhs_abs_max=float(np.abs(derivative).max()), svd=svd_summary(design),
                analytic_rank=0, analytic_nullity=len(parameters), exact_zero_singular_values=min(design.shape),
                recovery_curve=recovery_curve(design, parameters),
                conclusion='c and beta cannot be distinguished on this equilibrium trajectory')


def pure_amplitude(system=None):
    """a decays with no phase/speed/p/e excitation; beta remains invisible."""
    system = make_system() if system is None else system
    n = system['n']
    times = np.arange(11)*.01
    amplitude = np.linspace(.2, .6, n)
    states = np.zeros((11, 5, n))
    states[:, 3] = np.exp(-.6*times[:, None])*amplitude
    rhs = np.stack([truth_model.rhs(system, state) for state in states])
    expected = np.zeros_like(rhs)
    expected[:, 3] = -.6*states[:, 3]
    parameters = candidate_parameters(system)
    design = np.zeros((11*n, len(parameters)))
    return dict(scope='exact amplitude-only trajectory with theta=omega=p=e=0 in this NumPy ODE',
                analytic_rhs_residual_abs_max=float(np.max(np.abs(rhs-expected))),
                observed_motion_abs_max=float(np.abs(states[:, :2]).max()),
                observed_rhs_abs_max=float(np.abs(rhs[:, :2]).max()),
                amplitude_norm=float(np.linalg.norm(states[:, 3])), svd=svd_summary(design),
                analytic_rank=0, analytic_nullity=len(parameters), exact_zero_singular_values=min(design.shape),
                recovery_curve=recovery_curve(design, parameters),
                conclusion='nonzero amplitude alone does not excite coupling; c and beta are unobservable here')


def same_frequency_cancellation(system=None):
    """y=(A c+q)f: an exact linear observation companion, not the full ODE."""
    system = make_system() if system is None else system
    n, edge_count = system['n'], len(system['edges'])
    incidence = np.zeros((n, edge_count))
    for index, (left, right) in enumerate(system['edges']):
        incidence[left, index], incidence[right, index] = 1., -1.
    times = np.arange(11)*.01
    frequency = np.cos(2*np.pi*times)
    mixing = np.concatenate((incidence, np.eye(n)), axis=1)
    design = (frequency[:, None, None]*mixing).reshape(11*n, edge_count+n)
    parameters = np.concatenate((system['c'], np.linspace(-.2, .2, n)))
    # Remove one strictly positive edge while preserving every observation.
    delta_c = np.zeros(edge_count)
    delta_c[0] = -system['c'][0]
    null_vector = np.concatenate((delta_c, -incidence@delta_c))
    clean = design@parameters
    alternative = design@(parameters+null_vector)
    return dict(scope='linear observation companion y(t)=(A c+q) f(t); no claim of full ODE equivalence',
                observation_frames=11, observed_duration_s=.1,
                null_vector_residual_abs_max=float(np.max(np.abs(design@null_vector))),
                alternative_observation_abs_difference=float(np.max(np.abs(clean-alternative))),
                alternative_parameter_l2=float(np.linalg.norm(null_vector)),
                alternative_c_nonnegative=bool((parameters[:edge_count]+delta_c >= 0).all()),
                true_edge_count=int(np.count_nonzero(parameters[:edge_count])),
                alternative_edge_count=int(np.count_nonzero(parameters[:edge_count]+delta_c)),
                analytic_rank=n, analytic_nullity=edge_count, exact_zero_singular_values=edge_count,
                svd=svd_summary(design), recovery_curve=recovery_curve(design, parameters),
                conclusion='(c,q) and (c+delta,q-A delta) have identical companion observations')


def conditioning_grid():
    """Positive and degenerate known-state torque observations on the same dictionary."""
    system = make_system()
    n = system['n']
    left, right = system['candidate_edges'].T
    m = len(left)
    incidence = np.zeros((n, m))
    incidence[left, np.arange(m)] = 1.
    incidence[right, np.arange(m)] = -1.
    rng = np.random.default_rng(42033)
    phases = rng.standard_normal((66, n))
    amplitudes = .4*rng.standard_normal((66, n))
    parameters = np.r_[candidate_parameters(system), .4]
    shared = np.full(66*n, 1/np.sqrt(66*n))
    rows = []
    for excitation in (0., .01, .05, .25, 1.):
        angles = (phases[:, left]-phases[:, right])*excitation
        base = np.sin(angles)
        beta = base*np.expm1(np.tanh(amplitudes[:, left]+amplitudes[:, right]))
        design = np.concatenate([(features[:, None,:]*incidence[None]).reshape(66*n,m)
                                 for features in (base,beta)], axis=1)
        reciprocal = design[:,0]
        size = np.linalg.norm(reciprocal)
        match = reciprocal / size if size else reciprocal
        for overlap in (0., .9, .999, 1.):
            environment = overlap*match + np.sqrt(1-overlap**2)*shared
            combined = np.column_stack((design, environment))
            svd = svd_summary(combined)
            full = svd['rank'] == combined.shape[1]
            sigma_min = svd['singular_values'][-1]
            rows.append(dict(excitation=excitation, environmental_waveform_overlap=overlap,
                             algebraic_class=('no_edge_excitation' if not excitation else
                                              'full_rank' if full else 'equivalent_parameters'),
                             svd=svd, inverse_smallest_singular_value=1/sigma_min if full else None,
                             recovery_curve=recovery_curve(combined,parameters)))
    return dict(scope='known-state total-torque design; theta and a supplied for this diagnostic only',
                samples=66, candidates=m, rows=rows)


def audit():
    """Small algebraic/analytic controls only; creates no simulated episodes."""
    result = dict(synchronization=synchronization(), pure_amplitude=pure_amplitude(),
                  same_frequency_cancellation=same_frequency_cancellation())
    for name in ('synchronization', 'pure_amplitude'):
        assert result[name]['observed_rhs_abs_max'] == 0.
        assert result[name]['svd']['rank'] == 0
    assert result['pure_amplitude']['analytic_rhs_residual_abs_max'] == 0.
    assert result['same_frequency_cancellation']['null_vector_residual_abs_max'] < 1e-14
    assert result['same_frequency_cancellation']['alternative_c_nonnegative']
    assert result['same_frequency_cancellation']['alternative_edge_count'] == result['same_frequency_cancellation']['true_edge_count']-1
    assert result['same_frequency_cancellation']['svd']['nullity'] >= 12
    result['condition_map'] = conditioning_grid()
    return result
