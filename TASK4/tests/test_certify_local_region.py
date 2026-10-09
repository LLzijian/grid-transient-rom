"""Small synthetic interval proof checks; no real model/event payload is read.

Root runs these under the project budget. Their author does not execute them
during the concurrent scientific training stage.
"""
from fractions import Fraction
import importlib.util
import json
import math
import time

import numpy as np
import pytest

from task4.paths import ROOT


@pytest.fixture
def certifier(monkeypatch):
    pytest.importorskip('mpmath')
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('interval_region_test', ROOT / 'scripts/certify_local_region.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_spec(n=2, *, learned=False, port_slope=0.0):
    """Complete JSON fixture with every outer mechanism available, no Torch."""
    rank, hidden = 2, 2
    incidence = np.zeros((n, n-1))
    for edge in range(n-1):
        incidence[edge, edge], incidence[edge+1, edge] = 1, -1
    m, sdim = 2+rank, 1+rank
    sizes = {
        'node_energy': (sdim+4, 1), 'edge_energy': (2+2*sdim+8, 1),
        'node_exchange': (m+4, m*(m-1)//2), 'node_dissipation': (m+4, m),
        'edge_exchange': (2+2*m+8, m*m), 'edge_dissipation': (2+2*m+8, m),
        'port': (m+4, 2),
    }
    rng = np.random.default_rng(101+n)
    functions = {}
    for name, (inputs, outputs) in sizes.items():
        powers = [[0]*inputs]
        if learned:
            for i in range(inputs):
                for degree in (1, 2):
                    row = [0]*inputs
                    row[i] = degree
                    powers.append(row)
            row = [0]*inputs
            row[0], row[1], row[-1] = 1, 1, 1
            powers.append(row)
        coefficients = (rng.integers(-3, 4, (len(powers), outputs)) / 256 if learned
                        else np.zeros((len(powers), outputs)))
        if name == 'port' and port_slope:
            powers.append([1]+[0]*(inputs-1))
            coefficients = np.concatenate((coefficients, [[port_slope, 0.0]]))
        functions[name] = {'type': 'polynomial', 'inputs': inputs, 'outputs': outputs,
                           'powers': powers, 'coefficients': coefficients.tolist()}
    scales = [.25, .5, .125] if learned else [1., 1., 1.]
    mass = np.arange(1, n+1, dtype=float) if learned else np.ones(n)
    mu = (np.arange(n*4).reshape(n, 4)/16).tolist() if learned else np.zeros((n, 4)).tolist()
    return {
        'format': 'task4-polynomial-powergraph', 'version': 1, 'n': n, 'rank': rank,
        'strength': .03125 if learned else (1.0 if port_slope else 0.0),
        'physics': {'mass': mass.tolist(), 'incidence': incidence.tolist(),
                    'delta_star': np.zeros(n).tolist(), 'pm_star': np.ones(n).tolist(),
                    'coupling': np.ones(n-1).tolist(), 'damping': np.ones(n).tolist(),
                    'rate': np.ones(n).tolist(), 'gain': np.ones(n).tolist()},
        'log_coupling': np.zeros(n-1).tolist(), 'scales': scales,
        'mu': mu, 'phase': (np.arange(n-1)/8 if learned else np.zeros(n-1)).tolist(),
        'edges': [[i, i+1] for i in range(n-1)], 'functions': functions,
        'encoder': {
            'type': 'shared_gru', 'history_samples': 11, 'history_seconds': .1,
            'sample_dt_seconds': .01, 'gate_order': ['reset', 'update', 'new'],
            'hidden_size': hidden, 'mass': mass.tolist(), 'adjacency': np.eye(n).tolist(),
            'scales': scales, 'mu': mu,
            'weight_ih_l0': np.zeros((3*hidden, 9)).tolist(),
            'weight_hh_l0': np.zeros((3*hidden, hidden)).tolist(),
            'bias_ih_l0': np.zeros(3*hidden).tolist(), 'bias_hh_l0': np.zeros(3*hidden).tolist(),
            'head_weight': np.zeros((rank, hidden)).tolist(), 'head_bias': np.zeros(rank).tolist(),
        },
    }


def midpoint_array(module, matrix):
    return np.asarray([[float(sum(module.bounds(value), Fraction(0))/2) for value in row] for row in matrix])


@pytest.mark.parametrize('value', [0.0, .1, -.3, math.nextafter(0., 1.), 1e200])
def test_interval_conversion_preserves_exact_binary64(certifier, value):
    ctx, _ = certifier.backend()
    interval = certifier.iv_exact(ctx, value)
    low, high = certifier.bounds(interval)
    exact = Fraction.from_float(value)
    assert low <= exact <= high
    assert low == high == exact
    encoded = certifier.serialize_interval(interval)
    assert Fraction(encoded['lower_exact_rational']) == exact
    json.dumps(encoded, allow_nan=False)


def test_nonfinite_interval_and_json_fail_closed(certifier, tmp_path):
    ctx, _ = certifier.backend()
    for interval in (ctx.mpf(['-inf', 'inf']), ctx.mpf('inf'), ctx.mpf('nan')):
        with pytest.raises(ValueError):
            certifier.bounds(interval)
    source = tmp_path/'source.json'
    for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
        source.write_text(text)
        with pytest.raises(ValueError):
            certifier.strict_json_load(source)
    assert json.loads(json.dumps(certifier.json_safe({'x': float('nan')}), allow_nan=False)) == {'x': None}


def test_endpoint_memory_capability_guard(certifier):
    with pytest.raises(certifier.UnsupportedIntervalBackend, match='serialization capability'):
        certifier.endpoint_fraction((0, 1, certifier.MAX_ENDPOINT_ABS_EXPONENT+1, 1))
    with pytest.raises(certifier.UnsupportedIntervalBackend, match='serialization capability'):
        certifier.endpoint_fraction((0, 1 << certifier.MAX_ENDPOINT_MANTISSA_BITS, 0,
                                     certifier.MAX_ENDPOINT_MANTISSA_BITS+1))


def test_interval_ad_extrema_and_zero_polynomial_derivative(certifier):
    ctx, _ = certifier.backend()
    x = certifier.AD(ctx, ctx.mpf([-2, 2]), {0: ctx.mpf(1)})
    y = x**2
    assert certifier.bounds(y.value) == (Fraction(0), Fraction(4))
    low, high = certifier.bounds(x.sin().gradient[0])
    assert low <= 0 < 1 <= high  # cos extrema inside the interval, not just endpoints
    z = certifier.AD(ctx, ctx.mpf(0), {0: ctx.mpf(1)})
    cubic = z**3
    assert certifier.bounds(cubic.gradient[0]) == (Fraction(0), Fraction(0))


@pytest.mark.parametrize('n', [2, 3])
def test_full_rhs_interval_ad_matches_independent_numpy_center_difference(certifier, n):
    ctx, _ = certifier.backend()
    model = certifier.IntervalPowerGraph(synthetic_spec(n, learned=True), ctx)
    intervals, _ = model.jacobian_box(0.0)
    actual = midpoint_array(certifier, intervals)
    _, _, diagnostic = certifier.numerical_candidate(model)
    np.testing.assert_allclose(actual, diagnostic['A'], rtol=2e-8, atol=2e-9)
    # This checks full learned energy, skew, squared node/edge factors and both
    # centered ports. Independent NumPy midpoint is only a parity diagnostic.
    assert model.dim == 5*n-1
    assert len(intervals) == 5*n-1
    assert len(model.scale[n-1:2*n-1]) == n  # every pi/common omega retained
    assert np.array_equal(model.scale[3*n-1:], np.ones(2*n))


@pytest.mark.parametrize('n', [2, 3])
def test_baseline_origin_interval_jacobian_contains_analytic_matrix(certifier, n):
    ctx, _ = certifier.backend()
    model = certifier.IntervalPowerGraph(synthetic_spec(n), ctx)
    actual, _ = model.jacobian_box(0.0)
    # Explicit component equations, without implementation lift or AD rules.
    expected = np.zeros((5*n-1, 5*n-1))
    incidence = np.asarray(model.reference.incidence)
    lap = incidence @ incidence.T
    for i in range(n-1):
        expected[i, n-1+i] = 1
        expected[i, 2*n-2] = -1
    for i in range(n):
        expected[n-1+i, :n-1] = -lap[i, :n-1]
        expected[n-1+i, n-1+i] = -1
        expected[n-1+i, 2*n-1+i] = 1
        expected[2*n-1+i, n-1+i] = -1
        expected[2*n-1+i, 2*n-1+i] = -1
    expected[3*n-1:, 3*n-1:] = -.25*np.eye(2*n)
    for i, row in enumerate(actual):
        for j, value in enumerate(row):
            low, high = certifier.bounds(value)
            assert low <= Fraction.from_float(expected[i, j]) <= high


def test_linear_stable_region_proof_and_known_unstable_rejection(certifier):
    ctx, _ = certifier.backend()
    p, r = np.eye(3), np.eye(3)
    positive, lower_p = certifier.validate_p(ctx, p, r)
    assert positive['strict_positive'] and lower_p == Fraction(1, 3)
    good_j = certifier.fixed_matrix(ctx, -np.diag([1., 2., .5]))
    valid, lower_q = certifier.validate_q(ctx, p, good_j)
    assert valid['strict_positive'] and lower_q == 1
    region = certifier.proof_chain(.01, lower_p, lower_q)
    rho = Fraction(region['rho_exact_rational'])
    assert rho == lower_p * Fraction.from_float(.01)**2 / 2
    assert region['V'] == 'xi.T P xi (no factor one half)'
    bad_j = certifier.fixed_matrix(ctx, np.diag([1., -2., -.5]))
    failed, _ = certifier.validate_q(ctx, p, bad_j)
    assert not failed['strict_positive']
    singular_r = np.diag([1., 1., 0.])
    failed, lower_p = certifier.validate_p(ctx, p, singular_r)
    assert not failed['strict_positive'] and lower_p is None


def test_straddling_or_zero_row_margin_is_not_certified(certifier):
    ctx, _ = certifier.backend()
    matrix = [[ctx.mpf([-.1, .1]), ctx.mpf(0)], [ctx.mpf(0), ctx.mpf(1)]]
    result, lower = certifier.gershgorin(matrix)
    assert lower < 0 and not result['strict_positive']
    matrix[0][0] = ctx.mpf(0)
    result, lower = certifier.gershgorin(matrix)
    assert lower == 0 and not result['strict_positive']


def test_baseline_real_nonlinear_box_and_port_are_included(certifier):
    ctx, _ = certifier.backend()
    model = certifier.IntervalPowerGraph(synthetic_spec(2), ctx)
    p, r, _ = certifier.numerical_candidate(model)
    assert p is not None and r is not None
    positive, lower_p = certifier.validate_p(ctx, p, r)
    assert positive['strict_positive'] and lower_p > 0
    jacobian, _ = model.jacobian_box(1e-6)
    valid, lower_q = certifier.validate_q(ctx, p, jacobian)
    assert valid['strict_positive'] and lower_q > 0
    # A large linear learned pi port invalidates the same frozen candidate.
    port_model = certifier.IntervalPowerGraph(synthetic_spec(2, port_slope=1000.), ctx)
    bad, _ = port_model.jacobian_box(1e-6)
    invalid, _ = certifier.validate_q(ctx, p, bad)
    assert not invalid['strict_positive']
    difference = midpoint_array(certifier, bad) - midpoint_array(certifier, jacobian)
    assert difference[1, 1] > 900  # full supply term survived interval evaluation


def test_interval_box_derivatives_enclose_exact_baseline_cos_values(certifier):
    ctx, _ = certifier.backend()
    model = certifier.IntervalPowerGraph(synthetic_spec(2), ctx)
    jacobian, _ = model.jacobian_box(.1)
    # delta_1-delta_2=xi_0; pi_1 derivative=-cos(xi_0), with true cosine
    # bound evaluated independently through interval elementary operations.
    low, high = certifier.bounds(jacobian[1][0])
    cosine = -ctx.cos(certifier.iv_exact(ctx, .1))
    c_low, c_high = certifier.bounds(cosine)
    assert low <= -1 <= high
    assert low <= c_low <= c_high <= high


def test_exact_symmetry_dimension_and_degree_guards(certifier):
    ctx, _ = certifier.backend()
    p = np.eye(2)
    p[0, 1] = 1e-30
    with pytest.raises(ValueError, match='exactly symmetric'):
        certifier.validate_p(ctx, p, np.eye(2))
    source = synthetic_spec(2)
    source['functions']['port']['powers'][0][0] = 4
    with pytest.raises(ValueError, match='degree'):
        certifier.IntervalPowerGraph(source, ctx)


def test_authorization_precedes_model_read_and_output_creation(certifier, monkeypatch, tmp_path):
    def denied():
        raise PermissionError('gate')
    monkeypatch.setattr(certifier, 'check_authorized', denied)
    monkeypatch.setattr(certifier, 'strict_json_load', lambda path: pytest.fail('model read before authority'))
    output = tmp_path/'forbidden'
    with pytest.raises(PermissionError, match='gate'):
        certifier.certify('missing.json', output)
    assert not output.exists()


def test_deadline_guard(certifier):
    with pytest.raises(TimeoutError):
        certifier.check_deadline(0)


def test_backend_implementation_hashes_are_archived(certifier):
    before = certifier.backend_sources()
    assert before == certifier.backend_sources()
    for name in ('mpmath.ctx_iv', 'mpmath.libmp.libmpf', 'mpmath.libmp.libmpi', 'mpmath.libmp.libelefun'):
        assert len(before['files'][name]['sha256']) == 64


def test_complete_synthetic_certificate_is_revoked_after_backend_change(certifier, monkeypatch, tmp_path):
    source = tmp_path/'fixed.json'
    source.write_text(json.dumps(synthetic_spec(2)), encoding='utf-8')
    monkeypatch.setattr(certifier, 'check_authorized', lambda: None)
    monkeypatch.setattr(certifier, 'project_path', lambda path: type(source)(path))
    # pytest temp output is outside TASK4; replace only this test's writer,
    # while retaining strict JSON serialization. Production stays root-gated.
    monkeypatch.setattr(certifier, 'write_json',
                        lambda path, data: type(source)(path).write_text(
                            json.dumps(data, allow_nan=False), encoding='utf-8'))
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', str(time.time()+120))
    calls = []
    def changing_hashes():
        calls.append(None)
        return {'files': {'test-backend': {'path': 'synthetic', 'sha256': str(len(calls))}}}
    monkeypatch.setattr(certifier, 'backend_sources', changing_hashes)
    report = certifier.certify(source, tmp_path/'result')
    assert not report['L2_region_certificate']
    assert report['stop_reason'] == 'interval_backend_changed_or_unavailable'
    assert all(not row['certified'] for row in report['attempts'])
    assert not report['interval_backend_unchanged']
    assert len(report['attempts']) == len(certifier.RADII)
    loaded = json.loads((tmp_path/'result'/'certificate.json').read_text())
    assert loaded['status'] == 'inconclusive'
    with pytest.raises(FileExistsError):
        certifier.certify(source, tmp_path/'result')
