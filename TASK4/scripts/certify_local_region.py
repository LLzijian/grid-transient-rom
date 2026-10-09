"""Validate a quadratic local region of one fixed, complete explicit T4-S ODE.

No trajectories, event data, Torch predictor, FOM, or initializer are evaluated.
All JSON binary64 numbers are exact dyadic parameters of the continuous
mathematical ODE. Derived sin/cos/exp/sqrt constants are mathematical real
functions, enclosed with mpmath.iv. This does not certify floating point
integration or transfer a guarantee to the original higher order system.
"""
import _bootstrap  # noqa: F401

import argparse
from fractions import Fraction
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from task4.paths import project_path, write_json
from task4.symbolic_runtime import FUNCTION_NAMES, SymbolicPowerGraph
from task4.workflow import Workflow


RADII = (1e-6, 1e-5, 1e-4, 1e-3, .003, .01, .03, .1)
INTERVAL_DPS = 50
MAX_DIMENSION = 49
MAX_POLYNOMIAL_DEGREE = 3
MAX_POLYNOMIAL_ROWS = 256
MAX_INTERVAL_WORK_SECONDS = 900.0
MAX_ENDPOINT_ABS_EXPONENT = 8192
MAX_ENDPOINT_MANTISSA_BITS = 2048
CENTRAL_STEP = 5e-6
BACKEND_REFERENCE = 'https://mpmath.org/doc/current/contexts.html'


class UnsupportedIntervalBackend(RuntimeError):
    pass


def check_authorized():
    """Same managed gate as training, without importing Torch or event code."""
    run_id = os.environ.get('TASK4_MANAGED_RUN_ID')
    state = Workflow().load()
    if (not run_id or run_id not in state['active_jobs']
            or state['status'] != 'RUNNING' or state.get('pending_direction')):
        raise PermissionError('Region validation requires the authorized project.py runner')


def check_deadline(deadline):
    if time.time() + 3.0 >= deadline:
        raise TimeoutError('Managed region-validation save deadline reached')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_load(path):
    def bad(value):
        raise ValueError('Nonfinite JSON literal: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=bad,
                      object_pairs_hook=unique)


def exact_float(value):
    """A JSON number is interpreted exactly as its loaded binary64 value."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Finite binary64 parameter required')
    return Fraction.from_float(number)


def rational_text(value):
    value = Fraction(value)
    return str(value.numerator) + '/' + str(value.denominator)


def endpoint_fraction(raw):
    """Decode a finite libmp dyadic endpoint without a float round trip.

    This deliberately fails closed for unsupported endpoint formats. The four
    entries are sign, mantissa, exponent, and mantissa bit count. Nonfinite
    libmp sentinels have negative bit count; they must never enter a proof.
    """
    if not isinstance(raw, tuple) or len(raw) != 4:
        raise UnsupportedIntervalBackend('Unsupported interval endpoint representation')
    sign, mantissa, exponent, bits = raw
    if sign not in (0, 1) or bits < 0 or mantissa < 0:
        raise ValueError('Nonfinite or invalid interval endpoint')
    if (abs(exponent) > MAX_ENDPOINT_ABS_EXPONENT
            or bits > MAX_ENDPOINT_MANTISSA_BITS
            or int(mantissa).bit_length() > MAX_ENDPOINT_MANTISSA_BITS):
        raise UnsupportedIntervalBackend('Interval endpoint exceeds bounded serialization capability; no stability inference')
    if mantissa == 0:
        if bits != 0:
            raise ValueError('Invalid zero endpoint')
        return Fraction(0)
    numerator = -int(mantissa) if sign else int(mantissa)
    return Fraction(numerator << exponent) if exponent >= 0 else Fraction(numerator, 1 << -exponent)


def bounds(value):
    raw = getattr(value, '_mpi_', None)
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise UnsupportedIntervalBackend('Expected a real mpmath interval')
    low, high = endpoint_fraction(raw[0]), endpoint_fraction(raw[1])
    if low > high:
        raise ValueError('Reversed interval endpoints')
    return low, high


def serialize_interval(value):
    low, high = bounds(value)
    return {'lower_exact_rational': rational_text(low),
            'upper_exact_rational': rational_text(high)}


def interval_matrix_payload(matrix):
    return [[serialize_interval(value) for value in row] for row in matrix]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Fraction):
        return rational_text(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def backend():
    try:
        import mpmath
        from mpmath.ctx_iv import MPIntervalContext
    except ImportError as exc:
        raise UnsupportedIntervalBackend('Existing mpmath interval dependency unavailable; no installation attempted') from exc
    ctx = MPIntervalContext()
    ctx.dps = INTERVAL_DPS
    # Only documented real interval elementary operations are used. These
    # structural smoke checks are capability guards, not a library proof.
    try:
        zero, one = ctx.mpf(0), ctx.mpf(1)
        if bounds(zero) != (Fraction(0), Fraction(0)) or bounds(one) != (Fraction(1), Fraction(1)):
            raise UnsupportedIntervalBackend('Exact interval integer conversion failed')
        for value, target in ((ctx.sin(zero), 0), (ctx.cos(zero), 1),
                              (ctx.exp(zero), 1), (ctx.sqrt(one), 1)):
            low, high = bounds(value)
            if not low <= target <= high:
                raise UnsupportedIntervalBackend('Elementary interval capability check failed')
    except (AttributeError, NotImplementedError) as exc:
        raise UnsupportedIntervalBackend('Required real interval functions unavailable') from exc
    return ctx, mpmath.__version__


def iv_exact(ctx, value):
    value = exact_float(value) if not isinstance(value, Fraction) else value
    return ctx.mpf(value.numerator) / ctx.mpf(value.denominator)


def backend_sources():
    """Archive actual trusted interval implementation paths and byte hashes."""
    names = ('mpmath.ctx_iv', 'mpmath.libmp', 'mpmath.libmp.backend',
             'mpmath.libmp.libmpf', 'mpmath.libmp.libmpi',
             'mpmath.libmp.libelefun', 'mpmath.libmp.libintmath')
    result = {}
    for name in names:
        module = importlib.import_module(name)
        location = getattr(module, '__file__', None)
        if location is None or not Path(location).is_file():
            raise UnsupportedIntervalBackend('Cannot hash required interval backend implementation: ' + name)
        result[name] = {'path': str(Path(location).resolve()), 'sha256': sha256(location)}
    kernel = importlib.import_module('mpmath.libmp.backend')
    # A loaded GMP extension is part of the trusted arithmetic component when
    # mpmath selected it. Do not install/import a new optional dependency.
    for name in sorted(sys.modules):
        if name in ('gmpy2', 'gmpy') or name.startswith(('gmpy2.', 'gmpy.')):
            location = getattr(sys.modules[name], '__file__', None)
            if location is None or not Path(location).is_file():
                raise UnsupportedIntervalBackend('Cannot hash loaded optional arithmetic extension: ' + name)
            result[name] = {'path': str(Path(location).resolve()), 'sha256': sha256(location)}
    return {'selected_backend': str(getattr(kernel, 'BACKEND', 'unavailable')), 'files': result}


class AD:
    """Sparse first derivative with interval values, in quotient coordinates."""
    __slots__ = ('ctx', 'value', 'gradient')

    def __init__(self, ctx, value, gradient=None):
        self.ctx = ctx
        self.value = value
        self.gradient = {} if gradient is None else gradient

    def constant(self, value):
        if isinstance(value, AD):
            if value.ctx is not self.ctx:
                raise ValueError('Mixed interval contexts')
            return value
        return AD(self.ctx, value if hasattr(value, '_mpi_') else iv_exact(self.ctx, value))

    def __add__(self, other):
        other = self.constant(other)
        keys = self.gradient.keys() | other.gradient.keys()
        zero = self.ctx.mpf(0)
        return AD(self.ctx, self.value + other.value,
                  {key: self.gradient.get(key, zero) + other.gradient.get(key, zero) for key in keys})

    __radd__ = __add__

    def __neg__(self):
        return AD(self.ctx, -self.value, {key: -value for key, value in self.gradient.items()})

    def __sub__(self, other):
        return self + -self.constant(other)

    def __rsub__(self, other):
        return self.constant(other) + -self

    def __mul__(self, other):
        other = self.constant(other)
        keys = self.gradient.keys() | other.gradient.keys()
        zero = self.ctx.mpf(0)
        return AD(self.ctx, self.value * other.value,
                  {key: self.gradient.get(key, zero) * other.value
                   + self.value * other.gradient.get(key, zero) for key in keys})

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.constant(other)
        low, high = bounds(other.value)
        if low <= 0 <= high:
            raise ValueError('Interval division through zero')
        keys = self.gradient.keys() | other.gradient.keys()
        zero = self.ctx.mpf(0)
        return AD(self.ctx, self.value / other.value,
                  {key: (self.gradient.get(key, zero) * other.value
                         - self.value * other.gradient.get(key, zero)) / (other.value ** 2)
                   for key in keys})

    def __pow__(self, power):
        if type(power) is not int or power < 0:
            raise ValueError('AD powers must be nonnegative integers')
        if power == 0:
            return self.constant(1)
        factor = power * self.value ** (power - 1)
        return AD(self.ctx, self.value ** power,
                  {key: factor * value for key, value in self.gradient.items()})

    def sin(self):
        cosine = self.ctx.cos(self.value)
        return AD(self.ctx, self.ctx.sin(self.value),
                  {key: cosine * value for key, value in self.gradient.items()})

    def cos(self):
        negative_sine = -self.ctx.sin(self.value)
        return AD(self.ctx, self.ctx.cos(self.value),
                  {key: negative_sine * value for key, value in self.gradient.items()})


class IntervalPolynomial:
    def __init__(self, ctx, spec):
        self.ctx = ctx
        self.inputs, self.outputs = spec['inputs'], spec['outputs']
        self.rows = []
        for powers, coefficients in zip(spec['powers'], spec['coefficients']):
            if sum(powers) > MAX_POLYNOMIAL_DEGREE:
                raise ValueError('Polynomial degree exceeds registered interval scope')
            active = [(index, int(power)) for index, power in enumerate(powers) if power]
            self.rows.append((active, [iv_exact(ctx, value) for value in coefficients]))
        if len(self.rows) > MAX_POLYNOMIAL_ROWS:
            raise ValueError('Polynomial row count exceeds registered interval scope')

    def evaluate(self, values, derivative=None):
        if len(values) != self.inputs:
            raise ValueError('Polynomial input count differs from explicit JSON')
        zero = AD(self.ctx, self.ctx.mpf(0))
        output = [zero for _ in range(self.outputs)]
        for powers, coefficients in self.rows:
            multiplier = 1
            if derivative is not None:
                multiplier = next((power for index, power in powers if index == derivative), 0)
                if multiplier == 0:
                    continue
            atom = zero.constant(multiplier)
            for index, power in powers:
                exponent = power - (1 if derivative == index else 0)
                if exponent:
                    atom = atom * (values[index] ** exponent)
            for index, coefficient in enumerate(coefficients):
                if bounds(coefficient) != (Fraction(0), Fraction(0)):
                    output[index] = output[index] + atom * coefficient
        return output

    def value_gradient(self, values, active_inputs=None):
        count = self.inputs if active_inputs is None else active_inputs
        return self.evaluate(values), [self.evaluate(values, derivative=index) for index in range(count)]


class IntervalPowerGraph:
    """Full symbolic RHS including centered energies, skew, PSD and both ports."""
    def __init__(self, spec, ctx, *, deadline=float('inf')):
        # Check bounded polynomial scope before NumPy compiles any factors.
        if (type(spec.get('n')) is not int or type(spec.get('rank')) is not int
                or spec['n'] < 1 or spec['rank'] < 1
                or (3+spec['rank'])*spec['n']-1 > MAX_DIMENSION):
            raise ValueError('Positive integer dimensions within registered interval scope required')
        if set(spec.get('functions', {})) != set(FUNCTION_NAMES):
            raise ValueError('Exactly seven explicit polynomial functions required')
        for function in spec['functions'].values():
            if function.get('type') != 'polynomial':
                raise ValueError('Only complete explicit T4-S polynomials are supported')
            powers = function.get('powers')
            if not isinstance(powers, list) or not 0 < len(powers) <= MAX_POLYNOMIAL_ROWS:
                raise ValueError('Polynomial row count exceeds registered interval scope')
            for row in powers:
                if (not isinstance(row, list) or any(type(v) is not int or v < 0 for v in row)
                        or sum(row) > MAX_POLYNOMIAL_DEGREE):
                    raise ValueError('Polynomial degree exceeds registered interval scope')
        # Independent NumPy parser checks the entire export, including encoder
        # metadata, but neither the GRU nor any data-dependent initializer runs.
        self.reference = SymbolicPowerGraph(spec)
        self.ctx, self.deadline = ctx, deadline
        self.n, self.rank = self.reference.n, self.reference.rank
        self.dim, self.full_dim = self.reference.dim - 1, self.reference.dim
        if self.dim > MAX_DIMENSION:
            raise ValueError('Quotient dimension exceeds registered interval scope')
        self.sdim, self.m = self.reference.sdim, self.reference.m
        self.edges, self.pairs = self.reference.edges.tolist(), self.reference.pairs
        self.zero = AD(ctx, ctx.mpf(0))
        self.strength = iv_exact(ctx, spec['strength'])
        self.mass = [iv_exact(ctx, v) for v in spec['physics']['mass']]
        self.gain = [iv_exact(ctx, v) for v in spec['physics']['gain']]
        self.scale = np.concatenate((np.full(self.n - 1, self.reference.scales[0]),
                                     self.reference.mass * self.reference.scales[1],
                                     np.full(self.n, self.reference.scales[2]),
                                     np.ones(self.n * self.rank)))
        self.mapping_scale = [iv_exact(ctx, v) for v in self.scale]
        self.omega_scale, self.power_scale = [iv_exact(ctx, v) for v in spec['scales'][1:]]
        self.mu = [[self.zero.constant(v) for v in row] for row in spec['mu']]
        self.phase = [self.zero.constant(v) for v in spec['phase']]
        self.sin0, self.cos0 = [v.sin() for v in self.phase], [v.cos() for v in self.phase]
        self.coupling = [iv_exact(ctx, k) * ctx.exp(iv_exact(ctx, logk))
                         for k, logk in zip(spec['physics']['coupling'], spec['log_coupling'])]
        self.rho = [[ctx.sqrt(iv_exact(ctx, spec['physics']['damping'][i])),
                     ctx.sqrt(iv_exact(ctx, spec['physics']['rate'][i]) * self.gain[i])]
                    + [iv_exact(ctx, .5)] * self.rank for i in range(self.n)]
        self.fn = {name: IntervalPolynomial(ctx, spec['functions'][name]) for name in FUNCTION_NAMES}
        self.node_origin = []
        for i in range(self.n):
            _, gradient = self.fn['node_energy'].value_gradient([self.zero] * self.sdim + self.mu[i], self.sdim)
            self.node_origin.append([row[0] for row in gradient[:self.sdim]])
        self.edge_origin = [self.edge_energy_gradient(edge, self.zero, [self.zero] * self.sdim,
                                                      [self.zero] * self.sdim) for edge in range(len(self.edges))]
        self.port_origin = [self.fn['port'].evaluate([self.zero] * self.m + self.mu[i]) for i in range(self.n)]

    def edge_input(self, edge, eta, left, right, swap=False):
        i, j = self.edges[edge]
        s = (self.phase[edge] + eta).sin() - self.sin0[edge]
        c = (self.phase[edge] + eta).cos() - self.cos0[edge]
        if swap:
            return [-s, c] + right + left + self.mu[j] + self.mu[i]
        return [s, c] + left + right + self.mu[i] + self.mu[j]

    def edge_energy_gradient(self, edge, eta, left, right):
        _, ga = self.fn['edge_energy'].value_gradient(self.edge_input(edge, eta, left, right), 2+2*self.sdim)
        _, gb = self.fn['edge_energy'].value_gradient(self.edge_input(edge, eta, left, right, True), 2+2*self.sdim)
        ga, gb = [row[0] for row in ga], [row[0] for row in gb]
        angle = self.phase[edge] + eta
        geta = .5 * ((ga[0] - gb[0]) * angle.cos() - (ga[1] + gb[1]) * angle.sin())
        gleft = [.5 * (ga[2+k] + gb[2+self.sdim+k]) for k in range(self.sdim)]
        gright = [.5 * (ga[2+self.sdim+k] + gb[2+k]) for k in range(self.sdim)]
        return [geta] + gleft + gright

    def full_rhs(self, state):
        if len(state) != self.full_dim:
            raise ValueError('Wrong full state dimension')
        check_deadline(self.deadline)
        n = self.n
        delta, pi, power = state[:n], state[n:2*n], state[2*n:3*n]
        q = [state[3*n+i*self.rank:3*n+(i+1)*self.rank] for i in range(n)]
        omega = [pi[i] / self.mass[i] for i in range(n)]
        local = [[omega[i] / self.omega_scale, power[i] / self.power_scale] + q[i] for i in range(n)]
        stored = [[power[i] / self.power_scale] + q[i] for i in range(n)]
        node_gradient = []
        for i in range(n):
            check_deadline(self.deadline)
            _, gradient = self.fn['node_energy'].value_gradient(stored[i] + self.mu[i], self.sdim)
            node_gradient.append([gradient[k][0] - self.node_origin[i][k] for k in range(self.sdim)])
        gd = [self.zero for _ in range(n)]
        etas = []
        for edge, (i, j) in enumerate(self.edges):
            check_deadline(self.deadline)
            eta = delta[i] - delta[j]
            etas.append(eta)
            ge = self.edge_energy_gradient(edge, eta, stored[i], stored[j])
            ge = [value - origin for value, origin in zip(ge, self.edge_origin[edge])]
            base = ((self.phase[edge] + eta).sin() - self.sin0[edge]) * self.coupling[edge]
            angle_force = base + ge[0] * self.strength
            gd[i], gd[j] = gd[i] + angle_force, gd[j] - angle_force
            for k in range(self.sdim):
                node_gradient[i][k] = node_gradient[i][k] + ge[1+k]
                node_gradient[j][k] = node_gradient[j][k] + ge[1+self.sdim+k]
        effort = [[omega[i], power[i] / self.gain[i] + node_gradient[i][0] * self.strength / self.power_scale]
                  + [q[i][k] + node_gradient[i][1+k] * self.strength for k in range(self.rank)] for i in range(n)]
        force = [[self.zero for _ in range(self.m)] for _ in range(n)]
        for i in range(n):
            check_deadline(self.deadline)
            feat = local[i] + self.mu[i]
            skew = [value * self.strength for value in self.fn['node_exchange'].evaluate(feat)]
            skew[self.pairs.index((0, 2))] = skew[self.pairs.index((0, 2))] + self.strength
            for index, (a, b) in enumerate(self.pairs):
                force[i][a] = force[i][a] + skew[index] * effort[i][b]
                force[i][b] = force[i][b] - skew[index] * effort[i][a]
            ell = [value * self.strength + self.rho[i][k]
                   for k, value in enumerate(self.fn['node_dissipation'].evaluate(feat))]
            force[i] = [force[i][k] - ell[k] ** 2 * effort[i][k] for k in range(self.m)]
        for edge, (i, j) in enumerate(self.edges):
            check_deadline(self.deadline)
            forward = self.edge_input(edge, etas[edge], local[i], local[j])
            reverse = self.edge_input(edge, etas[edge], local[i], local[j], True)
            k1, k2 = self.fn['edge_exchange'].evaluate(forward), self.fn['edge_exchange'].evaluate(reverse)
            e1, e2 = self.fn['edge_dissipation'].evaluate(forward), self.fn['edge_dissipation'].evaluate(reverse)
            for a in range(self.m):
                for b in range(self.m):
                    kab = (k1[a*self.m+b] - k2[b*self.m+a]) * self.strength * .5
                    force[i][a] = force[i][a] + kab * effort[j][b]
                    force[j][b] = force[j][b] - kab * effort[i][a]
                le = (e1[a] + e2[a]) * self.strength * .5
                exchange = .5 * le ** 2 * (effort[i][a] - effort[j][a])
                force[i][a], force[j][a] = force[i][a] - exchange, force[j][a] + exchange
        pi_dot, p_dot, q_dot = [], [], []
        for i in range(n):
            ports = self.fn['port'].evaluate(local[i] + self.mu[i])
            ports = [(value - origin) * self.strength for value, origin in zip(ports, self.port_origin[i])]
            pi_dot.append(-gd[i] + force[i][0] + power[i] + ports[0])
            p_dot.append(force[i][1] + ports[1] - omega[i] * self.gain[i])
            q_dot.extend(force[i][2:])
        return omega + pi_dot + p_dot + q_dot

    def scaled_rhs(self, coordinates):
        if len(coordinates) != self.dim:
            raise ValueError('Wrong quotient state dimension')
        reduced = [value * scale for value, scale in zip(coordinates, self.mapping_scale)]
        full = reduced[:self.n-1] + [self.zero] + reduced[self.n-1:]
        rhs = self.full_rhs(full)
        projected = [rhs[i] - rhs[self.n-1] for i in range(self.n-1)] + rhs[self.n:]
        return [value / scale for value, scale in zip(projected, self.mapping_scale)]

    def jacobian_box(self, radius):
        radius = exact_float(radius)
        if radius < 0:
            raise ValueError('Radius must be nonnegative')
        # Hull exact singleton endpoints, retaining all binary64 radius bits.
        interval = self.ctx.mpf([iv_exact(self.ctx, -radius).a, iv_exact(self.ctx, radius).b])
        variables = [AD(self.ctx, interval, {i: self.ctx.mpf(1)}) for i in range(self.dim)]
        result = self.scaled_rhs(variables)
        zero = self.ctx.mpf(0)
        jacobian = [[row.gradient.get(i, zero) for i in range(self.dim)] for row in result]
        # Reject any nonfinite derivative enclosure before proof arithmetic.
        for row in jacobian:
            for value in row:
                bounds(value)
        return jacobian, [row.value for row in result]

    def coordinates_description(self):
        return {
            'full_dimension': self.full_dim, 'quotient_dimension': self.dim,
            'full_order': ['delta[N]', 'pi[N]', 'p[N]', 'q[N,rank] machine-major'],
            'quotient_order': ['delta_i-delta_last [N-1]', 'pi[N]', 'p[N]', 'q[N,rank]'],
            'lift': 'last delta=0; all pi/p/q retained', 'projection_lift_left_inverse': 'exact by algebra',
            'xi_definition': 'xi=y/s; each stored s is a fixed exact binary64 dyadic',
            'scale': self.scale.tolist(),
            'scale_exact_rationals': [rational_text(exact_float(v)) for v in self.scale],
            'pi_scale_construction': 'mass*saved TRAIN omega scale rounded once to binary64 and then fixed exactly',
            'scale_units': ['delta: rad', 'pi: mass unit * rad/s', 'p: system pu', 'q: latent unit'],
            'jacobian_unit': '1/s', 'time_unit': 's', 'fitting_performed': False,
        }


def interval_matmul(ctx, left, right, deadline=float('inf')):
    rows, middle, cols = len(left), len(right), len(right[0])
    if not rows or any(len(row) != middle for row in left) or any(len(row) != cols for row in right):
        raise ValueError('Interval matrix dimensions disagree')
    zero = ctx.mpf(0)
    out = []
    for i in range(rows):
        check_deadline(deadline)
        out.append([sum((left[i][k] * right[k][j] for k in range(middle)), zero) for j in range(cols)])
    return out


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def fixed_matrix(ctx, matrix):
    return [[iv_exact(ctx, value) for value in row] for row in matrix]


def gershgorin(matrix):
    """Exact-rational conservative eigenvalue lower bounds for symmetric sets.

    Callers supply interval enclosures of a mathematically symmetric matrix.
    Each row uses diagonal lower bound minus absolute off-diagonal upper bounds.
    All decisions here use exact integers; float diagnostics do not gate proof.
    """
    n = len(matrix)
    if n < 1 or any(len(row) != n for row in matrix):
        raise ValueError('Square interval matrix required')
    row_bounds = []
    for i in range(n):
        diagonal = bounds(matrix[i][i])[0]
        radius = sum((max(abs(v) for v in bounds(matrix[i][j])) for j in range(n) if j != i), Fraction(0))
        row_bounds.append(diagonal - radius)
    lower = min(row_bounds)
    return {'strict_positive': lower > 0, 'lower_exact_rational': rational_text(lower),
            'row_lower_exact_rationals': [rational_text(value) for value in row_bounds]}, lower


def validate_p(ctx, matrix_p, preconditioner, *, deadline=float('inf')):
    matrix_p, preconditioner = np.asarray(matrix_p, dtype=float), np.asarray(preconditioner, dtype=float)
    if (matrix_p.ndim != 2 or matrix_p.shape[0] != matrix_p.shape[1]
            or preconditioner.shape != matrix_p.shape or not np.isfinite(matrix_p).all()
            or not np.isfinite(preconditioner).all() or not np.array_equal(matrix_p, matrix_p.T)):
        raise ValueError('P must be exactly symmetric finite binary64; R must be matching finite square')
    pi, ri = fixed_matrix(ctx, matrix_p), fixed_matrix(ctx, preconditioner)
    c = interval_matmul(ctx, interval_matmul(ctx, ri, pi, deadline), transpose(ri), deadline)
    result, lower_c = gershgorin(c)
    # R's squared Frobenius norm is summed as exact rational binary64 squares.
    # This is an exact value and therefore also an upper bound; no BLAS norm.
    norm_squared = sum((exact_float(value) ** 2 for value in preconditioner.flat), Fraction(0))
    lower_p = lower_c / norm_squared if result['strict_positive'] and norm_squared > 0 else None
    result.update(preconditioned_matrix='C=R P R.T',
                  R_squared_frobenius_exact=rational_text(norm_squared),
                  P_eigenvalue_lower_exact=None if lower_p is None else rational_text(lower_p),
                  R_invertibility='C>0 implies square R full rank; no numerical invertibility assumption',
                  interval_matrix_C=interval_matrix_payload(c))
    return result, lower_p


def validate_q(ctx, matrix_p, jacobian, *, deadline=float('inf')):
    pi = fixed_matrix(ctx, matrix_p)
    pj = interval_matmul(ctx, pi, jacobian, deadline)
    q = [[-pj[i][j] - pj[j][i] for j in range(len(pj))] for i in range(len(pj))]
    result, lower_q = gershgorin(q)
    result.update(matrix='Q(xi)=-(P Jg(xi)+Jg(xi).T P)',
                  interval_matrix_Q=interval_matrix_payload(q))
    return result, lower_q


def numerical_candidate(model):
    """Independent NumPy finite differences propose P; never certify it."""
    d = model.dim
    scale = model.scale
    n = model.n
    def rhs(x):
        y = np.asarray(x, dtype=float) * scale
        full = np.concatenate((y[:, :n-1], np.zeros((len(y), 1)), y[:, n-1:]), axis=1)
        out = model.reference.field(full)
        return np.concatenate((out[:, :n-1] - out[:, n-1:n], out[:, n:]), axis=1) / scale
    probes = np.eye(d) * CENTRAL_STEP
    check_deadline(model.deadline)
    matrix_a = ((rhs(probes) - rhs(-probes)) / (2 * CENTRAL_STEP)).T
    if not np.isfinite(matrix_a).all():
        raise ValueError('Nonfinite numerical candidate Jacobian')
    eigenvalues = np.linalg.eigvals(matrix_a)
    diagnostics = {'central_step': CENTRAL_STEP, 'A': matrix_a.tolist(),
                   'spectral_abscissa_per_s': float(np.max(eigenvalues.real)),
                   'numeric_hurwitz': bool(np.all(eigenvalues.real < 0)),
                   'zero_probe_scaled_rhs': rhs(np.zeros((1, d)))[0].tolist(),
                   'numerical_origin_probe_is_not_exact_equilibrium_proof': True}
    if not diagnostics['numeric_hurwitz']:
        return None, None, diagnostics
    check_deadline(model.deadline)
    raw = solve_continuous_lyapunov(matrix_a.T, -np.eye(d))
    # Select each symmetric dyadic coefficient once; mirror exactly.
    p = np.empty_like(raw)
    for i in range(d):
        for j in range(i, d):
            value = float(.5 * raw[i, j] + .5 * raw[j, i])
            p[i, j] = p[j, i] = value
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite P candidate')
    try:
        r = np.linalg.solve(np.linalg.cholesky(p), np.eye(d))
    except np.linalg.LinAlgError:
        diagnostics['preconditioner_constructed'] = False
        return p, None, diagnostics
    diagnostics.update(preconditioner_constructed=True,
                       lyapunov_residual_frobenius=float(np.linalg.norm(matrix_a.T @ p + p @ matrix_a + np.eye(d))),
                       P_exact_binary64_coefficients=p.tolist(), R_exact_binary64_coefficients=r.tolist(),
                       P_exact_rationals=[[rational_text(exact_float(value)) for value in row] for row in p],
                       R_exact_rationals=[[rational_text(exact_float(value)) for value in row] for row in r],
                       candidate_fit='single continuous Lyapunov solve at fixed numerical origin Jacobian',
                       numerical_conditions_are_candidate_generation_only=True)
    diagnostics['fixed_candidate_parameter_sha256'] = hashlib.sha256(
        json.dumps({'P': diagnostics['P_exact_rationals'], 'R': diagnostics['R_exact_rationals']},
                   sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()
    return p, r, diagnostics


EQUILIBRIUM_PROOF = [
    'At delta=pi=p=q=0, every angle difference eta is exactly zero.',
    'Each sin(phase+eta)-sin(phase) and cos(phase+eta)-cos(phase) is identically zero at eta=0; the same mathematical phase functions occur in both terms.',
    'Node and symmetric edge energy gradients subtract their values at that identical zero state with fixed mu; base potential gradient, p/gain, q and pi/mass also vanish.',
    'All skew and squared dissipative forces multiply zero effort. Learned ports subtract that identical zero local input; fixed +p and -gain*omega vanish.',
    'Thus full f(0)=0 exactly, its common-angle gauge is exact, and quotient g(0)=0. An interval merely containing zero is not used to establish this identity.',
]


def proof_chain(radius, lower_p, lower_q):
    r = exact_float(radius)
    if lower_p <= 0 or lower_q <= 0 or r <= 0:
        raise ValueError('Strict positive proof constants required')
    rho = lower_p * r * r / 2
    return {
        'V': 'xi.T P xi (no factor one half)', 'rho_exact_rational': rational_text(rho),
        'V_and_rho_unit': 'formal dimensionless quadratic storage in the declared scaled coordinates; not SI energy',
        'm_P_exact_lower': rational_text(lower_p), 'm_Q_exact_lower_per_s': rational_text(lower_q),
        'Omega': '{xi:xi.T P xi<=rho}',
        'proof': [
            'Fixed exact symmetric P is positive definite by validated R P R.T>0; lambda_min(P)>=m_P.',
            'Omega is a compact convex ellipsoid containing the origin and is its entire connected component.',
            'On Omega, ||xi||_2^2<=rho/m_P=r^2/2, hence every |xi_i|<r and Omega lies strictly inside X=[-r,r]^d.',
            'The complete polynomial/trigonometric field is continuously differentiable. For xi in X, t*xi remains in X and g(xi)=integral_0^1 Jg(t*xi)xi dt by g(0)=0.',
            'Validated Q(X)>=m_Q I implies Vdot=2 xi.T P g(xi)<=-m_Q||xi||_2^2<0 for xi!=0 throughout X.',
            'On the nonzero boundary V=rho, strict decrease forbids outward crossing; compactness inside the smooth field domain gives forward invariance and global-in-time solutions within Omega.',
            'Strict Lyapunov decrease plus positive definiteness implies convergence to the origin for every initial quotient state in Omega.',
        ],
        'common_frequency_power_and_all_latent_states_retained': True,
        'claim_scope': 'local asymptotic stability and forward-invariant inner region for the continuous explicit mathematical ROM quotient only',
        'full_angle_space_interpretation': 'The common-angle saturation (a noncompact cylinder) is forward invariant; a fixed delta_last=0 lift slice is only a coordinate representative and need not be invariant. No asymptotic stability of the 50-dimensional point origin is asserted.',
        'excluded_claims': ['FOM guarantee', 'all observed fault states lie in this region',
                            'initializer error-set containment', 'float64 integrator verification',
                            'prediction accuracy', 'global stability', 'original 13-mode match'],
    }


def certify(symbolic, output):
    check_authorized()  # Before source/model read or creating any output.
    deadline_text = os.environ.get('TASK4_SAVE_DEADLINE_UNIX')
    if deadline_text is None or not math.isfinite(float(deadline_text)):
        raise RuntimeError('Managed finite save deadline required')
    deadline = min(float(deadline_text), time.time() + MAX_INTERVAL_WORK_SECONDS)
    out = project_path(output)
    out.mkdir(parents=True, exist_ok=False)
    source = project_path(symbolic)
    started = time.monotonic()
    report = {
        'status': 'inconclusive', 'complete': False, 'L2_region_certificate': False,
        'source_model_path': str(source), 'source_model_sha256': None,
        'role': 'post-hoc sufficient region validation of a fixed explicit T4-S; no model selection or fitting',
        'quadratic_candidate_is_full_P_not_H_plus_few_cross_terms': True,
        'radii_scaled_box': list(RADII), 'interval_dps': INTERVAL_DPS,
        'max_interval_work_seconds': MAX_INTERVAL_WORK_SECONDS,
        'serialization_capability': {'max_endpoint_absolute_binary_exponent': MAX_ENDPOINT_ABS_EXPONENT,
                                     'max_endpoint_mantissa_bits': MAX_ENDPOINT_MANTISSA_BITS,
                                     'exceeded_is_unsupported_not_instability': True},
        'source_mutation': False, 'event_payload_read': False, 'test_payload_read': False,
        'teacher_or_torch_used': False, 'new_fom_data_generated': False,
        'full_autonomous_field_including_all_ports': True, 'removed_field_terms': [],
        'finite_samples_used_as_proof': False, 'equilibrium_exact_algebra': EQUILIBRIUM_PROOF,
        'attempts': [], 'uncomputed_radii': list(RADII), 'stop_reason': 'started',
        'arithmetic_assurance': 'validated enclosure conditional on the correctness of mpmath.iv elementary interval arithmetic, not proof-assistant checked',
        'backend_reference': BACKEND_REFERENCE,
        'backend_documentation_note': 'The official documentation states the interval inclusion guarantee and notes that interval support is experimental; only basic arithmetic, integer powers, sin, cos, exp, sqrt are used.',
        'continuous_ode_parameter_interpretation': 'all loaded finite JSON numbers fixed as exact binary64 dyadics; sin/cos/exp/sqrt interpreted as mathematical real functions',
    }
    def persist():
        report['process_wall_seconds'] = time.monotonic() - started
        write_json(out / 'certificate.json', json_safe(report))
    persist()
    caught = None
    try:
        check_deadline(deadline)
        report['source_model_sha256'] = sha256(source)
        spec = strict_json_load(source)
        ctx, version = backend()
        report['mpmath_version'] = version
        report['interval_backend_sources_before'] = backend_sources()
        report['implementation_sha256'] = sha256(Path(__file__))
        runtime_source = Path(__file__).resolve().parents[1] / 'src/task4/symbolic_runtime.py'
        report['numpy_runtime_source_sha256'] = sha256(runtime_source)
        model = IntervalPowerGraph(spec, ctx, deadline=deadline)
        report['coordinates'] = model.coordinates_description()
        report['source_sha256_after_load'] = sha256(source)
        if report['source_sha256_after_load'] != report['source_model_sha256']:
            raise ValueError('Fixed symbolic source changed during loading')
        persist()
        p, r, diagnostic = numerical_candidate(model)
        report['quadratic_candidate'] = diagnostic
        if p is None or r is None:
            report.update(status='not_certified', complete=True, stop_reason='numerical_quadratic_candidate_unavailable')
        else:
            check_deadline(deadline)
            positive, lower_p = validate_p(ctx, p, r, deadline=deadline)
            report['validated_P'] = positive
            persist()
            if lower_p is None:
                report.update(status='not_certified', complete=True, stop_reason='P_positive_definiteness_not_validated')
            else:
                # Validate every registered radius even after one succeeds.
                for index, radius in enumerate(RADII):
                    check_deadline(deadline)
                    attempt = {'radius_scaled_box': radius, 'radius_exact_rational': rational_text(exact_float(radius)),
                               'complete': False, 'certified': False, 'stop_reason': 'started'}
                    report['attempts'].append(attempt)
                    persist()
                    try:
                        jacobian, field = model.jacobian_box(radius)
                        path = out / f'radius_{index+1:02d}_intervals.json'
                        payload = {'radius_exact_rational': attempt['radius_exact_rational'],
                                   'scaled_field_enclosure': [serialize_interval(value) for value in field],
                                   'jacobian_enclosure': interval_matrix_payload(jacobian)}
                        write_json(path, payload)
                        attempt['jacobian_artifact'] = path.name
                        check_deadline(deadline)
                        q_validation, lower_q = validate_q(ctx, p, jacobian, deadline=deadline)
                        payload['Q_validation'] = q_validation
                        write_json(path, payload)
                        attempt.update(complete=True, certified=bool(q_validation['strict_positive']),
                                       Q_gershgorin_lower_exact=q_validation['lower_exact_rational'],
                                       stop_reason='certified' if q_validation['strict_positive'] else 'interval_sufficient_condition_unresolved')
                        if attempt['certified']:
                            attempt['region'] = proof_chain(radius, lower_p, lower_q)
                    except BaseException as exc:
                        attempt.update(stop_reason=type(exc).__name__ + ': ' + str(exc))
                        raise
                    finally:
                        report['uncomputed_radii'] = [v for j, v in enumerate(RADII)
                                                     if j >= len(report['attempts']) or not report['attempts'][j]['complete']]
                        persist()
                report.update(complete=True, status='certified' if any(row['certified'] for row in report['attempts']) else 'not_certified',
                              stop_reason='fixed_radius_validation_complete')
                report['L2_region_certificate'] = any(row['certified'] for row in report['attempts'])
        check_deadline(deadline)
        report['source_sha256_after_validation'] = sha256(source)
        report['source_unchanged'] = report['source_sha256_after_validation'] == report['source_model_sha256']
        if not report['source_unchanged']:
            report.update(status='inconclusive', complete=False, L2_region_certificate=False, stop_reason='source_model_changed')
    except UnsupportedIntervalBackend as exc:
        report.update(status='unsupported', complete=False, stop_reason=str(exc))
    except TimeoutError as exc:
        report.update(status='inconclusive', complete=False, L2_region_certificate=False, stop_reason='save_deadline', limitation=str(exc))
    except BaseException as exc:
        caught = exc
        report.update(status='inconclusive', complete=False, L2_region_certificate=False,
                      stop_reason=type(exc).__name__ + ': ' + str(exc))
    finally:
        # Final hash check also applies to interrupted/failed runs. A partial
        # positive radius does not become a certificate after a source change.
        if report['source_model_sha256'] is not None:
            try:
                report['source_sha256_final'] = sha256(source)
                report['source_unchanged'] = report['source_sha256_final'] == report['source_model_sha256']
            except OSError:
                report['source_unchanged'] = False
            report['source_mutation'] = not report['source_unchanged']
            if not report['source_unchanged']:
                report.update(status='inconclusive', complete=False, L2_region_certificate=False,
                              stop_reason='source_model_changed_or_unavailable')
                for attempt in report['attempts']:
                    attempt['certified'] = False
        if 'interval_backend_sources_before' in report:
            try:
                report['interval_backend_sources_after'] = backend_sources()
                report['interval_backend_unchanged'] = (report['interval_backend_sources_after']
                                                       == report['interval_backend_sources_before'])
            except BaseException as exc:
                report['interval_backend_unchanged'] = False
                report['backend_final_hash_error'] = type(exc).__name__ + ': ' + str(exc)
            if not report['interval_backend_unchanged']:
                report.update(status='inconclusive', complete=False, L2_region_certificate=False,
                              stop_reason='interval_backend_changed_or_unavailable')
                for attempt in report['attempts']:
                    attempt['certified'] = False
        report['limitations'] = [
            'An unresolved interval bound means not certified; it is not a proof of instability.',
            'Certificates cover the continuous explicit mathematical ROM quotient, with the stated exact parameter interpretation.',
            'No trajectory accuracy, FOM compatibility/guarantee, initializer uncertainty containment, or discrete floating point solver guarantee is inferred.',
            'All common frequency, p and latent q coordinates are retained; only uniform angle gauge is removed.',
            'The arithmetic library remains a trusted component; this result is not a formal proof assistant certificate.',
        ]
        persist()
    if caught is not None:
        raise caught
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbolic', required=True)
    parser.add_argument('--output', required=True)
    report = certify(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ('status', 'complete', 'L2_region_certificate', 'stop_reason', 'process_wall_seconds')}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
