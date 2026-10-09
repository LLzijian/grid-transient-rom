"""Independent NumPy evaluator for the polynomial-only T4-S outer structure.

Only ``export_symbolic`` imports training code. Loading JSON, the shared GRU
initializer, analytic energy gradients, and the autonomous field need NumPy and
the standard library only. The exported GRU is used once at t=0.1 s; no neural
function or teacher is evaluated by the subsequent ODE. This is an executable
representation, not evidence that a compressed candidate predicts accurately or
has a certified stability region.
"""
from itertools import combinations
import json
from pathlib import Path

import numpy as np


FUNCTION_NAMES = (
    'node_energy', 'edge_energy', 'node_exchange', 'node_dissipation',
    'edge_exchange', 'edge_dissipation', 'port',
)
FORMAT = 'task4-polynomial-powergraph'


def _array(value, shape, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'Invalid {name}: expected finite array of shape {shape}')
    return out.copy()


class _Polynomial:
    """Vector polynomial and its analytic derivative, with no Torch dependency."""

    def __init__(self, spec, inputs, outputs):
        if spec.get('type') != 'polynomial':
            raise ValueError('Only explicit polynomial functions are supported')
        if spec.get('inputs') != inputs or spec.get('outputs') != outputs:
            raise ValueError('Polynomial input/output dimensions disagree with structure')
        pw = np.asarray(spec['powers'], dtype=np.float64)
        if (pw.ndim != 2 or pw.shape[1] != inputs or len(pw) == 0
                or not np.isfinite(pw).all() or (pw < 0).any()
                or (pw != np.floor(pw)).any() or (pw > 10000).any()):
            raise ValueError('Polynomial powers must be nonnegative bounded integers')
        self.powers = pw.astype(np.int64)
        self.coefficients = _array(spec['coefficients'], (len(pw), outputs), 'coefficients')
        self.inputs, self.outputs = inputs, outputs
        self._indices, self._mask = self._compile_factors(self.powers)
        self._derivatives = []
        for d in range(inputs):
            rows = np.flatnonzero(self.powers[:, d] > 0)
            derivative = self.powers[rows].copy()
            multiplier = derivative[:, d].copy()
            derivative[:, d] -= 1
            indices, mask = self._compile_factors(derivative)
            self._derivatives.append((rows, multiplier, indices, mask))

    @staticmethod
    def _compile_factors(powers):
        degree = powers.sum(axis=-1)
        width = max(1, int(degree.max()) if len(degree) else 0)
        indices = np.zeros((len(powers), width), dtype=np.int64)
        mask = np.zeros((len(powers), width), dtype=np.bool_)
        for row, exponents in enumerate(powers):
            variables = np.repeat(np.arange(powers.shape[1], dtype=np.int64), exponents)
            indices[row, :len(variables)] = variables
            mask[row, :len(variables)] = True
        return indices, mask

    @staticmethod
    def _factor_product(x, indices, mask):
        factors = np.where(mask, x[..., indices], 1.0)
        atoms = factors[..., 0]
        for position in range(1, factors.shape[-1]):
            atoms = atoms * factors[..., position]
        return atoms

    def value(self, x):
        return self._factor_product(x, self._indices, self._mask) @ self.coefficients

    def value_gradient(self, x):
        value = self.value(x)
        grad = np.empty((*x.shape[:-1], self.outputs, self.inputs), dtype=np.float64)
        # Only monomials depending on this variable contribute to its derivative.
        for d, (rows, multiplier, indices, mask) in enumerate(self._derivatives):
            if len(rows) == 0:
                grad[..., d] = 0.0
                continue
            atoms = self._factor_product(x, indices, mask) * multiplier
            grad[..., d] = atoms @ self.coefficients[rows]
        return value, grad


def export_symbolic(model, encoder, path=None, *, omega_s=None, metadata=None):
    """Return/save complete explicit JSON; reject any uncompressed RHS function.

    ``omega_s`` is optional metadata because PowerGraph does not retain it; pass
    the source Physics value when available. Input histories are already in rad/s
    and no guessed base frequency is used by this evaluator. ``metadata`` should
    identify the candidate, source data split, and subsequent validation report.
    """
    from .dynamics import PowerGraph
    from .encoder import HistoryEncoder
    from .symbolic import PolynomialFunction

    if type(model) is not PowerGraph:
        raise ValueError('Only the exact structured PowerGraph is exportable')
    if type(encoder) is not HistoryEncoder or (encoder.n, encoder.rank) != (model.n, model.rank):
        raise ValueError('Expected a matching shared HistoryEncoder')
    functions = model.function_modules()
    unsupported = [name for name in FUNCTION_NAMES if type(functions[name]) is not PolynomialFunction]
    if unsupported:
        raise ValueError('Uncompressed/unsupported functions: ' + ', '.join(unsupported))
    gru = encoder.gru
    if (gru.num_layers != 1 or gru.bidirectional or gru.input_size != 9
            or not gru.bias or not gru.batch_first):
        raise ValueError('Only the shared one-layer, biased, forward GRU(9, hidden) is supported')
    if omega_s is not None and (not np.isfinite(omega_s) or omega_s <= 0):
        raise ValueError('omega_s must be positive and finite when provided')

    def values(value):
        return value.detach().cpu().tolist()

    physics_keys = ('mass', 'incidence', 'delta_star', 'pm_star', 'coupling', 'damping', 'rate', 'gain')
    spec = {
        'format': FORMAT, 'version': 1, 'n': model.n, 'rank': model.rank,
        'strength': model.strength,
        'physics': {key: values(getattr(model, key)) for key in physics_keys},
        'omega_s': None if omega_s is None else float(omega_s),
        'scales': values(model.scales), 'log_coupling': values(model.log_coupling),
        # Record the actual buffers rather than silently recomputing features.
        'mu': values(model.mu), 'phase': values(model.phase), 'edges': values(model.edges),
        'functions': {name: functions[name].export() for name in FUNCTION_NAMES},
        'encoder': {
            'type': 'shared_gru', 'history_samples': 11, 'history_seconds': 0.1,
            'sample_dt_seconds': 0.01, 'gate_order': ['reset', 'update', 'new'],
            'new_gate_equation': 'tanh(input_new + reset * hidden_new)',
            'hidden_size': gru.hidden_size,
            **{name: values(getattr(encoder, name)) for name in ('mass', 'adjacency', 'scales', 'mu')},
            **{name: values(getattr(gru, name)) for name in
               ('weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0')},
            'head_weight': values(encoder.head.weight), 'head_bias': values(encoder.head.bias),
            'centering': 'raw(history) - raw(zeros_like(history)); mu retained in both',
            'deployment_use': 'once at t=0.1 s; no future observations or latent resets',
        },
        'coordinates': {
            'state_order': ['delta[N]', 'pi[N]', 'p[N]', 'q[N,rank] flattened machine-major'],
            'history_order': ['delta deviation', 'omega deviation', 'p deviation'],
            'time_unit': 's', 'delta_unit': 'rad', 'omega_unit': 'rad/s',
            'p_unit': 'system per-unit', 'pi_definition': 'mass * omega',
            'pi_unit': 'mass unit * rad/s',
            'q_unit': 'latent storage coordinate; numerical coefficient of q^2/2 is 1',
            'H_unit': 'formal model storage; SI joule calibration is not asserted',
            'physics_unit_convention': {
                'mass': 'p * s^2 / rad, from inertia / omega_s',
                'coupling': 'system per-unit', 'pm_star': 'system per-unit',
                'delta_star': 'rad', 'damping': 'p * s / rad',
                'rate': '1/s', 'gain': 'p / rad', 'incidence': 'dimensionless signed incidence',
            },
            'scale_units': ['rad', 'rad/s', 'system per-unit'],
            'mu': 'log of numerical [mass, max(damping,1e-8), rate, gain] in source units',
            'function_inputs': {
                'node_energy': '[p/scale_p, q..., mu...]',
                'edge_energy': '[sin(phase+eta)-sin(phase), cos(phase+eta)-cos(phase), s_i..., s_j..., mu_i..., mu_j...]',
                'node_exchange': '[omega/scale_omega, p/scale_p, q..., mu...]',
                'node_dissipation': '[omega/scale_omega, p/scale_p, q..., mu...]',
                'edge_exchange': '[sin difference, cos difference, local_i..., local_j..., mu_i..., mu_j...]',
                'edge_dissipation': '[sin difference, cos difference, local_i..., local_j..., mu_i..., mu_j...]',
                'port': '[omega/scale_omega, p/scale_p, q..., mu...]',
            },
            'coefficient_convention': 'F_o(v)=sum_a coefficients[a,o]*product_d v[d]**powers[a,d]; coefficients use the declared numerical coordinates and outer equations',
        },
        'structure': {
            'kinematics': 'delta_dot=pi/mass',
            'energy': 'sum(pi^2/(2*mass))+sum(k*(cos(phase)-cos(phase+eta)-sin(phase)*eta))+sum(p^2/(2*gain))+sum(q^2/2)+strength*(sum(centered_node)+sum(centered_symmetric_edge))',
            'centering': 'F(v)-F(0)-gradient(F)(0) dot v; physical mu is held fixed',
            'edge_symmetry': 'average F([sin_diff,cos_diff,s_i,s_j,mu_i,mu_j]) and F([-sin_diff,cos_diff,s_j,s_i,mu_j,mu_i]) before centering in [eta,s_i,s_j]',
            'exchange': 'node skew pair matrix; edge K=strength/2*(F(forward)-F(reverse)^T)',
            'node_exchange_offset': 'add strength to pair (pi,q_0)',
            'dissipation': 'node (rho0+strength*F)^2; edge (strength/2*(F(forward)+F(reverse)))^2/2 on effort difference',
            'ports': 'two fixed directions pi,p; learned F(local,mu)-F(0,mu); fixed +p in pi_dot and -gain*omega in p_dot',
            'claims': 'power construction only; no automatic Poisson, global stability, accuracy, speed, or region certificate',
        },
        'metadata': {} if metadata is None else metadata,
    }
    # Validate the export through the independent implementation, before writing.
    SymbolicPowerGraph(spec)
    encoded = json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False)
    if path is not None:
        Path(path).write_text(encoded + '\n', encoding='utf-8')
    return spec


class SymbolicPowerGraph:
    """Pure NumPy polynomial H/J/R model with optional leading batch dimensions."""

    def __init__(self, spec):
        if spec.get('format') != FORMAT or spec.get('version') != 1:
            raise ValueError('Unsupported symbolic format/version')
        self.n, self.rank = int(spec['n']), int(spec['rank'])
        if self.n < 1 or self.rank < 1 or self.n != spec['n'] or self.rank != spec['rank']:
            raise ValueError('n and rank must be positive integers')
        self.dim = (3 + self.rank) * self.n
        self.m, self.sdim = 2 + self.rank, 1 + self.rank
        self.strength = float(spec['strength'])
        if not np.isfinite(self.strength):
            raise ValueError('Nonfinite strength')
        physics = spec['physics']
        incidence = np.asarray(physics['incidence'], dtype=np.float64)
        if incidence.ndim != 2 or incidence.shape[0] != self.n:
            raise ValueError('Invalid incidence')
        self.ne = incidence.shape[1]
        self.incidence = _array(incidence, (self.n, self.ne), 'incidence')
        ends = []
        for col in self.incidence.T:
            plus, minus = np.flatnonzero(col == 1), np.flatnonzero(col == -1)
            if len(plus) != 1 or len(minus) != 1 or np.count_nonzero(col) != 2:
                raise ValueError('Expected signed two-endpoint incidence')
            ends.append([int(plus[0]), int(minus[0])])
        self.edges = np.asarray(ends, dtype=np.int64).reshape(self.ne, 2)
        if not np.array_equal(np.asarray(spec['edges']).reshape(self.ne, 2), self.edges):
            raise ValueError('Edges disagree with signed incidence')
        for key in ('mass', 'delta_star', 'pm_star', 'damping', 'rate', 'gain'):
            setattr(self, key, _array(physics[key], (self.n,), key))
        if ((self.mass <= 0).any() or (self.rate <= 0).any()
                or (self.gain <= 0).any() or (self.damping < 0).any()):
            raise ValueError('Invalid physical parameters')
        self.coupling = _array(physics['coupling'], (self.ne,), 'coupling')
        self.log_coupling = _array(spec['log_coupling'], (self.ne,), 'log_coupling')
        self.scales = _array(spec['scales'], (3,), 'scales')
        if (self.scales <= 0).any():
            raise ValueError('Scales must be positive')
        self.mu = _array(spec['mu'], (self.n, 4), 'mu')
        self.phase = _array(spec['phase'], (self.ne,), 'phase')
        self.pairs = list(combinations(range(self.m), 2))
        sizes = {
            'node_energy': (self.sdim + 4, 1),
            'edge_energy': (2 + 2 * self.sdim + 8, 1),
            'node_exchange': (self.m + 4, len(self.pairs)),
            'node_dissipation': (self.m + 4, self.m),
            'edge_exchange': (2 + 2 * self.m + 8, self.m * self.m),
            'edge_dissipation': (2 + 2 * self.m + 8, self.m),
            'port': (self.m + 4, 2),
        }
        if set(spec['functions']) != set(FUNCTION_NAMES):
            raise ValueError('Expected exactly the seven polynomial functions')
        for name, (inputs, outputs) in sizes.items():
            setattr(self, name, _Polynomial(spec['functions'][name], inputs, outputs))
        self._hn0, gn0 = self.node_energy.value_gradient(self._node_input(np.zeros((self.n, self.sdim))))
        self._hn0 = self._hn0[..., 0]
        self._gn0 = gn0[..., 0, :self.sdim]
        self._he0, self._ge0 = self._edge_energy_value_gradient(np.zeros((self.ne, 1 + 2 * self.sdim)))
        self._port0 = self.port.value(self._node_input(np.zeros((self.n, self.m))))
        self._load_encoder(spec['encoder'])

    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def _state(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim < 1 or x.shape[-1] != self.dim or not np.isfinite(x).all():
            raise ValueError(f'Expected finite state [...,{self.dim}]')
        return x

    def split(self, x):
        n = self.n
        return x[..., :n], x[..., n:2*n], x[..., 2*n:3*n], x[..., 3*n:].reshape(*x.shape[:-1], n, self.rank)

    def assemble(self, d, pi, p, q):
        return np.concatenate([d, pi, p, q.reshape(*q.shape[:-2], self.n*self.rank)], axis=-1)

    def physical(self, x):
        d, pi, p, _ = self.split(self._state(x))
        return np.stack([d, pi / self.mass, p], axis=-1)

    def from_physical(self, y, q):
        y, q = np.asarray(y, dtype=np.float64), np.asarray(q, dtype=np.float64)
        if y.shape[-2:] != (self.n, 3) or q.shape != (*y.shape[:-2], self.n, self.rank):
            raise ValueError('Physical and latent shapes disagree')
        return self._state(self.assemble(y[..., 0], y[..., 1] * self.mass, y[..., 2], q))

    def _node_input(self, v):
        return np.concatenate([v, np.broadcast_to(self.mu, (*v.shape[:-1], 4))], axis=-1)

    def _edge_input(self, eta, v, swap=False):
        i, j = self.edges.T
        si, sj = v[..., i, :], v[..., j, :]
        mu = np.broadcast_to(self.mu, (*v.shape[:-1], 4))
        mi, mj = mu[..., i, :], mu[..., j, :]
        s = np.sin(self.phase + eta) - np.sin(self.phase)
        c = np.cos(self.phase + eta) - np.cos(self.phase)
        if swap:
            return np.concatenate([-s[..., None], c[..., None], sj, si, mj, mi], axis=-1)
        return np.concatenate([s[..., None], c[..., None], si, sj, mi, mj], axis=-1)

    def _edge_energy_value_gradient(self, v):
        eta, si, sj = v[..., 0], v[..., 1:1+self.sdim], v[..., 1+self.sdim:]
        i, j = self.edges.T
        mi = np.broadcast_to(self.mu[i], (*v.shape[:-1], 4))
        mj = np.broadcast_to(self.mu[j], (*v.shape[:-1], 4))
        s = np.sin(self.phase + eta) - np.sin(self.phase)
        c = np.cos(self.phase + eta) - np.cos(self.phase)
        a = np.concatenate([s[..., None], c[..., None], si, sj, mi, mj], axis=-1)
        b = np.concatenate([-s[..., None], c[..., None], sj, si, mj, mi], axis=-1)
        va, ga = self.edge_energy.value_gradient(a)
        vb, gb = self.edge_energy.value_gradient(b)
        ga, gb = ga[..., 0, :], gb[..., 0, :]
        geta = (ga[..., 0] - gb[..., 0]) * np.cos(self.phase + eta)
        geta -= (ga[..., 1] + gb[..., 1]) * np.sin(self.phase + eta)
        gsi = ga[..., 2:2+self.sdim] + gb[..., 2+self.sdim:2+2*self.sdim]
        gsj = ga[..., 2+self.sdim:2+2*self.sdim] + gb[..., 2:2+self.sdim]
        return 0.5 * (va[..., 0] + vb[..., 0]), 0.5 * np.concatenate([geta[..., None], gsi, gsj], axis=-1)

    def energy_gradient(self, x):
        """Return H and dH/dx analytically, including both centering derivatives."""
        d, pi, p, q = self.split(self._state(x))
        eta = d @ self.incidence
        k = self.coupling * np.exp(self.log_coupling)
        h = 0.5 * np.sum(pi*pi/self.mass, axis=-1)
        h += np.sum(k * (np.cos(self.phase) - np.cos(self.phase + eta) - np.sin(self.phase)*eta), axis=-1)
        h += 0.5 * np.sum(p*p/self.gain, axis=-1) + 0.5 * np.sum(q*q, axis=(-1, -2))
        gd = (k * (np.sin(self.phase + eta) - np.sin(self.phase))) @ self.incidence.T
        sn = np.concatenate([p[..., None]/self.scales[2], q], axis=-1)
        hn, gn = self.node_energy.value_gradient(self._node_input(sn))
        hn = hn[..., 0] - self._hn0 - np.sum(self._gn0 * sn, axis=-1)
        gs = gn[..., 0, :self.sdim] - self._gn0
        i, j = self.edges.T
        ve = np.concatenate([eta[..., None], sn[..., i, :], sn[..., j, :]], axis=-1)
        he, ge = self._edge_energy_value_gradient(ve)
        he = he - self._he0 - np.sum(self._ge0 * ve, axis=-1)
        ge = ge - self._ge0
        for edge, (ii, jj) in enumerate(self.edges):
            gs[..., ii, :] += ge[..., edge, 1:1+self.sdim]
            gs[..., jj, :] += ge[..., edge, 1+self.sdim:]
        h += self.strength * (np.sum(hn, axis=-1) + np.sum(he, axis=-1))
        gd += self.strength * (ge[..., 0] @ self.incidence.T)
        gp = p/self.gain + self.strength*gs[..., 0]/self.scales[2]
        gq = q + self.strength*gs[..., 1:]
        return h, self.assemble(gd, pi/self.mass, gp, gq)

    def energy(self, x):
        return self.energy_gradient(x)[0]

    def field(self, x, details=False):
        """Autonomous field; only x is consumed after one history initialization."""
        x = self._state(x)
        h, gradient = self.energy_gradient(x)
        d, pi, p, q = self.split(x)
        ed, ew, ep, eq = self.split(gradient)
        w, eta = pi/self.mass, d @ self.incidence
        local = np.concatenate([w[..., None]/self.scales[1], p[..., None]/self.scales[2], q], axis=-1)
        feat = self._node_input(local)
        effort = np.concatenate([ew[..., None], ep[..., None], eq], axis=-1)
        skew = self.strength * self.node_exchange.value(feat)
        skew[..., self.pairs.index((0, 2))] += self.strength
        force = np.zeros_like(effort)
        for a, (i, j) in enumerate(self.pairs):
            force[..., i] += skew[..., a] * effort[..., j]
            force[..., j] -= skew[..., a] * effort[..., i]
        rho0 = np.concatenate([np.sqrt(self.damping)[:, None], np.sqrt(self.rate*self.gain)[:, None], np.full((self.n, self.rank), 0.5)], axis=-1)
        ell = rho0 + self.strength * self.node_dissipation.value(feat)
        force -= ell**2 * effort
        diss = np.sum(ell**2 * effort**2, axis=(-1, -2))
        fi, fj = self._edge_input(eta, local), self._edge_input(eta, local, True)
        k1 = self.edge_exchange.value(fi).reshape(*fi.shape[:-1], self.m, self.m)
        k2 = self.edge_exchange.value(fj).reshape(*fj.shape[:-1], self.m, self.m)
        k = 0.5 * self.strength * (k1 - np.swapaxes(k2, -1, -2))
        i, j = self.edges.T
        ei, ej = effort[..., i, :], effort[..., j, :]
        di = np.einsum('...eab,...eb->...ea', k, ej)
        dj = -np.einsum('...eba,...eb->...ea', k, ei)
        le = 0.5 * self.strength * (self.edge_dissipation.value(fi) + self.edge_dissipation.value(fj))
        diff = ei - ej
        di -= 0.5 * le**2 * diff
        dj += 0.5 * le**2 * diff
        for edge, (ii, jj) in enumerate(self.edges):
            force[..., ii, :] += di[..., edge, :]
            force[..., jj, :] += dj[..., edge, :]
        diss += 0.5 * np.sum(le**2 * diff**2, axis=(-1, -2))
        ports = self.strength * (self.port.value(feat) - self._port0)
        ud, up = ports[..., 0], ports[..., 1] - self.gain*w
        f = self.assemble(w, -ed + force[..., 0] + p + ud, force[..., 1] + up, force[..., 2:])
        if not details:
            return f
        supply = np.sum(ew*(p + ud) + ep*up, axis=-1)
        return f, {'H': h, 'gradient': gradient, 'Hdot': np.sum(gradient*f, axis=-1),
                   'dissipation': diss, 'supply': supply, 'expected_Hdot': supply-diss,
                   'port_pi': ud, 'port_p': up}

    def _load_encoder(self, enc):
        if (enc.get('type') != 'shared_gru' or enc.get('history_samples') != 11
                or enc.get('gate_order') != ['reset', 'update', 'new']
                or enc.get('history_seconds') != 0.1 or enc.get('sample_dt_seconds') != 0.01):
            raise ValueError('Unsupported initializer contract')
        size = int(enc['hidden_size'])
        if size < 1 or size != enc['hidden_size']:
            raise ValueError('Invalid GRU hidden size')
        self.encoder_hidden = size
        shapes = {
            'mass': (self.n,), 'adjacency': (self.n, self.n), 'scales': (3,), 'mu': (self.n, 4),
            'weight_ih_l0': (3*size, 9), 'weight_hh_l0': (3*size, size),
            'bias_ih_l0': (3*size,), 'bias_hh_l0': (3*size,),
            'head_weight': (self.rank, size), 'head_bias': (self.rank,),
        }
        self.enc = {key: _array(enc[key], shape, 'encoder.'+key) for key, shape in shapes.items()}
        if (self.enc['mass'] <= 0).any() or (self.enc['scales'] <= 0).any():
            raise ValueError('Invalid encoder masses/scales')

    def _history_features(self, history):
        d, w = history[..., 0], history[..., 1]
        mass, adjacency, scales = self.enc['mass'], self.enc['adjacency'], self.enc['scales']
        dc = d - np.sum(d*mass, axis=-1, keepdims=True)/mass.sum()
        common = np.sum(w*mass, axis=-1, keepdims=True)/mass.sum()
        nd, nw = d @ adjacency.T - d, w @ adjacency.T - w
        f = np.stack([dc/scales[0], w/scales[1], nd/scales[0], nw/scales[1],
                      np.broadcast_to(common, w.shape)/scales[1]], axis=-1)
        return np.concatenate([f, np.broadcast_to(self.enc['mu'], (*f.shape[:-1], 4))], axis=-1)

    @staticmethod
    def _sigmoid(x):
        z = np.exp(-np.abs(x))
        return np.where(x >= 0, 1/(1+z), z/(1+z))

    def _raw_history(self, history):
        f = self._history_features(history)
        batch, steps, n, channels = f.shape
        seq = f.transpose(0, 2, 1, 3).reshape(batch*n, steps, channels)
        hidden = np.zeros((batch*n, self.encoder_hidden), dtype=np.float64)
        enc = self.enc
        for t in range(steps):
            ig = seq[:, t] @ enc['weight_ih_l0'].T + enc['bias_ih_l0']
            hg = hidden @ enc['weight_hh_l0'].T + enc['bias_hh_l0']
            ir, iz, inn = np.split(ig, 3, axis=-1)
            hr, hz, hn = np.split(hg, 3, axis=-1)
            reset, update = self._sigmoid(ir + hr), self._sigmoid(iz + hz)
            new = np.tanh(inn + reset*hn)
            hidden = (1-update)*new + update*hidden
        return (hidden @ enc['head_weight'].T + enc['head_bias']).reshape(batch, n, self.rank)

    def encode_history(self, history):
        history = np.asarray(history, dtype=np.float64)
        if history.ndim != 4 or history.shape[1:] != (11, self.n, 3) or not np.isfinite(history).all():
            raise ValueError('Expected finite history [batch,11,N,3]')
        # Match HistoryEncoder's same-shaped reference evaluation. A cached B=1
        # baseline can use a different BLAS reduction for B>1, leaving a spurious
        # roundoff latent state for an exactly zero history.
        return self._raw_history(history) - self._raw_history(np.zeros_like(history))

    def initial(self, history):
        history = np.asarray(history, dtype=np.float64)
        q = self.encode_history(history)
        return self.from_physical(history[:, -1], q)
