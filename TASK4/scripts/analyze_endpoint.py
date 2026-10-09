"""Read-only diagnostics of one fixed complete ROM endpoint; no region proof.

Only the saved checkpoint/explicit symbolic JSON is read. No event payload,
FOM simulator, future observation, fitting, model selection, or 13-mode
matching is used. The entire autonomous field, including ports, is retained.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from task4.paths import ROOT, project_path, write_json


DIFFERENCE_STEPS = (1e-5, 5e-6)
RADII = (1e-4, 1e-3, 1e-2, .03, .1)
DIRECTION_COUNT = 128
DIRECTION_SEED = 4439
ORIGIN_TOLERANCE = 1e-10
JACOBIAN_ATOL = 1e-6
JACOBIAN_RTOL = 1e-5
LYAPUNOV_RESIDUAL_TOLERANCE = 1e-8


def check_deadline(deadline, reserve=3.0):
    if time.time() + reserve >= deadline:
        raise TimeoutError('Managed endpoint-analysis save deadline reached')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


class QuotientCoordinates:
    """Remove only common angle; retain all pi, p and q coordinates.

    Full order is delta[N], pi[N], p[N], q[N*rank]. Reduced y is
    delta[:-1]-delta[-1], pi, p, q, and its lift fixes delta[-1]=0.
    The analyzed dimensionless coordinate is xi=y/s, with saved TRAIN scales
    and s=(scale_delta, mass*scale_omega, scale_p, 1 for every latent q).
    """
    def __init__(self, n, rank, mass, scales):
        if int(n) != n or n < 1 or int(rank) != rank or rank < 0:
            raise ValueError('n must be positive and rank nonnegative integers')
        self.n, self.rank = int(n), int(rank)
        self.mass = np.asarray(mass, dtype=float)
        self.scales = np.asarray(scales, dtype=float)
        if (self.mass.shape != (self.n,) or self.scales.shape != (3,)
                or not np.isfinite(self.mass).all() or not np.isfinite(self.scales).all()
                or np.any(self.mass <= 0) or np.any(self.scales <= 0)):
            raise ValueError('Positive finite machine masses and three saved scales are required')
        self.full_dim = (3 + self.rank) * self.n
        self.dim = self.full_dim - 1
        self.scale = np.concatenate((np.full(self.n - 1, self.scales[0]),
                                     self.mass * self.scales[1],
                                     np.full(self.n, self.scales[2]),
                                     np.ones(self.n * self.rank)))

    def reduce(self, state):
        x = np.asarray(state)
        if x.ndim < 1 or x.shape[-1] != self.full_dim:
            raise ValueError('Wrong full-state dimension')
        angles = x[..., :self.n - 1] - x[..., self.n - 1:self.n]
        return np.concatenate((angles, x[..., self.n:]), axis=-1)

    def lift(self, reduced):
        y = np.asarray(reduced)
        if y.ndim < 1 or y.shape[-1] != self.dim:
            raise ValueError('Wrong quotient-state dimension')
        return np.concatenate((y[..., :self.n - 1], np.zeros((*y.shape[:-1], 1), dtype=y.dtype),
                               y[..., self.n - 1:]), axis=-1)

    def scaled_to_full(self, scaled):
        return self.lift(np.asarray(scaled) * self.scale)

    def scaled_field(self, full_field, scaled):
        return self.reduce(full_field(self.scaled_to_full(scaled))) / self.scale

    def physical_readout_matrices(self):
        """Dimensionless, gauge-invariant physical outputs of scaled states.

        Shares below are squared right-eigenvector readout amplitudes, not
        left/right participation factors or fractions of physical energy.
        Separate outputs retain relative angle, relative and common speed, p.
        """
        lift_scaled = self.scaled_to_full(np.eye(self.dim)).T
        weights = self.mass / self.mass.sum()
        projector = np.eye(self.n) - np.ones((self.n, 1)) * weights[None, :]
        delta = lift_scaled[:self.n]
        omega = lift_scaled[self.n:2 * self.n] / self.mass[:, None]
        power = lift_scaled[2 * self.n:3 * self.n]
        return {
            'relative_delta': projector @ delta / self.scales[0],
            'relative_omega': projector @ omega / self.scales[1],
            'common_omega': weights[None, :] @ omega / self.scales[1],
            'p': power / self.scales[2],
        }

    def describe(self):
        return {
            'n': self.n, 'rank': self.rank, 'full_dimension': self.full_dim,
            'quotient_dimension': self.dim,
            'full_order': ['delta[N]', 'pi[N]', 'p[N]', 'q[N*rank] machine-major'],
            'quotient_order': ['delta_i-delta_last [N-1]', 'pi[N]', 'p[N]', 'q[N*rank]'],
            'lift': 'last delta is exactly zero; all pi, p and q retained',
            'scaled_coordinate': 'xi=y/s', 'scaled_field': 'g(xi)=project(f(lift(s*xi)))/s',
            'coordinate_scale': self.scale.tolist(), 'saved_training_scales': self.scales.tolist(),
            'mass': self.mass.tolist(),
            'scale_units': ['delta: rad', 'pi: mass*(rad/s)', 'p: system pu', 'q: one latent unit'],
            'jacobian_unit': '1/s',
            'scale_fit_in_this_diagnostic': False,
        }


class CountedField:
    def __init__(self, function, dimension):
        self.function = function
        self.dimension = dimension
        self.calls = 0
        self.states = 0
        self.wall_seconds = 0.0

    def __call__(self, states):
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != self.dimension:
            raise ValueError('Field calls use [batch, full_dimension] arrays')
        self.calls += 1
        self.states += len(states)
        start = time.monotonic()
        try:
            result = np.asarray(self.function(states), dtype=float)
            if result.shape != states.shape:
                raise ValueError('Field result shape differs from input')
            return result
        finally:
            self.wall_seconds += time.monotonic() - start


def safe_field_batch(field, points):
    """Keep individual numerical failures as missing results and diagnostics."""
    points = np.asarray(points, dtype=float)
    errors = [None] * len(points)
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            values = np.asarray(field(points), dtype=float)
        if values.shape != points.shape:
            raise ValueError('Quotient field shape differs from input')
        return values, errors
    except (FloatingPointError, RuntimeError, ValueError, OverflowError):
        values = np.full_like(points, np.nan)
        for index, point in enumerate(points):
            try:
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    value = np.asarray(field(point[None]), dtype=float)
                if value.shape != (1, points.shape[1]):
                    raise ValueError('Quotient field shape differs from input')
                values[index] = value[0]
            except (FloatingPointError, RuntimeError, ValueError, OverflowError) as exc:
                errors[index] = type(exc).__name__ + ': ' + str(exc)
        return values, errors


def central_jacobian(field, dimension, step, *, deadline=float('inf')):
    """Centered differences in scaled coordinates, with partial evidence kept."""
    if dimension < 1 or not np.isfinite(step) or step <= 0:
        raise ValueError('Positive dimension and finite-difference step required')
    points = np.eye(dimension) * step
    plus = np.full((dimension, dimension), np.nan)
    minus = np.full_like(plus, np.nan)
    jacobian = np.full_like(plus, np.nan)
    errors = []
    completed = 0
    stop_reason = 'complete'
    start = time.monotonic()
    for column in range(dimension):
        try:
            check_deadline(deadline)
        except TimeoutError:
            stop_reason = 'save_deadline'
            break
        values, failures = safe_field_batch(field, np.stack((points[column], -points[column])))
        plus[column], minus[column] = values
        jacobian[:, column] = (plus[column] - minus[column]) / (2 * step)
        errors.extend({'column': column, 'sign': sign, 'error': error}
                      for sign, error in zip((1, -1), failures) if error is not None)
        completed += 1
    return {
        'jacobian': jacobian, 'positive_points': points, 'negative_points': -points,
        'positive_fields': plus, 'negative_fields': minus,
        'completed_columns': completed, 'complete': completed == dimension,
        'all_finite': bool(np.isfinite(jacobian).all()), 'step': float(step),
        'errors': errors, 'stop_reason': stop_reason, 'wall_seconds': time.monotonic() - start,
    }


def spectrum_rows(matrix, coordinates):
    values, vectors = np.linalg.eig(matrix)
    order = np.lexsort((values.imag, values.real))
    values, vectors = values[order], vectors[:, order]
    readouts = coordinates.physical_readout_matrices()
    physical_units = {'relative_delta': ('rad', coordinates.scales[0]),
                      'relative_omega': ('rad/s', coordinates.scales[1]),
                      'common_omega': ('rad/s', coordinates.scales[1]),
                      'p': ('system pu', coordinates.scales[2])}
    rows = []
    for index, (value, vector) in enumerate(zip(values, vectors.T)):
        vector = vector / np.linalg.norm(vector)
        squared = {name: float(np.sum(np.abs(readout @ vector) ** 2))
                   for name, readout in readouts.items()}
        total = sum(squared.values())
        output = {
            name: {'scaled_squared_amplitude': amount,
                   'share_of_total_scaled_readout_squared_amplitude': amount / total if total > 0 else None,
                   'physical_amplitude_l2': float(np.sqrt(amount) * physical_units[name][1]),
                   'physical_amplitude_unit': physical_units[name][0]}
            for name, amount in squared.items()
        }
        rows.append({
            'index': index, 'eigenvalue_real_per_s': float(value.real),
            'eigenvalue_imag_per_s': float(value.imag),
            'oscillation_frequency_hz': float(abs(value.imag) / (2 * np.pi)),
            'natural_frequency_hz': float(abs(value) / (2 * np.pi)),
            'damping_ratio': float(-value.real / abs(value)) if abs(value) > 0 else None,
            'strictly_negative_real_part': bool(value.real < 0),
            'physical_readout': output,
            'latent_scaled_right_eigenvector_squared_share': float(np.sum(np.abs(vector[3 * coordinates.n - 1:]) ** 2)),
        })
    denominator = max(np.linalg.norm(matrix) * np.linalg.norm(vectors), np.finfo(float).tiny)
    residual = float(np.linalg.norm(matrix @ vectors - vectors * values) / denominator)
    return rows, values, vectors, {
        'spectral_abscissa_per_s': float(values.real.max()),
        'hurwitz_by_computed_eigenvalues': bool(np.all(values.real < 0)),
        'near_imaginary_axis_count_abs_real_le_1e_8': int(np.sum(np.abs(values.real) <= 1e-8)),
        'eigenvector_matrix_condition': float(np.linalg.cond(vectors)),
        'relative_eigen_equation_residual': residual,
        'original_13_mode_matching_performed': False, 'original_13_mode_match_result': None,
        'participation_definition': 'squared right-eigenvector physical-readout amplitudes in saved physical scales; not biorthogonal participation factors or physical energy fractions',
    }


def fixed_directions(dimension):
    random = np.random.default_rng(DIRECTION_SEED)
    positive = random.normal(size=(DIRECTION_COUNT // 2, dimension))
    positive /= np.linalg.norm(positive, axis=1, keepdims=True)
    result = np.empty((DIRECTION_COUNT, dimension))
    result[::2], result[1::2] = positive, -positive
    return result


def sample_radius(field, coordinates, matrix_p, directions, radius, *, deadline=float('inf')):
    points = directions * radius
    derivatives = np.full_like(points, np.nan)
    completed = np.zeros(len(points), dtype=bool)
    errors = []
    start = time.monotonic()
    stop_reason = 'complete'
    for first in range(0, len(points), 8):
        try:
            check_deadline(deadline)
        except TimeoutError:
            stop_reason = 'save_deadline'
            break
        last = min(first + 8, len(points))
        values, failures = safe_field_batch(field, points[first:last])
        derivatives[first:last] = values
        completed[first:last] = True
        errors.extend({'direction_index': first + index, 'error': error}
                      for index, error in enumerate(failures) if error is not None)
    with np.errstate(over='ignore', invalid='ignore'):
        v = np.einsum('bi,ij,bj->b', points, matrix_p, points)
        vdot = 2 * np.einsum('bi,ij,bj->b', points, matrix_p, derivatives)
    finite = completed & np.isfinite(derivatives).all(1) & np.isfinite(v) & np.isfinite(vdot)
    bad = completed & (~finite | (v <= 0) | (vdot >= 0))
    usable = np.flatnonzero(finite)
    summary = {
        'radius_scaled_euclidean': float(radius), 'planned_directions': len(points),
        'completed_directions': int(completed.sum()), 'complete': bool(completed.all()),
        'finite_directions': int(finite.sum()), 'nonfinite_or_exception_count': int((completed & ~finite).sum()),
        'nonpositive_v_count': int((finite & (v <= 0)).sum()),
        'nonnegative_vdot_count': int((finite & (vdot >= 0)).sum()),
        'bad_direction_indices': np.flatnonzero(bad).tolist(),
        'uncomputed_direction_indices': np.flatnonzero(~completed).tolist(),
        'min_v': float(v[usable].min()) if len(usable) else None,
        'max_vdot': float(vdot[usable].max()) if len(usable) else None,
        'min_vdot': float(vdot[usable].min()) if len(usable) else None,
        'max_vdot_over_radius_squared': float(vdot[usable].max() / radius ** 2) if len(usable) else None,
        'sampled_strict_decrease_only': bool(finite.all() and np.all(v > 0) and np.all(vdot < 0)),
        'region_positive_definiteness_or_invariance_proven': False,
        'errors': errors, 'stop_reason': stop_reason, 'wall_seconds': time.monotonic() - start,
    }
    arrays = {
        'directions': directions, 'scaled_points': points,
        'full_states': coordinates.scaled_to_full(points), 'scaled_full_field_quotient': derivatives,
        'V': v, 'Vdot': vdot, 'computed_mask': completed, 'finite_mask': finite,
        'bad_indices': np.flatnonzero(bad), 'bad_scaled_points': points[bad],
        'bad_full_states': coordinates.scaled_to_full(points[bad]),
        'bad_V': v[bad], 'bad_Vdot': vdot[bad],
    }
    return summary, arrays


def load_endpoint(checkpoint=None, symbolic=None):
    """Called only after managed authorization; never opens event data."""
    if (checkpoint is None) == (symbolic is None):
        raise ValueError('Provide exactly one checkpoint or symbolic JSON')
    if checkpoint is not None:
        import torch
        from task4.training import load_checkpoint
        torch.set_num_threads(1)
        model, _, _, obj = load_checkpoint(checkpoint, device='cpu')
        model.eval()
        saved_scales = np.asarray(obj['scales'], dtype=float)
        if not np.allclose(saved_scales, model.scales.detach().cpu().numpy(), rtol=0, atol=0):
            raise ValueError('Saved normalization and model buffers disagree')
        coordinates = QuotientCoordinates(model.n, model.rank, model.mass.detach().cpu().numpy(), saved_scales)

        def field(states):
            with torch.no_grad():
                return model.field(torch.as_tensor(states, dtype=torch.float64), create_graph=False).cpu().numpy()

        metadata = {'format': 'torch_checkpoint', 'kind': obj.get('kind'),
                    'arm': obj.get('config', {}).get('arm'), 'step': obj.get('step'),
                    'synthetic': obj.get('synthetic'),
                    'scale_source': 'saved TRAIN normalization' if obj.get('synthetic') is False else 'saved source normalization; TRAIN origin not confirmed'}
    else:
        from task4.symbolic_runtime import SymbolicPowerGraph
        spec = json.loads(Path(symbolic).read_text(encoding='utf-8'))
        model = SymbolicPowerGraph(spec)
        coordinates = QuotientCoordinates(model.n, model.rank, model.mass, model.scales)
        field = model.field
        metadata = {'format': 'explicit_symbolic_json', 'kind': 'T4-S',
                    'export_metadata': spec.get('metadata', {}),
                    'scale_source': 'saved symbolic-export normalization; copied without any fit'}
    return CountedField(field, coordinates.full_dim), coordinates, metadata


def analyze(*, output, checkpoint=None, symbolic=None):
    from task4.training import check_authorized
    check_authorized(False)  # Must precede opening/loading either model format.
    if (checkpoint is None) == (symbolic is None):
        raise ValueError('Provide exactly one checkpoint or symbolic JSON')
    deadline_text = os.environ.get('TASK4_SAVE_DEADLINE_UNIX')
    if deadline_text is None:
        raise RuntimeError('Managed save deadline is required; no new budget is assumed')
    deadline = float(deadline_text)
    if not np.isfinite(deadline):
        raise ValueError('Managed save deadline must be finite')
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = project_path(checkpoint if checkpoint is not None else symbolic)
    started = time.monotonic()
    field = None
    report = {
        'status': 'inconclusive', 'complete': False, 'stop_reason': 'started',
        'source_model_path': str(source), 'source_model_sha256': None,
        'analysis_role': 'post-hoc diagnostic of one fixed final endpoint; never training or model selection',
        'full_autonomous_field_including_all_ports': True, 'removed_field_terms': [],
        'test_payload_read': False, 'event_payload_read': False, 'new_fom_data_generated': False,
        'all_analysis_samples_are_rom_only': True, 'original_13_mode_matching_performed': False,
        'L2_region_certificate': False, 'global_stability_claim': False, 'high_order_system_guarantee': False,
        'finite_samples_are_region_proof': False,
        'policy': {'difference_steps_scaled': DIFFERENCE_STEPS, 'radii_scaled': RADII,
                   'directions_per_radius': DIRECTION_COUNT, 'direction_seed': DIRECTION_SEED,
                   'directions': '64 fixed seeded Gaussian unit vectors and exact negative pairs, reused at all radii',
                   'origin_scaled_inf_tolerance': ORIGIN_TOLERANCE,
                   'jacobian_consistency_atol': JACOBIAN_ATOL, 'jacobian_consistency_rtol': JACOBIAN_RTOL,
                   'lyapunov_relative_residual_tolerance': LYAPUNOV_RESIDUAL_TOLERANCE,
                   'tolerances_are_numerical_diagnostics_not_proof_bounds': True},
        'stages': [], 'sampling': [], 'managed_save_deadline_unix': deadline,
    }

    def persist(stage=None):
        if stage is not None:
            report['stages'].append(stage)
        report['process_wall_seconds_including_load_and_all_stages'] = time.monotonic() - started
        report['field_cost'] = None if field is None else {
            'calls': field.calls, 'states_evaluated': field.states, 'wall_seconds': field.wall_seconds}
        write_json(output / 'analysis.json', json_safe(report))

    persist()
    caught = None
    try:
        check_deadline(deadline)
        report['source_model_sha256'] = sha256(source)
        field, coordinates, metadata = load_endpoint(
            checkpoint=str(source) if checkpoint is not None else None,
            symbolic=str(source) if symbolic is not None else None)
        report['source_metadata'] = metadata
        report['coordinates'] = coordinates.describe()
        report['source_sha256_after_load'] = sha256(source)
        if report['source_sha256_after_load'] != report['source_model_sha256']:
            raise ValueError('Source endpoint changed during loading; use a fixed snapshot')
        persist({'name': 'source_and_coordinates', 'complete': True})
        check_deadline(deadline)
        zero = np.zeros((1, coordinates.full_dim))
        origin = field(zero)[0]
        shifted = zero.copy()
        shifted[:, :coordinates.n] = .123
        shifted_origin = field(shifted)[0]
        scaled_origin = coordinates.reduce(origin) / coordinates.scale
        gauge_difference = coordinates.reduce(shifted_origin - origin) / coordinates.scale
        origin_finite = bool(np.isfinite(origin).all() and np.isfinite(scaled_origin).all())
        origin_norm = float(np.max(np.abs(scaled_origin)))
        gauge_norm = float(np.max(np.abs(gauge_difference)))
        report['origin'] = {'full_field': origin.tolist(), 'scaled_quotient_field': scaled_origin.tolist(),
                            'all_finite': origin_finite, 'scaled_inf_norm': origin_norm,
                            'numerically_zero': bool(origin_finite and origin_norm <= ORIGIN_TOLERANCE),
                            'common_angle_shift_probe_rad': .123, 'gauge_probe_scaled_inf_difference': gauge_norm,
                            'gauge_probe_pass': bool(np.isfinite(gauge_norm) and gauge_norm <= ORIGIN_TOLERANCE)}
        save_npz(output / 'origin.npz', full_state=zero[0], full_field=origin,
                 scaled_quotient_field=scaled_origin, common_shift_field=shifted_origin,
                 coordinate_scale=coordinates.scale)
        persist({'name': 'origin_complete_field', 'complete': True})
        scaled_field = lambda states: coordinates.scaled_field(field, states)
        differences = []
        for index, step in enumerate(DIFFERENCE_STEPS):
            check_deadline(deadline)
            result = central_jacobian(scaled_field, coordinates.dim, step, deadline=deadline)
            filename = f'jacobian_{index + 1}.npz'
            save_npz(output / filename, **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})
            stage = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
            stage.update(name='central_jacobian', artifact=filename)
            differences.append(result)
            persist(stage)
            if not result['complete']:
                raise TimeoutError('Finite-difference stage interrupted at the shared deadline')
        first, second = differences[0]['jacobian'], differences[1]['jacobian']
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            report['stop_reason'] = 'nonfinite_jacobian'
            report['complete'] = False
            return report
        difference = second - first
        maximum = float(np.max(np.abs(difference)))
        relative = float(np.linalg.norm(difference) / max(np.linalg.norm(second), np.finfo(float).tiny))
        consistent = bool(np.all(np.abs(difference) <= JACOBIAN_ATOL + JACOBIAN_RTOL * np.abs(second)))
        report['jacobian_consistency'] = {'max_absolute_difference_per_s': maximum,
                                           'relative_frobenius_difference': relative, 'consistent': consistent,
                                           'selected_step_is_fixed_smaller_step': DIFFERENCE_STEPS[1]}
        check_deadline(deadline)
        rows, eigenvalues, eigenvectors, spectral = spectrum_rows(second, coordinates)
        first_eigenvalues = np.linalg.eigvals(first)
        first_hurwitz = bool(np.all(first_eigenvalues.real < 0))
        spectral.update(first_step_spectral_abscissa_per_s=float(first_eigenvalues.real.max()),
                        first_step_hurwitz=first_hurwitz,
                        both_steps_hurwitz=bool(first_hurwitz and spectral['hurwitz_by_computed_eigenvalues']))
        report['spectrum'] = {'summary': spectral, 'all_modes': rows}
        save_npz(output / 'spectrum.npz', eigenvalues=eigenvalues, right_eigenvectors=eigenvectors,
                 first_step_eigenvalues=first_eigenvalues,
                 **{'readout_' + key: value for key, value in coordinates.physical_readout_matrices().items()})
        persist({'name': 'full_quotient_spectrum', 'complete': True, 'modes': len(rows)})
        if not spectral['both_steps_hurwitz']:
            report.update(complete=True, stop_reason='quadratic_candidate_not_constructed_non_hurwitz_linearization')
            report['lyapunov'] = {'constructed': False, 'reason': 'both numerical Jacobians must be Hurwitz'}
        else:
            check_deadline(deadline)
            p_raw = solve_continuous_lyapunov(second.T, -np.eye(coordinates.dim))
            matrix_p = (p_raw + p_raw.T) / 2
            save_npz(output / 'lyapunov_candidate_raw.npz', P_raw=p_raw, P_symmetric=matrix_p)
            p_eigenvalues = np.linalg.eigvalsh(matrix_p)
            residual = second.T @ matrix_p + matrix_p @ second + np.eye(coordinates.dim)
            relative_residual = float(np.linalg.norm(residual) / np.sqrt(coordinates.dim))
            finite_p = bool(np.isfinite(matrix_p).all() and np.isfinite(p_eigenvalues).all())
            positive_p = bool(finite_p and np.all(p_eigenvalues > 0))
            residual_pass = bool(np.isfinite(relative_residual) and relative_residual <= LYAPUNOV_RESIDUAL_TOLERANCE)
            report['lyapunov'] = {
                'constructed': True, 'candidate': 'V(xi)=xi.T @ P @ xi',
                'equation': 'A.T @ P + P @ A = -I, using the fixed smaller-step scaled Jacobian',
                'P_all_finite': finite_p, 'P_positive_definite_numerically': positive_p,
                'P_min_eigenvalue': float(p_eigenvalues.min()), 'P_max_eigenvalue': float(p_eigenvalues.max()),
                'P_condition_number': float(np.linalg.cond(matrix_p)),
                'raw_symmetry_defect': float(np.linalg.norm(p_raw - p_raw.T)),
                'relative_equation_residual': relative_residual, 'residual_pass': residual_pass,
                'region_certificate': False,
            }
            save_npz(output / 'lyapunov_candidate.npz', P=matrix_p, P_raw=p_raw,
                     P_eigenvalues=p_eigenvalues, equation_residual=residual, coordinate_scale=coordinates.scale)
            persist({'name': 'quadratic_candidate', 'complete': True})
            if not finite_p:
                report.update(complete=False, stop_reason='nonfinite_quadratic_candidate')
            else:
                directions = fixed_directions(coordinates.dim)
                save_npz(output / 'fixed_directions.npz', directions=directions)
                for index, radius in enumerate(RADII):
                    check_deadline(deadline)
                    sample, arrays = sample_radius(scaled_field, coordinates, matrix_p, directions, radius, deadline=deadline)
                    filename = f'radius_{index + 1:02d}.npz'
                    save_npz(output / filename, **arrays)
                    sample['artifact'] = filename
                    report['sampling'].append(sample)
                    persist({'name': 'fixed_radius_sampling', 'radius': radius, 'complete': sample['complete']})
                    if not sample['complete']:
                        raise TimeoutError('Fixed-direction sampling interrupted at the shared deadline')
                sample_finite = all(row['nonfinite_or_exception_count'] == 0 for row in report['sampling'])
                prerequisites = (consistent and report['origin']['numerically_zero']
                                 and report['origin']['gauge_probe_pass'] and positive_p and residual_pass and sample_finite)
                report.update(complete=True, stop_reason='fixed_diagnostic_complete',
                              status='candidate_only_no_region_certificate' if prerequisites else 'inconclusive')
                report['sampled_counterexample_count'] = sum(len(row['bad_direction_indices']) for row in report['sampling'])
                report['candidate_status_does_not_assert_all_radii_pass'] = True
        check_deadline(deadline)
        report['source_sha256_after_analysis'] = sha256(source)
        report['source_unchanged'] = report['source_sha256_after_analysis'] == report['source_model_sha256']
        if not report['source_unchanged']:
            report.update(status='inconclusive', complete=False, stop_reason='source_model_changed_during_analysis')
    except TimeoutError as exc:
        report.update(status='inconclusive', complete=False, stop_reason='save_deadline', limitation=str(exc))
    except BaseException as exc:
        caught = exc
        report.update(status='inconclusive', complete=False, stop_reason=type(exc).__name__ + ': ' + str(exc))
    finally:
        report['limitations'] = [
            'Numerical linearization and negative real parts are not a nonlinear stability-region certificate.',
            'Numerical eigenvalues of P are a positivity diagnostic; validated bounds and full-field region conditions are still needed for a nonlinear region/invariance certificate.',
            'The 640 fixed ROM-only directional samples can expose failures but cannot prove decrease between samples.',
            'No original 13-mode matching, L2 success, global stability, or high-order-system guarantee is asserted.',
        ]
        persist()
    if caught is not None:
        raise caught
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--symbolic')
    parser.add_argument('--output', required=True)
    result = analyze(**vars(parser.parse_args()))
    print(json.dumps(json_safe({'status': result['status'], 'complete': result['complete'],
                                'stop_reason': result['stop_reason'],
                                'seconds': result['process_wall_seconds_including_load_and_all_stages']}), indent=2))


if __name__ == '__main__':
    main()
