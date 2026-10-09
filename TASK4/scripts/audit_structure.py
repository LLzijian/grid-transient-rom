"""Bounded, read-only structure audit of a registered frozen P-N/P-S/A endpoint.

Only explicit TASK4 checkpoint/evaluation/cache paths are followed. Archive
hashing reads bytes, but NPZ deserialization is limited to prediction/physics
members: histories, truth arrays and HDF5 sources are never opened. No new
trajectory, fitting, data selection, model promotion or stability proof occurs.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from task4.dynamics import PowerGraph
from task4.paths import project_path, read_json, write_json
from task4.training import check_authorized, load_checkpoint


COMPONENTS = ('kinematics', 'canonical_energy_force', 'fixed_p', 'exchange_J',
              'dissipation_R', 'learned_port_pi', 'learned_port_p', 'fixed_governor_port')
SUPPLY_COMPONENTS = ('fixed_p', 'learned_port_pi', 'learned_port_p', 'fixed_governor_port')
BLOCKS = ('delta', 'omega', 'p', 'q')
ATOL, RTOL = 1e-9, 1e-8
RMS_FLOOR = 1e-12


def check_deadline(deadline, reserve=5.0):
    if time.time() + reserve >= deadline:
        raise TimeoutError('Managed structure-audit save deadline reached')


def sha256(path, deadline=float('inf')):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            check_deadline(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def nonfinite_paths(value, path='$'):
    """Name every nullified arithmetic result, including aggregate overflows."""
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in nonfinite_paths(child, f'{path}.{key}')]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return nonfinite_paths(value.item(), path)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [item for index, child in enumerate(value) for item in nonfinite_paths(child, f'{path}[{index}]')]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return [path]
    return []


def report_for_json(value):
    paths = nonfinite_paths(value)
    result = json_safe(value)
    result['nonfinite_json_fields'] = [{'path': path, 'reason': 'nonfinite_arithmetic', 'value': None}
                                      for path in paths]
    return result


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def same_array(left, right, name):
    left, right = np.asarray(left), np.asarray(right)
    if (left.shape != right.shape or not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or not np.allclose(left, right, rtol=1e-12, atol=1e-14)):
        raise ValueError(f'Checkpoint/evaluation/cache mismatch: {name}')


def verify_generation_hashes(declared, prediction_digest, metadata_digest):
    """Legacy manifests remain explicit; new anchors must match before deserialization."""
    expected = (declared.get('sha256'), declared.get('metadata_sha256'))
    if expected == (None, None):
        return False
    for label, anchor, actual in zip(('prediction', 'metadata'), expected,
                                    (prediction_digest, metadata_digest)):
        if (not isinstance(anchor, str) or len(anchor) != 64
                or any(c not in '0123456789abcdef' for c in anchor)):
            raise ValueError(f'Missing or malformed {label} cache-generation SHA256 anchor')
        if anchor != actual:
            raise ValueError(f'{label} cache changed since evaluation generation')
    return True


def fixed_plan(report, points_per_event=8, max_states=1024):
    """Choose indices before reading predictions; all report events are retained.

    Equal counts per event preserve population coverage under the total cap.
    Both initialization and final forecast grid points must fit; no difficult
    event, nonfinite point, or post-truth-prefix point is replaced or dropped.
    """
    if not 2 <= points_per_event <= 8 or not 2 <= max_states <= 1024:
        raise ValueError('Require 2..8 points per event and 2..1024 total states')
    steps = report.get('forecast_steps')
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('Evaluation must declare a positive forecast_steps')
    batches = report.get('cache', {}).get('batches', [])
    ids = [str(event) for batch in batches for event in batch['event_ids']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Explicit cache list is empty or duplicates event IDs')
    if len(ids) != report.get('eligible_event_count'):
        raise ValueError('Explicit cache population differs from evaluation')
    prefix = [row for row in report.get('rows', []) if row.get('horizon') == 'prefix']
    prefix_map = {str(row['id']): row for row in prefix}
    if len(prefix_map) != len(prefix) or set(prefix_map) != set(ids):
        raise ValueError('Evaluation prefix metadata and cache identities disagree')
    count = min(points_per_event, max_states // len(ids), steps + 1)
    if count < 2:
        raise ValueError('Bound cannot cover every event at both forecast endpoints')
    indices = np.unique(np.linspace(0, steps, count, dtype=np.int64))
    interval = float(report['decision_horizon_s']) / steps
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError('Invalid declared forecast interval')
    rows = []
    for batch_index, batch in enumerate(batches):
        for local, event in enumerate(batch['event_ids']):
            event = str(event)
            meta = prefix_map[event]
            available = int(meta['true_available_steps'])
            for index in indices:
                rows.append({'id': event, 'group': str(meta['group']), 'label': meta['label'],
                             'batch_index': batch_index, 'cache_local_index': local,
                             'output_index': int(index), 'forecast_elapsed_s': float(index * interval),
                             'postfault_time_s': float(.1 + index * interval),
                             'within_true_prefix_metadata': bool(index <= available),
                             'true_available_steps_metadata': available, 'status': 'not_processed'})
    return rows


def validate_endpoint(model, checkpoint, report, checkpoint_digest):
    if type(model) is not PowerGraph:
        raise ValueError('Structure audit requires the exact PowerGraph; U/M3 are unsupported')
    endpoint = report.get('endpoint')
    expected = {'P-N': ('P', 'T4-N'), 'P-S': ('P', 'T4-S'), 'A': ('A', 'T4-N')}
    if endpoint not in expected:
        raise ValueError('Require a registered frozen P-N/P-S/A evaluation')
    arm, kind = expected[endpoint]
    if checkpoint['config']['arm'] != arm or checkpoint['kind'] != kind:
        raise ValueError('Registered endpoint and checkpoint arm/kind disagree')
    identity = report.get('model_identity') or {}
    if identity.get('checkpoint_sha256') != checkpoint_digest or identity.get('kind') != kind:
        raise ValueError('Evaluation identity does not match the supplied checkpoint hash/kind')
    facts = report.get('source_facts', {})
    if report.get('test_payload_read') or facts.get('test_payload_read'):
        raise PermissionError('TEST evidence is outside this audit authorization')
    if facts.get('split') not in ('train', 'validation', 'synthetic'):
        raise PermissionError('Evaluation split must explicitly be train/validation/synthetic')
    if bool(checkpoint.get('synthetic')) != (facts['split'] == 'synthetic'):
        raise ValueError('Synthetic/real checkpoint and evaluation mismatch')
    if report.get('history_ms') != 100 or report.get('initializations_per_event') != 1:
        raise ValueError('Audit requires the registered single 100 ms initialization')
    context = report['comparison_context']
    same_array(context['saved_training_scales'], checkpoint['scales'], 'saved TRAIN scales')
    same_array(context['mass'], checkpoint['physics']['mass'], 'mass')
    same_array(model.scales.detach().cpu().numpy(), checkpoint['scales'], 'model scales')
    for key, value in checkpoint['physics'].items():
        same_array(context['source_physics'][key], value, f'source physics {key}')
        if hasattr(model, key):
            same_array(getattr(model, key).detach().cpu().numpy(), value, f'model physics {key}')


def read_prediction_batch(cache, metadata, declared, report, model):
    """Strict whitelist. Never materialize history/truth/substep NPZ members."""
    ids = [str(value) for value in cache['event_ids'].tolist()]
    if ids != list(declared['event_ids']) or ids != [str(row['id']) for row in metadata]:
        raise ValueError('Cache/metadata/manifest event order mismatch')
    same_array(cache['mass'], model.mass.detach().cpu().numpy(), 'cache mass')
    same_array(cache['p_scale'], float(model.scales[2]), 'cache p scale')
    same_array(cache['delta_star'], model.delta_star.detach().cpu().numpy(), 'cache delta_star')
    same_array(cache['omega_s'], report['comparison_context']['source_physics']['omega_s'], 'cache omega_s')
    same_array(cache['decision_horizon_s'], report['decision_horizon_s'], 'cache decision horizon')
    same_array(cache['sample_interval_s'], float(report['decision_horizon_s']) / report['forecast_steps'],
               'cache output interval')
    states = np.asarray(cache['fine_states'], dtype=np.float64)
    flags = np.asarray(cache['fine_state_finite'])
    if states.shape != (len(ids), report['forecast_steps'] + 1, model.dim):
        raise ValueError('Prediction state shape differs from registered complete output grid')
    if flags.shape != states.shape[:2] or flags.dtype.kind != 'b':
        raise ValueError('Prediction finite flags must be a matching boolean grid')
    return states, flags


def decompose_state(model, state):
    """One independently assembled global J/R calculation, without _force.

    Fitted scalar functions are shared, but feature construction, matrix
    assembly, finite directions and state layout are written independently.
    The effort comes from autograd of energy(), not energy_gradient().
    """
    if type(model) is not PowerGraph:
        raise ValueError('Only exact PowerGraph has this registered decomposition')
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (model.dim,) or not np.isfinite(state).all():
        raise ValueError('One finite complete state is required')
    device = model.mass.device
    x = torch.tensor(state[None], dtype=torch.float64, device=device, requires_grad=True)
    with torch.enable_grad():
        energy = model.energy(x)
        gradient = torch.autograd.grad(energy.sum(), x)[0].detach().cpu().numpy()[0]
    with torch.no_grad():
        analytic = model.energy_gradient(x).detach().cpu().numpy()[0]
        field = model.field(x, create_graph=False).cpu().numpy()[0]
    array = lambda name: getattr(model, name).detach().cpu().numpy()
    n, rank, width = model.n, model.rank, 2 + model.rank
    mass, incidence = array('mass'), array('incidence')
    damping, rate, gain = array('damping'), array('rate'), array('gain')
    scales, strength = array('scales'), model.strength
    delta, pi, p = state[:n], state[n:2*n], state[2*n:3*n]
    q, omega = state[3*n:].reshape(n, rank), pi / mass
    local = np.column_stack((omega / scales[1], p / scales[2], q))
    # Rebuild mu and edges from authoritative physical buffers, not model helpers.
    mu = np.log(np.column_stack((mass, np.maximum(damping, 1e-8), rate, gain)))
    ends = [(int(np.flatnonzero(column > 0)[0]), int(np.flatnonzero(column < 0)[0]))
            for column in incidence.T]
    left, right = np.asarray(ends, dtype=np.int64).T
    phase = array('delta_star') @ incidence
    angle = delta @ incidence
    sine = np.sin(phase + angle) - np.sin(phase)
    cosine = np.cos(phase + angle) - np.cos(phase)
    node = np.concatenate((local, mu), axis=-1)
    node0 = np.concatenate((np.zeros_like(local), mu), axis=-1)
    forward = np.column_stack((sine, cosine, local[left], local[right], mu[left], mu[right]))
    reverse = np.column_stack((-sine, cosine, local[right], local[left], mu[right], mu[left]))

    def function(name, features):
        with torch.no_grad():
            argument = torch.as_tensor(features[None], dtype=torch.float64, device=device)
            return getattr(model, name)(argument).detach().cpu().numpy()[0]

    pairs = list(combinations(range(width), 2))
    skew = strength * function('node_exchange', node)
    skew[:, pairs.index((0, 2))] += strength
    ell = np.column_stack((np.sqrt(damping), np.sqrt(rate * gain),
                           np.full((n, rank), .5))) + strength * function('node_dissipation', node)
    edge_exchange = .5 * strength * (
        function('edge_exchange', forward).reshape(-1, width, width)
        - function('edge_exchange', reverse).reshape(-1, width, width).transpose(0, 2, 1))
    edge_ell = .5 * strength * (function('edge_dissipation', forward)
                                + function('edge_dissipation', reverse))
    ports = strength * (function('port', node) - function('port', node0))
    matrix_j, matrix_r = np.zeros((n * width, n * width)), np.zeros((n * width, n * width))
    for machine in range(n):
        offset = machine * width
        for pair, (i, j) in enumerate(pairs):
            matrix_j[offset+i, offset+j] += skew[machine, pair]
            matrix_j[offset+j, offset+i] -= skew[machine, pair]
        block = slice(offset, offset + width)
        matrix_r[block, block] += np.diag(ell[machine] ** 2)
    for edge, (i, j) in enumerate(ends):
        a, b = slice(i * width, (i+1) * width), slice(j * width, (j+1) * width)
        matrix_j[a, b] += edge_exchange[edge]
        matrix_j[b, a] -= edge_exchange[edge].T
        edge_r = np.diag(.5 * edge_ell[edge] ** 2)
        matrix_r[a, a] += edge_r
        matrix_r[b, b] += edge_r
        matrix_r[a, b] -= edge_r
        matrix_r[b, a] -= edge_r
    effort = np.column_stack((gradient[n:2*n], gradient[2*n:3*n],
                               gradient[3*n:].reshape(n, rank))).reshape(-1)

    def force_layout(force):
        force = force.reshape(n, width)
        return np.concatenate((np.zeros(n), force[:, 0], force[:, 1], force[:, 2:].reshape(-1)))

    parts = {name: np.zeros(model.dim) for name in COMPONENTS}
    parts['kinematics'][:n] = omega
    parts['canonical_energy_force'][n:2*n] = -gradient[:n]
    parts['fixed_p'][n:2*n] = p
    parts['exchange_J'] = force_layout(matrix_j @ effort)
    parts['dissipation_R'] = force_layout(-matrix_r @ effort)
    parts['learned_port_pi'][n:2*n] = ports[:, 0]
    parts['learned_port_p'][2*n:3*n] = ports[:, 1]
    parts['fixed_governor_port'][2*n:3*n] = -gain * omega
    manual = np.sum(list(parts.values()), axis=0)
    powers = {name: float(gradient @ part) for name, part in parts.items()}
    supply = float(sum(powers[name] for name in SUPPLY_COMPONENTS))
    dissipation = float(effort @ matrix_r @ effort)
    hdot = float(gradient @ field)
    power_residual = hdot - (supply - dissipation)
    gross = abs(supply) + abs(dissipation)
    port_gross = sum(abs(powers[name]) for name in SUPPLY_COMPONENTS)
    cancellation = float(np.clip(1 - abs(supply-dissipation) / gross, 0, 1)) if gross else 0.
    port_cancellation = float(np.clip(1 - abs(supply) / port_gross, 0, 1)) if port_gross else 0.
    statistics = {
        'energy': float(energy.detach().cpu()[0]), 'Hdot_complete_field': hdot,
        'supply': supply, 'dissipation': dissipation, 'net_supply_minus_dissipation': supply-dissipation,
        'power_identity_residual': power_residual, 'gross_abs_supply_plus_dissipation': gross,
        'gross_individual_supply_abs_power': port_gross,
        'supply_dissipation_cancellation': cancellation, 'internal_supply_cancellation': port_cancellation,
        'component_power': powers,
        'canonical_power_residual': powers['kinematics'] + powers['canonical_energy_force'],
        'canonical_gross_power': abs(powers['kinematics']) + abs(powers['canonical_energy_force']),
        'J_skew_defect_maxabs': float(np.max(np.abs(matrix_j + matrix_j.T))),
        'R_symmetry_defect_maxabs': float(np.max(np.abs(matrix_r - matrix_r.T))),
        'J_power_residual': powers['exchange_J'],
        'J_power_absolute_product_sum': float(np.sum(np.abs(gradient * parts['exchange_J']))),
        'R_power_residual': powers['dissipation_R'] + dissipation,
        'gradient_maxabs_difference': float(np.max(np.abs(gradient - analytic))),
        'field_maxabs_difference': float(np.max(np.abs(field - manual))),
    }
    finite = all(np.isfinite(a).all() for a in (gradient, analytic, field, manual, matrix_j, matrix_r))
    finite = bool(finite and all(np.isfinite(v) for v in powers.values())
                  and all(np.isfinite(v) for k, v in statistics.items() if k != 'component_power'))
    verified = bool(finite and np.allclose(gradient, analytic, atol=ATOL, rtol=RTOL)
                    and np.allclose(field, manual, atol=ATOL, rtol=RTOL)
                    and abs(power_residual) <= ATOL + RTOL * max(abs(hdot), gross)
                    and abs(statistics['canonical_power_residual']) <= ATOL + RTOL * statistics['canonical_gross_power']
                    and abs(statistics['J_power_residual']) <= ATOL + RTOL * statistics['J_power_absolute_product_sum']
                    and abs(statistics['R_power_residual']) <= ATOL + RTOL * abs(dissipation)
                    and dissipation >= -ATOL
                    and statistics['J_skew_defect_maxabs'] <= ATOL
                    and statistics['R_symmetry_defect_maxabs'] <= ATOL)
    return {'gradient_autograd': gradient, 'gradient_analytic': analytic, 'field': field,
            'manual_field': manual, 'components': parts, 'J': matrix_j, 'R': matrix_r,
            'statistics': statistics, 'all_finite': finite, 'numerically_verified': verified}


def inspect_state(model, state, cache_finite):
    """Keep failure as a row at the same chosen index; never substitute a state."""
    if not cache_finite or not np.isfinite(state).all():
        return {'status': 'cached_nonfinite_or_invalid', 'cache_finite': bool(cache_finite),
                'state_all_finite': bool(np.isfinite(state).all()), 'numerically_verified': False}, None
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = decompose_state(model, state)
            derived = {'physical_state': physical_blocks(np.asarray(state), model),
                       'physical_field': physical_blocks(result['field'], model),
                       'physical_components': {name: physical_blocks(value, model)
                                               for name, value in result['components'].items()}}
            arithmetic_failures = nonfinite_paths({'statistics': result['statistics'], **derived})
        if arithmetic_failures:
            result['all_finite'] = result['numerically_verified'] = False
        status = ('verified' if result['numerically_verified'] else
                  'nonfinite_audit' if not result['all_finite'] else 'independent_formula_mismatch')
        return {'status': status, 'cache_finite': True, 'state_all_finite': True,
                'numerically_verified': result['numerically_verified'],
                'nonfinite_arithmetic_fields': arithmetic_failures, **result['statistics']}, result
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        return {'status': 'audit_exception', 'cache_finite': True, 'state_all_finite': True,
                'numerically_verified': False, 'exception': type(exc).__name__ + ': ' + str(exc)}, None


def rms(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    if not np.isfinite(values).all():
        return float('nan')
    amplitude = float(np.max(np.abs(values)))
    return 0. if amplitude == 0 else float(amplitude * np.sqrt(np.mean((values / amplitude) ** 2)))


def physical_blocks(values, model):
    n = model.n
    return {'delta': values[..., :n], 'omega': values[..., n:2*n] / model.mass.detach().cpu().numpy(),
            'p': values[..., 2*n:3*n], 'q': values[..., 3*n:]}


def summarize(rows, arrays, model):
    valid_states = arrays['cache_valid'] & np.isfinite(arrays['states']).all(axis=1)
    valid_fields = np.asarray([row['status'] in ('verified', 'independent_formula_mismatch') for row in rows])
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        all_physical = physical_blocks(arrays['states'], model)
    physical_finite = np.logical_and.reduce([np.isfinite(value).all(axis=1) for value in all_physical.values()])
    conversion_failures = int((valid_states & ~physical_finite).sum())
    valid_states &= physical_finite
    scales = {name: rms(value[valid_states]) for name, value in all_physical.items()}
    denominators = {name: None if value is None else max(value, RMS_FLOOR) for name, value in scales.items()}
    normalized, raw = {}, {}
    for component in ('field',) + COMPONENTS:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            blocks = physical_blocks(arrays[component][valid_fields], model)
            raw[component] = {name: rms(value) for name, value in blocks.items()}
            normalized[component] = {name: None if raw[component][name] is None or denominators[name] is None
                                     else float(np.divide(raw[component][name], denominators[name])) for name in BLOCKS}
    q = arrays['states'][valid_states, 3*model.n:]
    statuses = {name: sum(row['status'] == name for row in rows) for name in sorted({r['status'] for r in rows})}
    arithmetic_failures = nonfinite_paths({'actual_state_rms': scales, 'raw_component_derivative_rms': raw,
                                           'normalized_component_derivative_rms_per_s': normalized})
    return {'planned_states': len(rows), 'status_counts': statuses,
            'normalization_state_count': int(valid_states.sum()), 'contribution_state_count': int(valid_fields.sum()),
            'physical_conversion_failure_count': conversion_failures,
            'nonfinite_arithmetic_fields': arithmetic_failures,
            'all_preselected_states_verified': bool(not arithmetic_failures and not conversion_failures
                                                    and all(row['status'] == 'verified' for row in rows)),
            'actual_state_rms': scales, 'normalization_denominators': denominators, 'normalization_floor': RMS_FLOOR,
            'normalization_scope': 'finite preselected cached full-grid states; raw delta gauge retained; no TRAIN refit',
            'actual_state_units': {'delta': 'rad', 'omega': 'rad/s', 'p': 'system pu', 'q': 'unidentified latent units'},
            'raw_component_derivative_rms': raw, 'normalized_component_derivative_rms_per_s': normalized,
            'derivative_units': {'delta': 'rad/s', 'omega': 'rad/s^2', 'p': 'system pu/s', 'q': 'latent units/s'},
            'q_amplitude': {'rms': rms(q), 'maxabs': float(np.max(np.abs(q))) if q.size else None,
                            'rms_by_machine_and_coordinate': [[rms(q[:, machine*model.rank+coordinate])
                                for coordinate in range(model.rank)] for machine in range(model.n)]},
            'failure_policy': 'descriptive finite-only aggregates never erase failed planned rows; no replacement'}


def audit(checkpoint, evaluation, output, points_per_event=8, max_states=1024):
    started = time.monotonic()
    # Must precede even checkpoint/report/cache reads and output creation.
    check_authorized(False)
    deadline = float(os.environ.get('TASK4_SAVE_DEADLINE_UNIX', 'nan'))
    if not np.isfinite(deadline):
        raise PermissionError('A finite managed save deadline is required')
    check_deadline(deadline)
    source, evaluation_path, output = map(project_path, (checkpoint, evaluation, output))
    output.mkdir(parents=True, exist_ok=False)
    result = {'format': 1, 'status': 'incomplete', 'complete': False, 'inputs': [], 'rows': [],
              'cache_generation_anchor_status': 'not_checked',
              'cache_generation_anchor_checks': [],
              'scope': 'fixed indices across complete fine prediction output grid, including post-truth-prefix states',
              'selection': {'requested_points_per_event': points_per_event, 'total_state_bound': max_states,
                            'rule': 'equal count for all explicit events; unique integer linspace(0,forecast_steps,count)',
                            'no_replacement': True, 'truth_prefix_used_to_select': False},
              'source_payloads_read': False, 'history_or_truth_members_deserialized': False,
              'test_payload_read': False, 'new_rollouts': 0, 'optimization_steps': 0, 'model_promotion': False,
              'stability_certificate': False, 'unique_physical_mechanism': False,
              'independent_personnel_review': False,
              'independent_formula_path': 'autograd energy vs analytic gradient; NumPy global block J/R and fixed ports vs field',
              'state_order': 'delta[N], pi[N], p[N], q[N*rank] machine-major; pi=mass*omega',
              'matrix_order': 'node-major [pi_i,p_i,q_i]; canonical delta-pi terms are reported separately',
              'power_unit': 'model H-units/s; learned storage is not calibrated SI energy',
              'cancellation_definitions': {
                  'supply_dissipation': '1-|S-D|/(|S|+|D|), zero when both are zero',
                  'internal_supply': '1-|sum individual port powers|/sum |individual port powers|, zero for zero denominator'},
              'tolerances': {'absolute': ATOL, 'relative': RTOL},
              'limitations': ['Finite sampled-state algebra audit is not a stability-region or trajectory-validity certificate.',
                             'Before/end hashes establish input stability during this audit; generation integrity additionally requires matching anchors in the supplied evaluation report.',
                             'Fitted scalar functions are shared; assembly and energy differentiation are independent paths.',
                             'q and learned H are model coordinates, not uniquely identified physical states or SI energy.',
                             'Cancellation and contribution sizes describe this fixed representation, not causal identification.',
                             'Existing prediction failures and post-prefix states remain visible; no extra truth is synthesized.']}
    arrays, model, caught = None, None, None

    def persist():
        result['process_wall_seconds_including_load_hash_and_audit'] = time.monotonic() - started
        if arrays is not None:
            result['summary'] = summarize(result['rows'], arrays, model)
            result['numerically_verified'] = bool(result['complete']
                                                 and result['summary']['all_preselected_states_verified'])
            if result['complete'] and not result['numerically_verified']:
                result['status'] = 'audit_complete_with_failures'
            save_npz(output / 'selected_states_and_contributions.npz', **arrays)
        write_json(output / 'structure_audit.json', report_for_json(result))

    def record_input(path, role):
        check_deadline(deadline)
        digest = sha256(path, deadline)
        result['inputs'].append({'path': str(path), 'role': role, 'sha256': digest})
        return digest

    try:
        record_input(evaluation_path, 'evaluation_report')
        report = read_json(evaluation_path)
        # Reject TEST provenance before opening any cache or checkpoint tensors.
        if report.get('source_facts', {}).get('split') not in ('train', 'validation', 'synthetic'):
            raise PermissionError('Evaluation provenance does not authorize cache access')
        if report.get('test_payload_read') or report.get('source_facts', {}).get('test_payload_read'):
            raise PermissionError('TEST evidence cannot be audited in this run')
        digest = record_input(source, 'checkpoint')
        check_deadline(deadline)
        torch.set_num_threads(1)
        model, encoder, head, saved = load_checkpoint(source, device='cpu')
        del encoder, head
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        validate_endpoint(model, saved, report, digest)
        result.update(endpoint=report['endpoint'], evidence_role=report['evidence_role'],
                      source_split=report['source_facts']['split'], checkpoint_kind=saved['kind'],
                      checkpoint_step=saved['step'], saved_training_scales=saved['scales'],
                      physics=saved['physics'], rank=model.rank, n=model.n,
                      raw_event_count=report['raw_event_count'], eligible_event_count=report['eligible_event_count'],
                      latency_uncovered_count=report['latency_uncovered_count'],
                      forecast_steps=report['forecast_steps'], decision_horizon_s=report['decision_horizon_s'])
        result['rows'] = fixed_plan(report, points_per_event, max_states)
        size = (len(result['rows']), model.dim)
        arrays = {name: np.full(size, np.nan) for name in
                  ('states', 'gradient_autograd', 'gradient_analytic', 'field', 'manual_field') + COMPONENTS}
        arrays['cache_valid'] = np.zeros(size[0], dtype=bool)
        arrays['event_ids'] = np.asarray([row['id'] for row in result['rows']], dtype=str)
        arrays['output_indices'] = np.asarray([row['output_index'] for row in result['rows']], dtype=np.int64)
        persist()
        for batch_index, declared in enumerate(report['cache']['batches']):
            check_deadline(deadline)
            path, metadata_path = project_path(declared['path']), project_path(declared['metadata_path'])
            prediction_digest = record_input(path, 'explicit_prediction_cache')
            metadata_digest = record_input(metadata_path, 'explicit_cache_metadata')
            anchored = verify_generation_hashes(declared, prediction_digest, metadata_digest)
            result['cache_generation_anchor_checks'].append({'batch_index': batch_index, 'verified': anchored})
            metadata = read_json(metadata_path)
            with np.load(path, allow_pickle=False) as cache:
                states, flags = read_prediction_batch(cache, metadata, declared, report, model)
            selected = [(i, row) for i, row in enumerate(result['rows']) if row['batch_index'] == batch_index]
            # Record all chosen inputs/flags in this batch before evaluating any.
            for index, row in selected:
                local, point = row['cache_local_index'], row['output_index']
                if str(metadata[local]['group']) != row['group'] or metadata[local]['label'] != row['label']:
                    raise ValueError('Cache label/group differs from saved evaluation metadata')
                arrays['states'][index] = states[local, point]
                arrays['cache_valid'][index] = flags[local, point]
                row['cached_forecast_failure'] = metadata[local].get('fine_failure')
            for index, row in selected:
                check_deadline(deadline)
                facts, decomposed = inspect_state(model, arrays['states'][index], arrays['cache_valid'][index])
                row.update(facts)
                if decomposed is not None:
                    for key in ('gradient_autograd', 'gradient_analytic', 'field', 'manual_field'):
                        arrays[key][index] = decomposed[key]
                    for key in COMPONENTS:
                        arrays[key][index] = decomposed['components'][key]
            persist()
        # Rehash every explicitly consumed asset before calling this a frozen audit.
        for item in result['inputs']:
            item['sha256_after'] = sha256(item['path'], deadline)
            item['unchanged'] = item['sha256_after'] == item['sha256']
        frozen = all(item['unchanged'] for item in result['inputs'])
        result['cache_generation_anchor_status'] = (
            'fully_verified' if all(row['verified'] for row in result['cache_generation_anchor_checks'])
            else 'unavailable_for_legacy_manifest_batches')
        result.update(complete=frozen, status='audit_complete' if frozen else 'input_changed',
                      inputs_unchanged=frozen,
                      stop_reason='all_fixed_points_accounted_for' if frozen else 'source_changed_during_audit')
    except TimeoutError as exc:
        result.update(status='incomplete', complete=False, stop_reason='save_deadline', limitation=str(exc))
    except BaseException as exc:
        caught = exc
        result.update(status='incomplete', complete=False, stop_reason=type(exc).__name__ + ': ' + str(exc))
    finally:
        persist()
    if caught is not None:
        raise caught
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--points-per-event', type=int, default=8)
    parser.add_argument('--max-states', type=int, default=1024)
    result = audit(**vars(parser.parse_args()))
    print(json.dumps(report_for_json({'status': result['status'], 'complete': result['complete'],
                                     'numerically_verified': result.get('numerically_verified', False),
                                     'stop_reason': result['stop_reason'], 'summary': result.get('summary')}), indent=2))


if __name__ == '__main__':
    main()
