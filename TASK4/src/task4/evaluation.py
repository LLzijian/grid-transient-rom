"""Frozen, single-initialization TASK4 evaluation with retained failure evidence.

Accuracy uses genuine truth intervals and the original max-normalized
delta/omega/common/p step discrepancy. LOS uses the separately ported KIKI
RMS/trigger gate on the complete post-decision window. These gates are not
interchanged. All trajectories are initialized from eleven observations;
failed future states are missing (NaN), never copied truth or frozen states.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch

from .classification import (
    DELTA_SCALE, OMEGA_SCALE, classify_forecast, numerical_pair,
    summarize_classifications, trigger_times,
)
from .paths import project_path, write_json
from .dynamics import PowerGraph

SAMPLE_DT = 0.01
DECISION_HORIZON = 8.0


def cache_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenM3(torch.nn.Module):
    """Fixed physical M3 reference, with no latent state or encoder call."""
    def __init__(self, physics, scales):
        super().__init__()
        physics.check()
        self.n = physics.n
        self.dim = 3 * self.n
        self.omega_s = float(physics.omega_s)
        for name in ('mass', 'incidence', 'delta_star', 'coupling', 'damping', 'rate', 'gain'):
            self.register_buffer(name, torch.as_tensor(getattr(physics, name), dtype=torch.float64))
        self.register_buffer('scales', torch.as_tensor(scales, dtype=torch.float64))
        self.register_buffer('phase', self.delta_star @ self.incidence)

    def initial(self, history, encoder=None):
        y = history[:, -1]
        return torch.cat((y[..., 0], y[..., 1] * self.mass, y[..., 2]), -1)

    def physical(self, state):
        n = self.n
        return torch.stack((state[..., :n], state[..., n:2 * n] / self.mass,
                            state[..., 2 * n:3 * n]), -1)

    def field(self, state, create_graph=False):
        n = self.n
        d, w, p = state[..., :n], state[..., n:2 * n] / self.mass, state[..., 2 * n:3 * n]
        electric = ((torch.sin(self.phase + d @ self.incidence) - torch.sin(self.phase))
                    * self.coupling) @ self.incidence.T
        derivative = torch.cat((w, p - electric - self.damping * w,
                                -self.rate * p - self.gain * w), -1)
        return derivative if create_graph else derivative.detach()


def coi(x, mass):
    return x - (x * mass).sum(-1, keepdims=True) / mass.sum()


def metric(pred, truth, mass, p_scale):
    """Existing metric API, with SSE/counts added for physical-unit pooling."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    mass = np.asarray(mass)
    if pred.shape != truth.shape or pred.ndim != 3 or len(pred) < 2:
        raise ValueError('Metrics require equal [time>=2,machine,3] arrays')
    err = pred[1:] - truth[1:]
    de, we = coi(err[..., 0], mass), coi(err[..., 1], mass)
    common = (err[..., 1] * mass).sum(-1) / mass.sum()
    ss = .5 * np.sum((de / DELTA_SCALE) ** 2 + (we / OMEGA_SCALE) ** 2)
    count = de.size
    return {
        'n': int(count), 'normalized_sse': float(ss), 'rmse': float(np.sqrt(ss / count)),
        'delta_rmse_rad': float(np.sqrt(np.mean(de * de))),
        'omega_rmse_rad_s': float(np.sqrt(np.mean(we * we))),
        'common_rmse_rad_s': float(np.sqrt(np.mean(common * common))),
        'p_rmse_pu': float(np.sqrt(np.mean(err[..., 2] ** 2))),
        'p_normalized_rmse': float(np.sqrt(np.mean((err[..., 2] / p_scale) ** 2))),
        'delta_sse_rad2': float(np.sum(de * de)),
        'omega_sse_rad_s2': float(np.sum(we * we)),
        'common_sse_rad_s2': float(np.sum(common * common)), 'common_n': int(common.size),
        'p_sse_pu2': float(np.sum(err[..., 2] ** 2)),
        'p_normalized_sse': float(np.sum((err[..., 2] / p_scale) ** 2)),
    }


def aggregate(rows):
    """Keep the original fail-closed aggregate API and add physical metrics."""
    result = {'count': len(rows), 'mean': None, 'pooled': None, 'tail': None,
              'numerically_valid': False}
    if not rows:
        return result
    unresolved = [r['id'] for r in rows if not r['qualified']]
    if unresolved:
        return {**result, 'unresolved_ids': unresolved}
    values = np.array([r['metrics']['rmse'] for r in rows])
    tail_n = max(1, math.ceil(.1 * len(values)))
    result.update(mean=float(values.mean()),
                  pooled=float(np.sqrt(sum(r['metrics']['normalized_sse'] for r in rows)
                                       / sum(r['metrics']['n'] for r in rows))),
                  tail=float(np.sort(values)[-tail_n:].mean()), numerically_valid=True)
    components = {
        'delta_rad': ('delta_rmse_rad', 'delta_sse_rad2', 'n'),
        'omega_rad_s': ('omega_rmse_rad_s', 'omega_sse_rad_s2', 'n'),
        'common_rad_s': ('common_rmse_rad_s', 'common_sse_rad_s2', 'common_n'),
        'p_pu': ('p_rmse_pu', 'p_sse_pu2', 'n'),
        'p_normalized': ('p_normalized_rmse', 'p_normalized_sse', 'n'),
    }
    result['physical'] = {}
    for name, (rmse_key, sse_key, n_key) in components.items():
        if not all(all(key in row['metrics'] for key in (rmse_key, sse_key, n_key)) for row in rows):
            continue  # Compatible with callers using the previous minimal metric dict.
        vals = np.array([row['metrics'][rmse_key] for row in rows])
        result['physical'][name] = {
            'mean': float(vals.mean()), 'tail': float(np.sort(vals)[-tail_n:].mean()),
            'pooled': float(np.sqrt(sum(row['metrics'][sse_key] for row in rows)
                                    / sum(row['metrics'][n_key] for row in rows))),
        }
    return result


def _validate_origins(origins, batch):
    """All five existing PowerGraph origins must have an explicit batch axis."""
    if (not isinstance(origins, tuple) or len(origins) != 5
            or any(not isinstance(term, torch.Tensor) or term.ndim < 1 or term.shape[0] != batch for term in origins)):
        raise ValueError('Origin terms must be five batch-first tensors matching the current state batch')


def _safe_field(model, states, origins=None):
    """Isolate exceptional rows only if a batched field evaluation raises."""
    if origins is not None:
        _validate_origins(origins, len(states))
    errors = [None] * len(states)
    try:
        with torch.no_grad():
            values = (model.field(states, create_graph=False) if origins is None else
                      model.field(states, create_graph=False, origins=origins)).detach()
        if values.shape != states.shape:
            raise ValueError('field must preserve the ODE state shape')
        return values, errors
    except (FloatingPointError, RuntimeError):
        values = torch.full_like(states, float('nan'))
        for i in range(len(states)):
            try:
                with torch.no_grad():
                    single_origins = None if origins is None else tuple(term[i:i + 1] for term in origins)
                    value = (model.field(states[i:i + 1], create_graph=False) if single_origins is None else
                             model.field(states[i:i + 1], create_graph=False, origins=single_origins)).detach()
                if value.shape != states[i:i + 1].shape:
                    raise ValueError('field must preserve the ODE state shape')
                values[i] = value[0]
            except (FloatingPointError, RuntimeError) as exc:
                errors[i] = f'{type(exc).__name__}: {exc}'
        return values, errors


def forecast_batch(model, z0, *, steps=800, max_step=.005, sample_dt=SAMPLE_DT):
    """RK4 without gradient graphs, removing only failed events from each stage.

    All accepted internal states are cached. A nonfinite RK stage/derivative
    immediately invalidates that event's current substep and later states.
    Earlier complete output intervals remain usable. No solver retry or
    alternative step size is chosen from validation performance.
    """
    if z0.ndim != 2 or not 1 <= len(z0) <= 8:
        raise ValueError('Forecast batches must contain 1..8 flat ODE states')
    if int(steps) != steps or steps < 1 or not np.isfinite(max_step) or max_step <= 0:
        raise ValueError('steps and max_step must be positive')
    if not np.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError('sample_dt must be positive')
    reuse = bool(getattr(model, 'reuse_origin_terms', False))
    if reuse:
        if type(model) is not PowerGraph:
            raise ValueError('Evaluation origin reuse supports only the exact PowerGraph')
        if getattr(model, '_compiled_field', None) is None or not getattr(model, '_compiled_uses_origins', False):
            raise ValueError('Evaluation origin reuse requires a trace built with explicit origin inputs')
    origins = None
    if reuse:
        with torch.no_grad():
            origins = model.origin_terms(z0)
        _validate_origins(origins, len(z0))
        ne = model.incidence.shape[1]
        expected = ((len(z0), model.n, 1, model.sdim + 4), (len(z0), ne),
                    (len(z0), ne, model.sdim), (len(z0), ne, model.sdim), (len(z0), model.n, 2))
        if tuple(tuple(term.shape) for term in origins) != expected:
            raise ValueError('Origin terms layout does not match the existing PowerGraph contract')
    sub = max(1, int(math.ceil(sample_dt / max_step - 1e-10)))
    dt = sample_dt / sub
    total = int(steps) * sub
    batch, width = z0.shape
    n = int(model.n)
    states = np.full((batch, total + 1, width), np.nan, dtype=float)
    physical = np.full((batch, total + 1, n, 3), np.nan, dtype=float)
    finite = np.zeros((batch, total + 1), dtype=bool)
    failures = [None] * batch
    z = z0.detach().clone()
    active = torch.isfinite(z).all(-1)

    def fail(indices, step, stage, stage_time, errors=None):
        ids = indices.detach().cpu().tolist()
        for pos, identity in enumerate(ids):
            if failures[identity] is None:
                failures[identity] = {
                    'first_bad_substep': int(step),
                    'first_missing_output_index': int(math.ceil(step / sub)),
                    'time_decision_s': float(stage_time),
                    'time_record_s': float(stage_time + .1), 'stage': stage,
                    'reason': errors[pos] if errors and errors[pos] else 'nonfinite_state_or_derivative',
                }
        active[indices] = False

    def save(step):
        ids = torch.nonzero(active, as_tuple=False).flatten()
        if not len(ids):
            return
        with torch.no_grad():
            view = model.physical(z[ids]).detach()
        if view.shape != (len(ids), n, 3):
            raise ValueError('physical readout must be [batch,machine,3]')
        good = torch.isfinite(view).all(dim=(-1, -2))
        if not bool(good.all()):
            fail(ids[~good], step, 'physical_readout', step * dt)
        ids = ids[good]
        indices = ids.detach().cpu().numpy()
        states[indices, step] = z[ids].detach().cpu().numpy()
        physical[indices, step] = view[good].cpu().numpy()
        finite[indices, step] = True

    fail(torch.nonzero(~active, as_tuple=False).flatten(), 0, 'initial_state', 0.0)
    save(0)
    for step in range(1, total + 1):
        if not bool(active.any()):
            break
        stages = []
        for stage, fraction in enumerate((0.0, .5, .5, 1.0)):
            incoming = z if stage == 0 else z + dt * fraction * stages[-1]
            ids = torch.nonzero(active, as_tuple=False).flatten()
            derivative = torch.zeros_like(z)
            if len(ids):
                good_input = torch.isfinite(incoming[ids]).all(-1)
                if not bool(good_input.all()):
                    fail(ids[~good_input], step, f'k{stage + 1}_input', (step - 1 + fraction) * dt)
                ids = ids[good_input]
            if len(ids):
                active_origins = None if origins is None else tuple(term[ids] for term in origins)
                values, errors = _safe_field(model, incoming[ids], active_origins)
                good = torch.isfinite(values).all(-1)
                if not bool(good.all()):
                    bad_errors = [error for error, valid in zip(errors, good.cpu().tolist()) if not valid]
                    fail(ids[~good], step, f'k{stage + 1}_field', (step - 1 + fraction) * dt, bad_errors)
                derivative[ids[good]] = values[good]
            stages.append(derivative)
        candidate = z + dt * (stages[0] + 2 * stages[1] + 2 * stages[2] + stages[3]) / 6
        good_final = torch.isfinite(candidate).all(-1)
        fail(torch.nonzero(active & ~good_final, as_tuple=False).flatten(), step, 'rk4_result', step * dt)
        z = candidate.detach()
        save(step)
    return {
        'states': states[:, ::sub], 'physical': physical[:, ::sub],
        'state_finite': finite[:, ::sub], 'substep_states': states,
        'substep_physical': physical, 'substep_state_finite': finite,
        'failures': failures, 'substeps_per_output': sub, 'actual_step_s': dt,
    }


def numerical_difference(first, second, mass, p_scale):
    """Original max component step difference on future scoring samples."""
    if first.shape != second.shape or len(first) < 2:
        return None
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        diff = first[1:] - second[1:]
        result = {
            'delta': float(np.max(np.abs(coi(diff[..., 0], mass))) / DELTA_SCALE),
            'omega': float(np.max(np.abs(coi(diff[..., 1], mass))) / OMEGA_SCALE),
            'common': float(np.max(np.abs((diff[..., 1] * mass).sum(-1) / mass.sum())) / OMEGA_SCALE),
            'p': float(np.max(np.abs(diff[..., 2])) / p_scale),
        }
    return result if all(np.isfinite(value) for value in result.values()) else None


def _optional_float(value):
    if value is None or str(value).strip() == '':
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError('Timing metadata must be finite when supplied')
    return result


def _timing(event, index):
    metadata = event.metadata or {}
    clearing = _optional_float(metadata.get('clearing_time_s'))
    delay = _optional_float(metadata.get('record_start_delay_s'))
    true_time = _optional_float(metadata.get('event_time_s'))
    record = None if index is None else index * SAMPLE_DT
    times = {
        'predicted_los_time_record_s': record,
        'predicted_los_time_decision_s': None if record is None else record - .1,
        'predicted_los_time_absolute_s': None,
    }
    # A present blank delay has the source evaluator's documented zero-delay
    # semantics. An absent metadata field is not invented as zero.
    delay_known = 'record_start_delay_s' in metadata
    if clearing is not None and delay_known:
        times = trigger_times(index, clearing_time_s=clearing, record_start_delay_s=delay)
    effective_delay = (0.0 if delay is None else delay) if delay_known else None
    return {
        **times, 'true_event_time_s': true_time, 'clearing_time_s': clearing,
        'record_start_delay_s': delay,
        'predicted_los_time_since_clearing_s': None if record is None or effective_delay is None else record + effective_delay,
        'true_event_time_since_clearing_s': None if true_time is None or effective_delay is None else true_time + effective_delay,
        'trigger_time_error_s': None if record is None or true_time is None else record - true_time,
        'absolute_timing_available': clearing is not None and delay_known,
    }


def _summary(rows, population):
    scorable = [row for row in rows if row['scorable']]
    return {
        **aggregate(scorable), 'eligible_denominator': population,
        'truth_scorable_denominator': len(scorable),
        'truth_unavailable_count': len(rows) - len(scorable),
        'numerically_qualified_count': sum(row['qualified'] for row in scorable),
        'numerically_unresolved_count': sum(not row['qualified'] for row in scorable),
        'truth_unavailable_ids': [row['id'] for row in rows if not row['scorable']],
    }


def evaluate(model, encoder, events, output, horizons=(1., 2., 5., 8.),
             fine_dt=.005, coarse_dt=.01, tolerance=.001, *,
             physics=None, facts=None, batch_size=8, decision_horizon_s=DECISION_HORIZON,
             endpoint=None):
    """Evaluate frozen candidates; optional shorter horizons are engineering-only.

    ``physics`` supplies authoritative omega_s/delta_star. Without it the
    trajectory metrics remain available but physical classification is
    explicitly unavailable. ``facts`` carries the raw population and
    metadata-only latency exclusions. No data source is opened here.
    """
    torch.set_num_threads(1)
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
        raise ValueError('batch_size must be in 1..8')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive')
    steps = round(float(decision_horizon_s) / SAMPLE_DT)
    if steps < 1 or not np.isclose(steps * SAMPLE_DT, decision_horizon_s, atol=1e-10, rtol=0):
        raise ValueError('decision horizon must be on the 10-ms grid')
    horizons = tuple(float(h) for h in horizons)
    if not horizons or any(h <= 0 or h > decision_horizon_s or not np.isclose(round(h / SAMPLE_DT) * SAMPLE_DT, h) for h in horizons):
        raise ValueError('Scoring horizons must lie on the forecast grid within the decision horizon')
    if len(set(horizons)) != len(horizons):
        raise ValueError('Duplicate scoring horizons')
    if endpoint is not None and endpoint not in ('P-N', 'P-S', 'A', 'U', 'M3'):
        raise ValueError('Unknown registered endpoint')
    events = list(events)
    identities = [event.id for event in events]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate evaluation event IDs')
    source_facts = facts or {}
    excluded = [dict(row, group=row.get('group', 'unknown')) for row in source_facts.get('excluded', [])]
    if any(row.get('reason') != 'latency_uncovered' for row in excluded):
        raise ValueError('Unrecognized excluded event reason; reconcile population before scoring')
    raw_count = int(source_facts.get('raw_count', len(events) + len(excluded)))
    if raw_count != len(events) + len(excluded) or int(source_facts.get('eligible_count', len(events))) != len(events):
        raise ValueError('Facts and event populations disagree; cannot alter denominators')
    if source_facts.get('test_payload_read', False):
        raise PermissionError('TEST payload evidence is not authorized')
    output = project_path(output)
    cache_dir = project_path(output.with_suffix('.cache'))
    cache_dir.mkdir(parents=True, exist_ok=False)
    mass = model.mass.detach().cpu().numpy()
    p_scale = float(model.scales[2])
    if not np.isfinite(p_scale) or p_scale <= 0:
        raise ValueError('Power scale must be finite and positive')
    if physics is not None and not np.allclose(np.asarray(physics.mass), mass):
        raise ValueError('Evaluation physics mass differs from model')
    model.eval()
    if encoder is not None:
        encoder.eval()
    rows, classification_rows, numerical_rows, caches, failures = [], [], [], [], []
    device = model.mass.device
    start = time.monotonic()
    initial_calls = 0
    for offset in range(0, len(events), batch_size):
        batch_events = events[offset:offset + batch_size]
        for event in batch_events:
            if event.label not in ('stable', 'unstable') or len(event.y) < 11:
                raise ValueError(f'Invalid event label/history: {event.id}')
            if not np.isfinite(event.y[:11]).all():
                raise ValueError(f'Nonfinite causal history: {event.id}')
        history = np.stack([event.y[:11] for event in batch_events])
        hist = torch.as_tensor(history, dtype=torch.float64, device=device)
        with torch.no_grad():
            z0 = model.initial(hist, encoder).detach()
        initial_calls += 1
        fine = forecast_batch(model, z0, steps=steps, max_step=fine_dt)
        coarse = forecast_batch(model, z0, steps=steps, max_step=coarse_dt)
        arrays = {
            'event_ids': np.asarray([event.id for event in batch_events], dtype=str),
            'history': history, 'initial_state': z0.cpu().numpy(), 'mass': mass,
            'p_scale': np.asarray(p_scale), 'sample_interval_s': np.asarray(SAMPLE_DT),
            'decision_horizon_s': np.asarray(decision_horizon_s),
            'fine_states': fine['states'], 'coarse_states': coarse['states'],
            'fine_physical': fine['physical'], 'coarse_physical': coarse['physical'],
            'fine_state_finite': fine['state_finite'], 'coarse_state_finite': coarse['state_finite'],
            'fine_substep_states': fine['substep_states'], 'coarse_substep_states': coarse['substep_states'],
            'fine_substep_physical': fine['substep_physical'], 'coarse_substep_physical': coarse['substep_physical'],
            'fine_substep_state_finite': fine['substep_state_finite'],
            'coarse_substep_state_finite': coarse['substep_state_finite'],
            'fine_substeps_per_output': np.asarray(fine['substeps_per_output']),
            'coarse_substeps_per_output': np.asarray(coarse['substeps_per_output']),
        }
        if physics is not None:
            arrays.update(delta_star=np.asarray(physics.delta_star), omega_s=np.asarray(physics.omega_s))
        batch_metadata = []
        for local, event in enumerate(batch_events):
            # Save true ragged arrays directly. No extension/padding/freezing.
            arrays[f'truth_{local:03d}'] = np.asarray(event.y)
            arrays[f'truth_time_{local:03d}'] = np.asarray(event.t)
            batch_metadata.append({'id': event.id, 'label': event.label, 'group': event.group,
                                   'source': event.source, 'metadata': event.metadata,
                                   'fine_failure': fine['failures'][local], 'coarse_failure': coarse['failures'][local]})
            failures.append({'id': event.id, 'fine': fine['failures'][local], 'coarse': coarse['failures'][local]})
            available = len(event.y) - 11
            requests = [(str(h), round(h / SAMPLE_DT), 'complete') for h in horizons]
            requests.append(('prefix', min(steps, available), 'available_prefix'))
            for name, count, scope in requests:
                scorable = count >= 1 and available >= count
                row = {'id': event.id, 'group': event.group, 'label': event.label,
                       'horizon': name, 'scope': scope, 'steps': count,
                       'true_available_steps': available, 'scorable': scorable,
                       'qualified': False, 'metrics': None, 'numerical_difference': None}
                if not scorable:
                    row['failure'] = 'truth_interval_unavailable'
                    rows.append(row)
                    continue
                f, c = fine['physical'][local, :count + 1], coarse['physical'][local, :count + 1]
                fine_finite = bool(fine['substep_state_finite'][local, :count * fine['substeps_per_output'] + 1].all())
                coarse_finite = bool(coarse['substep_state_finite'][local, :count * coarse['substeps_per_output'] + 1].all())
                row.update(finite_fine=fine_finite, finite_coarse=coarse_finite)
                difference = numerical_difference(f, c, mass, p_scale)
                row['numerical_difference'] = difference
                row['qualified'] = bool(fine_finite and coarse_finite and difference is not None
                                        and max(difference.values()) <= tolerance)
                if np.isfinite(f).all():
                    with np.errstate(over='ignore', invalid='ignore'):
                        metrics = metric(f, event.y[10:11 + count], mass, p_scale)
                    if all(np.isfinite(value) for value in metrics.values()):
                        row['metrics'] = metrics
                    else:
                        row['qualified'] = False
                        row['failure'] = 'nonfinite_metric_statistic'
                if not row['qualified']:
                    reason = 'nonfinite_prediction' if not (fine_finite and coarse_finite) else 'step_discrepancy'
                    row.setdefault('failure', reason)
                rows.append(row)
            if physics is not None:
                pair, passed = numerical_pair(
                    fine['physical'][local], coarse['physical'][local], history[local],
                    mass, physics.delta_star, physics.omega_s,
                    first_latent_finite=fine['state_finite'][local],
                    second_latent_finite=coarse['state_finite'][local],
                    first_substeps=fine['substep_physical'][local],
                    first_substeps_latent_finite=fine['substep_state_finite'][local],
                    substep_ratio=fine['substeps_per_output'], decision_horizon_s=decision_horizon_s,
                )
                endpoint_index = max(0, pair['comparison_endpoint_index'] - 10)
                # The original coarse grid is 10 ms. If a caller requests
                # smaller coarse substeps, include their finite-state audit.
                coarse_to_endpoint = bool(coarse['substep_state_finite'][local, :endpoint_index * coarse['substeps_per_output'] + 1].all())
                passed = bool(passed and coarse_to_endpoint)
                pair.update(id=event.id, screen_pass=passed,
                            coarse_substeps_finite_to_endpoint=coarse_to_endpoint,
                            supplemental_max_component_difference=numerical_difference(
                                fine['physical'][local, :endpoint_index + 1],
                                coarse['physical'][local, :endpoint_index + 1], mass, p_scale),
                            supplemental_differences_change_original_los_gate=False)
                nonfinite_statistics = [key for key, value in pair.items()
                                        if isinstance(value, float) and not np.isfinite(value)]
                if nonfinite_statistics:
                    for key in nonfinite_statistics:
                        pair[key] = None
                    pair['nonfinite_screen_statistics'] = nonfinite_statistics
                    passed = False
                    pair['screen_pass'] = False
                numerical_rows.append(pair)
                true_event_time = _optional_float((event.metadata or {}).get('event_time_s'))
                category = classify_forecast(
                    history[local], fine['physical'][local], mass, physics.delta_star, physics.omega_s,
                    latent_finite=fine['state_finite'][local], true_event_time_s=true_event_time,
                    numerical_passed=passed, decision_horizon_s=decision_horizon_s,
                )
                classification_rows.append({'id': event.id, 'group': event.group, 'label': event.label,
                                            **category, **_timing(event, category['trigger_index'])})
        batch_number = offset // batch_size
        cache = cache_dir / f'batch_{batch_number:04d}.npz'
        np.savez_compressed(cache, **arrays)
        write_json(cache_dir / f'batch_{batch_number:04d}.json', batch_metadata)
        caches.append({'path': str(cache), 'metadata_path': str(cache.with_suffix('.json')),
                       'event_ids': [event.id for event in batch_events],
                       'sha256': cache_sha256(cache),
                       'metadata_sha256': cache_sha256(cache.with_suffix('.json'))})
    summary = {str(h): _summary([row for row in rows if row['horizon'] == str(h)], len(events)) for h in horizons}
    summary['prefix'] = _summary([row for row in rows if row['horizon'] == 'prefix'], len(events))
    severe_rows = [row for row in rows if row['horizon'] == 'prefix' and row['label'] == 'unstable']
    summary['severe_prefix'] = _summary(severe_rows, sum(event.label == 'unstable' for event in events))
    raw_unstable = sum(event.label == 'unstable' for event in events) + sum(row['label'] == 'unstable' for row in excluded)
    for name, item in summary.items():
        item['raw_denominator'] = raw_unstable if name == 'severe_prefix' else raw_count
        item['latency_uncovered_count'] = sum(row['label'] == 'unstable' for row in excluded) if name == 'severe_prefix' else len(excluded)
    if physics is not None:
        classification = summarize_classifications(classification_rows, excluded)
        classification['status'] = 'frozen_los_and_window_screen'
        classification['evidence_role'] = 'development' if source_facts.get('split') == 'validation' else 'engineering_or_train'
    else:
        classification = {'status': 'unavailable_missing_authoritative_physics',
                          'event_count_raw': raw_count, 'eligible_event_count': len(events),
                          'latency_uncovered_count': len(excluded), 'latency_uncovered_rows': excluded}
    result = {
        'summary': summary, 'rows': rows, 'severe_prefix_rows': severe_rows,
        'classification': classification, 'classification_numerical': numerical_rows,
        'classification_status': classification['status'], 'integration_failures': failures,
        'history_ms': 100, 'encoder_calls_per_event': 0 if isinstance(model, FrozenM3) else 1,
        'encoder_calls_per_forecast': 0 if isinstance(model, FrozenM3) else 1,
        'initializations_per_event': 1,
        'initialization_batch_calls': initial_calls, 'batch_size': batch_size,
        'forecast_steps': steps, 'decision_horizon_s': float(decision_horizon_s),
        'test_payload_read': False, 'raw_event_count': raw_count, 'eligible_event_count': len(events),
        'latency_uncovered_count': len(excluded), 'source_facts': source_facts,
        'scope': 'genuine-truth accuracy intervals; full fixed decision-window classification',
        'fine_dt': fine_dt, 'coarse_dt': coarse_dt, 'tolerance': tolerance,
        'classification_rms_tolerance': .001, 'classification_hit_time_tolerance_s': .02,
        'endpoint': endpoint, 'evidence_role': 'development' if source_facts.get('split') == 'validation' else 'engineering_or_train',
        'model_identity': getattr(model, 'evaluation_identity', None),
        'comparison_context': {
            'saved_training_scales': np.asarray(model.scales.detach().cpu()).tolist(),
            'metric_scales_delta_omega': [DELTA_SCALE, OMEGA_SCALE],
            'mass': mass.tolist(),
            'source_physics': physics.as_dict() if callable(getattr(physics,'as_dict',None)) else None,
            'truth_time_origin': 'postfault t=0; initializer ends at t=0.1 s',
            'error_time_origin': 'forecast elapsed after t=0.1 s',
            'history_samples': 11,
        },
        'cache': {'directory': str(cache_dir), 'batches': caches,
                  'generation_hash_algorithm': 'SHA256 after each cache/metadata file is written',
                  'missing_predictions': 'NaN after first invalid RK stage; never truth padding',
                  'truth_layout': 'unmodified ragged truth_000, truth_time_000 etc. within each batch'},
        'evaluation_wall_seconds': time.monotonic() - start,
        'timing': {'status': 'not_a_same_accuracy_speed_benchmark'},
    }
    write_json(output, result)
    return result
