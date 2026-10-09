"""Bounded deployment timing of two explicitly frozen endpoints, TRAIN only.

Run only through the TASK4 managed runner. These runtime ratios are neither
equal-accuracy comparisons nor speedups over FOM. No endpoint is selected here.
"""
import os

# Set BLAS limits before any NumPy/Torch import in the standalone process.
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
THREAD_LIMITS_SET_BEFORE_IMPORT = __name__ == '__main__'
if THREAD_LIMITS_SET_BEFORE_IMPORT:
    for _name in THREAD_ENV:
        os.environ[_name] = '1'

import _bootstrap  # noqa: E402,F401
import argparse
from contextlib import nullcontext
import hashlib
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from task4.paths import ROOT, project_path, read_json, source_file, source_root, write_json

HORIZON, SAMPLE_DT, STEP = 5.0, 0.01, 0.0025
PANEL_LIMIT, WARMUPS, REPEATS = 8, 1, 3
PARITY_TOLERANCE = 1e-6


def check_deadline(deadline):
    if time.time() >= deadline - 5.0:
        raise TimeoutError('Managed deadline reached; reserve remaining time for partial evidence')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def file_stat(path):
    stat = Path(path).stat()
    return {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def output_grid(horizon=HORIZON, sample_dt=SAMPLE_DT, step=STEP):
    if not all(np.isfinite(x) and x > 0 for x in (horizon, sample_dt, step)):
        raise ValueError('Positive finite time increments required')
    samples, substeps = round(horizon / sample_dt), round(sample_dt / step)
    if (samples < 1 or substeps < 1
            or not np.isclose(samples * sample_dt, horizon, rtol=0, atol=1e-12)
            or not np.isclose(substeps * step, sample_dt, rtol=0, atol=1e-12)):
        raise ValueError('Horizon and output interval must be exact multiples of the fixed steps')
    return np.arange(samples + 1, dtype=np.float64) * sample_dt, substeps


def deploy(model, history, *, backend, deadline, encoder=None, reuse_origins=False,
           horizon=HORIZON, sample_dt=SAMPLE_DT, step=STEP):
    """One initializer call and autonomous RK4, retaining only output samples.

    Stage inputs AND derivatives include all latent coordinates in finite tests.
    A failed batch is invalid as a whole; no survivor-only timing is substituted.
    Host output preparation, origin creation, finite checks, and conversion are
    included in the measured deployment call. File I/O is outside this timer.
    """
    started = time.perf_counter()
    grid, substeps = output_grid(horizon, sample_dt, step)
    if backend not in ('torch', 'numpy'):
        raise ValueError('Unknown deployment backend')
    is_torch = backend == 'torch'
    batch = len(history)
    values = np.full((batch, len(grid), model.n, 3), np.nan, dtype=np.float64)
    states = np.full((batch, len(grid), model.dim), np.nan, dtype=np.float64)
    report = {'status': 'incomplete', 'complete': False, 'initial_calls': 0,
              'output_samples_completed': 0, 'first_bad': None, 'error': None,
              'retained_fine_substep_states': False}
    cursor = {'output_index': 0, 'fine_step_index': 0, 'stage': 'initial',
              'forecast_elapsed_seconds': 0.0}
    try:
        check_deadline(deadline)
        with torch.no_grad() if is_torch else nullcontext():
            data = (torch.as_tensor(history, dtype=torch.float64, device='cpu')
                    if is_torch else np.asarray(history, dtype=np.float64))
            report['initial_calls'] += 1
            z = model.initial(data, encoder) if is_torch else model.initial(data)

            def finite(value, stage):
                cursor['stage'] = stage
                mask = torch.isfinite(value) if is_torch else np.isfinite(value)
                okay = bool(mask.all())
                if not okay:
                    per_event = mask.reshape(batch, -1).all(-1)
                    bad = (~per_event).nonzero().flatten().tolist() if is_torch else np.flatnonzero(~per_event).tolist()
                    report['first_bad'] = {**cursor, 'batch_indices': bad}
                    raise FloatingPointError('Nonfinite full state or field at ' + stage)
                return value

            finite(z, 'initial')
            origins = model.origin_terms(z) if reuse_origins else None
            if origins is not None:
                for origin in origins:
                    finite(origin, 'origin_terms')

            def field(value, stage):
                finite(value, stage + '_input')
                cursor['stage'] = stage + '_field'
                if is_torch:
                    result = (model.field(value, create_graph=False, origins=origins)
                              if reuse_origins else model.field(value, create_graph=False))
                else:
                    result = model.field(value)
                return finite(result, stage + '_derivative')

            def save_sample(index):
                physical = finite(model.physical(z), 'physical_output')
                values[:, index] = physical.detach().cpu().numpy() if is_torch else physical
                states[:, index] = z.detach().cpu().numpy() if is_torch else z
                report['output_samples_completed'] = index + 1

            save_sample(0)
            for index in range(1, len(grid)):
                check_deadline(deadline)  # Every 10 ms output sample, for both backends.
                cursor['output_index'] = index
                for fine in range(substeps):
                    cursor['fine_step_index'] = (index - 1) * substeps + fine
                    cursor['forecast_elapsed_seconds'] = cursor['fine_step_index'] * step
                    a = field(z, 'k1')
                    cursor['forecast_elapsed_seconds'] += step / 2
                    b = field(z + step * a / 2, 'k2')
                    c = field(z + step * b / 2, 'k3')
                    cursor['forecast_elapsed_seconds'] += step / 2
                    d = field(z + step * c, 'k4')
                    z = finite(z + step * (a + 2 * b + 2 * c + d) / 6, 'accepted_state')
                save_sample(index)
            check_deadline(deadline)
            report.update(status='complete', complete=True)
    except (Exception, KeyboardInterrupt) as exc:
        report['status'] = 'deadline' if isinstance(exc, TimeoutError) else 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        if report['first_bad'] is None:
            report['first_bad'] = dict(cursor)
    report['wall_seconds'] = time.perf_counter() - started
    report['all_output_finite'] = bool(np.isfinite(values).all())
    report['all_sampled_full_states_finite'] = bool(np.isfinite(states).all())
    report['full_state_dimension_checked'] = model.dim
    report['every_rk_stage_full_state_and_derivative_checked'] = True
    report['valid_runtime'] = bool(report['complete'] and report['all_output_finite']
                                   and report['all_sampled_full_states_finite'])
    return values, states, report


def select_panel(infos, limit=PANEL_LIMIT):
    """Metadata-only deterministic selection before opening an event group."""
    selected, raw, eligible, excluded, seen_pairs, seen_ids = [], [], [], [], set(), set()
    for relative, hp, splits, rows in infos:
        for eid in splits['train']:
            if eid in seen_ids:
                raise ValueError('Duplicate cross-dataset TRAIN ID: ' + eid)
            seen_ids.add(eid)
            row = rows[eid]
            if row['label'] not in ('stable', 'unstable'):
                raise ValueError('Unrecognized original TRAIN label')
            entry = {'id': eid, 'group': str(row['fault_bus']), 'label': row['label'],
                     'source': str(relative)}
            raw.append(entry)
            end = float(row['trajectory_end_s'])
            if not np.isfinite(end) or end < 0:
                raise ValueError('Invalid TRAIN trajectory end metadata')
            if end <= 0.1 + 1e-9:
                excluded.append(entry)
                continue
            eligible.append(entry)
            pair = (entry['group'], entry['label'])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                if len(selected) < limit:
                    selected.append((entry, hp))
    if not selected:
        raise ValueError('No eligible TRAIN history')
    return selected, {'raw': raw, 'eligible': eligible, 'latency_uncovered': excluded}


def read_panel(config, checkpoint_physics, expected_train_ids, deadline):
    from task4.data import load_physics, split_metadata
    kiki = source_root(config)
    metadata_hashes = {}
    for path in [source_file(kiki, config['baseline'])] + [
            source_file(kiki, relative) / name
            for relative in config['datasets'] for name in ('splits.json', 'dataset_index.csv')]:
        metadata_hashes[str(path)] = sha256(path)
    physics = load_physics(config)
    for key, value in physics.as_dict().items():
        if key != 'omega_s' and not np.allclose(value, checkpoint_physics[key], rtol=1e-9, atol=1e-12):
            raise ValueError('Checkpoint disagrees with original physical baseline: ' + key)
    if config['sample_dt'] != SAMPLE_DT or config['history_ms'] != 100:
        raise ValueError('Expected 11 samples over 100 ms')
    infos = split_metadata(config)
    selected, facts = select_panel(infos)
    eligible_ids = [entry['id'] for entry in facts['eligible']]
    if len(expected_train_ids) != 2 or any(ids != eligible_ids for ids in expected_train_ids):
        raise ValueError('Both frozen checkpoint TRAIN ID lists/order must match legal source splits before payload access')
    for metric, value in (('raw', len(facts['raw'])), ('eligible', len(eligible_ids)),
                          ('unstable', sum(e['label'] == 'unstable' for e in facts['eligible']))):
        expected = config.get('expected', {}).get('train_' + metric)
        if expected is not None and expected != value:
            raise ValueError('Legal source TRAIN metadata count changed: ' + metric)
    histories, hdf_stats = [], {}
    for entry, hp in selected:
        check_deadline(deadline)
        hdf_stats.setdefault(str(hp), file_stat(hp))
        with h5py.File(hp, 'r') as handle:
            ws = float(handle.attrs['omega_s'])
            inertia = np.asarray(handle['inertia'], dtype=np.float64)
            delta0 = np.asarray(handle['equilibrium/delta'], dtype=np.float64)
            pm0 = np.asarray(handle['equilibrium/Pm'], dtype=np.float64)
            if (not np.isclose(ws, checkpoint_physics['omega_s'], rtol=1e-12, atol=1e-12)
                    or not np.allclose(inertia / ws, physics.mass, rtol=1e-8, atol=1e-10)
                    or not np.allclose(pm0, physics.pm_star, rtol=1e-8, atol=1e-10)
                    or not np.allclose(delta0 @ physics.incidence,
                                       physics.delta_star @ physics.incidence, rtol=0, atol=1e-8)):
                raise ValueError('Source history physical units/equilibrium mismatch')
            # Explicit authorized ID and explicit 11-point slice, never group traversal.
            event = handle['events'][entry['id']]
            t = np.asarray(event['t'][:11], dtype=np.float64)
            delta = np.asarray(event['delta'][:11], dtype=np.float64) - delta0
            omega = (np.asarray(event['omega'][:11], dtype=np.float64) - 1.0) * ws
            power = np.asarray(event['Pm'][:11], dtype=np.float64) - pm0
            history = np.stack((delta, omega, power), axis=-1)
            if (t.shape != (11,) or not np.allclose(t, np.arange(11) * SAMPLE_DT, rtol=0, atol=1e-9)
                    or history.shape != (11, physics.n, 3) or not np.isfinite(history).all()):
                raise ValueError('Invalid selected history: ' + entry['id'])
            histories.append(history)
            entry['canonical_history_sha256'] = hashlib.sha256(history.tobytes()).hexdigest()
    facts.update(split='train', selected=[entry for entry, _ in selected],
                 raw_count=len(facts['raw']), eligible_count=len(facts['eligible']),
                 selected_count=len(selected),
                 panel_rule='first eligible per (group,label) in legal TRAIN order, at most 8',
                 payload_slices='selected events only: t/delta/omega/Pm[:11]',
                 future_train_payload_read=False, validation_payload_read=False,
                 test_payload_read=False, unselected_train_payload_read=False,
                 both_checkpoint_train_ids_exact_match_before_payload=True,
                 hdf5_stat_before=hdf_stats, source_metadata_sha256_before=metadata_hashes,
                 hdf5_whole_file_hash_performed=False,
                 hdf5_integrity_limit='file stat plus selected history hashes; full HDF5 hash would read unauthorized payloads')
    return np.stack(histories), facts


def parity(candidate, reference, scales):
    result = {'passed': False, 'valid_runtime': False, 'max_normalized_difference': None,
              'per_channel_max_normalized_difference': None,
              'threshold': PARITY_TOLERANCE, 'all_output_samples_including_initial': True,
              'full_delta_and_omega_including_common_motion': True}
    if (candidate.shape != reference.shape or candidate.ndim != 4 or candidate.shape[-1] != 3
            or candidate.size == 0 or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        return {**result, 'reason': 'incomplete or nonfinite outputs'}
    scale = np.asarray(scales, dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        return {**result, 'reason': 'invalid physical normalization scales'}
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        absolute = np.abs(candidate - reference)
        if not np.isfinite(absolute).all():
            return {**result, 'reason': 'difference overflow from finite physical inputs'}
        difference = absolute / scale
    if not np.isfinite(difference).all():
        return {**result, 'reason': 'normalization overflow from finite physical differences'}
    maximum = float(difference.max())
    passed = bool(maximum <= PARITY_TOLERANCE)
    return {**result, 'passed': passed, 'valid_runtime': passed, 'max_normalized_difference': maximum,
            'per_channel_max_normalized_difference': difference.max(axis=(0, 1, 2)).tolist(),
            'reason': None if passed else 'physical implementation difference exceeds fixed tolerance'}


def state_parity(candidate, reference, mass, scales, rank):
    """Report inferred latent agreement, without any latent-truth assertion."""
    result = {'complete_finite': False, 'passed': False, 'valid_runtime': False,
              'max_normalized_difference': None, 'per_state_max_normalized_difference': None,
              'scale_vector': None, 'latent_truth_claim': False,
              'passed_scope': 'finite diagnostic arithmetic only; no full-state tolerance gate',
              'role': 'diagnostic same-model implementation agreement; physical gate remains separate'}
    if (candidate.shape != reference.shape or candidate.ndim != 3 or candidate.size == 0
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        return {**result, 'reason': 'incomplete or nonfinite full-state outputs'}
    scales, mass = np.asarray(scales, dtype=np.float64), np.asarray(mass, dtype=np.float64)
    if (scales.shape != (3,) or mass.ndim != 1 or not np.isfinite(scales).all()
            or not np.isfinite(mass).all() or (scales <= 0).any() or (mass <= 0).any()):
        return {**result, 'reason': 'invalid full-state normalization scales or mass'}
    n = len(mass)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        scale = np.concatenate((np.full(n, scales[0]), mass * scales[1],
                                np.full(n, scales[2]), np.ones(n * rank)))
        if (scale.shape != candidate.shape[-1:] or not np.isfinite(scale).all()
                or (scale <= 0).any()):
            return {**result, 'reason': 'full-state scale product overflow/underflow or dimension mismatch'}
        result['scale_vector'] = scale.tolist()
        absolute = np.abs(candidate - reference)
        if not np.isfinite(absolute).all():
            return {**result, 'reason': 'difference overflow from finite full-state inputs'}
        difference = absolute / scale
    if not np.isfinite(difference).all():
        return {**result, 'reason': 'normalization overflow from finite full-state differences'}
    return {**result, 'complete_finite': True, 'passed': True, 'valid_runtime': True,
            'max_normalized_difference': float(difference.max()),
            'per_state_max_normalized_difference': difference.max(axis=(0, 1)).tolist(),
            'reason': None}


def resources():
    import psutil
    process = psutil.Process()
    memory, cpu = process.memory_info(), process.cpu_times()
    # Windows has an OS process high-water mark. On POSIX query getrusage.
    peak = getattr(memory, 'peak_wset', None)
    method = 'Windows OS peak_wset'
    if peak is None:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(peak if sys.platform == 'darwin' else peak * 1024)
        method = 'getrusage process high-water RSS'
    return {'rss_bytes': memory.rss, 'process_peak_rss_bytes': int(peak), 'peak_method': method,
            'cpu_user_seconds': cpu.user, 'cpu_system_seconds': cpu.system,
            'os_threads': process.num_threads(), 'peak_scope': 'whole process including model load and tracing'}


def save_arrays(path, grid, values, states):
    with Path(path).open('xb') as stream:
        np.savez(stream, forecast_elapsed_seconds=grid, physical=values, sampled_full_state=states)


def validate_models(neural, symbolic, runtime, encoder_s, obj_n, obj_s, spec, eligible_ids):
    from task4.dynamics import PowerGraph
    from task4.symbolic_runtime import export_symbolic
    for model, obj, kind in ((neural, obj_n, 'T4-N'), (symbolic, obj_s, 'T4-S')):
        if (type(model) is not PowerGraph or obj.get('kind') != kind
                or obj.get('synthetic') is not False or obj['config'].get('synthetic') is not False
                or obj['config'].get('arm') != 'P' or obj.get('event_ids') != eligible_ids):
            raise ValueError('Expected fixed nonsynthetic P endpoint with exact legal TRAIN IDs: ' + kind)
        if not np.array_equal(model.scales.detach().numpy(), obj['scales']):
            raise ValueError('Stored TRAIN scales disagree with endpoint buffers')
        for key, value in obj['physics'].items():
            if key != 'omega_s' and not np.allclose(getattr(model, key).detach().numpy(), value, rtol=0, atol=0):
                raise ValueError('Saved physics and model buffers disagree: ' + key)
    if neural.n != symbolic.n or neural.rank != symbolic.rank or symbolic.dim != runtime.dim:
        raise ValueError('Endpoint state dimensions differ')
    if obj_n['physics'] != obj_s['physics'] or obj_n['scales'] != obj_s['scales']:
        raise ValueError('Endpoints must use the same original physics and saved TRAIN scales')
    exported = export_symbolic(symbolic, encoder_s, omega_s=obj_s['physics']['omega_s'])
    # Metadata prose can differ; every executable scientific component must match.
    for key in ('format', 'version', 'n', 'rank', 'strength', 'physics', 'omega_s',
                'scales', 'log_coupling', 'mu', 'phase', 'edges', 'functions', 'encoder'):
        if exported[key] != spec[key]:
            raise ValueError('Symbolic JSON is not the supplied frozen Torch endpoint: ' + key)


def benchmark(*, neural_checkpoint, symbolic_checkpoint, symbolic_json, output):
    from task4.training import check_authorized, load_checkpoint
    check_authorized(False)  # Before opening any checkpoint, JSON, or source history.
    if not THREAD_LIMITS_SET_BEFORE_IMPORT:
        raise RuntimeError('Run this benchmark as a standalone child process so BLAS limits precede library imports')
    from task4.accelerate import trace_field
    from task4.symbolic_runtime import SymbolicPowerGraph
    deadline = float(os.environ['TASK4_SAVE_DEADLINE_UNIX'])
    if not np.isfinite(deadline):
        raise ValueError('Finite managed deadline required')
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = {'neural_checkpoint': project_path(neural_checkpoint),
             'symbolic_checkpoint': project_path(symbolic_checkpoint),
             'symbolic_json': project_path(symbolic_json), 'project_config': ROOT / 'configs/project.json'}
    sources = dict(paths)
    for name in ('benchmark_endpoints.py',):
        sources['script_' + name] = ROOT / 'scripts' / name
    for name in ('accelerate.py', 'dynamics.py', 'encoder.py', 'symbolic.py', 'symbolic_runtime.py', 'training.py', 'data.py'):
        sources['code_' + name] = ROOT / 'src/task4' / name
    started = time.perf_counter()
    report = {'status': 'inconclusive', 'complete': False, 'stop_reason': 'started',
              'managed_run_id': os.environ.get('TASK4_MANAGED_RUN_ID'),
              'managed_save_deadline_unix': deadline, 'sources': {k: str(v) for k, v in sources.items()},
              'source_sha256_before': {}, 'source_sha256_after': {}, 'stages': [], 'trials': [],
              'timing_summary': {}, 'runtime_ratios_N_over_S': {}, 'same_accuracy_claim': False,
              'fom_speedup_claim': False, 'fom_run': False, 'model_selection_performed': False,
              'test_payload_read': False, 'validation_payload_read': False,
              'observation_delay_seconds': 0.1,
              'precision_and_safety_status': 'requires corresponding frozen capability reports; timing alone cannot establish either',
              'policy': {'device': 'CPU', 'dtype': 'float64', 'compute_threads': 1,
                         'forecast_seconds': HORIZON, 'rk4_step_seconds': STEP, 'output_dt_seconds': SAMPLE_DT,
                         'warmups_per_backend_and_batch': WARMUPS, 'timed_repeats': REPEATS,
                         'maximum_batch': PANEL_LIMIT, 'parity_threshold': PARITY_TOLERANCE,
                         'timing_includes': 'one GRU initializer + origin preparation + full RK4 + stage finite checks + sampled outputs',
                         'timing_excludes': 'checkpoint/JSON read, history read, trace construction, file I/O, 100 ms observations',
                         'neural_backend': 'Torch traced field, once-per-rollout origins, no gradients',
                         'symbolic_backend': 'independent NumPy JSON runtime',
                         'torch_symbolic_role': 'one batch 5 s numerical reference only, excluded from timing backend'},
              'cost_seconds': {}}
    report['planned_trials'] = [
        {'endpoint': name, 'batch_mode': mode, 'trial_index': trial,
         'phase': 'warmup' if trial < WARMUPS else 'timed'}
        for mode in ('single', 'batch') for name in ('N', 'S')
        for trial in range(WARMUPS + REPEATS)]

    def persist(stage=None):
        if stage is not None:
            report['stages'].append(stage)
        report['process_wall_seconds'] = time.perf_counter() - started
        report['resources'] = resources()
        write_json(output / 'benchmark.json', report)

    persist()
    failure = None
    try:
        check_deadline(deadline)
        for name in THREAD_ENV:
            os.environ[name] = '1'
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float64)
        report['execution_environment'] = {'thread_environment': {k: os.environ[k] for k in THREAD_ENV},
                                           'thread_limits_set_before_numpy_torch_import': THREAD_LIMITS_SET_BEFORE_IMPORT,
                                           'torch_threads': torch.get_num_threads(),
                                           'torch_interop_threads': torch.get_num_interop_threads(),
                                           'numpy_version': np.__version__, 'torch_version': torch.__version__}
        report['source_sha256_before'] = {k: sha256(p) for k, p in sources.items()}
        for name, path in (('neural', paths['neural_checkpoint']), ('symbolic_torch', paths['symbolic_checkpoint'])):
            persist('load_' + name + '_started')
            check_deadline(deadline)
            begin = time.perf_counter()
            model, encoder, unused_head, obj = load_checkpoint(path, 'cpu')
            del unused_head
            model.double().eval().requires_grad_(False)
            encoder.double().eval().requires_grad_(False)
            if name == 'neural':
                neural, encoder_n, obj_n = model, encoder, obj
            else:
                symbolic, encoder_s, obj_s = model, encoder, obj
            report['cost_seconds']['model_load_' + name] = time.perf_counter() - begin
            persist('load_' + name + '_complete')
        check_deadline(deadline)
        begin = time.perf_counter()
        spec = read_json(paths['symbolic_json'])
        runtime = SymbolicPowerGraph(spec)
        report['cost_seconds']['model_load_symbolic_numpy'] = time.perf_counter() - begin
        check_deadline(deadline)
        begin = time.perf_counter()
        histories, facts = read_panel(read_json(paths['project_config']), obj_n['physics'],
                                      (obj_n.get('event_ids'), obj_s.get('event_ids')), deadline)
        report['data_facts'] = facts
        report['cost_seconds']['train_history_read'] = time.perf_counter() - begin
        sources.update({str(p): Path(p) for p in facts['source_metadata_sha256_before']})
        report['sources'].update({str(p): str(p) for p in facts['source_metadata_sha256_before']})
        report['source_sha256_before'].update(facts['source_metadata_sha256_before'])
        begin = time.perf_counter()
        validate_models(neural, symbolic, runtime, encoder_s, obj_n, obj_s, spec,
                        [entry['id'] for entry in facts['eligible']])
        report['cost_seconds']['endpoint_identity_validation'] = time.perf_counter() - begin
        report['saved_training_scales'] = runtime.scales.tolist()
        report['source_physics'] = obj_n['physics']
        report['dimensions'] = {'n': runtime.n, 'rank': runtime.rank, 'full_state': runtime.dim}
        with (output / 'train_histories.npz').open('xb') as stream:
            np.savez(stream, history=histories, observation_seconds=np.arange(11) * SAMPLE_DT)
        persist('source_and_history_validated')
        check_deadline(deadline)
        neural.reuse_origin_terms = True
        neural.checkpoint_steps = 0
        persist('neural_trace_started')
        begin = time.perf_counter()
        trace_field(neural)
        report['cost_seconds']['neural_trace_build'] = time.perf_counter() - begin
        persist('neural_trace_complete')
        check_deadline(deadline)
        grid, _ = output_grid()
        persist('torch_symbolic_reference_started')
        reference, reference_states, reference_report = deploy(symbolic, histories, backend='torch', encoder=encoder_s, deadline=deadline)
        reference_report['role'] = 'independent numerical reference; never included in timing summary'
        report['symbolic_torch_reference'] = reference_report
        report['cost_seconds']['symbolic_torch_reference'] = reference_report['wall_seconds']
        save_arrays(output / 'symbolic_torch_reference.npz', grid, reference, reference_states)
        persist('torch_symbolic_reference_saved')
        if not reference_report['valid_runtime']:
            raise RuntimeError('Torch symbolic reference incomplete/nonfinite; no valid runtime comparison')
        for batch_name, count in (('single', 1), ('batch', len(histories))):
            for name, model, encoder, backend in (('N', neural, encoder_n, 'torch'), ('S', runtime, None, 'numpy')):
                key = name + '_' + batch_name
                for trial in range(WARMUPS + REPEATS):
                    check_deadline(deadline)
                    phase = 'warmup' if trial < WARMUPS else 'timed'
                    persist(key + '_' + phase + '_' + str(trial) + '_started')
                    values, states, row = deploy(model, histories[:count], backend=backend, encoder=encoder,
                                                 reuse_origins=name == 'N', deadline=deadline)
                    row.update(endpoint=name, batch_mode=batch_name, batch_size=count, phase=phase,
                               trial_index=trial, event_ids=[e['id'] for e in facts['selected'][:count]])
                    row['parity'] = parity(values, reference[:count], runtime.scales) if name == 'S' else None
                    row['full_state_parity'] = (state_parity(states, reference_states[:count], runtime.mass,
                                                           runtime.scales, runtime.rank) if name == 'S' else None)
                    if name == 'S' and (not row['parity']['passed'] or not row['full_state_parity']['valid_runtime']):
                        row['valid_runtime'] = False
                    row['artifact'] = key + '_' + phase + '_' + str(trial) + '.npz'
                    report['trials'].append(row)
                    save_arrays(output / row['artifact'], grid, values, states)
                    persist(key + '_' + phase + '_' + str(trial) + '_saved')
                    if not row['valid_runtime']:
                        raise RuntimeError('Invalid deployment trial: ' + key + '; planned remaining trials not executed')
        report['complete'] = True
        report['stop_reason'] = 'All fixed trials finished; accuracy and safety require frozen evaluation evidence'
    except (Exception, KeyboardInterrupt) as exc:
        failure = exc
        report['stop_reason'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        # Hash only authorized endpoint/code/metadata assets, never entire HDF5.
        report['source_sha256_after'] = {k: sha256(p) if p.is_file() else None for k, p in sources.items()}
        report['sources_unchanged'] = bool(report['source_sha256_before'] and
                                          report['source_sha256_before'] == report['source_sha256_after'])
        if 'data_facts' in report:
            before = report['data_facts']['hdf5_stat_before']
            after = {p: file_stat(p) if Path(p).is_file() else None for p in before}
            report['data_facts']['hdf5_stat_after'] = after
            report['sources_unchanged'] &= before == after
        all_valid = (report['complete'] and report['sources_unchanged']
                     and len(report['trials']) == 4 * (WARMUPS + REPEATS)
                     and all(row['valid_runtime'] for row in report['trials']))
        report['valid_runtime_comparison'] = bool(all_valid)
        observed = {(row['endpoint'], row['batch_mode'], row['trial_index']) for row in report['trials']}
        report['unexecuted_trials'] = [row for row in report['planned_trials']
                                      if (row['endpoint'], row['batch_mode'], row['trial_index']) not in observed]
        if all_valid:
            for name in ('N', 'S'):
                for mode in ('single', 'batch'):
                    rows = [r for r in report['trials'] if r['endpoint'] == name and r['batch_mode'] == mode and r['phase'] == 'timed']
                    seconds = [r['wall_seconds'] for r in rows]
                    report['timing_summary'][name + '_' + mode] = {
                        'seconds': seconds, 'median_seconds': float(np.median(seconds)),
                        'min_seconds': min(seconds), 'max_seconds': max(seconds),
                        'batch_size': rows[0]['batch_size'],
                        'median_seconds_per_event': float(np.median(seconds)) / rows[0]['batch_size']}
            for mode in ('single', 'batch'):
                report['runtime_ratios_N_over_S'][mode] = (
                    report['timing_summary']['N_' + mode]['median_seconds'] /
                    report['timing_summary']['S_' + mode]['median_seconds'])
            report['status'] = 'complete_runtime_only'
        else:
            report['runtime_ratio_withheld_reason'] = 'Incomplete/failed trial, parity, or source-integrity check; raw costs are diagnostic only'
        persist('finished' if all_valid else 'partial_or_invalid_evidence_saved')
    if failure is not None:
        raise failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural-checkpoint', required=True)
    parser.add_argument('--symbolic-checkpoint', required=True)
    parser.add_argument('--symbolic-json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    benchmark(**vars(args))


if __name__ == '__main__':
    main()
