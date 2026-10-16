"""Registered 0100 paired-grid continuation; frozen 0098/0099 remain unchanged."""
from __future__ import annotations
import time
PROCESS_STARTED = time.perf_counter()  # Includes heavy imports in the process cap.
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
from train_local_memory_0098 import (load_model, optimizer_for, read_train, sources,
    NORM, check_structure, graph, histories_for, horizon, process_memory_bytes)
from train_joint_campaign import sample_batch
from train_joint_adequacy_0099 import digest, _json_digest, severe_transition
from monitor_joint_adequacy_0099 import build_manifest, run_monitor, _model_hash

SOURCE = ROOT/'private_data/task2_joint_adequacy_0099/formal_L10-S/last_step1167'
INITIAL_MONITOR = SOURCE.parent/'monitor_step0600'
START_STEP = 1167
EXPECTED = {
    'model.pt': 'b2f822d9f6dce2f27e944df6a3177bdb8229f90adb0358bb18485385946a5edc',
    'optimizer.pt': '5d8554c15ab73233a795ab5db6280d18929bfe704f7e50d7aa2dce1b574c22c5',
    'rng.pt': 'd5ce8d334b4f175426b33ea06fa1847bc459fd4c03e62c519f2fb62d2ce0368b',
    'checkpoint.json': '52635f34146646cc770fe6731b404a82d3a5bd4616463ee1db2acd581eb62762'}
MASK = [1., 1., 1., 1., 1., 1., 0., 1.]
MEMORY_CAP = 8*1024**3


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def parameter_layout(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [{'lr': float(g['lr']), 'parameter_names': [names[id(p)] for p in g['params']]}
            for g in optimizer.param_groups]


def validate_optimizer(model, optimizer, meta):
    required = [n for n, p in model.named_parameters() if p.requires_grad]
    if required != meta['requires_grad_parameters']:
        raise ValueError('requires_grad parameter identity/order mismatch')
    layout = parameter_layout(model, optimizer)
    if layout != meta['optimizer_parameter_names']:
        raise ValueError('Adam parameter names/order/rates mismatch')
    flat = [n for group in layout for n in group['parameter_names']]
    if len(flat) != len(set(flat)) or set(flat) != set(required):
        raise ValueError('Adam must include every allowed parameter exactly once')
    if len(optimizer.state) != len(flat):
        raise ValueError('Adam state count mismatch')
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if int(state['step']) != int(meta['step']):
                raise ValueError('Adam update counter mismatch')
            for key in ('exp_avg', 'exp_avg_sq'):
                if state[key].shape != p.shape or not torch.isfinite(state[key]).all():
                    raise ValueError('invalid Adam moment shape/values')


def validate_coverage(coverage, count):
    if sorted(coverage['cover_order']) != list(range(count)):
        raise ValueError('coverage must be a full TRAIN permutation')
    if not 0 <= coverage['cursor'] <= count:
        raise ValueError('invalid coverage cursor')


def restore(source, count):
    source = Path(source)
    if source.resolve() != SOURCE.resolve():
        raise ValueError('only the registered 0099 L10 step1167 source is allowed')
    if {name: digest(source/name) for name in EXPECTED} != EXPECTED:
        raise ValueError('pinned source hash mismatch')
    meta = json.loads((source/'checkpoint.json').read_text(encoding='utf-8'))
    if meta['step'] != START_STEP or meta['arm'] != 'L10-S':
        raise ValueError('wrong ancestor')
    model = load_model(source)
    if model.dim != 40 or model.template_mask.tolist() != MASK:
        raise ValueError('dimension or frozen support mismatch')
    if _model_hash(model) != meta['model_state_sha256']:
        raise ValueError('restored model state identity mismatch')
    optimizer = optimizer_for(model)
    # Verify ordering BEFORE loading the positional Adam state.
    if parameter_layout(model, optimizer) != meta['optimizer_parameter_names']:
        raise ValueError('pre-load Adam ordering mismatch')
    optimizer.load_state_dict(torch.load(source/'optimizer.pt', map_location='cpu', weights_only=False))
    validate_optimizer(model, optimizer, meta)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('nonfinite source model')
    coverage = copy.deepcopy(meta['coverage'])
    validate_coverage(coverage, count)
    state = torch.load(source/'rng.pt', map_location='cpu', weights_only=False)
    rng = np.random.default_rng()
    rng.bit_generator.state = state['numpy']
    torch.set_rng_state(state['torch'])
    return model, optimizer, rng, coverage, meta


def snapshot(source):
    paths = {'runner0100': Path(__file__), 'grid0100': ROOT/'experiments/joint_training_grid_0100.py',
             'card0100': ROOT/'reports/task2_joint_numerical_pair_0100.md',
             'monitor0099': ROOT/'experiments/monitor_joint_adequacy_0099.py',
             'runner0099': ROOT/'experiments/train_joint_adequacy_0099.py'}
    paths.update({f'ancestor_{name}': Path(source)/name for name in EXPECTED})
    return dict(sources(), **{k: {'path': str(v), 'sha256': digest(v)} for k, v in paths.items()})


def cached_initial_monitor(path, model, manifest):
    """A complete, possibly J-unresolved monitor is reusable with intact identity."""
    path = Path(path)
    if path.is_dir():
        path = path/'evidence.json'
    if not path.exists():
        return None, {'accepted': False, 'reason': 'missing_cache'}
    row = json.loads(path.read_text(encoding='utf-8'))
    state_hash = _model_hash(model)
    checks = {'complete': row.get('complete') is True,
              'model_before': row.get('model_state_sha256_before') == state_hash,
              'model_after': row.get('model_state_sha256_after') == state_hash,
              'manifest': row.get('manifest_sha256') == manifest['sha256'],
              'code': row.get('monitor_code_sha256') == digest(ROOT/'experiments/monitor_joint_adequacy_0099.py'),
              'source_before': row.get('source_before') == sources(),
              'source_after': row.get('source_after') == sources(),
              'source_integrity': row.get('sources_unchanged') is True,
              'rng_integrity': row.get('rng_unchanged') is True,
              'cost': isinstance(row.get('seconds'), (int, float)) and row.get('seconds', 0) > 0}
    # The frozen monitor stores an explicit RNG non-consumption assertion, not
    # an RNG fingerprint: this is sufficient because evaluation is deterministic.
    audit = {'accepted': all(checks.values()), 'checks': checks,
             'path': str(path), 'sha256': digest(path)}
    return (row if audit['accepted'] else None), audit


def next_samples(events, rng, coverage):
    before = copy.deepcopy(coverage)
    rng_before = _json_digest(rng.bit_generator.state)
    chosen, _ = sample_batch(events, rng, 10)
    for _ in range(2):
        if coverage['cursor'] >= len(events):
            coverage['cover_order'] = rng.permutation(len(events)).tolist()
            coverage['cursor'] = 0
        e = events[coverage['cover_order'][coverage['cursor']]]
        coverage['cursor'] += 1
        chosen.append((e, int(rng.integers(10, len(e['x'])-1))))
    return chosen, {'sample_ids': [e['id'] for e, _ in chosen],
                    'endpoints': [int(i) for _, i in chosen],
                    'coverage_before': before, 'coverage_after': copy.deepcopy(coverage),
                    'numpy_rng_before_sha256': rng_before,
                    'numpy_rng_after_sha256': _json_digest(rng.bit_generator.state)}


def reserve_seconds(last_monitor, recent_steps):
    return max(300., 1.5*last_monitor+60.) + 1.5*max(recent_steps, default=0.)


def estimated_cost(new_steps, costs, monitor_seconds):
    # Measured no-update backwards cost, scheduled monitors and a final reserve.
    return (sum(float(costs[str(int(horizon(START_STEP+i)))]) for i in range(1, new_steps+1))
            + (new_steps//100)*1.25*monitor_seconds
            + reserve_seconds(monitor_seconds, list(costs.values())))


def read_cost_evidence(path, dt):
    costs = json.loads(Path(path).read_text(encoding='utf-8'))
    if (costs.get('training_dt') != dt or costs.get('checkpoint_step') != START_STEP
            or costs.get('complete') is not True or costs.get('parameter_updates') is not False
            or costs.get('source_model_sha256') != EXPECTED['model.pt']
            or costs.get('objective_code_sha256') != digest(ROOT/'experiments/joint_training_grid_0100.py')
            or any(not np.isfinite(costs['step_seconds'][str(h)]) or costs['step_seconds'][str(h)] <= 0 for h in (1,2,5))):
        raise ValueError('invalid identity or grid-specific no-update QA costs')
    return costs


def save_checkpoint(output, name, model, optimizer, rng, coverage, step, dt,
                    manifest, source_snapshot, started, *, valid=True, attempted_step=None):
    directory = Path(output)/name
    directory.mkdir(exist_ok=False)
    torch.save(model.state_dict(), directory/'model.pt')
    torch.save(optimizer.state_dict(), directory/'optimizer.pt')
    torch.save({'numpy': rng.bit_generator.state, 'torch': torch.get_rng_state(),
                'numpy_global': np.random.get_state(), 'coverage': coverage}, directory/'rng.pt')
    meta = {'format': 'joint_numerical_pair_0100', 'arm': 'L10-dt10' if dt == .01 else 'L10-dt5',
            'step': step if valid else None, 'last_verified_step': step, 'attempted_step': attempted_step,
            'valid_training_state': valid, 'ancestor_step': START_STEP,
            'new_steps': step-START_STEP if valid else None, 'training_dt': dt,
            'physical_scoring_dt': .01, 'local_horizon': .2, 'latent_horizon': .01,
            'ancestor_rng_scope': 'torch CPU and NumPy Generator; global NumPy was not saved in ancestor',
            'numpy_global_rng_scope': 'captured at this 0100 checkpoint, not claimed restored from 0099',
            'dimension': model.dim, 'seed': 98, 'config': model.config(),
            'coverage': coverage, 'model_state_sha256': _model_hash(model),
            'monitor_manifest_sha256': _json_digest(manifest),
            'requires_grad_parameters': [n for n,p in model.named_parameters() if p.requires_grad],
            'requires_grad': {n: p.requires_grad for n,p in model.named_parameters()},
            'parameter_numel': {n: p.numel() for n,p in model.named_parameters()},
            'optimizer_parameter_names': parameter_layout(model, optimizer),
            'active_templates': model.template_mask.tolist(), 'source_snapshot': source_snapshot,
            'process_seconds_at_checkpoint': time.perf_counter()-started}
    write_json(directory/'checkpoint.json', meta)
    if valid and hasattr(model, 'export'):
        model.export(directory/'structured_numeric_export.json')
    write_json(directory/'artifact_hashes.json', {n: digest(directory/n) for n in EXPECTED})
    return str(directory)


def run(dt, source, output, new_steps=200, initial_monitor=INITIAL_MONITOR,
        cap_seconds=3500., qa_only=False, cost_evidence=None, process_start=None):
    if dt not in (.01, .005) or new_steps not in (100, 200) or not 0 < cap_seconds <= 3500:
        raise ValueError('unregistered grid, steps or cap')
    output, source = Path(output), Path(source)
    if output.exists():
        raise FileExistsError(output)
    started = PROCESS_STARTED if process_start is None else process_start
    deadline = started+cap_seconds
    torch.set_num_threads(1)
    def guard():
        if time.perf_counter() >= deadline-15:
            raise TimeoutError('serialization reserve reached')
        if process_memory_bytes() > MEMORY_CAP:
            raise MemoryError('8 GiB process cap')
    guard()
    before = snapshot(source)
    events, excluded = read_train()
    if len(events) != 346 or len(excluded) != 23 or sum(e['label']=='unstable' for e in events) != 35:
        raise ValueError('registered TRAIN pool changed')
    model, optimizer, rng, coverage, meta = restore(source, len(events))
    manifest = build_manifest(events)
    if meta['monitor_manifest_sha256'] != _json_digest(manifest):
        raise ValueError('source monitor TRAIN manifest differs')
    force_scale = float(np.load(NORM, allow_pickle=False)['force_scale'])
    initial, cache_audit = cached_initial_monitor(initial_monitor, model, manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json', manifest)
    monitors, traces = {}, []
    completed, attempted = START_STEP, None
    last_monitor = float(initial['seconds']) if initial else 0.
    stop, failure, last_path, estimates = None, None, None, None
    milestone = save_checkpoint(output, 'step0000', model, optimizer, rng, coverage,
                                completed, dt, manifest, before, started)
    def monitor_at(n):
        nonlocal last_monitor
        guard()
        if n == 0 and initial is not None:
            row = initial
        else:
            if time.perf_counter()+max(300., 1.5*last_monitor+60.) >= deadline:
                raise TimeoutError('insufficient full-monitor reserve')
            ts, ns, rs = torch.get_rng_state().clone(), np.random.get_state(), copy.deepcopy(rng.bit_generator.state)
            try:
                row = run_monitor(model, events, manifest, output/f'monitor_step{n:04d}',
                                  deadline-15, force_scale=force_scale, allow_refine=True)
                if not torch.equal(ts, torch.get_rng_state()) or _json_digest(rs) != _json_digest(rng.bit_generator.state):
                    raise RuntimeError('monitor consumed training RNG')
            finally:
                torch.set_rng_state(ts); np.random.set_state(ns); rng.bit_generator.state = rs
        monitors[str(n)] = row  # Do not relabel frozen component evidence.
        if row.get('complete'):
            last_monitor = float(row['seconds'])
        if before != snapshot(source):
            raise RuntimeError('source/code drift')
        return row
    try:
        costs = read_cost_evidence(cost_evidence, dt) if cost_evidence is not None else None
        if costs is not None:
            write_json(output/'cost_evidence.json', {'path': str(cost_evidence), 'sha256': digest(cost_evidence), 'evidence': costs})
            if initial is not None:
                estimates = {str(n): {'remaining_seconds': estimated_cost(n, costs['step_seconds'], last_monitor),
                                      'fits_remaining_cap': time.perf_counter()+estimated_cost(n, costs['step_seconds'], last_monitor)<deadline}
                             for n in (100,200)}
        if qa_only:
            stop = 'qa_restore_cache_complete' if initial is not None else 'qa_cache_not_reusable'
        else:
            if cost_evidence is None:
                raise ValueError('formal fit requires no-update 1/2/5-second cost evidence')
            row = monitor_at(0)
            baseline = row.get('raw_S_fine') or {}
            if not row.get('complete') or any(baseline.get(h) is None or not np.isfinite(baseline[h]) for h in ('1','2','5')):
                raise RuntimeError('complete finite initial severe monitor required')
            estimate = estimated_cost(new_steps, costs['step_seconds'], last_monitor)
            estimates = {str(n): {'remaining_seconds': estimated_cost(n, costs['step_seconds'], last_monitor),
                                  'fits_remaining_cap': time.perf_counter()+estimated_cost(n, costs['step_seconds'], last_monitor)<deadline}
                         for n in (100,200)}
            if time.perf_counter()+estimate >= deadline:
                raise TimeoutError('registered nominal step plan does not fit measured budget')
            from joint_training_grid_0100 import objective_at_grid
            check_structure(model)
            model.train()
            counters = {'1': 0, '2': 0, '5': 0}
            for step in range(START_STEP+1, START_STEP+new_steps+1):
                guard()
                recent = [r['seconds'] for r in traces[-10:]] or list(costs['step_seconds'].values())
                if time.perf_counter()+reserve_seconds(last_monitor, recent) >= deadline:
                    stop = 'wall_cap_reserve'; break
                if before != snapshot(source):
                    raise RuntimeError('source/code drift before update')
                attempted = step
                tic = time.perf_counter()
                samples, trace = next_samples(events, rng, coverage)
                trace.update(step=step, new_step=step-START_STEP, horizon=horizon(step), training_dt=dt)
                # Save sampled identity before an update that may fail.
                with (output/'samples.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(trace, allow_nan=False)+'\n')
                optimizer.zero_grad(set_to_none=True)
                with graph(model):
                    loss, terms = objective_at_grid(model, samples, histories_for(samples), horizon(step), force_scale, dt)
                    if not torch.isfinite(loss) or any(not torch.isfinite(v).all() for v in terms.values()):
                        raise FloatingPointError('nonfinite objective')
                    loss.backward()
                params = [p for p in model.parameters() if p.requires_grad]
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError('missing/nonfinite joint gradient')
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 100.))
                if not np.isfinite(grad_norm):
                    raise FloatingPointError('nonfinite gradient norm')
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('nonfinite post-update parameters')
                if model.template_mask.tolist() != MASK:
                    raise RuntimeError('frozen support drift')
                validate_optimizer(model, optimizer, dict(meta, step=step))
                check_structure(model)
                completed = step  # Only after finite and structure guards.
                trace.update(loss=float(loss.detach()), terms={k:float(v.detach()) for k,v in terms.items()},
                             gradient_norm=grad_norm, gradient_clipped=grad_norm>100.,
                             seconds=time.perf_counter()-tic, elapsed_seconds=time.perf_counter()-started)
                traces.append(trace)
                with (output/'trace.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(trace, allow_nan=False)+'\n')
                guard()
                if step-START_STEP in (100,200):
                    milestone = save_checkpoint(output, f'step{step:04d}', model, optimizer, rng,
                        coverage, completed, dt, manifest, before, started)
                    row = monitor_at(step-START_STEP)
                    if not row.get('complete'):
                        stop = 'partial_monitor'; break
                    stop = severe_transition(row, baseline, counters)
                    if stop:
                        break
            stop = stop or 'nominal_step_cap'
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc), 'attempted_step': attempted}
        stop = 'failed_state_preserved'
        save_checkpoint(output, 'failed_state', model, optimizer, rng, coverage, completed,
                        dt, manifest, before, started, valid=False, attempted_step=attempted)
    finally:
        if failure is None:
            last_path = save_checkpoint(output, f'last_step{completed:04d}', model, optimizer,
                        rng, coverage, completed, dt, manifest, before, started)
        after = snapshot(source)
        evidence = {'loop': '0100', 'training_dt': dt, 'ancestor_step': START_STEP,
                    'requested_new_steps': new_steps, 'verified_completed_updates': completed-START_STEP,
                    'final_step': completed if failure is None else None,
                    'status': stop, 'failure': failure, 'qa_only': qa_only, 'monitors': monitors,
                    'last_checkpoint': last_path, 'last_valid_milestone': milestone,
                    'cache_audit': cache_audit, 'source_before': before, 'source_after': after,
                    'nominal_cost_estimates': estimates,
                    'sources_unchanged': before == after, 'phase_cap_seconds': cap_seconds,
                    'seconds': time.perf_counter()-started, 'within_cap': time.perf_counter()<=deadline,
                    'test_read': False, 'validation_read': False, 'checkpoint_selection': None}
        if before != after:
            evidence['status'] = 'quarantined_source_drift'
        if not evidence['within_cap']:
            evidence['status'] = 'budget_exception'
        write_json(output/'evidence.json', evidence)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dt', type=float, choices=(.01,.005), required=True)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--new-steps', type=int, choices=(100,200), default=200)
    p.add_argument('--initial-monitor', type=Path, default=INITIAL_MONITOR)
    p.add_argument('--cap-seconds', type=float, default=3500.)
    p.add_argument('--cost-evidence', type=Path)
    p.add_argument('--qa-only', action='store_true', help='restore/cache checks only; no monitor integration')
    result = run(**vars(p.parse_args()), process_start=PROCESS_STARTED)
    print(json.dumps({'status': result['status'], 'final_step': result['final_step']}), flush=True)
    if result.get('failure') or result['status'] in ('budget_exception','quarantined_source_drift'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
