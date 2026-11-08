"""One bounded, train-only MAIN8/edge-deletion engineering audit; no fitted candidate."""
from __future__ import annotations
import time
START = time.perf_counter()  # before imports, including supervisor/child wall once
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'src')]
COHORT_SHA = '42228e81d9f3d15c94350b14ae138a39285f19448db4c2dbc5fd376c0e513685'
MAIN = ROOT/'results/closure_discovery_20261017/E/packages/main_2701/model.json'
TOLERANCES = {'field': [2e-11, 1e-11], 'loss': [2e-10, 2e-10],
              'gradient': [2e-9, 2e-9], 'adam': [2e-9, 2e-9], 'export': [1e-11, 0.]}


def terminate_owned_tree(process, runner=None, platform=None):
    """Terminate only this supervisor's Popen subtree, including Windows venv runtimes."""
    runner = subprocess.run if runner is None else runner
    platform = os.name if platform is None else platform
    if platform == 'nt':
        command = ['taskkill', '/PID', str(int(process.pid)), '/T', '/F']
        result = runner(command, capture_output=True, text=True, timeout=2, check=False)
        cleanup = {'command': command, 'returncode': result.returncode,
                   'stdout': result.stdout, 'stderr': result.stderr}
        # A failed tree termination is retained as an unresolved cleanup failure.
        if result.returncode != 0:
            cleanup['failure'] = 'owned subtree termination was not confirmed'
    else:
        process.kill()
        cleanup = {'command': ['Popen.kill', str(int(process.pid))], 'returncode': 0}
    try:
        cleanup['launcher_returncode'] = process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        cleanup['failure'] = 'launcher remains alive after owned subtree termination'
    return cleanup


def wait_owned(process, timeout, cleanup):
    """Every interrupted wait uses the same owned-tree cleanup, preserving the cause."""
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        try:
            cleanup.update(terminate_owned_tree(process))
        except BaseException as exc:
            cleanup['failure'] = repr(exc)
        raise


def supervise(args):
    """A terminating process deadline covers imports, loading and all child work."""
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), args.mode, '--worker',
               '--output', str(args.output), '--max-seconds', str(args.max_seconds)]
    status, error, child_pid = 'FAILED', None, None
    cleanup = {}
    try:
        with (args.output/'stdout.log').open('w') as out, (args.output/'stderr.log').open('w') as err:
            process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
            child_pid = process.pid
            try:
                code = wait_owned(process, max(.1, args.max_seconds-(time.perf_counter()-START)-3), cleanup)
            except subprocess.TimeoutExpired:
                raise TimeoutError('single cumulative process cap; child terminated')
        if code != 0:
            raise RuntimeError(f'worker exit {code}')
        status = 'COMPLETED'
    except BaseException as exc:
        error = repr(exc)
    finally:
        worker = None
        for metadata in ('progress.json', 'worker.json'):
            path = args.output/metadata
            if path.exists():
                try:
                    worker = json.loads(path.read_text(encoding='utf-8'))['pid']
                    break
                except (OSError, ValueError, KeyError):
                    pass
        receipt = {'status': status, 'error': error, 'pid': os.getpid(), 'child_pid': child_pid,
                   'launcher_pid': child_pid, 'actual_worker_pid': worker, 'cleanup': cleanup,
                   'command': command, 'mode': args.mode, 'seconds': time.perf_counter()-START,
                   'cap_seconds': args.max_seconds, 'includes_child_cost_once': True,
                   'category': 'SUPPORT' if args.mode == 'qa' else 'EXEC',
                   'val_read': False, 'test_read': False}
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    if status != 'COMPLETED':
        raise RuntimeError(error)


def load_runtime():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    global np, torch
    import numpy as np
    import torch
    torch.set_num_threads(1)


def structures(device):
    from closure_mechanism_study import source_model
    from symbolic_closure_rom import term_key, _BASE_TERMS
    # Read only the historical expression list; never adopt its fitted weights.
    public = json.loads(MAIN.read_text(encoding='utf-8'))
    terms = public['terms']
    terms_sha = hashlib.sha256(json.dumps(terms, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if terms_sha != '3081b3eaeb3b96dd3b938d276e5dc64c83e4b023fd988fa940a817f74489973f':
        raise ValueError('historical MAIN8 terms hash changed')
    expected = [_BASE_TERMS[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)]
    if [term_key(t) for t in terms] != [term_key(t) for t in expected]:
        raise ValueError('historical MAIN8 expression identity/order changed')
    full = source_model(device).with_terms(terms)
    deleted_key = term_key(_BASE_TERMS[2])
    deleted = full.with_terms([t for t in full.terms if term_key(t) != deleted_key])
    assert len(full.terms) == 8 and len(deleted.terms) == 7
    return {'full8': full, 'delete_edge_cosine': deleted}


def dependency_hashes():
    """Bind the local Python import closure, including imports inside functions."""
    from train_closure import sha
    pending = [Path(__file__).resolve(), ROOT/'tests/test_edge_pair_cost_gate_20261019.py']
    seen = {}
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen[path] = sha(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            modules = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                rel = Path(*module.split('.'))
                for base in (ROOT/'experiments', ROOT/'src'):
                    for candidate in (base/rel.with_suffix('.py'), base/rel/'__init__.py'):
                        if candidate.exists():
                            pending.append(candidate.resolve())
    return {str(p.relative_to(ROOT)).replace('\\', '/'): h for p, h in sorted(seen.items())}


def freeze(events, force, cohort, output):
    from train_closure import sha, write, RATES, horizon_for_update
    from train_joint_numerical_pair_0100 import next_samples
    from train_local_memory_0098 import sources
    from closure_mechanism_study import SOURCE, SOURCE_SHA
    if cohort['sha256'] != COHORT_SHA:
        raise ValueError('366 train identity changed')
    panels = {'distributed': np.linspace(0, len(events)-1, 12, dtype=int).tolist(),
              'longest': sorted(range(len(events)), key=lambda i: (-len(events[i]['x']), events[i]['id']))[:12],
              'unstable': sorted((i for i, e in enumerate(events) if e['label'] == 'unstable'),
                                 key=lambda i: (-len(events[i]['x']), events[i]['id']))[:12]}
    schedules = {}
    for seed in (6101, 6102, 6103):
        rng = np.random.default_rng(seed)
        coverage = {'cover_order': rng.permutation(len(events)).tolist(), 'cursor': 0}
        rows = []
        for step in range(1, 1025):
            samples, sampling = next_samples(events, rng, coverage)
            rows.append({'step': step, 'horizon_s': horizon_for_update(step),
                         'lr_multiplier': .3 if step > 512 else 1., 'sampling': sampling,
                         'actual_long_counts': [min(round(horizon_for_update(step)*100), len(e['x'])-11) for e, _ in samples]})
        included = sorted({eid for r in rows for eid in r['sampling']['sample_ids']})
        assert included == sorted(e['id'] for e in events)
        schedules[str(seed)] = {'rows': rows, 'included_ids': included, 'eligible_pool_count': len(events)}
    write(output/'schedule.json', schedules)
    metadata = [{'id': e['id'], 'group': e['group'], 'label': e['label'],
                 'length': len(e['x']), 'last_true_time_s': (len(e['x'])-1)*.01,
                 'future_steps_from_10': len(e['x'])-11,
                 'x_sha256': hashlib.sha256(np.ascontiguousarray(e['x']).tobytes()).hexdigest(),
                 'torque_sha256': hashlib.sha256(np.ascontiguousarray(e['torque']).tobytes()).hexdigest()} for e in events]
    identity = {'source_sha256': SOURCE_SHA, 'source_path': str(SOURCE), 'cohort': cohort,
                'events': metadata, 'old_sources': sources(), 'code': dependency_hashes(),
                'terms_only_reference_sha256': sha(MAIN), 'schedule_sha256': sha(output/'schedule.json'),
                'panels': {k: [events[i]['id'] for i in v] for k, v in panels.items()},
                'panel_endpoints': 10, 'force_scale': force, 'tolerances': TOLERANCES,
                'checkpoint_every': 50, 'milestones': [256, 512, 1024],
                'milestone_scope': 'full366 dual-step numeric/metrics plus local derivatives',
                'rates': RATES, 'final_decay': True, 'val_read': False, 'test_read': False}
    models = structures('cpu')
    identity['structures'] = {k: {'terms': m.terms, 'coefficients': m.coefficients.detach().tolist(),
                                  'template_mask': m.template_mask.tolist(),
                                  'input_scale': m.input_scale.tolist(), 'p_scale': m.p_scale.tolist(),
                                  'parameters': [(n, list(p.shape), p.requires_grad) for n,p in m.named_parameters()]}
                              for k,m in models.items()}
    write(output/'frozen_identity.json', identity)
    return identity, panels, schedules


def update(model, optimizer, store, samples, horizon, force):
    from closure_training_runtime import objective, all_finite, synchronize
    from train_local_memory_0098 import graph
    synchronize(model.mass.device); tick = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with graph(model):
        value, _ = objective(model, store, samples, horizon, force)
        if not all_finite([value]):
            raise FloatingPointError('nonfinite loss')
        value.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not all_finite([p.grad for p in parameters]):
        raise FloatingPointError('nonfinite gradient')
    torch.nn.utils.clip_grad_norm_(parameters, 100.)
    optimizer.step(); synchronize(model.mass.device)
    if not all_finite(parameters):
        raise FloatingPointError('nonfinite updated parameters')
    return time.perf_counter()-tick


def full_evaluation(model, events, output, check):
    from summarize_train_local_memory_0098 import event_metrics, build_summary
    from train_local_memory_0098 import graph
    from train_closure import write
    from evaluate_structured_rom import _safe
    records = {k: [] for k in ('records', 'common_frequency', 'normalized_p', 'objective_point_error', 'numerical')}
    # Existing metrics intentionally run on CPU; evaluate the exact exported source model there.
    model = copy.deepcopy(model).cpu()
    begin = time.perf_counter()
    output.mkdir()
    for first in range(0, len(events), 32):
        check(f'evaluation block {first}')
        block = events[first:first+32]
        histories = torch.as_tensor(np.stack([e['x'][:11] for e in block]), dtype=torch.float64)
        with torch.no_grad(), graph(model):
            fine = model.rollout(histories, 1600, .005).numpy()
            coarse = model.rollout(histories, 800, .01).numpy()
        for e, f, c in zip(block, fine, coarse):
            for k, rows in event_metrics(model, e, f, c).items():
                records[k].extend(rows)
        write(output/'partial_rows.json', _safe(records))
        check(f'evaluation metrics block {first}')
    write(output/'evidence.json', _safe({'rows': records, 'summary': build_summary(records),
          'seconds': time.perf_counter()-begin, 'event_count': len(events),
          'numerical_rule': 'original event_metrics/_scored_numerics .005/.01 finite all substeps; COI/common <=.001'}))
    return time.perf_counter()-begin


def measure(args):
    from closure_mechanism_study import train_cohort
    from closure_training_runtime import TensorEventStore, resolve_device, synchronize
    from train_closure import write, sha, make_optimizer, save_checkpoint, restore_checkpoint, environment
    from analyze_symbolic_closure import analyze_model
    device = resolve_device('cuda')
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'phase': 'load', 'rows': [], 'overhead': {},
             'val_read': False, 'test_read': False, 'environment': environment(device)}
    def check(phase):
        state.update(phase=phase, elapsed_worker_seconds=time.perf_counter()-START)
        write(args.output/'progress.json', state)
        if time.perf_counter()-START > args.max_seconds-15:
            raise TimeoutError('single cumulative deadline reserve')
    try:
        events, force, cohort = train_cohort()
        identity, panels, schedules = freeze(events, force, cohort, args.output)
        state['frozen_identity_sha256'] = sha(args.output/'frozen_identity.json')
        check('frozen before model results')
        store = TensorEventStore(events, device)
        for name in ('full8', 'delete_edge_cosine'):
            for horizon in (1., 2., 5.):
                for panel, indices in panels.items():
                    check(f'updates {name}/{horizon}/{panel}')
                    model = structures(device)[name]; opt = make_optimizer(model)
                    samples = [(events[i], 10) for i in indices]
                    row = {'structure': name, 'horizon': horizon, 'panel': panel, 'seconds': []}
                    state['rows'].append(row)
                    for repeat in range(2):
                        check(f'update {name}/{horizon}/{panel}/{repeat}')
                        row['seconds'].append(update(model, opt, store, samples, horizon, force))
                        check('completed discarded update')
                    if horizon == 5. and panel == 'unstable':
                        # These remain explicitly discarded probe weights, never final-eval sources.
                        check(f'populated checkpoint {name}')
                        probe = args.output/f'{name}_discarded_probe'; probe.mkdir()
                        rng = np.random.default_rng(6101)
                        trace = schedules['6101']['rows']
                        tick = time.perf_counter()
                        save_checkpoint(probe, model, opt, rng, {'cost_probe_only': True}, 2, identity, trace)
                        synchronize(device)
                        row['checkpoint_seconds'] = time.perf_counter()-tick
                        tick = time.perf_counter()
                        restore_checkpoint(probe, device, identity)
                        synchronize(device)
                        row['restore_seconds'] = time.perf_counter()-tick
        for name in ('full8', 'delete_edge_cosine'):
            check(f'fresh source final {name}')
            tick = time.perf_counter(); model = structures(device)[name]; opt = make_optimizer(model)
            overhead = {'load_seconds': time.perf_counter()-tick}
            state['overhead'][name] = overhead
            folder = args.output/name; folder.mkdir()
            rng = np.random.default_rng(6101)
            tick = time.perf_counter()
            save_checkpoint(folder, model, opt, rng, {'cursor': 0}, 0, identity, [])
            synchronize(device); overhead['checkpoint_seconds'] = time.perf_counter()-tick
            tick = time.perf_counter()
            restored = restore_checkpoint(folder, device, identity)[0]
            synchronize(device); overhead['restore_seconds'] = time.perf_counter()-tick
            check(f'local derivative {name}')
            tick = time.perf_counter()
            local, arrays = analyze_model(copy.deepcopy(restored).cpu())
            np.savez_compressed(folder/'local_derivatives.npz', **arrays)
            write(folder/'local_derivatives.json', local)
            overhead['jacobian_seconds'] = time.perf_counter()-tick
            check(f'full evaluation {name}')
            overhead['full_evaluation_seconds'] = full_evaluation(model, events, folder/'full_train', check)
        if identity['code'] != dependency_hashes():
            raise RuntimeError('source code changed during measurement')
        # max over every panel and first/repeat, then actual formal horizon counts.
        maxima = {name: {str(int(h)): max(t for r in state['rows'] if r['structure']==name and r['horizon']==h
                                           for t in r['seconds']) for h in (1.,2.,5.)}
                  for name in state['overhead']}
        estimates = []
        saves = sorted(set([0, 1024, 256, 512] + list(range(50, 1025, 50))))
        for name, o in state['overhead'].items():
            for seed, schedule in schedules.items():
                updates = sum(maxima[name][str(int(r['horizon_s']))] for r in schedule['rows'])
                probe = next(r for r in state['rows'] if r['structure']==name and 'checkpoint_seconds' in r)
                estimates.append({'structure': name, 'seed': int(seed), 'updates_seconds': updates,
                                  'checkpoint_seconds': len(saves)*max(o['checkpoint_seconds'],probe['checkpoint_seconds']),
                                  'milestones_seconds': 3*(o['full_evaluation_seconds']+o['jacobian_seconds']),
                                  'load_restore_seconds': o['load_seconds']+max(o['restore_seconds'],probe['restore_seconds'])})
        measured = sum(sum(v for k,v in row.items() if k.endswith('_seconds')) for row in estimates)
        startup_exit = time.perf_counter()-START
        total = (measured + startup_exit)*1.25
        state.update(status='COMPLETED', horizon_maxima_seconds=maxima, estimates=estimates,
                     raw_plan_seconds=measured, startup_exit_allowance_seconds=startup_exit,
                     estimated_with_25pct_margin_seconds=total, p2_cap_seconds=86400,
                     cost_gate='WITHIN_COST_CAP' if total <= 86400 else 'EXCEEDS_COST_CAP',
                     checkpoint_steps=saves, scientific_admission='root decision required; no P2 authorization')
    except BaseException as exc:
        state.update(status='UNRESOLVED', error=repr(exc), traceback=traceback.format_exc(), cost_gate='UNRESOLVED')
        raise
    finally:
        state['elapsed_worker_seconds'] = time.perf_counter()-START
        write(args.output/'measurement.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('qa', 'measure'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not (0 < args.max_seconds <= (600 if args.mode == 'qa' else 2800)):
        p.error('cap exceeds preregistered category')
    if not args.worker:
        return supervise(args)
    (args.output/'worker.json').write_text(json.dumps({'pid': os.getpid(), 'parent_pid': os.getppid(),
           'command': sys.argv, 'output': str(args.output.resolve())}), encoding='utf-8')
    load_runtime()
    if args.mode == 'measure':
        measure(args)
    else:
        sys.path.insert(0, str(ROOT/'tests'))
        from test_edge_pair_cost_gate_20261019 import run_qa
        run_qa(args.output)


if __name__ == '__main__':
    main()
