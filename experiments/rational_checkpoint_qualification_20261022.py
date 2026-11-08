"""One-shot, train-only CUDA checkpoint qualification for the registered repair.

This module has no command-line entry point. The source-bound native worker calls
qualify_cuda once and keeps the returned migrated models alive for continuation.
"""
from __future__ import annotations

import copy
import hashlib
import math
import traceback
from pathlib import Path


def _reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def _differences(left, right, limit=32):
    """Describe exact mismatches without accepting a numerical tolerance."""
    import numpy as np
    import torch

    result = []

    def visit(a, b, path):
        if len(result) >= limit:
            return
        if torch.is_tensor(a) or torch.is_tensor(b):
            if not (torch.is_tensor(a) and torch.is_tensor(b)):
                result.append({'path': path, 'reason': 'tensor/type mismatch'})
                return
            if a.dtype != b.dtype or a.shape != b.shape:
                result.append({'path': path, 'reason': 'tensor metadata differs',
                               'left': [str(a.dtype), list(a.shape)],
                               'right': [str(b.dtype), list(b.shape)]})
            elif not torch.equal(a, b):
                row = {'path': path, 'reason': 'tensor values differ',
                       'unequal_elements': int((a != b).sum().item())}
                if a.numel() and (a.is_floating_point() or a.is_complex()):
                    delta = (a - b).abs()
                    maximum = float(delta.max().item())
                    row['max_absolute_difference'] = maximum if math.isfinite(maximum) else None
                result.append(row)
            return
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                result.append({'path': path, 'reason': 'array/type mismatch'})
            elif a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                result.append({'path': path, 'reason': 'array metadata or values differ'})
            return
        if type(a) is not type(b):
            result.append({'path': path, 'reason': 'type differs',
                           'left_type': type(a).__name__, 'right_type': type(b).__name__})
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                result.append({'path': path, 'reason': 'mapping keys differ',
                               'left_keys': list(map(str, a)), 'right_keys': list(map(str, b))})
            for key in a:
                if key in b:
                    visit(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                result.append({'path': path, 'reason': 'sequence lengths differ',
                               'left': len(a), 'right': len(b)})
            for i, (av, bv) in enumerate(zip(a, b)):
                visit(av, bv, path + '/' + str(i))
        elif a != b:
            result.append({'path': path, 'reason': 'scalar values differ',
                           'left': repr(a), 'right': repr(b)})

    visit(left, right, 'state')
    return result


def qualify_cuda(manifest, base, commit):
    """Run exactly four 33-update paths, then migrate all three old step-32 states.

    Any error closes this batch with a durable failure report and propagates to
    the native supervisor. Existing qualification output is never overwritten.
    No val data, teacher generation, historical torch.load or retry occurs here;
    each historical load belongs to the single engine.migrate call for its row.
    """
    import numpy as np
    import torch
    from experiments import rational_checkpoint_recovery_20261022 as engine
    study = engine.study

    base = Path(base).resolve()
    out = base / 'qualification'
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'source_commit': commit,
              'actual_updates': 0,
              'qualification_updates_completed': 0, 'qualification_updates_attempted': 0,
              'registered_updates': 132, 'paths': [], 'comparisons': [], 'migrations': [],
              'seed': 6301, 'noise': 0, 'train_events': 16,
              'process_quotas': None, 'original_val_test_closed': True,
              'snapshots': [], 'manifest_rows': [], 'current_update': None}

    def write_report():
        study.atomic(out / 'result.json', report)

    def snapshot(name, state):
        path = out / (name + '.pt')
        if path.exists():
            raise ValueError('qualification snapshot already exists')
        torch.save(state, path)
        ref = _reference(path)
        report['snapshots'].append(ref)
        return ref

    def compare(name, left, right):
        equal = engine.state_equal(left, right)
        row = {'name': name, 'exact_equal': bool(equal),
               'components': {key: bool(engine.state_equal(left[key], right[key]))
                              for key in left if key in right}}
        if not equal:
            row['differences'] = _differences(left, right)
        report['comparisons'].append(row)
        write_report()
        if not equal:
            raise ValueError('exact CUDA state comparison failed: ' + name)

    try:
        if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('actual source commit required')
        rows = manifest['rows']
        if ([r['key'] for r in rows] != ['0_rational_6301', '0_rational_6302', '0_rational_6303']
                or [r['seed'] for r in rows] != [6301, 6302, 6303]):
            raise ValueError('fixed three source rows required in registered order')
        # Full hash/JSON/data checks precede any qualification update. The engine
        # does not deserialize old training.pt in validate_original.
        checked = [engine.validate_original(row) for row in rows]
        tr = checked[0]['tr']
        report['manifest_rows'] = copy.deepcopy(rows)
        if tr['observed'].shape != (16, 61, 30) or tr['observed'].dtype != np.float64:
            raise ValueError('original float64 16-event train observations required')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA qualification requires actual CUDA')
        # The original training worker set only this option. All deterministic
        # and backend flags are observed, never changed to rescue equality.
        torch.set_num_threads(1)
        report['configuration'] = {
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'num_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'deadline': None, 'dtype': 'torch.float64', 'batch_size': 12,
            'fit_rng_seed_sequence': [6301, 7]}
        obs = torch.as_tensor(tr['observed'], device='cuda')
        scale = torch.as_tensor(tr['scales'], device='cuda')
        rational32 = None
        for form in ('rational', 'tanh'):
            continuous = {}
            for mode in ('continuous', 'restored'):
                name = form + '_' + mode
                folder = out / name
                folder.mkdir()
                path_result = {'name': name, 'form': form, 'mode': mode,
                               'completed_updates': 0, 'attempted_updates': 0, 'status': 'RUNNING'}
                report['paths'].append(path_result)
                m, opt = study.make(6301, tr['observed'], form)
                if m.mass.dtype != torch.float64 or not m.mass.is_cuda:
                    raise ValueError('original float64 CUDA training model required')
                rng = np.random.default_rng(np.random.SeedSequence([6301, 7]))
                trace = []
                gradient = {'total_raw_norm': [], 'clip_factor': []}
                summary = {'first': {}, 'last': {}, 'max': {}}
                identity = {'source_commit': commit, 'qualification_only': True,
                            'key': name, 'seed': 6301, 'noise': 0, 'form': form,
                            'train_raw': _reference(Path(rows[0]['folder']) / 'train.npz')}
                initial = engine.capture_state(m, opt, rng, trace, gradient, summary)
                if mode == 'continuous':
                    continuous[0] = initial
                else:
                    compare(form + '_initial', continuous[0], initial)
                for step in range(1, 34):
                    report['current_update'] = {'path': name, 'step': step}
                    path_result['attempted_updates'] += 1
                    report['qualification_updates_attempted'] += 1
                    study.advance(m, opt, rng, obs, scale, None, trace, gradient, summary)
                    torch.cuda.synchronize()
                    path_result['completed_updates'] += 1
                    report['qualification_updates_completed'] += 1
                    report['actual_updates'] += 1
                    report['current_update'] = None
                    if step not in (32, 33):
                        continue
                    state = engine.capture_state(m, opt, rng, trace, gradient, summary)
                    snapshot(name + '_' + str(step), state)
                    if mode == 'continuous':
                        continuous[step] = state
                        if form == 'rational' and step == 32:
                            rational32 = state
                    elif step == 32:
                        compare(form + '_independent_32', continuous[32], state)
                        engine.publish_checkpoint(folder, m, opt, rng, 32, trace,
                                                  gradient, summary, identity)
                        compare(form + '_publication_preserves_32', state,
                                engine.capture_state(m, opt, rng, trace, gradient, summary))
                        m, opt = study.make(6301, tr['observed'], form)
                        rng = np.random.default_rng(np.random.SeedSequence([6301, 7]))
                        actual, trace, gradient, summary = engine.restore_complete(folder, m, opt, rng, identity)
                        if actual != 32:
                            raise ValueError('actual shared restore did not recover step 32')
                        restored = engine.capture_state(m, opt, rng, trace, gradient, summary)
                        snapshot(name + '_32_loaded', restored)
                        compare(form + '_actual_restore_32', state, restored)
                        path_result['actual_shared_restore_step'] = actual
                        path_result['checkpoint_files'] = [
                            _reference(p) for p in sorted(folder.rglob('*')) if p.is_file()]
                    else:
                        compare(form + '_continuation_33', continuous[33], state)
                path_result['status'] = 'PASS'
                write_report()
                del m, opt
        if report['qualification_updates_completed'] != 132:
            raise ValueError('fixed CUDA qualification update count differs')
        initial_states = {}
        for row in rows:
            key = row['key']
            state = engine.migrate(row, base / 'migration_read' / key,
                                   base / 'science' / 'run_0_rational' / key, commit)
            initial_states[key] = state
            engine.apply_global_rng(state['global_rng'])
            actual = engine.capture_state(state['m'], state['opt'], state['rng'],
                                          state['trace'], state['gradient'], state['summary'])
            snapshot(key + '_migrated_32', actual)
            if row['seed'] == 6301:
                compare('original_6301_32_vs_qualified_rational32', rational32, actual)
            science_folder = Path(state['folder'])
            report['migrations'].append({
                'key': key, 'step': state['step'], 'status': 'PASS',
                'source_commit': row['source_commit'], 'new_source_commit': commit,
                'old_files': copy.deepcopy(row['files']),
                'new_files': [_reference(p) for p in sorted(science_folder.rglob('*')) if p.is_file()]})
            write_report()
        report['status'] = 'PASS'
        report['formal_updates_executed_here'] = 0
        write_report()
        return report, initial_states
    except BaseException as exc:
        report['status'] = 'FAIL'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        report['formal_updates_executed_here'] = 0
        write_report()
        raise
