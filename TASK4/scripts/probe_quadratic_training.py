"""Three ordered TRAIN probes from S400; only diagnostic arrays are saved.
Timing includes first-backward effects, not deployment speed/accuracy evidence.
The checkpoint, support and scientific training configuration remain fixed.
"""
import _bootstrap  # noqa: F401

import argparse, json, os, time

import numpy as np
import torch

from audit_structure import check_deadline, report_for_json, same_array, save_npz, sha256
from task4.accelerate import trace_field
from task4.data import load_events, train_scales
from task4.dynamics import PowerGraph
from task4.objectives import ShootingBank, batch_objective
from task4.paths import ROOT, project_path, read_json, write_json
from task4.symbolic import PolynomialFunction
from task4.training import check_authorized, load_checkpoint, optimizer_groups

MODES = ('reference', 'energy_only', 'all_poly')
ATOL, RTOL = 1e-9, 1e-7  # Descriptive engineering check, not a scientific acceptance gate.


def fixed_panel(events):
    panel = [next((event for event in events if event.label == label), None) for label in ('stable', 'unstable')]
    if any(event is None for event in panel):
        raise ValueError('TRAIN must contain both a stable and an unstable eligible event')
    return panel


def validate_source(model, saved, events=None, physics=None, scales=None):
    cfg = saved['config']
    if (type(model) is not PowerGraph or saved.get('kind') != 'T4-S' or saved.get('synthetic') is not False
            or cfg.get('synthetic') is not False or cfg.get('arm') != 'P'
            or saved.get('step') != 400 or cfg.get('steps') != 400 or not saved.get('exact_resume_eligible', False)):
        raise ValueError('Require a real, complete-update P-arm S400 checkpoint')
    if (cfg.get('hidden') != 32 or cfg.get('latent_per_machine') != 2 or cfg.get('dtype') != 'float64'
            or not cfg.get('autonomous_only') or cfg.get('horizons') != [5.0] or cfg.get('integration_dt') != .01
            or cfg.get('shooting_fraction') != 0 or cfg.get('final_autonomous_fraction') != 1
            or cfg.get('batch_size') != 2 or not cfg.get('_optimizer_blocks') or saved.get('shooting_keys')):
        raise ValueError('Require the registered GRU32/rank2, single-start 5 s S400 training contract')
    if cfg.get('teacher_checkpoint') or any(cfg.get(key, 0) != 0 for key in (
            'anchor_weight', 'latent_weight', 'torque_weight', 'teacher_field_weight', 'teacher_gradient_weight', 'continuity_weight')):
        raise ValueError('All auxiliary weights and teacher input must be absent')
    modules = model.function_modules()
    if len(modules) != 7 or set(saved.get('symbolic_functions', {})) != set(modules):
        raise ValueError('All seven symbolic modules must be present')
    for module in modules.values():
        if not isinstance(module, PolynomialFunction) or module._quadratic_rejection is not None:
            raise ValueError('Require unique degree <=2 polynomial modules in every execution mode')
    if any(p.dtype != torch.float64 or p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('Probe requires float64 CPU parameters')
    if events is not None:
        ids = [event.id for event in events]
        if len(set(ids)) != len(ids) or saved.get('event_ids') != ids or cfg.get('_event_ids') != ids:
            raise ValueError('Checkpoint IDs/order differ from the complete legal TRAIN population')
        same_array(saved['scales'], scales, 'TRAIN scales')
        same_array(model.scales.numpy(), scales, 'model scales')
        for key, value in physics.as_dict().items():
            same_array(saved['physics'][key], value, 'physics '+key)
            if hasattr(model, key):
                same_array(getattr(model, key).numpy(), value, 'model physics '+key)


def configure_mode(model, mode):
    if mode not in MODES:
        raise ValueError('Unknown fixed execution mode')
    for name, module in model.function_modules().items():
        module.set_quadratic_execution(mode == 'all_poly' or (mode == 'energy_only' and name.endswith('_energy')))
    model.reuse_origin_terms, model.checkpoint_steps = True, 0


def difference(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        return {'finite': False, 'within_tolerance': False, 'reason': 'shape_mismatch'}
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        error = np.abs(a-b)
        absolute = float(np.max(error)) if error.size else 0.
        relative = absolute/max(float(np.max(np.abs(b))) if b.size else 0., 1e-12)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(error).all()
                      and np.isfinite(absolute) and np.isfinite(relative))
        return {'finite': finite, 'max_absolute': absolute, 'relative_to_reference_inf_norm_floor_1e_12': relative,
                'within_tolerance': bool(finite and np.all(error <= ATOL+RTOL*np.abs(b)))}


def run_mode(source, digest, mode, panel, deadline, row, arrays):
    def timed(name, function):
        check_deadline(deadline, max(15., 2*row['seconds'].get('loss', 0.)+15.) if name == 'backward' else 15.)
        started = time.perf_counter()
        try:
            return function()
        finally:
            row['seconds'][name] = time.perf_counter()-started

    model, encoder, head, saved = timed('load', lambda: load_checkpoint(source, device='cpu'))
    if sha256(source, deadline) != digest:
        raise ValueError('Source checkpoint changed before/during mode load')
    validate_source(model, saved)
    cfg = saved['config']  # No loss, learning-rate, clipping or horizon overrides.
    head.requires_grad_(False)
    if not all(p.requires_grad and p.dtype == torch.float64 and p.device.type == 'cpu'
               for module in (model, encoder) for p in module.parameters()):
        raise ValueError('Every ODE/GRU parameter must remain trainable float64 CPU')
    configure_mode(model, mode)
    model.save_deadline = deadline-15.
    groups = optimizer_groups(model, encoder, head, cfg)
    expected_groups = [(g['name'], g['lr']) for g in groups]
    optimizer = torch.optim.Adam(groups, lr=cfg['lr'])
    optimizer.load_state_dict(saved['optimizer'])
    for group, expected in zip(optimizer.param_groups, expected_groups):
        if (group.get('name'), group['lr']) != expected:
            raise ValueError('Restored Adam groups/learning rates differ from the source configuration')
    torch.set_rng_state(saved['rng_torch'])
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['rng_numpy']
    named = [('model.'+name, p) for name, p in model.named_parameters()]+[('encoder.'+name, p) for name, p in encoder.named_parameters()]
    row['parameters'] = [{'name': name, 'shape': list(p.shape), 'gradient_present': False} for name, p in named]
    arrays['parameter_names'] = np.asarray([name for name, _ in named], dtype=str)
    row['optimizer_groups'] = [{'name': g['name'], 'lr': g['lr'], 'parameter_count': len(g['params'])} for g in optimizer.param_groups]
    timed('trace', lambda: trace_field(model))
    history = torch.as_tensor(np.stack([event.y[:11] for event in panel]), dtype=torch.float64)
    with torch.no_grad():
        arrays['initial_field'] = timed('initial_field_check', lambda: model.field(model.initial(history, encoder), create_graph=False)).numpy().copy()
    if not np.isfinite(arrays['initial_field']).all():
        raise FloatingPointError('Nonfinite fixed initial field; no replacement state')
    before = {name: p.detach().clone() for name, p in named}
    optimizer.zero_grad(set_to_none=True)
    step_started = time.perf_counter()
    loss, terms = timed('loss', lambda: batch_objective(model, encoder, head, panel, 5., cfg, None, rng,
                                                       ShootingBank(), optimizer, None, 0., None))
    row['loss'], row['terms'] = float(loss.detach()), {key: float(value.detach()) for key, value in terms.items()}
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss; preserve this fixed panel and stop this mode')
    timed('backward', loss.backward)
    for index, (name, parameter) in enumerate(named):
        row['parameters'][index]['gradient_present'] = parameter.grad is not None
        arrays[f'gradient_{index}'] = (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).detach().numpy().copy()
    norm = timed('clip', lambda: torch.nn.utils.clip_grad_norm_([p for _, p in named], cfg['grad_clip'], error_if_nonfinite=True))
    row['raw_global_gradient_norm'] = float(norm)
    timed('adam', optimizer.step)
    for index, (name, parameter) in enumerate(named):
        arrays[f'update_{index}'] = (parameter.detach()-before[name]).numpy().copy()
    row['full_step_seconds_including_diagnostic_copies'] = time.perf_counter()-step_started
    row['loss_backward_clip_adam_seconds'] = sum(row['seconds'][key] for key in ('loss', 'backward', 'clip', 'adam'))
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key != 'parameter_names'):
        raise FloatingPointError('Nonfinite field/gradient/update retained in diagnostic NPZ')
    row['status'] = 'completed'


def compare_mode(row, arrays, reference_row, reference):
    if reference_row.get('status') != 'completed' or row.get('status') != 'completed':
        return {'available': False, 'reason': 'reference_or_candidate_step_failed_or_incomplete'}
    layout = lambda items: [(item['name'], item['shape']) for item in items]
    if layout(row['parameters']) != layout(reference_row['parameters']):
        return {'available': False, 'reason': 'parameter_order_or_shape_mismatch'}
    return {'available': True, 'initial_field': difference(arrays['initial_field'], reference['initial_field']),
            'loss': difference(row['loss'], reference_row['loss']),
            'parameters': [{'name': item['name'],
                            'gradient_presence_matches': item['gradient_present'] == reference_row['parameters'][i]['gradient_present'],
                            'gradient': difference(arrays[f'gradient_{i}'], reference[f'gradient_{i}']),
                            'update': difference(arrays[f'update_{i}'], reference[f'update_{i}'])}
                           for i, item in enumerate(row['parameters'])]}


def probe(checkpoint, output):
    started = time.monotonic()
    check_authorized(False)  # Before any checkpoint/TRAIN/output access.
    deadline = float(os.environ.get('TASK4_SAVE_DEADLINE_UNIX', 'nan'))
    if not np.isfinite(deadline):
        raise PermissionError('A finite managed save deadline is required')
    check_deadline(deadline, 15.)
    source, output = project_path(checkpoint), project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'status': 'incomplete', 'checkpoint': str(source), 'modes': [],
              'runtime_execution': {'trace_field': True, 'reuse_origin_terms': True, 'checkpoint_steps': 0,
                                    'reference': 'generic polynomial with the same executor combination; not original training cost'},
              'scientific_candidate_saved': False, 'optimization_steps_per_mode': 1, 'test_or_development_payload_read': False,
              'selection': 'first eligible stable and first eligible unstable in legal TRAIN order; no loss selection',
              'comparison_tolerance': {'absolute': ATOL, 'relative': RTOL},
              'limits': 'One ordered execution timing each; no accuracy improvement, deployment speed, or model promotion claim.'}
    digest, caught = None, None
    def persist():
        report['process_wall_seconds_including_all_stages'] = time.monotonic()-started
        write_json(output/'probe_summary.json', report_for_json(report))
    try:
        digest = report['checkpoint_sha256_before'] = sha256(source, deadline)
        model, encoder, head, saved = load_checkpoint(source, device='cpu')
        validate_source(model, saved)
        torch.set_num_threads(1)
        project = read_json(ROOT/'configs/project.json')
        check_deadline(deadline, 15.)
        events, physics, facts = load_events(project, 'train')
        scales = train_scales(events, physics)
        validate_source(model, saved, events, physics, scales)
        if facts.get('split') != 'train' or facts.get('test_payload_read'):
            raise PermissionError('Only legal TRAIN source data is allowed')
        panel = fixed_panel(events)
        report.update(source_facts=facts, all_train_ids=[event.id for event in events],
                      panel=[{'id': e.id, 'label': e.label, 'forecast_steps': min(500, len(e.y)-11)} for e in panel],
                      source_config=saved['config'], source_step=saved['step'])
        del model, encoder, head, saved
        reference, reference_row, estimated_step = None, {}, 0.
        persist()
        for mode in MODES:
            check_deadline(deadline, max(15., estimated_step*1.5))
            row, arrays = {'mode': mode, 'status': 'incomplete', 'seconds': {}}, {}
            report['modes'].append(row)
            persist()
            try:
                run_mode(source, digest, mode, panel, deadline, row, arrays)
            except (FloatingPointError, RuntimeError) as exc:
                row.update(status='failed', failure=type(exc).__name__+': '+str(exc))
            finally:
                save_npz(output/(mode+'_diagnostics.npz'), **arrays)
                row['artifact'] = mode+'_diagnostics.npz'
                persist()
            if mode == 'reference':
                reference, reference_row = arrays, row
            else:
                row['comparison_to_reference'] = compare_mode(row, arrays, reference_row, reference)
            estimated_step = max(estimated_step, row.get('full_step_seconds_including_diagnostic_copies', 0.))
            persist()
        passed = all(r['status'] == 'completed' for r in report['modes'])
        report.update(complete=True, status='probe_complete' if passed else 'probe_complete_with_failures', all_modes_completed=passed)
    except TimeoutError as exc:
        report.update(stop_reason='save_deadline', failure=str(exc))
    except BaseException as exc:
        caught = exc
        report.update(stop_reason=type(exc).__name__, failure=str(exc))
    finally:
        if digest is not None:
            try:
                report['checkpoint_sha256_after'] = sha256(source)
                report['checkpoint_unchanged'] = digest == report['checkpoint_sha256_after']
                if not report['checkpoint_unchanged']:
                    report.update(complete=False, status='source_changed')
            except OSError as exc:
                report.update(complete=False, status='source_guard_failed', failure=str(exc))
        persist()
    if caught is not None:
        raise caught
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    result = probe(**vars(parser.parse_args()))
    print(json.dumps({'status': result['status'], 'complete': result['complete']}))
