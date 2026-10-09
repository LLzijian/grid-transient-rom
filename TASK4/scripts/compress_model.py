"""Sequential TRAIN-only scalar compression and short joint adaptation.

This is one managed process, with no subprocess runner and no development/TEST
access. Final single-initialization 5 s truth fine-tuning and independent
evaluation remain required even when all seven functions have been exported.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from task4.data import load_events, train_scales
from task4.dynamics import PowerGraph
from task4.integrators import rollout
from task4.objectives import ShootingBank
from task4.paths import ROOT, project_path, read_json, write_json
from task4.symbolic import PolynomialFunction, fit_sparse_hermite
from task4.symbolic_runtime import FUNCTION_NAMES, export_symbolic
from task4.training import check_authorized, checkpoint as save_checkpoint, fit, load_checkpoint


HORIZONS = (0.25, 0.5, 1.0, 2.0, 5.0)


def safe_diagnostics(value):
    """Preserve diagnostic failures as null plus paths, never JSON NaN/Infinity."""
    issues = []

    def clean(item, path):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().tolist()
        elif isinstance(item, np.ndarray):
            item = item.tolist()
        elif isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            issues.append({'path': path, 'reason': 'nonfinite numeric diagnostic',
                           'representation': str(item)})
            return None
        if isinstance(item, dict):
            return {key: clean(child, path+'.'+str(key)) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child, path+f'[{index}]') for index, child in enumerate(item)]
        return item

    result = clean(value, '$')
    if issues and isinstance(result, dict):
        result['diagnostics_finite'] = False
        result['nonfinite_diagnostics'] = issues
        if 'complete' in result:
            result['complete'] = False
    return result, issues


def write_diagnostics(path, value):
    """Save evidence before rejecting an unexpected nonfinite diagnostic."""
    cleaned, issues = safe_diagnostics(value)
    write_json(path, cleaned)
    if issues:
        raise FloatingPointError('Nonfinite diagnostics saved: '+', '.join(item['path'] for item in issues))


def check_deadline(deadline, estimate=0.0):
    if time.time()+max(15.0, 1.5*estimate) >= deadline:
        raise TimeoutError('Compression save deadline reached')


def fixed_windows(length, count=8):
    if length < 12 or count < 1:
        raise ValueError('Expected an eligible real prefix and positive window count')
    return np.unique(np.linspace(10, length-2, count, dtype=np.int64))


def train_panel(events, limit=32):
    first = {}
    for index, event in enumerate(events):
        first.setdefault((event.group, event.label), index)
    return list(first.values())[:limit]


def validate_teacher(teacher, obj, events, physics, scales):
    """Refuse source/mode remapping before collecting any distillation state."""
    cfg = obj['config']
    if (type(teacher) is not PowerGraph or obj.get('kind') != 'T4-N'
            or obj.get('synthetic') is not False or cfg.get('synthetic') is not False
            or cfg.get('arm') != 'P' or obj.get('symbolic_functions')):
        raise ValueError('Expected an original nonsynthetic P-arm T4-N teacher')
    if cfg.get('hidden') != 32 or cfg.get('latent_per_machine') != 2:
        raise ValueError('Teacher must retain shared GRU32 and two latent states per machine')
    ids = [event.id for event in events]
    if len(set(ids)) != len(ids) or obj.get('event_ids') != ids:
        raise ValueError('Teacher full TRAIN ID list/order differs from the current legal TRAIN source')
    for candidate in (obj['scales'], teacher.scales.detach().cpu().numpy()):
        if not np.allclose(candidate, scales, rtol=1e-9, atol=1e-12):
            raise ValueError('Teacher and current full-TRAIN scales differ')
    for key, value in physics.as_dict().items():
        if key not in obj['physics'] or not np.allclose(value, obj['physics'][key], rtol=1e-9, atol=1e-12):
            raise ValueError('Teacher physics differs: '+key)
        if key != 'omega_s' and not np.allclose(
                value, getattr(teacher, key).detach().cpu().numpy(), rtol=1e-9, atol=1e-12):
            raise ValueError('Teacher state buffer differs from physical metadata: '+key)


def perturb_states(states, model, fraction=0.02, seed=404):
    """Coordinatewise bounded teacher inputs; these are never FOM trajectories."""
    if not 0 <= fraction <= 0.02:
        raise ValueError('Only perturbations up to 2% of each declared state scale are allowed')
    values = states.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite teacher states before perturbation')
    _, _, _, q = model.split(states)
    with np.errstate(over='ignore', invalid='ignore'):
        latent_scale = np.maximum(np.sqrt(np.mean(q.detach().cpu().numpy()**2, axis=0)), 1e-6)
    if not np.isfinite(latent_scale).all():
        raise FloatingPointError('Nonfinite latent perturbation scale from RMS arithmetic')
    s = model.scales.detach().cpu().numpy()
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        state_scale = np.concatenate([np.full(model.n, s[0]),
                                      model.mass.detach().cpu().numpy()*s[1],
                                      np.full(model.n, s[2]), latent_scale.reshape(-1)])
    if not np.isfinite(state_scale).all() or np.any(state_scale <= 0):
        raise FloatingPointError('Invalid full-state perturbation scale')
    rng = np.random.default_rng(seed)
    with np.errstate(over='ignore', invalid='ignore'):
        displacement = rng.uniform(-fraction, fraction, values.shape)*state_scale
        perturbed_values = values+displacement
    if not np.isfinite(displacement).all() or not np.isfinite(perturbed_values).all():
        raise FloatingPointError('Nonfinite displacement or perturbed teacher state')
    perturbed = torch.as_tensor(perturbed_values, dtype=torch.float64, device=states.device)
    return perturbed, state_scale


def _save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def collect_teacher_states(teacher, encoder, events, output, deadline, seed):
    """One fixed teacher state set reused across every scalar-function fit."""
    started = time.monotonic()
    states, origins = [], []
    manifest = {'complete': False, 'test_payload_read': False, 'split': 'train',
                'all_train_source_ids': [event.id for event in events],
                'window_rule': 'unique integer linspace(10,len(y)-2,8)',
                'panel_rule': 'first event per (group,label), at most 32, in legal TRAIN order',
                'horizons_after_initialization_seconds': list(HORIZONS),
                'sources': origins, 'generated_fom_data': False,
                'boundary': 'encoded q is inferred; autonomous/perturbed states are teacher inputs, never future truth'}
    failure = None
    try:
        with torch.no_grad():
            for event in events:
                check_deadline(deadline)
                indices = fixed_windows(len(event.y))
                histories = torch.as_tensor(np.stack([event.y[k-10:k+1] for k in indices]),
                                            dtype=torch.float64, device=teacher.mass.device)
                encoded = teacher.initial(histories, encoder).detach()
                if not torch.isfinite(encoded).all():
                    raise FloatingPointError('Nonfinite encoded TRAIN state: '+event.id)
                states.extend(encoded.unbind(0))
                origins.extend({'kind': 'TRAIN_window_encoded', 'event_id': event.id, 'sample_index': int(k)}
                               for k in indices)
            for index in train_panel(events):
                check_deadline(deadline)
                event = events[index]
                offsets = [(tau, round(tau/0.01)) for tau in HORIZONS
                           if round(tau/0.01) <= len(event.y)-11]
                if not offsets:
                    continue
                history = torch.as_tensor(event.y[:11], dtype=torch.float64, device=teacher.mass.device)[None]
                z0 = teacher.initial(history, encoder)
                teacher.save_deadline = deadline-15.0
                predicted = rollout(teacher, z0, offsets[-1][1], sample_dt=0.01, max_step=0.01,
                                    training=False)
                for tau, step in offsets:
                    states.append(predicted[0, step].detach())
                    origins.append({'kind': 'teacher_autonomous', 'event_id': event.id,
                                    'horizon_seconds': tau, 'sample_index': 10+step,
                                    'truth_prefix_end_index': len(event.y)-1})
        check_deadline(deadline)
        base = torch.stack(states)
        perturbed, scales = perturb_states(base, teacher, fraction=0.02, seed=seed+701)
        states.extend(perturbed.unbind(0))
        origins.extend({'kind': 'teacher_perturbed', 'parent_state_index': j,
                        'maximum_fraction_of_scale': 0.02} for j in range(len(base)))
        manifest.update(complete=True, perturbation_state_scales=scales.tolist(),
                        perturbation_seed=seed+701, perturbation_count=len(base))
    except BaseException as exc:
        failure = exc
        manifest['stop_reason'] = type(exc).__name__+': '+str(exc)
    values = torch.stack(states) if states else torch.empty((0, teacher.dim), dtype=torch.float64,
                                                           device=teacher.mass.device)
    _save_npz(output/'teacher_states.npz', states=values.detach().cpu().numpy())
    manifest.update(state_count=len(states), wall_seconds=time.monotonic()-started)
    write_diagnostics(output/'teacher_states_manifest.json', manifest)
    if failure is not None:
        raise failure
    return values, manifest


def function_features(model, states, name):
    """Return observation-free function inputs and separately reserved origins."""
    if name not in FUNCTION_NAMES:
        raise ValueError('Unknown function '+name)
    d, pi, p, q = model.split(states)
    eta = d@model.incidence
    v = model._scaled_s(p, q) if name.endswith('energy') else model._local(pi/model.mass, p, q)
    zero_v, zero_eta = torch.zeros_like(v[:1]), torch.zeros_like(eta[:1])
    if name.startswith('edge_'):
        features = torch.cat([model._edge_input(eta, v), model._edge_input(eta, v, True)], dim=-2)
        origins = torch.cat([model._edge_input(zero_eta, zero_v),
                             model._edge_input(zero_eta, zero_v, True)], dim=-2)
    else:
        features, origins = model._node_input(v), model._node_input(zero_v)
    return features.reshape(-1, features.shape[-1]).detach(), origins.reshape(-1, origins.shape[-1]).detach()


def select_features(features, origins, max_samples, seed):
    if max_samples < len(origins)+2:
        raise ValueError('max_samples must reserve all graph origins plus at least two state feature rows')
    if not torch.isfinite(features).all() or not torch.isfinite(origins).all():
        raise FloatingPointError('Nonfinite teacher function features')
    count = min(len(features), max_samples-len(origins))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(features), count, replace=False)
    # Explicitly reserve graph origins, with physical mu intact. These and the
    # random state rows are all TRAIN-derived; the last fifth is the inner check.
    selected = torch.cat([origins, features[torch.as_tensor(chosen, device=features.device)]], dim=0)
    return selected, {'state_feature_pool': len(features), 'origin_rows': len(origins),
                      'selected_rows': len(selected), 'seed': seed,
                      'selected_state_feature_indices': chosen.tolist()}


def fit_candidate_libraries(teacher_function, x, deadline, max_terms=16, on_candidate=None):
    candidates, facts = [], []
    previous_cost = 20.0
    for degree in (2, 3):
        check_deadline(deadline, previous_cost)
        tick = time.monotonic()
        detail = {'degree': degree}
        try:
            student, detail = fit_sparse_hermite(teacher_function, x, max_terms=max_terms,
                                                 degree=degree, ridge=1e-6, gradient_weight=0.1)
            detail, issues = safe_diagnostics(detail)
            if issues:
                raise FloatingPointError('Nonfinite sparse candidate diagnostics: '+', '.join(item['path'] for item in issues))
            if detail['inner_check_samples'] < 1:
                raise ValueError('No inner-TRAIN check samples')
            score = detail['inner_check_value_mse']+0.1*detail['inner_check_gradient_mse']
            detail['selection_score'] = score
            if not np.isfinite(score) or not torch.isfinite(student.coefficients).all():
                raise FloatingPointError('Nonfinite sparse candidate')
            if max(student.export()['nonzero_per_output']) > max_terms:
                raise ValueError('Sparse candidate exceeded the per-output support budget')
            detail.update(selection_score=score, selection_scope='inner-TRAIN rows only',
                          selection_rule='value_mse + 0.1 * gradient_mse; tie -> lower degree',
                          max_terms_per_output=max_terms, status='fitted')
            candidates.append((score, degree, student))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            detail, _ = safe_diagnostics(detail)
            detail.update(degree=degree, status='failed', reason=type(exc).__name__+': '+str(exc))
        previous_cost = time.monotonic()-tick
        detail['fit_wall_seconds'] = previous_cost
        facts.append(detail)
        if on_candidate is not None:
            on_candidate(detail, student if detail['status'] == 'fitted' else None)
    if not candidates:
        raise RuntimeError('Both finite polynomial libraries failed: '+json.dumps(facts, allow_nan=False))
    _, degree, student = min(candidates, key=lambda item: (item[0], item[1]))
    return student, {'candidates': facts, 'selected_degree': degree,
                     'selected_nonzero_per_output': student.export()['nonzero_per_output']}


def field_gap(model, probes, reference_field, reference_gradient):
    """Teacher agreement at fixed inputs, expressly not rollout truth accuracy."""
    with torch.no_grad():
        actual_field = model.field(probes, create_graph=False)
        actual_gradient = model.energy_gradient(probes)
    result = {'scope': 'fixed TRAIN-derived teacher probe states; not prediction truth metrics'}
    for name, value, reference in [('field', actual_field, reference_field),
                                    ('energy_gradient', actual_gradient, reference_gradient)]:
        finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
        result.update({name+'_inputs_finite': finite, name+'_finite': False,
                       name+'_rms_difference': None, name+'_teacher_rms': None,
                       name+'_relative_rms_difference': None, name+'_reason': None})
        try:
            if not finite:
                raise FloatingPointError('Nonfinite probe values or teacher reference')
            difference = value-reference
            if not torch.isfinite(difference).all():
                raise FloatingPointError('Difference overflow from finite probe inputs')
            error_squared, reference_squared = difference.square(), reference.square()
            if not torch.isfinite(error_squared).all() or not torch.isfinite(reference_squared).all():
                raise FloatingPointError('Squared difference or teacher square overflow')
            error = float(error_squared.mean().sqrt())
            rms = float(reference_squared.mean().sqrt())
            if not np.isfinite(error) or not np.isfinite(rms):
                raise FloatingPointError('Mean/square-root diagnostic overflow')
            relative = error/max(rms, 1e-12)
            if not np.isfinite(relative):
                raise FloatingPointError('Relative RMS diagnostic overflow')
            result.update({name+'_finite': True, name+'_rms_difference': error,
                           name+'_teacher_rms': rms, name+'_relative_rms_difference': relative})
        except FloatingPointError as exc:
            result[name+'_reason'] = str(exc)
    result['diagnostics_finite'] = result['field_finite'] and result['energy_gradient_finite']
    return result


def adaptation_config(base, teacher_path, adapt_steps, name, device, seed):
    cfg = dict(base)
    # C uses TRAIN teacher targets even when N was trained without auxiliaries.
    # Preserve the source's latent/torque weights, including explicit zeros.
    cfg.update(name='compress_adapt_'+name, synthetic=False, arm='P', hidden=32,
               latent_per_machine=2, steps=adapt_steps, horizons=[0.25], batch_size=2,
               lr=1e-4, shooting_fraction=0.0, final_autonomous_fraction=1.0,
               autonomous_only=False, anchor_weight=0.0,
               latent_weight=base.get('latent_weight', 0.0),
               torque_weight=base.get('torque_weight', 0.0),
               teacher_checkpoint=str(teacher_path),
               teacher_field_weight=0.1, teacher_gradient_weight=0.1,
               compression_source_autonomous_only=bool(base.get('autonomous_only', False)),
               teacher_usage='TRAIN_only_compression_adaptation',
               prediction_head_used_or_deployed=False,
               milestones=[], monitor_count=1, save_every=1, seed=seed, device=device,
               compression_stage=name, final_single_initialization_5s_finetune_completed=False)
    return cfg


def complexity(model):
    functions = {}
    for name, module in model.function_modules().items():
        if isinstance(module, PolynomialFunction):
            exported = module.export()
            functions[name] = {'nonzero_per_output': exported['nonzero_per_output'],
                               'union_atoms': len(exported['powers']),
                               'total_nonzero_coefficients': sum(exported['nonzero_per_output'])}
    return {'functions': functions, 'compressed_functions': len(functions),
            'total_nonzero_coefficients': sum(f['total_nonzero_coefficients'] for f in functions.values())}


def compress(checkpoint, output, *, adapt_steps=16, max_samples=2048, seed=419, device='cpu'):
    check_authorized(False)  # before any checkpoint/source payload reads or output creation
    if adapt_steps < 1 or max_samples < 2:
        raise ValueError('Positive adaptation steps and at least two feature samples are required')
    deadline = float(os.environ.get('TASK4_SAVE_DEADLINE_UNIX', 'nan'))
    if not np.isfinite(deadline):
        raise PermissionError('A finite managed save deadline is required')
    check_deadline(deadline)
    teacher_path = project_path(checkpoint)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report = {'complete': False, 'teacher_checkpoint': str(teacher_path),
              'adapt_steps_per_function': adapt_steps, 'max_samples_per_function': max_samples,
              'seed': seed, 'device': device, 'functions': [], 'stop_reason': 'not_started',
              'test_payload_read': False, 'validation_payload_read': False, 'generated_fom_data': False,
              'final_single_initialization_5s_finetune_completed': False,
              'independent_development_evaluation_completed': False,
              'required_followup': 'Complete single-initialization 5 s truth fine-tuning, then independent N/S evaluation',
              'acceptance': 'No automatic model promotion from function fitting or short adaptation'}
    write_diagnostics(output/'compression_summary.json', report)
    failure = None
    try:
        project_cfg = read_json(ROOT/'configs/project.json')
        if project_cfg['history_ms'] != 100 or project_cfg['sample_dt'] != 0.01:
            raise ValueError('The 100 ms/11-point observation contract must remain unchanged')
        teacher, teacher_encoder, _, original = load_checkpoint(teacher_path, device)
        check_deadline(deadline)
        events, physics, source_facts = load_events(project_cfg, 'train')
        scales = train_scales(events, physics)
        validate_teacher(teacher, original, events, physics, scales)
        write_diagnostics(output/'data_facts.json', {**source_facts, 'all_train_source_ids': [e.id for e in events],
                                                   'scales': scales.tolist(), 'teacher_source_checks_passed': True})
        for module in (teacher, teacher_encoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        model, encoder, head, _ = load_checkpoint(teacher_path, device)
        base = read_json(ROOT/'configs/experiment.json')
        base.update(original['config'])
        base.update(synthetic=False, arm='P', hidden=32, latent_per_machine=2,
                    _event_ids=[e.id for e in events], model_strength=model.strength)
        rng = np.random.default_rng(seed)

        def save_asset(path, stage):
            config = dict(base)
            config.update(compression_stage=stage, warm_start_only=True,
                          autonomous_only=False, anchor_weight=0.0,
                          latent_weight=base.get('latent_weight', 0.0),
                          torque_weight=base.get('torque_weight', 0.0),
                          teacher_checkpoint=str(teacher_path),
                          teacher_field_weight=0.1, teacher_gradient_weight=0.1,
                          compression_source_autonomous_only=bool(base.get('autonomous_only', False)),
                          teacher_usage='TRAIN_only_compression_adaptation',
                          prediction_head_used_or_deployed=False,
                          final_single_initialization_5s_finetune_completed=False)
            parameters = [parameter for module in (model, encoder, head)
                          for parameter in module.parameters() if parameter.requires_grad]
            optimizer = torch.optim.Adam(parameters, lr=1e-4)
            save_checkpoint(path, model, encoder, head, ShootingBank().to(device), optimizer,
                            config, physics, 0, rng, report['functions'], boundary_complete=False)

        save_asset(output/'last.pt', 'original_teacher_before_compression')
        states, state_facts = collect_teacher_states(teacher, teacher_encoder, events, output, deadline, seed)
        report['teacher_state_count'] = len(states)
        report['teacher_state_wall_seconds'] = state_facts['wall_seconds']
        probe_ids = np.random.default_rng(seed+902).choice(len(states), min(64, len(states)), replace=False)
        probes = states[torch.as_tensor(probe_ids, device=states.device)]
        check_deadline(deadline)
        with torch.no_grad():
            reference_field = teacher.field(probes, create_graph=False)
            reference_gradient = teacher.energy_gradient(probes)
        if not torch.isfinite(reference_field).all() or not torch.isfinite(reference_gradient).all():
            raise FloatingPointError('Original teacher is nonfinite on fixed distillation probes')
        report['field_probe_state_indices'] = probe_ids.tolist()
        for index, name in enumerate(FUNCTION_NAMES):
            check_deadline(deadline)
            row = {'name': name, 'fitting_complete': False, 'adaptation_complete': False,
                   'teacher': 'original fixed T4-N', 'feature_source': 'fixed TRAIN teacher states'}
            report['functions'].append(row)
            features, origins = function_features(teacher, states, name)
            x, sampling = select_features(features, origins, max_samples, seed+1000+index)
            del features, origins
            row['sampling'] = sampling
            row['empirical_feature_envelope'] = {
                'minimum': x.min(dim=0).values.detach().cpu().tolist(),
                'maximum': x.max(dim=0).values.detach().cpu().tolist(),
                'meaning': 'TRAIN teacher fitting sample envelope; not a certified accuracy or stability domain',
            }
            _save_npz(output/('features_'+name+'.npz'), features=x.detach().cpu().numpy())
            row['candidate_libraries'] = []
            def record_candidate(detail, candidate):
                path = output/f'candidate_{name}_degree{detail["degree"]}.json'
                write_diagnostics(path, {'facts': detail,
                                         'function': None if candidate is None else candidate.export()})
                row['candidate_libraries'].append(str(path))
                write_diagnostics(output/'compression_summary.json', report)
            scalar, row['fitting'] = fit_candidate_libraries(getattr(teacher, name), x, deadline,
                                                             on_candidate=record_candidate)
            setattr(model, name, scalar)
            row['fitting_complete'] = True
            mixed = output/('mixed_'+name+'.pt')
            save_asset(mixed, name+'_before_adaptation')
            save_asset(output/'last.pt', name+'_before_adaptation')
            row['mixed_checkpoint'] = str(mixed)
            check_deadline(deadline)
            row['before_adaptation'] = field_gap(model, probes, reference_field, reference_gradient)
            write_diagnostics(output/'compression_summary.json', report)
            if not row['before_adaptation']['diagnostics_finite']:
                raise FloatingPointError('Nonfinite before-adaptation probe diagnostics; last checkpoint retained')
            check_deadline(deadline, 20.0)
            cfg = adaptation_config(base, teacher_path, adapt_steps, name, device, seed+2000+index)
            adapted_directory = output/('adapt_'+name)
            # Same Python process and original managed run; no nested billing or
            # exact resume. fit reloads legal TRAIN and computes physical losses.
            row['adaptation'] = fit(cfg, project_cfg, adapted_directory, warm_start=mixed)
            adapted_path = adapted_directory/'last.pt'
            model, encoder, head, adapted = load_checkpoint(adapted_path, device)
            row['adaptation_checkpoint'] = str(adapted_path)
            row['adaptation_complete'] = (row['adaptation']['steps_completed'] == adapt_steps
                                          and row['adaptation']['stop_reason'] == 'steps_completed')
            save_asset(output/'last.pt', name+'_after_adaptation')
            row['complexity_after_adaptation'] = complexity(model)
            if not row['adaptation_complete']:
                raise TimeoutError('Adaptation did not finish before its managed save deadline')
            check_deadline(deadline)
            row['after_adaptation'] = field_gap(model, probes, reference_field, reference_gradient)
            write_diagnostics(output/'compression_summary.json', report)
            if not row['after_adaptation']['diagnostics_finite']:
                raise FloatingPointError('Nonfinite after-adaptation probe diagnostics; last checkpoint retained')
        check_deadline(deadline)
        export_symbolic(model, encoder, output/'model_symbolic.json', omega_s=physics.omega_s,
                        metadata={'teacher_checkpoint': str(teacher_path), 'source': 'TRAIN-only sequential compression',
                                  'final_single_initialization_5s_finetune_completed': False,
                                  'prediction_accuracy_or_stability_claim': False})
        save_asset(output/'last.pt', 'all_functions_short_adapted_requires_full_truth_finetune')
        report.update(complete=True, stop_reason='seven_functions_compressed_and_short_adapted',
                      final_checkpoint=str(output/'last.pt'), symbolic_export=str(output/'model_symbolic.json'),
                      complexity=complexity(model))
    except TimeoutError as exc:
        report.update(stop_reason='save_deadline', limitation=str(exc))
    except BaseException as exc:
        failure = exc
        report.update(stop_reason=type(exc).__name__+': '+str(exc))
    finally:
        report['fitted_functions'] = sum(row['fitting_complete'] for row in report['functions'])
        report['adapted_functions'] = sum(row['adaptation_complete'] for row in report['functions'])
        report['wall_seconds_including_data_and_all_substages'] = time.monotonic()-started
        report, issues = safe_diagnostics(report)
        if issues:
            report.update(complete=False, stop_reason='nonfinite_diagnostics_saved')
            if failure is None:
                failure = FloatingPointError('Nonfinite compression diagnostics; last checkpoint retained')
        write_json(output/'compression_summary.json', report)
    if failure is not None:
        raise failure
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapt-steps', type=int, default=16)
    parser.add_argument('--max-samples', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=419)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    print(json.dumps(compress(**vars(parser.parse_args())), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
