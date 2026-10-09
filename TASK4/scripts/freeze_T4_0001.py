"""Freeze the already registered first-loop endpoints before development reads.

Metadata/checkpoint audit only: no event loading or model integration. This
protocol-specific manifest does not select, train, evaluate, or accept a model.
"""
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from task4.paths import ROOT, project_path, read_json, write_json, source_root, source_file
from task4.training import check_authorized, load_checkpoint
from task4.symbolic import PolynomialFunction


ENDPOINTS = {
    'P-N': ('P_long_cached', 'T4-N', 'P', 'P_short_resume'),
    'P-S': ('S_final', 'T4-S', 'P', None),
    'A': ('A_long', 'T4-N', 'A', 'A_short'),
    'U': ('U_long', 'U-control', 'U', 'U_short'),
}


def freeze(output):
    check_authorized(False)
    deadline = float(os.environ['TASK4_SAVE_DEADLINE_UNIX'])
    if not (time.time() < deadline < float('inf')):
        raise PermissionError('Finite managed deadline required')
    destination = project_path(output)
    if destination.exists():
        raise FileExistsError(destination)
    inputs = {}

    def digest(path):
        path = Path(path)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                if time.time() >= deadline - 5:
                    raise TimeoutError('Save deadline')
                h.update(block)
        return h.hexdigest()

    def capture(path):
        path = Path(path)
        inputs[str(path)] = digest(path)
        return inputs[str(path)]

    capture(Path(__file__).resolve())
    project_file = ROOT / 'configs/project.json'
    capture(project_file)
    project = read_json(project_file)
    lineage_file = ROOT / 'reports/T4_0001_run_lineage.json'
    capture(lineage_file)
    lineage = read_json(lineage_file)
    baseline = source_file(source_root(project), project['baseline'])
    report = {'protocol': 'T4_0001', 'evidence_role': 'development',
              'candidate_selection_performed': False, 'model_acceptance': False,
              'event_payload_read': False, 'test_payload_read': False,
              'fine_dt': .0025, 'coarse_dt': .005, 'history_ms': 100,
              'decision_horizon_s': 8., 'endpoints': {},
              'TRAIN_identity_audit': 'Common checkpoint-recorded TRAIN IDs; relies on existing loader split audit, no independent split reread',
              'physical_baseline_role': 'legacy physical warm start; not independent new-group confirmation'}
    report['endpoints']['M3'] = {'kind': 'fixed_M3', 'source': str(baseline),
        'source_sha256': capture(baseline), 'state_dimension': 30,
        'deployment_neural_parameter_slots': 0, 'training_scales_source': 'TRAIN'}
    reference_physics, reference_scales, reference_ids = None, None, None
    for label, (directory, kind, arm, short) in ENDPOINTS.items():
        folder = ROOT / 'artifacts/T4_0001' / directory
        checkpoint = folder / 'last.pt'
        source_hash = capture(checkpoint)
        capture(folder / 'training_summary.json')
        summary = read_json(folder / 'training_summary.json')
        model, encoder, head, obj = load_checkpoint(checkpoint)
        cfg = obj['config']
        expected_name = 'T4_0001_' + ('P_long' if label == 'P-N' else directory)
        if (obj['kind'] != kind or cfg['arm'] != arm or obj['step'] != 400
                or cfg['steps'] != 400 or obj['synthetic'] is not False
                or not obj['exact_resume_eligible'] or model.dim != 50 or model.rank != 2
                or cfg['hidden'] != 32 or cfg['dtype'] != 'float64' or cfg['name'] != expected_name
                or summary['requested_total_steps'] != 400
                or summary['final_single_start_phase_completed'] is not True
                or summary['stop_reason'] != 'steps_completed'
                or summary['test_payload_read'] is not False
                or summary['future_input_at_deployment'] is not False):
            raise ValueError('Incomplete/wrong registered endpoint: ' + label)
        for section in ('model', 'encoder', 'head'):
            if any(t.is_floating_point() and t.dtype != torch.float64 for t in obj[section].values()):
                raise ValueError('Actual saved tensor precision differs from float64: ' + label)
        for name in ('mass', 'incidence', 'delta_star', 'pm_star', 'coupling', 'damping', 'rate', 'gain'):
            if not torch.equal(getattr(model, name), torch.as_tensor(obj['physics'][name], dtype=torch.float64)):
                raise ValueError('Fixed physical buffer differs from checkpoint metadata: ' + label + '/' + name)
        if not torch.equal(model.scales, torch.as_tensor(obj['scales'], dtype=torch.float64)):
            raise ValueError('Actual scale buffer differs from checkpoint metadata: ' + label)
        for name in ('mass', 'adjacency', 'scales', 'mu'):
            if not torch.equal(getattr(encoder, name), getattr(model, name)):
                raise ValueError('Initializer physical/scale buffers disagree: ' + label + '/' + name)
        if label == 'A' and cfg['anchor_weight'] != 0:
            raise ValueError('A is not the registered no-anchor arm')
        if label == 'U' and cfg['u_hidden'] != 64:
            raise ValueError('U vector network width differs from registration')
        if label == 'P-S':
            weights = ('anchor_weight', 'latent_weight', 'torque_weight', 'continuity_weight',
                       'teacher_field_weight', 'teacher_gradient_weight')
            if (not cfg.get('autonomous_only') or any(cfg.get(k, 0) != 0 for k in weights)
                    or cfg['shooting_fraction'] != 0
                    or cfg['horizons'] != [5.0] or cfg['final_autonomous_fraction'] != 1.0
                    or summary['declared_final_autonomous_horizon_seconds'] != 5.0
                    or summary['autonomous_only'] is not True
                    or summary['prediction_head_used_or_optimized'] is not False
                    or not all(isinstance(fn, PolynomialFunction) for fn in model.function_modules().values())):
                raise ValueError('S is not the completed pure autonomous symbolic endpoint')
        ids = sorted(obj['event_ids'])
        if len(ids) != 346 or len(set(ids)) != 346:
            raise ValueError('Registered TRAIN identity mismatch')
        if reference_physics is None:
            reference_physics, reference_scales, reference_ids = obj['physics'], obj['scales'], ids
        elif obj['physics'] != reference_physics or obj['scales'] != reference_scales or ids != reference_ids:
            raise ValueError('Endpoint physics/scales/TRAIN cohort mismatch')
        modules = ((model.node_rhs, model.edge_rhs) if label == 'U'
                   else tuple(model.function_modules().values()) if label != 'P-S' else ())
        deployed_nn = sum(p.numel() for module in modules for p in module.parameters())
        encoder_slots = sum(p.numel() for p in encoder.parameters())
        row = {'checkpoint': str(checkpoint), 'checkpoint_sha256': source_hash,
               'kind': kind, 'step': obj['step'], 'config': cfg, 'completion_summary': summary,
               'state_dimension': model.dim, 'deployed_function_neural_parameter_slots': deployed_nn,
               'initializer_neural_parameter_slots': encoder_slots,
               'deployment_neural_parameter_slots': deployed_nn + encoder_slots,
               'stored_dynamics_parameter_slots': sum(p.numel() for p in model.parameters()),
               'auxiliary_head_parameter_slots': sum(p.numel() for p in head.parameters()),
               'learned_coupling_parameter_slots': model.log_coupling.numel(),
               'inactive_inherited_structural_nn_slots': (sum(p.numel() for fn in model.function_modules().values()
                                                               for p in fn.parameters()) if label == 'U' else 0)}
        row['managed_run_lineage'] = lineage[label]
        if not row['managed_run_lineage']:
            raise ValueError('Missing managed run lineage: ' + label)
        if short:
            short_folder = ROOT / 'artifacts/T4_0001' / short
            capture(short_folder / 'last.pt')
            capture(short_folder / 'training_summary.json')
            preceding = read_json(short_folder / 'training_summary.json')
            preceding_cp = torch.load(short_folder / 'last.pt', map_location='cpu', weights_only=True)
            if (preceding_cp['step'] != 200 or preceding_cp['config']['arm'] != arm
                    or preceding_cp['kind'] != ('U-control' if label == 'U' else 'T4-N')
                    or preceding_cp['synthetic'] is not False or not preceding_cp['exact_resume_eligible']
                    or preceding['requested_total_steps'] != 200
                    or preceding['final_single_start_phase_completed'] is not True
                    or preceding['stop_reason'] != 'steps_completed'
                    or preceding_cp['physics'] != obj['physics'] or preceding_cp['scales'] != obj['scales']
                    or sorted(preceding_cp['event_ids']) != ids):
                raise ValueError('Missing completed short200 predecessor: ' + label)
            row['short200_predecessor'] = {'directory': str(short_folder), 'summary': preceding,
                                         'checkpoint_sha256': inputs[str(short_folder / 'last.pt')],
                                         'managed_run_lineage': lineage[short]}
        report['endpoints'][label] = row
        del model, encoder, head, obj
    report['source_run_archives'] = []
    for entry in read_json(ROOT / 'state/research.json')['history']:
        if entry.get('kind') == 'PROCESS_CLOSED':
            archive = ROOT / 'logs' / entry['run_id'] / 'source.zip'
            row = {'run_id': entry['run_id'], 'source_zip': str(archive) if archive.is_file() else None,
                   'sha256': capture(archive) if archive.is_file() else None,
                   'accounted_seconds': entry['seconds'], 'exit_code': entry['exit_code'],
                   'exact_per_run_source_archive_available': archive.is_file()}
            process_log = archive.parent / 'process.log'
            if process_log.is_file():
                row.update(process_log=str(process_log), process_log_sha256=capture(process_log))
            if not archive.is_file():
                anchor = ROOT / 'artifacts/code_snapshots/T4_0001_START.zip'
                row.update(source_provenance_limitation='Historical per-run archive absent; nearest recorded project anchor is not asserted to be exact run source',
                           nearest_recorded_anchor=str(anchor), nearest_recorded_anchor_sha256=capture(anchor))
            report['source_run_archives'].append(row)
    report['training_scales'] = reference_scales
    report['reference_physics'] = reference_physics
    report['TRAIN_ids'] = reference_ids
    known_runs = {item['run_id'] for item in report['source_run_archives']}
    if any(run not in known_runs for row in lineage.values() for run in row):
        raise ValueError('Lineage names an unfinished or unknown managed run')
    report['lineage_source_archive_gaps'] = [item['run_id'] for item in report['source_run_archives']
        if item['run_id'] in {run for row in lineage.values() for run in row}
        and not item['exact_per_run_source_archive_available']]
    if time.time() >= deadline - 5 or any(digest(Path(p)) != h for p, h in inputs.items()):
        raise RuntimeError('Deadline or frozen source changed')
    report['input_sha256'] = inputs
    report['complete'] = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze(args.output)
