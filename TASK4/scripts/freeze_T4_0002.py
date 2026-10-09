"""Metadata-only freeze of registered N800 -> TRAIN C -> final autonomous S.

All paths and warm/resume segments are explicit. No event loader, rollout,
development selection, or automatic acceptance is called. Historical archive
gaps are retained from the first-loop manifest, never reconstructed as exact.
"""
import _bootstrap  # noqa: F401

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import torch

from task4.paths import ROOT, project_path, source_root, source_file
from task4.training import check_authorized, load_checkpoint
from task4.symbolic import PolynomialFunction
from task4.symbolic_runtime import FUNCTION_NAMES


OLD_N400_SHA256 = 'c217b86ce4c8af0220c9ba44fb0e90aca76142b9a58d9c5083573b559a53a356'
PHYSICS_KEYS = ('mass', 'incidence', 'delta_star', 'pm_star', 'coupling', 'damping', 'rate', 'gain')
AUX_WEIGHTS = ('anchor_weight', 'latent_weight', 'torque_weight', 'continuity_weight',
               'teacher_field_weight', 'teacher_gradient_weight')
DERIVED_CONFIG_KEYS = {'_event_ids', '_optimizer_blocks', 'torque_scale'}
CODE_PATHS = [Path(__file__).resolve(), *[ROOT / 'src/task4' / (name + '.py')
              for name in ('training', 'dynamics', 'encoder', 'symbolic', 'symbolic_runtime', 'paths')]]


def strict_json(content):
    def invalid(value):
        raise ValueError('Nonfinite JSON input: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    value = json.loads(content.decode('utf-8-sig'), parse_constant=invalid, object_pairs_hook=unique)
    # Also reject numeric literals such as 1e999, which parse_constant misses.
    json.dumps(value, allow_nan=False)
    return value


class Inputs:
    def __init__(self, deadline):
        self.deadline = deadline
        self.hashes = {}

    def check(self):
        if time.time() >= self.deadline - 5:
            raise TimeoutError('Managed freeze save deadline reached')

    def digest(self, path):
        self.check()
        result = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                self.check()
                result.update(block)
        return result.hexdigest()

    def capture(self, path):
        path = Path(path).resolve()
        value = self.digest(path)
        if str(path) in self.hashes and self.hashes[str(path)] != value:
            raise ValueError('Frozen input changed: ' + str(path))
        self.hashes[str(path)] = value
        return value

    def read(self, path):
        path = Path(path).resolve()
        self.capture(path)
        self.check()
        value = strict_json(path.read_bytes())
        self.capture(path)
        return value

    def end_guard(self):
        self.check()
        for path, expected in self.hashes.items():
            if self.digest(path) != expected:
                raise ValueError('Frozen input changed: ' + path)
        self.check()


def raw_checkpoint(path, inputs):
    """Inspect serialized scientific tensors before reconstruction can cast."""
    inputs.capture(path)
    inputs.check()
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if type(obj.get('step')) is not int or obj['step'] < 0:
        raise ValueError('Saved checkpoint step must be a nonnegative integer')
    for section in ('model', 'encoder', 'head', 'shooting'):
        if section not in obj or not isinstance(obj[section], dict):
            raise ValueError('Missing raw checkpoint state: ' + section)
        for name, tensor in obj[section].items():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu':
                raise ValueError('Expected CPU scientific tensor: ' + section + '/' + name)
            expected_dtype = (torch.int64 if section == 'model' and (name == 'edges' or name.endswith('.powers'))
                              else torch.bool if section == 'model' and name.endswith('.support') else torch.float64)
            if tensor.dtype != expected_dtype or (tensor.is_floating_point() and not torch.isfinite(tensor).all()):
                raise ValueError('Nonfinite/non-double raw scientific tensor: ' + section + '/' + name)
    if obj.get('synthetic') is not False or obj['config'].get('synthetic') is not False:
        raise ValueError('A real registered checkpoint is required')
    for value in (obj['config'], obj['physics'], obj['scales'], obj['event_ids']):
        json.dumps(value, allow_nan=False)
    inputs.capture(path)
    return obj


def autonomous_config(cfg, label):
    expected_steps = 800 if label == 'N' else cfg.get('steps')
    if (type(expected_steps) is not int or not 600 <= expected_steps <= 800
            or cfg.get('steps') != expected_steps or cfg.get('arm') != 'P'
            or cfg.get('synthetic') is not False or cfg.get('autonomous_only') is not True
            or cfg.get('hidden') != 32 or cfg.get('latent_per_machine') != 2
            or cfg.get('dtype') != 'float64' or cfg.get('device') != 'cpu'
            or any(cfg.get(key, 0) != 0 for key in AUX_WEIGHTS)
            or cfg.get('teacher_checkpoint') or cfg.get('initialization_checkpoint')
            or cfg.get('shooting_fraction') != 0 or cfg.get('horizons') != [5.0]
            or cfg.get('final_autonomous_fraction') != 1.0
            or not str(cfg.get('name', '')).startswith('T4_0002_')
            or 'cost' in str(cfg.get('name', '')).lower()):
        raise ValueError('Wrong registered full-autonomous ' + label + ' configuration')
    if DERIVED_CONFIG_KEYS & set(cfg):
        raise ValueError('Formal configuration cannot supply derived TRAIN metadata')


def compare_config(actual, formal):
    wanted = {**formal, 'quadratic_functions': formal.get('quadratic_functions', 'off')}
    if {key: value for key, value in actual.items() if key not in DERIVED_CONFIG_KEYS} != wanted:
        raise ValueError('Checkpoint differs from complete locked formal configuration')


def compare_identity(obj, reference):
    if (obj['physics'] != reference['physics'] or obj['scales'] != reference['scales']
            or obj['event_ids'] != reference['event_ids']
            or obj['config'].get('_event_ids') != reference['event_ids']):
        raise ValueError('Checkpoint physics/scales/ordered TRAIN identity mismatch')
    for key in ('mu', 'phase', 'edges', 'adjacency'):
        if key not in obj['model'] or not torch.equal(obj['model'][key], reference['model'][key]):
            raise ValueError('Fixed physical derived buffer differs from N400: model/' + key)
    for key in ('mass', 'adjacency', 'scales', 'mu'):
        if key not in obj['encoder'] or not torch.equal(obj['encoder'][key], reference['encoder'][key]):
            raise ValueError('Fixed initializer buffer differs from N400: encoder/' + key)


def persistent_polynomials(obj):
    specs = obj.get('symbolic_functions', {})
    if obj.get('kind') != 'T4-S' or set(specs) != set(FUNCTION_NAMES):
        raise ValueError('Seven explicit polynomial functions are required')
    support_slots, active_slots = {}, {}
    dimensions = dict(zip(FUNCTION_NAMES, ((7, 1), (16, 1), (8, 6), (8, 4), (18, 16), (18, 4), (8, 2))))
    for name in FUNCTION_NAMES:
        prefix = name + '.'
        state = obj['model']
        if any(prefix + key not in state for key in ('powers', 'coefficients', 'support')):
            raise ValueError('Missing persistent polynomial support: ' + name)
        powers, coefficients, support = [state[prefix + key] for key in ('powers', 'coefficients', 'support')]
        spec = specs[name]
        if (powers.dtype != torch.int64 or support.dtype != torch.bool
                or powers.ndim != 2 or coefficients.ndim != 2
                or support.shape != coefficients.shape or powers.shape[0] != coefficients.shape[0]
                or torch.any(powers < 0) or torch.any(powers > 3) or torch.any(powers.sum(dim=1) > 3)
                or torch.any(support.sum(dim=0) > 16) or spec.get('type') != 'polynomial'
                or (powers.shape[1], coefficients.shape[1]) != dimensions[name]
                or spec.get('inputs') != powers.shape[1] or spec.get('outputs') != coefficients.shape[1]
                or powers.tolist() != spec['powers'] or (coefficients * support).tolist() != spec['coefficients']):
            raise ValueError('Polynomial export/persistent support mismatch: ' + name)
        support_slots[name] = int(support.sum())
        active_slots[name] = int(torch.count_nonzero(coefficients * support))
    return {'frozen_training_support_slots_by_function': support_slots,
            'active_nonzero_support_slots_by_function': active_slots,
            'support_source': 'checkpoint persistent bool mask; not inferred from exported nonzeros'}


def symbolic_json(path, obj, inputs):
    """Check the independently executable export against actual saved state."""
    spec = inputs.read(path)
    enc = spec.get('encoder', {})
    coordinates = spec.get('coordinates', {})
    if (spec.get('format') != 'task4-polynomial-powergraph' or spec.get('version') != 1
            or spec.get('n') != 10 or spec.get('rank') != 2
            or spec.get('physics') != {key: obj['physics'][key] for key in PHYSICS_KEYS}
            or spec.get('omega_s') != obj['physics']['omega_s']
            or spec.get('scales') != obj['scales'] or spec.get('functions') != obj['symbolic_functions']
            or spec.get('strength') != obj['config'].get('model_strength', .01)
            or enc.get('type') != 'shared_gru' or enc.get('hidden_size') != 32
            or enc.get('history_samples') != 11 or enc.get('history_seconds') != .1
            or enc.get('sample_dt_seconds') != .01 or enc.get('gate_order') != ['reset', 'update', 'new']
            or enc.get('new_gate_equation') != 'tanh(input_new + reset * hidden_new)'
            or enc.get('centering') != 'raw(history) - raw(zeros_like(history)); mu retained in both'
            or enc.get('deployment_use') != 'once at t=0.1 s; no future observations or latent resets'
            or coordinates.get('state_order') != ['delta[N]', 'pi[N]', 'p[N]', 'q[N,rank] flattened machine-major']
            or coordinates.get('history_order') != ['delta deviation', 'omega deviation', 'p deviation']
            or coordinates.get('time_unit') != 's' or coordinates.get('delta_unit') != 'rad'
            or coordinates.get('omega_unit') != 'rad/s' or coordinates.get('p_unit') != 'system per-unit'
            or coordinates.get('pi_definition') != 'mass * omega'):
        raise ValueError('Independent symbolic JSON differs from registered checkpoint/initialization')
    for key in ('log_coupling', 'mu', 'phase', 'edges'):
        if spec.get(key) != obj['model'][key].tolist():
            raise ValueError('Symbolic JSON scientific state differs: ' + key)
    for key in ('mass', 'adjacency', 'scales', 'mu'):
        if enc.get(key) != obj['encoder'][key].tolist():
            raise ValueError('Symbolic JSON initializer buffer differs: ' + key)
    for key in ('weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'):
        if enc.get(key) != obj['encoder']['gru.' + key].tolist():
            raise ValueError('Symbolic JSON GRU weight differs: ' + key)
    for key in ('weight', 'bias'):
        if enc.get('head_' + key) != obj['encoder']['head.' + key].tolist():
            raise ValueError('Symbolic JSON initializer projection differs: ' + key)
    return {'path': str(path), 'sha256': inputs.hashes[str(path.resolve())],
            'checkpoint_and_initializer_exactly_matched': True}


def m3_physics(path):
    """Read only the known fixed-baseline physical arrays, never an event."""
    with np.load(path, allow_pickle=False) as asset:
        return {key: np.asarray(asset[key], dtype=np.float64).tolist() for key in PHYSICS_KEYS}


def launch_receipt(path, segment, stage, config_path, inputs):
    """Bind argv using an explicitly trusted root governance receipt.

    This checks a root-captured live-job record. It does not authenticate an
    independent third party or pretend its later capture happened at launch.
    """
    receipt = inputs.read(path)
    command = receipt.get('command')
    if (type(receipt.get('schema_version')) is not int or receipt.get('schema_version') != 1
            or receipt.get('run_id') != segment['run_id']
            or receipt.get('loop_id') != 'T4_0002' or receipt.get('capture_origin') != 'live_active_job_state'
            or type(receipt.get('pid')) is not int or receipt['pid'] <= 0
            or type(receipt.get('project_root')) is not str or not receipt['project_root']
            or Path(receipt['project_root']).resolve() != ROOT.resolve()
            or not isinstance(command, list) or len(command) < 2
            or any(type(part) is not str or not part for part in command)):
        raise ValueError('Invalid trusted live-launch receipt')
    try:
        started, captured = [datetime.fromisoformat(receipt[key]) for key in ('started_at', 'captured_at')]
        if started.tzinfo is None or captured.tzinfo is None or captured < started:
            raise ValueError('Launch receipt capture precedes launch or lacks time zone')
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid launch/capture timestamp') from exc
    wanted_script = ROOT / 'scripts' / ('compress_model.py' if stage == 'C' else 'train.py')
    if (not re.fullmatch(r'python(?:3(?:\.\d+)?)?(?:\.exe)?', Path(command[0]).name.lower())
            or project_path(command[1]) != wanted_script.resolve()):
        raise ValueError('Receipt launched the wrong registered stage script')
    options = {}
    index = 2
    while index < len(command):
        option = command[index]
        if not option.startswith('--'):
            raise ValueError('Unexpected launch argv positional argument')
        if '=' in option:
            option, value = option.split('=', 1)
            index += 1
        else:
            if index + 1 >= len(command):
                raise ValueError('Launch argv option has no value')
            value = command[index + 1]
            index += 2
        if option in options or not value:
            raise ValueError('Duplicate/empty launch argv option')
        options[option] = value
    common = {'--output': str(project_path(segment['output']))}
    if stage == 'C':
        permitted = {'--checkpoint', '--output', '--adapt-steps', '--max-samples', '--seed', '--device'}
        required = {'--checkpoint', '--output', '--adapt-steps', '--max-samples', '--seed'}
        if not required <= set(options) or not set(options) <= permitted:
            raise ValueError('C launch must explicitly lock its registered compression options')
        if (options['--adapt-steps'] != '16' or options['--max-samples'] != '2048'
                or options['--seed'] != '419' or options.get('--device', 'cpu') != 'cpu'
                or receipt.get('formal_config_sha256') is not None):
            raise ValueError('C launch parameters differ from seed419/adapt16/max2048/CPU')
        common['--checkpoint'] = str(project_path(segment['source_checkpoint']))
    else:
        mode = segment.get('mode')
        if mode not in ('warm_start', 'resume'):
            raise ValueError('N/S launch needs an explicit warm_start/resume mode')
        source_option = '--warm-start' if mode == 'warm_start' else '--resume'
        if set(options) != {'--config', '--output', source_option}:
            raise ValueError('N/S launch argv overrides or differs from the locked stage')
        if receipt.get('formal_config_sha256') != inputs.capture(config_path):
            raise ValueError('Live-launch receipt formal configuration hash differs')
        common['--config'] = str(config_path)
        common[source_option] = str(project_path(segment['source_checkpoint']))
    for option, expected in common.items():
        if project_path(options[option]) != Path(expected):
            raise ValueError('Launch argv differs from registered config/output/source/mode: ' + option)
    return {'path': str(path), 'sha256': inputs.hashes[str(path.resolve())], 'command': command,
            'pid': receipt['pid'], 'started_at': receipt['started_at'], 'captured_at': receipt['captured_at'],
            'source_scope': 'Trusted root-captured live managed job; not independent third-party attestation'}


def archived_source(run_id, config_path, inputs, stage):
    folder = ROOT / 'logs' / run_id
    archive_path, index_path = folder / 'source.zip', folder / 'source_manifest.json'
    sha = inputs.capture(archive_path)
    index = inputs.read(index_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(index):
            raise ValueError('Incomplete/duplicate managed source archive')
        script = 'scripts/compress_model.py' if stage == 'C' else 'scripts/train.py'
        if script not in index:
            raise ValueError('Managed launch script is absent from its source archive')
        for name in names:
            inputs.check()
            if hashlib.sha256(archive.read(name)).hexdigest() != index[name]:
                raise ValueError('Managed source archive/index mismatch: ' + name)
        if config_path is not None:
            relative = config_path.relative_to(ROOT).as_posix()
            if relative not in index or index[relative] != inputs.capture(config_path):
                raise ValueError('Locked formal configuration differs from managed launch source')
    inputs.capture(archive_path)
    return {'run_id': run_id, 'source_zip': str(archive_path), 'sha256': sha,
            'source_manifest': str(index_path), 'source_manifest_sha256': inputs.hashes[str(index_path.resolve())],
            'exact_per_run_source_archive_available': True}


def completed_summary(summary, cfg, step, start, final):
    if (summary.get('synthetic') is not False or summary.get('requested_total_steps') != cfg['steps']
            or summary.get('steps_completed') != step - start
            or summary.get('test_payload_read') is not False or summary.get('future_input_at_deployment') is not False
            or summary.get('autonomous_only') is not True
            or summary.get('declared_final_autonomous_horizon_seconds') != 5.0
            or summary.get('prediction_head_used_or_optimized') is not False
            or any(summary.get(key, False) is not False for key in
                   ('prediction_head_used', 'prediction_head_deployed', 'prediction_head_optimized', 'prediction_head_in_optimizer'))):
        raise ValueError('Autonomous completion summary disagrees with retained checkpoint')
    if final and (step != cfg['steps'] or summary.get('final_single_start_phase_completed') is not True
                  or summary.get('stop_reason') != 'steps_completed'):
        raise ValueError('Incomplete endpoint cannot receive final freeze qualification')


def freeze(output, *, previous_manifest, n_config, s_config, lineage):
    check_authorized(False)  # Must precede every business input read/write.
    deadline = float(os.environ.get('TASK4_SAVE_DEADLINE_UNIX', 'nan'))
    if not math.isfinite(deadline):
        raise PermissionError('Finite managed deadline required')
    inputs = Inputs(deadline)
    inputs.check()
    destination = project_path(output)
    if destination.exists():
        raise FileExistsError(destination)
    for path in CODE_PATHS:
        inputs.capture(path)
    project_pathname = ROOT / 'configs/project.json'
    project = inputs.read(project_pathname)
    if project.get('history_ms') != 100 or project.get('sample_dt') != .01:
        raise ValueError('The original 100 ms observation/time-grid contract changed')
    previous_path = project_path(previous_manifest)
    previous = inputs.read(previous_path)
    if (previous.get('protocol') != 'T4_0001' or previous.get('complete') is not True
            or previous.get('event_payload_read') is not False or previous.get('test_payload_read') is not False
            or previous.get('fine_dt') != .0025 or previous.get('coarse_dt') != .005
            or previous.get('history_ms') != 100 or previous.get('decision_horizon_s') != 8.):
        raise ValueError('An unchanged valid first-loop frozen manifest is required')
    old_row = previous['endpoints']['P-N']
    old_path = project_path(old_row['checkpoint'])
    if old_row.get('checkpoint_sha256') != OLD_N400_SHA256 or inputs.capture(old_path) != OLD_N400_SHA256:
        raise ValueError('The fixed old N400 warm source identity differs')
    reference = raw_checkpoint(old_path, inputs)
    ids = reference['event_ids']
    if (reference['kind'] != 'T4-N' or reference['step'] != 400 or reference['config']['steps'] != 400
            or reference.get('exact_resume_eligible') is not True
            or len(ids) != 346 or len(set(ids)) != 346
            or reference['config'].get('_event_ids') != ids
            or old_row['config'].get('_event_ids') != ids
            or reference['physics'] != previous['reference_physics']
            or reference['scales'] != previous['training_scales'] or sorted(ids) != previous['TRAIN_ids']):
        raise ValueError('Old N400/TRAIN/physics/scales identity mismatch')
    baseline = source_file(source_root(project), project['baseline']).resolve()
    m3 = previous['endpoints']['M3']
    if str(baseline) != str(Path(m3['source']).resolve()) or inputs.capture(baseline) != m3['source_sha256']:
        raise ValueError('Fixed legacy M3 asset identity changed')
    if m3_physics(baseline) != {key: reference['physics'][key] for key in PHYSICS_KEYS}:
        raise ValueError('Fixed M3 physical metadata differs')
    if len(reference['physics']['mass']) != 10:
        raise ValueError('Registered system must retain ten machines')
    config_paths = {'N': project_path(n_config), 'S': project_path(s_config)}
    configs = {label: inputs.read(path) for label, path in config_paths.items()}
    for label, cfg in configs.items():
        autonomous_config(cfg, label)
    lineage_path = project_path(lineage)
    registered = inputs.read(lineage_path)
    stages = registered.get('stages', {})
    if registered.get('protocol') != 'T4_0002' or set(stages) != {'N', 'C', 'S'}:
        raise ValueError('Explicit second-loop N/C/S lineage is required')
    state = inputs.read(ROOT / 'state/research.json')
    closed = {row['run_id']: row for row in state['history'] if row.get('kind') == 'PROCESS_CLOSED'}
    archives, audited, endpoints, used_runs = [], {}, {}, set()
    last_path, last_obj = old_path, reference
    for stage in ('N', 'C', 'S'):
        segments = stages[stage]
        if not isinstance(segments, list) or not segments:
            raise ValueError('Missing registered stage segments: ' + stage)
        audited[stage] = []
        stage_teacher = last_path
        for index, segment in enumerate(segments):
            inputs.check()
            run_id = segment['run_id']
            if not re.fullmatch(r'run_[0-9a-f]{12}', run_id) or run_id not in closed or run_id in used_runs:
                raise ValueError('Unknown/active/duplicate managed run lineage')
            used_runs.add(run_id)
            result_path = project_path(segment['run_result'])
            result = inputs.read(result_path)
            ended = closed[run_id]
            if (result.get('run_id') != run_id or result.get('exit_code') != ended['exit_code']
                    or result.get('reason') != ended['reason'] or result.get('seconds') != ended['seconds']
                    or result.get('log') != str(ROOT / 'logs' / run_id / 'process.log')
                    or type(result.get('exit_code')) is not int
                    or type(result.get('seconds')) not in (int, float) or result['seconds'] < 0):
                raise ValueError('Run result differs from closed managed accounting')
            final = index == len(segments) - 1
            if final and (result['exit_code'] != 0 or result['reason'] != 'exit'):
                raise ValueError('Final stage run did not complete normally')
            folder = project_path(segment['output'])
            cp = folder / 'last.pt'
            source = project_path(segment['source_checkpoint'])
            expected_source = stage_teacher if stage == 'C' else last_path
            if source != expected_source:
                raise ValueError('Warm/resume source breaks registered N -> C -> D lineage')
            inputs.capture(source)
            config_path = None if stage == 'C' else config_paths[stage]
            if config_path is not None and project_path(segment['config']) != config_path:
                raise ValueError('Segment must name the complete locked stage configuration')
            receipt = launch_receipt(project_path(segment['launch_receipt']), segment, stage, config_path, inputs)
            archived = archived_source(run_id, config_path, inputs, stage)
            archives.append({**archived, 'accounted_seconds': result['seconds'], 'exit_code': result['exit_code']})
            obj = raw_checkpoint(cp, inputs)
            compare_identity(obj, reference)
            if stage == 'C':
                summary_path = folder / 'compression_summary.json'
                summary = inputs.read(summary_path)
                support = persistent_polynomials(obj) if final else None
                if (final and (summary.get('complete') is not True or summary.get('fitted_functions') != 7
                               or summary.get('adapted_functions') != 7
                               or summary.get('stop_reason') != 'seven_functions_compressed_and_short_adapted'
                               or summary.get('final_checkpoint') != str(cp))
                        or summary.get('teacher_checkpoint') != str(stage_teacher)
                        or summary.get('test_payload_read') is not False or summary.get('validation_payload_read') is not False
                        or summary.get('final_single_initialization_5s_finetune_completed') is not False
                        or obj['config'].get('teacher_usage') != 'TRAIN_only_compression_adaptation'
                        or obj['config'].get('teacher_checkpoint') != str(stage_teacher)
                        or obj['config'].get('compression_source_autonomous_only') is not True
                        or obj['config'].get('autonomous_only') is not False
                        or any(obj['config'].get(key) != 0 for key in ('anchor_weight', 'latent_weight', 'torque_weight'))
                        or any(obj['config'].get(key) != .1 for key in ('teacher_field_weight', 'teacher_gradient_weight'))
                        or obj['config'].get('prediction_head_used_or_deployed') is not False
                        or obj['config'].get('seed') != configs['N'].get('seed')
                        or summary.get('seed') != 419 or summary.get('adapt_steps_per_function') != 16
                        or summary.get('max_samples_per_function') != 2048):
                    raise ValueError('Not the completed current-loop TRAIN compression source')
                export = None
                if final:
                    export_path = folder / 'model_symbolic.json'
                    if summary.get('symbolic_export') != str(export_path):
                        raise ValueError('C must identify its actual completed symbolic export')
                    export = symbolic_json(export_path, obj, inputs)
            else:
                cfg = configs[stage]
                compare_config(obj['config'], cfg)
                if (obj.get('exact_resume_eligible') is not True or obj['step'] > cfg['steps']
                        or obj.get('shooting_keys') or obj['shooting']):
                    raise ValueError('Incomplete/foreign checkpoint boundary')
                expected_kind = 'T4-N' if stage == 'N' else 'T4-S'
                if obj['kind'] != expected_kind or (stage == 'N' and obj.get('symbolic_functions')):
                    raise ValueError('Wrong registered endpoint kind')
                expected_mode = 'warm_start' if index == 0 else 'resume'
                expected_start = 0 if index == 0 else last_obj['step']
                if (segment.get('mode') != expected_mode or type(segment.get('start_step')) is not int
                        or segment.get('start_step') != expected_start):
                    raise ValueError('Warm/resume stage step accounting mismatch')
                if not expected_start <= obj['step'] <= cfg['steps']:
                    raise ValueError('Retained stage checkpoint step moved backwards')
                summary_path = folder / 'training_summary.json'
                summary = inputs.read(summary_path)
                completed_summary(summary, cfg, obj['step'], expected_start, final)
                run_cfg = inputs.read(folder / 'run_config.json')
                if run_cfg != obj['config']:
                    raise ValueError('Saved run configuration and checkpoint differ')
                support = persistent_polynomials(obj) if stage == 'S' else None
                if stage == 'S':
                    for name in FUNCTION_NAMES:
                        for key in ('powers', 'support'):
                            if not torch.equal(obj['model'][name + '.' + key], last_obj['model'][name + '.' + key]):
                                raise ValueError('Final truth fine-tuning changed compressed library/support')
                export = symbolic_json(project_path(segment['symbolic_json']), obj, inputs) if stage == 'S' and final else None
            audited[stage].append({**segment, 'checkpoint': str(cp), 'checkpoint_sha256': inputs.hashes[str(cp.resolve())],
                                   'completion_summary': str(summary_path),
                                   'completion_summary_sha256': inputs.hashes[str(summary_path.resolve())],
                                   'run_result_sha256': inputs.hashes[str(result_path.resolve())],
                                   'step': obj['step'], 'kind': obj['kind'], 'symbolic_json': export,
                                   'launch_receipt_audit': receipt})
            if final and stage in ('N', 'S'):
                inputs.check()
                model, encoder, head, loaded = load_checkpoint(cp, 'cpu')
                if (model.dim != 50 or model.rank != 2 or loaded['config'] != obj['config']
                        or loaded['physics'] != obj['physics'] or loaded['scales'] != obj['scales']
                        or loaded['event_ids'] != obj['event_ids'] or loaded['step'] != obj['step']):
                    raise ValueError('Reconstructed endpoint disagrees with audited raw checkpoint')
                for key in PHYSICS_KEYS:
                    if not torch.equal(getattr(model, key), torch.as_tensor(obj['physics'][key], dtype=torch.float64)):
                        raise ValueError('Saved physical buffer/metadata mismatch: ' + key)
                if not torch.equal(model.scales, torch.as_tensor(obj['scales'], dtype=torch.float64)):
                    raise ValueError('Saved scale buffer/metadata mismatch')
                for key in ('mass', 'adjacency', 'scales', 'mu'):
                    if not torch.equal(getattr(encoder, key), getattr(model, key)):
                        raise ValueError('Initializer physical/scale buffers differ: ' + key)
                if stage == 'S' and not all(type(fn) is PolynomialFunction for fn in model.function_modules().values()):
                    raise ValueError('Final S contains a neural/unrecognized function')
                label = 'P-N' if stage == 'N' else 'P-S'
                endpoints[label] = {'checkpoint': str(cp), 'checkpoint_sha256': inputs.hashes[str(cp.resolve())],
                                    'kind': obj['kind'], 'step': obj['step'], 'config': obj['config'],
                                    'completion_summary': summary, 'state_dimension': 50, 'quotient_dimension': 49,
                                    'managed_run_lineage': audited[stage], 'formal_config': str(config_path),
                                    'formal_config_sha256': inputs.hashes[str(config_path.resolve())]}
                if support is not None:
                    endpoints[label]['polynomial_support_audit'] = support
                    endpoints[label]['symbolic_json'] = export
                del model, encoder, head, loaded
            last_path, last_obj = cp, obj
    endpoints['M3'] = {**m3, 'physical_metadata_audited': True, 'state_dimension': 30,
                       'physical_baseline_role': 'same legacy asset; not independent new-group confirmation'}
    report = {'protocol': 'T4_0002', 'complete': True, 'evidence_role': 'development',
              'candidate_selection_performed': False, 'model_acceptance': False,
              'event_payload_read': False, 'validation_payload_read': False, 'test_payload_read': False,
              'fine_dt': .0025, 'coarse_dt': .005, 'history_ms': 100, 'decision_horizon_s': 8.,
              'state_dimension': 50, 'quotient_dimension': 49, 'endpoints': endpoints,
              'TRAIN_ids': ids, 'TRAIN_order_preserved': True,
              'TRAIN_identity_audit': 'Exact saved 346 IDs/order inherited from fixed N400; no event or split payload reread',
              'training_scales': reference['scales'], 'reference_physics': reference['physics'],
              'raw_scientific_precision_audit': 'CPU float64 model/encoder/head/shooting before reconstruction; int64 edges/powers and bool support retained; Adam scalar counters are outside scientific precision audit',
              'previous_manifest': str(previous_path), 'previous_manifest_sha256': inputs.hashes[str(previous_path.resolve())],
              'predecessor_N400': old_row,
              'update_accounting': {'N_this_stage': 800, 'N_prior_long_stage_total': 400,
                                    'N_cumulative_long_stage': 1200, 'S_this_stage': configs['S']['steps'],
                                    'historical_anchor_and_short_stages': 'Separate predecessor anchor/short stages; not included in this stage800 or reported long400+800=1200'},
              'managed_stage_lineage': audited, 'source_run_archives': archives,
              'launch_identity_audit_scope': 'Explicit trusted root receipts captured from live active jobs, not independent third-party provenance; actual later capture timestamps retained',
              'registered_compression': {'seed': 419, 'adapt_steps_per_function': 16, 'max_samples_per_function': 2048,
                                         'asset_config_seed_role': 'Inherited N source seed; not the compression RNG seed',
                                         'maximum_polynomial_degree': 3, 'maximum_persistent_support_per_output': 16},
              'historical_source_run_archives': previous.get('source_run_archives', []),
              'lineage_source_archive_gaps': previous.get('lineage_source_archive_gaps', []),
              'historical_archive_audit_scope': 'Retained SHA-anchored first-loop manifest metadata; historical archive paths are not followed or synthesized',
              'source_identity_unchanged': True, 'evidence_validity': 'valid_metadata_freeze',
              'input_sha256': inputs.hashes}
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    inputs.end_guard()
    destination.parent.mkdir(parents=True, exist_ok=True)
    inputs.check()
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    try:
        inputs.end_guard()
    except (OSError, ValueError, TimeoutError) as exc:
        report.update(complete=False, source_identity_unchanged=None if isinstance(exc, TimeoutError) else False,
                      evidence_validity='invalid_deadline' if isinstance(exc, TimeoutError) else 'invalid_source_changed')
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'previous-manifest', 'n-config', 's-config', 'lineage'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze(args.output, previous_manifest=args.previous_manifest, n_config=args.n_config,
           s_config=args.s_config, lineage=args.lineage)


if __name__ == '__main__':
    main()
