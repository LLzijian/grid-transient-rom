"""Constructed metadata/tensors only; checkpoint/M3/model loaders are stubs.

No event source, model field, integration, TRAIN/Dev/TEST payload, or real
checkpoint is accessed. The fixture creates all archive/JSON/text sources.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from task4.paths import ROOT


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


@pytest.fixture
def freeze_fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('freeze_second_loop_test', ROOT / 'scripts/freeze_T4_0002.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'project'
    root.mkdir()
    monkeypatch.setattr(module, 'ROOT', root)
    def path(value):
        value = Path(value)
        resolved = value.resolve() if value.is_absolute() else (root / value).resolve()
        assert resolved.is_relative_to(root), 'TASK4 inputs must be explicit constructed paths'
        return resolved
    monkeypatch.setattr(module, 'project_path', path)
    calls, cp_objects, folders = [], {}, {}
    authorized = {'allowed': True}
    def check_authorized(synthetic):
        calls.append(('authorized', synthetic))
        if not authorized['allowed']:
            raise PermissionError('constructed unauthorized run')
    monkeypatch.setattr(module, 'check_authorized', check_authorized)
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', '1000000')
    monkeypatch.setattr(module.time, 'time', lambda: 100.)
    code = root / 'checker-source.py'
    code.write_text('constructed checker provenance bytes', encoding='utf-8')
    monkeypatch.setattr(module, 'CODE_PATHS', [code])
    for script_name in ('train.py', 'compress_model.py'):
        script = root / 'scripts' / script_name
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text('constructed launch source: ' + script_name, encoding='utf-8')
    ids = [f'constructed_TRAIN_{index:03d}' for index in reversed(range(346))]
    physics = {'mass': [1.] * 10, 'incidence': [[1.], [-1.]] + [[0.]] * 8,
               'delta_star': [0.] * 10, 'pm_star': [0.] * 10,
               'coupling': [1.], 'damping': [1.] * 10, 'rate': [1.] * 10,
               'gain': [1.] * 10, 'omega_s': 377.}
    scales = [.47675314588683515, 1.0033667249602924, .1]
    base = {'arm': 'P', 'seed': 401, 'synthetic': False, 'autonomous_only': True, 'hidden': 32,
            'latent_per_machine': 2, 'dtype': 'float64', 'device': 'cpu',
            'anchor_weight': 0., 'latent_weight': 0., 'torque_weight': 0.,
            'continuity_weight': 0., 'teacher_field_weight': 0., 'teacher_gradient_weight': 0.,
            'shooting_fraction': 0., 'horizons': [5.], 'final_autonomous_fraction': 1.,
            'quadratic_functions': 'off'}
    n_cfg = {**base, 'name': 'T4_0002_N_autonomous', 'steps': 800}
    s_cfg = {**base, 'name': 'T4_0002_S_final_D', 'steps': 600}
    n_path, s_path = root / 'configs/n-locked.json', root / 'configs/s-locked.json'
    save(n_path, n_cfg); save(s_path, s_cfg)
    def make_cp(cfg, kind, step, symbolic=False):
        actual = {**cfg, '_event_ids': list(ids), '_optimizer_blocks': True, 'torque_scale': 1.}
        state = {key: torch.tensor(value, dtype=torch.float64) for key, value in physics.items() if key != 'omega_s'}
        state.update(scales=torch.tensor(scales, dtype=torch.float64), mu=torch.zeros(10, 4, dtype=torch.float64),
                     adjacency=torch.zeros(10, 10, dtype=torch.float64), phase=torch.zeros(1, dtype=torch.float64),
                     edges=torch.tensor([[0, 1]], dtype=torch.int64), log_coupling=torch.zeros(1, dtype=torch.float64))
        encoder = {key: state[key].clone() for key in ('mass', 'adjacency', 'scales', 'mu')}
        encoder.update({'gru.weight_ih_l0': torch.zeros(96, 9, dtype=torch.float64),
                        'gru.weight_hh_l0': torch.zeros(96, 32, dtype=torch.float64),
                        'gru.bias_ih_l0': torch.zeros(96, dtype=torch.float64),
                        'gru.bias_hh_l0': torch.zeros(96, dtype=torch.float64),
                        'head.weight': torch.zeros(2, 32, dtype=torch.float64),
                        'head.bias': torch.zeros(2, dtype=torch.float64)})
        obj = {'kind': kind, 'step': step, 'config': actual, 'synthetic': False,
               'exact_resume_eligible': True, 'model': state, 'encoder': encoder,
               'head': {'constructed.weight': torch.zeros(1, dtype=torch.float64)},
               'shooting': {}, 'shooting_keys': [], 'physics': copy.deepcopy(physics),
               'scales': list(scales), 'event_ids': list(ids)}
        if symbolic:
            functions = {}
            for name, (inputs, outputs) in zip(module.FUNCTION_NAMES,
                    ((7, 1), (16, 1), (8, 6), (8, 4), (18, 16), (18, 4), (8, 2))):
                powers = torch.zeros(2, inputs, dtype=torch.int64)
                powers[1, 0] = 1
                coefficients = torch.zeros(2, outputs, dtype=torch.float64)
                coefficients[0] = 1.
                support = torch.ones(2, outputs, dtype=torch.bool)
                state.update({name + '.powers': powers, name + '.coefficients': coefficients,
                              name + '.support': support})
                functions[name] = {'type': 'polynomial', 'powers': powers.tolist(),
                                   'coefficients': coefficients.tolist(), 'inputs': inputs,
                                   'outputs': outputs, 'nonzero_per_output': [1] * outputs}
            obj['symbolic_functions'] = functions
        return obj
    def add_cp(folder, obj):
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / 'last.pt'
        checkpoint.write_text('constructed source identity: ' + folder.name, encoding='utf-8')
        cp_objects[checkpoint] = obj
        return checkpoint
    old_cfg = {**base, 'name': 'T4_0001_P_long', 'steps': 400, 'autonomous_only': False}
    old = make_cp(old_cfg, 'T4-N', 400)
    old['shooting'] = {'states.historical': torch.ones(1, dtype=torch.float64)}
    old['shooting_keys'] = ['historical']
    old_cp = add_cp(root / 'artifacts/old_N400', old)
    monkeypatch.setattr(module, 'OLD_N400_SHA256', hashlib.sha256(old_cp.read_bytes()).hexdigest())
    baseline = tmp_path / 'kiki/baseline-identity.txt'
    baseline.parent.mkdir()
    baseline.write_text('constructed fixed M3 metadata identity; not an NPZ', encoding='utf-8')
    monkeypatch.setattr(module, 'source_root', lambda config: baseline.parent)
    monkeypatch.setattr(module, 'source_file', lambda source, relative: baseline)
    monkeypatch.setattr(module, 'm3_physics', lambda p: {key: copy.deepcopy(physics[key]) for key in module.PHYSICS_KEYS})
    previous_path = root / 'old-manifest.json'
    gap = {'run_id': 'run_000000000099', 'source_zip': None, 'sha256': None,
           'exact_per_run_source_archive_available': False,
           'source_provenance_limitation': 'Historical archive absent; nearest anchor is not exact run source'}
    previous = {'protocol': 'T4_0001', 'complete': True, 'event_payload_read': False,
                'test_payload_read': False, 'fine_dt': .0025, 'coarse_dt': .005,
                'history_ms': 100, 'decision_horizon_s': 8.,
                'TRAIN_ids': sorted(ids), 'reference_physics': copy.deepcopy(physics), 'training_scales': scales,
                'endpoints': {'P-N': {'checkpoint': str(old_cp), 'checkpoint_sha256': module.OLD_N400_SHA256,
                                     'config': old['config'], 'managed_run_lineage': ['run_000000000099']},
                              'M3': {'source': str(baseline), 'source_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest()}},
                'source_run_archives': [gap], 'lineage_source_archive_gaps': [gap['run_id']]}
    save(previous_path, previous)
    save(root / 'configs/project.json', {'history_ms': 100, 'sample_dt': .01, 'baseline': baseline.name})
    n_cp = add_cp(root / 'artifacts/N800', make_cp(n_cfg, 'T4-N', 800))
    c_cfg = {**n_cfg, 'autonomous_only': False, 'teacher_usage': 'TRAIN_only_compression_adaptation',
             'teacher_checkpoint': str(n_cp), 'compression_source_autonomous_only': True,
             'teacher_field_weight': .1, 'teacher_gradient_weight': .1,
             'prediction_head_used_or_deployed': False}
    c_cp = add_cp(root / 'artifacts/C', make_cp(c_cfg, 'T4-S', 0, True))
    cp_objects[c_cp]['exact_resume_eligible'] = False
    s_cp = add_cp(root / 'artifacts/S600', make_cp(s_cfg, 'T4-S', 600, True))
    def export(cp, output):
        obj = cp_objects[cp]
        enc = {key: obj['encoder'][key].tolist() for key in ('mass', 'adjacency', 'scales', 'mu')}
        enc.update(type='shared_gru', hidden_size=32, history_samples=11, history_seconds=.1,
                   sample_dt_seconds=.01, gate_order=['reset', 'update', 'new'],
                   new_gate_equation='tanh(input_new + reset * hidden_new)',
                   centering='raw(history) - raw(zeros_like(history)); mu retained in both',
                   deployment_use='once at t=0.1 s; no future observations or latent resets')
        enc.update({key: obj['encoder']['gru.' + key].tolist() for key in
                    ('weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0')})
        enc.update({ 'head_' + key: obj['encoder']['head.' + key].tolist() for key in ('weight', 'bias')})
        result = {'format': 'task4-polynomial-powergraph', 'version': 1, 'n': 10, 'rank': 2,
                  'physics': {key: physics[key] for key in module.PHYSICS_KEYS}, 'omega_s': physics['omega_s'],
                  'scales': scales, 'strength': .01, 'functions': obj['symbolic_functions'], 'encoder': enc,
                  'coordinates': {'state_order': ['delta[N]', 'pi[N]', 'p[N]', 'q[N,rank] flattened machine-major'],
                                  'history_order': ['delta deviation', 'omega deviation', 'p deviation'],
                                  'time_unit': 's', 'delta_unit': 'rad', 'omega_unit': 'rad/s',
                                  'p_unit': 'system per-unit', 'pi_definition': 'mass * omega'},
                  **{key: obj['model'][key].tolist() for key in ('log_coupling', 'mu', 'phase', 'edges')}}
        save(output, result)
    c_json, s_json = c_cp.parent / 'model_symbolic.json', s_cp.parent / 'model_symbolic.json'
    export(c_cp, c_json); export(s_cp, s_json)
    def summary(cfg, step, start=0):
        return {'synthetic': False, 'requested_total_steps': cfg['steps'], 'steps_completed': step - start,
                'test_payload_read': False, 'future_input_at_deployment': False, 'autonomous_only': True,
                'declared_final_autonomous_horizon_seconds': 5., 'prediction_head_used_or_optimized': False,
                'final_single_start_phase_completed': step == cfg['steps'],
                'stop_reason': 'steps_completed' if step == cfg['steps'] else 'save_deadline'}
    for label, cp in (('N', n_cp), ('S', s_cp)):
        obj = cp_objects[cp]
        save(cp.parent / 'training_summary.json', summary(obj['config'], obj['step']))
        save(cp.parent / 'run_config.json', obj['config'])
        folders[label] = cp.parent
    folders['C'] = c_cp.parent
    save(c_cp.parent / 'compression_summary.json',
         {'complete': True, 'fitted_functions': 7, 'adapted_functions': 7,
          'stop_reason': 'seven_functions_compressed_and_short_adapted', 'final_checkpoint': str(c_cp),
          'teacher_checkpoint': str(n_cp), 'test_payload_read': False, 'validation_payload_read': False,
          'final_single_initialization_5s_finetune_completed': False, 'symbolic_export': str(c_json),
          'seed': 419, 'adapt_steps_per_function': 16, 'max_samples_per_function': 2048})
    state = {'history': []}
    def run(stage, number, folder, source, config=None, start=0, mode='warm_start'):
        run_id = f'run_{number:012x}'
        log_folder = root / 'logs' / run_id
        log_folder.mkdir(parents=True)
        manifest = {}
        script = root / 'scripts' / ('compress_model.py' if stage == 'C' else 'train.py')
        with zipfile.ZipFile(log_folder / 'source.zip', 'w') as archive:
            for code_path in [code, script] + ([config] if config is not None else []):
                content = code_path.read_bytes()
                relative = code_path.relative_to(root).as_posix()
                # The fixture mocks the deadline clock to epoch+100; ZIP's
                # header date must not inherit that pre-1980 synthetic time.
                archive.writestr(zipfile.ZipInfo(relative, (2020, 1, 1, 0, 0, 0)), content)
                manifest[relative] = hashlib.sha256(content).hexdigest()
        save(log_folder / 'source_manifest.json', manifest)
        result = {'run_id': run_id, 'seconds': float(number), 'exit_code': 0,
                  'reason': 'exit', 'log': str(log_folder / 'process.log')}
        result_path = root / f'reports/run-result-{number}.json'
        save(result_path, result)
        state['history'].append({'kind': 'PROCESS_CLOSED', **{k: v for k, v in result.items() if k != 'log'}})
        if stage == 'C':
            command = ['python.exe', 'scripts/compress_model.py', '--checkpoint', str(source),
                       '--output', str(folder), '--adapt-steps', '16', '--max-samples', '2048', '--seed', '419']
        else:
            command = ['python.exe', 'scripts/train.py', '--config', str(config), '--output', str(folder),
                       '--warm-start' if mode == 'warm_start' else '--resume', str(source)]
        receipt_path = root / f'reports/launch-receipt-{number}.json'
        save(receipt_path, {'schema_version': 1, 'run_id': run_id, 'loop_id': 'T4_0002', 'command': command,
                            'pid': number + 100, 'started_at': '2026-10-07T19:19:01+00:00',
                            'captured_at': '2026-10-07T19:57:27+00:00', 'capture_origin': 'live_active_job_state',
                            'project_root': str(root), 'formal_config_sha256': None if config is None else
                            hashlib.sha256(config.read_bytes()).hexdigest()})
        item = {'run_id': run_id, 'run_result': str(result_path), 'output': str(folder),
                'source_checkpoint': str(source), 'launch_receipt': str(receipt_path)}
        if stage != 'C':
            item.update(config=str(config), start_step=start, mode=mode)
        if stage == 'S':
            item['symbolic_json'] = str(s_json)
        return item
    lineage_path = root / 'lineage.json'
    lineage = {'protocol': 'T4_0002', 'stages': {
        'N': [run('N', 1, n_cp.parent, old_cp, n_path)],
        'C': [run('C', 2, c_cp.parent, n_cp)],
        'S': [run('S', 3, s_cp.parent, c_cp, s_path)]}}
    save(lineage_path, lineage); save(root / 'state/research.json', state)
    def raw_load(cp, map_location, weights_only):
        assert map_location == 'cpu' and weights_only is True
        calls.append(('raw', Path(cp)))
        return copy.deepcopy(cp_objects[Path(cp)])
    monkeypatch.setattr(module.torch, 'load', raw_load)
    change_on_load = {'path': None}
    def load(cp, device):
        assert device == 'cpu'
        cp = Path(cp)
        calls.append(('reconstruct', cp))
        assert ('raw', cp) in calls, 'Raw precision must be checked before any casting loader'
        obj = copy.deepcopy(cp_objects[cp])
        model = SimpleNamespace(dim=50, rank=2, **obj['model'])
        model.function_modules = lambda: {name: object.__new__(module.PolynomialFunction) for name in module.FUNCTION_NAMES}
        encoder = SimpleNamespace(**{key: obj['encoder'][key] for key in ('mass', 'adjacency', 'scales', 'mu')})
        if change_on_load['path'] is not None:
            target = change_on_load['path']
            target.write_bytes(target.read_bytes() + b'changed')
        return model, encoder, object(), obj
    monkeypatch.setattr(module, 'load_checkpoint', load)
    args = {'previous_manifest': previous_path, 'n_config': n_path, 's_config': s_path, 'lineage': lineage_path}
    return SimpleNamespace(module=module, root=root, args=args, output=root / 'new-manifest.json',
                           calls=calls, cp_objects=cp_objects, n_cp=n_cp, c_cp=c_cp, s_cp=s_cp, old_cp=old_cp,
                           n_cfg=n_cfg, s_cfg=s_cfg, n_path=n_path, s_path=s_path, s_json=s_json,
                           folders=folders, previous=previous, previous_path=previous_path, code=code,
                           lineage=lineage, lineage_path=lineage_path, state=state, run=run,
                           summary=summary, change=change_on_load, authorized=authorized)


def test_freeze_new_endpoints_preserves_order_history_gap_and_stage_accounting(freeze_fixture):
    f = freeze_fixture
    previous_bytes = f.previous_path.read_bytes()
    result = f.module.freeze(f.output, **f.args)
    assert result == json.loads(f.output.read_text(encoding='utf-8'))
    assert result['protocol'] == 'T4_0002' and result['complete'] is True
    assert set(result['endpoints']) == {'P-N', 'P-S', 'M3'}
    assert result['endpoints']['P-N']['step'] == 800
    assert result['endpoints']['P-S']['step'] == 600
    assert result['TRAIN_ids'] == f.cp_objects[f.old_cp]['event_ids']
    assert result['state_dimension'] == 50 and result['quotient_dimension'] == 49
    assert result['update_accounting']['N_cumulative_long_stage'] == 1200
    assert result['update_accounting']['historical_anchor_and_short_stages'] == (
        'Separate predecessor anchor/short stages; not included in this stage800 or reported long400+800=1200')
    assert result['lineage_source_archive_gaps'] == f.previous['lineage_source_archive_gaps']
    assert result['historical_source_run_archives'][0]['source_zip'] is None
    support = result['endpoints']['P-S']['polynomial_support_audit']
    assert support['frozen_training_support_slots_by_function']['node_energy'] == 2
    assert support['active_nonzero_support_slots_by_function']['node_energy'] == 1
    assert result['candidate_selection_performed'] is False and result['model_acceptance'] is False
    assert result['event_payload_read'] is False and result['test_payload_read'] is False
    assert f.previous_path.read_bytes() == previous_bytes
    launch = result['managed_stage_lineage']['N'][0]['launch_receipt_audit']
    assert launch['started_at'] != launch['captured_at']
    assert 'not independent third-party' in launch['source_scope']
    assert result['registered_compression']['seed'] == 419
    assert f.cp_objects[f.c_cp]['config']['seed'] == 401
    with pytest.raises(FileExistsError):
        f.module.freeze(f.output, **f.args)


@pytest.mark.parametrize('reason', ('unauthorized', 'infinite_deadline', 'expired_deadline'))
def test_authorization_and_finite_deadline_precede_source_reads(freeze_fixture, monkeypatch, reason):
    f = freeze_fixture
    if reason == 'unauthorized':
        f.authorized['allowed'] = False
    else:
        monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', 'inf' if reason == 'infinite_deadline' else '100')
    monkeypatch.setattr(f.module.Inputs, 'digest', lambda *args: pytest.fail('No read before authorization/deadline'))
    with pytest.raises((PermissionError, TimeoutError)):
        f.module.freeze(f.output, **f.args)
    assert not f.output.exists()


@pytest.mark.parametrize('bad', ('n_incomplete', 's_incomplete', 's_steps_outside_lock', 'wrong_kind', 'float32', 'integer_weight',
                                 'reordered_train', 'wrong_scales', 'auxiliary', 'teacher', 'shooting',
                                 'missing_support', 'changed_support', 'wrong_json_initializer', 'wrong_json_scale',
                                 'old_source_sha', 'broken_lineage', 'unfinished_run', 'archive_tamper',
                                 'posthoc_config', 'json_nonfinite', 's_cost', 'wrong_c_teacher', 'wrong_m3_identity'))
def test_incomplete_or_foreign_evidence_cannot_be_frozen(freeze_fixture, bad):
    f = freeze_fixture
    n, s = f.cp_objects[f.n_cp], f.cp_objects[f.s_cp]
    if bad == 'n_incomplete': n['step'] = 799
    elif bad == 's_incomplete': s['step'] = 599
    elif bad == 's_steps_outside_lock':
        value = {**f.s_cfg, 'steps': 801}
        save(f.s_path, value)
    elif bad == 'wrong_kind': s['kind'] = 'T4-N'
    elif bad == 'float32': n['model']['mass'] = n['model']['mass'].float()
    elif bad == 'integer_weight': n['encoder']['gru.weight_ih_l0'] = n['encoder']['gru.weight_ih_l0'].long()
    elif bad == 'reordered_train': s['event_ids'] = list(reversed(s['event_ids']))
    elif bad == 'wrong_scales': s['scales'][2] += .1
    elif bad == 'auxiliary': n['config']['anchor_weight'] = .1
    elif bad == 'teacher': n['config']['teacher_checkpoint'] = str(f.old_cp)
    elif bad == 'shooting': n['shooting'] = {'states.forbidden': torch.zeros(1, dtype=torch.float64)}
    elif bad == 'missing_support': del s['model']['node_energy.support']
    elif bad == 'changed_support': s['model']['node_energy.support'][1, 0] = False
    elif bad in ('wrong_json_initializer', 'wrong_json_scale'):
        value = json.loads(f.s_json.read_text(encoding='utf-8'))
        if bad == 'wrong_json_initializer': value['encoder']['head_bias'][0] += .1
        else: value['scales'][0] += .1
        save(f.s_json, value)
    elif bad == 'old_source_sha': f.old_cp.write_text('different old identity', encoding='utf-8')
    elif bad == 'broken_lineage':
        f.lineage['stages']['S'][0]['source_checkpoint'] = str(f.old_cp)
        save(f.lineage_path, f.lineage)
    elif bad == 'unfinished_run':
        f.state['history'] = [row for row in f.state['history'] if row['run_id'] != 'run_000000000003']
        save(f.root / 'state/research.json', f.state)
    elif bad == 'archive_tamper':
        archive = f.root / 'logs/run_000000000003/source.zip'
        with zipfile.ZipFile(archive, 'a') as z:
            z.writestr(zipfile.ZipInfo('unindexed.py', (2020, 1, 1, 0, 0, 0)), b'changed')
    elif bad == 'posthoc_config':
        save(f.s_path, {**f.s_cfg, 'lr': .0002})
    elif bad == 'json_nonfinite': f.s_path.write_text('{"value":1e999}', encoding='utf-8')
    elif bad == 's_cost': save(f.s_path, {**f.s_cfg, 'name': 'T4_0002_S_cost'})
    elif bad == 'wrong_c_teacher': f.cp_objects[f.c_cp]['config']['teacher_checkpoint'] = str(f.old_cp)
    elif bad == 'wrong_m3_identity':
        f.previous['endpoints']['M3']['source_sha256'] = '0' * 64
        save(f.previous_path, f.previous)
    with pytest.raises(ValueError):
        f.module.freeze(f.output, **f.args)
    assert not f.output.exists()
    if bad in ('float32', 'integer_weight'):
        assert not any(kind == 'reconstruct' for kind, _ in f.calls)


@pytest.mark.parametrize('source', ('checkpoint', 'code', 'previous_manifest'))
def test_before_end_hash_rejects_source_changes(freeze_fixture, source):
    f = freeze_fixture
    f.change['path'] = {'checkpoint': f.n_cp, 'code': f.code, 'previous_manifest': f.previous_path}[source]
    with pytest.raises(ValueError, match='changed'):
        f.module.freeze(f.output, **f.args)
    assert not f.output.exists()


def test_last_guard_revokes_a_published_manifest_if_a_source_changes(freeze_fixture, monkeypatch):
    f = freeze_fixture
    original = f.module.Inputs.end_guard
    calls = []
    def late_change(inputs):
        calls.append(True)
        if len(calls) == 2:
            assert f.output.is_file()
            f.code.write_bytes(f.code.read_bytes() + b'late source change')
        return original(inputs)
    monkeypatch.setattr(f.module.Inputs, 'end_guard', late_change)
    with pytest.raises(ValueError, match='changed'):
        f.module.freeze(f.output, **f.args)
    invalid = json.loads(f.output.read_text(encoding='utf-8'))
    assert invalid['complete'] is False
    assert invalid['source_identity_unchanged'] is False
    assert invalid['evidence_validity'] == 'invalid_source_changed'


@pytest.mark.parametrize('bad', ('script', 'executable', 'config', 'output', 'source', 'mode', 'run_id',
                                'formal_hash', 'extra_arm', 'duplicate_option', 'capture_before_start',
                                'wrong_loop', 'capture_origin', 'c_seed', 'c_adapt', 'c_samples'))
def test_live_launch_receipt_binds_actual_stage_arguments(freeze_fixture, bad):
    f = freeze_fixture
    stage = 'C' if bad.startswith('c_') else 'N'
    segment = f.lineage['stages'][stage][0]
    path = Path(segment['launch_receipt'])
    receipt = json.loads(path.read_text(encoding='utf-8'))
    command = receipt['command']
    replacements = {'script': (None, 'scripts/compress_model.py'), 'config': ('--config', str(f.s_path)),
                    'output': ('--output', str(f.root / 'other-output')), 'source': ('--warm-start', str(f.s_cp)),
                    'c_seed': ('--seed', '401'), 'c_adapt': ('--adapt-steps', '15'), 'c_samples': ('--max-samples', '1024')}
    if bad in replacements:
        option, value = replacements[bad]
        command[1 if option is None else command.index(option) + 1] = value
    elif bad == 'executable': command[0] = 'cmd.exe'
    elif bad == 'mode': command[command.index('--warm-start')] = '--resume'
    elif bad == 'run_id': receipt['run_id'] = 'run_000000000099'
    elif bad == 'formal_hash': receipt['formal_config_sha256'] = '0' * 64
    elif bad == 'extra_arm': command.extend(['--arm', 'A'])
    elif bad == 'duplicate_option': command.extend(['--config', str(f.n_path)])
    elif bad == 'capture_before_start': receipt['captured_at'] = '2026-10-07T19:00:00+00:00'
    elif bad == 'wrong_loop': receipt['loop_id'] = 'T4_0001'
    elif bad == 'capture_origin': receipt['capture_origin'] = 'reconstructed_after_close'
    save(path, receipt)
    with pytest.raises(ValueError):
        f.module.freeze(f.output, **f.args)
    assert not f.output.exists()


@pytest.mark.parametrize('section,key,value', (
    ('config', 'autonomous_only', True), ('config', 'anchor_weight', .1),
    ('config', 'latent_weight', .1), ('config', 'torque_weight', .1),
    ('config', 'teacher_field_weight', 0.), ('config', 'teacher_gradient_weight', 0.),
    ('config', 'prediction_head_used_or_deployed', True), ('config', 'seed', 419),
    ('summary', 'seed', 401), ('summary', 'adapt_steps_per_function', 15),
    ('summary', 'max_samples_per_function', 1024),
))
def test_actual_c_asset_and_summary_retain_the_registered_compression_contract(freeze_fixture, section, key, value):
    f = freeze_fixture
    if section == 'config':
        f.cp_objects[f.c_cp]['config'][key] = value
    else:
        path = f.c_cp.parent / 'compression_summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        summary[key] = value
        save(path, summary)
    with pytest.raises(ValueError, match='compression source'):
        f.module.freeze(f.output, **f.args)
    assert not f.output.exists()


@pytest.mark.parametrize('key', ('mu', 'phase', 'edges', 'adjacency', 'encoder_only'))
def test_synchronized_derived_buffer_changes_cannot_redefine_the_fixed_graph(freeze_fixture, key):
    f = freeze_fixture
    # Make N/C/S and both symbolic JSONs internally agree: the unchanged N400
    # anchor must still reject the altered physical feature/topology buffers.
    for cp in (f.n_cp, f.c_cp, f.s_cp):
        obj = f.cp_objects[cp]
        if key == 'encoder_only': obj['encoder']['mu'][0, 0] += .1
        elif key == 'edges': obj['model'][key][0, 1] = 2
        else:
            obj['model'][key].flatten()[0] += .1
            if key in ('mu', 'adjacency'):
                obj['encoder'][key].copy_(obj['model'][key])
        if cp in (f.c_cp, f.s_cp):
            path = cp.parent / 'model_symbolic.json'
            export = json.loads(path.read_text(encoding='utf-8'))
            if key != 'encoder_only' and key != 'adjacency': export[key] = obj['model'][key].tolist()
            for buffer in ('mu', 'adjacency'):
                export['encoder'][buffer] = obj['encoder'][buffer].tolist()
            save(path, export)
    with pytest.raises(ValueError, match='buffer differs from N400'):
        f.module.freeze(f.output, **f.args)
    assert not any(kind == 'reconstruct' for kind, _ in f.calls)
    assert not f.output.exists()


def test_learned_log_coupling_remains_free_to_change(freeze_fixture):
    f = freeze_fixture
    for index, cp in enumerate((f.n_cp, f.c_cp, f.s_cp), 1):
        f.cp_objects[cp]['model']['log_coupling'][0] = .1 * index
        if cp in (f.c_cp, f.s_cp):
            path = cp.parent / 'model_symbolic.json'
            export = json.loads(path.read_text(encoding='utf-8'))
            export['log_coupling'] = f.cp_objects[cp]['model']['log_coupling'].tolist()
            save(path, export)
    assert f.module.freeze(f.output, **f.args)['complete'] is True


@pytest.mark.parametrize('bad', ('degree4', 'seventeen_persistent_zero_slots'))
def test_compression_library_caps_apply_to_persistent_support_not_final_nonzeros(freeze_fixture, bad):
    f = freeze_fixture
    obj = copy.deepcopy(f.cp_objects[f.c_cp])
    name = 'node_energy'
    if bad == 'degree4':
        obj['model'][name + '.powers'][1, 0] = 4
    else:
        obj['model'][name + '.powers'] = torch.zeros(17, 7, dtype=torch.int64)
        coefficients = torch.zeros(17, 1, dtype=torch.float64)
        coefficients[0, 0] = 1.
        obj['model'][name + '.coefficients'] = coefficients
        obj['model'][name + '.support'] = torch.ones(17, 1, dtype=torch.bool)
        assert int(torch.count_nonzero(coefficients)) == 1
    powers, coefficients, support = [obj['model'][name + '.' + key] for key in ('powers', 'coefficients', 'support')]
    obj['symbolic_functions'][name].update(powers=powers.tolist(), coefficients=(coefficients * support).tolist())
    with pytest.raises(ValueError, match='Polynomial'):
        f.module.persistent_polynomials(obj)


def test_explicit_warm_then_resume_segments_are_accounted_once(freeze_fixture):
    f = freeze_fixture
    n_old_folder = f.root / 'artifacts/N-partial400'
    n_old_folder.mkdir()
    partial = n_old_folder / 'last.pt'
    partial.write_text('constructed intermediate complete-update identity', encoding='utf-8')
    f.cp_objects[partial] = copy.deepcopy(f.cp_objects[f.n_cp])
    f.cp_objects[partial]['step'] = 400
    save(n_old_folder / 'training_summary.json', f.summary(f.n_cfg, 400))
    save(n_old_folder / 'run_config.json', f.cp_objects[partial]['config'])
    save(f.n_cp.parent / 'training_summary.json', f.summary(f.n_cfg, 800, start=400))
    first = f.lineage['stages']['N'][0]
    first['output'] = str(n_old_folder)
    receipt_path = Path(first['launch_receipt'])
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt['command'][receipt['command'].index('--output') + 1] = str(n_old_folder)
    save(receipt_path, receipt)
    second = f.run('N', 4, f.n_cp.parent, partial, f.n_path, start=400, mode='resume')
    f.lineage['stages']['N'].append(second)
    save(f.lineage_path, f.lineage); save(f.root / 'state/research.json', f.state)
    result = f.module.freeze(f.output, **f.args)
    assert [row['step'] for row in result['managed_stage_lineage']['N']] == [400, 800]
    assert [row['mode'] for row in result['managed_stage_lineage']['N']] == ['warm_start', 'resume']
    assert result['update_accounting']['N_this_stage'] == 800
    assert result['update_accounting']['N_cumulative_long_stage'] == 1200
