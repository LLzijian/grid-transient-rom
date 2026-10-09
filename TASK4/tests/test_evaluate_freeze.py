"""CLI source guards using text placeholders and stubbed evaluation only.

No checkpoint parsing, real event loading, model construction, integration, or
scientific metric calculation occurs in these tests.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from task4.paths import ROOT


@pytest.fixture
def frozen_cli(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('evaluation_freeze_cli_test', ROOT / 'scripts/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = tmp_path / 'baseline-text-placeholder.txt'
    checkpoint = tmp_path / 'checkpoint-text-placeholder.txt'
    manifest_path = tmp_path / 'frozen-manifest.json'
    output = tmp_path / 'evaluation.json'
    baseline.write_text('constructed fixed M3 source bytes', encoding='utf-8')
    checkpoint.write_text('constructed checkpoint identity bytes; not a Torch file', encoding='utf-8')
    physics_values = {'stub_physics_only': True}
    scales = [.2, .3, .4]
    physics = SimpleNamespace(as_dict=lambda: copy.deepcopy(physics_values))
    loaded_checkpoint = {'synthetic': False, 'kind': 'T4-N', 'step': 400,
                         'config': {'arm': 'P'}, 'physics': copy.deepcopy(physics_values), 'scales': list(scales)}
    calls, result_holder = [], {}
    change = {'before_evaluate': None, 'after_evaluate': None, 'during_checkpoint_load': None, 'delete_after_evaluate': None}

    def digest(path):
        path = Path(path).resolve()
        assert path in (baseline, checkpoint, manifest_path), 'Only explicit placeholder sources may be hashed'
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {'complete': True, 'protocol': 'T4_0001', 'fine_dt': .0025, 'coarse_dt': .005,
                'reference_physics': copy.deepcopy(physics_values), 'training_scales': list(scales),
                'endpoints': {'M3': {'source_sha256': digest(baseline)},
                              'P-N': {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                                      'kind': 'T4-N', 'step': 400}}}

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, allow_nan=False), encoding='utf-8')
    save_manifest()

    project = {'baseline': 'placeholder-only', 'kiki_root': 'not-read'}

    def read_json(path):
        path = Path(path)
        calls.append(('read_json', str(path)))
        if path == module.ROOT / 'configs/project.json':
            return project
        assert path == manifest_path, 'No other JSON source may be read'
        return json.loads(path.read_text(encoding='utf-8'))

    def mutate(path):
        if path is not None:
            Path(path).write_text(Path(path).read_text(encoding='utf-8') + '\nchanged placeholder source', encoding='utf-8')

    def load_checkpoint(path):
        calls.append(('load_checkpoint', str(path)))
        assert Path(path) == checkpoint
        mutate(change['during_checkpoint_load'])
        return SimpleNamespace(), object(), object(), copy.deepcopy(loaded_checkpoint)

    def load_events(config, split):
        calls.append(('load_events', split))
        assert config is project and split in ('validation', 'train')
        mutate(change['before_evaluate'])
        # The plain ID is the only event attribute used by the stubbed CLI.
        return [SimpleNamespace(id='constructed-' + split)], physics, {'split': split, 'test_payload_read': False}

    def fixed_model(given_physics, given_scales):
        calls.append(('FrozenM3', None))
        assert given_physics is physics and list(given_scales) == scales
        return SimpleNamespace()

    def write_json(path, value):
        path = Path(path)
        calls.append(('write_json', str(path)))
        assert path == output, 'Invalidation must rewrite the same JSON file, not an output directory'
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')

    def evaluate(model, encoder, events, given_output, **kwargs):
        calls.append(('evaluate', kwargs['endpoint']))
        assert Path(given_output) == output
        assert kwargs['fine_dt'] == .0025 and kwargs['coarse_dt'] == .005
        assert kwargs['horizons'] == (1., 2., 5., 8.)
        # Nothing is read from events/model/encoder except the CLI identity.
        result = {'summary': {'stub_evaluation_only': True},
                  'model_identity': copy.deepcopy(model.evaluation_identity)}
        result_holder['result'] = result
        write_json(output, result)
        mutate(change['after_evaluate'])
        if change['delete_after_evaluate'] is not None:
            Path(change['delete_after_evaluate']).unlink()
        return result

    monkeypatch.setattr(module, 'check_authorized', lambda synthetic: calls.append(('authorized', synthetic)))
    monkeypatch.setattr(module.torch, 'set_num_threads', lambda value: calls.append(('threads_stub', value)))
    monkeypatch.setattr(module, 'read_json', read_json)
    monkeypatch.setattr(module, 'project_path', lambda value: Path(value).resolve())
    monkeypatch.setattr(module, 'source_root', lambda config: tmp_path)
    monkeypatch.setattr(module, 'source_file', lambda root, relative: baseline)
    monkeypatch.setattr(module, 'digest', digest)
    monkeypatch.setattr(module, 'load_checkpoint', load_checkpoint)
    monkeypatch.setattr(module, 'load_events', load_events)
    monkeypatch.setattr(module, 'train_scales', lambda events, given_physics: list(scales))
    monkeypatch.setattr(module, 'FrozenM3', fixed_model)
    monkeypatch.setattr(module, 'evaluate', evaluate)
    monkeypatch.setattr(module, 'write_json', write_json)
    monkeypatch.setattr(module, 'fixture', lambda: pytest.fail('No synthetic/real fixture loading in CLI metadata tests'))

    def run(endpoint='P-N'):
        args = ['evaluate.py', '--baseline-m3'] if endpoint == 'M3' else ['evaluate.py', '--checkpoint', str(checkpoint)]
        args += ['--endpoint', endpoint, '--split', 'validation', '--frozen-manifest', str(manifest_path), '--output', str(output)]
        monkeypatch.setattr(sys, 'argv', args)
        module.main()

    return SimpleNamespace(module=module, baseline=baseline, checkpoint=checkpoint, manifest_path=manifest_path,
                           manifest=manifest, save_manifest=save_manifest, checkpoint_metadata=loaded_checkpoint,
                           output=output, calls=calls, change=change, digest=digest, run=run, result_holder=result_holder)


def calls_named(context, name):
    return [row for row in context.calls if row[0] == name]


@pytest.mark.parametrize('endpoint', ['M3', 'P-N'])
def test_m3_sha_mismatch_is_rejected_before_any_event_read(frozen_cli, endpoint):
    frozen_cli.manifest['endpoints']['M3']['source_sha256'] = '0' * 64
    frozen_cli.save_manifest()
    with pytest.raises(ValueError, match='Fixed M3 source differs'):
        frozen_cli.run(endpoint)
    assert not calls_named(frozen_cli, 'load_events')
    assert not calls_named(frozen_cli, 'load_checkpoint')
    assert not calls_named(frozen_cli, 'evaluate') and not frozen_cli.output.exists()


@pytest.mark.parametrize('mismatch', ['sha', 'path', 'step', 'kind', 'physics', 'scales'])
def test_checkpoint_identity_mismatch_cannot_read_development_events(frozen_cli, mismatch):
    identity = frozen_cli.manifest['endpoints']['P-N']
    if mismatch == 'sha':
        identity['checkpoint_sha256'] = '0' * 64
    elif mismatch == 'path':
        identity['checkpoint'] += '.different-placeholder'
    elif mismatch == 'step':
        frozen_cli.checkpoint_metadata['step'] = 399
    elif mismatch == 'kind':
        frozen_cli.checkpoint_metadata['kind'] = 'T4-S'
    elif mismatch == 'physics':
        frozen_cli.checkpoint_metadata['physics'] = {'different_stub_physics': True}
    else:
        frozen_cli.checkpoint_metadata['scales'] = [.2, .3, .5]
    frozen_cli.save_manifest()
    with pytest.raises(ValueError, match='Checkpoint differs|Loaded checkpoint disagrees'):
        frozen_cli.run()
    assert not calls_named(frozen_cli, 'load_events') and not calls_named(frozen_cli, 'evaluate')
    if mismatch in ('sha', 'path'):
        assert not calls_named(frozen_cli, 'load_checkpoint')


def test_legal_m3_identity_anchors_baseline_path_hash_and_manifest_hash(frozen_cli):
    frozen_cli.run('M3')
    identity = frozen_cli.result_holder['result']['model_identity']
    assert identity['kind'] == 'fixed_M3' and not identity['fitted_on_development']
    assert identity['baseline_source'] == str(frozen_cli.baseline)
    assert identity['baseline_source_sha256'] == frozen_cli.digest(frozen_cli.baseline)
    assert identity['frozen_manifest'] == str(frozen_cli.manifest_path)
    assert identity['frozen_manifest_sha256'] == frozen_cli.digest(frozen_cli.manifest_path)
    assert [row[1] for row in calls_named(frozen_cli, 'load_events')] == ['validation', 'train']
    assert frozen_cli.output.is_file() and not frozen_cli.output.is_dir()
    assert frozen_cli.calls[0] == ('authorized', False)


def test_legal_checkpoint_identity_preserves_registered_step_and_sha(frozen_cli):
    frozen_cli.run()
    identity = frozen_cli.result_holder['result']['model_identity']
    assert identity['checkpoint'] == str(frozen_cli.checkpoint)
    assert identity['checkpoint_sha256'] == frozen_cli.digest(frozen_cli.checkpoint)
    assert identity['kind'] == 'T4-N' and identity['step'] == 400
    assert identity['frozen_manifest_sha256'] == frozen_cli.digest(frozen_cli.manifest_path)
    assert [row[1] for row in calls_named(frozen_cli, 'load_events')] == ['validation']


def test_checkpoint_mutation_during_stub_load_is_rejected_before_events(frozen_cli):
    frozen_cli.change['during_checkpoint_load'] = frozen_cli.checkpoint
    with pytest.raises(ValueError, match='source changed'):
        frozen_cli.run()
    assert len(calls_named(frozen_cli, 'load_checkpoint')) == 1
    assert not calls_named(frozen_cli, 'load_events') and not calls_named(frozen_cli, 'evaluate')


@pytest.mark.parametrize('source', ['baseline', 'checkpoint', 'manifest_path'])
def test_changed_source_before_stub_evaluation_is_rejected(frozen_cli, source):
    frozen_cli.change['before_evaluate'] = getattr(frozen_cli, source)
    with pytest.raises(ValueError, match='source changed'):
        frozen_cli.run()
    assert len(calls_named(frozen_cli, 'load_events')) == 1
    assert not calls_named(frozen_cli, 'evaluate') and not frozen_cli.output.exists()


@pytest.mark.parametrize('source', ['baseline', 'checkpoint', 'manifest_path'])
def test_changed_source_after_stub_evaluation_invalidates_same_output_json(frozen_cli, source):
    frozen_cli.change['after_evaluate'] = getattr(frozen_cli, source)
    with pytest.raises(ValueError, match='source changed'):
        frozen_cli.run()
    assert len(calls_named(frozen_cli, 'evaluate')) == 1
    assert frozen_cli.output.is_file() and not frozen_cli.output.is_dir()
    saved = json.loads(frozen_cli.output.read_text(encoding='utf-8'))
    assert saved['source_identity_unchanged'] is False
    assert saved['evidence_validity'] == 'invalid_source_changed'
    assert len(calls_named(frozen_cli, 'write_json')) == 2
    assert all(row[1] == str(frozen_cli.output) for row in calls_named(frozen_cli, 'write_json'))


@pytest.mark.parametrize('source', ['baseline', 'checkpoint', 'manifest_path'])
def test_deleted_source_after_stub_evaluation_also_invalidates_saved_output(frozen_cli, source):
    frozen_cli.change['delete_after_evaluate'] = getattr(frozen_cli, source)
    with pytest.raises((ValueError, OSError)):
        frozen_cli.run()
    saved = json.loads(frozen_cli.output.read_text(encoding='utf-8'))
    assert saved['source_identity_unchanged'] is False
    assert saved['evidence_validity'] == 'invalid_source_changed'
    assert len(calls_named(frozen_cli, 'write_json')) == 2


@pytest.mark.parametrize('mismatch', ['incomplete', 'protocol', 'fine_dt', 'coarse_dt'])
def test_incompatible_manifest_is_rejected_before_checkpoint_or_event_loading(frozen_cli, mismatch):
    if mismatch == 'incomplete':
        frozen_cli.manifest['complete'] = False
    elif mismatch == 'protocol':
        frozen_cli.manifest['protocol'] = 'different-protocol'
    else:
        frozen_cli.manifest[mismatch] *= 2
    frozen_cli.save_manifest()
    with pytest.raises(ValueError, match='Incomplete or incompatible'):
        frozen_cli.run()
    assert not calls_named(frozen_cli, 'load_checkpoint') and not calls_named(frozen_cli, 'load_events')


@pytest.mark.parametrize('mismatch', [None, 'sha', 'step'])
def test_registered_second_protocol_preserves_frozen_source_guards(frozen_cli, mismatch):
    # The registered new phase can finish at 800; it is not an old 400-step
    # endpoint relabel. Sources are still plain placeholders, with no payload.
    frozen_cli.manifest['protocol'] = 'T4_0002'
    identity = frozen_cli.manifest['endpoints']['P-N']
    identity['step'] = 800
    frozen_cli.checkpoint_metadata['step'] = 800
    if mismatch == 'sha':
        identity['checkpoint_sha256'] = '0' * 64
    elif mismatch == 'step':
        frozen_cli.checkpoint_metadata['step'] = 799
    frozen_cli.save_manifest()
    if mismatch is not None:
        with pytest.raises(ValueError, match='Checkpoint differs|Loaded checkpoint disagrees'):
            frozen_cli.run()
        assert not calls_named(frozen_cli, 'load_events')
        assert not calls_named(frozen_cli, 'evaluate')
    else:
        frozen_cli.run()
        result = frozen_cli.result_holder['result']
        assert result['source_identity_unchanged'] is True
        assert result['model_identity']['frozen_manifest_sha256'] == frozen_cli.digest(frozen_cli.manifest_path)
        assert len(calls_named(frozen_cli, 'evaluate')) == 1


def test_authorization_failure_precedes_all_placeholder_source_reads(frozen_cli, monkeypatch):
    def deny(_):
        raise PermissionError('constructed denied authorization')
    monkeypatch.setattr(frozen_cli.module, 'check_authorized', deny)
    with pytest.raises(PermissionError, match='denied authorization'):
        frozen_cli.run()
    assert frozen_cli.calls == [] and not frozen_cli.output.exists()
