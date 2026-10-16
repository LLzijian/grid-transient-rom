"""Scientific recovery invariants, using synthetic models and no datasets."""
import json
import contextlib
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'experiments'),
                str(Path(__file__).resolve().parents[1] / 'src')]
import discover_closure as runner
import closure_discovery_support as support
from closure_discovery_benchmarks import toy_source


def completed_checkpoint(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    identity = out / 'implementation/identity.json'
    identity.parent.mkdir(parents=True)
    identity.write_text('{"implementation":"before_repair"}')
    cfg = {'rates': {'skeleton': .01, 'closure': .001, 'encoder': .0003}}
    monkeypatch.setattr(runner, 'context', lambda: (cfg, out))
    model = toy_source()
    model.provenance['fit_seed'] = 2701
    path = tmp_path / 'candidate'
    runner.checkpoint(path, model, runner.optimizer(model), np.random.default_rng(2701),
                      1024, [{'step': 1024, 'loss': 1.}],
                      coverage={'cover_order': [0], 'cursor': 0})
    original = {p.name: p.read_bytes() for p in path.iterdir()}
    identity.write_text('{"implementation":"after_repair"}')
    return model, path, original


def test_completed_fit_is_read_only_across_implementation_identity_change(tmp_path, monkeypatch):
    model, path, original = completed_checkpoint(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('completed fit must not optimize, evaluate loss, or write a checkpoint')
    for name in ('optimizer', 'loss', 'checkpoint'):
        monkeypatch.setattr(runner, name, forbidden)
    restored, result = runner.fit(model, [{'id': 'synthetic'}], 1., 1024, path, 2701, final=True)
    assert result['updates'] == 1024 and result['reused_checkpoint']
    assert restored.provenance == model.provenance
    assert {p.name: p.read_bytes() for p in path.iterdir()} == original


def test_fit_rejects_later_checkpoint_instead_of_relabelling(tmp_path, monkeypatch):
    model, path, original = completed_checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'optimizer', lambda *a: pytest.fail('no optimizer on rejected resume'))
    with pytest.raises(ValueError, match='exceeds requested adaptation target'):
        runner.fit(model, [{'id': 'synthetic'}], 1., 512, path, 2701)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == original


def test_fit_rejects_model_optimizer_update_disagreement(tmp_path, monkeypatch):
    model, path, _ = completed_checkpoint(tmp_path, monkeypatch)
    state = torch.load(path / 'training.pt', weights_only=False)
    state['step'] = 512
    torch.save(state, path / 'training.pt')
    meta = json.loads((path / 'checkpoint.json').read_text())
    meta['training_sha256'] = runner.sha(path / 'training.pt')
    (path / 'checkpoint.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='updates differ'):
        runner.fit(model, [{'id': 'synthetic'}], 1., 1024, path, 2701)


def test_structure_identity_includes_exchange_dissipation_mask():
    terms = toy_source().terms
    first = runner.structure_identity(terms, [1., 1., 1., 0.])
    assert first == runner.structure_identity(list(reversed(terms)), [1, 1, 1, 0])
    assert first != runner.structure_identity(terms, [1., 1., 1., 1.])


def test_partial_fit_matches_uninterrupted_optimizer_rng_and_coverage(tmp_path, monkeypatch):
    import train_local_memory_0098 as memory
    import train_joint_numerical_pair_0100 as joint
    cfg = {'rates': {'skeleton': .01, 'closure': .001, 'encoder': .0003}}
    monkeypatch.setattr(runner, 'context', lambda: (cfg, tmp_path / 'out'))
    monkeypatch.setattr(memory, 'graph', lambda model: contextlib.nullcontext())
    monkeypatch.setattr(memory, 'histories_for', lambda samples: torch.zeros(1, 11, 30))
    def next_samples(events, rng, coverage):
        index = int(rng.integers(len(events)))
        coverage['cursor'] += 1
        return [(events[index], 10)], {}
    monkeypatch.setattr(joint, 'next_samples', next_samples)
    def synthetic_loss(model, samples, history, horizon, scale):
        value = (1. + .1 * samples[0][0]['index']) * sum(
            p.square().mean() for p in model.parameters() if p.requires_grad and p.numel())
        return value, {'risk': value}
    monkeypatch.setattr(runner, 'loss', synthetic_loss)
    events = [{'id': 'a', 'index': 0}, {'id': 'b', 'index': 1}]
    base = toy_source()
    full, _ = runner.fit(copy.deepcopy(base), events, 1., 3, tmp_path / 'full', 2701)
    part, _ = runner.fit(copy.deepcopy(base), events, 1., 2, tmp_path / 'split', 2701)
    resumed, _ = runner.fit(part, events, 1., 3, tmp_path / 'split', 2701)
    for name, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    a = torch.load(tmp_path / 'full/training.pt', weights_only=False)
    b = torch.load(tmp_path / 'split/training.pt', weights_only=False)
    assert a['numpy_rng'] == b['numpy_rng']
    assert torch.equal(a['torch_rng'], b['torch_rng'])
    assert a['coverage'] == b['coverage']
    ta = json.loads((tmp_path / 'full/checkpoint.json').read_text())['trace']
    tb = json.loads((tmp_path / 'split/checkpoint.json').read_text())['trace']
    assert [{k: v for k, v in row.items() if k != 'seconds'} for row in ta] == [
        {k: v for k, v in row.items() if k != 'seconds'} for row in tb]


def test_later_level_failure_does_not_admit_third_score32_candidate(tmp_path, monkeypatch):
    class StopAfterLibraryRound(Exception):
        pass
    cfg = {'search_seeds': [1701], 'search': {'selected_rounds': 1}}
    monkeypatch.setattr(runner, 'context', lambda: (cfg, tmp_path))
    monkeypatch.setattr(runner, 'train_inputs', lambda: ([{'id': 'synthetic'}], [], 1.))
    monkeypatch.setattr(support, 'ProcessLedger', lambda p: SimpleNamespace(caps_seconds={}))
    choice = {'depth': 1}
    monkeypatch.setattr(support, 'select_depth', lambda *a, **k: choice)
    runner.write(tmp_path / 'A/depth_lock.json', {
        'choice': choice, 'timing': {}, 'fom_preflight_qualified': True,
        'full_expression_depth_priced': True, 'confirmation_execution_path_priced': True})
    model = toy_source()
    monkeypatch.setattr(runner, 'source', lambda: model)
    def proposals(beam, mode, seed, events):
        if mode == 'generated':
            raise StopAfterLibraryRound
        return ([{'terms': model.terms} for _ in range(3)], [])
    monkeypatch.setattr(runner, 'rank_proposals', proposals)
    folder = tmp_path / 'B/library/1701/round1'
    for index in range(3):
        path = folder / f'candidate{index:02d}'
        path.mkdir(parents=True)
        model.export(path / 'model.json')
        runner.write(path / 'checkpoint.json', {'step': 32})
    runner.write(folder / 'candidate00/candidate_failure.json', {
        'level': 512, 'status': 'numerically_unresolved', 'error': 'synthetic later failure'})
    monkeypatch.setattr(runner, 'score_at_level', lambda p, e, level, **k: {
        'risk': float(Path(p).name[-2:]), 'group_se': 0., 'events': []})
    fitted = []
    def fit(m, events, scale, target, path, seed, **kwargs):
        fitted.append((int(Path(path).name[-2:]), target))
        runner.write(Path(path) / 'checkpoint.json', {'step': target})
        return m, {}
    monkeypatch.setattr(runner, 'fit', fit)
    with pytest.raises(StopAfterLibraryRound):
        runner.stage_b(tmp_path, None)
    assert fitted == [(1, 128), (1, 512)]
    results = json.loads((tmp_path / 'B/search_results.json').read_text())
    assert len(results) == 1 and results[0]['path'].endswith('candidate01')


def setup_synthetic_c(tmp_path, monkeypatch, seeds):
    model = toy_source().with_terms([
        {'channel': 'port', 'expression': {'op': 'var', 'name': 'q'}}])
    row = {'terms': model.terms, 'direction_mask': model.template_mask[3:7].tolist()}
    runner.write(tmp_path / 'B/selection.json', {
        'library': row, 'generated': row, 'alternative': row})
    monkeypatch.setattr(runner, 'context', lambda: ({'refit_seeds': seeds}, tmp_path))
    monkeypatch.setattr(runner, 'source', lambda: model)
    monkeypatch.setattr(runner, 'train_inputs', lambda: ([{'id': 'synthetic'}], [], 1.))
    return model


def test_public_c_resume_keeps_failed_core_seed_and_continues_other_seed(tmp_path, monkeypatch):
    class StopAfterRecordedFailure(Exception):
        pass
    setup_synthetic_c(tmp_path, monkeypatch, [2701, 2702])
    path = tmp_path / 'C/fixed/2701/fit_failure.json'
    runner.write(path, {'reason': 'synthetic failure', 'completed_updates': 5,
                        'state_sha256': 'f' * 64, 'status': 'numerically_unresolved'})
    original = path.read_bytes()
    calls = []
    def fit(model, events, scale, target, path, seed, **kwargs):
        calls.append((str(path), seed))
        raise StopAfterRecordedFailure
    monkeypatch.setattr(runner, 'fit', fit)
    with pytest.raises(StopAfterRecordedFailure):
        runner.stage_c(tmp_path, None)
    assert len(calls) == 1 and calls[0][1] == 2702
    failure = json.loads((tmp_path / 'C/progress.json').read_text())['numerical_failures'][0]
    assert failure['name'] == 'fixed' and failure['seed'] == 2701
    assert failure['completed_updates'] == 5 and failure['state_sha256'] == 'f' * 64
    assert path.read_bytes() == original


def test_public_c_resume_never_retries_a_recorded_deletion_failure(tmp_path, monkeypatch):
    setup_synthetic_c(tmp_path, monkeypatch, [2701])
    failed_path = tmp_path / 'C/delete0/2701/fit_failure.json'
    runner.write(failed_path, {'reason': 'synthetic deletion failure', 'completed_updates': 7,
                               'status': 'numerically_unresolved'})
    calls = []
    def fit(model, events, scale, target, path, seed, **kwargs):
        path = Path(path)
        calls.append(str(path))
        path.mkdir(parents=True, exist_ok=True)
        model.export(path / 'model.json')
        return model, {}
    monkeypatch.setattr(runner, 'fit', fit)
    monkeypatch.setattr(runner, 'score', lambda *a, **k: {'risk': 0., 'events': []})
    monkeypatch.setattr(runner, 'physical_qa', lambda model: {'pass': True})
    with pytest.raises(RuntimeError, match='complete C comparison unresolved'):
        runner.stage_c(tmp_path, None)
    assert len(calls) == 4 and all('delete0' not in path for path in calls)
    failure = json.loads((tmp_path / 'C/progress.json').read_text())['numerical_failures'][0]
    assert failure['name'] == 'delete0' and failure['completed_updates'] == 7
    assert not (tmp_path / 'C/frozen.json').exists()
