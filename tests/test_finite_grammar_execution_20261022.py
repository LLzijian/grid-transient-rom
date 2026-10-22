"""One qualification collection; no formal seed fixtures are generated here."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src'), str(ROOT / 'tests')]
import finite_grammar_execution_20261022 as entry
import finite_grammar_discovery_20261021 as science
# Collect only this one original CUDA qualification, not the old test module.
from test_finite_grammar_discovery_20261021 import test_real_32_checkpoint_resume_34_exact_params_adam_rng_output


def binding():
    return json.loads((ROOT / 'private_data/closure_mechanism_20261019/P0/finite_grammar_discovery01/bindings/qa01.json').read_text())


def test_frozen_pool_teacher_and_selector_interface():
    b = binding(); pool = entry.frozen_pool(b); truth = science.teacher(b)
    assert len(pool['selected']) == 6 and truth['coefficient'] == .15
    assert set(inspect.signature(science.search_seed).parameters) == {
        'terms', 'keys', 'seed', 'coarse', 'fine', 'sc', 'output', 'commit', 'deadline'}
    changed = copy.deepcopy(b); changed['pool_manifest']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='pool identity'):
        entry.frozen_pool(changed)
    changed = copy.deepcopy(b); changed['teacher_identity']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='teacher identity'):
        entry.frozen_pool(changed)


def test_no_old_cost_gate_no_deadline_and_finite_workload():
    assert science.counts() == {'updates_per_seed': 768, 'total_updates': 2304,
        'checkpoints_per_seed': 15, 'total_checkpoints': 45, 'total_restores': 9,
        'total_rankings': 27, 'total_constructions': 27, 'final_evaluation_loads': 3}
    assert entry.SEEDS == (6101, 6102, 6103)
    tree = ast.parse(inspect.getsource(entry))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = {'passed', 'run_cost', 'run_study', 'main', 'execute'}
    assert not any(isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                   and n.func.value.id == 'science' and n.func.attr in forbidden for n in calls)
    for n in calls:
        if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'science':
            if n.func.attr in {'fixture', 'search_seed', 'predict', 'evaluate_study'}:
                assert isinstance(n.args[-1], ast.Constant) and n.args[-1].value is None
    assert 'resume' not in inspect.signature(entry.run_study).parameters


def test_chain_overlap_and_real_interval_full_event_qualification():
    rows = [{'key': k, 'score': {'mean': m, 'B': 1., 'valid': True}} for k, m in [('c', 0.), ('b', 1.5), ('a', 3.)]]
    selected, ambiguous = science.choose(rows, 2)
    assert ambiguous and [r['key'] for r in selected] == ['a', 'b']
    coarse = np.zeros((16, 511, 30)); fine = coarse.copy()
    fine[15, 0, 0] = .002
    q = science.qualify_teacher(coarse, fine, np.ones(30))
    assert not q['valid'] and q['count'] == 16 and sum(q['eligible']) == 15
    raw = np.zeros((16, 500, 30))
    score = science.score(raw, raw, raw, raw, np.ones(30))
    assert score['indices'] == [200, 500] and score['count'] == 16


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.5], dtype=torch.float64))
        self.register_buffer('mass', torch.ones(1, dtype=torch.float64))

    def export(self, path):
        Path(path).write_text(json.dumps({'qa_tiny_only': True, 'weight': self.weight.detach().tolist()}))


def tiny_checkpoint(tmp_path):
    from closure_causal_history_residual_20261019 import checkpoint
    model = Tiny(); opt = torch.optim.Adam(model.parameters(), lr=.001)
    trace = []
    for _ in range(2):
        opt.zero_grad(); loss = model.weight.square().sum(); loss.backward(); opt.step()
        trace.append(float(loss.detach()))
    identity = {'qa_only': True, 'seed': 9902}
    rng = np.random.default_rng(9902)
    checkpoint(tmp_path, model, opt, rng, 2, trace, identity)
    directory = tmp_path / 'checkpoint_00002'
    state = torch.load(directory / 'training.pt', map_location='cpu', weights_only=False)
    return directory, state, identity


def test_real_small_serializer_count_adam_rng_trace(tmp_path):
    directory, state, identity = tiny_checkpoint(tmp_path)
    assert json.loads((directory.parent / 'latest.json').read_text())['updates'] == 2
    proof = entry.validate_state(state, identity, 2)
    assert proof == {'updates': 2, 'adam_parameters': 1, 'trace_length': 2, 'cuda_rng_count': 0}
    # An actual step2 snapshot must never qualify as a scientific step512.
    with pytest.raises(ValueError, match='count'):
        entry.validate_state(state, identity, 512)
    for field, value, message in [
        ('completed_updates', True, 'integer'), ('trace', state['trace'][:1], 'trace'),
        ('identity', {'wrong': True}, 'identity'), ('torch_rng', torch.ones(3), 'CPU RNG')]:
        bad = copy.deepcopy(state); bad[field] = value
        with pytest.raises(ValueError, match=message):
            entry.validate_state(bad, identity, 2)


def test_real_adam_empty_and_parameter_step_rejected(tmp_path):
    _, state, identity = tiny_checkpoint(tmp_path)
    bad = copy.deepcopy(state); bad['optimizer']['state'] = {}
    with pytest.raises(ValueError, match='nonempty'):
        entry.validate_state(bad, identity, 2)
    bad = copy.deepcopy(state)
    next(iter(bad['optimizer']['state'].values()))['step'].fill_(1)
    with pytest.raises(ValueError, match='per-parameter'):
        entry.validate_state(bad, identity, 2)


def test_pending_and_hash_rejected_before_any_model_import(tmp_path):
    directory, _, identity = tiny_checkpoint(tmp_path)
    wrong = {'model.json': '0' * 64, 'training.pt': '0' * 64}
    with pytest.raises(ValueError, match='bytes'):
        entry.validate_checkpoint(directory, identity, 2, wrong, require_cuda=False)
    (directory.parent / 'pending_qa').mkdir()
    with pytest.raises(ValueError, match='pending'):
        entry.validate_checkpoint(directory, identity, 2, require_cuda=False)


@pytest.mark.parametrize('seeds', [[], [6101], [6101, 6102], [6101, 6102, 6102]])
def test_missing_or_duplicate_endpoint_barrier_before_val(seeds, tmp_path):
    selection = {'source_commit': 'a' * 40, 'selection': 'train_only',
                 'all_three_frozen': True, 'rows': [{'seed': s} for s in seeds]}
    with pytest.raises(ValueError, match='three registered seeds'):
        entry.validate_selection(selection, binding(), 'a' * 40)
    # No fixture or evaluation folder can be opened before the barrier.
    assert not list(tmp_path.iterdir())


def test_unresolved_endpoint_barrier_before_checkpoint_load():
    selection = {'source_commit': 'a' * 40, 'selection': 'train_only',
                 'all_three_frozen': True, 'rows': [{'seed': s, 'status': 'UNRESOLVED'} for s in entry.SEEDS]}
    with pytest.raises(ValueError, match='real frozen'):
        entry.validate_selection(selection, binding(), 'a' * 40)


@pytest.mark.parametrize('extra', ['checkpoint_00512', 'checkpoint_00032.diagnostics.json', 'pending_qa', 'unknown.json'])
def test_future_and_unknown_entries_rejected(tmp_path, extra):
    # Directory names are layout-only fixtures: they are never treated as an
    # actual optimizer endpoint and cannot pass validate_checkpoint.
    (tmp_path / 'checkpoint_00000').mkdir()
    (tmp_path / 'checkpoint_00032').mkdir()
    latest = {'updates': 32, 'checkpoint': 'checkpoint_00032', 'sha256': {}}
    (tmp_path / 'latest.json').write_text(json.dumps(latest))
    assert entry.validate_layout(tmp_path, [0, 32]) == latest
    (tmp_path / extra).write_text('uncommitted')
    with pytest.raises(ValueError, match='unknown/pending/future'):
        entry.validate_layout(tmp_path, [0, 32])


def test_latest_must_match_actual_stage_path(tmp_path):
    (tmp_path / 'checkpoint_00000').mkdir()
    (tmp_path / 'checkpoint_00032').mkdir()
    (tmp_path / 'latest.json').write_text(json.dumps({'updates': 512, 'checkpoint': 'checkpoint_00512'}))
    with pytest.raises(ValueError, match='latest checkpoint'):
        entry.validate_layout(tmp_path, [0, 32])
