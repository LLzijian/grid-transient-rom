"""One finite qualification collection, using independent 9922 observations."""
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
import bounded_edge_grammar_20261022 as entry
from test_finite_grammar_discovery_20261021 import capture, exact
from test_finite_grammar_execution_20261022 import tiny_checkpoint


def variable(name): return {'op': 'var', 'name': name}
def multiply(*args): return {'op': 'mul', 'args': list(args)}
def term(expression): return {'channel': 'edge', 'expression': expression}


def test_exact_degree_retains_cancellation_and_rejects_zero():
    q = variable('qi'); s = variable('s')
    q2 = {'op': 'pow2', 'arg': q}
    expression = {'op': 'add', 'args': [q2, multiply({'op': 'const', 'value': -1}, q2), multiply(q, s)]}
    assert entry.polynomial(expression) == {('qi', 's'): 1}
    assert entry.eligible(term(expression))
    assert not entry.eligible(term(multiply(variable('qi'), variable('qj'), s)))
    assert not entry.eligible(term({'op': 'add', 'args': [q2, multiply({'op': 'const', 'value': -1}, q2)]}))
    assert not entry.eligible(term({'op': 'pow2', 'arg': s}))
    # Raw q degree1 does not qualify when endpoint symmetry cancels all q
    # dependence and leaves a nonzero pure angular mechanism.
    cancels = {'op': 'add', 'args': [multiply(q, s), multiply(variable('qj'), s), {'op': 'pow2', 'arg': variable('c')}]}
    assert not entry.eligible(term(cancels))
    assert entry.centered_edge_polynomial(cancels) == {('c', 'c'): 1}


@pytest.mark.parametrize('bad', [
    multiply(variable('pi'), variable('qi'), variable('s')),
    multiply(variable('pj'), variable('qj'), variable('c')),
    multiply({'op': 'tanh', 'scale': 1, 'arg': variable('qi')}, variable('s')),
    multiply(variable('qi'), {'op': 'tanh', 'scale': 1, 'arg': variable('s')}),
    {'op': 'sin', 'arg': variable('qi')},
])
def test_forbidden_variables_and_functions(bad):
    assert not entry.eligible(term(bad))


def test_pool_freeze_identity_and_unique_teacher(tmp_path):
    from closure_discovery_support import term_key
    from closure_expression_identity import mechanism_signature
    import hashlib
    pool = entry.generate_pool()
    assert len(pool['selected']) == 6 and all(entry.eligible(t) for t in pool['eligible_pool'])
    assert len(set(mechanism_signature(t) for t in pool['eligible_pool'])) == len(pool['eligible_pool'])
    assert pool['selected'] == sorted(pool['eligible_pool'], key=lambda t: (hashlib.sha256(term_key(t).encode()).hexdigest(), term_key(t)))[:6]
    frozen = entry.freeze(tmp_path / 'pool')
    assert frozen['status'] == 'FROZEN'
    assert entry.read_pool(frozen) == pool
    teacher = entry.teacher(frozen)
    index = int(np.random.default_rng(np.random.SeedSequence([20261022, 6200])).integers(6))
    assert teacher['index'] == index and teacher['coefficient'] == .15
    assert teacher['term'] == pool['selected'][index]
    path = Path(frozen['teacher_identity']['path']); path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        entry.teacher(frozen)


def test_production_compiler_values_gradients_and_autograd():
    from symbolic_closure_rom import evaluate_expression_torch, evaluate_expression_numpy
    features = torch.tensor(np.random.default_rng(9922).normal(size=(7, 6)), dtype=torch.float64, requires_grad=True)
    for candidate in entry.generate_pool()['selected']:
        value, gradient = evaluate_expression_torch(candidate['expression'], features, 'edge')
        np_value, np_gradient = evaluate_expression_numpy(candidate['expression'], features.detach().numpy(), 'edge')
        np.testing.assert_allclose(value.detach().numpy(), np_value, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(gradient.detach().numpy(), np_gradient, rtol=1e-12, atol=1e-12)
        actual, = torch.autograd.grad(value.sum(), features)
        torch.testing.assert_close(actual, gradient, rtol=1e-12, atol=1e-12)


def test_selector_isolation_fixed_counts_and_no_early_elimination():
    assert set(inspect.signature(entry.select_seed).parameters) == {
        'terms', 'keys', 'seed', 'coarse', 'fine', 'sc', 'output', 'commit'}
    assert entry.counts() == {'updates_per_seed': 3072, 'total_updates': 9216, 'total_endpoints': 18,
        'total_checkpoints': 72, 'total_restores': 36, 'total_rankings': 54}
    assert entry.SEEDS == (6201, 6202, 6203) and entry.STEPS == (0, 32, 128, 512)
    tree = ast.parse(inspect.getsource(entry.select_seed))
    # All candidates traverse every target. Intermediate rankings cannot change
    # the model list: choose is called only after both training loops close.
    loops = [n for n in tree.body[0].body if isinstance(n, ast.For)]
    assert len(loops) == 2
    assert 'enumerate(terms)' in ast.unparse(loops[0].iter)
    nested = [n for n in ast.walk(loops[0]) if isinstance(n, ast.For)]
    assert any(ast.unparse(n.iter) == 'STEPS[1:]' for n in nested)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'choose' for n in ast.walk(loops[0]))


def test_real_small_state_count_adam_rng_identity_trace_negative(tmp_path):
    _, state, identity = tiny_checkpoint(tmp_path)  # exactly two real CPU updates
    assert entry.checks.validate_state(state, identity, 2)['updates'] == 2
    with pytest.raises(ValueError, match='count'):
        entry.checks.validate_state(state, identity, 512)
    for key, value in [('trace', []), ('identity', {}), ('torch_rng', torch.ones(3))]:
        bad = copy.deepcopy(state); bad[key] = value
        with pytest.raises(ValueError): entry.checks.validate_state(bad, identity, 2)
    bad = copy.deepcopy(state); bad['optimizer']['state'] = {}
    with pytest.raises(ValueError, match='nonempty'):
        entry.checks.validate_state(bad, identity, 2)
    bad = copy.deepcopy(state); next(iter(bad['optimizer']['state'].values()))['step'].fill_(1)
    with pytest.raises(ValueError, match='per-parameter'):
        entry.checks.validate_state(bad, identity, 2)


def test_eighteen_barrier_before_any_val(tmp_path):
    frozen = entry.freeze(tmp_path / 'pool')
    for all_done, rows in [(False, []), (True, [{'seed': 6201}]),
                           (True, [{'seed': s, 'status': 'UNRESOLVED'} for s in entry.SEEDS])]:
        selection = {'source_commit': 'a' * 40, 'selection': 'train_only', 'all_eighteen_frozen': all_done, 'rows': rows}
        with pytest.raises(ValueError): entry.validate_selection(selection, frozen, 'a' * 40)
    assert not (tmp_path / 'val').exists()


def test_interval_chain_and_all_sixteen_numerical_events():
    rows = [{'key': k, 'score': {'mean': m, 'B': 1., 'valid': True}} for k, m in [('c', 0.), ('b', 1.5), ('a', 3.)]]
    selected, ambiguous = entry.science.choose(rows, 1)
    assert ambiguous and selected[0]['overlap_group'] == ['a', 'b', 'c']
    c = np.zeros((16, 511, 30)); f = c.copy(); f[15, 0, 0] = .002
    q = entry.science.qualify_teacher(c, f, np.ones(30))
    assert not q['valid'] and q['count'] == 16 and sum(q['eligible']) == 15


def test_missing_one_of_eighteen_layouts_rejected_before_state_load(tmp_path):
    # Empty directories test only the preliminary layout barrier. They have no
    # training.pt and cannot qualify as real scientific endpoints.
    rows = []
    for group in range(3):
        folder = tmp_path / f'qa_group_{group}'; folder.mkdir()
        rows.append({'status': 'FROZEN', 'train_observations': {'path': str(folder / 'no_data.npz')}})
        for index in range(6):
            candidate = folder / f'candidate_{index}'; candidate.mkdir()
            for step in entry.STEPS: (candidate / f'checkpoint_{step:05d}').mkdir()
            (candidate / 'latest.json').write_text(json.dumps({'updates': 512, 'checkpoint': 'checkpoint_00512', 'sha256': {}}))
    entry.require_all_candidate_layouts(rows)
    (tmp_path / 'qa_group_2/candidate_5/checkpoint_00512').rmdir()
    with pytest.raises(ValueError, match='unknown/pending/future'):
        entry.require_all_candidate_layouts(rows)


def test_real_9922_checkpoint_32_to_34_and_common_zero_initialization(tmp_path):
    from closure_causal_history_residual_20261019 import checkpoint, restore, train_update, prediction
    torch.set_num_threads(1); assert torch.cuda.is_available(), 'registered CUDA QA cannot skip'
    observations = np.random.default_rng(9922).normal(0, .01, (16, 61, 30))
    pool = entry.generate_pool()
    first, opt = entry.science.make(pool['selected'][0], 9922, observations)
    second, _ = entry.science.make(pool['selected'][1], 9922, observations)
    assert torch.count_nonzero(first.base.coefficients) == 0
    assert torch.count_nonzero(second.base.coefficients) == 0
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    del second
    rng = np.random.default_rng(np.random.SeedSequence([9922, 7]))
    observed = torch.as_tensor(observations, device='cuda'); scales = torch.ones(30, device='cuda', dtype=torch.float64)
    trace = []; identity = {'qa_only': True, 'seed': 9922, 'namespace': 'independent observations'}
    for _ in range(32):
        value, _ = train_update(first, opt, rng, observed, scales, None); trace.append(value)
    checkpoint(tmp_path, first, opt, rng, 32, trace, identity)
    for _ in range(2):
        value, _ = train_update(first, opt, rng, observed, scales, None); trace.append(value)
    expected = capture(first, opt, rng)
    with torch.no_grad(): expected_output = prediction(first, observed).clone()
    torch.rand(7); torch.rand(7, device='cuda'); rng.random(7)
    step, restored_trace = restore(tmp_path, first, opt, rng, identity)
    assert step == 32
    for _ in range(2):
        value, _ = train_update(first, opt, rng, observed, scales, None); restored_trace.append(value)
    exact(expected, capture(first, opt, rng)); assert trace == restored_trace
    with torch.no_grad(): actual_output = prediction(first, observed)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    # Check the production evaluation load after the exact restoration proof;
    # this adds no updates and cannot alter that proof's RNG comparisons.
    loaded, complexity = entry.load_for_evaluation(tmp_path / 'checkpoint_00032/model.json')
    assert loaded.mass.is_cuda and isinstance(complexity, dict)
    assert 'symbolic_closure' in complexity
    json.dumps(complexity, allow_nan=False)
