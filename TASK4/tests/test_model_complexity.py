"""Static synthetic JSONs only; no models, checkpoints, events, or solver."""
import importlib.util
import json
from pathlib import Path
import time

import pytest

from task4.paths import ROOT


@pytest.fixture
def complexity(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('complexity_test', ROOT / 'scripts/model_complexity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zero_matrix(rows, columns):
    return [[0.] * columns for _ in range(rows)]


def constructed_spec(module):
    incidence = [[0.] for _ in range(10)]
    incidence[0][0], incidence[1][0] = 1., -1.
    spec = {'format': 'task4-polynomial-powergraph', 'version': 1, 'n': 10, 'rank': 2,
            'strength': .01, 'scales': [.4, 1.5, .5], 'edges': [[0, 1]],
            'phase': [.1], 'log_coupling': [.02], 'mu': zero_matrix(10, 4),
            'coordinates': {'state_order': module.STATE_ORDER}, 'functions': {},
            'physics': {'incidence': incidence, 'mass': [2.] * 10, 'delta_star': [0.] * 10,
                        'pm_star': [1.] * 10, 'damping': [.2] * 10, 'rate': [3.] * 10,
                        'gain': [4.] * 10, 'coupling': [5.]},
            'encoder': {'type': 'shared_gru', 'hidden_size': 32, 'history_samples': 11,
                        'history_seconds': .1, 'sample_dt_seconds': .01,
                        'gate_order': ['reset', 'update', 'new'],
                        'weight_ih_l0': zero_matrix(96, 9), 'weight_hh_l0': zero_matrix(96, 32),
                        'bias_ih_l0': [0.] * 96, 'bias_hh_l0': [0.] * 96,
                        'head_weight': zero_matrix(2, 32), 'head_bias': [0.] * 2}}
    for name, (inputs, outputs) in module.MODULES.items():
        powers = [[0] * inputs for _ in range(3)]
        powers[1][0], powers[2][1] = 2, 1
        coefficients = zero_matrix(3, outputs)
        coefficients[0][0], coefficients[1][0], coefficients[2][0] = 2., 1., -1.
        spec['functions'][name] = {'type': 'polynomial', 'inputs': inputs, 'outputs': outputs,
                                   'powers': powers, 'coefficients': coefficients}
    return spec


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, allow_nan=False), encoding='utf-8')


def allow_constructed(module, monkeypatch):
    monkeypatch.setattr('task4.training.check_authorized', lambda synthetic: None)
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', str(time.time() + 60))
    monkeypatch.setattr(module, 'project_path', lambda value: Path(value))
    monkeypatch.setattr(module, 'write_json', write_json)


def test_binary_ast_counts_literals_powers_multiply_add_and_zero_output(complexity):
    spec = {'type': 'polynomial', 'inputs': 2, 'outputs': 3,
            'powers': [[0, 0], [1, 1], [2, 0]],
            'coefficients': [[1e-20, 0., 0.], [1., 0., 0.], [-1., 0., 0.]]}
    counted = complexity.polynomial_complexity(spec, 2, 3)
    output = counted['outputs'][0]
    # Constant:1; c*v0*v1: coefficient+2 variables+2 multiplies=5;
    # c*pow(v0,2): coefficient+variable+exponent+pow+multiply=5;
    # two binary additions:2. +/-1 remain explicit coefficient literals.
    assert output['tree_nodes'] == 13
    assert output['tree_node_breakdown'] == {
        'coefficient_leaves': 3, 'variable_leaves': 3, 'exponent_integer_leaves': 1,
        'power_operators': 1, 'binary_multiply_operators': 3,
        'binary_add_operators': 2, 'zero_literal_leaves': 0}
    assert [row['tree_nodes'] for row in counted['outputs']] == [13, 1, 1]
    assert counted['scalar_tree_nodes'] == 15
    assert counted['nonzero_coefficients'] == counted['union_active_nonzero_support_size'] == 3
    assert counted['dense_coefficient_slots'] == 9
    assert output['active_nonzero_support_row_indices'] == [0, 1, 2] and output['active_max_degree'] == 2
    assert output['frozen_training_support_slots'] is None
    assert counted['frozen_training_support_status'] == 'not_available_from_S_JSON'


def test_final_json_is_counted_without_claiming_rhs_or_dag_nodes(complexity):
    result = complexity.analyze_spec(constructed_spec(complexity))
    assert result['module_count'] == 7 and result['shared_scalar_polynomial_count'] == 34
    assert result['totals']['nonzero_coefficients'] == 21
    assert result['totals']['dense_coefficient_slots'] == 102
    assert result['totals']['union_active_nonzero_support_size'] == 21
    assert result['totals']['scalar_tree_nodes'] == 104
    assert result['shared_dag_nodes'] is result['fully_distributed_rhs_nodes'] is None
    assert result['initializer']['stored_trainable_parameter_slots'] == 4194
    assert result['initializer']['nonzero_parameter_values'] == 0
    assert result['learned_log_coupling_parameter_slots'] == 1
    assert result['T4_N_parameter_count'] is result['U_parameter_count'] is None
    assert not result['model_executed'] and not result['automatic_model_acceptance']
    assert result['frozen_training_support_slots'] is None
    assert result['frozen_training_support_status'] == 'not_available_from_S_JSON'


def test_final_zero_coefficient_does_not_recover_frozen_training_support(complexity):
    spec = {'type': 'polynomial', 'inputs': 1, 'outputs': 1,
            'powers': [[0], [1]], 'coefficients': [[0.], [0.]]}
    result = complexity.polynomial_complexity(spec, 1, 1)
    assert result['nonzero_coefficients'] == result['union_active_nonzero_support_size'] == 0
    assert result['outputs'][0]['active_nonzero_support_size'] == 0
    # The same all-zero JSON can arise from an absent mask or supported
    # coefficients reaching zero. Neither possibility is inferred here.
    assert result['frozen_training_support_slots'] is None
    assert result['outputs'][0]['frozen_training_support_slots'] is None
    assert 'support_size' not in result['outputs'][0]


def test_all_fifty_coordinate_rows_match_runtime_machine_major_order(complexity):
    spec = constructed_spec(complexity)
    result = complexity.analyze_spec(spec)
    rows = result['coordinate_rows']
    assert len(rows) == 50 and [row['index'] for row in rows] == list(range(50))
    assert rows[0] == {'index': 0, 'coordinate': 'delta_1', 'rhs': 'pi_1/M_1'}
    assert rows[10]['coordinate'] == 'pi_1' and '+ p_1 + u_(1,0)' in rows[10]['rhs']
    assert rows[20]['coordinate'] == 'p_1' and '- gain_1*pi_1/M_1' in rows[20]['rhs']
    assert [rows[index]['coordinate'] for index in (30, 31, 48, 49)] == ['q_1,0', 'q_1,1', 'q_10,0', 'q_10,1']
    rendered = complexity.rhs_markdown(spec, result)
    assert rendered.count('/dt = ') == 50
    assert '1_(a,b)=(0,2)' in rendered
    assert '[4*b+a]' in rendered and 'Jextra_j,i += -transpose(K_e)' in rendered
    assert 'R_i,j -= D_e; R_j,i -= D_e' in rendered
    assert 'sqrt(rate_i*gain_i)' in rendered and '不再额外加一次' in rendered
    assert 'grad_v(Q_e)(0)' in rendered and 'd F_o(v)/d v_d' in rendered


def test_render_keeps_tiny_coefficients_and_minus_one(complexity):
    spec = constructed_spec(complexity)
    spec['functions']['port']['coefficients'][0][1] = 1e-20
    result = complexity.analyze_spec(spec)
    text = complexity.rhs_markdown(spec, result)
    assert 'F_port[1] = (' + format(1e-20, '.17g') + ')' in text
    assert '(-1)*v1' in text
    assert result['totals']['nonzero_coefficients'] == 22


@pytest.mark.parametrize('corruption', ['rank', 'state_order', 'edge_sign', 'missing_function', 'nonfinite', 'duplicate_monomial'])
def test_static_structure_is_rejected_fail_closed(complexity, corruption):
    spec = constructed_spec(complexity)
    if corruption == 'rank':
        spec['rank'] = 1
    elif corruption == 'state_order':
        spec['coordinates']['state_order'] = ['other order']
    elif corruption == 'edge_sign':
        spec['edges'] = [[1, 0]]
    elif corruption == 'missing_function':
        spec['functions'].pop('port')
    elif corruption == 'nonfinite':
        spec['functions']['port']['coefficients'][0][0] = float('inf')
    else:
        spec['functions']['port']['powers'][1] = list(spec['functions']['port']['powers'][0])
    with pytest.raises(ValueError):
        complexity.analyze_spec(spec)


def test_authorization_and_finite_deadline_precede_any_path_access(complexity, monkeypatch):
    def deny(_):
        raise PermissionError('not authorized')
    monkeypatch.setattr('task4.training.check_authorized', deny)
    monkeypatch.setattr(complexity, 'project_path', lambda value: pytest.fail('Path touched before gate'))
    with pytest.raises(PermissionError):
        complexity.report_files('poisoned.json', 'output')
    monkeypatch.setattr('task4.training.check_authorized', lambda synthetic: None)
    monkeypatch.setenv('TASK4_SAVE_DEADLINE_UNIX', 'inf')
    with pytest.raises(ValueError, match='Finite managed deadline'):
        complexity.report_files('poisoned.json', 'output')


def test_static_report_hashes_source_and_code_and_refuses_existing_output(tmp_path, complexity, monkeypatch):
    allow_constructed(complexity, monkeypatch)
    source = tmp_path / 'constructed.json'
    write_json(source, constructed_spec(complexity))
    result = complexity.report_files(source, tmp_path / 'report')
    assert result['eligible_for_reporting'] and result['source_unchanged'] and result['code_unchanged']
    assert result['source_sha256_before'] == result['source_sha256_end']
    assert (tmp_path / 'report/rhs_factored.md').exists()
    with pytest.raises(FileExistsError):
        complexity.report_files(source, tmp_path / 'report')


def test_source_mutation_revokes_count_and_formula_reporting_qualification(tmp_path, complexity, monkeypatch):
    allow_constructed(complexity, monkeypatch)
    source = tmp_path / 'constructed.json'
    write_json(source, constructed_spec(complexity))
    original = complexity.analyze_spec
    def mutate(spec, deadline):
        result = original(spec, deadline)
        source.write_text(source.read_text(encoding='utf-8') + '\n', encoding='utf-8')
        return result
    monkeypatch.setattr(complexity, 'analyze_spec', mutate)
    result = complexity.report_files(source, tmp_path / 'changed')
    assert result['status'] == 'invalid_input_changed' and not result['eligible_for_reporting']
    assert not result['source_unchanged'] and result['source_sha256_before'] != result['source_sha256_end']
    assert (tmp_path / 'changed/rhs_factored.md').read_text(encoding='utf-8').startswith('> INVALID')


def test_nonfinite_json_literal_fails_with_strict_finite_saved_report(tmp_path, complexity, monkeypatch):
    allow_constructed(complexity, monkeypatch)
    source = tmp_path / 'invalid.json'
    source.write_text('{"strength": NaN}', encoding='utf-8')
    result = complexity.report_files(source, tmp_path / 'invalid_report')
    assert result['status'] == 'failed' and not result['eligible_for_reporting']
    assert 'Nonfinite JSON literal' in result['errors'][0]
    json.dumps(result, allow_nan=False)


def test_rendering_past_managed_deadline_cannot_publish_complete_report(tmp_path, complexity, monkeypatch):
    allow_constructed(complexity, monkeypatch)
    source = tmp_path / 'constructed.json'
    write_json(source, constructed_spec(complexity))
    rendered = False
    original_render = complexity.rhs_markdown
    original_check = complexity.check_deadline
    def render(spec, result):
        nonlocal rendered
        text = original_render(spec, result)
        rendered = True
        return text
    def expired_after_render(deadline):
        if rendered:
            raise TimeoutError('constructed rendering exhausted managed deadline')
        original_check(deadline)
    monkeypatch.setattr(complexity, 'rhs_markdown', render)
    monkeypatch.setattr(complexity, 'check_deadline', expired_after_render)
    result = complexity.report_files(source, tmp_path / 'expired')
    assert result['status'] == 'incomplete_deadline' and not result['eligible_for_reporting']
    assert not (tmp_path / 'expired/rhs_factored.md').exists()
