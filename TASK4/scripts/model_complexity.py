"""Static polynomial tree counts and the factored 50-component T4-S RHS.

This reads one explicit symbolic JSON, never checkpoints or event payloads, and
never instantiates or runs a model. Counts concern the 34 shared scalar
polynomials only, with binary add/multiply ASTs and no common-subexpression
elimination. The full RHS is displayed in factors, not distributed or counted.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from task4.paths import ROOT, project_path, write_json


MODULES = {
    'node_energy': (7, 1), 'edge_energy': (16, 1),
    'node_exchange': (8, 6), 'node_dissipation': (8, 4),
    'edge_exchange': (18, 16), 'edge_dissipation': (18, 4), 'port': (8, 2),
}
STATE_ORDER = ['delta[N]', 'pi[N]', 'p[N]', 'q[N,rank] flattened machine-major']
AST_RULE = {
    'representation': 'separate expanded binary expression tree for each shared scalar polynomial',
    'term': 'signed coefficient literal multiplied left-to-right by each variable factor with nonzero exponent',
    'exponent_one': 'one variable leaf',
    'exponent_above_one': 'one pow operator, one variable leaf, one integer exponent leaf',
    'multiplication': 'd binary multiply operators for d nonzero variable factors, retaining coefficient +/-1',
    'addition': 'T-1 binary add operators for T nonzero terms; zero terms omitted exactly',
    'constant_term': 'one signed coefficient literal; no multiply operator',
    'zero_output': 'one literal zero leaf',
    'negation': 'sign is part of coefficient literal; no separate negate operator',
    'sharing': 'none across terms/outputs; no CSE or DAG node count',
    'excluded': 'features, analytic derivatives, centering, outer H/J/R/ports, graph copies and GRU',
}


def check_deadline(deadline):
    if time.time() >= deadline - 5:
        raise TimeoutError('Managed save deadline reached')


def finite_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Expected finite numeric ' + name)
    return value


def vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError('Invalid shape: ' + name)
    return [finite_number(item, name) for item in value]


def matrix(value, rows, columns, name):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError('Invalid shape: ' + name)
    return [vector(row, columns, name) for row in value]


def strict_json(content):
    def reject(value):
        raise ValueError('Nonfinite JSON literal: ' + value)
    return json.loads(content.decode('utf-8-sig'), parse_constant=reject)


def polynomial_complexity(spec, inputs, outputs):
    if (spec.get('type') != 'polynomial' or type(spec.get('inputs')) is not int or spec['inputs'] != inputs
            or type(spec.get('outputs')) is not int or spec['outputs'] != outputs):
        raise ValueError('Explicit polynomial dimension/type mismatch')
    powers = spec.get('powers')
    if not isinstance(powers, list) or not powers:
        raise ValueError('Nonempty explicit powers required')
    powers = matrix(powers, len(powers), inputs, 'powers')
    if any(power < 0 or power > 10000 or int(power) != power for row in powers for power in row):
        raise ValueError('Powers must be bounded nonnegative integers')
    powers = [[int(power) for power in row] for row in powers]
    if len({tuple(row) for row in powers}) != len(powers):
        raise ValueError('Duplicate monomial powers are ambiguous for active-nonzero support counts')
    coefficients = matrix(spec.get('coefficients'), len(powers), outputs, 'coefficients')
    result = {'inputs': inputs, 'scalar_outputs': outputs, 'stored_monomial_rows': len(powers),
              'dense_coefficient_slots': len(powers) * outputs,
              'support_scope': 'active nonzero coefficient support; frozen training mask unavailable',
              'frozen_training_support_slots': None, 'frozen_training_support_status': 'not_available_from_S_JSON',
              'stored_library_max_degree': max(sum(row) for row in powers), 'outputs': []}
    union = set()
    for output in range(outputs):
        support = [index for index, row in enumerate(coefficients) if row[output] != 0]
        union.update(support)
        counts = {'coefficient_leaves': len(support), 'variable_leaves': 0, 'exponent_integer_leaves': 0,
                  'power_operators': 0, 'binary_multiply_operators': 0,
                  'binary_add_operators': max(0, len(support) - 1), 'zero_literal_leaves': int(not support)}
        for index in support:
            nonzero = [power for power in powers[index] if power]
            counts['variable_leaves'] += len(nonzero)
            counts['binary_multiply_operators'] += len(nonzero)
            counts['exponent_integer_leaves'] += sum(power > 1 for power in nonzero)
            counts['power_operators'] += sum(power > 1 for power in nonzero)
        result['outputs'].append({'output_index': output, 'nonzero_coefficients': len(support),
                                  'active_nonzero_support_row_indices': support, 'active_nonzero_support_size': len(support),
                                  'frozen_training_support_slots': None,
                                  'frozen_training_support_status': 'not_available_from_S_JSON',
                                  'dense_coefficient_slots': len(powers),
                                  'unique_active_monomials': len(support),
                                  'active_max_degree': max((sum(powers[index]) for index in support), default=0),
                                  'tree_nodes': sum(counts.values()), 'tree_node_breakdown': counts})
    result.update(union_active_nonzero_support_size=len(union), union_active_nonzero_support_row_indices=sorted(union),
                  nonzero_coefficients=sum(row['nonzero_coefficients'] for row in result['outputs']),
                  active_max_degree=max(row['active_max_degree'] for row in result['outputs']),
                  scalar_tree_nodes=sum(row['tree_nodes'] for row in result['outputs']))
    if 'nonzero_per_output' in spec and spec['nonzero_per_output'] != [row['nonzero_coefficients'] for row in result['outputs']]:
        raise ValueError('Declared nonzero_per_output disagrees with exact coefficients')
    return result


def validate_structure(spec):
    if spec.get('format') != 'task4-polynomial-powergraph' or type(spec.get('version')) is not int or spec['version'] != 1:
        raise ValueError('Unsupported symbolic format/version')
    if type(spec.get('n')) is not int or spec['n'] != 10 or type(spec.get('rank')) is not int or spec['rank'] != 2:
        raise ValueError('Registered TASK4 RHS requires N=10 and rank=2')
    if spec.get('coordinates', {}).get('state_order') != STATE_ORDER:
        raise ValueError('Explicit runtime state order required')
    finite_number(spec.get('strength'), 'strength')
    vector(spec.get('scales'), 3, 'scales')
    if any(value <= 0 for value in spec['scales']):
        raise ValueError('Scales must be positive')
    physics = spec.get('physics') or {}
    for name in ('mass', 'delta_star', 'pm_star', 'damping', 'rate', 'gain'):
        vector(physics.get(name), 10, 'physics.' + name)
    if any(value <= 0 for name in ('mass', 'rate', 'gain') for value in physics[name]) or any(value < 0 for value in physics['damping']):
        raise ValueError('Invalid fixed physical parameters')
    incidence = physics.get('incidence')
    if not isinstance(incidence, list) or len(incidence) != 10 or not isinstance(incidence[0], list):
        raise ValueError('Invalid incidence shape')
    edges = len(incidence[0])
    matrix(incidence, 10, edges, 'incidence')
    expected_edges = []
    for edge in range(edges):
        positive = [i for i in range(10) if incidence[i][edge] == 1]
        negative = [i for i in range(10) if incidence[i][edge] == -1]
        if len(positive) != 1 or len(negative) != 1 or sum(incidence[i][edge] != 0 for i in range(10)) != 2:
            raise ValueError('Expected signed two-endpoint incidence')
        expected_edges.append([positive[0], negative[0]])
    if spec.get('edges') != expected_edges:
        raise ValueError('Edge order/orientation disagrees with incidence')
    vector(physics.get('coupling'), edges, 'coupling')
    vector(spec.get('log_coupling'), edges, 'log_coupling')
    vector(spec.get('phase'), edges, 'phase')
    matrix(spec.get('mu'), 10, 4, 'mu')
    if set(spec.get('functions', {})) != set(MODULES):
        raise ValueError('Exactly seven explicit polynomial modules required')
    metadata = spec.get('metadata') or {}
    if metadata.get('test_payload_read') is True or metadata.get('split') == 'test':
        raise PermissionError('TEST provenance refused')
    return edges


def initializer_counts(spec):
    enc = spec.get('encoder') or {}
    if (enc.get('type') != 'shared_gru' or enc.get('hidden_size') != 32 or enc.get('history_samples') != 11
            or enc.get('history_seconds') != .1 or enc.get('sample_dt_seconds') != .01
            or enc.get('gate_order') != ['reset', 'update', 'new']):
        raise ValueError('Registered once-only GRU32 initializer required')
    tensor_shapes = {'weight_ih_l0': (96, 9), 'weight_hh_l0': (96, 32),
                     'bias_ih_l0': (96,), 'bias_hh_l0': (96,), 'head_weight': (2, 32), 'head_bias': (2,)}
    rows = {}
    for name, shape in tensor_shapes.items():
        values = vector(enc.get(name), shape[0], name) if len(shape) == 1 else matrix(enc.get(name), *shape, name)
        flattened = values if len(shape) == 1 else [item for row in values for item in row]
        rows[name] = {'shape': list(shape), 'stored_scalar_slots': len(flattened),
                      'nonzero_scalar_values': sum(value != 0 for value in flattened)}
    return {'tensors': rows, 'stored_trainable_parameter_slots': sum(row['stored_scalar_slots'] for row in rows.values()),
            'nonzero_parameter_values': sum(row['nonzero_scalar_values'] for row in rows.values()),
            'used_once_after_100ms': True, 'included_in_polynomial_tree_nodes': False}


def coordinate_rows():
    rows = [{'index': i - 1, 'coordinate': f'delta_{i}', 'rhs': f'pi_{i}/M_{i}'} for i in range(1, 11)]
    for i in range(1, 11):
        rows.append({'index': 9 + i, 'coordinate': f'pi_{i}', 'rhs': f'-G_delta_{i} + sum_(j=1..10,b=0..3) (Jextra_({i},0),(j,b)-R_({i},0),(j,b))*E_(j,b) + p_{i} + u_({i},0)'})
    for i in range(1, 11):
        rows.append({'index': 19 + i, 'coordinate': f'p_{i}', 'rhs': f'sum_(j=1..10,b=0..3) (Jextra_({i},1),(j,b)-R_({i},1),(j,b))*E_(j,b) + u_({i},1) - gain_{i}*pi_{i}/M_{i}'})
    for i in range(1, 11):
        for latent in range(2):
            rows.append({'index': 30 + (i - 1) * 2 + latent, 'coordinate': f'q_{i},{latent}',
                         'rhs': f'sum_(j=1..10,b=0..3) (Jextra_({i},{latent + 2}),(j,b)-R_({i},{latent + 2}),(j,b))*E_(j,b)'})
    return rows


def analyze_spec(spec, deadline=float('inf')):
    edge_count = validate_structure(spec)
    modules = {}
    for name, dimensions in MODULES.items():
        check_deadline(deadline)
        modules[name] = polynomial_complexity(spec['functions'][name], *dimensions)
    result = {'status': 'complete_static_analysis', 'eligible_for_reporting': True, 'automatic_model_acceptance': False,
              'model_executed': False, 'event_payload_read': False, 'n': 10, 'latent_per_machine': 2, 'state_width': 50,
              'module_count': 7, 'shared_scalar_polynomial_count': 34, 'modules': modules, 'ast_counting_rule': AST_RULE,
              'totals': {key: sum(module[key] for module in modules.values()) for key in
                         ('dense_coefficient_slots', 'nonzero_coefficients', 'union_active_nonzero_support_size', 'scalar_tree_nodes')},
              'frozen_training_support_slots': None, 'frozen_training_support_status': 'not_available_from_S_JSON',
              'shared_dag_nodes': None, 'shared_dag_count_status': 'not_implemented',
              'fully_distributed_rhs_nodes': None, 'fully_distributed_rhs_status': 'not_implemented; no large symbolic expansion',
              'rhs_display': 'complete 50 coordinate rows with explicitly defined H gradient, Jextra, R and two ports as factors',
              'union_active_nonzero_support_total_scope': 'sum of seven module active-nonzero union sizes; no sharing across modules is inferred',
              'initializer': initializer_counts(spec), 'learned_log_coupling_parameter_slots': edge_count,
              'T4_N_parameter_count': None, 'U_parameter_count': None,
              'neural_control_count_status': 'not_available_from_S_JSON; no checkpoint/source weight loading performed',
              'coordinate_rows': coordinate_rows()}
    result['totals']['active_polynomial_coefficient_slots'] = result['totals']['nonzero_coefficients']
    return result


def number(value):
    return format(float(value), '.17g')


def rhs_markdown(spec, result):
    text = ['# TASK4 T4-S 静态复杂度与因式 RHS', '',
            '这里只展示显式结构和静态计数，不构成预测精度、速度或稳定域证据。', '',
            f"七个共享模块、34个共享标量多项式：非零系数 {result['totals']['nonzero_coefficients']}，直接二元 AST 树节点 {result['totals']['scalar_tree_nodes']}。",
            'active_nonzero_support 仅指最终导出系数非零的行/槽；JSON 保存 coefficients*support，没有冻结训练 support mask。因此 frozen_training_support_slots=None（not_available_from_S_JSON），不会从零系数反推训练支持。',
            '系数数与节点数是不同统计。此树计数不含特征、导数、中心化、图复制、GRU 或 H/J/R 外层；未实施共享 DAG 和完整 RHS 分布展开，二者节点数均未报告。', '',
            '## 确切坐标、尺度和系数', '',
            '机器编号为 1..10，局部 effort 编号为 0..3，潜坐标编号为 0..1；JSON/运行时的机器与边索引为零起点。',
            '`x[0:10]=delta_1..delta_10; x[10:20]=pi_1..pi_10; x[20:30]=p_1..p_10; x[30:50]=(q_1,0,q_1,1,...,q_10,0,q_10,1)`。',
            '`pi_i=M_i*omega_i`，时间为秒，角为 rad，omega 为 rad/s，p 为系统 pu；q 为模型储能坐标。',
            f"`alpha={number(spec['strength'])}; scale_delta={number(spec['scales'][0])}; scale_omega={number(spec['scales'][1])}; scale_p={number(spec['scales'][2])}`。",
            'scale_delta 用于初始化器；ODE 的下述变量采用实际函数输入约定。', '',
            '| i | M_i | damping_i | rate_i | gain_i | mu_i[0..3] |', '|---|---|---|---|---|---|']
    physics = spec['physics']
    for i in range(10):
        text.append('| ' + str(i + 1) + ' | ' + ' | '.join(number(physics[key][i]) for key in ('mass', 'damping', 'rate', 'gain')) + ' | ' + ', '.join(number(v) for v in spec['mu'][i]) + ' |')
    text.extend(['', '| e | i_e (+) | j_e (-) | coupling_e | log_coupling_e | phase_e |', '|---|---|---|---|---|---|'])
    for edge, (i, j) in enumerate(spec['edges']):
        text.append('| ' + ' | '.join((str(edge + 1), str(i + 1), str(j + 1), number(physics['coupling'][edge]), number(spec['log_coupling'][edge]), number(spec['phase'][edge]))) + ' |')
    text.extend(['', 'phase 和 mu 使用 JSON 中保存的实际缓冲值。`k_e=coupling_e*exp(log_coupling_e)`，`eta_e=delta_i_e-delta_j_e`，B 为上表确定的有向 incidence。', '',
        '## 储能与 G=grad(H)', '', '```text',
        's_i=(p_i/scale_p,q_i,0,q_i,1)',
        'v_i=(pi_i/(M_i*scale_omega),p_i/scale_p,q_i,0,q_i,1,mu_i[0..3])',
        'v_i^0=(0,0,0,0,mu_i[0..3])',
        'a_e(z_i,z_j)=(sin(phase_e+eta_e)-sin(phase_e), cos(phase_e+eta_e)-cos(phase_e), z_i,z_j,mu_i,mu_j)',
        'b_e(z_i,z_j)=(-sin(phase_e+eta_e)+sin(phase_e), cos(phase_e+eta_e)-cos(phase_e), z_j,z_i,mu_j,mu_i)',
        'N_i(s)=F_node_energy(s,mu_i)',
        'Ntilde_i(s)=N_i(s)-N_i(0)-grad_s(N_i)(0) dot s',
        'v_e=(eta_e,s_i_e,s_j_e)',
        'Q_e(v_e)=(F_edge_energy(a_e(s_i_e,s_j_e))+F_edge_energy(b_e(s_i_e,s_j_e)))/2',
        'Qtilde_e(v_e)=Q_e(v_e)-Q_e(0)-grad_v(Q_e)(0) dot v_e',
        'H=sum_i(pi_i^2/(2*M_i)+p_i^2/(2*gain_i)+(q_i,0^2+q_i,1^2)/2)',
        '  +sum_e k_e*(cos(phase_e)-cos(phase_e+eta_e)-sin(phase_e)*eta_e)',
        '  +alpha*(sum_i Ntilde_i(s_i)+sum_e Qtilde_e(v_e))',
        'A_i=grad_s(Ntilde_i)(s_i)+sum_(e:i_e=i) grad_s_i(Qtilde_e)+sum_(e:j_e=i) grad_s_j(Qtilde_e)',
        'G_delta_i=sum_e B_i,e*[k_e*(sin(phase_e+eta_e)-sin(phase_e))+alpha*d_eta(Qtilde_e)]',
        'G_pi_i=pi_i/M_i; G_p_i=p_i/gain_i+alpha*A_i[0]/scale_p',
        'G_q_i,0=q_i,0+alpha*A_i[1]; G_q_i,1=q_i,1+alpha*A_i[2]',
        'E_i=(G_pi_i,G_p_i,G_q_i,0,G_q_i,1)',
        'd F_o(v)/d v_d = sum_(r:powers[r,d]>0) coefficients[r,o]*powers[r,d]',
        '                    *product_l v_l^(powers[r,l]-1_(l=d))', '```', '',
        '中心化时 mu 与 phase 固定；Q 的导数包含正反向输入及 sin/cos 链式导数。`grad_v(Q_e)(0)` 中 eta、两端 s 全为零，不把 mu 清零。', '',
        '## 实际 Jextra、R 与两端口', '', '```text',
        'node_exchange scalar output order: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)',
        'S_i[a,b]=alpha*F_node_exchange(v_i)[pair(a,b)]+alpha*1_(a,b)=(0,2), for a<b',
        'S_i[b,a]=-S_i[a,b]; S_i[a,a]=0',
        'rho0_i=(sqrt(damping_i),sqrt(rate_i*gain_i),0.5,0.5)',
        'ell_i=rho0_i+alpha*F_node_dissipation(v_i)',
        'local_i=(pi_i/(M_i*scale_omega),p_i/scale_p,q_i,0,q_i,1)',
        'K_e[a,b]=(alpha/2)*(F_edge_exchange(a_e(local_i,local_j))[4*a+b]',
        '                       -F_edge_exchange(b_e(local_i,local_j))[4*b+a])',
        'lambda_e=(alpha/2)*(F_edge_dissipation(a_e(local_i,local_j))+F_edge_dissipation(b_e(local_i,local_j)))',
        'D_e=diag(lambda_e[0]^2,lambda_e[1]^2,lambda_e[2]^2,lambda_e[3]^2)/2',
        'Jextra_i,i=S_i; R_i,i=diag(ell_i[0]^2,ell_i[1]^2,ell_i[2]^2,ell_i[3]^2)',
        'For each signed edge e=(i,j), add Jextra_i,j += K_e; Jextra_j,i += -transpose(K_e)',
        'For each signed edge e=(i,j), add R_i,i += D_e; R_j,j += D_e; R_i,j -= D_e; R_j,i -= D_e',
        'All other blocks start at zero; multiple edge contributions add.',
        'u_i,0=alpha*(F_port(v_i)[0]-F_port(v_i^0)[0])',
        'u_i,1=alpha*(F_port(v_i)[1]-F_port(v_i^0)[1])', '```', '',
        'Jextra 的局部 0/1/2/3 行分别作用于 pi/p/q0/q1；canonical 的 delta-pi 配对显式写在下列 RHS。固定 `+p_i` 与 `-gain_i*omega_i` 同样显式写出，不并入 learned port。',
        '`-rate_i*p_i` 已通过 R 的 p 对角项嵌入：未学习修正时，`ell_i[1]^2=rate_i*gain_i` 且 `G_p_i=p_i/gain_i`，二者乘积给出 `-rate_i*p_i`。有能量/耗散修正时使用完整 `-ell_i[1]^2*G_p_i`；不再额外加一次 `-rate_i*p_i`。', '',
        '## 全部 50 个因式坐标行', '', '```text'])
    for row in result['coordinate_rows']:
        text.append(f"x[{row['index']}]: d({row['coordinate']})/dt = {row['rhs']}")
    text.extend(['```', '', '## 七模块的共享标量显式公式', '',
        '每一函数以 v0,v1,... 表示该函数的实际输入次序；输出为零起点。edge exchange 输出为 row-major 4x4。系数以17位有效数字打印，不阈值删除小系数。',
        '输入次序：node_energy=(s_i,mu_i)；edge_energy=a_e(s_i,s_j) 或 b_e(s_i,s_j)；node_exchange/node_dissipation/port=v_i；edge_exchange/edge_dissipation=a_e(local_i,local_j) 或 b_e(local_i,local_j)。', ''])
    for name, fn in spec['functions'].items():
        text.extend(['### ' + name, '', '```text'])
        for output in range(fn['outputs']):
            terms = []
            for powers, coefficients in zip(fn['powers'], fn['coefficients']):
                coefficient = coefficients[output]
                if coefficient == 0:
                    continue
                factors = [f'v{index}' + (f'^{int(power)}' if power != 1 else '') for index, power in enumerate(powers) if power]
                terms.append('(' + number(coefficient) + ')' + (('*' + '*'.join(factors)) if factors else ''))
            text.append(f'F_{name}[{output}] = ' + (' + '.join(terms) if terms else '0'))
        text.extend(['```', ''])
    text.extend(['## AST 计数约定', '', '```json', json.dumps(AST_RULE, ensure_ascii=False, indent=2), '```', '',
                 '本报告将每个共享标量多项式分别计数；不会将34个多项式的树节点总数冒称50分量完整 RHS 节点数。'])
    return '\n'.join(text) + '\n'


def report_files(symbolic, output):
    from task4.training import check_authorized
    check_authorized(False)
    deadline = float(os.environ['TASK4_SAVE_DEADLINE_UNIX'])
    if not math.isfinite(deadline):
        raise ValueError('Finite managed deadline required')
    check_deadline(deadline)
    source, destination = project_path(symbolic), project_path(output)
    if source.suffix.lower() != '.json':
        raise ValueError('Explicit symbolic JSON required')
    destination.mkdir(parents=True, exist_ok=False)
    result = {'status': 'incomplete', 'eligible_for_reporting': False, 'source': str(source), 'errors': []}
    write_json(destination / 'complexity.json', result)
    before = None
    code_paths = []
    try:
        content = source.read_bytes()
        before = hashlib.sha256(content).hexdigest()
        code_paths = [Path(__file__).resolve(), ROOT / 'src/task4/symbolic_runtime.py']
        code_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
        result['code_hashes_before'] = code_hashes
        spec = strict_json(content)
        result.update(analyze_spec(spec, deadline))
        result['source_sha256_before'] = before
        check_deadline(deadline)
        markdown = rhs_markdown(spec, result)
        after = hashlib.sha256(source.read_bytes()).hexdigest()
        code_after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
        result.update(source_sha256_end=after, source_unchanged=before == after, code_hashes_end=code_after,
                      code_unchanged=code_hashes == code_after)
        if not result['source_unchanged'] or not result['code_unchanged']:
            result.update(status='invalid_input_changed', eligible_for_reporting=False)
            result['errors'].append('Source or reporting/runtime code changed during static analysis')
            markdown = '> INVALID: source or code changed; these counts/formulas are not qualified for reporting.\n\n' + markdown
        check_deadline(deadline)
        (destination / 'rhs_factored.md').write_text(markdown, encoding='utf-8')
    except Exception as exc:
        result.update(status='incomplete_deadline' if isinstance(exc, TimeoutError) else 'failed', eligible_for_reporting=False)
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
        result['source_sha256_before'] = before
        if before is not None:
            try:
                result['source_sha256_end'] = hashlib.sha256(source.read_bytes()).hexdigest()
                result['source_unchanged'] = result['source_sha256_end'] == before
            except OSError:
                result['source_sha256_end'] = None
                result['source_unchanged'] = False
        if code_paths:
            try:
                result['code_hashes_end'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
                result['code_unchanged'] = result.get('code_hashes_before') == result['code_hashes_end']
            except OSError:
                result['code_unchanged'] = False
    write_json(destination / 'complexity.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbolic', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report_files(args.symbolic, args.output)
    print(json.dumps({'status': result['status'], 'eligible_for_reporting': result['eligible_for_reporting'],
                      'totals': result.get('totals'), 'errors': result['errors']}, ensure_ascii=False, allow_nan=False))
    if not result['eligible_for_reporting']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
