"""Audit frozen cached arrays without importing kiki, runner or solver code."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = 'gftd_bounded_initialization_20261109'
RESULTS = Path('results') / RUN
ARTIFACTS = Path('artifacts') / RUN
PILOT = {0, 8, 10, 27, 57, 72, 89, 99, 117, 124, 127, 129, 133, 138, 152, 181, 204, 221, 236, 251, 265, 267, 268, 282, 296}
GRID = np.linspace(0., 1., 101)
LEDGER = {}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def check(condition, message):
    if not condition:
        raise AssertionError(message)

def read(path, digest, size=None):
    path = Path(path)
    target = (ROOT / path).resolve()
    check(target.is_relative_to(ROOT) and 'private_data' not in path.parts, 'read boundary')
    raw = target.read_bytes()
    check(sha(raw) == digest and (size is None or len(raw) == size), str(path))
    LEDGER[path.as_posix()] = dict(sha256=digest, bytes=len(raw))
    return raw

def ref(item, expected):
    check(item['path'] == Path(expected).as_posix(), 'reference path')
    return read(expected, item['sha256'], item['bytes'])

def arrays(item, expected):
    with np.load(io.BytesIO(ref(item, expected)), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}

def close(actual, expected, name):
    if actual is None or expected is None:
        check(actual is None and expected is None, name)
    else:
        check(np.isclose(actual, expected, atol=3e-13, rtol=2e-11), f'{name}: {actual} != {expected}')

def observe(states, mass):
    theta = states[:, :len(mass)].copy()
    theta -= np.average(theta, weights=mass, axis=1)[:, None]
    return np.concatenate((theta, states[:, len(mass):2*len(mass)]), axis=1) / .1

def compare(left, right, mass, completed):
    check(left.shape == right.shape == (101, 50), 'reference shape')
    count = 0
    for x, y in zip(left, right):
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            break
        count += 1
    check(count > 0, 'initial point missing')
    check(np.isnan(right[count:]).all(), 'reference is not a continuous saved prefix')
    x, y = left[:count], right[:count]
    difference = observe(x, mass) - observe(y, mass)
    rms = float(np.sqrt(np.sum(difference*difference)/difference.size))
    peak = float(np.max(np.abs(difference)))
    relative = float(np.max(np.abs(x-y)/(1. + np.maximum(np.abs(x), np.abs(y)))))
    return dict(common_rows=count, common_through_s=float(GRID[count-1]), observation_rms=rms,
                observation_peak=peak, full_state_max_relative=relative,
                confirmed=bool(completed and count == 101 and rms <= 1e-6 and peak <= 1e-5 and relative <= 1e-5))

def check_gate(record):
    if record is None or 'error' in record:
        return False
    values = [*record['roundtrip_scaled_max'], record['history_main_fine_rms'],
              record['future_main_fine_rms'], record['future_main_fine_peak'], record['p_endpoint_error']]
    passed = all(np.isfinite(v) and v <= cap for v, cap in zip(values, (1e-7, 1e-7, 1e-7, 1e-5, 1e-4, 0.)))
    check(passed == record['passed'], 'recorded six-gate decision')
    return passed

def audit_job(index, item, old, prepared, mass):
    phase = Path(item['path']).parts[-3]
    check(phase in ('pilot', 'full'), 'job phase')
    job = json.loads(ref(item, RESULTS / phase / 'jobs' / f'{index:03d}.json'))
    check(job['status'] == 'completed' and job['index'] == index, 'completed job')
    check(job['event_id'] == old['event_id'] and job['label'] == old['label'], 'job event identity')
    check(job['model_sha256'] == prepared['model']['sha256'] and job['config_sha256'] == prepared['config_sha256'], 'job model identity')
    claim = json.loads(ref(job['claim'], RESULTS / phase / 'claims' / f'{index:03d}.json'))
    for key in ('run', 'phase', 'action_id', 'release_sha256', 'prepare_sha256', 'index', 'event_id', 'label', 'input_sha256', 'model_sha256', 'config_sha256'):
        check(claim[key] == job[key], f'claim {key}')
    folder = ARTIFACTS / phase / f'{index:03d}'
    diagnostic = json.loads(gzip.decompress(ref(job['diagnostics'], folder / 'diagnostics.json.gz')))
    saved = {} if job['initialization_arrays'] is None else arrays(job['initialization_arrays'], folder / 'initialization.npz')
    if saved:
        check(np.array_equal(saved['future_times'], GRID), 'DP8 times')
    counts = job['initialization_counts']
    check(counts == diagnostic['counts'], 'counts in original diagnostics')
    residuals = 0
    for name, cap in (('stage1', 600), ('stage2', 1200)):
        stage = diagnostic[name]
        if stage is None:
            continue
        count = stage['actual_residual_calls']
        check(count == stage['counts']['actual_residual_calls'] and 0 <= count <= cap, 'residual cap')
        check(stage['counts']['actual_rhs_calls'] <= count*480, 'historical RHS cap')
        residuals += count
    check(residuals == counts['actual_residual_calls'], 'residual sum')
    check(12*counts['fixed_dp8_steps_completed'] <= counts['actual_rhs_calls'] <= 12*counts['fixed_dp8_steps_attempted'] <= 480*residuals+12960, 'DP8 work cap')
    check(sum(job['initialization_rhs_by_operation'].values()) == counts['actual_rhs_calls'], 'RHS operation sum')
    first_pass, final_pass = check_gate(job['stage1_gates']), check_gate(job['final_gates'])
    for key, endpoint_sha in job['dp8_capture_endpoints'].items():
        path = saved[key]
        check(path.shape == (101, 50) and path.dtype == np.float64 and np.isfinite(path).all(), 'stored DP8 states')
        check(sha(path[0].tobytes()) == endpoint_sha, 'DP8 endpoint hash')
    future_checks = 0
    for number in (1, 2):
        main, fine = f'dp8_gate{number}_main', f'dp8_gate{number}_fine'
        if main in saved and fine in saved:
            delta = observe(saved[main], mass) - observe(saved[fine], mass)
            gate = job['stage1_gates'] if number == 1 else job['final_gates']
            if 'future_main_fine_rms' in gate:
                close(np.sqrt(np.mean(delta**2)), gate['future_main_fine_rms'], 'future gate rms')
                close(np.max(np.abs(delta)), gate['future_main_fine_peak'], 'future gate peak')
            future_checks += 1
    formal = job['formal_point']
    check(job['six_gates_passed'] == (formal is not None), 'formal flag')
    comparison = job['comparison_stage']
    history = None if comparison is None else diagnostic[comparison]['best']['history_mse']
    result = dict(index=index, event_id=old['event_id'], label=old['label'], six_gates_passed=formal is not None,
                  history_mse=history, selected_stage=job['selected_stage'], future_gate_comparisons=future_checks,
                  reference_status=job['reference']['status'], fine_reference_confirmed=None, main_reference_confirmed=None)
    if formal is None:
        check(job['reference']['status'] == 'not_run', 'reference without formal endpoint')
        return result, job
    check(first_pass and final_pass and formal['p0_exact'], 'formal qualification')
    state = saved['formal_state']
    selected = job['selected_stage']
    check(sha(state.tobytes()) == formal['state_sha256'], 'formal state hash')
    check(np.array_equal(state, np.asarray(diagnostic[selected]['best']['state'])), 'selected original point')
    loss = diagnostic['stage2']['best']['observation_sse']
    baseline = diagnostic['stage2']['same_step_feasible_sse']
    improved = baseline is not None and loss < baseline-1e-14*max(1., abs(baseline))
    check(selected == ('stage2' if improved else 'stage1'), 'registered selection rule')
    number = 1 if job['final_check_reused_stage1'] else 2
    reference = job['reference']
    check(reference['actual_rhs_calls'] == sum(reference['rhs_calls_by_operation'].values()) <= 24000, 'reference callback budget')
    check((reference['rtol'], reference['atol'], reference['max_step']) == (1e-11, 1e-13, .005), 'strict settings')
    radau = arrays(job['reference_arrays'], folder / 'radau.npz')
    check(np.array_equal(radau['times'], GRID) and np.array_equal(radau['states'][0], state), 'reference times/state')
    for kind in ('main', 'fine'):
        path = saved[f'dp8_gate{number}_{kind}']
        check(np.array_equal(path[0], state), 'same initial state')
        metrics = compare(path, radau['states'], mass, reference['status'] == 'completed')
        for key, value in metrics.items():
            if key in ('confirmed', 'common_rows'):
                check(value == reference[kind][key], f'reference {kind}/{key}')
            else:
                close(value, reference[kind][key], f'reference {kind}/{key}')
        check(metrics['common_rows'] == reference['output_rows'], 'saved continuous reference rows')
        close(metrics['common_through_s'], reference['last_output_time'], 'reference window')
        result[kind+'_reference_confirmed'] = metrics['confirmed']
        result[kind+'_metrics'] = metrics
    return result, job

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'full'), required=True)
    parser.add_argument('--execution-sha', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    execution = json.loads(read(RESULTS/args.phase/'execution.json', args.execution_sha))
    check(execution['status'] == 'completed' and not execution['partial_jobs'] and not execution['unknown_jobs'] and not execution['unclaimed_jobs'], 'complete execution')
    prepared = json.loads(read(RESULTS/'prepare.json', execution['bindings']['prepare_sha256']))
    qualification = json.loads(read(RESULTS/'qualification.json', execution['bindings']['qualification_sha256']))
    check(qualification['passed'] and not qualification['violations'], 'qualification')
    check(qualification['source_sha256'] == prepared['source_sha256'], 'qualified source identity')
    for path, digest in prepared['source_sha256'].items():
        read(path, digest)
    for path, digest in prepared['frozen_sha256'].items():
        read(path, digest)
    for name in ('design_adoption', 'implementation_review'):
        item = execution['bindings'][name]
        ref(item, item['path'])
    ref(execution['claim'], RESULTS/args.phase/'started.json')
    model = json.loads(ref(prepared['model'], ARTIFACTS/'candidate_model.json'))
    check(model['format'] == 'gftd_kuramoto_5n_v3' and model['functional']['modulation_kernel'] == 'bounded_exp_tanh_v1', 'kernel')
    mass = np.asarray(model['functional']['mass'])
    oldpath = 'results/gftd_initialization_precision_20261109/candidate/full/rows.json'
    old = json.loads(read(oldpath, prepared['frozen_sha256'][oldpath]))
    old_exp = {row['index']: row for row in old if row['mode'] == 'modulated'}
    old_static = {row['index']: row for row in old if row['mode'] == 'static'}
    check(set(old_exp) == set(old_static) == set(range(297)), 'original denominator')
    new = {int(key): item for key, item in execution['jobs'].items()}
    reused = {int(key): item for key, item in execution['reused_pilot'].items()}
    planned = PILOT if args.phase == 'pilot' else set(range(297))-PILOT
    check(set(new) == set(execution['planned_jobs']) == planned, 'planned one-time jobs')
    check(set(reused) == (set() if args.phase == 'pilot' else PILOT), 'pilot reuse set')
    references = reused | new
    table = json.loads(ref(execution['rows'], RESULTS/args.phase/'rows.json'))
    check([row['index'] for row in table] == list(range(297)), 'full table order')
    audited, new_jobs = [], []
    for index, row in enumerate(table):
        check(row['event_id'] == old_exp[index]['event_id'] and row['label'] == old_exp[index]['label'], 'table identity')
        for name, original in (('old_exp', old_exp[index]), ('old_static', old_static[index])):
            check(row[name]['passed'] == original['candidate']['passed'], 'old pass flag')
            close(row[name]['history_mse'], original['candidate']['history_mse']['candidate'], 'old DP8 historical point')
        if index not in references:
            check(row['evidence'] is None and row['six_gates_passed'] is None and 'not_run' in row['tags'], 'unrun stays null')
            continue
        check(row['evidence'] == references[index], 'row evidence')
        result, job = audit_job(index, references[index], old_exp[index], prepared, mass)
        for field in ('six_gates_passed', 'fine_reference_confirmed', 'main_reference_confirmed'):
            check(result[field] == row[field], f'row {field}')
        close(result['history_mse'], row['history_mse'], 'new history')
        previous = row['old_exp']['history_mse']
        paired = None if previous is None or result['history_mse'] is None else result['history_mse']-previous
        close(paired, row['paired_history_mse_difference'], 'paired history difference')
        result['paired_history_mse_difference'] = paired
        audited.append(result)
        if index in new:
            check(all(job[key] == value for key, value in execution['action'].items()), 'phase action')
            new_jobs.append(job)
    totals = dict(residual_calls=sum(j['initialization_counts']['actual_residual_calls'] for j in new_jobs),
                  initialization_rhs_calls=sum(j['initialization_counts']['actual_rhs_calls'] for j in new_jobs),
                  reference_rhs_calls=sum(j['reference'].get('actual_rhs_calls', 0) for j in new_jobs),
                  worker_wall_seconds=sum(j['worker_wall_seconds'] for j in new_jobs),
                  worker_cpu_seconds=sum(j['worker_cpu_seconds'] for j in new_jobs))
    for key, value in totals.items():
        close(value, execution['known_new_costs'][key], f'known cost {key}')
        close(value, execution['actual_new_costs'][key], f'actual cost {key}')
    summaries = {}
    for group, denominator in (('all', 297), ('stable', 267), ('unstable', 30)):
        subset = [r for r in audited if group == 'all' or r['label'] == group]
        original = execution['summary'][group]
        check(original['denominator'] == denominator and original['completed'] == len(subset), 'summary denominator')
        current = dict(denominator=denominator, executed=len(subset), reference_status=dict(Counter(r['reference_status'] for r in subset)))
        for field in ('six_gates_passed', 'fine_reference_confirmed', 'main_reference_confirmed'):
            ids = [r['event_id'] for r in subset if r[field] is True]
            check(ids == original[field], f'summary {group}/{field}')
            current[field] = ids
        newly = [r['event_id'] for r in subset if r['six_gates_passed'] and not old_exp[r['index']]['candidate']['passed']]
        lost = [r['event_id'] for r in subset if not r['six_gates_passed'] and old_exp[r['index']]['candidate']['passed']]
        check(newly == original['newly_passed'] and lost == original['lost_pass'], 'paired pass identities')
        current.update(newly_passed=newly, lost_pass=lost)
        pairs = [r for r in subset if r['paired_history_mse_difference'] is not None]
        check([r['event_id'] for r in pairs] == original['paired_history_ids'], 'history subset identity')
        differences = [r['paired_history_mse_difference'] for r in pairs]
        current['paired_history'] = dict(count=len(pairs), mean=None if not pairs else float(np.mean(differences)), median=None if not pairs else float(np.median(differences)))
        summaries[group] = current
    eligible = args.phase == 'pilot' and any(r['six_gates_passed'] for r in audited)
    check(eligible == execution['eligible_for_full'], 'registered full trigger')
    output = dict(run=RUN, phase=args.phase, passed=True, execution_sha256=args.execution_sha,
                  audit_source_sha256=sha(Path(__file__).read_bytes()), checked_files=len(LEDGER), file_identities=LEDGER,
                  rows=audited, summaries=summaries, new_costs=totals, eligible_for_full=eligible,
                  arithmetic=dict(dp8_future_comparisons=sum(r['future_gate_comparisons'] for r in audited),
                                  reference_comparisons=sum(2 for r in audited if r['fine_reference_confirmed'] is not None)),
                  new_rhs=0, new_hamiltonian=0, new_gradients=0, new_ode=0, original_input_reads=0,
                  limits=['Historical roundtrip metrics are checked against stored decisions; four raw gate histories are not archived.',
                          'p0 exactness relies on the reviewed execution-time anchor check; original input is not reread.'],
                  audit_wall_seconds=time.perf_counter()-started)
    path = ROOT/RESULTS/(args.phase+'_independent_audit.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, phase=args.phase, rows=len(audited), checked_files=len(LEDGER), new_costs=totals, output_sha256=sha(path.read_bytes()))))

if __name__ == '__main__':
    main()
