"""Fixed original thirteen-mode comparison; no events, FOM solve, or model tuning.

Port sources (read-only; never imported or executed):
  kiki/experiments/evaluate_local_memory_0098.py
    SHA256 386a8cbd964618602fe7298d6a20c47ec876e9ad70e011c96f3f55e6c44d5034
  kiki/experiments/compare_physical_modes.py
    SHA256 0688c0c69778399524065152917c01de13348a4a55c414731454dd9af5c0f3ce

The original physical output, QA thresholds, positive-imaginary candidates,
pair cost, and Hungarian dummy columns are preserved. TASK4 pi is converted to
omega before QA and modal analysis. Added checks only reject nonfinite evidence,
malformed inputs, changed sources, or incomplete managed execution.
"""
import os
if __name__ == '__main__':
    for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[_key] = '1'

import _bootstrap  # noqa: F401
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from scipy.linalg import block_diag, null_space
from scipy.optimize import linear_sum_assignment

from task4.paths import ROOT, project_path, read_json, source_file, source_root, write_json

TARGET_IDS = [62, 64, 66, 68, 71, 73, 77, 79, 84, 91, 98, 100, 104]
OMEGA_S = 2 * np.pi * 60.0
S_DELTA, S_OMEGA = 0.47675314588683515, 1.0033667249602924
FREQ_GATE, DAMP_GATE, MAC_GATE = .15, .05, .8
THRESHOLDS = {'frequency_relative': FREQ_GATE, 'damping_absolute': DAMP_GATE, 'mac': MAC_GATE}
CACHE_RELATIVE = 'private_data/task2_control_modes_0078'
PORT_SOURCES = {
    'experiments/evaluate_local_memory_0098.py': '386a8cbd964618602fe7298d6a20c47ec876e9ad70e011c96f3f55e6c44d5034',
    'experiments/compare_physical_modes.py': '0688c0c69778399524065152917c01de13348a4a55c414731454dd9af5c0f3ce',
}


def check_deadline(deadline):
    if not np.isfinite(deadline) or time.time() + 5 >= deadline:
        raise TimeoutError('Managed modal save deadline reached or unavailable')


def fixed_cache(config):
    """Keep the original cache identity while respecting the authorized source root."""
    return source_file(source_root(config), CACHE_RELATIVE)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def json_safe(value, issues=None, path='$'):
    issues = [] if issues is None else issues
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        issues.append({'path': path, 'reason': 'nonfinite numeric diagnostic', 'representation': str(value)})
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v, issues, path+'.'+str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, issues, path+f'[{i}]') for i, v in enumerate(value)]
    return value


def require_finite(value, description):
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite ' + description)
    return value


def checked_norm(value, description):
    with np.errstate(over='ignore', invalid='ignore'):
        norm = float(np.linalg.norm(value))
    require_finite(norm, description)
    return norm


def physical_output(vector, mass, embed=None, omega_scale=1.0):
    """Exact original readout: COI delta / Sdelta and FULL omega / Somega."""
    full = np.asarray(vector) if embed is None else np.asarray(embed) @ np.asarray(vector)
    mass = np.asarray(mass)
    if mass.shape != (10,) or full.ndim != 1 or len(full) < 20:
        raise ValueError('Original readout requires ten delta and ten omega states')
    require_finite(full, 'physical-output state vector')
    require_finite(mass, 'physical masses')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        delta, omega = full[:10], full[10:20] * omega_scale
        coi = delta - np.sum(mass * delta) / np.sum(mass)
        output = np.r_[coi / S_DELTA, omega / S_OMEGA]
    return require_finite(output, 'physical output arithmetic')


def interface_field(model, interface):
    """Similarity coordinates u=(delta,omega,p,q), internal z=(delta,M*omega,p,q)."""
    n = model.n
    lift = torch.cat((torch.ones(n, dtype=interface.dtype, device=interface.device),
                      model.mass.to(interface), torch.ones(model.dim-2*n, dtype=interface.dtype, device=interface.device)))
    state = interface * lift
    return model.field(state.unsqueeze(0), create_graph=torch.is_grad_enabled())[0] / lift


def common_angle_embedding(dim, n=10):
    if dim < 3*n or n < 2:
        raise ValueError('Expected delta, omega, p and optional latent coordinates')
    return block_diag(null_space(np.ones((1, n))), np.eye(dim-n))


def jacobian_qa(model, deadline, *, arrays=None, qa=None):
    """Original unscaled interface-coordinate thresholds and finite differences."""
    arrays = {} if arrays is None else arrays
    qa = {} if qa is None else qa
    qa.update({'pass': False, 'finite': False, 'f0_gate': False, 'gauge_gate': False,
               'fd_gate': False, 'difference_steps': [1e-5, 1e-6], 'fd_columns_completed': {}})
    check_deadline(deadline)
    dim = model.dim
    zero = torch.zeros(dim, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        f0 = interface_field(model, zero.detach())
        shifted = zero.detach().clone()
        shifted[:model.n] += .37
        gauge = interface_field(model, shifted) - f0
    arrays['origin_field'], arrays['gauge_field_difference'] = f0.numpy(), gauge.numpy()
    require_finite(arrays['origin_field'], 'origin field')
    require_finite(arrays['gauge_field_difference'], 'angle gauge field difference')
    check_deadline(deadline)
    with torch.enable_grad():
        matrix = torch.autograd.functional.jacobian(lambda x: interface_field(model, x), zero).detach().numpy()
    arrays['jacobian_delta_omega_p_q'] = matrix
    if matrix.shape != (dim, dim):
        raise ValueError('Autograd field must preserve full state dimension')
    require_finite(matrix, 'autograd Jacobian')
    g = np.r_[np.ones(model.n), np.zeros(dim-model.n)]
    with np.errstate(over='ignore', invalid='ignore'):
        jg = matrix @ g
    require_finite(jg, 'Jacobian gauge direction')
    qa.update(f0_max_abs=float(f0.abs().max()), gauge_max_abs=float(gauge.abs().max()),
              jacobian_gauge_max_abs=float(np.max(np.abs(jg))), finite=True)
    matrix_norm = checked_norm(matrix, 'Jacobian Frobenius norm')
    for h in (1e-5, 1e-6):
        fd = np.full_like(matrix, np.nan)
        arrays[f'jacobian_fd_{h}'] = fd
        with torch.no_grad():
            for j in range(dim):
                check_deadline(deadline)
                plus, minus = zero.detach().clone(), zero.detach().clone()
                plus[j] += h
                minus[j] -= h
                fd[:, j] = ((interface_field(model, plus) - interface_field(model, minus)) / (2*h)).numpy()
                require_finite(fd[:, j], f'finite difference column {j}, h={h}')
                qa['fd_columns_completed'][str(h)] = j+1
        with np.errstate(over='ignore', invalid='ignore'):
            difference = fd-matrix
        require_finite(difference, 'Jacobian finite difference error')
        ratio = checked_norm(difference, 'Jacobian error Frobenius norm') / max(1e-300, matrix_norm)
        require_finite(ratio, 'Jacobian relative Frobenius error')
        qa[f'fd_{h}_relative_fro'] = ratio
        qa[f'fd_{h}_max_abs'] = float(np.max(np.abs(difference)))
    qa['f0_gate'] = qa['f0_max_abs'] <= 1e-10
    qa['gauge_gate'] = max(qa['gauge_max_abs'], qa['jacobian_gauge_max_abs']) <= 1e-10
    qa['fd_gate'] = all(qa[f'fd_{h}_relative_fro'] <= 1e-7 for h in (1e-5, 1e-6))
    qa['pass'] = all(qa[k] for k in ('finite', 'f0_gate', 'gauge_gate', 'fd_gate'))
    return matrix, qa


def validate_cached_targets(evidence, data, mass):
    targets = evidence['targets']
    if [t['index'] for t in targets] != TARGET_IDS:
        raise ValueError('Frozen thirteen target identities/order changed')
    if evidence.get('integrity_pass') is not True or evidence.get('hashes_unchanged') is not True:
        raise ValueError('Cached FOM source integrity assertions unavailable/failed')
    if evidence['thresholds'] != THRESHOLDS:
        raise ValueError('Historical matching thresholds changed')
    indices = np.asarray(data['target_recomputed_indices'])
    values, vectors = np.asarray(data['descriptor_eigenvalues']), np.asarray(data['descriptor_right_vectors'])
    if (indices.shape != (13,) or indices.dtype.kind not in 'iu' or len(set(indices.tolist())) != 13
            or not np.array_equal(data['target_ids'], TARGET_IDS)):
        raise ValueError('Cached descriptor target mapping malformed')
    if (np.asarray(data['mass']).shape != (10,) or np.asarray(mass).shape != (10,)
            or not np.isfinite(mass).all() or (np.asarray(mass) <= 0).any()
            or not np.allclose(data['mass'], mass, rtol=1e-10, atol=1e-12)):
        raise ValueError('Candidate and cached FOM physical masses differ')
    if (values.ndim != 1 or vectors.ndim != 2 or vectors.shape[0] < 20 or vectors.shape[1] != len(values)
            or np.any(indices < 0) or np.any(indices >= len(values))):
        raise ValueError('Cached descriptor indices outside pole/vector arrays')
    require_finite(values, 'cached descriptor eigenvalues')
    require_finite(vectors, 'cached descriptor right vectors')
    outputs = [physical_output(vectors[:, i], mass, omega_scale=OMEGA_S) for i in indices]
    target_outputs = np.asarray(data['target_outputs'])
    if target_outputs.shape != (20, 13) or not np.allclose(np.column_stack(outputs), target_outputs, rtol=1e-10, atol=1e-12):
        raise ValueError('Cached descriptor physical output mapping changed')
    for target, value, output in zip(targets, values[indices], outputs, strict=True):
        old = complex(float(target['real']), float(target['imag']))
        with np.errstate(over='ignore', invalid='ignore'):
            error = abs(old-value) / max(1., abs(old))
        require_finite(error, 'cached pole alignment error')
        if error > 1e-6 or float(target['frequency_hz']) <= 0:
            raise ValueError('Target pole and recomputed descriptor index do not align')
        require_finite([target['real'], target['imag'], target['frequency_hz'], target['damping_ratio']], 'target descriptors')
        if checked_norm(output, 'cached target output norm') <= 1e-12:
            raise ValueError('Invalid null physical target output')
    return targets, outputs


def ordinary_modes(matrix, *, arrays=None, diagnostics=None):
    arrays = {} if arrays is None else arrays
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update({'pass': False, 'residual_bound': 1e-8, 'residuals': []})
    require_finite(matrix, 'ROM matrix')
    values, vectors = np.linalg.eig(matrix)
    arrays.update(eigenvalues=values, right_vectors=vectors)
    require_finite(values, 'ROM eigenvalues')
    require_finite(vectors, 'ROM eigenvectors')
    matrix_norm, residuals = checked_norm(matrix, 'ROM matrix norm'), []
    for i in range(len(values)):
        with np.errstate(over='ignore', invalid='ignore'):
            difference = matrix @ vectors[:, i] - values[i] * vectors[:, i]
            denominator = matrix_norm * checked_norm(vectors[:, i], 'eigenvector norm')
        require_finite(denominator, 'eigen residual denominator')
        residual = checked_norm(difference, 'eigen residual numerator') / max(1., denominator)
        require_finite(residual, 'eigen residual')
        residuals.append(residual)
        diagnostics['residuals'].append(residual)
    arrays['eigen_residuals'] = np.asarray(residuals)
    diagnostics['max_residual'] = max(residuals)
    if max(residuals) > 1e-8:
        raise ValueError('ROM eigen residual exceeds original 1e-8 bound')
    diagnostics['pass'] = True
    return values, vectors, residuals


def mac(a, b):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        den = float(np.vdot(a, a).real * np.vdot(b, b).real)
        require_finite(den, 'MAC denominator')
        if den <= 0:
            return 0.0
        numerator = abs(np.vdot(a, b)) ** 2
        require_finite(numerator, 'MAC numerator')
        value = float(np.clip(numerator / den, 0., 1.))
    return float(require_finite(value, 'MAC quotient'))


def assignment(targets, target_vectors, values, vectors, mass, embed):
    """Original Hungarian assignment, including original unmatched dummy costs."""
    require_finite(values, 'candidate poles')
    outputs = [physical_output(vectors[:, i], mass, embed) for i in range(len(values))]
    norms = [checked_norm(v, 'candidate physical output norm') for v in outputs]
    candidates = [i for i, v in enumerate(values) if v.imag > 1e-6 and norms[i] > 1e-12]
    metrics, failed_pairs = [], []
    cost = np.full((len(targets), len(candidates)+len(targets)), 1e6, dtype=float)
    for ti, (target, fv) in enumerate(zip(targets, target_vectors, strict=True)):
        f, z = float(target['frequency_hz']), float(target['damping_ratio'])
        require_finite([f, z], 'target frequency/damping')
        if f <= 0:
            raise ValueError('Target frequency must be positive')
        for cj, ri in enumerate(candidates):
            row = {'fom_index': int(target['index']), 'rom_index': int(ri), 'eligible': False,
                   **{key: None for key in ('frequency_hz', 'damping_ratio', 'frequency_relative_difference',
                                            'damping_absolute_difference', 'd_f', 'd_z', 'd_m', 'mac', 'cost')}}
            try:
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    rv, rz = values[ri], float(-values[ri].real / abs(values[ri]))
                    rf, m = float(abs(rv.imag) / (2*np.pi)), mac(fv, outputs[ri])
                    df_raw, dz, dm = abs(rf-f)/f, abs(rz-z), (1-m)/.2
                    df, dz_norm = df_raw/FREQ_GATE, dz/DAMP_GATE
                require_finite([rf, rz, m, df_raw, dz, dm, df, dz_norm], 'pair metrics')
                ok = df_raw <= FREQ_GATE and dz <= DAMP_GATE and m >= MAC_GATE
                pair_cost = (df*df + dz_norm*dz_norm + dm*dm)/1000 if ok else 1e6
                require_finite(pair_cost, 'pair assignment cost')
                cost[ti, cj] = pair_cost
                row.update(frequency_hz=rf, damping_ratio=rz, frequency_relative_difference=df_raw,
                           damping_absolute_difference=dz, d_f=df, d_z=dz_norm, d_m=dm, mac=m,
                           eligible=bool(ok), cost=float(pair_cost), numerical_failure=None)
            except (FloatingPointError, OverflowError) as exc:
                row['numerical_failure'] = type(exc).__name__+': '+str(exc)
                failed_pairs.append({'fom_index': row['fom_index'], 'rom_index': row['rom_index'], 'reason': row['numerical_failure']})
            metrics.append(row)
    cost[:, len(candidates):] = 1.0
    result = {'all_pair_metrics': metrics, 'matched': [], 'unmatched_fom': [int(t['index']) for t in targets],
              'unmatched_rom': candidates, 'eligible_pairs': [r for r in metrics if r['eligible']],
              'failed_pairs': failed_pairs, 'oscillatory_rom_count': len(candidates),
              'excluded_null_output_rom': [i for i, v in enumerate(values) if v.imag > 1e-6 and norms[i] <= 1e-12],
              'excluded_nonpositive_imaginary_rom': [i for i, v in enumerate(values) if v.imag <= 1e-6],
              'assignment_executed': False, 'pass': False,
              'dummy_columns': len(targets), 'dummy_cost': 1., 'ineligible_pair_cost': 1e6,
              'cost_matrix': cost.tolist(), 'selected_assignment': []}
    if failed_pairs:
        result['unmatched_meaning'] = 'unresolved; numerical pair failure prevents any matching claim'
        return result
    ri, ci = linear_sum_assignment(cost)
    used = set()
    for r, c in zip(ri, ci, strict=True):
        result['selected_assignment'].append({'fom_index': int(targets[r]['index']), 'column': int(c),
                                             'rom_index': int(candidates[c]) if c < len(candidates) else None,
                                             'cost': float(cost[r, c])})
        if c < len(candidates) and cost[r, c] < 1.:
            result['matched'].append(metrics[r*len(candidates)+c])
            used.add(candidates[c])
    matched_ids = {row['fom_index'] for row in result['matched']}
    result.update({'pass': True, 'assignment_executed': True,
                   'unmatched_fom': sorted(set(int(t['index']) for t in targets)-matched_ids),
                   'unmatched_rom': [int(i) for i in candidates if i not in used],
                   'unmatched_meaning': 'original pair gates and one-to-one Hungarian assignment'})
    return result


def run(checkpoint, output):
    from task4.training import check_authorized, load_checkpoint
    from task4.dynamics import PowerGraph
    check_authorized(False)  # Precedes every input asset hash/read and model load.
    deadline = float(os.environ.get('TASK4_SAVE_DEADLINE_UNIX', 'nan'))
    check_deadline(deadline)
    cache = fixed_cache(read_json(ROOT/'configs/project.json'))
    paths = {'checkpoint': project_path(checkpoint), 'evidence': cache/'evidence.json', 'arrays': cache/'modal_arrays.npz'}
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    arrays = {}
    report = {'status': 'inconclusive', 'complete': False, 'matched_count': None, 'target_count': 13,
              'target_ids': TARGET_IDS, 'thresholds': THRESHOLDS, 'qa': {}, 'stages': [], 'stop_reason': 'started',
              'checkpoint': str(paths['checkpoint']), 'fixed_cache_directory': str(cache),
              'event_payload_read': False, 'test_payload_read': False, 'fom_recomputed': False,
              'model_tuning': False, 'port_sources_sha256_at_implementation': PORT_SOURCES,
              'scope': 'original L3 supplemental screen; no effect on L1 eligibility and no L2 region certificate',
              'coordinate_change': 'u=(delta,omega,p,q); z=diag(1,M,1,1)u; full common omega retained',
              'physical_readout_scales': {'delta': S_DELTA, 'omega_rad_s': S_OMEGA, 'cached_fom_omega_scale': OMEGA_S},
              'managed_save_deadline_unix': deadline, 'source_hashes_before': {},
              'source_hashes_after_load': {}, 'source_hashes_end': {},
              'preexisting_independent_cache_hash_anchor': None,
              'integrity_scope': 'original cached integrity assertions plus this run before/after-load/end asset hashes',
              'qa_thresholds': {'f0_max_abs': 1e-10, 'gauge_max_abs': 1e-10, 'jacobian_gauge_max_abs': 1e-10,
                                'fd_relative_fro': 1e-7, 'fd_steps': [1e-5, 1e-6], 'eigen_residual': 1e-8}}

    def persist(stage):
        report['stages'].append(stage)
        report['wall_seconds'] = time.monotonic()-started
        issues = []
        clean = json_safe(report, issues)
        if issues:
            clean.update(status='inconclusive', complete=False, matched_count=None,
                         matching_result_valid=False, nonfinite_diagnostics=issues)
        write_json(output/'evidence.json', clean)
        if arrays:
            temporary = output/'modal_arrays.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(output/'modal_arrays.npz')
        if issues:
            raise FloatingPointError('Nonfinite modal diagnostics saved with null values')

    persist('started')
    failure = None
    try:
        report['source_hashes_before'] = {key: sha256(path) for key, path in paths.items()}
        check_deadline(deadline)
        model, unused_encoder, unused_head, obj = load_checkpoint(paths['checkpoint'], 'cpu')
        del unused_encoder, unused_head
        if (type(model) is not PowerGraph or model.n != 10 or model.rank != 2 or model.dim != 50
                or obj.get('kind') not in ('T4-N', 'T4-S') or obj.get('synthetic') is not False
                or obj['config'].get('arm') != 'P' or obj['config'].get('synthetic') is not False):
            raise ValueError('Expected a fixed nonsynthetic P endpoint with ten machines and full dimension 50')
        model.double().eval().requires_grad_(False)
        mass = model.mass.detach().numpy()
        if not np.allclose(mass, obj['physics']['mass'], rtol=0, atol=0) or not np.isclose(obj['physics']['omega_s'], OMEGA_S, rtol=1e-12, atol=1e-12):
            raise ValueError('Checkpoint mass/omega_s convention differs from original modal contract')
        check_deadline(deadline)
        evidence = read_json(paths['evidence'])
        with np.load(paths['arrays'], allow_pickle=False) as data:
            targets, target_outputs = validate_cached_targets(evidence, data, mass)
        report['targets'] = targets
        report['cached_integrity_pass'] = evidence['integrity_pass']
        report['cached_hashes_unchanged'] = evidence['hashes_unchanged']
        report['source_hashes_after_load'] = {key: sha256(path) for key, path in paths.items()}
        if report['source_hashes_after_load'] != report['source_hashes_before']:
            raise ValueError('Source changed while loading fixed modal inputs')
        arrays.update(target_ids=np.asarray(TARGET_IDS), target_outputs=np.column_stack(target_outputs), mass=mass)
        report.update(full_dimension=50, reduced_dimension=49, endpoint_kind=obj['kind'])
        persist('cached_sources_validated')
        matrix, _ = jacobian_qa(model, deadline, arrays=arrays, qa=report['qa'])
        persist('jacobian_qa_saved')
        if not report['qa']['pass']:
            raise ValueError('Original equilibrium/gauge/Jacobian QA failed; no matching result')
        check_deadline(deadline)
        embedding = common_angle_embedding(50)
        reduced = embedding.T @ matrix @ embedding
        require_finite(reduced, 'common-angle reduced Jacobian')
        arrays.update(embedding=embedding, reduced_jacobian=reduced)
        report['eigen_qa'] = {}
        values, vectors, residuals = ordinary_modes(reduced, arrays=arrays, diagnostics=report['eigen_qa'])
        report.update(spectral_alpha=float(np.max(values.real)), max_eigen_residual=max(residuals),
                      poles=[{'index': i, 'real': float(v.real), 'imag': float(v.imag)} for i, v in enumerate(values)])
        arrays['physical_output_vectors'] = np.column_stack([physical_output(vectors[:, i], mass, embedding) for i in range(len(values))])
        persist('full_spectrum_saved')
        check_deadline(deadline)
        matches = assignment(targets, target_outputs, values, vectors, mass, embedding)
        report['matching'] = matches
        persist('assignment_saved')
        if not matches['pass']:
            raise FloatingPointError('Pair diagnostics incomplete; no matching count can be claimed')
        check_deadline(deadline)
        report.update(status='complete_original_modal_screen', complete=True,
                      matched_count=len(matches['matched']), stop_reason='original thirteen-target comparison completed')
    except (Exception, KeyboardInterrupt) as exc:
        failure = exc
        report.update(status='inconclusive', complete=False, matched_count=None, stop_reason=type(exc).__name__+': '+str(exc))
    finally:
        try:
            report['source_hashes_end'] = {key: sha256(path) if path.is_file() else None for key, path in paths.items()}
            report['source_integrity_pass'] = bool(report['source_hashes_before'] and
                                                   report['source_hashes_before'] == report['source_hashes_after_load'] == report['source_hashes_end'])
            if not report['source_integrity_pass']:
                report.update(status='inconclusive', complete=False, matched_count=None,
                              integrity_failure_reason='before/after-load/end source hashes differ or are incomplete')
        except Exception as exc:
            report.update(status='inconclusive', complete=False, matched_count=None, source_integrity_pass=False,
                          integrity_failure_reason=type(exc).__name__+': '+str(exc))
        report['matching_result_valid'] = bool(report['complete'] and report.get('source_integrity_pass')
                                                and report['qa'].get('pass') and report.get('matching', {}).get('pass'))
        try:
            persist('final_evidence_saved')
        except FloatingPointError as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure
    return json_safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    run(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
