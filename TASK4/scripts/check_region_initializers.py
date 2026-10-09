"""Check cached initializer centers against an already certified fixed ROM ellipsoid.

Only event_ids and initial_state are decoded from declared NPZ caches. Whole
NPZ byte hashes include all compressed members, but no history, forecast,
truth, checkpoint, or source HDF payload is decoded. Membership of the exact
cached binary64 center does not cover initializer uncertainty or imply FOM
stability or prediction accuracy. No new radius is searched or model fitted.
"""
import _bootstrap  # noqa: F401
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from task4.paths import project_path, write_json
from task4.workflow import Workflow


RADII = (1e-6, 1e-5, 1e-4, 1e-3, .003, .01, .03, .1)
HORIZONS = {'1.0', '2.0', '5.0', '8.0', 'prefix'}
FULL_ORDER = ['delta[N]', 'pi[N]', 'p[N]', 'q[N,rank] machine-major']
QUOTIENT_ORDER = ['delta_i-delta_last [N-1]', 'pi[N]', 'p[N]', 'q[N,rank]']


def check_authorized():
    rid = os.environ.get('TASK4_MANAGED_RUN_ID')
    state = Workflow().load()
    if not rid or rid not in state['active_jobs'] or state['status'] != 'RUNNING' or state.get('pending_direction'):
        raise PermissionError('Managed authorized project.py run required')


def check_deadline(deadline):
    if time.time()+3 >= deadline:
        raise TimeoutError('Managed initializer-membership save deadline')


def sha256(path, deadline=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            if deadline is not None:
                check_deadline(deadline)
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def bad(value):
        raise ValueError('Nonfinite JSON literal: '+value)
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key: '+key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=bad, object_pairs_hook=unique)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(value):
    require(isinstance(value, str) and len(value) <= 4096
            and re.fullmatch(r'-?[0-9]+/[1-9][0-9]*', value), 'Missing/invalid exact rational metadata')
    return Fraction(value)


def exact(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Finite binary64 number required')
    return Fraction.from_float(float(value))


def text(value):
    return str(value.numerator)+'/'+str(value.denominator)


def identity(row):
    require(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id']
            and row.get('label') in ('stable', 'unstable') and isinstance(row.get('group'), str),
            'Missing original ID/label/group metadata')
    return {'id': row['id'], 'label': row['label'], 'group': row['group']}


def population(evaluation):
    require(evaluation.get('endpoint') == 'P-S' and evaluation.get('source_identity_unchanged') is True,
            'Incomplete or changed fixed P-S evaluation')
    require(not str(evaluation.get('evidence_validity', '')).startswith('invalid')
            and evaluation.get('test_payload_read') is False, 'Invalid/TEST evaluation provenance')
    for key, value in {'history_ms': 100, 'initializations_per_event': 1, 'encoder_calls_per_event': 1,
                       'forecast_steps': 800, 'decision_horizon_s': 8., 'fine_dt': .0025, 'coarse_dt': .005}.items():
        require(type(evaluation.get(key)) in (int, float) and evaluation[key] == value,
                'Missing/mismatched fixed evaluation policy: '+key)
    eligible, horizons = {}, {}
    for row in evaluation['rows']:
        meta = identity(row)
        eid = meta['id']
        require(eid not in eligible or eligible[eid] == meta, 'Conflicting evaluation labels/groups')
        eligible[eid] = meta
        require(row.get('horizon') in HORIZONS and row['horizon'] not in horizons.setdefault(eid, set()),
                'Missing/duplicate/unknown evaluation horizon')
        horizons[eid].add(row['horizon'])
    require(eligible and all(value == HORIZONS for value in horizons.values()), 'Incomplete eligible evaluation population')
    facts = evaluation['source_facts']
    require(facts.get('split') in ('train', 'validation') and facts.get('test_payload_read') is False,
            'Missing authorized split provenance')
    excluded = [identity(row) for row in facts['excluded']]
    require(all(row.get('reason') == 'latency_uncovered' for row in facts['excluded']), 'Unknown excluded population reason')
    excluded_ids = [row['id'] for row in excluded]
    require(len(set(excluded_ids)) == len(excluded_ids) and not set(excluded_ids) & set(eligible), 'Duplicate population ID')
    raw = len(eligible)+len(excluded)
    for key, count in (('raw_event_count', raw), ('eligible_event_count', len(eligible)),
                       ('latency_uncovered_count', len(excluded))):
        require(type(evaluation.get(key)) is int and evaluation[key] == count, 'Evaluation denominator mismatch: '+key)
    for key, count in (('raw_count', raw), ('eligible_count', len(eligible)),
                       ('unstable_count', sum(row['label'] == 'unstable' for row in eligible.values()))):
        require(type(facts.get(key)) is int and facts[key] == count, 'Source denominator mismatch: '+key)
    classified = [identity(row) for row in evaluation['classification']['rows']]
    require(len(classified) == raw and len({row['id'] for row in classified}) == raw,
            'Classification population incomplete/duplicated')
    require({row['id']: row for row in classified} == {**eligible, **{row['id']: row for row in excluded}},
            'Classification labels/groups differ from original population')
    latency = {row['id'] for row in evaluation['classification']['rows'] if row.get('category') == 'latency_uncovered'}
    require(latency == set(excluded_ids), 'Latency population mismatch')
    return eligible, excluded


def verify_region(certificate, symbolic, evaluation, symbolic_path, symbolic_hash):
    for key in ('complete', 'L2_region_certificate', 'source_unchanged', 'interval_backend_unchanged'):
        require(certificate.get(key) is True, 'Unverified certificate flag: '+key)
    require(certificate.get('status') == 'certified' and certificate.get('source_mutation') is False
            and certificate.get('uncomputed_radii') == [], 'Incomplete fixed certificate')
    require(project_path(certificate['source_model_path']) == symbolic_path, 'Certificate source model path mismatch')
    for key in ('source_model_sha256', 'source_sha256_after_load', 'source_sha256_after_validation', 'source_sha256_final'):
        require(certificate.get(key) == symbolic_hash, 'Certificate fixed S hash mismatch: '+key)
    require(certificate.get('interval_backend_sources_before') == certificate.get('interval_backend_sources_after')
            and certificate.get('interval_backend_sources_before'), 'Missing/inconsistent frozen backend metadata')
    from certify_local_region import backend_sources
    import mpmath
    require(certificate.get('mpmath_version') == mpmath.__version__, 'Interval backend version changed')
    current_backend = backend_sources()
    require(current_backend == certificate['interval_backend_sources_before'], 'Interval backend files changed')
    require(symbolic.get('format') == 'task4-polynomial-powergraph' and symbolic.get('version') == 1,
            'Unsupported explicit S format')
    names = {'node_energy', 'edge_energy', 'node_exchange', 'node_dissipation',
             'edge_exchange', 'edge_dissipation', 'port'}
    require(set(symbolic.get('functions', {})) == names
            and all(value.get('type') == 'polynomial' for value in symbolic['functions'].values()),
            'Incomplete explicit seven-function T4-S')
    model = evaluation['model_identity']
    metadata = symbolic['metadata']
    require(model.get('kind') == 'T4-S' and metadata.get('checkpoint_sha256') == model.get('checkpoint_sha256')
            and re.fullmatch('[0-9a-f]{64}', str(model.get('checkpoint_sha256'))), 'S/evaluation checkpoint SHA mismatch')
    require(project_path(metadata['checkpoint']) == project_path(model['checkpoint'])
            and metadata.get('step') == model.get('step') and metadata.get('final_steps_complete') is True,
            'S/evaluation checkpoint metadata mismatch')
    n, rank = symbolic['n'], symbolic['rank']
    require(type(n) is int and type(rank) is int and n >= 1 and rank == 2 and (3+rank)*n-1 <= 49,
            'Unsupported fixed per-machine coordinate dimensions')
    coordinates = certificate['coordinates']
    dimension = (3+rank)*n-1
    require(coordinates.get('full_dimension') == dimension+1 and coordinates.get('quotient_dimension') == dimension
            and coordinates.get('full_order') == FULL_ORDER and coordinates.get('quotient_order') == QUOTIENT_ORDER
            and coordinates.get('lift') == 'last delta=0; all pi/p/q retained'
            and coordinates.get('projection_lift_left_inverse') == 'exact by algebra'
            and coordinates.get('xi_definition') == 'xi=y/s; each stored s is a fixed exact binary64 dyadic'
            and coordinates.get('pi_scale_construction') == 'mass*saved TRAIN omega scale rounded once to binary64 and then fixed exactly'
            and coordinates.get('scale_units') == ['delta: rad', 'pi: mass unit * rad/s', 'p: system pu', 'q: latent unit']
            and coordinates.get('time_unit') == 's',
            'Missing/changed certificate coordinate contract')
    scales = [ratio(value) for value in coordinates['scale_exact_rationals']]
    require(len(scales) == dimension and all(value > 0 for value in scales)
            and scales == [exact(value) for value in coordinates['scale']], 'Scale rational/binary64 mismatch')
    expected = ([symbolic['scales'][0]]*(n-1)
                + [float(mass)*float(symbolic['scales'][1]) for mass in symbolic['physics']['mass']]
                + [symbolic['scales'][2]]*n + [1.]*(n*rank))
    require(scales == [exact(value) for value in expected], 'Certificate scales differ from fixed S')
    context = evaluation['comparison_context']
    require(context.get('saved_training_scales') == symbolic['scales']
            and context.get('mass') == symbolic['physics']['mass'], 'Evaluation state units differ from fixed S')
    require(context.get('history_samples') == 11
            and context.get('truth_time_origin') == 'postfault t=0; initializer ends at t=0.1 s',
            'Missing initialization time metadata')
    candidate = certificate['quadratic_candidate']
    p = [[ratio(value) for value in row] for row in candidate['P_exact_rationals']]
    require(len(p) == dimension and all(len(row) == dimension for row in p)
            and all(p[i][j] == p[j][i] for i in range(dimension) for j in range(dimension)), 'P exact shape/symmetry mismatch')
    require(p == [[exact(value) for value in row] for row in candidate['P_exact_binary64_coefficients']],
            'P rational/binary64 mismatch')
    digest = hashlib.sha256(json.dumps({'P': candidate['P_exact_rationals'], 'R': candidate['R_exact_rationals']},
                                      sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()
    require(digest == candidate.get('fixed_candidate_parameter_sha256'), 'P/R candidate coefficient hash mismatch')
    require(certificate['validated_P'].get('strict_positive') is True, 'P positivity unverified')
    lower_p = ratio(certificate['validated_P']['P_eigenvalue_lower_exact'])
    require(lower_p > 0 and certificate.get('radii_scaled_box') == list(RADII), 'Missing fixed radius/P policy')
    attempts = certificate['attempts']
    require(len(attempts) == len(RADII) and all(row.get('complete') is True for row in attempts), 'Incomplete radius attempts')
    valid = []
    for radius, row in zip(RADII, attempts):
        require(row.get('radius_scaled_box') == radius and ratio(row['radius_exact_rational']) == exact(radius),
                'Radius metadata mismatch')
        if row.get('certified') is True:
            region = row['region']
            rho = ratio(region['rho_exact_rational'])
            require(region.get('V') == 'xi.T P xi (no factor one half)'
                    and ratio(region['m_P_exact_lower']) == lower_p
                    and ratio(region['m_Q_exact_lower_per_s']) > 0
                    and ratio(region['m_Q_exact_lower_per_s']) == ratio(row['Q_gershgorin_lower_exact'])
                    and rho == lower_p*exact(radius)**2/2, 'Missing/inconsistent certified ellipsoid metadata')
            valid.append((rho, row))
    require(valid, 'No fixed certified ellipsoid available')
    rho, selected = max(valid, key=lambda pair: pair[0])
    return n, rank, scales, p, rho, selected['radius_scaled_box'], current_backend


def membership(state, n, rank, scales, p, rho):
    require(len(state) == (3+rank)*n, 'Cached initial-state dimension mismatch')
    full = [exact(float(value)) for value in state]
    reduced = [full[i]-full[n-1] for i in range(n-1)]+full[n:]
    xi = [value/scale for value, scale in zip(reduced, scales)]
    require(len(xi) == len(p), 'Quotient dimension mismatch')
    v = sum((xi[i]*p[i][j]*xi[j] for i in range(len(xi)) for j in range(len(xi))), Fraction(0))
    require(v >= 0, 'Negative quadratic value contradicts certified P')
    result = {'status': 'inside' if v <= rho else 'outside', 'V_exact': text(v),
              'V_over_rho_exact': text(v/rho), 'rho_exact': text(rho),
              'max_abs_scaled_coordinate_exact': text(max(abs(value) for value in xi))}
    try:
        result['V_over_rho'] = float(v/rho)
    except OverflowError:
        result['V_over_rho'] = None
    return result


def cache_manifest(evaluation, eligible):
    cache = evaluation['cache']
    require(cache.get('generation_hash_algorithm') == 'SHA256 after each cache/metadata file is written',
            'Missing cache generation hashes')
    directory = project_path(cache['directory'])
    batches, ids, paths = [], [], set()
    for batch in cache['batches']:
        path, meta_path = project_path(batch['path']), project_path(batch['metadata_path'])
        require(path.suffix == '.npz' and meta_path == path.with_suffix('.json') and path.is_relative_to(directory)
                and path not in paths, 'Cache path layout/duplication mismatch')
        paths.add(path)
        batch_ids = batch['event_ids']
        require(isinstance(batch_ids, list) and batch_ids and all(isinstance(v, str) and v in eligible for v in batch_ids),
                'Cache IDs not in declared eligible population')
        ids.extend(batch_ids)
        for key in ('sha256', 'metadata_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', str(batch.get(key))), 'Missing generation hash: '+key)
        batches.append((batch, path, meta_path))
    require(len(ids) == len(eligible) and len(set(ids)) == len(ids) and set(ids) == set(eligible),
            'Cache population missing/duplicated/different IDs')
    return batches


def summary(rows):
    counts = {status: sum(row['status'] == status for row in rows) for status in ('inside', 'outside', 'invalid')}
    groups = sorted({row['group'] for row in rows})
    return {'eligible_centers': len(rows), **counts,
            'by_group_label': [{'group': group, 'label': label, 'total': sum(row['group'] == group and row['label'] == label for row in rows),
                                **{status: sum(row['group'] == group and row['label'] == label and row['status'] == status for row in rows)
                                   for status in counts}} for group in groups for label in ('stable', 'unstable')]}


def check(certificate, evaluation, symbolic, output):
    check_authorized()
    value = os.environ.get('TASK4_SAVE_DEADLINE_UNIX')
    require(value is not None and math.isfinite(float(value)), 'Finite managed deadline required')
    deadline = float(value)
    check_deadline(deadline)  # Before all input/output file I/O.
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'inconclusive', 'complete': False, 'rows': [], 'cache_audits': [],
              'initializer_uncertainty_covered': False, 'FOMguarantee': False, 'prediction_accuracy_claim': False,
              'new_radius_search': False, 'model_modified': False, 'checkpoint_read': False,
              'source_HDF_read': False, 'TEST_payload_read': False,
              'allowed_npz_decodes': ['event_ids', 'initial_state'], 'decoded_npz_members': [],
              'byte_hash_scope': 'full NPZ compressed file bytes, including undecoded members; decoded arrays are only event_ids and initial_state',
              'membership_definition': 'exact cached binary64 center in the largest already certified closed quotient ellipsoid',
              'stop_reason': 'started', 'input_identity_unchanged': False}
    paths, hashes = {}, {}
    def persist():
        report['summary'] = summary(report['rows'])
        report['process_wall_seconds'] = time.monotonic()-started
        write_json(output/'membership.json', report)
    persist()
    caught = None
    try:
        for key, name in (('certificate', certificate), ('evaluation', evaluation), ('symbolic', symbolic)):
            paths[key] = project_path(name)
            check_deadline(deadline)
            hashes[key] = sha256(paths[key], deadline)
        report['inputs'] = {key: {'path': str(path), 'sha256_before': hashes[key]} for key, path in paths.items()}
        cert, ev, spec = [read_json(paths[key]) for key in ('certificate', 'evaluation', 'symbolic')]
        eligible, excluded = population(ev)
        report['rows'] = [{**meta, 'status': 'invalid', 'reason': 'not_computed'} for meta in eligible.values()]
        by_id = {row['id']: row for row in report['rows']}
        report['population'] = {'raw': ev['raw_event_count'], 'eligible': len(eligible),
                                'latency_uncovered': len(excluded), 'excluded_metadata_only': excluded,
                                'split': ev['source_facts']['split']}
        n, rank, scales, p, rho, radius, backend_before = verify_region(cert, spec, ev, paths['symbolic'], hashes['symbolic'])
        report['selected_region'] = {'radius_scaled_box': radius, 'rho_exact': text(rho), 'V': 'xi.T P xi (no factor one half)',
                                     'common_angle_quotient_only': True, 'dimension': len(p)}
        report['backend_before'] = backend_before
        batches = cache_manifest(ev, eligible)
        persist()
        for batch, path, meta_path in batches:
            check_deadline(deadline)
            audit = {'path': str(path), 'metadata_path': str(meta_path), 'event_ids': batch['event_ids'], 'valid': False,
                     'generation_sha256': batch['sha256'], 'generation_metadata_sha256': batch['metadata_sha256']}
            report['cache_audits'].append(audit)
            try:
                audit['sha256_before'] = sha256(path, deadline)
                audit['metadata_sha256_before'] = sha256(meta_path, deadline)
                require(audit['sha256_before'] == batch['sha256'] and audit['metadata_sha256_before'] == batch['metadata_sha256'],
                        'Cache generation hash mismatch before decode')
                metadata = [identity(row) for row in read_json(meta_path)]
                require(metadata == [eligible[eid] for eid in batch['event_ids']], 'Cache metadata ID/label/group mismatch')
                # Do not iterate archive members or request history/forecast/truth.
                with np.load(path, allow_pickle=False) as cache:
                    event_ids = cache['event_ids']
                    if 'event_ids' not in report['decoded_npz_members']:
                        report['decoded_npz_members'].append('event_ids')
                    states = cache['initial_state']
                    if 'initial_state' not in report['decoded_npz_members']:
                        report['decoded_npz_members'].append('initial_state')
                require(event_ids.ndim == 1 and event_ids.dtype.kind == 'U'
                        and event_ids.tolist() == batch['event_ids'], 'Decoded cache event IDs mismatch')
                require(states.dtype == np.dtype('float64') and states.shape == (len(event_ids), (3+rank)*n),
                        'Decoded initial-state shape/dtype mismatch')
                for eid, state in zip(batch['event_ids'], states):
                    check_deadline(deadline)
                    try:
                        by_id[eid].update(membership(state, n, rank, scales, p, rho), reason=None)
                    except ValueError as exc:
                        by_id[eid].update(status='invalid', reason=str(exc))
                audit['sha256_after'] = sha256(path, deadline)
                audit['metadata_sha256_after'] = sha256(meta_path, deadline)
                require(audit['sha256_after'] == batch['sha256'] and audit['metadata_sha256_after'] == batch['metadata_sha256'],
                        'Cache changed during decode/membership')
                audit['valid'] = True
            except Exception as exc:
                audit['error'] = type(exc).__name__+': '+str(exc)
                for eid in batch['event_ids']:
                    by_id[eid].update(status='invalid', reason=audit['error'])
                if isinstance(exc, TimeoutError):
                    raise
            finally:
                for key, file in (('sha256_after', path), ('metadata_sha256_after', meta_path)):
                    if key not in audit:
                        try:
                            audit[key] = sha256(file, deadline)
                        except Exception as exc:
                            audit[key] = None
                            audit['after_hash_error'] = str(exc)
                persist()
        report.update(complete=True, status='complete_center_membership' if all(row['valid'] for row in report['cache_audits'])
                      else 'inconclusive', stop_reason='declared_population_processed')
    except Exception as exc:
        caught = exc
        report.update(status='inconclusive', complete=False, stop_reason=type(exc).__name__+': '+str(exc))
    finally:
        unchanged = True
        for key, path in paths.items():
            try:
                after = sha256(path)
                report.setdefault('inputs', {}).setdefault(key, {})['sha256_after'] = after
                unchanged = unchanged and after == hashes.get(key)
            except OSError:
                unchanged = False
        if 'backend_before' in report:
            try:
                from certify_local_region import backend_sources
                report['backend_after'] = backend_sources()
                unchanged = unchanged and report['backend_after'] == report['backend_before']
            except Exception as exc:
                unchanged = False
                report['backend_final_error'] = str(exc)
        report['input_identity_unchanged'] = unchanged and len(hashes) == 3
        if not report['input_identity_unchanged']:
            report.update(status='inconclusive', complete=False, stop_reason='source_or_backend_changed_or_unavailable')
            for row in report['rows']:
                row.update(status='invalid', reason='source_or_backend_changed_or_unavailable')
        by_id = {row['id']: row for row in report['rows']}
        cache_unchanged = bool(report['cache_audits'])
        for audit in report['cache_audits']:
            try:
                audit['sha256_final'] = sha256(audit['path'], deadline)
                audit['metadata_sha256_final'] = sha256(audit['metadata_path'], deadline)
                audit['identity_unchanged_at_end'] = (audit['sha256_final'] == audit['generation_sha256']
                                                     and audit['metadata_sha256_final'] == audit['generation_metadata_sha256'])
            except Exception as exc:
                audit['identity_unchanged_at_end'] = False
                audit['final_hash_error'] = str(exc)
            if not audit['identity_unchanged_at_end']:
                cache_unchanged = False
                audit['valid'] = False
                for eid in audit['event_ids']:
                    by_id[eid].update(status='invalid', reason='cache_changed_or_unavailable_at_end')
        report['cache_identity_unchanged_at_end'] = cache_unchanged
        if not cache_unchanged and report['complete']:
            report.update(status='inconclusive', complete=False, stop_reason='cache_changed_or_unavailable_at_end')
        report['limitations'] = ['Outside the ellipsoid means unproved by this inner bound, not unstable.',
                                 'Only exact cached centers are checked; no initialization uncertainty set is covered.',
                                 'Membership is in the continuous ROM quotient; no FOM or prediction-accuracy guarantee follows.']
        persist()
    if caught is not None:
        raise caught
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('certificate', 'evaluation', 'symbolic', 'output'):
        parser.add_argument('--'+name, required=True)
    result = check(**vars(parser.parse_args()))
    print(json.dumps({'status': result['status'], **result['summary']}, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
