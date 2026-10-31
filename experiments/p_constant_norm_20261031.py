"""One released exact screen of published binary64 norms; standard library only."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = 'p_constant_norm_20261031'
PRIVATE = ROOT / 'private_data' / RUN
PUBLIC = ROOT / 'results' / RUN
FROZEN = {
    'scientist': ('reports/' + RUN + '_scientist.md', 'f2609ff0f26570a22a95240715022b2799de8885ef5ada2ab76f6c9149fbb916'),
    'protocol': ('reports/' + RUN + '_protocol.md', 'f0b6ac793378160f08e0f76cf317030172282672be3da8eb96ea7faf9a18673b'),
    'source_bindings': ('results/' + RUN + '/source_bindings.json', '65cf35d51aae1c59ddc2c23feb703fe77ab2695a5a88f32e76fd77e279785b9e'),
    'authorization': ('results/' + RUN + '/authorization.json', 'e5edca5ed289047ad97e61d196bfba0f0b6cce529565c449750d6c24ec1b3552'),
    'design_review': ('results/' + RUN + '/design_review_at_adoption.md', 'bceb234c85b0bdd4473fdb45e0b9d05d67d1599cff5c7fbf4619b9f01bd3bc60'),
    'coverage_review': ('results/' + RUN + '/coverage_review_at_adoption.md', 'ebbe8ce400d489ec918b15ea3a7e54ab20152d5da49d6ebe7c0660fcaefec2b9'),
}
DENOM = {'all': 297, 'stable': 267, 'unstable': 30}
FIELDS = ('R_Q_before', 'R_L_before', 'nu', 'mu')
MAX_FINITE = sys.float_info.max

class QualificationUnknown(ValueError):
    pass

class NumberToken(str):
    pass

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise QualificationUnknown('duplicate JSON key')
        out[key] = value
    return out

def reject_constant(token):
    raise QualificationUnknown('nonfinite JSON constant')

def parse_document(text, tokens=False):
    options = dict(object_pairs_hook=unique_object, parse_constant=reject_constant)
    if tokens:
        options.update(parse_int=NumberToken, parse_float=NumberToken)
    result = json.loads(text, **options)
    def walk(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise QualificationUnknown('overflowed JSON number')
        if isinstance(value, dict):
            for v in value.values(): walk(v)
        elif isinstance(value, list):
            for v in value: walk(v)
    walk(result)
    return result

def read_control(path):
    return parse_document(Path(path).read_text(encoding='utf-8-sig'))

def release_expected():
    """Control identity only: never open a bound parent input or do arithmetic."""
    review = PUBLIC / 'implementation_review_at_release.md'
    return {
        'source_sha256': sha(Path(__file__)),
        'tests_sha256': sha(ROOT / 'tests' / ('test_' + RUN + '.py')),
        **{key + '_sha256': sha(ROOT / path) for key, (path, _) in FROZEN.items()},
        'implementation_review_sha256': sha(review) if review.is_file() else None,
        'allowed_command': 'analyze', 'normal_jobs_upper_bound': 1,
        'source_files_upper_bound': 8, 'original_cells_upper_bound': 180,
        'token_fields_upper_bound': 720, 'constraints_upper_bound': 360,
        'machine_intersections_upper_bound': 10,
        **{key + '_upper_bound': 0 for key in (
            'parameter_estimates', 'QR', 'MP_mean', 'new_residual_evaluations',
            'aggregate_recomputations', 'integrals', 'bounds', 'filters',
            'initializers', 'optimizer', 'ROM', 'FOM', 'training', 'downloads',
            'model_reads_writes', 'private_scientific_payload_reads')},
    }

def checked_release():
    for _, (path, expected) in FROZEN.items():
        if sha(ROOT / path) != expected:
            raise QualificationUnknown('frozen control identity differs')
    release = read_control(PUBLIC / 'execution_release.json')
    expected = release_expected()
    if expected['implementation_review_sha256'] is None:
        raise PermissionError('implementation review snapshot required')
    if release.get('released') is not True or release.get('allowed_command') != 'analyze':
        raise PermissionError('root analyze release required')
    for key, value in expected.items():
        if release.get(key) != value or (type(value) is int and type(release.get(key)) is not int):
            raise PermissionError('release identity or bound differs: ' + key)
    return {'identities': expected, 'release_sha256': sha(PUBLIC / 'execution_release.json')}

def nearest_even(token, value):
    """Independent exact decimal token containment in the float rounding cell."""
    if not isinstance(token, NumberToken) or type(value) not in (int, float):
        raise QualificationUnknown('numeric field or token type differs')
    try:
        v = float(value)
        if not math.isfinite(v) or v < 0 or v >= MAX_FINITE:
            raise QualificationUnknown('unsupported binary64 domain')
        dec = Decimal(str(token))
        if not dec.is_finite():
            raise QualificationUnknown('nonfinite token')
        exact_token = Fraction(*dec.as_integer_ratio())
    except (OverflowError, ValueError) as exc:
        raise QualificationUnknown('unsupported numeric token') from exc
    exact = Fraction(*v.as_integer_ratio())
    prev = Fraction(*math.nextafter(v, -math.inf).as_integer_ratio())
    nxt = Fraction(*math.nextafter(v, math.inf).as_integer_ratio())
    low, high = (prev + exact) / 2, (exact + nxt) / 2
    even = (struct.unpack('>Q', struct.pack('>d', v))[0] & 1) == 0
    if not (low < exact_token < high or even and exact_token in (low, high)):
        raise QualificationUnknown('nearest-even token identity failed')
    return exact

def strict_int(value):
    if type(value) is not int:
        raise QualificationUnknown('integer field must not be bool/float')
    return value

def cell_key(row, kind):
    return (strict_int(row['machine']), kind, row['group'], strict_int(row['steps']),
            '' if kind == 0 else row['phase'])

def qualify(normal, tokens, progress=None):
    expected = {(m, 0, g, s, '') for m, g, s in itertools.product(range(10), DENOM, (4, 8))}
    expected |= {(m, 1, g, s, p) for m, g, s, p in itertools.product(
        range(10), ('stable', 'unstable'), (4, 8), ('history', 'cross_t0', 'future'))}
    qualified = []
    seen = set()
    for kind, name, length in ((0, 'overall60', 60), (1, 'stage120', 120)):
        rows, raw = normal[name], tokens[name]
        if type(rows) is not list or type(raw) is not list or len(rows) != length or len(raw) != length:
            raise QualificationUnknown('complete axes required')
        for row, token_row in zip(rows, raw):
            key = cell_key(row, kind)
            if key not in expected or key in seen:
                raise QualificationUnknown('unknown or duplicate axis')
            seen.add(key)
            m, _, group, steps, phase = key
            n = DENOM[group]
            if strict_int(row['N_original']) != n or strict_int(row['windows']) <= 0:
                raise QualificationUnknown('original denominator/window differs')
            available = n
            if kind and group == 'unstable':
                if phase == 'future': available = 27 if steps == 4 else 22
                elif phase == 'cross_t0': available = 27
            if strict_int(row['N_available']) != available:
                raise QualificationUnknown('available denominator differs')
            values = {}
            for f in FIELDS:
                values[f] = nearest_even(token_row[f], row[f])
                if progress is not None: progress['token_fields_verified'] += 1
            if not (0 < values['mu'] <= 1) or kind == 0 and values['mu'] != 1:
                raise QualificationUnknown('invalid weight mass')
            qualified.append((key, values, {f: row[f] for f in ('N_original', 'N_available', 'windows')}))
            if progress is not None: progress['original_cells_qualified'] += 1
    if seen != expected:
        raise QualificationUnknown('incomplete axes')
    return sorted(qualified)

def pair_reduce(n, d):
    if d <= 0: raise QualificationUnknown('nonpositive denominator')
    g = math.gcd(n, d)
    return n // g, d // g

def int_endpoints(r, nu):
    """Rebuild independently using integer products, without Fraction arithmetic."""
    rn, rd = r; nn, nd = nu
    low_n = rn * nd - 10 * nn * rd
    high_n = rn * nd + 10 * nn * rd
    den = rd * nd
    return pair_reduce(max(0, low_n), den), pair_reduce(high_n, den)

def cmp_pair(a, b):
    return (a[0] * b[1] > b[0] * a[1]) - (a[0] * b[1] < b[0] * a[1])

def integer_certificate(constraints, lower_key, upper_key, lower, upper):
    ordered = sorted(constraints, key=lambda row: row['key'])
    rebuilt = []
    for row in ordered:
        lo, hi = int_endpoints(row['R_ratio'], row['nu_ratio'])
        if lo != row['lower_ratio'] or hi != row['upper_ratio']:
            raise QualificationUnknown('integer endpoint reconstruction differs')
        rebuilt.append((row['key'], lo, hi))
    if not rebuilt: raise QualificationUnknown('empty certificate input')
    low_w = high_w = rebuilt[0]
    for item in rebuilt[1:]:
        if cmp_pair(item[1], low_w[1]) > 0: low_w = item
        if cmp_pair(item[2], high_w[2]) < 0: high_w = item
    if (low_w[0] != lower_key or high_w[0] != upper_key or low_w[1] != lower or high_w[2] != upper):
        raise QualificationUnknown('integer witness or fixed tie order differs')
    for _, lo, hi in rebuilt:
        if cmp_pair(lo, lower) > 0 or cmp_pair(upper, hi) > 0:
            raise QualificationUnknown('integer extremum inequality failed')
    return {'status': 'verified', 'endpoint_reconstructions': len(rebuilt) * 2,
            'extremum_inequalities': len(rebuilt) * 2,
            'intersection_order': cmp_pair(lower, upper),
            'closed_equality_is_nonempty': True}

def rational_output(value):
    with localcontext() as ctx:
        ctx.prec = 18
        display = str(Decimal(value.numerator) / Decimal(value.denominator))
    return {'numerator': str(value.numerator), 'denominator': str(value.denominator), 'display': display}

def solve(qualified, progress=None):
    constraints = []
    for key, values, coverage in qualified:
        for rep, field in enumerate(('R_Q_before', 'R_L_before')):
            r, nu = values[field], values['nu']
            constraints.append({'key': key + (rep,), 'R': r, 'nu': nu,
                                'lower': max(Fraction(0), r - 10 * nu), 'upper': r + 10 * nu,
                                'coverage': coverage, 'mu': values['mu']})
            if progress is not None: progress['constraints_computed'] += 1
    constraints.sort(key=lambda x: x['key'])
    machines = []
    for machine in range(10):
        rows = [r for r in constraints if r['key'][0] == machine]
        if len(rows) != 36: raise QualificationUnknown('all36 constraints required')
        lower = max(x['lower'] for x in rows); upper = min(x['upper'] for x in rows)
        lw = next(x['key'] for x in rows if x['lower'] == lower)
        uw = next(x['key'] for x in rows if x['upper'] == upper)
        cert_rows = [{'key': x['key'], **{name + '_ratio': (x[name].numerator, x[name].denominator)
                     for name in ('R', 'nu', 'lower', 'upper')}} for x in rows]
        cert = integer_certificate(cert_rows, lw, uw,
                                   (lower.numerator, lower.denominator), (upper.numerator, upper.denominator))
        machines.append({'machine': machine, 'constraints': 36, 'L': rational_output(lower),
                         'U': rational_output(upper), 'lower_witness': lw, 'upper_witness': uw,
                         'classification': 'empty' if lower > upper else 'not_excluded', 'integer_certificate': cert})
        if progress is not None: progress['machine_intersections_computed'] += 1
    public = [{'key': x['key'], 'R_hat': rational_output(x['R']), 'nu_hat': rational_output(x['nu']),
               'lower': rational_output(x['lower']), 'upper': rational_output(x['upper']),
               'coverage': x['coverage'], 'mu': rational_output(x['mu'])} for x in constraints]
    return public, machines

def native_identity():
    result = {'pid': os.getpid(), 'creation_FILETIME_100ns': None}
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        times = [wintypes.FILETIME() for _ in range(4)]
        if kernel.GetProcessTimes(kernel.GetCurrentProcess(), *[ctypes.byref(x) for x in times]):
            result['creation_FILETIME_100ns'] = times[0].dwHighDateTime << 32 | times[0].dwLowDateTime
    return result

def write_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')

def analyze():
    release = checked_release()
    PRIVATE.mkdir(parents=True, exist_ok=False)
    write_once(PRIVATE / 'started.json', {'native_interpreter': native_identity(), 'release': release})
    start = time.perf_counter(); cpu = time.process_time()
    counts = {'source_files_verified': 0, 'source_files_reverified': 0,
              'numeric_JSON_documents_read': 0, 'JSON_parse_paths': 0,
              'original_cells_qualified': 0, 'token_fields_verified': 0,
              'constraints_computed': 0, 'machine_intersections_computed': 0}
    try:
        bindings = read_control(ROOT / FROZEN['source_bindings'][0])
        if len(bindings['files']) != 8: raise QualificationUnknown('exact8 assets required')
        paths = []
        for e in bindings['files']:
            p = (ROOT / e['path']).resolve(); p.relative_to(ROOT.resolve())
            if p.stat().st_size != e['bytes'] or sha(p) != e['sha256']:
                raise QualificationUnknown('source asset identity differs')
            paths.append(p); counts['source_files_verified'] += 1
        if len(set(paths)) != 8: raise QualificationUnknown('duplicate source assets')
        numeric = [e for e in bindings['files'] if e['role'] == 'published_numeric_source']
        if len(numeric) != 1 or numeric[0]['path'] != bindings['numeric_source']:
            raise QualificationUnknown('unique numeric source differs')
        text = (ROOT / numeric[0]['path']).read_text(encoding='utf-8-sig')
        counts['numeric_JSON_documents_read'] = 1
        normal = parse_document(text); counts['JSON_parse_paths'] += 1
        tokens = parse_document(text, True); counts['JSON_parse_paths'] += 1
        qualified = qualify(normal, tokens, counts)
        constraints, machines = solve(qualified, counts)
        for e, p in zip(bindings['files'], paths):
            if p.stat().st_size != e['bytes'] or sha(p) != e['sha256']:
                raise QualificationUnknown('source asset changed')
            counts['source_files_reverified'] += 1
        result = {'status': 'COMPLETED', 'qualification': 'verified', 'release': release,
                  'counts': counts, 'constraints': constraints, 'machines': machines,
                  'scope': 'Published binary64 norms/nu treated as exact inputs only; exact integer10 threshold. No certified source-vector or physical-system impossibility.',
                  'units': 's^-2', 'all8_source_assets_unchanged': True,
                  'input_rounding_error_envelope': 'unknown', 'new_parameter_estimates': 0,
                  'wall_seconds': time.perf_counter() - start, 'interpreter_CPU_seconds': time.process_time() - cpu,
                  'project_total_CPU_seconds': None, 'LLM_seconds': None,
                  'exact_operation_count': None, 'integer_endpoint_reconstructions': 720,
                  'integer_extremum_inequalities': 720, 'model_adopted': False}
        write_once(PRIVATE / 'manifest.json', result)
        write_once(PUBLIC / 'analysis_summary.json', result)
        write_once(PRIVATE / 'complete.json', {'status': 'COMPLETED', 'manifest_sha256': sha(PRIVATE / 'manifest.json'),
                                              'summary_sha256': sha(PUBLIC / 'analysis_summary.json')})
        return result
    except BaseException as exc:
        detail = {'status': 'unknown', 'counts': counts, 'reason': str(exc),
                  'wall_seconds': time.perf_counter() - start, 'interpreter_CPU_seconds': time.process_time() - cpu,
                  'automatic_retry': False}
        write_once(PRIVATE / 'failure.json', {**detail, 'traceback': traceback.format_exc()})
        write_once(PUBLIC / 'analysis_unknown.json', detail)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['analyze'])
    parser.parse_args(); analyze()
