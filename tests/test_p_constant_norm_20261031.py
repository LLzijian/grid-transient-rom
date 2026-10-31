"""Deterministic scalar/schema fixtures only. Never open a real numeric source."""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('constant_norm_under_test', Path(__file__).resolve().parents[1] / 'experiments/p_constant_norm_20261031.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def decimal_token(frac):
    with localcontext() as ctx:
        ctx.prec = 1200
        return m.NumberToken(str(Decimal(frac.numerator) / Decimal(frac.denominator)))

def fixture():
    data = {'overall60': [], 'stage120': []}
    for machine in range(10):
        for group, steps in itertools.product(m.DENOM, (4, 8)):
            n = m.DENOM[group]
            data['overall60'].append(dict(machine=machine, group=group, steps=steps, N_original=n,
                N_available=n, windows=n, mu=1.0, R_Q_before=1.0, R_L_before=1.0, nu=0.125))
        for group, steps, phase in itertools.product(('stable', 'unstable'), (4, 8), ('history', 'cross_t0', 'future')):
            n = m.DENOM[group]; available = n
            if group == 'unstable':
                if phase == 'future': available = 27 if steps == 4 else 22
                elif phase == 'cross_t0': available = 27
            data['stage120'].append(dict(machine=machine, group=group, steps=steps, phase=phase,
                N_original=n, N_available=available, windows=available, mu=0.25,
                R_Q_before=1.0, R_L_before=1.0, nu=0.125))
    return data

def qualify(data):
    text = json.dumps(data)
    return m.qualify(m.parse_document(text), m.parse_document(text, True))

class ScalarParsing(unittest.TestCase):
    def test_ordinary_decimal(self):
        self.assertEqual(m.nearest_even(m.NumberToken('0.1'), 0.1), F(*(.1).as_integer_ratio()))

    def test_wrong_adjacent_float_rejected(self):
        with self.assertRaises(m.QualificationUnknown): m.nearest_even(m.NumberToken('1'), math.nextafter(1., math.inf))

    def test_even_tie_both_sides(self):
        v = 1.; exact = F(1); low = (F(*math.nextafter(v, -math.inf).as_integer_ratio()) + exact) / 2
        high = (exact + F(*math.nextafter(v, math.inf).as_integer_ratio())) / 2
        for token in (decimal_token(low), decimal_token(high)):
            self.assertEqual(m.nearest_even(token, v), exact)

    def test_odd_tie_rejected_both_sides(self):
        v = math.nextafter(1., math.inf); f = F(*v.as_integer_ratio())
        for neighbor in (math.nextafter(v, -math.inf), math.nextafter(v, math.inf)):
            with self.assertRaises(m.QualificationUnknown):
                m.nearest_even(decimal_token((f + F(*neighbor.as_integer_ratio())) / 2), v)

    def test_strictly_outside_rounding_cell(self):
        upper = (F(1) + F(*math.nextafter(1., math.inf).as_integer_ratio())) / 2
        with self.assertRaises(m.QualificationUnknown): m.nearest_even(decimal_token(upper + F(1, 2**110)), 1.)

    def test_zero_and_signed_zero(self):
        for token, value in [('0', 0), ('-0.0', -0.0), ('0e10', 0.)]:
            self.assertEqual(m.nearest_even(m.NumberToken(token), value), 0)

    def test_zero_subnormal_midpoint(self):
        smallest = math.ulp(0.); tie = F(*smallest.as_integer_ratio()) / 2
        self.assertEqual(m.nearest_even(decimal_token(tie), 0.), 0)
        with self.assertRaises(m.QualificationUnknown): m.nearest_even(decimal_token(tie), smallest)

    def test_smallest_and_other_subnormal(self):
        for value in (math.ulp(0.), 2 * math.ulp(0.), 101 * math.ulp(0.)):
            self.assertEqual(m.nearest_even(decimal_token(F(*value.as_integer_ratio())), value), F(*value.as_integer_ratio()))

    def test_maxfinite_negative_nonfinite_bool_rejected(self):
        for value in (sys.float_info.max, -1., math.inf, math.nan, True):
            with self.assertRaises(m.QualificationUnknown): m.nearest_even(m.NumberToken(str(value)), value)

    def test_largest_supported_neighbor(self):
        value = math.nextafter(sys.float_info.max, 0.)
        self.assertEqual(m.nearest_even(decimal_token(F(*value.as_integer_ratio())), value), F(*value.as_integer_ratio()))

    def test_duplicate_keys_both_paths(self):
        for tokens in (False, True):
            with self.assertRaises(m.QualificationUnknown): m.parse_document('{"a":1,"a":2}', tokens)

    def test_bad_json_nonfinite_and_overflow(self):
        for text in ('{"a":', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
            with self.assertRaises((ValueError, m.QualificationUnknown)): m.parse_document(text)

class SchemaAndArithmetic(unittest.TestCase):
    def test_complete_fixture_and_720_fields(self):
        data = fixture(); text = json.dumps(data); progress = {'token_fields_verified': 0, 'original_cells_qualified': 0}
        q = m.qualify(m.parse_document(text), m.parse_document(text, True), progress)
        self.assertEqual(len(q), 180); self.assertEqual(progress, {'token_fields_verified': 720, 'original_cells_qualified': 180})

    def test_missing_and_duplicate_axes(self):
        for change in ('missing', 'duplicate'):
            d = fixture()
            if change == 'missing': d['overall60'].pop()
            else: d['stage120'][1] = d['stage120'][0]
            with self.assertRaises(m.QualificationUnknown): qualify(d)

    def test_bool_and_float_identity_fields(self):
        for field, bad in [('machine', True), ('steps', 4.), ('N_original', True), ('N_available', True), ('windows', 1.)]:
            d = fixture(); d['overall60'][0][field] = bad
            with self.assertRaises(m.QualificationUnknown): qualify(d)

    def test_coverage_and_mass_rejected(self):
        for field, bad in [('N_available', 266), ('N_original', 30), ('mu', 0.), ('mu', 1.01), ('windows', 0)]:
            d = fixture(); d['stage120'][0][field] = bad
            with self.assertRaises(m.QualificationUnknown): qualify(d)

    def test_crossing_original_denominator_separate_from_available(self):
        q = qualify(fixture())
        rows = [x for x in q if x[0][1:] == (1, 'unstable', 8, 'cross_t0')]
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0][2]['N_original'], 30); self.assertEqual(rows[0][2]['N_available'], 27)

    def test_complete_constraints_and_fixed_tie_witness(self):
        constraints, machines = m.solve(qualify(fixture()))
        self.assertEqual(len(constraints), 360); self.assertEqual(len(machines), 10)
        for row in machines:
            self.assertEqual(row['classification'], 'not_excluded')
            self.assertEqual(row['lower_witness'], (row['machine'], 0, 'all', 4, '', 0))
            self.assertEqual(row['upper_witness'], row['lower_witness'])

    def test_closed_equal_empty_and_nonempty(self):
        q = qualify(fixture())
        for values, wanted in [((1, 0), 'not_excluded'), ((1, 1), 'not_excluded')]:
            changed = [(k, {**v, 'R_Q_before': F(values[0]), 'R_L_before': F(values[0]), 'nu': F(values[1])}, c) for k, v, c in q]
            _, rows = m.solve(changed); self.assertTrue(all(x['classification'] == wanted for x in rows))
        changed = [(k, {**v, 'R_Q_before': F(0 if k[1] == 0 else 2), 'R_L_before': F(0 if k[1] == 0 else 2), 'nu': F(0)}, c) for k, v, c in q]
        _, rows = m.solve(changed); self.assertTrue(all(x['classification'] == 'empty' for x in rows))

    def test_integer_certificate_rejects_fake_endpoint_and_witness(self):
        rows = [{'key': (0, 0), 'R_ratio': (2, 1), 'nu_ratio': (0, 1), 'lower_ratio': (2, 1), 'upper_ratio': (2, 1)},
                {'key': (0, 1), 'R_ratio': (2, 1), 'nu_ratio': (0, 1), 'lower_ratio': (2, 1), 'upper_ratio': (2, 1)}]
        cert = m.integer_certificate(rows, (0, 0), (0, 0), (2, 1), (2, 1)); self.assertEqual(cert['intersection_order'], 0)
        with self.assertRaises(m.QualificationUnknown): m.integer_certificate(rows, (0, 1), (0, 0), (2, 1), (2, 1))
        bad = copy.deepcopy(rows); bad[1]['lower_ratio'] = (3, 1)
        with self.assertRaises(m.QualificationUnknown): m.integer_certificate(bad, (0, 0), (0, 0), (2, 1), (2, 1))

    def test_integer_endpoint_independent_exact10(self):
        r, nu = F(7, 3), F(1, 11)
        lo, hi = m.int_endpoints((r.numerator, r.denominator), (nu.numerator, nu.denominator))
        self.assertEqual(F(*lo), r-10*nu); self.assertEqual(F(*hi), r+10*nu)

    def test_release_expected_never_reads_parent(self):
        opened = []
        def fake_sha(path): opened.append(str(path)); return 'fixture-hash'
        with patch.object(m, 'sha', fake_sha): result = m.release_expected()
        self.assertEqual(result['allowed_command'], 'analyze')
        self.assertFalse(any('p_constant_direction_20261031' in p or 'private_data' in p for p in opened))

if __name__ == '__main__': unittest.main()
