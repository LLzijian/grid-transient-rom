"""JSON aggregation checks only; no dataset, inference or val reads."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import time

started = time.perf_counter()
root = Path(__file__).resolve().parents[2]
source = root / 'private_data/closure_mechanism_20261019/P3/fresh_design_pair01/extract_metrics_once.py'
ast.parse(source.read_text(encoding='utf-8'))
m = runpy.run_path(str(source))
assert math.isclose(m['risk']([1., 2., 3.]), 2.2)
assert m['risk']([1., None]) is None
rows = []
for i, n, error in ((0, 2, 1.), (1, 8, 2.)):
    pair = dict(finite_to_endpoint=True, los_pair_pass=True,
                coi_normalized_rmse=0., common_normalized_rmse=0.,
                fine_classification=dict(category='window_no_los_unsettled', predicted_los=False))
    actual = dict(available=True, finite=True, numerically_admissible=True,
                  n_time_points=n, frequency_rmse_rad_s=error, power_rmse_system_pu=error,
                  coi_normalized_rmse=0., common_normalized_rmse=0., mean_point_error=error)
    physical = dict(available=True, frequency=dict(finite=True, physical_sse=error**2*n, n_scalar_points=n),
                    mechanical_power=dict(finite=True, physical_sse=error**2*n*10, n_scalar_points=n*10))
    rows.append(dict(event_id=str(i), label='unstable', scientific_numeric_8s=True,
                     classification_8s=pair, classification_15s=copy.deepcopy(pair),
                     actual_interval15=actual, physical_1_2_5_8s={'prefix': physical}))
for family in ('frequency', 'mechanical_power'):
    a = m['pooled'](rows, family, 'prefix')
    b = m['pooled'](rows, family, None, True)
    assert math.isclose(a['rmse'], math.sqrt(3.4)) and math.isclose(a['rmse'], b['rmse'])
bad = copy.deepcopy(rows)
bad[0]['scientific_numeric_8s'] = False
assert m['pooled'](bad, 'frequency', 'prefix')['rmse'] is None
assert m['pooled'](bad, 'frequency', None, True)['rmse'] is not None
assert m['classifications'](bad, 8)['numerical_unresolved'] == 1
assert m['classifications'](bad, 15)['missed'] == 2
assert m['classifications'](bad, 15)['undetermined'] == 2
models = {str(i): {'new_val': {'primary_original_ceil_risk': {'value': 1.},
             'classification_8s': {'numerical_unresolved': 0},
             'classification_15s': {'numerical_unresolved': 0}}} for i in range(6)}
assert m['val_coverage'](19, models)['whole24_resolved'] is False
assert m['val_coverage'](19, models)['missing_count'] == 5
assert m['val_coverage'](24, models)['whole24_resolved'] is True
ids = m['cohort_ids'](rows, 2)
assert m['cohort_ids'](list(reversed(rows)), 2, ids) == ids
try:
    m['cohort_ids'](rows, 2, {'different', 'cohort'})
except AssertionError:
    pass
else:
    raise AssertionError('mismatched IDs accepted')
print(json.dumps({'status': 'PASS', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'checks': ['ceil risk', 'SSE/scalar count pooling', '8s vs15s isolation',
                             'unresolved retention', 'classification counts', '24-attempt denominator', 'event identity'],
                  'seconds': time.perf_counter()-started, 'actual_data_read': False,
                  'val_paths_read': False, 'test_read': False}))
