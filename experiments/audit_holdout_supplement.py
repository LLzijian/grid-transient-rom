"""Mechanical TEST coverage QA only; never compute a predictor score."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from audit_private_dataset import audit


def main():
    old = Path('private_data/ieee39_expansion')
    new = Path('private_data/ieee39_holdout_supplement')
    audit(new)
    with (old/'dataset_index.csv').open() as stream:
        old_rows = list(csv.DictReader(stream))
    with (new/'dataset_index.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    splits = json.loads((new/'splits.json').read_text())
    assert not splits['train'] and not splits['validation'] and len(splits['test']) == len(rows)
    assert not {r['event_id'] for r in rows}&{r['event_id'] for r in old_rows}
    def key(row):
        return int(row['fault_bus']), float(row['fault_duration_s']), float(row['fault_xf'])
    assert not {key(r) for r in rows}&{key(r) for r in old_rows}
    assert len({key(r) for r in rows}) == len(rows)
    assert {int(r['fault_bus']) for r in rows} == {37, 38, 39}
    with h5py.File(old/'dataset.h5', 'r') as a, h5py.File(new/'dataset.h5', 'r') as b:
        for name in ('postfault_signature', 'case_sha256', 'state_mapping', 'algebraic_names', 'omega_s', 'mva_base'):
            assert a.attrs[name] == b.attrs[name], name
        for name in a['equilibrium']:
            np.testing.assert_array_equal(a['equilibrium'][name][:], b['equilibrium'][name][:])
        np.testing.assert_array_equal(a['inertia'][:], b['inertia'][:])
        ca, cb = json.loads(a.attrs['config']), json.loads(b.attrs['config'])
        for name in ca:
            if name not in ('seed', 'pilot_buses', 'pilot_durations_s', 'validation_buses', 'notes'):
                assert ca[name] == cb[name], name
        assert int(b.attrs['attempted']) == 42
        rejected = json.loads(b.attrs['discard_counts'])
        generation_commit = b.attrs['git_commit']
        signature = b.attrs['postfault_signature']
    coverage = {str(t): {str(bus): {label: sum(int(r['fault_bus']) == bus and r['label'] == label
                    and float(r['trajectory_end_s']) > t+1e-9 for r in rows) for label in ('stable', 'unstable')}
                    for bus in (37, 38, 39)} for t in (.05, .1, .2)}
    passed = sum(v['unstable'] for v in coverage['0.2'].values()) >= 1
    result = {'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'generation_commit': generation_commit, 'postfault_signature': signature, 'raw_qa_pass': True,
        'coverage_pass': passed, 'attempted': 42, 'accepted': len(rows), 'discarded': rejected,
        'counts': {label: sum(r['label'] == label for r in rows) for label in ('stable', 'unstable')},
        'predictive_coverage_by_window_and_bus': coverage, 'split_counts': {k: len(v) for k, v in splits.items()},
        'hashes': {str(root/name): hashlib.sha256((root/name).read_bytes()).hexdigest()
                   for root in (old, new) for name in ('dataset_index.csv', 'splits.json')},
        'scope': 'Raw TEST-only coverage QA. No model score, parameter, observer, or threshold selection.'}
    with (new/'coverage_evidence.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
