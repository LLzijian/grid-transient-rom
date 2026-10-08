"""Bounded reuse of the frozen trajectory/mode and full-TRAIN evaluators.

The nested files retain their inherited implementation label 0098; this outer
manifest identifies the actual experiment/checkpoint without rewriting evidence.
No fitting, checkpoint selection or TEST access is performed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(checkpoint, output, cap=750.):
    if not 0 < cap <= 750:
        raise ValueError('one checkpoint evaluation allocation must be <=750 seconds')
    start = time.perf_counter()
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [checkpoint / 'model.pt', checkpoint / 'checkpoint.json', Path(__file__),
             ROOT / 'experiments/evaluate_local_memory_0098.py',
             ROOT / 'experiments/summarize_train_local_memory_0098.py']
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    evidence = {'loop': '0099', 'scope': 'frozen endpoint evaluation; not selection',
                'checkpoint': str(checkpoint), 'cap_seconds': cap, 'jobs': [],
                'test_read': False, 'source_before': before}
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    try:
        for kind, script in [('validation', 'evaluate_local_memory_0098.py'),
                             ('train', 'summarize_train_local_memory_0098.py')]:
            remaining = cap - (time.perf_counter() - start) - 15
            if remaining <= 0:
                evidence['status'] = 'resource_limited'; break
            command = [sys.executable, str(ROOT/'experiments'/script),
                       '--checkpoint', str(checkpoint), '--output', str(output/kind)]
            row = {'kind': kind, 'command': command}
            began = time.perf_counter()
            with (output/f'{kind}_stdout.log').open('w', encoding='utf-8') as stdout, \
                 (output/f'{kind}_stderr.log').open('w', encoding='utf-8') as stderr:
                try:
                    result = subprocess.run(command, cwd=ROOT, env=environment,
                                            stdout=stdout, stderr=stderr, timeout=remaining)
                    row.update(exit_code=result.returncode,
                               status='complete' if result.returncode == 0 else 'failed')
                except subprocess.TimeoutExpired:
                    row['status'] = 'resource_limited'
            row['seconds'] = time.perf_counter() - began
            evidence['jobs'].append(row)
            print(json.dumps({'kind': kind, 'status': row['status'], 'seconds': row['seconds']}), flush=True)
            if row['status'] != 'complete':
                evidence['status'] = row['status']; break
        else:
            evidence['status'] = 'complete'
    finally:
        evidence['source_after'] = hashes()
        evidence['sources_unchanged'] = before == evidence['source_after']
        if not evidence['sources_unchanged']:
            evidence['status'] = 'quarantined_source_drift'
        evidence['seconds'] = time.perf_counter() - start
        evidence['within_cap'] = evidence['seconds'] <= cap
        if not evidence['within_cap']:
            evidence['status'] = 'budget_exception'
        (output/'evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cap', type=float, default=750.)
    args = parser.parse_args()
    result = run(args.checkpoint, args.output, args.cap)
    print(json.dumps({'status': result['status'], 'seconds': result['seconds']}))
    raise SystemExit(0 if result['status'] == 'complete' else 2)
