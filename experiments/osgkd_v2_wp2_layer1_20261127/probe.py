"""Isolated CPU float64 long-window resource probe. Never updates a parameter."""

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, process_time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import torch

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
from experiments.osgkd_v2_wp2_layer1_20261127 import engine, models, synthetic
from kiki.data.osgkd_synthetic import model_constants, motion_packet


RUN = 'osgkd_v2_wp2_layer1_20261127'
SIZES = (8, 16, 32)
BATCH_SIZES = (1, 2, 4)
WINDOWS = (200, 400)
DT = .01
REPEATS = 3
REQUIRED_SOURCES = {
    'campaigns/osgkd_v2/wp2_layer1_execution_protocol.md', 'campaigns/osgkd_v2/sprint3_layer1_protocol.md',
    'src/kiki/data/osgkd_synthetic.py', 'src/kiki/models/osgkd.py', 'src/kiki/models/osgkd_graph.py',
    'src/kiki/models/osgkd_baselines.py', 'src/kiki/models/osgkd_fitting.py',
    'experiments/osgkd_v2_sprint2a_20261125/resume_host.py',
    *('experiments/' + RUN + '/' + name for name in ('probe.py', 'engine.py', 'models.py', 'synthetic.py', 'launch.ps1')),
}


class ProcessMemory(ctypes.Structure):
    _fields_ = [
        ('length', wintypes.DWORD), ('fault_count', wintypes.DWORD),
        *((name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged_pool', 'paged_pool',
           'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile')),
    ]


def process_peak_rss():
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = wintypes.HANDLE
    read = ctypes.windll.psapi.GetProcessMemoryInfo
    read.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    read.restype = wintypes.BOOL
    counters = ProcessMemory()
    counters.length = ctypes.sizeof(counters)
    if not read(process(), ctypes.byref(counters), counters.length):
        raise OSError('GetProcessMemoryInfo failed')
    return counters.peak_rss


def parameter_digest(model):
    digest = hashlib.sha256()
    for param in model.parameters():
        digest.update(param.detach().numpy().tobytes())
    return digest.hexdigest()


def cell(n, batch_size, steps):
    torch.set_num_threads(1)
    system = synthetic.make_system(n=n)
    observations = []
    mesh_checks = []
    for i in range(batch_size):
        seed = 73100 + i
        kind = synthetic.EXCITATIONS[i]
        observed, truth = synthetic.generate_episode(system, seed, kind, forecast_s=4., noise_levels=(.01,))
        _, fine_truth = synthetic.generate_episode(system, seed, kind, forecast_s=4., noise_levels=(.01,), internal_dt=.00125)
        mesh_checks.append(float(np.max(np.abs(truth['state'] - fine_truth['state']))))
        observations.append(observed['0.01'])
    io.require(max(mesh_checks) <= 1e-6, 'synthetic 4.1s truth grid did not qualify')
    constants = model_constants(motion_packet(observations[0]))
    model = models.make_model('osgkd', constants, 24121)
    batch = engine.Episodes.from_events(observations)
    before = parameter_digest(model)
    memory_before = io.memory()
    rss_before = process_peak_rss()
    measures = []
    for iteration in range(REPEATS+1):
        model.zero_grad(set_to_none=True)
        wall, cpu = perf_counter(), process_time()
        value = engine.loss(model, batch, steps, DT)
        value.backward()
        end_wall, end_cpu = perf_counter()-wall, process_time()-cpu
        io.require(bool(value.detach().isfinite()) and all(p.grad is None or bool(p.grad.isfinite().all())
                   for p in model.parameters()), 'nonfinite dry-run gradient')
        if iteration:
            measures.append({'wall_s': end_wall, 'cpu_s': end_cpu, 'process_peak_rss_bytes': process_peak_rss()})
    memory_after = io.memory()
    after = parameter_digest(model)
    io.require(after == before, 'dry-run parameters changed')
    return dict(status='completed', n=n, candidate_edges=len(system['candidate_edges']), batch_size=batch_size,
                prediction_s=steps*.01, internal_step=DT, repeats=measures,
                median_wall_s=float(np.median([row['wall_s'] for row in measures])),
                peak_rss_bytes=max(row['process_peak_rss_bytes'] for row in measures), pre_step_peak_rss_bytes=rss_before,
                system_memory_before=memory_before, system_memory_after=memory_after,
                truth_mesh_max_difference=max(mesh_checks), initial_parameter_sha256=before,
                final_parameter_sha256=after, parameters_identical=True,
                updates=0, weights_written=False, ieee39_event_access=False, dtype='float64', threads=1)


def matrix(release):
    folder = ROOT/'artifacts'/RUN/'dry01'
    out = ROOT/'results'/RUN/'dry01/summary.json'
    folder.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    io.require(release == ROOT/'results'/RUN/'dry_release.json' and not out.exists(), 'one registered dry run required')
    io.require(Path(sys.executable).resolve() == (ROOT.parent/'.venv/Scripts/python.exe').resolve(),
               'shared Python environment required')
    registration = io.read(release)
    io.require(registration['run_id'] == RUN and registration['released'] is True and
               registration['scientist']['configured_model'] == 'gpt-6-astra' and
               registration['independent_preflight']['status'] == 'passed', 'dry release or preflight differs')
    io.require(registration['sizes'] == list(SIZES) and registration['batch_sizes'] == list(BATCH_SIZES)
               and registration['windows'] == list(WINDOWS) and registration['model_dt'] == DT
               and registration['repeats'] == REPEATS and registration['updates'] == 0
               and registration['new_fom'] == 0 and registration['test_trajectory_access'] is False,
               'registered matrix differs')
    io.require(REQUIRED_SOURCES <= set(registration['source_hashes']), 'required dry-run sources missing')
    for source, digest in registration['source_hashes'].items():
        path = (ROOT/source).resolve()
        io.require(path.is_relative_to(ROOT) and io.identity(path)['sha256'] == digest, 'source differs: '+source)
    ref = io.identity(release)
    env = os.environ | dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    rows = []
    for n in SIZES:
        for b in BATCH_SIZES:
            for steps in WINDOWS:
                stem = f'n{n}_b{b}_t{steps}'
                receipt = folder/(stem+'.json')
                if receipt.exists():
                    row = io.read(receipt)
                    io.require(row['release'] == ref, 'dry cell release differs')
                else:
                    argv = [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), '--cell', str(n), str(b), str(steps)]
                    start = perf_counter()
                    run = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
                    process_wall = perf_counter() - start
                    (folder / (stem+'_stderr.log')).write_text(run.stderr, encoding='utf-8')
                    if run.returncode == 0:
                        row = json.loads(run.stdout)
                    else:
                        row = {'status': 'failed', 'exit_code': run.returncode, 'n': n, 'batch_size': b,
                               'prediction_s': steps*.01, 'total_cpu_s': None, 'total_peak_rss_bytes': None}
                    row.update(process_wall_s=process_wall, exit_code=run.returncode)
                    if row['status'] == 'failed':
                        row['stderr'] = io.identity(folder/(stem+'_stderr.log'))
                    row['release'] = ref
                    io.write(receipt, row)
                rows.append(row)
                print(n, b, steps, row['status'], row.get('median_wall_s'), flush=True)
    io.write(out, dict(run_id=RUN, job_id='dry01', status='completed', cells=rows, release=ref,
                       machine_context=registration['machine_context'],
                       updates=0, new_fom=0, test_trajectory_access=False, completed_utc=io.utc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--release', type=Path)
    mode.add_argument('--cell', nargs=3, type=int)
    args = parser.parse_args()
    if args.cell:
        n, b, steps = args.cell
        if n not in SIZES or b not in BATCH_SIZES or steps not in WINDOWS:
            parser.error('cell outside the registered matrix')
        start, cpu_start = perf_counter(), process_time()
        try:
            row = cell(n, b, steps)
        except Exception as error:
            traceback.print_exc()
            row = dict(status='failed', error=repr(error), n=n, batch_size=b, prediction_s=steps*.01)
        row.update(total_cell_wall_s=perf_counter()-start, total_cpu_s=process_time()-cpu_start,
                   total_peak_rss_bytes=process_peak_rss())
        print(json.dumps(row, allow_nan=False))
    else:
        matrix(args.release.resolve())
