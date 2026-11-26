"""Run the exact algebraic controls and existing escape check before any fit."""

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

from experiments.osgkd_v2_layer1.escape_example import run_checks as escape_checks
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
from experiments.osgkd_v2_wp2_layer1_20261127 import negative_controls


RUN = 'osgkd_v2_wp2_layer1_20261127'
PUBLIC = ROOT/'results'/RUN
SOURCES = {
    'campaigns/osgkd_v2/wp2_layer1_execution_protocol.md', 'campaigns/osgkd_v2/wp2_layer1_fit_schedule.md',
    'paper/osgkd_v2_methods_and_theory.md', 'src/kiki/data/osgkd_synthetic.py',
    'experiments/osgkd_v2_sprint2a_20261125/resume_host.py', 'experiments/osgkd_v2_layer1/escape_example.py',
    *('experiments/'+RUN+'/'+name for name in ('controls.py','negative_controls.py','synthetic.py')),
}


def execute(path):
    source = io.read(path)
    io.require(path == PUBLIC/'controls_release.json' and source['run_id']==RUN and source['released'] is True,
               'canonical control release required')
    io.require(Path(sys.executable).resolve()==(ROOT.parent/'.venv/Scripts/python.exe').resolve()
               and source['new_fom']==0 and source['test_trajectory_access'] is False,
               'shared interpreter and control-only scope required')
    io.require(SOURCES <= set(source['source_hashes']) and source['scientist']['configured_model']=='gpt-6-astra'
               and source['independent_preflight']['status']=='passed', 'complete controls source review required')
    for name, digest in source['source_hashes'].items():
        local = (ROOT/name).resolve()
        io.require(local.is_relative_to(ROOT) and io.identity(local)['sha256']==digest, 'control source differs: '+name)
    output = PUBLIC/'controls.json'
    io.require(not output.exists(), 'frozen control results already exist')
    start,cpu=perf_counter(),process_time()
    checks = negative_controls.audit()
    grid = checks['condition_map']['rows']
    io.require(len(grid)==20 and all(row['algebraic_class']=='full_rank' for row in grid if row['excitation']>0
               and row['environmental_waveform_overlap']<1), 'positive-control design lacks specified rank')
    io.require(all(row['algebraic_class']=='equivalent_parameters' for row in grid if row['excitation']>0
               and row['environmental_waveform_overlap']==1), 'exact overlap does not induce equivalence')
    result = dict(run_id=RUN, job_id='controls01', status='completed', release=io.identity(path),
                  negative_controls=checks, escape_example=escape_checks(), new_fom=0,
                  test_trajectory_access=False, ieee39_event_access=False, updates=0,
                  wall_s=perf_counter()-start, cpu_s=process_time()-cpu, completed_utc=io.utc())
    io.write(output,result)
    print(io.identity(output),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    execute(parser.parse_args().release.resolve())
