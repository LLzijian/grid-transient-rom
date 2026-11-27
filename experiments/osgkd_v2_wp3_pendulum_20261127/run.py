"""Release-bound train-only pendulum matrix; target val belongs to score.py."""

import argparse
from functools import lru_cache
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
for env_name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[env_name] = '1'

import numpy as np
import torch

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
from experiments.osgkd_v2_wp2_layer1_20261127 import engine, training
from experiments.osgkd_v2_wp3_pendulum_20261127 import pendulum
from kiki.models.osgkd import OSGKDConfig


RUN = 'osgkd_v2_wp3_pendulum_20261127'
SOURCE_RUN = 'osgkd_v2_wp2_layer1_20261127'
SOURCE_TASK = 'n8_open_noise0.01_s24121'
LOCAL = ROOT/'artifacts'/RUN/'fit'
PUBLIC = ROOT/'results'/RUN
SYSTEM_SEEDS = (31, 41, 51)
OPTIMIZATION_SEED = 24121
METHODS = ('osgkd_full', 'osgkd_few', 'source_frozen', 'random_frozen',
           'neural_full', 'neural_few', 'closed_passive', 'closed_conservative')
FEW_METHODS = ('osgkd_few', 'source_frozen', 'random_frozen', 'neural_few')
WEIGHTS = ((0, 100, .4), (100, 200, .3), (200, 400, .3))
HORIZON_BLOCKS = ({'attempts': 64, 'steps': 100}, {'attempts': 64, 'steps': 200},
                  {'attempts': 64, 'steps': 400})
HORIZONS = [100]*64+[200]*64+[400]*64
CONSTANTS = ('mass', 'damping', 'F', 'Sp', 'Se', 'omega_scale', 'theta_ref', 'edge_index', 'gravity')
REQUIRED_SOURCES = {
    'campaigns/osgkd_v2/wp3_pendulum_protocol.md',
    'src/kiki/data/osgkd_synthetic.py', 'src/kiki/models/osgkd.py',
    'src/kiki/models/osgkd_graph.py', 'src/kiki/models/osgkd_baselines.py',
    'src/kiki/models/osgkd_fitting.py',
    'experiments/osgkd_v2_sprint2a_20261125/resume_host.py',
    *('experiments/'+SOURCE_RUN+'/'+name for name in ('engine.py', 'models.py', 'synthetic.py', 'training.py')),
    *('experiments/'+RUN+'/'+name for name in ('pendulum.py', 'run.py', 'score.py', 'launch.ps1')),
}


def job_key(system_seed, method):
    return f'sys{system_seed}_{method}'


def expected_matrix():
    return [dict(id=job_key(system_seed, method), system_seed=system_seed,
                 optimization_seed=OPTIMIZATION_SEED, method=method,
                 train_count=16 if method in FEW_METHODS else 64)
            for system_seed in SYSTEM_SEEDS for method in METHODS]


def val_definition():
    """Freeze generation identity without generating or opening target val."""
    return dict(n=8, system_seeds=list(SYSTEM_SEEDS), event_seeds=list(range(2000, 2016)),
                excitations=list(pendulum.EXCITATIONS), per_excitation=4, noise_levels=[0., .01],
                history_points=11, forecast_s=4., absolute_lab_angles=True)


def create_model(method, constants, seed=OPTIMIZATION_SEED):
    """Scratch target ODE; source encoder assignment is a separate operation."""
    if method not in METHODS:
        raise ValueError('unknown registered pendulum method')
    torch.manual_seed(seed)
    if method.startswith('neural_'):
        cls = pendulum.PendulumGraphNeuralODE
    elif method in ('closed_passive', 'closed_conservative'):
        cls = pendulum.PendulumClosedPassive
    else:
        cls = pendulum.PendulumOSGKDGraph
    inputs = {key: constants[key] for key in CONSTANTS}
    if method == 'closed_conservative':
        inputs['damping'] = np.zeros_like(np.asarray(constants['damping']))
    model = cls(OSGKDConfig('C1', hidden_size=20, a_init_bound=2.), **inputs).cpu().double()
    with torch.no_grad():
        for name in ('mechanical_coeff', 'amplitude_coeff', 'memory_coeff'):
            if hasattr(model, name):
                getattr(model, name)[0].fill_(-1.)
    if method in ('source_frozen', 'random_frozen', 'closed_passive', 'closed_conservative'):
        model.encoder.requires_grad_(False)
    return model


def _no_ieee_or_test(record):
    io.require(record.get('new_fom') == 0 and record.get('test_trajectory_access') is False
               and record.get('ieee39_event_access') is False, 'source access scope differs')


def source_endpoint(value):
    """Verify the fixed WP2 full endpoint and its independently bound val cell."""
    io.require(value['source_task'] == SOURCE_TASK, 'fixed source task differs')
    fit = io.read(io.verified(value['source_fit_summary']))
    evaluated = io.read(io.verified(value['source_evaluation']))
    for book in (fit, evaluated):
        io.require(book['run_id'] == SOURCE_RUN and book['status'] == 'completed', 'completed synthetic source required')
        _no_ieee_or_test(book)
    io.require(fit['synthetic_val_access'] is False, 'source fitting used val')
    evaluation_release = io.read(io.verified(evaluated['release']))
    io.require(evaluation_release['fit_summary'] == value['source_fit_summary'] and
               evaluation_release['fit_release'] == fit['release'], 'source score/fit lineage differs')
    fit_release = io.read(io.verified(fit['release']))
    io.require(fit_release['run_id'] == SOURCE_RUN and fit_release['released'] is True
               and fit_release['default_model'] == 'none', 'source release differs')
    terminal = io.read(io.verified(fit['cells'][SOURCE_TASK]['methods']['osgkd']))
    source_cell = io.read(io.verified(evaluated['cells'][SOURCE_TASK]))
    task = dict(n=8, condition='open', noise='0.01', seed=OPTIMIZATION_SEED)
    io.require(all(terminal['binding']['cell'][key] == val and source_cell['job'][key] == val
                   for key, val in task.items()), 'source physical/noise/optimization identity differs')
    io.require(terminal['binding']['release'] == fit['release'], 'source method authority differs')
    endpoint = terminal['variant_results']['full']
    validation = source_cell['methods']['osgkd']['variants']['full']
    io.require(endpoint['status'] == 'completed' and endpoint['checkpoint'] is not None and
               endpoint['accepted'] == 128 and endpoint['binding']['variant'] == 'full', 'fixed full source endpoint required')
    io.require(source_cell['val_count'] == 24 and source_cell['event_sha256'] == value['source_val_sha256']
               and validation['evaluated'] is True and validation['fit']['checkpoint'] == endpoint['checkpoint'],
               'source val identity or evaluated endpoint differs')
    saved = torch.load(io.verified(endpoint['checkpoint']), map_location='cpu', weights_only=True)
    io.require(saved['binding'] == endpoint['binding'] and len(saved['rows']) == 128,
               'source checkpoint stage differs')
    state = {key.removeprefix('encoder.'): tensor.detach().clone() for key, tensor in saved['state'].items()
             if key.startswith('encoder.')}
    io.require(state and all(tensor.isfinite().all() for tensor in state.values()), 'finite source encoder required')
    return state, dict(task=SOURCE_TASK, checkpoint=endpoint['checkpoint'],
                       encoder_tensor_sha256=parameter_digest(state), val_sha256=value['source_val_sha256'])


def parameter_digest(state):
    import hashlib
    result = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        result.update(key.encode())
        result.update(repr((str(tensor.dtype), tuple(tensor.shape))).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def validate(path):
    value = io.read(path)
    io.require(path == PUBLIC/'release.json' and value['run_id'] == RUN and value['released'] is True,
               'canonical released WP3 training authority required')
    io.require(Path(sys.executable).resolve() == (ROOT.parent/'.venv/Scripts/python.exe').resolve(),
               'shared Python interpreter required')
    io.require(REQUIRED_SOURCES <= set(value['source_hashes']), 'missing pendulum source binding')
    for name, sha in value['source_hashes'].items():
        document = (ROOT/name).resolve()
        io.require(document.is_relative_to(ROOT) and io.identity(document)['sha256'] == sha, 'source differs: '+name)
    scientist, reviewer = value['scientist'], value['independent_preflight']
    io.require(scientist['configured_model'] == 'gpt-6-astra' and scientist['task_name'] == '/root/wp3_scientist'
               and reviewer['status'] == 'passed' and reviewer['task_name'] != scientist['task_name'],
               'separate actual scientific/reviewer records required')
    for record in (scientist, reviewer):
        io.verified(record['evidence_ref'])
    io.require(value['jobs'] == expected_matrix() and value['forecast_s'] == 4. and value['attempts'] == 192
               and value['horizon_blocks'] == list(HORIZON_BLOCKS)
               and value['loss_weights'] == [list(item) for item in WEIGHTS]
               and value['target_val'] == val_definition(), 'registered pendulum schedule/data identity differs')
    io.require(value['microbatch'] in (1, 2, 4) and type(value['microbatch']) is int,
               'registered microbatch required')
    io.require(value['new_fom'] == 0 and value['test_trajectory_access'] is False
               and value['ieee39_event_access'] is False and value['default_model'] == 'none', 'access scope differs')
    io.require(value['train_noise'] == '0.00' and value['train_seed'] == 1000
               and value['truth_meshes'] == [.0025, .00125], 'training observation contract differs')
    source_endpoint(value)
    return value


@lru_cache(maxsize=1)
def train_observations(system_seed):
    """Only train generation; no call to make_dataset or target val generation."""
    io.require(system_seed in SYSTEM_SEEDS, 'registered target system seed required')
    system = pendulum.make_system(n=8, seed=system_seed)
    events, gaps = [], []
    for index in range(64):
        seed, excitation = 1000+index, pendulum.EXCITATIONS[index//16]
        observed, coarse = pendulum.generate_episode(system, seed, excitation, noise_levels=(0.,), forecast_s=4.)
        _, fine = pendulum.generate_episode(system, seed, excitation, noise_levels=(0.,), forecast_s=4., internal_dt=.00125)
        gaps.append(float(np.max(np.abs(coarse['state']-fine['state']))))
        events.append(observed['0.00'])
    io.require(max(gaps) <= 1e-6, 'pendulum train truth grids disagree')
    constants = pendulum.model_constants(pendulum.motion_packet(events[0]))
    return events, constants, max(gaps)


def subset_indices():
    return [16*group+offset for group in range(4) for offset in range(4)]


def apply_source_encoder(model, source_state):
    """Encoder-only copy: preserve all scratch target ODE parameters/buffers."""
    before = {key: value.detach().clone() for key, value in model.state_dict().items()
              if not key.startswith('encoder.')}
    model.encoder.load_state_dict(source_state, strict=True)
    model.encoder.requires_grad_(False)
    io.require(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()),
               'source assignment changed target dynamics')


def execute(path):
    release = validate(path)
    authority = io.identity(path)
    source_state, source_info = source_endpoint(release)
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = LOCAL/'writer.lock'
    # A killed worker retains its lock for PID inspection before recovery.
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    torch.set_num_threads(1)
    started, cpu = perf_counter(), process_time()
    completed = {}
    try:
        for job in release['jobs']:
            key, method = job['id'], job['method']
            folder, public = LOCAL/key, PUBLIC/key
            folder.mkdir(parents=True, exist_ok=True)
            public.mkdir(parents=True, exist_ok=True)
            binding = dict(release=authority, job=job, source=source_info if method == 'source_frozen' else None,
                           weights=[list(item) for item in WEIGHTS], internal_step=.01)
            io.write(LOCAL/'live.json', dict(job=key, pid=os.getpid(), utc=io.utc()))
            terminal = public/'summary.json'
            if terminal.exists():
                receipt = io.read(terminal)
                io.require(receipt['binding'] == binding, 'terminal pendulum job binding differs')
                if receipt['result']['checkpoint'] is not None:
                    io.verified(receipt['result']['checkpoint'])
            else:
                events, constants, gap = train_observations(job['system_seed'])
                selected = subset_indices() if job['train_count'] == 16 else list(range(64))
                ep = engine.Episodes.from_events([events[index] for index in selected])
                model = create_model(method, constants, job['optimization_seed'])
                if method == 'source_frozen':
                    apply_source_encoder(model, source_state)
                initial_encoder_sha = parameter_digest(model.encoder.state_dict())
                plan = engine.plan(len(ep), 4, 192, job['optimization_seed'])
                order_path = folder/'batch_plan.json'
                if order_path.exists():
                    io.require(io.read(order_path) == plan, 'saved pendulum batch plan differs')
                else:
                    io.write(order_path, plan)
                result = training.fit_stage(model, ep, folder/'stage', binding, plan, HORIZONS,
                                            microbatch=release['microbatch'], weights=WEIGHTS)
                frozen = method in ('source_frozen', 'random_frozen', 'closed_passive', 'closed_conservative')
                io.require(not frozen or parameter_digest(model.encoder.state_dict()) == initial_encoder_sha,
                           'frozen encoder changed')
                receipt = dict(run_id=RUN, job_id=key, job=job, binding=binding, result=result,
                               train_count=len(ep), train_indices=selected, train_event_ids=[events[i]['id'] for i in selected],
                               truth_mesh_gap=gap, batch_plan_ref=io.identity(order_path),
                               initial_encoder_sha256=initial_encoder_sha,
                               final_encoder_sha256=parameter_digest(model.encoder.state_dict()),
                               declared_damping='zero' if method == 'closed_conservative' else 'known_physical',
                               synthetic_val_access=False, ieee39_event_access=False,
                               test_trajectory_access=False, new_fom=0)
                io.write(terminal, receipt)
            completed[key] = dict(job=job, summary_ref=io.identity(terminal), status=receipt['result']['status'])
            print(key, completed[key]['status'], flush=True)
        summary = dict(run_id=RUN, status='completed', release=authority, jobs=completed,
                       source=source_info, synthetic_val_access=False, ieee39_event_access=False,
                       test_trajectory_access=False, new_fom=0, default_model='none',
                       wall_s=perf_counter()-started, cpu_s=process_time()-cpu, completed_utc=io.utc())
        io.write(PUBLIC/'fit_summary.json', summary)
        io.write(LOCAL/'live.json', dict(status='completed', pid=os.getpid(), utc=io.utc()))
    except BaseException as error:
        io.write(LOCAL/f'failed_{os.getpid()}.json', dict(error=repr(error), release=authority, utc=io.utc()))
        raise
    else:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    execute(parser.parse_args().release.resolve())
