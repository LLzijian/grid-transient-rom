"""Score the pendulum matrix at its fixed endpoints on synthetic held-out data."""

import argparse
from functools import lru_cache
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import numpy as np
import torch

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
from experiments.osgkd_v2_wp2_layer1_20261127 import engine, training
from experiments.osgkd_v2_wp3_pendulum_20261127 import pendulum, run


@lru_cache(maxsize=1)
def val_bank(seed):
    system = pendulum.make_system(n=8, seed=seed)
    output = {'0.00': [], '0.01': []}
    gaps = []
    for index in range(16):
        key, kind = 2000+index, pendulum.EXCITATIONS[index//4]
        event, coarse = pendulum.generate_episode(system, key, kind, forecast_s=4.)
        _, fine = pendulum.generate_episode(system, key, kind, forecast_s=4., internal_dt=.00125, noise_levels=(0.,))
        gaps.append(float(np.max(np.abs(coarse['state']-fine['state']))))
        for noise in output:
            output[noise].append(event[noise])
    io.require(max(gaps) <= 1e-6, 'pendulum val truth mesh differs')
    constants = pendulum.model_constants(pendulum.motion_packet(output['0.00'][0]))
    return output, constants, max(gaps)


def verify_terminal(item, job, fit_info, authority):
    key = job['id']
    ref = item['summary_ref']
    io.require(ref['path'] == f'results/{run.RUN}/{key}/summary.json', 'terminal pendulum path differs')
    receipt = io.read(io.verified(ref))
    binding = dict(release=authority, job=job, source=fit_info['source'] if job['method']=='source_frozen' else None,
                   weights=[list(w) for w in run.WEIGHTS], internal_step=.01)
    io.require(receipt['run_id'] == run.RUN and receipt['job_id'] == key and receipt['job'] == item['job'] == job
               and receipt['binding'] == binding and receipt['synthetic_val_access'] is False,
               'terminal pendulum identity differs')
    result = receipt['result']
    io.require(all(result['binding'].get(k) == v for k,v in binding.items()), 'training checkpoint bindings differ')
    io.require(result['status'] == item['status'] and result['status'] in
               {'completed','scientific_stopped','qualification_failed'}, 'invalid fit stage status')
    if result['status'] == 'completed':
        io.require(result['attempts'] == result['accepted'] == 192 and result['checkpoint']['path'] ==
                   f'artifacts/{run.RUN}/fit/{key}/stage/checkpoint_0192.pt', 'fixed 192 endpoint differs')
        io.verified(result['checkpoint'])
    else:
        io.require(result['checkpoint'] is None, 'stopped job must have no scored endpoint')
    return receipt


def power_check(model, motion):
    # Coordinate gradient is an independent check of the reported power terms.
    x = model.initialize(motion.theta_history[:4], motion.omega_history[:4]).detach().requires_grad_(True)
    try:
        power = model.storage_power(x)
    except NotImplementedError:
        return None
    chain = (torch.autograd.grad(power.H.sum(), x)[0] * model.rhs(None, x)).sum(-1).detach()
    return dict(chain_residual_abs_max=float((chain-power.total_power.detach()).abs().max()),
                reported_power_min=float(power.total_power.detach().min()),
                reported_power_max=float(power.total_power.detach().max()))


def adaptation_check(table):
    scores = []
    for seed in run.SYSTEM_SEEDS:
        values = []
        for method in ('source_frozen', 'osgkd_few', 'random_frozen'):
            metric = table[run.job_key(seed,method)].get('nominal', {}).get('windows', {}).get('400')
            if metric is None:
                return dict(passed=None, reason='incomplete_fixed_endpoint_coverage')
            values.append(metric['risk_mean_tail'])
        scores.append(values)
    a = np.asarray(scores)
    means = a.mean(axis=0)
    passes = int(((a[:,0] <= .9*a[:,1]) & (a[:,0] <= .9*a[:,2])).sum())
    return dict(passed=bool(passes >= 2 and means[0]<=.9*means[1] and means[0]<=.9*means[2]),
                paired_systems_passing=passes, systems=len(a), ordering=['source_frozen','osgkd_few','random_frozen'],
                per_system_risk=a.tolist(), mean_risk=means.tolist())


def evaluate(path):
    value = io.read(path)
    io.require(path==run.PUBLIC/'evaluate_release.json' and value['run_id']==run.RUN and value['released'] is True,
               'canonical pendulum score release required')
    release = run.validate(io.verified(value['fit_release']))
    fit = io.read(io.verified(value['fit_summary']))
    io.require(fit['run_id']==run.RUN and fit['status']=='completed' and fit['release']==value['fit_release']
               and fit['synthetic_val_access'] is False and fit['test_trajectory_access'] is False,
               'closed train-only pendulum ledger required')
    io.require(set(fit['jobs'])=={job['id'] for job in release['jobs']}, 'complete matrix required')
    verified = {job['id']:verify_terminal(fit['jobs'][job['id']],job,fit,value['fit_release']) for job in release['jobs']}
    out = run.PUBLIC/'evaluate01'
    private = ROOT/'artifacts'/run.RUN/'evaluate01'
    io.require(not out.exists() and not private.exists(), 'one pendulum score per run')
    out.mkdir(parents=True)
    private.mkdir(parents=True)
    torch.set_num_threads(1)
    start, cpu = perf_counter(), process_time()
    table, refs = {}, {}
    for job in release['jobs']:
        key, method = job['id'], job['method']
        receipt, result = verified[key], verified[key]['result']
        entry = dict(job=job, fit=result, fit_summary=fit['jobs'][key]['summary_ref'])
        if result['status'] == 'completed':
            bank, constants, gap = val_bank(job['system_seed'])
            model = run.create_model(method, constants, job['optimization_seed'])
            saved = training.restore(result['checkpoint'],model)
            io.require(saved['binding'] == result['binding'] and len(saved['rows']) == 192
                       and run.parameter_digest(model.encoder.state_dict()) == receipt['final_encoder_sha256'],
                       'loaded pendulum weights differ')
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.eval().requires_grad_(False)
            entry.update(truth_mesh_gap=gap, parameters=sum(p.numel() for p in model.parameters()),
                         trainable_parameters=trainable,
                         encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
                         evolved_state_dimension=getattr(model,'effective_state_dimension',5*model.n))
            for noise, label in (('0.00','nominal'),('0.01','noisy')):
                observations = engine.Episodes.from_events(bank[noise])
                with torch.inference_mode():
                    engine.trajectory(model, observations.select(slice(0,1)), 2, .005)
                readout = engine.score_events(model, observations, (100,200,400), release['microbatch'],
                                               event_ids=[event['id'] for event in bank[noise]])
                detail = private/(key+'_'+label+'_rows.json')
                io.write(detail, readout.pop('rows'))
                entry[label] = {**readout, 'rows_ref':io.identity(detail), 'val_data_sha256':training.data_digest(observations)}
                if label=='nominal':
                    entry['power_check'] = power_check(model,observations)
        single=out/(key+'.json')
        io.write(single, entry)
        refs[key]=io.identity(single)
        table[key]=entry
        print(key, result['status'], flush=True)
    summary = dict(run_id=run.RUN, status='completed', release=io.identity(path), terminal_jobs=refs,
                   adaptation_gate=adaptation_check(table), ieee39_event_access=False, test_trajectory_access=False,
                   new_fom=0, default_model='none', wall_s=perf_counter()-start, cpu_s=process_time()-cpu, completed_utc=io.utc())
    io.write(out/'summary.json',summary)
    print(io.identity(out/'summary.json'),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    evaluate(parser.parse_args().release.resolve())
