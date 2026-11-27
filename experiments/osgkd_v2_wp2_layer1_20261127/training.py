"""Persist one fixed-budget synthetic fit stage without access to held-out data."""

import hashlib
import math
from pathlib import Path
import os
from statistics import mean
from time import perf_counter, process_time

import torch

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io

from . import engine


def data_digest(episodes):
    result = hashlib.sha256()
    for value in (episodes.theta_history, episodes.omega_history, episodes.targets):
        result.update(repr(tuple(value.shape)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def seal(path, payload):
    if path.exists():
        raise FileExistsError(f'immutable training checkpoint exists: {path}')
    temp = path.with_suffix('.pt.tmp')
    with temp.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    io.replace_with_retry(temp, path)
    return io.identity(path)


def restore(ref, model):
    saved = torch.load(io.verified(ref), map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state'], strict=True)
    return saved


def fit_stage(model, episodes, folder, binding, plan, steps, microbatch=4, masks=None, weights=None):
    """Use the same event list and batch plan across paired paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    masks = {} if masks is None else masks
    if isinstance(steps, int):
        steps = [steps] * len(plan)
    io.require(len(plan) == len(steps) and plan, 'fixed stage plan required')
    binding = {**binding, 'data_sha256': data_digest(episodes), 'attempts': len(plan),
               'microbatch': microbatch, 'horizons': sorted(set(steps))}
    terminal = folder/'terminal.json'
    if terminal.exists():
        result = io.read(terminal)
        io.require(result['binding'] == binding, 'saved stage binding differs')
        if result['status'] == 'completed':
            saved = restore(result['checkpoint'], model)
            io.require(saved['binding'] == binding and len(saved['rows']) == len(plan), 'saved endpoint differs')
        return result

    opt = engine.optimizer(model)
    fixed_batch = episodes.select(plan[0][:1])
    horizon = max(steps)
    model.train()
    rows = []
    start_wall, start_cpu = perf_counter(), process_time()
    resume = sorted(folder.glob('checkpoint_*.pt'))
    if resume:
        saved = restore(io.identity(resume[-1]), model)
        io.require(saved['binding'] == binding, 'resume stage binding differs')
        rows = saved['rows']
        opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        guard, initial = saved['guard'], saved['initial_loss']
    else:
        guard = engine.qualification(model, fixed_batch, horizon, weights=weights, masks=masks)
        if not guard['passed']:
            result = dict(binding=binding, status='qualification_failed', attempts=0, accepted=0, qualification=guard,
                          wall_s=perf_counter()-start_wall, cpu_s=process_time()-start_cpu, checkpoint=None)
            io.write(terminal, result)
            return result
        with torch.no_grad():
            initial = float(engine.loss(model, fixed_batch, horizon, .01, weights))
    reason = None
    failure = None
    for step in range(len(rows), len(plan)):
        engine.set_schedule(opt, step, len(plan))
        io.write(folder/'live.json', dict(attempt=step+1, pid=os.getpid(), utc=io.utc()))
        wall, cpu = perf_counter(), process_time()
        try:
            value, norm, components = engine.fit_step(model, opt, episodes.select(plan[step]), steps[step], .01,
                                                        microbatch, masks, weights)
        except FloatingPointError as error:
            reason = str(error)
            snapshot = seal(folder/f'failed_attempt_{os.getpid()}.pt', dict(state=model.state_dict(), optimizer=opt.state_dict(),
                          torch_rng=torch.get_rng_state(), binding=binding, attempt=step+1))
            failure = dict(attempt=step+1, events=plan[step], horizon=steps[step], stop_reason=reason,
                           wall_s=perf_counter()-wall, cpu_s=process_time()-cpu, snapshot=snapshot)
            io.write(folder/f'failed_attempt_{os.getpid()}.json', failure)
            break
        scientific = dict(attempt=step+1, events=plan[step], loss=value, gradient_norm=norm,
                          component_mse=components, horizon=steps[step], clipped=norm > 5.)
        record = folder/f'attempt_{step+1:04d}.json'
        if record.exists():
            old = io.read(record)
            io.require(old['scientific'] == scientific, 'replayed scientific row differs')
        else:
            old = dict(scientific=scientific, wall_s=perf_counter()-wall, cpu_s=process_time()-cpu)
            io.write(record, old)
        rows.append(old)
        if len(rows) % 64 == 0 or len(rows) == len(plan):
            checkpoint_path = folder/f'checkpoint_{len(rows):04d}.pt'
            seal(checkpoint_path, dict(state=model.state_dict(), optimizer=opt.state_dict(), rows=rows,
                                      binding=binding, guard=guard, initial_loss=initial, torch_rng=torch.get_rng_state()))
    measured = rows + ([failure] if failure else [])
    elapsed = dict(wall_s=perf_counter()-start_wall, cpu_s=process_time()-start_cpu,
                   attempted_update_wall_s=sum(row['wall_s'] for row in measured),
                   attempted_update_cpu_s=sum(row['cpu_s'] for row in measured))
    result = dict(binding=binding, status='scientific_stopped' if reason else 'completed',
                  attempts=len(rows)+int(reason is not None), accepted=len(rows), stop_reason=reason,
                  qualification=guard, checkpoint=None, failure=failure, clipped=sum(row['scientific']['clipped'] for row in rows),
                  initial_fixed_batch_loss=initial, **elapsed)
    if reason is None:
        with torch.no_grad():
            final = float(engine.loss(model, fixed_batch, horizon, .01, weights))
        if not math.isfinite(final):
            result.update(status='scientific_stopped', stop_reason='nonfinite fixed endpoint diagnostic')
        else:
            losses = [row['scientific']['loss'] for row in rows if row['scientific']['horizon'] == horizon]
            result.update(final_fixed_batch_loss=final, last32_horizon_mean_loss=mean(losses[-32:]),
                          first32_horizon_mean_loss=mean(losses[:32]), final_grad_norm=rows[-1]['scientific']['gradient_norm'],
                          checkpoint=io.identity(folder/f'checkpoint_{len(plan):04d}.pt'))
    io.write(terminal, result)
    return result
