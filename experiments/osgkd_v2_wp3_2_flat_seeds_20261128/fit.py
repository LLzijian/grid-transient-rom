"""Fresh matched flat seeds using the unchanged Sprint2A scientific recipe."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]

import torch

from experiments.osgkd_v2_sprint2a_20261125 import train, run, resume_host as io, resume_fit
from experiments.osgkd_v2_sprint2b_baselines_20261126 import fit as durable
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events


RUN = 'osgkd_v2_wp3_2_flat_seeds_20261128'
JOBS = {'flat_24121': ('flat', 24121), 'flat_24131': ('flat', 24131)}
ORDER = tuple(JOBS)
PUBLIC = ROOT/'results'/RUN
LOCAL = ROOT/'artifacts'/RUN/'durable'
PROTOCOL = 'campaigns/osgkd_v2/wp3_2_flat_seeds_protocol.md'
PLAN_SHA = 'b11463b14f01ed1c26f60b6227f7ce0072a90d5f465e61593fbc056878e8ae55'
CORE_FIELDS = ('run_id', 'released', 'jobs', 'source_hashes', 'input_refs', 'device', 'threads', 'recompute',
               'training_internal_step', 'initialization', 'batch_plan_sha256', 'new_fom',
               'test_trajectory_access', 'default_model')
REQUIRED_SOURCES = run.SOURCES | {PROTOCOL,
    'experiments/osgkd_v2_sprint2a_20261125/resume_host.py',
    'experiments/osgkd_v2_sprint2a_20261125/resume_fit.py',
    'experiments/osgkd_v2_sprint2b_baselines_20261126/fit.py',
    *('experiments/'+RUN+'/'+name for name in ('fit.py', 'host.py', 'launch_durable.ps1', 'score.py', 'postprocess.py'))}
require, identity, verified = io.require, io.identity, io.verified
write, guarded, payload, save_tensor = durable.write, durable.guarded, durable.payload, durable.save_tensor
equal, finite = run.old.prior.p1a.tree_equal, run.old.prior.finite_tree


def validate(path, job=None, qualification=False):
    path = path.resolve()
    release = io.read(path)
    require(path == PUBLIC/('qualify_release.json' if qualification else 'release.json'), 'canonical release path required')
    require(release['run_id'] == RUN and release['released'] is True and release['jobs'] == list(ORDER), 'released flat seed order differs')
    require(Path(sys.executable).resolve() == (ROOT.parent/'.venv/Scripts/python.exe').resolve()
            and sys.version_info[:2] == (3, 12), 'shared Python 3.12 required')
    require((release['device'], release['threads'], release['recompute'], release['training_internal_step'])
            == ('cpu', 1, False, .01), 'original execution contract required')
    require(release['initialization'] == 'fresh_empty_adam' and release['batch_plan_sha256'] == PLAN_SHA,
            'fresh initialization or batch course differs')
    require(release['new_fom'] == 0 and release['test_trajectory_access'] is False
            and release['default_model'] == 'none', 'scope differs')
    require(release['scientist']['configured_model'] == 'gpt-6-astra'
            and release['independent_preflight']['status'] == 'passed', 'scientist/preflight required')
    require(REQUIRED_SOURCES <= set(release['source_hashes']), 'source coverage missing')
    for name, digest in release['source_hashes'].items():
        source = (ROOT/name).resolve()
        require(source.is_relative_to(ROOT) and identity(source)['sha256'] == digest, 'source differs: '+name)
    require(set(release['input_refs']) == set(run.INPUTS), 'original input bindings required')
    inputs = {}
    for key, canonical in run.INPUTS.items():
        ref = release['input_refs'][key]
        require(ref['path'] == canonical, 'input path differs: '+key)
        inputs[key] = io.read(verified(ref))
    run.old.prior.validate_baseline(inputs['baseline_summary'], inputs['baseline_archive'], release['input_refs']['baseline_summary'])
    require(release['input_refs']['anchor_analysis']['sha256'] == run.previous.ANCHOR_SHA
            and inputs['anchor_analysis']['status'] == 'completed'
            and inputs['anchor_analysis']['encoder_parameters'] == 6174, 'fixed anchor differs')
    require(release['input_refs']['probe_summary']['sha256'] == run.PROBE_SHA
            and inputs['probe_summary']['status'] == 'completed', 'fixed spectral probe differs')
    if job is not None:
        require(job in JOBS, 'unknown flat seed')
    if qualification:
        require('qualification_refs' not in release, 'qualification release cannot consume itself')
    else:
        require(set(release['qualification_refs']) == set(JOBS), 'both independent qualifications required')
        for name, ref in release['qualification_refs'].items():
            require(ref['path'] == f'results/{RUN}/{name}/qualification.json', 'qualification path differs')
            q = io.read(verified(ref))
            require(q['run_id'] == RUN and q['job_id'] == name and q['status'] == 'completed'
                    and q['result']['passed'] and q['result']['official_updates'] == 0
                    and q['result']['fresh_empty_adam'] and q['result']['parameters_unchanged'], 'qualification failed')
            require(all(q['release'][key] == release[key] for key in CORE_FIELDS),
                    'qualification and fit input/source/execution bindings differ')
            require(q['release_ref'] == identity(PUBLIC/'qualify_release.json'), 'qualification authority differs')
    return release, inputs


def new_model(constants, job):
    return train.new_model(constants, *JOBS[job])


def stopped(rows, guards):
    if any(not guard['passed'] for guard in guards):
        return 'stage_guard_failed'
    rejected = sum(not row['accepted'] for row in rows)
    if rejected >= 16 or len(rows) >= 3 and all(not row['accepted'] for row in rows[-3:]):
        return 'nonfinite_rejection_limit'
    return None


def assert_no_scientific_stop(path, job):
    """Deleting a stale writer lock must not reopen a stopped scientific path."""
    authority = identity(path)
    for previous in ORDER[:ORDER.index(job)+1]:
        sentinel = LOCAL/previous/'scientific_stop.json'
        if sentinel.exists():
            stop = io.read(sentinel)
            require(stop['release_ref'] != authority,
                    f"scientific stop forbids {job} under this release: {previous}: {stop['reason']}")


def execute(path, job, slice_id=1, qualification=False):
    require(type(slice_id) is int and slice_id >= 1, 'positive slice required')
    release, inputs = validate(path, job, qualification)
    if not qualification:
        assert_no_scientific_stop(path, job)
    torch.set_num_threads(1)
    local, public = LOCAL/job, PUBLIC/job
    local.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    suffix = 'qualification' if qualification else 'summary'
    require(not (public/(suffix+'.json')).exists(), 'terminal receipt already exists')
    lock = local/'writer.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    context = {}
    write(local/f'slice_{slice_id}_started.json', dict(job_id=job, slice=slice_id, pid=os.getpid(),
          qualification=qualification, release_ref=identity(path), utc=io.utc()))
    try:
        raw, constants = run.load_train(inputs)
        sequence = train.plan(raw)
        canonical = (json.dumps(sequence, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
        require(hashlib.sha256(canonical).hexdigest() == PLAN_SHA, 'original train batch course differs')
        plan_path = local/'batch_plan.json'
        if plan_path.exists():
            require(io.read(plan_path) == sequence, 'saved plan differs')
        else:
            run.baseline.atomic_write(plan_path, sequence)
        plan_ref = identity(plan_path)
        require(plan_ref['sha256'] == PLAN_SHA, 'batch plan bytes differ')
        lookup = {event['id']: event for event in tensor_events(raw, 'cpu', max_steps=800)}
        model = new_model(constants, job)
        opt = train.new_optimizer(model)
        require(not opt.state, 'fresh Adam must be empty')
        rows, guards = [], []
        work, guard_work = IntegrationWork(), IntegrationWork()
        context.update(model=model, opt=opt, rows=rows, work=work, guard_work=guard_work)
        metadata = dict(seed=JOBS[job][1], kind='flat', encoder_parameters=train.encoder_parameters(model),
                        parameters=sum(p.numel() for p in model.parameters()), embedded_state_dimension=5*model.n)
        if qualification:
            before = copy.deepcopy(model.checkpoint())
            adam_before = copy.deepcopy(opt.state_dict())
            rng = torch.get_rng_state().clone()
            first = [lookup[key] for key in sequence['batches'][0]]
            result = guarded(model, first, constants, guard_work, 800)
            require(equal(before, model.checkpoint()) and equal(adam_before, opt.state_dict()) and not opt.state,
                    'zero-update qualification changed parameters or Adam')
            require(torch.equal(rng, torch.get_rng_state()), 'qualification changed RNG')
            result.update(metadata, official_updates=0, fresh_empty_adam=True, parameters_unchanged=True,
                          rng_unchanged=True, qualification_event_ids=[event['id'] for event in first],
                          process_peak_rss_bytes=run.peak_memory_bytes(), training_internal_step=.01, plan_ref=plan_ref)
        else:
            transactions = [io.read(p) for p in local.glob('transaction_*.json')]
            if transactions:
                tx = max(transactions, key=lambda value: (value['attempt'], value['slice'], value['guard_count']))
                require(slice_id > tx['slice'], 'resume slice must advance')
                saved = torch.load(verified(tx['checkpoint']), map_location='cpu', weights_only=True)
                require(saved['release'] == release and saved['plan_ref'] == plan_ref and saved['job_id'] == job,
                        'resume authority differs')
                rows, guards = io.read(verified(tx['attempts_ref'])), io.read(verified(tx['guards_ref']))
                require(saved['attempt'] == len(rows) and saved['guards'] == guards
                        and saved['accepted'] == sum(r['accepted'] for r in rows)
                        and saved['rejected'] == sum(not r['accepted'] for r in rows), 'sealed history differs')
                require(all(row['attempt'] == i+1 and row['event_ids'] == sequence['batches'][i]
                            for i, row in enumerate(rows)), 'sealed batches differ')
                model, opt = resume_fit.restore(saved, constants, *JOBS[job])
                work, guard_work = resume_fit.restore_work(saved['work']), resume_fit.restore_work(saved['guard_work'])
                write(local/f'restore_slice_{slice_id}.json', dict(checkpoint_ref=tx['checkpoint'],
                      model_equal=True, adam_equal=True, rng_equal=True, attempt=len(rows)))
                context.update(model=model, opt=opt, rows=rows, work=work, guard_work=guard_work)
            latest = None
            def seal():
                stem = f'{len(rows):04d}_guards_{len(guards)}_slice_{slice_id}'
                saved = payload(model, opt, rows, guards, work, guard_work, release, plan_ref)
                saved['job_id'] = job
                cp_path = local/f'checkpoint_{stem}.pt'
                cp = identity(cp_path) if cp_path.exists() else save_tensor(cp_path, saved)
                require(equal(torch.load(verified(cp), map_location='cpu', weights_only=True), saved), 'checkpoint roundtrip differs')
                tx = dict(attempt=len(rows), slice=slice_id, guard_count=len(guards), checkpoint=cp,
                          attempts_ref=write(local/f'attempts_{stem}.json', rows),
                          guards_ref=write(local/f'guards_{stem}.json', guards), release_ref=identity(path), job_id=job)
                existing = local/f'transaction_{stem}.json'
                if existing.exists():
                    require(io.read(existing) == tx, 'existing transaction differs')
                else:
                    write(existing, tx)
                print(json.dumps(dict(event='checkpoint', job_id=job, attempt=len(rows), checkpoint=cp)), flush=True)
                return tx
            reason = stopped(rows, guards)
            for k in range(len(rows), 2048):
                if reason:
                    break
                write(local/'state.json', dict(slice=slice_id, pid=os.getpid(), next_attempt=k+1, status='running'))
                if (local/f'pause_{slice_id}.json').exists():
                    seal()
                    lock.unlink()
                    return 75
                batch = [lookup[key] for key in sequence['batches'][k]]
                if k in (512, 896):
                    guard = dict(attempt=k+1, **guarded(model, [batch[0], batch[-1]], constants, guard_work, train.horizon(k)))
                    for tail in local.glob(f'guard_{k+1:04d}_slice_*.json'):
                        require(resume_fit.scientific(io.read(tail)) == resume_fit.scientific(guard), 'unsealed guard replay differs')
                    write(local/f'guard_{k+1:04d}_slice_{slice_id}.json', guard)
                    guards.append(guard)
                    if not guard['passed']:
                        reason = 'stage_guard_failed'
                        latest = seal()
                        break
                expected = 4*sum(min(train.horizon(k), len(event['targets']['theta'])-1) for event in batch)
                require(work.rhs['forward']+expected <= train.CAP, 'original training cap exceeded')
                before = work.rhs['forward']
                started, cpu = perf_counter(), process_time()
                context['attempt'] = k+1
                row = train.attempt(model, opt, batch, k, constants, work, .01)
                require(work.rhs['forward']-before == expected, 'original training RHS differs')
                row.update(wall_s=perf_counter()-started, cpu_s=process_time()-cpu, peak_memory_bytes=run.peak_memory_bytes())
                for tail in local.glob(f'attempt_{k+1:04d}_slice_*.json'):
                    require(resume_fit.scientific(io.read(tail)) == resume_fit.scientific(row), 'unsealed attempt replay differs')
                    write(local/f'replay_{k+1:04d}_slice_{slice_id}.json', dict(previous_ref=identity(tail), matched=True))
                write(local/f'attempt_{k+1:04d}_slice_{slice_id}.json', row)
                rows.append(row)
                reason = stopped(rows, guards)
                if (k+1) % 64 == 0 or reason:
                    latest = seal()
            if latest is None or latest['attempt'] != len(rows):
                latest = seal()
            accepted = sum(row['accepted'] for row in rows)
            valid = len(rows) == 2048 and accepted >= 2033 and reason is None
            require(finite(model.state_dict()) and finite(opt.state_dict()), 'nonfinite terminal model or Adam')
            result = dict(**metadata, fixed_endpoint=2048, actual_endpoint=len(rows), attempts=len(rows),
                          accepted=accepted, rejected=len(rows)-accepted, valid_endpoint=valid,
                          stop_reason=reason, checkpoints={'2048': latest['checkpoint']} if valid else {},
                          terminal_checkpoint=latest['checkpoint'], attempts_ref=latest['attempts_ref'],
                          guards_ref=latest['guards_ref'], guards=guards, plan_ref=plan_ref,
                          work=work.snapshot(), guard_work=guard_work.snapshot(), training_internal_step=.01,
                          spectral_jacobian_calls=sum(row['horizon_s'] == 8 for row in rows),
                          guard_spectral_jacobian_calls=sum(g['spectral_jacobian_calls'] for g in guards),
                          process_peak_rss_bytes=run.peak_memory_bytes(),
                          train_cost=dict(wall_s=sum(r['wall_s'] for r in rows), cpu_s=sum(r['cpu_s'] for r in rows)),
                          replay_refs=[identity(p) for p in sorted(local.glob('replay_*.json'))],
                          health=dict(model_finite=True, adam_finite=True, terminal_checkpoint_roundtrip_equal=True))
        receipt = dict(run_id=RUN, job_id=job, status='completed' if qualification or result['valid_endpoint'] else 'scientific_stopped',
                       release=release, release_ref=identity(path), source_hashes=release['source_hashes'],
                       input_refs=release['input_refs'], training_internal_step=.01, result=result)
        summary_ref = write(public/(suffix+'.json'), receipt)
        write(local/f'slice_{slice_id}_completed.json', dict(summary_ref=summary_ref, status=receipt['status'], utc=io.utc()))
        write(local/'state.json', dict(slice=slice_id, pid=os.getpid(), next_attempt=len(rows)+1, status=receipt['status']))
        lock.unlink()
        return 0
    except BaseException as error:
        scientific_reason = None
        if str(error).startswith('nonfinite parameter or Adam'):
            scientific_reason = 'nonfinite_parameter_or_adam'
        elif str(error).startswith('nonfinite projection'):
            scientific_reason = 'nonfinite_projection'
        elif str(error).startswith('nonfinite terminal model or Adam'):
            scientific_reason = 'nonfinite_terminal_model_or_adam'
        if scientific_reason:
            sentinel = local/'scientific_stop.json'
            if not sentinel.exists():
                write(sentinel, dict(run_id=RUN, job_id=job, status='scientific_stop', reason=scientific_reason,
                      original_error=str(error), release_ref=identity(path), slice=slice_id,
                      attempt=context.get('attempt'), recorded_attempts=len(context.get('rows', [])), utc=io.utc()))
        snapshot = None
        if context.get('model') is not None:
            snapshot = save_tensor(local/f'failure_slice_{slice_id}.pt', dict(model=context['model'].checkpoint(),
                     optimizer=context['opt'].state_dict(), torch_rng=torch.get_rng_state(), attempt=context.get('attempt')))
        write(local/f'failure_slice_{slice_id}.json', dict(status='scientific_stop' if scientific_reason else 'execution_failed',
              scientific_terminal=scientific_reason is not None, error_category=scientific_reason or 'runtime_error',
              error=repr(error), snapshot_ref=snapshot, attempt=context.get('attempt'),
              recorded_attempts=len(context.get('rows', [])), utc=io.utc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--job', required=True, choices=JOBS)
    parser.add_argument('--slice', type=int, default=1)
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    sys.exit(execute(args.release.resolve(), args.job, args.slice, args.qualify))
