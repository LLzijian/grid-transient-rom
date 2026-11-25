"""Durable continuation of the frozen Sprint2A fit; never edits original evidence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import torch
from experiments.osgkd_v2_sprint2a_20261125 import run, resume_host
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events
fit, require = run.fit, run.require
write = run.baseline.atomic_write
COST = {'wall_s', 'cpu_s', 'peak_memory_bytes'}
BOUNDARIES = {'graph_24111': 896, 'flat_24111': 896, 'graph_24121': 832, 'graph_24131': 0}


def scientific(value):
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items() if k not in COST}
    if isinstance(value, list):
        return [scientific(v) for v in value]
    return value


def restore_work(snapshot):
    work = IntegrationWork()
    work.rhs = {'forward': snapshot['rhs_forward'], 'recompute': snapshot['rhs_recompute']}
    work.encoder_events = snapshot['encoder_events']
    return work


def load(ref):
    return torch.load(run.verified(ref), map_location='cpu', weights_only=True)


def restore(saved, constants, kind, seed):
    model = fit.new_model(constants, kind, seed)
    require(saved['model']['version'] == model.checkpoint_version, 'model version differs')
    model.load_state_dict(saved['model']['state_dict'], strict=True)
    opt = fit.new_optimizer(model)
    opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng'])
    require(run.old.prior.p1a.tree_equal(model.checkpoint(), saved['model']), 'model restoration differs')
    require(run.old.prior.p1a.tree_equal(opt.state_dict(), saved['optimizer']), 'Adam restoration differs')
    require(torch.equal(torch.get_rng_state(), saved['torch_rng']), 'RNG restoration differs')
    return model, opt


def state(local, slice_id, next_attempt, status):
    path = local / 'state.json'
    temp = local / f'state_{slice_id}.tmp'
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(dict(slice=slice_id, next_attempt=next_attempt, attempt=next_attempt-1,
                       status=status, pid=os.getpid()), stream)
        stream.flush(); os.fsync(stream.fileno())
    retries = resume_host.replace_with_retry(temp, path)
    if retries:
        print(json.dumps(dict(status='state_write_retry', slice=slice_id, next_attempt=next_attempt,
                              retries=retries)), flush=True)


def emit_progress(job, slice_id, attempt, status):
    print(json.dumps(dict(job=job, slice=slice_id, attempt=attempt, next_attempt=attempt+1, status=status, pid=os.getpid())), flush=True)


def verify_seal(transaction, model, opt, rows, guards, work, guard_work, rejected):
    saved = load(transaction['checkpoint'])
    require(saved['attempt'] == len(rows) and saved['rejected'] == rejected, 'existing seal counters differ')
    require(run.old.prior.p1a.tree_equal(saved['model'], model.checkpoint()), 'existing seal model differs')
    require(run.old.prior.p1a.tree_equal(saved['optimizer'], opt.state_dict()), 'existing seal Adam differs')
    require(torch.equal(saved['torch_rng'], torch.get_rng_state()), 'existing seal RNG differs')
    require(saved['work'] == work.snapshot() and saved['guard_work'] == guard_work.snapshot(), 'existing seal work differs')
    require(run.read(transaction['attempts_ref']) == rows and run.read(transaction['guards_ref']) == guards, 'existing seal history differs')
    return transaction['checkpoint'], transaction['attempts_ref']


def execute(job, slice_id, resume_release):
    require(type(slice_id) is int and slice_id >= 1, 'positive slice required')
    resume_host.release_inputs(resume_release)
    torch.set_num_threads(1)
    original = ROOT / 'artifacts' / run.RUN / job
    local = ROOT / 'artifacts' / run.RUN / 'durable' / job
    public = ROOT / 'results' / run.RUN / job
    local.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    require(not (public/'summary.json').exists(), 'completed summary already exists')
    # The exclusive live claim is released only on a clean exit. A crash needs host inspection.
    lock = local/'writer.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    context = {}
    try:
        release_path = ROOT/'results'/run.RUN/f'{job}_release.json'
        release = json.loads(release_path.read_bytes())
        inputs = run.validate(release, job)
        interruption_path = ROOT/'results'/run.RUN/'interruption.json'
        interruption = json.loads(interruption_path.read_bytes())
        require(interruption['run_id'] == run.RUN and interruption['status'] == 'interrupted', 'interruption differs')
        interruption_ref = run.identity(interruption_path)
        boundary = BOUNDARIES[job]
        prior_transactions = [json.loads(p.read_bytes()) for p in local.glob('transaction_*.json')]
        planned_start = max((tx['attempt'] for tx in prior_transactions), default=boundary)
        state(local, slice_id, planned_start+1, 'restoring')
        raw, constants = run.load_train(inputs)
        sequence = fit.plan(raw)
        require(len(sequence['batches']) == 2048, 'fixed plan length differs')
        plan_path = local/'batch_plan.json'
        if boundary:
            require(json.loads((original/'batch_plan.json').read_bytes()) == sequence, 'original batch plan differs')
        if plan_path.exists():
            require(json.loads(plan_path.read_bytes()) == sequence, 'durable batch plan differs')
            plan_ref = run.identity(plan_path)
        else:
            plan_ref = write(plan_path, sequence)
        kind, seed = run.JOBS[job]
        transactions = list(local.glob('transaction_*.json'))
        guards, rows, diagnostics = [], [], []
        if transactions:
            transaction = max((json.loads(path.read_bytes()) for path in transactions), key=lambda tx: (tx['attempt'], tx['slice']))
            require(slice_id > transaction['slice'], 'slice must advance beyond completed transaction')
            require(transaction['release_ref'] == run.identity(release_path) and transaction['interruption_ref'] == interruption_ref, 'transaction authority differs')
            saved = load(transaction['checkpoint'])
            rows = run.read(transaction['attempts_ref'])
            guards = run.read(transaction['guards_ref'])
            diagnostics = transaction['diagnostics']
            start = saved['attempt']
            require(len(rows) == start and transaction['attempt'] == start, 'transaction history differs')
            require(saved['release'] == release and saved['plan_ref'] == plan_ref, 'durable checkpoint authority differs')
            require(all(row['attempt'] == i and row['event_ids'] == sequence['batches'][i-1] for i, row in enumerate(rows, 1)), 'durable row history differs')
        elif boundary:
            info = interruption['fit'][job]
            require(info['checkpoint_attempt'] == boundary and info['rejected'] == 0, 'original boundary differs')
            saved = load(info['checkpoint'])
            require(saved['attempt'] == boundary and saved['release'] == release, 'checkpoint authority differs')
            run.verified(saved['plan_ref'])
            require(run.read(saved['plan_ref']) == sequence, 'checkpoint plan differs')
            run.verified(info['last_attempt_ref']); run.verified(info['progress_ref'])
            rows = [json.loads((original/f'attempt_{i:04d}.json').read_bytes()) for i in range(1, boundary+1)]
            require(all(row['attempt'] == i and row['event_ids'] == sequence['batches'][i-1] and row['accepted'] for i, row in enumerate(rows, 1)), 'original history differs')
            start = boundary
            # Original checkpoint retained guard work but omitted its detailed result.
            guard_saved = load(run.identity(original/'attempt_512.pt'))
            gm, go = restore(guard_saved, constants, kind, seed)
            gw = IntegrationWork(); lookup = {e['id']: e for e in tensor_events(raw, 'cpu', max_steps=800)}
            bids = sequence['batches'][512]; t, c = perf_counter(), process_time()
            g = fit.guard(gm, [lookup[bids[0]], lookup[bids[-1]]], constants, gw, limits=(fit.horizon(512),))
            g['cost'] = dict(wall_s=perf_counter()-t, cpu_s=process_time()-c)
            guards = [dict(attempt=513, **g)]
            require(gw.snapshot() == saved['guard_work'], 'reconstructed guard work differs')
            diagnostics = [dict(kind='guard513_reconstruction', added_work=gw.snapshot(), cost=g['cost'])]
            del gm, go
        else:
            require(interruption['queued_without_start'] == job, 'fresh job was not queued')
            start, saved = 0, None
        if saved:
            model, opt = restore(saved, constants, kind, seed)
            work, guard_work = restore_work(saved['work']), restore_work(saved['guard_work'])
            rejected = saved['rejected']
            require(rejected == sum(not row['accepted'] for row in rows), 'rejection count differs')
        else:
            model = fit.new_model(constants, kind, seed); opt = fit.new_optimizer(model)
            work, guard_work, rejected = IntegrationWork(), IntegrationWork(), 0
        require(inputs['prerequisite']['result']['training_internal_step'] == .01, 'frozen .01 mesh required')
        write(local/f'slice_{slice_id}_started.json', dict(pid=os.getpid(), slice=slice_id, started_utc=datetime.now(timezone.utc).isoformat(), restored_step=start, next_attempt=start+1, release_ref=run.identity(release_path), interruption_ref=interruption_ref))
        state(local, slice_id, start+1, 'running')
        emit_progress(job, slice_id, start, 'started')
        lookup = {e['id']: e for e in tensor_events(raw, 'cpu', max_steps=800)}
        consecutive = 0
        for row in reversed(rows):
            if row['accepted']: break
            consecutive += 1
        context.update(model=model, opt=opt, rows=rows)

        def seal(step):
            stem = f'{step:04d}_slice_{slice_id}'
            existing = local/f'transaction_{stem}.json'
            if existing.exists():
                tx = json.loads(existing.read_bytes())
                require(tx['release_ref'] == run.identity(release_path) and tx['interruption_ref'] == interruption_ref, 'existing seal authority differs')
                return verify_seal(tx, model, opt, rows, guards, work, guard_work, rejected)
            cp = run.checkpoint(local/f'attempt_{stem}.pt', model, opt, step, release, plan_ref, work, guard_work, rejected)
            history = write(local/f'attempts_{stem}.json', rows)
            gr = write(local/f'guards_{stem}.json', guards)
            tx = dict(attempt=step, slice=slice_id, checkpoint=cp, attempts_ref=history, guards_ref=gr, diagnostics=diagnostics,
                      release_ref=run.identity(release_path), interruption_ref=interruption_ref)
            write(local/f'transaction_{stem}.json', tx)
            write(local/f'progress_{stem}.json', dict(attempt=step, accepted=step-rejected, rejected=rejected, checkpoint=cp, work=work.snapshot(), guard_work=guard_work.snapshot()))
            emit_progress(job, slice_id, step, 'checkpoint')
            return cp, history

        checkpoint_ref, attempts_ref = (transaction['checkpoint'], transaction['attempts_ref']) if start == 2048 else (None, None)
        for k in range(start, 2048):
            state(local, slice_id, k+1, 'running')
            if (local/f'pause_{slice_id}.json').exists():
                seal(k); state(local, slice_id, k+1, 'paused'); emit_progress(job, slice_id, k, 'paused'); lock.unlink(); return 75
            batch = [lookup[x] for x in sequence['batches'][k]]
            if k in (512, 896):
                t, c = perf_counter(), process_time()
                g = fit.guard(model, [batch[0], batch[-1]], constants, guard_work, limits=(fit.horizon(k),))
                g['cost'] = dict(wall_s=perf_counter()-t, cpu_s=process_time()-c)
                guards.append(dict(attempt=k+1, **g))
                require(g['passed'], 'stage guard failed; stopped')
            expected = 4*sum(min(fit.horizon(k), len(e['targets']['theta'])-1) for e in batch)
            require(work.rhs['forward']+expected <= fit.CAP, 'training cap would exceed')
            t, c, before = perf_counter(), process_time(), work.rhs['forward']
            row = fit.attempt(model, opt, batch, k, constants, work, .01)
            require(work.rhs['forward']-before == expected, 'RHS accounting differs')
            row.update(wall_s=perf_counter()-t, cpu_s=process_time()-c, peak_memory_bytes=run.peak_memory_bytes())
            replay = original/f'attempt_{k+1:04d}.json'
            if replay.exists():
                previous = json.loads(replay.read_bytes())
                matches = scientific(previous) == scientific(row)
                write(local/f'replay_{k+1:04d}_slice_{slice_id}.json', dict(attempt=k+1, matched=matches, original_ref=run.identity(replay), expected=scientific(previous), actual=scientific(row)))
                require(matches, 'original scientific replay mismatch')
            for tail in local.glob(f'attempt_{k+1:04d}_slice_*.json'):
                require(scientific(json.loads(tail.read_bytes())) == scientific(row), 'durable tail replay mismatch: '+tail.name)
            # Each slice owns its files: aborted tails are retained without replacement.
            write(local/f'attempt_{k+1:04d}_slice_{slice_id}.json', row)
            rows.append(row)
            rejected += not row['accepted']; consecutive = 0 if row['accepted'] else consecutive+1
            require(rejected == 0, 'zero-rejection continuation protection')
            require(consecutive < 3 and rejected < 16, 'original rejection protection')
            paused = k+1 < 2048 and (local/f'pause_{slice_id}.json').exists()
            if (k+1)%64 == 0 or paused:
                checkpoint_ref, attempts_ref = seal(k+1)
            state(local, slice_id, k+2, 'paused' if paused else 'running')
            if paused:
                emit_progress(job, slice_id, k+1, 'paused'); lock.unlink(); return 75
        replay_records = {}
        for replay_path in local.glob('replay_*_slice_*.json'):
            replay_row = json.loads(replay_path.read_bytes())
            require(replay_row['matched'], 'unmatched replay receipt exists')
            replay_records[replay_row['attempt']] = run.identity(replay_path)
        expected_replay = {'graph_24111': 8, 'flat_24111': 10, 'graph_24121': 35, 'graph_24131': 0}[job]
        require(len(replay_records) == expected_replay, 'required original replay coverage differs')
        replay_aggregate = dict(expected=expected_replay, matched=len(replay_records), records_ref=write(local/f'replay_index_slice_{slice_id}.json', replay_records))
        result = dict(fixed_endpoint=2048, actual_endpoint=2048, attempts=2048, accepted=2048-rejected, rejected=rejected,
                      valid_endpoint=True, stopped_early=False, checkpoints={'2048': checkpoint_ref}, plan_ref=plan_ref,
                      attempts_ref=attempts_ref, work=work.snapshot(), guard_work=guard_work.snapshot(), guards=guards,
                      train_cost=dict(wall_s=sum(r['wall_s'] for r in rows), cpu_s=sum(r['cpu_s'] for r in rows)),
                      encoder_parameters=fit.encoder_parameters(model), state_dimension=50, job=job, seed=seed,
                      training_internal_step=.01, spectral_jacobian_calls=sum(r['horizon_s']==8 for r in rows),
                      guard_spectral_jacobian_calls=sum(g['spectral_jacobian_calls'] for g in guards),
                      cost=dict(peak_memory_bytes=run.peak_memory_bytes(), training_wall_s=sum(r['wall_s'] for r in rows), training_cpu_s=sum(r['cpu_s'] for r in rows), guard_wall_s=sum(g['cost']['wall_s'] for g in guards), guard_cpu_s=sum(g['cost']['cpu_s'] for g in guards)),
                      health=dict(model_finite=run.old.prior.finite_tree(model.state_dict()), adam_finite=run.old.prior.finite_tree(opt.state_dict()), clipped=sum(r.get('clipped',False) for r in rows)), continuation_diagnostics=diagnostics, replay=replay_aggregate)
        receipt = dict(run_id=run.RUN, job_id=job, status='completed', release=release, release_ref=run.identity(release_path), result=result, completed_utc=datetime.now(timezone.utc).isoformat())
        ref = write(public/'summary.json', receipt, public=True)
        write(local/'completed.json', dict(status='completed', summary_ref=ref))
        state(local, slice_id, 2049, 'completed'); emit_progress(job, slice_id, 2048, 'completed'); lock.unlink(); return 0
    except BaseException as error:
        write(local/f'slice_{slice_id}_failed.json', dict(status='failed', error=repr(error), pid=os.getpid(), recorded_attempts=len(context.get('rows', [])), utc=datetime.now(timezone.utc).isoformat()))
        emit_progress(job, slice_id, len(context.get('rows', [])), 'failed')
        # Preserve the claim and all bytes for host diagnosis after any abnormal exit.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=run.JOBS, required=True)
    parser.add_argument('--slice', type=int, required=True)
    parser.add_argument('--resume-release', type=Path, required=True)
    args = parser.parse_args()
    sys.exit(execute(args.job, args.slice, args.resume_release))
