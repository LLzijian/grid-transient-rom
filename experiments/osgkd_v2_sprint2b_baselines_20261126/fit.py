"""Release-bound, train-only baseline qualification and durable fixed-mesh fitting."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import torch
from experiments.osgkd_v2_sprint2a_20261125 import train, run, resume_host as io, resume_fit
from experiments.osgkd_v2_discovery import prepare
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events
RUN = 'osgkd_v2_sprint2b_baselines_20261126'
JOBS = {'closed_passive_24111': 'closed_passive_kuramoto', 'neural_ode_24111': 'graph_neural_ode'}
ORDER = tuple(JOBS)
require, identity, verified = io.require, io.identity, io.verified


def write(path, value):
    io.write(path, value)
    return identity(path)


def validate(path, job, qualification=False):
    release = io.read(path)
    require(release['run_id'] == RUN and release['released'] is True, 'unreleased baseline run')
    require(release['training_internal_step'] == .01, 'frozen .01 mesh required')
    require(release.get('device') == 'cpu' and release.get('threads') == 1 and release.get('recompute') is False, 'execution contract differs')
    required = run.SOURCES | {f'experiments/{RUN}/{name}' for name in ('fit.py', 'host.py', 'launch_durable.ps1')} | {'experiments/osgkd_v2_discovery/prepare.py', 'src/kiki/models/osgkd_baselines.py', 'campaigns/osgkd_v2/sprint2b_discovery_baselines_protocol.md'}
    require(required <= set(release['source_hashes']), 'baseline source binding missing')
    for name, digest in release['source_hashes'].items():
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT) and identity(target)['sha256'] == digest, 'source digest differs: '+name)
    require(set(release['input_refs']) == set(run.INPUTS), 'matched input bindings required')
    inputs = {}
    for key, canonical in run.INPUTS.items():
        ref = release['input_refs'][key]
        require(ref['path'] == canonical, 'input path differs: '+key)
        inputs[key] = io.read(verified(ref))
    require(release['batch_plan_sha256'] == 'b11463b14f01ed1c26f60b6227f7ce0072a90d5f465e61593fbc056878e8ae55', 'registered batch course differs')
    require(job in JOBS, 'unknown baseline')
    if not qualification:
        qref = release['qualification_refs'][job]
        q = io.read(verified(qref))
        require(q['run_id'] == RUN and q['job_id'] == job and q['status'] == 'completed' and q['result']['passed'] and q['result']['official_updates'] == 0, 'successful train-only qualification required')
        for key in ('source_hashes', 'input_refs', 'training_internal_step'):
            require(q[key] == release[key], 'qualification binding differs: '+key)
    return release, inputs


def new_model(constants, job):
    model = prepare.new_baseline(JOBS[job], constants, 24111).cpu().double()
    rng = torch.get_rng_state().clone()
    template = train.new_model(constants, 'graph', 24111)
    require(run.old.prior.p1a.tree_equal(model.encoder.state_dict(), template.encoder.state_dict()), 'shared initial encoder differs')
    torch.set_rng_state(rng)
    return model


def counts(model, job):
    encoder = sum(p.numel() for p in model.encoder.parameters())
    dynamics = sum(p.numel() for p in model.parameters()) - encoder
    require(encoder == 1928 and dynamics == (45 if job == ORDER[0] else 671), 'baseline parameter count differs')
    return dict(encoder_parameters=encoder, effective_encoder_parameters=0 if job == ORDER[0] else encoder,
                ode_parameters=dynamics, active_parameters=dynamics+(0 if job == ORDER[0] else encoder),
                embedded_state_dimension=5*model.n, effective_state_dimension=2*model.n if job == ORDER[0] else 5*model.n)


def new_optimizer(model):
    # Sprint 2A puts all non-encoder parameters in one group. The two
    # baselines do not expose its OSGKD-specific c/beta parameter pair.
    encoder = list(model.encoder.parameters()) + list(model.initial_head.parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    dynamics = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    return torch.optim.Adam(
        [{"params": encoder, "lr": .001}, {"params": dynamics, "lr": .0003}],
        betas=(.9, .999), eps=1e-8, weight_decay=0,
    )


def guarded(model, batch, constants, work, limit):
    started, cpu = perf_counter(), process_time()
    try:
        result = train.guard(model, batch, constants, work, limits=(limit,))
    except (RuntimeError, ValueError) as error:
        if str(error) != 'nonfinite fine qualification: blocked':
            raise
        result = dict(passed=False, error=repr(error), work=work.snapshot(), spectral_jacobian_calls=2*int(limit == 800))
    return dict(**result, cost=dict(wall_s=perf_counter()-started, cpu_s=process_time()-cpu))


def payload(model, opt, rows, guards, work, guard_work, release, plan_ref):
    return dict(model=copy.deepcopy(model.checkpoint()), optimizer=copy.deepcopy(opt.state_dict()), torch_rng=torch.get_rng_state().clone(),
                attempt=len(rows), accepted=sum(r['accepted'] for r in rows), rejected=sum(not r['accepted'] for r in rows),
                work=work.snapshot(), guard_work=guard_work.snapshot(), guards=copy.deepcopy(guards), release=release, plan_ref=plan_ref)


def restore(saved, constants, job):
    model = new_model(constants, job)
    require(saved['model']['version'] == model.checkpoint_version, 'checkpoint class differs')
    model.load_state_dict(saved['model']['state_dict'], strict=True)
    opt = new_optimizer(model); opt.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['torch_rng'])
    require(run.old.prior.p1a.tree_equal(model.checkpoint(), saved['model']), 'model restoration differs')
    require(run.old.prior.p1a.tree_equal(opt.state_dict(), saved['optimizer']), 'Adam restoration differs')
    require(torch.equal(torch.get_rng_state(), saved['torch_rng']), 'RNG restoration differs')
    return model, opt


def save_tensor(path, value):
    require(not path.exists(), 'immutable checkpoint exists')
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    io.replace_with_retry(temporary, path)
    return identity(path)


def execute(path, job, slice_id=1, qualification=False):
    torch.set_num_threads(1)
    release, inputs = validate(path, job, qualification)
    local = ROOT/'artifacts'/RUN/'durable'/job
    public = ROOT/'results'/RUN/job
    local.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    suffix = 'qualification' if qualification else 'summary'
    require(not (public/(suffix+'.json')).exists(), 'terminal receipt already exists')
    lock = local/'writer.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    context = {}
    write(local/f'slice_{slice_id}_started.json',dict(job_id=job,slice=slice_id,pid=os.getpid(),release_ref=identity(path),qualification=qualification,started_utc=io.utc()))
    try:
        raw, constants = run.load_train(inputs)
        sequence = train.plan(raw)
        plan_path = local/'batch_plan.json'
        if plan_path.exists(): require(io.read(plan_path) == sequence, 'batch plan differs')
        else: run.baseline.atomic_write(plan_path, sequence)
        plan_ref = identity(plan_path)
        canonical = (json.dumps(sequence, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
        # The registered course uses the original atomic writer, not the host status formatter.
        import hashlib
        require(plan_ref['sha256'] == hashlib.sha256(canonical).hexdigest() == release['batch_plan_sha256'], 'frozen batch plan digest differs')
        lookup = {e['id']: e for e in tensor_events(raw, 'cpu', max_steps=800)}
        model = new_model(constants, job); opt = new_optimizer(model)
        rows, guards = [], []
        work, guard_work = IntegrationWork(), IntegrationWork()
        context.update(model=model, opt=opt, rows=rows, work=work, guard_work=guard_work)
        metadata = counts(model, job)
        import hashlib
        metadata['initial_encoder_tensors'] = {key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for key,value in model.encoder.state_dict().items()}
        metadata['initial_encoder_array_equal_to_shared_graph'] = True
        write(local/f'initial_encoder_slice_{slice_id}.json', metadata['initial_encoder_tensors'])
        if qualification:
            first = [lookup[x] for x in sequence['batches'][0]]
            result = dict(**guarded(model, first, constants, guard_work, 800), official_updates=0, **metadata)
            result.update(training_internal_step=.01)
        else:
            transactions = [io.read(p) for p in local.glob('transaction_*.json')]
            if transactions:
                tx = max(transactions, key=lambda x:(x['attempt'], x['slice'], x['guard_count']))
                require(slice_id > tx['slice'], 'slice must advance')
                saved = torch.load(verified(tx['checkpoint']), map_location='cpu', weights_only=True)
                require(saved['release'] == release and saved['plan_ref'] == plan_ref, 'checkpoint authority differs')
                rows = io.read(verified(tx['attempts_ref'])); guards = io.read(verified(tx['guards_ref']))
                require(len(rows) == saved['attempt'] and guards == saved['guards'], 'sealed history differs')
                require(saved['accepted'] == sum(r['accepted'] for r in rows), 'accepted count differs')
                require(saved['rejected'] == sum(not r['accepted'] for r in rows), 'rejected count differs')
                require(all(r['attempt'] == i and r['event_ids'] == sequence['batches'][i-1] for i,r in enumerate(rows,1)), 'sealed batches differ')
                model,opt = restore(saved, constants, job)
                write(local/f'restore_slice_{slice_id}.json',dict(checkpoint_ref=tx['checkpoint'],model_equal=True,adam_equal=True,rng_equal=True,attempt=saved['attempt'],accepted=saved['accepted'],rejected=saved['rejected'],work=saved['work'],guard_work=saved['guard_work']))
                work,guard_work = resume_fit.restore_work(saved['work']),resume_fit.restore_work(saved['guard_work'])
                context.update(model=model,opt=opt,rows=rows,work=work,guard_work=guard_work)
            rejected = sum(not r['accepted'] for r in rows)
            consecutive = 0
            for row in reversed(rows):
                if row['accepted']: break
                consecutive += 1
            stop_reason = None
            latest = None
            def seal():
                stem = f'{len(rows):04d}_guards_{len(guards)}_slice_{slice_id}'
                saved = payload(model,opt,rows,guards,work,guard_work,release,plan_ref)
                existing=local/f'transaction_{stem}.json'
                if existing.exists():
                    tx=io.read(existing)
                    old_saved=torch.load(verified(tx['checkpoint']),map_location='cpu',weights_only=True)
                    require(run.old.prior.p1a.tree_equal(saved,old_saved),'existing seal differs')
                    require(io.read(verified(tx['attempts_ref']))==rows and io.read(verified(tx['guards_ref']))==guards,'existing seal history differs')
                    return tx
                cp = save_tensor(local/f'checkpoint_{stem}.pt', saved)
                # Validate the serialized model, optimizer, RNG and counters before publishing its transaction.
                loaded = torch.load(verified(cp),map_location='cpu',weights_only=True)
                require(run.old.prior.p1a.tree_equal(saved,loaded), 'checkpoint roundtrip differs')
                tx = dict(attempt=len(rows),slice=slice_id,guard_count=len(guards),checkpoint=cp,attempts_ref=write(local/f'attempts_{stem}.json',rows),guards_ref=write(local/f'guards_{stem}.json',guards))
                write(local/f'transaction_{stem}.json',tx)
                print(json.dumps(dict(event='checkpoint',job_id=job,slice=slice_id,attempt=len(rows),accepted=saved['accepted'],rejected=saved['rejected'],checkpoint=cp,work=work.snapshot(),guard_work=guard_work.snapshot())),flush=True)
                return tx
            for k in range(len(rows),2048):
                write(local/'state.json',dict(slice=slice_id,pid=os.getpid(),next_attempt=k+1,status='running'))
                if (local/f'pause_{slice_id}.json').exists():
                    latest = seal(); lock.unlink(); return 75
                batch = [lookup[x] for x in sequence['batches'][k]]
                if k in (512,896):
                    g = dict(attempt=k+1,**guarded(model,[batch[0],batch[-1]],constants,guard_work,train.horizon(k)))
                    guards.append(g)
                    if not g['passed']:
                        stop_reason='stage_guard_failed'; latest=seal(); break
                expected = 4*sum(min(train.horizon(k),len(e['targets']['theta'])-1) for e in batch)
                require(work.rhs['forward']+expected <= train.CAP,'training cap exceeded')
                before = work.rhs['forward']; started,cpu = perf_counter(),process_time()
                context['attempt']=k+1
                row = train.attempt(model,opt,batch,k,constants,work,.01)
                require(work.rhs['forward']-before == expected,'training RHS differs')
                row.update(wall_s=perf_counter()-started,cpu_s=process_time()-cpu,peak_memory_bytes=run.peak_memory_bytes())
                for tail in local.glob(f'attempt_{k+1:04d}_slice_*.json'):
                    require(resume_fit.scientific(io.read(tail)) == resume_fit.scientific(row),'restored tail replay differs')
                    write(local/f'replay_{k+1:04d}_slice_{slice_id}.json',dict(matched=True,previous_ref=identity(tail),actual=resume_fit.scientific(row)))
                write(local/f'attempt_{k+1:04d}_slice_{slice_id}.json',row); rows.append(row)
                rejected += not row['accepted']; consecutive=0 if row['accepted'] else consecutive+1
                stop_reason='nonfinite_rejection_limit' if consecutive >= 3 or rejected >= 16 else None
                if (k+1)%64 == 0 or stop_reason: latest=seal()
                if stop_reason: break
            if latest is None or latest['attempt'] != len(rows): latest=seal()
            valid = len(rows)==2048 and 2048-rejected>=2033 and stop_reason is None
            terminal_saved=torch.load(verified(latest['checkpoint']),map_location='cpu',weights_only=True)
            require(run.old.prior.p1a.tree_equal(terminal_saved['model'],model.checkpoint()) and run.old.prior.p1a.tree_equal(terminal_saved['optimizer'],opt.state_dict()),'terminal checkpoint unseal differs')
            require(run.old.prior.finite_tree(model.state_dict()) and run.old.prior.finite_tree(opt.state_dict()),'nonfinite terminal model or Adam')
            result=dict(**metadata,fixed_endpoint=2048,actual_endpoint=len(rows),attempts=len(rows),accepted=len(rows)-rejected,rejected=rejected,
                        valid_endpoint=valid,stopped_early=not valid,stop_reason=stop_reason,checkpoints={'2048':latest['checkpoint']} if valid else {},
                        terminal_checkpoint=latest['checkpoint'],attempts_ref=latest['attempts_ref'],guards_ref=latest['guards_ref'],guards=guards,
                        plan_ref=plan_ref,work=work.snapshot(),guard_work=guard_work.snapshot(),training_internal_step=.01,
                        spectral_jacobian_calls=sum(r['horizon_s']==8 for r in rows),guard_spectral_jacobian_calls=sum(g['spectral_jacobian_calls'] for g in guards),
                        train_cost=dict(wall_s=sum(r['wall_s'] for r in rows),cpu_s=sum(r['cpu_s'] for r in rows)),
                        health=dict(model_finite=True,adam_finite=True,terminal_checkpoint_unsealed=True),replay_refs=[identity(p) for p in sorted(local.glob('replay_*.json'))])
        receipt = dict(run_id=RUN,job_id=job,status='completed' if qualification or result['valid_endpoint'] else 'scientific_stopped',
                       release=release,release_ref=identity(path),source_hashes=release['source_hashes'],input_refs=release['input_refs'],training_internal_step=.01,result=result)
        summary_ref=write(public/(suffix+'.json'),receipt)
        write(local/f'slice_{slice_id}_completed.json',dict(summary_ref=summary_ref,status=receipt['status'],utc=io.utc()))
        write(local/'state.json',dict(slice=slice_id,pid=os.getpid(),next_attempt=len(rows)+1,status=receipt['status']))
        lock.unlink(); return 0
    except BaseException as error:
        snapshot = None
        if context.get('model') is not None:
            snapshot=save_tensor(local/f'failure_slice_{slice_id}.pt',dict(model=context['model'].checkpoint(),optimizer=context['opt'].state_dict(),torch_rng=torch.get_rng_state(),attempt=context.get('attempt')))
        write(local/f'failure_slice_{slice_id}.json',dict(status='failed',error=repr(error),snapshot_ref=snapshot,recorded_attempts=len(context.get('rows',[])),attempt=context.get('attempt'),model_finite=run.old.prior.finite_tree(context['model'].state_dict()) if 'model' in context else None,adam_finite=run.old.prior.finite_tree(context['opt'].state_dict()) if 'opt' in context else None,work=context['work'].snapshot() if 'work' in context else None,guard_work=context['guard_work'].snapshot() if 'guard_work' in context else None))
        # Retain the claim after catastrophic failure; host inspection is required.
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True); parser.add_argument('--job',choices=JOBS,required=True)
    parser.add_argument('--slice',type=int,default=1); parser.add_argument('--qualify',action='store_true')
    args=parser.parse_args(); sys.exit(execute(args.release,args.job,args.slice,args.qualify))
