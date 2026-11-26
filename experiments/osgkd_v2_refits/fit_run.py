"""Single-process, train-only WP1A refits with immutable resume transactions."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import torch

from experiments.osgkd_v2_discovery import prepare
from experiments.osgkd_v2_refits import recipe
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io, resume_fit, run as matched
from kiki.models.osgkd_fitting import IntegrationWork, tensor_events
from kiki.models.osgkd_graph import OSGKDGraph


RUN = 'osgkd_v2_wp1a_refits_20261127'
PUBLIC = ROOT / 'results' / RUN
LOCAL = ROOT / 'artifacts' / RUN / 'fit'
PLAN_SHA = 'b11463b14f01ed1c26f60b6227f7ce0072a90d5f465e61593fbc056878e8ae55'
VARIANTS = ('full', 'static', 'top30', 'top20', 'no_beta', 'no_memory', 'no_coupling', 'frozen_p')
MIN_ACCEPTED = 113
FULL_GUARD = dict(stable_8s_abs=.2, stable_8s_rel=.25, additional_stable_fp=1,
                  unstable_hit_drop=1, positive_real_growth_increase=.15)
REQUIRED_SOURCES = matched.SOURCES | {
    'experiments/osgkd_v2_refits/' + name for name in ('optimizer.py', 'recipe.py', 'fit_run.py', 'launch.ps1', 'score.py')
} | {'experiments/osgkd_v2_discovery/prepare.py', 'src/kiki/models/osgkd_restricted.py',
     'campaigns/osgkd_v2/wp1a_matched_refit_protocol.md',
     'experiments/osgkd_v2_sprint2a_20261125/resume_host.py',
     'experiments/osgkd_v2_sprint2a_20261125/resume_fit.py'}
require = io.require
tree_equal = matched.old.prior.p1a.tree_equal
finite_tree = matched.old.prior.finite_tree


def registered_multipliers():
    warmup = [.1 + .4 * k / 7 for k in range(8)]
    cosine = [.1 + .2 * (1 + math.cos(math.pi * k / 119)) for k in range(120)]
    return warmup + cosine


def write(path, value):
    io.write(path, value)
    return io.identity(path)


def validate(path):
    release = io.read(path)
    require(path == PUBLIC / 'release.json' and release['run_id'] == RUN and release['released'] is True,
            'canonical released WP1A manifest required')
    require(Path(sys.executable).resolve() == (ROOT.parent / '.venv/Scripts/python.exe').resolve(),
            'shared interpreter required')
    require(REQUIRED_SOURCES <= set(release['source_hashes']), 'source binding coverage missing')
    for name, digest in release['source_hashes'].items():
        source = (ROOT / name).resolve()
        require(source.is_relative_to(ROOT) and io.identity(source)['sha256'] == digest, 'source differs: ' + name)
    require(release['plan_sha256'] == PLAN_SHA, 'common full batch plan differs')
    require(release['min_accepted'] == MIN_ACCEPTED and release['multipliers'] == registered_multipliers(),
            'registered learning rate or acceptance floor differs')
    require(release['full_retention_guard'] == FULL_GUARD, 'registered full retention guard differs')
    require(release['scientist']['configured_model'] == 'gpt-6-astra' and
            release['independent_preflight']['status'] == 'passed', 'scientific and separate review required')
    parent = io.read(io.verified(release['parent_evaluation']))
    wp0 = io.read(io.verified(release['wp0_summary']))
    require(parent['run_id'] == matched.RUN and parent['status'] == 'completed', 'completed parent evaluation required')
    require(wp0['run_id'] == 'osgkd_v2_wp0_20261127' and wp0['status'] == 'completed' and
            wp0['official_updates'] == 0, 'completed zero-update WP0 required')
    wp0_release = io.read(io.verified(wp0['release']))
    require(wp0_release['parent'] == release['parent_evaluation'], 'WP0 parent differs')
    edges = io.read(io.verified(wp0['edges']['details_ref']))
    stable = wp0['edges']['statistics']['stable_K']
    require(all(stable[k] is True for k in ('30', '20')) and stable['12'] is False,
            'WP0-gated variants differ')
    require(tuple(v for v in recipe.VARIANTS if v != 'top12') == VARIANTS, 'variant definitions differ')
    expected = [(seed, variant) for seed in prepare.GRAPH_SEEDS for variant in VARIANTS]
    require([(job['seed'], job['variant']) for job in release['jobs']] == expected,
            'same selected variants in canonical seed-major order required')
    for job in release['jobs']:
        require(job['id'] == f"{job['variant']}_{job['seed']}", 'canonical job id required')
        plan = edges['per_seed'][f"graph_{job['seed']}"]['restriction_plan'][job['variant']]
        require(job['restrictions'] == plan, 'WP0 restriction differs: ' + job['id'])
    return release, parent


def load_parents(parent):
    models, refs = {}, {}
    for seed in prepare.GRAPH_SEEDS:
        key = f'graph_{seed}'
        receipt = io.read(io.verified(parent['release']['fit_refs'][key]))
        result = receipt['result']
        ref = result['checkpoints']['2048']
        require(result['valid_endpoint'] and result['attempts'] == 2048 and
                parent['result']['val'][key]['checkpoint_ref'] == ref, 'fixed parent endpoint differs')
        saved = torch.load(io.verified(ref), map_location='cpu', weights_only=True)
        require(saved['attempt'] == 2048 and saved['release'] == receipt['release'], 'parent authority differs')
        models[seed] = OSGKDGraph.from_checkpoint(saved['model']).eval().requires_grad_(False)
        refs[seed] = ref
    return models, refs


def save_tensor(path, saved):
    require(not path.exists(), 'immutable checkpoint exists')
    temp = path.with_suffix('.pt.tmp')
    with temp.open('wb') as stream:
        torch.save(saved, stream)
        stream.flush()
        os.fsync(stream.fileno())
    io.replace_with_retry(temp, path)
    return io.identity(path)


def tensor_hashes(value):
    """Component hashes supplement the full serialized checkpoint byte hash."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def fit_job(job, parent, parent_ref, release, release_ref, plan_ref, lookup, batches, constants):
    name = job['id']
    local, public = LOCAL / name, PUBLIC / name
    local.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    summary_path = public / 'summary.json'
    if summary_path.exists():
        receipt = io.read(summary_path)
        require(receipt['release_ref'] == release_ref and receipt['job'] == job, 'terminal job authority differs')
        if receipt['result'].get('terminal_checkpoint'):
            io.verified(receipt['result']['terminal_checkpoint'])
        return receipt
    spec = prepare.Restrictions(**job['restrictions'])
    torch.manual_seed(job['seed'])
    model = recipe.new_model(parent, spec)
    opt = recipe.new_optimizer(model)
    rows, diagnostics = [], []
    work, guard_work, diagnostic_work = IntegrationWork(), IntegrationWork(), IntegrationWork()
    qualification = None
    checkpoints, seals = {}, []
    fixed_batch = [lookup[key] for key in batches[0]]
    stop_reason = None
    active_index = None
    started, cpu = perf_counter(), process_time()

    def seal():
        count = len(rows)
        saved = copy.deepcopy(dict(model=model.checkpoint(), optimizer=opt.state_dict(),
                    torch_rng=torch.get_rng_state(), attempt=count, rows=rows, diagnostics=diagnostics,
                    qualification=qualification, work=work.snapshot(), guard_work=guard_work.snapshot(),
                    diagnostic_work=diagnostic_work.snapshot(), release_ref=release_ref, job=job,
                    parent_ref=parent_ref, plan_ref=plan_ref))
        hashes = {key: tensor_hashes(saved[key]) for key in ('model', 'optimizer', 'torch_rng')}
        checkpoint_path = local / f'checkpoint_{count:03d}.pt'
        if checkpoint_path.exists():
            checkpoint_ref = io.identity(checkpoint_path)
            require(tree_equal(torch.load(io.verified(checkpoint_ref), map_location='cpu', weights_only=True), saved),
                    'orphan checkpoint differs from deterministic replay')
        else:
            checkpoint_ref = save_tensor(checkpoint_path, saved)
        loaded = torch.load(io.verified(checkpoint_ref), map_location='cpu', weights_only=True)
        require(tree_equal(saved, loaded), 'checkpoint roundtrip differs')
        transaction = dict(attempt=count, checkpoint=checkpoint_ref, component_sha256=hashes,
                           attempts_ref=write(local / f'attempts_{count:03d}.json', rows),
                           diagnostics_ref=write(local / f'diagnostics_{count:03d}.json', diagnostics),
                           qualification_ref=qualification_ref, release_ref=release_ref)
        transaction_ref = write(local / f'transaction_{count:03d}.json', transaction)
        seals.append(transaction_ref)
        checkpoints[str(count)] = checkpoint_ref
        return transaction

    try:
        transactions = [io.read(path) for path in sorted(local.glob('transaction_*.json'))]
        if transactions:
            for transaction in transactions:
                require(transaction['release_ref'] == release_ref, 'resume authority differs')
                io.verified(transaction['checkpoint'])
                checkpoints[str(transaction['attempt'])] = transaction['checkpoint']
                seals.append(io.identity(local / f"transaction_{transaction['attempt']:03d}.json"))
            transaction = max(transactions, key=lambda row: row['attempt'])
            saved = torch.load(io.verified(transaction['checkpoint']), map_location='cpu', weights_only=True)
            require(saved['release_ref'] == release_ref and saved['job'] == job and
                    saved['parent_ref'] == parent_ref and saved['plan_ref'] == plan_ref, 'resume bindings differ')
            require(saved['rows'] == io.read(io.verified(transaction['attempts_ref'])) and
                    saved['diagnostics'] == io.read(io.verified(transaction['diagnostics_ref'])), 'resume history differs')
            for key in ('model', 'optimizer', 'torch_rng'):
                require(tensor_hashes(saved[key]) == transaction['component_sha256'][key], 'resume component hash differs')
            model.load_state_dict(saved['model']['state_dict'], strict=True)
            opt.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng'])
            require(tree_equal(model.checkpoint(), saved['model']) and tree_equal(opt.state_dict(), saved['optimizer']),
                    'restored model or Adam differs')
            rows, diagnostics, qualification = saved['rows'], saved['diagnostics'], saved['qualification']
            work, guard_work, diagnostic_work = [resume_fit.restore_work(saved[key])
                                                for key in ('work', 'guard_work', 'diagnostic_work')]
            qualification_ref = transaction['qualification_ref']
            require(io.read(io.verified(qualification_ref)) == qualification, 'qualification binding differs')
        else:
            try:
                qualification = recipe.qualify(model, [fixed_batch[0], fixed_batch[-1]], constants, guard_work)
            except ValueError as error:
                if str(error) != 'nonfinite fine qualification: blocked':
                    raise
                qualification = dict(passed=False, error=str(error), official_updates=0, work=guard_work.snapshot())
            qualification_ref = write(local / 'qualification.json', qualification)
            if qualification['passed']:
                diagnostics.append(dict(attempt=0, **recipe.diagnostic(model, fixed_batch, constants, diagnostic_work)))
                require(diagnostics[0]['finite'], 'nonfinite initial diagnostic')
                seal()
        if not qualification['passed']:
            stop_reason = 'qualification_failed_fixed_mesh'
        else:
            for k in range(len(rows), recipe.ATTEMPTS):
                rejected = sum(not row['accepted'] for row in rows)
                consecutive = 0
                for row in reversed(rows):
                    if row['accepted']:
                        break
                    consecutive += 1
                if consecutive >= 3 or rejected >= 16:
                    stop_reason = 'nonfinite_rejection_limit'
                    break
                batch = [lookup[key] for key in batches[k]]
                write(local / 'state.json', dict(status='running', index=k, pid=os.getpid(), utc=io.utc()))
                before = work.rhs['forward']
                wall_start, cpu_start = perf_counter(), process_time()
                active_index = k
                row = recipe.attempt(model, opt, batch, k, constants, work, release['multipliers'])
                expected = 4*sum(min(800, len(event['targets']['theta'])-1) for event in batch)
                require(work.rhs['forward']-before == expected, 'fixed-mesh RHS count differs')
                row.update(wall_s=perf_counter()-wall_start, cpu_s=process_time()-cpu_start)
                tail = local / f'attempt_{k+1:03d}.json'
                if tail.exists():
                    previous = io.read(tail)
                    require(resume_fit.scientific(previous) == resume_fit.scientific(row), 'unsealed replay differs')
                    io.append(local / 'replays.jsonl', dict(attempt=k+1, wall_s=row['wall_s'], cpu_s=row['cpu_s']))
                    row = previous
                else:
                    write(tail, row)
                rows.append(row)
                active_index = None
                if len(rows) == 128:
                    diagnostics.append(dict(attempt=128, **recipe.diagnostic(model, fixed_batch, constants, diagnostic_work)))
                if len(rows) in (64, 128):
                    seal()
            rejected = sum(not row['accepted'] for row in rows)
            if rejected >= 16 or (len(rows) >= 3 and all(not row['accepted'] for row in rows[-3:])):
                stop_reason = 'nonfinite_rejection_limit'
            if not checkpoints or str(len(rows)) not in checkpoints:
                seal()
        accepted = sum(row['accepted'] for row in rows)
        valid = (qualification['passed'] and len(rows) == 128 and stop_reason is None and
                 accepted >= release['min_accepted'] and diagnostics[-1]['finite'])
        if not valid and stop_reason is None:
            stop_reason = 'min_accepted_not_met' if accepted < release['min_accepted'] else 'nonfinite_endpoint_diagnostic'
        result = dict(valid_endpoint=valid, fixed_endpoint=128, actual_endpoint=len(rows), attempts=len(rows),
                      accepted=accepted, rejected=len(rows)-accepted, stop_reason=stop_reason,
                      checkpoints={key: value for key, value in checkpoints.items() if key in ('64', '128')},
                      terminal_checkpoint=checkpoints.get(str(len(rows))), qualification_ref=qualification_ref,
                      attempts_ref=write(local / 'attempts.json', rows),
                      diagnostics_ref=write(local / 'diagnostics.json', diagnostics), seals=seals,
                      work=work.snapshot(), guard_work=guard_work.snapshot(), diagnostic_work=diagnostic_work.snapshot(),
                      plan_ref=plan_ref, parent_ref=parent_ref, training_internal_step=.01,
                      spectral_jacobian_calls=0, cost=dict(wall_s=perf_counter()-started, cpu_s=process_time()-cpu))
        status = 'completed' if valid else 'scientific_stopped'
    except Exception as error:
        failure_ref = save_tensor(local / f'failure_{os.getpid()}.pt', dict(model=model.checkpoint(), optimizer=opt.state_dict(),
                                 torch_rng=torch.get_rng_state(), attempt=len(rows), release_ref=release_ref, job=job))
        fatal_nonfinite = str(error).startswith(('nonfinite parameter or Adam', 'nonfinite projection',
                                                'nonfinite initial diagnostic'))
        write(local / f'failure_{os.getpid()}.json', dict(error=repr(error), failure_ref=failure_ref,
              technical_interruption=not fatal_nonfinite, recorded_attempts=len(rows), utc=io.utc()))
        if not fatal_nonfinite:
            # No terminal public receipt: restart replays from the latest seal.
            raise
        fatal_count = int(active_index is not None)
        result = dict(valid_endpoint=False, actual_endpoint=len(rows), attempts=len(rows)+fatal_count,
                      recorded_attempts=len(rows),
                      accepted=sum(row['accepted'] for row in rows), rejected=sum(not row['accepted'] for row in rows)+fatal_count,
                      stop_reason='fatal_nonfinite', error=repr(error), failure_ref=failure_ref,
                      fatal_attempt_index=active_index,
                      attempts_ref=write(local / 'attempts.json', rows),
                      diagnostics_ref=write(local / 'diagnostics.json', diagnostics),
                      terminal_checkpoint=None, checkpoints={}, parent_ref=parent_ref, plan_ref=plan_ref,
                      work=work.snapshot(), guard_work=guard_work.snapshot(), diagnostic_work=diagnostic_work.snapshot())
        status = 'scientific_stopped'
    receipt = dict(run_id=RUN, job_id=name, job=job, seed=job['seed'], variant=job['variant'], status=status,
                   release_ref=release_ref, source_hashes=release['source_hashes'], result=result, completed_utc=io.utc())
    summary_ref = write(summary_path, receipt)
    write(local / 'state.json', dict(status=status, summary_ref=summary_ref, pid=os.getpid(), utc=io.utc()))
    return receipt


def execute(path):
    release, parent = validate(path)
    release_ref = io.identity(path)
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = LOCAL / 'writer.lock'
    # A hard crash retains this claim; inspect the PID before manual removal.
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        torch.set_num_threads(1)
        models, parent_refs = load_parents(parent)
        anchor = io.read(io.verified(parent['release']['input_refs']['anchor_analysis']))
        raw, constants = matched.load_train({'anchor_analysis': anchor})
        plan = prepare.matched.plan(raw)
        canonical = (json.dumps(plan, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
        require(hashlib.sha256(canonical).hexdigest() == release['plan_sha256'], 'train common batch plan differs')
        plan_path = LOCAL / 'batch_plan.json'
        if plan_path.exists():
            require(io.read(plan_path) == plan, 'saved batch plan differs')
        else:
            matched.baseline.atomic_write(plan_path, plan)
        plan_ref = io.identity(plan_path)
        require(plan_ref['sha256'] == PLAN_SHA, 'plan file bytes differ')
        batches = plan['batches'][-128:]
        require(len({event_id for batch in batches for event_id in batch}) == len(raw) == 385,
                '128 batches must cover the complete training population')
        lookup = {event['id']: event for event in tensor_events(raw, 'cpu', max_steps=800)}
        require(raw and next(e for e in raw if e['id'] == batches[0][0])['label'] == 'stable' and
                next(e for e in raw if e['id'] == batches[0][-1])['label'] == 'unstable', 'qualification pair labels differ')
        results, wave2 = {}, None
        for job in release['jobs']:
            if job['seed'] != 24111 and wave2 is None:
                wave2 = results['full_24111']['result']['valid_endpoint']
                write(PUBLIC / 'wave2_gate.json', dict(release_ref=release_ref, passed=wave2,
                      authority=io.identity(PUBLIC / 'full_24111/summary.json'), basis='train execution validity only'))
            if job['seed'] != 24111 and not wave2:
                public = PUBLIC / job['id']
                public.mkdir(parents=True, exist_ok=True)
                receipt = dict(run_id=RUN, job_id=job['id'], job=job, seed=job['seed'], variant=job['variant'],
                               status='held', release_ref=release_ref,
                               result=dict(valid_endpoint=False, attempts=0, accepted=0, rejected=0,
                                           stop_reason='wave1_full_invalid', checkpoints={}, terminal_checkpoint=None))
                if (public / 'summary.json').exists():
                    require(io.read(public / 'summary.json') == receipt, 'held receipt differs')
                else:
                    write(public / 'summary.json', receipt)
                results[job['id']] = receipt
                continue
            io.write(LOCAL / 'live.json', dict(job_id=job['id'],pid=os.getpid(),utc=io.utc()))
            results[job['id']] = fit_job(job, models[job['seed']], parent_refs[job['seed']], release,
                                        release_ref, plan_ref, lookup, batches, constants)
            print(json.dumps({"job_id":job["id"],"status":results[job["id"]]["status"],
                              "accepted":results[job["id"]]["result"]["accepted"]}),flush=True)
        terminal = dict(run_id=RUN, status='completed' if wave2 else 'wave2_held', release_ref=release_ref,
                        wave2_released=wave2, jobs={key: dict(status=value['status'],
                        valid_endpoint=value['result']['valid_endpoint'],
                        summary_ref=io.identity(PUBLIC / key / 'summary.json')) for key, value in results.items()},
                        val_access=False, test_access=False, completed_utc=io.utc())
        write(PUBLIC / 'fit_summary.json', terminal)
        io.write(LOCAL / 'live.json', dict(status='terminal',pid=os.getpid(),utc=io.utc()))
    except BaseException as error:
        write(LOCAL / f'failure_{os.getpid()}.json', dict(error=repr(error), release_ref=release_ref, utc=io.utc()))
        raise
    else:
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    execute(parser.parse_args().release.resolve())
