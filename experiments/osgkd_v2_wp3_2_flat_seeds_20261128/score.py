"""One official original-val scoring pass for the two supplemental fixed endpoints."""

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch

from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io, run as matched
from experiments.osgkd_v2_sprint2b_baselines_20261126 import score as shared_score
from experiments.osgkd_v2_wp3_2_flat_seeds_20261128 import fit, postprocess
from kiki.data.osgkd_train_val import load_val
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_matched_flat import OSGKDMatchedFlat


PARENT = ROOT / 'results/osgkd_v2_sprint2a_20261125/evaluate01/summary.json'
PARENT_SHA = '7645c8b006b9dd6e71838e7e875da69117ed3df9011938d36f46cf126c6b3fbb'
CONSTANTS = ROOT / 'results/osgkd_fit297_20261113/port_audit.json'


def validate(path):
    public = ROOT / 'results' / fit.RUN
    permit = io.read(path)
    io.require(path == public / 'evaluate_release.json' and permit['released'] is True
               and permit['run_id'] == fit.RUN and permit['original_val_passes'] == 1,
               'canonical once-only val release required')
    io.require(Path(sys.executable).resolve() == (ROOT.parent / '.venv/Scripts/python.exe').resolve()
               and permit['test_trajectory_access'] is False and permit['new_fom'] == 0
               and permit['default_model'] == 'none', 'fixed scoring scope differs')
    parent = io.identity(PARENT)
    io.require(parent == permit['parent_evaluation'] and parent['sha256'] == PARENT_SHA,
               'historical four-endpoint scores differ')
    io.require(permit['constants'] == io.identity(CONSTANTS), 'frozen constants differ')
    claim = io.verified(permit['fit_release'])
    authority, _ = fit.validate(claim, fit.ORDER[0], False)
    sources = authority['source_hashes']
    for source in ('score.py', 'postprocess.py'):
        name = f'experiments/{fit.RUN}/{source}'
        io.require(sources[name] == io.identity(ROOT / name)['sha256'], 'registered score source differs: ' + name)
    training = io.read(io.verified(permit['training_summary']))
    io.require(training['run_id'] == fit.RUN and training['state'] == 'completed'
               and training['fit_release'] == permit['fit_release'] and list(training['serial_order']) == list(fit.ORDER)
               and set(training['jobs']) == set(fit.ORDER), 'two normally completed fixed fits required')
    native = io.read(io.verified(training['native_host_exit']))
    io.require(native['mode'] == 'fit' and native['native_exit_code'] == 0
               and native['release_ref']['sha256'] == permit['fit_release']['sha256'], 'fit host native exit differs')
    records = {}
    for job in fit.ORDER:
        fit.validate(claim, job, False)
        record = training['jobs'][job]
        receipt = io.read(io.verified(record['summary_ref']))
        exit_note = io.read(io.verified(record['native_exit_ref']))
        result = receipt['result']
        io.require(record['summary_ref']['path'] == f'results/{fit.RUN}/{job}/summary.json'
                   and receipt['run_id'] == fit.RUN and receipt['job_id'] == job and receipt['status'] == 'completed'
                   and receipt['release_ref'] == permit['fit_release'] and receipt['release'] == authority
                   and exit_note['job'] == job and exit_note['native_exit_code'] == 0
                   and result['valid_endpoint'] and result['attempts'] == 2048 and result['accepted'] >= 2033,
                   'training terminal differs: ' + job)
        io.verified(result['checkpoints']['2048'])
        records[job] = receipt
    return permit, records, io.read(PARENT), io.read(io.verified(permit['constants']))['constants']


def evaluate(path):
    permit, records, old, constants = validate(path)
    reference = io.identity(path)
    public = ROOT / 'results' / fit.RUN / 'evaluate01'
    private = ROOT / 'artifacts' / fit.RUN / 'evaluate01'
    io.require(not public.exists() and not private.exists(), 'one original-val score already claimed')
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    io.write(private / 'started.json', dict(release_ref=reference, pid=os.getpid(), started_utc=io.utc()))
    torch.set_num_threads(1)
    started, cpu = perf_counter(), process_time()
    try:
        events = load_val(constants)
        data_identity = matched.baseline.data_identity(events)
        io.require(data_identity == old['result']['val_identity'], 'original val identity differs')
        groups = shared_score.rollout_groups(events)
        endpoints = dict(old['result']['val'])
        timing = {}
        for job, item in endpoints.items():
            cls = OSGKDGraph if item['encoder'] == 'graph' else OSGKDMatchedFlat
            prior = io.read(io.verified(old['release']['fit_refs'][job]))
            model = shared_score.from_endpoint(item['checkpoint_ref'], cls, prior['release'])
            shared_score.warm_up(model)
            timing[job] = shared_score.time_archived_endpoint(model, groups)
            print(job, 'historical endpoint inference timing', flush=True)

        counts = dict(encoder_parameters=6144, effective_encoder_parameters=6144, ode_parameters=680,
                      active_parameters=6824, embedded_state_dimension=50, effective_state_dimension=50)
        scored = {}
        for job, receipt in records.items():
            result = receipt['result']
            model = shared_score.from_endpoint(result['checkpoints']['2048'], OSGKDMatchedFlat, receipt['release'])
            io.require(matched.fit.encoder_parameters(model) == 6144
                       and sum(p.numel() for p in model.parameters()) == 6824, 'supplemental endpoint size differs')
            shared_score.warm_up(model)
            item = shared_score.score_endpoint(job, model, events, groups, constants, private / job,
                                               dict(counts, checkpoints=result['checkpoints']))
            item.update(seed=int(job.rsplit('_', 1)[1]), encoder='flat', encoder_parameters=6144)
            io.require(item['score']['data_identity'] == data_identity, 'supplemental val identity differs')
            scored[job] = item
            timing[job] = item['timing']
            print(job, item['score']['numerical_qualified_events'], item['spectrum']['qualified'], flush=True)
        endpoints.update(scored)
        comparison = postprocess.compare(endpoints, timing)
        eligible = all(row['score']['gate_passed'] and row['score']['numerical_qualified_events'] == 107
                       and row['spectrum']['qualified'] and row['spectrum']['dimension'] == 49
                       for row in endpoints.values())
        summary = dict(run_id=fit.RUN, status='completed', job_id='evaluate01', release_ref=reference,
                       parent_evaluation=permit['parent_evaluation'], training_summary=permit['training_summary'],
                       val_identity=data_identity, val_raw=117, val_initializable=107, val=scored, timings=timing,
                       six_endpoints_qualified=eligible, comparison=comparison, original_val_passes=1,
                       new_fom=0, test_trajectory_access=False, default_model='none',
                       cost=dict(wall_s=perf_counter()-started, cpu_s=process_time()-cpu,
                                 peak_working_set_bytes=matched.peak_memory_bytes()), completed_utc=io.utc())
        ref = matched.baseline.atomic_write(public / 'summary.json', summary, public=True)
        io.write(private / 'completed.json', dict(status='completed', summary_ref=ref))
        print(ref, flush=True)
    except BaseException as error:
        io.write(private / 'failed.json', dict(status='failed', release_ref=reference, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    evaluate(parser.parse_args().release.resolve())
