"""Execute v2 only after an explicit grant tied to the exact frozen manifest."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import torch

from experiments.osgkd_v2_frozen_confirmation_20261127 import data, run as prior, score
from experiments.osgkd_v2_frozen_confirmation_v2_20261128 import freeze_spec as spec
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io


def validate_frozen(path):
    io.require(path == spec.PUBLIC / 'freeze.json', 'canonical v2 freeze required')
    book = io.read(path)
    io.require(book['run_id'] == spec.RUN and book['evaluation_id'] == spec.EVALUATION_ID and book['status'] == 'frozen'
               and book['user_test_authorized'] is False and book['test_trajectory_access'] is False
               and book['default_model'] == 'none' and book['official_updates'] == book['new_fom'] == 0
               and book['independent_freeze_passed'] is True, 'reviewed frozen v2 required')
    io.verified(book['independent_freeze_review'])
    legacy, old = spec.prior_v1()
    io.require(book['supersedes'] == legacy and book['data_identity'] == old['data_identity']
               and book['constants_source'] == old['constants_source'] and book['numerics'] == old['numerics'] == prior.numerics(),
               'same data, reference and numerical contract required')
    prerequisites = spec.closed_prerequisites()
    decision = io.read(io.verified(book['wp3_2_decision']))
    io.require(decision['status'] == 'closed' and decision['extra_flat_seeds'] == 'completed'
               and decision['local_stability_and_modal'] == 'deferred'
               and decision['supplemental_flat_evidence'] == prerequisites[-1], 'closed supplementary decision required')
    io.verified(book['claim_ledger'])
    io.require(set(book['source_hashes']) == spec.evidence_sources(), 'v2 source inventory differs')
    for path, sha in book['source_hashes'].items():
        absolute = (ROOT/path).resolve()
        io.require(absolute.is_relative_to(ROOT) and io.identity(absolute)['sha256'] == sha, 'frozen source differs: '+path)
    endpoints, alias = spec.fixed_endpoints()
    io.require(book['prerequisites'] == prerequisites and book['endpoints'] == endpoints and book['alias'] == alias
               and book['comparisons'] == spec.fixed_comparisons(), 'frozen v2 endpoints or comparisons differ')
    return book


def exposure_claim(authority, whitelist_sha256):
    """Only a separately authorized v2 process reaches the shared holdout claim."""
    path = spec.READ_CLAIM
    value = dict(evaluation_id=spec.EVALUATION_ID, authorization=authority, whitelist_sha256=whitelist_sha256)
    if path.exists():
        io.require(io.read(path) == value, 'the shared holdout already has another authorized evaluation')
        return io.identity(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return io.identity(path)


def approved_data(metadata, constants, authority, claim):
    local = spec.LOCAL
    access = local/'test_access_receipt.json'
    complete = local/'data_complete.json'
    if complete.exists():
        record = io.read(complete)
        io.require(record['authorization'] == authority and record['shared_claim'] == claim,
                   'saved v2 cache authorization differs')
        io.verified(record['access'])
        stored = torch.load(io.verified(record['cache']), map_location='cpu', weights_only=True)
        io.require(stored['authorization'] == authority, 'private cache authority differs')
        return prior.unpack(stored['events']), record['cache']
    io.require(not access.exists(), 'holdout already entered without a complete private cache')
    record = dict(authorization=authority, shared_claim=claim, opened_utc=io.utc(), reads=[], official_updates=0)
    io.write(access, record)

    def on_read(note):
        record['reads'].append({**note, 'utc': io.utc()})
        io.write(access, record)

    events = data.load_allowed_events(ROOT, metadata, constants, on_read)
    io.require(len(events) == 86 and Counter(e['label'] for e in events) == {'stable': 67, 'unstable': 19},
               'raw v2 test population differs')
    counts = Counter(item['phase'] for item in record['reads'])
    io.require(counts['opened_shard'] == 2 and counts['read_event'] == 86, 'physical holdout read activity differs')
    cache = local/'fixed_input_packets.pt'
    temp = cache.with_suffix('.pt.tmp')
    with temp.open('xb') as stream:
        torch.save(dict(events=prior.pack(events), authorization=authority), stream)
        stream.flush()
        os.fsync(stream.fileno())
    io.replace_with_retry(temp, cache)
    ref = io.identity(cache)
    record.update(closed_utc=io.utc(), event_trajectory_reads=86, physical_shard_opens=2, cache=ref)
    io.write(access, record)
    io.write(complete, dict(authorization=authority, shared_claim=claim, cache=ref, access=io.identity(access)))
    return events, ref


def sealed_endpoint(entry, model, events, constants, authority, cache):
    name = entry['id']
    folder = spec.LOCAL / 'endpoints' / name
    folder.mkdir(parents=True, exist_ok=True)
    terminal = folder/'terminal.json'
    if not terminal.exists():
        ready = sorted(folder.glob('work*/endpoint_summary.json'))
        io.require(len(ready) <= 1, 'multiple complete endpoint scores found')
        result = io.read(ready[0]) if ready else score.score_endpoint(model, events, constants, folder/f'work_{os.getpid()}')
        io.write(terminal, dict(entry=entry, authorization=authority, cache=cache, result=result))
    row = io.read(terminal)
    io.require(row['entry'] == entry and row['authorization'] == authority and row['cache'] == cache,
               'sealed endpoint authority differs')
    io.verified(row['result']['rows_ref'])
    return row['result']


def write_public(path, value):
    io.require(len(json.dumps(value, allow_nan=False).encode('utf-8')) <= 512*1024, 'public evidence exceeds size limit')
    path.parent.mkdir(parents=True, exist_ok=True)
    io.write(path, value)
    return io.identity(path)


def execute(path):
    io.require(path == spec.PUBLIC/'authorized_release.json', 'explicit v2 grant path required')
    grant = io.read(path)
    io.require(grant['run_id'] == spec.RUN and grant['frozen_evaluation_id'] == spec.EVALUATION_ID
               and grant['user_authorized_once'] is True and bool(grant['user_authorization_reference']),
               'explicit authorization for this v2 evaluation required')
    io.require(Path(sys.executable).resolve() == (ROOT.parent/'.venv/Scripts/python.exe').resolve(), 'shared Python required')
    book = validate_frozen(io.verified(grant['freeze']))
    metadata = data.read_source_metadata(ROOT)
    io.require(prior.data_identity(metadata) == book['data_identity'], 'v2 holdout metadata differs')
    torch.set_num_threads(1)
    models, constants = prior.load_models(book)
    authority = io.identity(path)
    spec.LOCAL.mkdir(parents=True, exist_ok=True)
    io.require(not (spec.LOCAL/'scientific_stop.json').exists(), 'previous scientific stop requires a separate decision')
    lease = spec.LOCAL/'score_writer.lock'
    with lease.open('x') as stream:
        stream.write(str(os.getpid()))
    started, cpu = perf_counter(), process_time()
    try:
        out = spec.PUBLIC/'summary.json'
        io.require(not out.exists(), 'v2 evaluation has already finished')
        claim = exposure_claim(authority, book['data_identity']['whitelist_sha256'])
        events, cache = approved_data(metadata, constants, authority, claim)
        results, refs = {}, {}
        for entry in book['endpoints']:
            name = entry['id']
            results[name] = sealed_endpoint(entry, models[name], events, constants, authority, cache)
            refs[name] = write_public(spec.PUBLIC/'endpoints'/f'{name}.json', dict(endpoint=name, result=results[name]))
            print(name, 'sealed', flush=True)
        pairwise = []
        sources = {e['id']: e['metadata']['source_root'] for e in events}
        for pair in book['comparisons']:
            left, right = [io.read(io.verified(results[pair[k]]['rows_ref'])) for k in ('reference', 'candidate')]
            outcome = prior.paired_rows(left, right, sources)
            name = f"{pair['role']}_{pair['reference']}_vs_{pair['candidate']}.json"
            file = spec.LOCAL / 'pairs' / name
            file.parent.mkdir(parents=True, exist_ok=True)
            io.write(file, outcome.pop('disagreements'))
            ref = write_public(spec.PUBLIC/'pairs'/name, {**pair, **outcome, 'disagreements_ref': io.identity(file)})
            pairwise.append({**pair, 'paired_result': ref})
        write_public(out, dict(run_id=spec.RUN, status='completed', evaluation_id=spec.EVALUATION_ID, release=authority,
                               shared_claim=claim, cache=cache, access_receipt=io.identity(spec.LOCAL/'test_access_receipt.json'),
                               endpoint_results=refs, alias=book['alias'], paired_comparisons=pairwise,
                               raw_events=len(events), new_fom=0, official_updates=0, default_model='none',
                               wall_s=perf_counter()-started, cpu_s=process_time()-cpu, completed_utc=io.utc()))
    except BaseException as error:
        technical = isinstance(error, (OSError, MemoryError, KeyboardInterrupt))
        failure = dict(error=repr(error), authorization=authority, technical=technical, utc=io.utc())
        io.write(spec.LOCAL/f'failed_{os.getpid()}.json', failure)
        if not technical:
            io.write(spec.LOCAL/'scientific_stop.json', failure)
        raise
    finally:
        lease.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    execute(parser.parse_args().release.resolve())
