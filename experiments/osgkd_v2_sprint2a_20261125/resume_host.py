"""One-shot Windows supervisor for the two authorized Sprint 2A batches."""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import errno
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RUN = 'osgkd_v2_sprint2a_20261125'
LOCAL = ROOT / 'artifacts' / RUN / 'durable'
PUBLIC = ROOT / 'results' / RUN
SOURCES = ROOT / 'experiments' / RUN
PROTOCOL = 'campaigns/osgkd_v2/sprint2a_spectral_matched_protocol.md'
WAVES = (('graph_24111', 'flat_24111'), ('graph_24121', 'graph_24131'))
LAST = {'graph_24111': 896, 'flat_24111': 896, 'graph_24121': 832, 'graph_24131': 0}
MIN_AVAILABLE = .30
PAUSED_EXIT = 75


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    data, _ = retry_windows_io(path.read_bytes)
    return json.loads(data)


def identity(path):
    path = path.resolve()
    data, _ = retry_windows_io(path.read_bytes)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def verified(ref):
    path = (ROOT / ref['path']).resolve()
    require(path.is_relative_to(ROOT), 'receipt path is outside the repository')
    require(identity(path) == ref, f'receipt bytes differ: {ref["path"]}')
    return path


def retry_windows_io(operation, *args):
    deadline = time.monotonic() + 8.
    retries = 0
    while True:
        try:
            return operation(*args), retries
        except PermissionError as error:
            winerror = getattr(error, 'winerror', None)
            sharing_error = winerror in (5, 32, 33) or (winerror is None and error.errno == errno.EACCES)
            if os.name != 'nt' or not sharing_error or time.monotonic() >= deadline:
                raise
            retries += 1
            time.sleep(.025)


def replace_with_retry(temp, path):
    _, retries = retry_windows_io(os.replace, temp, path)
    return retries


def write(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temp, path)


def append(path, value):
    with path.open('a', encoding='utf8', newline='\n') as stream:
        print(json.dumps(value, sort_keys=True, allow_nan=False), file=stream, flush=True)
        os.fsync(stream.fileno())


class PerformanceInformation(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD)] + [(key, ctypes.c_size_t) for key in (
        'commit_total', 'commit_limit', 'commit_peak', 'physical_total', 'physical_available',
        'system_cache', 'kernel_total', 'kernel_paged', 'kernel_nonpaged', 'page_size',
    )] + [(key, wintypes.DWORD) for key in ('handles', 'processes', 'threads')]


def memory():
    counters = PerformanceInformation()
    counters.cb = ctypes.sizeof(counters)
    query = ctypes.windll.psapi.GetPerformanceInfo
    query.argtypes = [ctypes.POINTER(PerformanceInformation), wintypes.DWORD]
    query.restype = wintypes.BOOL
    require(bool(query(ctypes.byref(counters), counters.cb)), 'GetPerformanceInfo failed')
    size = counters.page_size
    return {
        'physical_total_bytes': counters.physical_total * size,
        'physical_available_bytes': counters.physical_available * size,
        'physical_available_fraction': counters.physical_available / counters.physical_total,
        'system_commit_bytes': counters.commit_total * size,
        'system_commit_limit_bytes': counters.commit_limit * size,
        'system_commit_peak_since_boot_bytes': counters.commit_peak * size,
    }


def memory_cut(next_attempts, available_fraction):
    return sum(attempt >= 897 for attempt in next_attempts) >= 2 and available_fraction < MIN_AVAILABLE


def release_inputs(path):
    path = path.resolve()
    lineage = {
        'resume_release_v2.json': ('resume_release.json', 'resume_runtime_retry.json'),
        'resume_release_v3.json': ('resume_release_v2.json', 'resume_runtime_read_retry.json'),
    }
    require(path.parent == PUBLIC and (path.name == 'resume_release.json' or path.name in lineage), 'canonical release path required')
    release = read(path)
    require((release['run_id'], release['released'], release['memory_reserve_fraction']) == (RUN, True, MIN_AVAILABLE),
            'invalid supervisor release')
    require(tuple(map(tuple, release['waves'])) == WAVES and release['restored_attempts'] == LAST, 'batch order differs')
    if path.name in lineage:
        prior, incident = lineage[path.name]
        require(verified(release['supersedes']) == PUBLIC / prior, 'preceding release reference differs')
        require(verified(release['runtime_retry_ref']) == PUBLIC / incident, 'runtime repair reference differs')
    shared = (ROOT.parent / '.venv' / 'Scripts' / 'python.exe').resolve()
    require(Path(sys.executable).resolve() == shared, 'shared Python interpreter required')
    require(identity(ROOT / PROTOCOL)['sha256'] == release['protocol_sha256'], 'frozen protocol differs')
    for name, digest in release['source_hashes'].items():
        require(identity(ROOT / name)['sha256'] == digest, f'frozen source differs: {name}')
    gate = read(verified(release['phase1_ref']))
    require(gate['resume_authorized'] and gate['official_updates'] == 0, 'zero-update safety gate has not passed')
    require(set(release['original_releases']) == set(LAST), 'all original releases required')
    for job, ref in release['original_releases'].items():
        require(verified(ref) == PUBLIC / f'{job}_release.json', 'original release path differs')
    return release


def endpoint(job, release):
    summary = PUBLIC / job / 'summary.json'
    if not summary.is_file():
        return None
    receipt = read(summary)
    result = receipt['result']
    require((receipt['job_id'], receipt['status'], receipt['release_ref']) ==
            (job, 'completed', release['original_releases'][job]), 'terminal identity differs')
    require(result['valid_endpoint'] and (result['attempts'], result['accepted'], result['rejected']) == (2048, 2048, 0),
            f'{job} has no zero-rejection fixed endpoint')
    verified(result['checkpoints']['2048'])
    ref = identity(summary)
    complete = read(LOCAL / job / 'completed.json')
    require(complete['summary_ref'] == ref, 'completion receipt differs')
    return ref


def next_attempt(job, slice_id=None):
    path = LOCAL / job / 'state.json'
    if path.is_file():
        state = read(path)
        if slice_id is None or state['slice'] == slice_id:
            return state['next_attempt']
    return LAST[job] + 1


def start(job, release_path):
    directory = LOCAL / job
    directory.mkdir(exist_ok=True)
    launched = sorted(int(p.stem.split('_')[1]) for p in directory.glob('slice_*_launch.json'))
    for previous in launched:
        require((directory / f'slice_{previous}_exit.json').is_file(), f'{job} previous launch has no exit receipt')
    slice_id = max(launched, default=0) + 1
    with (directory / 'stdout.log').open('ab', buffering=0) as stdout, (directory / 'stderr.log').open('ab', buffering=0) as stderr:
        stdout.write(f'\n[{utc()}] slice {slice_id}\n'.encode())
        stderr.write(f'\n[{utc()}] slice {slice_id}\n'.encode())
        command = [sys.executable, '-X', 'utf8', '-B', '-u', str(SOURCES / 'resume_fit.py'),
                   '--job', job, '--slice', str(slice_id), '--resume-release', str(release_path)]
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    launch = {'job': job, 'slice': slice_id, 'launcher_pid': process.pid, 'started_utc': utc()}
    write(directory / f'slice_{slice_id}_launch.json', launch)
    append(LOCAL / 'host_events.jsonl', {'event': 'worker_launch', **launch})
    return slice_id, process


def request_pause(job, slice_id, reason):
    pause = LOCAL / job / f'pause_{slice_id}.json'
    if not pause.exists():
        request = {'job': job, 'slice': slice_id, 'requested_utc': utc(), 'reason': reason}
        write(pause, request)
        append(LOCAL / 'host_events.jsonl', {'event': 'pause_requested', **request})


def observe_exit(job, slice_id, process):
    code = process.poll()
    if code is not None:
        exit_receipt = {'job': job, 'slice': slice_id, 'launcher_pid': process.pid, 'exit_code': code, 'utc': utc()}
        write(LOCAL / job / f'slice_{slice_id}_exit.json', exit_receipt)
        append(LOCAL / 'host_events.jsonl', {'event': 'worker_exit', **exit_receipt})
    return code


def supervise(release_path):
    import msvcrt

    release = release_inputs(release_path)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'host.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'1')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        return supervise_locked(release_path, release)


def supervise_locked(release_path, release):
    active, finished = {}, {}
    log = LOCAL / 'host_events.jsonl'
    prior_serial = {row['wave'] for row in map(json.loads, log.read_text(encoding='utf8').splitlines())
                    if row['event'] == 'serial_batch'} if log.exists() else set()
    start_ref = {'pid': os.getpid(), 'parent_pid': os.getppid(), 'started_utc': utc(), 'release_ref': identity(release_path)}
    write(LOCAL / 'host_started.json', start_ref)
    append(log, {'event': 'host_start', **start_ref})
    peak_commit, min_available, peak_physical_used = 0, 1., 0
    wave_number = 0
    observed = memory()

    def status(state, **other):
        write(LOCAL / 'host_status.json', {
            'state': state, 'utc': utc(), 'pid': os.getpid(), 'wave': wave_number,
            'active': {job: {'slice': number, 'launcher_pid': process.pid, 'next_attempt': next_attempt(job, number)}
                       for job, (number, process) in active.items()},
            'complete': finished, 'serial_waves': sorted(prior_serial), 'memory': observed,
            'peak_system_commit_bytes': peak_commit, 'min_physical_available_fraction': min_available,
            'peak_physical_used_bytes': peak_physical_used, **other,
        })

    try:
        for wave_number, jobs in enumerate(WAVES, 1):
            for job in jobs:
                result = endpoint(job, release)
                if result:
                    finished[job] = result
            next_sample = 0.
            while any(job not in finished for job in jobs):
                observed = memory()
                available = observed['physical_available_fraction']
                peak_commit = max(peak_commit, observed['system_commit_bytes'])
                min_available = min(min_available, available)
                peak_physical_used = max(peak_physical_used, observed['physical_total_bytes'] - observed['physical_available_bytes'])
                for job, (number, process) in list(active.items()):
                    code = observe_exit(job, number, process)
                    if code is None:
                        continue
                    del active[job]
                    if code == 0:
                        result = endpoint(job, release)
                        require(result is not None, f'{job} exited without a fixed endpoint')
                        finished[job] = result
                    else:
                        require(code == PAUSED_EXIT and (LOCAL / job / f'pause_{number}.json').is_file(),
                                f'{job} slice {number} exited with code {code}')
                pending = [job for job in jobs if job not in active and job not in finished]
                running = [next_attempt(job, number) for job, (number, _) in active.items()]
                eligible = running + [next_attempt(job) for job in pending]
                if wave_number not in prior_serial and memory_cut(eligible, available):
                    prior_serial.add(wave_number)
                    append(log, {'event': 'serial_batch', 'wave': wave_number, 'utc': utc(), **observed})
                    if len(active) == 2:
                        later = jobs[1]
                        request_pause(later, active[later][0], 'physical memory below 30% with both workers in stage 8s')
                slots = (1 if wave_number in prior_serial else 2) - len(active)
                for job in pending[:max(0, slots)]:
                    active[job] = start(job, release_path)
                if time.monotonic() >= next_sample:
                    append(LOCAL / 'memory.jsonl', {'utc': utc(), 'wave': wave_number, **observed})
                    status('running')
                    next_sample = time.monotonic() + 10.
                if active:
                    time.sleep(1.)
        status('completed', exit_code=0)
        return 0
    except BaseException as error:
        for job, (number, _) in active.items():
            request_pause(job, number, 'supervisor stopped after a peer or runtime failure')
        status('stopping', error=repr(error))
        for job, (number, process) in list(active.items()):
            process.wait()
            observe_exit(job, number, process)
            del active[job]
        status('failed', error=repr(error), exit_code=1)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--release', type=Path)
    mode.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        assert memory_cut((897, 960), .299) and not memory_cut((896, 897), .299)
        assert not memory_cut((897, 960), .30)
        print(json.dumps(memory(), sort_keys=True))
    else:
        sys.exit(supervise(args.release))
