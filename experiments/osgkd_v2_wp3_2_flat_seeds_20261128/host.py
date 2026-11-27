"""One Windows child at a time; never silently restart an unobserved worker."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from experiments.osgkd_v2_wp3_2_flat_seeds_20261128 import fit

io, RUN, ORDER, LOCAL, PUBLIC = fit.io, fit.RUN, fit.ORDER, fit.LOCAL, fit.PUBLIC


def prior_launches(directory):
    launches = [io.read(path) for path in directory.glob('slice_*_launch.json')]
    for launch in launches:
        path = directory/f"slice_{launch['slice']}_exit.json"
        io.require(path.is_file(), 'previous native child exit unobserved; inspect before restart')
        receipt = io.read(path)
        io.require(receipt['child_pid'] == launch['child_pid'] and receipt['slice'] == launch['slice']
                   and receipt['job'] == launch['job'] == directory.name,
                   'native child exit identity differs')
    return launches


def latest_exit(job, path, qualification=False):
    directory = LOCAL/job
    launches = [row for row in prior_launches(directory) if row['qualification'] == qualification]
    io.require(launches, 'no matching native child launch')
    launch = max(launches, key=lambda row: row['slice'])
    io.require(launch['release_ref'] == io.identity(path), 'native child release differs')
    ref = io.identity(directory/f"slice_{launch['slice']}_exit.json")
    receipt = io.read(io.verified(ref))
    io.require(receipt['native_exit_code'] == 0, 'native child exit is not zero')
    return ref


def terminal(job, path):
    summary = PUBLIC/job/'summary.json'
    if not summary.exists():
        return None
    row = io.read(summary)
    io.require(row['run_id'] == RUN and row['job_id'] == job and row['release_ref'] == io.identity(path),
               'terminal identity differs')
    io.require(row['status'] in ('completed', 'scientific_stopped'), 'nonterminal receipt')
    result = row['result']
    for key in ('terminal_checkpoint', 'attempts_ref', 'guards_ref', 'plan_ref'):
        io.verified(result[key])
    if row['status'] == 'completed':
        io.require(result['valid_endpoint'] and result['attempts'] == 2048 and result['accepted'] >= 2033,
                   'invalid fixed endpoint')
        io.verified(result['checkpoints']['2048'])
    else:
        io.require(not result['valid_endpoint'] and result['stop_reason'] is not None, 'unexplained stop')
    return dict(summary_ref=io.identity(summary), status=row['status'], valid_endpoint=result['valid_endpoint'],
                stop_reason=result.get('stop_reason'))


def assert_no_other_worker():
    """Check current repository scientific entrypoints before each child launch."""
    io.require(os.name == 'nt', 'native Windows host required')
    script = r'''
$ErrorActionPreference='Stop'
$rootPath=$env:KIKI_EXCLUSIVE_ROOT.ToLowerInvariant().Replace([char]92,[char]47)
$ownPid=[int]$env:KIKI_EXCLUSIVE_PID
$parentPid=[int]$env:KIKI_EXCLUSIVE_PARENT_PID
$processes=@(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'")
$self=@($processes | Where-Object ProcessId -eq $ownPid)
$parent=@($processes | Where-Object ProcessId -eq $parentPid)
$excludeParent=$false
if ($self.Count -eq 1 -and $parent.Count -eq 1 -and $self[0].ParentProcessId -eq $parentPid -and $parent[0].ExecutablePath) {
    $sameLauncher=[string]::Equals([IO.Path]::GetFullPath($parent[0].ExecutablePath),[IO.Path]::GetFullPath($env:KIKI_SHARED_PYTHON),[StringComparison]::OrdinalIgnoreCase)
    $ownMatch=[regex]::Match($self[0].CommandLine,'(?s)^\s*(?:"[^"]+"|\S+)(\s+.+)$')
    $parentMatch=[regex]::Match($parent[0].CommandLine,'(?s)^\s*(?:"[^"]+"|\S+)(\s+.+)$')
    $excludeParent=$sameLauncher -and $ownMatch.Success -and $parentMatch.Success -and ($ownMatch.Groups[1].Value -ceq $parentMatch.Groups[1].Value)
}
$ids=@($processes | Where-Object {
    $line=if ($_.CommandLine) {$_.CommandLine.ToLowerInvariant().Replace([char]92,[char]47)} else {''}
    $_.ProcessId -ne $ownPid -and -not ($excludeParent -and $_.ProcessId -eq $parentPid) -and
    ($line.Contains($rootPath) -or $line -match '(^|[" ])(?:\./)?experiments/') -and
    $line -match 'experiments/.*/(?:fit|host|run|resume_fit|probe|train)[^ /" ]*\.py'
} | ForEach-Object {[int]$_.ProcessId})
ConvertTo-Json -InputObject $ids -Compress
'''
    environment = dict(os.environ, KIKI_EXCLUSIVE_ROOT=str(ROOT), KIKI_EXCLUSIVE_PID=str(os.getpid()),
                       KIKI_EXCLUSIVE_PARENT_PID=str(os.getppid()),
                       KIKI_SHARED_PYTHON=str((ROOT.parent/'.venv/Scripts/python.exe').resolve()))
    shell = str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe')
    result = subprocess.run([shell, '-NoLogo', '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden',
                             '-Command', script], capture_output=True, text=True, check=True, env=environment,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    ids = json.loads(result.stdout.strip())
    io.require(isinstance(ids, list) and not ids, 'another repository qualification/training worker is live: '+str(ids))


def score_ready(path):
    release = io.read(path)
    io.require(path.resolve() == PUBLIC/'evaluate_release.json' and release['run_id'] == RUN
               and release['released'] is True, 'canonical released score authority required')
    fit_path = io.verified(release['fit_release'])
    registered, _ = fit.validate(fit_path)
    io.require(release['source_hashes'] == registered['source_hashes'] and set(release['fit_refs']) == set(ORDER),
               'frozen score sources or endpoint set differ')
    training = io.read(io.verified(release['training_summary']))
    io.require(training['run_id'] == RUN and training['state'] == 'completed'
               and training['fit_release'] == release['fit_release']
               and training['serial_order'] == list(ORDER) and set(training['jobs']) == set(ORDER),
               'typed training summary differs')
    native = io.read(io.verified(training['native_host_exit']))
    io.require(native['mode'] == 'fit' and native['native_exit_code'] == 0
               and native['release_ref']['sha256'] == release['fit_release']['sha256'], 'host native exit differs')
    receipts = {}
    for job in ORDER:
        fit.assert_no_scientific_stop(fit_path, job)
        endpoint = terminal(job, fit_path)
        io.require(endpoint is not None and endpoint['valid_endpoint'], 'two effective fixed endpoints required')
        io.require(endpoint['summary_ref'] == release['fit_refs'][job], 'score endpoint binding differs')
        receipts[job] = dict(**endpoint, native_exit_ref=latest_exit(job, fit_path))
        io.require(training['jobs'][job] == receipts[job], 'training summary/native job differs')
    return receipts


def supervise(path, qualification=False):
    import msvcrt
    path = path.resolve()
    fit.validate(path, qualification=qualification)
    LOCAL.mkdir(parents=True, exist_ok=True)
    mode = 'qualify' if qualification else 'fit'
    log = LOCAL/f'{mode}_host_events.jsonl'
    finished, active = {}, None
    with (LOCAL/'host.lock').open('a+b') as lock:
        lock.write(b'1')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            for job in ORDER:
                if not qualification:
                    fit.assert_no_scientific_stop(path, job)
                if not qualification and job == ORDER[1]:
                    predecessor = terminal(ORDER[0], path)
                    exit_ref = latest_exit(ORDER[0], path)
                    if predecessor is None or not predecessor['valid_endpoint']:
                        evidence = dict(state='successor_blocked', job=job, predecessor=predecessor,
                                        predecessor_native_exit_ref=exit_ref, reason='first flat has no effective 2048 endpoint', utc=io.utc())
                        io.write(LOCAL/f'{mode}_host_status.json', evidence)
                        io.append(log, evidence)
                        return 2
                directory = LOCAL/job
                directory.mkdir(parents=True, exist_ok=True)
                launches = prior_launches(directory)
                if qualification:
                    qpath = PUBLIC/job/'qualification.json'
                    if qpath.exists():
                        q = io.read(qpath)
                        io.require(q['release_ref'] == io.identity(path), 'existing qualification authority differs')
                        finished[job] = dict(summary_ref=io.identity(qpath), passed=q['result']['passed'],
                                             native_exit_ref=latest_exit(job, path, True))
                        continue
                else:
                    endpoint = terminal(job, path)
                    if endpoint:
                        finished[job] = dict(**endpoint, native_exit_ref=latest_exit(job, path))
                        continue
                io.require(not (directory/'writer.lock').exists(), 'orphan writer claim requires inspection')
                assert_no_other_worker()
                number = max((row['slice'] for row in launches), default=0)+1
                command = [str(ROOT.parent/'.venv/Scripts/python.exe'), '-X', 'utf8', '-B', '-u',
                           str(Path(__file__).with_name('fit.py')), '--release', str(path), '--job', job, '--slice', str(number)]
                if qualification:
                    command.append('--qualify')
                with (directory/f'slice_{number}_stdout.log').open('ab', buffering=0) as stdout, (directory/f'slice_{number}_stderr.log').open('ab', buffering=0) as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                               creationflags=subprocess.CREATE_NO_WINDOW)
                active = (job, number, process)
                launch = dict(slice=number, job=job, child_pid=process.pid, host_pid=os.getpid(), utc=io.utc(),
                              release_ref=io.identity(path), qualification=qualification)
                io.write(directory/f'slice_{number}_launch.json', launch)
                io.append(log, dict(event='child_launch', **launch))
                next_sample = 0.
                while process.poll() is None:
                    if time.monotonic() >= next_sample:
                        step = io.read(directory/'state.json') if (directory/'state.json').exists() else None
                        sample = dict(state='running', job=job, slice=number, child_pid=process.pid, step=step,
                                      finished=finished, memory=io.memory(), utc=io.utc())
                        io.write(LOCAL/f'{mode}_host_status.json', sample)
                        io.append(LOCAL/f'{mode}_memory.jsonl', sample)
                        next_sample = time.monotonic()+10.
                    time.sleep(1.)
                code = process.returncode
                exit_row = dict(job=job, slice=number, child_pid=process.pid, native_exit_code=code, utc=io.utc())
                io.write(directory/f'slice_{number}_exit.json', exit_row)
                io.append(log, dict(event='child_exit', **exit_row))
                active = None
                io.require(code == 0, f'{job} child exit {code}; successor not launched')
                if qualification:
                    qpath = PUBLIC/job/'qualification.json'
                    q = io.read(qpath)
                    io.require(q['release_ref'] == io.identity(path), 'qualification authority differs')
                    finished[job] = dict(summary_ref=io.identity(qpath), passed=q['result']['passed'],
                                         native_exit_ref=latest_exit(job, path, True))
                else:
                    endpoint = terminal(job, path)
                    io.require(endpoint is not None, 'native exit without terminal summary')
                    finished[job] = dict(**endpoint, native_exit_ref=latest_exit(job, path))
            valid = qualification or all(row['valid_endpoint'] for row in finished.values())
            io.write(LOCAL/f'{mode}_host_status.json', dict(run_id=RUN,
                     state='completed' if valid else 'scientific_stopped', release_ref=io.identity(path),
                     fit_release=io.identity(path) if not qualification else None,
                     serial_order=list(ORDER), jobs=finished, finished=finished, utc=io.utc()))
            return 0 if valid else 2
        except BaseException as error:
            if active:
                job, number, process = active
                if process.poll() is None and not qualification:
                    io.write(LOCAL/job/f'pause_{number}.json', dict(reason='host exception; seal and pause', utc=io.utc()))
                process.wait()
                exit_row = dict(job=job, slice=number, child_pid=process.pid,
                                native_exit_code=process.returncode, utc=io.utc())
                io.write(LOCAL/job/f'slice_{number}_exit.json', exit_row)
                io.append(log, dict(event='child_exit_after_host_error', **exit_row))
            io.write(LOCAL/f'{mode}_host_status.json', dict(state='failed', error=repr(error), utc=io.utc()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--score-gate', action='store_true')
    args = parser.parse_args()
    if args.score_gate:
        print(json.dumps(score_ready(args.release.resolve())))
    else:
        sys.exit(supervise(args.release, args.qualify))
