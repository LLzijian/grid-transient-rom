"""Lightweight external Windows host; one baseline child at a time."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT)]
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io
RUN='osgkd_v2_sprint2b_baselines_20261126'
ORDER=('closed_passive_24111','neural_ode_24111')
LOCAL=ROOT/'artifacts'/RUN/'durable'


def terminal(job,path):
    summary=ROOT/'results'/RUN/job/'summary.json'
    if not summary.exists(): return None
    row=io.read(summary)
    io.require(row['release_ref']==io.identity(path),'terminal release differs')
    io.require(row['status'] in ('completed','scientific_stopped'),'nonterminal receipt')
    result=row['result']
    for key in ('terminal_checkpoint','attempts_ref','guards_ref'): io.verified(result[key])
    if row['status']=='completed':
        io.require(result['valid_endpoint'] and result['attempts']==2048 and result['accepted']>=2033,'invalid official endpoint')
        io.verified(result['checkpoints']['2048'])
    else: io.require(not result['valid_endpoint'] and result['stop_reason'] is not None,'unexplained scientific stop')
    return dict(summary_ref=io.identity(summary),status=row['status'],valid_endpoint=result['valid_endpoint'])


def prior_launches(directory):
    launches=[io.read(p) for p in directory.glob('slice_*_launch.json')]
    for launch in launches:
        io.require((directory/f"slice_{launch['slice']}_exit.json").is_file(),'previous native child exit not observed; inspect before restart')
    return launches


def latest_fit_exit(job,path):
    directory=LOCAL/job
    launches=[r for r in prior_launches(directory) if not r['qualification']]
    io.require(bool(launches),'predecessor has no native fit launch receipt')
    launch=max(launches,key=lambda r:r['slice'])
    io.require(launch['release_ref']==io.identity(path),'predecessor fit release differs')
    exit_path=directory/f"slice_{launch['slice']}_exit.json"
    receipt=io.read(exit_path)
    io.require(receipt['native_exit_code']==0 and receipt['child_pid']==launch['child_pid'],'predecessor native fit exit is not zero')
    return io.identity(exit_path)


def supervise(path,qualification=False):
    import msvcrt
    path=path.resolve();LOCAL.mkdir(parents=True,exist_ok=True)
    mode='qualify' if qualification else 'fit'
    log=LOCAL/f'{mode}_host_events.jsonl'
    finished={};active=None
    with (LOCAL/'host.lock').open('a+b') as lock:
        lock.write(b'1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            for job in ORDER:
                if not qualification and job==ORDER[1]:
                    previous=terminal(ORDER[0],path)
                    exit_ref=latest_fit_exit(ORDER[0],path)
                    if not previous or not previous['valid_endpoint']:
                        evidence=dict(state='successor_blocked',job=job,predecessor=previous,predecessor_native_exit_ref=exit_ref,reason='closed passive requires a valid official 2048 endpoint',utc=io.utc())
                        io.write(LOCAL/f'{mode}_host_status.json',evidence);io.append(log,evidence)
                        return 2
                directory=LOCAL/job;directory.mkdir(parents=True,exist_ok=True)
                launches=prior_launches(directory)
                if not qualification:
                    previous=terminal(job,path)
                    if previous: finished[job]=previous;continue
                else:
                    qpath=ROOT/'results'/RUN/job/'qualification.json'
                    if qpath.exists():
                        q=io.read(qpath);io.require(q['release_ref']==io.identity(path),'previous qualification release differs')
                        finished[job]=dict(summary_ref=io.identity(qpath),passed=q['result']['passed']);continue
                io.require(not (directory/'writer.lock').exists(),'orphan writer claim requires inspection')
                number=max((r['slice'] for r in launches),default=0)+1
                command=[str(ROOT.parent/'.venv/Scripts/python.exe'),'-X','utf8','-B','-u',str(Path(__file__).with_name('fit.py')),'--release',str(path),'--job',job,'--slice',str(number)]
                if qualification:command.append('--qualify')
                with (directory/f'slice_{number}_stdout.log').open('ab',buffering=0) as stdout,(directory/f'slice_{number}_stderr.log').open('ab',buffering=0) as stderr:
                    process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                active=(job,number,process)
                launch=dict(slice=number,job=job,child_pid=process.pid,host_pid=os.getpid(),utc=io.utc(),release_ref=io.identity(path),qualification=qualification)
                io.write(directory/f'slice_{number}_launch.json',launch);io.append(log,dict(event='child_launch',**launch))
                next_sample=0.
                while process.poll() is None:
                    if time.monotonic()>=next_sample:
                        step=io.read(directory/'state.json') if (directory/'state.json').exists() else None
                        sample=dict(state='running',job=job,slice=number,child_pid=process.pid,step=step,native_exit_code=None,finished=finished,memory=io.memory(),utc=io.utc())
                        io.write(LOCAL/f'{mode}_host_status.json',sample);io.append(LOCAL/f'{mode}_memory.jsonl',sample);next_sample=time.monotonic()+10.
                    time.sleep(1.)
                code=process.returncode
                exit_row=dict(job=job,slice=number,child_pid=process.pid,native_exit_code=code,utc=io.utc())
                io.write(directory/f'slice_{number}_exit.json',exit_row);io.append(log,dict(event='child_exit',**exit_row));active=None
                io.require(code==0,f'{job} child failed with native exit {code}; no successor launched')
                if qualification:
                    qpath=ROOT/'results'/RUN/job/'qualification.json';q=io.read(qpath)
                    io.require(q['release_ref']==io.identity(path),'qualification release differs')
                    finished[job]=dict(summary_ref=io.identity(qpath),passed=q['result']['passed'])
                else:
                    finished[job]=terminal(job,path);io.require(finished[job] is not None,'child exited without terminal summary')
            io.write(LOCAL/f'{mode}_host_status.json',dict(state='completed',finished=finished,utc=io.utc()))
            return 0
        except BaseException as error:
            if active:
                job,number,process=active
                if process.poll() is None and not qualification:
                    pause=LOCAL/job/f'pause_{number}.json'
                    io.write(pause,dict(reason='host exception; seal and pause before host exit',utc=io.utc()))
                io.write(LOCAL/f'{mode}_host_status.json',dict(state='stopping',error=repr(error),job=job,slice=number,utc=io.utc()))
                process.wait()
                exit_row=dict(job=job,slice=number,child_pid=process.pid,native_exit_code=process.returncode,utc=io.utc())
                io.write(LOCAL/job/f'slice_{number}_exit.json',exit_row);io.append(log,dict(event='child_exit_after_host_error',**exit_row))
            io.write(LOCAL/f'{mode}_host_status.json',dict(state='failed',error=repr(error),utc=io.utc()));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True);parser.add_argument('--qualify',action='store_true')
    args=parser.parse_args();sys.exit(supervise(args.release,args.qualify))
