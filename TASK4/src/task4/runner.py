"""Synchronous process supervisor. No background API and no automatic scientific decisions."""
import os, signal, subprocess, time, uuid, zipfile, json, hashlib
from pathlib import Path
import psutil
from .paths import ROOT, project_path
from .workflow import Workflow

def stop_tree(proc):
    try:
        parent=psutil.Process(proc.pid);children=parent.children(recursive=True)
        for c in children:
            try:c.terminate()
            except psutil.Error:pass
        try:parent.terminate()
        except psutil.Error:pass
        _,alive=psutil.wait_procs(children+[parent],timeout=5)
        for p in alive:
            try:p.kill()
            except psutil.Error:pass
    except psutil.NoSuchProcess:pass
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()

def snapshot_source(root,folder):
    """Freeze only this project's source before a child imports it."""
    files=[]
    for name in ('src','scripts','tests','configs'):
        files.extend(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'))
    files.extend(root/name for name in ('pyproject.toml','requirements.txt') if (root/name).is_file())
    hashes={}
    with zipfile.ZipFile(folder/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            content=path.read_bytes();name=path.relative_to(root).as_posix()
            archive.writestr(name,content);hashes[name]=hashlib.sha256(content).hexdigest()
    (folder/'source_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')

def run_managed(command,loop_id,cap_seconds,memory_gib=8,root=ROOT):
    root=Path(root).resolve();wf=Workflow(root);s=wf.load()
    if s['status']!='RUNNING' or s['active_loop']!=loop_id:raise PermissionError('No authorized active loop, or user approval pending')
    if s['active_jobs']:raise RuntimeError('A managed job is already active')
    loop=s['loops'][loop_id]
    left=min(loop['reserved_seconds']-loop['spent_seconds'],s['budget']['cap_seconds']-s['budget']['spent_seconds'])
    cap=min(float(cap_seconds),left)
    if cap<=5:raise RuntimeError('Insufficient registered budget')
    rid='run_'+uuid.uuid4().hex[:12];folder=project_path('logs/'+rid,root);folder.mkdir(parents=True)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    env['PYTHONPATH']=os.pathsep.join([str(root/'src'),str(root/'.runtime'),env.get('PYTHONPATH','')])
    env.setdefault('OMP_NUM_THREADS','1');env.setdefault('MKL_NUM_THREADS','1')
    env['TASK4_MANAGED_RUN_ID']=rid;env['TASK4_LOOP_ID']=loop_id
    # Child receives a save deadline; supervisor hard stop leaves several seconds to exit.
    env['TASK4_SAVE_DEADLINE_UNIX']=str(time.time()+max(1,cap-10))
    log=open(folder/'process.log','w',encoding='utf-8')
    proc=None;registered=False;t0=time.monotonic();reason='exit'
    try:
        snapshot_source(root,folder)
        proc=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=(os.name!='nt'))
        wf.process_started(loop_id,rid,proc.pid,command);registered=True
        while proc.poll() is None:
            elapsed=time.monotonic()-t0
            if elapsed>=cap:
                reason='time_cap';stop_tree(proc);break
            try:
                p=psutil.Process(proc.pid)
                rss=p.memory_info().rss+sum(c.memory_info().rss for c in p.children(recursive=True) if c.is_running())
                if rss>memory_gib*1024**3:reason='memory_cap';stop_tree(proc);break
            except psutil.Error:pass
            time.sleep(min(1,max(.05,cap-elapsed)))  # Cheap internal safety checks, not user-facing polling.
    except BaseException:
        reason='interrupted'
        if proc and proc.poll() is None:stop_tree(proc)
        raise
    finally:
        seconds=time.monotonic()-t0;log.close()
        if registered:wf.process_finished(rid,seconds,proc.returncode,reason)
    return {'run_id':rid,'seconds':seconds,'exit_code':proc.returncode,'reason':reason,'log':str(folder/'process.log')}
