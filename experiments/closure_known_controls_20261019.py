"""Run the registered P1 known-support controls sequentially within one cap.

This is the fitting-adequacy substage, not a completed structure search. Parent
wall time includes children once. Failed/partial attempts remain immutable.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(temp,path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args()
    if not 0<a.max_seconds<=28000:p.error('known-control cap must be <=28000 seconds')
    start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).with_name('closure_causal_recovery_20261019.py')
    source_sha=hashlib.sha256(script.read_bytes()).hexdigest()
    state_path=a.output/'batch.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'jobs':[],'source_sha256':source_sha,'prior_seconds':0.}
    if state['source_sha256']!=source_sha:raise ValueError('cannot mix control code versions')
    prior=state.get('elapsed_total_seconds',0.)
    if prior>=a.max_seconds:raise ValueError('cumulative batch cap exhausted')
    state.update(status='RUNNING',pid=os.getpid(),child_pid=None,scope='known support only; no search success claim')
    write(state_path,state)
    try:
        for case in ('node','edge','saturation'):
            for noise in (0.,.01):
                for seed in (5101,5102,5103):
                    key=f'{case}_noise{noise:g}_seed{seed}'
                    if any(r['key']==key and r['status']=='complete' for r in state['jobs']):continue
                    remaining=a.max_seconds-prior-(time.perf_counter()-start)
                    if remaining<180:raise TimeoutError('insufficient remaining time to launch another fit')
                    folder=a.output/key
                    command=[sys.executable,str(script),'run','--case',case,'--seed',str(seed),'--noise',str(noise),
                        '--updates','512','--known-support-only','--output',str(folder),'--max-seconds',str(remaining-30)]
                    if folder.exists():command.append('--restart')
                    log=a.output/f'{key}_{len(state["jobs"])}.log'
                    tick=time.perf_counter()
                    with log.open('w',encoding='utf-8') as stream:
                        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                        state.update(child_pid=child.pid,current=key,command=command);write(state_path,state)
                        try:code=child.wait(timeout=remaining-10)
                        except BaseException:
                            child.terminate();child.wait(timeout=10);raise
                    state['jobs'].append({'key':key,'status':'complete' if code==0 else 'failed',
                        'exit_code':code,'seconds':time.perf_counter()-tick,'log':str(log)})
                    state.update(child_pid=None,elapsed_total_seconds=prior+time.perf_counter()-start)
                    write(state_path,state)
        state['status']='COMPLETED' if all(r['status']=='complete' for r in state['jobs']) else 'COMPLETED_WITH_FAILURES'
    except BaseException as error:
        state.update(status='INTERRUPTED',error=repr(error));raise
    finally:
        state.update(elapsed_total_seconds=prior+time.perf_counter()-start,child_pid=None)
        write(state_path,state)
        write(a.output/'receipt.json',{'stage':'P1','seconds':state['elapsed_total_seconds'],
            'includes_children':True,'status':state['status'],'pid':os.getpid()})

if __name__=='__main__':main()
