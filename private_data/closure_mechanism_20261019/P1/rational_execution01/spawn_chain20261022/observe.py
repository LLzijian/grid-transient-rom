"""Exactly five registered native cases, then qualify one new runtime selection."""
import argparse,ctypes,json,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import python_command,validate_selection
BASE=HERE.parent;PROBE=BASE/'reconstruction20261022/probe_worker.py'
AUTH=ROOT/'results/rational_execution_20261021/spawn_chain_authority_20261022.json'
AUTH_SHA='fcef6abc075be703711ff5cc0b88967b34a7c65ed93f910488f0006ec5c3db68'
CASES=('normal','grandchild','pipes','orphan','supervisor_exit')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def name(path):
    path=Path(path).resolve()
    try:return path.relative_to(ROOT).as_posix()
    except ValueError:return str(path)
def ref(path):return {'path':name(path),'sha256':job.sha(path)}
def witness(target,stop):
    handle=None;k=job.api()
    try:
        while not (target/'child.json').is_file():
            if stop.wait(.01):return
            if (target/'worker.json').is_file() and job.identity_state(read(target/'worker.json'))!='RUNNING':
                raise ValueError('supervisor exited before child handshake')
        child=read(target/'child.json')
        handle=job.checked(k.OpenProcess(0x1000|0x100000,False,child['pid']))
        identity=job.times(k,handle)
        if identity['creation_filetime_utc']!=child['creation_filetime_utc'] or k.WaitForSingleObject(handle,0)!=258:
            raise ValueError('witness requires exact live handshaken child')
        job.atomic(target/'witness_ready.json',{'pid':child['pid'],'creation_filetime_utc':child['creation_filetime_utc'],
            'wait_result':258,'physical_image':job.physical_image(k,handle)})
        waited=int(k.WaitForSingleObject(handle,0xffffffff)) # Native INFINITE, no elapsed cap.
        code=job.W.DWORD();job.checked(k.GetExitCodeProcess(handle,ctypes.byref(code)))
        job.atomic(target/'handle_end.json',{'pid':child['pid'],**job.times(k,handle),'wait_result':waited,'exit_code':int(code.value)})
    except BaseException as error:job.atomic(target/'witness_error.json',{'error':repr(error)})
    finally:
        if handle:k.CloseHandle(handle)
def check_case(row,commit,files,command):
    proof=read(ROOT/row['receipt']['path']);case=row['name'];expected=121 if case=='supervisor_exit' else 0
    assert proof['source_commit']==commit and proof['phase']=='runtime_observation' and proof['group']==case
    assert proof['command']==command and proof['source_before']==proof['source_after']==files
    assert proof['status']==('FAILED' if case in ('orphan','supervisor_exit') else 'COMPLETED')
    assert proof['cleanup_verified'] is True and type(proof['job_active_processes']) is int and proof['job_active_processes']==0
    assert all(type(proof[key]) is int and proof[key]==expected for key in ('exit_code','launcher_exit_code','worker_exit_code'))
    assert proof['actual_worker_state']=='GONE' and type(proof['orphans_cleaned']) is bool
    assert proof['orphans_cleaned'] is (case=='orphan')
    identities=[read(ROOT/value['path']) for value in row['identities']]
    assert len(identities)==(3 if case=='grandchild' else 2)
    for identity in identities:
        assert type(identity['pid']) is int and type(identity['creation_filetime_utc']) is int
        assert identity['pid']>0 and identity['creation_filetime_utc']>0 and job.identity_state(identity)=='GONE'
        assert Path(identity['executable']).resolve()==Path(command['logical_executable'])
    assert [(v['pid'],v['creation_filetime_utc']) for v in identities][:1]==[(proof['worker']['pid'],proof['worker']['creation_filetime_utc'])]
    records=[value for trace in row['spawns'] for value in read(ROOT/trace['path'])]
    assert {v['pid'] for v in identities[1:]}=={v['child_pid'] for v in records}
    for value in records:
        assert value['via']==('native_job' if case=='supervisor_exit' else 'subprocess')
        assert type(value['creationflags']) is int and value['creationflags']==(525324 if case=='supervisor_exit' else 8)
        assert value['executable']==command['executable'] and value['logical_executable']==command['logical_executable']
    if case=='orphan':
        live={(v['pid'],v.get('creation_filetime_utc')) for v in proof['active_members_observed'] if v.get('in_owned_job') is True and v.get('wait_result')==258}
        assert all((v['pid'],v['creation_filetime_utc']) in live for v in identities[1:])
    if case=='supervisor_exit':
        ends=[read(ROOT/value['path']) for value in row['handle_ends']]
        assert len(ends)==1
        assert ends[0]['pid']==identities[1]['pid'] and ends[0]['creation_filetime_utc']==identities[1]['creation_filetime_utc']
        assert type(ends[0]['wait_result']) is int and ends[0]['wait_result']==0 and type(ends[0]['exit_code']) is int
    if case=='pipes':
        content=read(HERE/case/'pipes.json')
        assert content=={'stdout':'fixed stdout\n','stderr':'fixed stderr\n','returncode':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);a=p.parse_args()
    if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('full real source commit required')
    if job.sha(AUTH)!=AUTH_SHA or read(AUTH).get('human_answer')!='允许修复、验收并继续训练':raise ValueError('exact actual human authority required')
    if (HERE/'batch.json').exists() or (HERE/'active.lock').exists():raise ValueError('fixed batch already started; no repeat')
    from experiments import rational_execution_20261021 as execution
    old=read(BASE/'bindings/qa05.json')
    sources={ROOT/path for path in old['files'] if path.endswith('.py')}
    sources.update(ROOT/path for path in execution.qa06_source_names())
    files={str(path.resolve()):job.sha(path) for path in sorted(sources)}
    job.atomic(HERE/'sources.json',files)
    batch={'status':'RUNNING','source_commit':a.source_commit,'authority':ref(AUTH),'cases':[]}
    job.atomic(HERE/'batch.json',batch)
    try:
        for case in CASES:
            target=HERE/case;target.mkdir(exist_ok=False)
            args=[str(PROBE),'--role',case,'--scheme','direct_base_venv','--out',str(target),'--source-commit',a.source_commit]
            command=python_command(ROOT/'.venv-gpu/Scripts/python.exe',args,'direct_base_venv')
            stop=threading.Event();thread=None
            if case=='supervisor_exit':
                thread=threading.Thread(target=witness,args=(target,stop),daemon=False);thread.start()
            def owned(worker):
                (target/'owned').write_text('owned',encoding='utf-8')
                if thread:
                    # Prove inner Job-close termination on a retained process handle
                    # before the outer Job can perform any cleanup of its own.
                    thread.join()
                    if (target/'witness_error.json').is_file():raise RuntimeError((target/'witness_error.json').read_text(encoding='utf-8'))
            try:
                job.supervise(command,HERE/('native_'+case),HERE/'active.lock',files,ROOT,
                    worker_receipt=target/'worker.json',on_worker_owned=owned,
                    provenance={'source_commit':a.source_commit,'phase':'runtime_observation','group':case})
            finally:
                stop.set()
                if thread:thread.join()
            identities=[ref(target/'worker.json'),ref(target/'child.json')]
            spawns=[ref(target/'spawns_parent.json')]
            if case=='grandchild':identities.append(ref(target/'grandchild.json'));spawns.append(ref(target/'spawns_child.json'))
            row={'name':case,'receipt':ref(HERE/('native_'+case)/'receipt.json'),'identities':identities,
                'spawns':spawns,'handle_ends':[ref(target/'handle_end.json')] if case=='supervisor_exit' else []}
            if case=='pipes':row['pipe_result']=ref(target/'pipes.json')
            batch['cases'].append(row);job.atomic(HERE/'batch.json',batch)
            check_case(row,a.source_commit,files,command)
        batch['status']='QUALIFIED';job.atomic(HERE/'batch.json',batch)
        baseline=BASE/'reconstruction20261022/reference1/normal/metadata.json'
        candidate=HERE/'normal/metadata.json'
        parity={'baseline':ref(baseline),'candidate':ref(candidate),
            'child_identity':ref(HERE/'orphan/child.json'),'normal_child_identity':ref(HERE/'normal/child.json')}
        job.atomic(HERE/'parity.json',parity)
        low=python_command(ROOT/'.venv-gpu/Scripts/python.exe',[],'direct_base_venv')
        selection={'status':'SELECTED','scheme':'direct_base_venv','observation_attempt':1,
            'observation_source_commit':a.source_commit,**{key:low[key] for key in ('logical_executable','executable','expected_worker_image')},
            'normal_receipt':batch['cases'][0]['receipt'],'orphan_receipt':batch['cases'][3]['receipt'],
            'baseline_reference':ref(baseline),'parity_evidence':ref(HERE/'parity.json')}
        dependencies={Path(path) for path in files}
        dependencies.update([AUTH,HERE/'batch.json',baseline,candidate,HERE/'parity.json',
            Path(low['logical_executable']),Path(low['executable']),Path(low['expected_worker_image']),
            ROOT/'.venv-gpu/pyvenv.cfg',Path(read(baseline)['python_dll']['path'])])
        for row in batch['cases']:
            dependencies.add(ROOT/row['receipt']['path'])
            for key in ('identities','spawns','handle_ends'):dependencies.update(ROOT/value['path'] for value in row[key])
            if row['name']=='pipes':dependencies.add(ROOT/row['pipe_result']['path'])
        selection['files']={name(path):job.sha(path) for path in sorted(dependencies)}
        job.atomic(HERE/'runtime_selection.json',selection)
        validate_selection(HERE/'runtime_selection.json')
        job.atomic(HERE/'qualification.json',{'status':'PASS','source_commit':a.source_commit,'batch':ref(HERE/'batch.json'),
            'selection':ref(HERE/'runtime_selection.json'),'cases':5,'wall_seconds_sum':sum(read(ROOT/row['receipt']['path'])['wall_seconds'] for row in batch['cases']),
            'job_cpu_user_seconds_sum':sum(read(ROOT/row['receipt']['path'])['job_cpu_user_seconds'] for row in batch['cases']),
            'job_cpu_kernel_seconds_sum':sum(read(ROOT/row['receipt']['path'])['job_cpu_kernel_seconds'] for row in batch['cases'])})
        print(json.dumps(read(HERE/'qualification.json')))
    except BaseException as error:
        batch.update(status='FAILED',error=repr(error));job.atomic(HERE/'batch.json',batch)
        raise

if __name__=='__main__':main()
