"""Finite engineering phases on the existing uncapped Windows Job runtime."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'src')]
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import selected_python_command,command_environment
SELECTION=ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01/spawn_chain20261022/runtime_selection.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return {'path':str(Path(p).resolve()),'sha256':job.sha(p)}
def check(binding,commit):
    if binding.get('source_commit')!=commit or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit) or binding.get('no_process_quotas') is not True:raise ValueError('registered uncapped source required')
    job.verify_files(binding['files']);selected_python_command([],SELECTION)
    native=read(ROOT/binding['old_native']['path']);complete=read(ROOT/binding['old_complete']['path'])
    if (native.get('status')!='COMPLETED' or native.get('source_commit')!='ebd45248a25905dbf5af4a3cea8c4073a02994db'
        or not native.get('source_before') or native['source_before']!=native.get('source_after')
        or native.get('cleanup_verified') is not True or native.get('actual_worker_state')!='GONE'
        or native.get('orphans_cleaned') is not False or type(native.get('job_active_processes')) is not int or native['job_active_processes']!=0
        or any(type(native.get(k)) is not int or native[k]!=0 for k in ('exit_code','worker_exit_code','launcher_exit_code'))
        or complete.get('actual_updates')!=24576 or complete.get('all_twelve_frozen') is not True):raise ValueError('old clean science completion required')
    for name,digest in native['source_before'].items():
        if binding['files'].get(str((ROOT/name).resolve()))!=digest:raise ValueError('missing old closed source')
    names=['experiments/closure_training_efficiency_20261024.py','tests/test_closure_training_efficiency_20261024.py',
        str((BASE/'launch.py').relative_to(ROOT)),str((BASE/'make_binding.py').relative_to(ROOT)),
        'reports/training_efficiency_20261024_protocol.md','reports/astra6_training_efficiency_20261024_raw.txt','reports/training_efficiency_20261024_review.md']
    for name in names:
        if binding['files'].get(str((ROOT/name).resolve()))!=job.sha(ROOT/name):raise ValueError('missing new frozen source')
def validate_qa(binding,commit):
    native=read(BASE/'native_qa/receipt.json');result=read(BASE/'qa/complete.json')
    if (native.get('status')!='COMPLETED' or native.get('source_commit')!=commit or native.get('phase')!='training_efficiency_qa'
        or native.get('cleanup_verified') is not True or native.get('orphans_cleaned') is not False
        or native.get('actual_worker_state')!='GONE' or type(native.get('job_active_processes')) is not int or native['job_active_processes']!=0
        or not native.get('source_before') or native['source_before']!=native.get('source_after')
        or any(type(native.get(k)) is not int or native[k]!=0 for k in ('exit_code','worker_exit_code','launcher_exit_code'))
        or result.get('source_commit')!=commit or result.get('status')!='PASS' or type(result.get('pytest_returncode')) is not int or result['pytest_returncode']!=0):raise ValueError('actual same-source CPU negative QA required')
    for name,digest in binding['files'].items():
        if native['source_before'].get(name)!=digest:raise ValueError('QA source changed')
def worker(phase,binding,commit):
    check(binding,commit);out=BASE/phase;out.mkdir(exist_ok=False)
    k=job.api();command=selected_python_command([],SELECTION)
    job.atomic(out/'worker.json',{**job.identity(),'physical_image':job.physical_image(k,k.GetCurrentProcess()),
        'logical_executable':command['logical_executable'],'runtime_scheme':command['runtime_scheme'],
        'runtime_selection':command['runtime_selection'],'source_commit':commit,'phase':'training_efficiency_'+phase})
    while not (out/'owned').is_file():time.sleep(.01)
    try:
        if phase=='qa':
            temp=ROOT/'private_data/qte1'
            if temp.exists():raise ValueError('preserve existing QA temp')
            cmd=selected_python_command(['-m','pytest','-q',str(ROOT/'tests/test_closure_training_efficiency_20261024.py'),'--basetemp',str(temp)],SELECTION)
            proc=subprocess.run([cmd['executable'],*cmd['arguments']],cwd=ROOT,env=command_environment(cmd),capture_output=True,text=True,encoding='utf-8',creationflags=subprocess.DETACHED_PROCESS)
            (out/'stdout.txt').write_text(proc.stdout,encoding='utf-8');(out/'stderr.txt').write_text(proc.stderr,encoding='utf-8')
            job.atomic(out/'complete.json',{'status':'PASS' if proc.returncode==0 else 'FAIL','source_commit':commit,'pytest_returncode':proc.returncode,'planned_cpu_optimizer_updates':2,'gpu_optimizer_updates':0})
            return 0 if proc.returncode==0 else 2
        validate_qa(binding,commit)
        import torch
        torch.set_num_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('registered GPU unavailable')
        from experiments import closure_training_efficiency_20261024 as study
        actual=dict(binding)
        if phase=='opt1':
            admitted=read(BASE/'opt1_inputs.json')
            if admitted['source_commit']!=commit:raise ValueError('reference source differs')
            job.verify_files(admitted['files']);actual['reference_result']=admitted['reference_result']
        result=study.benchmark(out/'benchmark',actual,commit,phase)
        job.atomic(out/'complete.json',{'status':'COMPLETED','source_commit':commit,'result':ref(out/'benchmark/result.json'),'ledger':result['ledger']})
        return 0
    except BaseException as error:
        import traceback
        job.atomic(out/'failure.json',{'status':'UNRESOLVED','error':repr(error),'traceback':traceback.format_exc()});raise
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('qa','reference','opt1'));p.add_argument('--source-commit',required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    binding=read(BASE/'binding.json');check(binding,a.source_commit)
    if a.worker:return worker(a.phase,binding,a.source_commit)
    if (BASE/a.phase).exists() or (BASE/('native_'+a.phase)).exists():raise ValueError('sole phase already exists; preserve evidence')
    for path in [BASE/'active.lock',*(ROOT/'private_data/closure_mechanism_20261019/P1').glob('*/active.lock'),ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01/native_active.lock']:
        if path.exists():raise ValueError('existing job lock; no duplicate')
    files=dict(binding['files']);files[str((BASE/'binding.json').resolve())]=job.sha(BASE/'binding.json')
    if a.phase!='qa':
        validate_qa(binding,a.source_commit)
        for path in (BASE/'native_qa/receipt.json',BASE/'qa/complete.json'):files[str(path.resolve())]=job.sha(path)
    if a.phase=='opt1':
        result_path=BASE/'reference/benchmark/result.json';result=read(result_path);native=read(BASE/'native_reference/receipt.json')
        if result.get('status')!='PASS' or result.get('source_commit')!=a.source_commit or native.get('status')!='COMPLETED' or native.get('source_commit')!=a.source_commit or native['source_before']!=native.get('source_after'):raise ValueError('successful same-source reference required')
        reference_files={str(result_path.resolve()):job.sha(result_path),str((BASE/'native_reference/receipt.json').resolve()):job.sha(BASE/'native_reference/receipt.json')}
        for row in result['rows']:
            path=ROOT/row['final_state']['path']
            if job.sha(path)!=row['final_state']['sha256']:raise ValueError('reference state changed')
            reference_files[str(path.resolve())]=row['final_state']['sha256']
        admitted={'source_commit':a.source_commit,'files':reference_files,'reference_result':ref(result_path)}
        with (BASE/'opt1_inputs.json').open('x',encoding='utf-8') as f:json.dump(admitted,f,indent=2,sort_keys=True)
        files.update(reference_files);files[str((BASE/'opt1_inputs.json').resolve())]=job.sha(BASE/'opt1_inputs.json')
    command=selected_python_command([str(Path(__file__).resolve()),a.phase,'--worker','--source-commit',a.source_commit],SELECTION)
    receipt=job.supervise(command,BASE/('native_'+a.phase),BASE/'active.lock',files,ROOT,worker_receipt=BASE/a.phase/'worker.json',
        on_worker_owned=lambda ident:(BASE/a.phase/'owned').write_text('owned',encoding='utf-8'),provenance={'source_commit':a.source_commit,'phase':'training_efficiency_'+a.phase})
    return 0 if receipt['status']=='COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
