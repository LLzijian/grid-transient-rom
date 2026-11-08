"""Source-bound Windows Job wrapper for the finite train-only horizon pair."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import selected_python_command,command_environment
SELECTION=ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01/spawn_chain20261022/runtime_selection.json'
REQUIRED=('experiments/horizon_pair_20261023.py','tests/test_horizon_pair_20261023.py',
    str((BASE/'launch.py').relative_to(ROOT)),str((BASE/'make_binding.py').relative_to(ROOT)),
    'reports/horizon_pair_20261023_protocol.md','reports/astra6_horizon_pair_20261023.md',
    'reports/astra6_horizon_pair_20261023_raw.txt','reports/horizon_pair_20261023_review.md')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def ref(path):return {'path':str(Path(path).resolve()),'sha256':job.sha(path)}
def check(binding,commit):
    if binding.get('source_commit')!=commit or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('registered exact source required')
    if binding.get('no_process_quotas') is not True:raise ValueError('uncapped process authority required')
    files={str((ROOT/name).resolve()):digest for name,digest in binding['files'].items()}
    job.verify_files(files);selected_python_command([],SELECTION)
    for name in REQUIRED:
        path=(ROOT/name).resolve()
        if files.get(str(path))!=job.sha(path):raise ValueError('missing frozen new source '+name)
    for name in ('old_selection','old_complete','old_native','pool_manifest','teacher_identity'):
        path=(ROOT/binding[name]['path']).resolve()
        if files.get(str(path))!=binding[name]['sha256'] or job.sha(path)!=binding[name]['sha256']:
            raise ValueError('old proof/input differs '+name)
    old_native=read(ROOT/binding['old_native']['path'])
    if (old_native.get('status')!='COMPLETED' or old_native.get('source_commit')!='c9ec666fc7e8dd5ad04f63d9bea3ada33d4ec7a2'
        or not old_native.get('source_before') or old_native['source_before']!=old_native.get('source_after')
        or old_native.get('cleanup_verified') is not True or old_native.get('actual_worker_state')!='GONE'
        or old_native.get('orphans_cleaned') is not False or type(old_native.get('job_active_processes')) is not int
        or old_native['job_active_processes']!=0
        or any(type(old_native.get(k)) is not int or old_native[k]!=0 for k in ('exit_code','worker_exit_code','launcher_exit_code'))):
        raise ValueError('old clean Job completion required')
    for name,digest in old_native['source_before'].items():
        path=(ROOT/name).resolve()
        if files.get(str(path))!=digest or job.sha(path)!=digest:raise ValueError('old closed source changed '+name)
    for group in ('train_inputs','train_qualifications'):
        if set(binding[group])!={'6201','6202','6203'}:raise ValueError('three old train streams required')
        for value in binding[group].values():
            path=(ROOT/value['path']).resolve()
            if files.get(str(path))!=value['sha256'] or job.sha(path)!=value['sha256']:raise ValueError('train source differs')
def validate_qa(binding,commit):
    native=read(BASE/'native_qa/receipt.json');result=read(BASE/'qa/complete.json')
    if (native.get('source_commit')!=commit or native.get('phase')!='horizon_pair_qa' or native.get('status')!='COMPLETED'
        or native.get('cleanup_verified') is not True or native.get('orphans_cleaned') is not False
        or native.get('actual_worker_state')!='GONE' or type(native.get('job_active_processes')) is not int
        or native['job_active_processes']!=0 or not native.get('source_before') or native['source_before']!=native.get('source_after')
        or any(type(native.get(k)) is not int or native[k]!=0 for k in ('exit_code','worker_exit_code','launcher_exit_code'))
        or result.get('source_commit')!=commit or result.get('status')!='PASS'
        or type(result.get('pytest_returncode')) is not int or result['pytest_returncode']!=0):
        raise ValueError('actual same-source related QA required')
    for name,digest in binding['files'].items():
        if native['source_before'].get(str((ROOT/name).resolve()))!=digest:raise ValueError('QA sources differ')
def worker(phase,binding,commit):
    check(binding,commit);out=BASE/phase;out.mkdir(exist_ok=False)
    command=selected_python_command([],SELECTION);k=job.api()
    job.atomic(out/'worker.json',{**job.identity(),'physical_image':job.physical_image(k,k.GetCurrentProcess()),
        'logical_executable':command['logical_executable'],'runtime_scheme':command['runtime_scheme'],
        'runtime_selection':command['runtime_selection'],'source_commit':commit,'phase':'horizon_pair_'+phase})
    while not (out/'owned').is_file():time.sleep(.01)
    try:
        if phase=='qa':
            temp=ROOT/'private_data/qhp1'
            if temp.exists():raise ValueError('preserve existing QA temp; no overwrite')
            cmd=selected_python_command(['-m','pytest','-q',str(ROOT/'tests/test_horizon_pair_20261023.py'),'--basetemp',str(temp)],SELECTION)
            result=subprocess.run([cmd['executable'],*cmd['arguments']],cwd=ROOT,env=command_environment(cmd),
                capture_output=True,text=True,encoding='utf-8',creationflags=subprocess.DETACHED_PROCESS)
            (out/'stdout.txt').write_text(result.stdout,encoding='utf-8');(out/'stderr.txt').write_text(result.stderr,encoding='utf-8')
            job.atomic(out/'complete.json',{'status':'PASS' if result.returncode==0 else 'FAIL','source_commit':commit,
                'pytest_returncode':result.returncode,'pytest_creation_flags':subprocess.DETACHED_PROCESS,
                'planned_qualification_cuda_updates':72,'planned_qualification_cpu_updates':2})
            return 0 if result.returncode==0 else 2
        validate_qa(binding,commit)
        import torch
        torch.set_num_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('registered CUDA unavailable')
        from experiments import horizon_pair_20261023 as study
        result=study.run_study(BASE/'science',binding,commit)
        job.atomic(out/'complete.json',{'status':'COMPLETED' if result.get('all_twelve_frozen') else 'UNRESOLVED',
            'source_commit':commit,'selection_manifest':ref(BASE/'science/selection.json'),
            'actual_updates':result['actual_updates'],'actual_restores':result['actual_restores'],
            'all_twelve_frozen':result['all_twelve_frozen'],'synthetic_val_accessed':False,'project_val_accessed':False,'project_test_accessed':False})
        return 0 if result.get('all_twelve_frozen') else 2
    except BaseException as error:
        import traceback
        job.atomic(out/'failure.json',{'status':'UNRESOLVED','source_commit':commit,'error':repr(error),'traceback':traceback.format_exc()})
        raise
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('qa','run'));p.add_argument('--source-commit',required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args();binding=read(BASE/'binding.json');check(binding,a.source_commit)
    if a.worker:return worker(a.phase,binding,a.source_commit)
    if (BASE/a.phase).exists() or (BASE/('native_'+a.phase)).exists():raise ValueError('sole execution exists; preserve evidence')
    for path in (BASE/'active.lock',ROOT/'private_data/closure_mechanism_20261019/P1/bounded_edge_grammar01/active.lock',
        ROOT/'private_data/closure_mechanism_20261019/P1/rational_checkpoint02/active.lock',
        ROOT/'private_data/closure_mechanism_20261019/P1/finite_grammar_execution01/active.lock',
        ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01/native_active.lock'):
        if path.exists():raise ValueError('existing owned job; no duplicate')
    files=dict(binding['files']);files[str((BASE/'binding.json').resolve())]=job.sha(BASE/'binding.json')
    if a.phase=='run':
        validate_qa(binding,a.source_commit)
        for path in (BASE/'native_qa/receipt.json',BASE/'qa/complete.json'):files[str(path.resolve())]=job.sha(path)
    command=selected_python_command([str(Path(__file__).resolve()),a.phase,'--worker','--source-commit',a.source_commit],SELECTION)
    receipt=job.supervise(command,BASE/('native_'+a.phase),BASE/'active.lock',files,ROOT,
        worker_receipt=BASE/a.phase/'worker.json',on_worker_owned=lambda ident:(BASE/a.phase/'owned').write_text('owned',encoding='utf-8'),
        provenance={'source_commit':a.source_commit,'phase':'horizon_pair_'+a.phase})
    return 0 if receipt['status']=='COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
