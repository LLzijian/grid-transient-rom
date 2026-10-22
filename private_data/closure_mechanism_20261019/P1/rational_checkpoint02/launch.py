"""Finite checkpoint continuation: preserved Job qualification, new source-bound work."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import selected_python_command,command_environment
OLD=ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01'
SELECTION=OLD/'spawn_chain20261022/runtime_selection.json'
ORIGINS=ROOT/'results/rational_execution_20261021/checkpoint_origins_20261022.json'
ORIGIN_SHA='f7848dd683e254632f8c8e574db374c72d1f51ce17980ec7981d329cdbd66bae'
REQUIRED=('experiments/rational_checkpoint_recovery_20261022.py',
    'experiments/rational_checkpoint_qualification_20261022.py',
    'tests/test_rational_checkpoint_recovery_20261022.py',
    'private_data/closure_mechanism_20261019/P1/rational_checkpoint02/launch.py',
    'reports/astra6_checkpoint_continuation_20261022_raw.txt',
    'reports/astra6_checkpoint_continuation_20261022.md',
    'reports/checkpoint_continuation_20261022_protocol.md',
    'reports/checkpoint_continuation_20261022_design_review.md',
    'results/rational_execution_20261021/checkpoint_origins_20261022.json',
    'results/rational_execution_20261021/first_gpu_failure_summary.json')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def ref(path):return {'path':str(Path(path).resolve()),'sha256':job.sha(path)}
def check(binding,commit):
    if binding.get('source_commit')!=commit or len(commit)!=40 or any(v not in '0123456789abcdef' for v in commit):raise ValueError('exact new source required')
    if job.sha(ORIGINS)!=ORIGIN_SHA or binding.get('origins')!=ref(ORIGINS):raise ValueError('fixed complete three-row origins required')
    if [r['seed'] for r in read(ORIGINS)['rows']]!=[6301,6302,6303]:raise ValueError('all three nonselective origins required')
    files={str(Path(path).resolve()):digest for path,digest in binding.get('files',{}).items()}
    for name in REQUIRED:
        path=(ROOT/name).resolve()
        if files.get(str(path))!=job.sha(path):raise ValueError('required new source/plan not bound: '+name)
    origins=read(ORIGINS)
    for item in [origins['old_group_complete'],origins['old_native_receipt'],*[item for row in origins['rows'] for item in row['files']]]:
        if files.get(str(Path(item['path']).resolve()))!=item['sha256']:raise ValueError('complete original checkpoint evidence not bound')
    from experiments.rational_execution_20261021 import validate_qa06_pass
    old_binding=read(OLD/'bindings/run_0_rational.json')
    validate_qa06_pass(old_binding,'5ffe63f5f9bd71c68e417da4759fdbd183299799')
    selected_python_command([],SELECTION) # Old qualified sources stay byte-for-byte unchanged.
    job.verify_files(binding['files'])
def validate_unit(commit,binding):
    native=read(BASE/'native_unit/receipt.json');result=read(BASE/'unit/complete.json')
    if (native.get('source_commit')!=commit or native.get('phase')!='checkpoint_unit'
            or native.get('status')!='COMPLETED' or native.get('orphans_cleaned') is not False
            or native.get('cleanup_verified') is not True or type(native.get('job_active_processes')) is not int
            or native['job_active_processes']!=0 or native.get('source_before')!=native.get('source_after')
            or not native.get('source_before') or native.get('actual_worker_state')!='GONE'
            or any(type(native.get(k)) is not int or native[k]!=0 for k in ('exit_code','worker_exit_code','launcher_exit_code'))
            or result.get('status')!='PASS' or result.get('pytest_returncode')!=0 or result.get('source_commit')!=commit
            or result.get('pytest_creation_flags')!=8):
        raise ValueError('actual sole related serializer unit suite must pass')
    if any(native['source_before'].get(str(Path(path).resolve()))!=digest for path,digest in binding['files'].items()):
        raise ValueError('related serializer unit source differs from actual run binding')
    return native,result
def worker(phase,binding,commit):
    check(binding,commit);out=BASE/phase
    out.mkdir(exist_ok=False)
    cmd=selected_python_command([],SELECTION);k=job.api()
    job.atomic(out/'worker.json',{**job.identity(),'physical_image':job.physical_image(k,k.GetCurrentProcess()),
        'logical_executable':cmd['logical_executable'],'runtime_scheme':cmd['runtime_scheme'],
        'runtime_selection':cmd['runtime_selection'],'source_commit':commit,'phase':'checkpoint_'+phase,'group':None})
    while not (out/'owned').is_file():time.sleep(.01)
    try:
        if phase=='unit':
            temp=ROOT/'private_data/qcp1'
            if temp.exists():raise ValueError('related test temp exists; preserve evidence')
            command=selected_python_command(['-m','pytest','-q',str(ROOT/'tests/test_rational_checkpoint_recovery_20261022.py'),
                '--basetemp',str(temp)],SELECTION)
            flags=subprocess.DETACHED_PROCESS if os.name=='nt' else 0
            result=subprocess.run([command['executable'],*command['arguments']],cwd=ROOT,env=command_environment(command),
                capture_output=True,text=True,encoding='utf-8',creationflags=flags)
            (out/'stdout.txt').write_text(result.stdout,encoding='utf-8');(out/'stderr.txt').write_text(result.stderr,encoding='utf-8')
            report={'status':'PASS' if result.returncode==0 else 'FAIL','source_commit':commit,
                'pytest_returncode':result.returncode,'pytest_creation_flags':flags}
            job.atomic(out/'complete.json',report)
            return 0 if result.returncode==0 else 2
        validate_unit(commit,binding)
        import torch
        torch.set_num_threads(1)
        from experiments import rational_checkpoint_recovery_20261022 as recovery
        from experiments import rational_checkpoint_qualification_20261022 as qualification
        report,states=qualification.qualify_cuda(read(ORIGINS),BASE,commit)
        if report.get('status')!='PASS' or report.get('actual_updates')!=132 or len(states)!=3:raise ValueError('fixed four paths and three migrations required')
        rows=[]
        for group in recovery.study.GROUPS:
            target=BASE/'science'/('run_'+group);target.mkdir(parents=True,exist_ok=group=='0_rational')
            result=recovery.train_group(target,commit,group,initial_states=states if group=='0_rational' else None)
            job.atomic(target/'complete.json',{'status':'COMPLETED','result':result})
            if not result.get('all_group_frozen'):raise ValueError('scientific group unresolved: '+group)
            rows.extend(result['rows'])
            if group=='0_rational':states.clear()
        frozen={'source_commit':commit,'all_twelve_frozen':True,'rows':rows,'historical_updates':96,'new_formal_updates':6048,
            'validation_updates':132,'origin_manifest':ref(ORIGINS)}
        recovery.study.validate_selection(frozen,commit)
        manifest=BASE/'science/selection.json';job.atomic(manifest,frozen)
        evaluation_out=BASE/'evaluate01';evaluation_out.mkdir(exist_ok=False)
        evaluation=recovery.study.evaluate(evaluation_out,{'selection_manifest':ref(manifest)},commit,None)
        job.atomic(evaluation_out/'result.json',evaluation)
        job.atomic(out/'complete.json',{'status':'COMPLETED','source_commit':commit,'all_twelve_frozen':True,
            'selection_manifest':ref(manifest),'evaluation':ref(evaluation_out/'result.json'),'new_formal_updates':6048,'validation_updates':132})
        return 0
    except BaseException as error:
        import traceback
        job.atomic(out/'failure.json',{'status':'UNRESOLVED','source_commit':commit,'error':repr(error),'traceback':traceback.format_exc()})
        raise
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('unit','run'));p.add_argument('--source-commit',required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    binding=read(BASE/'binding.json');check(binding,a.source_commit)
    if a.worker:return worker(a.phase,binding,a.source_commit)
    if (BASE/a.phase).exists() or (BASE/('native_'+a.phase)).exists():raise ValueError('sole attempt already exists; no overwrite or retry')
    if (BASE/'active.lock').exists() or (OLD/'native_active.lock').exists():raise ValueError('owned job already present; no duplicate queue')
    files=dict(binding['files']);files[str((BASE/'binding.json').resolve())]=job.sha(BASE/'binding.json')
    if a.phase=='run':
        validate_unit(a.source_commit,binding)
        for path in (BASE/'native_unit/receipt.json',BASE/'unit/complete.json'):
            files[str(path.resolve())]=job.sha(path)
    command=selected_python_command([str(Path(__file__).resolve()),a.phase,'--worker','--source-commit',a.source_commit],SELECTION)
    receipt=job.supervise(command,BASE/('native_'+a.phase),BASE/'active.lock',files,ROOT,
        worker_receipt=BASE/a.phase/'worker.json',
        on_worker_owned=lambda ident:(BASE/a.phase/'owned').write_text('owned',encoding='utf-8'),
        provenance={'source_commit':a.source_commit,'phase':'checkpoint_'+a.phase,'group':None})
    return 0 if receipt['status']=='COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
