"""Source-bound native Job entry for the fixed three-seed grammar study."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from experiments import uncapped_windows_job_20261021 as job
from experiments.rational_python_runtime_20261022 import selected_python_command,command_environment
SELECTION=ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01/spawn_chain20261022/runtime_selection.json'
REQUIRED=('experiments/bounded_edge_grammar_20261022.py','tests/test_bounded_edge_grammar_20261022.py',
    'private_data/closure_mechanism_20261019/P1/bounded_edge_grammar01/launch.py',
    'private_data/closure_mechanism_20261019/P1/bounded_edge_grammar01/make_binding.py',
    'reports/bounded_edge_grammar_20261022_protocol.md','reports/astra6_bounded_edge_20261022.md',
    'reports/astra6_bounded_edge_20261022_raw.txt','reports/bounded_edge_grammar_20261022_review.md',
    'results/bounded_edge_grammar_20261022/preflight.json',
    'experiments/finite_grammar_execution_20261022.py',
    'tests/test_finite_grammar_execution_20261022.py')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p):return {'path':str(Path(p).resolve()),'sha256':job.sha(p)}
def check(binding,commit):
    if binding.get('source_commit')!=commit or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('exact registered source required')
    files={str(Path(p).resolve()):v for p,v in binding.get('files',{}).items()}
    for name in REQUIRED:
        p=(ROOT/name).resolve()
        if files.get(str(p))!=job.sha(p):raise ValueError('missing bound source '+name)
    if binding.get('no_process_quotas') is not True:raise ValueError('prospective uncapped authority required')
    job.verify_files(binding['files'])
    selected_python_command([],SELECTION)
    for name in (('old_qa_complete','old_qa_receipt') if binding.get('phase')=='freeze' else ('pool_manifest','teacher_identity','old_qa_complete','old_qa_receipt')):
        p=ROOT/binding[name]['path']
        if files.get(str(p.resolve()))!=binding[name]['sha256'] or job.sha(p)!=binding[name]['sha256']:
            raise ValueError('changed frozen science or QA proof '+name)
    qa=read(ROOT/binding['old_qa_complete']['path'])
    if qa.get('status')!='COMPLETED' or qa.get('result',{}).get('status')!='PASS' or qa['result'].get('pytest_returncode')!=0:
        raise ValueError('old unchanged scientific QA unavailable')
    native=read(ROOT/binding['old_qa_receipt']['path'])
    if (binding['old_qa_receipt']['sha256']!='8989b1e74b70c52e0aeb98de79adc3a57d4a53104005a8af93ae1a609dd1c60a'
        or native.get('status')!='COMPLETED' or native.get('source_commit')!='f41cb671a48d102705362fe84f6f2e16e00cbc2a'
        or native.get('phase_gate_pass') is not True or not native.get('source_before')
        or native['source_before']!=native.get('source_after')
        or native.get('complete_sha256')!=binding['old_qa_complete']['sha256']):
        raise ValueError('actual unchanged old scientific QA receipt required')
    inner=native.get('inner',{});outer=native.get('outer',{})
    if (inner.get('status')!='COMPLETED' or inner.get('cleanup_verified') is not True
        or inner.get('native_receipt_complete') is not True or inner.get('fixture_identity_after')!='GONE'
        or inner.get('supervised_identity_after')!='GONE' or inner.get('errors')
        or outer.get('status')!='COMPLETED' or outer.get('timed_out') is not False
        or any(type(v) is not int or v!=0 for v in (inner.get('exit_code'),outer.get('exit_code')))):
        raise ValueError('old scientific QA outer/inner native failure')
    python_sources={name:digest for name,digest in native['source_before'].items() if name.endswith('.py')}
    if len(python_sources)!=69:raise ValueError('old scientific QA Python closure changed')
    for name,digest in python_sources.items():
        path=(ROOT/name).resolve()
        if files.get(str(path))!=digest or job.sha(path)!=digest:
            raise ValueError('old scientific QA Python source changed '+name)
    if binding.get('phase')!='freeze':
        for name in ('freeze_receipt','freeze_complete'):
            path=ROOT/binding[name]['path']
            if files.get(str(path.resolve()))!=binding[name]['sha256'] or job.sha(path)!=binding[name]['sha256']:
                raise ValueError('changed actual syntax-freeze proof')
        native=read(ROOT/binding['freeze_receipt']['path']);result=read(ROOT/binding['freeze_complete']['path'])
        if (native.get('status')!='COMPLETED' or native.get('phase')!='bounded_edge_freeze'
            or native.get('cleanup_verified') is not True or native.get('orphans_cleaned') is not False
            or type(native.get('job_active_processes')) is not int or native['job_active_processes']!=0
            or native.get('actual_worker_state')!='GONE' or not native.get('source_before')
            or native['source_before']!=native.get('source_after')
            or any(type(native.get(k)) is not int or native[k]!=0 for k in ('exit_code','worker_exit_code','launcher_exit_code'))
            or result.get('source_commit')!=native.get('source_commit')
            or result.get('status')!='COMPLETED' or result.get('result',{}).get('status') not in ('PASS','FROZEN','READY')):
            raise ValueError('actual successful pure-syntax freeze required')
        for key,filename in (('pool_manifest','pool.json'),('teacher_identity','teacher.json')):
            actual=result['result'].get(key,{})
            expected=binding[key]
            required=(BASE/'frozen_pool'/filename).resolve()
            if ((ROOT/actual.get('path','')).resolve()!=required or (ROOT/expected['path']).resolve()!=required
                or actual.get('sha256')!=expected['sha256'] or job.sha(required)!=expected['sha256']):
                raise ValueError('formal pool/teacher must be actual frozen outputs')
        for name in REQUIRED:
            path=(ROOT/name).resolve()
            if native['source_before'].get(str(path))!=files[str(path)]:
                raise ValueError('frozen syntax source differs before formal work')
def validate_qa(binding,commit):
    n=read(BASE/'native_qa/receipt.json');q=read(BASE/'qa/complete.json')
    if (n.get('source_commit')!=commit or n.get('phase')!='bounded_edge_qa' or n.get('status')!='COMPLETED'
        or n.get('cleanup_verified') is not True or n.get('orphans_cleaned') is not False
        or n.get('actual_worker_state')!='GONE' or type(n.get('job_active_processes')) is not int
        or n['job_active_processes']!=0 or not n.get('source_before') or n['source_before']!=n.get('source_after')
        or any(type(n.get(k)) is not int or n[k]!=0 for k in ('exit_code','worker_exit_code','launcher_exit_code'))
        or q.get('source_commit')!=commit or q.get('status')!='PASS' or q.get('pytest_returncode')!=0):
        raise ValueError('actual same-source related QA required')
    for path,digest in binding['files'].items():
        if n['source_before'].get(str(Path(path).resolve()))!=digest:raise ValueError('QA sources differ')
def worker(phase,binding,commit):
    check(binding,commit);out=BASE/phase;out.mkdir(exist_ok=False)
    command=selected_python_command([],SELECTION);k=job.api()
    job.atomic(out/'worker.json',{**job.identity(),'physical_image':job.physical_image(k,k.GetCurrentProcess()),
        'logical_executable':command['logical_executable'],'runtime_scheme':command['runtime_scheme'],
        'runtime_selection':command['runtime_selection'],'source_commit':commit,'phase':'bounded_edge_'+phase})
    while not (out/'owned').is_file():time.sleep(.01)
    try:
        if phase=='freeze':
            from experiments import bounded_edge_grammar_20261022 as study
            report=study.freeze(BASE/'frozen_pool')
            job.atomic(out/'complete.json',{'status':'COMPLETED','source_commit':commit,'result':report})
            return 0 if report.get('status') in ('PASS','FROZEN','READY') else 2
        if phase=='qa':
            temp=ROOT/'private_data/qbg1'
            if temp.exists():raise ValueError('retain existing QA temp; no overwrite')
            cmd=selected_python_command(['-m','pytest','-q',str(ROOT/'tests/test_bounded_edge_grammar_20261022.py'),'--basetemp',str(temp)],SELECTION)
            result=subprocess.run([cmd['executable'],*cmd['arguments']],cwd=ROOT,env=command_environment(cmd),
                capture_output=True,text=True,encoding='utf-8',creationflags=subprocess.DETACHED_PROCESS)
            (out/'stdout.txt').write_text(result.stdout,encoding='utf-8');(out/'stderr.txt').write_text(result.stderr,encoding='utf-8')
            report={'status':'PASS' if result.returncode==0 else 'FAIL','source_commit':commit,
                'pytest_returncode':result.returncode,'pytest_creation_flags':subprocess.DETACHED_PROCESS,
                'qualification_workload':'one actual32-to34 shared-serializer CUDA test; other new tests use no formal data'}
            job.atomic(out/'complete.json',report);return 0 if result.returncode==0 else 2
        validate_qa(binding,commit)
        import torch
        torch.set_num_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('registered GPU unavailable')
        from experiments import bounded_edge_grammar_20261022 as study
        science=BASE/'science'
        result=study.run_study(science,binding,commit)
        if not result.get('all_eighteen_frozen'):raise ValueError('all18 endpoints unresolved; no synthetic val')
        manifest=science/'selection.json'
        study.validate_selection(read(manifest),binding,commit)
        evaluation_out=BASE/'evaluate01'
        eval_binding={**binding,'selection_manifest':ref(manifest)}
        evaluation=study.evaluate_study(evaluation_out,eval_binding,commit)
        job.atomic(evaluation_out/'result.json',evaluation)
        job.atomic(out/'complete.json',{'status':'COMPLETED','source_commit':commit,
            'all_eighteen_frozen':True,'selection_manifest':ref(manifest),'evaluation':ref(evaluation_out/'result.json'),
            'formal_updates_planned':9216,'science_result':result.get('recorded_completed_counts')})
        return 0
    except BaseException as error:
        import traceback
        job.atomic(out/'failure.json',{'status':'UNRESOLVED','source_commit':commit,'error':repr(error),'traceback':traceback.format_exc()})
        raise
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('freeze','qa','run'));p.add_argument('--source-commit',required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args();binding=read(BASE/('freeze_binding.json' if a.phase=='freeze' else 'binding.json'));check(binding,a.source_commit)
    if (a.phase=='freeze')!=(binding.get('phase')=='freeze'):raise ValueError('phase binding mismatch')
    if a.worker:return worker(a.phase,binding,a.source_commit)
    if (BASE/a.phase).exists() or (BASE/('native_'+a.phase)).exists():raise ValueError('sole execution already exists; preserve evidence')
    for lock in (BASE/'active.lock',ROOT/'private_data/closure_mechanism_20261019/P1/rational_checkpoint02/active.lock',
                 ROOT/'private_data/closure_mechanism_20261019/P1/finite_grammar_execution01/active.lock',
                 ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01/native_active.lock'):
        if lock.exists():raise ValueError('existing owned job; no duplicate')
    files=dict(binding['files']);binding_path=BASE/('freeze_binding.json' if a.phase=='freeze' else 'binding.json');files[str(binding_path.resolve())]=job.sha(binding_path)
    if a.phase=='run':
        validate_qa(binding,a.source_commit)
        for p in (BASE/'native_qa/receipt.json',BASE/'qa/complete.json'):files[str(p.resolve())]=job.sha(p)
    command=selected_python_command([str(Path(__file__).resolve()),a.phase,'--worker','--source-commit',a.source_commit],SELECTION)
    receipt=job.supervise(command,BASE/('native_'+a.phase),BASE/'active.lock',files,ROOT,
        worker_receipt=BASE/a.phase/'worker.json',
        on_worker_owned=lambda ident:(BASE/a.phase/'owned').write_text('owned',encoding='utf-8'),
        provenance={'source_commit':a.source_commit,'phase':'bounded_edge_'+a.phase})
    return 0 if receipt['status']=='COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
