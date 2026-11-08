"""Two one-shot P0 native phases. No P1 execution authority."""
import argparse,ast,ctypes,hashlib,json,math,os,subprocess,sys,time,traceback
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
PHASES=('qa','cost')
REPORTS=('reports/long_horizon_discrimination_20261021_protocol.md','reports/long_horizon_discrimination_20261021_plan_review.md','reports/astra6_after_edge_deletion_20261021.md','AGENTS.md')
def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def identity():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;kernel.GetProcessTimes.restype=wintypes.BOOL
    times=[wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(),*[ctypes.byref(x) for x in times]):raise ctypes.WinError(ctypes.get_last_error())
    return {'pid':os.getpid(),'creation_filetime_utc':(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime,'executable':str(Path(sys.executable).resolve())}
def verify(binding):
    files=binding.get('files');require(isinstance(files,dict) and files,'explicit nonempty binding required');result={}
    for name,digest in files.items():
        path=(ROOT/name).resolve();require(not Path(name).is_absolute() and path.is_relative_to(ROOT) and path.suffix.lower() not in ('.h5','.hdf5'),'forbidden bound path')
        require(isinstance(digest,str) and len(digest)==64,'bound hash invalid')
        actual=sha(path);require(actual==digest,'bound input changed: '+name);result[name]=actual
    return result
def scientific_closure():
    pending=[ROOT/'experiments/fixed_edge_pair_20261019.py',ROOT/'experiments/long_horizon_discrimination_20261021.py',ROOT/'tests/test_long_horizon_discrimination_20261021.py'];seen=set()
    while pending:
        path=pending.pop().resolve()
        if path in seen:continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[(n.name,0) for n in node.names] if isinstance(node,ast.Import) else []
            if isinstance(node,ast.ImportFrom):
                names=[(node.module or '',node.level)]+[(((node.module+'.') if node.module else '')+n.name,node.level) for n in node.names if n.name!='*']
            for module,level in names:
                roots=[path.parent.joinpath(*(['..']*(level-1)))] if level else [ROOT,ROOT/'experiments',ROOT/'src',ROOT/'tests']
                for root in roots:
                    target=root.joinpath(*module.split('.')) if module else root
                    for item in (target.with_suffix('.py'),target/'__init__.py'):
                        if item.is_file() and item.resolve().is_relative_to(ROOT):pending.append(item)
                    for count in range(1,len(module.split('.'))):
                        item=root.joinpath(*module.split('.')[:count])/'__init__.py'
                        if item.is_file():pending.append(item)
    return {p.relative_to(ROOT).as_posix() for p in seen}

ENDPOINT_SHA='42c624470156e2e2b438671131f3ca7aecfed834b00b9344b7a09a2ac44ad627'
def endpoint_inputs(path):
    path=Path(path).resolve()
    require(path==BASE/'endpoints.json' and sha(path)==ENDPOINT_SHA,'fixed frozen endpoint manifest bytes required')
    value=read(path);files=value.get('files')
    require(value.get('seeds')==[5101,5102,5103] and isinstance(files,dict) and len(files)==25,'fixed endpoint seeds and explicit25 input identities')
    for name,h in files.items():
        p=(ROOT/name).resolve()
        require(not Path(name).is_absolute() and p.is_relative_to(ROOT) and p.suffix.lower() not in ('.h5','.hdf5') and isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'invalid explicit endpoint input')
    rows=value.get('rows');require(isinstance(rows,list) and len(rows)==3 and {r['seed'] for r in rows}=={5101,5102,5103},'three unique endpoint rows')
    pairs=set()
    for row in rows:
        require(isinstance(row.get('scales'),list) and len(row['scales'])==30 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in row['scales']),'fixed finite positive scales')
        for key in ('old_val_clean_sha256','teacher_source_model_sha256'):
            h=row.get(key);require(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'fixed teacher/history identities')
        require(set(row.get('models',{}))=={'sin','cos','empty'},'all three fixed structures')
        for role,item in row['models'].items():
            for key,hashkey in (('path','model_sha256'),('training_path','training_sha256')):
                p=Path(item[key]);require(p.is_absolute() and p.resolve().is_relative_to(ROOT) and p.name==('model.json' if key=='path' else 'training.pt'),'fixed model/training path')
                name=p.resolve().relative_to(ROOT).as_posix()
                require(name in files and files[name]==item[hashkey] and name not in pairs,'unique frozen endpoint hash absent');pairs.add(name)
    require(len(pairs)==18,'all nine model/training pairs required')
    return files

def command_gate(a,command,binding):
    expected_output=BASE/(a.phase+'01')
    require(a.output==BASE/('native_'+a.phase+'01') and a.worker==expected_output/'worker.json','fixed phase outputs required')
    require(a.command==BASE/'commands'/f'{a.phase}01.json' and a.prior==BASE/'prior'/f'{a.phase}01.json' and a.binding==BASE/'bindings'/f'{a.phase}01.json','fixed exact command/prior/binding paths')
    expected=['-m','experiments.long_horizon_discrimination_20261021',a.phase,'--output',str(expected_output),'--max-seconds','210.0','--source-commit',a.source_commit,'--binding',str(a.binding)]
    require(command['arguments']==expected and command['arguments_line']==subprocess.list2cmdline(expected) and Path(command['executable']).resolve()==ROOT/'.venv-gpu/Scripts/python.exe','fixed native GPU argv required')
    required=[Path(__file__),BASE/'make_invocation.py',BASE/'support_cost.json',a.command,a.prior,ROOT/'experiments/long_horizon_discrimination_20261021.py',ROOT/'tests/test_long_horizon_discrimination_20261021.py',ROOT/'experiments/fixed_edge_pair_20261019.py',ROOT/'experiments/native_process_receipt_20261020.ps1',*[ROOT/n for n in REPORTS]]
    require(all(p.relative_to(ROOT).as_posix() in binding['files'] for p in required),'required source/command binding absent')
    science=binding.get('immutable_science_files');require(isinstance(science,dict) and science,'explicit immutable scientific closure required')
    mandatory=[ROOT/'experiments/long_horizon_discrimination_20261021.py',ROOT/'tests/test_long_horizon_discrimination_20261021.py',ROOT/'experiments/fixed_edge_pair_20261019.py',Path(__file__),BASE/'make_invocation.py']
    require(all(x.relative_to(ROOT).as_posix() in science for x in mandatory) and all(binding['files'].get(k)==h for k,h in science.items()),'scientific/native closure is not fully bound')
    require(scientific_closure()<=set(science),'actual runner/test/native local dependency closure absent')
    manifest=BASE/'endpoints.json';endpointfiles=endpoint_inputs(manifest)
    require(binding.get('endpoint_manifest')=={'path':manifest.relative_to(ROOT).as_posix(),'sha256':sha(manifest)},'exact frozen endpoint manifest identity')
    require(binding['files'].get(manifest.relative_to(ROOT).as_posix())==sha(manifest) and all(binding['files'].get(k)==h==science.get(k) for k,h in endpointfiles.items()),'all frozen endpoint inputs must be bound and immutable')
    generation=Path(binding['generation_path']).resolve()
    require(generation==BASE/'generation'/f'{a.phase}01.json' and sha(generation)==binding['generation_sha256'] and binding['files'].get(generation.relative_to(ROOT).as_posix())==sha(generation),'generation record changed')
    require('$MaxSeconds -gt 3600' in (ROOT/'experiments/native_process_receipt_20261020.ps1').read_text(encoding='utf-8-sig'),'qualified PS bound changed')
    require(0<210<225<240<=3600,'fixed cleanup margin')
def prior_gate(a,binding):
    names=read(a.prior);paths=[Path(x).resolve() for x in names]
    require(len(paths)==len(set(paths)) and set(paths)=={p.resolve() for p in BASE.glob('native_*/execution.json')},'prior receipts incomplete/duplicate')
    require({p.parent for p in paths}=={p.parent.resolve() for p in BASE.glob('native_*/launch.json')},'unreconciled prior launch')
    total=0.;phases=[]
    for path in paths:
        require(path.is_relative_to(BASE),'foreign prior forbidden');r=read(path);inner=r.get('inner',{});outer=r.get('outer',{})
        require(r.get('phase') in PHASES and finite(r.get('charged_seconds')) and 0<=r['charged_seconds']<=240,'unreconciled phase cost')
        require(inner.get('cleanup_verified') is True and inner.get('fixture_identity_after')==inner.get('supervised_identity_after')=='GONE' and inner.get('native_receipt_complete') is True and outer.get('native_receipt_complete') is True,'prior process ownership unresolved')
        require(r.get('source_before')==r.get('source_after'),'prior source discrepancy')
        total+=r['charged_seconds'];phases.append(r['phase'])
        require(binding['files'].get(path.relative_to(ROOT).as_posix())==sha(path),'prior receipt binding absent')
    if a.phase=='qa':require(not paths,'QA single attempt already used')
    else:
        require(phases==['qa'] and len(paths)==1,'cost only after one QA')
        qa=read(paths[0]);require(qa.get('status')=='COMPLETED' and qa.get('phase_gate_pass') is True and qa.get('source_commit')==a.source_commit,'cost requires actual same-source QA pass')
        require(qa.get('inner',{}).get('exit_code')==qa.get('outer',{}).get('exit_code')==0,'QA actual two-layer exits must pass')
        science=binding['immutable_science_files'];require(science==qa.get('immutable_science_files'),'cost scientific identity differs from actual QA')
        require(all(qa['source_before'].get(k)==h==qa['source_after'].get(k) and binding['files'].get(k)==h for k,h in science.items()),'scientific/native dependencies were not actually QA checked')
        item=binding.get('qa_result') or {};marker=BASE/'qa01/complete.json'
        require((ROOT/item.get('path','')).resolve()==marker and item.get('sha256')==sha(marker) and binding['files'].get(marker.relative_to(ROOT).as_posix())==sha(marker),'actual QA result binding required')
        result=read(marker);require(result.get('status')=='COMPLETED' and result.get('result',{}).get('status')=='PASS' and result==qa.get('complete') and sha(marker)==qa.get('complete_sha256'),'actual unchanged QA PASS required')
    support=read(BASE/'support_cost.json')
    require(support.get('support_held_seconds')==520 and support.get('cap_seconds')==1000,'registered P0 support and total cap')
    remaining=240*(2-len(paths))
    require(total+remaining+520<=1000,'complete P0 native plus support budget')
    return {'prior_native_seconds':total,'remaining_native_cap_including_current':remaining,'support_hold':520,'total_cap':1000}
def execute(a):
    require(len(a.source_commit)==40 and all(c in '0123456789abcdef' for c in a.source_commit),'full source commit required')
    for name in ('command','output','worker','binding','prior'):setattr(a,name,getattr(a,name).resolve())
    require(not a.output.exists() and not a.worker.parent.exists(),'exclusive outputs required')
    binding=read(a.binding);command=read(a.command);command_gate(a,command,binding);before=verify(binding);budget=prior_gate(a,binding)
    a.output.mkdir(exist_ok=False)
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    native=[str(ps),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ROOT/'experiments/native_process_receipt_20261020.ps1'),'-CommandJson',str(a.command),'-WorkingDirectory',str(ROOT),'-Output',str(a.output/'inner'),'-TimeoutSeconds','225','-MaxSeconds','240','-SourceCommit',a.source_commit,'-FixtureIdentityJson',str(a.worker)]
    record={'phase':a.phase,'category':a.phase,'status':'RUNNING','source_commit':a.source_commit,'source_before':before,'immutable_science_files':binding['immutable_science_files'],'binding_sha256':sha(a.binding),'command_sha256':sha(a.command),'worker_command':command,'budget_gate':budget,'native_cap_seconds':240,'includes_children_once':True}
    write(a.output/'launch.json',record);started=time.perf_counter()
    try:
        sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
        from fixed_edge_pair_20261019 import native_job
        a.native_invoked=True
        outer=native_job(native,a.output/'outer',240.,worker_json=a.worker);record['outer']=outer;record['charged_seconds']=outer['seconds']
        inner=read(a.output/'inner/receipt.json');record['inner']=inner
        worker=read(a.worker);record['worker']=worker
        complete=read(a.worker.parent/'complete.json');record['complete']=complete;record['complete_sha256']=sha(a.worker.parent/'complete.json')
        record['source_after']=verify(binding)
        okay=(outer.get('status')=='COMPLETED' and outer.get('exit_code')==inner.get('exit_code')==0 and inner.get('timeout') is False and outer.get('native_receipt_complete') is True and inner.get('native_receipt_complete') is True and inner.get('cleanup_verified') is True and inner.get('fixture_identity_after')==inner.get('supervised_identity_after')=='GONE' and worker.get('pid')==outer.get('actual_worker_pid')==inner.get('fixture_pid') and isinstance(worker.get('creation_filetime_utc'),int) and worker.get('creation_filetime_utc')==inner.get('fixture_creation_filetime_utc') and Path(worker.get('executable','')).resolve()==ROOT/'.venv-gpu/Scripts/python.exe' and complete.get('status')=='COMPLETED' and record['source_before']==record['source_after'])
        record['phase_gate_pass']=okay and complete.get('result',{}).get('status')=='PASS'
        record['status']='COMPLETED' if okay else 'UNRESOLVED_EXECUTION'
    except BaseException as exc:
        record.update(status='UNRESOLVED_EXECUTION',phase_gate_pass=False,error=repr(exc),traceback=traceback.format_exc())
        for key,path in [('outer',a.output/'outer/native.json'),('inner',a.output/'inner/receipt.json'),('worker',a.worker)]:
            if path.is_file():record[key]=read(path)
        if 'outer' in record:record['charged_seconds']=record['outer'].get('seconds')
        try:record['source_after']=verify(binding)
        except BaseException as check_error:record['source_after_error']=repr(check_error)
        write(a.output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
    finally:record['launcher_seconds']=time.perf_counter()-started;write(a.output/'execution.json',record)
    print(json.dumps({'status':record['status'],'phase':a.phase,'phase_gate_pass':record.get('phase_gate_pass',False),'charged_seconds':record.get('charged_seconds'),'receipt':str(a.output/'execution.json')}))
    return 0 if record['status']=='COMPLETED' and record.get('phase_gate_pass') is True else 1
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('command','output','worker','binding','prior'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--phase',choices=PHASES,required=True);a=p.parse_args()
    require(BASE==(ROOT/'private_data/closure_mechanism_20261019/P0/long_horizon_discrimination01').resolve(),'fixed P0 study root')
    a.native_invoked=False
    owner=identity();BASE.mkdir(parents=True,exist_ok=True);lock=BASE/'native_active.lock'
    write(lock,owner)
    try:return execute(a)
    finally:
        require(read(lock)==owner,'owner identity changed; preserve lock')
        safe=not a.native_invoked
        if a.native_invoked:
            receipt=a.output/'inner/receipt.json'
            if receipt.is_file():
                inner=read(receipt)
                safe=inner.get('native_receipt_complete') is True and inner.get('cleanup_verified') is True and inner.get('fixture_identity_after')==inner.get('supervised_identity_after')=='GONE'
                if a.worker.is_file():
                    worker=read(a.worker);safe=safe and worker.get('pid')==inner.get('fixture_pid') and worker.get('creation_filetime_utc')==inner.get('fixture_creation_filetime_utc')
        if safe:lock.unlink()
        else:raise RuntimeError('native cleanup not proved; owner lock retained for PID+creation-time reconciliation')
if __name__=='__main__':raise SystemExit(main())

