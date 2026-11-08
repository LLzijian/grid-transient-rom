"""Finite pool freeze, QA and cost native supervision; no scientific execution on import."""
import argparse,ast,ctypes,hashlib,json,math,os,subprocess,sys,time,traceback
from ctypes import wintypes
from pathlib import Path
LAUNCH_START=time.perf_counter()
ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
PHASES=('qa','probe','benchmark')
REPORTS=('reports/edge_polynomial_runtime_20261021_protocol.md','reports/edge_polynomial_runtime_20261021_plan_review.md','reports/astra6_after_finite_grammar_20261021.md','reports/edge_polynomial_runtime_20261021_implementation_review.md','reports/edge_polynomial_runtime_20261021_execution_setup.md','AGENTS.md')
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
    pending=[ROOT/'experiments/fixed_edge_pair_20261019.py',ROOT/'experiments/edge_polynomial_study_20261021.py',ROOT/'tests/test_edge_polynomial_study_20261021.py',ROOT/'experiments/edge_polynomial_runtime_20261021.py',ROOT/'tests/test_edge_polynomial_runtime_20261021.py'];seen=set()
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


PREVIOUS=ROOT/'private_data/closure_mechanism_20261019/P0/long_horizon_discrimination01'
POOL=ROOT/'private_data/closure_mechanism_20261019/P0/finite_grammar_discovery01/freeze_pool01'
POOL_SHA='00040fbf83705eb660474de660ef96fec0424a299fd3650df4c1f77b605a8925'
TEACHER_SHA='5e31e4a454c45d56aa8ca2586cb39cde1ce96cbcb485fd7316fdbda6af36157c'
FORMULA={'T':'2304*u_max+27*r_max+3*g_pair+3*v_final+c_all','T_cap':576,'margin':1.25,'support':180,'P1_complete_cap_seconds':900,'P1_native_cap_seconds':720,'startup':'2*max(three complete cold native launch wall times)'}
def previous_science():
    r=read(PREVIOUS/'native_qa01/execution.json')
    require(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and r.get('source_before')==r.get('source_after'),'actual prior qualified proof')
    science=r['immutable_science_files'];require(len(science)==96,'inherited96 identities');return science
def complete_record(p,allow_failure=False):
    r=read(p);i=r.get('inner',{});o=r.get('outer',{})
    require(r.get('phase') in PHASES and finite(r.get('charged_seconds')) and r['charged_seconds']>=0 and i.get('native_receipt_complete') is True and o.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE' and r.get('source_before')==r.get('source_after'),'unreconciled prior')
    if not allow_failure:require(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and i.get('exit_code')==o.get('exit_code')==0,'actual completed passing prior')
    return r
def phase_plan(phase,source,approval=None):
    require(phase in PHASES and len(source)==40 and all(c in '0123456789abcdef' for c in source),'registered phase/source')
    paths=sorted(BASE.glob('native_*/execution.json'));require({p.parent for p in paths}=={p.parent for p in BASE.glob('native_*/launch.json')},'pending native launch')
    rows=[complete_record(p,True) for p in paths];qas=[r for r in rows if r['phase']=='qa'];probes=[r for r in rows if r['phase']=='probe'];benches=[r for r in rows if r['phase']=='benchmark']
    require(len(qas)<=2 and len(probes)<=3 and len(benches)<=1 and sum(r['charged_seconds'] for r in probes)<=90 and sum(r['charged_seconds'] for r in benches)<=1000,'phase counts/caps')
    if phase=='qa':
        require(not probes and not benches and len(qas)<2 and not any(r.get('phase_gate_pass') is True for r in qas),'QA unique success before probes')
        if not qas:require(approval is None,'QA01 no repair authority');name='qa01';caps=(650.,635.,620.)
        else:
            require(approval is not None and qas[0]['charged_seconds']<=650,'explicit defect repair only')
            value=read(approval);require(value.get('status')=='APPROVED' and value.get('target')=='qa02' and value.get('source_commit')==source and value.get('failed_execution_sha256')==sha(BASE/'native_qa01/execution.json') and value.get('declared_file_diff'),'bound root implementation defect/diff approval')
            name='qa02';caps=(300.,285.,270.)
    else:
        require(approval is None and qas and qas[-1].get('phase_gate_pass') is True and not benches,'actual QA pass before cold probes/benchmark')
        qp=next(p for p in paths if read(p)['phase']=='qa' and read(p).get('phase_gate_pass') is True);require(complete_record(qp)['source_commit']==source,'same passed QA source')
        for p in paths:
            if read(p)['phase']=='probe':require(complete_record(p)['source_commit']==source,'same probe source')
        if phase=='probe':require(len(probes)<3,'three unique probes');name='probe0'+str(len(probes)+1);caps=(30.,25.,20.)
        else:require(len(probes)==3,'three complete cold launches required');name='benchmark01';caps=(1000.,980.,960.)
    support=read(BASE/'support_cost.json');require(support=={'support_held_seconds':1500,'QA_cap_seconds':650,'repair_cap_seconds':300,'benchmark_cap_seconds':1000,'metadata_cap_seconds':150,'cap_seconds':3600},'registered full stage ledger')
    require(sum(r['charged_seconds'] for r in qas)<=950 and 1500+650+300+1000+150<=3600,'full stage hard cap')
    return name,caps,paths,{'support':1500,'QA':650,'repair':300,'benchmark':1000,'metadata':150,'total':3600}
def pool_identity():
    require(sha(POOL/'pool.json')==POOL_SHA and sha(POOL/'teacher.json')==TEACHER_SHA,'immutable original pool and teacher')
    return {'pool_manifest':{'path':(POOL/'pool.json').relative_to(ROOT).as_posix(),'sha256':POOL_SHA},'teacher_identity':{'path':(POOL/'teacher.json').relative_to(ROOT).as_posix(),'sha256':TEACHER_SHA}}
def arguments(phase,name,caps,source,bp):
    return ['-m','experiments.edge_polynomial_study_20261021',phase,'--output',str(BASE/name),'--max-seconds',str(caps[2]),'--source-commit',source,'--binding',str(bp)]
def binding_gate(binding,phase,name,source,paths,approval):
    if approval:require(binding.get('qa_retry_approval')=={'path':Path(approval).resolve().relative_to(ROOT).as_posix(),'sha256':sha(approval)},'exact explicit root repair authority binding')
    science=binding['immutable_science_files'];require(scientific_closure()<=set(science) and all(binding['files'].get(k)==h for k,h in science.items()),'complete immutable dependency closure')
    if approval:
        authority=read(approval);before=complete_record(BASE/'native_qa01/execution.json',True)['immutable_science_files']
        changed={k:{'before_sha256':before.get(k),'after_sha256':science.get(k)} for k in set(before)|set(science) if before.get(k)!=science.get(k)}
        require(changed and authority.get('declared_file_diff')==changed,'actual old/new scientific dependency diff must exactly equal root repair declaration')
    require(all(science.get(k)==h for k,h in previous_science().items()),'prior qualified identities retained')
    for k,v in pool_identity().items():require(binding.get(k)==v and binding['files'].get(v['path'])==v['sha256'],'pool and teacher fully bound')
    required=[Path(__file__),BASE/'make_invocation.py',BASE/'support_cost.json',PREVIOUS/'native_qa01/execution.json',ROOT/'experiments/native_process_receipt_20261020.ps1',*[ROOT/n for n in REPORTS],BASE/'commands'/f'{name}.json',BASE/'prior'/f'{name}.json',*paths]
    if approval:required.append(Path(approval).resolve())
    for p in required:require(binding['files'].get(p.relative_to(ROOT).as_posix())==sha(p),'exact required input')
    g=BASE/'generation'/f'{name}.json';require(binding.get('generation_path')==str(g) and binding.get('generation_sha256')==sha(g) and binding['files'].get(g.relative_to(ROOT).as_posix())==sha(g),'exact generation')
    require(binding.get('cost_formula')==FORMULA,'fixed cost timing policy')
    if phase!='qa':
        qp=next(p for p in paths if read(p)['phase']=='qa' and read(p).get('phase_gate_pass') is True);qa=complete_record(qp)
        require(qa['immutable_science_files']==science and all(qa['source_before'].get(k)==h==qa['source_after'].get(k) for k,h in science.items()),'same actual QA science')
        marker=BASE/qp.parent.name.removeprefix('native_')/'complete.json'
        require(binding.get('qa_result')=={'path':marker.relative_to(ROOT).as_posix(),'sha256':sha(marker)} and read(marker)==qa['complete'] and sha(marker)==qa['complete_sha256'] and read(marker)['result']['status']=='PASS','actual QA marker')
    if phase=='benchmark':
        items=binding.get('cold_probe_receipts',[]);expected=[p for p in paths if read(p)['phase']=='probe']
        require(items==[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in expected] and len(items)==3,'three exact cold receipts')
        wallpath=BASE/'cold_probe_wall.json';proof=read(wallpath)
        require(binding.get('cold_probe_wall_proof')=={'path':wallpath.relative_to(ROOT).as_posix(),'sha256':sha(wallpath)} and binding['files'].get(wallpath.relative_to(ROOT).as_posix())==sha(wallpath),'fixed outside-observed cold proof must be bound')
        require(proof.get('status')=='COMPLETED' and proof.get('source_commit')==source and len(proof.get('rows',[]))==3,'three actual full launch observations')
        for observed,p in zip(proof['rows'],expected):
            r=complete_record(p);wall=observed.get('full_startup_seconds');generation=observed.get('generation_shell_seconds');launch=observed.get('launch_shell_seconds')
            require(all(finite(x) and x>0 for x in (generation,launch,wall)) and wall==generation+launch,'actual generation and full launch timing sum')
            require(observed.get('native_receipt_path')==p.relative_to(ROOT).as_posix() and observed.get('native_receipt_sha256')==sha(p) and observed.get('attempt')==r.get('attempt') and finite(wall) and wall>=max(r['charged_seconds'],r['complete_launch_wall_seconds']),'outside launch identity/time must dominate actual internal/native durations')
def generate(phase,source,approval=None):
    started=time.perf_counter();require(not (BASE/'native_active.lock').exists(),'reconcile live owner')
    name,caps,paths,budget=phase_plan(phase,source,approval);require(not (BASE/name).exists() and not (BASE/('native_'+name)).exists(),'exclusive outputs')
    files=dict(previous_science());names=set(files)|scientific_closure()|{Path(__file__).relative_to(ROOT).as_posix(),(BASE/'make_invocation.py').relative_to(ROOT).as_posix(),(ROOT/'experiments/native_process_receipt_20261020.ps1').relative_to(ROOT).as_posix(),*REPORTS}
    def bind(p):
        p=Path(p).resolve();require(p.is_relative_to(ROOT) and p.suffix.lower() not in ('.h5','.hdf5'),'explicit bound input');files[p.relative_to(ROOT).as_posix()]=sha(p)
    command=BASE/'commands'/f'{name}.json';pr=BASE/'prior'/f'{name}.json';bp=BASE/'bindings'/f'{name}.json';g=BASE/'generation'/f'{name}.json'
    for p in (command,pr,bp,g):p.parent.mkdir(parents=True,exist_ok=True)
    args=arguments(phase,name,caps,source,bp);write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)});write(pr,[str(p) for p in paths])
    for n in names:bind(ROOT/n)
    for p in [BASE/'support_cost.json',PREVIOUS/'native_qa01/execution.json',command,pr,*paths]:bind(p)
    extras={**pool_identity(),'cost_formula':FORMULA,'qa_result':None}
    for key in ('pool_manifest','teacher_identity'):bind(ROOT/extras[key]['path']);names.add(extras[key]['path'])
    if approval:bind(approval);extras['qa_retry_approval']={'path':Path(approval).resolve().relative_to(ROOT).as_posix(),'sha256':sha(approval)}
    if phase!='qa':
        qp=next(p for p in paths if read(p)['phase']=='qa' and read(p).get('phase_gate_pass') is True);marker=BASE/qp.parent.name.removeprefix('native_')/'complete.json';bind(marker);extras['qa_result']={'path':marker.relative_to(ROOT).as_posix(),'sha256':sha(marker)}
    if phase=='benchmark':
        extras['cold_probe_receipts']=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in paths if read(p)['phase']=='probe']
        wallpath=BASE/'cold_probe_wall.json';bind(wallpath);extras['cold_probe_wall_proof']={'path':wallpath.relative_to(ROOT).as_posix(),'sha256':sha(wallpath)}
    write(g,{'phase':phase,'source_commit':source,'cost_formula':FORMULA,'no_launch':True,'seconds_before_record':time.perf_counter()-started});bind(g)
    binding={**extras,'files':dict(sorted(files.items())),'immutable_science_files':{k:files[k] for k in sorted(names)},'generation_path':str(g),'generation_sha256':sha(g)}
    binding_gate(binding,phase,name,source,paths,approval);write(bp,binding)
    argv=[str(BASE/'launch_stage.py'),'--phase',phase,'--source-commit',source]
    if approval:argv+=['--qa-retry-approval',str(Path(approval).resolve())]
    write(BASE/'commands'/f'{name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':argv,'arguments_line':subprocess.list2cmdline(argv)})
    print(json.dumps({'name':name,'caps':caps,'files':len(files),'seconds':time.perf_counter()-started,'no_launch':True}))
def execute(a):
    approval=a.qa_retry_approval.resolve() if a.qa_retry_approval else None
    name,caps,paths,budget=phase_plan(a.phase,a.source_commit,approval)
    a.output=BASE/('native_'+name);a.worker=BASE/name/'worker.json'
    commandpath=BASE/'commands'/f'{name}.json';priorpath=BASE/'prior'/f'{name}.json';bp=BASE/'bindings'/f'{name}.json'
    require(not a.output.exists() and not a.worker.parent.exists(),'exclusive outputs; no retry of existing attempt')
    binding=read(bp);command=read(commandpath)
    expected=arguments(a.phase,name,caps,a.source_commit,bp)
    require(command.get('arguments')==expected and command.get('arguments_line')==subprocess.list2cmdline(expected) and Path(command.get('executable','')).resolve()==ROOT/'.venv-gpu/Scripts/python.exe','exact fixed GPU command')
    require(read(priorpath)==[str(p) for p in paths],'complete exact prior list')
    binding_gate(binding,a.phase,name,a.source_commit,paths,approval);before=verify(binding)
    a.output.mkdir(exist_ok=False)
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    cap,timeout,selfcap=caps
    native=[str(ps),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ROOT/'experiments/native_process_receipt_20261020.ps1'),'-CommandJson',str(commandpath),'-WorkingDirectory',str(ROOT),'-Output',str(a.output/'inner'),'-TimeoutSeconds',str(timeout),'-MaxSeconds',str(cap),'-SourceCommit',a.source_commit,'-FixtureIdentityJson',str(a.worker)]
    record={'phase':a.phase,'attempt':name,'category':'metadata' if False else a.phase,'status':'RUNNING','source_commit':a.source_commit,'source_before':before,'immutable_science_files':binding['immutable_science_files'],'binding_sha256':sha(bp),'command_sha256':sha(commandpath),'worker_command':command,'budget_gate':budget,'native_cap_seconds':cap,'includes_children_once':True}
    write(a.output/'launch.json',record);started=time.perf_counter()
    try:
        sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
        from fixed_edge_pair_20261019 import native_job
        a.native_invoked=True
        outer=native_job(native,a.output/'outer',cap,worker_json=a.worker);record['outer']=outer;record['charged_seconds']=outer['seconds']
        inner=read(a.output/'inner/receipt.json');record['inner']=inner
        worker=read(a.worker);record['worker']=worker
        complete=read(a.worker.parent/'complete.json');record['complete']=complete;record['complete_sha256']=sha(a.worker.parent/'complete.json')
        record['source_after']=verify(binding);record['complete_launch_wall_seconds']=time.perf_counter()-LAUNCH_START
        okay=(outer.get('status')=='COMPLETED' and outer.get('exit_code')==inner.get('exit_code')==0 and inner.get('timeout') is False and outer.get('native_receipt_complete') is True and inner.get('native_receipt_complete') is True and inner.get('cleanup_verified') is True and inner.get('fixture_identity_after')==inner.get('supervised_identity_after')=='GONE' and worker.get('pid')==outer.get('actual_worker_pid')==inner.get('fixture_pid') and isinstance(worker.get('creation_filetime_utc'),int) and worker.get('creation_filetime_utc')==inner.get('fixture_creation_filetime_utc') and Path(worker.get('executable','')).resolve()==ROOT/'.venv-gpu/Scripts/python.exe' and complete.get('status')=='COMPLETED' and record['source_before']==record['source_after'])
        record['phase_gate_pass']=okay and complete.get('result',{}).get('status') in (('PASS','COMPLETED') if a.phase=='freeze-pool' else ('PASS',))
        record['status']='COMPLETED' if okay else 'UNRESOLVED_EXECUTION'
    except BaseException as exc:
        record.update(status='UNRESOLVED_EXECUTION',phase_gate_pass=False,error=repr(exc),traceback=traceback.format_exc())
        for key,path in [('outer',a.output/'outer/native.json'),('inner',a.output/'inner/receipt.json'),('worker',a.worker)]:
            if path.is_file():record[key]=read(path)
        if 'outer' in record:record['charged_seconds']=record['outer'].get('seconds')
        try:record['source_after']=verify(binding)
        except BaseException as e:record['source_after_error']=repr(e)
        write(a.output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
    finally:record['launcher_seconds']=time.perf_counter()-started;write(a.output/'execution.json',record)
    print(json.dumps({'status':record['status'],'phase':a.phase,'phase_gate_pass':record.get('phase_gate_pass',False),'charged_seconds':record.get('charged_seconds'),'receipt':str(a.output/'execution.json')}))
    return 0 if record.get('phase_gate_pass') is True else 1
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=PHASES,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--qa-retry-approval',type=Path);a=p.parse_args()
    require(BASE==(ROOT/'private_data/closure_mechanism_20261019/P0/edge_polynomial_runtime01').resolve(),'fixed study root')
    a.native_invoked=False
    owner=identity();BASE.mkdir(parents=True,exist_ok=True);lock=BASE/'native_active.lock';write(lock,owner)
    try:return execute(a)
    finally:
        require(read(lock)==owner,'owner identity changed; retain lock')
        safe=not a.native_invoked
        if a.native_invoked:
            receipt=a.output/'inner/receipt.json'
            if receipt.is_file():
                inner=read(receipt);safe=inner.get('native_receipt_complete') is True and inner.get('cleanup_verified') is True and inner.get('fixture_identity_after')==inner.get('supervised_identity_after')=='GONE'
                if a.worker.is_file():
                    worker=read(a.worker);safe=safe and worker.get('pid')==inner.get('fixture_pid') and worker.get('creation_filetime_utc')==inner.get('fixture_creation_filetime_utc')
        if safe:lock.unlink()
        else:raise RuntimeError('cleanup unknown; retain PID/creation-time owner lock')
if __name__=='__main__':raise SystemExit(main())


