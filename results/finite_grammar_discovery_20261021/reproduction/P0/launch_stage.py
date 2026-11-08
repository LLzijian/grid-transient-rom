"""Finite pool freeze, QA and cost native supervision; no scientific execution on import."""
import argparse,ast,ctypes,hashlib,json,math,os,subprocess,sys,time,traceback
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
PHASES=('freeze-pool','qa','cost')
REPORTS=('reports/finite_grammar_discovery_20261021_protocol.md','reports/finite_grammar_discovery_20261021_plan_review.md','reports/astra6_after_long_horizon_20261021.md','reports/finite_grammar_discovery_20261021_root_review_response.md','reports/finite_grammar_discovery_20261021_implementation_review.md','reports/finite_grammar_discovery_20261021_execution_setup.md','AGENTS.md')
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
    pending=[ROOT/'experiments/fixed_edge_pair_20261019.py',ROOT/'experiments/finite_grammar_discovery_20261021.py',ROOT/'tests/test_finite_grammar_discovery_20261021.py'];seen=set()
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
FORMULA={'formula':'1.25*T+180<=900','T':'2304*u_max+27*r_max+3*g_pair+3*v_final+c_all','margin':1.25,'P1_native_cap_seconds':720,'P1_complete_cap_seconds':900}
def previous_science():
    path=PREVIOUS/'native_qa01/execution.json';r=read(path);i=r.get('inner',{});o=r.get('outer',{})
    require(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and r.get('source_before')==r.get('source_after') and i.get('exit_code')==o.get('exit_code')==0 and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE','actual qualified preceding QA proof')
    science=r.get('immutable_science_files');require(isinstance(science,dict) and len(science)==96 and all(r['source_before'].get(k)==h for k,h in science.items()),'preceding96 actual scientific identities')
    return science
def complete_record(path,allow_failure=False):
    r=read(path);i=r.get('inner',{});o=r.get('outer',{})
    require(r.get('phase') in PHASES and finite(r.get('charged_seconds')) and r['charged_seconds']>=0 and i.get('native_receipt_complete') is True and o.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE' and r.get('source_before')==r.get('source_after'),'prior fee/source/ownership unresolved')
    if not allow_failure:require(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and i.get('exit_code')==o.get('exit_code')==0,'prior phase did not pass')
    return r
def phase_plan(phase,source,approval=None):
    require(approval is None,'QA02 authority withdrawn prospectively; unique QA only')
    require(phase in PHASES and len(source)==40 and all(c in '0123456789abcdef' for c in source),'phase and archived source')
    paths=sorted(BASE.glob('native_*/execution.json'))
    require({p.parent for p in paths}=={p.parent for p in BASE.glob('native_*/launch.json')},'pending prior launch requires reconciliation')
    rows=[complete_record(p,allow_failure=True) for p in paths]
    freeze=[r for r in rows if r['phase']=='freeze-pool'];qas=[r for r in rows if r['phase']=='qa'];costs=[r for r in rows if r['phase']=='cost']
    require(len(freeze)<=1 and len(qas)<=2 and len(costs)<=1,'duplicate phase receipts')
    spentqa=sum(r['charged_seconds'] for r in qas);spentmeta=sum(r['charged_seconds'] for r in freeze);spentcost=sum(r['charged_seconds'] for r in costs)
    require(spentqa<=450 and spentmeta<=60 and spentcost<=120,'registered phase costs exceeded')
    if phase=='freeze-pool':
        require(not rows and approval is None,'unique pool freeze before teacher or QA');name='freeze_pool01';caps=(60.,50.,40.)
    else:
        require(len(freeze)==1,'actual pool freeze precedes QA/cost');complete_record(BASE/'native_freeze_pool01/execution.json')
        if phase=='qa':
            require(not qas,'unique QA only; no retry after any outcome')
            require(not costs and len(qas)<2 and not any(r.get('phase_gate_pass') is True for r in qas),'QA cannot repeat success')
            if not qas:require(approval is None,'initial QA does not take repair authority');name='qa01';caps=(450.,435.,420.)
            else:
                require(approval is not None,'root targeted-retry approval required')
                approval=Path(approval).resolve();require(approval.is_relative_to(BASE),'local explicit approval required')
                authority=read(approval);failed=BASE/'native_qa01/execution.json'
                require(authority.get('status')=='APPROVED' and authority.get('target')=='qa02' and authority.get('source_commit')==source and authority.get('failed_execution_sha256')==sha(failed) and authority.get('targeted_checks_only') is True,'root bounded targeted repair authority missing')
                require(qas[0].get('phase_gate_pass') is False,'retry only after actual failed QA')
                remaining=450.-spentqa;require(remaining>30,'insufficient remaining cumulative QA cap');name='qa02';caps=(remaining,remaining-15,remaining-30)
        else:
            require(approval is None and not costs and qas and qas[-1].get('phase_gate_pass') is True,'unique cost requires actual QA pass');name='cost01';caps=(120.,110.,100.)
            passed=next(p for p,r in zip(paths,rows) if r['phase']=='qa' and r.get('phase_gate_pass') is True);qa=complete_record(passed)
            require(qa.get('source_commit')==source,'cost science commit differs from passed QA')
    support=read(BASE/'support_cost.json');require(support=={'support_held_seconds':1100,'metadata_cap_seconds':130,'QA_cap_seconds':450,'cost_cap_seconds':120,'cap_seconds':1800},'fixed1800 P0 ledger')
    require(1100+130+450+120<=1800,'full stage cap')
    require(0<caps[2]<caps[1]<caps[0]<=3600,'actual PS/self/native cap order')
    return name,caps,paths,{'prior_QA_seconds':spentqa,'prior_metadata_seconds':spentmeta,'prior_cost_seconds':spentcost,'support_hold':1100,'metadata_reserve':130,'QA_cap':450,'cost_cap':120,'total_cap':1800}
def pool_identity():
    folder=BASE/'freeze_pool01'
    p=folder/'pool.json';t=folder/'teacher.json';pool=read(p);teacher=read(t)
    require(isinstance(pool.get('eligible_pool'),list) and len(pool['eligible_pool'])>=6 and len(pool.get('selected',[]))==len(pool.get('keys',[]))==6 and len(set(pool['keys']))==6,'frozen six candidate pool')
    require(teacher.get('pool_sha256')==sha(p) and teacher.get('coefficient')==.15 and isinstance(teacher.get('index'),int) and not isinstance(teacher['index'],bool) and 0<=teacher['index']<6 and teacher.get('key')==pool['keys'][teacher['index']] and teacher.get('term')==pool['selected'][teacher['index']],'teacher selected only after immutable pool')
    return {'pool_manifest':{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)},'teacher_identity':{'path':t.relative_to(ROOT).as_posix(),'sha256':sha(t)}}
def arguments(phase,name,caps,source,binding):
    return ['-m','experiments.finite_grammar_discovery_20261021',phase,'--output',str(BASE/name),'--max-seconds',str(caps[2]),'--source-commit',source,'--binding',str(binding)]
def binding_gate(binding,phase,name,source,paths,approval):
    require(binding.get('cost_formula')==FORMULA,'registered cost formula binding')
    science=binding.get('immutable_science_files');require(isinstance(science,dict) and scientific_closure()<=set(science),'complete static scientific/native closure required')
    require(all(binding['files'].get(k)==h==science.get(k) for k,h in previous_science().items()),'preceding96 science must remain unchanged')
    required=[Path(__file__),BASE/'make_invocation.py',BASE/'support_cost.json',PREVIOUS/'native_qa01/execution.json',ROOT/'experiments/native_process_receipt_20261020.ps1',*[ROOT/n for n in REPORTS],BASE/'commands'/f'{name}.json',BASE/'prior'/f'{name}.json',*paths]
    if approval:required.append(Path(approval).resolve())
    for p in required:require(binding['files'].get(p.relative_to(ROOT).as_posix())==sha(p),'required exact explicit input absent')
    generation=BASE/'generation'/f'{name}.json'
    require(binding.get('generation_path')==str(generation) and binding.get('generation_sha256')==sha(generation) and binding['files'].get(generation.relative_to(ROOT).as_posix())==sha(generation),'generation identity changed')
    if approval:require(binding.get('qa_retry_approval')=={'path':Path(approval).resolve().relative_to(ROOT).as_posix(),'sha256':sha(approval)},'retry authority must be explicitly bound')
    if phase!='freeze-pool':
        freeze=complete_record(BASE/'native_freeze_pool01/execution.json')
        require(freeze.get('source_commit')==source,'frozen pool source commit differs')
        require(all(freeze['source_before'].get(k)==h==freeze['source_after'].get(k) and science.get(k)==h for k,h in freeze['immutable_science_files'].items()),'frozen-pool scientific dependencies differ from current phase')
        ids=pool_identity()
        for k,v in ids.items():require(binding.get(k)==v and binding['files'].get(v['path'])==v['sha256'],'actual frozen pool/teacher must be bound')
    if phase=='cost':
        qp=next(p for p in paths if read(p)['phase']=='qa' and read(p).get('phase_gate_pass') is True);qa=complete_record(qp)
        require(science==qa.get('immutable_science_files') and all(qa['source_before'].get(k)==h==qa['source_after'].get(k) for k,h in science.items()),'cost scientific identity differs from actual passed QA')
        marker=BASE/qp.parent.name.removeprefix('native_')/'complete.json';c=read(marker)
        require(binding.get('qa_result')=={'path':marker.relative_to(ROOT).as_posix(),'sha256':sha(marker)} and binding['files'].get(marker.relative_to(ROOT).as_posix())==sha(marker) and c==qa.get('complete') and sha(marker)==qa.get('complete_sha256') and c.get('status')=='COMPLETED' and c.get('result',{}).get('status')=='PASS','actual unchanged QA result must pass')
def generate(phase,source,approval=None):
    started=time.perf_counter();require(not (BASE/'native_active.lock').exists(),'reconcile live owner first')
    name,caps,paths,budget=phase_plan(phase,source,approval);output=BASE/name
    require(not output.exists() and not (BASE/('native_'+name)).exists(),'exclusive unique outputs')
    files=dict(previous_science());science_names=set(files)|scientific_closure()
    science_names|={Path(__file__).relative_to(ROOT).as_posix(),(BASE/'make_invocation.py').relative_to(ROOT).as_posix(),(ROOT/'experiments/native_process_receipt_20261020.ps1').relative_to(ROOT).as_posix(),*[p for p in REPORTS]}
    command=BASE/'commands'/f'{name}.json';prior=BASE/'prior'/f'{name}.json';bp=BASE/'bindings'/f'{name}.json';generation=BASE/'generation'/f'{name}.json'
    args=arguments(phase,name,caps,source,bp);command.parent.mkdir(parents=True,exist_ok=True);prior.parent.mkdir(parents=True,exist_ok=True)
    write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)});write(prior,[str(p) for p in paths])
    def bind(p):
        p=Path(p).resolve();require(p.is_relative_to(ROOT) and p.suffix.lower() not in ('.h5','.hdf5'),'explicit permitted files only');files[p.relative_to(ROOT).as_posix()]=sha(p)
    for k in science_names:bind(ROOT/k)
    for p in [BASE/'support_cost.json',PREVIOUS/'native_qa01/execution.json',command,prior,*paths]:bind(p)
    extras={'cost_formula':FORMULA,'qa_result':None}
    if approval:
        bind(approval);extras['qa_retry_approval']={'path':Path(approval).resolve().relative_to(ROOT).as_posix(),'sha256':sha(approval)}
    if phase!='freeze-pool':
        extras.update(pool_identity())
        for key in ('pool_manifest','teacher_identity'):bind(ROOT/extras[key]['path'])
        science_names|={extras[k]['path'] for k in ('pool_manifest','teacher_identity')}
    if phase=='cost':
        qp=next(p for p in paths if read(p)['phase']=='qa' and read(p).get('phase_gate_pass') is True)
        marker=BASE/qp.parent.name.removeprefix('native_')/'complete.json';bind(marker);extras['qa_result']={'path':marker.relative_to(ROOT).as_posix(),'sha256':sha(marker)}
    generation.parent.mkdir(parents=True,exist_ok=True);write(generation,{'phase':phase,'source_commit':source,'previous_QA_receipt_sha256':sha(PREVIOUS/'native_qa01/execution.json'),'registered_formula':FORMULA,'seconds_before_record':time.perf_counter()-started,'no_launch':True});bind(generation)
    binding={**extras,'files':dict(sorted(files.items())),'immutable_science_files':{k:files[k] for k in sorted(science_names)},'generation_path':str(generation),'generation_sha256':sha(generation)}
    binding_gate(binding,phase,name,source,paths,approval);bp.parent.mkdir(parents=True,exist_ok=True);write(bp,binding)
    argv=[str(BASE/'launch_stage.py'),'--phase',phase,'--source-commit',source]
    if approval:argv+=['--qa-retry-approval',str(Path(approval).resolve())]
    write(BASE/'commands'/f'{name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':argv,'arguments_line':subprocess.list2cmdline(argv)})
    print(json.dumps({'name':name,'source_files':len(files),'prior_count':len(paths),'caps':caps,'budget':budget,'seconds':time.perf_counter()-started,'no_launch':True}))

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
    record={'phase':a.phase,'attempt':name,'category':'metadata' if a.phase=='freeze-pool' else a.phase,'status':'RUNNING','source_commit':a.source_commit,'source_before':before,'immutable_science_files':binding['immutable_science_files'],'binding_sha256':sha(bp),'command_sha256':sha(commandpath),'worker_command':command,'budget_gate':budget,'native_cap_seconds':cap,'includes_children_once':True}
    write(a.output/'launch.json',record);started=time.perf_counter()
    try:
        sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
        from fixed_edge_pair_20261019 import native_job
        a.native_invoked=True
        outer=native_job(native,a.output/'outer',cap,worker_json=a.worker);record['outer']=outer;record['charged_seconds']=outer['seconds']
        inner=read(a.output/'inner/receipt.json');record['inner']=inner
        worker=read(a.worker);record['worker']=worker
        complete=read(a.worker.parent/'complete.json');record['complete']=complete;record['complete_sha256']=sha(a.worker.parent/'complete.json')
        record['source_after']=verify(binding)
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
    require(BASE==(ROOT/'private_data/closure_mechanism_20261019/P0/finite_grammar_discovery01').resolve(),'fixed study root')
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

