"""Four fixed rational/tanh three-seed runs, unified twelve-endpoint freeze, one evaluation."""
import argparse,hashlib,json,math,os,runpy,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
P0=ROOT/'private_data/closure_mechanism_20261019/P0/rational_control01'
SEEDS=(6301,6302,6303);GROUPS=('0_rational','0_tanh','1_rational','1_tanh');CAP={'run':(2750.,2730.,2700.),'evaluate':(650.,630.,600.)}
def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def support():
    path=P0/'launch_stage.py';expected=read(P0/'bindings/cost01.json')['files'][path.relative_to(ROOT).as_posix()]
    require(sha(path)==expected,'qualified P0 support source changed')
    return runpy.run_path(str(path))

def plan(phase,group,source):
    require(phase in CAP and len(source)==40 and all(c in '0123456789abcdef' for c in source),'registered phase/source')
    require((phase=='run' and group in GROUPS) or (phase=='evaluate' and group is None),'fixed group role')
    name='run_'+group if phase=='run' else 'evaluate01';output=BASE/name
    args=['-m','experiments.rational_control_20261021',phase,'--output',str(output),'--max-seconds',str(CAP[phase][2]),'--source-commit',source,'--binding',str(BASE/'bindings'/f'{name}.json')]
    if group is not None:args+=['--group',group]
    return name,output,args
def proof(source):
    b=read(P0/'bindings/cost01.json');q=read(P0/'native_qa01/execution.json');r=read(P0/'native_cost01/execution.json')
    for record in (q,r):
        i=record['inner'];o=record['outer']
        require(record.get('status')=='COMPLETED' and record.get('phase_gate_pass') is True and record.get('source_commit')==source and record.get('source_before')==record.get('source_after'),'actual same-source P0 execution')
        require(i.get('exit_code')==o.get('exit_code')==0 and i.get('native_receipt_complete') is True and o.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE','P0 actual native ownership')
    science=q['immutable_science_files']
    require(science==r['immutable_science_files'] and r['source_before']==b['files'] and all(q['source_before'].get(k)==h==r['source_before'].get(k) and sha(ROOT/k)==h for k,h in science.items()),'same exact QA/cost/current science')
    for phase,record,status in (('qa',q,'PASS'),('cost',r,'COMPLETED')):
        path=P0/(phase+'01')/'complete.json';c=read(path)
        require(c==record['complete'] and sha(path)==record['complete_sha256'] and c.get('status')=='COMPLETED' and c.get('result',{}).get('status')==status,'actual P0 marker')
    admission=read(BASE/'admission.json');payload=admission.get('result',admission)
    require(payload.get('status')=='PASS' and payload.get('P1_complete_cap_seconds')==12000 and payload.get('numerical_source_commit')==source and payload.get('immutable_science_files')==science,'root complete actual admission and numerical identity')
    return b,science
def prior(phase,group,source):
    paths=sorted(BASE.glob('native_*/execution.json'))
    require({p.parent for p in paths}=={p.parent for p in BASE.glob('native_*/launch.json')},'unreconciled launch')
    rows=[]
    for p in paths:
        r=read(p);i=r.get('inner',{});o=r.get('outer',{});value=r.get('charged_seconds')
        require(r.get('phase') in CAP and isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value) and 0<=value<=CAP[r['phase']][0],'actual prior fee')
        require(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and i.get('exit_code')==o.get('exit_code')==0 and i.get('native_receipt_complete') is True and o.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE' and r.get('source_before')==r.get('source_after') and r.get('source_commit')==source,'prior execution failure/ownership/source requires root assessment')
        rows.append(r)
    runs=[r for r in rows if r['phase']=='run'];evaluations=[r for r in rows if r['phase']=='evaluate']
    require(all(r.get('complete',{}).get('result',{}).get('all_group_frozen') is True and r['complete']['result'].get('source_commit')==source and r['complete']['result'].get('group')==r.get('group') for r in runs),'any incomplete512 group stops all later queues; no scientific retry')
    require(len({r.get('group') for r in runs})==len(runs) and len(evaluations)<=1,'duplicate role')
    if phase=='run':require(not evaluations and len(runs)<4 and {r['group'] for r in runs}==set(GROUPS[:len(runs)]) and group==GROUPS[len(runs)],'next fixed group once')
    else:require(len(runs)==4 and {r['group'] for r in runs}==set(GROUPS) and not evaluations,'four complete runs before once evaluate')
    require(read(BASE/'support_cost.json')=={'support_held_seconds':300,'metadata_cap_seconds':50,'run_cap_seconds':11000,'evaluate_cap_seconds':650,'cap_seconds':12000},'P1 fixed budget')
    spent=sum(r['charged_seconds'] for r in rows);remaining=2750*(4-len(runs))+650*(1-len(evaluations))
    require(spent+remaining+50+300<=12000,'P1 complete cumulative native/metadata/support budget')
    return paths
def selection(binding,source):
    path=BASE/'selection.json';require(binding.get('selection_manifest')=={'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)},'exact root unified selection')
    frozen=read(path);rows=frozen.get('rows',[])
    require(frozen.get('source_commit')==source and frozen.get('all_twelve_frozen') is True and len(rows)==12 and {r['key'] for r in rows}=={g+'_'+str(seed) for g in GROUPS for seed in SEEDS},'twelve fixed all-frozen endpoints')
    for group in GROUPS:
        groupdir=BASE/('run_'+group);actual=read(groupdir/'selection.json')
        require(actual.get('status')=='COMPLETED' and actual.get('source_commit')==source and actual.get('group')==group and actual.get('all_group_frozen') is True,'complete fixed group selection')
        require([r for r in rows if r['key'].startswith(group+'_')]==actual['rows'],'root selection must preserve exact group rows')
    for row in rows:
        group=row['key'].rsplit('_',1)[0];folder=Path(row['path']).resolve()
        require(row.get('status')=='FROZEN' and row['seed'] in SEEDS and row.get('noise')==int(group[0]) and row.get('form')==group[2:] and folder.is_relative_to(BASE/('run_'+group)) and folder.name=='checkpoint_00512' and set(row['files'])=={'model.json','training.pt'},'fixed endpoint role and both files')
        require(row['identity']['source_commit']==source and row['identity']['key']==row['key'],'endpoint source identity')
        for n,h in row['files'].items():require(sha(folder/n)==h,'frozen endpoint changed')
        for k in ('train_raw','trace'):require(Path(row[k]['path']).resolve().is_relative_to(BASE/('run_'+group)) and sha(row[k]['path'])==row[k]['sha256'],'frozen observations/trace changed')
    # Full step512/trace512/nonempty Adam and pair-data checks are performed by the frozen numerical module before val.
    return frozen
def bind_new(files,path):
    p=Path(path).resolve();require(p.is_relative_to(ROOT) and p.suffix.lower() not in ('.h5','.hdf5'),'explicit safe input');files[p.relative_to(ROOT).as_posix()]=sha(p)
def generate(phase,group,source):
    start=time.perf_counter();name,output,args=plan(phase,group,source);require(not (BASE/'native_active.lock').exists(),'reconcile owner')
    paths=prior(phase,group,source);old,science=proof(source);require(not output.exists() and not (BASE/('native_'+name)).exists(),'exclusive output')
    files=dict(old['files']);command=BASE/'commands'/f'{name}.json';pr=BASE/'prior'/f'{name}.json';bp=BASE/'bindings'/f'{name}.json'
    write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)});write(pr,[str(p) for p in paths])
    extras={'admission_result':{'path':(BASE/'admission.json').relative_to(ROOT).as_posix(),'sha256':sha(BASE/'admission.json')}}
    for p in [BASE/'admission.json',BASE/'support_cost.json',Path(__file__),BASE/'make_invocation.py',BASE/'observe_launch.ps1',P0/'bindings/cost01.json',P0/'native_qa01/execution.json',P0/'native_cost01/execution.json',P0/'qa01/complete.json',P0/'cost01/complete.json',command,pr,*paths]:bind_new(files,p)
    if phase=='evaluate':
        f=BASE/'selection.json';extras['selection_manifest']={'path':f.relative_to(ROOT).as_posix(),'sha256':sha(f)}
        frozen=selection(extras,source);bind_new(files,f)
        for g in GROUPS:
            for n in ('selection.json','result.json','complete.json'):bind_new(files,BASE/('run_'+g)/n)
        for row in frozen['rows']:
            for n in ('model.json','training.pt'):bind_new(files,Path(row['path'])/n)
            for k in ('train_raw','trace'):bind_new(files,row[k]['path'])
    generation=BASE/'generation'/f'{name}.json';write(generation,{'source_commit':source,'phase':phase,'group':group,'base_binding_sha256':sha(P0/'bindings/cost01.json'),'seconds_before_record':time.perf_counter()-start,'no_launch':True});bind_new(files,generation)
    binding={**old,**extras,'files':dict(sorted(files.items())),'immutable_science_files':science,'generation_path':str(generation),'generation_sha256':sha(generation)}
    write(bp,binding);argv=[str(BASE/'launch_stage.py'),'--phase',phase,'--source-commit',source]
    if group is not None:argv+=['--group',group]
    write(BASE/'commands'/f'{name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':argv,'arguments_line':subprocess.list2cmdline(argv)})
    print(json.dumps({'name':name,'seconds':time.perf_counter()-start,'source_files':len(files),'no_launch':True}))
def execute(phase,group,source):
    name,output,args=plan(phase,group,source);paths=prior(phase,group,source);old,science=proof(source);folder=BASE/('native_'+name);require(not output.exists() and not folder.exists(),'exclusive attempt')
    binding=read(BASE/'bindings'/f'{name}.json');command=read(BASE/'commands'/f'{name}.json');require(read(BASE/'prior'/f'{name}.json')==[str(p) for p in paths],'complete exact prior list')
    require(command['arguments']==args and command['arguments_line']==subprocess.list2cmdline(args) and Path(command['executable']).resolve()==ROOT/'.venv-gpu/Scripts/python.exe','exact fixed GPU argv')
    require(all(binding['files'].get(k)==h for k,h in old['files'].items()) and binding.get('immutable_science_files')==science,'immutable P0 sources differ')
    generation=BASE/'generation'/f'{name}.json';require(binding.get('generation_path')==str(generation) and binding.get('generation_sha256')==sha(generation) and binding['files'].get(generation.relative_to(ROOT).as_posix())==sha(generation),'exact generation identity required')
    for p in [Path(__file__),BASE/'make_invocation.py',BASE/'observe_launch.ps1',BASE/'admission.json',BASE/'support_cost.json',P0/'bindings/cost01.json',P0/'native_qa01/execution.json',P0/'native_cost01/execution.json',P0/'qa01/complete.json',P0/'cost01/complete.json',BASE/'commands'/f'{name}.json',BASE/'prior'/f'{name}.json',*paths]:require(binding['files'].get(p.relative_to(ROOT).as_posix())==sha(p),'required P1 input bound')
    require(binding.get('admission_result')=={'path':(BASE/'admission.json').relative_to(ROOT).as_posix(),'sha256':sha(BASE/'admission.json')},'actual admission frozen')
    if phase=='evaluate':
        frozen=selection(binding,source)
        selection_path=BASE/'selection.json';require(binding['files'].get(selection_path.relative_to(ROOT).as_posix())==sha(selection_path),'unified twelve-endpoint selection file must be source bound')
        for group_name in GROUPS:
            for filename in ('selection.json','result.json','complete.json'):
                path=BASE/('run_'+group_name)/filename;require(binding['files'].get(path.relative_to(ROOT).as_posix())==sha(path),'all four complete group result files must be bound')
        for row in frozen['rows']:
            for n in ('model.json','training.pt'):require(binding['files'].get((Path(row['path'])/n).resolve().relative_to(ROOT).as_posix())==sha(Path(row['path'])/n),'all endpoint bytes bound')
            for key in ('train_raw','trace'):require(binding['files'].get(Path(row[key]['path']).resolve().relative_to(ROOT).as_posix())==row[key]['sha256'],'all observation/trace identities bound')
    h=support();before=h['verify'](binding);folder.mkdir(exist_ok=False);worker=output/'worker.json';cap,timeout,selfcap=CAP[phase]
    require(0<selfcap<timeout<cap<=3600,'fixed PS budget boundary')
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    native=[str(ps),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ROOT/'experiments/native_process_receipt_20261020.ps1'),'-CommandJson',str(BASE/'commands'/f'{name}.json'),'-WorkingDirectory',str(ROOT),'-Output',str(folder/'inner'),'-TimeoutSeconds',str(timeout),'-MaxSeconds',str(cap),'-SourceCommit',source,'-FixtureIdentityJson',str(worker)]
    record={'phase':phase,'category':phase,'group':group,'status':'RUNNING','source_commit':source,'source_before':before,'immutable_science_files':science,'binding_sha256':sha(BASE/'bindings'/f'{name}.json'),'command_sha256':sha(BASE/'commands'/f'{name}.json'),'worker_command':command,'native_cap_seconds':cap,'ps_timeout_seconds':timeout,'worker_max_seconds':selfcap,'includes_children_once':True};write(folder/'launch.json',record);started=time.perf_counter()
    try:
        sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
        from fixed_edge_pair_20261019 import native_job
        RUN['spawned']=True
        o=native_job(native,folder/'outer',cap,worker_json=worker);record['outer']=o;record['charged_seconds']=o['seconds']
        i=read(folder/'inner/receipt.json');record['inner']=i;w=read(worker);record['worker']=w;c=read(output/'complete.json');record['complete']=c;record['source_after']=h['verify'](binding)
        okay=o.get('status')=='COMPLETED' and i.get('status')=='COMPLETED' and i.get('timeout') is False and o.get('exit_code')==i.get('exit_code')==0 and i.get('native_receipt_complete') is True and o.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE' and w.get('pid')==i.get('fixture_pid')==o.get('actual_worker_pid') and isinstance(w.get('creation_filetime_utc'),int) and w.get('creation_filetime_utc')==i.get('fixture_creation_filetime_utc') and Path(w.get('executable','')).resolve()==ROOT/'.venv-gpu/Scripts/python.exe' and record['source_before']==record['source_after']
        record['status']='COMPLETED' if okay else 'UNRESOLVED_EXECUTION';record['phase_gate_pass']=okay and c.get('status')=='COMPLETED' and c.get('result',{}).get('status')=='COMPLETED'
    except BaseException as exc:
        record.update(status='UNRESOLVED_EXECUTION',phase_gate_pass=False,error=repr(exc),traceback=traceback.format_exc())
        for key,p in [('inner',folder/'inner/receipt.json'),('outer',folder/'outer/native.json'),('worker',worker)]:
            if p.is_file():record[key]=read(p)
        if 'outer' in record:record['charged_seconds']=record['outer'].get('seconds')
        try:record['source_after']=h['verify'](binding)
        except BaseException as e:record['source_after_error']=repr(e)
        write(folder/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
    finally:record['launcher_seconds']=time.perf_counter()-started;write(folder/'execution.json',record)
    print(json.dumps({'status':record['status'],'phase_gate_pass':record.get('phase_gate_pass'),'charged_seconds':record.get('charged_seconds'),'receipt':str(folder/'execution.json')}))
    return 0 if record.get('phase_gate_pass') else 1
RUN={'spawned':False}
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=tuple(CAP),required=True);p.add_argument('--group',choices=GROUPS);p.add_argument('--source-commit',required=True);a=p.parse_args();name,output,args=plan(a.phase,a.group,a.source_commit)
    h=support();owner=h['identity']();lock=BASE/'native_active.lock';BASE.mkdir(parents=True,exist_ok=True);write(lock,owner)
    try:return execute(a.phase,a.group,a.source_commit)
    finally:
        require(read(lock)==owner,'owner changed; retain lock');safe=not RUN['spawned'];rp=BASE/('native_'+name)/'inner/receipt.json'
        if RUN['spawned'] and rp.is_file():
            i=read(rp);safe=i.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE'
            if (output/'worker.json').is_file():
                w=read(output/'worker.json');safe=safe and w.get('pid')==i.get('fixture_pid') and w.get('creation_filetime_utc')==i.get('fixture_creation_filetime_utc')
        if safe:lock.unlink()
        else:raise RuntimeError('missing cleanup proof; retain owner lock for reconciliation')
if __name__=='__main__':raise SystemExit(main())



