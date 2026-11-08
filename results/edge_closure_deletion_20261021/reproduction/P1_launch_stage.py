"""Finite P1 empty-closure execution; immutable qualified P0 scientific sources."""
import argparse,hashlib,json,math,os,runpy,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
P0=ROOT/'private_data/closure_mechanism_20261019/P0/edge_closure_deletion01'
SEEDS=(5101,5102,5103);CAP={'train':(500.,480.,450.),'freeze':(50.,45.,40.),'evaluate':(50.,45.,35.)}
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
def plan(phase,seed,source):
    require(phase in CAP and len(source)==40 and all(c in '0123456789abcdef' for c in source),'fixed P1 phase/full source required')
    require((phase=='freeze' and seed is None) or (phase!='freeze' and seed in SEEDS),'exact seed role required')
    name='freeze01' if phase=='freeze' else ('train' if phase=='train' else 'eval')+str(seed)
    output=BASE/'freeze01' if phase=='freeze' else BASE/('models' if phase=='train' else 'evaluation')/f'{seed}_empty'
    args=['-m','experiments.edge_closure_deletion_20261021',phase,'--output',str(output),'--max-seconds',str(CAP[phase][2]),'--source-commit',source,'--binding',str(BASE/'bindings'/f'{name}.json')]
    if seed is not None:args+=['--seed',str(seed)]
    if phase=='evaluate':args+=['--model',str(endpoint(seed)/'model.json')]
    return name,output,args
def endpoint(seed):return BASE/'models'/f'{seed}_empty/empty_{seed}/checkpoint_00512'
def proof():
    b=read(P0/'bindings/cost01.json');require(len(b['files'])==104,'fixed P0 source table count')
    r=read(P0/'native_cost01/execution.json');q=read(P0/'native_qa01/execution.json')
    for record in (r,q):
        require(record.get('status')=='COMPLETED' and record.get('phase_gate_pass') is True and record['source_before']==record['source_after'],'actual P0 source/pass proof missing')
        require(record['inner']['exit_code']==record['outer']['exit_code']==0 and record['inner']['cleanup_verified'] is True and record['inner']['fixture_identity_after']==record['inner']['supervised_identity_after']=='GONE','P0 cleanup/exit proof missing')
    require(r['source_before']==b['files'],'P0 actual cost source map differs')
    science=q['immutable_science_files'];require(len(science)==98 and science.get('experiments/edge_closure_deletion_20261021.py')=='831785b3edcf67390ef3935af25cc22c5a00e0704255bf67b1c561330a749f41' and q.get('source_commit')==r.get('source_commit')=='0a6ef1c5f01abfb28d38b90d18131d3d565a254c','fixed actual P0 scientific source required');require(science==r['immutable_science_files'] and all(q['source_before'].get(k)==h==r['source_before'].get(k) for k,h in science.items()),'actual QA/cost scientific identity differs')
    admission=read(BASE/'admission.json');require(admission.get('status')=='PASS' and admission.get('P1_complete_cap_seconds')==2450,'P1 root admission missing')
    require(admission.get('numerical_source_commit')==r.get('source_commit')==q.get('source_commit'),'admission numerical source differs from actual P0')
    require(admission.get('numerical_runner_sha256')==science.get('experiments/edge_closure_deletion_20261021.py') and admission.get('tests_sha256')==science.get('tests/test_edge_closure_deletion_20261021.py'),'admission scientific file hashes differ from actual QA/cost')
    marker=P0/'cost01/complete.json';complete=read(marker)
    require(complete==r.get('complete') and sha(marker)==r.get('complete_sha256') and complete.get('status')=='COMPLETED' and complete.get('result',{}).get('status')=='PASS','actual cost completion identity/PASS differs')
    return b,science
def prior(phase,seed,source):
    paths=sorted(BASE.glob('native_*/execution.json'))
    require({p.parent for p in paths}=={p.parent for p in BASE.glob('native_*/launch.json')},'pending/unreconciled prior')
    records=[]
    for p in paths:
        r=read(p);i=r.get('inner',{});o=r.get('outer',{});s=r.get('charged_seconds')
        require(r.get('phase') in CAP and isinstance(s,(int,float)) and not isinstance(s,bool) and math.isfinite(s) and 0<=s<=CAP[r['phase']][0],'invalid prior fee')
        require(i.get('native_receipt_complete') is True and o.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE','prior ownership unresolved')
        require(r.get('source_commit')==source and r.get('source_before')==r.get('source_after'),'prior source differs');records.append(r)
    trains=[r for r in records if r['phase']=='train'];evaluations=[r for r in records if r['phase']=='evaluate'];freezes=[r for r in records if r['phase']=='freeze']
    require(len({r['seed'] for r in trains})==len(trains) and len({r['seed'] for r in evaluations})==len(evaluations) and len(freezes)<=1,'duplicate phase role')
    require(all(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True for r in records),'failed prior stops the fixed study; no partial continuation')
    if phase=='train':require(not freezes and not evaluations and len(trains)<3 and {r['seed'] for r in trains}==set(SEEDS[:len(trains)]) and seed==SEEDS[len(trains)],'strict next-seed train, no retry')
    else:
        require(len(trains)==3 and {r['seed'] for r in trains}==set(SEEDS) and all(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True for r in trains),'all three valid512 endpoints required')
        if phase=='freeze':require(not freezes and not evaluations,'freeze is unique and precedes evaluation')
        else:require(len(freezes)==1 and freezes[0].get('status')=='COMPLETED' and freezes[0].get('phase_gate_pass') is True and len(evaluations)<3 and {r['seed'] for r in evaluations}==set(SEEDS[:len(evaluations)]) and seed==SEEDS[len(evaluations)],'actual unique freeze and next once-evaluation required')
    cost=read(BASE/'metadata_cost.json');prep=cost['preparation_charged_or_held_seconds'];closure=cost['closure_reserved_seconds']
    require(cost['metadata_cap_seconds']==100 and cost['freeze_cap_seconds']==50 and prep==40 and closure==10 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0 for x in (prep,closure)) and prep+50+closure<=100,'metadata preparation/freeze/closure budget')
    train_spent=sum(r['charged_seconds'] for r in trains);eval_spent=sum(r['charged_seconds'] for r in evaluations);freeze_spent=sum(r['charged_seconds'] for r in freezes)
    require(train_spent+500*(3-len(trains))<=1500 and eval_spent+50*(3-len(evaluations))<=150 and prep+(freeze_spent if freezes else 50)+closure<=100,'category cumulative cap')
    require(train_spent+500*(3-len(trains))+eval_spent+50*(3-len(evaluations))+100+700<=2450,'complete P1 cap; metadata counted once')
    return paths
def bind_new(files,path):
    p=Path(path).resolve();require(p.is_relative_to(ROOT) and p.suffix.lower() not in ('.h5','.hdf5'),'forbidden explicit input');files[p.relative_to(ROOT).as_posix()]=sha(p)

OLD_SELECTION=ROOT/'results/edge_memory_discrimination_20261021/frozen_selection.json'
OLD_SELECTION_SHA='d1706d146857946a771596adbc3576df5fb2b0014fd6f21ceb8a7aa9064a1c61'
IDENTITY_LIMIT=65536
def selection_files():
    require(sha(OLD_SELECTION)==OLD_SELECTION_SHA,'exact old cosine selection required')
    rows=read(OLD_SELECTION)['rows'];require(len(rows)==3 and {r['seed'] for r in rows}==set(SEEDS),'three fixed old cosine roles')
    result={OLD_SELECTION.relative_to(ROOT).as_posix():OLD_SELECTION_SHA}
    for row in rows:
        folder=Path(row['path']);require(folder.is_absolute() and folder.is_relative_to(ROOT) and folder.name=='checkpoint_00512' and folder.parent.name==f"cos_{row['seed']}",'exact selected cosine endpoint role')
        require(set(row['files'])=={'model.json','training.pt'},'exact old endpoint file pair')
        for name,expected in row['files'].items():
            path=folder/name;require(sha(path)==expected,'selected cosine endpoint changed');result[path.relative_to(ROOT).as_posix()]=expected
    return result
def identity_bound(binding,seed,source):
    contracts=read(P0/'qa01/complete.json')['result']['contracts']
    require(len(contracts)==3 and {r['seed'] for r in contracts}==set(SEEDS),'actual QA fixture identities')
    row=next(r for r in contracts if r['seed']==seed)
    require(row['fixture']['seed']==seed,'actual fixture seed mismatch')
    value={'source_commit':source,'binding':binding,'seed':seed,'fixture':row['fixture'],'terms':[],'closure_coefficients':0}
    n=len(json.dumps(value,sort_keys=True,allow_nan=False).encode('utf8'))
    require(n+512*32+4096<=IDENTITY_LIMIT,'final formal identity exceeds cost-measured padding capacity')
    return {'identity_json_bytes':n,'trace_allowance_bytes':512*32,'reserve_bytes':4096,'identity_padding_bytes':IDENTITY_LIMIT,'within_measured_capacity':True}
def json_bytes(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')
def write_bytes(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data)

def generate(phase,seed,source):
    started=time.perf_counter();name,output,args=plan(phase,seed,source);require(not (BASE/'native_active.lock').exists(),'reconcile current owner first')
    paths=prior(phase,seed,source);old,science=proof();require(not output.exists() and not (BASE/('native_'+name)).exists(),'exclusive output')
    files=dict(old['files']);files.update(selection_files());command=BASE/'commands'/f'{name}.json';pr=BASE/'prior'/f'{name}.json';bp=BASE/'bindings'/f'{name}.json'
    write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)});write(pr,[str(p) for p in paths])
    for p in [Path(__file__),BASE/'make_invocation.py',BASE/'metadata_cost.json',BASE/'admission.json',P0/'bindings/cost01.json',P0/'native_cost01/execution.json',P0/'cost01/complete.json',P0/'native_qa01/execution.json',P0/'qa01/complete.json',OLD_SELECTION,ROOT/'reports/edge_closure_deletion_20261021_P1_registration.md',command,pr,*paths]:bind_new(files,p)
    extras={}
    if phase in ('freeze','evaluate'):
        extras['endpoints']={str(s):str(endpoint(s)) for s in SEEDS}
        for s in SEEDS:
            for step in (0,512):
                folder=endpoint(s).parent/f'checkpoint_{step:05d}'
                for n in ('model.json','training.pt'):bind_new(files,folder/n)
    if phase=='evaluate':
        frozen=BASE/'freeze01/frozen_selection.json';bind_new(files,frozen);extras.update(freeze_manifest=str(frozen),freeze_manifest_sha256=sha(frozen))
    generation=BASE/'generation'/f'{name}.json'
    gen={'source_commit':source,'phase':phase,'seed':seed,'base_binding_sha256':sha(P0/'bindings/cost01.json'),'seconds_before_record':time.perf_counter()-started,'no_launch':True,'identity_capacity':{'identity_padding_bytes':65536,'trace_allowance_bytes':512*32,'reserve_bytes':4096}}
    raw=json_bytes(gen);generation_sha=hashlib.sha256(raw).hexdigest();files[generation.relative_to(ROOT).as_posix()]=generation_sha
    binding={**old,**extras,'files':dict(sorted(files.items())),'admission':{'path':str(BASE/'admission.json'),'sha256':sha(BASE/'admission.json')},'old_selection':{'path':OLD_SELECTION.relative_to(ROOT).as_posix(),'sha256':OLD_SELECTION_SHA},'immutable_science_files':science,'generation_path':str(generation),'generation_sha256':generation_sha}
    padding=identity_bound(binding,seed,source) if phase=='train' else None
    write_bytes(generation,raw)
    write(bp,binding)
    launch=[str(BASE/'launch_stage.py'),'--phase',phase,'--source-commit',source]
    if seed is not None:launch+=['--seed',str(seed)]
    write(BASE/'commands'/f'{name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':launch,'arguments_line':subprocess.list2cmdline(launch)})
    print(json.dumps({'name':name,'files':len(files),'seconds':time.perf_counter()-started,'no_launch':True,'identity_capacity_proof':padding}))
def execute(phase,seed,source):
    name,output,args=plan(phase,seed,source);paths=prior(phase,seed,source);old,science=proof();folder=BASE/('native_'+name);require(not output.exists() and not folder.exists(),'exclusive attempt')
    binding=read(BASE/'bindings'/f'{name}.json');command=read(BASE/'commands'/f'{name}.json');require(read(BASE/'prior'/f'{name}.json')==[str(p) for p in paths],'complete exact prior list')
    require(command['arguments']==args and command['arguments_line']==subprocess.list2cmdline(args) and Path(command['executable']).resolve()==ROOT/'.venv-gpu/Scripts/python.exe','exact fixed GPU argv')
    require(all(binding['files'].get(k)==h for k,h in old['files'].items()) and binding.get('immutable_science_files')==science,'immutable P0 sources differ')
    generation=BASE/'generation'/f'{name}.json';require(binding.get('generation_path')==str(generation) and binding.get('generation_sha256')==sha(generation) and binding['files'].get(generation.relative_to(ROOT).as_posix())==sha(generation),'exact generation identity required')
    for p in [Path(__file__),BASE/'make_invocation.py',BASE/'admission.json',BASE/'metadata_cost.json',BASE/'commands'/f'{name}.json',BASE/'prior'/f'{name}.json',*paths]:require(binding['files'].get(p.relative_to(ROOT).as_posix())==sha(p),'required new binding absent')
    if phase in ('freeze','evaluate'):
        require(binding.get('endpoints')=={str(s):str(endpoint(s)) for s in SEEDS},'three exact role endpoints required')
        for s in SEEDS:
            for n in ('model.json','training.pt'):require(binding['files'].get((endpoint(s)/n).relative_to(ROOT).as_posix())==sha(endpoint(s)/n),'frozen endpoint changed')
    if phase=='evaluate':
        f=BASE/'freeze01/frozen_selection.json';require(binding.get('freeze_manifest')==str(f) and binding.get('freeze_manifest_sha256')==sha(f) and binding['files'].get(f.relative_to(ROOT).as_posix())==sha(f),'exact frozen selection required')
    require(binding.get('old_selection')=={'path':OLD_SELECTION.relative_to(ROOT).as_posix(),'sha256':OLD_SELECTION_SHA},'exact old train selection identity required')
    require(all(binding['files'].get(k)==v for k,v in selection_files().items()),'old selection and selected endpoints must remain bound')
    if phase=='train':identity_bound(binding,seed,source)
    h=support();before=h['verify'](binding);folder.mkdir(exist_ok=False);worker=output/'worker.json';cap,timeout,selfcap=CAP[phase]
    require(0<selfcap<timeout<cap<=3600,'fixed PS budget boundary')
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    native=[str(ps),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ROOT/'experiments/native_process_receipt_20261020.ps1'),'-CommandJson',str(BASE/'commands'/f'{name}.json'),'-WorkingDirectory',str(ROOT),'-Output',str(folder/'inner'),'-TimeoutSeconds',str(timeout),'-MaxSeconds',str(cap),'-SourceCommit',source,'-FixtureIdentityJson',str(worker)]
    record={'phase':phase,'category':phase,'seed':seed,'status':'RUNNING','source_commit':source,'source_before':before,'binding_sha256':sha(BASE/'bindings'/f'{name}.json'),'command_sha256':sha(BASE/'commands'/f'{name}.json'),'worker_command':command,'native_cap_seconds':cap,'ps_timeout_seconds':timeout,'worker_max_seconds':selfcap,'includes_children_once':True};write(folder/'launch.json',record);started=time.perf_counter()
    try:
        sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
        from fixed_edge_pair_20261019 import native_job
        RUN['spawned']=True
        o=native_job(native,folder/'outer',cap,worker_json=worker);record['outer']=o;record['charged_seconds']=o['seconds']
        i=read(folder/'inner/receipt.json');record['inner']=i;w=read(worker);record['worker']=w;c=read(output/'complete.json');record['complete']=c;record['source_after']=h['verify'](binding)
        okay=o.get('status')=='COMPLETED' and i.get('status')=='COMPLETED' and i.get('timeout') is False and o.get('exit_code')==i.get('exit_code')==0 and i.get('native_receipt_complete') is True and o.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE' and w.get('pid')==i.get('fixture_pid')==o.get('actual_worker_pid') and isinstance(w.get('creation_filetime_utc'),int) and w.get('creation_filetime_utc')==i.get('fixture_creation_filetime_utc') and Path(w.get('executable','')).resolve()==ROOT/'.venv-gpu/Scripts/python.exe' and record['source_before']==record['source_after']
        record['status']='COMPLETED' if okay else 'UNRESOLVED_EXECUTION';record['phase_gate_pass']=okay and c.get('status')=='COMPLETED' and c.get('result',{}).get('status')=='COMPLETED'
        if phase=='train':record['phase_gate_pass']=record['phase_gate_pass'] and c.get('result',{}).get('terms')==[] and c.get('result',{}).get('closure_coefficients')==0
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
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=tuple(CAP),required=True);p.add_argument('--seed',type=int);p.add_argument('--source-commit',required=True);a=p.parse_args();name,output,args=plan(a.phase,a.seed,a.source_commit)
    h=support();owner=h['identity']();lock=BASE/'native_active.lock';BASE.mkdir(parents=True,exist_ok=True);write(lock,owner)
    try:return execute(a.phase,a.seed,a.source_commit)
    finally:
        require(read(lock)==owner,'owner changed; retain lock');safe=not RUN['spawned'];rp=BASE/('native_'+name)/'inner/receipt.json'
        if RUN['spawned'] and rp.is_file():
            i=read(rp);safe=i.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE'
            if (output/'worker.json').is_file():
                w=read(output/'worker.json');safe=safe and w.get('pid')==i.get('fixture_pid') and w.get('creation_filetime_utc')==i.get('fixture_creation_filetime_utc')
        if safe:lock.unlink()
        else:raise RuntimeError('missing cleanup proof; retain owner lock for reconciliation')
if __name__=='__main__':raise SystemExit(main())

