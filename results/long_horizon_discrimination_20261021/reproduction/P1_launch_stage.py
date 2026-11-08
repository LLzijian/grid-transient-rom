"""Three once-only P1 long-horizon evaluations; immutable qualified P0 scientific sources."""
import argparse,hashlib,json,math,os,runpy,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
P0=ROOT/'private_data/closure_mechanism_20261019/P0/long_horizon_discrimination01'
SEEDS=(5101,5102,5103);CAP={'evaluate':(280.,260.,240.)}
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
    require(phase=='evaluate' and seed in SEEDS and len(source)==40 and all(c in '0123456789abcdef' for c in source),'exact evaluation seed and full source required')
    name='eval'+str(seed);output=BASE/'evaluation'/str(seed)
    args=['-m','experiments.long_horizon_discrimination_20261021',phase,'--output',str(output),'--max-seconds','240.0','--source-commit',source,'--binding',str(BASE/'bindings'/f'{name}.json'),'--seed',str(seed)]
    return name,output,args
def proof():
    b=read(P0/'bindings/cost01.json');require(len(b['files'])==102,'fixed P0 source table count')
    r=read(P0/'native_cost01/execution.json');q=read(P0/'native_qa01/execution.json')
    for record in (r,q):
        require(record.get('status')=='COMPLETED' and record.get('phase_gate_pass') is True and record['source_before']==record['source_after'],'actual P0 source/pass proof missing')
        require(record['inner']['exit_code']==record['outer']['exit_code']==0 and record['inner'].get('native_receipt_complete') is True and record['outer'].get('native_receipt_complete') is True and record['inner']['cleanup_verified'] is True and record['inner']['fixture_identity_after']==record['inner']['supervised_identity_after']=='GONE','P0 cleanup/exit proof missing')
    require(r['source_before']==b['files'],'P0 actual cost source map differs')
    science=q['immutable_science_files'];require(len(science)==96 and science.get('experiments/long_horizon_discrimination_20261021.py')=='02d314e0f5d2bed7832977d4aa37c63969c2dc44e1ab3205b68e7e2a53f6d5c9' and q.get('source_commit')==r.get('source_commit')=='4533cbab3131b681a892afc4849b0742770883c5','fixed actual P0 scientific source required');require(science==r['immutable_science_files'] and all(q['source_before'].get(k)==h==r['source_before'].get(k) for k,h in science.items()),'actual QA/cost scientific identity differs')
    admission=read(BASE/'admission.json');require(admission.get('status')=='PASS' and admission.get('P1_complete_cap_seconds')==1400,'P1 root admission missing')
    require(admission.get('numerical_source_commit')==r.get('source_commit')==q.get('source_commit'),'admission numerical source differs from actual P0')
    require(admission.get('numerical_runner_sha256')==science.get('experiments/long_horizon_discrimination_20261021.py') and admission.get('tests_sha256')==science.get('tests/test_long_horizon_discrimination_20261021.py'),'admission scientific file hashes differ from actual QA/cost')
    marker=P0/'cost01/complete.json';complete=read(marker)
    require(complete==r.get('complete') and sha(marker)==r.get('complete_sha256') and complete.get('status')=='COMPLETED' and complete.get('result',{}).get('status')=='PASS','actual cost completion identity/PASS differs')
    require(science.get('tests/test_long_horizon_discrimination_20261021.py')=='e4198f2361a4c8e10feec4a8b971d7017539b7d0c9158eafde1d106e9e548cb5' and all(sha(ROOT/k)==h for k,h in science.items()),'qualified96 scientific bytes changed')
    qmarker=P0/'qa01/complete.json';qcomplete=read(qmarker)
    require(qcomplete==q.get('complete') and sha(qmarker)==q.get('complete_sha256') and qcomplete.get('status')=='COMPLETED' and qcomplete.get('result',{}).get('status')=='PASS','actual unchanged QA PASS')
    manifest=P0/'endpoints.json';require(sha(manifest)=='42c624470156e2e2b438671131f3ca7aecfed834b00b9344b7a09a2ac44ad627','nine endpoints manifest changed')
    require(b.get('endpoint_manifest')=={'path':manifest.relative_to(ROOT).as_posix(),'sha256':sha(manifest)} and b['files'].get(manifest.relative_to(ROOT).as_posix())==sha(manifest) and all(b['files'].get(k)==h for k,h in read(manifest)['files'].items()),'frozen endpoint identities must remain fully bound')
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
    require(len(records)<3 and len({r['seed'] for r in records})==len(records) and {r['seed'] for r in records}==set(SEEDS[:len(records)]) and seed==SEEDS[len(records)],'strict next-seed once-evaluation; no extra queues')
    require(all(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and r.get('inner',{}).get('exit_code')==r.get('outer',{}).get('exit_code')==0 for r in records),'execution failure requires root assessment; no retry')
    cost=read(BASE/'metadata_cost.json')
    require(cost=={'metadata_cap_seconds':40,'metadata_charged_or_held_seconds':40,'summary_cap_seconds':60,'support_held_seconds':460,'total_cap_seconds':1400},'exact registered P1 budget contract')
    spent=sum(r['charged_seconds'] for r in records)
    require(spent+280*(3-len(records))<=840 and spent+280*(3-len(records))+40+60+460<=1400,'complete P1 budget; metadata counted once')
    return paths
def bind_new(files,path):
    p=Path(path).resolve();require(p.is_relative_to(ROOT) and p.suffix.lower() not in ('.h5','.hdf5'),'forbidden explicit input');files[p.relative_to(ROOT).as_posix()]=sha(p)

def generate(phase,seed,source):
    started=time.perf_counter();name,output,args=plan(phase,seed,source);require(not (BASE/'native_active.lock').exists(),'reconcile current owner first')
    paths=prior(phase,seed,source);old,science=proof();require(not output.exists() and not (BASE/('native_'+name)).exists(),'exclusive output')
    files=dict(old['files']);command=BASE/'commands'/f'{name}.json';pr=BASE/'prior'/f'{name}.json';bp=BASE/'bindings'/f'{name}.json'
    write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)});write(pr,[str(p) for p in paths])
    for p in [Path(__file__),BASE/'make_invocation.py',BASE/'metadata_cost.json',BASE/'admission.json',P0/'bindings/cost01.json',P0/'native_cost01/execution.json',P0/'cost01/complete.json',P0/'native_qa01/execution.json',P0/'qa01/complete.json',ROOT/'reports/long_horizon_discrimination_20261021_P1_admission.md',command,pr,*paths]:bind_new(files,p)
    generation=BASE/'generation'/f'{name}.json'
    write(generation,{'source_commit':source,'phase':phase,'seed':seed,'base_binding_sha256':sha(P0/'bindings/cost01.json'),'seconds_before_record':time.perf_counter()-started,'no_launch':True});bind_new(files,generation)
    binding={**old,'files':dict(sorted(files.items())),'admission':{'path':str(BASE/'admission.json'),'sha256':sha(BASE/'admission.json')},'immutable_science_files':science,'generation_path':str(generation),'generation_sha256':sha(generation)}
    write(bp,binding)
    launch=[str(BASE/'launch_stage.py'),'--phase',phase,'--seed',str(seed),'--source-commit',source]
    write(BASE/'commands'/f'{name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':launch,'arguments_line':subprocess.list2cmdline(launch)})
    print(json.dumps({'name':name,'files':len(files),'seconds':time.perf_counter()-started,'no_launch':True}))
def execute(phase,seed,source):
    name,output,args=plan(phase,seed,source);paths=prior(phase,seed,source);old,science=proof();folder=BASE/('native_'+name);require(not output.exists() and not folder.exists(),'exclusive attempt')
    binding=read(BASE/'bindings'/f'{name}.json');command=read(BASE/'commands'/f'{name}.json');require(read(BASE/'prior'/f'{name}.json')==[str(p) for p in paths],'complete exact prior list')
    require(command['arguments']==args and command['arguments_line']==subprocess.list2cmdline(args) and Path(command['executable']).resolve()==ROOT/'.venv-gpu/Scripts/python.exe','exact fixed GPU argv')
    require(all(binding['files'].get(k)==h for k,h in old['files'].items()) and binding.get('immutable_science_files')==science,'immutable P0 sources differ')
    generation=BASE/'generation'/f'{name}.json';require(binding.get('generation_path')==str(generation) and binding.get('generation_sha256')==sha(generation) and binding['files'].get(generation.relative_to(ROOT).as_posix())==sha(generation),'exact generation identity required')
    for p in [Path(__file__),BASE/'make_invocation.py',BASE/'admission.json',BASE/'metadata_cost.json',P0/'bindings/cost01.json',P0/'native_cost01/execution.json',P0/'cost01/complete.json',P0/'native_qa01/execution.json',P0/'qa01/complete.json',ROOT/'reports/long_horizon_discrimination_20261021_P1_admission.md',BASE/'commands'/f'{name}.json',BASE/'prior'/f'{name}.json',*paths]:require(binding['files'].get(p.relative_to(ROOT).as_posix())==sha(p),'required new binding absent')
    require(binding.get('endpoint_manifest')==old['endpoint_manifest'],'exact P0 endpoint manifest required')
    require(binding.get('admission')=={'path':str(BASE/'admission.json'),'sha256':sha(BASE/'admission.json')},'current root admission must be frozen')
    h=support();before=h['verify'](binding);folder.mkdir(exist_ok=False);worker=output/'worker.json';cap,timeout,selfcap=CAP[phase]
    require(0<selfcap<timeout<cap<=3600,'fixed PS budget boundary')
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    native=[str(ps),'-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(ROOT/'experiments/native_process_receipt_20261020.ps1'),'-CommandJson',str(BASE/'commands'/f'{name}.json'),'-WorkingDirectory',str(ROOT),'-Output',str(folder/'inner'),'-TimeoutSeconds',str(timeout),'-MaxSeconds',str(cap),'-SourceCommit',source,'-FixtureIdentityJson',str(worker)]
    record={'phase':phase,'category':phase,'seed':seed,'status':'RUNNING','source_commit':source,'source_before':before,'immutable_science_files':science,'binding_sha256':sha(BASE/'bindings'/f'{name}.json'),'command_sha256':sha(BASE/'commands'/f'{name}.json'),'worker_command':command,'native_cap_seconds':cap,'ps_timeout_seconds':timeout,'worker_max_seconds':selfcap,'includes_children_once':True};write(folder/'launch.json',record);started=time.perf_counter()
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
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=tuple(CAP),required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--source-commit',required=True);a=p.parse_args();name,output,args=plan(a.phase,a.seed,a.source_commit)
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


