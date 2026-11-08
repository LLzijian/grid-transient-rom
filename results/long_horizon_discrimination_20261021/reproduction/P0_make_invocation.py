"""Generate one bounded P0 command; no execution, data loading or training."""
import argparse,ast,hashlib,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
REPORTS=('reports/long_horizon_discrimination_20261021_protocol.md','reports/long_horizon_discrimination_20261021_plan_review.md','reports/astra6_after_edge_deletion_20261021.md','AGENTS.md')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(ok,message):
    if not ok:raise ValueError(message)
def digest(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('qa','cost'));p.add_argument('--source-commit',required=True);p.add_argument('--endpoints',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter();source=a.source_commit
    require(len(source)==40 and all(c in '0123456789abcdef' for c in source),'full archived source commit required')
    require(not (BASE/'native_active.lock').exists(),'existing owner must be reconciled')
    endpointpath=a.endpoints.resolve();endpointfiles=endpoint_inputs(endpointpath)
    files=dict(endpointfiles) # Native consumers rehash all actual bytes before and after.
    phase=a.phase;name=phase+'01';output=BASE/name
    require(not output.exists() and not (BASE/('native_'+name)).exists(),'exclusive phase output required')
    receipts=sorted(BASE.glob('native_*/execution.json'))
    require({x.parent for x in receipts}=={x.parent for x in BASE.glob('native_*/launch.json')},'unreconciled native prior')
    if phase=='qa':require(not receipts,'unique QA, no retry')
    else:
        require(len(receipts)==1 and receipts[0]==BASE/'native_qa01/execution.json','cost requires unique QA prior')
        r=read(receipts[0]);require(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and r.get('source_commit')==source,'current QA must actually pass')
    for receipt in receipts:
        r=read(receipt);i=r.get('inner',{});o=r.get('outer',{})
        require(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and r.get('source_commit')==source and r.get('source_before')==r.get('source_after'),'actual same-source QA completion required')
        require(isinstance(r.get('charged_seconds'),(int,float)) and not isinstance(r.get('charged_seconds'),bool) and math.isfinite(r['charged_seconds']) and 0<=r['charged_seconds']<=240,'prior native cap')
        require(i.get('exit_code')==o.get('exit_code')==0 and i.get('native_receipt_complete') is True and o.get('native_receipt_complete') is True and i.get('cleanup_verified') is True and i.get('fixture_identity_after')==i.get('supervised_identity_after')=='GONE','QA native ownership/exit unresolved')
    support=read(BASE/'support_cost.json')
    require(support.get('support_held_seconds')==520 and support.get('cap_seconds')==1000 and sum(read(x)['charged_seconds'] for x in receipts)+240*(2-len(receipts))+520<=1000,'whole registered P0 cost cannot fit')
    command=BASE/'commands'/f'{name}.json';prior=BASE/'prior'/f'{name}.json';binding=BASE/'bindings'/f'{name}.json';generation=BASE/'generation'/f'{name}.json'
    args=['-m','experiments.long_horizon_discrimination_20261021',phase,'--output',str(output),'--max-seconds','210.0','--source-commit',source,'--binding',str(binding)]
    write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)})
    write(prior,[str(x.resolve()) for x in receipts]);new={}
    def bind(path):
        path=Path(path).resolve();require(path.is_relative_to(ROOT) and path.suffix.lower() not in ('.h5','.hdf5'),'forbidden input')
        name=path.relative_to(ROOT).as_posix();files[name]=new[name]=sha(path)
    pending=[ROOT/'experiments/long_horizon_discrimination_20261021.py',ROOT/'tests/test_long_horizon_discrimination_20261021.py',ROOT/'experiments/fixed_edge_pair_20261019.py'];seen=set()
    while pending:
        path=pending.pop().resolve()
        if path in seen:continue
        seen.add(path);bind(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            imports=[]
            if isinstance(node,ast.Import):imports=[(n.name,0) for n in node.names]
            elif isinstance(node,ast.ImportFrom):
                imports=[(node.module or '',node.level)]
                imports += [(((node.module+'.') if node.module else '')+n.name,node.level) for n in node.names if n.name!='*']
            for module,level in imports:
                roots=[path.parent.joinpath(*(['..']*(level-1)))] if level else [ROOT,ROOT/'experiments',ROOT/'src',ROOT/'tests']
                for folder in roots:
                    target=folder.joinpath(*module.split('.')) if module else folder
                    for candidate in (target.with_suffix('.py'),target/'__init__.py'):
                        if candidate.is_file() and candidate.resolve().is_relative_to(ROOT):pending.append(candidate)
                    parts=module.split('.')
                    for count in range(1,len(parts)):
                        init=folder.joinpath(*parts[:count])/'__init__.py'
                        if init.is_file():pending.append(init)
    for path in [Path(__file__),BASE/'launch_stage.py',BASE/'support_cost.json',endpointpath,command,prior,ROOT/'experiments/native_process_receipt_20261020.ps1',*[ROOT/n for n in REPORTS],*receipts]:bind(path)
    science_names=set(endpointfiles)|{x.relative_to(ROOT).as_posix() for x in seen}|{x.relative_to(ROOT).as_posix() for x in [Path(__file__).resolve(),BASE/'launch_stage.py',endpointpath,ROOT/'experiments/native_process_receipt_20261020.ps1',*[ROOT/n for n in REPORTS]]}
    science={k:files[k] for k in sorted(science_names)};qa_result=None
    if phase=='cost':
        qa=read(receipts[0]);require(qa.get('immutable_science_files')==science,'scientific dependency identity differs from actual QA')
        require(all(qa['source_before'].get(k)==h==qa['source_after'].get(k) and sha(ROOT/k)==h for k,h in science.items()),'current scientific dependency changed since QA')
        marker=BASE/'qa01/complete.json';complete=read(marker)
        require(complete.get('status')=='COMPLETED' and complete.get('result',{}).get('status')=='PASS' and complete==qa.get('complete') and sha(marker)==qa.get('complete_sha256'),'actual QA completion must pass unchanged')
        bind(marker);qa_result={'path':marker.relative_to(ROOT).as_posix(),'sha256':sha(marker)}
    write(generation,{'phase':phase,'source_commit':source,'endpoint_manifest_path':str(endpointpath),'endpoint_manifest_sha256':sha(endpointpath),'new_input_hashes':new,'generation_seconds_before_record':time.perf_counter()-started,'no_launch':True})
    bind(generation)
    write(binding,{'files':dict(sorted(files.items())),'immutable_science_files':science,'qa_result':qa_result,'endpoint_manifest':{'path':endpointpath.relative_to(ROOT).as_posix(),'sha256':sha(endpointpath)},'generation_path':str(generation),'generation_sha256':sha(generation)})
    argv=[str(BASE/'launch_stage.py'),'--command',str(command),'--output',str(BASE/('native_'+name)),'--worker',str(output/'worker.json'),'--binding',str(binding),'--prior',str(prior),'--source-commit',source,'--phase',phase]
    write(BASE/'commands'/f'{name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':argv,'arguments_line':subprocess.list2cmdline(argv)})
    print(json.dumps({'phase':phase,'source_files':len(files),'prior_count':len(receipts),'seconds':time.perf_counter()-started,'no_launch':True}))
if __name__=='__main__':main()

