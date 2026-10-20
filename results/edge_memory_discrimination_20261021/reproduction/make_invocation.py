"""Generate one bounded P0 command; no execution, data loading or training."""
import argparse,ast,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BASE=Path(__file__).resolve().parent
REPORTS=('reports/edge_memory_discrimination_20261021_protocol.md','reports/edge_memory_discrimination_20261021_plan_review.md','reports/astra6_after_frequency_20261021_addendum2.md','AGENTS.md')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(ok,message):
    if not ok:raise ValueError(message)
def digest(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('qa','cost'));p.add_argument('--source-commit',required=True);p.add_argument('--old-inputs',type=Path,required=True);p.add_argument('--static-cost',type=Path,action='append',default=[])
    a=p.parse_args();started=time.perf_counter();source=a.source_commit
    require(len(source)==40 and all(c in '0123456789abcdef' for c in source),'full archived source commit required')
    require(not (BASE/'native_active.lock').exists(),'existing owner must be reconciled')
    oldpath=a.old_inputs.resolve();require(oldpath==BASE/'old_contract_inputs.json','fixed explicit old input table required')
    require(sha(oldpath)=='6af592c1374f0c1aed1e55b0c7c239ba2b4b0fc4b5356e092bf9269525314576','registered old allowlist bytes changed')
    old=read(oldpath);oldfiles=old.get('files');require(isinstance(oldfiles,dict) and oldfiles,'nonempty explicit old input table required')
    files={}
    for name,h in oldfiles.items():
        path=(ROOT/name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(ROOT) and path.suffix.lower() not in ('.h5','.hdf5') and digest(h),'invalid old allowlist entry')
        files[name]=h # Deliberately do not rehash the large old assets here.
    phase=a.phase;name=phase+'01';output=BASE/name
    require(not output.exists() and not (BASE/('native_'+name)).exists(),'exclusive phase output required')
    receipts=sorted(BASE.glob('native_*/execution.json'))
    require({x.parent for x in receipts}=={x.parent for x in BASE.glob('native_*/launch.json')},'unreconciled native prior')
    if phase=='qa':require(not receipts,'unique QA, no retry')
    else:
        require(len(receipts)==1 and receipts[0]==BASE/'native_qa01/execution.json','cost requires unique QA prior')
        r=read(receipts[0]);require(r.get('status')=='COMPLETED' and r.get('phase_gate_pass') is True and r.get('source_commit')==source,'current QA must actually pass')
    command=BASE/'commands'/f'{name}.json';prior=BASE/'prior'/f'{name}.json';binding=BASE/'bindings'/f'{name}.json';generation=BASE/'generation'/f'{name}.json'
    args=['-m','experiments.edge_memory_discrimination_20261021',phase,'--output',str(output),'--max-seconds','540.0','--source-commit',source,'--binding',str(binding)]
    write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)})
    write(prior,[str(x.resolve()) for x in receipts]);new={}
    def bind(path):
        path=Path(path).resolve();require(path.is_relative_to(ROOT) and path.suffix.lower() not in ('.h5','.hdf5'),'forbidden input')
        name=path.relative_to(ROOT).as_posix();files[name]=new[name]=sha(path)
    pending=[ROOT/'experiments/edge_memory_discrimination_20261021.py',ROOT/'tests/test_edge_memory_discrimination_20261021.py',ROOT/'experiments/fixed_edge_pair_20261019.py'];seen=set()
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
    costs=[x.resolve() for x in a.static_cost];require(len(costs)==len(set(costs)),'duplicate static fee file')
    for path in costs:require(path.is_relative_to(BASE) and path!=BASE/'implementation_cost.json','explicit immutable local fee supplement required')
    for path in [Path(__file__),BASE/'launch_stage.py',BASE/'implementation_cost.json',oldpath,command,prior,ROOT/'experiments/native_process_receipt_20261020.ps1',*[ROOT/n for n in REPORTS],*costs,*receipts]:bind(path)
    science_names=set(oldfiles)|{x.relative_to(ROOT).as_posix() for x in seen}|{x.relative_to(ROOT).as_posix() for x in [Path(__file__).resolve(),BASE/'launch_stage.py',oldpath,ROOT/'experiments/native_process_receipt_20261020.ps1',*[ROOT/n for n in REPORTS]]}
    science={k:files[k] for k in sorted(science_names)};qa_result=None
    if phase=='cost':
        qa=read(receipts[0]);require(qa.get('immutable_science_files')==science,'scientific dependency identity differs from actual QA')
        require(all(qa['source_before'].get(k)==h==qa['source_after'].get(k) and sha(ROOT/k)==h for k,h in science.items()),'current scientific dependency changed since QA')
        marker=BASE/'qa01/complete.json';complete=read(marker)
        require(complete.get('status')=='COMPLETED' and complete.get('result',{}).get('status')=='PASS' and complete==qa.get('complete') and sha(marker)==qa.get('complete_sha256'),'actual QA completion must pass unchanged')
        bind(marker);qa_result={'path':marker.relative_to(ROOT).as_posix(),'sha256':sha(marker)}
    write(generation,{'phase':phase,'source_commit':source,'old_table_path':str(oldpath),'old_table_sha256':sha(oldpath),'new_input_hashes':new,'generation_seconds_before_record':time.perf_counter()-started,'static_cost_paths':[str(x) for x in costs],'no_launch':True})
    bind(generation)
    write(binding,{'files':dict(sorted(files.items())),'immutable_science_files':science,'qa_result':qa_result,'old_contract_inputs':{'path':oldpath.relative_to(ROOT).as_posix(),'sha256':sha(oldpath),'files':oldfiles},'static_cost_paths':[str(x) for x in costs],'generation_path':str(generation),'generation_sha256':sha(generation)})
    argv=[str(BASE/'launch_stage.py'),'--command',str(command),'--output',str(BASE/('native_'+name)),'--worker',str(output/'worker.json'),'--binding',str(binding),'--prior',str(prior),'--source-commit',source,'--phase',phase]
    write(BASE/'commands'/f'{name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':argv,'arguments_line':subprocess.list2cmdline(argv)})
    print(json.dumps({'phase':phase,'source_files':len(files),'prior_count':len(receipts),'seconds':time.perf_counter()-started,'no_launch':True}))
if __name__=='__main__':main()
