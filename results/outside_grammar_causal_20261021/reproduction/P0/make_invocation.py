"""Create one exact outside-grammar P0 invocation; never execute it."""
import argparse,hashlib,json,runpy,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('qa','cost'));p.add_argument('--source-commit',required=True);a=p.parse_args()
    begin=time.perf_counter();h=runpy.run_path(str(BASE/'launch_stage.py'));require=h['require'];read=h['read'];sha=h['sha'];write=h['write']
    source=a.source_commit;require(len(source)==40 and all(c in '0123456789abcdef' for c in source),'archived full source required')
    require(not (BASE/'native_active.lock').exists(),'reconcile existing owner')
    name=a.phase+'01';output=BASE/name;native=BASE/('native_'+name)
    require(not output.exists() and not native.exists(),'exclusive unique phase outputs')
    paths=sorted(BASE.glob('native_*/execution.json'));require({q.parent for q in paths}=={q.parent for q in BASE.glob('native_*/launch.json')},'unreconciled prior native attempt')
    if a.phase=='qa':require(not paths,'one QA only, no retry')
    else:require(paths==[BASE/'native_qa01/execution.json'],'cost only after unique QA')
    old=h['previous_science']();names=set(old)|h['scientific_closure']()
    names|={(BASE/'observe_launch.ps1').relative_to(ROOT).as_posix(),Path(__file__).resolve().relative_to(ROOT).as_posix(),(BASE/'launch_stage.py').relative_to(ROOT).as_posix(),(ROOT/'experiments/native_process_receipt_20261020.ps1').relative_to(ROOT).as_posix(),*h['REPORTS']}
    command=BASE/'commands'/f'{name}.json';pr=BASE/'prior'/f'{name}.json';bp=BASE/'bindings'/f'{name}.json';generation=BASE/'generation'/f'{name}.json'
    for q in (command,pr,bp,generation):q.parent.mkdir(parents=True,exist_ok=True)
    args=['-m','experiments.outside_grammar_causal_20261021',a.phase,'--output',str(output),'--max-seconds','370.0','--source-commit',source,'--binding',str(bp)]
    write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)});write(pr,[str(q.resolve()) for q in paths])
    files={}
    def bind(q):
        q=Path(q).resolve();require(q.is_relative_to(ROOT) and q.suffix.lower() not in ('.h5','.hdf5'),'explicit source files only')
        k=q.relative_to(ROOT).as_posix();value=sha(q);require(k not in old or old[k]==value,'qualified historical source changed: '+k);files[k]=value
    for key in names:bind(ROOT/key)
    previous=h['PREVIOUS']
    for q in [BASE/'support_cost.json',command,pr,previous/'native_qa01/execution.json',previous/'qa01/complete.json',*paths]:bind(q)
    science={k:files[k] for k in sorted(names)};marker=None
    if a.phase=='cost':
        qa=read(paths[0]);require(qa.get('source_commit')==source and qa.get('immutable_science_files')==science and qa.get('source_before')==qa.get('source_after'),'same actual QA scientific identity')
        require(all(qa['source_before'].get(k)==v for k,v in science.items()),'all source assets actually QA checked')
        q=BASE/'qa01/complete.json';bind(q);marker={'path':q.relative_to(ROOT).as_posix(),'sha256':sha(q)}
    write(generation,{'phase':a.phase,'source_commit':source,'previous_QA_receipt_sha256':sha(previous/'native_qa01/execution.json'),'generation_seconds_before_record':time.perf_counter()-begin,'no_launch':True});bind(generation)
    binding={'files':dict(sorted(files.items())),'immutable_science_files':science,'qa_result':marker,'generation_path':str(generation),'generation_sha256':sha(generation)}
    actual=argparse.Namespace(command=command,output=native,worker=output/'worker.json',binding=bp,prior=pr,source_commit=source,phase=a.phase)
    h['command_gate'](actual,read(command),binding);budget=h['prior_gate'](actual,binding)
    write(bp,binding)
    argv=[str(BASE/'launch_stage.py'),'--phase',a.phase,'--source-commit',source,'--command',str(command),'--output',str(native),'--worker',str(output/'worker.json'),'--binding',str(bp),'--prior',str(pr)]
    write(BASE/'commands'/f'{name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':argv,'arguments_line':subprocess.list2cmdline(argv)})
    print(json.dumps({'phase':a.phase,'source_files':len(files),'prior_count':len(paths),'budget':budget,'seconds':time.perf_counter()-begin,'no_launch':True}))
if __name__=='__main__':main()

