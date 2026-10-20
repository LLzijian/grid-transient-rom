"""Generate exactly one synthetic audit invocation; never launch it."""
from pathlib import Path
import argparse,ast,json,hashlib,sys,subprocess
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('category',choices=('prepare','qa','benchmark'));p.add_argument('source_commit');p.add_argument('cap',type=float);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
limits={'prepare':(100.,40.),'qa':(400.,340.),'benchmark':(1800.,1740.)}
if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in a.name) or len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit) or a.cap!=limits[a.category][0]:raise ValueError('fixed lowercase invocation/cap/commit')
a.output=a.output.resolve();allowed={'prepare':{'prepare01'},'qa':{'qa01','qa02'},'benchmark':{'benchmark01'}}
if a.output.parent!=BASE or a.output.name not in allowed[a.category] or a.output.exists():raise ValueError('exclusive fixed output')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
command=BASE/'commands'/f'{a.name}.json';prior=BASE/'prior'/f'{a.name}.json';binding=BASE/'bindings'/f'{a.name}.json'
args=([str(BASE/'qa_driver.py')] if a.category=='qa' else ['-m','experiments.fused_closure_audit_20261021',a.category])+['--output',str(a.output),'--max-seconds',str(limits[a.category][1]),'--source-commit',a.source_commit,'--binding',str(binding)]
write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args,'arguments_line':subprocess.list2cmdline(args)})
receipts=sorted(str(p.resolve()) for p in BASE.glob('native_*/execution.json'));write(prior,receipts)
pending=[ROOT/n for n in ('experiments/fused_closure_audit_20261021.py','experiments/fused_closure_runtime_20261021.py','tests/test_fused_closure_audit_20261021.py','tests/test_fused_closure_runtime_20261021.py')];seen=set();files={}
while pending:
    f=pending.pop().resolve()
    if f in seen:continue
    seen.add(f);files[f.relative_to(ROOT).as_posix()]=sha(f)
    for node in ast.walk(ast.parse(f.read_text(encoding='utf-8-sig'))):
        names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        for name in names:
            if name.startswith('experiments.'):name=name[12:]
            for folder in (ROOT/'experiments',ROOT/'src',ROOT/'tests'):
                candidate=folder.joinpath(*name.split('.')).with_suffix('.py')
                if candidate.is_file():pending.append(candidate)
                parts=name.split('.')
                for count in range(1,len(parts)+1):
                    init=folder.joinpath(*parts[:count])/'__init__.py'
                    if init.is_file():pending.append(init)
for f in [Path(__file__),BASE/'launch_stage.py',BASE/'qa_driver.py',BASE/'implementation_cost.json',command,prior,ROOT/'experiments/native_process_receipt_20261020.ps1',ROOT/'reports/fused_closure_runtime_20261021_protocol.md',ROOT/'reports/fused_closure_runtime_20261021_plan_review.md',ROOT/'reports/astra6_after_cubic_cost_20261021.md',ROOT/'tests/test_linear_cache_runtime_20261019.py',*[Path(p) for p in receipts]]:files[f.relative_to(ROOT).as_posix()]=sha(f)
if a.category!='prepare':
    for name in ('manifest.json','model_N.json','model_D.json','synthetic.npz','rng.pt'):
        f=BASE/'prepare01'/name;files[f.relative_to(ROOT).as_posix()]=sha(f)
if a.output.name=='qa02':
    f=BASE/'qa02_authorization.json';files[f.relative_to(ROOT).as_posix()]=sha(f)
write(binding,{'files':dict(sorted(files.items())),'synthetic_only':True,'old_scientific_assets_read':False})
launch=[str(BASE/'launch_stage.py'),'--command',str(command),'--output',str(BASE/f'native_{a.name}'),'--worker',str(a.output/'worker.json'),'--binding',str(binding),'--prior',str(prior),'--source-commit',a.source_commit,'--category',a.category,'--cap',str(a.cap)]
write(BASE/'commands'/f'{a.name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':launch,'arguments_line':subprocess.list2cmdline(launch)})
print(json.dumps({'name':a.name,'binding_files':len(files),'prior_count':len(receipts),'no_launch':True}))
