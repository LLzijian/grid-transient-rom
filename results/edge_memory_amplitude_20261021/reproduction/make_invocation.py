"""Explicit E8 command generation from the unchanged P8 source identity table.
Does not launch, import training code, or rehash frozen train/baseline leaves.
All inputs are fully rehashed by the qualified native consumer before/after execution.
"""
from pathlib import Path
import argparse,ast,json,hashlib,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
SCIENCE=ROOT/'private_data/closure_mechanism_20261019/P2/quartic_memory01/prepare01'
BASE_TABLE=ROOT/'private_data/closure_mechanism_20261019/P2/cubic_port_only01/bindings/qa01.json'
EXPECTED_BASE_COUNT=323
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2);f.write('\n')
def generate(a):
    start=time.perf_counter()
    caps={'qa':(200.,160.),'cost':(500.,440.),'train':(2500.,2440.),'evaluate':(240.,180.)}
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in a.name) or a.cap!=caps[a.category][0]:raise ValueError('fixed lowercase name and cap required')
    if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit):raise ValueError('full frozen source commit required')
    if (BASE/'native_active.lock').exists():raise ValueError('native attempt is live or unreconciled')
    args=a.extra
    def flag(n):
        if args.count(n)!=1 or args.index(n)+1>=len(args):raise ValueError('one explicit '+n)
        return args[args.index(n)+1]
    output=a.output.resolve()
    if a.category=='train':
        seed=flag('--seed')
        if seed not in ('6101','6102','6103') or flag('--variant')!='E' or flag('--target-step')!='1024' or a.name!=f'train_{seed}_e' or output!=BASE/'models'/f'{seed}_E':raise ValueError('exact E train role required')
    elif a.category=='evaluate':
        roles={BASE/'models'/f'{s}_E/checkpoints/step001024/model.json':s for s in (6101,6102,6103)}
        model=Path(flag('--model')).resolve()
        if model not in roles or a.name!=f'eval_{roles[model]}_e' or output!=BASE/'evaluation'/f'{roles[model]}_E/step001024':raise ValueError('exact E evaluation role required')
    elif a.category=='cost':
        if a.name!='cost01' or output!=BASE/'cost01':raise ValueError('unique cost required')
    elif a.name not in ('qa01','qa02') or output!=BASE/a.name:raise ValueError('exact QA required')
    if output.exists():raise FileExistsError('output already exists')
    if any(x in args for x in ('--resume','--external-recovery','--worker-output')):raise ValueError('no recovery or substitute worker')
    receipts=sorted(BASE.glob('native_*/execution.json'))
    if {p.parent for p in receipts}!={p.parent for p in BASE.glob('native_*/launch.json')}:raise ValueError('unreconciled native prior')
    for p in receipts:
        e=read(p)
        if e.get('status')=='RUNNING' or not e.get('inner',{}).get('cleanup_verified') or e['inner'].get('fixture_identity_after')!='GONE':raise ValueError('prior ownership unresolved')
    qa=[p for p in receipts if read(p).get('category')=='qa']
    if a.category=='qa':
        if len(qa)>1 or any(read(p).get('status')=='COMPLETED' for p in qa):raise ValueError('QA success or second attempt cannot repeat')
        if a.name!=('qa02' if qa else 'qa01'):raise ValueError('QA attempt sequence differs')
        if qa:
            auth=read(BASE/'qa02_authorization.json')
            if auth.get('authorized') is not True or auth.get('pure_engineering_fault') is not True or auth.get('source_commit')!=a.source_commit or auth.get('qa01_execution_sha256')!=sha(qa[0]):raise ValueError('root engineering-only QA02 authorization required')
    elif not qa or read(qa[-1]).get('status')!='COMPLETED' or read(qa[-1]).get('source_commit')!=a.source_commit:raise ValueError('successful current E QA required')
    old=read(BASE_TABLE)
    if not isinstance(old.get('files'),dict) or len(old['files'])!=EXPECTED_BASE_COUNT:raise ValueError('exact registered source table count required')
    provenance=BASE_TABLE.parents[1]/'native_qa01/execution.json'
    oldreceipt=read(provenance)
    if oldreceipt['source_before']!=oldreceipt['source_after'] or oldreceipt['source_before']!=old['files']:raise ValueError('P8 table is not unchanged; no claim of QA success')
    # Preserve all frozen identities, including historical code as provenance.
    files=dict(old['files']);new={}
    def bind(p):
        p=Path(p).resolve()
        if not p.is_relative_to(ROOT) or p.suffix.lower() in ('.h5','.hdf5'):raise ValueError('forbidden input')
        k=p.relative_to(ROOT).as_posix();files[k]=new[k]=sha(p)
    command=BASE/'commands'/f'{a.name}.json';prior=BASE/'prior'/f'{a.name}.json';binding=BASE/'bindings'/f'{a.name}.json'
    argv=([str(BASE/'qa_driver.py')] if a.category=='qa' else ['-m','experiments.edge_memory_amplitude_20261021',a.category])+['--output',str(output),'--max-seconds',str(caps[a.category][1]),'--source-commit',a.source_commit]+([] if a.category=='qa' else ['--binding',str(binding)])+args
    write(command,{'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':argv,'arguments_line':subprocess.list2cmdline(argv)})
    write(prior,[str(p.resolve()) for p in receipts])
    pending=[ROOT/n for n in ('experiments/edge_memory_amplitude_20261021.py','experiments/edge_memory_amplitude_runtime_20261021.py','tests/test_edge_memory_amplitude_20261021.py','tests/test_edge_memory_amplitude_runtime_20261021.py','tests/test_edge_memory_amplitude_native_20261021.py')]
    seen=set()
    while pending:
        p=pending.pop().resolve()
        if p in seen:continue
        seen.add(p);key=p.relative_to(ROOT).as_posix()
        if key not in files:bind(p)
        # Existing baseline table already freezes its full dependency closure.
        if key in old['files']:continue
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[v.name for v in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            for name in names:
                if name.startswith('experiments.'):name=name[12:]
                for folder in (ROOT/'experiments',ROOT/'src',ROOT/'tests'):
                    f=folder.joinpath(*name.split('.')).with_suffix('.py')
                    if f.is_file():pending.append(f)
    for p in [Path(__file__),BASE/'launch_stage.py',BASE/'qa_driver.py',BASE/'implementation_cost.json',BASE_TABLE,provenance,command,prior,ROOT/'reports/edge_memory_amplitude_20261021_protocol.md',ROOT/'reports/edge_memory_amplitude_20261021_plan_review.md',ROOT/'reports/astra6_after_port_only_20261021.md',*receipts]:bind(p)
    for p in sorted(BASE.glob('static_*cost*.json')):bind(p)
    if a.category!='qa':
        for p in (SCIENCE/'prepare.json',SCIENCE/'schedule.json'):bind(p)
    if a.name=='qa02':bind(BASE/'qa02_authorization.json')
    for n in ('--schedule-path','--admission','--model'):
        if n in argv:bind(Path(argv[argv.index(n)+1]))
    if a.category=='evaluate':
        bind(BASE/'frozen_evaluation_endpoints.json')
        for name in ('training.pt','checkpoint.json'):bind(Path(flag('--model')).parent/name)
    record=BASE/'generation'/f'{a.name}.json'
    write(record,{'base_table':str(BASE_TABLE),'base_table_byte_sha256':sha(BASE_TABLE),'base_table_was_successful_QA':False,'new_input_hashes':new,'seconds_before_record':time.perf_counter()-start,'arguments':vars(a)|{'output':str(output)},'no_launch':True})
    bind(record);write(binding,{'files':dict(sorted(files.items())),'base_table_sha256':sha(BASE_TABLE),'all_files_rehashed_by_native_before_after':True})
    launch=[str(BASE/'launch_stage.py'),'--command',str(command),'--output',str(BASE/('native_'+a.name)),'--worker',str(output/'worker.json'),'--binding',str(binding),'--prior',str(prior),'--source-commit',a.source_commit,'--category',a.category,'--cap',str(a.cap)]
    write(BASE/'commands'/f'{a.name}_launcher.json',{'executable':str(Path(sys.executable).resolve()),'arguments':launch,'arguments_line':subprocess.list2cmdline(launch)})
    print(json.dumps({'name':a.name,'source_count':len(files),'base_table_sha256':sha(BASE_TABLE),'prior_count':len(receipts),'seconds':time.perf_counter()-start,'no_launch':True}))
def main():
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('category',choices=('qa','cost','train','evaluate'));p.add_argument('source_commit');p.add_argument('cap',type=float);p.add_argument('--output',type=Path,required=True);p.add_argument('--extra',nargs=argparse.REMAINDER,default=[])
    generate(p.parse_args())
if __name__=='__main__':main()
