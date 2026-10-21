"""Fixed rational comparison launcher. No time, CPU, count or cost quotas."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
GROUPS=('0_rational','0_tanh','1_rational','1_tanh')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def item(binding,key):
    i=binding[key];p=(ROOT/i['path']).resolve();require(p.is_relative_to(ROOT) and sha(p)==i['sha256'],'changed explicit '+key);return p,read(p)
def plan(phase,group,resume):
    require(phase in ('qa','run','evaluate') and ((phase=='run' and group in GROUPS) or (phase!='run' and group is None)),'fixed scientific role')
    require(not resume or phase=='run','resume only interrupted run')
    return ('run_'+group if phase=='run' else phase+'01')+('_resume01' if resume else '')
def mathematical_proof(binding):
    _,proof=item(binding,'mathematical_verification');_,old=item(binding,'original_qa_receipt')
    require(proof.get('verified') is True and len(proof.get('exact_ast_closure',[]))==54,'verified54 mathematical closure')
    require(old.get('status')=='COMPLETED' and old.get('phase_gate_pass') is True and old['inner']['exit_code']==old['outer']['exit_code']==0 and old['inner']['cleanup_verified'] is True and old['inner']['fixture_identity_after']==old['inner']['supervised_identity_after']=='GONE','actual old mathematical QA')
    files=binding['files']
    for row in proof['exact_ast_closure']:
        name=row['path'];p=(ROOT/name).resolve();raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+bytes([0])+raw).hexdigest()
        require(digest==old['immutable_science_files'].get(name)==old['source_before'].get(name)==old['source_after'].get(name) and files.get(name)==digest and blob in (row['sha'],row['lf_sha']),'mathematical byte identity differs: '+name)
def execute(a):
    from uncapped_windows_job_20261021 import supervise,verify_files
    require(len(a.source_commit)==40 and all(c in '0123456789abcdef' for c in a.source_commit),'full code source identity')
    bp=a.binding.resolve();require(bp.is_relative_to(BASE),'this round binding only');binding=read(bp)
    _,authority=item(binding,'authority')
    require(authority.get('user_instruction')=='不设置进程限制' and authority.get('explicit_clarification')=='全部取消进程额度限制' and binding.get('source_commit')==a.source_commit,'explicit human no-process-quota authority and bound source')
    mathematical_proof(binding)
    name=plan(a.phase,a.group,a.resume);logical='run_'+a.group if a.phase=='run' else a.phase+'01';out=BASE/logical;native=BASE/('native_'+name)
    require(not native.exists(),'unique native attempt, never overwrite')
    files={(ROOT/k).resolve():v for k,v in binding['files'].items()}
    files[bp]=sha(bp) # Runtime bind the input JSON itself without a self-hash cycle.
    required_current=[]
    for name0 in ('AGENTS.md','docs/engineering_loop.md','reports/rational_execution_20261021_protocol.md','reports/astra6_rational_execution_20261021.md','experiments/rational_execution_20261021.py','experiments/uncapped_windows_job_20261021.py','tests/test_uncapped_windows_job_20261021.py','tests/test_rational_execution_20261021.py',str(Path(__file__).relative_to(ROOT).as_posix())):
        p=ROOT/name0;require(p.resolve() in files and sha(p)==files[p.resolve()],'required source/policy binding')
        required_current.append(p.resolve())
    verify_files(files)
    paths=sorted(BASE.glob('native_*/receipt.json'));declared=[(ROOT/p).resolve() for p in binding.get('prior_receipts',[])]
    require(set(declared)==set(paths) and len(declared)==len(set(declared)),'complete actual own prior receipts')
    require({p.parent for p in paths}==set(BASE.glob('native_*')),'pending prior native job must be reconciled')
    rows=[read(p) for p in paths]
    for p,r in zip(paths,rows):
        require(r.get('cleanup_verified') is True and r.get('job_active_processes')==0 and r.get('source_before')==r.get('source_after') and r.get('source_commit')==a.source_commit and files.get(p.resolve())==sha(p),'prior Job closure/source/hash')
    if a.phase=='qa':require(not paths and not a.resume,'first wiring QA only')
    else:
        qa=BASE/'native_qa01/receipt.json';require(qa in paths and read(qa).get('status')=='COMPLETED','actual new native QA pass')
        qa_receipt=read(qa)
        require(all(qa_receipt['source_before'].get(str(p))==files[p]==qa_receipt['source_after'].get(str(p)) for p in required_current),'same actual new QA source and authority protocol')
        marker=read(BASE/'qa01/complete.json');require(marker['status']=='COMPLETED' and marker['result']['status']=='PASS','actual new worker QA pass')
        completed=[]
        for group in GROUPS:
            markerpath=BASE/('run_'+group)/'complete.json'
            if markerpath.exists():
                marker=read(markerpath);result=marker.get('result',{})
                require(marker.get('status')=='COMPLETED' and result.get('status')=='COMPLETED' and result.get('source_commit')==a.source_commit and result.get('group')==group,'group marker exact executed role/source')
                completed.append(group)
        if a.phase=='run':
            require(len(completed)<4 and completed==list(GROUPS[:len(completed)]) and a.group==GROUPS[len(completed)],'fixed next scientific group')
            if a.resume:
                _,authorization=item(binding,'resume_authorization')
                require(authorization.get('source_commit')==a.source_commit and authorization.get('group')==a.group and authorization.get('output')==str(out) and authorization.get('nonselective_interruption') is True and authorization.get('resume_count')==1 and authorization.get('previous_job_closed') is True,'one root-directed interruption resume')
                previous_path,previous=item(authorization,'previous_native_receipt')
                require(previous_path in paths and previous.get('job_active_processes')==0 and previous.get('cleanup_verified') is True and previous.get('source_before')==previous.get('source_after') and previous.get('source_commit')==a.source_commit and previous.get('group')==a.group,'actual same-source interrupted Job fully closed')
                from uncapped_windows_job_20261021 import identity_state
                require(identity_state(previous['worker'])=='GONE','previous worker exact creation identity must be GONE')
                require(files.get(previous_path)==sha(previous_path),'resume closure evidence must be source bound')
                require(not any('_resume01' in p.parent.name for p in paths),'one external resume for this round')
            else:require(not out.exists(),'new group output must be exclusive')
        else:
            require(completed==list(GROUPS) and not out.exists(),'all twelve before one evaluation')
            require(all(read(BASE/('run_'+group)/'complete.json')['result'].get('all_group_frozen') is True for group in GROUPS),'all groups frozen before val')
            _,selection=item(binding,'selection_manifest');require(selection.get('all_twelve_frozen') is True and selection.get('source_commit')==a.source_commit and len(selection.get('rows',[]))==12,'actual twelve-endpoint freeze')
            for row in selection['rows']:
                for filename,digest in row['files'].items():
                    p=(Path(row['path'])/filename).resolve();require(p in files and files[p]==digest==sha(p),'all frozen model/Adam files bound')
                for key in ('train_raw','trace'):
                    p=Path(row[key]['path']).resolve();require(p in files and files[p]==row[key]['sha256']==sha(p),'frozen observations/trace bound')
    args=['-m','experiments.rational_execution_20261021',a.phase,'--output',str(out),'--binding',str(bp),'--source-commit',a.source_commit]
    if a.group:args+=['--group',a.group]
    if a.resume:args+=['--resume']
    command={'executable':str(ROOT/'.venv-gpu/Scripts/python.exe'),'arguments':args}
    result=supervise(command,native,BASE/'native_active.lock',files,ROOT,worker_receipt=out/'worker.json',provenance={'source_commit':a.source_commit,'phase':a.phase,'group':a.group,'binding_sha256':sha(bp)})
    if result['status']!='COMPLETED':return 1
    complete=read(out/'complete.json')
    return 0 if complete.get('status')=='COMPLETED' and complete.get('result',{}).get('status') in ('PASS','COMPLETED') else 1
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('qa','run','evaluate'));p.add_argument('--group',choices=GROUPS);p.add_argument('--binding',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    return execute(a)
if __name__=='__main__':raise SystemExit(main())

