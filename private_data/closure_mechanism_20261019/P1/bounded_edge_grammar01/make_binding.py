"""One frozen syntax pass, followed by independent source-bound QA and science."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
OLD=ROOT/'private_data/closure_mechanism_20261019/P0/finite_grammar_discovery01'
OUT=ROOT/'results/bounded_edge_grammar_20261022'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(Path(p).resolve().relative_to(ROOT)),'sha256':sha(p)}
def write_new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as stream:json.dump(v,stream,indent=2,sort_keys=True,allow_nan=False)
old=read(OLD/'bindings/qa01.json')
old_python={name:digest for name,digest in old['files'].items() if name.endswith('.py')}
if len(old_python)!=69 or any(sha(ROOT/n)!=d for n,d in old_python.items()):raise ValueError('old science closure changed')
new_names=['experiments/bounded_edge_grammar_20261022.py','tests/test_bounded_edge_grammar_20261022.py',
    str((BASE/'launch.py').relative_to(ROOT)),str((BASE/'make_binding.py').relative_to(ROOT))]
for name in new_names:ast.parse((ROOT/name).read_text(encoding='utf-8'))
if sys.argv[1]=='preflight':
    write_new(OUT/'preflight.json',{'status':'STATIC_PASS_NOT_RUNTIME_QA','old_python_sources':69,
        'old_python_all_unchanged':True,'new_source_syntax':{name:sha(ROOT/name) for name in new_names},
        'seeds':[6201,6202,6203],'data_namespace':[20261022,6200],
        'teacher_draw_seed':[20261022,6200],'teacher_coefficient':.15,'candidate_count':6,
        'formal_updates':9216,'checkpoint_directories':72,'internal_restores':36,'candidate_rankings':54,
        'qualification_cuda_updates_planned':36,'qualification_cpu_updates_planned':2,'project_val_test_closed':True,'TASK4_paused':True,
        'old_failure_retained':'e03b21bf7f72b01067763bc108fde2d2ca29f08c','no_process_quotas':True})
else:
    phase,commit=sys.argv[1:3]
    if phase not in ('freeze','run') or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('registered source phase')
    files={str((ROOT/name).resolve()):digest for name,digest in old_python.items()}
    selection=ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01/spawn_chain20261022/runtime_selection.json'
    for name,digest in read(selection)['files'].items():
        path=(ROOT/name).resolve()
        if sha(path)!=digest:raise ValueError('qualified runtime changed')
        files[str(path)]=digest
    extras=new_names+['experiments/finite_grammar_execution_20261022.py','tests/test_finite_grammar_execution_20261022.py','AGENTS.md','docs/engineering_loop.md',
        'reports/astra6_bounded_edge_20261022.md','reports/astra6_bounded_edge_20261022_raw.txt',
        'reports/bounded_edge_grammar_20261022_protocol.md','reports/bounded_edge_grammar_20261022_review.md',
        'results/bounded_edge_grammar_20261022/preflight.json',str(selection.relative_to(ROOT)),
        str((OLD/'bindings/qa01.json').relative_to(ROOT))]
    value={'source_commit':commit,'phase':phase,'no_process_quotas':True}
    for key,path in [('old_qa_complete',OLD/'qa01/complete.json'),('old_qa_receipt',OLD/'native_qa01/execution.json')]:
        value[key]=ref(path);extras.append(str(path.relative_to(ROOT)))
    if phase=='run':
        for key,path in [('pool_manifest',BASE/'frozen_pool/pool.json'),('teacher_identity',BASE/'frozen_pool/teacher.json'),
            ('freeze_receipt',BASE/'native_freeze/receipt.json'),('freeze_complete',BASE/'freeze/complete.json')]:
            value[key]=ref(path);extras.append(str(path.relative_to(ROOT)))
        extras.append(str((BASE/'freeze_binding.json').relative_to(ROOT)))
    for name in extras:files[str((ROOT/name).resolve())]=sha(ROOT/name)
    value['files']=dict(sorted(files.items()))
    dest=BASE/('freeze_binding.json' if phase=='freeze' else 'binding.json')
    write_new(dest,value)
    print(json.dumps({'source_commit':commit,'phase':phase,'files':len(files),'binding_sha256':sha(dest)}))
