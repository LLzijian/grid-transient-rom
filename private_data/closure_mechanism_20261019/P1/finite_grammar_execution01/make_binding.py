"""Freeze only registered source and existing evidence; never runs science."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).resolve().parent
OLD=ROOT/'private_data/closure_mechanism_20261019/P0/finite_grammar_discovery01'
OUT=ROOT/'results/finite_grammar_execution_20261022'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(Path(p).resolve().relative_to(ROOT)),'sha256':sha(p)}
def write_new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as stream:json.dump(v,stream,indent=2,sort_keys=True,allow_nan=False)
new_names=['experiments/finite_grammar_execution_20261022.py','tests/test_finite_grammar_execution_20261022.py',
    str((BASE/'launch.py').relative_to(ROOT)),str((BASE/'make_binding.py').relative_to(ROOT))]
old=read(OLD/'bindings/qa01.json')
old_python={name:digest for name,digest in old['files'].items() if name.endswith('.py')}
if len(old_python)!=69 or any(sha(ROOT/name)!=digest for name,digest in old_python.items()):
    raise ValueError('old science/QA Python closure changed')
for name in new_names:ast.parse((ROOT/name).read_text(encoding='utf-8'))
if sys.argv[1]=='preflight':
    summary={'status':'STATIC_PREPARATION_PASS_NOT_RUNTIME_QA','old_python_sources':69,'old_python_all_unchanged':True,
        'new_source_syntax':{name:sha(ROOT/name) for name in new_names},
        'frozen_pool':ref(OLD/'freeze_pool01/pool.json'),'frozen_teacher':ref(OLD/'freeze_pool01/teacher.json'),
        'formal_stream_audit':{'formal_seeds':[6101,6102,6103],'default_namespace':6100,
            'QA_independent_observations_seed':9902,'cost_seeds':[7901,7902],'cost_timing_namespace':991,
            'old_P1_formal_directory_exists':(ROOT/'private_data/closure_mechanism_20261019/P1/finite_grammar_discovery01').exists(),
            'old_QA_cost_do_not_call_formal_seed_fixture':True},
        'workload':{'seeds':3,'candidate_count':6,'updates_per_seed':768,'formal_updates':2304,
            'checkpoint_directories':45,'restores':9,'rankings':27,'qualification_cuda_updates':36,'qualification_cpu_updates':6},
        'old_P1_cost_outcome':'CLOSED_NO_FORMAL_TRAINING_RETAINED','no_process_quotas':True,
        'project_val_test_closed':True,'TASK4_paused':True}
    if summary['formal_stream_audit']['old_P1_formal_directory_exists']:raise ValueError('old formal evidence requires root reconciliation')
    write_new(OUT/'preflight.json',summary)
else:
    commit=sys.argv[1]
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('full frozen source')
    files={str((ROOT/name).resolve()):digest for name,digest in old_python.items()}
    selection_path=ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01/spawn_chain20261022/runtime_selection.json'
    for name,digest in read(selection_path)['files'].items():
        path=(ROOT/name).resolve()
        if sha(path)!=digest:raise ValueError('old selected runtime evidence changed')
        files[str(path)]=digest
    extras=new_names+['AGENTS.md','docs/engineering_loop.md',
        'reports/astra6_after_rational_20261022.md','reports/astra6_after_rational_20261022_raw.txt',
        'reports/finite_grammar_execution_20261022_protocol.md','reports/finite_grammar_execution_20261022_review.md',
        'results/finite_grammar_execution_20261022/preflight.json',str(selection_path.relative_to(ROOT)),
        str((OLD/'bindings/qa01.json').relative_to(ROOT))]
    result={'source_commit':commit,'no_process_quotas':True}
    for key,path in [('pool_manifest',OLD/'freeze_pool01/pool.json'),('teacher_identity',OLD/'freeze_pool01/teacher.json'),
        ('old_qa_complete',OLD/'qa01/complete.json'),('old_qa_receipt',OLD/'native_qa01/execution.json')]:
        result[key]=ref(path);extras.append(str(path.relative_to(ROOT)))
    for name in extras:files[str((ROOT/name).resolve())]=sha(ROOT/name)
    result['files']=dict(sorted(files.items()))
    write_new(BASE/'binding.json',result)
    print(json.dumps({'source_commit':commit,'bound_files':len(files),'binding_sha256':sha(BASE/'binding.json')}))
