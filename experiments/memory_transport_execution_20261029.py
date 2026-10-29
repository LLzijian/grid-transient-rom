"""Finite native A/S/P owner. UNKNOWN release blocks every executable entrance."""
from __future__ import annotations
import argparse,importlib,json,math,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.r_memory_execution_20261025 import read,sha,write_new,commit_id,source_closure,python_command,typed_identity
BASE=ROOT/'private_data/memory_transport_20261029/study01'
PARENT=ROOT/'private_data/weak_force_identification_20261029/study01'
PARENT_COMMIT='3b01f19055203734c370d32eba4cea8a0555f8d3'
PARENT_COMPLETE_SHA='e834b61ee975192946f1fe8ecdaf1cdd9d4eaea0bc22b0f25a1ee3fe26e7c611'
PARENT_SUMMARY_SHA='be95331073bd0fc039e5d23561c44724bd04334ef120e212b917d4bd7bada94e'
RELEASE_RECORD=ROOT/'results/memory_transport_20261029/execution_release.json';RELEASE_SHA='UNKNOWN'
PHASES=('prepare','qualify','qa','study','summary')
OUTPUTS=dict(prepare='prepare.json',qualify='qualification/summary.json',qa='qa/summary.json',study='study/summary.json',summary='summary.json')
STUDY_MODULE='experiments.memory_transport_study_20261029'
SOURCE_ROOTS=('experiments/memory_transport_execution_20261029.py','experiments/memory_transport_study_20261029.py',
 'experiments/memory_transport_model_20261029.py','tests/test_memory_transport_model_20261029.py',
 'tests/test_memory_transport_execution_20261029.py','tests/test_memory_transport_study_20261029.py')
DOCUMENTS=('AGENTS.md','docs/engineering_loop.md','docs/research_execution_20261028.md',
 'reports/memory_transport_20261029_protocol.md','reports/astra6_force_structure_next_20261029.md',
 'reports/force_structure_next_20261029_challenge.md','reports/memory_transport_20261029_model_review.md')
FIXED_RECORDS={'experiments/memory_transport_model_20261029.py':'c9040bd65026edd0939b6c1046e21205388ffe1d2d5d69752d67144416101494',
 'tests/test_memory_transport_model_20261029.py':'0e043c1200aed9de6d11e21dc57900c23ec6922dd75d71ea649e9dc39ba9989a',
 'reports/memory_transport_20261029_protocol.md':'3fcb1ed8ece5a6465c2b912e83645c2b90a1306a9a299ba5e351728e659d0509',
 'results/memory_transport_20261029/model_cpu_qualification.json':'4d3068e30635c2cf06a7cbe4b35d47681833475001a09e8bfb755db498082c41',
 'results/memory_transport_20261029/model_cuda_qualification.json':'5f0f37f28f48749ea9bf196c0e4c8d1c827958bc25abe8ebf483aad40025ea9f'}


def checked_base(base=BASE):
    base=Path(base).resolve()
    if base!=BASE.resolve():raise ValueError('registered memory transport study01 only')
    return base


def require_release():
    if len(RELEASE_SHA)!=64 or any(c not in '0123456789abcdef' for c in RELEASE_SHA):raise ValueError('root execution release UNKNOWN; no executable phase admitted')
    if sha(RELEASE_RECORD)!=RELEASE_SHA:raise ValueError('root release bytes changed')
    value=read(RELEASE_RECORD)
    if any(value.get(k) is not True for k in ('parent_remote_archival_verified','independent_implementation_review_passed','root_execution_authorized')):
        raise ValueError('actual archival/review/execution release incomplete')
    if value.get('parent_complete_sha256')!=PARENT_COMPLETE_SHA or value.get('optimizer_upper_bound')!=1848:raise ValueError('released parent/scope differs')
    return value


def fixed_evidence():
    files={}
    for name,digest in FIXED_RECORDS.items():
        path=ROOT/name
        if sha(path)!=digest:raise ValueError('frozen numerical model/protocol/evidence changed: '+name)
        files[str(path)]=digest
    return files


def parent_proof(files,*,state=None):
    from experiments import weak_force_execution_20261029 as old
    from experiments import uncapped_windows_job_20261021 as job
    if sha(PARENT/'complete.json')!=PARENT_COMPLETE_SHA or sha(PARENT/'summary.json')!=PARENT_SUMMARY_SHA:raise ValueError('closed weak parent terminal differs')
    if (PARENT/'active.lock').exists():raise ValueError('closed weak parent retains active lock')
    binding=read(PARENT/'binding.json')
    if binding['source_commit']!=PARENT_COMMIT:raise ValueError('weak parent source differs')
    state=job.identity_state if state is None else state;identities={};receipts={}
    for phase in old.PHASES:
        paths,_=old.completed_phase(PARENT,PARENT_COMMIT,phase);files.update(paths)
        path=PARENT/'native'/phase/'receipt.json';receipt=old.clean_receipt(read(path),PARENT_COMMIT,phase)
        current=state(receipt['worker'])
        if current not in ('GONE','REUSED_PID'):raise ValueError('recorded closed parent identity live/unknown: '+str(current))
        identities[phase]=dict(recorded_worker_identity=dict(receipt['worker']),current_identity_state=current);receipts[phase]=sha(path)
    for name in ('complete.json','summary.json','binding.json','prepare.json','schedule.json'):
        files[str(PARENT/name)]=sha(PARENT/name)
    runtime=binding['runtime']
    for name in runtime['files']:
        if name not in binding['files']:raise ValueError('runtime missing from proven parent binding')
        files[name]=binding['files'][name]
    return runtime,dict(parent_complete_sha256=PARENT_COMPLETE_SHA,parent_summary_sha256=PARENT_SUMMARY_SHA,
        parent_source_commit=PARENT_COMMIT,six_parent_receipt_hashes=receipts,current_parent_worker_identities=identities,
        parent_optimizer_updates=0,parent_QA_executed=False)


def check_locks(base=BASE,owned_identity=None):
    from experiments import weak_force_execution_20261029 as old
    checked_base(base);old.check_locks(PARENT);path=Path(base)/'active.lock'
    if path.exists():
        actual=typed_identity(read(path).get('worker'))
        if owned_identity is None or any(actual[k]!=owned_identity[k] for k in ('pid','creation_filetime_utc')):raise ValueError('new native lock is not owned')
    elif owned_identity is not None:raise ValueError('owned worker requires actual native lock')


def freeze(source_commit,base=BASE):
    require_release();commit_id(source_commit);base=checked_base(base)
    if base.exists():raise FileExistsError('preserve sole source freeze and scientific outputs')
    check_locks(base)
    from experiments import memory_transport_study_20261029 as study
    files={str(p.resolve()):sha(p) for p in source_closure(ROOT,SOURCE_ROOTS)+[ROOT/n for n in DOCUMENTS]+[RELEASE_RECORD]};files.update(fixed_evidence())
    study.source_identity()
    for name,digest in study.SOURCE_SHA.items():files[str(study.SOURCE/name)]=digest
    files[str(study.PLANE)]=study.PLANE_SHA;files[str(study.PLANE_PUBLIC)]=study.PLANE_PUBLIC_SHA
    for name,digest in study.EVIDENCE_SHA.items():
        path=study.SGM/name
        if sha(path)!=digest:raise ValueError('reusable source/reference evidence differs')
        files[str(path)]=digest
    for name in ('qualification/R_N_fit297/evidence.json','study/A_28311/fit000096/evidence.json'):
        for entry in read(study.SGM/name).get('prediction_files',[]):
            if sha(entry['path'])!=entry['sha256']:raise ValueError('source cached prediction changed')
            files[str(Path(entry['path']).resolve())]=entry['sha256']
    profile=study.SGM/'study/A_28311/fit000096/state_gated_profile.json'
    previous_files=read(PARENT/'binding.json')['files'];previous_profile_sha=previous_files.get(str(profile))
    if previous_profile_sha is None or sha(profile)!=previous_profile_sha:raise ValueError('A96 profile differs from closed parent binding')
    files[str(profile)]=previous_profile_sha
    runtime,proof=parent_proof(files)
    binding=dict(source_commit=source_commit,base=str(base),files=files,runtime=runtime,parent_proof=proof,
        release_sha256=RELEASE_SHA,no_process_quotas=True,optimizer_call_upper_bound=1848,category_limits=study.LIMITS,
        source_plane_previously_consumed=1,source_eigh_calls_this_loop=0,NNLS_calls=0,mathematical_suite_rerun=False,
        automatic_resume=False,project_val_test_accessed=False)
    base.mkdir(parents=True,exist_ok=False);write_new(base/'binding.json',binding);return binding


def check_binding(binding,base=BASE):
    require_release();base=checked_base(base)
    from experiments import uncapped_windows_job_20261021 as job
    from experiments import memory_transport_study_20261029 as study
    if (binding.get('base')!=str(base) or binding.get('release_sha256')!=RELEASE_SHA or binding.get('optimizer_call_upper_bound')!=1848
            or binding.get('category_limits')!=study.LIMITS or binding.get('no_process_quotas') is not True
            or binding.get('source_eigh_calls_this_loop')!=0 or binding.get('NNLS_calls')!=0):raise ValueError('current finite reviewed binding required')
    commit_id(binding['source_commit']);job.verify_files(binding['files'])
    for name,digest in fixed_evidence().items():
        if binding['files'].get(name)!=digest:raise ValueError('required qualified bytes missing from native binding')
    command=python_command([])
    if any(command.get(k)!=v for k,v in binding['runtime']['command'].items()):raise ValueError('actual runtime command differs')
    return command


def clean_receipt(value,source_commit,phase):
    from experiments import weak_force_execution_20261029 as old
    return old.clean_receipt(value,source_commit,phase)


def cross_model_evidence(base,result):
    path=(base/'qualification/A_A96_compatibility.json').resolve();link=result.get('A_A96_compatibility',{})
    if result.get('A_A96_compatibility_qualified') is not True or link.get('path')!=str(path) or link.get('sha256')!=sha(path):raise ValueError('A0/A96 exact path/hash/qualified required')
    value=read(path);ids=read(base/'prepare.json').get('fit_ids',[])
    from experiments import memory_transport_study_20261029 as study
    if (value.get('qualified') is not True or value.get('record_count')!=297 or len(ids)!=297 or len(set(ids))!=297 or value.get('event_ids')!=ids
            or value.get('original_schema_aliases')!={'R_N':'frozen_A96','A':'current_A0'}
            or value.get('source_evidence_sha256')!=study.EVIDENCE_SHA['study/A_28311/fit000096/evidence.json']
            or value.get('source_A96_prediction_recomputed') is not False or value.get('scope')!='A96_A0_fit297_bidirectional_RK4_fine005_base01_8s_actual15'
            or value.get('bidirectional') is not True or any(value.get(k)!=v for k,v in dict(history_rows=11,initial_observation_index=10,fine_dt=.005,output_dt=.01,coarse_stride=2).items())):
        raise ValueError('A0/A96 source/identity/scope differs')
    rows=value.get('rows',[])
    if len(rows)!=297 or [r.get('event_id') for r in rows]!=ids:raise ValueError('all registered fit297 rows required')
    from experiments.fixed_edge_pair_20261019 import scientific_qualification
    from experiments.frequency_dissipation_20261021 import actual15_qualified
    for row in rows:
        n=row.get('actual_steps')
        if type(n) is not int or not 1<=n<=1500 or row.get('qualified') is not True:raise ValueError('unresolved genuine interval')
        for scope in ('8s','actual15'):
            interval=row.get('intervals',{}).get(scope,{})
            if interval.get('qualified') is not True or interval.get('actual_output_steps')!=(800 if scope=='8s' else n) or any(interval.get(k) is not True for k in ('full_R_N_fine_finite','full_A_fine_finite')):raise ValueError('A0/A96 full fine scope unresolved')
            for direction in ('R_N_to_A','A_to_R_N'):
                pair=interval.get(direction+'_pair');scored=interval.get(direction+'_scored_numerics')
                if not isinstance(pair,dict) or not isinstance(scored,list) or not scientific_qualification(pair,scored):raise ValueError('original bidirectional numerical gate failed')
                if scope=='actual15' and not actual15_qualified(interval.get(direction+'_actual_interval',{}),pair):raise ValueError('original genuine-interval gate failed')
    return {str(path):sha(path)}


def qa_evidence(base,result):
    from experiments import memory_transport_study_20261029 as study
    eligible=result.get('eligible_variants',[]);models=result.get('models',[])
    if len(eligible)!=len(set(eligible)) or not set(eligible)<=set(study.VARIANTS) or len(models)!=len({r['variant'] for r in models}) or not {r['variant'] for r in models}<=set(study.VARIANTS):raise ValueError('unique registered QA models required')
    if set(eligible)!={r['variant'] for r in models if r.get('status')=='PASS'}:raise ValueError('eligible QA models differ from actual passes')
    counts=study.counts(base);ledger=base/'optimizer_calls.jsonl';records=[] if not ledger.exists() else [json.loads(l) for l in ledger.read_text(encoding='utf-8').splitlines()]
    projection=[r for r in records if r['category']=='qa'];positions=[j for j,r in enumerate(records) if r['category']=='qa']
    if positions and any(r['category']=='formal' and j<max(positions) for j,r in enumerate(records)):raise ValueError('formal call preceded QA closure')
    allowed={f'qa_{v}_{label}' for v in study.VARIANTS for label in ('continuous','restored')}
    if any(r['run_id'] not in allowed for r in projection):raise ValueError('unregistered QA trajectory')
    for variant in eligible:
        row=next(r for r in models if r['variant']==variant)
        if row.get('actual_optimizer_calls')!=36:raise ValueError('each qualified model needs real36 QA')
        for label in ('continuous','restored'):
            run=f'qa_{variant}_{label}'
            for event in ('optimizer_step_attempt','optimizer_step_completed'):
                if sorted(r['step'] for r in projection if r['run_id']==run and r['event']==event)!=list(range(1,19)):raise ValueError('each qualified QA trajectory needs actual18 steps')
    if set(eligible)==set(study.VARIANTS) and counts['categories']['qa']!={'attempted':108,'completed':108}:raise ValueError('full QA must close actual108/108')
    if any(r['category']=='qa' and r['run_id'].split('_')[1] in eligible for r in counts['pending']):raise ValueError('qualified QA has pending actual call')
    path=(base/'qa/optimizer_projection.json').resolve();link=result.get('qa_journal_projection',{})
    if link.get('path')!=str(path) or link.get('sha256')!=sha(path) or read(path)!=projection:raise ValueError('immutable QA projection differs from actual ledger')
    return {str(path):sha(path)}


def performance_evidence(base,variant,row):
    """Check the fixed two actual J/backward records without new computation."""
    folder=(base/'qualification'/(variant+'_performance')).resolve();path=folder/'summary.json';link=row.get('performance_evidence',{})
    if link.get('path')!=str(path) or link.get('sha256')!=sha(path):raise ValueError('exact performance path/hash required')
    value=read(path);prepared=read(base/'prepare.json')
    if (value.get('status')!='PASS' or value.get('equivalent') is not True or value.get('variant')!=variant
            or value.get('event_ids')!=prepared['cost12_ids'] or len(value.get('event_ids',[]))!=12
            or any(value.get(k)!=v for k,v in dict(horizon_s=8.,method='rk4',dtype='float64',optimizer_calls=0,atol=1e-10,rtol=1e-8).items())):
        raise ValueError('fixed performance sample/range/precision differs')
    rows=value.get('ordered_runs',[])
    if [r.get('implementation') for r in rows]!=['uncached','cached'] or {p.name for p in folder.glob('*_attempt.json')}!={'uncached_attempt.json','cached_attempt.json'}:
        raise ValueError('only the fixed uncached-then-cached pair is registered')
    files={str(path):sha(path)}
    expected={'kappa_raw','lambda_raw','a_bar'}|({'s_raw'} if variant=='S' else {'d_raw'} if variant=='P' else set())
    def flat(value):
        return [x for child in value for x in flat(child)] if isinstance(value,list) else [value]
    vectors=[]
    for r in rows:
        name=r['implementation'];attempt=folder/(name+'_attempt.json');actual=folder/(name+'_result.json')
        if read(attempt)!=dict(variant=variant,implementation=name,optimizer_calls=0) or read(actual)!=r or r.get('all_raw_gradients') is not True or set(r.get('gradients',{}))!=expected:
            raise ValueError('actual performance/gradient records differ')
        vector=[r.get('J')]+[x for key in sorted(expected) for x in flat(r['gradients'][key])]
        if len(vector)!=(48 if variant=='A' else 49) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in vector):
            raise ValueError('finite complete J/raw gradient vector required')
        vectors.append(vector);files.update({str(p):sha(p) for p in (attempt,actual)})
    if any(abs(a-b)>1e-10+1e-8*abs(b) for a,b in zip(*vectors)):raise ValueError('original complete J/all-gradient equivalence failed')
    return files


def completed_phase(base,source_commit,phase):
    folder=base/'ownership'/phase;rp=base/'native'/phase/'receipt.json';receipt=clean_receipt(read(rp),source_commit,phase)
    binding=read(base/'binding.json');admission=read(folder/'admission.json')
    if receipt['source_before'].get(str(folder/'admission.json'))!=sha(folder/'admission.json') or any(receipt['source_before'].get(n)!=d for n,d in {**binding['files'],**admission['inputs']}.items()):raise ValueError('actual native admission/source identity changed')
    path=base/OUTPUTS[phase];result=read(path);complete=read(folder/'complete.json')
    if complete.get('status')!='COMPLETED' or complete.get('phase')!=phase or complete.get('source_commit')!=source_commit or complete.get('scientific_output_sha256')!=sha(path):raise ValueError('native completion/scientific output differs')
    if result.get('val_read') is not False or result.get('test_read') is not False:raise ValueError('closed val/test must be explicit')
    extra={}
    if phase=='qualify':
        from experiments import memory_transport_study_20261029 as study
        eligible=result.get('eligible_variants',[])
        if len(set(eligible))!=len(eligible) or not set(eligible)<=set(study.VARIANTS) or set(result.get('per_variant',{}))!=set(study.VARIANTS):raise ValueError('explicit per-variant admission required')
        for v,row in result['per_variant'].items():
            if row.get('training_admission') is not (v in eligible):raise ValueError('variant qualification/admission differs')
            if v in eligible and any(row.get(k) is not True for k in ('cpu_qualified','cpu_cuda_pair_pass','performance_equivalent','finite_12x8_backward','fullfit297_qualified')):raise ValueError('numerical/performance gate missing')
            if v in eligible:extra.update(performance_evidence(base,v,row))
        attempts=list((base/'qualification').glob('*_performance/*_attempt.json'))
        if len(attempts)>6 or result.get('actual_performance_attempts')!=len(attempts) or result.get('registered_performance_attempts_upper_bound')!=6:
            raise ValueError('actual fixed performance attempt count differs')
        if 'A' in eligible:extra.update(cross_model_evidence(base,result['per_variant']['A']))
    if phase=='qa':
        qualified=read(base/'qualification/summary.json').get('eligible_variants',[])
        if {r['variant'] for r in result.get('models',[])}!=set(qualified):raise ValueError('actual QA scope differs from numerically qualified variants')
        extra.update(qa_evidence(base,result))
    return {str(rp):sha(rp),str(path):sha(path),**extra},result


def phase_inputs(base,source_commit,phase):
    if phase not in PHASES:raise ValueError('unregistered phase')
    files={str(base/'binding.json'):sha(base/'binding.json')}
    for previous in PHASES[:PHASES.index(phase)]:
        paths,_=completed_phase(base,source_commit,previous);files.update(paths);path=base/OUTPUTS[previous]
        if previous=='prepare':files[str(base/'schedule.json')]=sha(base/'schedule.json')
        else:
            for item in path.parent.rglob('*'):
                if item.is_file():files[str(item.resolve())]=sha(item)
    # Never freeze the append-only optimizer ledger. Immutable QA projection is
    # checked above; failed quarantined variants cannot become eligible again.
    from experiments import memory_transport_study_20261029 as study
    for pending in study.counts(base)['pending']:
        if pending['category']=='qa' and phase in ('study','summary'):
            variant=pending['run_id'].split('_')[1] if pending['run_id'].startswith('qa_') else None
            rows=read(base/'qa/summary.json').get('models',[]);row=next((r for r in rows if r['variant']==variant),{})
        elif pending['category']=='formal' and phase=='summary':
            row=read(base/'study'/pending['run_id']/'result.json')
        else:raise ValueError('pending actual attempt has no closed affected-run evidence')
        if row.get('status')!='NUMERICAL_UNRESOLVED' or row.get('failure_is_physical_instability') is not False or row.get('quarantine') is None:
            raise ValueError('pending call must stay quarantined in its failed run; no automatic restore')
    return files


def worker(base,source_commit,phase):
    require_release()
    from experiments import uncapped_windows_job_20261021 as job
    base=checked_base(base);binding=read(base/'binding.json');folder=base/'ownership'/phase;command=binding['runtime']['command']
    if binding['source_commit']!=source_commit:raise ValueError('worker source commit differs')
    identity={**job.identity(),'physical_image':job.physical_image(job.api(),job.api().GetCurrentProcess()),'logical_executable':command['logical_executable'],
        'runtime_scheme':command['runtime_scheme'],'source_commit':source_commit,'phase':phase}
    identity.update(pid_decimal=str(identity['pid']),creation_filetime_decimal=str(identity['creation_filetime_utc']));write_new(folder/'worker.json',identity)
    while not (folder/'owned').is_file():time.sleep(.02)
    try:
        check_binding(binding,base);job.verify_files(read(folder/'admission.json')['inputs']);phase_inputs(base,source_commit,phase);check_locks(base,identity)
        module=importlib.import_module(STUDY_MODULE);result=getattr(module,phase)(base) if phase in ('prepare','summary') else getattr(module,phase)(base,device='cuda')
        write_new(folder/'complete.json',dict(status='COMPLETED',source_commit=source_commit,phase=phase,scientific_output_sha256=sha(base/OUTPUTS[phase]),optimizer_calls=module.counts(base)))
    except BaseException as exc:
        write_new(folder/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),phase=phase,source_commit=source_commit));raise


def launch(base,source_commit,phase):
    require_release()
    from experiments import uncapped_windows_job_20261021 as job
    base=checked_base(base);binding=read(base/'binding.json')
    if binding['source_commit']!=commit_id(source_commit):raise ValueError('source commit differs')
    check_binding(binding,base);check_locks(base);inputs=phase_inputs(base,source_commit,phase);folder=base/'ownership'/phase;native=base/'native'/phase
    if folder.exists() or native.exists() or (base/OUTPUTS[phase]).exists():raise FileExistsError('sole attempt already exists; no automatic retry/recovery')
    folder.mkdir(parents=True,exist_ok=False);write_new(folder/'admission.json',dict(source_commit=source_commit,phase=phase,inputs=inputs,binding_sha256=sha(base/'binding.json'),dtype='float64',automatic_resume=False))
    files={**binding['files'],**inputs,str(folder/'admission.json'):sha(folder/'admission.json')}
    command=python_command(['-B',str(Path(__file__).resolve()),phase,'--source-commit',source_commit,'--base',str(base),'--worker'])
    def owned(_identity):
        with (folder/'owned').open('x',encoding='utf-8') as stream:stream.write('owned\n')
    receipt=job.supervise(command,native,base/'active.lock',files,ROOT,worker_receipt=folder/'worker.json',on_worker_owned=owned,provenance=dict(source_commit=source_commit,phase=phase))
    clean_receipt(receipt,source_commit,phase);completed_phase(base,source_commit,phase);return receipt


def run(base,source_commit):
    require_release();base=checked_base(base)
    if not (base/'binding.json').exists():freeze(source_commit,base)
    for phase in PHASES:launch(base,source_commit,phase)
    from experiments import memory_transport_study_20261029 as study
    write_new(base/'complete.json',dict(status='COMPLETED',source_commit=source_commit,optimizer_calls=study.counts(base),model_adopted=False))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('freeze',*PHASES,'run'))
    parser.add_argument('--source-commit',required=True,type=commit_id);parser.add_argument('--base',type=Path,default=BASE);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv);base=checked_base(args.base)
    if args.worker:
        if args.phase not in PHASES:parser.error('only registered phase workers')
        worker(base,args.source_commit,args.phase)
    elif args.phase=='freeze':freeze(args.source_commit,base)
    elif args.phase=='run':run(base,args.source_commit)
    else:launch(base,args.source_commit,args.phase)


if __name__=='__main__':main()
