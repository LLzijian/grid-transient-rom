"""Native ownership and immutable phase admission for the finite D/W study.

Import performs no event access, numerical work, Job launch or quota change.
UNKNOWN execution release is an intentional hard stop, including standalone phases.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.r_memory_execution_20261025 import read,sha,write_new,commit_id,source_closure,python_command,typed_identity
BASE=ROOT/'private_data/weak_force_identification_20261029/study01'
PARENT=ROOT/'private_data/state_gated_memory_20261028/study01'
PARENT_COMMIT='0215a9f6a5582d93bd7f7ebfd5e34d7be1d1e1c6'
PARENT_COMPLETE_SHA='dc10355064f293f4b05eb26f92429317c4d6038ebe4f94159d146bb62bc9bf1e'
PARENT_SUMMARY_SHA='7e77c7f4140d0522a8bb8bd373a3d0a793bbcffeda3b867c0fab67cb821eb525'
PARENT_QUALIFICATION_SHA='a169a7e798b87ce244b0c45eb27b6ce75e38f55ed014e28382e71d918c113288'
RELEASE_RECORD=ROOT/'results/weak_force_identification_20261029/execution_release.json'
RELEASE_SHA='UNKNOWN'  # Root binds after actual review + parent public remote archival.
PHASES=('prepare','weak','qualify','qa','study','summary')
OUTPUTS=dict(prepare='prepare.json',weak='weak/summary.json',qualify='qualification/summary.json',
             qa='qa/summary.json',study='study/summary.json',summary='summary.json')
SOURCE_ROOTS=('experiments/weak_force_execution_20261029.py','experiments/weak_force_study_20261029.py',
 'experiments/weak_force_identification_20261029.py','experiments/test_weak_force_identification_20261029.py',
 'tests/test_weak_force_execution_20261029.py','tests/test_weak_force_study_20261029.py')
DOCUMENTS=('AGENTS.md','docs/engineering_loop.md','reports/weak_force_identification_20261029_protocol.md',
 'reports/astra6_weak_force_execution_plan_20261029.md','reports/astra6_weak_force_next_design_20261029.md',
 'reports/weak_force_next_design_20261029_challenge.md')
STUDY_MODULE='experiments.weak_force_study_20261029'


def checked_base(base=BASE):
    base=Path(base).resolve()
    if base!=BASE.resolve():raise ValueError('only registered new study01 directory is allowed')
    return base


def require_release():
    if len(RELEASE_SHA)!=64 or any(c not in '0123456789abcdef' for c in RELEASE_SHA):
        raise ValueError('root execution release is UNKNOWN; no phase or fixture is authorized')
    if sha(RELEASE_RECORD)!=RELEASE_SHA:raise ValueError('frozen root release changed')
    value=read(RELEASE_RECORD)
    if any(value.get(k) is not True for k in ('parent_remote_archival_verified','independent_implementation_review_passed','root_execution_authorized')):
        raise ValueError('root review/archive/execution release incomplete')
    if value.get('parent_complete_sha256')!=PARENT_COMPLETE_SHA or value.get('optimizer_upper_bound')!=976:
        raise ValueError('root release parent identity or finite scientific allowance differs')
    return value


def parent_proof(files,*,state=None):
    from experiments import state_gated_memory_execution_20261028 as sgm
    from experiments import uncapped_windows_job_20261021 as job
    if sha(PARENT/'complete.json')!=PARENT_COMPLETE_SHA or sha(PARENT/'summary.json')!=PARENT_SUMMARY_SHA:
        raise ValueError('actual closed parent terminal evidence differs')
    if sha(PARENT/'qualification/summary.json')!=PARENT_QUALIFICATION_SHA:
        raise ValueError('frozen reusable R_N profile/timing changed')
    binding=read(PARENT/'binding.json')
    if binding['source_commit']!=PARENT_COMMIT:raise ValueError('parent source commit differs')
    if (PARENT/'active.lock').exists():raise ValueError('parent active lock still present')
    state=job.identity_state if state is None else state
    receipt_hashes={}
    for phase in sgm.PHASES:
        paths,_=sgm._completed_phase(PARENT,PARENT_COMMIT,phase);files.update(paths)
        receipt_path=PARENT/'native'/phase/'receipt.json';receipt=sgm.clean_receipt(read(receipt_path),PARENT_COMMIT,phase)
        if state(receipt['worker'])!='GONE':raise ValueError('parent owner is live/unknown/reused')
        receipt_hashes[phase]=sha(receipt_path)
    for name in ('complete.json','summary.json','qualification/summary.json','binding.json','prepare.json','schedule.json'):
        files[str(PARENT/name)]=sha(PARENT/name)
    # The prior full runtime proof is retained; no new runtime executable scan.
    runtime=binding['runtime']
    for name in runtime['files']:
        if name not in binding['files']:raise ValueError('runtime not in proven parent binding')
        files[name]=binding['files'][name]
    return runtime,dict(parent_complete_sha256=PARENT_COMPLETE_SHA,parent_summary_sha256=PARENT_SUMMARY_SHA,
                        five_parent_receipt_hashes=receipt_hashes,parent_source_commit=PARENT_COMMIT)


def check_locks(base=BASE,owned_identity=None):
    from experiments import state_gated_memory_execution_20261028 as sgm
    checked_base(base);sgm.check_locks(PARENT)
    own=Path(base)/'active.lock'
    if own.exists():
        actual=typed_identity(read(own).get('worker'))
        if owned_identity is None or any(actual[k]!=owned_identity[k] for k in ('pid','creation_filetime_utc')):
            raise ValueError('unowned current native lock')
    elif owned_identity is not None:raise ValueError('owned worker requires active lock')


def freeze(source_commit,base=BASE):
    require_release();commit_id(source_commit);base=checked_base(base)
    if base.exists():raise FileExistsError('immutable new study already exists')
    check_locks(base)
    from experiments import weak_force_study_20261029 as study
    paths=source_closure(ROOT,SOURCE_ROOTS)+[ROOT/p for p in DOCUMENTS]+[RELEASE_RECORD]
    files={str(p.resolve()):sha(p) for p in paths}
    for name,digest in study.SOURCE_SHA.items():
        path=study.SOURCE/name
        if sha(path)!=digest:raise ValueError('frozen A96 asset changed')
        files[str(path)]=digest
    for name,digest in study.EVIDENCE_SHA.items():
        path=PARENT/name
        if sha(path)!=digest:raise ValueError('frozen reusable evidence changed')
        files[str(path)]=digest
    # Profiles and cached prediction assets are bound once, without reading events.
    for name in ('qualification/R_N_fit297/evidence.json','study/A_28311/fit000096/evidence.json'):
        evidence=read(PARENT/name)
        for entry in evidence.get('prediction_files',[]):
            if sha(entry['path'])!=entry['sha256']:raise ValueError('source cached prediction changed')
            files[str(Path(entry['path']).resolve())]=entry['sha256']
    profile=PARENT/'study/A_28311/fit000096/state_gated_profile.json';files[str(profile)]=sha(profile)
    runtime,proof=parent_proof(files)
    binding=dict(source_commit=source_commit,base=str(base),files=files,runtime=runtime,parent_proof=proof,
        release_sha256=RELEASE_SHA,no_process_quotas=True,optimizer_call_upper_bound=976,
        category_limits=study.LIMITS,automatic_resume=False,real_nnls_upper_bound=1,
        scientific_scope='fixed A weak start versus continuous A96; no physical recovery claim',project_val_test_accessed=False)
    base.mkdir(parents=True,exist_ok=False);write_new(base/'binding.json',binding);return binding


def check_binding(binding,base=BASE):
    require_release();base=checked_base(base)
    from experiments import uncapped_windows_job_20261021 as job
    from experiments import weak_force_study_20261029 as study
    if binding.get('base')!=str(base) or binding.get('release_sha256')!=RELEASE_SHA or binding.get('optimizer_call_upper_bound')!=976 or binding.get('category_limits')!=study.LIMITS or binding.get('no_process_quotas') is not True:
        raise ValueError('current finite root-reviewed binding required')
    commit_id(binding['source_commit']);job.verify_files(binding['files'])
    command=python_command([])
    if any(command.get(k)!=v for k,v in binding['runtime']['command'].items()):raise ValueError('runtime command changed')
    return command


def clean_receipt(value,source_commit,phase):
    from experiments import state_gated_memory_execution_20261028 as sgm
    return sgm.clean_receipt(value,source_commit,phase)


def completed_phase(base,source_commit,phase):
    folder=base/'ownership'/phase;receipt_path=base/'native'/phase/'receipt.json'
    receipt=clean_receipt(read(receipt_path),source_commit,phase)
    binding=read(base/'binding.json');admission=read(folder/'admission.json')
    if receipt['source_before'].get(str(folder/'admission.json'))!=sha(folder/'admission.json'):
        raise ValueError('native admission identity changed')
    if any(receipt['source_before'].get(n)!=d for n,d in {**binding['files'],**admission['inputs']}.items()):
        raise ValueError('native phase did not use exact bound scientific inputs')
    result_path=base/OUTPUTS[phase];complete=read(folder/'complete.json');result=read(result_path)
    if complete.get('status')!='COMPLETED' or complete.get('source_commit')!=source_commit or complete.get('phase')!=phase or complete.get('scientific_output_sha256')!=sha(result_path):
        raise ValueError('phase completion/result differs')
    if result.get('val_read') is not False or result.get('test_read') is not False:raise ValueError('closed val/test must be explicit')
    if phase=='qualify' and result.get('training_admission'):
        if result.get('eligible_variants')!=['D','W']:raise ValueError('no unpaired training admission')
        for v in ('D','W'):
            if any(result['per_variant'][v].get(k) is not True for k in ('cpu_qualified','cpu_cuda_pair_pass','finite_12x8_backward','fullfit297_qualified','training_admission')):
                raise ValueError('D/W original numerical guard missing')
    if phase=='qa' and result.get('training_admission'):
        if result.get('eligible_variants')!=['D','W'] or len(result.get('models',[]))!=2 or any(r.get('status')!='PASS' or r.get('actual_optimizer_calls')!=36 for r in result['models']):
            raise ValueError('paired actual 36+36 QA required')
    return {str(receipt_path):sha(receipt_path),str(result_path):sha(result_path)},result


def phase_inputs(base,source_commit,phase):
    if phase not in PHASES:raise ValueError('unregistered phase')
    files={str(base/'binding.json'):sha(base/'binding.json')}
    for previous in PHASES[:PHASES.index(phase)]:
        paths,_=completed_phase(base,source_commit,previous);files.update(paths)
    # Every prior immutable scientific artifact is included, including candidate
    # coefficients, prediction caches and real NNLS attempt/completion evidence.
    for previous in PHASES[:PHASES.index(phase)]:
        path=base/OUTPUTS[previous]
        if previous=='prepare':
            files[str(base/'schedule.json')]=sha(base/'schedule.json')
        else:
            for item in path.parent.rglob('*'):
                if item.is_file():files[str(item.resolve())]=sha(item)
    from experiments import weak_force_study_20261029 as study
    if study.counts(base)['pending'] and phase!='summary':raise ValueError('pending actual optimizer attempt requires root reconciliation')
    return files


def worker(base,source_commit,phase):
    require_release()
    from experiments import uncapped_windows_job_20261021 as job
    base=checked_base(base);binding=read(base/'binding.json');folder=base/'ownership'/phase
    command=binding['runtime']['command']
    identity={**job.identity(),'physical_image':job.physical_image(job.api(),job.api().GetCurrentProcess()),
        'logical_executable':command['logical_executable'],'runtime_scheme':command['runtime_scheme'],
        'source_commit':source_commit,'phase':phase}
    identity.update(pid_decimal=str(identity['pid']),creation_filetime_decimal=str(identity['creation_filetime_utc']))
    write_new(folder/'worker.json',identity)
    while not (folder/'owned').is_file():time.sleep(.02)
    try:
        check_binding(binding,base);job.verify_files(read(folder/'admission.json')['inputs'])
        phase_inputs(base,source_commit,phase);check_locks(base,owned_identity=identity)
        module=importlib.import_module(STUDY_MODULE)
        result=getattr(module,phase)(base) if phase in ('prepare','summary') else getattr(module,phase)(base,device='cuda')
        write_new(folder/'complete.json',dict(status='COMPLETED',source_commit=source_commit,phase=phase,
            scientific_output_sha256=sha(base/OUTPUTS[phase]),optimizer_calls=module.counts(base)))
    except BaseException as exc:
        write_new(folder/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),phase=phase,source_commit=source_commit));raise


def launch(base,source_commit,phase):
    require_release()
    from experiments import uncapped_windows_job_20261021 as job
    base=checked_base(base);binding=read(base/'binding.json')
    if binding['source_commit']!=commit_id(source_commit):raise ValueError('source commit differs')
    check_binding(binding,base);check_locks(base);inputs=phase_inputs(base,source_commit,phase)
    folder=base/'ownership'/phase;native=base/'native'/phase
    if folder.exists() or native.exists() or (base/OUTPUTS[phase]).exists():
        raise FileExistsError('preserve sole phase attempt; no automatic repair/resume or NNLS retry')
    folder.mkdir(parents=True,exist_ok=False)
    write_new(folder/'admission.json',dict(source_commit=source_commit,phase=phase,inputs=inputs,
        binding_sha256=sha(base/'binding.json'),dtype='float64',automatic_resume=False))
    files={**binding['files'],**inputs,str(folder/'admission.json'):sha(folder/'admission.json')}
    command=python_command(['-B',str(Path(__file__).resolve()),phase,'--source-commit',source_commit,'--base',str(base),'--worker'])
    def owned(_identity):
        with (folder/'owned').open('x',encoding='utf-8') as stream:stream.write('owned\n')
    receipt=job.supervise(command,native,base/'active.lock',files,ROOT,worker_receipt=folder/'worker.json',
        on_worker_owned=owned,provenance=dict(source_commit=source_commit,phase=phase))
    clean_receipt(receipt,source_commit,phase);completed_phase(base,source_commit,phase)
    return receipt


def run(base,source_commit):
    require_release();base=checked_base(base)
    if not (base/'binding.json').exists():freeze(source_commit,base)
    for phase in PHASES:launch(base,source_commit,phase)
    from experiments import weak_force_study_20261029 as study
    write_new(base/'complete.json',dict(status='COMPLETED',source_commit=source_commit,optimizer_calls=study.counts(base),
        model_adopted=False,scope='finite preregistered execution; scientific decision remains root-owned'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze',*PHASES,'run'));parser.add_argument('--source-commit',required=True,type=commit_id)
    parser.add_argument('--base',type=Path,default=BASE);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv);base=checked_base(args.base)
    if args.worker:
        if args.phase not in PHASES:parser.error('only phase workers are registered')
        worker(base,args.source_commit,args.phase)
    elif args.phase=='freeze':freeze(args.source_commit,base)
    elif args.phase=='run':run(base,args.source_commit)
    else:launch(base,args.source_commit,args.phase)


if __name__=='__main__':main()
