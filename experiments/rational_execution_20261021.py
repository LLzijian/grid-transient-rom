"""Prospective finite science execution without process quotas or cost admission."""
from __future__ import annotations
import argparse,copy,ctypes,json,math,os,shutil,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from finite_grammar_discovery_20261021 import sha,bound,receipt
import rational_control_20261021 as study
SAVE_STEPS=(0,32,128,512)
_REPLACE_DELAYS=(.01,.02,.04,.08,.16)
ROUND_PATH='private_data/closure_mechanism_20261019/P1/rational_execution01'
# Exact archived human authority and actual raw Astra6 return/root adoption.
REPAIR_INPUTS={
    'additional_repair_authority':('results/rational_execution_20261021/additional_repair_authority_20261022.json','191281af9ee91d3d125d7bdbfb1c7bb00c4c74e3bcc1929a5cd45984c1ea70b0'),
    'repair_amendment':('reports/rational_execution_additional_repair_20261022_protocol.md','7062b1355bfa0e595423443fbf39821ad075765f05e65d4b1d91fee172e56dd3'),
    'repair_astra_plan':('reports/astra6_authorized_executor_repair_20261022.md','902757b852038cf8615ef5de7cb0fa0e84d12084151ff35e8b6dc0db7bb8e2a6'),
    'recorder_amendment':('reports/rational_execution_recorder_amendment_20261022.md','dffb6b1316942c10aec252c976591897a72b84054e4096e9a28fcf71f2bc79b2')}
HISTORICAL_QA={
    1:('0a395e3f593a82f8462884e3079b5cb0d54bc70e','UNRESOLVED','bf71624142ee68ca9c7e99d2cd50cf827c2e618cd0754bc0238d19d4bc0ccdfc'),
    2:('b00db35e23adcd035ed08fb3f3d365fceda09042','FAILED','7a465f7b5bc87d7b9f1cf3b7e06333c4fa186bf686236ca6c888cede10d850b2')}
REPAIR_SOURCE_FILES=('AGENTS.md','docs/engineering_loop.md','reports/rational_execution_20261021_protocol.md',
    'reports/astra6_rational_execution_20261021.md','reports/rational_execution_20261021_repair.md',
    'results/rational_execution_20261021/authority.json','experiments/rational_execution_20261021.py',
    'experiments/uncapped_windows_job_20261021.py','tests/test_rational_execution_20261021.py',
    'tests/test_uncapped_windows_job_20261021.py',ROUND_PATH+'/launch.py',ROUND_PATH+'/make_binding.py')


def repair_source_names():
    return REPAIR_SOURCE_FILES+tuple(path for path,_ in REPAIR_INPUTS.values())


# Separate prospective authority. Historical QA03 admission remains available only
# to its retained tests; current science always uses validate_qa06_pass below.
RECONSTRUCTION_INPUTS={
    'reconstruction_authority':('results/rational_execution_20261021/reconstruction_authority_20261022.json','4327713dead80850fc1f973c3959113dbc09987de772c79ae1e4a9b2a143ed4c'),
    'reconstruction_protocol':('reports/executor_reconstruction_20261022_protocol.md','5d1aa6a177c991649f6fea02f574c075332f5d5a1840267c8d01668735f85f1a'),
    'reconstruction_astra_plan':('reports/astra6_executor_reconstruction_20261022.md','08713b8e0b0b5e07fd6376a3d01d5ef038e0376d9c0cd94337d18a1e0af38581'),
    'qa03_failed_summary':('results/rational_execution_20261021/qa03_summary.json','8783eca36518ecd722e737216db897ae4137d463a704c2a5edb69b940ef3e4d4')}
QA03_HISTORY=('95ee7cfe3d0c9dbdc065a306fcdc1d60fd0078d3','FAILED','d2893e3d14924058c3a5e695a3b103329b7fb7d59a5b2590cfdc57883eb19baa')
RUNTIME_SELECTION=ROUND_PATH+'/spawn_chain20261022/runtime_selection.json'
DESCENDANT_BATCH=ROUND_PATH+'/spawn_chain20261022/batch.json'
QA06_INPUTS={
    'spawn_chain_authority':('results/rational_execution_20261021/spawn_chain_authority_20261022.json','fcef6abc075be703711ff5cc0b88967b34a7c65ed93f910488f0006ec5c3db68'),
    'spawn_chain_astra':('reports/astra6_spawn_chain_20261022.md','cbcbed1467214632827cf46e1925306886ed54c698fa34105d8c621fad6752fd'),
    'spawn_chain_protocol':('reports/spawn_chain_repair_20261022_protocol.md','67f48c19b7c8c0cdbaf1d8b5a53e5ea2b18ab510a4383ed6c9036e14d611ad56'),
    'qa05_failed_summary':('results/rational_execution_20261021/qa05_summary.json','9f00062094935b0323ba115f741c49c9165cb10a0240d6d837c1a3e540c52ea1')}
QA05_HISTORY=('5b7722c0cf66b3eb5313d9f4efd6d2f007a551e4','FAILED','34283382ab6da9e5901062986bdfacd3301a1379cf3d423ef576e0fb3f4dd548')


def qa06_source_names():
    return tuple(dict.fromkeys(qa05_source_names()+tuple(p for p,_ in QA06_INPUTS.values())+(
        ROUND_PATH+'/spawn_chain20261022/observe.py',ROUND_PATH+'/reconstruction20261022/probe_worker.py')))


def validate_qa06_authority(binding,commit):
    for key,(name,digest) in QA06_INPUTS.items():
        if (binding.get(key)!={'path':name,'sha256':digest} or binding.get('files',{}).get(name)!=digest
                or not (ROOT/name).is_file() or sha(ROOT/name)!=digest):
            raise ValueError('missing/changed exact QA06 authority: '+key)
    validate_qa05_authority(binding,commit)
    authority=bound_json(binding,QA06_INPUTS['spawn_chain_authority'][0])
    expected={'human_answer':'允许修复、验收并继续训练','fixed_native_cases':5,
        'additional_full_acceptance_executions':1,'expected_qa_attempt':6,'original_scientific_models':12,
        'updates_per_model':512,'no_process_quotas':True,'original_val_test_closed':True,'TASK4_paused':True,
        'scientific_gates_unchanged':True,'historical_QA01_QA02_QA03_QA04_QA05_preserved':True,'requested_model':'gpt-6-astra'}
    if any(k not in authority or type(authority[k]) is not type(v) or authority[k]!=v for k,v in expected.items()):
        raise ValueError('exact typed QA06 permission required')
    for name in qa06_source_names():
        if not (ROOT/name).is_file() or binding['files'].get(name)!=sha(ROOT/name):
            raise ValueError('changed current QA06 source: '+name)
    validate_descendant_batch(binding,commit)


def validate_qa06_history(binding):
    validate_qa05_history(binding)
    name=ROUND_PATH+'/native_qa05/receipt.json';row=bound_json(binding,name)
    commit,status,digest=QA05_HISTORY
    if (sha(ROOT/name)!=digest or binding.get('prior_receipts',[]).count(name)!=1
            or row.get('source_commit')!=commit or row.get('status')!=status or row.get('qa_attempt')!=5
            or row.get('phase')!='qa' or row.get('cleanup_verified') is not True
            or type(row.get('job_active_processes')) is not int or row['job_active_processes']!=0
            or not row.get('source_before') or row['source_before']!=row.get('source_after')):
        raise ValueError('actual immutable QA05 failed history required')


def require_unused_qa06_temp():
    path=ROOT/'private_data/q6'
    if path.exists() or path.is_symlink():raise ValueError('QA06 short temporary root must not exist; preserve existing evidence')
    return path


def validate_descendant_batch(binding,commit):
    selection=bound_json(binding,RUNTIME_SELECTION)
    selected=validate_runtime_binding(binding)
    files={(ROOT/name).resolve():digest for name,digest in binding.get('files',{}).items()}
    selected_files={(ROOT/name).resolve():digest for name,digest in selection.get('files',{}).items()}
    def read_ref(ref):
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('exact descendant raw reference required')
        path=(ROOT/ref['path']).resolve();digest=ref['sha256']
        if (not path.is_file() or files.get(path)!=digest or selected_files.get(path)!=digest or sha(path)!=digest):
            raise ValueError('source-bound descendant raw reference required')
        return json.loads(path.read_text(encoding='utf-8'))
    reference=binding.get('descendant_batch',{})
    if reference.get('path')!=DESCENDANT_BATCH:raise ValueError('exact descendant batch required')
    batch=read_ref(reference)
    authority_path,authority_sha=QA06_INPUTS['spawn_chain_authority']
    if (batch.get('status')!='QUALIFIED' or batch.get('source_commit')!=commit
            or batch.get('authority')!={'path':authority_path,'sha256':authority_sha}):
        raise ValueError('descendant batch authority/source differs')
    read_ref(batch['authority'])
    cases=batch.get('cases',[])
    if not isinstance(cases,list) or [row.get('name') for row in cases]!=['normal','grandchild','pipes','orphan','supervisor_exit']:
        raise ValueError('exact ordered five descendant cases required')
    if (selection.get('observation_source_commit')!=commit
            or selection.get('normal_receipt')!=cases[0].get('receipt')
            or selection.get('orphan_receipt')!=cases[3].get('receipt')):
        raise ValueError('selected normal/orphan differs from registered batch')
    low_command={key:value for key,value in selected.items() if key!='runtime_selection'}
    python_sources={path:digest for path,digest in selected_files.items() if path.suffix.lower()=='.py'}
    if not python_sources:raise ValueError('descendant source files required')
    for row in cases:
        name=row['name'];proof=read_ref(row.get('receipt'))
        exit_code=121 if name=='supervisor_exit' else 0
        if (proof.get('source_commit')!=commit or proof.get('phase')!='runtime_observation' or proof.get('group')!=name
                or proof.get('status')!=('FAILED' if name in ('orphan','supervisor_exit') else 'COMPLETED')
                or proof.get('cleanup_verified') is not True or type(proof.get('job_active_processes')) is not int
                or proof['job_active_processes']!=0
                or any(type(proof.get(k)) is not int or proof[k]!=exit_code for k in ('exit_code','launcher_exit_code','worker_exit_code'))
                or type(proof.get('orphans_cleaned')) is not bool
                or proof['orphans_cleaned'] is not (name=='orphan')
                or not proof.get('source_before') or proof['source_before']!=proof.get('source_after')
                or proof.get('command')!=dict(low_command,arguments=proof.get('command',{}).get('arguments'))):
            raise ValueError('actual descendant native case differs: '+name)
        if any(proof['source_before'].get(str(path))!=digest for path,digest in python_sources.items()):
            raise ValueError('descendant observation source bytes differ: '+name)
        if name=='pipes':
            pipe=read_ref(row.get('pipe_result'))
            if (pipe.get('stdout')!='fixed stdout\n' or pipe.get('stderr')!='fixed stderr\n'
                    or type(pipe.get('returncode')) is not int or pipe['returncode']!=0):
                raise ValueError('actual descendant pipe contents differ')
        identities=[read_ref(ref) for ref in row.get('identities',[])]
        minimum=3 if name=='grandchild' else 2
        if len(identities)<minimum:raise ValueError('descendant identity chain incomplete: '+name)
        identity_keys=[]
        for identity in identities:
            if (type(identity.get('pid')) is not int or identity['pid']<=0
                    or type(identity.get('creation_filetime_utc')) is not int or identity['creation_filetime_utc']<=0
                    or Path(identity.get('logical_executable',identity.get('executable',''))).resolve()!=Path(selected['logical_executable']).resolve()
                    or ('executable' in identity and Path(identity['executable']).resolve()!=Path(selected['logical_executable']).resolve())):
                raise ValueError('actual descendant identity differs: '+name)
            prior_worker_gone(identity);identity_keys.append((identity['pid'],identity['creation_filetime_utc']))
        if len(set(identity_keys))!=len(identity_keys):raise ValueError('duplicate descendant identity: '+name)
        parent=proof.get('worker',{})
        if (identity_keys[0]!=(parent.get('pid'),parent.get('creation_filetime_utc'))
                or Path(parent.get('physical_image','')).resolve()!=Path(selected['expected_worker_image']).resolve()
                or proof.get('actual_worker_state')!='GONE'):
            raise ValueError('descendant parent/native identity differs: '+name)
        spawns=[]
        for ref in row.get('spawns',[]):
            records=read_ref(ref)
            if not isinstance(records,list):raise ValueError('actual descendant spawn records required')
            spawns.extend(records)
        if len(spawns)<minimum-1:raise ValueError('descendant spawn chain incomplete: '+name)
        for spawn in spawns:
            expected_flags=525324 if name=='supervisor_exit' else 8
            expected_via='native_job' if name=='supervisor_exit' else 'subprocess'
            if (type(spawn.get('creationflags')) is not int or spawn['creationflags']!=expected_flags
                    or spawn.get('via')!=expected_via
                    or Path(spawn.get('executable','')).resolve()!=Path(selected['executable']).resolve()
                    or Path(spawn.get('logical_executable','')).resolve()!=Path(selected['logical_executable']).resolve()
                    or type(spawn.get('child_pid')) is not int or spawn['child_pid']<=0
                    or not isinstance(spawn.get('arguments'),list) or not all(isinstance(v,str) for v in spawn['arguments'])):
                raise ValueError('actual descendant DETACHED spawn differs: '+name)
        if not {pid for pid,_ in identity_keys[1:]}.issubset({spawn['child_pid'] for spawn in spawns}):
            raise ValueError('descendant spawn/identity coverage differs: '+name)
        if name=='orphan':
            live={(value.get('pid'),value.get('creation_filetime_utc')) for value in proof.get('active_members_observed',[])
                  if value.get('in_owned_job') is True and value.get('wait_result')==258}
            if not set(identity_keys[1:]).issubset(live):raise ValueError('actual handshaken orphan observation required')
        ends=[read_ref(ref) for ref in row.get('handle_ends',[])]
        if name=='supervisor_exit':
            if not ends:raise ValueError('retained descendant handle termination required')
            ended=set()
            for end in ends:
                if (type(end.get('pid')) is not int or type(end.get('creation_filetime_utc')) is not int
                        or type(end.get('wait_result')) is not int or end['wait_result']!=0
                        or type(end.get('exit_code')) is not int):raise ValueError('actual descendant handle end differs')
                ended.add((end['pid'],end['creation_filetime_utc']))
            if not set(identity_keys[1:]).issubset(ended):raise ValueError('retained handle/child identity differs')
    return batch
QA05_INPUTS={
    'qa05_authority':('results/rational_execution_20261021/qa05_authority_20261022.json','a493f77a01de06d7b3bff49930900ffabb776ec15ddf59a856828c51ce3e0e79'),
    'qa05_astra_raw':('reports/astra6_qa05_20261022_raw.txt','1b7f849ba85b2695b9aa50967ffe6e1510122a27bc7140a3ef9ded1e1236667f'),
    'qa05_astra_decision':('reports/astra6_qa05_20261022.md','50062eb20cac73cb0f7223888adaafabf4cb25348ed48945c7eb177ac8b4409c'),
    'qa05_protocol':('reports/qa05_recheck_20261022_protocol.md','4eeb6514ba048454bfc8614f2070fd5e9f61c154f1e839b46718471c0cad5742'),
    'qa04_failed_summary':('results/rational_execution_20261021/qa04_summary.json','c218e27718a8524e840286696419d6d694f0e1b8b4d570d4893db6c42bd1fdbc')}
QA04_HISTORY=('6eb63542ee4da28be4227d653d32f3d48bd8367f','FAILED','d49f6db12049ecef2160460037237fffe46b96178ef13435e6fc1e8c6138aa12')


def qa05_source_names():
    return tuple(dict.fromkeys(reconstruction_source_names()+tuple(p for p,_ in QA05_INPUTS.values())))


def validate_qa05_authority(binding,commit):
    for key,(name,digest) in QA05_INPUTS.items():
        if (binding.get(key)!={'path':name,'sha256':digest} or binding.get('files',{}).get(name)!=digest
                or not (ROOT/name).is_file() or sha(ROOT/name)!=digest):
            raise ValueError('missing/changed exact QA05 authority: '+key)
    validate_reconstruction_authority(binding,commit)
    authority=bound_json(binding,QA05_INPUTS['qa05_authority'][0])
    expected={'human_answer':'允许复验，通过后继续训练','additional_full_acceptance_executions':1,
        'expected_qa_attempt':5,'original_scientific_models':12,'updates_per_model':512,
        'no_process_quotas':True,'original_val_test_closed':True,'TASK4_paused':True,
        'scientific_gates_unchanged':True,'historical_QA01_QA02_QA03_QA04_preserved':True,'requested_model':'gpt-6-astra'}
    if any(k not in authority or type(authority[k]) is not type(v) or authority[k]!=v for k,v in expected.items()):
        raise ValueError('exact typed QA05 permission required')
    for name in qa05_source_names():
        if not (ROOT/name).is_file() or binding['files'].get(name)!=sha(ROOT/name):
            raise ValueError('changed current QA05 source: '+name)


def validate_qa05_history(binding):
    validate_qa04_history(binding)
    name=ROUND_PATH+'/native_qa04/receipt.json';row=bound_json(binding,name)
    commit,status,digest=QA04_HISTORY
    if (sha(ROOT/name)!=digest or binding.get('prior_receipts',[]).count(name)!=1
            or row.get('source_commit')!=commit or row.get('status')!=status or row.get('qa_attempt')!=4
            or row.get('phase')!='qa' or row.get('cleanup_verified') is not True
            or type(row.get('job_active_processes')) is not int or row['job_active_processes']!=0
            or row.get('source_before')!=row.get('source_after')):
        raise ValueError('actual immutable QA04 failed history required')


def require_unused_qa05_temp():
    path=ROOT/'private_data/q5'
    if path.exists() or path.is_symlink():raise ValueError('QA05 short temporary root must not exist; preserve existing evidence')
    return path


def reconstruction_source_names():
    return tuple(dict.fromkeys(repair_source_names()+tuple(p for p,_ in RECONSTRUCTION_INPUTS.values())+(
        'experiments/rational_python_runtime_20261022.py','tests/test_rational_python_runtime_20261022.py',
        'tests/test_rational_control_20261021.py','reports/executor_reconstruction_20261022_independent_review.md')))


def bound_json(binding,name):
    path=(ROOT/name).resolve()
    if not path.is_file() or binding.get('files',{}).get(name)!=sha(path):
        raise ValueError('source-bound reconstruction evidence required: '+name)
    return json.loads(path.read_text(encoding='utf-8'))


def validate_runtime_binding(binding):
    reference=binding.get('runtime_selection',{})
    if reference.get('path')!=RUNTIME_SELECTION:
        raise ValueError('exact root runtime selection required')
    selection=bound_json(binding,RUNTIME_SELECTION)
    if reference.get('sha256')!=binding['files'][RUNTIME_SELECTION]:
        raise ValueError('runtime selection binding differs')
    from rational_python_runtime_20261022 import selected_python_command
    # The helper validates raw observation proofs, never an asserted PASS flag.
    command=selected_python_command([],ROOT/RUNTIME_SELECTION)
    bound_files={(ROOT/name).resolve():digest for name,digest in binding['files'].items()}
    for name,digest in selection.get('files',{}).items():
        path=(ROOT/name).resolve()
        if not path.is_file() or sha(path)!=digest or bound_files.get(path)!=digest:
            raise ValueError('runtime dependency/probe proof differs: '+name)
    if not selection.get('files'):raise ValueError('actual runtime/probe files required')
    evidence={}
    for key in ('normal_receipt','orphan_receipt','baseline_reference','parity_evidence'):
        ref=selection.get(key,{})
        path=(ROOT/ref.get('path','')).resolve();digest=ref.get('sha256')
        if (not path.is_file() or sha(path)!=digest or bound_files.get(path)!=digest
                or not any((ROOT/name).resolve()==path and value==digest for name,value in selection['files'].items())):
            raise ValueError('actual source-bound runtime observation required: '+key)
        evidence[key]=json.loads(path.read_text(encoding='utf-8'))
    for key in ('normal_receipt','orphan_receipt'):
        proof=evidence[key];negative=key=='orphan_receipt';worker=proof.get('worker',{})
        if (proof.get('source_commit')!=selection.get('observation_source_commit')
                or proof.get('status')!=('FAILED' if negative else 'COMPLETED')
                or proof.get('cleanup_verified') is not True or proof.get('orphans_cleaned') is not negative
                or type(proof.get('job_active_processes')) is not int or proof['job_active_processes']!=0
                or proof.get('actual_worker_state')!='GONE' or proof.get('exit_code')!=0
                or proof.get('launcher_exit_code')!=0 or proof.get('worker_exit_code')!=0
                or not proof.get('source_before') or proof['source_before']!=proof.get('source_after')
                or type(worker.get('pid')) is not int or worker['pid']<=0
                or type(worker.get('creation_filetime_utc')) is not int or worker['creation_filetime_utc']<=0
                or Path(worker.get('physical_image',worker.get('executable',''))).resolve()!=Path(command['expected_worker_image']).resolve()):
            raise ValueError('actual normal/orphan native observation differs: '+key)
    return command


def validate_reconstruction_authority(binding,commit):
    if (len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)
            or binding.get('source_commit')!=commit):raise ValueError('full same-source reconstruction identity required')
    for key,(name,digest) in RECONSTRUCTION_INPUTS.items():
        if binding.get(key)!={'path':name,'sha256':digest} or binding.get('files',{}).get(name)!=digest:
            raise ValueError('missing/changed exact reconstruction authority: '+key)
        if sha(ROOT/name)!=digest:raise ValueError('missing/changed exact reconstruction authority: '+key)
    authority=bound_json(binding,RECONSTRUCTION_INPUTS['reconstruction_authority'][0])
    expected={'human_answer':'允许重构并继续训练','maximum_preregistered_startup_schemes':2,
        'final_full_acceptance_executions':1,'original_scientific_models':12,'updates_per_model':512,
        'no_process_quotas':True,'original_val_test_closed':True,'TASK4_paused':True,
        'scientific_gates_unchanged':True,'historical_failed_QA01_QA02_QA03_preserved':True,'requested_model':'gpt-6-astra'}
    if any(k not in authority or type(authority[k]) is not type(v) or authority[k]!=v for k,v in expected.items()):
        raise ValueError('exact typed reconstruction permission required')
    for name in reconstruction_source_names():
        if not (ROOT/name).is_file() or binding['files'].get(name)!=sha(ROOT/name):
            raise ValueError('changed current reconstruction source: '+name)
    validate_runtime_binding(binding)


def validate_qa04_history(binding):
    validate_qa03_history(binding)
    name=ROUND_PATH+'/native_qa03/receipt.json';row=bound_json(binding,name)
    commit,status,digest=QA03_HISTORY
    if (sha(ROOT/name)!=digest or binding.get('prior_receipts',[]).count(name)!=1
            or row.get('source_commit')!=commit or row.get('status')!=status or row.get('qa_attempt')!=3
            or row.get('phase')!='qa' or row.get('cleanup_verified') is not True
            or row.get('job_active_processes')!=0 or row.get('source_before')!=row.get('source_after')):
        raise ValueError('actual immutable QA03 failed history required')


def validate_qa04_pass(binding,commit):
    validate_reconstruction_authority(binding,commit);validate_qa04_history(binding)
    native=bound_json(binding,ROUND_PATH+'/native_qa04/receipt.json')
    marker=bound_json(binding,ROUND_PATH+'/qa04/complete.json')
    command=validate_runtime_binding(binding)
    if (native.get('status')!='COMPLETED' or native.get('phase')!='qa' or native.get('qa_attempt')!=4
            or native.get('source_commit')!=commit or native.get('cleanup_verified') is not True
            or native.get('orphans_cleaned') is not False or native.get('job_active_processes')!=0
            or native.get('actual_worker_state')!='GONE' or native.get('source_before')!=native.get('source_after')
            or any(type(native.get(key)) is not int or native[key]!=0 for key in ('exit_code','launcher_exit_code','worker_exit_code'))
            or native.get('command')!=dict(command,arguments=native.get('command',{}).get('arguments'))):
        raise ValueError('actual same-source native QA04 pass required')
    worker=native.get('worker',{});actual=native.get('worker_receipt',{})
    if (type(worker.get('pid')) is not int or worker['pid']<=0
            or type(worker.get('creation_filetime_utc')) is not int or worker['creation_filetime_utc']<=0
            or any(worker.get(key)!=actual.get(key) for key in ('pid','creation_filetime_utc','physical_image','logical_executable','runtime_scheme','runtime_selection'))
            or Path(worker.get('physical_image','')).resolve()!=Path(command['expected_worker_image']).resolve()
            or Path(worker.get('logical_executable','')).resolve()!=Path(command['logical_executable']).resolve()
            or worker.get('runtime_scheme')!=command['runtime_scheme']
            or worker.get('runtime_selection')!={'path':RUNTIME_SELECTION,'sha256':binding['files'][RUNTIME_SELECTION]}
            or any(k not in native or native[k] is not None for k in ('runtime_limits','cpu_limits','active_process_limit'))):
        raise ValueError('actual QA04 selected worker identity differs')
    runtime_files={(ROOT/name).resolve() for name in bound_json(binding,RUNTIME_SELECTION)['files']}
    for name,digest in binding['files'].items():
        if name in (ROUND_PATH+'/native_qa04/receipt.json',ROUND_PATH+'/qa04/complete.json'):continue
        if name in reconstruction_source_names() or name==RUNTIME_SELECTION or (ROOT/name).resolve() in runtime_files:
            if native.get('source_before',{}).get(str((ROOT/name).resolve()))!=digest:
                raise ValueError('QA04 source/authority/runtime differs')
    result=marker.get('result',{})
    if (marker.get('status')!='COMPLETED' or result.get('status')!='PASS' or result.get('source_commit')!=commit
            or result.get('qa_attempt')!=4 or result.get('pytest_returncode')!=0
            or result.get('runtime_selection_sha256')!=binding['files'][RUNTIME_SELECTION]):
        raise ValueError('actual same-source inner QA04 pass required')


def validate_qa05_pass(binding,commit):
    validate_qa05_authority(binding,commit);validate_qa05_history(binding)
    native=bound_json(binding,ROUND_PATH+'/native_qa05/receipt.json')
    marker=bound_json(binding,ROUND_PATH+'/qa05/complete.json')
    command=validate_runtime_binding(binding)
    if (native.get('status')!='COMPLETED' or native.get('phase')!='qa' or native.get('qa_attempt')!=5
            or native.get('source_commit')!=commit or native.get('cleanup_verified') is not True
            or native.get('orphans_cleaned') is not False or native.get('job_active_processes')!=0
            or native.get('actual_worker_state')!='GONE' or native.get('source_before')!=native.get('source_after')
            or any(type(native.get(key)) is not int or native[key]!=0 for key in ('exit_code','launcher_exit_code','worker_exit_code'))
            or native.get('command')!=dict(command,arguments=native.get('command',{}).get('arguments'))):
        raise ValueError('actual same-source native QA05 pass required')
    worker=native.get('worker',{});actual=native.get('worker_receipt',{})
    if (type(worker.get('pid')) is not int or worker['pid']<=0
            or type(worker.get('creation_filetime_utc')) is not int or worker['creation_filetime_utc']<=0
            or any(worker.get(key)!=actual.get(key) for key in ('pid','creation_filetime_utc','physical_image','logical_executable','runtime_scheme','runtime_selection'))
            or Path(worker.get('physical_image','')).resolve()!=Path(command['expected_worker_image']).resolve()
            or Path(worker.get('logical_executable','')).resolve()!=Path(command['logical_executable']).resolve()
            or worker.get('runtime_scheme')!=command['runtime_scheme']
            or worker.get('runtime_selection')!={'path':RUNTIME_SELECTION,'sha256':binding['files'][RUNTIME_SELECTION]}
            or any(k not in native or native[k] is not None for k in ('runtime_limits','cpu_limits','active_process_limit'))):
        raise ValueError('actual QA05 selected worker identity differs')
    runtime_files={(ROOT/name).resolve() for name in bound_json(binding,RUNTIME_SELECTION)['files']}
    for name,digest in binding['files'].items():
        if name in (ROUND_PATH+'/native_qa05/receipt.json',ROUND_PATH+'/qa05/complete.json'):continue
        if name in qa05_source_names() or name==RUNTIME_SELECTION or (ROOT/name).resolve() in runtime_files:
            if native.get('source_before',{}).get(str((ROOT/name).resolve()))!=digest:
                raise ValueError('QA05 source/authority/runtime differs')
    result=marker.get('result',{})
    if (marker.get('status')!='COMPLETED' or result.get('status')!='PASS' or result.get('source_commit')!=commit
            or result.get('qa_attempt')!=5 or result.get('pytest_returncode')!=0
            or result.get('runtime_selection_sha256')!=binding['files'][RUNTIME_SELECTION]):
        raise ValueError('actual same-source inner QA05 pass required')


def validate_qa06_pass(binding,commit):
    validate_qa06_authority(binding,commit);validate_qa06_history(binding)
    native=bound_json(binding,ROUND_PATH+'/native_qa06/receipt.json')
    marker=bound_json(binding,ROUND_PATH+'/qa06/complete.json')
    command=validate_runtime_binding(binding)
    if (native.get('status')!='COMPLETED' or native.get('phase')!='qa' or native.get('qa_attempt')!=6
            or native.get('source_commit')!=commit or native.get('cleanup_verified') is not True
            or native.get('orphans_cleaned') is not False or native.get('job_active_processes')!=0
            or native.get('actual_worker_state')!='GONE' or native.get('source_before')!=native.get('source_after')
            or any(type(native.get(key)) is not int or native[key]!=0 for key in ('exit_code','launcher_exit_code','worker_exit_code'))
            or native.get('command')!=dict(command,arguments=native.get('command',{}).get('arguments'))):
        raise ValueError('actual same-source native QA06 pass required')
    worker=native.get('worker',{});actual=native.get('worker_receipt',{})
    if (type(worker.get('pid')) is not int or worker['pid']<=0
            or type(worker.get('creation_filetime_utc')) is not int or worker['creation_filetime_utc']<=0
            or any(worker.get(key)!=actual.get(key) for key in ('pid','creation_filetime_utc','physical_image','logical_executable','runtime_scheme','runtime_selection'))
            or Path(worker.get('physical_image','')).resolve()!=Path(command['expected_worker_image']).resolve()
            or Path(worker.get('logical_executable','')).resolve()!=Path(command['logical_executable']).resolve()
            or worker.get('runtime_scheme')!=command['runtime_scheme']
            or worker.get('runtime_selection')!={'path':RUNTIME_SELECTION,'sha256':binding['files'][RUNTIME_SELECTION]}
            or any(k not in native or native[k] is not None for k in ('runtime_limits','cpu_limits','active_process_limit'))):
        raise ValueError('actual QA06 selected worker identity differs')
    runtime_files={(ROOT/name).resolve() for name in bound_json(binding,RUNTIME_SELECTION)['files']}
    for name,digest in binding['files'].items():
        if name in (ROUND_PATH+'/native_qa06/receipt.json',ROUND_PATH+'/qa06/complete.json'):continue
        if name in qa06_source_names() or name==RUNTIME_SELECTION or (ROOT/name).resolve() in runtime_files:
            if native.get('source_before',{}).get(str((ROOT/name).resolve()))!=digest:
                raise ValueError('QA06 source/authority/runtime differs')
    result=marker.get('result',{})
    if (marker.get('status')!='COMPLETED' or result.get('status')!='PASS' or result.get('source_commit')!=commit
            or result.get('qa_attempt')!=6 or result.get('pytest_returncode')!=0
            or result.get('runtime_selection_sha256')!=binding['files'][RUNTIME_SELECTION]
            or type(result.get('pytest_creation_flags')) is not int or result['pytest_creation_flags']!=8):
        raise ValueError('actual same-source inner QA06 pass required')


def validate_repair_authority(binding,commit):
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit) or binding.get('source_commit')!=commit:
        raise ValueError('full same-source repair identity required')
    files=binding.get('files',{})
    for key,(name,digest) in REPAIR_INPUTS.items():
        if binding.get(key)!={'path':name,'sha256':digest} or files.get(name)!=digest or not (ROOT/name).is_file() or sha(ROOT/name)!=digest:
            raise ValueError('missing/changed exact repair authority: '+key)
    authority=json.loads((ROOT/REPAIR_INPUTS['additional_repair_authority'][0]).read_text(encoding='utf-8'))
    expected={'human_answer':'同意修复并继续训练','additional_implementation_repairs':1,'additional_qa_executions':1,
              'expected_qa_attempt':3,'scientific_models':12,'updates_per_model':512,'original_val_test_closed':True,
              'TASK4_paused':True,'prior_qa_failures_preserved':True,'no_mathematical_or_selection_changes':True,
              'requested_model':'gpt-6-astra','process_count_cap':None,'process_runtime_cap_seconds':None,
              'stage_or_parent_process_cap_seconds':None}
    if any(k not in authority or type(authority[k]) is not type(v) or authority[k]!=v for k,v in expected.items()):
        raise ValueError('exact typed additional repair permission required')
    for name in repair_source_names():
        if not (ROOT/name).is_file() or files.get(name)!=sha(ROOT/name):raise ValueError('changed current repair source: '+name)


def historical_qa_receipt(path,row):
    for attempt,(commit,status,digest) in HISTORICAL_QA.items():
        if Path(path).resolve()==(ROOT/ROUND_PATH/f'native_qa{attempt:02d}/receipt.json').resolve():
            return (sha(path)==digest and row.get('source_commit')==commit and row.get('status')==status
                    and row.get('phase')=='qa' and row.get('qa_attempt')==(None if attempt==1 else attempt)
                    and row.get('cleanup_verified') is True and row.get('job_active_processes')==0
                    and row.get('source_before')==row.get('source_after'))
    return False


def validate_qa03_history(binding):
    base=ROOT/ROUND_PATH;files=binding.get('files',{})
    for attempt in HISTORICAL_QA:
        path=base/f'native_qa{attempt:02d}/receipt.json';name=path.relative_to(ROOT).as_posix()
        if not path.is_file():raise ValueError('actual immutable QA01/QA02 failure history required')
        row=json.loads(path.read_text(encoding='utf-8'))
        if (not historical_qa_receipt(path,row) or files.get(name)!=sha(path)
                or binding.get('prior_receipts',[]).count(name)!=1):
            raise ValueError('actual immutable QA01/QA02 failure history required')


def validate_qa03_pass(binding,commit):
    validate_repair_authority(binding,commit);validate_qa03_history(binding)
    base=ROOT/ROUND_PATH
    def bound_result(relative):
        name=(base/relative).relative_to(ROOT).as_posix();path=ROOT/name
        if not path.is_file() or binding.get('files',{}).get(name)!=sha(path):raise ValueError('source-bound QA03 evidence required')
        return json.loads(path.read_text(encoding='utf-8'))
    native=bound_result('native_qa03/receipt.json');marker=bound_result('qa03/complete.json')
    if (native.get('status')!='COMPLETED' or native.get('phase')!='qa' or native.get('qa_attempt')!=3
            or native.get('source_commit')!=commit or native.get('cleanup_verified') is not True
            or native.get('job_active_processes')!=0 or native.get('source_before')!=native.get('source_after')):
        raise ValueError('actual same-source native QA03 pass required')
    for name in repair_source_names():
        if native.get('source_before',{}).get(str((ROOT/name).resolve()))!=binding['files'][name]:
            raise ValueError('QA03 source/authority/amendment differs')
    result=marker.get('result',{})
    if marker.get('status')!='COMPLETED' or result.get('status')!='PASS' or result.get('source_commit')!=commit or result.get('qa_attempt')!=3:
        raise ValueError('actual same-source inner QA03 pass required')


def validate_qa_attempt(binding,commit,attempt):
    if attempt not in (1,2,3,4,5,6):raise ValueError('only registered QA attempts1 or2 or authorized3 or reconstruction4 or recheck5')
    if attempt==6:
        if binding.get('phase')!='qa' or binding.get('qa_attempt')!=6 or binding.get('group') is not None:
            raise ValueError('QA06 binding role differs')
        validate_qa06_authority(binding,commit);validate_qa06_history(binding)
        expected={(ROOT/ROUND_PATH/f'native_qa{i:02d}/receipt.json').resolve() for i in (1,2,3,4,5)}
        actual={p.resolve() for p in (ROOT/ROUND_PATH).glob('native_*/receipt.json')}
        if actual!=expected:raise ValueError('QA06 only after exactly five actual failed attempts')
        require_unused_qa06_temp()
    if attempt==5:
        if binding.get('phase')!='qa' or binding.get('qa_attempt')!=5 or binding.get('group') is not None:
            raise ValueError('QA05 binding role differs')
        validate_qa05_authority(binding,commit);validate_qa05_history(binding)
        expected={(ROOT/ROUND_PATH/f'native_qa{i:02d}/receipt.json').resolve() for i in (1,2,3,4)}
        actual={p.resolve() for p in (ROOT/ROUND_PATH).glob('native_*/receipt.json')}
        if actual!=expected:raise ValueError('QA05 only after exactly four actual failed attempts')
        require_unused_qa05_temp()
    if attempt==4:
        if binding.get('phase')!='qa' or binding.get('qa_attempt')!=4 or binding.get('group') is not None:
            raise ValueError('QA04 binding role differs')
        validate_reconstruction_authority(binding,commit);validate_qa04_history(binding)
        expected={(ROOT/ROUND_PATH/f'native_qa{i:02d}/receipt.json').resolve() for i in (1,2,3)}
        actual={p.resolve() for p in (ROOT/ROUND_PATH).glob('native_*/receipt.json')}
        if actual!=expected:raise ValueError('QA04 only after exactly three actual failed attempts')
    if attempt==3:
        validate_repair_authority(binding,commit);validate_qa03_history(binding)
        if binding.get('phase')!='qa' or binding.get('qa_attempt')!=3 or binding.get('group') is not None:
            raise ValueError('QA03 binding role differs')
        expected={str((ROOT/ROUND_PATH/f'native_qa{i:02d}/receipt.json').resolve()) for i in (1,2)}
        actual={str(p.resolve()) for p in (ROOT/ROUND_PATH).glob('native_*/receipt.json')}
        # The running outer QA03 has its unique native directory but no final receipt yet.
        if actual!=expected:raise ValueError('QA03 only after exactly two actual failed attempts')

def atomic(path,value):
    """Bounded Windows sharing-conflict repair for entry metadata only."""
    path=Path(path);encoded=json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode('utf-8')
    if len(encoded)>262144:raise ValueError('registered JSON serialization envelope exceeded')
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as file:file.write(encoded)
    for attempt in range(len(_REPLACE_DELAYS)+1):
        try:os.replace(temporary,path);return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32,33) or attempt==len(_REPLACE_DELAYS):raise
            time.sleep(_REPLACE_DELAYS[attempt])
    # On permanent failure the uniquely named temp file is retained as evidence.

def publish_worker_receipt(out,phase,commit,binding,group,qa_attempt):
    # Publish source/role and actual creation identity together, avoiding a
    # supervisor observing an incomplete identity during a later metadata edit.
    pending=out/('worker_identity_'+uuid.uuid4().hex);pending.mkdir()
    receipt(pending,phase,commit,binding)
    value=json.loads((pending/'worker.json').read_text(encoding='utf-8'))
    value.update(group=group,qa_attempt=qa_attempt)
    if qa_attempt in (4,5,6) or phase!='qa':
        from rational_python_runtime_20261022 import selected_python_command
        command=selected_python_command([],ROOT/RUNTIME_SELECTION)
        k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=ctypes.c_void_p
        k.QueryFullProcessImageNameW.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_uint32)]
        buffer=ctypes.create_unicode_buffer(32768);length=ctypes.c_uint32(len(buffer))
        if not k.QueryFullProcessImageNameW(k.GetCurrentProcess(),0,buffer,ctypes.byref(length)):raise ctypes.WinError(ctypes.get_last_error())
        if Path(buffer.value).resolve()!=Path(command['expected_worker_image']).resolve():raise ValueError('actual worker physical image differs')
        if Path(sys.executable).resolve()!=Path(command['logical_executable']).resolve():raise ValueError('actual worker logical executable differs')
        value.update(runtime_scheme=command['runtime_scheme'],logical_executable=sys.executable,
                     physical_image=buffer.value,runtime_selection={'path':RUNTIME_SELECTION,'sha256':sha(ROOT/RUNTIME_SELECTION)})
    atomic(out/'worker.json',value)
def new_or_equal(path,value):
    if Path(path).exists():
        if json.loads(Path(path).read_text(encoding='utf-8'))!=value:raise ValueError('existing evidence differs; refusing overwrite')
    else:atomic(path,value)


def checkpoint_api():
    from closure_causal_history_residual_20261019 import checkpoint,restore
    return checkpoint,restore
def publish_checkpoint(folder,m,opt,rng,step,trace,gradient,summary,identity):
    checkpoint,_=checkpoint_api();checkpoint(folder,m,opt,rng,step,trace,identity)
    sidecar=folder/f'checkpoint_{step:05d}.diagnostics.json'
    atomic(sidecar,{'step':step,'identity':identity,'loss':trace,'gradient':gradient,'parameter_summary':summary})
    checkpoint_dir=folder/f'checkpoint_{step:05d}'
    atomic(folder/'resume.json',{'step':step,'checkpoint':checkpoint_dir.name,'identity':identity,
                               'files':{n:sha(checkpoint_dir/n) for n in ('model.json','training.pt')},
                               'diagnostics':study.fileid(sidecar)})
def complete_sidecar(folder,identity):
    pointer=json.loads((folder/'resume.json').read_text(encoding='utf-8'))
    latest=json.loads((folder/'latest.json').read_text(encoding='utf-8'))
    step=pointer.get('step')
    if step not in SAVE_STEPS or pointer.get('checkpoint')!=f'checkpoint_{step:05d}' or pointer.get('identity')!=identity:
        raise ValueError('invalid complete resume identity')
    if list(folder.glob('pending_*')):raise ValueError('pending checkpoint retained; no silent fallback')
    if latest.get('checkpoint')!=pointer['checkpoint'] or latest.get('updates')!=step or latest.get('sha256')!=pointer.get('files'):
        raise ValueError('latest checkpoint is ahead of complete diagnostics or differs')
    directory=folder/pointer['checkpoint']
    if set(pointer['files'])!={'model.json','training.pt'}:raise ValueError('incomplete model/state pair')
    if any(sha(directory/n)!=h for n,h in pointer['files'].items()):raise ValueError('checkpoint changed')
    diag_path=Path(pointer['diagnostics']['path']).resolve()
    if diag_path!=(folder/f'checkpoint_{step:05d}.diagnostics.json').resolve() or sha(diag_path)!=pointer['diagnostics']['sha256']:
        raise ValueError('diagnostic identity changed')
    d=json.loads(diag_path.read_text(encoding='utf-8'))
    if set(d)!={'step','identity','loss','gradient','parameter_summary'} or d['step']!=step or d['identity']!=identity or len(d['loss'])!=step:
        raise ValueError('missing or inconsistent diagnostic fields')
    if set(d['gradient'])!={'total_raw_norm','clip_factor'} or any(len(v)!=step for v in d['gradient'].values()):raise ValueError('gradient trajectory incomplete')
    if set(d['parameter_summary'])!={'first','last','max'}:raise ValueError('parameter gradient summary incomplete')
    if step and any(not d['parameter_summary'][key] for key in ('first','last','max')):raise ValueError('empty nonzero-step diagnostics')
    return step,d
def restore_complete(folder,m,opt,rng,identity):
    step,d=complete_sidecar(folder,identity);_,restore=checkpoint_api()
    actual,trace=restore(folder,m,opt,rng,identity)
    if actual!=step or trace!=d['loss']:raise ValueError('model/RNG checkpoint and diagnostics disagree')
    return actual,trace,copy.deepcopy(d['gradient']),copy.deepcopy(d['parameter_summary'])


def prior_worker_gone(worker):
    """Windows creation identity and exit state; no age-based PID assumption."""
    class FT(ctypes.Structure):_fields_=[('low',ctypes.c_uint32),('high',ctypes.c_uint32)]
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];k.OpenProcess.restype=ctypes.c_void_p
    k.CloseHandle.argtypes=[ctypes.c_void_p];k.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(FT)]*4
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    handle=k.OpenProcess(0x1000|0x100000,False,int(worker['pid']))
    if not handle:
        if ctypes.get_last_error()==87:return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        times=[FT() for _ in range(4)]
        if not k.GetProcessTimes(handle,*[ctypes.byref(v) for v in times]):raise ctypes.WinError(ctypes.get_last_error())
        creation=(int(times[0].high)<<32)|int(times[0].low)
        if creation!=worker['creation_filetime_utc']:raise ValueError('PID reused; previous ownership needs reconciliation')
        if k.WaitForSingleObject(handle,0)!=0:raise ValueError('prior worker still alive')
    finally:k.CloseHandle(handle)
def validate_native_resume(binding,approval,worker,commit,group):
    reference=approval.get('previous_native_receipt',{})
    path=(ROOT/reference.get('path','')).resolve()
    files={(ROOT/p).resolve():h for p,h in binding.get('files',{}).items()}
    if not path.is_file() or files.get(path)!=reference.get('sha256') or sha(path)!=reference.get('sha256'):
        raise ValueError('actual source-bound native Job receipt required')
    proof=json.loads(path.read_text(encoding='utf-8'))
    native_worker=proof.get('worker',{})
    if (proof.get('source_commit')!=commit or proof.get('group')!=group or proof.get('phase')!='run'
            or type(native_worker.get('pid')) is not int or type(native_worker.get('creation_filetime_utc')) is not int
            or native_worker['pid']<=0 or native_worker['creation_filetime_utc']<=0
            or native_worker.get('pid')!=worker.get('pid') or native_worker.get('creation_filetime_utc')!=worker.get('creation_filetime_utc')
            or Path(native_worker.get('executable','')).resolve()!=Path(worker.get('executable','')).resolve()):
        raise ValueError('native Job/worker creation identity differs')
    if proof.get('cleanup_verified') is not True or type(proof.get('job_active_processes')) is not int or proof['job_active_processes']!=0:
        raise ValueError('actual Job subtree cleanup not verified')
    if type(proof.get('job_total_processes')) is not int or proof['job_total_processes']<1:
        raise ValueError('actual Job accounting missing')
    for key in ('job_cpu_user_seconds','job_cpu_kernel_seconds','wall_seconds'):
        value=proof.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('actual Job accounting missing')
    before=proof.get('source_before')
    if not isinstance(before,dict) or not before or before!=proof.get('source_after'):
        raise ValueError('actual source before/after verification missing')
    if any(k not in proof or proof[k] is not None for k in ('runtime_limits','cpu_limits','active_process_limit')):
        raise ValueError('typed uncapped native receipt required')
    if binding.get('reconstruction_authority'):
        command=validate_runtime_binding(binding)
        for row in (worker,native_worker):
            if (row.get('runtime_scheme')!=command['runtime_scheme']
                    or Path(row.get('logical_executable','')).resolve()!=Path(command['logical_executable']).resolve()
                    or Path(row.get('physical_image','')).resolve()!=Path(command['expected_worker_image']).resolve()
                    or row.get('runtime_selection')!={'path':RUNTIME_SELECTION,'sha256':binding['files'][RUNTIME_SELECTION]}):
                raise ValueError('resume selected runtime/physical/logical identity differs')
    return proof
def authorize_resume(out,binding,commit,group,rootdir):
    if binding.get('reconstruction_authority'):validate_qa06_pass(binding,commit)
    approval=bound(binding,'resume_authorization')
    if (approval.get('source_commit')!=commit or approval.get('group')!=group or Path(approval.get('output','')).resolve()!=out
            or approval.get('nonselective_interruption') is not True or approval.get('previous_job_closed') is not True or approval.get('resume_count')!=1):
        raise ValueError('explicit same-source nonselective interruption evidence required')
    if (out/'complete.json').exists() or (out/'selection.json').exists():raise ValueError('completed scientific attempts cannot be resumed')
    previous=approval.get('previous_worker_receipt',{})
    proof_path=(ROOT/previous.get('path','')).resolve()
    files={(ROOT/p).resolve():h for p,h in binding.get('files',{}).items()}
    if any(p.resolve() in files for p in (out/'worker.json',out/'progress.json')):raise ValueError('mutable resume output must not be source bound')
    if (proof_path==(out/'worker.json').resolve() or not proof_path.is_file() or files.get(proof_path)!=previous.get('sha256')
            or sha(proof_path)!=previous.get('sha256') or sha(out/'worker.json')!=previous.get('sha256')):
        raise ValueError('immutable previous worker proof differs from actual old receipt')
    worker=json.loads(proof_path.read_text(encoding='utf-8'))
    command=worker.get('command',[])
    if (worker.get('source_commit')!=commit or worker.get('phase')!='run'
            or '--group' not in command or command.index('--group')+1>=len(command)
            or command[command.index('--group')+1]!=group):raise ValueError('previous worker source/phase/group differs')
    validate_native_resume(binding,approval,worker,commit,group);prior_worker_gone(worker)
    record={'approval':approval,'previous_worker_receipt':previous,'time_utc_unix':time.time()}
    with (rootdir/'resume_used.json').open('x',encoding='utf-8') as file:json.dump(record,file,indent=2,allow_nan=False)
    for current,backup in ((out/'worker.json',out/'worker_before_resume.json'),(out/'progress.json',out/'progress_before_resume.json')):
        if current.is_file():
            if backup.exists():
                if sha(backup)!=sha(current):raise ValueError('previous interruption proof already differs')
            else:shutil.copyfile(current,backup)


def train_group(out,commit,group,resume=False):
    import numpy as np,torch
    noise=int(group[0]);form=group[2:];rows=[]
    for seed in study.SEEDS:
        key=group+'_'+str(seed);folder=out/key;existing=folder.exists();trace=[];gradient={'total_raw_norm':[],'clip_factor':[]};summary={'first':{},'last':{},'max':{}}
        try:
            if existing:
                if not resume:raise ValueError('model output already exists')
                pointer=json.loads((folder/'resume.json').read_text(encoding='utf-8'));identity=pointer['identity'];tr=study.load_data(folder/'train.npz')
                expected={'clean_sha256':study.array_sha(tr['coarse']),'fine_sha256':study.array_sha(tr['fine']),
                          'observed_sha256':study.array_sha(tr['observed']),'scales_sha256':study.array_sha(tr['scales']),'seed':seed,'noise':noise,'split':0}
                train_raw=study.fileid(folder/'train.npz');train_raw['bytes']=(folder/'train.npz').stat().st_size
                if identity!={'source_commit':commit,'key':key,'data_identity':expected,'train_raw':train_raw,'form':form}:raise ValueError('resume source/data identity differs')
                if not study.qualify(tr['coarse'],tr['fine'],tr['scales'])['valid']:raise ValueError('saved teacher is not numerically qualified')
            else:
                folder.mkdir();tr=study.data(seed,0,noise,None);train_raw=study.save_data(folder/'train.npz',tr)
                if not tr['qualification']['valid']:
                    rows.append({'key':key,'status':'UNRESOLVED','qualification':tr['qualification'],'train_raw':train_raw});continue
                identity={'source_commit':commit,'key':key,'data_identity':tr['identity'],'train_raw':train_raw,'form':form}
            m,opt=study.make(seed,tr['observed'],form);rng=np.random.default_rng(np.random.SeedSequence([seed,7]))
            if existing:
                step,trace,gradient,summary=restore_complete(folder,m,opt,rng,identity)
                previous=json.loads((out/'progress_before_resume.json').read_text(encoding='utf-8')) if (out/'progress_before_resume.json').exists() else {}
                if previous.get('key')==key:
                    atomic(folder/'interruption_recovery.json',{'restored_step':step,'known_completed_updates_recomputed':max(0,previous.get('last_completed',step)-step),
                        'possibly_partial_update':previous.get('inflight'),'possible_additional_recomputed_updates':int(previous.get('inflight') is not None),
                        'checkpoint_interval_upper_bound':{0:32,32:96,128:384,512:0}[step],'previous_progress':previous})
            else:step=0;publish_checkpoint(folder,m,opt,rng,0,trace,gradient,summary,identity)
            obs=torch.as_tensor(tr['observed'],device='cuda');scale=torch.as_tensor(tr['scales'],device='cuda')
            # Checkpoint32 is always restored once within the scientific512;
            # resuming exactly32 already performs that original continuation.
            while step<512:
                atomic(out/'progress.json',{'key':key,'last_completed':step,'inflight':step+1,'source_commit':commit})
                study.advance(m,opt,rng,obs,scale,None,trace,gradient,summary);step+=1
                atomic(out/'progress.json',{'key':key,'last_completed':step,'inflight':None,'source_commit':commit})
                if step in (32,128,512):publish_checkpoint(folder,m,opt,rng,step,trace,gradient,summary,identity)
                if step==32:
                    m,opt=study.make(seed,tr['observed'],form);step,trace,gradient,summary=restore_complete(folder,m,opt,rng,identity)
            new_or_equal(folder/'trace.json',{'loss':trace,'gradient':gradient,'parameter_summary':summary})
            path=folder/'checkpoint_00512';row={'key':key,'seed':seed,'noise':noise,'form':form,'status':'FROZEN','path':str(path),
                'files':{n:sha(path/n) for n in ('model.json','training.pt')},'train_raw':train_raw,'trace':study.fileid(folder/'trace.json'),'identity':identity,
                'loss_first32':sum(trace[:32])/32,'loss_last32':sum(trace[-32:])/32,'resume_pointer':study.fileid(folder/'resume.json')}
            study.verify(row);rows.append(row)
        except (RuntimeError,FloatingPointError,ValueError,FileNotFoundError,KeyError) as error:
            atomic(folder/'failed_trace.json',{'loss':trace,'gradient':gradient,'parameter_summary':summary})
            rows.append({'key':key,'status':'UNRESOLVED','error':repr(error),'trace':study.fileid(folder/'failed_trace.json'),
                         'raw_files':[study.fileid(p) for p in folder.glob('*.npz')]})
    result={'status':'COMPLETED','source_commit':commit,'group':group,'rows':rows,'all_group_frozen':all(r['status']=='FROZEN' for r in rows)}
    atomic(out/'selection.json',result);return result
def execute(phase,out,binding,commit,group,resume,qa_attempt=1):
    if phase=='qa':
        validate_qa_attempt(binding,commit,qa_attempt)
        if qa_attempt==3 and Path(out).resolve()!=(ROOT/ROUND_PATH/'qa03').resolve():raise ValueError('QA03 requires exact authorized output')
        if qa_attempt==4 and Path(out).resolve()!=(ROOT/ROUND_PATH/'qa04').resolve():raise ValueError('QA04 requires exact authorized output')
        if qa_attempt==5 and Path(out).resolve()!=(ROOT/ROUND_PATH/'qa05').resolve():raise ValueError('QA05 requires exact authorized output')
        if qa_attempt==6 and Path(out).resolve()!=(ROOT/ROUND_PATH/'qa06').resolve():raise ValueError('QA06 requires exact authorized output')
        import subprocess
        command=[sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_rational_execution_20261021.py'),
                 str(ROOT/'tests/test_uncapped_windows_job_20261021.py'),'--basetemp',str(out/'pytest_tmp')]
        environment=None
        if qa_attempt in (4,5,6):
            # Keep nested fixture copies and atomic filenames below Windows MAX_PATH.
            # QA04 remains historical; QA05 has separate authority for its short root.
            command[-1]=str(require_unused_qa06_temp() if qa_attempt==6 else require_unused_qa05_temp() if qa_attempt==5 else ROOT/'private_data/q4')
            from rational_python_runtime_20261022 import selected_python_command,command_environment
            arguments=command[1:]
            arguments[5:5]=[str(ROOT/'tests/test_rational_python_runtime_20261022.py'),str(ROOT/'tests/test_rational_control_20261021.py')]
            selected=selected_python_command(arguments,ROOT/RUNTIME_SELECTION)
            command=[selected['executable'],*selected['arguments']];environment=command_environment(selected)
        pytest_creation_flags=subprocess.DETACHED_PROCESS if os.name=='nt' else 0
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env=environment,creationflags=pytest_creation_flags)
        (out/'pytest_stdout.txt').write_text(result.stdout,encoding='utf-8');(out/'pytest_stderr.txt').write_text(result.stderr,encoding='utf-8')
        row={'status':'PASS' if result.returncode==0 else 'FAIL','qa_attempt':qa_attempt,'pytest_returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        if qa_attempt in (4,5,6):row['runtime_selection_sha256']=sha(ROOT/RUNTIME_SELECTION)
        if qa_attempt==6:row['pytest_creation_flags']=pytest_creation_flags
        return row
    validate_qa06_pass(binding,commit)
    import torch
    torch.set_num_threads(1)
    if phase=='run':return train_group(out,commit,group,resume)
    return study.evaluate(out,binding,commit,None)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('qa','run','evaluate'))
    for name in ('output','binding'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--source-commit',required=True);p.add_argument('--group',choices=study.GROUPS);p.add_argument('--resume',action='store_true')
    p.add_argument('--qa-attempt',type=int,choices=(1,2,3,4,5,6));a=p.parse_args(argv)
    if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit) or (a.phase=='run')!=(a.group is not None) or (a.resume and a.phase!='run') or (a.phase!='qa' and a.qa_attempt is not None):p.error('full source commit/run-only group/resume and qa-only attempt required')
    out=a.output.resolve();rootdir=(ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01').resolve()
    qa_attempt=(a.qa_attempt or 1) if a.phase=='qa' else None
    if a.phase=='qa' and out.name!=f'qa{qa_attempt:02d}':raise ValueError('QA attempt must use its own exclusive output directory')
    if a.phase=='qa' and qa_attempt==3 and out!=rootdir/'qa03':raise ValueError('QA03 requires exact authorized output')
    if a.phase=='qa' and qa_attempt==4 and out!=rootdir/'qa04':raise ValueError('QA04 requires exact authorized output')
    if a.phase=='qa' and qa_attempt==5 and out!=rootdir/'qa05':raise ValueError('QA05 requires exact authorized output')
    if a.phase=='qa' and qa_attempt==6 and out!=rootdir/'qa06':raise ValueError('QA06 requires exact authorized output')
    qa_root=(ROOT/'private_data/closure_mechanism_20261019/P0/rational_execution01').resolve()
    if out in (rootdir,qa_root) or not (out.is_relative_to(rootdir) or (a.phase=='qa' and out.is_relative_to(qa_root))):raise ValueError('exclusive new round output required')
    binding=json.loads(a.binding.read_text(encoding='utf-8'))
    if a.phase!='qa' or qa_attempt in (4,5,6):
        if (binding.get('phase')!=a.phase or binding.get('group')!=a.group
                or binding.get('qa_attempt')!=qa_attempt):raise ValueError('exact reconstruction binding role required')
    if a.phase=='qa':validate_qa_attempt(binding,a.source_commit,qa_attempt)
    else:validate_qa06_pass(binding,a.source_commit)
    if out.exists():
        if not a.resume:raise ValueError('output exists; explicit verified resume required')
        authorize_resume(out,binding,a.source_commit,a.group,rootdir)
    else:
        if a.resume:raise ValueError('resume requires actual interrupted output')
        out.mkdir(parents=True)
    # pythonw may leave Python's streams unset even with explicit native handles.
    # Establish real logs after authorization, without intercepting exceptions.
    if sys.stdout is None:sys.stdout=(out/'pythonw_stdout.txt').open('a',encoding='utf-8',buffering=1)
    if sys.stderr is None:sys.stderr=(out/'pythonw_stderr.txt').open('a',encoding='utf-8',buffering=1)
    publish_worker_receipt(out,a.phase,a.source_commit,a.binding,a.group,qa_attempt)
    try:
        result=execute(a.phase,out,binding,a.source_commit,a.group,a.resume,qa_attempt);result['source_commit']=a.source_commit
        atomic(out/'result.json',result);atomic(out/'complete.json',{'status':'COMPLETED','result':result})
        return 0 if result['status'] in ('PASS','COMPLETED') else 2
    except BaseException as error:
        import traceback
        atomic(out/'failure.json',{'status':'UNRESOLVED','error':repr(error),'traceback':traceback.format_exc()});raise
if __name__=='__main__':raise SystemExit(main())
