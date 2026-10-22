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
    if attempt not in (1,2,3):raise ValueError('only registered QA attempts1 or2 or authorized3')
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
    return proof
def authorize_resume(out,binding,commit,group,rootdir):
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
                pointer=json.loads((folder/'resume.json').read_text());identity=pointer['identity'];tr=study.load_data(folder/'train.npz')
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
                previous=json.loads((out/'progress_before_resume.json').read_text()) if (out/'progress_before_resume.json').exists() else {}
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
        import subprocess
        command=[sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_rational_execution_20261021.py'),
                 str(ROOT/'tests/test_uncapped_windows_job_20261021.py'),'--basetemp',str(out/'pytest_tmp')]
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (out/'pytest_stdout.txt').write_text(result.stdout,encoding='utf-8');(out/'pytest_stderr.txt').write_text(result.stderr,encoding='utf-8')
        return {'status':'PASS' if result.returncode==0 else 'FAIL','qa_attempt':qa_attempt,'pytest_returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    validate_qa03_pass(binding,commit)
    import torch
    torch.set_num_threads(1)
    if phase=='run':return train_group(out,commit,group,resume)
    return study.evaluate(out,binding,commit,None)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('qa','run','evaluate'))
    for name in ('output','binding'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--source-commit',required=True);p.add_argument('--group',choices=study.GROUPS);p.add_argument('--resume',action='store_true')
    p.add_argument('--qa-attempt',type=int,choices=(1,2,3));a=p.parse_args(argv)
    if len(a.source_commit)!=40 or any(c not in '0123456789abcdef' for c in a.source_commit) or (a.phase=='run')!=(a.group is not None) or (a.resume and a.phase!='run') or (a.phase!='qa' and a.qa_attempt is not None):p.error('full source commit/run-only group/resume and qa-only attempt required')
    out=a.output.resolve();rootdir=(ROOT/'private_data/closure_mechanism_20261019/P1/rational_execution01').resolve()
    qa_attempt=(a.qa_attempt or 1) if a.phase=='qa' else None
    if a.phase=='qa' and out.name!=f'qa{qa_attempt:02d}':raise ValueError('QA attempt must use its own exclusive output directory')
    if a.phase=='qa' and qa_attempt==3 and out!=rootdir/'qa03':raise ValueError('QA03 requires exact authorized output')
    qa_root=(ROOT/'private_data/closure_mechanism_20261019/P0/rational_execution01').resolve()
    if out in (rootdir,qa_root) or not (out.is_relative_to(rootdir) or (a.phase=='qa' and out.is_relative_to(qa_root))):raise ValueError('exclusive new round output required')
    binding=json.loads(a.binding.read_text(encoding='utf-8'))
    if a.phase=='qa':validate_qa_attempt(binding,a.source_commit,qa_attempt)
    else:validate_qa03_pass(binding,a.source_commit)
    if out.exists():
        if not a.resume:raise ValueError('output exists; explicit verified resume required')
        authorize_resume(out,binding,a.source_commit,a.group,rootdir)
    else:
        if a.resume:raise ValueError('resume requires actual interrupted output')
        out.mkdir(parents=True)
    publish_worker_receipt(out,a.phase,a.source_commit,a.binding,a.group,qa_attempt)
    try:
        result=execute(a.phase,out,binding,a.source_commit,a.group,a.resume,qa_attempt);result['source_commit']=a.source_commit
        atomic(out/'result.json',result);atomic(out/'complete.json',{'status':'COMPLETED','result':result})
        return 0 if result['status'] in ('PASS','COMPLETED') else 2
    except BaseException as error:
        import traceback
        atomic(out/'failure.json',{'status':'UNRESOLVED','error':repr(error),'traceback':traceback.format_exc()});raise
if __name__=='__main__':raise SystemExit(main())
