"""Thin scheduling/recovery tests only; existing numerical QA is reused."""
import ast,copy,json,sys,types
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
import rational_execution_20261021 as execution


@pytest.fixture
def fakes(monkeypatch):
    calls={'updates':[],'saves':[],'restores':[],'data':[]}
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(as_tensor=lambda a,**kw:a))
    def data(seed,split,noise,deadline):
        assert deadline is None and split==0;calls['data'].append((seed,noise,deadline))
        a=np.zeros((16,61,30));sc=np.ones(30)
        identity={'clean_sha256':execution.study.array_sha(a),'fine_sha256':execution.study.array_sha(a),
                  'observed_sha256':execution.study.array_sha(a),'scales_sha256':execution.study.array_sha(sc),'seed':seed,'noise':noise,'split':0}
        return {'coarse':a,'fine':a.copy(),'observed':a.copy(),'scales':sc,'identity':identity,'qualification':{'valid':True}}
    monkeypatch.setattr(execution.study,'data',data)
    monkeypatch.setattr(execution.study,'make',lambda seed,obs,form:({'seed':seed,'form':form},None))
    monkeypatch.setattr(execution.study,'verify',lambda row:None)
    def advance(m,opt,rng,obs,sc,deadline,trace,gradient,summary):
        assert deadline is None;calls['updates'].append((m['seed'],m['form'],len(trace)+1,deadline));trace.append(float(len(trace)+1))
        gradient['total_raw_norm'].append(.1);gradient['clip_factor'].append(1.)
        for k in ('first','last','max'):summary[k]={'p':.1}
    monkeypatch.setattr(execution.study,'advance',advance)
    def checkpoint(folder,m,opt,rng,step,trace,identity):
        calls['saves'].append((identity['key'],step));directory=folder/f'checkpoint_{step:05d}';directory.mkdir()
        execution.atomic(directory/'model.json',{'qa_only':True,'step':step})
        execution.atomic(directory/'training.pt',{'qa_only':True,'completed_updates':step,'trace':trace,'identity':identity})
        execution.atomic(folder/'latest.json',{'updates':step,'checkpoint':directory.name,
                                             'sha256':{n:execution.sha(directory/n) for n in ('model.json','training.pt')}})
    def restore(folder,m,opt,rng,identity):
        latest=json.loads((folder/'latest.json').read_text(encoding='utf-8'));state=json.loads((folder/latest['checkpoint']/'training.pt').read_text(encoding='utf-8'))
        assert state['identity']==identity;calls['restores'].append((identity['key'],state['completed_updates']))
        return state['completed_updates'],state['trace']
    monkeypatch.setattr(execution,'checkpoint_api',lambda:(checkpoint,restore))
    return calls


def test_complete_twelve_models_exact_schedule_and_none(fakes,tmp_path):
    results=[]
    for group in execution.study.GROUPS:
        out=tmp_path/group;out.mkdir();results.append(execution.train_group(out,'a'*40,group))
    assert all(r['all_group_frozen'] and len(r['rows'])==3 for r in results)
    assert len(fakes['updates'])==12*512 and len(fakes['saves'])==48 and len(fakes['restores'])==12
    assert all(step==32 for _,step in fakes['restores'])
    for result in results:
        for row in result['rows']:
            assert [step for key,step in fakes['saves'] if key==row['key']]==[0,32,128,512]
            trace=json.loads(Path(row['trace']['path']).read_text(encoding='utf-8'));assert len(trace['loss'])==512


def saved(folder,fakes,step=32):
    folder.mkdir();identity={'source_commit':'a'*40,'key':'qa_only'};trace=[float(i) for i in range(step)]
    gradient={'total_raw_norm':[.1]*step,'clip_factor':[1.]*step};summary={k:{'p':.1} for k in ('first','last','max')}
    execution.publish_checkpoint(folder,None,None,None,step,trace,gradient,summary,identity)
    return identity,trace,gradient,summary


def test_sidecar_exact_restore_and_missing_fields_rejected(fakes,tmp_path):
    folder=tmp_path/'model';identity,trace,gradient,summary=saved(folder,fakes)
    step,actual,g,s=execution.restore_complete(folder,None,None,None,identity)
    assert step==32 and actual==trace and g==gradient and s==summary
    sidecar=folder/'checkpoint_00032.diagnostics.json';value=json.loads(sidecar.read_text(encoding='utf-8'));del value['gradient']
    execution.atomic(sidecar,value)
    pointer=json.loads((folder/'resume.json').read_text(encoding='utf-8'));pointer['diagnostics']['sha256']=execution.sha(sidecar);execution.atomic(folder/'resume.json',pointer)
    with pytest.raises(ValueError,match='diagnostic fields'):execution.complete_sidecar(folder,identity)


def test_latest_ahead_of_sidecar_refuses_fallback(fakes,tmp_path):
    folder=tmp_path/'model';identity,*_=saved(folder,fakes)
    latest=json.loads((folder/'latest.json').read_text(encoding='utf-8'));latest['checkpoint']='checkpoint_00128';latest['updates']=128;execution.atomic(folder/'latest.json',latest)
    with pytest.raises(ValueError,match='ahead'):execution.complete_sidecar(folder,identity)


def test_pending_and_changed_checkpoint_rejected(fakes,tmp_path):
    folder=tmp_path/'model';identity,*_=saved(folder,fakes);(folder/'pending_partial').mkdir()
    with pytest.raises(ValueError,match='pending'):execution.complete_sidecar(folder,identity)


def test_evaluate_uses_existing_full_barrier_and_none(monkeypatch,tmp_path):
    # Isolate the unchanged scientific barrier; separate tests exercise QA03 admission.
    monkeypatch.setattr(execution,'validate_qa05_pass',lambda binding,commit:None)
    def evaluate(out,binding,commit,deadline):
        assert deadline is None
        execution.study.validate_selection({},commit)
    monkeypatch.setattr(execution.study,'evaluate',evaluate)
    import torch
    with pytest.raises(ValueError,match='all12'):execution.execute('evaluate',tmp_path,{},'a'*40,None,False)


def test_no_runtime_quota_or_admission_path():
    source=Path(execution.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
    assert 'admission_result' not in source and 'max-seconds' not in source
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            assert not any(k.arg=='timeout' for k in node.keywords)
            assert not (isinstance(node.func,ast.Attribute) and node.func.attr=='Timer')


def test_existing_evidence_cannot_be_overwritten(tmp_path):
    path=tmp_path/'evidence.json';execution.new_or_equal(path,{'value':1})
    execution.new_or_equal(path,{'value':1})
    with pytest.raises(ValueError,match='refusing overwrite'):execution.new_or_equal(path,{'value':2})


def test_resume_requires_actual_source_bound_native_receipt(tmp_path):
    worker={'pid':12345,'creation_filetime_utc':134370000000000001,'executable':sys.executable}
    with pytest.raises(ValueError,match='native Job receipt'):
        execution.validate_native_resume({'files':{}},{'previous_job_closed':True},worker,'a'*40,'0_rational')


@pytest.mark.parametrize('active',[1,2])
def test_root_gone_does_not_prove_child_subtree_empty(tmp_path,active):
    worker={'pid':12345,'creation_filetime_utc':134370000000000001,'executable':sys.executable}
    # Explicit negative engineering fixture: a GONE root/status flag cannot
    # override an actual receipt reporting remaining owned children.
    proof={'source_commit':'a'*40,'group':'0_rational','phase':'run','worker':worker,
           'actual_worker_state':'GONE','status':'COMPLETED','cleanup_verified':True,
           'job_active_processes':active,'job_total_processes':3,'job_cpu_user_seconds':.1,
           'job_cpu_kernel_seconds':.1,'wall_seconds':.2,'source_before':{'source.py':'f'*64},
           'source_after':{'source.py':'f'*64},'runtime_limits':None,'cpu_limits':None,'active_process_limit':None}
    path=tmp_path/'negative_native_receipt.json';execution.atomic(path,proof);identity={'path':str(path),'sha256':execution.sha(path)}
    with pytest.raises(ValueError,match='subtree cleanup'):
        execution.validate_native_resume({'files':{str(path):identity['sha256']}},
            {'previous_job_closed':True,'previous_native_receipt':identity},worker,'a'*40,'0_rational')


def test_resume_proof_copy_is_immutable_before_new_worker_receipt(monkeypatch,tmp_path):
    out=tmp_path/'run';out.mkdir();rootdir=tmp_path
    worker={'pid':12345,'creation_filetime_utc':134370000000000001,'executable':sys.executable,
            'source_commit':'a'*40,'phase':'run','command':['module','run','--group','0_rational']}
    execution.atomic(out/'worker.json',worker)
    approval={'source_commit':'a'*40,'group':'0_rational','output':str(out.resolve()),
              'nonselective_interruption':True,'previous_job_closed':True,'resume_count':1,
              'previous_worker_receipt':{'path':str(out/'worker.json'),'sha256':execution.sha(out/'worker.json')}}
    ap=tmp_path/'approval.json';execution.atomic(ap,approval)
    binding={'resume_authorization':{'path':str(ap),'sha256':execution.sha(ap)},
             'files':{str(out/'worker.json'):execution.sha(out/'worker.json')}}
    monkeypatch.setattr(execution,'prior_worker_gone',lambda worker:None)
    with pytest.raises(ValueError,match='mutable resume output'):
        execution.authorize_resume(out,binding,'a'*40,'0_rational',rootdir)
    assert not (rootdir/'resume_used.json').exists()


def test_qa_collects_both_new_suites_only(monkeypatch,tmp_path):
    import subprocess
    commands=[]
    def run(command,**kwargs):
        assert 'timeout' not in kwargs;commands.append(command)
        return types.SimpleNamespace(returncode=0,stdout='substitute only',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    execution.execute('qa',tmp_path,{},'a'*40,None,False)
    files=[Path(x).name for x in commands[0] if x.endswith('.py')]
    assert files==['test_rational_execution_20261021.py','test_uncapped_windows_job_20261021.py']


def sharing_error(code):
    error=PermissionError('simulated Windows metadata sharing conflict');error.winerror=code;return error


def test_metadata_atomic_recovers_transient_sharing_conflict(monkeypatch,tmp_path):
    actual=execution.os.replace;attempts=[];sleeps=[]
    def replace(source,target):
        attempts.append(1)
        if len(attempts)<=2:raise sharing_error(5 if len(attempts)==1 else 32)
        return actual(source,target)
    monkeypatch.setattr(execution.os,'replace',replace);monkeypatch.setattr(execution.time,'sleep',sleeps.append)
    path=tmp_path/'progress.json';execution.atomic(path,{'last_completed':388})
    assert json.loads(path.read_text(encoding='utf-8'))=={'last_completed':388}
    assert len(attempts)==3 and sleeps==[.01,.02] and not list(tmp_path.glob('*.tmp'))


def test_metadata_atomic_permanent_conflict_preserves_old_and_temp(monkeypatch,tmp_path):
    attempts=[];sleeps=[];path=tmp_path/'progress.json';path.write_text('{"last_completed":32}',encoding='utf-8')
    def replace(source,target):attempts.append(1);raise sharing_error(33)
    monkeypatch.setattr(execution.os,'replace',replace);monkeypatch.setattr(execution.time,'sleep',sleeps.append)
    with pytest.raises(PermissionError):execution.atomic(path,{'last_completed':128})
    assert len(attempts)==6 and sleeps==list(execution._REPLACE_DELAYS)
    assert json.loads(path.read_text(encoding='utf-8'))=={'last_completed':32}
    temporary=list(tmp_path.glob('*.tmp'));assert len(temporary)==1
    assert json.loads(temporary[0].read_text(encoding='utf-8'))=={'last_completed':128}


def test_metadata_atomic_does_not_retry_other_permission_errors(monkeypatch,tmp_path):
    attempts=[]
    def replace(source,target):attempts.append(1);raise sharing_error(87)
    monkeypatch.setattr(execution.os,'replace',replace)
    with pytest.raises(PermissionError):execution.atomic(tmp_path/'progress.json',{'value':1})
    assert len(attempts)==1


def test_second_qa_attempt_has_own_collection_and_no_unauthorized_third(monkeypatch,tmp_path):
    import subprocess
    commands=[]
    def run(command,**kwargs):
        assert 'timeout' not in kwargs;commands.append(command)
        return types.SimpleNamespace(returncode=0,stdout='substitute only',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    out=tmp_path/'qa02';out.mkdir()
    result=execution.execute('qa',out,{},'a'*40,None,False,qa_attempt=2)
    assert result['qa_attempt']==2 and commands[0][-1]==str(out/'pytest_tmp')
    with pytest.raises(ValueError,match='repair identity'):execution.execute('qa',out,{},'a'*40,None,False,qa_attempt=3)
    with pytest.raises(ValueError,match='binding role'):execution.execute('qa',out,{},'a'*40,None,False,qa_attempt=4)
    with pytest.raises(ValueError,match='binding role'):execution.execute('qa',out,{},'a'*40,None,False,qa_attempt=5)
    with pytest.raises(ValueError,match='attempts1 or2'):execution.execute('qa',out,{},'a'*40,None,False,qa_attempt=6)
    args=['qa','--output',str(tmp_path/'qa01'),'--binding',str(tmp_path/'missing.json'),'--source-commit','a'*40,'--qa-attempt','2']
    with pytest.raises(ValueError,match='own exclusive'):execution.main(args)


@pytest.fixture
def repair_binding(monkeypatch,tmp_path):
    """Exact archived authority/history copied as read-only fixture inputs."""
    original=execution.ROOT
    names=set(execution.repair_source_names())
    names.update(execution.ROUND_PATH+f'/native_qa{i:02d}/receipt.json' for i in (1,2))
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((original/name).read_bytes())
    monkeypatch.setattr(execution,'ROOT',tmp_path)
    binding={'source_commit':'a'*40,'phase':'qa','group':None,'qa_attempt':3,
             'files':{name:execution.sha(tmp_path/name) for name in names},
             'prior_receipts':[execution.ROUND_PATH+f'/native_qa{i:02d}/receipt.json' for i in (1,2)]}
    for key,(name,digest) in execution.REPAIR_INPUTS.items():binding[key]={'path':name,'sha256':digest}
    return binding


def test_third_qa_requires_exact_actual_authority_and_two_failures(repair_binding):
    execution.validate_qa_attempt(repair_binding,'a'*40,3)


def test_authorized_third_collects_same_two_suites_in_own_directory(monkeypatch,repair_binding):
    import subprocess
    commands=[]
    def run(command,**kwargs):
        assert 'timeout' not in kwargs;commands.append(command)
        return types.SimpleNamespace(returncode=0,stdout='substitute only',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    out=execution.ROOT/execution.ROUND_PATH/'qa03';out.mkdir()
    result=execution.execute('qa',out,repair_binding,'a'*40,None,False,qa_attempt=3)
    assert result['qa_attempt']==3 and commands[0][-1]==str(out/'pytest_tmp')
    assert [Path(x).name for x in commands[0] if x.endswith('.py')]==['test_rational_execution_20261021.py','test_uncapped_windows_job_20261021.py']


def test_third_qa_missing_declared_history_rejected(repair_binding):
    repair_binding['prior_receipts'].pop()
    with pytest.raises(ValueError,match='immutable QA01/QA02'):execution.validate_qa_attempt(repair_binding,'a'*40,3)


def test_third_qa_short_or_wrong_source_rejected(repair_binding):
    for commit in ('a'*39,'b'*40):
        with pytest.raises(ValueError,match='repair identity'):execution.validate_qa_attempt(repair_binding,commit,3)


def test_third_qa_tampered_authority_rejected_even_with_new_binding_hash(repair_binding):
    name,_=execution.REPAIR_INPUTS['additional_repair_authority'];path=execution.ROOT/name
    authority=json.loads(path.read_text(encoding='utf-8'));authority['additional_qa_executions']=2;path.write_text(json.dumps(authority),encoding='utf-8')
    repair_binding['additional_repair_authority']['sha256']=repair_binding['files'][name]=execution.sha(path)
    with pytest.raises(ValueError,match='exact repair authority'):execution.validate_qa_attempt(repair_binding,'a'*40,3)


@pytest.mark.parametrize('key',list(execution.REPAIR_INPUTS))
def test_third_qa_missing_or_forged_authority_rejected(repair_binding,key):
    del repair_binding[key]
    with pytest.raises(ValueError,match='exact repair authority'):execution.validate_qa_attempt(repair_binding,'a'*40,3)


@pytest.mark.parametrize('attempt',[1,2])
def test_historical_receipt_cannot_be_relabelled_or_rehashed(repair_binding,attempt):
    name=execution.ROUND_PATH+f'/native_qa{attempt:02d}/receipt.json';path=execution.ROOT/name
    row=json.loads(path.read_text(encoding='utf-8'));row['status']='COMPLETED';path.write_text(json.dumps(row),encoding='utf-8')
    repair_binding['files'][name]=execution.sha(path)
    with pytest.raises(ValueError,match='immutable QA01/QA02'):execution.validate_qa_attempt(repair_binding,'a'*40,3)


def test_third_qa_changed_current_source_rejected(repair_binding):
    name='experiments/rational_execution_20261021.py';(execution.ROOT/name).write_text('# changed',encoding='utf-8')
    with pytest.raises(ValueError,match='current repair source'):execution.validate_qa_attempt(repair_binding,'a'*40,3)


def test_third_qa_rejects_fourth_receipt_and_wrong_role(repair_binding):
    repair_binding['qa_attempt']=4
    with pytest.raises(ValueError,match='binding role'):execution.validate_qa_attempt(repair_binding,'a'*40,3)
    repair_binding['qa_attempt']=3
    path=execution.ROOT/execution.ROUND_PATH/'native_qa04/receipt.json';path.parent.mkdir();path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='exactly two'):execution.validate_qa_attempt(repair_binding,'a'*40,3)


def install_qa03_pass(binding):
    base=execution.ROOT/execution.ROUND_PATH
    source={str((execution.ROOT/name).resolve()):binding['files'][name] for name in execution.repair_source_names()}
    native={'status':'COMPLETED','phase':'qa','qa_attempt':3,'source_commit':'a'*40,'cleanup_verified':True,
            'job_active_processes':0,'source_before':source,'source_after':dict(source)}
    marker={'status':'COMPLETED','result':{'status':'PASS','qa_attempt':3,'source_commit':'a'*40}}
    for name,row in (('native_qa03/receipt.json',native),('qa03/complete.json',marker)):
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row),encoding='utf-8')
        binding['files'][path.relative_to(execution.ROOT).as_posix()]=execution.sha(path)
    binding['phase']='run';binding['qa_attempt']=None;binding['group']='0_rational'
    return native,marker


def test_formal_admission_requires_inner_outer_same_source_qa03(repair_binding):
    with pytest.raises(ValueError,match='QA03 evidence'):execution.validate_qa03_pass(repair_binding,'a'*40)
    install_qa03_pass(repair_binding);execution.validate_qa03_pass(repair_binding,'a'*40)


@pytest.mark.parametrize('target,field,value',[
    ('native','status','FAILED'),('native','source_commit','b'*40),('native','qa_attempt',2),
    ('native','cleanup_verified',False),('native','job_active_processes',1),
    ('marker','status','UNRESOLVED'),('inner','status','FAIL'),('inner','source_commit','b'*40),('inner','qa_attempt',2)])
def test_formal_admission_rejects_wrong_qa03_even_when_rehashed(repair_binding,target,field,value):
    native,marker=install_qa03_pass(repair_binding)
    row={'native':native,'marker':marker,'inner':marker['result']}[target];row[field]=value
    name='native_qa03/receipt.json' if target=='native' else 'qa03/complete.json'
    path=execution.ROOT/execution.ROUND_PATH/name;path.write_text(json.dumps(native if target=='native' else marker),encoding='utf-8')
    repair_binding['files'][path.relative_to(execution.ROOT).as_posix()]=execution.sha(path)
    with pytest.raises(ValueError,match='QA03 pass'):execution.validate_qa03_pass(repair_binding,'a'*40)


def test_formal_admission_rejects_wrong_qa03_authority_snapshot(repair_binding):
    native,_=install_qa03_pass(repair_binding)
    name=execution.REPAIR_INPUTS['additional_repair_authority'][0]
    key=str((execution.ROOT/name).resolve());native['source_before'][key]=native['source_after'][key]='f'*64
    path=execution.ROOT/execution.ROUND_PATH/'native_qa03/receipt.json';path.write_text(json.dumps(native),encoding='utf-8')
    repair_binding['files'][path.relative_to(execution.ROOT).as_posix()]=execution.sha(path)
    with pytest.raises(ValueError,match='authority/amendment differs'):execution.validate_qa03_pass(repair_binding,'a'*40)


def test_core_cli_rejects_missing_third_authority_before_creating_output(monkeypatch,tmp_path):
    monkeypatch.setattr(execution,'ROOT',tmp_path)
    path=tmp_path/'binding.json';path.write_text('{}',encoding='utf-8');out=tmp_path/execution.ROUND_PATH/'qa03'
    argv=['qa','--output',str(out),'--binding',str(path),'--source-commit','a'*40,'--qa-attempt','3']
    with pytest.raises(ValueError,match='repair identity'):execution.main(argv)
    assert not out.exists()
    argv[-1]='4'
    with pytest.raises(ValueError,match='own exclusive|exact authorized output'):execution.main(argv)
    assert not out.exists()


@pytest.fixture
def reconstruction_binding(monkeypatch,tmp_path):
    original=execution.ROOT
    names=set(execution.reconstruction_source_names())
    names.update(execution.ROUND_PATH+f'/native_qa{i:02d}/receipt.json' for i in (1,2,3))
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((original/name).read_bytes())
    monkeypatch.setattr(execution,'ROOT',tmp_path)
    binding={'source_commit':'a'*40,'phase':'qa','group':None,'qa_attempt':4,
        'files':{name:execution.sha(tmp_path/name) for name in names},
        'prior_receipts':[execution.ROUND_PATH+f'/native_qa{i:02d}/receipt.json' for i in (1,2,3)]}
    for key,(name,digest) in execution.RECONSTRUCTION_INPUTS.items():binding[key]={'path':name,'sha256':digest}
    path=tmp_path/execution.RUNTIME_SELECTION;path.parent.mkdir(parents=True,exist_ok=True)
    execution.atomic(path,{'files':{}})
    binding['files'][execution.RUNTIME_SELECTION]=execution.sha(path)
    binding['runtime_selection']={'path':execution.RUNTIME_SELECTION,'sha256':execution.sha(path)}
    # Raw runtime-proof negatives live in the helper suite; isolate admission here.
    command={'executable':str(tmp_path/'base.exe'),'logical_executable':str(tmp_path/'venv.exe'),
             'expected_worker_image':str(tmp_path/'base.exe'),'arguments':[],
             'runtime_scheme':'direct_base_venv','environment_overrides':{'PYTHONIOENCODING':'utf-8'}}
    monkeypatch.setattr(execution,'validate_runtime_binding',lambda binding:copy.deepcopy(command))
    return binding,command


def test_qa04_separate_authority_and_three_actual_failures(reconstruction_binding):
    binding,_=reconstruction_binding
    execution.validate_qa_attempt(binding,'a'*40,4)
    assert execution.QA03_HISTORY[0]=='95ee7cfe3d0c9dbdc065a306fcdc1d60fd0078d3'
    binding['prior_receipts'].pop()
    with pytest.raises(ValueError,match='immutable QA03'):execution.validate_qa_attempt(binding,'a'*40,4)


@pytest.mark.parametrize('key',list(execution.RECONSTRUCTION_INPUTS))
def test_qa04_rejects_unbound_or_replaced_authority(reconstruction_binding,key):
    binding,_=reconstruction_binding;binding.pop(key)
    with pytest.raises(ValueError,match='exact reconstruction authority'):execution.validate_qa_attempt(binding,'a'*40,4)


@pytest.mark.parametrize('field,value',[('phase','run'),('group','0_tanh'),('qa_attempt',5)])
def test_qa04_rejects_wrong_role_before_proof_io(reconstruction_binding,field,value):
    binding,_=reconstruction_binding;binding[field]=value
    with pytest.raises(ValueError,match='binding role'):execution.validate_qa_attempt(binding,'a'*40,4)


def test_qa04_rejects_relabelled_historical_qa03(reconstruction_binding):
    binding,_=reconstruction_binding
    name=execution.ROUND_PATH+'/native_qa03/receipt.json';path=execution.ROOT/name
    row=json.loads(path.read_text(encoding='utf-8'));row['status']='COMPLETED';execution.atomic(path,row)
    binding['files'][name]=execution.sha(path)
    with pytest.raises(ValueError,match='immutable QA03'):execution.validate_qa_attempt(binding,'a'*40,4)


def install_qa04(binding,command):
    base=execution.ROOT/execution.ROUND_PATH
    source={str((execution.ROOT/name).resolve()):binding['files'][name] for name in execution.reconstruction_source_names()+(execution.RUNTIME_SELECTION,)}
    native={'status':'COMPLETED','phase':'qa','qa_attempt':4,'source_commit':'a'*40,'cleanup_verified':True,
        'job_active_processes':0,'orphans_cleaned':False,'actual_worker_state':'GONE',
        'exit_code':0,'launcher_exit_code':0,'worker_exit_code':0,
        'source_before':source,'source_after':dict(source),'command':dict(command,arguments=['-m','qa'])}
    worker={'pid':12345,'creation_filetime_utc':134370000000000001,
        'physical_image':command['expected_worker_image'],'logical_executable':command['logical_executable'],
        'runtime_scheme':command['runtime_scheme'],
        'runtime_selection':{'path':execution.RUNTIME_SELECTION,'sha256':binding['files'][execution.RUNTIME_SELECTION]}}
    native.update(worker=worker,worker_receipt=dict(worker),runtime_limits=None,cpu_limits=None,active_process_limit=None)
    marker={'status':'COMPLETED','result':{'status':'PASS','qa_attempt':4,'source_commit':'a'*40,
        'pytest_returncode':0,'runtime_selection_sha256':binding['files'][execution.RUNTIME_SELECTION]}}
    for name,row in (('native_qa04/receipt.json',native),('qa04/complete.json',marker)):
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);execution.atomic(path,row)
        binding['files'][path.relative_to(execution.ROOT).as_posix()]=execution.sha(path)
    binding.update(phase='run',group='0_rational',qa_attempt=None)
    return native,marker


def test_formal_admission_now_requires_inner_outer_qa04(reconstruction_binding):
    binding,command=reconstruction_binding
    with pytest.raises(ValueError,match='QA04/receipt'):execution.validate_qa04_pass(binding,'a'*40)
    install_qa04(binding,command);execution.validate_qa04_pass(binding,'a'*40)


@pytest.mark.parametrize('target,field,value',[
    ('native','status','FAILED'),('native','source_commit','b'*40),('native','qa_attempt',3),
    ('native','orphans_cleaned',True),('native','job_active_processes',1),('native','actual_worker_state','ALIVE'),
    ('native','exit_code',1),('native','launcher_exit_code',True),('native','worker_exit_code',None),
    ('inner','status','FAIL'),('inner','pytest_returncode',1),('inner','runtime_selection_sha256','f'*64)])
def test_qa04_pass_rejects_rehashed_bad_native_or_inner(reconstruction_binding,target,field,value):
    binding,command=reconstruction_binding;native,marker=install_qa04(binding,command)
    row=native if target=='native' else marker['result'];row[field]=value
    name=execution.ROUND_PATH+('/native_qa04/receipt.json' if target=='native' else '/qa04/complete.json')
    execution.atomic(execution.ROOT/name,native if target=='native' else marker)
    binding['files'][name]=execution.sha(execution.ROOT/name)
    with pytest.raises(ValueError,match='QA04 pass'):execution.validate_qa04_pass(binding,'a'*40)


def test_core_unauthorized_qa04_and_qa05_leave_no_output(monkeypatch,tmp_path):
    monkeypatch.setattr(execution,'ROOT',tmp_path)
    binding=tmp_path/'binding.json';execution.atomic(binding,{});out=tmp_path/execution.ROUND_PATH/'qa04'
    argv=['qa','--output',str(out),'--binding',str(binding),'--source-commit','a'*40,'--qa-attempt','4']
    with pytest.raises(ValueError,match='binding role'):execution.main(argv)
    assert not out.exists()
    argv[-1]='5';out=tmp_path/execution.ROUND_PATH/'qa05';argv[2]=str(out)
    execution.atomic(binding,{'source_commit':'a'*40,'phase':'qa','qa_attempt':5,'group':None})
    with pytest.raises(ValueError,match='exact QA05 authority'):execution.main(argv)
    assert not out.exists()
    argv[-1]='6'
    with pytest.raises(SystemExit):execution.main(argv)
    assert not out.exists()


def test_qa04_collects_all_four_suites_using_selected_runtime(monkeypatch,reconstruction_binding):
    import subprocess
    binding,selected=reconstruction_binding;calls=[]
    def build(arguments,path):
        assert path==execution.ROOT/execution.RUNTIME_SELECTION
        return dict(selected,arguments=list(arguments))
    helper=types.SimpleNamespace(selected_python_command=build,command_environment=lambda command:{'QA_UTF8':'中文'})
    monkeypatch.setitem(sys.modules,'rational_python_runtime_20261022',helper)
    def run(command,**kwargs):
        calls.append((command,kwargs));return types.SimpleNamespace(returncode=0,stdout='fixture only',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    out=execution.ROOT/execution.ROUND_PATH/'qa04';out.mkdir()
    result=execution.execute('qa',out,binding,'a'*40,None,False,4)
    command,kwargs=calls[0]
    assert command[0]==selected['executable'] and kwargs['env']=={'QA_UTF8':'中文'} and kwargs['encoding']=='utf-8'
    assert [Path(x).name for x in command if x.endswith('.py')]==[
        'test_rational_execution_20261021.py','test_uncapped_windows_job_20261021.py',
        'test_rational_python_runtime_20261022.py','test_rational_control_20261021.py']
    assert result['runtime_selection_sha256']==binding['files'][execution.RUNTIME_SELECTION]


@pytest.mark.parametrize('field,value',[('physical_image','wrong.exe'),('logical_executable','wrong.exe'),('runtime_scheme','venv_pythonw')])
def test_qa04_rejects_rehashed_wrong_runtime_identity(reconstruction_binding,field,value):
    binding,command=reconstruction_binding;native,_=install_qa04(binding,command)
    native['worker'][field]=native['worker_receipt'][field]=value
    name=execution.ROUND_PATH+'/native_qa04/receipt.json';execution.atomic(execution.ROOT/name,native)
    binding['files'][name]=execution.sha(execution.ROOT/name)
    with pytest.raises(ValueError,match='worker identity differs'):execution.validate_qa04_pass(binding,'a'*40)


def install_qa05(binding,command):
    base=execution.ROOT/execution.ROUND_PATH
    source={str((execution.ROOT/name).resolve()):binding['files'][name] for name in execution.qa05_source_names()+(execution.RUNTIME_SELECTION,)}
    native={'status':'COMPLETED','phase':'qa','qa_attempt':5,'source_commit':'a'*40,'cleanup_verified':True,
        'job_active_processes':0,'orphans_cleaned':False,'actual_worker_state':'GONE',
        'exit_code':0,'launcher_exit_code':0,'worker_exit_code':0,
        'source_before':source,'source_after':dict(source),'command':dict(command,arguments=['-m','qa'])}
    worker={'pid':12345,'creation_filetime_utc':134370000000000001,
        'physical_image':command['expected_worker_image'],'logical_executable':command['logical_executable'],
        'runtime_scheme':command['runtime_scheme'],
        'runtime_selection':{'path':execution.RUNTIME_SELECTION,'sha256':binding['files'][execution.RUNTIME_SELECTION]}}
    native.update(worker=worker,worker_receipt=dict(worker),runtime_limits=None,cpu_limits=None,active_process_limit=None)
    marker={'status':'COMPLETED','result':{'status':'PASS','qa_attempt':5,'source_commit':'a'*40,
        'pytest_returncode':0,'runtime_selection_sha256':binding['files'][execution.RUNTIME_SELECTION]}}
    for name,row in (('native_qa05/receipt.json',native),('qa05/complete.json',marker)):
        path=base/name;path.parent.mkdir(parents=True,exist_ok=True);execution.atomic(path,row)
        binding['files'][path.relative_to(execution.ROOT).as_posix()]=execution.sha(path)
    binding.update(phase='run',group='0_rational',qa_attempt=None)
    return native,marker


@pytest.fixture
def qa05_binding(reconstruction_binding):
    binding,command=reconstruction_binding
    names=set(execution.qa05_source_names())
    names.add(execution.ROUND_PATH+'/native_qa04/receipt.json')
    for name in names:
        path=execution.ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes());binding['files'][name]=execution.sha(path)
    for key,(name,digest) in execution.QA05_INPUTS.items():binding[key]={'path':name,'sha256':digest}
    binding['prior_receipts'].append(execution.ROUND_PATH+'/native_qa04/receipt.json')
    binding['qa_attempt']=5
    return binding,command


def test_qa05_separate_authority_and_four_actual_failures(qa05_binding):
    binding,_=qa05_binding;execution.validate_qa_attempt(binding,'a'*40,5)
    binding['prior_receipts'].pop()
    with pytest.raises(ValueError,match='immutable QA04'):execution.validate_qa_attempt(binding,'a'*40,5)


@pytest.mark.parametrize('key',list(execution.QA05_INPUTS))
def test_qa05_missing_authority_rejected(qa05_binding,key):
    binding,_=qa05_binding;binding.pop(key)
    with pytest.raises(ValueError,match='exact QA05 authority'):execution.validate_qa_attempt(binding,'a'*40,5)


@pytest.mark.parametrize('field,value',[('phase','run'),('group','0_rational'),('qa_attempt',4)])
def test_qa05_wrong_role_rejected(qa05_binding,field,value):
    binding,_=qa05_binding;binding[field]=value
    with pytest.raises(ValueError,match='binding role'):execution.validate_qa_attempt(binding,'a'*40,5)


@pytest.mark.parametrize('field,value',[('status','COMPLETED'),('source_commit','a'*40),('cleanup_verified',False),('job_active_processes',1),('qa_attempt',3)])
def test_qa05_rehashed_historical_qa04_rejected(qa05_binding,field,value):
    binding,_=qa05_binding;name=execution.ROUND_PATH+'/native_qa04/receipt.json'
    path=execution.ROOT/name;row=json.loads(path.read_text(encoding='utf-8'));row[field]=value
    execution.atomic(path,row);binding['files'][name]=execution.sha(path)
    with pytest.raises(ValueError,match='immutable QA04'):execution.validate_qa_attempt(binding,'a'*40,5)


def test_formal_execution_requires_qa05_not_qa04(monkeypatch,qa05_binding):
    binding,command=qa05_binding
    # A historical QA04 validator returning successfully cannot grant admission.
    monkeypatch.setattr(execution,'validate_qa04_pass',lambda *args:None)
    with pytest.raises(ValueError,match='native_qa05'):execution.execute('run',execution.ROOT,binding,'a'*40,'0_rational',False)
    install_qa05(binding,command);execution.validate_qa05_pass(binding,'a'*40)


@pytest.mark.parametrize('target,field,value',[
    ('native','status','FAILED'),('native','source_commit','b'*40),('native','qa_attempt',4),
    ('native','orphans_cleaned',True),('native','job_active_processes',1),('native','actual_worker_state','RUNNING'),
    ('native','exit_code',1),('native','launcher_exit_code',True),('native','worker_exit_code',None),
    ('inner','status','FAIL'),('inner','source_commit','b'*40),('inner','pytest_returncode',1),('inner','runtime_selection_sha256','f'*64)])
def test_qa05_rehashed_native_or_inner_failure_rejected(qa05_binding,target,field,value):
    binding,command=qa05_binding;native,marker=install_qa05(binding,command)
    row=native if target=='native' else marker['result'];row[field]=value
    name=execution.ROUND_PATH+('/native_qa05/receipt.json' if target=='native' else '/qa05/complete.json')
    execution.atomic(execution.ROOT/name,native if target=='native' else marker);binding['files'][name]=execution.sha(execution.ROOT/name)
    with pytest.raises(ValueError,match='QA05 pass'):execution.validate_qa05_pass(binding,'a'*40)


@pytest.mark.parametrize('field,value',[('physical_image','wrong.exe'),('logical_executable','wrong.exe'),('runtime_scheme','venv_pythonw')])
def test_qa05_wrong_actual_runtime_identity_rejected(qa05_binding,field,value):
    binding,command=qa05_binding;native,_=install_qa05(binding,command)
    native['worker'][field]=native['worker_receipt'][field]=value
    name=execution.ROUND_PATH+'/native_qa05/receipt.json';execution.atomic(execution.ROOT/name,native)
    binding['files'][name]=execution.sha(execution.ROOT/name)
    with pytest.raises(ValueError,match='worker identity differs'):execution.validate_qa05_pass(binding,'a'*40)


def test_qa05_wrong_source_snapshot_rejected(qa05_binding):
    binding,command=qa05_binding;native,_=install_qa05(binding,command)
    name=execution.QA05_INPUTS['qa05_astra_raw'][0];key=str((execution.ROOT/name).resolve())
    native['source_before'][key]=native['source_after'][key]='f'*64
    name=execution.ROUND_PATH+'/native_qa05/receipt.json';execution.atomic(execution.ROOT/name,native)
    binding['files'][name]=execution.sha(execution.ROOT/name)
    with pytest.raises(ValueError,match='source/authority/runtime differs'):execution.validate_qa05_pass(binding,'a'*40)


def test_qa05_collects_four_suites_with_short_unused_temp(monkeypatch,qa05_binding):
    import subprocess
    binding,selected=qa05_binding;calls=[]
    helper=types.SimpleNamespace(selected_python_command=lambda args,path:dict(selected,arguments=list(args)),
        command_environment=lambda command:{'QA_UTF8':'中文'})
    monkeypatch.setitem(sys.modules,'rational_python_runtime_20261022',helper)
    def run(command,**kwargs):
        calls.append((command,kwargs));return types.SimpleNamespace(returncode=0,stdout='fixture only',stderr='')
    monkeypatch.setattr(subprocess,'run',run)
    out=execution.ROOT/execution.ROUND_PATH/'qa05';out.mkdir()
    result=execution.execute('qa',out,binding,'a'*40,None,False,5)
    command,kwargs=calls[0]
    assert command[-2:]==['--basetemp',str(execution.ROOT/'private_data/q5')]
    assert command[0]==selected['executable'] and kwargs['env']=={'QA_UTF8':'中文'} and kwargs['encoding']=='utf-8'
    assert [Path(x).name for x in command if x.endswith('.py')]==[
        'test_rational_execution_20261021.py','test_uncapped_windows_job_20261021.py',
        'test_rational_python_runtime_20261022.py','test_rational_control_20261021.py']
    assert result['qa_attempt']==5 and result['runtime_selection_sha256']==binding['files'][execution.RUNTIME_SELECTION]


def test_qa05_existing_temp_preserved_without_starting_pytest(monkeypatch,qa05_binding):
    import subprocess
    binding,_=qa05_binding;temp=execution.ROOT/'private_data/q5';temp.mkdir()
    sentinel=temp/'preserve.txt';sentinel.write_text('existing evidence',encoding='utf-8')
    monkeypatch.setattr(subprocess,'run',lambda *args,**kwargs:pytest.fail('must reject before process'))
    with pytest.raises(ValueError,match='must not exist'):execution.validate_qa_attempt(binding,'a'*40,5)
    assert sentinel.read_text(encoding='utf-8')=='existing evidence'


def test_qa05_repeat_native_receipt_rejected(qa05_binding):
    binding,_=qa05_binding;path=execution.ROOT/execution.ROUND_PATH/'native_qa05/receipt.json'
    path.parent.mkdir();path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='exactly four'):execution.validate_qa_attempt(binding,'a'*40,5)


def test_qa05_resume_and_attempt6_rejected_without_outputs(monkeypatch,tmp_path):
    monkeypatch.setattr(execution,'ROOT',tmp_path)
    binding=tmp_path/'binding.json';execution.atomic(binding,{})
    out=tmp_path/execution.ROUND_PATH/'qa05'
    argv=['qa','--output',str(out),'--binding',str(binding),'--source-commit','a'*40,'--qa-attempt','5','--resume']
    with pytest.raises(SystemExit):execution.main(argv)
    assert not out.exists()
    argv=argv[:-1];argv[-1]='6'
    with pytest.raises(SystemExit):execution.main(argv)
    assert not out.exists()

