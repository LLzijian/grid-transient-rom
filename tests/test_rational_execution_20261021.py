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
        latest=json.loads((folder/'latest.json').read_text());state=json.loads((folder/latest['checkpoint']/'training.pt').read_text())
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
            trace=json.loads(Path(row['trace']['path']).read_text());assert len(trace['loss'])==512


def saved(folder,fakes,step=32):
    folder.mkdir();identity={'source_commit':'a'*40,'key':'qa_only'};trace=[float(i) for i in range(step)]
    gradient={'total_raw_norm':[.1]*step,'clip_factor':[1.]*step};summary={k:{'p':.1} for k in ('first','last','max')}
    execution.publish_checkpoint(folder,None,None,None,step,trace,gradient,summary,identity)
    return identity,trace,gradient,summary


def test_sidecar_exact_restore_and_missing_fields_rejected(fakes,tmp_path):
    folder=tmp_path/'model';identity,trace,gradient,summary=saved(folder,fakes)
    step,actual,g,s=execution.restore_complete(folder,None,None,None,identity)
    assert step==32 and actual==trace and g==gradient and s==summary
    sidecar=folder/'checkpoint_00032.diagnostics.json';value=json.loads(sidecar.read_text());del value['gradient']
    execution.atomic(sidecar,value)
    pointer=json.loads((folder/'resume.json').read_text());pointer['diagnostics']['sha256']=execution.sha(sidecar);execution.atomic(folder/'resume.json',pointer)
    with pytest.raises(ValueError,match='diagnostic fields'):execution.complete_sidecar(folder,identity)


def test_latest_ahead_of_sidecar_refuses_fallback(fakes,tmp_path):
    folder=tmp_path/'model';identity,*_=saved(folder,fakes)
    latest=json.loads((folder/'latest.json').read_text());latest['checkpoint']='checkpoint_00128';latest['updates']=128;execution.atomic(folder/'latest.json',latest)
    with pytest.raises(ValueError,match='ahead'):execution.complete_sidecar(folder,identity)


def test_pending_and_changed_checkpoint_rejected(fakes,tmp_path):
    folder=tmp_path/'model';identity,*_=saved(folder,fakes);(folder/'pending_partial').mkdir()
    with pytest.raises(ValueError,match='pending'):execution.complete_sidecar(folder,identity)


def test_evaluate_uses_existing_full_barrier_and_none(monkeypatch,tmp_path):
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
           'worker_identity_after':'GONE','status':'COMPLETED','cleanup_verified':True,
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
