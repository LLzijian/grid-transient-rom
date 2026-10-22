import json
import copy
import hashlib
from pathlib import Path

import pytest

import experiments.rational_python_runtime_20261022 as runtime
from experiments.rational_python_runtime_20261022 import (
    command_environment,
    python_command,
    selected_python_command,
)


def test_direct_base_venv_constructs_base_command(tmp_path):
    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = C:/Python312\nversion = 3.12.14\n", encoding="utf-8")
    logical = venv / "Scripts" / "python.exe"
    cmd = python_command(logical, ["-c", "print(1)"], "direct_base_venv")
    assert cmd["executable"].endswith("Python312/python.exe") or cmd["executable"].endswith("Python312\\python.exe")
    assert cmd["logical_executable"] == str(logical.resolve())
    assert cmd["environment_overrides"]["__PYVENV_LAUNCHER__"] == str(logical.resolve())
    assert command_environment(cmd)["__PYVENV_LAUNCHER__"] == str(logical.resolve())


def test_pythonw_distinguishes_logical_launcher_and_physical_image(tmp_path):
    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv/'pyvenv.cfg').write_text('home = C:/Python312\n',encoding='utf-8')
    logical = venv / "Scripts" / "python.exe"
    cmd = python_command(logical, ["probe.py"], "venv_pythonw")
    assert Path(cmd['logical_executable'])==logical.with_name('pythonw.exe').resolve()
    assert cmd['executable']==cmd['logical_executable']
    assert Path(cmd['expected_worker_image'])==Path('C:/Python312/pythonw.exe').resolve()
    assert cmd['expected_worker_image']!=cmd['executable']


def test_rejects_unsupported_scheme_and_environment(tmp_path):
    with pytest.raises(ValueError):
        python_command("python.exe", [], "unknown")
    with pytest.raises(ValueError):
        command_environment({"environment_overrides": {"PATH": "bad"}})


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def selected_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    venv=tmp_path/'中文venv';base=tmp_path/'base';(venv/'Scripts').mkdir(parents=True);base.mkdir()
    cfg=venv/'pyvenv.cfg';cfg.write_text('home = '+str(base)+'\n',encoding='utf-8')
    logical=venv/'Scripts/python.exe';logical.write_bytes(b'logical launcher')
    (base/'python.exe').write_bytes(b'physical interpreter');dll=base/'python312.dll';dll.write_bytes(b'Python DLL')
    command=python_command(logical,[], 'direct_base_venv')
    selection={'status':'SELECTED','scheme':'direct_base_venv',
        'logical_executable':command['logical_executable'],'executable':command['executable'],
        'expected_worker_image':command['expected_worker_image'],'observation_source_commit':'b'*40,
        'observation_attempt':1,'files':{}}
    def put(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value,ensure_ascii=False),encoding='utf-8')
        selection['files'][str(path)]=digest(path)
        return {'path':str(path),'sha256':digest(path)}
    metadata={'python_version':'3.12.14','python_dll':{'path':str(dll),'sha256':digest(dll)},
        'logical_executable':command['logical_executable'],'physical_executable':command['expected_worker_image'],
        'prefix':str(venv),'base_prefix':str(base),'package_paths':[str(venv/'Lib/site-packages')],
        'torch_version':'fixture','torch_file':str(venv/'torch.py'),'cuda_version':'fixture','cuda_device':'fixture',
        'deterministic_algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,
        'num_threads':1,'interop_threads':1,'tensor14':list(range(14))}
    baseline=put('baseline.json',metadata);candidate=put('candidate.json',metadata)
    child=put('child.json',{'pid':4321,'creation_filetime_utc':12345,'executable':command['logical_executable']})
    normal_child=put('normal_child.json',{'pid':4322,'creation_filetime_utc':12346,'executable':command['logical_executable']})
    selection['baseline_reference']=baseline
    selection['parity_evidence']=put('parity.json',{'baseline':baseline,'candidate':candidate,'child_identity':child,
        'normal_child_identity':normal_child,'checks':{'assertion_only':True}})
    source=tmp_path/'probe.py';source.write_text('# frozen observation source\n',encoding='utf-8')
    (tmp_path/'experiments').mkdir()
    sources=[source]
    for filename in ('rational_python_runtime_20261022.py','uncapped_windows_job_20261021.py'):
        target=tmp_path/'experiments'/filename;target.write_text('# source fixture\n',encoding='utf-8');sources.append(target)
    before={str(target):digest(target) for target in sources}
    normal={'source_commit':'b'*40,'status':'COMPLETED','cleanup_verified':True,'orphans_cleaned':False,
        'job_active_processes':0,'job_total_processes':2,'actual_worker_state':'GONE','exit_code':0,
        'launcher_exit_code':0,'worker_exit_code':0,'source_before':before,'source_after':dict(before),
        'worker':{'pid':1234,'creation_filetime_utc':12340,'physical_image':command['expected_worker_image']},'command':command}
    selection['normal_receipt']=put('normal.json',normal)
    orphan=copy.deepcopy(normal);orphan.update(status='FAILED',orphans_cleaned=True)
    selection['orphan_receipt']=put('orphan.json',orphan)
    for path in (cfg,logical,base/'python.exe',dll,*sources):selection['files'][str(path)]=digest(path)
    path=tmp_path/'selection.json';path.write_text(json.dumps(selection),encoding='utf-8')
    import experiments.uncapped_windows_job_20261021 as job
    monkeypatch.setattr(job,'identity_state',lambda identity:'GONE')
    return path,selection,put


def save_selection(path,value):path.write_text(json.dumps(value),encoding='utf-8')


def test_selected_runtime_requires_actual_raw_proofs(selected_fixture):
    path,selection,_=selected_fixture
    command=selected_python_command(['-m','pytest'],path)
    assert command['arguments']==['-m','pytest']
    assert command['runtime_selection']['sha256']==digest(path)
    selection.pop('orphan_receipt');save_selection(path,selection)
    with pytest.raises(ValueError,match='raw reference'):selected_python_command([],path)


@pytest.mark.parametrize('kind,field,value',[
    ('normal_receipt','status','FAILED'),('normal_receipt','orphans_cleaned',True),
    ('orphan_receipt','status','COMPLETED'),('orphan_receipt','orphans_cleaned',False),
    ('orphan_receipt','job_active_processes',1),('orphan_receipt','worker_exit_code',2),
    ('orphan_receipt','actual_worker_state','ALIVE'),('orphan_receipt','source_commit','c'*40)])
def test_rehashed_native_proof_cannot_relax_selection(selected_fixture,kind,field,value):
    path,selection,put=selected_fixture;ref=selection[kind]
    proof=json.loads(Path(ref['path']).read_text(encoding='utf-8'));proof[field]=value
    selection[kind]=put(Path(ref['path']).name,proof);save_selection(path,selection)
    with pytest.raises(ValueError,match='native normal/orphan'):selected_python_command([],path)


def test_parity_checks_boolean_cannot_override_actual_tensor(selected_fixture):
    path,selection,put=selected_fixture
    parity=json.loads(Path(selection['parity_evidence']['path']).read_text(encoding='utf-8'))
    metadata=json.loads(Path(parity['candidate']['path']).read_text(encoding='utf-8'));metadata['tensor14'][0]=999
    parity['candidate']=put('candidate.json',metadata);selection['parity_evidence']=put('parity.json',parity)
    save_selection(path,selection)
    with pytest.raises(ValueError,match='actual computation/runtime parity'):selected_python_command([],path)


def test_selected_interpreter_change_is_rejected(selected_fixture):
    path,selection,_=selected_fixture;Path(selection['executable']).write_bytes(b'changed interpreter')
    with pytest.raises(ValueError,match='runtime file changed'):selected_python_command([],path)


def test_actual_orphan_identity_must_be_gone(selected_fixture,monkeypatch):
    path,_,_=selected_fixture
    import experiments.uncapped_windows_job_20261021 as job
    monkeypatch.setattr(job,'identity_state',lambda identity:'ALIVE')
    with pytest.raises(ValueError,match='not GONE'):selected_python_command([],path)


def test_pythonw_does_not_inherit_venv_launcher_marker(monkeypatch):
    monkeypatch.setenv('__PYVENV_LAUNCHER__','unrelated.exe')
    assert '__PYVENV_LAUNCHER__' not in command_environment({'environment_overrides':{}})
