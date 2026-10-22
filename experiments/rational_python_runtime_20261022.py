"""Small, explicit Python command builder for the Windows runtime reconstruction."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import hashlib

ROOT=Path(__file__).resolve().parents[1]


SCHEMES = {"direct_base_venv", "venv_pythonw"}


def _absolute(value: str | os.PathLike[str]) -> str:
    return str(Path(value).expanduser().resolve())


def python_command(logical_executable, arguments, scheme):
    logical = _absolute(logical_executable)
    args = [str(item) for item in arguments]
    if scheme not in SCHEMES:
        raise ValueError(f"unsupported runtime scheme: {scheme}")
    cfg = Path(logical).parent.parent / "pyvenv.cfg"
    if not cfg.is_file():raise ValueError(f"venv configuration not found: {cfg}")
    home = None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("home ="):
            home = line.split("=", 1)[1].strip();break
    if not home:raise ValueError(f"venv base home missing: {cfg}")
    if scheme == "direct_base_venv":
        logical=_absolute(Path(logical).with_name('python.exe'))
        executable = _absolute(Path(home) / "python.exe")
        overrides = {"__PYVENV_LAUNCHER__": logical}
        expected_worker_image=executable
    else:
        logical=_absolute(Path(logical).with_name('pythonw.exe'))
        executable = logical
        expected_worker_image=_absolute(Path(home)/'pythonw.exe')
        overrides = {}
    return {
        "executable": executable,
        "logical_executable": logical,
        "arguments": args,
        "environment_overrides": overrides,
        "runtime_scheme": scheme,
        "expected_worker_image": expected_worker_image,
    }


def command_environment(command):
    env = os.environ.copy()
    overrides = command.get("environment_overrides", {})
    if set(overrides) - {"__PYVENV_LAUNCHER__"}:
        raise ValueError("unsupported environment override")
    env.pop('__PYVENV_LAUNCHER__',None)
    env.update({str(k): str(v) for k, v in overrides.items()})
    return env


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve(value):return (ROOT/value).resolve()


def validate_selection(selection_path):
    """Re-read raw receipts and calculations; an asserted SELECTED is insufficient."""
    path=Path(selection_path).resolve();selection=json.loads(path.read_text(encoding='utf-8'))
    if selection.get('status')!='SELECTED' or selection.get('scheme') not in SCHEMES:
        raise ValueError('runtime selection is not qualified')
    commit=selection.get('observation_source_commit','')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('full observation source required')
    if type(selection.get('observation_attempt')) is not int or selection['observation_attempt'] not in (1,2):
        raise ValueError('registered observation attempt required')
    files={_resolve(name):digest for name,digest in selection.get('files',{}).items()}
    if not files:raise ValueError('raw runtime files required')
    for target,digest in files.items():
        if not target.is_file() or _sha(target)!=digest:raise ValueError('runtime file changed: '+str(target))
    required_sources={ROOT/'experiments/rational_python_runtime_20261022.py',ROOT/'experiments/uncapped_windows_job_20261021.py'}
    python_sources={target for target in files if target.suffix.lower()=='.py'}
    if not required_sources.issubset(python_sources) or len(python_sources)<3:
        raise ValueError('runtime/helper/probe source hashes required')
    def read_ref(ref):
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('exact raw reference required')
        target=_resolve(ref['path'])
        if not target.is_file() or files.get(target)!=ref['sha256'] or _sha(target)!=ref['sha256']:
            raise ValueError('source-bound raw reference required')
        return json.loads(target.read_text(encoding='utf-8'))
    command=python_command(selection['logical_executable'],[],selection['scheme'])
    for key in ('logical_executable','executable','expected_worker_image'):
        if _absolute(selection.get(key,''))!=command[key]:raise ValueError('selected runtime image differs: '+key)
        if _resolve(command[key]) not in files:raise ValueError('selected interpreter hash required')
    cfg=Path(command['logical_executable']).parent.parent/'pyvenv.cfg'
    if cfg.resolve() not in files:raise ValueError('selected venv configuration hash required')
    proofs={key:read_ref(selection.get(key)) for key in ('normal_receipt','orphan_receipt','baseline_reference','parity_evidence')}
    for key in ('normal_receipt','orphan_receipt'):
        proof=proofs[key];negative=key=='orphan_receipt';worker=proof.get('worker',{});raw_command=proof.get('command',{})
        if (proof.get('source_commit')!=commit or proof.get('status')!=('FAILED' if negative else 'COMPLETED')
                or proof.get('cleanup_verified') is not True or proof.get('orphans_cleaned') is not negative
                or type(proof.get('job_active_processes')) is not int or proof['job_active_processes']!=0
                or type(proof.get('job_total_processes')) is not int or proof['job_total_processes']<2
                or proof.get('actual_worker_state')!='GONE'
                or any(type(proof.get(k)) is not int or proof[k]!=0 for k in ('exit_code','launcher_exit_code','worker_exit_code'))
                or not proof.get('source_before') or proof['source_before']!=proof.get('source_after')
                or type(worker.get('pid')) is not int or worker['pid']<=0
                or type(worker.get('creation_filetime_utc')) is not int or worker['creation_filetime_utc']<=0
                or _absolute(worker.get('physical_image',worker.get('executable','')))!=command['expected_worker_image']
                or any(raw_command.get(k)!=v for k,v in command.items() if k!='arguments')):
            raise ValueError('actual native normal/orphan proof differs: '+key)
        for target,digest in files.items():
            if target.suffix.lower()=='.py' and proof['source_before'].get(str(target))!=digest:
                raise ValueError('observation source bytes differ: '+str(target))
    parity=proofs['parity_evidence']
    baseline=read_ref(parity.get('baseline'));candidate=read_ref(parity.get('candidate'))
    if parity.get('baseline')!=selection.get('baseline_reference') or baseline!=proofs['baseline_reference']:
        raise ValueError('parity baseline differs')
    required=('python_version','python_dll','prefix','base_prefix','package_paths','torch_version','torch_file',
        'cuda_version','cuda_device','deterministic_algorithms','cudnn_deterministic','cudnn_benchmark',
        'num_threads','interop_threads','tensor14')
    if any(key not in baseline or key not in candidate
            or json.dumps(baseline[key],sort_keys=True,allow_nan=False)!=json.dumps(candidate[key],sort_keys=True,allow_nan=False)
            for key in required):
        raise ValueError('actual computation/runtime parity differs')
    for field,command_key in (('logical_executable','logical_executable'),('physical_executable','expected_worker_image')):
        if field not in baseline or field not in candidate or _absolute(candidate[field])!=command[command_key]:
            raise ValueError('candidate metadata process identity differs')
        expected=Path(command[command_key])
        if selection['scheme']=='venv_pythonw':expected=expected.with_name('python.exe')
        if _absolute(baseline[field])!=str(expected.resolve()):raise ValueError('baseline metadata process identity differs')
    dll=baseline.get('python_dll',{})
    if not isinstance(dll,dict) or files.get(_resolve(dll.get('path','')))!=dll.get('sha256'):
        raise ValueError('actual Python DLL hash required')
    if (not isinstance(candidate['tensor14'],list) or len(candidate['tensor14'])!=14
            or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not __import__('math').isfinite(v) for v in candidate['tensor14'])):
        raise ValueError('actual fixed tensor14 result required')
    child=read_ref(parity.get('child_identity'))
    if (type(child.get('pid')) is not int or child['pid']<=0
            or type(child.get('creation_filetime_utc')) is not int or child['creation_filetime_utc']<=0
            or child['pid']==proofs['orphan_receipt']['worker']['pid']):raise ValueError('actual handshaken orphan identity required')
    if __package__:
        from .uncapped_windows_job_20261021 import identity_state
    else:
        from uncapped_windows_job_20261021 import identity_state
    if identity_state(child)!='GONE':raise ValueError('actual handshaken orphan is not GONE')
    return selection


def selected_python_command(arguments, selection_path):
    path = Path(selection_path).resolve()
    selection = validate_selection(path)
    if selection.get("status") != "SELECTED" or selection.get("scheme") not in SCHEMES:
        raise ValueError("runtime selection is not qualified")
    command = python_command(selection["logical_executable"], arguments, selection["scheme"])
    if _absolute(selection["executable"]) != command["executable"]:
        raise ValueError("selected physical executable does not match scheme")
    if _absolute(selection["expected_worker_image"]) != command["expected_worker_image"]:
        raise ValueError("selected worker image does not match scheme")
    command["runtime_selection"] = {"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}
    return command
