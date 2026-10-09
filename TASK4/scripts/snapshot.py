"""Non-destructive, expected-current guarded source snapshot/rollback.
Only src/scripts/tests/configs and pyproject/requirements are managed. Reports,
state, models, user notes and KIKI are never deleted or restored by this tool.
"""
import _bootstrap
import argparse,hashlib,json,zipfile
from pathlib import Path
from task4.paths import ROOT,project_path

PREFIXES=('src/','scripts/','tests/','configs/')
TOP={'pyproject.toml','requirements.txt'}
def included(name):
    return (name.startswith(PREFIXES) or name in TOP) and '__pycache__' not in name and not name.endswith(('.pyc','.pyo'))
def current():
    return {str(p.relative_to(ROOT)).replace('\\','/'):p.read_bytes() for p in ROOT.rglob('*') if p.is_file() and included(str(p.relative_to(ROOT)).replace('\\','/'))}
def bundle(path):
    with zipfile.ZipFile(path) as z:
        out={}
        for name in z.namelist():
            if not included(name) or '..' in Path(name).parts or Path(name).is_absolute():raise ValueError('Invalid snapshot entry')
            project_path(name);out[name]=z.read(name)
        return out
p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
a=s.add_parser('create');a.add_argument('--output',required=True)
a=s.add_parser('restore');a.add_argument('--snapshot',required=True);a.add_argument('--expected-current',required=True)
a=p.parse_args()
if a.cmd=='create':
    path=project_path(a.output)
    if path.exists():raise FileExistsError('Snapshots are immutable')
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(current().items()):z.writestr(name,content)
    print(path)
else:
    old=bundle(project_path(a.snapshot));expected=bundle(project_path(a.expected_current));actual=current()
    if actual!=expected:raise RuntimeError('Working code differs from expected candidate snapshot; preserve/reconcile user edits before rollback')
    for name in set(expected)-set(old):project_path(name).unlink()
    for name,content in old.items():
        path=project_path(name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    print('Rolled back managed source only; evidence/model/state files unchanged.')
