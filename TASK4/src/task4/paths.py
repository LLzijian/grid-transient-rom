"""Explicit read-only source paths; every managed output stays in TASK4."""
from pathlib import Path
import json, os, tempfile
ROOT = Path(__file__).resolve().parents[2]

def project_path(value, root=ROOT):
    p = Path(value)
    p = (root/p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(root.resolve()):
        raise PermissionError(f"Output escapes TASK4: {p}")
    return p

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, data, root=ROOT):
    path=project_path(path, root);path.parent.mkdir(parents=True, exist_ok=True)
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    fd,name=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(text)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def source_root(config, root=ROOT):
    p=Path(os.environ.get("TASK4_KIKI_ROOT",config["kiki_root"]))
    p=(root/p).resolve() if not p.is_absolute() else p.resolve()
    if p==root.resolve() or p.is_relative_to(root.resolve()):
        raise ValueError("KIKI must be separate from TASK4")
    return p

def source_file(kiki, relative):
    p=(kiki/relative).resolve()
    if not p.is_relative_to(kiki.resolve()):raise PermissionError("Source path escapes KIKI")
    return p
