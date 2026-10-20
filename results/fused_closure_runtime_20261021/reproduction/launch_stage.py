"""Fixed native envelope for the preregistered quadratic memory edge study."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
if BASE != (ROOT / "private_data/closure_mechanism_20261019/P0/fused_runtime01").resolve():
    raise ValueError("fixed first-physical-solve launcher location required")
CAPS = {"prepare":100.,"qa":400.,"benchmark":1800.}
CLASS = {}
CLASS_CAPS = {"prepare":100.,"qa":800.,"benchmark":1800.}



def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_binding(binding):
    if not binding.get('files'):raise ValueError('explicit audit source binding required')
    actual={}
    for name,h in binding['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or path.suffix.lower() in ('.h5','.hdf5') or name.startswith('configs/') or (name.startswith('private_data/') and not path.is_relative_to(BASE)):raise ValueError('no scientific assets')
        actual[name]=sha(path)
        if actual[name]!=h:raise ValueError('source/fixture changed:'+name)
    return actual



def ps_limits(timeout,maximum):
    if not 0<timeout<maximum<=3600:raise ValueError('actual PS time limit violated')
    return True

def native_limits(category,cap,worker):
    expected={'prepare':(100.,40.),'qa':(400.,340.),'benchmark':(1800.,1740.)}
    if expected.get(category)!=(cap,worker):raise ValueError('fixed audit native limits required')
    ps_limits(cap-30,cap)
    if not 0<worker<cap-30:raise ValueError('cleanup margin missing')
    if '$MaxSeconds -gt 3600' not in (ROOT/'experiments/native_process_receipt_20261020.ps1').read_text(encoding='utf-8-sig'):raise ValueError('actual native PS cap changed')
    return {'max_seconds':cap,'timeout_seconds':cap-30,'worker_seconds':worker}


def execution_contract():
    for mode,cap,selfcap in [('prepare',100.,40.),('qa',400.,340.),('benchmark',1800.,1740.)]:native_limits(mode,cap,selfcap)
    if 100+100+800+1800+200+600!=3600:raise ValueError('whole finite audit budget inconsistent')
    return True


def command_gate(a,command,binding):
    args=command['arguments'];execution_contract()
    def flag(n):
        if args.count(n)!=1 or args.index(n)+1>=len(args):raise ValueError('unique explicit flag required:'+n)
        return args[args.index(n)+1]
    expected=([str(BASE/'qa_driver.py')] if a.category=='qa' else ['-m','experiments.fused_closure_audit_20261021',a.category])
    if args[:len(expected)]!=expected or flag('--source-commit')!=a.source_commit:raise ValueError('exact frozen audit entry required')
    native_limits(a.category,a.cap,float(flag('--max-seconds')))
    output=Path(flag('--output'));allowed={'prepare':{'prepare01'},'benchmark':{'benchmark01'},'qa':{'qa01','qa02'}}
    if not output.is_absolute() or output.parent!=BASE or output.name not in allowed[a.category] or a.worker!=output/'worker.json':raise ValueError('exclusive current-stage output/worker required')
    required={(BASE/'implementation_cost.json').relative_to(ROOT).as_posix(),a.command.relative_to(ROOT).as_posix(),a.prior.relative_to(ROOT).as_posix(),Path(__file__).relative_to(ROOT).as_posix(),'experiments/fused_closure_audit_20261021.py','experiments/fused_closure_runtime_20261021.py','reports/fused_closure_runtime_20261021_protocol.md','reports/astra6_after_cubic_cost_20261021.md','experiments/native_process_receipt_20261020.ps1'}
    if not required<=set(binding['files']):raise ValueError('required source/command/cost binding absent')
    if output.name=='qa02' and (BASE/'qa02_authorization.json').relative_to(ROOT).as_posix() not in binding['files']:raise ValueError('root repair authorization must be bound')
    if a.category!='prepare':
        for name in ('manifest.json','model_N.json','model_D.json','synthetic.npz','rng.pt'):
            if (BASE/'prepare01'/name).relative_to(ROOT).as_posix() not in binding['files']:raise ValueError('actual frozen fixture missing')



def prior_cost_gate(a,command):
    prior=json.loads(a.prior.read_text());totals={k:0. for k in CLASS_CAPS};seen=set();attempts=[]
    prep_path=BASE/'implementation_cost.json';static=0.
    if prep_path.is_file():static=json.loads(prep_path.read_text())['static_seconds']
    if type(static) not in (int,float) or not 0<=static<=100:raise ValueError('static preparation cap exceeded')
    for name in prior:
        path=Path(name).resolve()
        if not path.is_relative_to(BASE) or path in seen:raise ValueError('foreign/duplicate prior')
        seen.add(path);r=json.loads(path.read_text());inner=r.get('inner',{});outer=r.get('outer',{})
        if r.get('status')=='RUNNING' or r.get('category') not in totals or type(r.get('charged_seconds')) not in (int,float) or not 0<=r['charged_seconds']<36000:raise ValueError('unreconciled prior cost')
        if inner.get('cleanup_verified') is not True or inner.get('fixture_identity_after')!='GONE' or inner.get('native_receipt_complete') is not True or outer.get('native_receipt_complete') is not True:raise ValueError('prior ownership cleanup unresolved')
        totals[r['category']]+=r['charged_seconds']
        if r['category']==a.category:attempts.append(r)
    recorded={p.resolve() for p in BASE.glob('native_*/execution.json')};launched={p.parent.resolve() for p in BASE.glob('native_*/launch.json')}
    if recorded!=seen or launched!={p.parent for p in seen}:raise ValueError('prior receipts incomplete')
    if totals[a.category]+a.cap>CLASS_CAPS[a.category] or sum(totals.values())+static+200+600+a.cap>3600:raise ValueError('cumulative audit cap exceeded')
    if a.category in ('prepare','benchmark') and attempts:raise ValueError('single attempt already used')
    output=Path(command['arguments'][command['arguments'].index('--output')+1])
    if a.category in ('qa','benchmark'):
        prepared=[json.loads(p.read_text()) for p in seen if json.loads(p.read_text())['category']=='prepare']
        if len(prepared)!=1 or prepared[0]['status']!='COMPLETED':raise ValueError('unique qualified asset preparation required')
    if a.category=='qa':
        if output.name=='qa01' and attempts:raise ValueError('QA01 already used')
        if output.name=='qa02':
            auth=BASE/'qa02_authorization.json'
            if len(attempts)!=1 or attempts[0]['status']=='COMPLETED' or not auth.is_file():raise ValueError('root repair QA02 authorization required')
            value=json.loads(auth.read_text())
            if value.get('root_release') is not True or value.get('source_commit')!=a.source_commit:raise ValueError('root repair QA02 identity mismatch')
        if len(attempts)>=2:raise ValueError('QA attempts exhausted')
    if a.category=='benchmark':
        qas=[json.loads(p.read_text()) for p in seen if json.loads(p.read_text())['category']=='qa']
        if not any(r['status']=='COMPLETED' and r.get('source_commit')==a.source_commit for r in qas):raise ValueError('QA pass before benchmark required')
    return {'prior_native_seconds':totals,'static_seconds':static,'support_hold_seconds':600,'metadata_reserve_seconds':200,'new_cap_seconds':a.cap}



def main_inner():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--command", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--category", choices=tuple(CAPS), required=True)
    p.add_argument("--cap", type=float, required=True)
    a = p.parse_args()
    # Validate and forward exactly the same canonical values, including when the
    # operator supplies a relative CLI path. Native PS remains strict.
    for name in ("command", "output", "worker", "binding", "prior"):
        setattr(a, name, getattr(a, name).resolve())
    if not 40 == len(a.source_commit) or any(c not in "0123456789abcdef" for c in a.source_commit):
        p.error("archived full source commit required")
    if not 60 < a.cap <= CAPS[a.category]:
        p.error("registered category cap exceeded")
    for path in (a.output, a.worker, a.command, a.binding, a.prior):
        if not path.resolve().is_relative_to(BASE):
            p.error("explicit campaign-local paths required")
    if a.output.exists():
        p.error("exclusive native attempt directory required")
    binding = json.loads(a.binding.read_text(encoding="utf-8"))
    before = verify_binding(binding)
    command = json.loads(a.command.read_text(encoding="utf-8"))
    executable = (ROOT / ".venv-gpu/Scripts/python.exe").resolve()
    if (Path(command["executable"]).resolve() != executable or
            command["arguments_line"] != subprocess.list2cmdline(command["arguments"])):
        raise ValueError("registered executable/argument identity changed")
    command_gate(a, command, binding)
    budget = prior_cost_gate(a, command)
    a.output.mkdir(parents=True, exist_ok=False)
    ps = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    native = [str(ps), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
              "-File", str(ROOT / "experiments/native_process_receipt_20261020.ps1"),
              "-CommandJson", str(a.command.resolve()), "-WorkingDirectory", str(ROOT),
              "-Output", str(a.output / "inner"), "-TimeoutSeconds", str(a.cap-30),
              "-MaxSeconds", str(a.cap), "-SourceCommit", a.source_commit,
              "-FixtureIdentityJson", str(a.worker.resolve())]
    sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
    from fixed_edge_pair_20261019 import native_job
    started = time.perf_counter()
    record = {"category": a.category, "status": "RUNNING", "source_commit": a.source_commit,
              "source_before": before, "binding_sha256": sha(a.binding),
              "command_sha256": sha(a.command), "worker_command": command, "category_envelope_cap": a.cap,
              "worker_identity_path": str(a.worker), "includes_children_once": True}
    record["budget_gate"] = budget
    write_new(a.output / "launch.json", record)
    try:
        outer = native_job(native, a.output / "outer", a.cap, worker_json=a.worker)
        record["outer"] = outer
        record["charged_seconds"] = outer["seconds"]
        inner_path = a.output / "inner/receipt.json"
        inner = json.loads(inner_path.read_text(encoding="utf-8")) if inner_path.is_file() else {}
        record["inner"] = inner
        worker = json.loads(a.worker.read_text(encoding="utf-8")) if a.worker.is_file() else {}
        record["worker"] = worker
        record["source_after"] = verify_binding(binding)
        valid = (outer.get("status") == "COMPLETED" and outer.get("exit_code") == 0
                 and outer.get("native_receipt_complete") is True
                 and inner.get("status") == "COMPLETED" and inner.get("exit_code") == 0
                 and inner.get("native_receipt_complete") is True and inner.get("cleanup_verified") is True
                 and inner.get("fixture_identity_after") == "GONE" and inner.get("timeout") is False
                 and worker.get("pid") == outer.get("actual_worker_pid") == inner.get("fixture_pid")
                 and worker.get("creation_filetime_utc") == inner.get("fixture_creation_filetime_utc")
                 and isinstance(worker.get("creation_filetime_utc"), int)
                 and Path(worker.get("executable", "")).resolve() == executable)
        record["status"] = "COMPLETED" if valid else "UNRESOLVED_EXECUTION"
    except BaseException as exc:
        record.update(status="UNRESOLVED_EXECUTION", error=repr(exc))
    finally:
        record["launcher_seconds"] = time.perf_counter()-started
        write_new(a.output / "execution.json", record)
    print(json.dumps({"status": record["status"], "category": a.category,
                      "charged_seconds": record.get("charged_seconds"),
                      "receipt": str(a.output / "execution.json")}))
    return 0 if record["status"] == "COMPLETED" else 1


def main():
    # A single live native attempt per fixed study; a crash leaves this claim
    # for explicit PID+creation-time reconciliation, never automatic deletion.
    from closure_p3_disagreement_20261020 import own_creation_filetime_utc
    identity={"pid":os.getpid(),"creation_filetime_utc":own_creation_filetime_utc()}
    BASE.mkdir(parents=True,exist_ok=True)
    lock=BASE/"native_active.lock"
    with lock.open("x",encoding="utf-8") as stream: json.dump(identity,stream)
    try:
        return main_inner()
    finally:
        if json.loads(lock.read_text(encoding="utf-8"))!=identity:
            raise RuntimeError("native ownership claim changed")
        lock.unlink()


if __name__ == "__main__":
    sys.path[:0]=[str(ROOT/"experiments"),str(ROOT/"src")]
    raise SystemExit(main())
