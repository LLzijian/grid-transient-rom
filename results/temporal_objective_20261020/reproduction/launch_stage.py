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
if BASE != (ROOT / "private_data/closure_mechanism_20261019/P2/temporal_objective01").resolve():
    raise ValueError("fixed first-physical-solve launcher location required")
CAPS = {"qa":1200., "prepare":180., "cost":2400., "train":4000., "evaluate":400.}
CLASS = {"prepare":"qa", "cost":"qa"}
CLASS_CAPS = {"qa":4200., "train":30000., "evaluate":4000.}



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
    if not binding.get("files"):
        raise ValueError("empty explicit source binding")
    actual = {}
    for name, expected in binding["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.suffix.lower() in (".h5", ".hdf5"):
            raise ValueError("forbidden source binding")
        actual[name] = sha(path)
        if actual[name] != expected:
            raise ValueError("source/model/allowed-input changed: " + name)
    return actual


def command_gate(a, command, binding):
    args = command["arguments"]
    required = {(BASE/"implementation_cost.json").relative_to(ROOT).as_posix(), Path(__file__).resolve().relative_to(ROOT).as_posix(),
        a.command.relative_to(ROOT).as_posix(), a.prior.relative_to(ROOT).as_posix(),
        "experiments/native_process_receipt_20261020.ps1",
        "experiments/temporal_objective_runtime_20261020.py",
        "experiments/temporal_objective_20261020.py",
        "tests/test_temporal_objective_runtime_20261020.py",
        "tests/test_temporal_objective_20261020.py",
        "tests/test_temporal_native_envelope_20261020.py",
        "reports/temporal_objective_20261020_protocol.md",
        "reports/astra6_after_quadratic_memory_20261020.md"}
    if not required <= set(binding["files"]):
        raise ValueError("required source/entry/command/protocol binding absent")
    def flag(name):
        if args.count(name)!=1 or args.index(name)+1>=len(args): raise ValueError("one explicit value required: "+name)
        return args[args.index(name)+1]
    expected = (["-m", "experiments.temporal_objective_20261020", a.category]
                if a.category!="qa" else [str(BASE/"qa_driver.py")])
    if args[:len(expected)]!=expected: raise ValueError("registered category entry required")
    if a.category=="qa" and (BASE/"qa_driver.py").relative_to(ROOT).as_posix() not in binding["files"]:
        raise ValueError("QA driver must be bound")
    if flag("--source-commit")!=a.source_commit: raise ValueError("source commit differs")
    output=Path(flag("--output")).resolve()
    if not Path(flag("--output")).is_absolute() or not output.is_relative_to(BASE): raise ValueError("absolute current output required")
    worker_output=Path(flag("--worker-output")).resolve() if "--worker-output" in args else output
    if not worker_output.is_relative_to(BASE) or a.worker!=worker_output/"worker.json": raise ValueError("worker identity path differs")
    seconds=float(flag("--max-seconds"))
    if not 0<seconds<=a.cap-60: raise ValueError("self-stop requires 60 seconds margin")


def prior_cost_gate(a, command):
    prior=json.loads(a.prior.read_text(encoding="utf-8"))
    totals={k:0. for k in CLASS_CAPS}; seen=set(); model_total=0.; attempts=[]
    preparation=BASE/"implementation_cost.json"
    preparation_seconds=0.
    if preparation.is_file():
        preparation_seconds=json.loads(preparation.read_text(encoding="utf-8"))["qa_seconds"]
        if type(preparation_seconds) not in (int,float) or not 0<=preparation_seconds<=4200:
            raise ValueError("invalid preparation cost")
        totals["qa"]+=preparation_seconds
    def flag(args,n): return args[args.index(n)+1] if n in args else None
    def model(args): return (flag(args,"--seed"),flag(args,"--variant"))
    key=model(command["arguments"]); recoveries=0
    for name in prior:
        path=Path(name).resolve()
        if not path.is_relative_to(BASE) or path in seen: raise ValueError("nonlocal/duplicate prior receipt")
        seen.add(path); r=json.loads(path.read_text(encoding="utf-8"))
        seconds=r.get("charged_seconds")
        if not isinstance(seconds,(int,float)) or not 0<=seconds<36000 or r.get("status")=="RUNNING": raise ValueError("unreconciled prior cost")
        inner=r.get("inner",{}); outer=r.get("outer",{})
        if (inner.get("cleanup_verified") is not True or inner.get("fixture_identity_after")!="GONE"
                or inner.get("native_receipt_complete") is not True or outer.get("native_receipt_complete") is not True):
            raise ValueError("prior native ownership/cleanup unresolved: actual PID+creation reconciliation required")
        category=CLASS.get(r["category"],r["category"]); totals[category]+=seconds
        args=r.get("worker_command",{}).get("arguments",[])
        recoveries += int("--external-recovery" in args)
        if r["category"]==a.category: attempts.append((r,args))
        if category=="train" and model(args)==key: model_total+=seconds
    recorded={p.resolve() for p in BASE.glob("native_*/execution.json")}
    launched={p.parent.resolve() for p in BASE.glob("native_*/launch.json")}
    if recorded!=seen or launched!={p.parent for p in recorded}:
        raise ValueError("prior receipts incomplete or an earlier native attempt remains unresolved")
    category=CLASS.get(a.category,a.category)
    if totals[category]+a.cap>CLASS_CAPS[category] or sum(totals.values())+3800+a.cap>42000: raise ValueError("cumulative category/stage cap exceeded")
    if a.category=="qa" and (len(attempts)>=2 or any(r["status"]=="COMPLETED" for r,_ in attempts)): raise ValueError("one failed QA repair only")
    if a.category in ("prepare","cost") and attempts: raise ValueError("stage single attempt already consumed")
    if a.category=="train":
        if key[0] not in ("6101","6102","6103") or key[1] not in ("O","F","T"):raise ValueError("unregistered temporal training identity")
        if flag(command["arguments"],"--target-step")!="512":raise ValueError("only512 temporal endpoint")
        if model_total+a.cap>(2000 if key[1]=="O" else 4000): raise ValueError("model cumulative group cap exceeded")
        args=command["arguments"]
        if "--external-recovery" in args:
            if recoveries or a.cap>1200: raise ValueError("single nested external recovery cap exceeded")
        elif any(model(old)==key and flag(old,"--target-step")==flag(args,"--target-step") for _,old in attempts):
            raise ValueError("model endpoint already attempted")
    if a.category=="evaluate":
        target=flag(command["arguments"],"--model")
        if any(flag(old,"--model")==target for _,old in attempts): raise ValueError("endpoint evaluation already attempted")
    return {"prior_native_seconds":totals,"prior_model_seconds":model_total,"preparation_qa_seconds":preparation_seconds,"support_hold_seconds":3800,"new_cap_seconds":a.cap}


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
