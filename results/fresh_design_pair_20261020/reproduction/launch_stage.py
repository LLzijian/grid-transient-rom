"""Local native envelope for the separately preregistered fresh 384-pool campaign."""
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
if BASE != (ROOT / "private_data/closure_mechanism_20261019/P3/fresh_design_pair01").resolve():
    raise ValueError("fixed first-physical-solve launcher location required")
CAPS = {"qa": 1000., "prepare": 180., "short": 1500., "score": 750.,
        "train": 3000., "schedule": 180., "adapt": 3000.,
        "evaluate_train": 3125., "freeze_val": 180., "val": 3000.,
        "evaluate_val": 1875., "summary": 180.}
CLASS = {"prepare": "qa", "schedule": "adapt", "evaluate_train": "evaluate_train",
         "freeze_val": "evaluate_val", "summary": "evaluate_val"}
CLASS_CAPS = {"qa":1000., "short":1500., "score":750., "train":3000.,
              "adapt":18000., "evaluate_train":3125., "val":3000., "evaluate_val":1875.}


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
    required = {Path(__file__).resolve().relative_to(ROOT).as_posix(),
        a.command.resolve().relative_to(ROOT).as_posix(), a.prior.resolve().relative_to(ROOT).as_posix(),
        "experiments/native_process_receipt_20261020.ps1",
        "experiments/fixed_edge_pair_20261019.py", "experiments/linear_cache_cost_gate_20261019.py",
        "experiments/fresh_design_pair_20261020.py", "tests/test_fresh_design_pair_20261020.py",
        "experiments/fom_execution_environment_20261020.py",
        "experiments/active_fom_pair_20261020.py", "experiments/active_fom_pair_fit_20261020.py",
        "experiments/closure_discovery_data.py", "configs/private_dataset.json",
        "reports/fresh_design_pair_20261020_protocol.md", "reports/astra6_after_first_physics_QA_20261020.md"}
    if not required <= set(binding["files"]):
        raise ValueError("required source/entry/command/protocol binding absent")
    def flag(name):
        if args.count(name) != 1 or args.index(name)+1 >= len(args):
            raise ValueError("one explicit value required: " + name)
        return args[args.index(name)+1]
    if args[:2] != ["-m", "experiments.fresh_design_pair_20261020"]:
        raise ValueError("registered wrapper entry required")
    if a.category in ("qa", "prepare"):
        mode = "qa" if a.category == "qa" else "prepare-binding"
        if args[2] != mode: raise ValueError("QA/prepare wrapper mode mismatch")
    elif a.category in ("short", "score", "train", "val"):
        if args[2] != "acquire" or flag("--stage") != a.category:
            raise ValueError("acquisition role/category mismatch")
        if Path(flag("--campaign")).resolve() != BASE / "campaign01":
            raise ValueError("fixed campaign differs")
    else:
        modes = {"schedule":"schedule", "adapt":"adapt", "evaluate_train":"evaluate",
                 "evaluate_val":"evaluate", "freeze_val":"freeze-val", "summary":"summary"}
        if args[2:4] != ["fit", modes[a.category]]:
            raise ValueError("fit wrapper category mismatch")
    if flag("--source-commit") != a.source_commit:
        raise ValueError("archived source identity differs")
    if a.worker.name != "wrapper_worker.json" or Path(flag("--wrapper-output")).resolve() != a.worker.parent.resolve():
        raise ValueError("early wrapper worker/output mismatch")
    if not Path(flag("--output")).resolve().is_relative_to(BASE):
        raise ValueError("scientific output outside fixed BASE")
    for name in ("--output", "--wrapper-output", "--campaign", "--source-manifest", "--freeze-manifest",
                 "--events-spec", "--source-output", "--source-identity", "--stage-identity", "--schedule-path",
                 "--model", "--frozen-predictions", "--manifest"):
        if name in args and not Path(flag(name)).is_absolute():
            raise ValueError("worker command must bind absolute paths: " + name)
    seconds = float(flag("--max-seconds"))
    if not 0 < seconds <= a.cap-60:
        raise ValueError("worker must self-stop at least 60s before envelope cap")



def prior_cost_gate(a, command):
    # Explicit prior native receipts only: never scan data folders or infer success.
    prior = json.loads(a.prior.read_text(encoding="utf-8"))
    totals = {k:0. for k in CLASS_CAPS}; seen = set(); model_total = 0.; counts = {}; prior_completed = {}
    current_args = command["arguments"]
    def model(args):
        return tuple(args[args.index(n)+1] for n in ("--seed","--structure")) if "--seed" in args and "--structure" in args else None
    key = model(current_args)
    for name in prior:
        path = Path(name).resolve()
        if not path.is_relative_to(BASE) or path in seen:
            raise ValueError("nonlocal/duplicate prior receipt")
        seen.add(path); r = json.loads(path.read_text(encoding="utf-8"))
        native = r.get("outer", {})
        seconds = (r.get("prelaunch_seconds") if r.get("cost_kind")=="prelaunch_tool_envelope"
                   and not native else native.get("seconds"))
        if not isinstance(seconds, (int,float)) or not 0 <= seconds < 40000 or r.get("status") == "RUNNING":
            raise ValueError("unreconciled prior native cost")
        category = CLASS.get(r["category"],r["category"])
        counts[r["category"]] = counts.get(r["category"],0)+1
        prior_completed[r["category"]] = prior_completed.get(r["category"],False) or r.get("status")=="COMPLETED"
        totals[category] += seconds
        if category == "adapt" and key is not None and model(r.get("worker_command",{}).get("arguments",[])) == key:
            model_total += seconds
    category = CLASS.get(a.category,a.category)
    if a.category in ("prepare","short","score","train","schedule","freeze_val","val","summary") and counts.get(a.category,0):
        raise ValueError("single-attempt scientific stage already attempted")
    if a.category == "qa" and (counts.get("qa",0)>=2 or prior_completed.get("qa")):
        raise ValueError("only one failed-QA repair/retest allowed; passed QA cannot repeat")
    if totals[category]+a.cap > CLASS_CAPS[category] or sum(totals.values())+3250+a.cap > 40000:
        raise ValueError("cumulative category/parent allocation exceeded")
    if a.category == "adapt" and model_total+a.cap > 3000:
        raise ValueError("cumulative model attempts exceed 3000")
    return {"prior_native_seconds":totals,"prior_model_seconds":model_total,
            "support_hold_seconds":3250,"new_cap_seconds":a.cap}


def main():
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
    executable = (Path("E:/03-nn_Transfer_Func/kron-reduce/.venv/Scripts/python.exe")
                  if a.category in ("short", "train", "val") else ROOT / ".venv-gpu/Scripts/python.exe").resolve()
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


if __name__ == "__main__":
    raise SystemExit(main())
