"""Single registered, no-solver environment QA envelope."""
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sources(binding):
    required = {"experiments/fom_execution_environment_20261020.py",
                "tests/test_fom_execution_environment_20261020.py",
                "experiments/active_fom_pair_20261020.py",
                "experiments/closure_discovery_data.py",
                "configs/private_dataset.json",
                "experiments/native_process_receipt_20261020.ps1",
                "experiments/fixed_edge_pair_20261019.py",
                "experiments/linear_cache_cost_gate_20261019.py",
                "reports/fom_environment_repair_20261020_protocol.md",
                "reports/astra6_fom_environment_repair_20261020.md",
                "private_data/closure_mechanism_20261019/repair/fom_environment01/launch_qa.py"}
    if not required <= binding.keys():
        raise ValueError("required source dependency absent")
    actual = {}
    for name, expected in binding.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.suffix.lower() in (".h5", ".hdf5", ".pt", ".npz"):
            raise ValueError("engineering source list cannot read trajectories/weights/external files")
        actual[name] = sha(path)
        if actual[name] != expected:
            raise ValueError("frozen engineering source changed: " + name)
    return actual


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-commit", required=True)
    p.add_argument("--attempt", choices=("qa01", "qa02"), default="qa01")
    p.add_argument("--cap", type=float, default=450.)
    a = p.parse_args()
    if len(a.source_commit) != 40 or any(c not in "0123456789abcdef" for c in a.source_commit):
        p.error("archived source commit required")
    prior = BASE / "native_qa01/execution.json"
    previous = 0.
    if a.attempt == "qa02":
        old = json.loads(prior.read_text(encoding="utf-8"))
        previous = old["outer"]["seconds"]
        if old.get("status") == "COMPLETED":
            raise ValueError("passed QA must not be repeated")
    if not 90 <= a.cap <= min(450., 600.-previous):
        p.error("cumulative QA execution cap exceeded")
    out = BASE / ("native_" + a.attempt)
    if out.exists():
        raise FileExistsError(out)
    command_path = BASE / "commands" / (a.attempt + ".json")
    binding_path = BASE / ("binding_" + a.attempt + ".json")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))["source"]
    if command_path.relative_to(ROOT).as_posix() not in binding:
        raise ValueError("actual command is not bound")
    before = sources(binding)
    command = json.loads(command_path.read_text(encoding="utf-8"))
    expected = ["-m", "experiments.fom_execution_environment_20261020", "qa",
                "--output", str(BASE / a.attempt), "--max-seconds", str(int(a.cap-90))]
    if (Path(command["executable"]).resolve() != (ROOT / ".venv-gpu/Scripts/python.exe").resolve()
            or command["arguments"] != expected or command["arguments_line"] != subprocess.list2cmdline(expected)):
        raise ValueError("registered engineering QA command changed")
    out.mkdir(exist_ok=False)
    worker = BASE / a.attempt / "worker.json"
    ps = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    argv = [str(ps), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(ROOT / "experiments/native_process_receipt_20261020.ps1"),
            "-CommandJson", str(command_path), "-WorkingDirectory", str(ROOT),
            "-Output", str(out / "inner"), "-TimeoutSeconds", str(a.cap-30),
            "-MaxSeconds", str(a.cap), "-SourceCommit", a.source_commit,
            "-FixtureIdentityJson", str(worker)]
    sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
    from fixed_edge_pair_20261019 import native_job
    record = {"status":"UNRESOLVED", "source_commit":a.source_commit, "source_before":before,
              "source_binding_sha256":sha(binding_path), "command_sha256":sha(command_path),
              "previous_native_seconds":previous, "support_hold_seconds":600}
    try:
        outer = native_job(argv, out / "outer", a.cap, worker_json=worker)
        record["outer"] = outer
        inner = json.loads((out / "inner/receipt.json").read_text(encoding="utf-8"))
        record["inner"] = inner
        identity = json.loads(worker.read_text(encoding="utf-8"))
        record["worker"] = identity
        record["source_after"] = sources(binding)
        result = json.loads((BASE / a.attempt / "final.json").read_text(encoding="utf-8"))
        record["qa"] = result
        valid = (outer.get("status") == "COMPLETED" and outer.get("exit_code") == 0
                 and outer.get("native_receipt_complete") is True
                 and inner.get("status") == "COMPLETED" and inner.get("exit_code") == 0
                 and inner.get("native_receipt_complete") is True and inner.get("cleanup_verified") is True
                 and inner.get("fixture_identity_after") == "GONE" and inner.get("timeout") is False
                 and outer.get("actual_worker_pid") == identity.get("pid") == inner.get("fixture_pid")
                 and identity.get("creation_filetime_utc") == inner.get("fixture_creation_filetime_utc")
                 and result.get("status") == "PASS")
        record["status"] = "COMPLETED" if valid else "UNRESOLVED"
    except BaseException as exc:
        record["error"] = repr(exc)
        try:
            record["source_after"] = sources(binding)
        except BaseException as other:
            record["source_after_error"] = repr(other)
    finally:
        record["charged_seconds"] = record.get("outer", {}).get("seconds")
        write_new(out / "execution.json", record)
    print(json.dumps({"status":record["status"], "charged_seconds":record["charged_seconds"],
                      "receipt":str(out / "execution.json")}))
    return 0 if record["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
