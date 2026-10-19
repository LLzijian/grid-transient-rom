"""One bounded, real Windows PowerShell acceptance run for the native receipt interface."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


KERNEL = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 258
if KERNEL is not None:
    KERNEL.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    KERNEL.OpenProcess.restype = ctypes.c_void_p
    KERNEL.GetProcessTimes.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p)
    KERNEL.GetProcessTimes.restype = ctypes.c_int
    KERNEL.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    KERNEL.WaitForSingleObject.restype = ctypes.c_ulong
    KERNEL.CloseHandle.argtypes = (ctypes.c_void_p,)
    KERNEL.CloseHandle.restype = ctypes.c_int


def creation_filetime(pid):
    handle = KERNEL.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess({pid})")
    try:
        created = ctypes.c_ulonglong()
        exited = ctypes.c_ulonglong()
        kernel = ctypes.c_ulonglong()
        user = ctypes.c_ulonglong()
        if not KERNEL.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(kernel), ctypes.byref(user)):
            raise OSError(ctypes.get_last_error(), f"GetProcessTimes({pid})")
        return created.value
    finally:
        KERNEL.CloseHandle(handle)


def identity_state(record):
    pid = int(record["pid"])
    handle = KERNEL.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return "GONE"
        return f"CHECK_ERROR:{error}"
    try:
        created = ctypes.c_ulonglong()
        exited = ctypes.c_ulonglong()
        kernel = ctypes.c_ulonglong()
        user = ctypes.c_ulonglong()
        if not KERNEL.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(kernel), ctypes.byref(user)):
            return f"CHECK_ERROR:{ctypes.get_last_error()}"
        if created.value != int(record["creation_filetime_utc"]):
            return "PID_REUSED"
        return "GONE" if KERNEL.WaitForSingleObject(handle, 0) != WAIT_TIMEOUT else "ALIVE"
    finally:
        KERNEL.CloseHandle(handle)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def fixture_main(args):
    identity = {"pid": os.getpid(), "creation_filetime_utc": creation_filetime(os.getpid())}
    write_json(Path(args.identity), identity)
    if args.worker == "child":
        while True:
            time.sleep(60)
    print("FIXTURE_STDOUT", flush=True)
    print("FIXTURE_STDERR", file=sys.stderr, flush=True)
    if args.worker == "timeout_tree":
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "child",
                                  "--identity", args.child_identity], stdin=subprocess.DEVNULL)
        while not Path(args.child_identity).is_file():
            if child.poll() is not None:
                raise RuntimeError("fixture child exited before identity publication")
            time.sleep(0.01)
        while True:
            time.sleep(60)
    return 0 if args.worker == "quick_exit_0" else 7


def taskkill_owned(record, seconds=10):
    state = identity_state(record)
    if state != "ALIVE":
        return {"before": state, "taskkill_exit_code": None}
    result = subprocess.run([str(Path(os.environ["SystemRoot"]) / "System32" / "taskkill.exe"),
                             "/PID", str(record["pid"]), "/T", "/F"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=seconds)
    return {"before": state, "taskkill_exit_code": result.returncode,
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace")}


def main(args):
    if os.name != "nt":
        raise RuntimeError("Windows native process QA only")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Exclusive QA output already exists: {output}")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.max_seconds > 300:
        raise ValueError("max-seconds must be in (0, 300]")
    output.mkdir(parents=True)
    write_json(output / "worker.json", {"pid": os.getpid(),
                                         "creation_filetime_utc": creation_filetime(os.getpid())})
    start = time.perf_counter()
    qa = {"status": "FAILED", "source_commit": args.source_commit, "cases": [],
          "max_seconds": args.max_seconds, "python_launcher": sys.executable,
          "powershell_executable": str(Path(os.environ["SystemRoot"]) / "System32" /
                                       "WindowsPowerShell" / "v1.0" / "powershell.exe")}
    script = Path(__file__).resolve().parents[1] / "experiments" / "native_process_receipt_20261020.ps1"
    try:
        for name in ("quick_exit_0", "quick_exit_7", "timeout_tree"):
            case = output / name
            case.mkdir()
            worker_identity = case / "worker.json"
            child_identity = case / "child.json"
            command = {"executable": sys.executable,
                       "arguments": [str(Path(__file__).resolve()), "--worker", name,
                                     "--identity", str(worker_identity),
                                     "--child-identity", str(child_identity)]}
            command["arguments_line"] = subprocess.list2cmdline(command["arguments"])
            write_json(case / "command.json", command)
            cmd = [qa["powershell_executable"], "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                   "-File", str(script), "-CommandJson", str(case / "command.json"),
                   "-WorkingDirectory", str(case), "-Output", str(case / "native"),
                   "-TimeoutSeconds", "10" if name == "timeout_tree" else "20",
                   "-MaxSeconds", "29", "-SourceCommit", args.source_commit,
                   "-FixtureIdentityJson", str(worker_identity)]
            if name == "timeout_tree":
                cmd.extend(["-OwnedChildIdentityJson", str(child_identity)])
            case_start = time.perf_counter()
            outer = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            outer_identity = {"pid": outer.pid, "creation_filetime_utc": creation_filetime(outer.pid)}
            timed_out = False
            outer_cleanup = None
            try:
                remaining = min(30, args.max_seconds - (time.perf_counter() - start))
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, 0)
                out, err = outer.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                outer_cleanup = taskkill_owned(outer_identity, min(10, max(1, args.max_seconds - (time.perf_counter() - start))))
                try:
                    out, err = outer.communicate(timeout=min(10, max(1, args.max_seconds - (time.perf_counter() - start))))
                except subprocess.TimeoutExpired:
                    out, err = b"", b"OUTER_DRAIN_TIMEOUT"
            (case / "outer_stdout.log").write_bytes(out)
            (case / "outer_stderr.log").write_bytes(err)
            native_path = case / "native" / "receipt.json"
            native = json.loads(native_path.read_text(encoding="utf-8-sig")) if native_path.is_file() else None
            identities = {"powershell": {"identity": outer_identity, "state": identity_state(outer_identity)}}
            for label, path in (("fixture", worker_identity), ("child", child_identity)):
                identities[label] = None
                if path.is_file():
                    record = json.loads(path.read_text(encoding="utf-8"))
                    identities[label] = {"identity": record, "state": identity_state(record)}
            rescue_cleanup = {}
            for label in ("fixture", "child"):
                entry = identities[label]
                if entry is not None and entry["state"] == "ALIVE":
                    rescue_cleanup[label] = taskkill_owned(entry["identity"])
                    entry["state_after_rescue"] = identity_state(entry["identity"])
            elapsed = time.perf_counter() - case_start
            actual = {"case": name, "command": cmd, "elapsed_seconds": elapsed,
                      "outer_exit_code": outer.returncode, "outer_timed_out": timed_out,
                      "outer_cleanup": outer_cleanup, "rescue_cleanup": rescue_cleanup,
                      "identities": identities,
                      "native_receipt": native}
            write_json(case / "outer_native_receipt.json", actual)
            qa["cases"].append(actual)
            expected_exit = 0 if name == "quick_exit_0" else None
            expected_status = {"quick_exit_0": "COMPLETED", "quick_exit_7": "FAILED_NONZERO",
                               "timeout_tree": "TIMEOUT"}[name]
            check = not timed_out and elapsed <= 30 and outer.returncode is not None
            check &= native is not None and native["status"] == expected_status
            if native is not None:
                check &= native["timeout"] == (name == "timeout_tree")
                check &= bool(native["native_receipt_complete"])
                check &= bool(native["exit_code_obtained"]) and bool(native["terminated_observed"])
                check &= bool(native["cleanup_verified"])
            check &= identities["powershell"]["state"] == "GONE"
            check &= identities["fixture"] is not None and identities["fixture"]["state"] == "GONE"
            if name == "timeout_tree":
                check &= identities["child"] is not None and identities["child"]["state"] == "GONE"
            if native is not None and identities["fixture"] is not None:
                check &= native["fixture_pid"] == identities["fixture"]["identity"]["pid"]
            if native is None:
                check = False
            elif name == "quick_exit_0":
                check &= outer.returncode == 0 and native["exit_code"] == expected_exit
            elif name == "quick_exit_7":
                check &= outer.returncode != 0 and native["exit_code"] == 7
            else:
                check &= outer.returncode != 0 and isinstance(native["exit_code"], int)
                check &= identities["child"] is not None
            if native is not None:
                check &= "FIXTURE_STDOUT" in (case / "native" / "stdout.log").read_text(encoding="utf-8")
                check &= "FIXTURE_STDERR" in (case / "native" / "stderr.log").read_text(encoding="utf-8")
            actual["accepted"] = bool(check)
            write_json(case / "outer_native_receipt.json", actual)
            if not check:
                qa["failure"] = f"First nonconforming case: {name}"
                break
        qa["elapsed_seconds"] = time.perf_counter() - start
        if len(qa["cases"]) == 3 and all(x["accepted"] for x in qa["cases"]) and qa["elapsed_seconds"] <= args.max_seconds:
            qa["status"] = "ENGINEERING_ACCEPTED"
    except Exception as exc:
        qa["failure"] = repr(exc)
        qa["elapsed_seconds"] = time.perf_counter() - start
    finally:
        write_json(output / "qa.json", qa)
        print(json.dumps(qa, sort_keys=True), flush=True)
    return 0 if qa["status"] == "ENGINEERING_ACCEPTED" else 1


def outer_main(args):
    if os.name != "nt":
        raise RuntimeError("Windows native process QA only")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or args.max_seconds > 300:
        raise ValueError("QA max-seconds must be in (0, 300]")
    output = Path(args.output).resolve()
    receipt_path = output.with_name(output.name + ".native.receipt.json")
    running_path = output.with_name(output.name + ".running.json")
    if output.exists() or receipt_path.exists() or running_path.exists():
        raise FileExistsError("QA output or outer receipt already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(running_path, {"status": "RUNNING", "requested_output": str(output),
                              "source_commit": args.source_commit})
    command = [sys.executable, str(Path(__file__).resolve()), "--qa-worker", "--output", str(output),
               "--max-seconds", str(args.max_seconds), "--source-commit", args.source_commit]
    receipt = {"status": "FAILED", "command": command, "output": str(output),
               "source_commit": args.source_commit, "max_seconds": args.max_seconds,
               "launcher_pid": None, "launcher_creation_filetime_utc": None,
               "worker_pid": None, "worker_creation_filetime_utc": None,
               "worker_exit_code": None, "outer_timeout": False, "cleanup": {},
               "launcher_state": None, "worker_state": None, "elapsed_seconds": None,
               "native_receipt_complete": False}
    outer = None
    try:
        outer = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        launcher = {"pid": outer.pid, "creation_filetime_utc": creation_filetime(outer.pid)}
        receipt["launcher_pid"] = launcher["pid"]
        receipt["launcher_creation_filetime_utc"] = launcher["creation_filetime_utc"]
        try:
            remaining = max(0, args.max_seconds - 15 - (time.perf_counter() - started))
            out, err = outer.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            receipt["outer_timeout"] = True
            receipt["cleanup"]["launcher"] = taskkill_owned(launcher, 10)
            try:
                out, err = outer.communicate(timeout=4)
            except subprocess.TimeoutExpired:
                out, err = b"", b"OUTER_DRAIN_TIMEOUT"
        receipt["worker_exit_code"] = outer.returncode
        receipt["launcher_state"] = identity_state(launcher)
        receipt["stdout_log"] = str(output.with_name(output.name + ".outer.stdout.log"))
        receipt["stderr_log"] = str(output.with_name(output.name + ".outer.stderr.log"))
        Path(receipt["stdout_log"]).write_bytes(out)
        Path(receipt["stderr_log"]).write_bytes(err)
        worker_path = output / "worker.json"
        worker = json.loads(worker_path.read_text(encoding="utf-8")) if worker_path.is_file() else None
        if worker is not None:
            receipt["worker_pid"] = worker["pid"]
            receipt["worker_creation_filetime_utc"] = worker["creation_filetime_utc"]
            receipt["worker_state"] = identity_state(worker)
            if receipt["worker_state"] == "ALIVE":
                receipt["cleanup"]["worker"] = taskkill_owned(worker, 10)
                receipt["worker_state_after_rescue"] = identity_state(worker)
        qa_path = output / "qa.json"
        qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.is_file() else None
        receipt["qa_status"] = qa["status"] if qa else None
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["native_receipt_complete"] = (
            not receipt["outer_timeout"] and isinstance(receipt["worker_exit_code"], int)
            and receipt["launcher_state"] == "GONE" and receipt["worker_state"] == "GONE"
            and receipt["elapsed_seconds"] <= args.max_seconds and qa is not None)
        if (receipt["native_receipt_complete"] and receipt["worker_exit_code"] == 0
                and receipt["qa_status"] == "ENGINEERING_ACCEPTED"):
            receipt["status"] = "ENGINEERING_ACCEPTED"
    except Exception as exc:
        receipt["error"] = repr(exc)
        if outer is not None and outer.poll() is None and receipt["launcher_pid"] is not None:
            launcher = {"pid": receipt["launcher_pid"],
                        "creation_filetime_utc": receipt["launcher_creation_filetime_utc"]}
            receipt["cleanup"]["exception"] = taskkill_owned(launcher, 10)
        receipt["elapsed_seconds"] = time.perf_counter() - started
    finally:
        write_json(receipt_path, receipt)
        running_path.unlink(missing_ok=True)
        print(json.dumps({"status": receipt["status"], "worker_exit_code": receipt["worker_exit_code"],
                          "elapsed_seconds": receipt["elapsed_seconds"],
                          "outer_receipt": str(receipt_path)}, sort_keys=True), flush=True)
    return 0 if receipt["status"] == "ENGINEERING_ACCEPTED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("quick_exit_0", "quick_exit_7", "timeout_tree", "child"))
    parser.add_argument("--identity")
    parser.add_argument("--child-identity")
    parser.add_argument("--output")
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--qa-worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    sys.exit(fixture_main(options) if options.worker else main(options) if options.qa_worker else outer_main(options))
