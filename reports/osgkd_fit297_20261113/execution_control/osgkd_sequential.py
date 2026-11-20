"""Queue the existing C1 process behind C0; resume in finally, without restarting either."""
import ctypes
from ctypes import wintypes as w
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/osgkd_fit297_20261113/resource_queue.json"
IDENTITY = ROOT / "artifacts/osgkd_fit297_20261113/resource_queue_processes.json"
k = ctypes.WinDLL("kernel32", use_last_error=True)
n = ctypes.WinDLL("ntdll")
p = ctypes.WinDLL("psapi", use_last_error=True)
k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
k.OpenProcess.restype = w.HANDLE
k.CloseHandle.argtypes = [w.HANDLE]
k.CloseHandle.restype = w.BOOL
k.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
k.WaitForSingleObject.restype = w.DWORD
for fn in (n.NtSuspendProcess, n.NtResumeProcess):
    fn.argtypes = [w.HANDLE]
    fn.restype = w.LONG
p.EmptyWorkingSet.argtypes = [w.HANDLE]
p.EmptyWorkingSet.restype = w.BOOL


def save(value):
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temp.replace(OUT)


assert not OUT.exists(), "resource queue already claimed"
identities = json.loads(IDENTITY.read_bytes())
for variant, pid in (("C0",33128),("C1",10940)):
    row = identities[variant]
    assert row["ProcessId"] == pid
    assert "osgkd_fit297_20261113/run.py" in row["CommandLine"]
    assert f"--job-id train_{variant} --variant {variant}" in row["CommandLine"]
first = k.OpenProcess(0x100000, False, 33128)
second = k.OpenProcess(0x0800 | 0x0400 | 0x0100, False, 10940)
if not first or not second:
    raise ctypes.WinError(ctypes.get_last_error())
receipt = dict(run_id="osgkd_fit297_20261113", helper_pid=os.getpid(), C0_pid=33128, C1_pid=10940,
               status="starting", reason="available physical memory below 1 GB and active paging",
               source_or_optimizer_changed=False, new_training_jobs=0,
               started_utc=datetime.now(timezone.utc).isoformat())
suspended = False
start = time.perf_counter()
try:
    assert k.WaitForSingleObject(first,0) == 258, "C0 already exited"
    status = n.NtSuspendProcess(second)
    if status != 0:
        raise RuntimeError(f"NtSuspendProcess status {status}")
    suspended = True
    receipt.update(status="C1_waiting_for_C0", suspend_status=status,
                   working_set_trimmed=bool(p.EmptyWorkingSet(second)))
    save(receipt)
    while True:
        status = k.WaitForSingleObject(first,1000)
        if status == 0:
            break
        if status != 258:
            raise ctypes.WinError(ctypes.get_last_error())
except BaseException as error:
    receipt["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    if suspended:
        receipt["resume_status"] = n.NtResumeProcess(second)
        receipt["status"] = "C1_resumed" if receipt["resume_status"] == 0 else "resume_failed"
        receipt["queued_wall_s"] = time.perf_counter()-start
        receipt["resumed_utc"] = datetime.now(timezone.utc).isoformat()
        save(receipt)
    k.CloseHandle(first)
    k.CloseHandle(second)
