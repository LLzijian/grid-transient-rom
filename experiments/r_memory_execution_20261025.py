"""Explicit native ownership for the registered, finite r-memory comparison.

This controller freezes inputs and launches one independent Windows Job per
phase.  It never reuses the retired runtime selection or an old output folder.
Scientific functions remain in r_memory_study_20261025; importing this file
does not import torch, read event arrays, or start a process.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "private_data/r_memory_20261025/run01"
LOGICAL_PYTHON = ROOT / ".venv-gpu/Scripts/python.exe"
MODES = ("A", "B", "C")
SEEDS = (25101, 25102)
PHASES = ("runtime-normal", "runtime-exception", "prepare", "preflight", "qa", "calibrate", "fit", "noise", "summary")
LEGACY_LOCK = "private_data/closure_mechanism_20261019/P1/training_efficiency01/active.lock"
LEGACY_SHA = "f9d72f0d5fccd8b5349907d69299aad543aa9b936f3e016971a7e24acd57c3e4"
RECOVERY = "results/training_efficiency_20261024/recovery_20261025.json"
LEGACY_IDENTITIES = {
    "supervisor": (27704, 134373316685022826),
    "worker": (54432, 134373316690097373),
}
CHECKPOINT = "private_data/closure_mechanism_20261019/P2/quartic_memory01/models/6101_N/checkpoints/step001024"
SOURCE_ASSETS = {
    CHECKPOINT + "/model.json": "c926caa685f2105d908c5d642f5944cc78f467744b3f9bd24831085eeca700a5",
    CHECKPOINT + "/training.pt": "7fc8d7b9fd79d2356dcd21b808f1632db7b5ec701bea5d92b6c879290855ccaf",
    CHECKPOINT + "/checkpoint.json": "08af27cd4c53b9698e6c8f88a84de54886c220c359446435878f8e71d13f12c7",
}
SOURCE_ROOTS = (
    "experiments/r_memory_model_20261025.py", "experiments/r_memory_study_20261025.py",
    "experiments/r_memory_execution_20261025.py", "experiments/uncapped_windows_job_20261021.py",
    "experiments/rational_python_runtime_20261022.py", "tests/test_r_memory_model_20261025.py",
    "tests/test_r_memory_study_20261025.py", "tests/test_r_memory_execution_20261025.py",
)
DOCUMENTS = (
    "AGENTS.md", "docs/engineering_loop.md", "reports/r_memory_20261025_protocol.md",
    "reports/astra6_r_memory_20261025_raw.md", RECOVERY, LEGACY_LOCK,
    "results/quartic_memory_20261020/frozen_evaluation_endpoints.json",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def commit_id(value):
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("a full verified archival commit is required")
    return value


def phase_key(phase, mode=None, seed=None, resume=False):
    if phase not in PHASES:
        raise ValueError("unregistered phase")
    if phase == "fit":
        if mode not in MODES or type(seed) is not int or seed not in SEEDS:
            raise ValueError("unregistered model or optimization seed")
        return f"fit_{mode}_{seed}" + ("_resume1" if resume else "")
    if mode is not None or seed is not None or resume:
        raise ValueError("only fit accepts model, seed, or resume")
    return phase.replace("-", "_")


def typed_identity(value):
    if (not isinstance(value, dict) or type(value.get("pid")) is not int or value["pid"] <= 0
            or type(value.get("creation_filetime_utc")) is not int or value["creation_filetime_utc"] <= 0):
        raise ValueError("typed positive PID and creation FILETIME required")
    return value


def legacy_lock_allowed(path, *, root=ROOT, state=None):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path != (root / LEGACY_LOCK).resolve() or sha(path) != LEGACY_SHA:
        raise ValueError("only the exact retained historical lock is exempt")
    value, recovery = read(path), read(root / RECOVERY)
    if (recovery.get("status") != "INTERRUPTED_RUNTIME" or recovery.get("lock_sha256") != LEGACY_SHA
            or recovery.get("lock_preserved") is not True):
        raise ValueError("root interruption record is required")
    if state is None:
        from experiments.uncapped_windows_job_20261021 import identity_state
        state = identity_state
    for name, expected in LEGACY_IDENTITIES.items():
        identity = typed_identity(value.get(name))
        if (identity["pid"], identity["creation_filetime_utc"]) != expected:
            raise ValueError("historical owner changed")
        recorded = recovery.get(name, {})
        if (recorded.get("pid"), recorded.get("creation_filetime_utc"), recorded.get("state")) != (*expected, "GONE"):
            raise ValueError("root owner reconciliation differs")
        if state(identity) != "GONE":
            raise ValueError("historical owner is live, unknown, or reused")
    return {"path": str(path), "sha256": LEGACY_SHA, "supervisor_state": "GONE", "worker_state": "GONE",
            "basis": "root-recorded historical interruption; no measured Job accounting claimed"}


def check_locks(*, root=ROOT, base=BASE, state=None):
    root, base = Path(root).resolve(), Path(base).resolve()
    paths = [base / "active.lock", root / "private_data/closure_mechanism_20261019/P1/rational_execution01/native_active.lock"]
    paths.extend((root / "private_data/closure_mechanism_20261019/P1").glob("*/active.lock"))
    admitted = []
    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        if path.resolve() == (root / LEGACY_LOCK).resolve():
            admitted.append(legacy_lock_allowed(path, root=root, state=state))
        else:
            raise ValueError("existing lock requires separate root reconciliation: " + str(path))
    return admitted


def _local_module(name, source, level, root):
    parts = name.split(".") if name else []
    if level:
        parent = source.parent
        for _ in range(level - 1):
            parent = parent.parent
        bases = [parent]
    else:
        bases = [root, root / "experiments", root / "src"]
    for base in bases:
        target = base.joinpath(*parts)
        for candidate in (target.with_suffix(".py"), target / "__init__.py"):
            if candidate.is_file() and candidate.resolve().is_relative_to(root):
                return candidate.resolve()
    return None


def source_closure(root=ROOT, roots=SOURCE_ROOTS):
    """Finite closure of local import statements; third-party runtime is separate."""
    root = Path(root).resolve()
    queue = [(root / name).resolve() for name in roots]
    found = set()
    while queue:
        path = queue.pop()
        if path in found:
            continue
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("required local source missing or outside workspace: " + str(path))
        found.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [(alias.name, 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "", node.level)]
                names.extend(((node.module + "." if node.module else "") + alias.name, node.level)
                             for alias in node.names if alias.name != "*")
            for name, level in names:
                target = _local_module(name, path, level, root)
                if target is not None and target not in found:
                    queue.append(target)
    return sorted(found, key=str)


def python_command(arguments, logical=LOGICAL_PYTHON):
    from experiments.rational_python_runtime_20261022 import python_command as build
    return build(logical, arguments, "direct_base_venv")


def runtime_metadata(logical=LOGICAL_PYTHON):
    """Identify current packages and native binaries without CUDA initialization."""
    from experiments.rational_python_runtime_20261022 import command_environment
    code = r'''
import json,sys,pathlib,importlib.metadata as md
import torch,numpy,h5py
paths=[pathlib.Path(torch.__file__),pathlib.Path(torch._C.__file__),pathlib.Path(numpy.__file__),pathlib.Path(h5py.__file__)]
for name in ('torch','numpy','h5py'):
    distribution=md.distribution(name)
    paths.extend(pathlib.Path(distribution.locate_file(item)) for item in (distribution.files or []) if str(item).endswith('.dist-info/METADATA'))
for name in ('c10.dll','torch_cpu.dll','torch_cuda.dll','c10_cuda.dll'):
    path=pathlib.Path(torch.__file__).parent/'lib'/name
    if path.is_file():paths.append(path)
home=pathlib.Path(sys.base_prefix)
paths.extend([home/'python.exe',home/f'python{sys.version_info.major}{sys.version_info.minor}.dll'])
print(json.dumps({'python':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':numpy.__version__,'h5py':h5py.__version__,'files':sorted({str(p.resolve()) for p in paths})}))
'''
    command = python_command(["-B", "-c", code], logical)
    process = subprocess.run([command["executable"], *command["arguments"]], check=True,
        env=command_environment(command), capture_output=True, text=True, encoding="utf-8",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    metadata = json.loads(process.stdout)
    metadata["command"] = {key: value for key, value in command.items() if key != "arguments"}
    metadata["files"].extend([str(Path(command["logical_executable"]).resolve()),
        str((Path(command["logical_executable"]).parent.parent / "pyvenv.cfg").resolve())])
    metadata["files"] = sorted(set(metadata["files"]))
    return metadata


def freeze(source_commit, *, root=ROOT, base=BASE, logical=LOGICAL_PYTHON):
    root, base = Path(root).resolve(), Path(base).resolve()
    commit_id(source_commit)
    if not base.is_relative_to(root / "private_data/r_memory_20261025"):
        raise ValueError("new study output must stay in its registered workspace directory")
    if (base / "binding.json").exists():
        raise FileExistsError("preserve the existing binding")
    admitted = check_locks(root=root, base=base)
    for name, expected in SOURCE_ASSETS.items():
        if sha(root / name) != expected:
            raise ValueError("registered N6101 source changed: " + name)
    runtime = runtime_metadata(logical)
    paths = source_closure(root) + [root / name for name in DOCUMENTS] + [root / name for name in SOURCE_ASSETS]
    paths.extend(Path(name) for name in runtime["files"])
    files = {}
    for path in paths:
        path = Path(path).resolve()
        if path.suffix.lower() in (".h5", ".hdf5"):
            raise ValueError("mixed HDF5 files cannot enter this binding")
        files[str(path)] = sha(path)
    binding = {"source_commit": source_commit, "scope": "r_memory_20261025 finite A/B/C comparison",
        "no_process_quotas": True, "base": str(base), "logical_executable": str(Path(logical).resolve()),
        "runtime": runtime, "files": files, "historical_lock_admission": admitted,
        "source_assets": SOURCE_ASSETS, "project_val_test_accessed": False}
    write_new(base / "binding.json", binding)
    return binding


def check_binding(binding, source_commit, *, base=BASE):
    from experiments import uncapped_windows_job_20261021 as job
    if (binding.get("source_commit") != commit_id(source_commit) or binding.get("no_process_quotas") is not True
            or Path(binding.get("base", "")).resolve() != Path(base).resolve() or not binding.get("files")):
        raise ValueError("registered current-source binding required")
    job.verify_files(binding["files"])
    current = python_command([], binding["logical_executable"])
    expected = binding["runtime"]["command"]
    if any(current.get(name) != expected.get(name) for name in current if name != "arguments"):
        raise ValueError("current runtime command differs from freeze")
    return current


def clean_receipt(value, source_commit, phase, *, allow_exception=False):
    worker = typed_identity(value.get("worker"))
    expected_status, expected_exit = ("FAILED", 1) if allow_exception else ("COMPLETED", 0)
    if (value.get("source_commit") != source_commit or value.get("phase") != phase
            or value.get("status") != expected_status or value.get("cleanup_verified") is not True
            or value.get("orphans_cleaned") is not False or value.get("actual_worker_state") != "GONE"
            or type(value.get("job_active_processes")) is not int or value["job_active_processes"] != 0
            or any(type(value.get(name)) is not int or value[name] != expected_exit
                   for name in ("exit_code", "worker_exit_code", "launcher_exit_code"))
            or not value.get("source_before") or value["source_before"] != value.get("source_after")):
        raise ValueError("actual clean, same-source native phase receipt required")
    if Path(worker.get("logical_executable", worker.get("executable", ""))).resolve() != LOGICAL_PYTHON.resolve():
        raise ValueError("actual worker logical executable differs")
    return value


def _require_phase(base, name, source_commit):
    path = Path(base) / "native" / name / "receipt.json"
    clean_receipt(read(path), source_commit, name)
    return path


def journal_counts(path, checkpoint_step):
    """Count durable calls, including a possible attempt interrupted in Adam."""
    path = Path(path)
    attempts = completed = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event") == "optimizer_step_attempt" and attempts == completed:
                attempts += 1
            elif row.get("event") == "optimizer_step_completed" and attempts == completed + 1:
                completed += 1
            else:
                raise ValueError("optimizer call journal is not an ordered attempt/completion record")
    if not 0 <= completed - checkpoint_step <= attempts - checkpoint_step <= 32:
        raise ValueError("recorded unpersisted calls exceed the single 32-call allowance")
    return {"attempted": attempts, "completed": completed,
            "unpersisted_attempts": attempts - checkpoint_step,
            "unpersisted_completions": completed - checkpoint_step}


def resume_proof(base, source_commit, mode, seed, expected_files, *, state=None):
    """Validate a sole external interruption without inventing a native receipt."""
    from experiments import uncapped_windows_job_20261021 as job
    if state is None:
        state = job.identity_state
    base = Path(base).resolve()
    key = phase_key("fit", mode, seed)
    native, owner = base / "native" / key, base / "ownership" / key
    run = base / "models" / f"{mode}_{seed}"
    for path in (native / "receipt.json", owner / "failure.json", owner / "complete.json",
                 run / "failure.json", run / "complete.json", run / "resume_once.json"):
        if path.exists():
            raise ValueError("completed or failed phases cannot be relabeled an external interruption")
    lock_path = base / "active.lock"
    lock, started, actual = read(lock_path), read(native / "started.json"), read(native / "actual_worker.json")
    worker_record = read(owner / "worker.json")
    if (lock.get("supervisor") != started.get("supervisor") or lock.get("worker") != actual.get("worker")
            or lock.get("launcher") != actual.get("launcher") or started.get("source_before") != expected_files
            or worker_record.get("source_commit") != source_commit or worker_record.get("phase") != key):
        raise ValueError("interrupted native source and owner records differ")
    for name in ("supervisor", "worker", "launcher"):
        if state(typed_identity(lock.get(name))) != "GONE":
            raise ValueError("interrupted owner is live, unknown, or reused")
    if (worker_record.get("pid"), worker_record.get("creation_filetime_utc")) != (
            lock["worker"]["pid"], lock["worker"]["creation_filetime_utc"]):
        raise ValueError("typed actual worker receipt differs")
    job.verify_files(expected_files)
    pointer = read(run / "checkpoints/latest.json")
    directory = pointer.get("directory", "")
    if not isinstance(directory, str) or Path(directory).name != directory or directory in ("", ".", ".."):
        raise ValueError("checkpoint pointer must name an immediate checkpoint directory")
    folder = (run / "checkpoints" / directory).resolve()
    if folder.parent != (run / "checkpoints").resolve() or sha(folder / "checkpoint.json") != pointer.get("checkpoint_sha256"):
        raise ValueError("interrupted checkpoint pointer identity changed")
    meta = read(folder / "checkpoint.json")
    step = meta.get("step")
    if type(step) is not int or step < 0 or step > 2048 or step % 32:
        raise ValueError("a real registered 32-step checkpoint is required")
    if directory != f"step{step:06d}":
        raise ValueError("checkpoint directory and persisted step differ")
    if meta.get("mode") != mode or meta.get("identity", {}).get("seed") != seed:
        raise ValueError("interrupted checkpoint configuration differs")
    checkpoint_files = {}
    for name in ("model.json", "training.pt", "checkpoint.json"):
        path = folder / name
        expected = pointer["checkpoint_sha256"] if name == "checkpoint.json" else meta.get(name.split(".")[0] + "_sha256")
        if sha(path) != expected:
            raise ValueError("interrupted checkpoint artifact changed")
        checkpoint_files[str(path)] = expected
    # Deserialization is restricted to this hash-checked, newly generated local
    # checkpoint.  The study subsequently validates complete state/RNG equality.
    import torch
    payload = torch.load(folder / "training.pt", map_location="cpu", weights_only=False)
    required = {"optimizer", "numpy_rng", "numpy_global_rng", "python_rng", "torch_rng", "cuda_rng", "coverage", "step"}
    if not required.issubset(payload) or payload["step"] != step or not meta.get("parameter_layout"):
        raise ValueError("complete Adam/RNG/schedule checkpoint required")
    if step and not payload["optimizer"].get("state"):
        raise ValueError("nonempty Adam state required after actual updates")
    counts = journal_counts(run / "optimizer_calls.jsonl", step)
    return {"status": "INTERRUPTED_RUNTIME", "source_commit": source_commit, "phase": key,
        "lock_sha256": sha(lock_path), "owner": lock, "checkpoint": str(folder), "step": step,
        "checkpoint_files": checkpoint_files, "optimizer_calls": counts,
        "job_active_processes": None, "cleanup_verified": None,
        "descendant_basis": "frozen noninherited anonymous Job/KILL_ON_JOB_CLOSE contract and exact GONE owners"}


def preserve_interrupted_lock(base, folder, proof, *, state=None):
    from experiments import uncapped_windows_job_20261021 as job
    state = job.identity_state if state is None else state
    base, folder = Path(base).resolve(), Path(folder).resolve()
    source, target = (base / "active.lock").resolve(), (folder / "interrupted_active.lock").resolve()
    if not source.is_relative_to(base) or not target.is_relative_to(base) or target.exists():
        raise ValueError("interrupted lock move must stay in the new study and preserve prior evidence")
    if sha(source) != proof["lock_sha256"] or read(source) != proof["owner"]:
        raise ValueError("interrupted lock changed before preservation")
    for name in ("supervisor", "worker", "launcher"):
        if state(typed_identity(proof["owner"][name])) != "GONE":
            raise ValueError("interrupted owner changed before preservation")
    source.rename(target)
    if sha(target) != proof["lock_sha256"]:
        raise ValueError("preserved lock bytes differ")
    return target


def phase_inputs(base, phase, source_commit, mode=None, seed=None):
    base = Path(base)
    files = [base / "binding.json"]
    if phase not in ("runtime-normal", "runtime-exception"):
        binding = read(base / "binding.json")
        expected_probe_sources = dict(binding["files"])
        expected_probe_sources[str((base / "binding.json").resolve())] = sha(base / "binding.json")
        for key, exception in (("runtime_normal", False), ("runtime_exception", True)):
            path = base / "native" / key / "receipt.json"
            receipt = clean_receipt(read(path), source_commit, key, allow_exception=exception)
            if receipt["source_before"] != expected_probe_sources:
                raise ValueError("current binding and actual runtime probe source closure differ")
            files.append(path)
    if phase not in ("runtime-normal", "runtime-exception", "prepare"):
        files.append(_require_phase(base, "prepare", source_commit))
        files.extend(base / name for name in ("prepare.json", "schedule.json", "fit_events.npz", "no_update_events.npz"))
    if phase in ("qa", "calibrate", "fit", "noise"):
        files.append(_require_phase(base, "preflight", source_commit))
        preflight = base / "preflight/summary.json"
        result = read(preflight)
        eligible = result.get("eligible", {})
        if (result.get("status") not in ("PASS", "PARTIAL") or result.get("actual_optimizer_updates") != 0
                or set(eligible) != set(MODES) or any(type(eligible[item]) is not bool for item in MODES)
                or not any(eligible.values())):
            raise ValueError("registered zero-update full-objective preflight required")
        if phase == "fit" and eligible.get(mode) is not True:
            raise ValueError("configuration isolated by full-objective preflight")
        if phase == "fit" and mode in ("B", "C") and not all(eligible[item] for item in ("B", "C")):
            raise ValueError("both B/C configurations must qualify before their paired comparison")
        files.append(preflight)
    if phase in ("calibrate", "fit", "noise"):
        files.append(_require_phase(base, "qa", source_commit))
        summary = base / "qa/summary.json"
        result = read(summary)
        if result.get("status") != "PASS" or result.get("actual_optimizer_updates") != 204:
            raise ValueError("registered 204-update fixture QA required")
        files.append(summary)
    if phase in ("fit", "noise"):
        files.append(_require_phase(base, "calibrate", source_commit))
        selection = base / "calibration/selection.json"
        chosen = read(selection).get("selected", {})
        files.append(selection)
        if phase == "fit" and chosen.get(mode) not in (1, 3):
            raise ValueError("configuration isolated by finite calibration")
        if phase == "fit" and mode in ("B", "C") and (chosen.get("B") not in (1, 3) or chosen.get("B") != chosen.get("C")):
            raise ValueError("B/C require the same jointly qualified calibration multiplier")
    if phase == "noise":
        for item in MODES:
            for item_seed in SEEDS:
                run = base / "models" / f"{item}_{item_seed}"
                done = run / "complete.json"
                result = read(done)
                if result.get("status") != "COMPLETED" or result.get("step") != 2048:
                    raise ValueError("all six fixed formal endpoints required before noise")
                normal_key = phase_key("fit", item, item_seed)
                resumed_key = phase_key("fit", item, item_seed, True)
                receipt_key = resumed_key if (base / "native" / resumed_key / "receipt.json").exists() else normal_key
                files.append(_require_phase(base, receipt_key, source_commit))
                files.append(done)
                files.extend((run / "checkpoints/step002048" / name) for name in ("model.json", "training.pt", "checkpoint.json"))
    return {str(path.resolve()): sha(path) for path in files}


def _study_call(base, phase, mode, seed, device, resume):
    import torch
    torch.set_num_threads(1)
    study = importlib.import_module("experiments.r_memory_study_20261025")
    if phase in ("preflight", "qa", "calibrate"):
        return getattr(study, phase)(base, device=device)
    if phase == "fit":
        return study.fit(base, mode, seed, device=device, resume=resume)
    if phase == "summary":
        return (getattr(study, "summary", None) or study.summarize)(base)
    return getattr(study, phase)(base)


def worker(base, binding, source_commit, phase, mode=None, seed=None, device="cuda", resume=False):
    from experiments import uncapped_windows_job_20261021 as job
    base = Path(base)
    command = check_binding(binding, source_commit, base=base)
    key = phase_key(phase, mode, seed, resume)
    folder = base / "ownership" / key
    identity = {**job.identity(), "physical_image": job.physical_image(job.api(), job.api().GetCurrentProcess()),
        "logical_executable": command["logical_executable"], "runtime_scheme": command["runtime_scheme"],
        "source_commit": source_commit, "phase": key}
    write_new(folder / "worker.json", identity)
    while not (folder / "owned").is_file():
        time.sleep(.02)
    try:
        job.verify_files(read(folder / "admission.json")["inputs"])
        if phase == "runtime-normal":
            result = {"status": "PASS", "optimizer_updates": 0, "GPU_initialized": False}
        elif phase == "runtime-exception":
            raise RuntimeError("registered zero-optimizer ownership exception probe")
        else:
            result = _study_call(base, phase, mode, seed, device, resume)
        write_new(folder / "complete.json", {"status": "COMPLETED", "source_commit": source_commit,
            "phase": key, "scientific_result_type": type(result).__name__})
        return 0
    except BaseException as error:
        write_new(folder / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "source_commit": source_commit, "phase": key})
        raise


def launch(base, source_commit, phase, mode=None, seed=None, device="cuda", resume=False):
    from experiments import uncapped_windows_job_20261021 as job
    base = Path(base).resolve()
    binding = read(base / "binding.json")
    check_binding(binding, source_commit, base=base)
    key = phase_key(phase, mode, seed, resume)
    folder = base / "ownership" / key
    native = base / "native" / key
    if folder.exists() or native.exists():
        raise FileExistsError("preserve the sole native phase attempt")
    inputs = phase_inputs(base, phase, source_commit, mode, seed)
    files = dict(binding["files"])
    files.update(inputs)
    proof = resume_proof(base, source_commit, mode, seed, files) if resume else None
    if not resume:
        check_locks(base=base)
    folder.mkdir(parents=True, exist_ok=False)
    if resume:
        # This exclusive protection is durable before moving the old lock or
        # creating a process.  Study fit owns its separate model resume guard.
        write_new(folder / "resume_once.json", proof)
        preserve_interrupted_lock(base, folder, proof)
        files.update(proof["checkpoint_files"])
    write_new(folder / "admission.json", {"phase": key, "source_commit": source_commit,
        "binding_sha256": sha(base / "binding.json"), "inputs": inputs, "device": device,
        "historical_locks": check_locks(base=base), "resume": resume})
    arguments = ["-B", str(Path(__file__).resolve()), phase, "--source-commit", source_commit,
                 "--base", str(base), "--device", device, "--worker"]
    if phase == "fit":
        arguments.extend(["--mode", mode, "--seed", str(seed)])
    if resume:
        arguments.append("--resume")
    command = python_command(arguments, binding["logical_executable"])
    receipt = job.supervise(command, native, base / "active.lock", files, ROOT,
        worker_receipt=folder / "worker.json",
        on_worker_owned=lambda _: (folder / "owned").write_text("owned\n", encoding="utf-8"),
        provenance={"source_commit": source_commit, "phase": key})
    clean_receipt(receipt, source_commit, key, allow_exception=phase == "runtime-exception")
    return receipt


def run(base, source_commit, device="cuda"):
    for phase in ("runtime-normal", "runtime-exception", "prepare", "preflight", "qa", "calibrate"):
        launch(base, source_commit, phase, device=device)
    skipped, failed = [], []
    selected = read(Path(base) / "calibration/selection.json")["selected"]
    eligible = read(Path(base) / "preflight/summary.json")["eligible"]
    for mode in MODES:
        for seed in SEEDS:
            reason = None
            if eligible.get(mode) is not True:
                reason = "full-objective preflight isolated mode"
            elif mode in ("B", "C") and not all(eligible[item] for item in ("B", "C")):
                reason = "paired B/C preflight qualification unavailable"
            elif selected.get(mode) not in (1, 3):
                reason = "finite calibration isolated mode"
            elif mode in ("B", "C") and selected.get("B") != selected.get("C"):
                reason = "joint B/C calibration multiplier unavailable"
            if reason is not None:
                skipped.append({"mode": mode, "seed": seed, "reason": reason})
                continue
            try:
                launch(base, source_commit, "fit", mode, seed, device)
            except Exception as error:
                binding = read(Path(base) / "binding.json")
                check_binding(binding, source_commit, base=base)
                key = phase_key("fit", mode, seed)
                clean_receipt(read(Path(base) / "native" / key / "receipt.json"), source_commit,
                    key, allow_exception=True)
                if not (Path(base) / "models" / f"{mode}_{seed}" / "failure.json").is_file():
                    raise
                failed.append({"mode": mode, "seed": seed, "error": repr(error)})
                check_locks(base=base)
    if not skipped and not failed:
        launch(base, source_commit, "noise", device=device)
    launch(base, source_commit, "summary", device=device)
    write_new(Path(base) / "ownership/run_complete.json", {"source_commit": source_commit,
        "status": "COMPLETED" if not failed and not skipped else "PARTIAL", "skipped": skipped,
        "failed": failed, "scientific_decision_owned_by": "root"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", *PHASES))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.base.resolve() != BASE.resolve():
        parser.error("only the registered run01 output is allowed")
    commit_id(args.source_commit)
    if args.phase == "freeze":
        if args.worker or args.mode is not None or args.seed is not None or args.resume:
            parser.error("freeze accepts only source commit and registered output")
        freeze(args.source_commit, base=args.base)
    elif args.phase == "run":
        if args.worker or args.mode is not None or args.seed is not None or args.resume:
            parser.error("run performs only the registered fresh phases")
        run(args.base, args.source_commit, args.device)
    elif args.worker:
        return worker(args.base, read(args.base / "binding.json"), args.source_commit,
            args.phase, args.mode, args.seed, args.device, args.resume)
    else:
        launch(args.base, args.source_commit, args.phase, args.mode, args.seed, args.device, args.resume)
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "src")]
    raise SystemExit(main())
