"""Finite native-Job controller for the registered 2026-10-28 force study.

Imports are data/torch free. Scientific functions, sampling, checkpoints and
decisions stay in force_graph_study_20261028. No automatic retries, optimizer
resume, QA repair, process limits, or historical lock moves are provided here.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.r_memory_execution_20261025 import (
    commit_id, read, sha, write_new, typed_identity, source_closure,
    runtime_metadata, python_command, clean_receipt,
)

BASE = ROOT / "private_data/force_graph_20261028/study01"
STUDY_MODULE = "experiments.force_graph_study_20261028"
PHASES = ("prepare", "qualify", "qa", "calibrate", "study", "summary")
MODES = ("L", "S", "M")
PHASE_OUTPUTS = {"prepare": "prepare.json", "qualify": "qualification/summary.json",
                 "qa": "qa/summary.json", "calibrate": "calibration/summary.json",
                 "study": "study/summary.json", "summary": "summary.json"}
SOURCE_ROOTS = (
    "experiments/force_graph_execution_20261028.py",
    "experiments/force_graph_study_20261028.py",
    "experiments/force_graph_model_20261028.py",
    "tests/test_force_graph_execution_20261028.py",
    "tests/test_force_graph_study_20261028.py",
    "tests/test_force_graph_model_20261028.py",
)
DOCUMENTS = ("AGENTS.md", "docs/engineering_loop.md", "docs/research_execution_20261028.md",
    "reports/force_graph_20261028_protocol.md", "reports/astra6_force_memory_20261028_raw.md",
    "results/force_graph_20261028/model_pre_freeze_qa.json",
    "results/force_graph_20261028/execution_pre_freeze_qa.json",
    "results/r_memory_20261025/recovery_20261028.json",
    "private_data/r_memory_20261025/run01_repair01/binding.json",
    "private_data/r_memory_20261025/run01_repair01/prepare.json",
    "private_data/r_memory_20261025/run01_repair01/schedule.json")
OLD = "private_data/r_memory_20261025/run01_repair01"
OLD_LOCK_SHA = "d764e2c272182acc6c54dda0a9edf1309c3475b1e79916305652b2adfeb123bf"
OLD_IDENTITIES = {"supervisor": (51852, 134374012174040563),
                  "worker": (61692, 134374012197238172)}
R0 = "private_data/r_memory_20261028/recovery_eval01"
R0_PHASE = "recovery_evaluate_A_25101_step001152"
SOURCE_ASSETS = {
    OLD + "/models/A_25101/checkpoints/step000512/model.json": "6710d648b2edc8e4c579480ba76e3282ed03b9647b507f4ef51b21fca0e65636",
    OLD + "/models/A_25101/checkpoints/step000512/training.pt": "61871814c64f7ca4f947a3896e6aef5f1619fa6ff30639f1fb7cb2d118fdb96c",
    OLD + "/models/A_25101/checkpoints/step000512/checkpoint.json": "87607ea701fda7a4834255dad4782d06ce123a058f24f90d4a4d42dcbcb4b5aa",
}
N_SOURCE = "private_data/closure_mechanism_20261019/P2/quartic_memory01/models/6101_N/checkpoints/step001024"
SOURCE_ASSETS.update({N_SOURCE + "/model.json": "c926caa685f2105d908c5d642f5944cc78f467744b3f9bd24831085eeca700a5",
    N_SOURCE + "/training.pt": "7fc8d7b9fd79d2356dcd21b808f1632db7b5ec701bea5d92b6c879290855ccaf",
    N_SOURCE + "/checkpoint.json": "08af27cd4c53b9698e6c8f88a84de54886c220c359446435878f8e71d13f12c7"})
NORMAL_CATEGORY_LIMITS = {"qa": 108, "calibration": 48, "formal": 960, "replay": 80}
TOTAL_LIMIT = 1304  # includes 108 reserved implementation-repair calls, not enabled here


def checked_base(base=BASE):
    base = Path(base).resolve()
    if base != (ROOT / "private_data/force_graph_20261028/study01"):
        raise ValueError("only the preregistered, non-aliased study01 output is allowed")
    return base


def check_locks(base=BASE, *, state=None, owned_identity=None):
    from experiments import uncapped_windows_job_20261021 as job
    from experiments.r_memory_execution_20261025 import LEGACY_LOCK, legacy_lock_allowed
    base = checked_base(base)
    own = base / "active.lock"
    if own.exists():
        owner = read(own)
        actual = typed_identity(owner.get("worker"))
        if (owned_identity is None or
                (actual["pid"], actual["creation_filetime_utc"]) !=
                (owned_identity["pid"], owned_identity["creation_filetime_utc"])):
            raise ValueError("the current study lock belongs to another or unowned worker")
    elif owned_identity is not None:
        raise ValueError("an owned native worker requires its actual study lock")
    old_root = ROOT / "private_data/closure_mechanism_20261019/P1"
    paths = [old_root / "rational_execution01/native_active.lock", *old_root.glob("*/active.lock")]
    admitted = []
    for path in dict.fromkeys(paths):
        if path.exists():
            if path.resolve() != ROOT / LEGACY_LOCK:
                raise ValueError("an unrelated historical lock requires root reconciliation: " + str(path))
            admitted.append(legacy_lock_allowed(path, root=ROOT, state=state))
    old = ROOT / OLD / "active.lock"
    recovery = read(ROOT / "results/r_memory_20261025/recovery_20261028.json")
    if (sha(old) != OLD_LOCK_SHA or recovery.get("lock_sha256") != OLD_LOCK_SHA
            or recovery.get("lock_preserved") is not True
            or recovery.get("status") != "INTERRUPTED_RUNTIME"):
        raise ValueError("only the exact retained interrupted r-memory lock is exempt")
    value = read(old)
    state = job.identity_state if state is None else state
    owners = {}
    for name, expected in OLD_IDENTITIES.items():
        identity = typed_identity(value.get(name))
        recorded = recovery["owners"][name]
        if ((identity["pid"], identity["creation_filetime_utc"]) != expected
                or recorded["pid_decimal"] != str(expected[0])
                or recorded["creation_filetime_utc_decimal"] != str(expected[1])
                or recorded["identity_state"] != "GONE" or state(identity) != "GONE"):
            raise ValueError("retained owner is changed, live, unknown or a reused PID")
        owners[name] = {"pid_decimal": str(expected[0]),
                        "creation_filetime_utc_decimal": str(expected[1]), "state": "GONE"}
    return admitted + [{"path": str(old), "sha256": OLD_LOCK_SHA, "owners": owners,
                        "old_lock_preserved": True, "old_JobCPU": None}]


def reused_runtime_proof(runtime, files):
    """Reuse the fresh, clean R0 Job and the once-full old runtime audit."""
    recovery = read(ROOT / "results/r_memory_20261025/recovery_20261028.json")
    full = recovery["binding_comparison"]
    if (full["total"] != 87 or full["matched"] != 87 or full["changed"] or full["missing"]
            or len(full["files"]) != 87 or any(row["state"] != "MATCH"
                or row["expected_sha256"] != row["actual_sha256"] for row in full["files"])):
        raise ValueError("the actual once-full 87-file match is required")
    old_hashes = {row["path"]: row["actual_sha256"] for row in full["files"]}
    for name in runtime["files"]:
        if files[name] != old_hashes.get(name):
            raise ValueError("runtime identity changed since its qualified owned-Job proof: " + name)
    r0 = ROOT / R0
    completion = read(r0 / "complete.json")
    receipt_path = r0 / "native" / R0_PHASE / "receipt.json"
    receipt = clean_receipt(read(receipt_path), completion["source_commit"], R0_PHASE)
    if (completion.get("status") != "COMPLETED" or completion.get("optimizer_updates") != 0
            or sha(receipt_path) != completion["native_receipt_sha256"]
            or receipt.get("worker_receipt", {}).get("source_commit") != completion["source_commit"]):
        raise ValueError("the fresh, actual zero-optimizer R0 receipt is required")
    expected = runtime["command"]
    if any(receipt["command"].get(k) != v for k, v in expected.items()):
        raise ValueError("R0 and current direct-base runtime commands differ")
    for name in ("experiments/uncapped_windows_job_20261021.py", "experiments/rational_python_runtime_20261022.py"):
        key = str(ROOT / name)
        if receipt["source_before"].get(key) != files.get(key):
            raise ValueError("native wrapper changed since the actual R0 ownership proof")
    paths = [r0 / "complete.json", r0 / "binding.json", receipt_path]
    files.update({str(p): sha(p) for p in paths})
    return {"receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path),
            "source_commit": completion["source_commit"], "fresh_native_probe_rerun": False,
            "old_binding_full_match_reference": str(ROOT / "results/r_memory_20261025/recovery_20261028.json")}


def freeze(source_commit, base=BASE):
    commit_id(source_commit)
    base = checked_base(base)
    if base.exists():
        raise FileExistsError("preserve the existing study output; do not re-freeze")
    locks = check_locks(base)
    runtime = runtime_metadata()
    paths = source_closure(ROOT, SOURCE_ROOTS) + [ROOT / n for n in DOCUMENTS]
    paths += [ROOT / n for n in SOURCE_ASSETS]
    paths += [Path(n) for n in runtime["files"]]
    files = {str(p.resolve()): sha(p) for p in paths}
    for name, expected in SOURCE_ASSETS.items():
        if files[str(ROOT / name)] != expected:
            raise ValueError("the fixed A512 or N6101 source changed: " + name)
    prepared = read(ROOT / OLD / "prepare.json")
    for name, key in (("fit_events.npz", "fit_cache_sha256"), ("no_update_events.npz", "no_update_cache_sha256")):
        path = ROOT / OLD / name
        if sha(path) != prepared[key]:
            raise ValueError("original prepared event cache bytes changed")
        files[str(path)] = prepared[key]  # bytes only; controller never decodes tensors
    runtime_proof = reused_runtime_proof(runtime, files)
    binding = {"source_commit": source_commit, "scope": "force_graph_20261028 finite L/S/M study",
        "base": str(base), "files": files, "runtime": runtime, "source_assets": SOURCE_ASSETS,
        "historical_locks": locks, "runtime_proof": runtime_proof,
        "no_process_quotas": True, "project_val_test_accessed": False,
        "optimizer_call_upper_bound": TOTAL_LIMIT, "normal_category_limits": NORMAL_CATEGORY_LIMITS,
        "reserved_implementation_repair_calls": 108, "automatic_resume": False,
        "repair_or_external_resume_requires_separate_explicit_root_execution": True}
    base.mkdir(parents=True, exist_ok=False)
    write_new(base / "binding.json", binding)
    return binding


def check_binding(binding, source_commit, base=BASE):
    from experiments import uncapped_windows_job_20261021 as job
    base = checked_base(base)
    if (binding.get("source_commit") != commit_id(source_commit) or binding.get("base") != str(base)
            or binding.get("no_process_quotas") is not True or not binding.get("files")
            or binding.get("optimizer_call_upper_bound") != TOTAL_LIMIT
            or binding.get("normal_category_limits") != NORMAL_CATEGORY_LIMITS):
        raise ValueError("a current same-source finite-study binding is required")
    job.verify_files(binding["files"])
    command = python_command([])
    expected = binding["runtime"]["command"]
    if any(command.get(k) != v for k, v in expected.items()):
        raise ValueError("current direct-base runtime differs from frozen command")
    return command


def call_counts(base=BASE):
    path = Path(base) / "optimizer_calls.jsonl"
    counts = {key: {"attempted": 0, "completed": 0} for key in NORMAL_CATEGORY_LIMITS}
    pending = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            category = row.get("category")
            if category not in counts:
                raise ValueError("unregistered optimizer category")
            key = (category, row.get("run_id", "__single_run_fixture__"))
            if row.get("event") == "optimizer_step_attempt" and key not in pending:
                counts[category]["attempted"] += 1
                pending[key] = row.get("step")
            elif row.get("event") == "optimizer_step_completed" and key in pending and pending[key] == row.get("step"):
                counts[category]["completed"] += 1
                del pending[key]
            else:
                raise ValueError("optimizer journal must alternate ordered attempt/completion rows")
    if any(v["attempted"] > NORMAL_CATEGORY_LIMITS[k] for k, v in counts.items()):
        raise ValueError("a registered category optimizer-call bound was exceeded")
    total = sum(v["attempted"] for v in counts.values())
    if total > TOTAL_LIMIT:
        raise ValueError("the cumulative 1304-call bound was exceeded")
    return {"categories": counts, "attempted_total": total,
            "completed_total": sum(v["completed"] for v in counts.values()),
            "pending_attempt_category": next(iter(pending))[0] if pending else None,
            "pending_attempts": [{"category": key[0], "run_id": key[1], "step": step}
                                 for key, step in pending.items()], "reserved_repair_calls_used": 0}


def _completed_phase(base, source_commit, phase):
    receipt_path = base / "native" / phase / "receipt.json"
    receipt = clean_receipt(read(receipt_path), source_commit, phase)
    bound_files = read(base / "binding.json")["files"]
    if any(receipt["source_before"].get(name) != digest for name, digest in bound_files.items()):
        raise ValueError("the completed phase was not executed against this exact source binding")
    folder = base / "ownership" / phase
    admission_path = folder / "admission.json"
    admission = read(admission_path)
    if (admission.get("source_commit") != source_commit or admission.get("phase") != phase
            or receipt["source_before"].get(str(admission_path)) != sha(admission_path)
            or any(receipt["source_before"].get(name) != digest or sha(name) != digest
                   for name, digest in admission["inputs"].items())):
        raise ValueError("the completed phase prerequisite evidence or admission changed")
    result_path = base / PHASE_OUTPUTS[phase]
    result = read(result_path)
    completion = read(folder / "complete.json")
    if (completion.get("status") != "COMPLETED" or completion.get("source_commit") != source_commit
            or completion.get("phase") != phase or completion.get("scientific_output_sha256") != sha(result_path)):
        raise ValueError("the native worker's completed scientific output changed")
    if phase == "prepare":
        if result.get("val_read") is not False or result.get("test_read") is not False:
            raise ValueError("prepared cohort must explicitly preserve closed val/test")
    elif phase == "qualify":
        method = result.get("training_method")
        if (result.get("status") not in ("PASS", "COMPLETED") or method not in ("fast", "rk4")
                or type(result.get("all_fast_qualified")) is not bool
                or (method == "fast" and result["all_fast_qualified"] is not True)):
            raise ValueError("one common qualified fast method or registered RK4 fallback is required")
    elif phase == "qa":
        if result.get("status") != "PASS" or result.get("actual_optimizer_updates") != 108:
            raise ValueError("actual 108-update save/restore QA must pass")
    elif phase == "calibrate":
        validate_calibration(result)
    # A completed study/summary can honestly contain UNRESOLVED science. This
    # controller does not turn a native exit into a model-qualification decision.
    return {str(receipt_path): sha(receipt_path), str(result_path): sha(result_path)}, result


def validate_calibration(result):
    chosen, scores = result.get("multipliers", {}), result.get("scores", {})
    if (result.get("status") != "COMPLETED" or result.get("all_trials_processed") is not True
            or set(chosen) != set(MODES) or set(scores) != set(MODES)):
        raise ValueError("all six bounded calibration trials must be processed")
    attempted = completed = 0
    for mode in MODES:
        if set(scores[mode]) != {"1", "3"}:
            raise ValueError("only the six registered calibration trials are allowed")
        for trial in scores[mode].values():
            a, c = trial.get("optimizer_attempts"), trial.get("optimizer_completed")
            if (type(a) is not int or type(c) is not int or not 0 <= c <= a <= 8 or a != c
                    or (trial.get("qualified") is True and (c != 8 or len(trial.get("trace", [])) != 8))):
                raise ValueError("calibration calls must be bounded and reconciled; eligible trials require eight updates")
            attempted += a; completed += c
        value = chosen[mode]
        if (value is not None and (type(value) is not int or value not in (1, 3)
                or scores[mode][str(value)].get("qualified") is not True)):
            raise ValueError("chosen calibration trial must be fully qualified")
        if value is None and any(t.get("qualified") is True for t in scores[mode].values()):
            raise ValueError("a healthy model must retain its eligible calibration trial")
    if (result.get("actual_optimizer_updates") != completed
            or result.get("actual_optimizer_attempts") != attempted or attempted > 48):
        raise ValueError("calibration totals must match all six actual trial records")
    return {"attempted": attempted, "completed": completed}


def phase_inputs(base, source_commit, phase):
    if phase not in PHASES:
        raise ValueError("unregistered native phase")
    files = {str(base / "binding.json"): sha(base / "binding.json")}
    dependencies = {"prepare": (), "qualify": ("prepare",), "qa": ("prepare", "qualify"),
        "calibrate": ("prepare", "qualify", "qa"),
        "study": ("prepare", "qualify", "qa", "calibrate"),
        "summary": ("prepare", "qualify", "qa", "calibrate", "study")}
    for previous in dependencies[phase]:
        paths, _ = _completed_phase(base, source_commit, previous)
        files.update(paths)
    counts = call_counts(base)
    if counts["pending_attempts"] and phase != "summary":
        raise ValueError("an interrupted/failed optimizer attempt needs explicit root reconciliation")
    for previous, category, expected in (("qa", "qa", 108), ("calibrate", "calibration", None)):
        if previous in dependencies[phase]:
            actual = counts["categories"][category]
            if previous == "calibrate":
                expected = read(base / PHASE_OUTPUTS[previous])["actual_optimizer_updates"]
            if actual != {"attempted": expected, "completed": expected}:
                raise ValueError("actual optimizer journal differs from completed " + previous)
    return files


def _study_call(base, phase):
    module = importlib.import_module(STUDY_MODULE)
    return getattr(module, phase)(base) if phase in ("prepare", "summary") else getattr(module, phase)(base, device="cuda")


def worker(base, source_commit, phase):
    from experiments import uncapped_windows_job_20261021 as job
    base = checked_base(base)
    binding = read(base / "binding.json")
    command = binding["runtime"]["command"]
    folder = base / "ownership" / phase
    identity = {**job.identity(), "physical_image": job.physical_image(job.api(), job.api().GetCurrentProcess()),
        "logical_executable": command["logical_executable"], "runtime_scheme": command["runtime_scheme"],
        "source_commit": source_commit, "phase": phase}
    identity["pid_decimal"] = str(identity["pid"])
    identity["creation_filetime_decimal"] = str(identity["creation_filetime_utc"])
    write_new(folder / "worker.json", identity)
    while not (folder / "owned").is_file():
        time.sleep(.02)
    try:
        check_binding(binding, source_commit, base)
        job.verify_files(read(folder / "admission.json")["inputs"])
        phase_inputs(base, source_commit, phase)
        check_locks(base, owned_identity=identity)
        result = _study_call(base, phase)
        counts = call_counts(base)
        write_new(folder / "complete.json", {"status": "COMPLETED", "source_commit": source_commit,
            "phase": phase, "scientific_result_type": type(result).__name__, "optimizer_calls": counts,
            "scientific_output_sha256": sha(base / PHASE_OUTPUTS[phase])})
    except BaseException as error:
        write_new(folder / "failure.json", {"source_commit": source_commit, "phase": phase,
            "error": repr(error), "traceback": traceback.format_exc()})
        raise


def launch(base, source_commit, phase):
    from experiments import uncapped_windows_job_20261021 as job
    base = checked_base(base)
    binding = read(base / "binding.json")
    check_binding(binding, source_commit, base)
    check_locks(base)
    inputs = phase_inputs(base, source_commit, phase)
    folder, native = base / "ownership" / phase, base / "native" / phase
    if folder.exists() or native.exists() or (base / PHASE_OUTPUTS[phase]).exists():
        raise FileExistsError("preserve the sole phase attempt; no automatic resume/repair")
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "admission.json", {"source_commit": source_commit, "phase": phase,
        "binding_sha256": sha(base / "binding.json"), "inputs": inputs,
        "historical_locks": check_locks(base), "device": "cuda" if phase not in ("prepare", "summary") else "cpu",
        "dtype": "float64", "automatic_resume": False})
    files = dict(binding["files"])
    files.update(inputs)
    files[str(folder / "admission.json")] = sha(folder / "admission.json")
    command = python_command(["-B", str(Path(__file__).resolve()), phase,
        "--source-commit", source_commit, "--base", str(base), "--worker"])

    def owned(_identity):
        with (folder / "owned").open("x", encoding="utf-8") as stream:
            stream.write("owned\n")

    receipt = job.supervise(command, native, base / "active.lock", files, ROOT,
        worker_receipt=folder / "worker.json", on_worker_owned=owned,
        provenance={"source_commit": source_commit, "phase": phase})
    clean_receipt(receipt, source_commit, phase)
    _completed_phase(base, source_commit, phase)
    return receipt


def run(base, source_commit):
    base = checked_base(base)
    if not (base / "binding.json").exists():
        freeze(source_commit, base)
    for phase in PHASES:
        # Completed phases can be inspected but never silently rerun. This also
        # permits run after an explicit standalone freeze, before any Job exists.
        launch(base, source_commit, phase)
    write_new(base / "complete.json", {"status": "COMPLETED", "source_commit": source_commit,
        "optimizer_calls": call_counts(base), "model_adopted": False,
        "scope": "finite registered execution ended; scientific decisions remain root-owned"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", *PHASES, "run"))
    parser.add_argument("--source-commit", required=True, type=commit_id)
    parser.add_argument("--base", default=BASE, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    base = checked_base(args.base)
    if args.worker:
        if args.phase not in PHASES:
            parser.error("only an actual registered phase may be a worker")
        worker(base, args.source_commit, args.phase)
    elif args.phase == "freeze":
        freeze(args.source_commit, base)
    elif args.phase == "run":
        run(base, args.source_commit)
    else:
        launch(base, args.source_commit, args.phase)


if __name__ == "__main__":
    main()
