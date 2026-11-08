"""R0: one new CPU evaluation of the fixed, latest durable A/25101 model.

The old optimizer, six-fit queue, locks, and evaluations are never resumed or
written. ``check`` uses only Python source and JSON metadata, without torch or
event-array imports. ``run`` requires the root-verified remote source commit.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.r_memory_execution_20261025 import commit_id, read, sha, write_new

BASE = ROOT / "private_data/r_memory_20261028/recovery_eval01"
OLD = ROOT / "private_data/r_memory_20261025/run01_repair01"
CHECKPOINT = OLD / "models/A_25101/checkpoints/step001152"
RECOVERY = ROOT / "results/r_memory_20261025/recovery_20261028.json"
PROTOCOL = ROOT / "reports/r_memory_recovery_20261028_protocol.md"
PHASE = "recovery_evaluate_A_25101_step001152"
OLD_COMMIT = "6409d55f48aff1fd7c0091a26f44252a170502a3"
TARGET_SHA = {
    "model.json": "5e8bba7c406d12ec9745664c7aeb8cd8d1aebe29f3c0257b9ef13e0109f12aac",
    "training.pt": "a90aa8a672fae69083e5fc16888227835f8daedaf20cbf162c6060a3b55ed109",
    "checkpoint.json": "b5de24d436c6ae70af29b1b4942ca510e9047367165eb4c0fcec89dd8a4800b5",
}
COUNTS = {"fit": 297, "no_update": 88, "common385": 385,
          "original366": 366, "fresh19": 19}


def check_target_path():
    expected = ROOT / "private_data/r_memory_20261028/recovery_eval01"
    if BASE != expected or BASE.resolve() != expected or BASE.is_relative_to(OLD):
        raise ValueError("R0 must own its exact new, non-aliased output path")


def helper_signatures():
    path = ROOT / "experiments/r_memory_study_20261025.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    actual = {node.name: ([a.arg for a in node.args.args],
                         [a.arg for a in node.args.kwonlyargs])
              for node in tree.body if isinstance(node, ast.FunctionDef)}
    expected = {"checked_preparation": (["base"], ["fit_only"]),
                "load_export": (["path", "mode", "device"], []),
                "evaluate_model": (["model", "events", "output", "prepared"], ["noisy"])}
    if any(actual.get(name) != signature for name, signature in expected.items()):
        raise ValueError("the previously inspected study helper signatures changed")
    return expected


def frozen_target():
    hashes = {name: sha(CHECKPOINT / name) for name in TARGET_SHA}
    if hashes != TARGET_SHA:
        raise ValueError("fixed A1152 source bytes changed")
    meta = read(CHECKPOINT / "checkpoint.json")
    identity = meta["identity"]
    if (meta["step"] != 1152 or meta["mode"] != "A" or identity["seed"] != 25101
            or identity["mode"] != "A" or meta["scope"] != "research_candidate"
            or meta["model_sha256"] != hashes["model.json"]
            or meta["training_sha256"] != hashes["training.pt"]
            or meta["val_read"] is not False or meta["test_read"] is not False
            or read(CHECKPOINT.parent / "latest.json") != {
                "directory": "step001152", "checkpoint_sha256": hashes["checkpoint.json"]}):
        raise ValueError("latest durable A1152 role or checkpoint identity differs")
    return {"mode": "A", "seed": 25101, "step": 1152,
            "selection": "latest complete durable checkpoint, fixed before R0 scores",
            "role": "prospective recovery evaluation; not the old 2048 endpoint",
            "directory": str(CHECKPOINT), "files": hashes,
            "training_pt_loaded": False, "optimizer_updates": 0}


def cohort_metadata():
    prepared = read(OLD / "prepare.json")
    identity = read(CHECKPOINT / "checkpoint.json")["identity"]
    if (sha(OLD / "prepare.json") != identity["prepare_sha256"]
            or sha(OLD / "schedule.json") != identity["schedule_sha256"]):
        raise ValueError("the original checkpoint-bound preparation or schedule changed")
    fit, held, fresh = (set(prepared[k]) for k in ("fit_ids", "no_update_ids", "fresh19_ids"))
    if (len(fit) != 297 or len(held) != 88 or fit & held or len(fresh) != 19
            or not fresh <= fit | held or len((fit | held) - fresh) != 366
            or prepared["val_read"] is not False or prepared["test_read"] is not False):
        raise ValueError("original train385 preparation or group counts differ")
    return prepared


def prior_evaluations(recovery):
    refs = {}
    for step in (0, 512):
        folder = OLD / "evaluations" / f"A_25101_step{step:06d}"
        prior = recovery["evaluations"][str(step)]
        complete = read(folder / "complete.json")
        files = {str(folder / name): sha(folder / name)
                 for name in ("complete.json", "frozen.json", "science/evidence.json")}
        if (complete["status"] != "COMPLETED"
                or files[str(folder / "complete.json")] != prior["complete_sha256"]
                or files[str(folder / "frozen.json")] != prior["frozen_sha256"]
                or complete["evidence_sha256"] != prior["evidence_sha256"]
                or files[str(folder / "science/evidence.json")] != prior["evidence_sha256"]):
            raise ValueError("retained A0/A512 evidence changed")
        refs[str(step)] = {"reuse_only": True, "reevaluated": False, "files": files}
    return refs


def light_source_files():
    """Reuse the once-full 87-file audit; rehash critical small inputs now.

    Large third-party DLL/PYD and old non-target optimizer files are referenced
    through the immutable full audit, not repeatedly hashed by each Job check.
    """
    recovery = read(RECOVERY)
    comparison = recovery["binding_comparison"]
    if (comparison["total"] != 87 or comparison["matched"] != 87
            or comparison["changed"] or comparison["missing"]
            or sha(OLD / "binding.json") != recovery["binding_sha256"]):
        raise ValueError("the once-full matching old binding proof is required")
    files = {}
    for row in comparison["files"]:
        p = Path(row["path"])
        if row["state"] != "MATCH" or row["actual_sha256"] != row["expected_sha256"]:
            raise ValueError("the original 87-file audit is not fully matching")
        if p.suffix.lower() in (".pt", ".pyd"):
            continue
        if p.suffix.lower() == ".dll" and p.name.lower() != "python312.dll":
            continue
        files[str(p)] = row["actual_sha256"]
    for p in (Path(__file__).resolve(), PROTOCOL, RECOVERY, OLD / "binding.json",
              OLD / "prepare.json", OLD / "schedule.json", OLD / "active.lock"):
        files[str(p)] = sha(p)
    if files[str(OLD / "active.lock")] != recovery["lock_sha256"]:
        raise ValueError("retain the exact old active.lock")
    for name, digest in TARGET_SHA.items():
        files[str(CHECKPOINT / name)] = digest
    return files, recovery


def old_owners_gone(recovery):
    from experiments import uncapped_windows_job_20261021 as job
    lock = read(OLD / "active.lock")
    if sha(OLD / "active.lock") != recovery["lock_sha256"]:
        raise ValueError("old lock changed")
    result = {}
    for name in ("supervisor", "worker"):
        identity = lock[name]
        expected = recovery["owners"][name]
        if (type(identity["pid"]) is not int or type(identity["creation_filetime_utc"]) is not int
                or str(identity["pid"]) != expected["pid_decimal"]
                or str(identity["creation_filetime_utc"]) != expected["creation_filetime_utc_decimal"]
                or job.identity_state(identity) != "GONE"):
            raise ValueError("old owner is changed, live or not provably GONE")
        result[name] = {"pid_decimal": str(identity["pid"]),
                        "creation_filetime_utc_decimal": str(identity["creation_filetime_utc"]),
                        "identity_state": "GONE"}
    return result


def reused_runtime_proofs(files):
    from experiments.r_memory_execution_20261025 import clean_receipt, python_command
    binding = read(OLD / "binding.json")
    command = python_command([], binding["logical_executable"])
    old_probe_sources = dict(binding["files"])
    old_probe_sources[str(OLD / "binding.json")] = sha(OLD / "binding.json")
    if any(command.get(k) != v for k, v in binding["runtime"]["command"].items()):
        raise ValueError("the current direct-base runtime command changed")
    result = {}
    for phase, exception in (("runtime_normal", False), ("runtime_exception", True)):
        path = OLD / "native" / phase / "receipt.json"
        receipt = clean_receipt(read(path), OLD_COMMIT, phase, allow_exception=exception)
        if any(receipt["command"].get(k) != v for k, v in command.items() if k != "arguments"):
            raise ValueError("the historical owned runtime proof uses a different command")
        if receipt["source_before"] != old_probe_sources:
            raise ValueError("the runtime/helper/probe source proof differs from the old full binding")
        files[str(path)] = sha(path)
        result[phase] = {"path": str(path), "sha256": files[str(path)],
                         "source_commit": OLD_COMMIT, "reused_without_rerun": True}
    return result


def pure_check():
    check_target_path()
    if BASE.exists():
        raise FileExistsError("R0 is single-attempt; preserve its output")
    helper_signatures()
    target = frozen_target()
    cohort_metadata()
    for invalid in (None, "main", "0" * 39, "G" * 40, 123):
        try:
            commit_id(invalid)
        except ValueError:
            continue
        raise AssertionError("source commit format accepted an invalid value")
    return {"status": "PASS", "output": str(BASE), "target": target,
            "helper_signatures": helper_signatures(), "event_arrays_loaded": False,
            "torch_imported": "torch" in sys.modules, "output_created": False}


def prepare(source_commit):
    from experiments import uncapped_windows_job_20261021 as job
    pure_check()
    commit_id(source_commit)
    files, recovery = light_source_files()
    owners = old_owners_gone(recovery)
    proofs = reused_runtime_proofs(files)
    job.verify_files(files)
    refs = prior_evaluations(recovery)
    for ref in refs.values():
        files.update(ref["files"])
    prepared = cohort_metadata()
    for name, key in (("fit_events.npz", "fit_cache_sha256"),
                      ("no_update_events.npz", "no_update_cache_sha256")):
        files[str(OLD / name)] = prepared[key]
    BASE.mkdir(parents=True, exist_ok=False)
    write_new(BASE / "frozen_target.json", frozen_target())
    write_new(BASE / "prepare.json", {"source_commit": source_commit,
        "scope": "new R0 A1152 train-only frozen evaluation", "optimizer_updates": 0,
        "old_preparation_path": str(OLD / "prepare.json"), "counts": COUNTS,
        "reused_A0_A512": refs, "runtime_proofs": proofs, "old_owners": owners,
        "old_lock_preserved": True, "val_read": False, "test_read": False,
        "TASK4": "PAUSED", "model_adopted": False})
    files.update({str(BASE / name): sha(BASE / name)
                  for name in ("frozen_target.json", "prepare.json")})
    binding = {"source_commit": source_commit, "base": str(BASE), "phase": PHASE,
        "files": files, "old_binding_full_audit_path": str(RECOVERY),
        "old_binding_all_87_matched_at_recovery": True,
        "large_runtime_files_rehash": False, "no_process_quotas": True,
        "runtime": read(OLD / "binding.json")["runtime"]["command"]}
    write_new(BASE / "binding.json", binding)
    return binding


def worker(source_commit):
    from experiments import uncapped_windows_job_20261021 as job
    folder = BASE / "ownership" / PHASE
    binding = read(BASE / "binding.json")
    command = binding["runtime"]
    identity = {**job.identity(), "physical_image": job.physical_image(job.api(), job.api().GetCurrentProcess()),
        "logical_executable": command["logical_executable"], "runtime_scheme": command["runtime_scheme"],
        "source_commit": source_commit, "phase": PHASE}
    write_new(folder / "worker.json", identity)
    while not (folder / "owned").is_file():
        time.sleep(.02)
    try:
        if (binding["source_commit"] != commit_id(source_commit)
                or binding["base"] != str(BASE) or binding["phase"] != PHASE):
            raise ValueError("R0 source, output or phase binding differs")
        job.verify_files(binding["files"])
        old_owners_gone(read(RECOVERY))
        if frozen_target() != read(BASE / "frozen_target.json"):
            raise ValueError("frozen R0 model identity changed")
        import torch
        torch.set_num_threads(1)
        from experiments import r_memory_study_20261025 as study
        prepared, events = study.checked_preparation(OLD, fit_only=False)
        if len(events) != 385 or {e["id"] for e in events} != set(prepared["fit_ids"]) | set(prepared["no_update_ids"]):
            raise ValueError("R0 must retain the complete prepared train385 cohort")
        model = study.load_export(CHECKPOINT / "model.json", "A", "cpu")
        result = study.evaluate_model(model, events, BASE / "science", prepared, noisy=False)
        if (result["event_count"] != 385 or result["val_read"] is not False
                or result["test_read"] is not False or result["noisy"] is not False
                or {k: v["event_count"] for k, v in result["profiles"].items()} != COUNTS
                or torch.get_num_threads() != 1 or torch.cuda.is_initialized()):
            raise ValueError("R0 returned a different cohort, compute mode or data scope")
        write_new(folder / "complete.json", {"source_commit": source_commit, "phase": PHASE,
            "status": "COMPLETED", "optimizer_updates": 0,
            "evidence_sha256": sha(BASE / "science/evidence.json"),
            "qualification_is_not_model_adoption": True})
    except BaseException as error:
        write_new(folder / "failure.json", {"source_commit": source_commit, "phase": PHASE,
            "error": repr(error), "traceback": traceback.format_exc()})
        raise


def run(source_commit):
    from experiments import uncapped_windows_job_20261021 as job
    from experiments.r_memory_execution_20261025 import python_command, clean_receipt
    binding = prepare(source_commit)
    folder = BASE / "ownership" / PHASE
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "admission.json", {"source_commit": source_commit, "phase": PHASE,
        "binding_sha256": sha(BASE / "binding.json"), "optimizer_updates": 0,
        "old_lock_preserved": True, "device": "cpu", "dtype": "float64", "threads": 1})
    command = python_command(["-B", str(Path(__file__).resolve()), "run",
        "--source-commit", source_commit, "--worker"])
    files = dict(binding["files"])
    files[str(BASE / "binding.json")] = sha(BASE / "binding.json")
    files[str(folder / "admission.json")] = sha(folder / "admission.json")

    def owned(_identity):
        with (folder / "owned").open("x", encoding="utf-8") as stream:
            stream.write("owned\n")

    receipt = job.supervise(command, BASE / "native" / PHASE, BASE / "active.lock",
        files, ROOT, worker_receipt=folder / "worker.json", on_worker_owned=owned,
        provenance={"source_commit": source_commit, "phase": PHASE})
    clean_receipt(receipt, source_commit, PHASE)
    if sha(OLD / "active.lock") != read(RECOVERY)["lock_sha256"]:
        raise ValueError("old lock preservation failed")
    write_new(BASE / "complete.json", {"status": "COMPLETED", "source_commit": source_commit,
        "phase": PHASE, "evidence_sha256": sha(BASE / "science/evidence.json"),
        "native_receipt_sha256": sha(BASE / "native" / PHASE / "receipt.json"),
        "optimizer_updates": 0, "old_lock_preserved": True, "model_adopted": False})
    return {"status": "COMPLETED", "output": str(BASE), "optimizer_updates": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--source-commit", type=commit_id)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.action == "check":
        if args.worker:
            parser.error("check cannot be a worker")
        result = pure_check()
    else:
        if args.source_commit is None:
            parser.error("run requires the full root-verified remote --source-commit")
        if args.worker:
            worker(args.source_commit)
            return
        result = run(args.source_commit)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
