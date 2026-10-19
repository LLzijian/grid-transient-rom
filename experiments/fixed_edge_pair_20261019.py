"""Finite, train-only fixed eight/seven-term paired campaign. No model adoption."""
from __future__ import annotations
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
PROTOCOL = ROOT / "reports/astra6_fixed_edge_pair_20261019.md"
COST = ROOT / "private_data/closure_mechanism_20261019/P0/linear_cache_cost01/measure01"
STRUCTURES = ("full8", "delete7")
MILESTONES = (256, 512, 1024)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def checkpoint_steps():
    return sorted({0, 256, 512, 1024, *range(50, 1001, 50)})


def task_order():
    return [{"seed": seed, "structure": s} for seed in (6101, 6102, 6103) for s in STRUCTURES]


def model_cap(structure):
    if structure not in STRUCTURES:
        raise ValueError("unregistered structure")
    return 12000. if structure == "full8" else 10000.


def lr_multiplier(step):
    if not isinstance(step, int) or not 1 <= step <= 1024:
        raise ValueError("registered update index required")
    return .3 if step > 512 else 1.


def validate_budget(mode, seconds):
    limits = {"qa": 600., "campaign": 69000., "analyze": 1800.}
    if mode not in limits or isinstance(seconds, bool) or not math.isfinite(seconds) or not 0 < seconds <= limits[mode]:
        raise ValueError("registered category limit exceeded")
    return float(seconds)


def recovery_accounting(previous_seconds, attempt_seconds, structure, replayed_steps,
                        recovery_seconds, used_before=False):
    values = (previous_seconds, attempt_seconds, recovery_seconds)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("finite nonnegative actual costs required")
    if used_before or not 0 <= replayed_steps <= 50 or recovery_seconds > 1200:
        raise ValueError("single bounded recovery exceeded")
    if recovery_seconds > attempt_seconds + 1e-6:
        raise ValueError("recovery subcost cannot exceed its native attempt envelope")
    total = previous_seconds + attempt_seconds
    if total > model_cap(structure):
        raise ValueError("cumulative model cap exceeded")
    return {"model_seconds": total, "recovery_seconds_subset": recovery_seconds,
            "replayed_steps": replayed_steps, "remaining_model_seconds": model_cap(structure)-total}


def claim_queue(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output / "campaign.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as stream:
        json.dump({"pid": os.getpid(), "created_unix": time.time()}, stream)
    return output / "campaign.lock"


def pause_requested(output):
    return (Path(output) / "pause.request").exists()


def dependency_hashes():
    # Extend the accepted dependency closure, including all new local imports.
    from linear_cache_cost_gate_20261019 import dependency_hashes as accepted_hashes
    accepted = accepted_hashes()
    pending = [Path(__file__), ROOT / "tests/test_fixed_edge_pair_20261019.py"]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found[path] = sha(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    base = ROOT / "src" if path.is_relative_to(ROOT / "src") else ROOT / "experiments"
                    parent = list(path.relative_to(base).parent.parts)
                    module = ".".join(parent[:len(parent)-node.level+1]+([module] if module else []))
                names = [module]+[module+"."+a.name for a in node.names if a.name != "*"]
            else:
                names = []
            for name in names:
                parts = name.split(".")
                for base in (ROOT / "experiments", ROOT / "src"):
                    candidates = [base.joinpath(*parts).with_suffix(".py")]
                    candidates += [base.joinpath(*parts[:i]) / "__init__.py" for i in range(1, len(parts)+1)]
                    pending.extend(p for p in candidates if p.is_file())
    result = {**accepted, **{p.relative_to(ROOT).as_posix(): h for p, h in sorted(found.items())}}
    result[PROTOCOL.relative_to(ROOT).as_posix()] = sha(PROTOCOL)
    return result


def verify_identity(expected, actual):
    from closure_checkpoint_identity_20261019 import identity_fingerprint
    if identity_fingerprint(expected) != identity_fingerprint(actual):
        raise ValueError("typed stage/source/data/code identity changed")


def save_safe(output, model, opt, rng, coverage, step, identity, trace, *, safe=True):
    from linear_cache_cost_gate_20261019 import finite
    from closure_checkpoint_identity_20261019 import save_identity_checkpoint
    from train_closure import save_checkpoint
    if not safe or not finite([model.state_dict(), opt.state_dict()]):
        raise FloatingPointError("unsafe state cannot publish a checkpoint")
    return save_identity_checkpoint(save_checkpoint, output, model, opt, rng, coverage,
                                    step, identity, trace)


def restore_safe(output, device, identity):
    from linear_cache_cost_gate_20261019 import restore_cached
    return restore_cached(output, device, identity)


def scientific_qualification(pair, scored):
    from evaluate_structured_rom import _needs_finer
    return not _needs_finer([pair] + [r for r in scored if r["available"]])


def primary_risk(rows, qualification):
    from train_local_memory_0098 import risk
    import torch
    selected = [r for r in rows if str(r["horizon"]) == "5" and r["label"] == "stable" and r["available"]]
    unresolved = [r["event_id"] for r in selected if not r.get("finite") or
                  not r.get("numerically_admissible") or not qualification.get(r["event_id"], False) or
                  r.get("mean_point_error") is None or not math.isfinite(r["mean_point_error"])]
    value = None if unresolved or not selected else float(risk(torch.tensor(
        [r["mean_point_error"] for r in selected], dtype=torch.float64)))
    return {"value": value, "eligible": len(selected), "unresolved_ids": unresolved,
            "definition": "original risk of stable complete-5s objective_point_error.mean_point_error"}


def physical_metrics(pred, event, mass, p_scale, mva_base=100):
    from closure_discovery_confirmation import _component_metrics
    if mva_base != 100:
        raise ValueError("registered system MVA base must be exactly 100")
    return _component_metrics(pred, event, mass, p_scale)


def train_steps(model, opt, store, events, rng, coverage, schedule, *, start_step=0,
                end_step=1024, force=.01, save=None, milestone=None, check=None,
                update=None, sampler=None):
    from linear_cache_cost_gate_20261019 import one_update
    from train_joint_numerical_pair_0100 import next_samples
    from train_closure import RATES
    update = update or one_update; sampler = sampler or next_samples
    trace = []
    if not 0 <= start_step <= end_step <= len(schedule):
        raise ValueError("illegal schedule range")
    if start_step == 0 and save:
        save(0, trace)
    for step in range(start_step+1, end_step+1):
        if check:
            check(step)
        samples, sampling = sampler(events, rng, coverage)
        row = schedule[step-1]
        if row["step"] != step:
            raise ValueError("schedule position mismatch")
        verify_identity(row["sampling"], sampling)
        if "numpy_rng_after" in row:
            verify_identity(row["numpy_rng_after"], rng.bit_generator.state)
        long_counts = [min(round(row["horizon_s"]*100), len(e["x"])-11) for e, _ in samples]
        local_counts = [min(20, len(e["x"])-i-1) for e, i in samples]
        if row["long_counts"] != long_counts or row["local_counts"] != local_counts:
            raise ValueError("genuine sample intervals differ")
        for group in opt.param_groups:
            group["lr"] = RATES[group["block"]] * lr_multiplier(step)
        result = update(model, opt, store, samples, row["horizon_s"], force)
        trace.append({"step": step, "horizon_s": row["horizon_s"], "sampling": sampling, **result})
        if save and step in checkpoint_steps():
            save(step, trace)
        if milestone and step in MILESTONES:
            milestone(step)
    return trace


def validate_classification_metadata(event, meta, reference):
    import numpy as np
    if reference["mva_base"] != 100:
        raise ValueError("physical system base differs from 100 MVA")
    for field, expected in (("equilibrium_delta", reference["equilibrium"]["delta"]), ("inertia", reference["inertia"])):
        if not np.array_equal(meta[field], expected):
            raise ValueError("classification system fingerprint mismatch")
    if meta["omega_s"] != reference["omega_s"]:
        raise ValueError("classification frequency units mismatch")
    t = meta["event_time_s"]
    end = (len(event["x"])-1)*.01
    if event["label"] not in ("stable", "unstable"):
        raise ValueError("physical outcome label required")
    if event["label"] == "unstable" and (t is None or not math.isfinite(t) or not 0 <= t <= end+1e-8):
        raise ValueError("unstable event time outside the record coordinate")
    if event["label"] == "stable" and t is not None:
        raise ValueError("stable record has physical event time")
    if "event_time_s" in event and event["event_time_s"] != t:
        raise ValueError("physical event time origin differs from record metadata")
    return True


def metadata(events):
    import numpy as np
    import h5py
    from closure_discovery_data import _load_reference_fingerprint
    reference = _load_reference_fingerprint()
    if reference["mva_base"] != 100:
        raise ValueError("physical units differ from 100 MVA system base")
    by_id = {e["id"]: e for e in events}
    for root_name in ("ieee39_postfault", "ieee39_expansion"):
        root = ROOT / "private_data" / root_name
        ids = {e["id"] for e in events if e["root"] == root_name}
        with (root / "dataset_index.csv").open() as stream:
            rows = {r["event_id"]: r for r in csv.DictReader(stream) if r["event_id"] in ids}
        if set(rows) != ids:
            raise ValueError("missing explicit train classification metadata")
        with h5py.File(root / "dataset.h5", "r") as file:
            if float(file.attrs["mva_base"]) != 100:
                raise ValueError("historical MVA base changed")
            eq = file["equilibrium"]["delta"][:]; inertia = file["inertia"][:]
            omega_s = float(file.attrs["omega_s"])
        for eid, row in rows.items():
            e = by_id[eid]
            event_time = None if row["event_time_s"] in ("", "None") else float(row["event_time_s"])
            if row["label"] != e["label"] or not math.isclose(float(row["trajectory_end_s"]), (len(e["x"])-1)*.01, abs_tol=1e-8):
                raise ValueError("train metadata label/time origin mismatch")
            if e["label"] == "unstable" and (event_time is None or not math.isclose(event_time, float(row["trajectory_end_s"]), abs_tol=1e-8)):
                raise ValueError("historical event time differs from its record endpoint")
            e["classification_meta"] = {"equilibrium_delta": eq.copy(), "inertia": inertia.copy(),
                "omega_s": omega_s, "event_time_s": event_time, "sample_interval_s": .01}
    for e in events:
        m = e["classification_meta"]
        validate_classification_metadata(e, m, reference)
        if e["root"] == "closure_discovery" and (m.get("system_fingerprint_sha256") != reference["fingerprint_sha256"] or not m.get("system_attestation_complete")):
            raise ValueError("new-train system attestation missing")
    return reference


def preflight(output, code):
    import numpy as np
    from linear_cache_cost_gate_20261019 import freeze, registered_terms
    from closure_mechanism_study import train_cohort
    output.mkdir(parents=True, exist_ok=True)
    events, force, cohort = train_cohort()
    reference = metadata(events)
    terms = registered_terms()
    identity, _, schedules = freeze(events, force, cohort, terms, output, code)
    # The accepted cost schedule contains full sampler RNG and coverage states.
    if sha(output / "schedule.json") != sha(COST / "schedule.json"):
        raise ValueError("full schedule file differs from accepted cost schedule")
    accepted = json.loads((COST / "schedule.json").read_text(encoding="utf-8"))
    verify_identity(accepted, schedules)
    identity["classification"] = [{"id": e["id"], "meta": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in e["classification_meta"].items()}} for e in events]
    identity["mva_base"] = 100.
    identity["stage_code"] = code
    # Bind every file in the explicitly authorized train artifacts, plus the
    # original train stores and fixed normalization/source files. No val paths.
    files = []
    for root_name in ("ieee39_postfault", "ieee39_expansion"):
        files += [ROOT / "private_data" / root_name / n for n in ("dataset_index.csv", "splits.json")]
    train_root = ROOT / "private_data/closure_discovery_20261017/D/new_data/train"
    files += [p for p in train_root.rglob("*") if p.is_file()]
    from train_local_memory_0098 import BASE, NORM
    from closure_mechanism_study import SOURCE
    files += [BASE, NORM, SOURCE, COST / "schedule.json"]
    identity["bound_files"] = {str(p.resolve()): sha(p) for p in files}
    identity["global_metadata_sha256"] = reference["fingerprint_sha256"]
    write(output / "stage_identity.json", identity)
    return events, force, terms, identity, schedules, reference


def evaluate_checkpoint(model, events, output, provenance, check):
    import numpy as np
    import torch
    from train_local_memory_0098 import graph
    from summarize_train_local_memory_0098 import event_metrics, build_summary
    from evaluate_structured_rom import _pair_row, _scored_numerics, _safe
    from analyze_symbolic_closure import analyze_model
    output.mkdir(parents=True, exist_ok=False)
    model = copy.deepcopy(model).cpu(); model.eval()
    records = {k: [] for k in ("records", "common_frequency", "normalized_p", "objective_point_error", "numerical")}
    extras = []; qualification = {}
    mass = model.mass.detach().numpy(); pscale = model.p_scale.detach().numpy()
    for first in range(0, len(events), 32):
        check()
        block = events[first:first+32]
        histories = np.stack([e["x"][:11] for e in block])
        with torch.no_grad(), graph(model):
            h = torch.tensor(histories, dtype=torch.float64)
            fine = model.rollout(h, 1600, .005).numpy()
            coarse = model.rollout(h, 800, .01).numpy()
        for e, history, f, c in zip(block, histories, fine, coarse):
            for key, rows in event_metrics(model, e, f, c).items():
                records[key].extend(rows)
            pair = _pair_row(f, c, history, e, e["classification_meta"], mass)
            scored = _scored_numerics(f, c, e, mass)
            qualification[e["id"]] = scientific_qualification(pair, scored)
            extras.append({"event_id": e["id"], "group": e["group"], "label": e["label"],
                "scientific_numeric": qualification[e["id"]], "pair": pair,
                "physical": physical_metrics(f[::2], e, mass, pscale, 100)})
        write(output / "partial.json", _safe({"provenance": provenance, "rows": records, "extras": extras}))
    if len(events) != 366 or any(len(v) != 1830 for v in records.values()):
        raise ValueError("full train denominator changed")
    with torch.enable_grad():
        local, arrays = analyze_model(model, source_sha256=provenance["model_sha256"], source_path=provenance["checkpoint"])
    np.savez_compressed(output / "local.npz", **arrays)
    write(output / "local.json", _safe(local))
    result = {"provenance": provenance, "rows": records, "extras": extras,
        "summary": build_summary(records), "primary": primary_risk(records["objective_point_error"], qualification),
        "scientific_numeric": qualification, "units": {"frequency": "rad/s", "power": "system pu", "mva_base": 100},
        "event_count": 366, "val_read": False, "test_read": False}
    write(output / "evidence.json", _safe(result))
    write(output / "complete.json", {"provenance": provenance, "evidence_sha256": sha(output / "evidence.json"),
                                    "local_sha256": sha(output / "local.json"), "status": "COMPLETED"})
    return result


def worker(args):
    from linear_cache_cost_gate_20261019 import load_runtime, fresh_model, restore_cached, finite
    load_runtime()
    import numpy as np
    import torch
    from train_closure import make_optimizer, environment
    from closure_training_runtime import TensorEventStore, synchronize
    key = str(args.seed)+"_"+args.structure
    output = args.output; folder = output / key; folder.mkdir(parents=True, exist_ok=True)
    attempt = folder / ("attempt_recovery" if args.recovery else "attempt_initial")
    attempt.mkdir(exist_ok=False)
    started = time.perf_counter()
    write(attempt / "worker.json", {"pid": os.getpid(), "argv": sys.argv})
    frozen = json.loads((output / "frozen_code.json").read_text())
    try:
        verify_identity(frozen, dependency_hashes())
        events, force, terms, common, schedules, _ = preflight(attempt / "preflight", frozen)
    except BaseException as exc:
        write(attempt / "result.json", {"status": "BLOCKED_IDENTITY", "error": "GLOBAL_IDENTITY_CHANGED: "+repr(exc)})
        raise
    device = torch.device("cuda", torch.cuda.current_device()); torch.cuda.get_rng_state(device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    identity = {"common": common, "seed": args.seed, "structure": args.structure,
                "environment": {k: str(v) if k == "torch" else v for k, v in environment(device).items()}}
    from closure_checkpoint_identity_20261019 import identity_fingerprint
    checkpoint = folder / "checkpoints"; checkpoint.mkdir(exist_ok=True)
    state = {"status": "RUNNING", "pid": os.getpid(), "seed": args.seed, "structure": args.structure,
             "last_completed_update": 0, "identity_sha256": identity_fingerprint(identity)}
    def check(step=None):
        write(attempt / "progress.json", state)
        if time.perf_counter()-started > args.max_seconds-8:
            raise TimeoutError("cumulative model deadline")
        if args.recovery and not (attempt / "recovery_complete.json").exists() and time.perf_counter()-started > 1200:
            raise TimeoutError("single recovery activity sublimit")
    def sources_check():
        try:
            verify_identity(frozen, dependency_hashes())
            if any(sha(path) != value for path, value in common["bound_files"].items()):
                raise ValueError("bound source/data payload changed")
            from train_closure import cohort_identity
            from closure_discovery_data import _load_reference_fingerprint
            if cohort_identity(events) != common["cohort"]["sha256"] or _load_reference_fingerprint()["fingerprint_sha256"] != common["global_metadata_sha256"]:
                raise ValueError("resident train arrays or global metadata changed")
        except BaseException as exc:
            raise RuntimeError("GLOBAL_IDENTITY_CHANGED: "+str(exc)) from exc
        from closure_mechanism_study import SOURCE, SOURCE_SHA
        if sha(SOURCE) != SOURCE_SHA:
            raise RuntimeError("GLOBAL_IDENTITY_CHANGED")
    try:
        if args.recovery:
            model, opt, rng, coverage, meta, changed = restore_cached(checkpoint, device, identity)
            if changed:
                raise ValueError("recovery environment changed")
            start_step = meta["step"]
            for committed in meta["trace"]:
                if committed["completed_step"] != start_step:
                    raise ValueError("checkpoint trace schedule position mismatch")
                raw = Path(committed["trace_file"]).read_bytes()[:committed["bytes"]]
                if hashlib.sha256(raw).hexdigest() != committed["prefix_sha256"]:
                    raise ValueError("safe checkpoint trace prefix changed")
            recovery = json.loads(args.recovery.read_text())
            target = recovery["last_completed_update"]
            if not start_step <= target <= start_step+50:
                raise ValueError("replay exceeds safe-checkpoint bound")
        else:
            model = fresh_model(args.structure, terms, device); opt = make_optimizer(model)
            rng = np.random.default_rng(args.seed)
            coverage = {"cover_order": rng.permutation(len(events)).tolist(), "cursor": 0}
            start_step = 0; target = 0
        state["last_completed_update"] = start_step
        if set(id(p) for g in opt.param_groups for p in g["params"]) != {id(p) for p in model.parameters() if p.requires_grad}:
            raise ValueError("optimizer does not cover all allowed parameters")
        def save(step, trace):
            state["last_completed_update"] = step
            if step:
                sources_check()
            trace_file = attempt / "trace.jsonl"
            compact = [{"trace_file": str(trace_file), "bytes": trace_file.stat().st_size,
                        "prefix_sha256": sha(trace_file), "completed_step": step}] if trace_file.exists() else []
            path = save_safe(checkpoint, model, opt, rng, coverage, step, identity, compact)
            write(attempt / "progress.json", state)
            return path
        def milestone(step):
            root = folder / ("milestone_"+str(step))
            pointer = json.loads((checkpoint / "latest.json").read_text())
            cp = checkpoint / pointer["directory"]
            provenance = {"seed": args.seed, "structure": args.structure, "step": step,
                "checkpoint": str(cp / "model.json"), "model_sha256": sha(cp / "model.json"),
                "identity": identity, "split": "train"}
            for prior in sorted(root.glob("attempt_*")) if root.exists() else []:
                complete = prior / "complete.json"
                if complete.exists():
                    record = json.loads(complete.read_text())
                    verify_identity(record["provenance"], provenance)
                    if sha(prior / "evidence.json") != record["evidence_sha256"] or sha(prior / "local.json") != record["local_sha256"]:
                        raise ValueError("completed milestone integrity failure")
                    return
            snapshot = copy.deepcopy(model.state_dict()); adam = copy.deepcopy(opt.state_dict())
            cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state(device).clone()
            numpy_rng = copy.deepcopy(rng.bit_generator.state)
            evaluate_checkpoint(model, events, root / ("attempt_"+str(time.time_ns())), provenance, check)
            from linear_cache_cost_gate_20261019 import _equal
            _equal(snapshot, model.state_dict(), exact=True); _equal(adam, opt.state_dict(), exact=True)
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
            verify_identity(numpy_rng, rng.bit_generator.state)
        def actual_update(*values):
            from linear_cache_cost_gate_20261019 import one_update, digest
            result = one_update(*values)
            state["last_completed_update"] += 1
            samples = values[3]
            with (attempt / "trace.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": state["last_completed_update"], **result,
                    "sample_ids": [e["id"] for e, _ in samples], "endpoints": [i for _, i in samples],
                    "numpy_rng_after_sha256": digest(rng.bit_generator.state),
                    "coverage_sha256": digest(coverage), "schedule_sha256": common["schedule_sha256"]}, allow_nan=False)+"\n")
            write(attempt / "progress.json", state)
            if pause_requested(output):
                save(state["last_completed_update"], [])
                raise InterruptedError("PAUSED_BY_USER")
            return result
        if args.recovery:
            if start_step in MILESTONES:
                milestone(start_step)
            if target > start_step:
                train_steps(model, opt, TensorEventStore(events, device), events, rng, coverage,
                    schedules[str(args.seed)]["rows"], start_step=start_step, end_step=target, force=force,
                    save=save, milestone=milestone, check=check, update=actual_update)
            check()
            write(attempt / "recovery_complete.json", {"seconds": time.perf_counter()-started,
                "completed_qpc": time.perf_counter(), "completed_unix": time.time(),
                "replayed_steps": target-start_step, "status": "COMPLETED"})
            start_step = target
            state["last_completed_update"] = target
        store = TensorEventStore(events, device)
        train_steps(model, opt, store, events, rng, coverage, schedules[str(args.seed)]["rows"],
            start_step=start_step, force=force, save=save, milestone=milestone, check=check, update=actual_update)
        sources_check()
        state["status"] = "COMPLETED"
    except BaseException as exc:
        state.update(status="PAUSED_BY_USER" if str(exc) == "PAUSED_BY_USER" else "FAILED",
                     error=repr(exc), traceback=traceback.format_exc())
        if "model" in locals():
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "safe": False}, attempt / "failed_state.pt")
        raise
    finally:
        state["worker_seconds"] = time.perf_counter()-started
        write(attempt / "result.json", state)


def native_job(command, folder, cap, recovery_marker=None, worker_json=None):
    """OS monotonic/QPC envelope: Popen startup, owned child exit and cleanup."""
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "native.json"
    from linear_cache_cost_gate_20261019 import wait_owned
    started = time.perf_counter(); unix = time.time(); proc = None; cleanup = {}
    result = {"started_unix": unix, "started_qpc": started, "exit_code": -1, "command": command,
              "includes_child_cost_once": True, "timed_out": False, "native_receipt_complete": False}
    try:
        with (folder / "stdout.log").open("w") as out, (folder / "stderr.log").open("w") as err:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            result["launcher_pid"] = proc.pid
            write(folder / "native_running.json", {**result, "status": "RUNNING", "supervisor_pid": os.getpid()})
            if recovery_marker is not None:
                while proc.poll() is None and not Path(recovery_marker).exists():
                    elapsed = time.perf_counter()-started
                    if elapsed >= min(1200., cap)-6:
                        wait_owned(proc, .001, cleanup)
                    time.sleep(.1)
            remaining = cap-(time.perf_counter()-started)-6
            result["exit_code"] = wait_owned(proc, max(.001, remaining), cleanup)
    except BaseException as exc:
        result.update(error=repr(exc), timed_out=isinstance(exc, subprocess.TimeoutExpired))
        if proc is not None and proc.poll() is None:
            try:
                wait_owned(proc, .001, cleanup)
            except BaseException:
                pass
    finally:
        result["seconds"] = time.perf_counter()-started
        result["cleanup"] = cleanup
        result["native_receipt_complete"] = (proc is not None and proc.poll() is not None and "failure" not in cleanup)
        result["actual_worker_pid"] = None
        if worker_json is not None:
            try:
                worker_path = Path(worker_json)
                if worker_path.exists() and worker_path.stat().st_mtime >= unix:
                    result["actual_worker_pid"] = json.loads(worker_path.read_text())["pid"]
            except (OSError, ValueError, KeyError) as exc:
                result["worker_identity_error"] = repr(exc)
        result["status"] = "COMPLETED" if result["exit_code"] == 0 and result["native_receipt_complete"] else "TIMED_OUT" if result["timed_out"] else "FAILED"
        write(receipt, result)
    return result


def capability_table(evidence):
    from train_local_memory_0098 import risk
    import torch
    table = {}
    qualified = evidence["scientific_numeric"]
    for label, horizons in (("stable", ("1", "2", "5", "8")), ("unstable", ("prefix",))):
        for horizon in horizons:
            all_rows = [r for r in evidence["rows"]["objective_point_error"] if r["label"] == label and str(r["horizon"]) == horizon]
            rows = [r for r in all_rows if r["available"]]
            bad = [r["event_id"] for r in rows if not r.get("finite") or not r.get("numerically_admissible") or
                   not qualified.get(r["event_id"], False) or r.get("mean_point_error") is None]
            values = [r["mean_point_error"] for r in rows]
            cell = {"label": label, "horizon": horizon, "all_events": len(all_rows), "eligible_events": len(rows),
                "unavailable_events": len(all_rows)-len(rows), "unresolved_ids": bad,
                "point_risk": None if bad or not values else float(risk(torch.tensor(values, dtype=torch.float64)))}
            for family, unit in (("frequency", "rad/s"), ("mechanical_power", "system pu (100 MVA)")):
                physical = [r for r in evidence["extras"] if r["label"] == label]
                eligible = [r for r in physical if r["physical"][horizon]["available"]]
                valid = [r for r in eligible if r["physical"][horizon][family]["finite"] and qualified[r["event_id"]]]
                sse = sum(r["physical"][horizon][family]["physical_sse"] for r in valid)
                count = sum(r["physical"][horizon][family]["n_scalar_points"] for r in valid)
                complete = len(valid) == len(eligible) and bool(eligible)
                cell[family] = {"unit": unit, "eligible_events": len(eligible), "qualified_events": len(valid),
                    "physical_sse": sse if complete else None, "n_scalar_points": count if complete else None,
                    "pooled_rmse": math.sqrt(sse/count) if complete and count else None,
                    "unresolved_ids": [r["event_id"] for r in eligible if r not in valid]}
            table[label+"_"+horizon] = cell
    classifications = []
    for row in evidence["extras"]:
        original = row["pair"]["fine_classification"]
        classifications.append({"event_id": row["event_id"], "label": row["label"],
            "category": original["category"] if row["scientific_numeric"] else "numerical_unresolved",
            "predicted_los": original["predicted_los"] if row["scientific_numeric"] else None})
    resolved = [r for r in classifications if r["category"] not in ("numerical_unresolved", "latency_uncovered")]
    table["classification"] = {"all_events": len(classifications), "resolved_events": len(resolved),
        "counts": {k: sum(r["category"] == k for r in classifications) for k in sorted({r["category"] for r in classifications})},
        "unstable_denominator": sum(r["label"] == "unstable" for r in classifications),
        "stable_denominator": sum(r["label"] == "stable" for r in classifications),
        "missed_resolved_unstable": sum(r["label"] == "unstable" and not r["predicted_los"] for r in resolved),
        "false_los_resolved_stable": sum(r["label"] == "stable" and r["predicted_los"] for r in resolved),
        "unresolved_events": sum(r["category"] == "numerical_unresolved" for r in classifications),
        "resolved_numeric_coverage_of_eligible": len(resolved)/len(classifications) if classifications else None}
    cohort = evidence.get("provenance", {}).get("identity", {}).get("common", {}).get("cohort", {})
    exclusions = [{"source": source, **row} for source in ("old_excluded", "new_excluded") for row in cohort.get(source, [])]
    original_count = len(classifications)+len(exclusions)
    reasons = {reason: sum(r.get("reason", "unspecified") == reason for r in exclusions) for reason in sorted({r.get("reason", "unspecified") for r in exclusions})}
    table["initialization_coverage"] = {"original_train_count": original_count,
        "eligible_train_count": len(classifications), "excluded_count": len(exclusions),
        "excluded_by_reason": reasons, "exclusions": exclusions,
        "causal_initialization_coverage_of_original": len(classifications)/original_count if original_count else None,
        "resolved_numeric_coverage_of_original": len(resolved)/original_count if original_count else None,
        "scope": "original exclusions retained; only eligible366 trained and evaluated"}
    return table


def summarize(output):
    """Read-only endpoint comparison; incomplete/invalid pairs remain unresolved."""
    output = Path(output); models = {}; pairs = []; capabilities = {}
    verify_identity(json.loads((output / "frozen_code.json").read_text()), dependency_hashes())
    for task in task_order():
        key = str(task["seed"])+"_"+task["structure"]
        endpoints = {}
        for step in MILESTONES:
            root = output / key / ("milestone_"+str(step))
            completed = [p.parent for p in root.glob("attempt_*/complete.json")]
            if completed:
                folder = completed[-1]; marker = json.loads((folder / "complete.json").read_text())
                if sha(folder / "evidence.json") != marker["evidence_sha256"]:
                    raise ValueError("analysis evidence integrity failure")
                endpoints[str(step)] = json.loads((folder / "evidence.json").read_text())
                capabilities[key+"_"+str(step)] = capability_table(endpoints[str(step)])
        models[key] = endpoints
    for seed in (6101, 6102, 6103):
        a = models[str(seed)+"_full8"].get("1024"); b = models[str(seed)+"_delete7"].get("1024")
        value = None if not a or not b or a["primary"]["value"] is None or b["primary"]["value"] is None else b["primary"]["value"]-a["primary"]["value"]
        distribution = []
        if a and b:
            aa = {(r["event_id"], r["horizon"]): r for r in a["rows"]["objective_point_error"]}
            for row in b["rows"]["objective_point_error"]:
                old = aa[(row["event_id"], row["horizon"])]
                x, y = old.get("mean_point_error"), row.get("mean_point_error")
                distribution.append({"event_id": row["event_id"], "group": row["group"], "horizon": row["horizon"],
                    "difference": None if x is None or y is None else y-x,
                    "qualified": bool(a["scientific_numeric"][row["event_id"]] and b["scientific_numeric"][row["event_id"]])})
        groups = {}
        for group in sorted({r["group"] for r in distribution}):
            groups[str(group)] = {}
            for horizon in ("1", "2", "5", "8", "prefix"):
                group_rows = [r for r in distribution if r["group"] == group and str(r["horizon"]) == horizon]
                vals = [r["difference"] for r in group_rows if r["qualified"] and r["difference"] is not None]
                groups[str(group)][horizon] = {"all_event_rows": len(group_rows), "qualified_finite_rows": len(vals),
                    "mean_point_error_difference": sum(vals)/len(vals) if vals else None,
                    "range": [min(vals), max(vals)] if vals else None,
                    "scope": "train descriptive subset; complete primary retained separately"}
            if a and b:
                ar = primary_risk([r for r in a["rows"]["objective_point_error"] if r["group"] == group], a["scientific_numeric"])
                br = primary_risk([r for r in b["rows"]["objective_point_error"] if r["group"] == group], b["scientific_numeric"])
                groups[str(group)]["stable_complete5_point_risk"] = {"full": ar, "delete": br,
                    "difference": None if ar["value"] is None or br["value"] is None else br["value"]-ar["value"]}
        pairs.append({"seed": seed, "delete_minus_full_primary": value, "events": distribution, "groups": groups})
    values = sorted(p["delete_minus_full_primary"] for p in pairs if p["delete_minus_full_primary"] is not None)
    all_numeric = all("1024" in endpoints and all(endpoints["1024"]["scientific_numeric"].values()) for endpoints in models.values())
    return {"status": "COMPLETED" if len(values) == 3 and all_numeric else "UNRESOLVED", "pairs": pairs,
        "all_final_event_numerics_qualified": all_numeric,
        "capability_table": capabilities,
        "primary_seed": 6101, "median_difference": values[1] if len(values) == 3 else None,
        "range": [values[0], values[-1]] if values else None,
        "positive_directions": sum(v > 0 for v in values), "models": models,
        "scope": "train fit and fixed retraining comparison; no significance, generalization or model adoption"}


def reconcile_receipts(output, interruption=None):
    """Reconcile only the twelve fixed model-attempt receipt locations."""
    output = Path(output); found = {}; errors = []
    if interruption is not None and not isinstance(interruption, dict):
        errors.append({"scope": "external interruption", "error": "mapping required"})
        interruption = None
    candidates = []
    for task in task_order():
        key = str(task["seed"])+"_"+task["structure"]
        for attempt in ("initial", "recovery"):
            candidates.append((task, attempt, output / key / ("native_"+attempt) / "native.json"))
    if interruption and interruption.get("verified_external") is True:
        try:
            task = {"seed": interruption["seed"], "structure": interruption["structure"]}
            if task in task_order():
                candidates.append((task, "initial", Path(interruption["native_receipt_path"])))
        except (KeyError, TypeError) as exc:
            errors.append({"scope": "external interruption receipt", "error": repr(exc)})
    for task, attempt, path in candidates:
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(record["seconds"], bool) or not isinstance(record["seconds"], (int, float)) or not math.isfinite(record["seconds"]) or record["seconds"] < 0:
                raise ValueError("invalid actual native cost")
            row = {**record, **task, "attempt": attempt, "receipt_path": str(path), "receipt_sha256": sha(path)}
            found[(task["seed"], task["structure"], attempt)] = row
            if not record.get("native_receipt_complete"):
                errors.append({"path": str(path), "error": "native process-tree exit not confirmed"})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": repr(exc)})
    return list(found.values()), errors


def campaign(args):
    started = time.perf_counter(); output = args.output.resolve()
    state = {"status": "STARTING", "pid": os.getpid(), "tasks": [], "owns_queue_attempt": False}
    interruption = None; prior_unallocated = 0.
    try:
        if args.recovery:
            interruption = json.loads(args.recovery.read_text())
            if not isinstance(interruption, dict):
                interruption = None
                raise ValueError("verified interruption mapping required")
            prior_unallocated = interruption.get("previous_queue_unallocated_seconds", 0.)
            if isinstance(prior_unallocated, bool) or not isinstance(prior_unallocated, (int, float)) or not math.isfinite(prior_unallocated) or prior_unallocated < 0:
                state["prior_accounting_unresolved"] = True
                prior_unallocated = 0.
                raise ValueError("valid prior native queue envelope required")
        before, _ = reconcile_receipts(output, interruption)
        baseline = {(r["seed"], r["structure"], r["attempt"]): r["receipt_sha256"] for r in before}
        _campaign_impl(args, state)
    except BaseException as exc:
        state.update(status="PAUSED_BY_USER" if str(exc) == "PAUSED_BY_USER" else "UNRESOLVED",
                     error=repr(exc), traceback=traceback.format_exc())
    finally:
        state["pid"] = os.getpid()
        state["active_job"] = None
        state["queue_attempt_seconds"] = time.perf_counter()-started
        rows, errors = reconcile_receipts(output, interruption)
        state["tasks"] = rows
        state["models_seconds"] = sum(r["seconds"] for r in rows)
        if "baseline" in locals():
            current = sum(r["seconds"] for r in rows if baseline.get((r["seed"], r["structure"], r["attempt"])) != r["receipt_sha256"])
            state["queue_unallocated_and_analysis_seconds"] = prior_unallocated+state["queue_attempt_seconds"]-current
            state["cumulative_science_seconds"] = state["models_seconds"]+state["queue_unallocated_and_analysis_seconds"]
        if errors:
            state["accounting_errors"] = errors
            if state["status"] == "COMPLETED":
                state["status"] = "UNRESOLVED"
        if state["status"] in ("RUNNING", "STARTING"):
            state["status"] = "UNRESOLVED"
        state["outer_native_calibration_required"] = True
        if state.get("owns_queue_attempt"):
            write(output / "campaign.json", state)
        else:
            # A rejected duplicate must never overwrite a running queue's file.
            write(output / ("startup_rejected_"+str(time.time_ns())+".json"), state)
    return state


def _campaign_impl(args, state):
    started = time.perf_counter(); output = args.output.resolve()
    recovery = None
    if args.recovery:
        recovery = json.loads(args.recovery.read_text())
        if recovery.get("kind") not in ("EXTERNAL_INTERRUPTION", "RUNTIME_INTERRUPTION") or recovery.get("verified_external") is not True:
            raise ValueError("only verified external/runtime interruption may recover")
        fd = os.open(output / "recovery.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        state["owns_queue_attempt"] = True
        frozen = json.loads((output / "frozen_code.json").read_text())
        verify_identity(frozen, dependency_hashes())
        state.update(json.loads((output / "campaign.json").read_text()))
        state["pid"] = os.getpid()
        state["owns_queue_attempt"] = True
        previous_unallocated = recovery["previous_queue_unallocated_seconds"]
        if not math.isfinite(previous_unallocated) or not 0 <= previous_unallocated <= 1800:
            raise ValueError("complete prior native queue overhead required")
        state["queue_unallocated_and_analysis_seconds"] = previous_unallocated
        native = json.loads(Path(recovery["native_receipt_path"]).read_text(encoding="utf-8-sig"))
        if native.get("native_receipt_complete") is not True or native.get("interrupted") is not True or not native.get("includes_child_cost_once"):
            raise ValueError("full actual interrupted model receipt required")
        recovery_accounting(native["seconds"], 0., recovery["structure"], 0, 0.)
        if not any(r.get("seed") == recovery["seed"] and r.get("structure") == recovery["structure"] for r in state["tasks"]):
            state["tasks"].append({"seed": recovery["seed"], "structure": recovery["structure"], **native})
            state["models_seconds"] += native["seconds"]
        state["status"] = "RUNNING"; state["recovery_used"] = True
    else:
        claim_queue(output)
        state["owns_queue_attempt"] = True
        frozen = dependency_hashes()
        state.update(status="RUNNING", pid=os.getpid(), tasks=[], models_seconds=0., recovery_used=False)
    qa = json.loads(args.qa_record.read_text())
    if qa.get("status") != "PASS":
        raise ValueError("matching synthetic QA PASS required")
    verify_identity(qa["code_sha256"], frozen)
    write(output / "frozen_code.json", frozen)
    # Hash all actual payload files once, then workers check on each boundary.
    previous_unallocated = state.get("queue_unallocated_and_analysis_seconds", 0.)
    initial_tasks = list(state["tasks"])
    unallocated = previous_unallocated
    write(output / "campaign.json", state)
    for task in task_order():
        prior = [r for r in state["tasks"] if r["seed"] == task["seed"] and r["structure"] == task["structure"]]
        is_recovery = recovery is not None and task["seed"] == recovery["seed"] and task["structure"] == recovery["structure"]
        if prior and not is_recovery:
            continue
        if pause_requested(output):
            state["status"] = "PAUSED_BY_USER"; break
        verify_identity(frozen, dependency_hashes())
        key = str(task["seed"])+"_"+task["structure"]
        spent = sum(r["seconds"] for r in prior)
        cap = model_cap(task["structure"])-spent
        if cap <= 20:
            state.setdefault("uncompleted_tasks", []).append({**task, "reason": "MODEL_BUDGET_EXHAUSTED", "remaining_seconds": cap})
            continue
        command = [sys.executable, str(Path(__file__)), "worker", "--output", str(output),
            "--seed", str(task["seed"]), "--structure", task["structure"], "--max-seconds", str(cap-8)]
        if is_recovery:
            command += ["--recovery", str(args.recovery.resolve())]
        state["active_job"] = {**task, "attempt": "recovery" if is_recovery else "initial", "status": "STARTING"}
        write(output / "campaign.json", state)
        receipt = native_job(command, output / key / ("native_recovery" if is_recovery else "native_initial"), cap-4,
            output / key / "attempt_recovery/recovery_complete.json" if is_recovery else None,
            output / key / ("attempt_recovery" if is_recovery else "attempt_initial") / "worker.json")
        state["active_job"] = None
        if is_recovery:
            marker = output / key / "attempt_recovery/recovery_complete.json"
            if not marker.exists():
                receipt["recovery_status"] = "UNRESOLVED"
            else:
                recovered = json.loads(marker.read_text())
                recovery_accounting(spent, receipt["seconds"], task["structure"],
                                    recovered["replayed_steps"], recovered["completed_qpc"]-receipt["started_qpc"])
        state["models_seconds"] += receipt["seconds"]
        state["tasks"].append({**task, **receipt})
        write(output / "campaign.json", state)
        result_path = output / key / ("attempt_recovery" if is_recovery else "attempt_initial") / "result.json"
        if result_path.exists() and "GLOBAL_IDENTITY_CHANGED" in result_path.read_text():
            state["status"] = "BLOCKED_IDENTITY"; break
        if result_path.exists() and "PAUSED_BY_USER" in result_path.read_text():
            state["status"] = "PAUSED_BY_USER"; break
        current_spend = sum(r["seconds"] for r in state["tasks"] if r not in initial_tasks)
        unallocated = previous_unallocated+time.perf_counter()-started-current_spend
        if unallocated > 1800 or state["models_seconds"]+unallocated > 69000-30 or time.perf_counter()-started > args.max_seconds-30:
            state["status"] = "BUDGET_EXHAUSTED"; break
    if state["status"] == "RUNNING":
        remaining = 1800-unallocated
        command = [sys.executable, str(Path(__file__)), "analyze", "--output", str(output), "--max-seconds", str(max(1, remaining-10))]
        state["analysis_native"] = native_job(command, output / "native_analysis", max(1, remaining-5),
                                             worker_json=output / "native_analysis/worker.json")
        latest = {(r["seed"], r["structure"]): r for r in state["tasks"]}
        state["status"] = "COMPLETED" if len(latest) == 6 and all(r["exit_code"] == 0 for r in latest.values()) and state["analysis_native"]["exit_code"] == 0 else "UNRESOLVED"
    state["queue_seconds"] = time.perf_counter()-started
    state["queue_unallocated_and_analysis_seconds"] = previous_unallocated+state["queue_seconds"]-sum(r["seconds"] for r in state["tasks"] if r not in initial_tasks)
    state["cumulative_science_seconds"] = state["models_seconds"]+state["queue_unallocated_and_analysis_seconds"]
    write(output / "campaign.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qa", "campaign", "worker", "analyze"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", required=True, type=float)
    parser.add_argument("--qa-record", type=Path)
    parser.add_argument("--seed", type=int, choices=(6101, 6102, 6103))
    parser.add_argument("--structure", choices=STRUCTURES)
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode == "worker":
        if args.seed is None or args.structure is None or not 0 < args.max_seconds <= model_cap(args.structure):
            parser.error("bounded registered worker required")
        return worker(args)
    validate_budget(args.mode, args.max_seconds)
    if args.mode == "campaign":
        if args.qa_record is None:
            parser.error("campaign requires --qa-record")
        return campaign(args)
    if args.mode == "qa":
        write(args.output / "worker.json", {"pid": os.getpid(), "argv": sys.argv})
        from linear_cache_cost_gate_20261019 import load_runtime
        load_runtime(); sys.path.insert(0, str(ROOT / "tests"))
        from test_fixed_edge_pair_20261019 import run_qa
        return run_qa(args.output)
    write(args.output / "native_analysis/worker.json", {"pid": os.getpid(), "argv": sys.argv})
    write(args.output / "analysis.json", summarize(args.output))


if __name__ == "__main__":
    main()
