"""Single bounded train cost measurement for the accepted linear-cache runtime.

Importing this module loads only the standard library. Native-process finalization
is deliberately separate: the workload cannot know its own complete exit cost.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
STRUCTURES = ("full8", "delete7")
PANELS = ("distributed", "longest", "unstable")
HORIZONS = (1., 2., 5.)
SEEDS = (6101, 6102, 6103)
SOURCE_SHA = "c9fe3f86cf093d1f6c4600c71ecd333200840b9be4c2b156c018e1ff09a1e238"
TERMS_SHA = "3081b3eaeb3b96dd3b938d276e5dc64c83e4b023fd988fa940a817f74489973f"
COHORT_SHA = "42228e81d9f3d15c94350b14ae138a39285f19448db4c2dbc5fd376c0e513685"
MAIN = ROOT / "results/closure_discovery_20261017/E/packages/main_2701/model.json"
SOURCE_PATH = ROOT / "private_data/task2_prediction_optimization_0102/formal_S_O/step00650/structured_numeric_export.json"
OVERHEAD_KEYS = ("save_empty", "save_filled", "restore_empty", "restore_filled",
                 "full_eval", "jacobian", "construction")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def checkpoint_steps():
    return sorted({0, 256, 512, 1024, *range(50, 1001, 50)})


def measurement_tasks():
    return [{"structure": s, "horizon_s": h, "panel": p, "repeat": r}
            for s in STRUCTURES for h in HORIZONS for p in PANELS for r in (0, 1)]


def validate_budget(mode, seconds):
    if mode not in ("qa", "measure"):
        raise ValueError("unknown cost category")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("numeric time cap required")
    if not math.isfinite(seconds) or not 0 < seconds <= (300 if mode == "qa" else 6000):
        raise ValueError("time cap exceeds the registered category")
    return float(seconds)


def terminate_owned_tree(proc):
    """Only the Popen subtree owned by this caller, including venv redirectors."""
    if os.name == "nt":
        cmd = ["taskkill", "/PID", str(int(proc.pid)), "/T", "/F"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3)
        receipt = {"command": cmd, "returncode": result.returncode,
                   "stdout": result.stdout, "stderr": result.stderr}
        if result.returncode:
            receipt["failure"] = "owned Windows subtree termination unconfirmed"
    else:
        proc.kill()
        receipt = {"command": ["Popen.kill", str(proc.pid)], "returncode": 0}
    try:
        receipt["launcher_returncode"] = proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        receipt["failure"] = "owned launcher survived termination"
    return receipt


def wait_owned(proc, timeout, cleanup):
    try:
        return proc.wait(timeout=timeout)
    except BaseException:
        try:
            cleanup.update(terminate_owned_tree(proc))
        except BaseException as exc:
            cleanup["failure"] = repr(exc)
        raise


def dependency_hashes():
    """Read local Python import closure, including imports inside functions."""
    pending = [Path(__file__).resolve(), ROOT / "tests/test_linear_cache_cost_gate_20261019.py"]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        found[path] = sha(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    base = ROOT / "src" if path.is_relative_to(ROOT / "src") else ROOT / "experiments"
                    context = list(path.relative_to(base).parent.parts)
                    module = ".".join(context[:len(context)-node.level+1] + ([module] if module else []))
                names = [module] + [module + "." + alias.name for alias in node.names if alias.name != "*"]
            else:
                continue
            for name in names:
                parts = name.split(".") if name else []
                for base in (ROOT / "experiments", ROOT / "src"):
                    for i in range(1, len(parts)+1):
                        package = base.joinpath(*parts[:i]) / "__init__.py"
                        if package.is_file():
                            pending.append(package)
                    if parts:
                        candidate = base.joinpath(*parts).with_suffix(".py")
                        if candidate.is_file():
                            pending.append(candidate)
    return {p.relative_to(ROOT).as_posix(): h for p, h in sorted(found.items())}


def _seconds(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("finite nonnegative seconds required")
    if not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative seconds required")
    return float(value)


def compute_cost(maxima, overhead, schedules, startup_envelope):
    """Pure formula, also usable with short hand-calculated QA schedules.

    The real 1024-row schedule contract is enforced separately by finalization.
    """
    envelope = _seconds(startup_envelope)
    if set(maxima) != set(STRUCTURES) or set(overhead) != set(STRUCTURES):
        raise ValueError("both registered structures required")
    if {str(k) for k in schedules} != {str(k) for k in SEEDS}:
        raise ValueError("three registered schedules required")
    normalized = {}
    for structure in STRUCTURES:
        normalized[structure] = {}
        for horizon in HORIZONS:
            values = maxima[structure]
            key = next((k for k in (horizon, str(horizon), str(int(horizon))) if k in values), None)
            if key is None:
                raise ValueError("missing measured horizon maximum")
            normalized[structure][horizon] = _seconds(values[key])
            if normalized[structure][horizon] <= 0:
                raise ValueError("measured update maximum must be positive")
        if not set(OVERHEAD_KEYS) <= set(overhead[structure]):
            raise ValueError("missing cost component")
        for key in OVERHEAD_KEYS:
            _seconds(overhead[structure][key])
    rows = []
    for structure in STRUCTURES:
        o = overhead[structure]
        for seed, schedule in schedules.items():
            schedule = schedule["rows"] if isinstance(schedule, dict) else schedule
            if not schedule:
                raise ValueError("nonempty schedule required")
            horizons = [float(row["horizon_s"]) for row in schedule]
            if any(h not in HORIZONS for h in horizons):
                raise ValueError("unknown scheduled horizon")
            updates = sum(normalized[structure][h] for h in horizons)
            save = len(checkpoint_steps()) * max(o["save_empty"], o["save_filled"])
            milestones = 3 * (o["full_eval"] + o["jacobian"])
            total = updates + save + milestones + o["construction"] + envelope
            rows.append({"structure": structure, "seed": int(seed), "updates_seconds": updates,
                         "save_seconds": save, "milestones_seconds": milestones,
                         "construction_seconds": o["construction"], "startup_exit_seconds": envelope,
                         "total_seconds": total})
    recoveries = {s: envelope + max(overhead[s]["restore_empty"], overhead[s]["restore_filled"])
                  + 50 * max(normalized[s].values()) for s in STRUCTURES}
    recovery = max(recoveries.values())
    raw = sum(row["total_seconds"] for row in rows) + recovery
    total = raw * 1.25
    return {"models": rows, "checkpoint_steps": checkpoint_steps(), "milestones": [256, 512, 1024],
            "recovery_by_structure_seconds": recoveries, "one_global_recovery_seconds": recovery,
            "raw_plan_seconds": raw, "estimated_with_25pct_margin_seconds": total,
            "p2_cap_seconds": 86400, "within_cap_arithmetic": total <= 86400,
            "scope": "formula only; complete real workload must be verified separately"}


def load_runtime():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    global np, torch
    import numpy as np
    import torch
    torch.set_num_threads(1)


def _tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [t for item in value.values() for t in _tensors(item)]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in _tensors(item)]
    return []


def finite(value):
    # Adam counters remain CPU tensors; moments may be on CUDA.
    return all(bool(torch.isfinite(t).all().item()) for t in _tensors(value))


def _equal(a, b, path="", exact=False):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError("tensor structure mismatch: " + path)
        if not finite([a, b]):
            raise FloatingPointError("nonfinite checkpoint tensor: " + path)
        if exact or path.endswith("/step") or not a.is_floating_point():
            if not torch.equal(a.to(b.device), b):
                raise AssertionError("exact tensor mismatch: " + path)
        else:
            torch.testing.assert_close(a.to(b.device), b, atol=2e-9, rtol=2e-9)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            _equal(a[key], b[key], path + "/" + str(key), exact)
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _equal(x, y, path + "/" + str(i), exact)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b), path
        assert a == b if exact else math.isclose(a, b, abs_tol=2e-9, rel_tol=2e-9), path
    else:
        assert a == b, path


def restore_cached(output, device, identity):
    from train_closure import restore_checkpoint
    from closure_checkpoint_identity_20261019 import restore_identity_checkpoint
    from linear_cache_runtime_20261019 import enable_linear_cache
    result = restore_identity_checkpoint(restore_checkpoint, output, device, identity)
    model, opt = result[:2]
    model_id = id(model)
    param_ids = tuple(id(p) for p in model.parameters())
    refs = tuple(tuple(id(p) for p in group["params"]) for group in opt.param_groups)
    enabled = enable_linear_cache(model)
    assert enabled is model and id(enabled) == model_id
    assert tuple(id(p) for p in enabled.parameters()) == param_ids
    assert refs == tuple(tuple(id(p) for p in group["params"]) for group in opt.param_groups)
    assert all(id(p) in param_ids for group in opt.param_groups for p in group["params"])
    return result


def one_update(model, opt, store, samples, horizon, force):
    from closure_training_runtime import objective
    from train_local_memory_0098 import graph
    opt.zero_grad(set_to_none=True)
    with graph(model):
        value, components = objective(model, store, samples, horizon, force)
        if not finite([value, components]):
            raise FloatingPointError("nonfinite complete objective")
        value.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    if not finite([p.grad for p in params]):
        raise FloatingPointError("nonfinite complete objective gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, 100.)
    if not finite(norm):
        raise FloatingPointError("nonfinite clipped gradient norm")
    opt.step()
    model.end_graph_cache()
    if not finite([model.state_dict(), opt.state_dict()]):
        raise FloatingPointError("nonfinite Adam update/state")
    assert getattr(model, "_linear_cache", None) is None
    assert getattr(model, "_runtime_cache", None) is None
    return {"loss": float(value.detach()), "gradient_norm": float(norm.detach()),
            "components": {k: float(v.detach()) for k, v in components.items()}}


def registered_terms():
    from symbolic_closure_rom import _BASE_TERMS, term_key
    terms = json.loads(MAIN.read_text(encoding="utf-8"))["terms"]
    if digest(terms) != TERMS_SHA:
        raise ValueError("public terms identity changed")
    expected = [_BASE_TERMS[i] for i in (0, 1, 2, 3, 4, 5, 7, 8)]
    if [term_key(t) for t in terms] != [term_key(t) for t in expected]:
        raise ValueError("registered terms order changed")
    return {"full8": terms, "delete7": [t for t in terms if term_key(t) != term_key(_BASE_TERMS[2])]}


def fresh_model(structure, terms, device):
    from closure_mechanism_study import source_model
    from linear_cache_runtime_20261019 import enable_linear_cache
    return enable_linear_cache(source_model("cpu").to(device).with_terms(terms[structure]))


def freeze(events, force, cohort, terms, output, code):
    from train_closure import RATES, horizon_for_update
    from train_joint_numerical_pair_0100 import next_samples
    from closure_mechanism_study import SOURCE
    if len(events) != 366 or cohort["sha256"] != COHORT_SHA or sha(SOURCE) != SOURCE_SHA:
        raise ValueError("registered source/train identity mismatch")
    if len({e["id"] for e in events}) != 366 or any(len(e["x"]) < 12 for e in events):
        raise ValueError("illegal or duplicate train events")
    panels = {"distributed": np.linspace(0, len(events)-1, min(12, len(events)), dtype=int).tolist(),
              "longest": sorted(range(len(events)), key=lambda i: (-len(events[i]["x"]), events[i]["id"]))[:12],
              "unstable": sorted((i for i, e in enumerate(events) if e["label"] == "unstable"),
                                 key=lambda i: events[i]["id"])[:12]}
    schedules = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        coverage = {"cover_order": rng.permutation(len(events)).tolist(), "cursor": 0}
        rows = []
        for step in range(1, 1025):
            rng_before = copy.deepcopy(rng.bit_generator.state)
            samples, sampling = next_samples(events, rng, coverage)
            horizon = horizon_for_update(step)
            rows.append({"step": step, "horizon_s": horizon, "sampling": sampling,
                "numpy_rng_before": rng_before, "numpy_rng_after": copy.deepcopy(rng.bit_generator.state),
                "long_counts": [min(round(horizon*100), len(e["x"])-11) for e, _ in samples],
                "local_counts": [min(20, len(e["x"])-i-1) for e, i in samples],
                "rates": {k: v*(.3 if step > 512 else 1.) for k, v in RATES.items()},
                "lr_multiplier": .3 if step > 512 else 1.})
        included = sorted({eid for row in rows for eid in row["sampling"]["sample_ids"]})
        assert included == sorted(e["id"] for e in events)
        assert [sum(r["horizon_s"] == h for r in rows) for h in HORIZONS] == [308, 408, 308]
        schedules[str(seed)] = {"rows": rows, "included_ids": included, "eligible_pool_count": 366,
                                "final_coverage": copy.deepcopy(coverage),
                                "final_numpy_rng": copy.deepcopy(rng.bit_generator.state)}
    write(output / "schedule.json", schedules)
    metadata = []
    for event in events:
        arrays = {}
        for key in ("x", "torque"):
            arr = np.ascontiguousarray(event[key])
            arrays[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype),
                           "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
        metadata.append({"id": event["id"], "group": event["group"], "label": event["label"],
                         "length": len(event["x"]), "arrays": arrays})
    identity = {"source_sha256": SOURCE_SHA, "public_terms_sha256": TERMS_SHA,
        "cohort": cohort, "events": metadata, "code_sha256": code,
        "schedule_sha256": sha(output / "schedule.json"), "terms": terms,
        "panels": {k: [{"id": events[i]["id"], "label": events[i]["label"],
                         "length": len(events[i]["x"])} for i in indices] for k, indices in panels.items()},
        "force_scale": force, "endpoint": 10, "dtype": "float64", "clip_norm": 100.,
        "rates": RATES,
        "seeds": list(SEEDS), "milestones": [256, 512, 1024], "checkpoint_steps": checkpoint_steps(),
        "train_only": True, "val_read": False, "test_read": False}
    write(output / "frozen_identity.json", identity)
    return identity, panels, schedules


def full_evaluation(model, events, output, check):
    from summarize_train_local_memory_0098 import event_metrics, build_summary
    from evaluate_structured_rom import _safe
    from train_local_memory_0098 import graph
    # The caller times the CPU copy as part of this one mutually exclusive item.
    model = copy.deepcopy(model).cpu()
    model.eval()
    output.mkdir()
    records = {key: [] for key in ("records", "common_frequency", "normalized_p",
                                  "objective_point_error", "numerical")}
    completed = []
    for first in range(0, len(events), 32):
        check("full_train block " + str(first))
        block = events[first:first+32]
        history = torch.as_tensor(np.stack([e["x"][:11] for e in block]), dtype=torch.float64)
        with torch.no_grad(), graph(model):
            fine = model.rollout(history, 1600, .005).numpy()
            coarse = model.rollout(history, 800, .01).numpy()
        assert fine.shape[1:] == (1601, 40) and coarse.shape[1:] == (801, 40)
        for event, f, c in zip(block, fine, coarse):
            for key, rows in event_metrics(model, event, f, c).items():
                records[key].extend(rows)
            completed.append(event["id"])
        write(output / "partial_rows.json", _safe({"rows": records, "event_ids": completed}))
    assert len(completed) == len(set(completed)) == 366
    assert all(len(rows) == 366*5 for rows in records.values())
    summary = build_summary(records)
    numerical_ok = all((not row["available"]) or (row["finite_to_endpoint"] and
        row["coi_normalized_rmse"] <= .001 and row["common_normalized_rmse"] <= .001)
                       for row in records["numerical"])
    evidence = {"rows": records, "summary": summary, "event_count": 366,
                "event_ids": completed, "numerical_qualification": "PASS" if numerical_ok else "UNRESOLVED",
                "scope": "fresh S-O source; train only; no model adoption"}
    write(output / "evidence.json", _safe(evidence))
    return {"event_count": 366, "rows_by_family": {k: len(v) for k, v in records.items()},
            "numerical_qualification": evidence["numerical_qualification"],
            "evidence_sha256": sha(output / "evidence.json")}


def _checkpoint_trial(model, opt, output, identity, step, rng, coverage, measure):
    from closure_checkpoint_identity_20261019 import save_identity_checkpoint
    from train_closure import save_checkpoint
    output.mkdir()
    device = model.mass.device
    before_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before_opt = copy.deepcopy(opt.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    numpy_rng = copy.deepcopy(rng.bit_generator.state)
    measure("save", lambda: save_identity_checkpoint(save_checkpoint, output, model, opt, rng,
                                                    coverage, step, identity, []))
    restored = measure("restore", lambda: restore_cached(output, device, identity))
    rm, ro, rr, rc, meta, changed = restored
    assert not changed and meta["step"] == step
    _equal(before_model, rm.state_dict(), exact=(step == 0))
    _equal(before_opt, ro.state_dict())
    _equal(coverage, rc, exact=True)
    _equal(numpy_rng, rr.bit_generator.state, exact=True)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    return {"step": step, "identity_optimizer_rng_coverage": "PASS", "directory": str(output)}


def measure(args):
    from closure_mechanism_study import train_cohort, SOURCE
    from closure_training_runtime import TensorEventStore, resolve_device, synchronize
    from train_closure import make_optimizer, environment
    from analyze_symbolic_closure import analyze_model
    device = resolve_device("cuda")
    torch.cuda.get_rng_state(device)
    state = {"status": "RUNNING", "mode": "measure", "pid": os.getpid(), "argv": sys.argv,
        "rows": [], "components": [], "overhead": {s: {} for s in STRUCTURES},
        "checkpoints": [], "evaluations": {}, "jacobians": {}, "environment": environment(device),
        "val_read": False, "test_read": False, "scientific_admission": "NOT_AUTHORIZED"}
    def check(phase):
        state.update(phase=phase, elapsed_worker_seconds=time.perf_counter()-START)
        write(args.output / "progress.json", state)
        if time.perf_counter()-START > args.max_seconds-12:
            raise TimeoutError("single registered measurement deadline")
    def timed(kind, fn, structure=None, **metadata):
        synchronize(device)
        tick = time.perf_counter()
        result = fn()
        synchronize(device)
        seconds = time.perf_counter()-tick
        state["components"].append({"kind": kind, "structure": structure, "seconds": seconds,
                                     "start_offset": tick-START, "end_offset": tick-START+seconds,
                                     **metadata})
        return result, seconds
    try:
        qa = json.loads(args.qa_record.read_text(encoding="utf-8"))
        code = dependency_hashes()
        if qa.get("status") != "PASS" or qa.get("code_sha256") != code:
            raise ValueError("complete matching QA PASS required before real measurement")
        state["code_sha256"] = code
        state["qa_record_sha256"] = sha(args.qa_record)
        # One nonnested preparation interval includes true train loading, all
        # frozen schedules and tensor residency before the first trial update.
        def prepare():
            events, force, cohort = train_cohort()
            terms = registered_terms()
            identity, panels, schedules = freeze(events, force, cohort, terms, args.output, code)
            store = TensorEventStore(events, device)
            return events, force, terms, identity, panels, schedules, store
        payload, global_prepare = timed("global_prepare", prepare)
        events, force, terms, identity, panels, schedules, store = payload
        state["frozen_identity_sha256"] = sha(args.output / "frozen_identity.json")
        state["schedule_sha256"] = sha(args.output / "schedule.json")
        check("frozen before all trial updates")
        constructions = {s: [] for s in STRUCTURES}
        def construct(name):
            model, seconds = timed("construction", lambda: fresh_model(name, terms, device), name)
            constructions[name].append(seconds)
            return model
        def trial_checkpoint(name, model, opt, kind, step):
            durations = {}
            def cp_measure(operation, fn):
                value, seconds = timed(operation + "_" + kind, fn, name)
                durations[operation + "_" + kind] = seconds
                return value
            rng = np.random.default_rng(6101)
            coverage = {"cover_order": rng.permutation(len(events)).tolist(), "cursor": 0,
                        "cost_probe_only": True, "model_not_adopted": True}
            report = _checkpoint_trial(model, opt, args.output / (name + "_" + kind),
                                       identity, step, rng, coverage, cp_measure)
            state["checkpoints"].append({"structure": name, "kind": kind, **report})
            state["overhead"][name].update(durations)
            check("checkpoint completed " + name + "/" + kind)
        for name in STRUCTURES:
            empty = construct(name)
            trial_checkpoint(name, empty, make_optimizer(empty), "empty", 0)
            del empty
            for horizon in HORIZONS:
                for panel in PANELS:
                    check("configuration " + name + "/" + str(horizon) + "/" + panel)
                    model = construct(name)
                    opt = make_optimizer(model)
                    samples = [(events[i], 10) for i in panels[panel]]
                    if not samples:
                        raise ValueError("empty registered panel: " + panel)
                    for repeat in (0, 1):
                        row = {"structure": name, "horizon_s": horizon, "panel": panel,
                            "repeat": repeat, "status": "RUNNING", "sample_ids": [e["id"] for e, _ in samples],
                            "long_counts": [min(round(horizon*100), len(e["x"])-11) for e, _ in samples],
                            "local_counts": [min(20, len(e["x"])-11) for e, _ in samples]}
                        state["rows"].append(row)
                        check("update " + str(len(state["rows"])))
                        result, seconds = timed("update", lambda: one_update(model, opt, store, samples,
                            horizon, force), name, horizon_s=horizon, panel=panel, repeat=repeat)
                        row.update(status="COMPLETED", seconds=seconds, **result)
                        check("discarded update complete")
                    if horizon == 5. and panel == "unstable":
                        trial_checkpoint(name, model, opt, "filled", 2)
                    del model, opt
        for name in STRUCTURES:
            check("fresh source evaluation " + name)
            model = construct(name)
            fresh_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            evaluation, seconds = timed("full_eval", lambda: full_evaluation(
                model, events, args.output / (name + "_full_train"), check), name)
            state["overhead"][name]["full_eval"] = seconds
            state["evaluations"][name] = evaluation
            _equal(fresh_state, model.state_dict(), exact=True)
            def derivatives():
                with torch.enable_grad():
                    local, arrays = analyze_model(copy.deepcopy(model).cpu(), source_sha256=SOURCE_SHA,
                                                   source_path=str(SOURCE))
                np.savez_compressed(args.output / (name + "_local_derivatives.npz"), **arrays)
                write(args.output / (name + "_local_derivatives.json"), local)
                return {"result_sha256": sha(args.output / (name + "_local_derivatives.json")),
                        "grad_enabled": True, "scope": "cost only; no stability claim"}
            local, seconds = timed("jacobian", derivatives, name)
            state["overhead"][name]["jacobian"] = seconds
            state["jacobians"][name] = local
            # Include loading/preparation for each future independently started
            # model. This is an estimate, not a second measured interval.
            state["overhead"][name]["construction"] = max(constructions[name]) + global_prepare
            state["overhead"][name]["construction_only_max"] = max(constructions[name])
            state["overhead"][name]["global_prepare_allowance"] = global_prepare
            check("evaluation and local derivatives complete " + name)
            del model
        if code != dependency_hashes() or sha(SOURCE) != SOURCE_SHA:
            raise RuntimeError("frozen source/code changed during real measurement")
        state["maxima"] = {s: {str(int(h)): max(r["seconds"] for r in state["rows"] if
            r["structure"] == s and r["horizon_s"] == h) for h in HORIZONS} for s in STRUCTURES}
        assert len(state["rows"]) == 36 and all(r["status"] == "COMPLETED" for r in state["rows"])
        state.update(status="WORKLOAD_COMPLETED_AWAITING_NATIVE_FINALIZE", workload_complete=True,
                     cost_gate="UNRESOLVED", source_sha256=SOURCE_SHA, public_terms_sha256=TERMS_SHA,
                     cohort_sha256=COHORT_SHA, discarded_updates=36,
                     numerical_qualification={s: state["evaluations"][s]["numerical_qualification"] for s in STRUCTURES})
    except BaseException as exc:
        state.update(status="UNRESOLVED", workload_complete=False, cost_gate="UNRESOLVED",
                     error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        state["elapsed_worker_seconds"] = time.perf_counter()-START
        write(args.output / "measurement.json", state)
        write(args.output / "progress.json", state)


def finalize_cost(output, native_receipt_path):
    """After external native supervision exits; no torch/NumPy imports here."""
    output = Path(output).resolve()
    state = json.loads((output / "measurement.json").read_text(encoding="utf-8"))
    native = json.loads(Path(native_receipt_path).read_text(encoding="utf-8"))
    try:
        if state.get("status") != "WORKLOAD_COMPLETED_AWAITING_NATIVE_FINALIZE" or not state.get("workload_complete"):
            raise ValueError("complete unfinalized workload required")
        if state.get("code_sha256") != dependency_hashes():
            raise ValueError("source changed before finalization")
        if (state.get("source_sha256") != SOURCE_SHA or state.get("public_terms_sha256") != TERMS_SHA or
                state.get("cohort_sha256") != COHORT_SHA):
            raise ValueError("registered scientific identity mismatch")
        if (native.get("status") != "COMPLETED" or native.get("includes_child_cost_once") is not True
                or native.get("native_receipt_complete") is not True):
            raise ValueError("complete successful native process-tree receipt required")
        if native.get("mode") != "measure" or Path(native.get("output", "")).resolve() != output:
            raise ValueError("native receipt belongs to another measurement")
        if native.get("actual_worker_pid") != state["pid"] or native.get("exit_code") != 0:
            raise ValueError("native receipt does not confirm this worker exited successfully")
        native_seconds = _seconds(native["seconds"])
        if native_seconds > 6000:
            raise ValueError("native measurement exceeded the common hard cap")
        rows = state["rows"]
        observed = [{k: row[k] for k in ("structure", "horizon_s", "panel", "repeat")} for row in rows]
        if observed != measurement_tasks() or any(r.get("status") != "COMPLETED" for r in rows):
            raise ValueError("36 fixed complete trial updates required")
        if len(state["checkpoints"]) != 4 or any(r["identity_optimizer_rng_coverage"] != "PASS" for r in state["checkpoints"]):
            raise ValueError("four checkpoint/restore checks required")
        if {(r["structure"], r["kind"], r["step"]) for r in state["checkpoints"]} != {
                (s, kind, step) for s in STRUCTURES for kind, step in (("empty", 0), ("filled", 2))}:
            raise ValueError("registered empty/filled checkpoints required")
        if any(state["evaluations"][s]["event_count"] != 366 or not state["jacobians"][s]["grad_enabled"] for s in STRUCTURES):
            raise ValueError("two complete train evaluations and local derivative analyses required")
        for s in STRUCTURES:
            evaluation = state["evaluations"][s]
            if len(evaluation["rows_by_family"]) != 5 or any(v != 1830 for v in evaluation["rows_by_family"].values()):
                raise ValueError("missing train metric rows or changed denominator")
            if sha(output / (s + "_full_train") / "evidence.json") != evaluation["evidence_sha256"]:
                raise ValueError("train evaluation evidence changed")
            if sha(output / (s + "_local_derivatives.json")) != state["jacobians"][s]["result_sha256"]:
                raise ValueError("local derivative evidence changed")
        components = sorted(state["components"], key=lambda row: row["start_offset"])
        expected_counts = {"global_prepare": 1, "construction": 22, "update": 36,
            "save_empty": 2, "save_filled": 2, "restore_empty": 2, "restore_filled": 2,
            "full_eval": 2, "jacobian": 2}
        if {kind: sum(r["kind"] == kind for r in components) for kind in expected_counts} != expected_counts:
            raise ValueError("incomplete mutually exclusive component inventory")
        if any(r["kind"] not in expected_counts for r in components):
            raise ValueError("unregistered component accounting")
        previous = -1.
        for row in components:
            seconds = _seconds(row["seconds"])
            if (row["start_offset"] < 0 or row["start_offset"] < previous or
                    row["end_offset"] < row["start_offset"] or row["end_offset"] > native_seconds):
                raise ValueError("overlapping or invalid measured components")
            if not math.isclose(row["end_offset"]-row["start_offset"], seconds, abs_tol=1e-6):
                raise ValueError("inconsistent measured component duration")
            previous = row["end_offset"]
        allocated = sum(row["seconds"] for row in components)
        envelope = native_seconds - allocated
        if not math.isfinite(envelope) or envelope < 0:
            raise ValueError("native unallocated startup/exit envelope cannot be determined nonnegatively")
        actual_maxima = {s: {str(int(h)): max(r["seconds"] for r in rows if
            r["structure"] == s and r["horizon_s"] == h) for h in HORIZONS} for s in STRUCTURES}
        if actual_maxima != state["maxima"]:
            raise ValueError("update maxima do not match every retained raw trial")
        global_prepare = next(r["seconds"] for r in components if r["kind"] == "global_prepare")
        for s in STRUCTURES:
            o = state["overhead"][s]
            for kind in OVERHEAD_KEYS:
                if kind == "construction":
                    expected = max(r["seconds"] for r in components if r["kind"] == kind and r["structure"] == s) + global_prepare
                else:
                    items = [r["seconds"] for r in components if r["kind"] == kind and r["structure"] == s]
                    if len(items) != 1:
                        raise ValueError("ambiguous measured overhead component")
                    expected = items[0]
                if o[kind] != expected:
                    raise ValueError("future-model cost differs from measured component")
        schedules = json.loads((output / "schedule.json").read_text(encoding="utf-8"))
        frozen = json.loads((output / "frozen_identity.json").read_text(encoding="utf-8"))
        if (frozen["source_sha256"] != SOURCE_SHA or frozen["public_terms_sha256"] != TERMS_SHA or
                frozen["cohort"]["sha256"] != COHORT_SHA or frozen["cohort"]["count"] != 366 or
                frozen["code_sha256"] != state["code_sha256"] or not frozen["train_only"] or
                frozen["val_read"] or frozen["test_read"]):
            raise ValueError("incomplete frozen identity")
        event_lengths = {e["id"]: e["length"] for e in frozen["events"]}
        if len(frozen["events"]) != 366 or len(event_lengths) != 366 or any(n < 12 for n in event_lengths.values()):
            raise ValueError("invalid frozen event pool")
        if frozen["checkpoint_steps"] != checkpoint_steps() or frozen["milestones"] != [256, 512, 1024]:
            raise ValueError("registered checkpoints/milestones changed")
        if set(schedules) != {str(seed) for seed in SEEDS}:
            raise ValueError("three real registered schedules required")
        for schedule in schedules.values():
            schedule_rows = schedule["rows"]
            if len(schedule_rows) != 1024 or [r["step"] for r in schedule_rows] != list(range(1, 1025)):
                raise ValueError("complete real 1024-step schedule required")
            actual_horizons = [float(r["horizon_s"]) for r in schedule_rows]
            expected_horizons = [2. if step <= 100 else (1., 2., 5.)[(step-101) % 3]
                                 for step in range(1, 1025)]
            if actual_horizons != expected_horizons or [actual_horizons.count(h) for h in HORIZONS] != [308, 408, 308]:
                raise ValueError("real horizon schedule changed")
            if schedule["eligible_pool_count"] != 366 or schedule["included_ids"] != sorted(event_lengths):
                raise ValueError("schedule does not cover the complete legal pool")
            seen_ids = set()
            for row in schedule_rows:
                sampling = row["sampling"]
                ids, endpoints = sampling["sample_ids"], sampling["endpoints"]
                if len(ids) != len(endpoints) or len(ids) != 12 or any(eid not in event_lengths for eid in ids):
                    raise ValueError("illegal scheduled samples")
                if any(not 10 <= i < event_lengths[eid]-1 for eid, i in zip(ids, endpoints)):
                    raise ValueError("illegal scheduled endpoints")
                if row["long_counts"] != [min(round(row["horizon_s"]*100), event_lengths[eid]-11) for eid in ids]:
                    raise ValueError("scheduled genuine long counts changed")
                if row["local_counts"] != [min(20, event_lengths[eid]-i-1) for eid, i in zip(ids, endpoints)]:
                    raise ValueError("scheduled genuine local counts changed")
                for name in ("coverage_before", "coverage_after"):
                    coverage = sampling[name]
                    if sorted(coverage["cover_order"]) != list(range(366)) or not 0 <= coverage["cursor"] <= 366:
                        raise ValueError("invalid scheduled coverage state")
                if digest(row["numpy_rng_before"]) != sampling["numpy_rng_before_sha256"] or digest(row["numpy_rng_after"]) != sampling["numpy_rng_after_sha256"]:
                    raise ValueError("scheduled sampler RNG state changed")
                multiplier = .3 if row["step"] > 512 else 1.
                if row["lr_multiplier"] != multiplier or row["rates"] != {k: v*multiplier for k, v in frozen["rates"].items()}:
                    raise ValueError("scheduled learning rates changed")
                seen_ids.update(ids)
            if seen_ids != set(event_lengths):
                raise ValueError("not every train event is covered by the schedule")
        if sha(output / "schedule.json") != state["schedule_sha256"] or sha(output / "frozen_identity.json") != state["frozen_identity_sha256"]:
            raise ValueError("frozen local evidence changed")
        if sha(SOURCE_PATH) != SOURCE_SHA or digest(json.loads(MAIN.read_text(encoding="utf-8"))["terms"]) != TERMS_SHA:
            raise ValueError("registered source/term files changed before finalization")
        cost = compute_cost(state["maxima"], state["overhead"], schedules, envelope)
        gate = "WITHIN_COST_CAP" if cost["within_cap_arithmetic"] else "EXCEEDS_COST_CAP"
        final = {**state, **cost, "status": "COMPLETED", "native_seconds": native_seconds,
                 "cost_gate": gate,
                 "scope": "complete registered train engineering cost; no scientific admission",
                 "allocated_component_seconds": allocated, "startup_envelope_seconds": envelope,
                 "native_receipt_sha256": sha(native_receipt_path),
                 "engineering_cost_executable": gate == "WITHIN_COST_CAP",
                 "scientific_admission": "NOT_AUTHORIZED"}
    except BaseException as exc:
        final = {**state, "status": "UNRESOLVED", "cost_gate": "UNRESOLVED", "finalization_error": repr(exc)}
        write(output / "final_cost.json", final)
        raise
    write(output / "final_cost.json", final)
    return final


def supervise(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), args.mode, "--worker", "--output", str(args.output),
               "--max-seconds", str(args.max_seconds)]
    if args.qa_record is not None:
        command += ["--qa-record", str(args.qa_record.resolve())]
    cleanup = {}
    receipt = {"status": "FAILED", "mode": args.mode, "pid": os.getpid(), "output": str(args.output),
               "command": command, "includes_child_cost_once": True, "category": "QA" if args.mode == "qa" else "MEASURE"}
    try:
        with (args.output / "stdout.log").open("w") as out, (args.output / "stderr.log").open("w") as err:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
            receipt["launcher_pid"] = proc.pid
            code = wait_owned(proc, max(.1, args.max_seconds-(time.perf_counter()-START)-5), cleanup)
        receipt["exit_code"] = code
        if code:
            raise RuntimeError("worker exit " + str(code))
        receipt["status"] = "COMPLETED"
    except BaseException as exc:
        receipt["error"] = repr(exc)
    finally:
        worker = args.output / "worker.json"
        if worker.exists():
            receipt["actual_worker_pid"] = json.loads(worker.read_text(encoding="utf-8"))["pid"]
        receipt.update(seconds=time.perf_counter()-START, cap_seconds=args.max_seconds, cleanup=cleanup,
                       native_receipt_complete=False, val_read=False, test_read=False)
        write(args.output / "receipt.json", receipt)
    if receipt["status"] != "COMPLETED":
        raise RuntimeError(receipt.get("error", "supervisor failure"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qa", "measure"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", required=True, type=float)
    parser.add_argument("--qa-record", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        validate_budget(args.mode, args.max_seconds)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "measure" and args.qa_record is None:
        parser.error("measure requires the frozen --qa-record")
    if not args.worker:
        return supervise(args)
    args.output = args.output.resolve()
    write(args.output / "worker.json", {"pid": os.getpid(), "parent_pid": os.getppid(), "argv": sys.argv})
    load_runtime()
    if args.mode == "qa":
        sys.path.insert(0, str(ROOT / "tests"))
        from test_linear_cache_cost_gate_20261019 import run_qa
        run_qa(args.output)
    else:
        measure(args)


if __name__ == "__main__":
    main()
