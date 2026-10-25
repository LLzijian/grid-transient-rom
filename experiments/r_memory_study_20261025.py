"""Finite train-only A/B/C memory study; native ownership belongs to its caller.

Importing this module never opens a dataset or starts an optimizer. Historical
scientific runners and their immutable outputs are not modified. There are no
process-count or elapsed-time admission limits in this prospective runner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import uuid

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import numpy as np
import torch

from closure_training_runtime import (TensorEventStore, TrainingClosureROM,
    all_finite, resolve_device, rollout_errors, synchronize)
from train_local_memory_0098 import graph, risk
from train_structured_rom import _point_error

MODES = ("A", "B", "C")
SEEDS = (25101, 25102)
MILESTONES = (0, 512, 2048)
MULTIPLIERS = (1, 3)
RATES = {"skeleton": .0009, "closure": .00009, "encoder": .000027}
SOURCE = ROOT / "private_data/closure_mechanism_20261019/P2/quartic_memory01/models/6101_N/checkpoints/step001024/model.json"
SOURCE_SHA = "c926caa685f2105d908c5d642f5944cc78f467744b3f9bd24831085eeca700a5"
FORCE_SCALE = 1.0068379869405584
DATA_BASE = ROOT / "private_data/closure_mechanism_20261019/P2/quartic_memory01"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(safe(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(safe(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
        return
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_source():
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("registered N6101 source identity changed")
    return SOURCE


def code_identity():
    names = ("r_memory_study_20261025.py", "r_memory_model_20261025.py",
        "closure_training_runtime.py", "symbolic_closure_rom.py", "structured_rom.py",
        "local_memory_rom_0098.py", "train_closure.py", "train_local_memory_0098.py",
        "train_structured_rom.py", "train_joint_numerical_pair_0100.py",
        "quartic_memory_20261020.py", "active_fom_pair_fit_20261020.py",
        "fixed_edge_pair_20261019.py", "summarize_train_local_memory_0098.py",
        "evaluate_structured_rom.py", "closure_discovery_confirmation.py",
        "linear_cache_runtime_20261019.py", "frequency_dissipation_20261021.py",
        "closure_discovery_data.py", "closure_mechanism_study.py", "train_joint_campaign.py")
    paths = [ROOT / "experiments" / name for name in names]
    paths.extend(ROOT / name for name in ("src/kiki/data/private_dataset.py",
        "src/kiki/models/private_pm.py", "src/kiki/models/private_rom.py"))
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def runtime_identity(device):
    executable = Path(sys.executable).resolve()
    return {"executable": str(executable), "executable_sha256": sha(executable),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": "float64", "threads": torch.get_num_threads()}


def no_update_group(group):
    payload = ("r_memory_20261025|" + str(group)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % 5 == 0


def partition_events(events):
    fit, held = [], []
    for event in sorted(events, key=lambda e: e["id"]):
        (held if no_update_group(event["group"]) else fit).append(event)
    if not fit or not held:
        raise ValueError("registered group partition is empty or full; no automatic redraw")
    if {str(e["group"]) for e in fit} & {str(e["group"]) for e in held}:
        raise ValueError("event group partition overlaps")
    return fit, held


def panel_events(events):
    """Fixed group-round-robin panel, targeting nine stable and three unstable."""
    chosen = []
    for label, target in (("unstable", 3), ("stable", 9)):
        groups = sorted({str(e["group"]) for e in events if e["label"] == label})
        pools = [[e for e in sorted(events, key=lambda e: e["id"])
                  if e["label"] == label and str(e["group"]) == group] for group in groups]
        count = 0
        for offset in range(max((len(p) for p in pools), default=0)):
            for pool in pools:
                if offset < len(pool) and count < target:
                    chosen.append(pool[offset]); count += 1
    used = {e["id"] for e in chosen}
    for event in sorted(events, key=lambda e: e["id"]):
        if len(chosen) >= 12:
            break
        if event["id"] not in used:
            chosen.append(event); used.add(event["id"])
    if len(chosen) != 12:
        raise ValueError("fit pool cannot supply the registered twelve-event panel")
    return chosen


def horizon_for(step):
    if type(step) is not int or not 1 <= step <= 2048:
        raise ValueError("formal step must be an integer in 1..2048")
    return 1. if step <= 256 else (2. if step <= 512 else (5. if step % 2 else 8.))


def make_schedule(events, seed, updates=2048):
    from train_joint_numerical_pair_0100 import next_samples
    if seed not in SEEDS or updates != 2048:
        raise ValueError("only registered seed and 2048-update schedule")
    rng = np.random.default_rng(seed)
    coverage = {"cover_order": rng.permutation(len(events)).tolist(), "cursor": 0}
    initial = {"rng": copy.deepcopy(rng.bit_generator.state), "coverage": copy.deepcopy(coverage)}
    rows = []
    for step in range(1, updates + 1):
        before = digest(rng.bit_generator.state)
        samples, sampling = next_samples(events, rng, coverage)
        h = horizon_for(step)
        rows.append({"step": step, "horizon_s": h, "sampling_sha256": digest(sampling),
            "sample_ids": sampling["sample_ids"], "endpoints": sampling["endpoints"],
            "rng_before_sha256": before, "rng_after_sha256": digest(rng.bit_generator.state),
            "coverage_after_sha256": digest(coverage),
            "long_counts": [min(round(h * 100), len(e["x"]) - 11) for e, _ in samples],
            "local_counts": [min(20, len(e["x"]) - i - 1) for e, i in samples]})
    if {eid for row in rows for eid in row["sample_ids"]} != {e["id"] for e in events}:
        raise ValueError("frozen schedule failed full fit-event coverage")
    return {"initial": initial, "rows": rows}


def _save_event_cache(path, events):
    arrays, metadata = {}, []
    for index, event in enumerate(events):
        prefix = f"event_{index:04d}"
        arrays[prefix + "_x"] = np.asarray(event["x"], dtype=np.float64)
        arrays[prefix + "_torque"] = np.asarray(event["torque"], dtype=np.float64)
        metadata.append({k: safe(v) for k, v in event.items() if k not in ("x", "torque")})
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, allow_nan=False))
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _load_event_cache(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError("prepared train cache identity changed")
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata_json"]))
        return [{**event, "x": cached[f"event_{i:04d}_x"].copy(),
                 "torque": cached[f"event_{i:04d}_torque"].copy()} for i, event in enumerate(metadata)]


def prepare(base):
    from quartic_memory_20261020 import load_events
    from active_fom_pair_fit_20261020 import event_identity
    require_source()
    base = Path(base).resolve(); base.mkdir(parents=True, exist_ok=True)
    if (base / "prepare.json").exists():
        raise FileExistsError("study already prepared")
    events, excluded = load_events()
    if len(events) != 385 or sum(e["label"] == "stable" for e in events) != 342:
        raise ValueError("locked common385 train cohort changed")
    fit, held = partition_events(events)
    from frequency_dissipation_20261021 import cohort_sets
    legacy_groups = cohort_sets(events)
    prior = read(DATA_BASE / "models/6101_N/checkpoints/step001024/checkpoint.json")
    if event_identity(events) != prior["identity"]["data"]:
        raise ValueError("current train differs from frozen N6101 historical train")
    _save_event_cache(base / "fit_events.npz", fit)
    _save_event_cache(base / "no_update_events.npz", held)
    schedules = {str(seed): make_schedule(fit, seed) for seed in SEEDS}
    write(base / "schedule.json", schedules, exclusive=True)
    record = {"source": str(SOURCE), "source_sha256": SOURCE_SHA, "code": code_identity(),
        "cohort": event_identity(events), "cohort_sha256": digest(event_identity(events)),
        "fit_ids": [e["id"] for e in fit], "no_update_ids": [e["id"] for e in held],
        "fit_groups": sorted({str(e["group"]) for e in fit}),
        "no_update_groups": sorted({str(e["group"]) for e in held}),
        "source_already_exposed_to_all_groups": True, "exclusions": excluded,
        "exclusions_scope": "exact385-wrapper returned exclusions; earlier366 exclusions retain their original archived records",
        "fit_cache_sha256": sha(base / "fit_events.npz"),
        "no_update_cache_sha256": sha(base / "no_update_events.npz"),
        "schedule_sha256": sha(base / "schedule.json"),
        "calibration_panel": [e["id"] for e in panel_events(fit)],
        "original366_ids": sorted(legacy_groups["original366"]),
        "fresh19_ids": sorted(legacy_groups["new19"]),
        "force_scale": FORCE_SCALE, "val_read": False, "test_read": False}
    write(base / "prepare.json", record, exclusive=True)
    return record


def checked_preparation(base, *, fit_only=True):
    base = Path(base).resolve(); record = read(base / "prepare.json")
    require_source()
    if record["source_sha256"] != SOURCE_SHA or record["code"] != code_identity():
        raise ValueError("prepared source/code binding changed")
    if sha(base / "schedule.json") != record["schedule_sha256"]:
        raise ValueError("frozen schedule changed")
    fit = _load_event_cache(base / "fit_events.npz", record["fit_cache_sha256"])
    if [e["id"] for e in fit] != record["fit_ids"]:
        raise ValueError("fit cache event identity changed")
    if fit_only:
        return record, fit
    held = _load_event_cache(base / "no_update_events.npz", record["no_update_cache_sha256"])
    if [e["id"] for e in held] != record["no_update_ids"]:
        raise ValueError("no-update cache event identity changed")
    return record, sorted(fit + held, key=lambda e: e["id"])


def build_model(mode, seed, device="cuda"):
    if mode not in MODES or seed not in SEEDS:
        raise ValueError("unregistered model or optimization seed")
    require_source()
    source = TrainingClosureROM.from_export(SOURCE)
    if mode == "A":
        from linear_cache_runtime_20261019 import enable_linear_cache
        model = enable_linear_cache(source)
    else:
        from r_memory_model_20261025 import RMemoryROM
        model = RMemoryROM.from_source(source, mode=mode, seed=seed,
            source_identity={"path": str(SOURCE), "sha256": SOURCE_SHA})
    return model.to(resolve_device(device)).double()


def latent_scale(model, mode):
    if mode == "A":
        return model.mass.new_full((10,), .1)
    if mode not in ("B", "C"):
        raise ValueError("unregistered model")
    rstar = torch.as_tensor(model.rstar, dtype=model.mass.dtype, device=model.mass.device)
    if rstar.numel() not in (1, 10) or not torch.isfinite(rstar).all() or torch.any(rstar <= 0):
        raise ValueError("fixed positive physical rstar required")
    return .1 * rstar


def objective(model, mode, store, samples, horizon, force_scale=FORCE_SCALE):
    long_batch = store.prepare([(e, 10) for e, _ in samples], horizon)
    local_batch = store.prepare(samples, .2)
    history = local_batch.history
    long = rollout_errors(model, long_batch)
    state = torch.cat((history[:, -1], model.encode(history)), -1)
    local = rollout_errors(model, local_batch, initial=state).mean()
    latent = ((model.rk4(state, .01)[:, 30:] - model.encode(store.histories(samples, adjacent=True)))
              / latent_scale(model, mode)).square().mean()
    predicted = state[:, 20:30] - model.mass * model.field(state)[:, 10:20]
    target = torch.stack([store.events[e["id"]][1][i] for e, i in samples])
    torque = ((predicted - target) / force_scale).square().mean()
    deploy = 10 * risk(long)
    auxiliary = local + .1 * latent + .1 * torque
    terms = {"risk": risk(long), "mean": long.mean(), "local": local,
        "latent_consistency": latent, "torque": torque, "deploy_loss": deploy,
        "auxiliary_loss": auxiliary, "total_loss": deploy + auxiliary}
    return deploy + auxiliary, terms


def make_optimizer(model, mode, multiplier=1):
    if mode not in MODES or multiplier not in MULTIPLIERS:
        raise ValueError("unregistered optimizer calibration")
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "gru" in name or "head" in name:
            block = "encoder"
        elif mode == "A":
            block = "closure" if name in ("coefficients", "_template_scalars.3", "_template_scalars.4") else "skeleton"
        elif name.startswith("structured."):
            block = "skeleton"
        else:
            block = "closure"
        groups.setdefault(block, []).append(parameter)
    return torch.optim.Adam([{"params": params, "block": block,
        "lr": RATES[block] * multiplier} for block, params in groups.items()])


def one_update(model, mode, optimizer, store, samples, horizon, *, call_journal=None):
    device = model.mass.device; synchronize(device); started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with graph(model):
        loss, terms = objective(model, mode, store, samples, horizon)
        if not all_finite([loss]):
            raise FloatingPointError("nonfinite training objective")
        loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not all_finite([p.grad for p in parameters]):
        raise FloatingPointError("nonfinite training gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 100.)
    if not all_finite([norm]):
        raise FloatingPointError("nonfinite gradient norm")
    if call_journal:
        with Path(call_journal).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "optimizer_step_attempt", "pid": os.getpid()}) + "\n")
    before = {id(p): p.detach().clone() for group in optimizer.param_groups for p in group["params"]}
    optimizer.step(); synchronize(device)
    if not all_finite(parameters):
        raise FloatingPointError("nonfinite parameter after Adam")
    for state in optimizer.state.values():
        if not all_finite([v for v in state.values() if torch.is_tensor(v)]):
            raise FloatingPointError("nonfinite Adam state")
    if call_journal:
        with Path(call_journal).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "optimizer_step_completed", "pid": os.getpid()}) + "\n")
    relative = {}
    for group in optimizer.param_groups:
        old_norm = torch.stack([before[id(p)].square().sum() for p in group["params"]]).sum().sqrt()
        change = torch.stack([(p.detach() - before[id(p)]).square().sum() for p in group["params"]]).sum().sqrt()
        relative[group["block"]] = {"old_l2": float(old_norm), "change_l2": float(change),
            "relative_l2": float(change / old_norm) if float(old_norm) > 0 else None}
    return {**{k: float(v.detach()) for k, v in terms.items()}, "block_updates": relative,
        "gradient_norm": float(norm), "seconds": time.perf_counter() - started}


def _parameter_layout(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [{"block": group["block"], "lr": group["lr"],
             "names": [names[id(p)] for p in group["params"]]} for group in optimizer.param_groups]


def _cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def quarantine_failure(output, model, optimizer, rng, coverage, step, trace, error):
    """Preserve fault-time state, including NaNs; never publish a resume pointer."""
    folder = Path(output).resolve() / "QUARANTINED_NOT_RESUMABLE" / uuid.uuid4().hex
    record = {"status": "CAPTURE_FAILED", "resumable": False, "step": step,
        "trace_completed_updates": len(trace), "original_error": repr(error),
        "original_traceback": traceback.format_exc() if sys.exc_info()[0] is not None else None,
        "directory": str(folder),
        "payload_saved": False, "payload_sha256": None, "capture_error": None}
    try:
        folder.mkdir(parents=True, exist_ok=False)
        payload = {"resumable": False, "step": step, "trace": copy.deepcopy(trace),
            "model_state": _cpu_snapshot(model.state_dict()),
            "gradients": {name: None if parameter.grad is None else _cpu_snapshot(parameter.grad)
                for name, parameter in model.named_parameters()},
            "Adam": _cpu_snapshot(optimizer.state_dict()),
            "numpy_generator_state": copy.deepcopy(rng.bit_generator.state) if rng is not None else None,
            "numpy_generator_in_use": rng is not None, "coverage": copy.deepcopy(coverage),
            "numpy_global_rng": np.random.get_state(), "python_rng": random.getstate(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
        torch.save(payload, folder / "fault_state.pt")
        record.update(status="CAPTURED", payload_saved=True,
            payload_sha256=sha(folder / "fault_state.pt"))
    except BaseException as capture_error:
        record["capture_error"] = repr(capture_error)
        record["capture_traceback"] = traceback.format_exc()
    try:
        write(folder / "record.json", record, exclusive=True)
    except BaseException as record_error:
        record["capture_record_error"] = repr(record_error)
    return record


def save_checkpoint(output, model, mode, optimizer, rng, coverage, step, identity, trace):
    """Publish a real nonempty checkpoint without replacing any prior endpoint."""
    from train_closure import export_cpu
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    final = output / f"step{step:06d}"
    if final.exists():
        pointer = output / "latest.json"
        if pointer.exists() and read(pointer)["directory"] == final.name:
            raise FileExistsError("active immutable checkpoint already exists")
        orphan = output / ("orphan-" + final.name + "-" + uuid.uuid4().hex)
        if final.resolve().parent != output.resolve() or orphan.resolve().parent != output.resolve():
            raise ValueError("orphan preservation must stay within checkpoint output")
        final.rename(orphan)
    pending = output / (".pending-" + uuid.uuid4().hex); pending.mkdir()
    export_cpu(model, pending / "model.json")
    state = {"optimizer": optimizer.state_dict(), "numpy_rng": copy.deepcopy(rng.bit_generator.state),
        "numpy_global_rng": np.random.get_state(), "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if model.mass.device.type == "cuda" else None,
        "coverage": copy.deepcopy(coverage), "step": step}
    torch.save(state, pending / "training.pt")
    metadata = {"mode": mode, "step": step, "identity": identity,
        "model_sha256": sha(pending / "model.json"), "training_sha256": sha(pending / "training.pt"),
        "runtime": runtime_identity(model.mass.device), "trace": trace,
        "parameter_layout": _parameter_layout(model, optimizer),
        "requires_grad": [name for name, p in model.named_parameters() if p.requires_grad],
        "val_read": False, "test_read": False, "scope": "research_candidate"}
    write(pending / "checkpoint.json", metadata, exclusive=True)
    pending.rename(final)
    write(output / "latest.json", {"directory": final.name,
        "checkpoint_sha256": sha(final / "checkpoint.json")})
    return final


def load_export(path, mode, device="cpu"):
    if mode == "A":
        from linear_cache_runtime_20261019 import enable_linear_cache
        return enable_linear_cache(TrainingClosureROM.from_export(path)).to(resolve_device(device)).double()
    if mode not in ("B", "C"):
        raise ValueError("unregistered exported model")
    from r_memory_model_20261025 import RMemoryROM
    return RMemoryROM.from_export(path).to(resolve_device(device)).double()


def restore_checkpoint(output, device, expected_identity, *, multiplier=1):
    output = Path(output); pointer = read(output / "latest.json")
    directory = pointer["directory"]
    if Path(directory).name != directory:
        raise ValueError("checkpoint pointer must name an immediate directory")
    folder = output / directory
    if sha(folder / "checkpoint.json") != pointer["checkpoint_sha256"]:
        raise ValueError("checkpoint metadata identity changed")
    meta = read(folder / "checkpoint.json")
    if meta["identity"] != expected_identity:
        raise ValueError("source/data/code/seed identity changed")
    for filename in ("model.json", "training.pt"):
        if sha(folder / filename) != meta[filename.split(".")[0] + "_sha256"]:
            raise ValueError("checkpoint artifact identity changed")
    device = resolve_device(device)
    if meta["runtime"] != runtime_identity(device):
        raise ValueError("resume runtime changed; no silent numerical continuation")
    model = load_export(folder / "model.json", meta["mode"], device)
    optimizer = make_optimizer(model, meta["mode"], multiplier)
    required = [name for name, p in model.named_parameters() if p.requires_grad]
    if required != meta["requires_grad"]:
        raise ValueError("restored trainable parameter layout differs")
    state = torch.load(folder / "training.pt", map_location="cpu", weights_only=False)
    if state["step"] != meta["step"]:
        raise ValueError("model and Adam update counts differ")
    # Validate positional Adam names before copying its moments.
    current = _parameter_layout(model, optimizer)
    if [g["names"] for g in current] != [g["names"] for g in meta["parameter_layout"]]:
        raise ValueError("restored Adam parameter names/order differ")
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            moment = optimizer.state.get(parameter)
            if meta["step"] and (not moment or int(moment["step"]) != meta["step"]):
                raise ValueError("Adam counter is not the persisted real update count")
            if moment and not all_finite([v for v in moment.values() if torch.is_tensor(v)]):
                raise FloatingPointError("restored Adam contains nonfinite state")
    rng = np.random.default_rng(); rng.bit_generator.state = state["numpy_rng"]
    np.random.set_state(state["numpy_global_rng"]); random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        if state["cuda_rng"] is None:
            raise ValueError("CUDA RNG missing from CUDA recovery")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return model, optimizer, rng, state["coverage"], meta


def calibration_score(model, panel):
    """Five-second genuine-interval risk with existing real-system grid checks."""
    from evaluate_structured_rom import _rollouts, _pair_row, _scored_numerics, _needs_finer
    candidate = copy.deepcopy(model).cpu().double().eval()
    histories = np.stack([e["x"][:11] for e in panel])
    fine = _rollouts(candidate, histories, 1000, .005)
    coarse = _rollouts(candidate, histories, 500, .01)
    mass = candidate.mass.detach().numpy(); values, rows = [], []
    for event, history, f, c in zip(panel, histories, fine, coarse):
        n = min(500, len(event["x"]) - 11)
        shadow = {**event, "x": event["x"][:11 + n]}
        pair = _pair_row(f[:2*n + 1], c[:n + 1], history, shadow, event["classification_meta"], mass)
        scored = _scored_numerics(f, c, shadow, mass)
        qualified = not _needs_finer([pair] + [r for r in scored if r["available"]])
        with torch.no_grad():
            value = float(_point_error(torch.as_tensor(f[::2][1:n + 1, :30]),
                torch.as_tensor(event["x"][11:11 + n]), candidate).mean())
        rows.append({"id": event["id"], "genuine_future_steps": n,
            "pair": pair, "scored": scored, "qualified": qualified, "point_error": value})
        values.append(value)
    qualified = all(r["qualified"] for r in rows) and bool(np.isfinite(values).all())
    return {"risk": float(risk(torch.tensor(values, dtype=torch.float64))) if qualified else None,
        "qualified": qualified, "rows": safe(rows), "events": [e["id"] for e in panel],
        "score_scope": "all twelve events, each genuine future up to five seconds"}


def select_multipliers(scores):
    """Registered arithmetic mean for the joint B/C choice; ties prefer one."""
    valid_a, valid_bc = [], []
    for multiplier in MULTIPLIERS:
        a = scores["A"][str(multiplier)]
        if a["qualified"]:
            valid_a.append((a["ratio"], multiplier))
        b, c = (scores[mode][str(multiplier)] for mode in ("B", "C"))
        if b["qualified"] and c["qualified"]:
            valid_bc.append(((b["ratio"] + c["ratio"]) / 2., multiplier))
    a = min(valid_a)[1] if valid_a else None
    bc = min(valid_bc)[1] if valid_bc else None
    return {"A": a, "B": bc, "C": bc}


def calibrate(base, device="cuda"):
    base = Path(base).resolve(); prepared, fit = checked_preparation(base, fit_only=True)
    admission = read(base / "preflight/summary.json")
    if (admission.get("prepare_sha256") != sha(base / "prepare.json")
            or set(admission.get("eligible", {})) != set(MODES)
            or any(type(value) is not bool for value in admission["eligible"].values())):
        raise ValueError("calibration requires the bound per-model preflight eligibility")
    output = base / "calibration"; output.mkdir(exist_ok=False)
    pool = {e["id"]: e for e in fit}
    panel = [pool[eid] for eid in prepared["calibration_panel"]]
    write(output / "panel.json", {"ids": [e["id"] for e in panel],
        "labels": [e["label"] for e in panel], "groups": [e["group"] for e in panel]}, exclusive=True)
    device = resolve_device(device); store = TensorEventStore(panel, device)
    scores, attempts = {}, 0
    for mode in MODES:
        if not admission["eligible"][mode]:
            scores[mode] = {}
            for multiplier in MULTIPLIERS:
                folder = output / f"{mode}_x{multiplier}"; folder.mkdir()
                record = {"mode": mode, "multiplier": multiplier,
                    "qualified": False, "ratio": None, "status": "SKIPPED_PREFLIGHT",
                    "completed_updates": 0, "preflight_eligible": False}
                write(folder / "result.json", record, exclusive=True)
                scores[mode][str(multiplier)] = {key: record[key] for key in
                    ("qualified", "ratio", "status", "completed_updates")}
            continue
        zero = calibration_score(build_model(mode, 25101, device), panel)
        scores[mode] = {}
        for multiplier in MULTIPLIERS:
            folder = output / f"{mode}_x{multiplier}"; folder.mkdir()
            model = build_model(mode, 25101, device)
            optimizer = make_optimizer(model, mode, multiplier)
            trace = []; status = "ISOLATED_ZERO_NUMERICAL"; final = None; quarantine = None
            if zero["qualified"]:
                try:
                    for step in range(1, 17):
                        row = one_update(model, mode, optimizer, store,
                            [(e, 10) for e in panel], 1., call_journal=folder / "optimizer_calls.jsonl")
                        attempts += 1; trace.append({"step": step, **row})
                    final = calibration_score(model, panel)
                    status = "COMPLETED"
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    status = "FAILED"
                    quarantine = quarantine_failure(folder, model, optimizer, None, None,
                        len(trace) + 1, trace, error)
                    write(folder / "failure.json", {"error": repr(error),
                        "traceback": traceback.format_exc(), "quarantine": quarantine})
            qualified = bool(zero["qualified"] and final and final["qualified"])
            if not qualified and quarantine is None:
                quarantine = quarantine_failure(folder, model, optimizer, None, None, len(trace), trace,
                    ValueError("calibration zero/final numerical qualification unresolved"))
            ratio = final["risk"] / max(zero["risk"], 1e-12) if qualified else None
            record = {"mode": mode, "multiplier": multiplier, "zero": zero, "final": final,
                "ratio": ratio, "qualified": qualified, "status": status, "trace": trace,
                "completed_updates": len(trace), "trial_state_discarded_for_formal": True,
                "quarantine": quarantine}
            write(folder / "result.json", record, exclusive=True)
            # Preserve adverse models/Adam too, without adopting them as a source.
            if qualified and all_finite([p for p in model.parameters()]):
                rng = np.random.default_rng(25101)
                save_checkpoint(folder / "checkpoints", model, mode, optimizer, rng, {}, len(trace),
                    {"calibration_only": True, "source_sha256": SOURCE_SHA, "multiplier": multiplier}, trace)
            scores[mode][str(multiplier)] = {k: record[k] for k in ("ratio", "qualified", "status", "completed_updates")}
            write(output / "partial.json", {"scores": scores, "completed_updates": attempts})
    result = {"scores": scores, "selected": select_multipliers(scores),
        "completed_updates": attempts, "maximum_updates": 96,
        "no_update_cache_parsed": False, "no_update_data_used": False,
        "native_identity_checks_may_read_cache_bytes": True,
        "selection": "A minimum ratio; B/C arithmetic mean of ratios, both qualified; tie x1",
        "preflight_eligible": admission["eligible"],
        "preflight_sha256": sha(base / "preflight/summary.json"),
        "prepare_sha256": sha(base / "prepare.json"), "code": code_identity()}
    write(output / "selection.json", result, exclusive=True)
    return result


def noise_history(event, delta_scale, omega_scale):
    """Same deterministic noisy history for every model and optimization seed."""
    history = np.array(event["x"][:11], dtype=np.float64, copy=True)
    payload = ("r_memory_noise_20261025|" + event["id"]).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    scales = np.r_[np.broadcast_to(delta_scale, (10,)), np.broadcast_to(omega_scale, (10,))]
    history[:, :20] += rng.normal(size=(11, 20)) * (.01 * scales)
    return history


def preflight(base, device="cuda"):
    """Three registered real eight-second backwards, never an Adam update."""
    import gc
    base = Path(base).resolve(); prepared, fit_events = checked_preparation(base, fit_only=True)
    output = base / "preflight"; output.mkdir(exist_ok=False)
    panel = sorted((e for e in fit_events if e["label"] == "stable" and len(e["x"]) - 11 >= 800),
        key=lambda e: (-len(e["x"]), e["id"]))[:12]
    if len(panel) != 12:
        result = {"status": "INSUFFICIENT_REAL_8S_PANEL", "actual_optimizer_updates": 0,
            "actual_eligible_events": len(panel), "event_ids": [e["id"] for e in panel],
            "eligible": {mode: False for mode in MODES}, "prepare_sha256": sha(base / "prepare.json")}
        write(output / "summary.json", result, exclusive=True)
        return result
    device = resolve_device(device); rows = []
    for mode in MODES:
        model = store = loss = terms = None
        row = {"mode": mode, "event_ids": [e["id"] for e in panel], "horizon_s": 8.,
            "actual_optimizer_updates": 0, "score_used_for_selection": False,
            "source_before_sha256": sha(SOURCE)}
        started = time.perf_counter()
        try:
            model = build_model(mode, 25101, device)
            store = TensorEventStore(panel, device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                row["resident_allocated_bytes"] = torch.cuda.memory_allocated(device)
                row["resident_reserved_bytes"] = torch.cuda.memory_reserved(device)
            synchronize(device); tick = time.perf_counter()
            model.zero_grad(set_to_none=True)
            with graph(model):
                loss, terms = objective(model, mode, store, [(e, 10) for e in panel], 8.)
                if not all_finite([loss]):
                    raise FloatingPointError("nonfinite real8s preflight objective")
                loss.backward()
            missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
            if missing or not all_finite([p.grad for p in model.parameters() if p.requires_grad]):
                raise FloatingPointError("missing or nonfinite real8s gradient: " + repr(missing))
            if any(getattr(model, name, None) is not None for name in ("_runtime_cache", "_graph_cache", "_linear_cache")):
                raise ValueError("training cache not closed after real8s preflight")
            synchronize(device)
            row.update(status="PASS", synchronized_objective_backward_seconds=time.perf_counter() - tick,
                objective_terms={name: float(value.detach()) for name, value in terms.items()},
                gradient_parameter_count=sum(p.requires_grad for p in model.parameters()))
        except BaseException as error:
            row.update(status="FAILED", error=repr(error), traceback=traceback.format_exc())
        finally:
            row["source_after_sha256"] = sha(SOURCE)
            if row["source_before_sha256"] != row["source_after_sha256"]:
                row.update(status="FAILED", source_identity_changed=True)
            if device.type == "cuda":
                row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            row["total_seconds"] = time.perf_counter() - started
            rows.append(row); write(output / f"{mode}.json", row, exclusive=True)
            model = store = loss = terms = None; gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    eligible = {row["mode"]: row["status"] == "PASS" for row in rows}
    result = {"status": "PASS" if all(eligible.values()) else ("PARTIAL" if any(eligible.values()) else "FAILED"),
        "eligible": eligible,
        "actual_optimizer_updates": 0, "models": rows, "fit_only": True,
        "no_update_cache_parsed": False, "no_update_data_used": False,
        "native_identity_checks_may_read_cache_bytes": True,
        "prepare_sha256": sha(base / "prepare.json"),
        "val_read": False, "test_read": False}
    write(output / "summary.json", result, exclusive=True)
    return result


def _classification_counts(extras, key, qualifications):
    from frequency_dissipation_20261021 import actual15_qualified
    rows = []
    for extra in extras:
        pair = extra[key]
        numeric = qualifications.get(extra["event_id"], False)
        if key == "classification_15s":
            numeric = actual15_qualified(extra["actual_interval15"], pair)
        rows.append({"event_id": extra["event_id"], "label": extra["label"],
            "numerically_admissible": bool(numeric),
            "predicted_los": pair["fine_classification"]["predicted_los"],
            "window_settled": pair["fine_classification"].get("window_settled", False),
            "trigger_index": pair["fine_classification"].get("trigger_index")})
    resolved = [row for row in rows if row["numerically_admissible"]]
    return {"event_count": len(rows), "unstable_count": sum(r["label"] == "unstable" for r in rows),
        "numerical_unresolved": len(rows) - len(resolved),
        "missed_resolved_unstable": sum(r["label"] == "unstable" and not r["predicted_los"] for r in resolved),
        "false_los_resolved_stable": sum(r["label"] == "stable" and r["predicted_los"] for r in resolved),
        "rows": rows}


def group_summary(records, extras, ids):
    from summarize_train_local_memory_0098 import build_summary
    from fixed_edge_pair_20261019 import primary_risk
    ids = set(ids)
    selected = {key: [row for row in rows if row["event_id"] in ids] for key, rows in records.items()}
    detail = [row for row in extras if row["event_id"] in ids]
    qualification = {row["event_id"]: row["scientific_numeric_8s"] for row in detail}
    from frequency_dissipation_20261021 import actual15_qualified
    detail_by_id = {row["event_id"]: row for row in detail}
    horizon_risks = {}
    for h in ("1", "2", "5", "8"):
        scored = [row for row in selected["objective_point_error"]
            if str(row["horizon"]) == h and row["label"] == "stable" and row["available"]]
        unresolved = [row["event_id"] for row in scored if not row.get("finite")
            or not row.get("numerically_admissible") or (h in ("5", "8") and not qualification[row["event_id"]])
            or detail_by_id[row["event_id"]]["coarse_point_errors"].get(h) is None]
        fine_risk = coarse_risk = perturbation = None
        if scored and not unresolved:
            fine_risk = float(risk(torch.tensor([row["mean_point_error"] for row in scored], dtype=torch.float64)))
            coarse_risk = float(risk(torch.tensor([detail_by_id[row["event_id"]]["coarse_point_errors"][h]
                for row in scored], dtype=torch.float64)))
            perturbation = abs(fine_risk - coarse_risk)
        horizon_risks[h] = {"value": fine_risk, "coarse_value": coarse_risk,
            "numerical_score_perturbation": perturbation, "eligible": len(scored),
            "unresolved_ids": unresolved, "definition": "original ceil-top10 risk on stable complete horizon"}
    return {"event_count": len(ids), "summary": build_summary(selected),
        "primary": primary_risk(selected["objective_point_error"], qualification),
        "horizon_risks": horizon_risks,
        "classification_8s": _classification_counts(detail, "classification_8s", qualification),
        "classification_actual15s": _classification_counts(detail, "classification_15s", qualification),
        "all_8s_rows_qualified": bool(detail) and all(qualification.values()),
        "all_actual15s_rows_qualified": bool(detail) and all(actual15_qualified(row["actual_interval15"], row["classification_15s"]) for row in detail),
        "all_required_rows_qualified": bool(detail) and all(qualification.values())
            and all(actual15_qualified(row["actual_interval15"], row["classification_15s"]) for row in detail)}


def evaluate_model(model, events, output, prepared, *, noisy=False):
    """P2 arithmetic with model-agnostic orchestration, no forced class switch.

    evaluate_queue itself forces LinearCacheRuntime and rejects the new family.
    Every numerical/statistical function below is the existing P2 function.
    """
    from evaluate_structured_rom import _rollouts, _pair_row
    from summarize_train_local_memory_0098 import event_metrics
    from fixed_edge_pair_20261019 import scientific_qualification, physical_metrics
    from active_fom_pair_fit_20261020 import interval15, actual_interval_predictions
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    candidate = copy.deepcopy(model).cpu().double().eval()
    events = sorted(events, key=lambda e: e["id"])
    timing = {"initialization_seconds": 0., "initialization_calls": 0, "rollout_seconds": 0.}
    original_encode = candidate.encode

    def timed_encode(history):
        tick = time.perf_counter()
        result = original_encode(history)
        timing["initialization_seconds"] += time.perf_counter() - tick
        timing["initialization_calls"] += 1
        return result

    candidate.encode = timed_encode
    mass = candidate.mass.detach().numpy(); pscale = candidate.p_scale.detach().numpy()
    records = {key: [] for key in ("records", "common_frequency", "normalized_p", "objective_point_error", "numerical")}
    extras, prediction_files = [], []
    start = time.perf_counter()
    for first in range(0, len(events), 16):
        block = events[first:first + 16]
        histories = np.stack([noise_history(e, candidate.input_scale_delta.detach().numpy(),
            candidate.input_scale_omega.detach().numpy()) if noisy else e["x"][:11] for e in block])
        failure = None
        tick = time.perf_counter()
        try:
            fine = _rollouts(candidate, histories, 3000, .005)
            coarse = _rollouts(candidate, histories, 1500, .01)
        except Exception as error:
            failure = {"error": repr(error), "traceback": traceback.format_exc()}
            fine = np.full((len(block), 3001, 40), np.nan)
            coarse = np.full((len(block), 1501, 40), np.nan)
        finally:
            timing["rollout_seconds"] += time.perf_counter() - tick
        path = output / f"predictions_{first:04d}.npz"
        np.savez_compressed(path, event_ids=np.asarray([e["id"] for e in block]),
            histories=histories, fine=fine, coarse=coarse)
        prediction_files.append({"path": str(path.resolve()), "sha256": sha(path)})
        for event, history, f, c in zip(block, histories, fine, coarse):
            metrics = event_metrics(candidate, event, f[:1601], c[:801])
            for key, rows in metrics.items():
                records[key].extend(rows)
            pair8 = _pair_row(f[:1601], c[:801], history, event, event["classification_meta"], mass)
            actual_f, actual_c, n = actual_interval_predictions(event, f, c)
            pair15 = _pair_row(actual_f, actual_c, history, event, event["classification_meta"], mass)
            pair15.update(scope="actual_interval_up_to_15s", observed_future_steps=n,
                observed_future_seconds=n * .01)
            coarse_points = {}
            for h in ("1", "2", "5", "8"):
                steps = int(h) * 100
                if len(event["x"]) - 11 < steps or not np.isfinite(c[:steps + 1]).all():
                    coarse_points[h] = None
                else:
                    with torch.no_grad():
                        value = float(_point_error(torch.as_tensor(c[1:steps + 1, :30]),
                            torch.as_tensor(event["x"][11:11 + steps]), candidate).mean())
                    coarse_points[h] = value if np.isfinite(value) else None
            extras.append({"event_id": event["id"], "group": event["group"], "label": event["label"],
                "exception": failure, "classification_8s": pair8, "classification_15s": pair15,
                "coarse_point_errors": coarse_points,
                "actual_interval15": interval15(candidate, event, f, c),
                "scientific_numeric_8s": scientific_qualification(pair8, metrics["numerical"]),
                "physical_1_2_5_8s": physical_metrics(f[::2], event, mass, pscale)})
        write(output / "partial.json", {"rows": records, "extras": extras})
    all_ids = {e["id"] for e in events}
    original_ids = {e["id"] for e in events if e["id"] not in prepared.get("fresh19_ids", [])}
    scopes = {"common385" if not noisy else "no_update_noisy": all_ids,
        "fit": all_ids & set(prepared["fit_ids"]),
        "no_update": all_ids & set(prepared["no_update_ids"]),
        "original366": original_ids, "fresh19": all_ids - original_ids}
    timing["integration_and_rollout_overhead_seconds"] = timing["rollout_seconds"] - timing["initialization_seconds"]
    result = {"event_count": len(events), "rows": records, "extras": extras,
        "profiles": {name: group_summary(records, extras, ids) for name, ids in scopes.items() if ids},
        "prediction_files": prediction_files, "seconds": time.perf_counter() - start,
        "timing": timing,
        "timing_scope": "CPU double one-thread; rollout includes initialization, RK4 and output construction; total includes metrics and files",
        "complexity": candidate.complexity_counts(),
        "noisy": noisy, "truth_unchanged": True, "val_read": False, "test_read": False,
        "helper_sources": {name: sha(ROOT / "experiments" / name) for name in (
            "evaluate_structured_rom.py", "summarize_train_local_memory_0098.py",
            "fixed_edge_pair_20261019.py", "active_fom_pair_fit_20261020.py",
            "closure_discovery_confirmation.py")}}
    write(output / "evidence.json", result, exclusive=True)
    write(output / "complete.json", {"status": "COMPLETED", "event_count": len(events),
        "evidence_sha256": sha(output / "evidence.json")}, exclusive=True)
    return result


def endpoint_folder(base, mode, seed, step):
    if mode not in MODES or seed not in SEEDS or step not in MILESTONES:
        raise ValueError("unregistered scientific endpoint")
    return Path(base) / "models" / f"{mode}_{seed}" / "checkpoints" / f"step{step:06d}"


def evaluate_endpoint(base, mode, seed, step, *, resume_partial=False):
    base = Path(base).resolve(); prepared, events = checked_preparation(base, fit_only=False)
    folder = endpoint_folder(base, mode, seed, step)
    meta = read(folder / "checkpoint.json")
    if (meta["step"] != step or meta["mode"] != mode or meta["identity"]["seed"] != seed
            or meta["identity"]["mode"] != mode or meta["identity"]["source_sha256"] != SOURCE_SHA):
        raise ValueError("endpoint role mismatch")
    if sha(folder / "model.json") != meta["model_sha256"]:
        raise ValueError("endpoint model changed")
    if sha(folder / "training.pt") != meta["training_sha256"]:
        raise ValueError("endpoint training state changed")
    frozen = {"mode": mode, "seed": seed, "step": step,
        "model_path": str((folder / "model.json").resolve()), "model_sha256": meta["model_sha256"],
        "training_path": str((folder / "training.pt").resolve()), "training_sha256": meta["training_sha256"],
        "checkpoint_sha256": sha(folder / "checkpoint.json"),
        "configuration_sha256": digest(meta["identity"]),
        "prepared_sha256": sha(base / "prepare.json")}
    evaluation_root = base / "evaluations"
    resolved_root = evaluation_root.resolve()
    if resolved_root.parent != base:
        raise ValueError("evaluation directory must resolve inside the study base")
    output = evaluation_root / f"{mode}_{seed}_step{step:06d}"
    if output.resolve() != resolved_root / output.name:
        raise ValueError("evaluation output is an aliased or escaping path")
    if (output / "complete.json").exists():
        freeze = read(output / "frozen.json")
        complete = read(output / "complete.json")
        if freeze != frozen or complete["evidence_sha256"] != sha(output / "science/evidence.json"):
            raise ValueError("completed evaluation identity changed")
        return read(output / "science/evidence.json")
    if output.exists():
        if not resume_partial:
            raise FileExistsError("partial evaluation needs the single root-authorized fit resume")
        if read(output / "frozen.json") != frozen:
            raise ValueError("partial evaluation frozen source/configuration differs")
        run = base / "models" / f"{mode}_{seed}"
        authorization = read(run / "resume_once.json")
        expected_pointer = {"directory": folder.name, "checkpoint_sha256": frozen["checkpoint_sha256"]}
        if authorization.get("source") != expected_pointer:
            raise ValueError("partial evaluation is not the authorized persisted resume endpoint")
        # A completed science queue is never repeated, including numerical failures.
        if (output / "science/complete.json").exists():
            complete = read(output / "science/complete.json")
            if complete["evidence_sha256"] != sha(output / "science/evidence.json"):
                raise ValueError("published science completion identity changed")
            write(output / "complete.json", {"status": "COMPLETED",
                "evidence_sha256": complete["evidence_sha256"],
                "reconciled_science_completion_without_reevaluation": True}, exclusive=True)
            return read(output / "science/evidence.json")
        preserved = evaluation_root / (output.name + "_interrupted1")
        if preserved.exists() or preserved.resolve() != resolved_root / preserved.name:
            raise ValueError("the unique preserved evaluation target exists or escapes its owner")
        write(run / "evaluation_resume_once.json", {"original": str(output.resolve()),
            "preserved": str(preserved.resolve()), "frozen": frozen,
            "partial_frozen_sha256": sha(output / "frozen.json"),
            "same_endpoint_reevaluation_only": True}, exclusive=True)
        # Single-directory rename preserves every partial array/error in place.
        output.rename(preserved)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    pending = evaluation_root / (".pending_" + output.name + "_" + uuid.uuid4().hex)
    if pending.resolve().parent != resolved_root or output.exists():
        raise ValueError("new evaluation publication target is unsafe or already exists")
    pending.mkdir(exist_ok=False)
    write(pending / "frozen.json", frozen, exclusive=True)
    pending.rename(output)
    model = load_export(folder / "model.json", mode, "cpu")
    result = evaluate_model(model, events, output / "science", prepared)
    write(output / "complete.json", {"status": "COMPLETED", "evidence_sha256": sha(output / "science/evidence.json")}, exclusive=True)
    return result


def fit(base, mode, seed, device="cuda", resume=False):
    from train_joint_numerical_pair_0100 import next_samples
    base = Path(base).resolve(); prepared, events = checked_preparation(base, fit_only=True)
    selected = read(base / "calibration/selection.json")
    if selected["prepare_sha256"] != sha(base / "prepare.json") or selected["code"] != code_identity():
        raise ValueError("calibration identity changed")
    multiplier = selected["selected"].get(mode)
    if multiplier not in MULTIPLIERS:
        raise ValueError("configuration isolated by finite calibration")
    if mode not in MODES or seed not in SEEDS:
        raise ValueError("unregistered formal configuration")
    device = resolve_device(device)
    run = base / "models" / f"{mode}_{seed}"; checkpoints = run / "checkpoints"
    identity = {"mode": mode, "seed": seed, "source_sha256": SOURCE_SHA,
        "prepare_sha256": sha(base / "prepare.json"), "schedule_sha256": prepared["schedule_sha256"],
        "calibration_sha256": sha(base / "calibration/selection.json"), "code": code_identity(),
        "final_updates": 2048, "multiplier": multiplier, "rates": RATES}
    schedule = read(base / "schedule.json")[str(seed)]
    if resume:
        if (run / "failure.json").exists():
            raise ValueError("scientific/implementation failure cannot be relabeled external interruption")
        write(run / "resume_once.json", {"source": read(checkpoints / "latest.json"),
            "maximum_replayed_unpersisted_calls": 32}, exclusive=True)
        model, optimizer, rng, coverage, metadata = restore_checkpoint(checkpoints, device, identity,
            multiplier=multiplier)
        start_step, trace = metadata["step"], metadata["trace"]
    else:
        run.mkdir(parents=True, exist_ok=False)
        model = build_model(mode, seed, device); optimizer = make_optimizer(model, mode, multiplier)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(); rng.bit_generator.state = schedule["initial"]["rng"]
        coverage = copy.deepcopy(schedule["initial"]["coverage"])
        start_step, trace = 0, []
        save_checkpoint(checkpoints, model, mode, optimizer, rng, coverage, 0, identity, trace)
    store = TensorEventStore(events, device)

    def milestone(number, *, resume_partial=False):
        evaluation = evaluate_endpoint(base, mode, seed, number, resume_partial=resume_partial)
        profile = evaluation["profiles"]["common385"]
        unresolved = [scope for scope, key in (("8s", "all_8s_rows_qualified"),
            ("actual15s", "all_actual15s_rows_qualified")) if not profile[key]]
        if unresolved:
            capture = quarantine_failure(run / f"numerical_milestone_{number:06d}", model,
                optimizer, rng, coverage, number, trace,
                ValueError("numerical qualification unresolved in " + ",".join(unresolved)))
            write(run / f"numerical_milestone_{number:06d}.json", {
                "step": number, "unresolved_scopes": unresolved, "quarantine": capture,
                "registered_remaining_updates_continue": True})
        return evaluation

    try:
        if start_step in MILESTONES:
            milestone(start_step,
                resume_partial=bool(resume and start_step in MILESTONES))
        for step in range(start_step + 1, 2049):
            row = schedule["rows"][step - 1]
            if digest(rng.bit_generator.state) != row["rng_before_sha256"]:
                raise ValueError("frozen sampler RNG differs before update")
            samples, sampling = next_samples(events, rng, coverage)
            if digest(sampling) != row["sampling_sha256"] or digest(rng.bit_generator.state) != row["rng_after_sha256"]:
                raise ValueError("paired event/position/RNG schedule differs")
            if digest(coverage) != row["coverage_after_sha256"]:
                raise ValueError("coverage state differs")
            if row["long_counts"] != [min(round(row["horizon_s"] * 100), len(e["x"]) - 11) for e, _ in samples]:
                raise ValueError("genuine future interval changed")
            for group in optimizer.param_groups:
                group["lr"] = RATES[group["block"]] * multiplier * (.3 if step >= 1025 else 1.)
            result = one_update(model, mode, optimizer, store, samples, row["horizon_s"],
                call_journal=run / "optimizer_calls.jsonl")
            trace.append({"step": step, "horizon_s": row["horizon_s"],
                "sample_ids": row["sample_ids"], "endpoints": row["endpoints"], **result})
            if step % 32 == 0:
                save_checkpoint(checkpoints, model, mode, optimizer, rng, coverage, step, identity, trace)
            if step in MILESTONES:
                milestone(step)
        result = {"status": "COMPLETED", "mode": mode, "seed": seed, "step": 2048,
            "checkpoint": str(checkpoints / "step002048"), "source_sha256": SOURCE_SHA,
            "resumed": resume, "replayed_calls_not_counted_as_effective_updates": resume}
        write(run / "complete.json", result, exclusive=True)
        return result
    except (KeyboardInterrupt, SystemExit) as error:
        write(run / "interruption.json", {"error": repr(error), "last_persisted": read(checkpoints / "latest.json")})
        raise
    except BaseException as error:
        quarantine = quarantine_failure(run, model, optimizer, rng, coverage,
            locals().get("step", start_step), trace, error)
        write(run / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "last_persisted": read(checkpoints / "latest.json"), "quarantine": quarantine})
        raise


def noise(base):
    base = Path(base).resolve(); prepared, events = checked_preparation(base, fit_only=False)
    held = [e for e in events if e["id"] in set(prepared["no_update_ids"])]
    endpoints = []
    for mode in MODES:
        for seed in SEEDS:
            run = base / "models" / f"{mode}_{seed}"
            if read(run / "complete.json")["step"] != 2048:
                raise ValueError("all six effective2048 endpoints required before noise")
            folder = endpoint_folder(base, mode, seed, 2048)
            metadata = read(folder / "checkpoint.json")
            if sha(folder / "model.json") != metadata["model_sha256"]:
                raise ValueError("final endpoint changed")
            endpoints.append({"mode": mode, "seed": seed, "model_path": str(folder / "model.json"),
                "model_sha256": metadata["model_sha256"]})
    output = base / "noise"; output.mkdir(exist_ok=False)
    write(output / "frozen.json", {"endpoints": endpoints, "event_ids": [e["id"] for e in held],
        "stream": "first eight SHA256 bytes big endian of r_memory_noise_20261025|event_id; PCG64",
        "sigma": ".01 * frozen source delta/omega input_scale; p unchanged"}, exclusive=True)
    for endpoint in endpoints:
        model = load_export(endpoint["model_path"], endpoint["mode"], "cpu")
        evaluate_model(model, held, output / f"{endpoint['mode']}_{endpoint['seed']}", prepared, noisy=True)
    write(output / "complete.json", {"status": "COMPLETED", "endpoint_count": 6}, exclusive=True)
    return {"status": "COMPLETED", "endpoint_count": 6}


def synthetic_fixture():
    """Small nonzero, ragged implementation fixture, never a scientific event."""
    events = []
    for index, length in enumerate((17, 23)):
        t = np.arange(length) * .01
        phase = np.arange(10)[None, :] * .19 + index * .3
        d = .0002 * np.sin(t[:, None] * 2. + phase)
        w = .0004 * np.cos(t[:, None] * 2. + phase)
        p = .0001 * np.sin(t[:, None] + phase)
        events.append({"id": f"qa_{index}", "group": f"qa_group_{index}",
            "label": "stable" if index == 0 else "unstable", "root": "synthetic_qa_only",
            "x": np.c_[d, w, p], "torque": .0003 * np.cos(t[:, None] + phase)})
    return events


def cpu_cuda_comparison(mode, events, device):
    """Zero-update paired checks with the registered fixed device tolerances."""
    device = resolve_device(device)
    if device.type != "cuda":
        raise RuntimeError("CPU/CUDA qualification requires an available CUDA device")
    cpu = build_model(mode, 25101, "cpu")
    cuda = copy.deepcopy(cpu).to(device)
    histories = np.stack([event["x"][:11] for event in events])
    states = np.column_stack((np.stack([event["x"][10] for event in events]),
        np.full((len(events), 10), 1e-5)))
    differences = {}

    def compare(name, left, right):
        right_cpu = right.detach().cpu(); left_cpu = left.detach().cpu()
        if left_cpu.dtype != torch.float64 or right_cpu.dtype != torch.float64:
            raise ValueError("device comparison must retain float64: " + name)
        if not torch.isfinite(left_cpu).all() or not torch.isfinite(right_cpu).all():
            raise FloatingPointError("nonfinite CPU/CUDA comparison: " + name)
        torch.testing.assert_close(left_cpu, right_cpu, rtol=1e-8, atol=1e-10)
        differences[name] = float((left_cpu - right_cpu).abs().max()) if left_cpu.numel() else 0.

    with torch.no_grad():
        compare("field", cpu.field(torch.as_tensor(states)),
            cuda.field(torch.as_tensor(states, device=device)))
        with graph(cpu):
            left_path = cpu.rollout(torch.as_tensor(histories), 4, .01)
        with graph(cuda):
            right_path = cuda.rollout(torch.as_tensor(histories, device=device), 4, .01)
        compare("short_trajectory", left_path, right_path)
    samples = [(event, 10) for event in events]
    cpu.zero_grad(set_to_none=True); cuda.zero_grad(set_to_none=True)
    with graph(cpu):
        cpu_loss, cpu_terms = objective(cpu, mode, TensorEventStore(events, "cpu"), samples, .02)
        cpu_loss.backward()
    with graph(cuda):
        cuda_loss, cuda_terms = objective(cuda, mode, TensorEventStore(events, device), samples, .02)
        cuda_loss.backward()
    compare("complete_short_objective", cpu_loss, cuda_loss)
    for name in cpu_terms:
        compare("objective_" + name, cpu_terms[name], cuda_terms[name])
    cpu_parameters = {name: p for name, p in cpu.named_parameters() if p.requires_grad}
    cuda_parameters = {name: p for name, p in cuda.named_parameters() if p.requires_grad}
    if cpu_parameters.keys() != cuda_parameters.keys():
        raise ValueError("CPU/CUDA trainable parameter names differ")
    gradient_differences, dtypes = {}, {}
    for name in cpu_parameters:
        a, b = cpu_parameters[name], cuda_parameters[name]
        if a.grad is None or b.grad is None:
            raise ValueError("missing unclipped CPU/CUDA gradient: " + name)
        compare("gradient_" + name, a.grad, b.grad)
        gradient_differences[name] = differences["gradient_" + name]
        dtypes[name] = {"CPU": str(a.grad.dtype), "CUDA": str(b.grad.dtype)}
    return {"mode": mode, "status": "PASS", "actual_optimizer_updates": 0,
        "rtol": 1e-8, "atol": 1e-10, "maximum_absolute_differences": differences,
        "maximum_unclipped_gradient_difference": max(gradient_differences.values(), default=0.),
        "unclipped_gradient_differences_by_parameter": gradient_differences,
        "gradient_dtypes_by_parameter": dtypes,
        "parameter_names": list(cpu_parameters), "field_dtype": "torch.float64",
        "trajectory_dtype": "torch.float64", "objective_dtype": "torch.float64"}


def _qa_compare(left, right):
    if left.keys() != right.keys():
        raise ValueError("QA state names differ")
    maximum = 0.
    for name in left:
        a, b = left[name], right[name]
        if torch.is_tensor(a):
            if a.is_floating_point():
                torch.testing.assert_close(a, b, rtol=2e-9, atol=2e-9)
                maximum = max(maximum, float((a - b).abs().max())) if a.numel() else maximum
            elif not torch.equal(a, b):
                raise ValueError("integer QA state differs: " + name)
        elif a != b:
            raise ValueError("non-tensor QA state differs: " + name)
    return maximum


def _exact_nested(left, right, path="state"):
    """Serialization equality, kept separate from numerical continuation QA."""
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError("nonexact saved/restored tensor: " + path)
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError("nonexact saved/restored NumPy state: " + path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("saved/restored dictionary layout differs: " + path)
        for key in left:
            _exact_nested(left[key], right[key], path + "." + str(key))
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("saved/restored sequence layout differs: " + path)
        for index, (a, b) in enumerate(zip(left, right)):
            _exact_nested(a, b, path + "." + str(index))
    elif left != right:
        raise ValueError("saved/restored scalar differs: " + path)


def qa(base, device="cuda"):
    """Exactly204 real fixture updates: three modes * (34 + 32 + 2)."""
    from train_joint_numerical_pair_0100 import next_samples
    base = Path(base).resolve(); output = base / "qa"; output.mkdir(parents=True, exist_ok=False)
    device = resolve_device(device); events = synthetic_fixture(); store = TensorEventStore(events, device)
    comparisons = []
    try:
        for mode in MODES:
            comparisons.append(cpu_cuda_comparison(mode, events, device))
            write(output / "device_comparisons.json", {"models": comparisons,
                "actual_optimizer_updates": 0})
    except BaseException as error:
        write(output / "device_comparison_failure.json", {"models": comparisons,
            "actual_optimizer_updates": 0, "error": repr(error), "traceback": traceback.format_exc()}, exclusive=True)
        raise
    results = []
    for mode in MODES:
        folder = output / mode; folder.mkdir()
        original = build_model(mode, 25101, device)
        identity = {"qa_fixture_only": True, "mode": mode, "source_sha256": SOURCE_SHA,
            "code": code_identity(), "real_resume_step": 32}

        def initial():
            model = copy.deepcopy(original)
            optimizer = make_optimizer(model, mode, 1)
            torch.manual_seed(25101); np.random.seed(25101); random.seed(25101)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(25101)
            rng = np.random.default_rng(25101)
            coverage = {"cover_order": rng.permutation(len(events)).tolist(), "cursor": 0}
            return model, optimizer, rng, coverage

        def advance(model, optimizer, rng, coverage, count, journal):
            trace = []
            for step in range(count):
                samples, _sampling = next_samples(events, rng, coverage)
                # Exercise all captured RNGs without making a scientific change.
                random.random(); np.random.random(); torch.rand((), device=device)
                trace.append(one_update(model, mode, optimizer, store, samples, .02, call_journal=journal))
            return trace

        continuous, opt_a, rng_a, cov_a = initial()
        trace_a = advance(continuous, opt_a, rng_a, cov_a, 34, folder / "continuous_calls.jsonl")
        cpu_a = torch.get_rng_state().clone()
        cuda_a = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        numpy_global_a = np.random.get_state(); python_a = random.getstate()
        interrupted, opt_b, rng_b, cov_b = initial()
        trace_b = advance(interrupted, opt_b, rng_b, cov_b, 32, folder / "resumed_calls.jsonl")
        before_save = {"model": copy.deepcopy(interrupted.state_dict()),
            "Adam": copy.deepcopy(opt_b.state_dict()), "rng": copy.deepcopy(rng_b.bit_generator.state),
            "coverage": copy.deepcopy(cov_b), "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "numpy_global_rng": np.random.get_state(), "python_rng": random.getstate()}
        nonzero_moment = any(bool(torch.count_nonzero(v["exp_avg"])) for v in opt_b.state.values())
        if not nonzero_moment:
            raise ValueError("real32 checkpoint lacks a nonzero Adam first moment")
        saved = save_checkpoint(folder / "checkpoints", interrupted, mode, opt_b, rng_b, cov_b,
            32, identity, trace_b)
        if not opt_b.state or not all(int(v["step"]) == 32 for v in opt_b.state.values()):
            raise ValueError("QA did not save real nonempty32 Adam state")
        resumed, opt_c, rng_c, cov_c, metadata = restore_checkpoint(folder / "checkpoints", device, identity)
        immediately_restored = {"model": resumed.state_dict(), "Adam": opt_c.state_dict(),
            "rng": rng_c.bit_generator.state, "coverage": cov_c,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "numpy_global_rng": np.random.get_state(), "python_rng": random.getstate()}
        _exact_nested(before_save, immediately_restored)
        trace_c = advance(resumed, opt_c, rng_c, cov_c, 2, folder / "resumed_calls.jsonl")
        if metadata["step"] != 32:
            raise ValueError("QA did not restore the real32 checkpoint")
        parameter_error = _qa_compare(continuous.state_dict(), resumed.state_dict())
        state_a, state_c = opt_a.state_dict(), opt_c.state_dict()
        if state_a["param_groups"] != state_c["param_groups"]:
            raise ValueError("Adam group layout changed after real32 recovery")
        moment_error = max((_qa_compare(state_a["state"][key], state_c["state"][key])
            for key in state_a["state"]), default=0.)
        if digest(rng_a.bit_generator.state) != digest(rng_c.bit_generator.state) or cov_a != cov_c:
            raise ValueError("NumPy Generator or coverage not restored")
        if not torch.equal(cpu_a, torch.get_rng_state()) or digest(numpy_global_a) != digest(np.random.get_state()) or python_a != random.getstate():
            raise ValueError("CPU/Python/global NumPy RNG not restored")
        if cuda_a is not None and any(not torch.equal(a, b) for a, b in zip(cuda_a, torch.cuda.get_rng_state_all())):
            raise ValueError("CUDA RNG not restored")
        result = {"mode": mode, "status": "PASS", "actual_optimizer_updates": 68,
            "continuous": len(trace_a), "before_save": len(trace_b), "after_restore": len(trace_c),
            "saved_step": 32, "checkpoint": str(saved), "parameter_maxabs": parameter_error,
            "Adam_maxabs": moment_error, "runtime": runtime_identity(device),
            "serialization_model_Adam_RNG_exact": True, "nonzero_Adam_exp_avg": True,
            "continuation_claim": "same-environment numerical recovery at rtol/atol2e-9; RNG bitwise equal"}
        results.append(result); write(folder / "result.json", result, exclusive=True)
    summary = {"status": "PASS", "actual_optimizer_updates": 204, "models": results,
        "zero_update_CPU_CUDA_comparisons": comparisons,
        "fixture_only": True, "real_train_payloads_read": False, "val_read": False, "test_read": False}
    write(output / "summary.json", summary, exclusive=True)
    return summary


def summarize(base):
    base = Path(base).resolve()
    result = {"source_sha256": SOURCE_SHA, "val_read": False, "test_read": False,
        "qa": read(base / "qa/summary.json") if (base / "qa/summary.json").exists() else None,
        "calibration": read(base / "calibration/selection.json") if (base / "calibration/selection.json").exists() else None,
        "runs": {}, "scientific_decision_owned_by": "root"}
    for mode in MODES:
        for seed in SEEDS:
            run = base / "models" / f"{mode}_{seed}"
            if not run.exists():
                continue
            result["runs"][f"{mode}_{seed}"] = {"complete": read(run / "complete.json") if (run / "complete.json").exists() else None,
                "failure": read(run / "failure.json") if (run / "failure.json").exists() else None,
                "latest": read(run / "checkpoints/latest.json") if (run / "checkpoints/latest.json").exists() else None}
    comparisons = []
    for seed in SEEDS:
        for step in MILESTONES:
            evidence = {}
            for mode in ("B", "C"):
                folder = base / "evaluations" / f"{mode}_{seed}_step{step:06d}"
                if (folder / "complete.json").exists():
                    complete = read(folder / "complete.json")
                    if complete["evidence_sha256"] != sha(folder / "science/evidence.json"):
                        raise ValueError("comparison evidence identity changed")
                    evidence[mode] = read(folder / "science/evidence.json")
            if len(evidence) != 2:
                continue
            for scope in evidence["B"]["profiles"].keys() & evidence["C"]["profiles"].keys():
                for h in ("5", "8"):
                    b, c = (evidence[mode]["profiles"][scope]["horizon_risks"][h] for mode in ("B", "C"))
                    delta = None if b["value"] is None or c["value"] is None else b["value"] - c["value"]
                    bound = None if b["numerical_score_perturbation"] is None or c["numerical_score_perturbation"] is None else b["numerical_score_perturbation"] + c["numerical_score_perturbation"]
                    comparisons.append({"seed": seed, "step": step, "scope": scope, "horizon": h,
                        "B_minus_C_risk": delta, "sum_discrete_score_perturbations": bound,
                        "positive_gain_exceeds_numerical_range": bool(delta is not None and bound is not None and delta > bound),
                        "range_is_confidence_interval": False})
    result["paired_B_C_risks"] = comparisons
    write(base / "summary.json", result)
    return result


summary = summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "qa", "calibrate", "fit", "evaluate", "noise", "summary"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--step", type=int, choices=MILESTONES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)
    if args.stage in ("fit", "evaluate") and (args.mode is None or args.seed is None):
        parser.error("fit/evaluate require a registered mode and seed")
    if args.stage == "evaluate" and args.step is None:
        parser.error("evaluate requires a registered milestone")
    if args.resume and args.stage != "fit":
        parser.error("only formal fit accepts the single external-interruption resume")
    if args.source_commit is not None and (len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit)):
        parser.error("source commit must be a full archived hexadecimal SHA")
    torch.set_num_threads(1)
    if args.stage == "prepare":
        result = prepare(args.base)
    elif args.stage == "preflight":
        result = preflight(args.base, args.device)
    elif args.stage == "qa":
        result = qa(args.base, args.device)
    elif args.stage == "calibrate":
        result = calibrate(args.base, args.device)
    elif args.stage == "fit":
        result = fit(args.base, args.mode, args.seed, args.device, args.resume)
    elif args.stage == "evaluate":
        result = evaluate_endpoint(args.base, args.mode, args.seed, args.step)
    elif args.stage == "noise":
        result = noise(args.base)
    else:
        result = summarize(args.base)
    print(json.dumps({"stage": args.stage, "status": result.get("status", "COMPLETED"),
        "base": str(args.base.resolve())}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
