"""GPU-first closure training, with a synthetic doctor/benchmark and exact resume.

No old discovery protocol, checkpoint or confirmation cohort is modified.
The train command reads the existing train-only loader, never val or test.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]
import numpy as np
import torch

from closure_training_runtime import (TrainingClosureROM, TensorEventStore,
    all_finite, export_cpu, objective, resolve_device, synchronize)
from train_local_memory_0098 import graph

RATES = {"skeleton": .01, "closure": .001, "encoder": .0003}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def make_optimizer(model):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        block = "encoder" if "gru" in name or "head" in name else (
            "closure" if name in ("coefficients", "_template_scalars.3", "_template_scalars.4") else "skeleton")
        groups.setdefault(block, []).append(p)
    return torch.optim.Adam([{"params": p, "block": b, "lr": RATES[b]} for b, p in groups.items()])


def code_identity():
    # Bind the directly used math, loader and sampling helpers.
    names = ["train_closure.py", "closure_training_runtime.py", "symbolic_closure_rom.py",
             "local_memory_rom_0098.py", "structured_rom.py", "closure_expression_identity.py",
             "structure_grammar.py", "train_structured_rom.py", "train_local_memory_0098.py",
             "train_joint_numerical_pair_0100.py", "train_joint_campaign.py"]
    paths = [ROOT / "experiments" / name for name in names]
    paths += [ROOT / p for p in ("src/kiki/data/private_dataset.py", "src/kiki/models/private_pm.py",
                                 "src/kiki/models/private_rom.py", "src/kiki/physics/__init__.py")]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def environment(device):
    return {"torch": torch.__version__, "cuda_build": torch.version.cuda,
            "device": str(device), "dtype": "float64", "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def save_checkpoint(output, model, opt, rng, coverage, step, identity, trace):
    output = Path(output).resolve()
    final = output / f"step{step:06d}"
    folder = output / f".step{step:06d}.pending-{uuid.uuid4().hex}"
    folder.mkdir()
    export_cpu(model, folder / "model.json")
    device = model.mass.device
    torch.save({"optimizer": opt.state_dict(), "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "coverage": copy.deepcopy(coverage), "step": step}, folder / "training.pt")
    meta = {"step": step, "identity": identity, "environment": environment(device),
            "model_sha256": sha(folder / "model.json"), "training_sha256": sha(folder / "training.pt"),
            "trace": trace, "status": "research_candidate", "val_read": False, "test_read": False}
    write(folder / "checkpoint.json", meta)
    if final.exists():
        pointer = output / "latest.json"
        if pointer.exists() and json.loads(pointer.read_text())["directory"] == final.name:
            raise ValueError("cannot overwrite the active immutable checkpoint")
        # An interruption after directory publication but before pointer update
        # may leave an orphan. Preserve it without blocking a resumed save.
        orphan = output / f"orphan-{final.name}-{uuid.uuid4().hex}"
        if final.resolve().parent != output or orphan.resolve().parent != output:
            raise ValueError("checkpoint preservation must remain inside output")
        final.rename(orphan)
    folder.rename(final)
    write(output / "latest.json", {"directory": final.name,
                                   "checkpoint_sha256": sha(final / "checkpoint.json")})
    return final


def restore_checkpoint(output, device, expected_identity, allow_device_change=False):
    pointer = json.loads((Path(output) / "latest.json").read_text())
    directory = pointer["directory"]
    if Path(directory).name != directory:
        raise ValueError("checkpoint directory must be an immediate child")
    folder = Path(output) / directory
    if sha(folder / "checkpoint.json") != pointer["checkpoint_sha256"]:
        raise ValueError("checkpoint metadata hash changed")
    meta = json.loads((folder / "checkpoint.json").read_text())
    if meta["identity"] != expected_identity:
        raise ValueError("source, train, configuration or code identity changed")
    for file, key in [("model.json", "model_sha256"), ("training.pt", "training_sha256")]:
        if sha(folder / file) != meta[key]:
            raise ValueError("checkpoint file hash changed")
    old = meta["environment"]
    current_env = environment(device)
    changed = any(old[k] != current_env[k] for k in ("device", "torch", "gpu", "cuda_build", "threads"))
    if changed and not allow_device_change:
        raise ValueError("resume environment changed; --allow-device-change records a numerical continuation, not bitwise replay")
    model = TrainingClosureROM.from_export(folder / "model.json").to(device)
    opt = make_optimizer(model)
    state = torch.load(folder / "training.pt", weights_only=False, map_location="cpu")
    if state["step"] != meta["step"]:
        raise ValueError("optimizer/model update counts differ")
    opt.load_state_dict(state["optimizer"])
    # Adam migrates moments to parameter devices; its scalar step stays on CPU.
    rng = np.random.default_rng()
    rng.bit_generator.state = state["numpy_rng"]
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    elif device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(expected_identity["seed"])
    return model, opt, rng, state["coverage"], meta, changed


def horizon_for_update(step):
    return 2. if step <= 100 else (1., 2., 5.)[(step - 101) % 3]


def cohort_identity(events):
    digest = hashlib.sha256()
    for e in events:
        digest.update(json.dumps({k: e[k] for k in ("id", "group", "label")}, sort_keys=True).encode())
        for k in ("x", "torque"):
            a = np.ascontiguousarray(e[k], dtype=np.float64)
            digest.update(str(a.shape).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def train(args):
    started = time.perf_counter()
    from train_local_memory_0098 import read_train
    from train_joint_numerical_pair_0100 import next_samples
    device = resolve_device(args.device)
    output, source = args.output.resolve(), args.source.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("fresh training requires a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    events, exclusions = read_train()  # only this explicit command can read train
    force_scale = max(float(np.sqrt(np.mean(np.concatenate([e["torque"] for e in events]) ** 2))), 1e-8)
    identity = {"source_sha256": sha(source), "train_sha256": cohort_identity(events),
                "train_count": len(events), "force_scale": force_scale, "code": code_identity(),
                "rates": RATES, "seed": args.seed, "final_decay": args.final,
                "precision": "float64", "dt": .01, "clip_norm": 100.}
    step, trace, changed = 0, [], False
    if args.resume:
        model, opt, rng, coverage, meta, changed = restore_checkpoint(output, device, identity, args.allow_device_change)
        step, trace = meta["step"], meta["trace"]
    else:
        model = TrainingClosureROM.from_export(source).to(device)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        coverage = {"cover_order": rng.permutation(len(events)).tolist(), "cursor": 0}
        opt = make_optimizer(model)
    if step > args.updates:
        raise ValueError("checkpoint exceeds requested total updates")
    model.provenance["training_engine"] = "train_closure float64 device-aware runtime"
    model.provenance["training_source_sha256"] = identity["source_sha256"]
    model.provenance["training_code_identity"] = identity["code"]
    store = TensorEventStore(events, device)
    run_record = {"identity": identity, "environment": environment(device),
        "source": str(source), "updates_target": args.updates, "max_seconds": args.max_seconds,
        "resume_environment_changed": changed, "excluded_train": exclusions,
        "pid": os.getpid(), "command": sys.argv, "started_epoch": time.time(),
        "status": "RUNNING", "val_read": False, "test_read": False}
    write(output / "run.json", run_record)
    parameters = [p for p in model.parameters() if p.requires_grad]
    saved_step = step if args.resume else None
    if saved_step is None:
        save_checkpoint(output, model, opt, rng, coverage, step, identity, trace)
        saved_step = step
    phase, before_attempt = "complete", None
    try:
        while step < args.updates:
            if time.perf_counter() - started >= args.max_seconds:
                if saved_step != step:
                    save_checkpoint(output, model, opt, rng, coverage, step, identity, trace)
                    saved_step = step
                raise TimeoutError("execution cap reached; last completed update preserved")
            tick = time.perf_counter()
            before_attempt = {"numpy": copy.deepcopy(rng.bit_generator.state),
                              "coverage": copy.deepcopy(coverage), "torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            phase = "before_update"
            samples, sampling = next_samples(events, rng, coverage)
            for group in opt.param_groups:
                group["lr"] = RATES[group["block"]] * (.3 if args.final and step >= 512 else 1.)
            opt.zero_grad(set_to_none=True)
            with graph(model):
                value, terms = objective(model, store, samples, horizon_for_update(step + 1), force_scale)
                if not all_finite([value]):
                    raise FloatingPointError("nonfinite objective")
                value.backward()
            if not all_finite([p.grad for p in parameters]):
                raise FloatingPointError("nonfinite gradient")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 100.)
            phase = "updating"
            opt.step()
            if not all_finite(parameters):
                raise FloatingPointError("nonfinite updated parameters")
            values = torch.stack((value.detach(), terms["risk"].detach(), norm.detach())).cpu().tolist()
            trace.append({"step": step + 1, "loss": values[0], "risk": values[1], "gradient_norm": values[2],
                          "seconds": time.perf_counter() - tick, "sampling": sampling})
            step += 1
            phase = "complete"
            if step % args.checkpoint_every == 0 or step == args.updates:
                save_checkpoint(output, model, opt, rng, coverage, step, identity, trace)
                saved_step = step
                print(json.dumps({"step": step, "target": args.updates, "loss": values[0], "device": str(device)}), flush=True)
    except BaseException as error:
        # Preserve possibly invalid attempted parameters separately; never label
        # them as an ordinary resumable completed update.
        model.end_graph_cache()
        recovery_error = None
        if phase == "before_update":
            rng.bit_generator.state = before_attempt["numpy"]
            coverage = before_attempt["coverage"]
            torch.set_rng_state(before_attempt["torch"])
            if before_attempt["cuda"] is not None:
                torch.cuda.set_rng_state(before_attempt["cuda"], device)
        if phase != "updating" and saved_step != step:
            try:
                save_checkpoint(output, model, opt, rng, coverage, step, identity, trace)
                saved_step = step
            except BaseException as save_error:
                recovery_error = repr(save_error)
                # Publication can finish just before an interrupt prevents the
                # in-memory saved_step assignment. Trust only verified files.
                try:
                    pointer = json.loads((output / "latest.json").read_text())
                    if pointer["directory"] != f"step{step:06d}":
                        raise ValueError("latest is not the current step")
                    published = output / pointer["directory"]
                    meta = json.loads((published / "checkpoint.json").read_text())
                    if (sha(published / "checkpoint.json") != pointer["checkpoint_sha256"] or
                            meta["identity"] != identity or meta["step"] != step or
                            sha(published / "model.json") != meta["model_sha256"] or
                            sha(published / "training.pt") != meta["training_sha256"]):
                        raise ValueError("published checkpoint verification failed")
                    saved_step = step
                except (OSError, ValueError, KeyError):
                    pass  # last independently verified saved_step is retained
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "numpy_rng": rng.bit_generator.state, "coverage": coverage,
                    "torch_rng": torch.get_rng_state(), "trace": trace, "step": step,
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "phase": phase, "last_valid_saved_step": saved_step}, output / "failed_state.pt")
        write(output / "failure.json", {"error": repr(error), "last_completed_update": step,
              "phase": phase, "last_valid_saved_step": saved_step,
              "rollback_completed_updates": step - saved_step,
              "unsafe_attempt_not_resumable": phase == "updating", "safe_save_error": recovery_error,
              "test_read": False})
        write(output / "run.json", {**run_record, "status": "INTERRUPTED_RUNTIME" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "FAILED",
              "last_completed_update": step, "last_valid_saved_step": saved_step,
              "elapsed_seconds": time.perf_counter() - started})
        raise
    write(output / "run.json", {**run_record, "status": "COMPLETED", "completed_updates": step,
                                "elapsed_seconds": time.perf_counter() - started})
    write(output / "result.json", {"status": "completed", "updates": step,
          "elapsed_seconds": time.perf_counter() - started, "environment": environment(device),
          "model_role": "research_candidate", "val_read": False, "test_read": False})


def synthetic_fixture(batch):
    from closure_discovery_benchmarks import toy_source
    from symbolic_closure_rom import _BASE_TERMS
    source = toy_source().with_terms(_BASE_TERMS)
    with torch.no_grad():
        source.coefficients.copy_(torch.linspace(.001, .009, len(_BASE_TERMS), dtype=torch.float64))
    rng = np.random.default_rng(8821)
    events = [{"id": f"synthetic{i}", "group": str(i % 3), "label": "unstable" if i % 3 == 0 else "stable",
               "x": rng.normal(0, .005, (36 + i % 3 * 10, 30)),
               "torque": rng.normal(0, .005, (36 + i % 3 * 10, 10))} for i in range(batch)]
    return source, events


def benchmark(args):
    from discover_closure import loss as reference_loss
    from train_local_memory_0098 import histories_for
    device = resolve_device(args.device)
    source, events = synthetic_fixture(args.batch)
    samples = [(e, 10) for e in events]
    rows = []
    for name, target, prepared, accelerated in [("legacy_cpu", torch.device("cpu"), False, False),
                                               ("prepared_cpu", torch.device("cpu"), True, True),
                                               ("prepared_original_field", device, True, False),
                                               ("prepared_requested_device", device, True, True)]:
        setup_started = time.perf_counter()
        model = copy.deepcopy(source)
        if accelerated:
            # Reload through the same portable representation without touching private artifacts.
            model.__class__ = TrainingClosureROM
        model = model.to(target)
        store = TensorEventStore(events, target) if prepared else None
        synchronize(target)
        setup_seconds = time.perf_counter() - setup_started
        if target.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target)
        durations, losses = [], []
        for index in range(args.repeats + 1):
            model.zero_grad(set_to_none=True)
            synchronize(target)
            tick = time.perf_counter()
            with graph(model):
                value, _ = (objective(model, store, samples, args.horizon, 1.) if prepared else
                            reference_loss(model, samples, histories_for(samples), args.horizon, 1.))
                value.backward()
            synchronize(target)
            duration = time.perf_counter() - tick
            if index:
                durations.append(duration)
                losses.append(float(value.detach()))
        rows.append({"name": name, "environment": environment(target), "seconds": durations,
                     "median_seconds": float(np.median(durations)), "losses": losses,
                     "initial_model_and_data_setup_seconds": setup_seconds,
                     "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None})
        del value, model, store
    result = {"scope": "synthetic forward+backward, no optimizer updates; not FOM prediction or research training",
              "batch": args.batch, "horizon": args.horizon, "warmups_each": 1, "rows": rows,
              "timing_scope": "forward and backward including per-batch preparation; excludes one-time model/data residency, checkpoint and optimizer step; CUDA synchronized",
              "test_read": False, "private_data_read": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--device", default="cuda")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--device", default="cuda")
    bench.add_argument("--batch", type=int, default=12)
    bench.add_argument("--horizon", type=float, default=.2)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--source", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--updates", type=int, required=True)
    fit.add_argument("--max-seconds", type=float, required=True)
    fit.add_argument("--seed", type=int, default=2701)
    fit.add_argument("--checkpoint-every", type=int, default=128)
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--allow-device-change", action="store_true")
    fit.add_argument("--final", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "doctor":
        d = resolve_device(args.device)
        probe = torch.ones(16, device=d, dtype=torch.float64)
        (probe @ probe).item()
        print(json.dumps(environment(d)))
    elif args.command == "benchmark":
        if args.batch < 1 or args.repeats < 1 or not np.isfinite(args.horizon) or args.horizon <= 0:
            parser.error("positive batch, repeats and horizon required")
        benchmark(args)
    else:
        if args.updates < 1 or args.checkpoint_every < 1 or not np.isfinite(args.max_seconds) or args.max_seconds <= 0:
            parser.error("positive updates, checkpoint cadence and finite seconds cap required")
        train(args)


if __name__ == "__main__":
    main()
