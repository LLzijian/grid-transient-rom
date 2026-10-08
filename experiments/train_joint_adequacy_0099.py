"""Narrow, deterministic Loop 0099 resume runner.

The runner intentionally does not replace the 0098 trainer.  It restores one
of the two registered 0098 end states, replays the old sampler, and continues
the same joint objective at absolute update numbers.  Full-pool monitoring is
provided by ``monitor_joint_adequacy_0099``; this
file contains no TEST reader and does not run on import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
PROCESS_STARTED = time.perf_counter()
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(EXPERIMENTS))

from train_joint_campaign import sample_batch
from train_local_memory_0098 import (  # noqa: E402
    SEED, check_structure, graph, histories_for, horizon, load_model,
    objective, optimizer_for, read_train, sources, NORM, process_memory_bytes,
)

ARMS = ("G3-S", "L10-S")
START_STEP = {"G3-S": 570, "L10-S": 567}
MILESTONES = (200, 400, 600, 800)
FORMAL_CAP_SECONDS = 7500.0
QA_CAP_SECONDS = 800.0
ANCESTOR_CHARGED = 8623.467782599997
EXPECTED = {
    'G3-S': ('fe3178f9aec2366b6022ca6430b5094a7f07c45ab2cd5f5b94756c8d4c35f5c1',
             '61d261ceb46aa9bd8917d71b010048f308ec6eea59a551d3c513eb13ca787b9e'),
    'L10-S': ('f707ec9c912866ad73853eaf60d3bc361ceb45b6746e9098197ec8af064c5b85',
              '1e1d0bbb6bbbc0a0c648f44f8233da02a411f41c4729fc2fa41265923b22f304')}


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  default=str).encode()).hexdigest()


def _state_digest(model: torch.nn.Module) -> str:
    from monitor_joint_adequacy_0099 import _model_hash
    return _model_hash(model)


def _all_sources(source: Path) -> dict[str, Any]:
    """Hash the inherited sources plus the 0099 runner/monitor/card/checkpoint."""
    result = dict(sources())
    extra = {
        "runner": Path(__file__),
        "protocol": ROOT / "reports/task2_joint_adequacy_0099.md",
        "source_model": source / "model.pt",
        "source_optimizer": source / "optimizer.pt",
        "source_rng": source / ("rng.pt" if (source / "rng.pt").exists() else "rng_state.pt"),
        "source_metadata": source / "checkpoint.json",
    }
    monitor_path = ROOT / "experiments/monitor_joint_adequacy_0099.py"
    if monitor_path.exists():
        extra["monitor"] = monitor_path
    for key, path in extra.items():
        if Path(path).exists():
            result[key] = {"path": str(path), "sha256": digest(path)}
    return result


def _trace_rows(source: Path) -> list[dict[str, Any]]:
    path = source / "trace.jsonl"
    if not path.exists():
        path = source.parent / "trace.jsonl"
    if not path.exists():
        raise FileNotFoundError('exact resume requires the complete ancestor trace')
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _same_numpy_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # Generator state contains an ndarray under ``state``; JSON conversion may
    # turn it into a list, so compare through a canonical Generator copy.
    a, b = np.random.default_rng(), np.random.default_rng()
    try:
        a.bit_generator.state = left
        b.bit_generator.state = right
        return a.bit_generator.state == b.bit_generator.state
    except (TypeError, ValueError):
        return _json_digest(left) == _json_digest(right)


def replay_sampler(events: list[dict[str, Any]], source: str | Path,
                   *, expected_trace: Iterable[dict[str, Any]] | None = None,
                   seed: int = SEED) -> dict[str, Any]:
    """Replay every pre-resume formal sample without consuming the train RNG.

    The original formal loop first created one full-pool permutation, then for
    each step drew the ten stratified samples followed by two coverage events.
    Repeating those exact calls recovers the permutation cursor and numpy RNG
    state that were omitted from the old checkpoint format.
    """
    source = Path(source)
    rows = list(expected_trace) if expected_trace is not None else _trace_rows(source)
    meta_path = source / "checkpoint.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    step = int(meta.get("step", 0))
    if step < 0:
        raise ValueError("checkpoint step must be nonnegative")
    if len(rows) != step or [r.get('step') for r in rows] != list(range(1, step+1)):
        raise ValueError('ancestor trace must contain every completed step exactly once')
    pool = events
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    cursor = 0
    replayed = []
    for absolute in range(1, step + 1):
        samples, _ = sample_batch(pool, rng, 10)
        ids = [event["id"] for event, _ in samples]
        endpoints = [int(endpoint) for _, endpoint in samples]
        extra_ids, extra_endpoints = [], []
        for _ in range(2):
            if cursor >= len(pool):
                order = rng.permutation(len(pool)); cursor = 0
            event = pool[int(order[cursor])]; cursor += 1
            extra_ids.append(event["id"])
            extra_endpoints.append(int(rng.integers(10, len(event["x"]) - 1)))
        record = {"step": absolute, "sample_ids": ids + extra_ids,
                  "endpoints": endpoints + extra_endpoints}
        replayed.append(record)
        if rows:
            old = rows[absolute - 1]
            if list(old.get("sample_ids", ())) != record["sample_ids"] or list(old.get("endpoints", ())) != record["endpoints"]:
                raise RuntimeError(f"sampler replay mismatch at old step {absolute}")
    saved_rng = source / "rng.pt"
    if not saved_rng.exists():
        saved_rng = source / "rng_state.pt"
    if saved_rng.exists():
        payload = torch.load(saved_rng, map_location="cpu", weights_only=False)
        expected_np = payload.get("numpy")
        if expected_np is not None and not _same_numpy_state(rng.bit_generator.state, expected_np):
            raise RuntimeError("replayed numpy RNG does not match checkpoint RNG")
    else:
        raise FileNotFoundError(saved_rng)
    return {"seed": int(seed), "steps": step, "cover_order": order.astype(int).tolist(),
            "cursor": int(cursor), "pool_size": len(pool),
            "replayed_trace_digest": _json_digest(replayed),
            "final_numpy_state": rng.bit_generator.state,
            "records": replayed}


def _load_optimizer(source: Path, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = optimizer_for(model)
    path = source / "optimizer.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state)
    return optimizer


def load_resume(source: str | Path, arm: str, events: list[dict[str, Any]]) -> tuple[torch.nn.Module, torch.optim.Optimizer, np.random.Generator, dict[str, Any]]:
    source = Path(source)
    for filename, expected_hash in zip(('model.pt', 'optimizer.pt'), EXPECTED[arm]):
        if digest(source / filename) != expected_hash:
            raise ValueError(f'unregistered {arm} starting {filename}')
    meta = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    if meta.get("arm") != arm:
        raise ValueError(f"checkpoint arm {meta.get('arm')} does not match {arm}")
    expected = START_STEP[arm]
    if int(meta.get("step", -1)) != expected:
        raise ValueError(f"{arm} resume must start at absolute step {expected}")
    model = load_model(source)
    if model.dim != {'G3-S':33, 'L10-S':40}[arm]:
        raise ValueError("0099 accepts only G3-S/L10-S state dimensions")
    mask = model.template_mask.detach().cpu().tolist()
    if mask != [1.,1.,1.,1.,1.,1.,0.,1.]:
        raise ValueError("0099 requires the seven-template 0098 support with R_pi_q inactive")
    optimizer = _load_optimizer(source, model)
    if any(int(state['step']) != expected for state in optimizer.state.values()):
        raise ValueError('ancestor Adam update counters changed')
    check_structure(model)
    rng_path = source / "rng.pt"
    if not rng_path.exists():
        rng_path = source / "rng_state.pt"
    rng_payload = torch.load(rng_path, map_location="cpu", weights_only=False)
    if "torch" in rng_payload:
        torch.set_rng_state(rng_payload["torch"])
    rng = np.random.default_rng()
    replay = replay_sampler(events, source)
    if "numpy" in rng_payload:
        rng.bit_generator.state = rng_payload["numpy"]
    # Replay and checkpoint must agree; the restored RNG is intentionally the
    # checkpoint state, while replay supplies the omitted cover cursor/order.
    replay["restored_numpy_state_digest"] = _json_digest(rng.bit_generator.state)
    meta["replay"] = {key: value for key, value in replay.items() if key != "records"}
    return model, optimizer, rng, meta


def build_monitor_manifest(events: list[dict[str, Any]]) -> dict[str, Any]:
    from monitor_joint_adequacy_0099 import build_manifest
    return build_manifest(events)


def _monitor_callable():
    try:
        from monitor_joint_adequacy_0099 import run_monitor
        return run_monitor
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("formal 0099 execution requires monitor_joint_adequacy_0099") from exc


def _monitor_is_complete(row: dict[str, Any]) -> bool:
    return bool(row.get("complete", False) and row.get("numerical_pass", False)
                and row.get("J_fine") is not None)


def _eligible_800(monitors: dict[str, dict[str, Any]]) -> bool:
    required = [monitors.get(str(step)) for step in (200, 400, 600)]
    if any(item is None or not _monitor_is_complete(item) for item in required):
        return False
    js = [float(item["J_fine"]) for item in required]
    tol = lambda a, b: 1e-8 * max(1.0, abs(a), abs(b))
    if not all(js[i + 1] < js[i] - tol(js[i], js[i + 1]) for i in range(2)):
        return False
    severe = [item.get("S_fine") or {} for item in required]
    for key in ("1", "2", "5"):
        vals = [float(item[key]) for item in severe if item.get(key) is not None]
        if len(vals) != 3 or any(not np.isfinite(v) for v in vals):
            return False
        if any(vals[i + 1] > vals[i] + tol(vals[i], vals[i + 1]) for i in range(2)):
            return False
    return True


def _monitor_partial(row: dict[str, Any]) -> bool:
    return bool(row.get("partial", False) or row.get("deadline_exceeded", False)
                or (not row.get("complete", False) and row.get("J_fine") is None))


def _select_best(monitors: dict[str, dict[str, Any]]) -> str | None:
    eligible = [(int(step), row) for step, row in monitors.items()
                if str(step).isdigit() and _monitor_is_complete(row)]
    if not eligible:
        return None
    best_step, best_row = min(eligible, key=lambda item: item[0])
    for step, row in sorted(eligible):
        a, b = float(best_row['J_fine']), float(row['J_fine'])
        if b < a-1e-8*max(1.,abs(a),abs(b)):
            best_step, best_row = step, row
    return str(best_step)


def severe_transition(row, baseline, counters):
    """Unresolved finite raw S may stop for numerical risk, never imply physics."""
    for h in ('1','2','5'):
        value = row.get('S_fine', {}).get(h)
        numerical = value is None
        if value is None:
            value = row.get('raw_S_fine', {}).get(h)
        over = value is not None and np.isfinite(value) and value > 10*baseline[h]
        previous = counters[h]
        counters[h] = (-(abs(previous)+1) if numerical or previous<0 else previous+1) if over else 0
    if any(abs(v) >= 2 for v in counters.values()):
        return 'numerical_risk_10x_twice' if any(v<=-2 for v in counters.values()) else 'severe_prefix_10x_twice'
    return None


def _save_checkpoint(output: Path, name: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     rng: np.random.Generator, arm: str, absolute_step: int,
                     coverage: dict[str, Any], source_meta: dict[str, Any],
                     monitor_manifest: dict[str, Any] | None = None) -> Path:
    directory = output / name
    directory.mkdir(parents=False, exist_ok=False)
    torch.save(model.state_dict(), directory / "model.pt")
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    torch.save({"numpy": rng.bit_generator.state, "torch": torch.get_rng_state()}, directory / "rng.pt")
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    groups = []
    by_id = {id(p): n for n,p in model.named_parameters()}
    for group in optimizer.param_groups:
        groups.append({"lr": float(group["lr"]), "parameter_names": [by_id[id(p)] for p in group['params']]})
    meta = {
        "format": "joint_adequacy_0099", "arm": arm, "step": int(absolute_step),
        "ancestor_step": START_STEP[arm], "new_steps": int(absolute_step - START_STEP[arm]),
        "dimension": model.dim, "seed": 98,
        "ancestor_charged_seconds": ANCESTOR_CHARGED,
        "process_seconds_at_checkpoint": source_meta.get('process_seconds'),
        "active_templates": model.template_mask.detach().tolist(),
        "requires_grad_parameters": names, "optimizer_parameter_names": groups,
        "coverage": coverage, "source_meta": source_meta,
        "model_state_sha256": _state_digest(model),
        "monitor_manifest_sha256": None if monitor_manifest is None else _json_digest(monitor_manifest),
        "config": model.config() if hasattr(model, "config") else None,
    }
    (directory / "checkpoint.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    exporter = getattr(model, "export_numeric", None) or getattr(model, "export", None)
    if exporter is not None:
        exporter(directory / "structured_numeric_export.json")
    return directory


def run(arm: str, source: str | Path, output: str | Path, *, events: list[dict[str, Any]] | None = None,
        new_steps: int = 600, deadline: float | None = None, qa_only: bool = False,
        force_scale: float | None = None, allow_refine: bool = True,
        monitor_cache: str | Path | None = None, cap_seconds: float | None = None,
        process_start: float | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"0099 arm must be one of {ARMS}")
    if new_steps not in MILESTONES[:-1] + MILESTONES[-1:]:
        raise ValueError("new_steps must be one of 200, 400, 600, 800")
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    started = time.perf_counter() if process_start is None else process_start
    registered_cap = QA_CAP_SECONDS if qa_only else FORMAL_CAP_SECONDS
    deadline = min(started+registered_cap, deadline if deadline is not None else float('inf'))
    if cap_seconds is not None:
        if cap_seconds <= 0:
            raise ValueError("cap_seconds must be positive")
        deadline = min(deadline, started + float(cap_seconds))
    if time.perf_counter() >= deadline:
        raise TimeoutError("0099 deadline expired before recovery")
    source_snapshot = _all_sources(source)
    if events is None:
        events, _ = read_train()
    if len(events) != 346:
        raise ValueError("0099 formal runner requires the 346-event eligible TRAIN pool")
    registered_force_scale = float(np.load(NORM, allow_pickle=False)["force_scale"])
    if force_scale is not None and abs(float(force_scale) - registered_force_scale) > 1e-14:
        raise ValueError("0099 uses the registered NORM force_scale; overrides are not allowed")
    force_scale = registered_force_scale
    model, optimizer, rng, meta = load_resume(source, arm, events)
    replay = meta["replay"]
    coverage_order = np.asarray(replay["cover_order"], dtype=int)
    cursor = int(replay["cursor"])
    manifest = build_monitor_manifest(events)
    output.mkdir(parents=True)
    monitors: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    monitor_fn = _monitor_callable()
    last_monitor_seconds = 0.0
    severe_baseline: dict[str, float] | None = None
    severe_over_limit = {"1": 0, "2": 0, "5": 0}
    stop_reason: str | None = None
    failure: dict[str, Any] | None = None
    completed_step = START_STEP[arm]
    def checkpoint_meta():
        return {'source':str(source), 'ancestor_step':START_STEP[arm],
                'source_hashes':source_snapshot, 'process_seconds':time.perf_counter()-started}

    def cached_initial_monitor() -> dict[str, Any] | None:
        if monitor_cache is None:
            return None
        cache = Path(monitor_cache)
        candidates = [cache] if cache.is_file() else [cache / "monitor.json", cache / "evidence.json"]
        monitor_code = ROOT / "experiments/monitor_joint_adequacy_0099.py"
        expected_monitor_hash = digest(monitor_code) if monitor_code.exists() else None
        expected_source_before = sources()
        for candidate in candidates:
            if not candidate.exists():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            row = payload.get("monitor", payload)
            model_hash = payload.get("model_state_sha256_before", payload.get("model_state_sha256", row.get("model_state_sha256")))
            manifest_hash = payload.get("manifest_sha256", row.get("manifest_sha256"))
            code_hash = payload.get("monitor_code_sha256", row.get("monitor_code_sha256"))
            source_before = payload.get("source_before", row.get("source_before"))
            if (model_hash == _state_digest(model) and manifest_hash == manifest.get("sha256")
                    and code_hash == expected_monitor_hash and source_before == expected_source_before
                    and row.get('complete') and row.get('sources_unchanged') and row.get('rng_unchanged')
                    and row.get('model_state_sha256_after') == model_hash):
                return row
        raise RuntimeError("monitor cache exists but model/manifest/code/source hashes do not match")

    def monitor_at(step: int):
        nonlocal monitors, last_monitor_seconds
        if monitor_fn is None:
            return None
        cached = cached_initial_monitor() if step == 0 else None
        if cached is not None:
            monitors[str(step)] = cached
            last_monitor_seconds = float(cached.get("seconds", 0.0))
            return cached
        now = time.perf_counter()
        reserve = max(300.0, 1.5 * last_monitor_seconds + 60.0)
        if deadline is not None and deadline - now < reserve:
            return {"complete": False, "numerical_pass": False, "J_fine": None, "partial": True,
                    "reason": "reserve_before_monitor", "reserve_seconds": reserve}
        path = output / f"monitor_step{step:04d}"
        torch_before = torch.get_rng_state()
        numpy_before = np.random.get_state()
        monitor_started = time.perf_counter()
        try:
            row = monitor_fn(model, events, manifest, path,
                             deadline=deadline-15., allow_refine=allow_refine, force_scale=force_scale)
        finally:
            # A monitor is read-only evidence and must not advance the
            # optimizer's torch/global numpy RNG streams.
            torch.set_rng_state(torch_before)
            np.random.set_state(numpy_before)
        last_monitor_seconds = float(row.get("seconds", time.perf_counter() - monitor_started))
        monitors[str(step)] = row
        return row

    # Preserve the exact restored step as a first-class recoverable artifact.
    _save_checkpoint(output, "step0000", model, optimizer, rng, arm, START_STEP[arm],
                     {"cover_order": replay["cover_order"], "cursor": cursor},
                     checkpoint_meta(), manifest)
    initial = monitor_at(0)
    if initial is not None:
        monitors["0"] = initial
        initial_s = initial.get("raw_S_fine") or {}
        if all(initial_s.get(key) is not None and np.isfinite(initial_s[key]) for key in ("1", "2", "5")):
            severe_baseline = {key: float(initial_s[key]) for key in ("1", "2", "5")}
    if qa_only:
        evidence = {'status':'qa_complete', 'arm':arm,'qa_only':True,'monitor':initial,
                    'manifest':manifest,'replay':replay,'seconds':time.perf_counter()-started,
                    'source_snapshot':source_snapshot,'sources_unchanged':source_snapshot==_all_sources(source),
                    'within_cap':time.perf_counter()<=deadline,'test_read':False,'validation_read':False}
        (output/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        return evidence
    if initial is None or not initial.get('complete') or severe_baseline is None:
        raise RuntimeError('complete finite starting S and measured full monitor required before updates')
    estimate = 200*4.85 + max(300.,1.5*last_monitor_seconds+60.)
    if time.perf_counter()+estimate >= deadline:
        raise RuntimeError('remaining budget cannot accommodate first 200 updates and final monitor')
    target = START_STEP[arm] + new_steps
    step = START_STEP[arm]
    mask_before = model.template_mask.detach().clone()
    try:
        for step in range(START_STEP[arm] + 1, target + 1):
            recent_seconds = [float(row["seconds"]) for row in traces[-10:] if "seconds" in row]
            reserve = max(300.0, 1.5 * last_monitor_seconds + 60.0) + 1.5*(max(recent_seconds) if recent_seconds else 0.)
            if deadline is not None and time.perf_counter() + reserve >= deadline:
                stop_reason = "wall_cap"
                break
            tic = time.perf_counter()
            samples, _ = sample_batch(events, rng, 10)
            ids = [e["id"] for e, _ in samples]; endpoints = [int(i) for _, i in samples]
            extra_ids, extra_endpoints = [], []
            for _ in range(2):
                if cursor >= len(coverage_order):
                    coverage_order = rng.permutation(len(events)); cursor = 0
                event = events[int(coverage_order[cursor])]; cursor += 1
                endpoint = int(rng.integers(10, len(event["x"]) - 1))
                samples.append((event, endpoint)); extra_ids.append(event["id"]); extra_endpoints.append(endpoint)
            histories = histories_for(samples)
            optimizer.zero_grad(set_to_none=True)
            absolute_horizon = horizon(step)
            with graph(model):
                loss, terms = objective(model, samples, histories, absolute_horizon,
                                        force_scale)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                loss.backward()
            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if any(not torch.isfinite(p.grad).all() for p in params):
                raise FloatingPointError("nonfinite gradient")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 100.))
            optimizer.step()
            completed_step = step
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite parameters")
            if not torch.equal(model.template_mask.detach(), mask_before):
                raise RuntimeError("0099 template support changed during resume")
            check_structure(model)
            traces.append({"step": step, "horizon": absolute_horizon, "loss": float(loss.detach()),
                           "terms": {key: float(value.detach()) for key, value in terms.items()},
                           "gradient_norm": grad_norm, "sample_ids": ids + extra_ids,
                           "endpoints": endpoints + extra_endpoints, "seconds": time.perf_counter() - tic,
                           "cursor": cursor})
            traces[-1]['elapsed_seconds'] = time.perf_counter()-started
            with (output / "trace.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(traces[-1], allow_nan=False) + "\n")
            if (step-START_STEP[arm]) % 25 == 0:
                print(json.dumps({'arm':arm,'new_steps':step-START_STEP[arm],
                                  'seconds':time.perf_counter()-started,'loss':traces[-1]['loss']}),flush=True)
            if process_memory_bytes() > 8 * 1024**3:
                stop_reason = "memory_cap"
                break
            if step - START_STEP[arm] in (200, 400, 600, 800):
                row = monitor_at(step - START_STEP[arm])
                if row is not None and _monitor_partial(row):
                    stop_reason = "partial_monitor"
                    break
                if row is not None:
                    stop_reason = severe_transition(row, severe_baseline, severe_over_limit)
                _save_checkpoint(output, f"step{step:04d}", model, optimizer, rng, arm, step,
                                 {"cover_order": coverage_order.tolist(), "cursor": cursor},
                                 checkpoint_meta(), manifest)
                if source_snapshot != _all_sources(source):
                    raise RuntimeError('source drift during joint fit')
                if stop_reason:
                    break
                if step - START_STEP[arm] == 600 and new_steps == 800 and not _eligible_800(monitors):
                    stop_reason = "optional_800_gate"
                    break
                if step-START_STEP[arm] == 600 and time.perf_counter()+estimate >= deadline:
                    stop_reason = 'optional_800_budget'; break
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "J_fine": None}
        stop_reason = 'implementation_or_numerical_failure'
        raise
    finally:
        _save_checkpoint(output, f"last_step{completed_step:04d}", model, optimizer, rng, arm, completed_step,
                         {"cover_order": coverage_order.tolist(), "cursor": cursor},
                         checkpoint_meta(), manifest)
        after_snapshot = _all_sources(source)
        evidence = {"format": "joint_adequacy_0099", "arm": arm, "ancestor_step": START_STEP[arm],
                    "final_step": completed_step, "new_steps": completed_step - START_STEP[arm], "qa_only": False,
                    "monitors": monitors, "traces": traces, "manifest": manifest,
                    "source_snapshot": source_snapshot, "seconds": time.perf_counter() - started,
                    "source_snapshot_after": after_snapshot,
                    "sources_unchanged": source_snapshot == after_snapshot,
                    "best_monitor_step": _select_best(monitors),
                    "stop_reason": stop_reason, "failure": failure,
                    "test_read": False, "validation_read": False}
        best = evidence['best_monitor_step']
        evidence['selected_checkpoint'] = (str(output/('step0000' if best=='0' else f'step{START_STEP[arm]+int(best):04d}')) if best is not None else None)
        evidence['last_checkpoint'] = str(output/f'last_step{completed_step:04d}')
        evidence['phase_cap_seconds'] = deadline-started
        evidence['within_cap'] = time.perf_counter() <= deadline
        evidence['status'] = stop_reason or 'step_cap'
        if not evidence['sources_unchanged']:
            evidence.update(status='quarantined_source_drift', selected_checkpoint=None)
        if not evidence['within_cap']:
            evidence.update(status='budget_exception', selected_checkpoint=None)
        evidence['ancestor_charged_seconds'] = ANCESTOR_CHARGED
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="0099 exact warm-start joint optimizer resume")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-steps", type=int, choices=MILESTONES, default=600)
    parser.add_argument("--deadline", type=float, default=None,
                        help="absolute time.perf_counter deadline")
    parser.add_argument("--qa-only", action="store_true", help="run recovery/monitor QA without updates")
    parser.add_argument("--no-refine", action="store_true", help="disallow optional .0025 monitor refinement")
    parser.add_argument("--monitor-cache", type=Path, default=None,
                        help="validated initial monitor cache directory/file")
    parser.add_argument("--initial-monitor", type=Path, default=None,
                        help="strictly validated QA initial monitor cache (alias of --monitor-cache)")
    parser.add_argument("--cap-seconds", type=float, default=None,
                        help="relative per-process hard cap; never increases the registered cap")
    args = parser.parse_args()
    result = run(args.arm, args.source, args.output, new_steps=args.new_steps,
                 deadline=args.deadline, qa_only=args.qa_only, force_scale=None,
                 allow_refine=not args.no_refine,
                 monitor_cache=args.initial_monitor or args.monitor_cache,
                 cap_seconds=args.cap_seconds, process_start=PROCESS_STARTED)
    print(json.dumps({"status": result.get("status", "complete"), "output": str(args.output),
                      "final_step": result.get("final_step")}, default=str))


if __name__ == "__main__":
    main()
