"""Finite force-graph study. Import has no data access or optimizer activity.

Native process ownership belongs to force_graph_execution_20261028. This module
does not resume old training, read original val/test, or change scientific gates.
"""
from __future__ import annotations

import copy
import json
import math
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

import r_memory_study_20261025 as old
from closure_training_runtime import resolve_device, synchronize
from force_graph_model_20261028 import ForceGraphROM, MODES
from train_local_memory_0098 import graph, risk
from train_structured_rom import _point_error

sha, digest, read, write, safe = old.sha, old.digest, old.read, old.write, old.safe
OLD_BASE = ROOT / "private_data/r_memory_20261025/run01_repair01"
SOURCES = {"A512": OLD_BASE / "models/A_25101/checkpoints/step000512",
           "N6101": old.SOURCE.parent}
SEEDS = (28101, 28102)
PROTOCOL = ROOT / "reports/force_graph_20261028_protocol.md"


def code_identity():
    paths = (Path(__file__), ROOT / "experiments/force_graph_model_20261028.py", PROTOCOL)
    return {**old.code_identity(), **{p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}


def source_identities():
    # Byte hashing training.pt is identity verification, never tensor loading.
    return {key: {"directory": str(path.resolve()), "files": {
        name: sha(path / name) for name in ("model.json", "checkpoint.json", "training.pt")}}
        for key, path in SOURCES.items()}


def horizon_for(step):
    if type(step) is not int or not 1 <= step <= 192:
        raise ValueError("registered update index is 1..192")
    return 5. if step % 2 else 8.


def panel_events(events):
    chosen = []
    for label, target in (("stable", 36), ("unstable", 12)):
        eligible = [e for e in events if e["label"] == label and
                    (label != "stable" or len(e["x"]) - 11 >= 800)]
        groups = sorted({str(e["group"]) for e in eligible})
        pools = [[e for e in sorted(eligible, key=lambda e: e["id"]) if str(e["group"]) == g] for g in groups]
        selected = []
        for offset in range(max((len(p) for p in pools), default=0)):
            for pool in pools:
                if offset < len(pool) and len(selected) < target:
                    selected.append(pool[offset])
        if len(selected) != target:
            raise ValueError("ESCALATE: fixed fit panel cannot supply registered label/horizon quota")
        chosen.extend(selected)
    return chosen


def sample_step(events, rng, coverage):
    if not events:
        raise ValueError("nonempty fit events required")
    random_indices = rng.integers(0, len(events), size=8).tolist()
    cover = []
    # Coverage blocks never contain duplicates, including permutation boundaries.
    while len(cover) < min(4, len(events)):
        if coverage["cursor"] == len(coverage["cover_order"]):
            coverage["cover_order"] = rng.permutation(len(events)).tolist()
            coverage["cursor"] = 0
        index = coverage["cover_order"][coverage["cursor"]]
        if index in cover:
            # Defer this entry to a later batch rather than consume it unseen.
            pos = coverage["cursor"]
            replacement = next((j for j in range(pos + 1, len(events))
                                if coverage["cover_order"][j] not in cover), None)
            if replacement is None:
                raise ValueError("coverage permutation cannot supply distinct block")
            coverage["cover_order"][pos], coverage["cover_order"][replacement] = (
                coverage["cover_order"][replacement], coverage["cover_order"][pos])
            index = coverage["cover_order"][pos]
        cover.append(index); coverage["cursor"] += 1
    if len(cover) != 4:
        # QA fixture has two events and is deliberately not a scientific batch.
        cover = (cover * 4)[:4]
    indices = random_indices + cover
    return [(events[i], 10) for i in indices], {"random_indices": random_indices,
        "coverage_indices": cover, "sample_ids": [events[i]["id"] for i in indices],
        "endpoints": [10] * 12}


def make_schedule(events, seed):
    if seed not in SEEDS or len(events) != 297:
        raise ValueError("registered seed and fit297 required")
    rng = np.random.default_rng(seed)
    coverage = {"cover_order": rng.permutation(len(events)).tolist(), "cursor": 0}
    initial = {"rng": copy.deepcopy(rng.bit_generator.state), "coverage": copy.deepcopy(coverage)}
    rows = []
    for step in range(1, 193):
        samples, sampling = sample_step(events, rng, coverage)
        rows.append({"step": step, "horizon_s": horizon_for(step), **sampling,
            "rng_after": copy.deepcopy(rng.bit_generator.state), "coverage_after": copy.deepcopy(coverage),
            "genuine_steps": [min(round(horizon_for(step) * 100), len(e["x"]) - 11) for e, _ in samples]})
    covered = {events[i]["id"] for row in rows[:75] for i in row["coverage_indices"]}
    if covered != {e["id"] for e in events}:
        raise ValueError("first75 coverage did not reach every fit event")
    return {"initial": initial, "rows": rows}


def prepare(base):
    base = Path(base).resolve(); base.mkdir(parents=True, exist_ok=True)
    if (base / "prepare.json").exists():
        raise FileExistsError("study preparation is immutable")
    sources = source_identities()
    write(base / "sources.json", sources, exclusive=True)
    prior, fit = old.checked_preparation(OLD_BASE, fit_only=True)
    if len(fit) != 297 or sum(e["label"] == "stable" for e in fit) != 267 or len(prior["no_update_ids"]) != 88:
        raise ValueError("registered fit297/no-update88 cohort changed")
    common = set(prior["fit_ids"]) | set(prior["no_update_ids"])
    if (len(common) != 385 or set(prior["fit_ids"]) & set(prior["no_update_ids"])
            or len(prior["original366_ids"]) != 366 or len(prior["fresh19_ids"]) != 19
            or set(prior["original366_ids"]) & set(prior["fresh19_ids"])
            or set(prior["original366_ids"]) | set(prior["fresh19_ids"]) != common):
        raise ValueError("common385 original366/fresh19 identity partition changed")
    panel = panel_events(fit)
    schedule = {str(seed): make_schedule(fit, seed) for seed in SEEDS}
    write(base / "schedule.json", schedule, exclusive=True)
    record = {key: prior[key] for key in ("fit_ids", "no_update_ids", "original366_ids", "fresh19_ids",
              "fit_groups", "no_update_groups", "fit_cache_sha256", "no_update_cache_sha256")}
    record.update(status="COMPLETED", sources=sources, code=code_identity(),
        old_base=str(OLD_BASE), old_prepare_sha256=sha(OLD_BASE / "prepare.json"),
        schedule_sha256=sha(base / "schedule.json"), panel_ids=[e["id"] for e in panel],
        panel_denominators={str(h): sum(len(e["x"]) - 11 >= h * 100 for e in panel) for h in (1, 2, 5, 8)},
        source_already_exposed_to_all_groups=True, no_update_cache_parsed=False, val_read=False, test_read=False)
    write(base / "prepare.json", record, exclusive=True)
    return record


def checked_preparation(base, *, fit_only=True):
    base = Path(base); record = read(base / "prepare.json")
    if record["sources"] != source_identities() or record["code"] != code_identity():
        raise ValueError("force-graph source/code binding changed")
    if sha(base / "schedule.json") != record["schedule_sha256"] or sha(OLD_BASE / "prepare.json") != record["old_prepare_sha256"]:
        raise ValueError("prepared schedule/parent metadata changed")
    prior, events = old.checked_preparation(OLD_BASE, fit_only=fit_only)
    if any(prior[k] != record[k] for k in ("fit_ids", "no_update_ids", "fit_cache_sha256", "no_update_cache_sha256")):
        raise ValueError("prepared train identity changed")
    return record, events


def build_model(mode, source="A512", device="cuda"):
    if mode not in MODES or source not in SOURCES:
        raise ValueError("unregistered model/source")
    return ForceGraphROM.from_source_export(SOURCES[source] / "model.json", mode).to(resolve_device(device)).double()


class EventStore:
    """Only observed x enters this objective; no torque or hidden-state targets."""
    def __init__(self, events, device):
        self.events = {e["id"]: torch.as_tensor(np.asarray(e["x"]), dtype=torch.float64, device=device) for e in events}
        if len(self.events) != len(events) or any(x.ndim != 2 or x.shape[1] != 30 or len(x) <= 11 or not torch.isfinite(x).all() for x in self.events.values()):
            raise ValueError("unique finite genuine Nx30 events required")


def event_losses(model, store, samples, horizon, method="rk4"):
    if method not in ("rk4", "fast") or horizon <= 0 or not samples or any(i != 10 for _, i in samples):
        raise ValueError("registered integration and unique initial endpoint10 required")
    xs = [store.events[e["id"]] for e, _ in samples]
    counts = [min(round(horizon * 100), len(x) - 11) for x in xs]
    state = model.initialize(torch.stack([x[:11] for x in xs]))
    sums = state.new_zeros(len(xs))
    stepper = model.rk4 if method == "rk4" else model.fast_step
    for step in range(1, max(counts) + 1):
        active = [i for i, count in enumerate(counts) if count >= step]
        ix = torch.tensor(active, dtype=torch.long, device=state.device)
        next_state = stepper(state.index_select(0, ix), .01)
        truth = torch.stack([xs[i][10 + step] for i in active])
        sums = sums.index_add(0, ix, _point_error(next_state, truth, model))
        state = state.index_copy(0, ix, next_state)
    return sums / state.new_tensor(counts)


def objective(model, store, samples, horizon, method="rk4"):
    losses = event_losses(model, store, samples, horizon, method)
    r = risk(losses)
    return 10 * r, {"risk": r, "mean_point_error": losses.mean(), "objective_J": 10 * r}


def make_optimizer(model, multiplier=1):
    if multiplier not in (1, 3):
        raise ValueError("only registered LR multipliers1/3")
    return torch.optim.Adam([{"params": [p], "block": name, "lr": .01 * multiplier}
        for name, p in model.named_parameters() if p.requires_grad], betas=(.9, .999), eps=1e-8, weight_decay=0.)


def _journal(base, event, category, **fields):
    path = Path(base) / "optimizer_calls.jsonl"
    if event == "optimizer_step_attempt":
        calls = call_counts(base)
        if calls["attempts"] >= 1304:
            raise ValueError("registered total optimizer-call bound exhausted")
        limits = {"qa": 108, "calibration": 48, "formal": 960, "replay": 80}
        if calls["by_category"].get(category, {}).get("attempts", 0) >= limits[category]:
            raise ValueError("registered optimizer category exhausted")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": event, "category": category, "pid": os.getpid(),
            "time_unix": time.time(), **fields}, allow_nan=False) + "\n")


def call_counts(base):
    path = Path(base) / "optimizer_calls.jsonl"
    result = {"attempts": 0, "completed": 0, "by_category": {}}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); key = {"optimizer_step_attempt": "attempts", "optimizer_step_completed": "completed"}.get(row["event"])
            if key:
                result[key] += 1
                block = result["by_category"].setdefault(row["category"], {"attempts": 0, "completed": 0})
                block[key] += 1
    return result


def one_update(model, optimizer, store, samples, horizon, *, method, base, category, run_id, step, source):
    synchronize(model.mass.device); started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with graph(model):
        loss, terms = objective(model, store, samples, horizon, method)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite objective; not physical instability")
        loss.backward()
    parameters = list(model.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("missing/nonfinite gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, 100.)
    if not torch.isfinite(norm):
        raise FloatingPointError("nonfinite gradient norm")
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    fields = {"mode": model.mode, "source": source, "run_id": run_id, "step": step}
    _journal(base, "optimizer_step_attempt", category, **fields)
    optimizer.step()
    # Journal immediately records the actual completed call, even if subsequent validation fails.
    _journal(base, "optimizer_step_completed", category, **fields)
    model.project_nonnegative_(); synchronize(model.mass.device)
    if any(not torch.isfinite(p).all() for p in parameters) or not old._adam_states_finite(optimizer.state.values()):
        raise FloatingPointError("nonfinite model/Adam after optimizer")
    changes = {}
    for name, parameter in model.named_parameters():
        old_norm = before[name].norm(); change = (parameter.detach() - before[name]).norm()
        changes[name] = {"old_l2": float(old_norm), "change_l2": float(change),
                         "relative_l2": float(change / old_norm.clamp_min(1.))}
    return {"step": step, "horizon_s": horizon, "method": method,
        **{k: float(v.detach()) for k, v in terms.items()}, "gradient_norm": float(norm),
        "block_updates": changes, "seconds": time.perf_counter() - started}


def _seed(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _rng_state(rng, coverage):
    return {"numpy_rng": copy.deepcopy(rng.bit_generator.state), "coverage": copy.deepcopy(coverage),
        "numpy_global_rng": np.random.get_state(), "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def save_checkpoint(output, model, optimizer, rng, coverage, step, identity, trace):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    final = output / f"step{step:06d}"
    if final.exists():
        raise FileExistsError("immutable force-graph checkpoint exists")
    if model._cache is not None or any(not torch.isfinite(p).all() for p in model.parameters()) or not old._adam_states_finite(optimizer.state.values()):
        raise ValueError("checkpoint requires closed cache and finite model/Adam")
    pending = output / (".pending-" + uuid.uuid4().hex); pending.mkdir()
    model.export(pending / "model.json")
    torch.save({**_rng_state(rng, coverage), "optimizer": optimizer.state_dict(), "step": step}, pending / "training.pt")
    meta = {"mode": model.mode, "step": step, "identity": identity, "trace": trace,
        "parameter_layout": old._parameter_layout(model, optimizer), "runtime": old.runtime_identity(model.mass.device),
        "model_sha256": sha(pending / "model.json"), "training_sha256": sha(pending / "training.pt"),
        "scope": "research_candidate", "resumable": True}
    write(pending / "checkpoint.json", meta, exclusive=True); pending.rename(final)
    write(output / "latest.json", {"directory": final.name, "checkpoint_sha256": sha(final / "checkpoint.json")})
    return final


def restore_checkpoint(output, device, identity, *, multiplier=1):
    output = Path(output); pointer = read(output / "latest.json"); directory = pointer["directory"]
    if Path(directory).name != directory:
        raise ValueError("checkpoint must name immediate child")
    folder = output / directory
    if sha(folder / "checkpoint.json") != pointer["checkpoint_sha256"]:
        raise ValueError("checkpoint metadata hash changed")
    meta = read(folder / "checkpoint.json"); device = resolve_device(device)
    if meta["identity"] != identity or not meta["resumable"] or meta["runtime"] != old.runtime_identity(device):
        raise ValueError("checkpoint identity/runtime differs")
    for filename, key in (("model.json", "model_sha256"), ("training.pt", "training_sha256")):
        if sha(folder / filename) != meta[key]:
            raise ValueError("checkpoint artifact hash changed")
    model = ForceGraphROM.from_export(folder / "model.json").to(device).double()
    optimizer = make_optimizer(model, multiplier)
    if old._parameter_layout(model, optimizer) != meta["parameter_layout"]:
        raise ValueError("Adam parameter layout/LR differs")
    state = torch.load(folder / "training.pt", map_location="cpu", weights_only=False)
    if state["step"] != meta["step"]:
        raise ValueError("model/Adam step mismatch")
    optimizer.load_state_dict(state["optimizer"])
    if state["step"] and (not optimizer.state or any(int(s["step"]) != state["step"] for s in optimizer.state.values())):
        raise ValueError("Adam counter differs from real updates")
    if not old._adam_states_finite(optimizer.state.values()):
        raise FloatingPointError("restored Adam nonfinite")
    rng = np.random.default_rng(); rng.bit_generator.state = state["numpy_rng"]
    np.random.set_state(state["numpy_global_rng"]); random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        if state["cuda_rng"] is None:
            raise ValueError("missing CUDA RNG")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return model, optimizer, rng, state["coverage"], meta


def _risk_rows(rows, key="mean_point_error"):
    if not rows or any(not r.get("finite") or not r.get("numerically_admissible") or r.get(key) is None for r in rows):
        return None
    return float(risk(torch.tensor([r[key] for r in rows], dtype=torch.float64)))


def panel_summary(records, extras, *, seconds, parameter_count):
    risks = {}
    qualified = {r["event_id"]: r["scientific_numeric_8s"] for r in extras}
    by_id = {r["event_id"]: r for r in extras}
    for h in ("5", "8"):
        rows = [r for r in records["objective_point_error"] if r["horizon"] == h and r["label"] == "stable" and r["available"]]
        valid = bool(rows) and all(qualified.get(r["event_id"], False) for r in rows)
        value = _risk_rows(rows) if valid else None
        coarse = [by_id[r["event_id"]]["coarse_point_errors"][h] for r in rows]
        cv = float(risk(torch.tensor(coarse, dtype=torch.float64))) if value is not None and all(v is not None for v in coarse) else None
        risks[h] = {"value": value, "coarse_value": cv, "numerical_score_perturbation": abs(value - cv) if cv is not None else None,
            "eligible": len(rows), "unresolved_ids": [r["event_id"] for r in rows if not qualified.get(r["event_id"], False) or not r.get("finite") or not r.get("numerically_admissible")]}
    unstable = [r for r in records["objective_point_error"] if r["horizon"] == "prefix" and r["label"] == "unstable" and r["available"]]
    unstable_risk = _risk_rows(unstable) if all(qualified.get(r["event_id"], False) for r in unstable) else None
    def pooled(key):
        rows = [r for r in records[key] if r["horizon"] == "8" and r["label"] == "stable" and r["available"]]
        if not rows or any(not r["finite"] or not r["numerically_admissible"] for r in rows):
            return None
        return math.sqrt(sum(r["sse"] for r in rows) / sum(r["n"] for r in rows))
    classification = old._classification_counts(extras, "classification_8s", qualified)
    return {"horizon_risks": risks, "classification_8s": classification,
        "unstable_interval_risk": unstable_risk, "unstable_interval_eligible": len(unstable),
        "common_frequency_rmse_8s": pooled("common_frequency"), "normalized_p_rmse_8s": pooled("normalized_p"),
        "all_8s_rows_qualified": bool(extras) and all(qualified.values()),
        "parameter_count": parameter_count, "qualified_inference_seconds_per_event": seconds / len(extras) if extras else None,
        "timing_scope": "CPU float64 one thread; true RK4 .005/.01 including causal initialization, output, metrics"}


def score_panel(model, events):
    from evaluate_structured_rom import _rollouts, _pair_row
    from summarize_train_local_memory_0098 import event_metrics
    from fixed_edge_pair_20261019 import scientific_qualification
    torch.set_num_threads(1)
    started = time.perf_counter(); candidate = copy.deepcopy(model).cpu().double().eval()
    records = {key: [] for key in ("records", "common_frequency", "normalized_p", "objective_point_error", "numerical")}
    extras, diagnostics = [], []
    for first in range(0, len(events), 16):
        block = events[first:first + 16]; histories = np.stack([e["x"][:11] for e in block])
        with torch.no_grad():
            system = candidate.history_system(histories)
            diag = candidate.initialization_diagnostics(histories)
            diag.update(event_ids=[e["id"] for e in block],
                r0_over_rstar=(system["r0"] / candidate.rstar).cpu().tolist(),
                reconstructed_p_maxabs=float(system["p_history"].abs().max()))
            diagnostics.append(diag)
        error = None
        try:
            fine = _rollouts(candidate, histories, 1600, .005)
            coarse = _rollouts(candidate, histories, 800, .01)
        except Exception as exc:
            error = repr(exc); fine = np.full((len(block), 1601, 40), np.nan); coarse = np.full((len(block), 801, 40), np.nan)
        for event, history, f, c in zip(block, histories, fine, coarse):
            metrics = event_metrics(candidate, event, f, c)
            for key, rows in metrics.items():
                records[key].extend(rows)
            pair = _pair_row(f, c, history, event, event["classification_meta"], candidate.mass.numpy())
            points = {}
            for h in ("5", "8"):
                n = int(h) * 100
                points[h] = None
                if len(event["x"]) - 11 >= n and np.isfinite(c[:n + 1]).all():
                    with torch.no_grad():
                        points[h] = float(_point_error(torch.as_tensor(c[1:n + 1, :30]), torch.as_tensor(event["x"][11:11 + n]), candidate).mean())
            extras.append({"event_id": event["id"], "label": event["label"], "classification_8s": pair,
                "scientific_numeric_8s": scientific_qualification(pair, metrics["numerical"]), "coarse_point_errors": points, "error": error})
    profile = panel_summary(records, extras, seconds=time.perf_counter() - started,
        parameter_count=candidate.complexity_counts()["raw_trainable_scalar_count"])
    return {**profile, "rows": safe(records), "extras": safe(extras), "initialization_diagnostics": safe(diagnostics),
        "event_ids": [e["id"] for e in events], "no_update_data_used": False, "val_read": False, "test_read": False}


def _valid_profile(profile):
    return bool(profile.get("all_8s_rows_qualified")) and all(profile["horizon_risks"][h]["value"] is not None and
        profile["horizon_risks"][h]["numerical_score_perturbation"] is not None for h in ("5", "8")) and profile.get("unstable_interval_risk") is not None


def extension_decision(before, after, trace):
    recent = trace[-48:]
    clip_count = sum(row["gradient_norm"] > 100 for row in recent)
    groups = {name for row in recent for name in row["block_updates"]}
    jumps = {name: sum(row["block_updates"].get(name, {}).get("relative_l2", 0) > .25 for row in recent) for name in groups}
    optimization_unresolved = len(recent) != 48 or clip_count > 12 or any(v >= 12 for v in jumps.values())
    result = {"extend": False, "optimization_unresolved": optimization_unresolved, "clip_count_last48": clip_count,
              "large_change_counts_last48": jumps, "reason": None}
    if optimization_unresolved:
        return {**result, "reason": "optimization_state_unresolved"}
    if not _valid_profile(before) or not _valid_profile(after):
        return {**result, "reason": "numerical_or_score_unresolved"}
    gains = {}; harms = []
    for h in ("5", "8"):
        a, b = before["horizon_risks"][h], after["horizon_risks"][h]
        absolute = a["value"] - b["value"]
        gains[h] = absolute / max(a["value"], 1e-12) >= .01 and absolute > a["numerical_score_perturbation"] + b["numerical_score_perturbation"]
        if b["value"] > 1.05 * a["value"]:
            harms.append("risk_" + h)
    for key in ("missed_resolved_unstable", "false_los_resolved_stable"):
        if after["classification_8s"][key] > before["classification_8s"][key]:
            harms.append(key)
    if after["unstable_interval_risk"] > 1.05 * before["unstable_interval_risk"]:
        harms.append("unstable_interval_risk")
    return {**result, "extend": any(gains.values()) and not harms, "gains": gains, "harms": harms,
        "reason": "registered_extension" if any(gains.values()) and not harms else "tradeoff_or_insufficient_registered_gain"}


def select_multipliers(scores):
    selected = {}
    for mode in MODES:
        choices = [(entry["ratio"], int(mult)) for mult, entry in scores[mode].items()
                   if entry.get("qualified") and entry.get("ratio") is not None and math.isfinite(entry["ratio"])]
        selected[mode] = min(choices)[1] if choices else None
    return selected


def _comparison_vector(profile):
    if not _valid_profile(profile):
        return None
    values = [profile["horizon_risks"][h]["value"] for h in ("5", "8")]
    values += [profile.get("common_frequency_rmse_8s"), profile.get("normalized_p_rmse_8s"),
        profile["unstable_interval_risk"], profile["classification_8s"]["missed_resolved_unstable"],
        profile["classification_8s"]["false_los_resolved_stable"], profile["parameter_count"],
        profile.get("qualified_inference_seconds_per_event")]
    return values if all(v is not None and math.isfinite(v) for v in values) else None


def select_backup(profiles):
    if any(mode not in profiles for mode in MODES) or not _valid_profile(profiles["L"]):
        return {"structure": None, "control": None, "reason": "required_primary_fit_comparison_unresolved"}
    reference = profiles["L"]; candidates = []
    def order(mode):
        p = profiles[mode]
        return (p["classification_8s"]["missed_resolved_unstable"], p["classification_8s"]["false_los_resolved_stable"],
            max(p["horizon_risks"][h]["value"] / max(reference["horizon_risks"][h]["value"], 1e-12) for h in ("5", "8")),
            p["common_frequency_rmse_8s"], p["normalized_p_rmse_8s"], p["unstable_interval_risk"],
            p["qualified_inference_seconds_per_event"], mode)
    vectors = {mode: _comparison_vector(p) for mode, p in profiles.items()}
    for mode in ("S", "M"):
        p = profiles[mode]
        if vectors[mode] is None:
            continue
        gains, harms = [], []
        for h in ("5", "8"):
            a, b = reference["horizon_risks"][h], p["horizon_risks"][h]
            difference = a["value"] - b["value"]
            gains.append(difference / max(a["value"], 1e-12) >= .01 and difference > a["numerical_score_perturbation"] + b["numerical_score_perturbation"])
            harms.append(b["value"] > 1.05 * a["value"])
        harms += [p["classification_8s"][key] > reference["classification_8s"][key] for key in ("missed_resolved_unstable", "false_los_resolved_stable")]
        harms.append(p["unstable_interval_risk"] > 1.05 * reference["unstable_interval_risk"])
        dominated = any(other != mode and vector is not None and all(a <= b for a, b in zip(vector, vectors[mode]))
            and any(a < b for a, b in zip(vector, vectors[mode])) for other, vector in vectors.items())
        if any(gains) and not any(harms) and not dominated:
            candidates.append(mode)
    if not candidates:
        return {"structure": None, "control": None, "reason": "no_structure_meets_registered_fit_rule"}
    structure = min(candidates, key=order)
    controls = [mode for mode in MODES if mode != structure and vectors[mode] is not None]
    if not controls:
        return {"structure": None, "control": None, "reason": "qualified_control_unavailable"}
    return {"structure": structure, "control": min(controls, key=order), "reason": "registered_fit_only_source_sensitivity_check"}


def qualify(base, device="cuda"):
    from evaluate_structured_rom import _rollouts, _pair_row, _scored_numerics
    from fixed_edge_pair_20261019 import scientific_qualification
    torch.set_num_threads(1)
    base = Path(base); prepared, fit = checked_preparation(base)
    output = base / "qualification"; output.mkdir(exist_ok=False)
    panel = [e for e in fit if e["id"] in prepared["panel_ids"] and e["label"] == "stable"]
    selected = sorted(panel, key=lambda e: (-len(e["x"]), e["id"]))[:12]
    if len(selected) != 12 or any(len(e["x"]) - 11 < 800 for e in selected):
        raise ValueError("registered qualification panel unavailable")
    rows, all_fast, all_reference = [], True, True
    for mode in MODES:
        result = {"mode": mode, "events": [e["id"] for e in selected], "cost": {}, "numerics": []}
        for method in ("rk4", "fast"):
            model = build_model(mode, device=device); dev = model.mass.device; store = EventStore(selected, dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            synchronize(dev); tick = time.perf_counter()
            try:
                with graph(model):
                    loss, _ = objective(model, store, [(e, 10) for e in selected], 8., method)
                    loss.backward()
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("nonfinite complete objective/gradient")
                synchronize(dev)
                result["cost"][method] = {"qualified_finite_backward": True, "seconds": time.perf_counter() - tick,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
                    "objective_J": float(loss.detach()), "actual_optimizer_updates": 0}
            except Exception as exc:
                result["cost"][method] = {"qualified_finite_backward": False, "error": repr(exc), "seconds": time.perf_counter() - tick}
            del model, store
        candidate = build_model(mode, device="cpu").eval(); histories = np.stack([e["x"][:11] for e in selected])
        try:
            ref = _rollouts(candidate, histories, 1600, .005)
            coarse_ref = _rollouts(candidate, histories, 800, .01)
            with torch.no_grad():
                fast_fine = candidate.rollout(histories, 1600, .005, method="fast").numpy()
                fast_coarse = candidate.rollout(histories, 800, .01, method="fast").numpy()
            for e, h, f, c, ff, fc in zip(selected, histories, ref, coarse_ref, fast_fine, fast_coarse):
                checks = {}
                for key, a, b in (("rk4_coarse_fine", f, c), ("fast_coarse_fine", ff, fc), ("fast_against_fine_rk4", f, fc)):
                    pair = _pair_row(a, b, h, e, e["classification_meta"], candidate.mass.numpy())
                    scored = _scored_numerics(a, b, e, candidate.mass.numpy())
                    checks[key] = {"qualified": scientific_qualification(pair, scored), "pair": pair, "scored": scored}
                result["numerics"].append({"event_id": e["id"], **checks})
        except Exception as exc:
            result["numerical_exception"] = repr(exc)
        reference_ok = len(result["numerics"]) == 12 and all(row["rk4_coarse_fine"]["qualified"] for row in result["numerics"]) and result["cost"]["rk4"]["qualified_finite_backward"]
        fast_ok = reference_ok and all(row[key]["qualified"] for row in result["numerics"] for key in ("fast_coarse_fine", "fast_against_fine_rk4")) and result["cost"]["fast"]["qualified_finite_backward"]
        result.update(reference_qualified=reference_ok, fast_qualified=fast_ok)
        all_reference &= reference_ok; all_fast &= fast_ok; rows.append(result)
        write(output / (mode + ".json"), result, exclusive=True)
    result = {"status": "COMPLETED" if all_reference else "UNRESOLVED", "training_method": "fast" if all_fast else "rk4",
        "all_fast_qualified": bool(all_fast), "all_reference_qualified": bool(all_reference), "models": rows,
        "actual_optimizer_updates": 0, "no_update_data_used": False, "val_read": False, "test_read": False}
    write(output / "summary.json", result, exclusive=True); return result


def _method(base):
    result = read(Path(base) / "qualification/summary.json")
    if result["status"] != "COMPLETED" or not result["all_reference_qualified"]:
        raise ValueError("complete objective and RK4 numerical qualification required")
    method = result["training_method"]
    if method not in ("rk4", "fast") or (method == "fast" and not result["all_fast_qualified"]):
        raise ValueError("invalid common training method")
    return method


def qa(base, device="cuda"):
    base = Path(base); output = base / "qa"; output.mkdir(exist_ok=False)
    device = resolve_device(device)
    if device.type != "cuda":
        raise ValueError("registered save/recovery QA must execute on CUDA")
    torch.set_num_threads(1); events = old.synthetic_fixture(); store = EventStore(events, device)
    results = []
    for mode in MODES:
        folder = output / mode; folder.mkdir(); identity = {"fixture_only": True, "mode": mode, "real_resume_step": 16,
            "code": code_identity(), "sources": source_identities()}
        def initial():
            model = build_model(mode, device=device); optimizer = make_optimizer(model)
            _seed(28101, device); rng = np.random.default_rng(28101)
            return model, optimizer, rng, {"cover_order": rng.permutation(len(events)).tolist(), "cursor": 0}
        def advance(model, optimizer, rng, coverage, start, stop, label):
            trace = []
            for step in range(start + 1, stop + 1):
                samples, _ = sample_step(events, rng, coverage)
                random.random(); np.random.random(); torch.rand((), device=device)
                trace.append(one_update(model, optimizer, store, samples, .02, method="rk4", base=base,
                    category="qa", run_id=f"qa_{mode}_{label}", step=step, source="A512"))
            return trace
        model_a, opt_a, rng_a, cov_a = initial()
        current_model, current_opt, current_rng, current_cov, current_trace = model_a, opt_a, rng_a, cov_a, []
        try:
            current_trace = advance(model_a, opt_a, rng_a, cov_a, 0, 18, "continuous")
            final_rng = old._cpu_snapshot(_rng_state(rng_a, cov_a))
            model_b, opt_b, rng_b, cov_b = initial()
            current_model, current_opt, current_rng, current_cov, current_trace = model_b, opt_b, rng_b, cov_b, []
            trace_b = advance(model_b, opt_b, rng_b, cov_b, 0, 16, "restored")
            current_trace = trace_b
            snapshot = old._cpu_snapshot({"model": model_b.state_dict(), "Adam": opt_b.state_dict(), "RNG": _rng_state(rng_b, cov_b),
                "layout": old._parameter_layout(model_b, opt_b)})
            if not any(bool(torch.count_nonzero(s["exp_avg"])) for s in opt_b.state.values()):
                raise ValueError("real16 Adam first moment must be nonzero")
            checkpoint = save_checkpoint(folder / "checkpoints", model_b, opt_b, rng_b, cov_b, 16, identity, trace_b)
            restored, opt_c, rng_c, cov_c, meta = restore_checkpoint(folder / "checkpoints", device, identity)
            current_model, current_opt, current_rng, current_cov = restored, opt_c, rng_c, cov_c
            old._exact_nested(snapshot, old._cpu_snapshot({"model": restored.state_dict(), "Adam": opt_c.state_dict(),
                "RNG": _rng_state(rng_c, cov_c), "layout": old._parameter_layout(restored, opt_c)}))
            current_trace = trace_b + advance(restored, opt_c, rng_c, cov_c, 16, 18, "restored")
            parameter_error = old._qa_compare(old._cpu_snapshot(model_a.state_dict()), old._cpu_snapshot(restored.state_dict()))
            a, c = old._cpu_snapshot(opt_a.state_dict()), old._cpu_snapshot(opt_c.state_dict())
            old._exact_nested(a["param_groups"], c["param_groups"])
            moment_error = max(old._qa_compare(a["state"][k], c["state"][k]) for k in a["state"])
            old._exact_nested(final_rng, old._cpu_snapshot(_rng_state(rng_c, cov_c)))
            row = {"mode": mode, "status": "PASS", "actual_optimizer_updates": 36, "saved_step": meta["step"],
                "continuous": 18, "before_save": 16, "after_restore": 2, "checkpoint": str(checkpoint),
                "parameter_maxabs": parameter_error, "Adam_maxabs": moment_error,
                "serialization_model_Adam_RNG_layout_exact": True, "nonzero_Adam_exp_avg": True,
                "continuation_rtol": 2e-9, "continuation_atol": 2e-9}
            results.append(row); write(folder / "result.json", row, exclusive=True)
        except BaseException as exc:
            quarantine = old.quarantine_failure(folder, current_model, current_opt,
                current_rng, current_cov, len(current_trace), current_trace, exc)
            write(output / "failure.json", {"error": repr(exc), "quarantine": quarantine, "calls": call_counts(base)}, exclusive=True)
            raise
    result = {"status": "PASS", "actual_optimizer_updates": 108, "models": results,
        "fixture_only": True, "real_train_payloads_read": False, "val_read": False, "test_read": False}
    write(output / "summary.json", result, exclusive=True); return result


def calibrate(base, device="cuda"):
    torch.set_num_threads(1)
    base = Path(base); method = _method(base)
    if read(base / "qa/summary.json")["status"] != "PASS":
        raise ValueError("real CUDA resume QA required")
    prepared, events = checked_preparation(base); panel = [e for e in events if e["id"] in prepared["panel_ids"]]
    output = base / "calibration"; output.mkdir(exist_ok=False); schedule = read(base / "schedule.json")["28101"]
    lookup = {e["id"]: e for e in events}; store = EventStore(events, resolve_device(device)); scores = {mode: {} for mode in MODES}
    calls_before = call_counts(base)
    for mode in MODES:
        zero = score_panel(build_model(mode, device="cpu"), panel); write(output / f"{mode}_zero.json", zero, exclusive=True)
        for multiplier in (1, 3):
            folder = output / f"{mode}_{multiplier}"; folder.mkdir()
            model = build_model(mode, device=device); optimizer = make_optimizer(model, multiplier); _seed(28101, model.mass.device)
            rng = np.random.default_rng(); rng.bit_generator.state = schedule["initial"]["rng"]
            coverage = copy.deepcopy(schedule["initial"]["coverage"]); trace = []
            trial_before = call_counts(base)
            try:
                for row in schedule["rows"][:8]:
                    trace.append(one_update(model, optimizer, store, [(lookup[eid], 10) for eid in row["sample_ids"]], row["horizon_s"],
                        method=method, base=base, category="calibration", run_id=f"cal_{mode}_{multiplier}", step=row["step"], source="A512"))
                    rng.bit_generator.state = row["rng_after"]; coverage = copy.deepcopy(row["coverage_after"])
                profile = score_panel(model, panel)
                valid = len(trace) == 8 and _valid_profile(zero) and _valid_profile(profile)
                ratio = sum(profile["horizon_risks"][h]["value"] / max(zero["horizon_risks"][h]["value"], 1e-12) for h in ("5", "8")) / 2 if valid else None
                entry = {"qualified": valid, "ratio": ratio, "profile": profile, "trace": trace, "trial_state_discarded": True}
            except BaseException as exc:
                quarantine = old.quarantine_failure(folder, model, optimizer, rng, coverage, len(trace), trace, exc)
                entry = {"qualified": False, "ratio": None, "error": repr(exc), "quarantine": quarantine, "trace": trace, "trial_state_discarded": True}
            trial_after = call_counts(base)
            entry["optimizer_attempts"] = trial_after["attempts"] - trial_before["attempts"]
            entry["optimizer_completed"] = trial_after["completed"] - trial_before["completed"]
            scores[mode][str(multiplier)] = entry; write(folder / "result.json", entry, exclusive=True)
    calls_after = call_counts(base)
    actual = calls_after["completed"] - calls_before["completed"]
    attempts = calls_after["attempts"] - calls_before["attempts"]
    result = {"status": "COMPLETED" if actual == attempts and attempts <= 48 else "UNRESOLVED", "multipliers": select_multipliers(scores),
        "actual_optimizer_updates": actual, "actual_optimizer_attempts": attempts, "all_trials_processed": True,
        "scores": scores, "training_method": method, "no_update_data_used": False,
        "val_read": False, "test_read": False}
    write(output / "summary.json", result, exclusive=True); return result


def _formal_run(base, source, mode, seed, multiplier, prepared, fit, panel, method, device):
    run_id = f"{source}_{mode}_{seed}"; folder = Path(base) / "study" / run_id; folder.mkdir(exist_ok=False)
    model = build_model(mode, source, device=device); optimizer = make_optimizer(model, multiplier)
    _seed(seed, model.mass.device); schedule = read(Path(base) / "schedule.json")[str(seed)]
    rng = np.random.default_rng(); rng.bit_generator.state = schedule["initial"]["rng"]
    coverage = copy.deepcopy(schedule["initial"]["coverage"]); trace = []; profiles = {}
    identity = {"source": source, "mode": mode, "seed": seed, "prepare_sha256": sha(Path(base) / "prepare.json"),
                "multiplier": multiplier, "training_method": method, "sources": prepared["sources"], "code": prepared["code"]}
    store = EventStore(fit, model.mass.device); lookup = {e["id"]: e for e in fit}
    try:
        profiles["0"] = score_panel(model, panel); write(folder / "panel000000.json", profiles["0"], exclusive=True)
        save_checkpoint(folder / "checkpoints", model, optimizer, rng, coverage, 0, identity, trace)
        end, decision = 96, None
        for row in schedule["rows"]:
            step = row["step"]
            if step > end:
                break
            trace.append(one_update(model, optimizer, store, [(lookup[eid], 10) for eid in row["sample_ids"]], row["horizon_s"],
                method=method, base=base, category="formal", run_id=run_id, step=step, source=source))
            rng.bit_generator.state = row["rng_after"]; coverage = copy.deepcopy(row["coverage_after"])
            if step % 16 == 0:
                save_checkpoint(folder / "checkpoints", model, optimizer, rng, coverage, step, identity, trace)
            if step in (48, 96, 192):
                profiles[str(step)] = score_panel(model, panel); write(folder / f"panel{step:06d}.json", profiles[str(step)], exclusive=True)
            if step == 96:
                decision = extension_decision(profiles["48"], profiles["96"], trace)
                write(folder / "extension_decision.json", decision, exclusive=True)
                if decision["extend"]:
                    end = 192
        # Held payloads are opened only at the report endpoint, after the fit-only stop decision.
        _, all_events = checked_preparation(base, fit_only=False)
        full = old.evaluate_model(model, all_events, folder / "evaluation385", prepared)
        result = {"status": "COMPLETED", "run_id": run_id, "source": source, "mode": mode, "seed": seed,
            "updates": len(trace), "endpoint": str(folder / "checkpoints" / f"step{len(trace):06d}"),
            "profile": profiles[str(len(trace))], "profiles": {k: str(folder / f"panel{int(k):06d}.json") for k in profiles},
            "extension_decision": decision, "full_evaluation": str(folder / "evaluation385/evidence.json"),
            "optimization_scope": "finite adaptation; completing updates does not establish sufficient optimization",
            "upper_bound_still_improving": extension_decision(profiles["96"], profiles["192"], trace)["extend"] if len(trace) == 192 else False,
            "all385_required_rows_qualified": full["profiles"]["common385"]["all_required_rows_qualified"]}
        write(folder / "result.json", result, exclusive=True); return result
    except BaseException as exc:
        quarantine = old.quarantine_failure(folder, model, optimizer, rng, coverage, len(trace), trace, exc)
        result = {"status": "UNRESOLVED", "run_id": run_id, "source": source, "mode": mode, "updates": len(trace),
                  "error": repr(exc), "quarantine": quarantine, "resumable": False, "failure_is_physical_instability": False}
        write(folder / "result.json", result, exclusive=True); return result


def study(base, device="cuda"):
    torch.set_num_threads(1)
    base = Path(base); method = _method(base); calibration = read(base / "calibration/summary.json")
    from experiments.force_graph_execution_20261028 import validate_calibration, call_counts as checked_calls
    totals = validate_calibration(calibration)
    calls = checked_calls(base)
    if calls["pending_attempts"] or calls["categories"]["calibration"] != totals:
        raise ValueError("calibration requires an actual reconciled optimizer journal")
    prepared, fit = checked_preparation(base, fit_only=True)
    panel = [e for e in fit if e["id"] in prepared["panel_ids"]]
    output = base / "study"; output.mkdir(exist_ok=False)
    results, profiles = [], {}
    for mode in MODES:
        multiplier = calibration["multipliers"][mode]
        if multiplier is None:
            results.append({"mode": mode, "source": "A512", "status": "UNRESOLVED", "reason": "no_qualified_calibration_multiplier", "updates": 0})
            continue
        result = _formal_run(base, "A512", mode, 28101, multiplier, prepared, fit, panel, method, device)
        results.append(result)
        if result["status"] == "COMPLETED":
            profiles[mode] = result["profile"]
    selection = select_backup(profiles); write(output / "backup_selection.json", selection, exclusive=True)
    if selection["structure"] is not None:
        for mode in (selection["structure"], selection["control"]):
            results.append(_formal_run(base, "N6101", mode, 28102, calibration["multipliers"][mode], prepared, fit, panel, method, device))
    _, all_events = checked_preparation(base, fit_only=False)
    held = [e for e in all_events if e["id"] in prepared["no_update_ids"]]
    noisy = []
    for result in results:
        if result["status"] != "COMPLETED":
            continue
        model = ForceGraphROM.from_export(Path(result["endpoint"]) / "model.json").cpu().double()
        folder = output / result["run_id"] / "noise88"
        evidence = old.evaluate_model(model, held, folder, prepared, noisy=True)
        noisy.append({"run_id": result["run_id"], "evidence": str(folder / "evidence.json"),
            "event_count": evidence["event_count"], "noise_scope": "1% source-scaled delta/omega history; fixed per-event SHA256 stream; terminal p0 and genuine future unchanged; report only"})
    result = {"status": "COMPLETED" if all(r["status"] == "COMPLETED" for r in results) else "UNRESOLVED", "runs": results,
        "backup_selection": selection, "noise": noisy, "calls": call_counts(base), "default_model": None,
        "scientific_model_adoption": False, "val_read": False, "test_read": False}
    write(output / "summary.json", result, exclusive=True); return result


def summary(base):
    base = Path(base)
    paths = {"prepare": "prepare.json", "qualification": "qualification/summary.json", "qa": "qa/summary.json",
             "calibration": "calibration/summary.json", "study": "study/summary.json"}
    stages = {name: {"path": str(base / p), "sha256": sha(base / p), "status": read(base / p).get("status")} if (base / p).exists() else None for name, p in paths.items()}
    result = {"status": stages["study"]["status"] if stages["study"] else "INCOMPLETE", "stages": stages,
        "calls": call_counts(base), "default_model": None, "val_read": False, "test_read": False,
        "no_process_count_or_runtime_limit": True}
    write(base / "summary.json", result); return result
