"""Frozen-state numerical diagnostic. Importing this module does no scientific work."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
from itertools import combinations
import json
import os
from pathlib import Path, PureWindowsPath
import re
import time
import uuid

import numpy as np
import scipy
from scipy.integrate import DOP853, Radau
from scipy.optimize import linear_sum_assignment

from kiki.models import GFTDKuramotoModel, gftd_discovery as core, gftd_fit297
from kiki.physics.kuramoto_functional import bregman_kuramoto_potential


ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_initialization_precision_20261109"
PARENT = Path("private_data/gftd_fit297_20261107")
PARENT_RESULTS = Path("results/gftd_fit297_20261107")
PROTOCOL = Path("campaigns") / RUN / "protocol.md"
OUTPUT = Path("results") / RUN
PRIVATE = Path("artifacts") / RUN / "diagnostic"
PARENT_RECORDS = (
    PARENT_RESULTS / "raw_artifacts_manifest.json",
    PARENT_RESULTS / "initialization_evidence_digest.json",
    PARENT_RESULTS / "evidence_digest.json",
)
SOURCE_PATHS = (
    Path(__file__).resolve().relative_to(ROOT).as_posix(),
    "src/kiki/models/base.py",
    "src/kiki/models/gftd_discovery.py",
    "src/kiki/models/gftd_fit297.py",
    "src/kiki/models/gftd_fitting.py",
    "src/kiki/models/gftd_kuramoto.py",
    "src/kiki/physics/graph.py",
    "src/kiki/physics/kuramoto_functional.py",
)
MODES = ("static", "modulated")
GROUPS = ("stage2", "exp", "history", "future_rms_peak", "future_rms", "success")
GATE_NAMES = (
    "roundtrip_main_scaled_max", "roundtrip_fine_scaled_max", "history_main_fine_rms",
    "future_main_fine_rms", "future_main_fine_peak", "p_endpoint_error",
)
GATE_LIMITS = dict(zip(GATE_NAMES, (1e-7, 1e-7, 1e-7, 1e-5, 1e-4, 0.0), strict=True))
NUMERICAL_ERRORS = (FloatingPointError, OverflowError, ValueError)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, expected):
    path = Path(path)
    if path.is_absolute() or not (ROOT / path).resolve().is_relative_to(ROOT):
        raise ValueError(f"path must stay in the workspace: {path}")
    data = (ROOT / path).read_bytes()
    if sha(data) != expected:
        raise ValueError(f"frozen bytes changed: {path}")
    return data


def write_bytes(path, data):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f"{path.name}.pending-{uuid.uuid4().hex}")
    with pending.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(100):
        try:
            os.replace(pending, path)
            break
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(data), bytes=len(data))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def write_json(path, value):
    return write_bytes(path, json_bytes(value))


def write_arrays(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return write_bytes(path, buffer.getvalue())


def verify_release(release, phase):
    if (phase not in ("prepare", "prepare02", "diagnostic")
            or release.get("released") is not True or release.get("phase") != phase):
        raise PermissionError(f"root {phase} release required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", release.get("action_id", "")):
        raise ValueError("unique action_id required")
    sources, assets = release["source_sha256"], release["asset_sha256"]
    if not set(SOURCE_PATHS) <= sources.keys():
        raise ValueError("incomplete source bindings")
    if set(assets) != {path.as_posix() for path in PARENT_RECORDS}:
        raise ValueError("exact three public parent manifests required")
    for path, expected in {**sources, **assets, PROTOCOL.as_posix(): release["protocol_sha256"]}.items():
        if not (path.startswith(("src/", "experiments/", "tests/")) and path in sources
                or path in assets or path == PROTOCOL.as_posix()):
            raise ValueError(f"unexpected preclaim input: {path}")
        read_verified(path, expected)
    if phase == "prepare02":
        previous = json.loads(read_verified(OUTPUT / "prepare_execution.json", release["previous_prepare_sha256"]))
        if (previous["phase"] != "prepare" or previous["status"] != "INTERRUPTED_RUNTIME"
                or previous["completed"] or previous["action_id"] == release["action_id"]
                or (ROOT / OUTPUT / "selection.json").exists()):
            raise ValueError("second preparation requires the interrupted first preparation and no frozen selection")
    elif phase == "diagnostic":
        read_verified(OUTPUT / "selection.json", release["selection_sha256"])


def claim(phase, action_id, release_sha256):
    path = ROOT / OUTPUT / ("started.json" if phase == "diagnostic" else f"{phase}_started.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json_bytes(dict(action_id=action_id, release_sha256=release_sha256)))
        stream.flush()
        os.fsync(stream.fileno())


def gates(record):
    if record is None:
        return None
    values = {name: record.get(name) for name in GATE_NAMES}
    if "roundtrip_scaled_max" in record:
        values[GATE_NAMES[0]], values[GATE_NAMES[1]] = record["roundtrip_scaled_max"]
    return dict(passed=record["passed"], values=values, error=record.get("error"))


def failed_integration(counts):
    """Infer the failed leg and attempted step from the original per-check counters."""
    legs = ("history_back_main", "history_return_main", "history_back_fine", "history_return_fine",
            "future_main", "future_fine")
    lengths = (40, 40, 80, 80, 100, 200)
    complete = counts["integrations_completed"]
    attempted = counts["integrations_attempted"]
    if not 0 <= complete < len(legs) or attempted != complete + 1:
        raise ValueError("unexpected original interrupted integration counts")
    step = counts["rk4_steps_attempted"] - sum(lengths[:complete])
    if not 1 <= step <= lengths[complete] or counts["rk4_steps_completed"] + 1 != counts["rk4_steps_attempted"]:
        raise ValueError("unexpected original interrupted RK4 counts")
    return dict(leg=legs[complete], step=step)


def original_group(status, check):
    if status == "completed":
        return "success"
    if status == "numerically_unresolved":
        return "stage2"
    if check is None:
        return None
    if check["error"] and "in exp" in check["error"]:
        return "exp"
    values = check["values"]
    if any(values[name] is not None and values[name] > GATE_LIMITS[name] for name in GATE_NAMES[:3]):
        return "history"
    if values["future_main_fine_rms"] is not None and values["future_main_fine_rms"] > 1e-5:
        return "future_rms_peak" if values["future_main_fine_peak"] > 1e-4 else "future_rms"
    return None


def select_states(rows):
    selected = set()
    for mode in MODES:
        for group in GROUPS:
            indices = [r["index"] for r in rows if r["mode"] == mode and r["group"] == group]
            size = min(4, len(indices))
            selected.update((mode, indices[j * len(indices) // size], "selected") for j in range(size))
    for row in rows:
        key = (row["mode"], row["index"])
        if row["group"] in ("exp", "stage2"):
            selected.add((*key, "selected"))
        if row["group"] == "stage2":
            selected.add((*key, "stage1"))
    result = sorted(selected, key=lambda key: (MODES.index(key[0]), key[1], key[2]))
    if len(result) > 64:
        raise ValueError("frozen cohort exceeds the deterministic 64-state upper bound")
    return result


def recovered_state(model, packet, detail, role):
    if detail is None or detail.get("best") is None or detail["best"].get("state") is None:
        return None
    best = detail["best"]
    saved = np.asarray(best["state"], dtype=np.float64)
    model.functional.split_state(saved)
    rebuilt = gftd_fit297.endpoint(packet, model.functional, best["u"], model.fit_mode, role,
                                  model.inference_config)
    if saved.tobytes() != rebuilt.tobytes():
        raise ValueError("saved initializer state does not match the frozen variables and endpoint")
    return saved


def recover(release):
    raw, digest, summary = [json.loads(read_verified(p, release["asset_sha256"][p.as_posix()]))
                            for p in PARENT_RECORDS]
    hashes = {entry["path"]: entry["sha256"] for entry in raw["files"]}
    for path, expected in digest["input_sha256"].items():
        if path in hashes and hashes[path] != expected:
            raise ValueError(f"parent manifests disagree: {path}")

    def parent_bytes(path):
        return read_verified(path, hashes[path.as_posix()])

    frozen = json.loads(parent_bytes(PARENT / "resume01/all_initializations_frozen.json"))
    inputs = json.loads(parent_bytes(PARENT / "inputs.json"))
    ids, packets = inputs["receipt"]["legal_ids"], inputs["packets"]
    if ids != frozen["legal_ids"] or len(ids) != 297 or ids != sorted(set(ids)) or len(packets) != len(ids):
        raise ValueError("original fit297 identities or order changed")
    packets = [core.validate_observation({("p_end" if k == "p0" else k): v for k, v in p.items()}, 10)
               for p in packets]
    if any(p["sigma"] != 0 for p in packets):
        raise ValueError("original hard-endpoint contract changed")
    old_labels = [summary["L1"]["model_summaries"][mode]["event_facts"]["labels"] for mode in MODES]
    unstable = set(old_labels[0]["unstable"]["failure_ids"])
    if len(unstable) != 30 or unstable != set(old_labels[1]["unstable"]["failure_ids"]) or not unstable <= set(ids):
        raise ValueError("frozen 267/30 label identity changed")
    models, rows, states = {}, [], {}
    for mode in MODES:
        path = PARENT / f"{mode}_model.json"
        model_bytes = parent_bytes(path)
        model = GFTDKuramotoModel.load(ROOT / path)
        if sha(model_bytes) != frozen["models"][mode] or model_bytes != parent_bytes(path):
            raise ValueError(f"frozen model identity changed: {mode}")
        if model.fit_mode != mode or model.n_nodes != 10 or model.initialization_mode != "causal_two_stage":
            raise ValueError(f"unexpected frozen model format: {mode}")
        models[mode] = model
        for index, (event_id, packet) in enumerate(zip(ids, packets, strict=True)):
            prefix = PARENT if mode == "static" and index <= 40 else PARENT / "resume01"
            name = f"{mode}_{index:03d}"
            original = json.loads(parent_bytes(prefix / "initializations" / f"{name}.json"))
            diag = original["diagnostics"]
            status = diag["status"]
            completed = status == "completed"
            if original["event_id"] != event_id or completed != frozen["success"][mode][index]:
                raise ValueError(f"frozen event identity changed: {name}")
            stage1_check = gates(diag.get("stage1_numerical"))
            final_check = gates(diag.get("final_numerical"))
            selected_stage = diag.get("returned_point") or "stage1"
            if selected_stage not in ("stage1", "stage2"):
                raise ValueError(f"invalid chosen initialization stage: {name}")
            check = stage1_check if selected_stage == "stage1" else final_check
            group = original_group(status, stage1_check)
            state = recovered_state(model, packet, diag.get(selected_stage),
                                    "simple" if selected_stage == "stage1" else "smooth")
            if completed:
                npz_path = prefix / "initializations" / f"{name}.npz"
                asset = original["state"]
                if (PureWindowsPath(asset["path"]).as_posix() != npz_path.as_posix()
                        or asset["sha256"] != hashes[npz_path.as_posix()]):
                    raise ValueError(f"original output state identity changed: {name}")
                with np.load(io.BytesIO(parent_bytes(npz_path)), allow_pickle=False) as file:
                    saved = file["state"]
                if saved.dtype != np.float64 or saved.shape != (50,) or not np.isfinite(saved).all():
                    raise ValueError(f"invalid original output state: {name}")
                if state is not None and state.tobytes() != saved.tobytes():
                    raise ValueError(f"original output and diagnostic states disagree: {name}")
                state = saved
            elif original["state"] is not None:
                raise ValueError(f"failed original state unexpectedly published as output: {name}")
            if state is not None and state[20:30].tobytes() != packet["p_end"].tobytes():
                raise ValueError(f"original mechanical endpoint differs: {name}")
            states[(mode, index, "selected")] = state
            if group == "stage2":
                states[(mode, index, "stage1")] = recovered_state(model, packet, diag["stage1"], "simple")
            rows.append(dict(mode=mode, index=index, event_id=event_id,
                             label="unstable" if event_id in unstable else "stable", group=group,
                             status=status, selected_stage=selected_stage, old_gates=check,
                             old_stage1_gates=stage1_check, state_available=state is not None,
                             old_failure=failed_integration(diag["stage1_numerical"]["counts"]) if group == "exp" else None,
                             diagnostic=None, candidate=None))
    if tuple(sum(r["mode"] == mode and r["status"] == "completed" for r in rows) for mode in MODES) != (46, 1):
        raise ValueError("original successful counts changed")
    if sum(r["group"] == "exp" for r in rows) != 14 or sum(r["group"] == "stage2" for r in rows) != 5:
        raise ValueError("original overflow or stage2 counts changed")
    return models, rows, states


def metrics(delta):
    peak = float(np.max(np.abs(delta)))
    rms = peak * float(np.sqrt(np.mean((delta / peak)**2))) if peak else 0.
    return dict(rms=rms, peak=peak)


def edge_terms(functional, state):
    theta, _, _, amplitude, _ = functional.split_state(state)
    incidence = functional.incidence
    with np.errstate(over="raise", invalid="raise"):
        exponent = np.abs(incidence).T @ (functional.gamma_amp * amplitude)
        psi = bregman_kuramoto_potential(incidence.T @ theta, functional.eta_edge)
    if not np.isfinite(exponent).all() or not np.isfinite(psi).all():
        raise FloatingPointError("nonfinite edge terms")
    active = (functional.gates.s_edge * functional.k_edge * functional.gates.s_mod) > 0
    return exponent, psi, active


def edge_summary(functional, state):
    exponent, psi, active = edge_terms(functional, state)
    largest = {}
    for name, mask in (("active", active), ("inactive", ~active)):
        indices = np.flatnonzero(mask)
        if not len(indices):
            largest[name] = None
            continue
        minimum = indices[np.argmin(exponent[indices])]
        maximum = indices[np.argmax(exponent[indices])]
        largest[name] = dict(min=float(exponent[minimum]), max=float(exponent[maximum]),
                             min_edge=int(minimum), max_edge=int(maximum),
                             psi_at_max=float(psi[maximum]), psi_min=float(np.min(psi[indices])))
    return largest


class RhsLimit(RuntimeError):
    pass


class AuditedRhs:
    def __init__(self, functional, limit):
        self.functional = functional
        self.limit = limit
        self.calls = 0
        self.step = None
        self.substage = None
        self.operation = None
        self.time = 0.
        self.failure = None
        self.failure_state = None

    def split_state(self, state):
        return self.functional.split_state(state)

    def rhs(self, state):
        self.substage += 1
        time_offset = (0., .5, .5, 1.)[self.substage - 1]
        return self(self.time + self.h * time_offset, state)

    def __call__(self, t, state):
        if self.calls == self.limit:
            raise RhsLimit("registered RHS call limit reached")
        self.calls += 1
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = self.functional.rhs(state)
                self.split_state(result)
        except NUMERICAL_ERRORS as error:
            self.failure = dict(time=float(t), step=self.step, substage=self.substage, operation=self.operation,
                                error=repr(error), edges=None)
            self.failure_state = np.asarray(state).copy()
            try:
                self.failure["edges"] = edge_summary(self.functional, state)
            except (AttributeError, *NUMERICAL_ERRORS):
                pass
            raise
        return result


@dataclass
class Integration:
    path: np.ndarray
    last_state: np.ndarray
    failure_state: np.ndarray | None
    record: dict


def integrate(functional, state, duration, *, step=None, solver=None, rtol=None, atol=None,
              max_step=None, rhs_limit=None):
    grid = np.linspace(0., duration, round(abs(duration) / .01) + 1)
    path = np.full((len(grid), state.size), np.nan)
    path[0] = state
    count, attempted, t, last = 0, 0, 0., state.copy()
    adapter = AuditedRhs(functional, rhs_limit)
    begin, cpu_begin, error, instance = time.perf_counter(), time.process_time(), None, None
    try:
        if step is not None:
            adapter.h = np.copysign(step, duration)
            adapter.operation = "rk4"
            stride = round(.01 / step)
            for index in range(1, round(abs(duration) / step) + 1):
                adapter.time, adapter.step, adapter.substage = t, index, 0
                attempted += 1
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    next_state = core.rk4(adapter, last, adapter.h)
                last, t, count = next_state, index * adapter.h, index
                if index % stride == 0:
                    path[index // stride] = last
        else:
            adapter.operation = "solver_setup"
            instance = solver(adapter, 0., last, duration, rtol=rtol, atol=atol, max_step=max_step)
            direction, index = np.sign(duration), 1
            while instance.status == "running":
                attempted += 1
                adapter.step, adapter.operation = attempted, "solver_step"
                message = instance.step()
                if instance.status == "failed":
                    raise FloatingPointError(message)
                next_state = instance.y
                adapter.split_state(next_state)
                last, t, count = next_state.copy(), float(instance.t), count + 1
                if index < len(grid) and direction * (grid[index] - t) <= 0.:
                    adapter.operation = "dense_output"
                    dense = instance.dense_output()
                    while index < len(grid) and direction * (grid[index] - t) <= 0.:
                        value = dense(grid[index])
                        adapter.split_state(value)
                        path[index] = value
                        index += 1
    except (RhsLimit, *NUMERICAL_ERRORS) as exc:
        error = repr(exc)
        if adapter.failure is None:
            adapter.failure = dict(time=None, step=adapter.step, substage=None,
                                   operation=adapter.operation, error=error, edges=None)
    if error is None and adapter.calls == rhs_limit:
        error = "registered RHS call limit reached"
    completed = (error is None and abs(t - duration) < 1e-14
                 and path.dtype == np.float64 and np.isfinite(path).all()
                 and (step is not None or instance is not None and instance.status == "finished"))
    record = dict(completed=bool(completed), target_time=duration, last_accepted_time=t, step=step,
                  attempted_steps=attempted, accepted_steps=count, actual_rhs_calls=adapter.calls,
                  wall_seconds=time.perf_counter() - begin, cpu_seconds=time.process_time() - cpu_begin,
                  error=error if completed or error else "incomplete finite path", failure=adapter.failure)
    if solver is not None:
        record.update(solver=solver.__name__, rtol=rtol, atol=atol, max_step=max_step, rhs_limit=rhs_limit,
                      solver_nfev=None if instance is None else instance.nfev,
                      solver_njev=None if instance is None else instance.njev,
                      solver_nlu=None if instance is None else instance.nlu)
    return Integration(path, last, adapter.failure_state, record)


def path_delta(functional, left, right):
    if not (left.record["completed"] and right.record["completed"]):
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            projected = core.observations(left.path, functional.mass) - core.observations(right.path, functional.mass)
            delta = left.path - right.path
            return dict(observed=metrics(projected), state=metrics(delta / .1),
                        components={key: metrics(component) for key, component in
                                    zip(("theta", "omega", "p", "a", "r"), np.split(delta, 5, axis=1), strict=True)})
    except FloatingPointError:
        return None


def reference_agreement(functional, solutions, history):
    pairs = {f"{a}__{b}": path_delta(functional, solutions[a], solutions[b])
             for a, b in combinations(solutions, 2)}
    return dict(qualified=all(inside(delta, history) for delta in pairs.values()), pairs=pairs)


def inside(delta, history):
    if delta is None:
        return False
    observed = delta["observed"]
    return observed["rms"] <= (1e-8 if history else 1e-6) and (history or observed["peak"] <= 1e-5)


def convergence(functional, paths, references, history):
    adjacent = [path_delta(functional, a, b) for a, b in zip(paths, paths[1:])]
    observed = [None if item is None else item["observed"]["rms"] for item in adjacent]
    orders = []
    for coarse, fine in zip(observed, observed[1:]):
        if coarse is None or fine is None:
            orders.append(None)
        elif min(coarse, fine) <= 1e-12:
            orders.append("floor_limited")
        else:
            orders.append(float(np.log2(coarse) - np.log2(fine)))
    comparison = [path_delta(functional, path, references["radau_strict"]) for path in paths]
    order_evidence = all(isinstance(order, float) and 2.5 <= order <= 6 for order in orders)
    declining = all(a is not None and b is not None and a["observed"]["rms"] > b["observed"]["rms"]
                    for a, b in zip(comparison, comparison[1:]))
    agreement = reference_agreement(functional, references, history)
    evidence = agreement["qualified"] and (order_evidence and declining
                                           or all(inside(item, history) for item in comparison[-2:]))
    return dict(adjacent=adjacent, orders=orders, vs_radau_strict=comparison,
                reference=agreement, discretization_evidence=evidence)


def rk4_gates(functional, state, histories, returns, futures):
    roundtrips = [roundtrip_error(path, state) for path in returns[:2]]
    history = path_delta(functional, *histories[:2])
    future = path_delta(functional, *futures[:2])
    values = dict(zip(GATE_NAMES, (
        *roundtrips, None if history is None else history["observed"]["rms"],
        None if future is None else future["observed"]["rms"],
        None if future is None else future["observed"]["peak"], 0.,
    ), strict=True))
    passed = all(value is not None and value <= GATE_LIMITS[name] for name, value in values.items())
    return dict(passed=passed, values=values)


def roundtrip_error(path, state):
    if path is None or not path.record["completed"]:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.max(np.abs((path.last_state - state) / .1)))
    return value if np.isfinite(value) else None


def rk4_amplification(z):
    return np.abs(1 + z * (1 + z * (.5 + z * (1 / 6 + z / 24))))


def trajectory_sample(functional, state, t):
    adapter = AuditedRhs(functional, None)
    record = dict(time=t, error=None, rhs_calls=None, eigenvalue_difference=None, spectrum=None)
    arrays = {}
    try:
        rhs = adapter(t, state)
        gradients = functional.gradients(state)
        ra, rm = functional.dissipative_operators(state)
        _, _, gp, ga, gr = gradients
        omega = state[functional.n_nodes:2 * functional.n_nodes]
        chain = float(np.concatenate(gradients) @ rhs)
        dissipation = float((functional.damping * omega**2).sum() + (functional.rate_p * gp**2).sum()
                            + (ra * ga**2).sum() + gr @ rm @ gr)
        H = functional.hamiltonian(state)
        if not np.isfinite([chain, dissipation, H]).all():
            raise FloatingPointError("nonfinite power or energy")
        exponent, psi, active = edge_terms(functional, state)
        arrays.update(state=state, rhs=rhs, exponent=exponent, psi=psi, active=active, amplitude_gradient=ga)
        record.update(H=H, rhs_peak=float(np.max(np.abs(rhs))), amplitude_gradient=metrics(ga),
                      power=dict(chain=chain, expected=-dissipation, difference=abs(chain + dissipation)),
                      edges=edge_summary(functional, state))

        eigenvalues = []
        for factor in (1e-5, 5e-6):
            steps = factor * np.maximum(1., np.abs(state))
            perturbations = np.diag(steps)
            columns = [(adapter(t, state + shift) - adapter(t, state - shift)) / (2 * step)
                       for shift, step in zip(perturbations, steps, strict=True)]
            jacobian = np.column_stack(columns)
            if not np.isfinite(jacobian).all():
                raise FloatingPointError("nonfinite sampled Jacobian")
            eigenvalues.append(np.linalg.eigvals(jacobian))
        arrays.update(eigenvalues_coarse=eigenvalues[0], eigenvalues_fine=eigenvalues[1])
        difference = np.abs(eigenvalues[0][:, None] - eigenvalues[1])
        rows, cols = linear_sum_assignment(difference)
        record["eigenvalue_difference"] = float(np.max(difference[rows, cols]))
        poles = eigenvalues[-1]
        h = -.0025 if t < 0 else .01
        stable = poles.real < -1e-8
        stable_in_direction = (h * poles).real < -1e-8 * abs(h)
        amplified = rk4_amplification(h * poles) > 1.
        record["spectrum"] = dict(original_step=h, max_real=float(np.max(poles.real)),
                                  max_frequency_hz=float(np.max(np.abs(poles.imag)) / (2 * np.pi)),
                                  max_abs_real=float(np.max(np.abs(poles.real))),
                                  negative_real_rk4_outside=int(np.sum(stable & amplified)),
                                  decaying_in_integration_direction_rk4_outside=int(np.sum(stable_in_direction & amplified)))
        if t == 0:
            record["spectrum"]["history_from_endpoint_decaying_rk4_outside"] = int(np.sum(
                (poles.real > 1e-8) & (rk4_amplification(-.0025 * poles) > 1.)
            ))
    except (*NUMERICAL_ERRORS, np.linalg.LinAlgError) as exc:
        record.update(error=repr(exc), failure=adapter.failure)
        if adapter.failure_state is not None:
            arrays["failure_state"] = adapter.failure_state
    record["rhs_calls"] = adapter.calls
    return record, arrays


def summarize_path(functional, solution, arrays, name, origin_hash):
    arrays[f"{name}_path"] = solution.path
    arrays[f"{name}_last_state"] = solution.last_state
    if solution.failure_state is not None:
        arrays[f"{name}_failure_state"] = solution.failure_state
        try:
            exponent, psi, active = edge_terms(functional, solution.failure_state)
            arrays[f"{name}_failure_exponent"] = exponent
            arrays[f"{name}_failure_psi"] = psi
            arrays[f"{name}_failure_active_edges"] = active
        except NUMERICAL_ERRORS:
            pass
    saved = solution.path[np.isfinite(solution.path).all(axis=1)]
    components = {}
    for label, values, final in zip(("theta", "omega", "p", "a", "r"), np.split(saved, 5, axis=1),
                                    np.split(solution.last_state, 5), strict=True):
        components[label] = dict(min=float(np.min(values)), max=float(np.max(values)),
                                 last_accepted_max_abs=float(np.max(np.abs(final))))
    return dict(solution.record, start_state_sha256=origin_hash, component_ranges=components)


def diagnose(functional, state):
    start, cpu_start = time.perf_counter(), time.process_time()
    state_hash = sha(state.tobytes())
    arrays = {"initial_state": state, "future_times": np.linspace(0., 1., 101),
              "history_times": np.linspace(0., -.1, 11)}
    hist_steps = (.0025, .00125, .000625, .0003125)
    future_steps = (.01, .005, .0025, .00125)
    history, returns, future, receipts = [], [], [], {}
    for index, step in enumerate(hist_steps):
        back = integrate(functional, state, -.1, step=step)
        history.append(back)
        receipts[f"history_{index}"] = summarize_path(functional, back, arrays, f"history_{index}", state_hash)
        forward = integrate(functional, back.last_state, .1, step=step) if back.record["completed"] else None
        returns.append(forward)
        receipts[f"return_{index}"] = None if forward is None else summarize_path(
            functional, forward, arrays, f"return_{index}", sha(back.last_state.tobytes()))
    for index, step in enumerate(future_steps):
        solution = integrate(functional, state, 1., step=step)
        future.append(solution)
        receipts[f"future_{index}"] = summarize_path(functional, solution, arrays, f"future_{index}", state_hash)
    references = {"history": {}, "future": {}}
    for scope, duration, max_step in (("history", -.1, .00125), ("future", 1., .005)):
        for name, solver, rtol, atol in (("radau", Radau, 1e-9, 1e-11), ("radau_strict", Radau, 1e-11, 1e-13),
                                         ("dop853", DOP853, 1e-11, 1e-13)):
            solution = integrate(functional, state, duration, solver=solver, max_step=max_step,
                                 rtol=rtol, atol=atol, rhs_limit=20000)
            references[scope][name] = solution
            key = f"{scope}_{name}"
            receipts[key] = summarize_path(functional, solution, arrays, key, state_hash)
    samples = []
    for scope, index, t in (("history", 10, -.1), ("future", 0, 0.), ("future", 10, .1),
                            ("future", 50, .5), ("future", 100, 1.)):
        reference = references[scope]["radau_strict"].path[index]
        if t != 0 and not np.isfinite(reference).all():
            samples.append(dict(time=t, status="reference_unresolved", rhs_calls=0))
            continue
        record, values = trajectory_sample(functional, reference, t)
        arrays.update({f"sample_{len(samples)}_{name}": value for name, value in values.items()})
        samples.append(record)
    if sha(state.tobytes()) != state_hash:
        raise ValueError("frozen initial state changed during diagnostic")
    costs = [r for r in receipts.values() if r is not None]
    return dict(
        history=convergence(functional, history, references["history"], True),
        future=convergence(functional, future, references["future"], False),
        recomputed_rk4_gates=rk4_gates(functional, state, history, returns, future),
        roundtrip_scaled_max=[roundtrip_error(item, state) for item in returns],
        integrals=receipts, samples=samples, continuous_growth=None,
        costs=dict(integrations_attempted=len(costs), integrations_completed=sum(r["completed"] for r in costs),
                   rhs_integrations=sum(r["actual_rhs_calls"] for r in costs),
                   rhs_sampling_and_jacobians=sum(row["rhs_calls"] for row in samples),
                   wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu_start),
    ), arrays


def frozen_selection(rows, states):
    index = {(r["mode"], r["index"]): r for r in rows}
    selected = []
    for mode, event_index, role in select_states(rows):
        row = index[mode, event_index]
        state = states[mode, event_index, role]
        selected.append(dict(mode=mode, index=event_index, role=role, event_id=row["event_id"], group=row["group"],
                             original_state_gates=row["old_stage1_gates"] if role == "stage1" else row["old_gates"],
                             state_sha256=None if state is None else sha(state.tobytes())))
    return dict(selection=selected, rows=rows)


def run(release_path, phase):
    raw_release = Path(release_path).read_bytes()
    release = json.loads(raw_release)
    if phase == "prepare" and release.get("phase") == "prepare02":
        phase = "prepare02"
    verify_release(release, phase)
    claim(phase, release["action_id"], sha(raw_release))
    tick, cpu_tick = time.perf_counter(), time.process_time()
    log_path = OUTPUT / ("execution.json" if phase == "diagnostic" else f"{phase}_execution.json")
    receipt = dict(status="RUNNING", phase=phase, action_id=release["action_id"],
                   release_sha256=sha(raw_release), numpy=np.__version__, scipy=scipy.__version__,
                   selected_count=None, current=None, current_costs=None, completed=[])
    try:
        write_json(log_path, receipt)
        models, rows, states = recover(release)
        frozen = frozen_selection(rows, states)
        if phase in ("prepare", "prepare02"):
            receipt["selection"] = write_json(OUTPUT / "selection.json", frozen)
        else:
            actual = read_verified(OUTPUT / "selection.json", release["selection_sha256"])
            if actual != json_bytes(frozen):
                raise ValueError("recovered state bytes or selection changed after preflight")
            receipt["selected_count"] = len(frozen["selection"])
            indexed = {(r["mode"], r["index"]): r for r in rows}
            for item in frozen["selection"]:
                mode, index, role = item["mode"], item["index"], item["role"]
                name = f"{mode}_{index:03d}_{role}"
                receipt["current"] = name
                write_json(log_path, receipt)
                state = states[mode, index, role]
                if state is None:
                    details = dict(status="missing")
                else:
                    details, arrays = diagnose(models[mode].functional, state)
                    receipt["current_costs"] = details["costs"]
                    details["status"] = "measured"
                    details["arrays"] = write_arrays(PRIVATE / f"{name}.npz", arrays)
                evidence = write_json(OUTPUT / "states" / f"{name}.json", dict(item, **details))
                receipt["completed"].append(dict(state=name, evidence=evidence, costs=details.get("costs")))
                receipt["current_costs"] = None
                row = indexed[mode, index]
                if row["diagnostic"] is None:
                    row["diagnostic"] = {}
                row["diagnostic"][role] = evidence
                write_json(log_path, receipt)
            receipt["rows"] = write_json(OUTPUT / "rows.json", rows)
        receipt.update(status="COMPLETED", current=None)
    except BaseException as error:
        receipt.update(status="INTERRUPTED_RUNTIME", error=repr(error),
                       wall_seconds=time.perf_counter() - tick, cpu_seconds=time.process_time() - cpu_tick)
        try:
            write_json(log_path, receipt)
        except OSError as save_error:
            error.add_note(f"diagnostic receipt write failed: {save_error!r}")
        raise
    receipt.update(wall_seconds=time.perf_counter() - tick, cpu_seconds=time.process_time() - cpu_tick)
    write_json(log_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.release, "prepare" if arguments.prepare_only else "diagnostic")
