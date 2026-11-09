"""Run the complete fixed-DP8 initializer under separate pilot and full releases."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
import time
import uuid

import numpy as np
import scipy
from scipy.integrate import Radau

from kiki.models import GFTDKuramotoModel, gftd_fitting as fitting
from . import fixed_state, run as diagnostic


OUTPUT = diagnostic.OUTPUT / "candidate"
PRIVATE = diagnostic.PRIVATE.parent / "candidate"
SOURCE_PATHS = (*diagnostic.SOURCE_PATHS, *fixed_state.SOURCE_PATHS,
                Path(__file__).resolve().relative_to(diagnostic.ROOT).as_posix(),
                "tests/test_gftd_precision_candidate.py")
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
HORIZONS = (1, 2, 5, 8, 15)
METHOD = "fixed_dp8"
MODELS = {}


def thread_settings():
    settings = {key: os.environ.get(key) for key in THREADS}
    if set(settings.values()) != {"1"}:
        raise ValueError("candidate requires single-thread BLAS, OMP and MKL in each process")
    return settings


def event_key(row):
    return f"{row['mode']}_{row['index']:03d}"


def pilot_keys(selection):
    keys = {event_key(row) for row in selection["selection"]}
    if len(selection["selection"]) != 49 or len(keys) != 44:
        raise ValueError("expected 44 unique packets from the 49 frozen diagnostic states")
    return keys


def read_json(path, sha256):
    return json.loads(diagnostic.read_verified(path, sha256))


def read_asset(ref, path):
    if ref["path"] != path.as_posix():
        raise ValueError(f"unexpected evidence path: {ref['path']}")
    return diagnostic.read_verified(path, ref["sha256"])


def verify(release):
    phase, action = release.get("phase"), release.get("action_id", "")
    if (release.get("released") is not True or phase not in ("pilot", "full")
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", action)):
        raise PermissionError("distinct root pilot or full candidate release required")
    if type(release.get("workers")) is not int or not 1 <= release["workers"] <= 4:
        raise ValueError("one through four worker processes are allowed")
    thread_settings()
    sources, assets = release["source_sha256"], release["asset_sha256"]
    if set(sources) != set(SOURCE_PATHS) or set(assets) != {p.as_posix() for p in diagnostic.PARENT_RECORDS}:
        raise ValueError("candidate source or parent assets are incomplete")
    for path, digest in (sources | assets).items():
        diagnostic.read_verified(path, digest)
    diagnostic.read_verified(diagnostic.PROTOCOL, release["protocol_sha256"])
    adoption = read_json(diagnostic.OUTPUT / "candidate_adoption.json", release["candidate_adoption_sha256"])
    integrity = read_json(diagnostic.OUTPUT / "fixed_state_integrity.json", adoption["fixed_state_integrity_sha256"])
    pilot_ref = read_json(fixed_state.OUTPUT / "execution.json", integrity["fixed_state_execution_sha256"])
    if (adoption["selected_branch"] != "A: fixed_dp8" or pilot_ref["status"] != "COMPLETED"
            or not pilot_ref["decision"]["eligible_for_initializer_pilot"]
            or pilot_ref["selection_sha256"] != release["selection_sha256"]
            or adoption["source_precision_kernel_sha256"] != sources["src/kiki/numerics.py"]):
        raise ValueError("fixed-state decision does not authorize this candidate")
    selection = diagnostic.read_verified(diagnostic.OUTPUT / "selection.json", release["selection_sha256"])
    keys = pilot_keys(json.loads(selection))
    fixed_release = read_json(diagnostic.OUTPUT / "fixed_state_release.json", pilot_ref["release_sha256"])
    source_diagnostic = read_json(diagnostic.OUTPUT / "execution.json", fixed_release["diagnostic_execution_sha256"])
    bindings = {key: release[key] for key in ("protocol_sha256", "candidate_adoption_sha256", "selection_sha256",
                                             "source_sha256", "asset_sha256")}
    old_pilot = None
    if phase == "full":
        old_pilot = read_json(OUTPUT / "pilot/execution.json", release["pilot_execution_sha256"])
        if (old_pilot["phase"] != "pilot" or old_pilot["status"] != "COMPLETED"
                or old_pilot["action_id"] == action or old_pilot["bindings"] != bindings
                or old_pilot["runtime"] != runtime() or set(old_pilot["tasks"]["initialize"]) != keys
                or not old_pilot["decision"]["eligible_for_full"]):
            raise ValueError("full validation requires the complete eligible pilot under identical bindings")
    return selection, keys, bindings, old_pilot, source_diagnostic["rows"]


def runtime():
    return dict(numpy=np.__version__, scipy=scipy.__version__, threads=thread_settings())


def claim(path, payload):
    path = diagnostic.ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(diagnostic.json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint(phase, models, pilot):
    refs = {}
    for mode, model in models.items():
        config = fitting.make_fixed_dp8_config(model.functional, model.inference_config)
        model.inference_config, model.last_initialization = config, None
        path = PRIVATE / "pilot" / f"{mode}_model.json"
        config_sha = diagnostic.sha(diagnostic.json_bytes(config))
        if phase == "pilot":
            pending = diagnostic.ROOT / path.with_suffix(f".save-{uuid.uuid4().hex}.json")
            model.save(pending)
            ref = diagnostic.write_bytes(path, pending.read_bytes())
            try:
                pending.unlink()
            except PermissionError:
                ref["retained_checkpoint_copy"] = pending.relative_to(diagnostic.ROOT).as_posix()
        else:
            ref = pilot["models"][mode]
            raw = read_asset(ref, path)
            loaded = json.loads(raw)
            if (ref["config_sha256"] != config_sha or loaded["inference_config"] != config
                    or fitting._hash(fitting._physics(GFTDKuramotoModel.load(diagnostic.ROOT / path).functional))
                    != fitting._hash(fitting._physics(model.functional))):
                raise ValueError(f"pilot candidate model or configuration changed: {mode}")
        refs[mode] = dict(ref, config_sha256=config_sha)
    return refs


def load_packets(release):
    manifest = diagnostic.PARENT_RECORDS[0]
    raw = read_json(manifest, release["asset_sha256"][manifest.as_posix()])
    hashes = {item["path"]: item["sha256"] for item in raw["files"]}
    path = diagnostic.PARENT / "inputs.json"
    inputs = read_json(path, hashes[path.as_posix()])
    packets = {}
    for event_id, original in zip(inputs["receipt"]["legal_ids"], inputs["packets"], strict=True):
        packet = diagnostic.core.validate_observation(
            {("p_end" if key == "p0" else key): value for key, value in original.items()}, 10)
        if packet.pop("sigma") != 0.:
            raise ValueError("original endpoint must be hard anchored")
        packet["p0"] = packet.pop("p_end")
        packets[event_id] = packet
    return packets, hashes


def history_fit(diag, stage):
    task = diag.get(stage)
    best = None if task is None else task.get("best")
    return None if best is None else best["history_mse"]


def old_history(row, hashes):
    prefix = diagnostic.PARENT if row["mode"] == "static" and row["index"] <= 40 else diagnostic.PARENT / "resume01"
    path = prefix / "initializations" / f"{event_key(row)}.json"
    saved = read_json(path, hashes[path.as_posix()])
    if saved["event_id"] != row["event_id"]:
        raise ValueError("original history identity changed")
    return history_fit(saved["diagnostics"], row["selected_stage"])


def setup_worker(models):
    thread_settings()
    for mode, ref in models.items():
        path = PRIVATE / "pilot" / f"{mode}_model.json"
        read_asset(ref, path)
        model = GFTDKuramotoModel.load(diagnostic.ROOT / path)
        config_sha = diagnostic.sha(diagnostic.json_bytes(model.inference_config))
        if model.fit_mode != mode or config_sha != ref["config_sha256"] or model.inference_config["method"] != METHOD:
            raise ValueError(f"worker candidate config changed: {mode}")
        MODELS[mode] = model


def checked_state(state, p0):
    if (state.dtype != np.float64 or state.shape != (50,) or not np.isfinite(state).all()
            or state[20:30].tobytes() != p0.tobytes()):
        raise ValueError("candidate output violates the 50D finite state or true p0 anchor")


def initialize(model, packet, row, old_mse):
    state = None
    try:
        state = model.initialize(packet)
        detail = model.last_initialization
    except fitting.InitializationFailure as error:
        detail = error.diagnostics
    if detail.get("method") != METHOD:
        raise ValueError("complete initializer did not use the frozen fixed-DP8 method")
    counts = detail["counts"]
    stages = [detail.get(f"stage{number}") for number in (1, 2)]
    calls = [0 if task is None else task["actual_residual_calls"] for task in stages]
    if (not 0 <= calls[0] <= 600 or not 0 <= calls[1] <= 1200
            or sum(calls) != counts["actual_residual_calls"]
            or not 12 * counts["fixed_dp8_steps_completed"] <= counts["actual_rhs_calls"]
            <= 12 * counts["fixed_dp8_steps_attempted"]):
        raise ValueError("initializer work accounting or registered budget changed")
    final = diagnostic.gates(detail.get("final_numerical"))
    if state is not None:
        checked_state(state, packet["p0"])
        if final is None or final["passed"] is not True or detail["status"] != "completed":
            raise ValueError("returned candidate state failed original numerical gates")
    elif detail["status"] == "completed":
        raise ValueError("candidate reported completion without a state")
    stage = detail.get("returned_point") or "stage1"
    candidate_mse = history_fit(detail, stage)
    selected = detail.get(stage)
    bounds = None
    if selected is not None and selected.get("best") is not None:
        u = selected["best"]["u"]
        active = selected["active_bounds"]
        offset = len(u) - 10
        bounds = dict(a=sum(i < offset for i in active), r=sum(i >= offset for i in active),
                      a_fixed=offset == 0)
    public = dict(mode=row["mode"], index=row["index"], event_id=row["event_id"], label=row["label"],
                  old_status=row["status"], old_gates=row["old_gates"], status=detail["status"],
                  passed=state is not None, p0_exact=None if state is None else True, stage=stage,
                  stage1_gates=diagnostic.gates(detail.get("stage1_numerical")), final_gates=final,
                  history_mse=dict(old=old_mse, candidate=candidate_mse,
                                   change=None if old_mse is None or candidate_mse is None else candidate_mse - old_mse),
                  active_bounds=bounds, counts=counts)
    return public, state, detail


def state_from_result(result, p0, phase):
    path = PRIVATE / phase / "states" / f"{event_key(result)}.npz"
    data = read_asset(result["state"], path)
    with np.load(io.BytesIO(data), allow_pickle=False) as file:
        if set(file.files) != {"state"}:
            raise ValueError("unexpected persisted state keys")
        state = file["state"]
    checked_state(state, p0)
    if diagnostic.sha(state.tobytes()) != result["state_sha256"]:
        raise ValueError("candidate state bytes changed")
    return state


def prefix(solution, duration):
    limit = 100 * duration + 1
    data = solution.path[:limit]
    if len(data) != limit or not np.isfinite(data).all() or solution.record["last_accepted_time"] + 1e-14 < duration:
        return None
    return diagnostic.Integration(data, data[-1], None, {"completed": True})


def continuation(functional, state, new_pass):
    main = fixed_state.integrate(functional, state, 15., .01)
    fine = fixed_state.integrate(functional, state, 15., .005)
    arrays = {"initial_state": state, "times": np.linspace(0., 15., 1501)}
    origin = diagnostic.sha(state.tobytes())
    integrals = {name: diagnostic.summarize_path(functional, item, arrays, name, origin)
                 for name, item in (("main", main), ("fine", fine))}
    for leg in integrals.values():
        leg["first_failure_interval"] = None if leg["completed"] else [
            leg["last_accepted_time"], min(15., leg["last_accepted_time"] + leg["step"])]
    differences = {}
    for horizon in HORIZONS:
        a, b = prefix(main, horizon), prefix(fine, horizon)
        differences[str(horizon)] = None if a is None or b is None else diagnostic.path_delta(functional, a, b)
    reference = None
    if new_pass:
        reference = dict(status="reference_unresolved", attempted=False, reason="fine DP8 incomplete before 1 s")
        fine_one = prefix(fine, 1)
        if fine_one is not None:
            radau = diagnostic.integrate(functional, state, 1., solver=Radau, rtol=1e-11, atol=1e-13,
                                         max_step=.005, rhs_limit=20000)
            integrals["radau"] = diagnostic.summarize_path(functional, radau, arrays, "radau", origin)
            difference = diagnostic.path_delta(functional, fine_one, radau)
            confirmed = diagnostic.inside(difference, history=False)
            reference = dict(attempted=True, status="confirmed" if confirmed else "reference_unresolved",
                             difference=difference)
    return dict(differences=differences, integrals=integrals, single_radau_reference=reference,
                counts=dict(actual_rhs_calls=sum(r["actual_rhs_calls"] for r in integrals.values()),
                            integrations_attempted=len(integrals),
                            integrations_completed=sum(r["completed"] for r in integrals.values()))), arrays


def worker(phase, action, kind, row, packet, context):
    name = event_key(row)
    started, cpu = time.perf_counter(), time.process_time()
    claim(OUTPUT / phase / "claims" / kind / f"{name}.json",
          dict(action_id=action, event_id=row["event_id"], mode=row["mode"], index=row["index"], pid=os.getpid()))
    result = {}
    try:
        model = MODELS[row["mode"]]
        if kind == "initialize":
            result, state, details = initialize(model, packet, row, context)
            data = gzip.compress(diagnostic.json_bytes(details), mtime=0)
            result["diagnostics"] = diagnostic.write_bytes(PRIVATE / phase / "diagnostics" / f"{name}.json.gz", data)
            result["state"] = None
            if state is not None:
                result["state"] = diagnostic.write_arrays(PRIVATE / phase / "states" / f"{name}.npz", {"state": state})
                result["state_sha256"] = diagnostic.sha(state.tobytes())
        else:
            original_result, original_phase = context
            state = state_from_result(original_result, packet["p0"], original_phase)
            result, arrays = continuation(model.functional, state, original_result["old_status"] != "completed")
            result["arrays"] = diagnostic.write_arrays(PRIVATE / phase / "trajectories" / f"{name}.npz", arrays)
        result.update(phase=phase, action_id=action, event_id=row["event_id"], index=row["index"], mode=row["mode"],
                      worker_wall_seconds=time.perf_counter() - started, worker_cpu_seconds=time.process_time() - cpu)
        ref = diagnostic.write_json(OUTPUT / phase / kind / f"{name}.json", result)
    except BaseException as error:
        failure = dict(action_id=action, error=repr(error), counts=result.get("counts"),
                       worker_wall_seconds=time.perf_counter() - started, worker_cpu_seconds=time.process_time() - cpu)
        try:
            diagnostic.write_json(OUTPUT / phase / "interrupted" / kind / f"{name}.json", failure)
        except OSError as save_error:
            error.add_note(f"worker failure receipt write failed: {save_error!r}")
        raise
    return name, ref, result


def execute(pool, jobs, workers):
    iterator = iter(jobs)
    pending = {pool.submit(worker, *job) for _, job in zip(range(workers), iterator)}
    while pending:
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in finished:
            yield future.result()
        pending.update(pool.submit(worker, *job) for _, job in zip(finished, iterator))


def reuse_pilot(pilot, rows, packets):
    refs, results = {}, {}
    for name, ref in pilot["tasks"]["initialize"].items():
        result = json.loads(read_asset(ref, OUTPUT / "pilot/initialize" / f"{name}.json"))
        row = rows[name]
        if (result["phase"] != "pilot" or result["action_id"] != pilot["action_id"]
                or event_key(result) != name or result["event_id"] != row["event_id"]
                or result["old_status"] != row["status"]):
            raise ValueError("pilot result identity changed")
        read_asset(result["diagnostics"], PRIVATE / "pilot/diagnostics" / f"{name}.json.gz")
        if result["passed"]:
            state_from_result(result, packets[row["event_id"]]["p0"], "pilot")
        elif result["state"] is not None:
            raise ValueError("failed pilot result has a state artifact")
        refs[name], results[name] = ref, result
    return refs, results


def summarize(rows, results, futures, phase):
    groups = {}
    for mode in diagnostic.MODES:
        groups[mode] = {}
        for label in ("stable", "unstable"):
            keys = [key for key in results if rows[key]["mode"] == mode and rows[key]["label"] == label]
            groups[mode][label] = dict(
                attempted=len(keys), old_qualified=sum(results[key]["old_status"] == "completed" for key in keys),
                candidate_qualified=sum(results[key]["passed"] for key in keys),
                fifteen_second_pairs=sum(all(futures[key]["integrals"][side]["completed"] for side in ("main", "fine"))
                                          for key in keys if key in futures) if phase == "full" else None,
                new_confirmed_by_radau=sum(
                    futures[key]["single_radau_reference"]["status"] == "confirmed"
                    for key in keys if key in futures and futures[key]["single_radau_reference"] is not None
                ) if phase == "full" else None,
            )
    added = sorted(key for key, row in results.items() if row["old_status"] != "completed" and row["passed"])
    controls = [key for key, row in results.items() if row["old_status"] == "completed"]
    regressions = sorted(key for key in controls if not results[key]["passed"])
    return dict(groups=groups, added=added, controls=len(controls), regressions=regressions,
                eligible_for_full=bool(added and len(controls) == 5 and not regressions) if phase == "pilot" else None)


def state_table(rows, diagnostic_ref, results, futures, receipt):
    old_rows = json.loads(read_asset(diagnostic_ref, diagnostic.OUTPUT / "rows.json"))
    if [event_key(row) for row in old_rows] != list(rows):
        raise ValueError("original diagnostic summary order changed")
    table = []
    for row in old_rows:
        name = event_key(row)
        summary = None
        if name in results:
            summary = {key: results[name][key] for key in (
                "status", "passed", "p0_exact", "stage", "stage1_gates", "final_gates",
                "history_mse", "active_bounds", "counts", "worker_cpu_seconds", "worker_wall_seconds")}
            summary["evidence"] = receipt["tasks"]["initialize"].get(name, receipt["reused_pilot"].get(name))
        future = None
        if name in futures:
            future = {key: futures[name][key] for key in ("differences", "single_radau_reference", "counts")}
            future["failures"] = {side: {key: futures[name]["integrals"][side][key]
                                         for key in ("completed", "failure", "first_failure_interval")}
                                  for side in ("main", "fine")}
            future["evidence"] = receipt["tasks"]["future"][name]
        table.append(dict(row, candidate=summary, continuation=future))
    return table


def run(release_path):
    raw = Path(release_path).read_bytes()
    release = json.loads(raw)
    selection, keys, bindings, pilot, diagnostic_ref = verify(release)
    phase, action, workers = (release[key] for key in ("phase", "action_id", "workers"))
    sha = diagnostic.sha(raw)
    receipt = dict(status="RUNNING", phase=phase, action_id=action, method=METHOD, release_sha256=sha,
                   bindings=bindings, runtime=runtime(), workers=workers, tasks={"initialize": {}, "future": {}},
                   cost_scope="sum of new worker tasks; excludes worker startup and reused pilot work",
                   reused_pilot={}, new_costs=dict(worker_cpu_seconds=0., worker_wall_seconds=0.,
                                                   actual_rhs_calls=0, actual_residual_calls=0,
                                                   integrations_attempted=0, integrations_completed=0))
    claim(OUTPUT / phase / "started.json", dict(action_id=action, release_sha256=sha))
    started, cpu = time.perf_counter(), time.process_time()
    path = OUTPUT / phase / "execution.json"
    persist = lambda: diagnostic.write_json(path, receipt)
    try:
        persist()
        models, recovered_rows, states = diagnostic.recover(release)
        if diagnostic.json_bytes(diagnostic.frozen_selection(recovered_rows, states)) != selection:
            raise ValueError("candidate inputs or diagnostic selection changed")
        rows = {event_key(row): row for row in recovered_rows}
        packets, hashes = load_packets(release)
        receipt["models"] = checkpoint(phase, models, pilot)
        del models, states
        results, futures = {}, {}
        if pilot is not None:
            receipt["reused_pilot"], results = reuse_pilot(pilot, rows, packets)
        todo = [row for name, row in rows.items() if (name in keys if phase == "pilot" else name not in results)]
        if len(todo) != (44 if phase == "pilot" else 550):
            raise ValueError("candidate packet count changed")
        receipt["planned_initializations"] = [event_key(row) for row in todo]
        persist()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=setup_worker, initargs=(receipt["models"],))
        try:
            for kind, planned, output in (("initialize", todo, results), ("future", None, futures)):
                if kind == "future":
                    if phase == "pilot":
                        break
                    planned = [row for name, row in rows.items() if results[name]["passed"]]
                    receipt["planned_futures"] = [event_key(row) for row in planned]
                    persist()
                jobs = ((phase, action, kind, row, packets[row["event_id"]],
                         old_history(row, hashes) if kind == "initialize" else
                         (results[event_key(row)], "pilot" if event_key(row) in receipt["reused_pilot"] else "full"))
                        for row in planned)
                for name, ref, result in execute(pool, jobs, workers):
                    receipt["tasks"][kind][name], output[name] = ref, result
                    for metric in receipt["new_costs"]:
                        receipt["new_costs"][metric] += result.get(metric, result["counts"].get(metric, 0))
                    persist()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        receipt["decision"] = summarize(rows, results, futures, phase)
        table = state_table(rows, diagnostic_ref, results, futures, receipt)
        receipt["rows"] = diagnostic.write_json(OUTPUT / phase / "rows.json", table)
        receipt["status"] = "COMPLETED"
    except BaseException as error:
        receipt.update(status="INTERRUPTED_RUNTIME", error=repr(error),
                       host_wall_seconds=time.perf_counter() - started, host_cpu_seconds=time.process_time() - cpu)
        try:
            persist()
        except OSError as save_error:
            error.add_note(f"candidate receipt write failed: {save_error!r}")
        raise
    receipt.update(host_wall_seconds=time.perf_counter() - started, host_cpu_seconds=time.process_time() - cpu)
    persist()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    run(parser.parse_args().release)
