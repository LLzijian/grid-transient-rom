"""Apply the fixed DP8 formula to the frozen diagnostic states."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import scipy
from scipy.integrate import DOP853

from kiki.numerics import dop853_step
from . import run as diagnostic


OUTPUT = diagnostic.OUTPUT / "fixed_state"
PRIVATE = diagnostic.PRIVATE.parent / "fixed_state"
SOURCE_PATHS = (
    Path(__file__).resolve().relative_to(diagnostic.ROOT).as_posix(), "src/kiki/numerics.py",
    "tests/test_gftd_precision_fixed_state.py",
)
COEFFICIENTS_SHA256 = "123669c4e6be7b9e45302adb0f7714a9972c8f1e196728914b43b2bd9c2f6361"


def state_name(item):
    return f"{item['mode']}_{item['index']:03d}_{item['role']}"


def verify(release):
    action = release.get("action_id", "")
    if (release.get("released") is not True or release.get("phase") != "fixed_state"
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", action)):
        raise PermissionError("distinct root fixed-state release required")
    read = diagnostic.read_verified
    parent = json.loads(read(diagnostic.OUTPUT / "diagnostic_release.json", release["diagnostic_release_sha256"]))
    diagnostic.verify_release(parent, "diagnostic")
    execution = json.loads(read(diagnostic.OUTPUT / "execution.json", release["diagnostic_execution_sha256"]))
    selection = read(diagnostic.OUTPUT / "selection.json", release["selection_sha256"])
    read(diagnostic.OUTPUT / "design_adoption.json", release["design_adoption_sha256"])
    items = {state_name(item): item for item in json.loads(selection)["selection"]}
    rows = {row["state"]: row["evidence"] for row in execution["completed"]}
    if (execution["status"] != "COMPLETED" or execution["phase"] != "diagnostic"
            or execution["release_sha256"] != release["diagnostic_release_sha256"]
            or execution["action_id"] != parent["action_id"] or action == parent["action_id"]
            or parent["selection_sha256"] != release["selection_sha256"]
            or execution["selected_count"] != 49 or len(execution["completed"]) != 49 or len(items) != 49
            or items.keys() != rows.keys()):
        raise ValueError("fixed-state pilot requires the complete frozen diagnostic cohort")
    name, leg = release["trigger"]["state"], release["trigger"]["leg"]
    trigger = items[name]
    evidence = rows[name]
    expected = diagnostic.OUTPUT / "states" / f"{name}.json"
    detail = json.loads(read(expected, evidence["sha256"]))
    if (leg not in ("history", "future") or trigger["role"] != "selected" or trigger["state_sha256"] is None
            or trigger["original_state_gates"]["passed"] is not False or evidence["path"] != expected.as_posix()
            or any(detail.get(key) != value for key, value in trigger.items())
            or detail[leg]["discretization_evidence"] is not True):
        raise ValueError("fixed-state pilot requires an old failure with discretization evidence")
    sources = release["source_sha256"]
    if set(sources) != set(SOURCE_PATHS):
        raise ValueError("fixed-state source bindings changed")
    for path, expected_sha in sources.items():
        read(path, expected_sha)
    coefficients = b"".join(np.asarray(value, dtype="<f8").tobytes() for value in (DOP853.A, DOP853.B))
    if diagnostic.sha(coefficients) != COEFFICIENTS_SHA256:
        raise ValueError("frozen DOP853 coefficients changed")
    return parent, selection


def integrate(functional, state, duration, step):
    step = float(np.copysign(step, duration))
    path = np.full((round(abs(duration) / .01) + 1, state.size), np.nan)
    path[0] = last = state.copy()
    adapter = diagnostic.AuditedRhs(functional, None)
    adapter.operation = "fixed_dp8"
    stride, steps = round(.01 / abs(step)), round(duration / step)
    completed, attempted, error = 0, 0, None
    started, cpu_started = time.perf_counter(), time.process_time()

    def rhs(value):
        adapter.substage += 1
        return adapter(adapter.time + step * DOP853.C[adapter.substage - 1], value)

    for attempted in range(1, steps + 1):
        adapter.time, adapter.step, adapter.substage = completed * step, attempted, 0
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                last = dop853_step(rhs, last, step)
        except diagnostic.NUMERICAL_ERRORS as exc:
            error = repr(exc)
            break
        if adapter.substage != 12:
            raise RuntimeError("fixed DP8 step must use twelve RHS calls")
        completed = attempted
        if attempted % stride == 0:
            path[attempted // stride] = last
    record = dict(completed=completed == steps and abs(completed * step - duration) < 1e-14
                            and np.isfinite(path).all().item(), step=step,
                  target_time=duration, last_accepted_time=completed * step, attempted_steps=attempted,
                  accepted_steps=completed, actual_rhs_calls=adapter.calls,
                  interrupted_step_rhs_calls=adapter.calls - 12 * completed,
                  error=error, failure=adapter.failure or (
                      None if error is None else dict(step=attempted, substage=None, operation="step_assembly", error=error)),
                  wall_seconds=time.perf_counter() - started,
                  cpu_seconds=time.process_time() - cpu_started)
    return diagnostic.Integration(path, last, adapter.failure_state, record)


def qualify(functional, state):
    arrays = dict(initial_state=state, history_times=np.linspace(0., -.1, 11),
                  return_times=np.linspace(-.1, 0., 11), future_times=np.linspace(0., 1., 101))
    history = [integrate(functional, state, -.1, step) for step in (.0025, .00125)]
    returns = [integrate(functional, leg.last_state, .1, abs(leg.record["step"])) if leg.record["completed"] else None
               for leg in history]
    future = [integrate(functional, state, 1., step) for step in (.01, .005)]
    legs = dict(zip(("history_main", "history_fine", "return_main", "return_fine", "future_main", "future_fine"),
                    [*history, *returns, *future], strict=True))
    receipts = {name: None if leg is None else diagnostic.summarize_path(
        functional, leg, arrays, name, diagnostic.sha(leg.path[0].tobytes())) for name, leg in legs.items()}
    counts = [record for record in receipts.values() if record is not None]
    return dict(gates=diagnostic.rk4_gates(functional, state, history, returns, future), integrals=receipts,
                costs=dict(integrations_attempted=len(counts), integrations_completed=sum(c["completed"] for c in counts),
                           **{key: sum(c[key] for c in counts)
                              for key in ("actual_rhs_calls", "wall_seconds", "cpu_seconds")})), arrays


def decision(rows):
    old = [r for r in rows if r["role"] == "selected" and r["old_passed"] is True]
    stage1 = [r for r in rows if r["role"] == "stage1" and r["old_passed"] is True]
    added = [r["state"] for r in rows if r["role"] == "selected" and r["old_passed"] is False and r["dp8_passed"] is True]
    regressions = [r["state"] for r in old if r["dp8_passed"] is not True]
    return dict(newly_qualified=added, originally_qualified_controls=len(old), regressed_controls=regressions,
                stage1_controls=len(stage1), regressed_stage1_controls=[r["state"] for r in stage1 if r["dp8_passed"] is not True],
                eligible_for_initializer_pilot=bool(added and old and not regressions))


def run(release_path):
    raw = Path(release_path).read_bytes()
    release = json.loads(raw)
    parent, selection = verify(release)
    action = release["action_id"]
    receipt = dict(action_id=action, status="RUNNING", phase="fixed_state", method="fixed_dp8", current=None,
                   current_costs=None, release_sha256=diagnostic.sha(raw), completed=[],
                   selection_sha256=release["selection_sha256"], coefficients_sha256=COEFFICIENTS_SHA256,
                   numpy=np.__version__, scipy=scipy.__version__)
    claim = diagnostic.ROOT / OUTPUT / "started.json"
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open("xb") as stream:
        stream.write(diagnostic.json_bytes(dict(action_id=action, release_sha256=receipt["release_sha256"])))
        stream.flush()
        os.fsync(stream.fileno())
    started = time.perf_counter()
    result_path = OUTPUT / "execution.json"
    try:
        diagnostic.write_json(result_path, receipt)
        models, rows, states = diagnostic.recover(parent)
        if diagnostic.json_bytes(diagnostic.frozen_selection(rows, states)) != selection:
            raise ValueError("fixed-state cohort differs from the frozen diagnostic selection")
        for item in json.loads(selection)["selection"]:
            name = receipt["current"] = state_name(item)
            diagnostic.write_json(result_path, receipt)
            state = states[item["mode"], item["index"], item["role"]]
            old = item["original_state_gates"]
            summary = dict(item, status="missing", gates=None, costs=None)
            if state is not None:
                summary, arrays = qualify(models[item["mode"]].functional, state)
                if diagnostic.sha(state.tobytes()) != item["state_sha256"]:
                    raise ValueError("fixed initial state changed during the pilot")
                receipt["current_costs"] = summary["costs"]
                summary.update(item, status="measured", arrays=diagnostic.write_arrays(PRIVATE / f"{name}.npz", arrays))
            evidence = diagnostic.write_json(OUTPUT / "states" / f"{name}.json", summary)
            receipt["completed"].append(dict(state=name, role=item["role"], mode=item["mode"], evidence=evidence,
                                             old_passed=None if old is None else old["passed"],
                                             dp8_passed=None if summary["gates"] is None else summary["gates"]["passed"],
                                             costs=summary["costs"]))
            receipt.update(current=None, current_costs=None)
            diagnostic.write_json(result_path, receipt)
        receipt.update(status="COMPLETED", decision=decision(receipt["completed"]))
    except BaseException as error:
        receipt.update(status="INTERRUPTED_RUNTIME", error=repr(error), wall_seconds=time.perf_counter() - started)
        try:
            diagnostic.write_json(result_path, receipt)
        except OSError as save_error:
            error.add_note(f"fixed-state receipt write failed: {save_error!r}")
        raise
    receipt["wall_seconds"] = time.perf_counter() - started
    diagnostic.write_json(result_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    run(parser.parse_args().release)
