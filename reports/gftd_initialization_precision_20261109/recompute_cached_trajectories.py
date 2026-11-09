"""Recompute recorded trajectory differences without calling a model or solver."""

from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path("results/gftd_initialization_precision_20261109")
HORIZONS = (1, 2, 5, 8, 15)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_asset(reference):
    path = (ROOT / reference["path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"asset outside workspace: {path}")
    data = path.read_bytes()
    if len(data) != reference["bytes"] or sha(data) != reference["sha256"]:
        raise ValueError(f"frozen asset changed: {path}")
    return data


def compare(x, y, mass):
    def observe(values):
        theta = values[:, :10]
        centered = theta - (theta @ mass / mass.sum())[:, None]
        return np.column_stack((centered, values[:, 10:20])) / .1

    def metrics(values):
        peak = float(np.max(np.abs(values)))
        rms = peak * float(np.sqrt(np.mean((values / peak) ** 2))) if peak else 0.
        return {"rms": rms, "peak": peak}

    delta = x - y
    return {
        "observed": metrics(observe(x) - observe(y)),
        "state": metrics(delta / .1),
        "components": {
            name: metrics(block) for name, block in zip(
                ("theta", "omega", "p", "a", "r"), np.split(delta, 5, axis=1), strict=True
            )
        },
    }


def check(saved, recomputed):
    if recomputed is None:
        assert saved is None
    elif isinstance(recomputed, dict):
        assert set(saved) == set(recomputed)
        for key, value in recomputed.items():
            check(saved[key], value)
    else:
        assert math.isclose(saved, recomputed, rel_tol=3e-13, abs_tol=2e-15)


def prefix(arrays, side, leg, seconds):
    path = arrays[f"{side}_path"][:100 * seconds + 1]
    if len(path) == 100 * seconds + 1 and leg["last_accepted_time"] + 1e-14 >= seconds and np.isfinite(path).all():
        return path
    return None


def statistics(values):
    return dict(n=len(values), min=min(values), median=float(np.median(values)), max=max(values))


def recompute():
    execution_path = RESULTS / "candidate/full/execution.json"
    execution_bytes = (ROOT / execution_path).read_bytes()
    execution = json.loads(execution_bytes)
    assert execution["status"] == "COMPLETED" and execution["method"] == "fixed_dp8"
    rows = json.loads(read_asset(execution["rows"]))
    assert len(rows) == 594
    masses = {mode: np.asarray(json.loads(read_asset(ref))["functional"]["mass"])
              for mode, ref in execution["models"].items()}
    initializations = execution["reused_pilot"] | execution["tasks"]["initialize"]
    assert len(execution["reused_pilot"]) == 44 and len(initializations) == 594
    future = execution["tasks"]["future"]
    differences = defaultdict(list)
    references = defaultdict(list)
    added, controls, retained, checked = set(), set(), set(), set()
    failures, unresolved, anomalies = [], [], []

    for row in rows:
        mode, label = row["mode"], row["label"]
        name = f"{mode}_{row['index']:03d}"
        group = f"{mode}_{label}"
        init = json.loads(read_asset(initializations[name]))
        old = row["status"] == "completed"
        if old:
            controls.add(name)
        if not init["passed"]:
            assert name not in future
            continue
        if old:
            retained.add(name)
        else:
            added.add(name)
        result = json.loads(read_asset(future[name]))
        checked.add(name)
        legs = result["integrals"]
        with np.load(io.BytesIO(read_asset(result["arrays"])), allow_pickle=False) as arrays:
            assert sha(arrays["initial_state"].tobytes()) == init["state_sha256"]
            assert np.array_equal(arrays["times"], np.linspace(0., 15., 1501))
            incomplete = {}
            for side, leg in legs.items():
                path = arrays[f"{side}_path"]
                assert path[0].tobytes() == arrays["initial_state"].tobytes()
                assert path.dtype == np.float64 and path.shape == (101 if side == "radau" else 1501, 50)
                if side == "radau":
                    assert leg["solver"] == "Radau" and (leg["rtol"], leg["atol"], leg["max_step"]) == (1e-11, 1e-13, .005)
                    assert leg["actual_rhs_calls"] <= 20000
                else:
                    assert leg["step"] == (.01 if side == "main" else .005)
                    assert leg["actual_rhs_calls"] == 12 * leg["accepted_steps"] + leg["interrupted_step_rhs_calls"]
                    assert leg["completed"] == (leg["last_accepted_time"] == 15. and np.isfinite(path).all())
                    if not leg["completed"]:
                        last = leg["last_accepted_time"]
                        check(last, leg["accepted_steps"] * leg["step"])
                        for saved, expected in zip(leg["first_failure_interval"], (last, min(15., last + leg["step"])), strict=True):
                            check(saved, expected)
                        incomplete[side] = {key: leg[key] for key in (
                            "last_accepted_time", "first_failure_interval", "failure", "error"
                        )}
            if incomplete:
                failures.append(dict(task=name, event_id=row["event_id"], label=label, legs=incomplete))
            for seconds in HORIZONS:
                main = prefix(arrays, "main", legs["main"], seconds)
                fine = prefix(arrays, "fine", legs["fine"], seconds)
                computed = compare(main, fine, masses[mode]) if main is not None and fine is not None else None
                check(result["differences"][str(seconds)], computed)
                if computed is not None:
                    differences[group, seconds].append(computed)
            reference = result["single_radau_reference"]
            if old:
                assert reference is None and "radau" not in legs
                continue
            fine = prefix(arrays, "fine", legs["fine"], 1)
            assert reference["attempted"] == (fine is not None)
            if fine is None:
                anomalies.append(name)
                continue
            leg = legs["radau"]
            computed = compare(fine, arrays["radau_path"], masses[mode]) if leg["completed"] else None
            check(reference["difference"], computed)
            confirmed = computed is not None and computed["observed"]["rms"] <= 1e-6 and computed["observed"]["peak"] <= 1e-5
            assert confirmed == (reference["status"] == "confirmed")
            references[group].append(computed if confirmed else None)
            if not confirmed:
                unresolved.append(dict(task=name, event_id=row["event_id"], label=label, completed=leg["completed"],
                                       actual_rhs_calls=leg["actual_rhs_calls"], last_accepted_time=leg["last_accepted_time"],
                                       error=leg["error"], difference=computed))

    assert checked == set(future) and sorted(added) == execution["decision"]["added"]
    assert sorted(controls - retained) == execution["decision"]["regressions"]
    groups = {}
    for group in sorted({group for group, _ in differences}):
        summaries = {}
        for seconds in HORIZONS:
            entries = differences[group, seconds]
            summaries[str(seconds)] = {
                view: {metric: statistics([entry[view][metric] for entry in entries])
                       for metric in ("rms", "peak")}
                for view in ("observed", "state")
            } if entries else None
        ref = references[group]
        confirmed = [value["observed"] for value in ref if value is not None]
        groups[group] = dict(main_fine=summaries, radau_attempted=len(ref),
                             radau_confirmed=len(confirmed), radau_confirmed_observed={
                                 metric: statistics([value[metric] for value in confirmed]) for metric in ("rms", "peak")
                             } if confirmed else None)
    return dict(status="VERIFIED", execution_path=execution_path.as_posix(), execution_sha256=sha(execution_bytes),
                trajectory_tasks=len(checked), new_qualifications=len(added), old_controls=len(controls),
                old_controls_retained=len(retained), groups=groups,
                incomplete_15s=failures, radau_unresolved=unresolved, fine_before_1s_anomalies=anomalies)


if __name__ == "__main__":
    result = recompute()
    path = ROOT / RESULTS / "independent_trajectory_digest.json"
    path.write_bytes((json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))
