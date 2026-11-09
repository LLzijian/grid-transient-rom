"""Meter registered fixed points and mock tests, sharing the limits across corrections."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from scipy.integrate import DOP853, Radau, solve_ivp
from scipy.optimize import least_squares

from experiments.gftd_matched_prediction_20261109 import run
from kiki.evaluation import gftd_analysis
from kiki.models import gftd_discovery, gftd_fit297, gftd_fitting
from kiki.numerics import dop853_step, rk4_step
from kiki.physics import kuramoto_functional as physics

LIMITS = {"rhs": 128, "hamiltonian": 2000}


def qualify(path):
    release, release_sha = run.release_record(path, "qualification")
    prepared = run.bound_preparation(release)
    attempt = release["qualification_attempt"]
    out = run.attempt_path("qualification", attempt)
    if (ROOT / run.RESULTS / "fit/started.json").exists() or (ROOT / out).exists():
        raise FileExistsError("qualification cannot follow an existing outcome or scientific execution")
    used = Counter()
    previous_refs = []
    for prior in range(attempt):
        previous_path = run.attempt_path("qualification", prior)
        previous = json.loads(run.read_ref(release["previous"][prior], previous_path))
        if previous["passed"] is not False or previous["cost_unknown"] is not False:
            raise ValueError("previous qualification has an unknown cost or already passed")
        used.update({key: previous["counts"].get(key, 0) for key in LIMITS})
        previous_refs.append(release["previous"][prior])
    if len(release["previous"]) != attempt or any(not 0 <= used[key] < cap for key, cap in LIMITS.items()):
        raise ValueError("qualification correction exceeds the shared budget or lacks its prior ledger")
    claim = run.save_json(run.attempt_path("qualification_claim", attempt), dict(
        action_id=release["action_id"], release_sha256=release_sha, prepare_sha256=release["prepare_sha256"],
        previous=previous_refs, pid=os.getpid(), limits=LIMITS), exclusive=True)

    field = physics.GeneratingFunctionalKuramoto
    targets = {getattr(field, name).__code__: name for name in
               ("rhs", "hamiltonian", "gradients", "effective_edge_couplings", "power_balance_audit")}
    targets[gftd_discovery.conditional_dictionary.__code__] = "conditional_dictionary"
    targets[gftd_analysis.TaylorBudget.interval.__code__] = "taylor_budget_calls"
    targets[gftd_analysis.outward_interval.__code__] = "outward_interval"
    for name, value in vars(physics).items():
        if inspect.isfunction(value) and value.__module__ == physics.__name__:
            targets[value.__code__] = f"physics.{name}"
    targets.update({run.radau_reference.__code__: "runner_radau_entry",
                    run.reference_comparison.__code__: "runner_reference_comparison"})
    forbidden = {function: name for name, function in {
        "dop853_step": dop853_step, "rk4_step": rk4_step, "rk4": gftd_discovery.rk4,
        "time_rk4": gftd_discovery._time_rk4, "reconstruct_ports": gftd_discovery.reconstruct_ports,
        "infer_endpoint": gftd_fitting.infer_endpoint, "least_squares": least_squares,
        "legacy_fit": gftd_fitting.fit_shared, "legacy_fit297": gftd_fit297.fit_shared,
        "solve_ivp": solve_ivp, "solve_nnls": gftd_discovery.solve_nnls,
        "svd": np.linalg.svd, "eig": np.linalg.eig,
        "load_legal_packets": run.load_packets, "load_parent_checkpoint": run.load_parent,
        "score_true_future": run.score_stage,
    }.items()}
    solver_types = (Radau, DOP853)
    blocked_codes = {function.__code__: name for function, name in forbidden.items() if hasattr(function, "__code__")}
    blocked_codes.update({solver.__init__.__code__: solver.__name__ for solver in solver_types})
    counts, violations, restored = Counter(), [], []
    fixtures = {os.path.normcase(str(ROOT / path)): Path(path).stem for path in run.TESTS}

    def guard(function, name, *, blocked=False):
        @wraps(function)
        def call(*args, **kwargs):
            if blocked or used[name] + counts[name] >= LIMITS[name]:
                kind = "forbidden_scientific_call" if blocked else "shared_artificial_budget"
                violations.append(dict(kind=kind, function=name))
                raise RuntimeError(f"qualification blocks {name}")
            return function(*args, **kwargs)
        return call

    def replace(owner, name, replacement):
        restored.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def profile(frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        if code in blocked_codes:
            violations.append(dict(kind="forbidden_scientific_call", function=blocked_codes[code]))
            raise RuntimeError("unwrapped scientific call in qualification")
        name = targets.get(code)
        if name is not None:
            counts[name] += 1
        elif "test_gftd_" in code.co_filename and not code.co_name.startswith("<"):
            fixture = fixtures.get(os.path.normcase(code.co_filename))
            if fixture is not None:
                counts[f"fixture.{fixture}.{code.co_name}"] += 1

    wall, cpu = time.perf_counter(), time.process_time()
    exit_code, error, meter_lost = None, None, False
    try:
        for name in LIMITS:
            replace(field, name, guard(getattr(field, name), name))
        for solver in solver_types:
            replace(solver, "__init__", guard(solver.__init__, solver.__name__, blocked=True))
        aliases = {function: guard(function, name, blocked=True) for function, name in forbidden.items()}
        for module in tuple(sys.modules.values()):
            if module is None:
                continue
            for key, value in tuple(vars(module).items()):
                for function, replacement in aliases.items():
                    if value is function:
                        replace(module, key, replacement)
                        break
        sys.setprofile(profile)
        exit_code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp",
                                    str(ROOT / ".tmp" / f"{run.RUN}_qualification_{attempt}"), *run.TESTS]))
        meter_lost = sys.getprofile() is not profile
        run.verify_sources(prepared)
    except BaseException as exc:
        error = type(exc).__name__
        meter_lost = sys.getprofile() is not profile
        raise
    finally:
        sys.setprofile(None)
        for owner, name, original in reversed(restored):
            setattr(owner, name, original)
        cumulative = {key: used[key] + counts[key] for key in LIMITS}
        passed = (exit_code == 0 and error is None and not meter_lost and not violations
                  and all(0 < cumulative[key] <= cap for key, cap in LIMITS.items()))
        run.save_json(out, dict(run=run.RUN, qualification_attempt=attempt, claim=claim, passed=passed,
                               cost_unknown=meter_lost, prepare_sha256=release["prepare_sha256"],
                               source_sha256=prepared["source_sha256"], previous=previous_refs,
                               limits=LIMITS, counts=dict(counts), cumulative_counts=cumulative,
                               violations=violations, tests=list(run.TESTS), pytest_exit_code=exit_code, error_type=error,
                               worker_wall_seconds=time.perf_counter() - wall, worker_cpu_seconds=time.process_time() - cpu))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    raise SystemExit(qualify(parser.parse_args().release))
