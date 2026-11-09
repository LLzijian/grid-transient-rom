"""Run the fixed single-point and mocked checks once, with actual-call accounting."""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from functools import wraps
from pathlib import Path
import sys
import time

import pytest
from scipy.integrate import DOP853, Radau, solve_ivp

from kiki.models import gftd_discovery, gftd_fit297, gftd_fitting
from kiki.numerics import dop853_step
from kiki.physics import kuramoto_functional as physics


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN = "gftd_bounded_initialization_20261109"
OUTPUT = ROOT / "results" / RUN
TESTS = ("tests/test_gftd_bounded_kernel.py", "tests/test_gftd_bounded_initialization_runner.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prepare_path = OUTPUT / "prepare.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    for path, digest in prepare["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"qualification source changed: {path}")
    required = {*TESTS, Path(__file__).relative_to(ROOT).as_posix()}
    if not required <= set(prepare["source_sha256"]):
        raise ValueError("prepare must bind the qualification driver and both test files")
    if (OUTPUT / "qualification.json").exists():
        raise FileExistsError("qualification already has an outcome; no automatic replay")
    with (OUTPUT / "qualification_claim.json").open("x", encoding="utf-8") as stream:
        json.dump({"prepare_sha256": sha(prepare_path), "source_sha256": prepare["source_sha256"]}, stream, indent=2)
        stream.write("\n")

    field = physics.GeneratingFunctionalKuramoto
    targets = {getattr(field, name).__code__: name for name in
               ("rhs", "hamiltonian", "gradients", "effective_edge_couplings", "power_balance_audit")}
    targets[gftd_discovery.conditional_dictionary.__code__] = "conditional_dictionary"
    for name, value in vars(physics).items():
        if inspect.isfunction(value) and value.__module__ == physics.__name__:
            targets[value.__code__] = "physics." + name
    forbidden_functions = {
        "dop853_step": dop853_step, "rk4": gftd_discovery.rk4,
        "reconstruct_ports": gftd_discovery.reconstruct_ports,
        "infer_endpoint": gftd_fitting.infer_endpoint,
        "fit_shared": gftd_fitting.fit_shared, "fit297_shared": gftd_fit297.fit_shared,
        "Radau": Radau.__init__, "DOP853": DOP853.__init__, "solve_ivp": solve_ivp,
    }
    forbidden = {value.__code__: name for name, value in forbidden_functions.items()}
    counts, violations = Counter(), []
    limits = {"rhs": 128, "hamiltonian": 2000}
    restored = []

    def guarded(function, name, *, blocked=False):
        @wraps(function)
        def call(*args, **kwargs):
            if blocked or counts[name] >= limits[name]:
                violations.append({"kind": "forbidden_scientific_call" if blocked else "artificial_budget_exhausted",
                                   "function": name})
                raise RuntimeError(f"qualification blocks {name}")
            return function(*args, **kwargs)
        return call

    for name in limits:
        function = getattr(field, name)
        restored.append((field, name, function))
        setattr(field, name, guarded(function, name))
    # Guard known aliases before pytest imports its fixtures; mock replacements remain allowed.
    for name, function in forbidden_functions.items():
        replacement = guarded(function, name, blocked=True)
        if name in ("Radau", "DOP853"):
            owner = {"Radau": Radau, "DOP853": DOP853}[name]
            restored.append((owner, "__init__", function))
            setattr(owner, "__init__", replacement)
        else:
            for module in tuple(sys.modules.values()):
                if module is None:
                    continue
                for key, value in tuple(vars(module).items()):
                    if value is function:
                        restored.append((module, key, function))
                        setattr(module, key, replacement)

    def profile(frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        if code in forbidden:
            violations.append({"kind": "forbidden_scientific_call", "function": forbidden[code]})
        name = targets.get(code)
        if name is not None:
            counts[name] += 1

    started, cpu = time.perf_counter(), time.process_time()
    outcome, error = None, None
    try:
        sys.setprofile(profile)
        outcome = int(pytest.main(["-q", "-p", "no:cacheprovider", "--basetemp",
                                  str(ROOT / ".tmp" / (RUN + "_qualification")), *TESTS]))
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        sys.setprofile(None)
        for owner, name, function in reversed(restored):
            setattr(owner, name, function)
        result = {"run_id": RUN, "passed": outcome == 0 and not violations,
                  "prepare_sha256": sha(prepare_path), "source_sha256": prepare["source_sha256"],
                  "pytest_exit_code": outcome, "tests": list(TESTS), "counts": dict(counts),
                  "limits": limits, "violations": violations, "error": error,
                  "wall_seconds": time.perf_counter() - started,
                  "cpu_seconds": time.process_time() - cpu,
                  "count_scope": "Actual Python function calls; mocked replacements have distinct code and are not model RHS.",
                  "new_ode": 0 if not violations else None,
                  "new_initialization_optimization": 0 if not violations else None}
        temporary = OUTPUT / "qualification.json.tmp"
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(OUTPUT / "qualification.json")
    return outcome if outcome is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
