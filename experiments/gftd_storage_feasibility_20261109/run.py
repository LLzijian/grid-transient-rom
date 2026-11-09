"""Frozen same-state intervention for the GFTD bounded amplitude kernel."""

from __future__ import annotations

import argparse
from dataclasses import fields
import gzip
import hashlib
import io
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import h5py
import numpy as np
import scipy
from scipy.integrate import DOP853, Radau

from kiki.models.gftd_discovery import observations
from kiki.physics import GeneratingFunctionalKuramoto, StructuralGates, complete_incidence
from kiki.physics.kuramoto_functional import bregman_kuramoto_force, bregman_kuramoto_potential


ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_storage_feasibility_20261109"
VERSION = "bounded_exp_tanh_v1"
OUTPUT = Path("results") / RUN
ARRAYS = Path("artifacts") / RUN
PROTOCOL = (Path("campaigns") / RUN / "protocol.md", "c2dca7371ca684e50e5c49c77e1b31fa4c61abc2b5ade2f1d7815c34d0a93448")
DESIGN = (Path("designs") / f"{RUN}.md", "8fd416a7f7b57f4847e0ababd283255952eed602b146da7dcece427966a6bbb9")
ADOPTION = OUTPUT / "design_adoption.json"
REVIEW = Path("reports") / RUN / "implementation_review.md"
TEST = Path("tests/test_gftd_storage_feasibility.py")
PARENT = "gftd_initialization_precision_20261109"
OLD_RUN = "gftd_long_horizon_diagnostic_20261109"
OLD_RESULTS = Path("results") / OLD_RUN / "phase1"
OLD_ARRAYS = Path("artifacts") / OLD_RUN / "phase1"
MODEL = (Path("artifacts") / PARENT / "candidate/pilot/modulated_model.json",
         "59f73171eae057ec87b6d5618dbbe224687721d1fc117cd73a50aafdb8c78dac")
DIAGNOSTIC = (Path("artifacts") / PARENT / "candidate/full/diagnostics/modulated_268.json.gz",
              "1261b6e16aa854a679e503e24bd5b51911548803c32cef698ec33141f24e395d")
INIT = (Path("results") / PARENT / "candidate/full/initialize/modulated_268.json",
        "ef35115b6f90a8d19a6a0cfd6470b31925009246438283b629fbe138e82ec6d9")
OLD_EXECUTION = (OLD_RESULTS / "execution.json", "143fe9087f1599ae668481d1f1cf54a714f6f4ca39f7170430448de69abcccbb")
STATE_HASHES = {
    "stage1": "055b46d107410d868c10eb69a1b2c7490c933938a1ffe65f3066cbcba81f5a46",
    "stage2": "37382722d19bd318c616b8c59f43390d004c874f3c2c509ee7d01188dc36c469",
}
P0_HASH = "3cafb5e809889c5664c11dcb686bd5462308a3824c7f0029ebb164d396ea225f"
WITNESSES = {
    "modulated_008": (219, "9695f3f377d37c2fbc4d839ab8c96852981bc5bf3be00de51da94d81f6ecb91e"),
    "modulated_124": (108, "888e27575e3c6449cfc54902f8c1dd8b87ad7fe2c35a73416bb674d058db4508"),
    "modulated_138": (167, "30055f3752ff2de6506f87e0ce1f589c06dd5ea9f0a16d89a210b6fea9f43046"),
    "modulated_204": (433, "16c9660afaa7de6d39c180d3374f1ef2007541c7ce9efa7570ece11478528dd2"),
    "modulated_268_stage2": (92, "a7efbcaabf404fbd351e4ea10a090a6b4b9963f799c5740cbeb008b2a61f92b7"),
}
REFERENCES = {
    "radau": (Radau, 1e-9, 1e-11),
    "radau_strict": (Radau, 1e-11, 1e-13),
    "dop853_adaptive": (DOP853, 1e-11, 1e-13),
}
GRID = np.linspace(0., 1., 101)
RHS_LIMIT = 24000
NUMERICAL_ERRORS = (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError)
COMPONENTS = ("theta", "omega", "p", "a", "r")


class BoundedAmplitudeKuramoto(GeneratingFunctionalKuramoto):
    """The inherited H and RHS use the same bounded couplings and gradients."""

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.tau_mem_coi):
            raise ValueError("tau_mem_coi must be finite")

    @classmethod
    def from_functional(cls, functional):
        return cls(**{field.name: getattr(functional, field.name) for field in fields(functional)})

    def _kernel(self, amplitude):
        a = np.asarray(amplitude, dtype=np.float64)
        if a.shape != (self.n_nodes,) or not np.isfinite(a).all():
            raise ValueError("finite node-aligned amplitude required")
        h = np.abs(self.incidence).T @ (self.gamma_amp * a)
        t = np.tanh(h)
        exponential = np.exp(t)
        s = self.gates.s_mod
        return (1. - s) + s * exponential, s * exponential * (1. - t**2)

    def effective_edge_couplings(self, z_amp):
        phi, _ = self._kernel(z_amp)
        return self.gates.s_edge * self.k_edge * phi

    def gradients(self, state):
        theta, omega, p, a, r = self.split_state(state)
        phase = self.incidence.T @ theta
        psi = bregman_kuramoto_potential(phase, self.eta_edge)
        phi, slope = self._kernel(a)
        c = self.gates.s_edge * self.k_edge
        return (
            self.incidence @ (c * phi * bregman_kuramoto_force(phase, self.eta_edge)),
            self.mass * omega,
            self.storage_p * p,
            self.lambda_amp * a + self.gamma_amp * (np.abs(self.incidence) @ (c * slope * psi)),
            self.lambda_mem * r,
        )


def qualification():
    counts = dict(rhs=0, hamiltonian=0, gradients=0, effective_couplings=0, bounded_kernel=0, saturation_formula=0)
    limits = dict(rhs=64, hamiltonian=2000)

    def counted(kind, call):
        if kind in limits and counts[kind] >= limits[kind]:
            raise ValueError(f"artificial {kind} budget exhausted")
        counts[kind] += 1
        return call()

    n = 3
    incidence, _ = complete_incidence(n)
    one = np.ones(n)
    zero = np.zeros(n)
    theta_star = np.array([.1, -.05, 0.])
    constants = dict(mass=[1., 1.5, 2.], damping=.2 * one, k_edge=[1., .8, 1.2],
                     eta_edge=incidence.T @ theta_star, gamma_amp=[.5, -.25, .75], lambda_amp=one,
                     lambda_mem=one, tau_amp=one, tau_mem_sync=one, tau_mem_coi=1., j_omega_mem=one,
                     storage_p=one, gain_p=one, rate_p=one)
    state_i = np.r_[theta_star, zero, zero, zero, zero]
    state_ii = np.array([8., 0., 0., .1, -.1, .05, .2, 0., -.1, .1, -.2, .3, .01, -.02, .03])
    positive, negative = state_ii.copy(), state_ii.copy()
    positive[9:12] = [2000., 0., 0.]
    negative[9:12] = [-2000., 0., 0.]
    fixture = {
        "A": (StructuralGates(one, zero, one), (state_i, state_ii)),
        "B": (StructuralGates([1., 1., 0.], [.5, 0., 0.], one), (state_i, state_ii)),
        "C": (StructuralGates(one, [.5, 0., 1.], one), (state_i, state_ii, positive, negative)),
    }

    def sample(f, state):
        gradient = f.gradients
        actual = []

        def capture(value):
            result = counted("gradients", lambda: gradient(value))
            actual.append(result)
            return result

        a = f.split_state(state)[3]
        h = counted("hamiltonian", lambda: f.hamiltonian(state))
        c = counted("effective_couplings", lambda: f.effective_edge_couplings(a))
        f.gradients = capture
        try:
            rhs = counted("rhs", lambda: f.rhs(state))
        finally:
            del f.gradients
        if len(actual) != 1:
            raise ValueError("expected one gradient in the inherited RHS")
        g = f.pack_state(*actual[0])
        if not np.isfinite(np.r_[h, c, g, rhs]).all():
            raise ValueError("artificial H/coupling/gradient/RHS must be finite")
        return h, c, g, rhs

    records, kernel_at_zero = [], {}
    try:
        for name, (gates, states) in fixture.items():
            f = BoundedAmplitudeKuramoto(**constants, gates=gates)
            kernel = f._kernel
            f._kernel = lambda a: counted("bounded_kernel", lambda: kernel(a))
            phi_zero, first_zero = f._kernel(zero)
            s = gates.s_mod
            # d² exp(tanh h)/dh² = exp(t)(1-t²)[(1-t²)-2t], with t=0 here.
            t = 0.
            second_zero = first_zero * ((1. - t**2) - 2. * t)
            np.testing.assert_array_equal(phi_zero, one)
            np.testing.assert_array_equal(first_zero, s)
            np.testing.assert_array_equal(second_zero, s)
            kernel_at_zero[name] = dict(phi=phi_zero.tolist(), first=first_zero.tolist(), second=second_zero.tolist())
            for index, state in enumerate(states):
                h, c, g, rhs = sample(f, state)
                phi, slope = f._kernel(state[9:12])
                if np.any((f.gates.s_edge * f.k_edge * slope)[s == 0.] != 0.):
                    raise ValueError("nonzero derivative from a static or absent edge")
                if index >= 2:
                    expected = counted("saturation_formula", lambda: .5 + .5 * np.exp(1. if index == 2 else -1.))
                    if not np.isclose(phi[0], expected, atol=0., rtol=1e-15) or slope[0] != 0.:
                        raise ValueError("bounded kernel did not reach the registered saturation limit")
                ra, rm = f.dissipative_operators(state)
                _, omega, gp, ga, gr = f.split_state(g)
                diss = float(np.sum(f.damping * (omega / f.mass)**2 + f.rate_p * gp**2 + ra * ga**2) + gr @ rm @ gr)
                power = float(g @ rhs)
                power_error = abs(power + diss)
                if power_error > 1e-11 * max(1., abs(power), diss):
                    raise ValueError("artificial power balance failed")
                item = dict(gates=name, state=index + 1, hamiltonian=h, power=power, dissipation=diss,
                            power_residual=power_error, gradient_max_abs_error={})
                records.append(item)
                if index < 2:
                    for step in (1e-5, 5e-6):
                        fd = []
                        for direction in np.eye(state.size) * step:
                            plus = counted("hamiltonian", lambda: f.hamiltonian(state + direction))
                            minus = counted("hamiltonian", lambda: f.hamiltonian(state - direction))
                            fd.append((plus - minus) / (2. * step))
                        item["gradient_max_abs_error"][str(step)] = float(np.max(np.abs(fd - g)))
                        np.testing.assert_allclose(fd, g, rtol=1e-6, atol=1e-7)
                if name == "A":
                    original = GeneratingFunctionalKuramoto(**constants, gates=gates)
                    baseline = sample(original, state)
                    item["static_max_abs_error"] = []
                    for left, right in zip((h, c, g, rhs), baseline, strict=True):
                        item["static_max_abs_error"].append(float(np.max(np.abs(left - right))))
                        np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)
    except (*NUMERICAL_ERRORS, AssertionError) as exc:
        return dict(passed=False, error=str(exc), counts=counts, limits=limits, points=records, kernel_at_zero=kernel_at_zero)
    return dict(passed=True, error=None, counts=counts, limits=limits, points=records, kernel_at_zero=kernel_at_zero)


class RhsLimit(RuntimeError):
    pass


def integrate_reference(functional, state, method, *, rhs_limit=RHS_LIMIT):
    """Count actual SciPy callbacks, including Jacobian and dense-output work."""
    # Adaptive branch and accepted-grid accounting follow the closed 20261109 long-horizon diagnostic.
    solver, rtol, atol = REFERENCES[method]
    grid_states = np.full((len(GRID), state.size), np.nan)
    grid_states[0] = state
    accepted_times, accepted_states = [0.], [state.copy()]
    calls = dict(solver_setup=0, solver_step=0, dense_output=0)
    total, output_rows, attempted, dense_attempts = 0, 1, 0, 0
    operation = "solver_setup"
    failure = failed_state = instance = None
    last_attempt = last_attempt_state = None
    status, error = "completed", None
    wall, cpu = time.perf_counter(), time.process_time()

    def checked_state(t, value):
        nonlocal failure, failed_state
        try:
            functional.split_state(value)
        except NUMERICAL_ERRORS:
            failure = dict(operation=operation, time=float(t), evaluated=None, step=attempted)
            failed_state = np.asarray(value).copy()
            raise

    def rhs(t, value):
        nonlocal total, failure, failed_state, last_attempt, last_attempt_state
        evaluated = total < rhs_limit
        last_attempt = dict(operation=operation, time=float(t), evaluated=evaluated, step=attempted)
        last_attempt_state = np.asarray(value).copy()
        try:
            if not evaluated:
                raise RhsLimit("registered RHS call limit reached")
            calls[operation] += 1
            total += 1
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return functional.rhs(value)
        except (RhsLimit, *NUMERICAL_ERRORS):
            failure, failed_state = last_attempt, last_attempt_state
            raise

    try:
        instance = solver(rhs, 0., state.copy(), 1., rtol=rtol, atol=atol, max_step=.005)
        while instance.status == "running":
            operation = "solver_step"
            attempted += 1
            last_attempt = last_attempt_state = None
            message = instance.step()
            if instance.status == "failed":
                raise FloatingPointError(message)
            checked_state(instance.t, instance.y)
            accepted_times.append(float(instance.t))
            accepted_states.append(instance.y.copy())
            if output_rows < len(GRID) and GRID[output_rows] <= instance.t:
                operation = "dense_output"
                dense_attempts += 1
                last_attempt = last_attempt_state = None
                dense = instance.dense_output()
                while output_rows < len(GRID) and GRID[output_rows] <= instance.t:
                    value = dense(GRID[output_rows])
                    checked_state(GRID[output_rows], value)
                    grid_states[output_rows] = value
                    output_rows += 1
    except (RhsLimit, *NUMERICAL_ERRORS) as exc:
        status = ("rhs_limit" if isinstance(exc, RhsLimit) else "rhs_failure"
                  if failure is not None and failure["evaluated"] is True else "solver_failure")
        error = repr(exc)
    if error is None and (output_rows != len(GRID) or accepted_times[-1] != 1.):
        status, error = "incomplete", "target time or required output grid was incomplete"
    if error is not None and failure is None:
        failure = dict(operation=operation, time=None, evaluated=None, step=attempted)
    record = dict(method=method, rhs_version=VERSION, status=status, completed=status == "completed",
                  target_time=1., solver=solver.__name__, rtol=rtol, atol=atol, max_step=.005,
                  last_accepted_time=accepted_times[-1], last_output_time=float(GRID[output_rows - 1]),
                  output_rows=output_rows, external_step_attempts=attempted, accepted_steps=len(accepted_times) - 1,
                  dense_output_attempts=dense_attempts, internal_rejected_steps=None,
                  actual_rhs_calls=total, rhs_calls_by_operation=calls, rhs_limit=rhs_limit, error=error, failure=failure,
                  wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu,
                  **{f"solver_{key}": getattr(instance, key, None) for key in ("nfev", "njev", "nlu")})
    arrays = dict(grid_times=GRID, grid_states=grid_states, accepted_times=np.asarray(accepted_times),
                  accepted_states=np.asarray(accepted_states))
    if failed_state is not None:
        arrays["failure_state"] = failed_state
    elif error is not None and last_attempt is not None:
        record["last_attempted_callback"] = last_attempt
        arrays["last_attempted_state"] = last_attempt_state
    return record, arrays


def reference_distances(mass, x, y):
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.abs(observations(x, mass) - observations(y, mass)).reshape(len(x), -1)
        scale = 1. + np.maximum(np.abs(x), np.abs(y))
        relative = np.abs(x - y) / scale
        bad = ~np.isfinite(relative)
        relative[bad] = np.abs(x[bad] / scale[bad] - y[bad] / scale[bad])
    return delta, relative


def reference_prefix(mass, paths):
    """Return common finite row count and maximum continuous qualified row count."""
    counts = []
    for path in paths.values():
        bad = np.flatnonzero(~np.isfinite(path).all(axis=1))
        counts.append(int(bad[0]) if bad.size else len(path))
    common = min(counts)
    keep = common
    for left, right in combinations(paths.values(), 2):
        delta, relative = reference_distances(mass, left[:common], right[:common])
        out = np.flatnonzero(~((delta.max(axis=1) <= 1e-5) & (relative.max(axis=1) <= 1e-5)))
        if out.size:
            keep = min(keep, int(out[0]))
        rms = np.sqrt(np.cumsum(np.mean(delta[:keep]**2, axis=1)) / np.arange(1, keep + 1))
        out = np.flatnonzero(rms > 1e-6)
        if out.size:
            keep = int(out[0])
    return common, keep


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def workspace_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("path escapes the workspace")
    return path


def read_bound(path, digest, size=None):
    raw = workspace_path(path).read_bytes()
    if sha(raw) != digest or (size is not None and len(raw) != size):
        raise ValueError(f"frozen bytes changed: {path}")
    return raw


def read_ref(ref, expected):
    if ref["path"] != expected.as_posix():
        raise ValueError(f"unexpected frozen reference: {expected}")
    return read_bound(expected, ref["sha256"], ref.get("bytes"))


def save(path, data, *, exclusive=False):
    dest = workspace_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    target = dest if exclusive else dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.pending")
    with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if not exclusive:
        os.replace(target, dest)
    return dict(path=path.as_posix(), sha256=sha(data), bytes=len(data))


def bindings():
    for bound in (PROTOCOL, DESIGN):
        read_bound(*bound)
    source = {Path(__file__).resolve(), workspace_path(TEST)}
    for name, module in tuple(sys.modules.items()):
        if name == "kiki" or name.startswith("kiki."):
            source.add(Path(module.__file__).resolve())
    if any(not path.is_relative_to(ROOT) for path in source):
        raise ValueError("first-party import escaped the source workspace")
    return dict(version=VERSION, protocol_sha256=PROTOCOL[1], design_sha256=DESIGN[1],
                source_sha256={path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in sorted(source)},
                runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                             h5py=h5py.__version__, threads={key: os.getenv(key) for key in
                             ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))


def load_inputs():
    payload = json.loads(read_bound(*MODEL))
    if (payload["format"], payload["fit_mode"], payload["initialization_mode"], payload["inference_config"]["method"]) != (
            "gftd_kuramoto_5n_v2", "modulated", "causal_two_stage", "fixed_dp8"):
        raise ValueError("unexpected frozen model dispatch")
    parent = GeneratingFunctionalKuramoto(**payload["functional"], gates=StructuralGates(**payload["gates"]))
    if parent.n_nodes != 10 or len(set(parent.edges)) != 45 or any(i >= j for i, j in parent.edges):
        raise ValueError("expected the original ten-node simple graph")

    init = json.loads(read_bound(*INIT))
    if init["diagnostics"]["sha256"] != DIAGNOSTIC[1]:
        raise ValueError("frozen diagnostic hash changed")
    diagnostic = json.loads(gzip.decompress(read_ref(init["diagnostics"], DIAGNOSTIC[0])))
    if (init["event_id"] != "pilot_0255" or init["state"] is not None or init["stage"] != "stage2"
            or init["status"] != "numerically_unresolved" or diagnostic["status"] != init["status"]
            or diagnostic["method"] != "fixed_dp8" or diagnostic["returned_point"] != "stage2"
            or init["stage1_gates"]["passed"] is not True or init["final_gates"]["passed"] is not False
            or diagnostic["stage1_numerical"]["passed"] is not True or diagnostic["final_numerical"]["passed"] is not False
            or "underflow encountered in exp" not in diagnostic["final_numerical"]["error"]):
        raise ValueError("frozen 268 stage identity or original gate changed")
    states, mse = {}, {}
    for stage, digest in STATE_HASHES.items():
        best = diagnostic[stage]["best"]
        state = np.ascontiguousarray(best["state"], dtype="<f8")
        parent.split_state(state)
        if sha(state.tobytes()) != digest or sha(state[20:30].tobytes()) != P0_HASH:
            raise ValueError(f"frozen 268 bytes or endpoint anchor changed: {stage}")
        states[stage], mse[stage] = state, float(best["history_mse"])
    if states["stage1"][:30].tobytes() != states["stage2"][:30].tobytes() or not (
            np.isfinite(list(mse.values())).all() and mse["stage1"] > mse["stage2"]):
        raise ValueError("the stage comparison lost its frozen observation anchor or historical order")

    old = json.loads(read_bound(*OLD_EXECUTION))
    if old["status"] != "COMPLETED":
        raise ValueError("original reference execution is not complete")
    beta = parent.gates.s_edge * parent.k_edge * parent.gates.s_mod
    witnesses = {}
    for name, (index, digest) in WITNESSES.items():
        summary_ref = old["summaries"][name]
        summary = json.loads(read_ref(summary_ref, OLD_RESULTS / "summaries" / f"{name}.json"))
        array_ref = summary["arrays"]
        if array_ref["sha256"] != digest or summary["reference"]["agreement_end"] != index / 100.:
            raise ValueError(f"frozen witness identity changed: {name}")
        raw = read_ref(array_ref, OLD_ARRAYS / "summaries" / f"{name}.npz")
        prefix = f"sample_radau_strict_{index}_"
        with np.load(io.BytesIO(raw), allow_pickle=False) as source:
            state, active, h, cached_beta, psi, product, grad, rhs = (
                source[prefix + key] for key in
                ("state", "active", "endpoint_sum", "beta", "psi", "beta_psi", "amplitude_gradient", "rhs"))
        parent.split_state(state)
        parent.split_state(rhs)
        if (not all(np.isfinite(values).all() for values in (h, psi, product, grad))
                or not np.array_equal(active, beta > 0.) or not np.array_equal(cached_beta, beta)
                or not np.array_equal(product, beta * psi)):
            raise ValueError(f"frozen witness coefficients changed: {name}")
        edge = int(np.argmax(np.where(active, h, -np.inf)))
        nodes = np.array(parent.edges[edge])
        if product[edge] >= 0. or np.any(parent.gamma_amp[nodes] == 0.):
            raise ValueError(f"negative coupled witness condition failed: {name}")
        coupling_gradient = grad - parent.lambda_amp * state[30:40]
        witnesses[name] = dict(summary=summary_ref, arrays=array_ref, time=index / 100., edge=nodes.tolist(),
                               beta_psi=float(product[edge]), gamma=parent.gamma_amp[nodes].tolist(),
                               amplitude_coupling_gradient=coupling_gradient[nodes].tolist(),
                               amplitude_rhs=rhs[30:40][nodes].tolist())

    original, original_paths, original_evidence, old_growth = {}, {}, {}, {}
    for stage, state in states.items():
        name = f"modulated_268_{stage}"
        summary_ref = old["summaries"][name]
        summary = json.loads(read_ref(summary_ref, OLD_RESULTS / "summaries" / f"{name}.json"))
        old_growth[stage] = summary["growth"]
        original[stage], original_paths[stage], original_evidence[stage] = {}, {}, {}
        for method in REFERENCES:
            action_id = f"{old['action_id']}:{name}:{method}"
            ref = old["tasks"][action_id]
            job = json.loads(read_ref(ref, OLD_RESULTS / "jobs" / name / f"{method}.json"))
            if (job["action_id"], job["case_id"], job["method"], job["rhs_version"],
                    job["target_time"], job["state_sha256"]) != (action_id, name, method, "original_strict", 1., STATE_HASHES[stage]):
                raise ValueError("original reference Job identity changed")
            raw = read_ref(job["arrays"], OLD_ARRAYS / "jobs" / name / f"{method}.npz")
            with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
                grid, path = arrays["grid_times"], arrays["grid_states"]
            rows = job["output_rows"]
            if (not np.array_equal(grid, GRID) or path.shape != (len(GRID), 50) or path.dtype != np.float64
                    or type(rows) is not int or not 1 <= rows <= len(GRID)
                    or path[0].tobytes() != state.tobytes() or not np.isfinite(path[:rows]).all()
                    or not np.isnan(path[rows:]).all() or job["last_output_time"] != GRID[rows - 1]):
                raise ValueError("original reference grid identity changed")
            original[stage][method], original_paths[stage][method] = job, path
            original_evidence[stage][method] = dict(job=ref, arrays=job["arrays"])
        common, keep = reference_prefix(parent.mass, original_paths[stage])
        if (common - 1, keep - 1) != tuple(round(100 * summary["reference"][key]) for key in ("common_end", "agreement_end")):
            raise ValueError("original reference prefix differs from the frozen summary")
    inputs = dict(model=dict(path=MODEL[0].as_posix(), sha256=MODEL[1]), diagnostic=init["diagnostics"],
                  original_execution_sha256=OLD_EXECUTION[1], original_initialize_sha256=INIT[1],
                  stage_state_sha256=STATE_HASHES, p0_sha256=P0_HASH, original_state=None,
                  stage_history_mse=mse, stage1_original_gates=init["stage1_gates"], stage2_original_gates=init["final_gates"],
                  qualification_first_counterfactual_stage="stage1", witnesses=witnesses,
                  original_reference_jobs=original_evidence)
    return parent, states, original, original_paths, old_growth, inputs


def path_ranges(functional, path, bounded, counts):
    _, _, _, a, _ = np.split(path, 5, axis=1)
    gates = functional.gates
    active = gates.s_edge * functional.k_edge * gates.s_mod > 0.
    h = (a * functional.gamma_amp) @ np.abs(functional.incidence[:, active])
    t = np.tanh(h) if bounded else h
    with np.errstate(over="raise", invalid="raise"):
        s = gates.s_mod[active]
        phi = 1. - s + s * np.exp(t)
    counts["candidate" if bounded else "original"]["formula_calls"] += 1
    counts["candidate" if bounded else "original"]["exp_elements"] += int(t.size)

    def ranges(values):
        return dict(start=[float(values[0].min()), float(values[0].max())],
                    end=[float(values[-1].min()), float(values[-1].max())],
                    full=[float(values.min()), float(values.max())])

    amplitude = np.max(np.abs(a), axis=1)
    a0, end = float(amplitude[0]), float(amplitude[-1])
    return dict(A0=a0, Aend=end, Amax=float(amplitude.max()), deltaA=end - a0,
                Aend_over_A0=end / a0 if a0 else None, h=ranges(h), exponent_argument=ranges(t), phi=ranges(phi),
                components={key: ranges(values) for key, values in zip(COMPONENTS, np.split(path, 5, axis=1), strict=True)})


def summarize(parent, old_jobs, old_paths, new_jobs, new_paths, old_growth):
    algebra = {version: dict(formula_calls=0, exp_elements=0) for version in ("original", "candidate")}
    stages = {}
    for stage in STATE_HASHES:
        own = {}
        for label, jobs, paths in (("original", old_jobs, old_paths), ("candidate", new_jobs, new_paths)):
            common, keep = reference_prefix(parent.mass, paths[stage])
            if keep < 1:
                raise ValueError("same-kernel reference identity disagrees at the initial row")
            pair_metrics = {}
            for (left_name, left), (right_name, right) in combinations(paths[stage].items(), 2):
                delta, relative = reference_distances(parent.mass, left[:keep], right[:keep])
                pair_metrics[f"{left_name}:{right_name}"] = dict(
                    observation_rms=float(np.sqrt(np.mean(delta**2))), observation_peak=float(delta.max()),
                    state_relative_max=float(relative.max()))
            own[label] = dict(common_end=(common - 1) / 100., agreement_end=(keep - 1) / 100.,
                              agreement_rows=keep, qualified_full_second=keep == len(GRID), agreement_pairs=pair_metrics,
                              methods={method: {key: record[key] for key in
                                       ("completed", "status", "last_accepted_time", "last_output_time", "error")}
                                       for method, record in jobs[stage].items()})
        matched = min(item["agreement_rows"] for item in own.values())
        t_pair = (matched - 1) / 100.
        stats = {label: {method: path_ranges(parent, path[:matched], label == "candidate", algebra)
                        for method, path in paths[stage].items()}
                 for label, paths in (("original", old_paths), ("candidate", new_paths))}
        stages[stage] = dict(within_model=own, T_pair=t_pair, T_pair_rows=matched, mechanism_eligible=t_pair >= .1,
                             matched_window=stats, inherited_original_growth=old_growth[stage])
    return dict(stages=stages, algebra_counts=algebra, real_extra_rhs=0, real_extra_hamiltonian=0,
                real_extra_gradients=0, real_extra_power=0)


def prepare():
    if workspace_path(OUTPUT / "qualification_claim.json").exists():
        raise ValueError("preparation is frozen by the qualification claim")
    *_, inputs = load_inputs()
    return save(OUTPUT / "prepare.json", json_bytes(dict(bindings(), inputs=inputs)))


def qualified():
    path = OUTPUT / "prepare.json"
    data = workspace_path(path).read_bytes()
    prepared = json.loads(data)
    if {key: value for key, value in prepared.items() if key != "inputs"} != bindings():
        raise ValueError("qualification source or runtime differs from preparation")
    prep_sha = sha(data)
    claim = save(OUTPUT / "qualification_claim.json", json_bytes(dict(
        action_id=f"{RUN}:qualification", prepare_sha256=prep_sha, pid=os.getpid(),
        cumulative_limits=dict(rhs=64, hamiltonian=2000))), exclusive=True)
    wall, cpu = time.perf_counter(), time.process_time()
    record = qualification()
    record.update(version=VERSION, prepare_sha256=prep_sha, claim=claim, source_sha256=prepared["source_sha256"],
                  cpu_seconds=time.process_time() - cpu, wall_seconds=time.perf_counter() - wall)
    return save(OUTPUT / "qualification.json", json_bytes(record)), record["passed"]


def release_inputs(path):
    release_raw = workspace_path(path).read_bytes()
    release = json.loads(release_raw)
    if (set(release) != {"version", "action_id", "prepare_sha256", "qualification_sha256", "adoption",
                         "implementation_review"} or release["version"] != VERSION
            or not isinstance(release["action_id"], str) or not release["action_id"].strip()):
        raise ValueError("unexpected release fields or candidate version")
    for key, expected in (("adoption", ADOPTION), ("implementation_review", REVIEW)):
        read_ref(release[key], expected)
    prepared = json.loads(read_bound(OUTPUT / "prepare.json", release["prepare_sha256"]))
    qual = json.loads(read_bound(OUTPUT / "qualification.json", release["qualification_sha256"]))
    read_ref(qual["claim"], OUTPUT / "qualification_claim.json")
    if (qual["passed"] is not True or qual["version"] != VERSION
            or qual["prepare_sha256"] != release["prepare_sha256"] or qual["source_sha256"] != prepared["source_sha256"]
            or qual["limits"] != {"rhs": 64, "hamiltonian": 2000}
            or not 0 < qual["counts"]["rhs"] < 64 or not 0 < qual["counts"]["hamiltonian"] < 2000):
        raise ValueError("candidate qualification is missing, changed or failed")
    model, states, old_jobs, old_paths, old_growth, inputs = load_inputs()
    if prepared != dict(bindings(), inputs=inputs):
        raise ValueError("scientific inputs, source or runtime differ from preparation")
    return release, sha(release_raw), prepared, qual, model, states, old_jobs, old_paths, old_growth


def execute(path):
    host_wall, host_cpu = time.perf_counter(), time.process_time()
    release, release_sha, prepared, qual, parent, states, old_jobs, old_paths, old_growth = release_inputs(path)
    action_id = release["action_id"]
    action = dict(action_id=action_id, version=VERSION, release_sha256=release_sha, pid=os.getpid())
    claim = save(OUTPUT / "execution_claim.json", json_bytes(action), exclusive=True)
    records, paths, refs, claims, known = {}, {}, {}, {}, {}
    summary = failure = None
    try:
        f = BoundedAmplitudeKuramoto.from_functional(parent)
        for stage, state in states.items():
            records[stage], paths[stage], refs[stage] = {}, {}, {}
            for method in REFERENCES:
                job_id = f"{action_id}:{stage}:{method}"
                job_meta = dict(action, job_id=job_id, stage=stage, method=method, state_sha256=STATE_HASHES[stage])
                job_claim = save(OUTPUT / "claims" / stage / f"{method}.json", json_bytes(job_meta), exclusive=True)
                claims[job_id] = job_claim
                record, arrays = integrate_reference(f, state, method)
                record.update(job_meta, claim=job_claim, arrays=None)
                known[job_id] = record
                job_path = OUTPUT / "jobs" / stage / f"{method}.json"
                refs[stage][method] = save(job_path, json_bytes(record))
                archive = io.BytesIO()
                np.savez_compressed(archive, **arrays)
                ref = save(ARRAYS / "jobs" / stage / f"{method}.npz", archive.getvalue())
                record = dict(record, arrays=dict(ref, local_only=True, non_upload=True))
                refs[stage][method] = save(job_path, json_bytes(record))
                known[job_id] = record
                records[stage][method], paths[stage][method] = record, arrays["grid_states"]
        summary = save(OUTPUT / "summary.json", json_bytes(summarize(parent, old_jobs, old_paths, records, paths, old_growth)))
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        unknown = {job_id: ref for job_id, ref in claims.items() if job_id not in known}
        rhs = sum(job["actual_rhs_calls"] for job in known.values())
        cpu = sum(job["cpu_seconds"] for job in known.values())
        wall = sum(job["wall_seconds"] for job in known.values())
        record = dict(action, status="COMPLETED" if failure is None else "INTERRUPTED_RUNTIME", error=failure,
                      claim=claim, jobs=refs, summary=summary, claimed_job_count=len(claims),
                      returned_integration_jobs=len(known), unknown_jobs=unknown,
                      jobs_without_arrays=[job for job in known.values() if job["arrays"] is None],
                      prepare_sha256=release["prepare_sha256"], qualification_sha256=release["qualification_sha256"],
                      artificial_counts=qual["counts"], known_integration_rhs_calls=rhs, known_worker_cpu_seconds=cpu,
                      known_worker_wall_seconds=wall, actual_integration_rhs_calls=None if unknown else rhs,
                      worker_cpu_seconds=None if unknown else cpu, worker_wall_seconds=None if unknown else wall,
                      host_cpu_seconds=time.process_time() - host_cpu, host_wall_seconds=time.perf_counter() - host_wall,
                      source_sha256=prepared["source_sha256"], runtime=prepared["runtime"])
        try:
            ref = save(OUTPUT / "execution.json", json_bytes(record))
        except OSError:
            print(json.dumps({key: record[key] for key in
                              ("action_id", "status", "known_integration_rhs_calls", "returned_integration_jobs")}),
                  file=sys.stderr)
            raise
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--qualify", action="store_true")
    mode.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.qualify:
        ref, passed = qualified()
        print(json.dumps(ref))
        return 0 if passed else 1
    print(json.dumps(prepare() if args.prepare_only else execute(args.release)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
