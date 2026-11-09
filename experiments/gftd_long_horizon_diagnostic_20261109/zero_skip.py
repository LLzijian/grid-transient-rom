"""The conditional zero-coefficient evaluation for the frozen long-horizon Run."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import fields
import io
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from kiki.physics.kuramoto_functional import (
    GeneratingFunctionalKuramoto,
    StructuralGates,
    bregman_kuramoto_force,
    bregman_kuramoto_potential,
    latent_coupling_modulation,
)

from . import run


VERSION = "inactive_zero_skip"
OUTPUT = run.OUTPUT / VERSION
PRIVATE = run.PRIVATE / VERSION
PHASE1 = run.OUTPUT / "phase1"
PHASE1_RAW = run.PRIVATE / "phase1"
PHASE1_SHA = "143fe9087f1599ae668481d1f1cf54a714f6f4ca39f7170430448de69abcccbb"
PHASE1_ADOPTION_SHA = "922ccc7cb9cc5c802b5415b1a3cb1fc7584d71e6386eda13dad86c08af20f1e7"
BRANCH_TARGETS = {"modulated_138": 188, "modulated_204": 454, "modulated_268_stage2": 100, "modulated_296": 1500}
SOURCE_PATHS = (*run.SOURCE_PATHS, f"experiments/{run.RUN}/zero_skip.py", "tests/test_gftd_long_horizon_zero_skip.py")
POINT_TOL = 1e-12


class InactiveZeroSkip(GeneratingFunctionalKuramoto):
    """Evaluate the original functional with exponentials only on nonzero terms."""

    @classmethod
    def from_functional(cls, functional):
        return cls(**{field.name: getattr(functional, field.name) for field in fields(functional)})

    def __post_init__(self):
        super().__post_init__()
        # A rounded-to-zero product of nonzero factors does not authorize a skip.
        self._modulated = (self.gates.s_edge != 0.) & (self.k_edge != 0.) & (self.gates.s_mod != 0.)
        self._modulated_edges = tuple(edge for edge, active in zip(self.edges, self._modulated, strict=True) if active)
        self._modulated_gates = StructuralGates(self.gates.s_edge[self._modulated],
                                                self.gates.s_mod[self._modulated], self.gates.s_mem)
        self._absolute_incidence = np.abs(self.incidence)
        self._modulated_incidence = self._absolute_incidence.copy()
        self._modulated_incidence[:, ~self._modulated] = 0.
        self._modulated_nodes = np.any(self._modulated_incidence != 0., axis=1)

    def effective_edge_couplings(self, z_amp):
        amplitude = np.asarray(z_amp, dtype=np.float64).reshape(-1)
        if amplitude.shape != self.gamma_amp.shape or not np.isfinite(amplitude).all():
            raise ValueError("finite amplitude states must match the node dimension")
        modulation = np.ones(self.n_edges)
        if self._modulated_edges:
            modulation[self._modulated], _, _ = latent_coupling_modulation(
                amplitude, self._modulated_edges, self.gamma_amp, self._modulated_gates)
        return self.gates.s_edge * self.k_edge * modulation

    def gradients(self, state):
        theta, omega, p, z_amp, z_mem = self.split_state(state)
        phi = self.incidence.T @ theta
        psi = bregman_kuramoto_potential(phi, self.eta_edge)
        couplings = self.effective_edge_couplings(z_amp)
        grad_amp = self.lambda_amp * z_amp
        if self._modulated_edges:
            with np.errstate(over="raise", under="raise", invalid="raise"):
                products = np.multiply(self.gamma_amp, z_amp, out=np.zeros_like(z_amp), where=self._modulated_nodes)
                exponent = self._modulated_incidence.T @ products
                exp_h = np.exp(exponent, out=np.zeros_like(exponent), where=self._modulated)
            grad_amp += self.gamma_amp * (self._absolute_incidence @ (
                self.gates.s_edge * self.k_edge * self.gates.s_mod * exp_h * psi))
        return (
            self.incidence @ (couplings * bregman_kuramoto_force(phi, self.eta_edge)),
            self.mass * omega, self.storage_p * p, grad_amp, self.lambda_mem * z_mem,
        )


def evaluate_point(functional, state, budget):
    values, errors, counts = {}, {}, {}

    def evaluate(name, operation):
        counts[name] = counts.get(name, 0) + 1
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = np.asarray(operation())
            if not np.isfinite(result).all():
                raise FloatingPointError("nonfinite diagnostic output")
            values[name] = result
        except run.NUMERICAL_ERRORS as error:
            errors[name] = repr(error)

    _, omega, _, amplitude, _ = functional.split_state(state)
    evaluate("couplings", lambda: functional.effective_edge_couplings(amplitude))
    evaluate("hamiltonian", lambda: functional.hamiltonian(state))
    evaluate("gradients", lambda: functional.gradients(state))
    if budget[0] >= 500:
        raise RuntimeError("registered single-point RHS limit reached")
    budget[0] += 1
    evaluate("rhs", lambda: functional.rhs(state))

    def power():
        gradients = values["gradients"]
        _, _, gp, ga, gr = gradients
        ra, rm = functional.dissipative_operators(state)
        diss = ((functional.damping * omega**2).sum() + (functional.rate_p * gp**2).sum()
                + (ra * ga**2).sum() + gr @ rm @ gr)
        return [gradients.ravel() @ values["rhs"], -diss]

    if "rhs" in values and "gradients" in values:
        evaluate("power", power)
    return values, errors, counts


def relative_max(left, right):
    scale = 1. + np.maximum(np.abs(left), np.abs(right))
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(left - right) / scale
        difference = np.where(np.isfinite(difference), difference, np.abs(left / scale - right / scale))
    return float(np.max(difference))


def point_comparison(original, candidate, state, budget):
    left, left_errors, left_counts = evaluate_point(original, state, budget)
    right, right_errors, right_counts = evaluate_point(candidate, state, budget)
    differences = {name: relative_max(left[name], right[name]) for name in left if name in right}
    power_errors = {name: relative_max(*values["power"]) if "power" in values else None
                    for name, values in (("original", left), ("candidate", right))}
    counts = {}
    edge_info, edge_arrays = run.inspect_state(original, state, counts)
    labels = run.exponent_labels(edge_info["exponents"])
    matched = all(value <= POINT_TOL for value in differences.values())
    power_valid = all(value is None or value <= POINT_TOL for value in power_errors.values())
    inactive = "inactive_zero_exp_out_of_range" in labels
    active = any(label.startswith("active_exp_") for label in labels)
    def strict_exp_errors(errors, active_only=False):
        kinds = (kind for kind in ("overflow", "underflow") if not active_only or f"active_exp_{kind}" in labels)
        messages = tuple(f"{kind} encountered in exp" for kind in kinds)
        return bool(errors) and all(any(message in error for message in messages) for error in errors.values())

    strict_exception = strict_exp_errors(left_errors) and strict_exp_errors(right_errors, active_only=True) and active
    unblocked = strict_exp_errors(left_errors) and not right_errors and inactive and not active
    equivalent = not left_errors and not right_errors and matched and power_valid
    passed = bool(equivalent or (unblocked or strict_exception) and matched and power_valid)
    status = ("equivalent" if equivalent else "zero_term_unblocked" if unblocked and passed else
              "active_exception_preserved" if strict_exception and passed else "failed")
    record = dict(status=status, passed=passed, differences=differences, power_residual_relative=power_errors,
                  labels=labels, original_errors=left_errors, candidate_errors=right_errors,
                  original_counts=left_counts, candidate_counts=right_counts, edge_counts=counts)
    arrays = dict(edge_arrays)
    arrays.update({f"{side}_{name}": value for side, values in (("original", left), ("candidate", right))
                   for name, value in values.items()})
    return record, arrays


@contextmanager
def branch_paths():
    previous = run.OUTPUT, run.PRIVATE, run.RHS_VERSION
    run.OUTPUT, run.PRIVATE, run.RHS_VERSION = OUTPUT, PRIVATE, VERSION
    try:
        yield
    finally:
        run.OUTPUT, run.PRIVATE, run.RHS_VERSION = previous


def branch_worker(*args):
    with branch_paths():
        return run.run_case_jobs(*args)


def frozen_inputs():
    phase1 = json.loads(run.checked_bytes(PHASE1 / "execution.json", PHASE1_SHA))
    if phase1["status"] != "COMPLETED" or phase1["sample_rhs_calls"] != 171:
        raise ValueError("phase1 receipt did not preserve the registered single-point work")
    run.checked_bytes(run.PROTOCOL, phase1["protocol_sha256"])
    run.checked_bytes(run.OUTPUT / "design_adoption.json", phase1["design_adoption_sha256"])
    run.checked_bytes(Path("reports") / run.RUN / "design_review.md", phase1["design_review_sha256"])
    adoption = json.loads(run.checked_bytes(run.OUTPUT / "phase1_adoption.json", PHASE1_ADOPTION_SHA))
    branch = adoption["conditional_preparation"]
    if (adoption["execution_sha256"] != PHASE1_SHA or branch["real_zero_term_trigger_verified"] is not True
            or branch["states"] != list(BRANCH_TARGETS)
            or branch["approved_output_prefixes"] != [OUTPUT.as_posix(), PRIVATE.as_posix()]):
        raise ValueError("conditional preparation disagrees with phase1 adoption")
    for name, file in (("scientist", "phase1_scientific_assessment.md"), ("review", "phase1_review.md")):
        path = Path("reports") / run.RUN / file
        if adoption[f"{name}_reference"] != path.as_posix():
            raise ValueError(f"unexpected phase1 {name} reference")
        run.checked_bytes(path, adoption[f"{name}_sha256"])
    for path, digest in phase1["source_sha256"].items():
        run.checked_bytes(path, digest)
    summary_path = PHASE1 / "summaries/modulated_138.json"
    summary = json.loads(run.read_ref(phase1["summaries"]["modulated_138"], summary_path))
    if (not summary["inactive_zero_skip_candidate"] or
            "inactive_zero_exp_out_of_range" not in summary["terminal"]["fine"]["failure_exponent_labels"]):
        raise ValueError("the registered archived failure lacks a zero-term exponent trigger")
    prepared = run.checked_bytes(run.OUTPUT / "prepare.json", phase1["prepared_sha256"])
    models, cases, inventory = run.load_inputs()
    if run.json_bytes(inventory) != prepared:
        raise ValueError("original prepared input identity changed")
    if any(inventory["cases"][name]["target_samples"] != samples for name, samples in BRANCH_TARGETS.items()):
        raise ValueError("conditional trajectory targets changed")
    return phase1, models, cases, inventory, adoption


def read_job(phase1, case_id, method):
    path = PHASE1 / "jobs" / case_id / f"{method}.json"
    action = f"{phase1['action_id']}:{case_id}:{method}"
    ref = phase1["tasks"].get(action)
    if ref is None:
        return None, None
    record = json.loads(run.read_ref(ref, path))
    if (record["case_id"], record["method"], record["action_id"], record["rhs_version"]) != (
            case_id, method, action, "original_strict"):
        raise ValueError("original trajectory job identity changed")
    return record, ref


def point_plan(phase1, cases, inventory, adoption):
    points, states = [], []
    anomalies = ("modulated_008", "modulated_124", "modulated_138", "modulated_204")

    def add(case_id, role, state, t, source, array_key, receipt=None):
        state = run.checked_state(state)
        points.append(dict(index=len(points) + 1, case_id=case_id, role=role, time=t, source=source,
                           array_key=array_key, receipt=receipt, state_sha256=run.sha(state.tobytes())))
        states.append(state)

    def initial(case_id):
        row = inventory["cases"][case_id]
        key = "state" if row["state"] is not None else f"{row['stage']}.best.state"
        add(case_id, "initial", cases[case_id].state, 0., row["state"] or row["diagnostics"], key,
            row["initialization"])

    def cached(case_id, method, key):
        case, row = cases[case_id], inventory["cases"][case_id]
        leg = case.integrals[method]
        name = f"{method}_{key}"
        t = leg["failure"]["time"] if key == "failure_state" else leg["last_accepted_time"]
        suffix = "failure" if key == "failure_state" else "15s" if case_id == "modulated_296" else "last_accepted"
        add(case_id, f"cached_{method}_{suffix}", case.cache[name], t, row["arrays"], name, row["future"])

    for case_id in anomalies:
        initial(case_id)
        cached(case_id, "fine", "last_state")
        cached(case_id, "fine", "failure_state")
    for case_id in anomalies:
        cached(case_id, "main", "failure_state")
    case_id, method = "modulated_268_stage2", "main"
    initial(case_id)
    job, ref = read_job(phase1, case_id, method)
    if job["status"] != "rhs_failure" or job["failure"]["evaluated"] is not True:
        raise ValueError("selected stage2 failure no longer names an evaluated RHS state")
    raw_path = PHASE1_RAW / "jobs" / case_id / f"{method}.npz"
    with np.load(io.BytesIO(run.read_ref(job["arrays"], raw_path)), allow_pickle=False) as archive:
        add(case_id, "phase1_main_failure", archive["failure_state"], job["failure"]["time"],
            job["arrays"], "failure_state", ref)
    initial("modulated_296")
    cached("modulated_296", "fine", "last_state")
    labels = [f"{point['case_id']}:{point['role']}" for point in points]
    if len(points) != 20 or labels != adoption["conditional_preparation"]["real_point_selection_order"]:
        raise ValueError("conditional point plan changed")
    return dict(rhs_version=VERSION, phase1_execution_sha256=PHASE1_SHA, prepared_sha256=phase1["prepared_sha256"],
                phase1_adoption_sha256=PHASE1_ADOPTION_SHA, phase1_scientist_sha256=adoption["scientist_sha256"],
                phase1_review_sha256=adoption["review_sha256"],
                protocol_sha256=phase1["protocol_sha256"], points=points), states


def verify_release(path, phase):
    expected = OUTPUT / f"{phase}_release.json"
    if path != expected:
        raise ValueError(f"{phase} release must be {expected.as_posix()}")
    raw = run.workspace_path(path).read_bytes()
    release = json.loads(raw)
    if (release.get("released") is not True or release["phase"] != phase or release["rhs_version"] != VERSION
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", release["action_id"])
            or release["phase1_execution_sha256"] != PHASE1_SHA
            or release["phase1_adoption_sha256"] != PHASE1_ADOPTION_SHA
            or release["protocol_sha256"] != run.PROTOCOL_SHA256 or release["runtime"] != run.runtime()):
        raise PermissionError(f"frozen root {phase} release is required")
    if set(release["source_sha256"]) != set(SOURCE_PATHS):
        raise ValueError("conditional source identity is incomplete")
    for name, digest in release["source_sha256"].items():
        run.checked_bytes(name, digest)
    qualification = json.loads(run.checked_bytes(OUTPUT / "artificial_qualification.json",
                                                  release["artificial_qualification_sha256"]))
    fixtures = qualification["fixtures"]
    if (qualification["rhs_version"] != VERSION or qualification["actual_rhs_calls"] != 8
            or set(fixtures) != {"all_closed", "partially_closed", "inactive_overflow", "normal_active"}
            or any(row["passed"] is not True for row in fixtures.values())
            or fixtures["inactive_overflow"]["status"] != "zero_term_unblocked"):
        raise ValueError("four artificial equivalence checks must pass before evaluation")
    return release, run.sha(raw), qualification


def check_points(path):
    release, release_sha, qualification = verify_release(path, "point_checks")
    phase1, models, cases, inventory, adoption = frozen_inputs()
    manifest, states = point_plan(phase1, cases, inventory, adoption)
    frozen = run.checked_bytes(OUTPUT / "point_plan.json", release["point_plan_sha256"])
    if run.json_bytes(manifest) != frozen:
        raise ValueError("real single-point identities changed")
    action = release["action_id"]
    out = OUTPUT / "point_checks"
    identity = dict(action_id=action, release_sha256=release_sha, rhs_version=VERSION,
                    point_plan_sha256=release["point_plan_sha256"], source_sha256=release["source_sha256"],
                    artificial_qualification_sha256=release["artificial_qualification_sha256"], pid=os.getpid())
    receipt = dict(identity, status="RUNNING", points={})
    prior = phase1["sample_rhs_calls"] + qualification["actual_rhs_calls"]
    budget = [prior]
    candidates = {mode: InactiveZeroSkip.from_functional(model) for mode, model in models.items()}
    run.claim(out / "started.json", identity)
    try:
        for point, state in zip(manifest["points"], states, strict=True):
            key = f"{point['index']:02d}"
            job = dict(point, action_id=f"{action}:{key}", phase_action_id=action)
            run.claim(out / "jobs" / f"{key}_started.json", job)
            start = budget[0]
            try:
                mode = inventory["cases"][point["case_id"]]["mode"]
                record, arrays = point_comparison(models[mode], candidates[mode], state, budget)
                record.update(job, actual_rhs_calls=budget[0] - start)
                record["arrays"] = run.save_arrays(PRIVATE / "point_checks" / f"{key}.npz", arrays)
            except BaseException as error:
                record = dict(job, status="interrupted", passed=False, error=repr(error), actual_rhs_calls=budget[0] - start)
                run.save_asset(out / "jobs" / f"{key}.json", run.json_bytes(record))
                raise
            receipt["points"][key] = run.save_asset(out / "jobs" / f"{key}.json", run.json_bytes(record))
        receipt["passed"] = all(json.loads(run.read_ref(ref, out / "jobs" / f"{key}.json"))["passed"]
                                for key, ref in receipt["points"].items())
        receipt["status"] = "COMPLETED"
    except BaseException as error:
        receipt.update(status="INTERRUPTED", passed=False, error=repr(error))
        raise
    finally:
        receipt.update(actual_rhs_calls=budget[0] - prior, cumulative_single_point_rhs_calls=budget[0])
        run.save_asset(out / "execution.json", run.json_bytes(receipt))
    print(f"{receipt['status']} {len(receipt['points'])} paired single-point checks, passed={receipt['passed']}")


def version_comparison(phase1, case_id, functional, case, jobs):
    comparisons = {}
    for method, record in jobs.items():
        if case.cache is not None and method in ("main", "fine"):
            original = case.cache[f"{method}_path"]
        else:
            old, _ = read_job(phase1, case_id, method)
            if old is None:
                comparisons[method] = None
                continue
            old_path = PHASE1_RAW / "jobs" / case_id / f"{method}.npz"
            with np.load(io.BytesIO(run.read_ref(old["arrays"], old_path)), allow_pickle=False) as archive:
                original = archive["grid_states"]
        new_path = PRIVATE / "phase1/jobs" / case_id / f"{method}.npz"
        with np.load(io.BytesIO(run.read_ref(record["arrays"], new_path)), allow_pickle=False) as archive:
            candidate = archive["grid_states"]
        if any(values[0].tobytes() != case.state.tobytes() for values in (original, candidate)):
            raise ValueError("trajectory versions do not share the frozen initial state")
        paths = dict(original=original, candidate=candidate)
        common, agreed = run.reference_prefix(functional.mass, paths)
        delta = run.differences(functional.mass, original[:common], candidate[:common])
        comparisons[method] = dict(common_end=(common - 1) / 100, agreement_end=(agreed - 1) / 100,
                                   agreed_on_all_common=common == agreed and run.agreed(delta), differences=delta)
    return comparisons


def run_trajectories(path):
    release, release_sha, qualification = verify_release(path, "trajectories")
    phase1, models, cases, _, _ = frozen_inputs()
    checks = json.loads(run.checked_bytes(OUTPUT / "point_checks/execution.json", release["point_execution_sha256"]))
    if (checks["status"] != "COMPLETED" or checks["passed"] is not True or checks["rhs_version"] != VERSION
            or len(checks["points"]) != 20 or checks["actual_rhs_calls"] != 40
            or checks["source_sha256"] != release["source_sha256"]
            or checks["point_plan_sha256"] != release["point_plan_sha256"]
            or checks["artificial_qualification_sha256"] != release["artificial_qualification_sha256"]
            or type(release["workers"]) is not int or release["workers"] < 1):
        raise ValueError("qualified real single-point checks with the same source identity are required")
    review = Path("reports") / run.RUN / "zero_skip_implementation_review.md"
    run.checked_bytes(review, release["implementation_review_sha256"])
    run.checked_bytes(OUTPUT / "point_plan.json", release["point_plan_sha256"])
    budget = [phase1["sample_rhs_calls"] + qualification["actual_rhs_calls"] + checks["actual_rhs_calls"]]
    initial_budget = budget[0]
    if initial_budget != checks["cumulative_single_point_rhs_calls"] or initial_budget + 84 > 500:
        raise ValueError("conditional sampling would exceed the registered work")
    reference_limit = sum(BRANCH_TARGETS.values()) * 240 * len(run.REFERENCES)
    fixed_limit = sum(12 * samples * round(.01 / step) for samples in BRANCH_TARGETS.values()
                      for step in run.FIXED_STEPS.values())
    if (reference_limit, fixed_limit) != (1614240, 403560):
        raise ValueError("conditional integration budgets changed")

    action, workers = release["action_id"], release["workers"]
    candidate = InactiveZeroSkip.from_functional(models["modulated"])
    identity = dict(action_id=action, release_sha256=release_sha, rhs_version=VERSION, pid=os.getpid())
    out = OUTPUT / "phase1"
    run.claim(out / "started.json", identity)
    receipt = dict(identity, status="RUNNING", runtime=release["runtime"], workers=workers, jobs={}, summaries={},
                   phase1_execution_sha256=PHASE1_SHA, point_execution_sha256=release["point_execution_sha256"],
                   implementation_review_sha256=release["implementation_review_sha256"],
                   source_sha256=release["source_sha256"], reference_rhs_limit=reference_limit,
                   fixed_rhs_limit=fixed_limit)
    started = time.perf_counter()
    jobs = {name: {} for name in BRANCH_TARGETS}
    persist = lambda: run.save_asset(out / "execution.json", run.json_bytes(receipt))
    try:
        persist()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for methods in (tuple(run.REFERENCES), tuple(run.FIXED_STEPS)):
                pending = [pool.submit(branch_worker, action, name, candidate, cases[name].state, samples, methods)
                           for name, samples in BRANCH_TARGETS.items()]
                for task in pending:
                    name, records = task.result()
                    for method, (record, ref) in records.items():
                        jobs[name][method] = record
                        receipt["jobs"][record["action_id"]] = ref
                    persist()
        for name, samples in BRANCH_TARGETS.items():
            candidate_case = run.FrozenCase(cases[name].state, None, None)
            with branch_paths():
                summary, arrays = run.summarize_case(candidate, candidate_case, samples, jobs[name], budget)
            summary.pop("inactive_zero_skip_candidate")
            summary["rhs_version"] = VERSION
            summary["against_original"] = version_comparison(phase1, name, candidate, cases[name], jobs[name])
            summary["arrays"] = run.save_arrays(PRIVATE / "phase1/summaries" / f"{name}.npz", arrays)
            receipt["summaries"][name] = run.save_asset(out / "summaries" / f"{name}.json", run.json_bytes(summary))
        legs = [record for case_jobs in jobs.values() for record in case_jobs.values()]
        if len(legs) != 28 or budget[0] - initial_budget > 84:
            raise RuntimeError("conditional work differs from the registered plan")
        receipt.update(status="COMPLETED", actual_integration_rhs_calls=sum(leg["actual_rhs_calls"] for leg in legs),
                       worker_cpu_seconds=sum(leg["worker_cpu_seconds"] for leg in legs),
                       worker_wall_seconds=sum(leg["worker_wall_seconds"] for leg in legs))
    except BaseException as error:
        receipt.update(status="INTERRUPTED", error=repr(error))
        raise
    finally:
        receipt.update(host_wall_seconds=time.perf_counter() - started, sample_rhs_calls=budget[0] - initial_budget,
                       cumulative_single_point_rhs_calls=budget[0])
        persist()
    print(f"Completed {len(receipt['jobs'])} conditional integrations. {(out / 'execution.json').as_posix()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare-pairs", action="store_true")
    actions.add_argument("--point-release", type=Path)
    actions.add_argument("--trajectory-release", type=Path)
    args = parser.parse_args()
    if args.prepare_pairs:
        run.runtime()
        phase1, _, cases, inventory, adoption = frozen_inputs()
        manifest, _ = point_plan(phase1, cases, inventory, adoption)
        path = OUTPUT / "point_plan.json"
        digest = run.claim(path, manifest)
        print(f"Verified {len(manifest['points'])} saved points. {path.as_posix()} {digest}")
    elif args.point_release is not None:
        check_points(args.point_release)
    else:
        run_trajectories(args.trajectory_release)


if __name__ == "__main__":
    main()
