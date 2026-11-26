"""Score only valid frozen baseline endpoints in one val pass."""

import argparse
import os
from collections import defaultdict
from pathlib import Path
import sys
from time import perf_counter, process_time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch

from experiments.osgkd_v2_fast_zero_sum_20261124 import scoring
from experiments.osgkd_v2_p2a_equivariant_graph_20261124 import run as graph_run
from experiments.osgkd_v2_sprint2a_20261125 import resume_host as io, run as matched
from experiments.osgkd_v2_sprint2a_20261125.score import endpoint_spectrum
from experiments.osgkd_v2_sprint2b_baselines_20261126 import fit
from experiments.osgkd_v2_sprint2b_baselines_20261126.probe import PARENT, PARENT_SHA256
from kiki.data.osgkd_train_val import load_val
from kiki.models.osgkd_baselines import ClosedPassiveKuramoto, GraphNeuralODE
from kiki.models.osgkd_fitting import IntegrationWork
from kiki.models.osgkd_graph import OSGKDGraph
from kiki.models.osgkd_matched_flat import OSGKDMatchedFlat


PARENTS = {"graph_24111": OSGKDGraph, "flat_24111": OSGKDMatchedFlat}
BASELINES = {"closed_passive_24111": ClosedPassiveKuramoto, "neural_ode_24111": GraphNeuralODE}


class InferenceClock:
    """Measure the three original inference operations without instrumenting RHS calls."""

    def __init__(self, model):
        self.raw_initialize = model.initialize
        self.raw_integrate = scoring.integrate
        self.initialization = 0.
        self.integration = {.005: 0., .0025: 0.}
        self.events = 0
        self.outputs = {.005: 0, .0025: 0}

    def initialize(self, theta, omega):
        started = perf_counter()
        value = self.raw_initialize(theta, omega)
        self.initialization += perf_counter() - started
        self.events += len(theta)
        return value

    def integrate(self, model, state, steps, *, internal_step=.005, work=None):
        started = perf_counter()
        value = self.raw_integrate(model, state, steps, internal_step=internal_step, work=work)
        self.integration[internal_step] += perf_counter() - started
        self.outputs[internal_step] += len(state) * steps
        return value

    def summary(self, expected_outputs):
        io.require(self.events == 107 and all(value == expected_outputs for value in self.outputs.values()),
                   "timing workload differs from the official val inference")
        return {"initialize_s": self.initialization, "rk4_0_005_s": self.integration[.005],
                "rk4_0_0025_s": self.integration[.0025],
                "inference_wall_s": self.initialization + sum(self.integration.values()),
                "event_count": self.events, "event_output_steps": expected_outputs}


def rollout_groups(events):
    groups = defaultdict(list)
    for event in events:
        if event["history"] is not None:
            count = max(800, min(1500, len(event["t"]) - 1)) if event["label"] == "unstable" else 800
            groups[count].append(event)
    return groups


@torch.no_grad()
def warm_up(model):
    theta = model.theta_ref.expand(1, 11, -1)
    initial = model.initialize(theta, torch.zeros_like(theta))
    for step in (.005, .0025):
        scoring.integrate(model, initial, 1, internal_step=step)


@torch.no_grad()
def time_archived_endpoint(model, groups):
    """Use the same val motion packets and integrator without generating old scores."""
    work = IntegrationWork()
    clock = InferenceClock(model)
    for count, events in groups.items():
        for start in range(0, len(events), 16):
            batch = events[start:start + 16]
            theta, omega = [torch.as_tensor(np.stack([event["history"][key] for event in batch]),
                                           dtype=torch.float64) for key in ("theta", "omega")]
            initial = clock.initialize(theta, omega)
            for step in (.005, .0025):
                clock.integrate(model, initial, count, internal_step=step, work=work)
    expected_outputs = sum(count * len(events) for count, events in groups.items())
    io.require(work.rhs["forward"] == 24 * expected_outputs, "archived timing work differs")
    return clock.summary(expected_outputs)


def spectrum_effective_passive(model, embedded, detail):
    """P0 methods on the invariant 20-state physical restriction."""
    n = model.n
    dimension = 2 * n
    jacobian = np.asarray(detail["normalized_jacobian"])
    core = jacobian[:dimension, :dimension]
    io.require(np.count_nonzero(jacobian[:dimension, dimension:]) == 0
               and np.count_nonzero(jacobian[dimension:, :dimension]) == 0
               and np.array_equal(jacobian[dimension:, dimension:], -np.eye(3 * n)),
               "passive checkpoint does not have the expected invariant physical state")
    q = np.asarray(detail["basis"])[:dimension, :dimension - 1]
    quotient = q.T @ core @ q
    full_poles = np.linalg.eigvals(core)
    poles = np.linalg.eigvals(quotient)
    maximum = float(poles.real.max())

    x0 = torch.as_tensor(detail["reference"], dtype=torch.float64)
    scale = torch.as_tensor(detail["scales"], dtype=torch.float64)
    full_eye = torch.eye(5 * n, dtype=torch.float64)

    @torch.no_grad()
    def value(y):
        return (model.rhs(None, x0 + scale * y) / scale)[:dimension].numpy()

    finite_difference = []
    for step in (1e-5, 5e-6):
        fd = np.column_stack([(value(step * axis) - value(-step * axis)) / (2 * step)
                              for axis in full_eye[:dimension]])
        finite_difference.append(
            {"h": step, "max_real": float(np.linalg.eigvals(q.T @ fd @ q).real.max()),
             "max_real_full": float(np.linalg.eigvals(fd).real.max()),
             "relative_frobenius": float(np.linalg.norm(fd - core) / max(np.linalg.norm(core), 1e-300))})
    error = max(abs(row["max_real"] - maximum) for row in finite_difference)
    threshold = max(1e-7, error)
    qualified = bool(embedded["qualified"] and all(row["relative_frobenius"] <= 1e-6
                                                  for row in finite_difference))
    detail["effective_passive"] = {"jacobian": core.tolist(), "quotient_jacobian": quotient.tolist(),
                                   "full_poles": [[float(x.real), float(x.imag)] for x in full_poles],
                                   "quotient_poles": [[float(x.real), float(x.imag)] for x in poles]}
    return {"qualified": qualified, "state_dimension": dimension, "quotient_dimension": dimension - 1,
            "neutral_common_angle_mode_count": int(np.count_nonzero(abs(full_poles) <= threshold)),
            "embedded_dummy_modes_at_minus_one": 3 * n, "max_real_full": float(full_poles.real.max()),
            "max_real": maximum, "positive_modes": int(np.count_nonzero(poles.real > threshold)),
            "max_real_fd_difference": error, "finite_difference": finite_difference}


def from_endpoint(checkpoint, kind, release):
    saved = torch.load(io.verified(checkpoint), map_location="cpu", weights_only=True)
    io.require(saved["attempt"] == 2048 and saved["release"] == release, "endpoint authority differs")
    return kind.from_checkpoint(saved["model"]).eval().requires_grad_(False)


def metrics_row(summary, timing, scored, spectrum):
    auto = scored["autonomous"]
    for key, eligible, raw in (("stable_false_alarm_8s", 95, 95), ("unstable_recall_8s", 12, 22)):
        row = auto[key]
        io.require((row["input_eligible_denominator"], row["raw_denominator"]) == (eligible, raw),
                   "official val denominators differ")
    metrics = scored["summary"]["splits"]["val"]
    stable = {str(h): (metric["official"] or {}).get("motion_risk_v1")
              for h in (1, 2, 5, 8) for metric in (metrics[f"stable_{h}s"],)}
    actual = metrics["unstable_actual"]
    return {"status": "scored", "seed": 24111, "counts": {
                key: summary[key] for key in ("encoder_parameters", "effective_encoder_parameters",
                                               "ode_parameters", "active_parameters",
                                               "embedded_state_dimension", "effective_state_dimension")},
            "checkpoint_ref": summary["checkpoints"]["2048"],
            "stable_motion_risk": stable, "unstable_actual_risk_covered":
                (actual["covered_only_diagnostic"] or {}).get("motion_risk_v1"),
            "unstable_actual_risk_coverage": actual["covered"], "timing": timing,
            "score": scored, "spectrum": spectrum}


def score_endpoint(job, model, events, groups, constants, directory, info):
    clock = InferenceClock(model)
    with patch.object(model, "initialize", clock.initialize), patch.object(scoring, "integrate", clock.integrate):
        scored = graph_run.score(model, events, constants, "val", directory, {"work": {}})
    timing = clock.summary(sum(count * len(rows) for count, rows in groups.items()))
    spectrum = endpoint_spectrum(model, directory)
    if job == "closed_passive_24111" and spectrum["qualified"]:
        ref = spectrum["detail_ref"]
        detail = io.read(io.verified(ref))
        spectrum["effective_passive"] = spectrum_effective_passive(model, spectrum, detail)
        spectrum["effective_passive"]["detail_ref"] = graph_run.baseline.atomic_write(
            directory / "effective_spectrum.json", detail["effective_passive"])
    scored["data_identity_verified"] = True
    return metrics_row(info, timing, scored, spectrum)


def validate(path):
    release = io.read(path)
    io.require(release["run_id"] == fit.RUN and release["released"] is True,
               "official score release required")
    for source, digest in release["source_hashes"].items():
        io.require(io.identity(ROOT / source)["sha256"] == digest, "scoring source differs: " + source)
    io.require(io.identity(PARENT) == release["parent_evaluation"] and
               release["parent_evaluation"]["sha256"] == PARENT_SHA256, "parent evaluation differs")
    plan_path = io.verified(release["fit_release"])
    fit_release, inputs = fit.validate(plan_path, fit.ORDER[0], False)
    host = io.read(io.verified(release["training_summary"]))
    io.require(host["state"] == "completed" and host["native_exit_code"] == 0
               and set(host["jobs"]) == set(BASELINES), "both fit outcomes must be sealed")
    records = {}
    for job, record in host["jobs"].items():
        ref = record["summary"]
        result = io.read(io.verified(ref))
        io.require(ref["path"] == f"results/{fit.RUN}/{job}/summary.json"
                   and result["job_id"] == job and result["release"] == fit_release
                   and result["release_ref"] == release["fit_release"]
                   and record["native_exit_code"] == 0, "training outcome authority differs")
        metadata = result["result"]
        if metadata["valid_endpoint"]:
            io.require(result["status"] == "completed" and metadata["attempts"] == 2048
                       and metadata["accepted"] >= 2033, "invalid fixed endpoint")
            io.verified(metadata["checkpoints"]["2048"])
        else:
            io.require(result["status"] == "scientific_stopped" and metadata["stop_reason"] is not None,
                       "invalid stopped outcome")
        records[job] = result
    return release, inputs, records


def evaluate(path):
    release, inputs, records = validate(path)
    public = ROOT / "results" / fit.RUN / "evaluate01"
    local = ROOT / "artifacts" / fit.RUN / "evaluate01"
    io.require(not public.exists() and not local.exists(), "the one official val evaluation was already claimed")
    public.mkdir(parents=True)
    local.mkdir(parents=True)
    started, cpu = perf_counter(), process_time()
    reference = io.identity(path)
    io.write(local / "started.json", {"release_ref": reference, "pid": os.getpid(),
                                       "started_utc": io.utc()})
    try:
        torch.set_num_threads(1)
        parent = io.read(PARENT)
        raw, constants = matched.load_train(inputs)
        del raw
        events = load_val(constants)
        data_identity = matched.baseline.data_identity(events)
        io.require(data_identity == parent["result"]["val_identity"], "original val digest differs")
        groups = rollout_groups(events)
        parental = {}
        for job, kind in PARENTS.items():
            archived = parent["result"]["val"][job]
            old_fit = io.read(io.verified(parent["release"]["fit_refs"][job]))
            model = from_endpoint(archived["checkpoint_ref"], kind, old_fit["release"])
            warm_up(model)
            parental[job] = time_archived_endpoint(model, groups)
        baseline = {}
        for job, kind in BASELINES.items():
            receipt = records[job]
            info = receipt["result"]
            if not info["valid_endpoint"]:
                baseline[job] = {"status": receipt["status"], "reason": info["stop_reason"],
                                 "actual_endpoint": info["actual_endpoint"], "timing": None,
                                 "score": None, "spectrum": None}
                continue
            model = from_endpoint(info["checkpoints"]["2048"], kind, receipt["release"])
            warm_up(model)
            baseline[job] = score_endpoint(job, model, events, groups, constants, local / job, info)
            io.require(baseline[job]["score"]["data_identity"] == data_identity, "scored val identity differs")
        summary = {"run_id": fit.RUN, "status": "completed", "job_id": "evaluate01",
                   "release_ref": reference, "parent_evaluation": release["parent_evaluation"],
                   "training_summary": release["training_summary"],
                   "parent_timing_only": parental, "val": baseline, "val_identity": data_identity,
                   "cost": {"wall_s": perf_counter() - started, "cpu_s": process_time() - cpu,
                            "peak_working_set_bytes": matched.peak_memory_bytes()},
                   "test_trajectory_access": False, "new_fom": 0, "completed_utc": io.utc()}
        result = matched.baseline.atomic_write(public / "summary.json", summary, public=True)
        io.write(local / "completed.json", {"status": "completed", "summary_ref": result})
        return 0
    except BaseException as error:
        io.write(local / "failed.json", {"status": "failed", "error": repr(error), "release_ref": reference})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    sys.exit(evaluate(parser.parse_args().release.resolve()))
