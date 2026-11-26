"""Audit saved Sprint 2A scores and measure isolated Layer 1 train steps."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.osgkd_v2_sprint2a_20261125 import resume_host

RUN = "osgkd_v2_sprint2b_baselines_20261126"
PARENT = ROOT / "results/osgkd_v2_sprint2a_20261125/evaluate01/summary.json"
PARENT_SHA256 = "7645c8b006b9dd6e71838e7e875da69117ed3df9011938d36f46cf126c6b3fbb"
SIZES = (8, 16, 32)
BATCHES = (1, 2, 4, 8)
FUTURE_POINTS = 20


def audit_false_alarms():
    from kiki.evaluation.osgkd import aggregate_metrics
    from kiki.evaluation.osgkd_autonomous import autonomous_outcome

    source = resume_host.identity(PARENT)
    if source["sha256"] != PARENT_SHA256:
        raise ValueError("Sprint 2A evaluation does not match the frozen archive")
    official = resume_host.read(PARENT)["result"]["val"]
    results = {}
    for name in ("graph_24111", "graph_24131"):
        fit = official[name]
        ref = fit["score"]["rows_ref"]
        rows = resume_host.read(resume_host.verified(ref))
        stable = [row for row in rows if row["label"] == "stable"]
        false_alarms = [row for row in stable if autonomous_outcome(row) == "autonomous_trigger"]
        if len(rows) != 117 or len(stable) != 95 or len(false_alarms) != 1:
            raise ValueError("original event count or stable false alarm count differs")
        selected = false_alarms[0]
        cohort = [row for row in stable if row["event_id"] != selected["event_id"]]
        official_risk = aggregate_metrics(stable, "8")["official"]["motion_risk_v1"]
        sensitivity = aggregate_metrics(cohort, "8")["official"]["motion_risk_v1"]
        if not math.isclose(official_risk, fit["stable_motion_risk"]["8"], rel_tol=1e-13):
            raise ValueError("aggregate is not identical to formal risk")
        classification = selected["classification_8s"]
        predicted = classification["predicted_time_record_s"]
        results[name] = {
            "rows_ref": ref, "event_id": selected["event_id"],
            "trigger_record_s": predicted, "trigger_after_initialization_s": predicted - .1,
            "stable_8s_official_n": len(stable), "stable_8s_sensitivity_n": len(cohort),
            "excluded_event_motion_error": selected["scopes"]["8"]["metrics"]["mean_motion_error"],
            "stable_8s_official_risk": official_risk, "stable_8s_sensitivity_risk": sensitivity,
        }
    return {"summary_ref": source, "models": results, "reintegrated_events": 0,
            "same_false_alarm": len({item["event_id"] for item in results.values()}) == 1}


def dry_step(n, batch_size):
    import torch
    from experiments.osgkd_v2_sprint2a_20261125 import run as matched_run, train
    from kiki.data import osgkd_synthetic as synthetic
    from kiki.models.osgkd_fitting import IntegrationWork, tensor_events

    torch.set_num_threads(1)
    system = synthetic.make_system(n=n, seed=31, condition="open")
    events = [synthetic.generate_event(system, seed=1200 + index, duration=.30,
                                       noise_std=.01)[0] for index in range(batch_size)]
    constants = synthetic.model_constants(synthetic.motion_packet(events[0]))
    model = train.new_model(constants, "graph", 24111)
    batch = tensor_events(events, "cpu", max_steps=FUTURE_POINTS)
    if model.edge_index.shape != (2, n * (n - 1) // 2):
        raise ValueError("dry step must allocate the full candidate graph")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    memory_before = resume_host.memory()
    peak_before = matched_run.peak_memory_bytes()
    work = IntegrationWork()
    started = perf_counter()
    loss = train.objective(train.components(model, batch, constants, work, .005, FUTURE_POINTS))
    loss.backward()
    wall_s = perf_counter() - started
    peak_after = matched_run.peak_memory_bytes()
    memory_after = resume_host.memory()
    checks = [torch.equal(parameter, original) for parameter, original
              in zip(model.parameters(), before, strict=True)]
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if (not math.isfinite(float(loss.detach())) or not all(checks) or not gradients
            or not all(torch.isfinite(gradient).all() for gradient in gradients)):
        raise ArithmeticError("dry step must be finite and must preserve every parameter")
    if work.rhs["forward"] != batch_size * FUTURE_POINTS * 8 or not peak_after:
        raise ArithmeticError("dry step work or memory accounting differs")
    return {
        "n": n, "candidate_edges": model.edge_index.shape[1], "batch": batch_size,
        "wall_s": wall_s, "peak_working_set_bytes": peak_after,
        "peak_before_step_bytes": peak_before, "new_process_peak_bytes": peak_after - peak_before,
        "system_before": memory_before, "system_after": memory_after,
        "loss": float(loss.detach()), "work": work.snapshot(),
        "parameter_updates": 0, "parameters_identical": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, type=int)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(dry_step(*args.worker), allow_nan=False))
        return
    path = ROOT / "results" / RUN / "phase1_summary.json"
    if path.exists():
        raise ValueError("phase one evidence already exists")
    started = perf_counter()
    audit = audit_false_alarms()
    steps = []
    for n in SIZES:
        for batch in BATCHES:
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "-B", __file__, "--worker", str(n), str(batch)],
                check=True, capture_output=True, text=True, encoding="utf-8", timeout=90,
            )
            steps.append(json.loads(completed.stdout))
    result = {
        "run_id": RUN, "status": "completed", "official_updates": 0, "new_fom": 0,
        "raw_test_trajectory_access": False, "source": resume_host.identity(Path(__file__)),
        "false_alarm_audit": audit,
        "layer1_dry_run": {"grid_s": .005, "truth_grid_s": .0025, "observed_history_s": .1,
                           "future_horizon_s": .20, "device": "cpu", "dtype": "float64",
                           "torch_threads": 1, "one_fresh_process_per_case": True, "cases": steps},
        "wall_s": perf_counter() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    resume_host.write(path, result)
    print(json.dumps({"summary": resume_host.identity(path), "wall_s": result["wall_s"],
                      "false_alarms": audit["models"], "same_event": audit["same_false_alarm"],
                      "dry_steps": [{key: step[key] for key in ("n", "candidate_edges", "batch",
                                    "wall_s", "peak_working_set_bytes", "new_process_peak_bytes")}
                                    for step in steps]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
