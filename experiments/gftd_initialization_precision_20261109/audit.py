"""Audit completed precision-candidate receipts. This never imports a model or solver."""

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_initialization_precision_20261109"
RESULTS = Path("results") / RUN
CANDIDATE = RESULTS / "candidate"
PRIVATE = Path("artifacts") / RUN / "candidate"
PARENT_MANIFEST = "results/gftd_fit297_20261107/raw_artifacts_manifest.json"
INPUTS = "private_data/gftd_fit297_20261107/inputs.json"
HORIZONS = ("1", "2", "5", "8", "15")
BINDINGS = ("protocol_sha256", "candidate_adoption_sha256", "selection_sha256", "source_sha256", "asset_sha256")
GATES = dict(roundtrip_main_scaled_max=1e-7, roundtrip_fine_scaled_max=1e-7, history_main_fine_rms=1e-7,
             future_main_fine_rms=1e-5, future_main_fine_peak=1e-4, p_endpoint_error=0.)
COST_KEYS = ("worker_cpu_seconds", "worker_wall_seconds", "actual_rhs_calls", "actual_residual_calls",
             "integrations_attempted", "integrations_completed")
INIT_SUMMARY = ("status", "passed", "p0_exact", "stage", "stage1_gates", "final_gates", "history_mse",
                "active_bounds", "counts", "worker_cpu_seconds", "worker_wall_seconds")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None):
    path = Path(path)
    require(not path.is_absolute() and (ROOT / path).resolve().is_relative_to(ROOT), f"outside workspace: {path}")
    raw = (ROOT / path).read_bytes()
    require(expected is None or sha(raw) == expected, f"raw hash mismatch: {path}")
    return raw


def asset(ref, path):
    require(ref["path"] == path.as_posix(), f"unexpected asset path: {ref['path']}, expected {path}")
    raw = read(path, ref["sha256"])
    require(len(raw) == ref["bytes"], f"byte count mismatch: {path}")
    return raw


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def key(row):
    return f"{row['mode']}_{row['index']:03d}"


def execution(phase, expected=None):
    raw = read(CANDIDATE / phase / "execution.json", expected)
    run = json.loads(raw)
    require(run["status"] == "COMPLETED", f"{phase} execution is {run['status']}; completed receipt required")
    require(run["phase"] == phase and run["method"] == "fixed_dp8", f"unexpected execution phase or method: {phase}")
    release = json.loads(read(RESULTS / f"{phase}_release.json", run["release_sha256"]))
    require(release["released"] is True and all(run[name] == release[name] for name in ("phase", "action_id", "workers")),
            f"release identity differs: {phase}")
    require(run["bindings"] == {name: release[name] for name in BINDINGS}, f"execution bindings differ: {phase}")
    require(run["runtime"]["threads"] == dict(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            f"worker thread configuration differs: {phase}")
    started = json.loads(read(CANDIDATE / phase / "started.json"))
    require(started == {name: run[name] for name in ("action_id", "release_sha256")}, f"claim identity differs: {phase}")
    for binding in ("source_sha256", "source_closure_sha256", "asset_sha256", "review_sha256", "qualification_sha256"):
        for path, digest in release.get(binding, {}).items():
            read(path, digest)
    read(Path("campaigns") / RUN / "protocol.md", release["protocol_sha256"])
    for mode, ref in run["models"].items():
        model = json.loads(asset(ref, PRIVATE / "pilot" / f"{mode}_model.json"))
        config = model["inference_config"]
        require(sha(json_bytes(config)) == ref["config_sha256"] and model["fit_mode"] == mode
                and config["method"] == run["method"] and config["protocol_sha256"] == release["protocol_sha256"],
                f"model inference binding differs: {mode}")
    require(set(run["models"]) == {"static", "modulated"}, "candidate models are incomplete")
    return run, release, sha(raw)


def frozen_inputs(release):
    selection = json.loads(read(RESULTS / "selection.json", release["selection_sha256"]))["selection"]
    pilot_keys = {key(row) for row in selection}
    require(len(selection) == 49 and len(pilot_keys) == 44, "pilot selection changed")
    adoption = json.loads(read(RESULTS / "candidate_adoption.json", release["candidate_adoption_sha256"]))
    integrity = json.loads(read(RESULTS / "fixed_state_integrity.json", adoption["fixed_state_integrity_sha256"]))
    fixed = json.loads(read(RESULTS / "fixed_state/execution.json", integrity["fixed_state_execution_sha256"]))
    require(fixed["status"] == "COMPLETED" and fixed["decision"]["eligible_for_initializer_pilot"]
            and fixed["selection_sha256"] == release["selection_sha256"], "fixed-state authorization differs")
    fixed_release = json.loads(read(RESULTS / "fixed_state_release.json", fixed["release_sha256"]))
    previous = json.loads(read(RESULTS / "execution.json", fixed_release["diagnostic_execution_sha256"]))
    rows = json.loads(asset(previous["rows"], RESULTS / "rows.json"))
    require(previous["status"] == "COMPLETED" and len(rows) == len({key(row) for row in rows}) == 594,
            "original 594-row denominator differs")
    manifest = json.loads(read(PARENT_MANIFEST, release["asset_sha256"][PARENT_MANIFEST]))
    hashes = {item["path"]: item["sha256"] for item in manifest["files"]}
    packets = json.loads(read(INPUTS, hashes[INPUTS]))
    anchors = {event: np.asarray(packet["p0"], dtype=np.float64).tobytes()
               for event, packet in zip(packets["receipt"]["legal_ids"], packets["packets"], strict=True)}
    require(len(anchors) == 297 and {row["event_id"] for row in rows} == set(anchors), "frozen event identities differ")
    return rows, pilot_keys, anchors


def task(run, kind, row, ref):
    name, phase = key(row), run["phase"]
    result = json.loads(asset(ref, CANDIDATE / phase / kind / f"{name}.json"))
    require(all(result[field] == row[field] for field in ("mode", "index", "event_id"))
            and all(result[field] == run[field] for field in ("phase", "action_id")), f"{kind} identity differs: {name}")
    for field in ("worker_wall_seconds", "worker_cpu_seconds"):
        require(math.isfinite(result[field]) and result[field] >= 0, f"invalid {field}: {name}")
    return result


def initialization(run, row, ref, anchor):
    name, phase = key(row), run["phase"]
    result = task(run, "initialize", row, ref)
    diag = json.loads(gzip.decompress(asset(result["diagnostics"], PRIVATE / phase / "diagnostics" / f"{name}.json.gz")))
    counts = result["counts"]
    require(result["old_status"] == row["status"] and result["old_gates"] == row["old_gates"]
            and result["label"] == row["label"] and diag["method"] == run["method"]
            and diag["counts"] == counts and diag["status"] == result["status"]
            and result["passed"] == (result["status"] == "completed"), f"initialization summary differs: {name}")
    stages = [diag.get(stage) for stage in ("stage1", "stage2")]
    calls = [stage["actual_residual_calls"] if stage is not None else 0 for stage in stages]
    require(0 <= calls[0] <= 600 and 0 <= calls[1] <= 1200 and sum(calls) == counts["actual_residual_calls"]
            and 0 <= 12 * counts["fixed_dp8_steps_completed"] <= counts["actual_rhs_calls"]
            <= 12 * counts["fixed_dp8_steps_attempted"] and counts["rk4_steps_attempted"] == 0,
            f"initializer accounting differs: {name}")
    health = {f"stage{i}": None if stage is None else {field: stage[field] for field in
              ("status", "actual_residual_calls", "projected_gradient_kkt", "optimization_sufficiency")}
              for i, stage in enumerate(stages, 1)}
    if result["passed"]:
        gate = result["final_gates"]
        require(gate["passed"] is True and result["p0_exact"] is True
                and all(0 <= gate["values"][field] <= limit for field, limit in GATES.items()),
                f"returned state gate differs: {name}")
        with np.load(io.BytesIO(asset(result["state"], PRIVATE / phase / "states" / f"{name}.npz")), allow_pickle=False) as arrays:
            require(set(arrays.files) == {"state"}, f"state keys differ: {name}")
            state = arrays["state"]
        require(state.dtype == np.float64 and state.shape == (50,) and np.isfinite(state).all()
                and sha(state.tobytes()) == result["state_sha256"] and state[20:30].tobytes() == anchor
                and np.asarray(diag[diag["returned_point"]]["best"]["state"], dtype=np.float64).tobytes() == state.tobytes(),
                f"state hash, selected point or frozen p0 differs: {name}")
    else:
        require(result["state"] is None and result["p0_exact"] is None, f"failed initialization stores an output: {name}")
    return result, health


def continuation(run, row, ref, init, anomalies):
    name = key(row)
    result = task(run, "future", row, ref)
    legs, reference = result["integrals"], result["single_radau_reference"]
    expected = {"main", "fine"} | ({"radau"} if reference and reference["attempted"] else set())
    require(set(legs) == expected and set(result["differences"]) == set(HORIZONS), f"continuation records differ: {name}")
    with np.load(io.BytesIO(asset(result["arrays"], PRIVATE / "full/trajectories" / f"{name}.npz")), allow_pickle=False) as arrays:
        state = arrays["initial_state"]
        require(sha(state.tobytes()) == init["state_sha256"], f"continuation starts from a different state: {name}")
        for side, leg in legs.items():
            path = arrays[f"{side}_path"]
            require(path.dtype == np.float64 and path.shape == (101 if side == "radau" else 1501, 50)
                    and leg["start_state_sha256"] == init["state_sha256"] and path[0].tobytes() == state.tobytes(),
                    f"{side} trajectory origin or shape differs: {name}")
        fine_one = (legs["fine"]["last_accepted_time"] + 1e-14 >= 1.
                    and np.isfinite(arrays["fine_path"][:101]).all())
    added = init["old_status"] != "completed"
    require((reference is not None) == added and (not added or reference["attempted"] == fine_one),
            f"new qualification reference attempt differs: {name}")
    if added and not fine_one:
        anomalies.append(dict(task=name, reason="newly qualified state has incomplete fine DP8 before 1 s",
                              last_accepted_time=legs["fine"]["last_accepted_time"]))
    require(result["counts"] == dict(actual_rhs_calls=sum(leg["actual_rhs_calls"] for leg in legs.values()),
                                     integrations_attempted=len(legs), integrations_completed=sum(leg["completed"] for leg in legs.values())),
            f"continuation accounting differs: {name}")
    return result


def costs(results):
    results = list(results)
    return {metric: (math.fsum if metric.endswith("seconds") else sum)(
        row.get(metric, row["counts"].get(metric, 0)) for row in results) for metric in COST_KEYS}


def check_costs(run, results):
    total = costs(results)
    require(set(total) == set(run["new_costs"]) and all(
        math.isclose(value, run["new_costs"][metric], rel_tol=1e-12, abs_tol=1e-9) if metric.endswith("seconds")
        else value == run["new_costs"][metric] for metric, value in total.items()), "new worker cost sum differs")
    return total


def stats(values):
    values = list(values)
    return dict(n=len(values), min=min(values), median=median(values), max=max(values)) if values else dict(n=0, min=None, median=None, max=None)


def group_digest(names, init, health, future):
    initial = [init[name] for name in names if name in init]
    continuations = [future[name] for name in names if name in future]
    paired = [row["history_mse"] for row in initial if row["history_mse"]["change"] is not None]
    bounds = [row["active_bounds"] for row in initial if row["active_bounds"] is not None]
    stage_summary = {}
    for stage, limit in (("stage1", 600), ("stage2", 1200)):
        stages = [health[name][stage] for name in names if name in health and health[name][stage] is not None]
        stage_summary[stage] = dict(status_counts=dict(Counter(item["status"] for item in stages)),
                                    actual_residual_calls=sum(item["actual_residual_calls"] for item in stages),
                                    hit_residual_limit=sum(item["actual_residual_calls"] == limit for item in stages),
                                    projected_gradient_kkt=stats(item["projected_gradient_kkt"] for item in stages
                                                                 if item["projected_gradient_kkt"] is not None),
                                    optimization_sufficiency=dict(Counter(item["optimization_sufficiency"] for item in stages)))
    differences = {}
    for time in HORIZONS:
        comparable = [row["differences"][time] for row in continuations if row["differences"][time] is not None]
        differences[time] = {view: {metric: stats(item[view][metric] for item in comparable) for metric in ("rms", "peak")}
                             for view in ("observed", "state")}
    references = [row["single_radau_reference"] for row in continuations if row["single_radau_reference"] is not None]
    return dict(denominator=len(names), initialization_attempted=len(initial),
                old_qualified=sum(row["old_status"] == "completed" for row in initial),
                candidate_qualified=sum(row["passed"] for row in initial),
                initialization_status=dict(Counter(row["status"] for row in initial)), stage_health=stage_summary,
                selected_stage=dict(Counter(row["stage"] for row in initial if row["passed"])),
                history_mse_paired={field: stats(item[field] for item in paired) for field in ("old", "candidate", "change")},
                selected_active_bounds=dict(with_selected_point=len(bounds), a_fixed=sum(item["a_fixed"] for item in bounds),
                                            **{field: dict(events=sum(item[field] > 0 for item in bounds),
                                                           coordinates=sum(item[field] for item in bounds)) for field in ("a", "r")}),
                continuation_15s=dict(attempted=len(continuations), **{
                    f"{side}_completed": sum(row["integrals"][side]["completed"] for row in continuations)
                    for side in ("main", "fine")}, pair_completed=sum(all(row["integrals"][side]["completed"]
                    for side in ("main", "fine")) for row in continuations)),
                main_fine_differences=differences,
                newly_qualified_radau=dict(newly_qualified=len(references), attempted=sum(row["attempted"] for row in references),
                                           confirmed=sum(row["status"] == "confirmed" for row in references)))


def audit(phase):
    run, release, execution_sha = execution(phase)
    old_rows, pilot_keys, anchors = frozen_inputs(release)
    rows = {key(row): row for row in old_rows}
    init, health, future, anomalies = {}, {}, {}, []
    new_refs = run["tasks"]["initialize"]
    expected = pilot_keys if phase == "pilot" else set(rows) - pilot_keys
    require(len(new_refs) == (44 if phase == "pilot" else 550) and set(new_refs) == expected
            and run["planned_initializations"] == [name for name in rows if name in expected], "new initialization plan differs")
    require({file.stem for file in (ROOT / CANDIDATE / phase / "claims/initialize").glob("*.json")} == expected,
            "new initialization claims differ")
    if phase == "full":
        pilot, _, _ = execution("pilot", release["pilot_execution_sha256"])
        reused = run["reused_pilot"]
        require(reused == pilot["tasks"]["initialize"] and set(reused) == pilot_keys
                and run["bindings"] == pilot["bindings"] and run["models"] == pilot["models"]
                and run["runtime"] == pilot["runtime"] and run["action_id"] != pilot["action_id"]
                and pilot["decision"]["eligible_for_full"] is True, "full does not reuse the eligible pilot verbatim")
        for name, ref in reused.items():
            init[name], health[name] = initialization(pilot, rows[name], ref, anchors[rows[name]["event_id"]])
        check_costs(pilot, init.values())
    else:
        require(not run["reused_pilot"], "pilot unexpectedly reuses another phase")
    for name, ref in new_refs.items():
        init[name], health[name] = initialization(run, rows[name], ref, anchors[rows[name]["event_id"]])
    qualified = [name for name in rows if name in init and init[name]["passed"]]
    future_refs = run["tasks"]["future"]
    require((phase == "pilot" and not future_refs) or (phase == "full" and set(future_refs) == set(qualified)
            and run["planned_futures"] == qualified), "future plans differ from qualified initializations")
    for name, ref in future_refs.items():
        future[name] = continuation(run, rows[name], ref, init[name], anomalies)
    totals = check_costs(run, [*(init[name] for name in new_refs), *future.values()])
    table = json.loads(asset(run["rows"], CANDIDATE / phase / "rows.json"))
    for row, old in zip(table, old_rows, strict=True):
        name = key(old)
        require({k: v for k, v in row.items() if k not in ("candidate", "continuation")} ==
                {k: v for k, v in old.items() if k not in ("candidate", "continuation")}, f"old table row differs: {name}")
        candidate = dict({field: init[name][field] for field in INIT_SUMMARY},
                         evidence=new_refs.get(name, run["reused_pilot"].get(name))) if name in init else None
        require(row["candidate"] == candidate and (row["continuation"] is not None) == (name in future), f"table task differs: {name}")
        if name in future:
            require(row["continuation"]["evidence"] == future_refs[name] and all(
                row["continuation"][field] == future[name][field] for field in ("differences", "single_radau_reference", "counts")),
                f"table continuation differs: {name}")
    controls = sorted(name for name in init if rows[name]["status"] == "completed")
    regressions = sorted(name for name in controls if not init[name]["passed"])
    added = sorted(set(qualified) - set(controls))
    decision = run["decision"]
    eligible = bool(added and len(controls) == 5 and not regressions) if phase == "pilot" else None
    require((len(controls), regressions, added, eligible) == tuple(decision[field] for field in
            ("controls", "regressions", "added", "eligible_for_full")), "candidate decision differs")
    require(len(controls) == (5 if phase == "pilot" else 47), "original success denominator differs")
    denominator = {mode: dict(Counter(row["label"] for row in old_rows if row["mode"] == mode)) for mode in ("static", "modulated")}
    require(all(group == {"stable": 267, "unstable": 30} for group in denominator.values()), "original group denominators differ")
    summary = dict(status="TECHNICAL_ANOMALY" if anomalies else "VERIFIED", phase=phase, execution_sha256=execution_sha,
                   release_sha256=run["release_sha256"], source_maps_verified=["source_sha256", "source_closure_sha256"],
                   original_denominators=denominator, new_initializations=len(new_refs), reused_pilot=len(run["reused_pilot"]),
                   state_hash_and_p0_verified=len(qualified), future_tasks=len(future), controls=controls,
                   regressions=regressions, added=added, eligible_for_full=eligible, new_costs=totals,
                   all_initialization_costs_including_reused=costs(init.values()), host_wall_seconds=run["host_wall_seconds"],
                   technical_anomalies=anomalies)
    (ROOT / RESULTS / f"{phase}_integrity.json").write_bytes(json_bytes(summary))
    if phase == "full":
        groups = {mode: {label: group_digest([name for name, row in rows.items() if row["mode"] == mode and row["label"] == label],
                                             init, health, future) for label in ("stable", "unstable")} for mode in denominator}
        digest = dict(execution_sha256=execution_sha, method=run["method"], groups=groups, new_costs=totals,
                      host_wall_seconds=run["host_wall_seconds"], technical_anomalies=anomalies)
        (ROOT / RESULTS / "candidate_digest.json").write_bytes(json_bytes(digest))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "full"))
    result = audit(parser.parse_args().phase)
    print(json_bytes(result).decode(), end="")
    raise SystemExit(2 if result["technical_anomalies"] else 0)
