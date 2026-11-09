"""Independent frozen-array audit; no kiki imports or model evaluations."""

from itertools import combinations
from pathlib import Path
import hashlib
import io
import json
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_storage_feasibility_20261109"
OUT = ROOT / "results" / RUN
METHODS = ("radau", "radau_strict", "dop853_adaptive")
STAGES = ("stage1", "stage2")
BLOCKS = ("theta", "omega", "p", "a", "r")
VERIFIED = {}
ALGEBRA = {kind: dict(formula_calls=0, exp_elements=0, tanh_elements=0)
           for kind in ("original", "candidate")}


def bound(path, digest, size=None):
    file = (ROOT / path).resolve()
    assert file.is_relative_to(ROOT), path
    raw = file.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == digest, f"hash: {path}"
    assert size is None or len(raw) == size, f"size: {path}"
    VERIFIED[file.relative_to(ROOT).as_posix()] = dict(sha256=actual, bytes=len(raw))
    return raw


def ref(item):
    return bound(item["path"], item["sha256"], item.get("bytes"))


def references(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            ref(value)
        for child in value.values():
            references(child)
    elif isinstance(value, list):
        for child in value:
            references(child)


def metadata(path, digest):
    value = json.loads(bound(path, digest))
    references(value)
    return value


def close(actual, expected, *, atol=1e-12, rtol=1e-6):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            close(actual[key], expected[key], atol=atol, rtol=rtol)
    elif expected is None or isinstance(expected, (str, bool)):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def pair_errors(left, right, mass):
    n = len(mass)
    def measured(state):
        center = np.sum(state[:, :n] * mass, axis=1) / np.sum(mass)
        return np.concatenate((state[:, :n] - center[:, None], state[:, n:2*n]), axis=1) / .1
    error = measured(left) - measured(right)
    relative = np.abs(left-right) / (1. + np.maximum(np.abs(left), np.abs(right)))
    return dict(observation_rms=float(np.sqrt(np.mean(error*error))),
                observation_peak=float(np.max(np.abs(error))),
                state_relative_max=float(np.max(relative)))


def prefix(paths, mass):
    lengths = []
    for values in paths.values():
        count = 0
        for row in values:
            if not np.isfinite(row).all():
                break
            count += 1
        lengths.append(count)
    common = min(lengths)
    kept = 0
    # Recheck each whole prefix rather than reuse the producer's cumulative arrays.
    for end in range(1, common+1):
        valid = True
        for left, right in combinations(paths.values(), 2):
            errors = pair_errors(left[:end], right[:end], mass)
            valid &= (errors["observation_rms"] <= 1e-6
                      and errors["observation_peak"] <= 1e-5
                      and errors["state_relative_max"] <= 1e-5)
        if not valid:
            break
        kept = end
    pairs = {f"{a}:{b}": pair_errors(paths[a][:kept], paths[b][:kept], mass)
             for a, b in combinations(paths, 2)} if kept else {}
    return common, kept, pairs


def self_check():
    clean = np.zeros((25, 10))
    hidden = clean.copy()
    hidden[8, 6] = .01
    short = clean.copy()
    short[20:] = np.nan
    assert prefix({"a": clean, "b": hidden, "c": short}, np.ones(2))[:2] == (20, 8)


def extrema(values):
    return dict(start=[float(np.min(values[0])), float(np.max(values[0]))],
                end=[float(np.min(values[-1])), float(np.max(values[-1]))],
                full=[float(np.min(values)), float(np.max(values))])


def derived(values, kind, gamma, edges, mod):
    amp = values[:, 30:40]
    # Explicit endpoint sums are independent of the producer's incidence product.
    h = np.column_stack([gamma[i]*amp[:, i] + gamma[j]*amp[:, j] for i, j in edges])
    argument = np.tanh(h) if kind == "candidate" else h
    ALGEBRA[kind]["formula_calls"] += 1
    ALGEBRA[kind]["exp_elements"] += int(h.size)
    ALGEBRA[kind]["tanh_elements"] += int(h.size) if kind == "candidate" else 0
    with np.errstate(over="raise", invalid="raise"):
        phi = (1.-mod) + mod*np.exp(argument)
    return dict(A=np.max(np.abs(amp), axis=1), h=h, exponent_argument=argument, phi=phi)


def stats(values, quantities, rows):
    amplitude = quantities["A"][:rows]
    a0, end = float(amplitude[0]), float(amplitude[-1])
    output = dict(A0=a0, Aend=end, Amax=float(np.max(amplitude)), deltaA=end-a0,
                  Aend_over_A0=None if a0 == 0 else end/a0,
                  components={name: extrema(values[:rows, 10*i:10*(i+1)])
                              for i, name in enumerate(BLOCKS)})
    output.update({key: extrema(quantities[key][:rows]) for key in ("h", "exponent_argument", "phi")})
    return output


def audit():
    self_check()
    execution = metadata(f"results/{RUN}/execution.json", "e3d6cf2ea2e1c19af620321a4f1ca6802f1027545357f54a5d04cdebc7fc6c2a")
    summary = metadata(execution["summary"]["path"], "defb59e086515dc37496e53b40d3fc3050850b17424fe2a41060b859e888b7f4")
    prepared = metadata(f"results/{RUN}/prepare.json", execution["prepare_sha256"])
    qualified = metadata(f"results/{RUN}/qualification.json", execution["qualification_sha256"])
    release = metadata(f"results/{RUN}/release.json", execution["release_sha256"])
    adoption = json.loads(ref(release["adoption"]))
    references(adoption)
    for document in (execution, prepared, qualified, adoption):
        for path, digest in document["source_sha256"].items():
            bound(path, digest)
    assert execution["source_sha256"] == prepared["source_sha256"] == qualified["source_sha256"]
    assert execution["runtime"] == prepared["runtime"]
    assert execution["artificial_counts"] == qualified["counts"]
    assert qualified["passed"] and qualified["prepare_sha256"] == execution["prepare_sha256"]
    assert execution["status"] == "COMPLETED" and execution["error"] is None
    assert execution["unknown_jobs"] == {} and execution["jobs_without_arrays"] == []
    assert execution["returned_integration_jobs"] == execution["claimed_job_count"] == 6
    assert release["prepare_sha256"] == execution["prepare_sha256"]
    assert release["qualification_sha256"] == execution["qualification_sha256"]
    inputs = prepared["inputs"]
    model = json.loads(ref(inputs["model"]))
    mass = np.asarray(model["functional"]["mass"])
    gamma = np.asarray(model["functional"]["gamma_amp"])
    all_edges = list(combinations(range(10), 2))
    coefficient = np.asarray(model["functional"]["k_edge"])*np.asarray(model["gates"]["s_edge"])
    modulation = np.asarray(model["gates"]["s_mod"])
    active = coefficient*modulation > 0.
    edges = [edge for edge, selected in zip(all_edges, active) if selected]
    mod = modulation[active]
    original_exec = json.loads(bound("results/gftd_long_horizon_diagnostic_20261109/phase1/execution.json",
                                    inputs["original_execution_sha256"]))
    # Its hash binds all old metadata; do not follow unrelated old trajectory refs.
    original_init_path = "results/gftd_initialization_precision_20261109/candidate/full/initialize/modulated_268.json"
    original_init = json.loads(bound(original_init_path, inputs["original_initialize_sha256"]))
    assert original_init["state"] is None and inputs["original_state"] is None
    witnesses = {}
    for name, witness in inputs["witnesses"].items():
        assert witness["summary"] == original_exec["summaries"][name]
        data = json.loads(ref(witness["summary"]))
        assert data["arrays"] == witness["arrays"]
        tag = f"sample_radau_strict_{round(100*witness['time'])}_"
        with np.load(io.BytesIO(ref(witness["arrays"])), allow_pickle=False) as archive:
            cached = {key: archive[tag+key] for key in ("active", "endpoint_sum", "beta", "psi", "beta_psi")}
        np.testing.assert_array_equal(cached["active"], active)
        np.testing.assert_array_equal(cached["beta"], coefficient*modulation)
        np.testing.assert_array_equal(cached["beta_psi"], cached["beta"]*cached["psi"])
        edge_id = int(np.argmax(np.where(active, cached["endpoint_sum"], -np.inf)))
        assert list(all_edges[edge_id]) == witness["edge"]
        assert cached["beta_psi"][edge_id] < 0 and np.all(gamma[list(all_edges[edge_id])] != 0)
        witnesses[name] = dict(time=witness["time"], negative_active_edge=True, gamma_nonzero=True)

    all_results, job_records, path_data = {}, [], {}
    for stage in STAGES:
        versions, paths_by_version = {}, {}
        for kind in ("original", "candidate"):
            paths, jobs = {}, {}
            for method in METHODS:
                item = inputs["original_reference_jobs"][stage][method] if kind == "original" else execution["jobs"][stage][method]
                job = json.loads(ref(item["job"] if kind == "original" else item))
                references(job)
                if kind == "original":
                    assert original_exec["tasks"][job["action_id"]] == item["job"]
                    assert job["arrays"] == item["arrays"]
                    assert job["rhs_version"] == "original_strict"
                else:
                    assert job["action_id"] == execution["action_id"] and job["stage"] == stage
                    assert job["rhs_version"] == execution["version"] == "bounded_exp_tanh_v1"
                    assert job["release_sha256"] == execution["release_sha256"]
                    assert job["completed"] and job["status"] == "completed" and job["error"] is None
                    assert job["actual_rhs_calls"] == sum(job["rhs_calls_by_operation"].values()) <= 24000
                    assert job["actual_rhs_calls"] >= job["solver_nfev"]
                    job_records.append(dict(stage=stage, method=method, **{key:job[key] for key in
                        ("actual_rhs_calls", "rhs_calls_by_operation", "solver_nfev", "solver_njev", "solver_nlu", "wall_seconds", "cpu_seconds")}))
                assert job["method"] == method and job["target_time"] == 1.
                assert job["rtol"] == (1e-9 if method == "radau" else 1e-11)
                assert job["atol"] == (1e-11 if method == "radau" else 1e-13)
                with np.load(io.BytesIO(ref(job["arrays"])), allow_pickle=False) as archive:
                    path, grid = archive["grid_states"], archive["grid_times"]
                    accepted_t, accepted_x = archive["accepted_times"], archive["accepted_states"]
                assert path.shape == (101, 50) and path.dtype == np.float64
                np.testing.assert_allclose(grid, np.arange(101)/100., atol=1e-15, rtol=0)
                rows = job["output_rows"]
                assert np.isfinite(path[:rows]).all() and np.isnan(path[rows:]).all()
                assert job["last_output_time"] == grid[rows-1]
                assert accepted_t[0] == 0 and accepted_t[-1] == job["last_accepted_time"]
                assert np.all(np.diff(accepted_t) > 0) and np.max(np.diff(accepted_t)) <= .005+1e-14
                assert accepted_x.shape == (len(accepted_t), 50) and np.isfinite(accepted_x).all()
                assert job["accepted_steps"] == len(accepted_t)-1
                assert hashlib.sha256(path[0].astype("<f8").tobytes()).hexdigest() == inputs["stage_state_sha256"][stage]
                assert hashlib.sha256(path[0, 20:30].astype("<f8").tobytes()).hexdigest() == inputs["p0_sha256"]
                if kind == "candidate":
                    assert rows == 101 and accepted_t[-1] == 1.
                    np.testing.assert_allclose(path[-1], accepted_x[-1], atol=1e-12, rtol=1e-12)
                paths[method], jobs[method] = path, job
            common, keep, pair = prefix(paths, mass)
            frozen = summary["stages"][stage]["within_model"][kind]
            assert (common-1)/100. == frozen["common_end"]
            assert (keep-1)/100. == frozen["agreement_end"] and keep == frozen["agreement_rows"]
            close(pair, frozen["agreement_pairs"])
            versions[kind] = dict(common_end=(common-1)/100., agreement_end=(keep-1)/100.,
                                  agreement_rows=keep, pair_errors=pair,
                                  jobs={method: dict(status=jobs[method]["status"], last_output_time=jobs[method]["last_output_time"],
                                                     last_accepted_time=jobs[method]["last_accepted_time"]) for method in METHODS})
            paths_by_version[kind] = paths
        matched = min(value["agreement_rows"] for value in versions.values())
        assert (matched-1)/100. == summary["stages"][stage]["T_pair"]
        comparable, full = {}, {}
        for kind in versions:
            comparable[kind], full[kind] = {}, {}
            keep = versions[kind]["agreement_rows"]
            for method, path in paths_by_version[kind].items():
                quantities = derived(path[:keep], kind, gamma, edges, mod)
                comparable[kind][method] = stats(path, quantities, matched)
                full[kind][method] = stats(path, quantities, keep)
                close(comparable[kind][method], summary["stages"][stage]["matched_window"][kind][method], atol=1e-12, rtol=1e-12)
                np.testing.assert_array_equal(path[0], paths_by_version["original"][method][0])
        all_results[stage] = dict(within_model=versions, T_pair=(matched-1)/100.,
                                  matched_window=comparable, own_qualified_window=full)
        path_data[stage] = paths_by_version
    np.testing.assert_array_equal(path_data["stage1"]["original"]["radau"][0, :30],
                                  path_data["stage2"]["original"]["radau"][0, :30])
    rhs = sum(row["actual_rhs_calls"] for row in job_records)
    assert rhs == execution["actual_integration_rhs_calls"] == execution["known_integration_rhs_calls"] == 61899
    for key in ("cpu_seconds", "wall_seconds"):
        close(sum(row[key] for row in job_records), execution["worker_"+key], atol=1e-12)
    assert qualified["counts"]["rhs"] == 10 and qualified["counts"]["hamiltonian"] == 370
    for key in ("real_extra_rhs", "real_extra_hamiltonian", "real_extra_gradients", "real_extra_power"):
        assert summary[key] == 0
    matched_elements = len(edges)*3*sum(round(100*result["T_pair"])+1 for result in all_results.values())
    assert summary["algebra_counts"] == {kind: dict(formula_calls=6, exp_elements=matched_elements) for kind in versions}
    return dict(passed=True, independent_method="separate weighted-mean observations, full-prefix rescan, explicit edge endpoint sums",
                stages=all_results, witness_checks=witnesses, active_edge_count=len(edges), candidate_job_counts=job_records,
                verified_file_count=len(VERIFIED), verified_files=VERIFIED,
                integration_rhs=rhs, artificial_rhs=10, total_model_rhs=rhs+10, artificial_H=370,
                producer_algebra_counts=summary["algebra_counts"], audit_algebra_counts=ALGEBRA,
                audit_new_ode=0, audit_new_rhs=0, audit_new_H=0, audit_new_gradients=0,
                science_host_wall_seconds=execution["host_wall_seconds"], science_host_cpu_seconds=execution["host_cpu_seconds"],
                worker_wall_seconds=execution["worker_wall_seconds"], worker_cpu_seconds=execution["worker_cpu_seconds"])


if __name__ == "__main__":
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    result = {}
    try:
        result = audit()
    except BaseException as error:
        result = dict(passed=False, error=repr(error), audit_algebra_counts=ALGEBRA,
                      verified_file_count=len(VERIFIED), verified_files=VERIFIED,
                      audit_new_ode=0, audit_new_rhs=0, audit_new_H=0, audit_new_gradients=0)
        raise
    finally:
        result.update(audit_wall_seconds=time.perf_counter()-started_wall,
                      audit_cpu_seconds=time.process_time()-started_cpu,
                      audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (OUT / "independent_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("passed", "verified_file_count", "integration_rhs", "total_model_rhs", "audit_algebra_counts")}))
