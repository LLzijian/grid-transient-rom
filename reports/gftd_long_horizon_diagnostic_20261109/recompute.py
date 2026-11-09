"""Independently verify phase 1 using saved bytes and array algebra only."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = "gftd_long_horizon_diagnostic_20261109"
BASE = Path("results") / RUN
REFS = ("radau", "radau_strict", "dop853_adaptive")
STEPS = {"main": .01, "fine": .005, "finer": .0025, "finest": .00125}
TARGETS = dict(zip(("modulated_008", "modulated_124", "modulated_138", "modulated_204", "static_027",
                   "static_000", "modulated_296", "modulated_048", "modulated_268_stage2", "modulated_268_stage1"),
                  (241, 129, 188, 454, 1500, 1500, 1500, 110, 100, 100), strict=True))
LIMITS = (1e-6, 1e-5, 1e-5)
CHECKED = {}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bytes(relative, expected, size=None):
    path = ROOT / relative
    assert path.resolve().is_relative_to(ROOT) and not Path(relative).is_absolute()
    raw = path.read_bytes()
    assert digest(raw) == expected, f"hash differs: {relative}"
    assert size is None or len(raw) == size, f"size differs: {relative}"
    CHECKED[Path(relative).as_posix()] = expected
    return raw


def read_ref(ref, kind="json"):
    raw = read_bytes(ref["path"], ref["sha256"], ref["bytes"])
    if kind == "npz":
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    return json.loads(gzip.decompress(raw) if kind == "gzip" else raw)


def equal(actual, expected, location="root"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), location
        for key in expected:
            equal(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), location
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            equal(left, right, f"{location}[{index}]")
    elif isinstance(expected, float):
        assert np.isclose(actual, expected, rtol=2e-10, atol=2e-13), (location, actual, expected)
    else:
        assert actual == expected, (location, actual, expected)


def rows(path):
    finite = np.isfinite(path).all(axis=1)
    bad = np.flatnonzero(~finite)
    n = int(bad[0]) if bad.size else len(path)
    assert n and np.isnan(path[n:]).all()
    return n


def observed(states, mass):
    n = len(mass)
    theta = states[:, :n]
    return np.hstack((theta - (theta@mass/mass.sum())[:, None], states[:, n:2*n])) / .1


def magnitude(values):
    peak = float(np.max(np.abs(values)))
    if not np.isfinite(peak):
        return None
    return dict(rms=float(peak * np.linalg.norm(values / peak) / np.sqrt(values.size)) if peak else 0., peak=peak)


def difference(left, right, mass):
    with np.errstate(over="ignore", invalid="ignore"):
        obs = magnitude(observed(left, mass) - observed(right, mass))
        delta = left - right
        scale = 1 + np.maximum(abs(left), abs(right))
        relative = abs(delta) / scale
        bad = ~np.isfinite(relative)
        relative[bad] = abs(left[bad] / scale[bad] - right[bad] / scale[bad])
    if obs is None or not np.isfinite(relative).all():
        return None
    return dict(observed=obs, state_relative_max=float(relative.max()),
                components={name: magnitude(block) for name, block in
                            zip(("theta", "omega", "p", "a", "r"), np.split(delta, 5, axis=1), strict=True)})


def agreement(delta):
    return delta is not None and all(value <= limit for value, limit in zip(
        (delta["observed"]["rms"], delta["observed"]["peak"], delta["state_relative_max"]), LIMITS, strict=True))


def pair_metrics(paths, mass, n):
    return {f"{a}:{b}": difference(paths[a][:n], paths[b][:n], mass)
            if min(rows(paths[a]), rows(paths[b])) >= n else None for a, b in combinations(paths, 2)}


def prefix(paths, mass):
    common = min(map(rows, paths.values()))
    valid = np.ones(common, dtype=bool)
    for left, right in combinations(paths.values(), 2):
        x, y = left[:common], right[:common]
        with np.errstate(over="ignore", invalid="ignore"):
            d = abs(observed(x, mass) - observed(y, mass))
            scale = 1 + np.maximum(abs(x), abs(y))
            relative = abs(x - y) / scale
            bad = ~np.isfinite(relative)
            relative[bad] = abs(x[bad] / scale[bad] - y[bad] / scale[bad])
            accumulated_rms = np.sqrt(np.cumsum(np.mean(d*d, axis=1)) / np.arange(1, common + 1))
        valid &= (d.max(axis=1) <= LIMITS[1]) & (relative.max(axis=1) <= LIMITS[2]) & (accumulated_rms <= LIMITS[0])
    first_bad = np.flatnonzero(~valid)
    return common, int(first_bad[0]) if first_bad.size else common


def precision(paths, reference, mass, n):
    if any(name not in paths for name in STEPS):
        return None
    n = min(n, *(rows(paths[name]) for name in STEPS))
    deltas = {name: difference(paths[name][:n], reference[:n], mass) for name in STEPS}
    errors = [None if d is None else d["observed"]["rms"] for d in deltas.values()]
    decreasing = None not in errors and all(a > b for a, b in zip(errors, errors[1:])) and errors[-1] <= errors[1]/4
    fine_match = all(d is not None and d["observed"]["rms"] <= LIMITS[0] and d["observed"]["peak"] <= LIMITS[1]
                     for d in (deltas["finer"], deltas["finest"]))
    return dict(end_time=(n-1)/100, deltas=deltas,
                adjacent_rms_ratios=[None if a is None or b is None or min(a, b) <= 1e-12 else a/b
                                     for a, b in zip(errors, errors[1:])],
                floor_limited=[name for name, value in zip(STEPS, errors, strict=True) if value is not None and value <= 1e-12],
                decreasing=decreasing, finest_two_match_observations=fine_match,
                evidence=n >= 11 and (decreasing or fine_match))


def coefficients(model):
    f = {key: np.asarray(value) if isinstance(value, list) else value for key, value in model["functional"].items()}
    gates = {key: np.asarray(value) for key, value in model["gates"].items()}
    n = len(f["mass"])
    edges = np.array(list(combinations(range(n), 2)))
    incidence = np.zeros((n, len(edges)))
    incidence[edges[:, 0], np.arange(len(edges))] = 1
    incidence[edges[:, 1], np.arange(len(edges))] = -1
    beta = gates["s_edge"] * f["k_edge"] * gates["s_mod"]
    return f, gates, edges, incidence, beta


def growth(paths, model, n):
    f, _, edges, _, beta = coefficients(model)
    active = beta > 0
    if n < 11 or not active.any():
        return dict(window=None, applicable=bool(active.any()), evaluated=False, matching_windows=0)
    nodes = len(f["mass"])
    a = {name: value[:n, 3*nodes:4*nodes] for name, value in paths.items()}
    amplitude = {name: abs(value).max(axis=1) for name, value in a.items()}
    q = {name: ((value*f["gamma_amp"])[:, edges[active, 0]] +
                (value*f["gamma_amp"])[:, edges[active, 1]]).max(axis=1) for name, value in a.items()}
    first, matches, checked = None, 0, 0
    for start in range(n-5):
        for width in (5, 10, 20):
            end = start + width
            if end >= n:
                continue
            checked += 1
            values = {}
            for name in paths:
                A, Q = amplitude[name], q[name]
                af = float(np.mean(np.diff(A[start:end+1]) >= -1e-8))
                qf = float(np.mean(np.diff(Q[start:end+1]) >= -1e-8))
                if A[end] < max(1, 2*A[start]) or Q[end]-Q[start] < 2 or min(af, qf) < .8:
                    break
                values[name] = dict(A_start=float(A[start]), A_end=float(A[end]), q_start=float(Q[start]),
                                    q_end=float(Q[end]), A_increment_fraction=af, q_increment_fraction=qf)
            if len(values) != len(paths):
                continue
            deltas = pair_metrics({key: value[start:end+1] for key, value in paths.items()}, f["mass"], width+1)
            if not all(map(agreement, deltas.values())):
                continue
            matches += 1
            if first is None:
                first = dict(start=start/100, end=end/100, agreement=deltas, methods=values)
    return dict(window=first, applicable=True, evaluated=True, windows_checked=checked, matching_windows=matches)


def check_job(record, arrays, state, samples, method):
    assert record["state_sha256"] == digest(state.tobytes())
    assert record["rhs_version"] == "original_strict" and record["method"] == method
    assert record["target_time"] == samples/100
    grid, path = arrays["grid_times"], arrays["grid_states"]
    times, mesh = arrays["accepted_times"], arrays["accepted_states"]
    assert np.array_equal(grid, np.linspace(0, samples/100, samples+1))
    assert path.shape == (samples+1, 50) and mesh.shape == (len(times), 50)
    assert path.dtype == mesh.dtype == np.float64 and np.isfinite(mesh).all()
    assert path[0].tobytes() == mesh[0].tobytes() == state.tobytes()
    assert times[0] == 0 and np.all(np.diff(times) > 0)
    assert times[-1] == record["last_accepted_time"] and len(times)-1 == record["accepted_steps"]
    finite = rows(path)
    assert finite == record["output_rows"] and grid[finite-1] == record["last_output_time"]
    assert grid[finite-1] <= times[-1] and record["internal_rejected_steps"] is None
    calls = record["actual_rhs_calls"]
    assert calls == sum(record["rhs_calls_by_operation"].values()) and 0 < calls <= record["rhs_limit"]
    if method in STEPS:
        stride = round(.01/STEPS[method])
        assert record["step"] == STEPS[method] and record["rhs_limit"] == 12*samples*stride
        assert calls == 12*record["accepted_steps"] + record["interrupted_step_rhs_calls"]
        assert np.array_equal(times, np.arange(len(times))*STEPS[method])
        assert np.array_equal(path[:finite], mesh[::stride])
    else:
        assert record["rhs_limit"] == 240*samples and record["max_step"] == .005
        assert np.all(np.diff(times) <= .005 + 1e-14)
        assert record["rtol"] == (1e-9 if method == "radau" else 1e-11)
        assert record["atol"] == (1e-11 if method == "radau" else 1e-13)
    assert record["completed"] == (record["status"] == "completed")
    if record["completed"]:
        assert finite == samples+1 and times[-1] == samples/100 and record["failure"] is None
    if "terminal_trial_states" in arrays:
        length = len(arrays["terminal_trial_states"])
        assert all(len(arrays[key]) == length for key in
                   ("terminal_trial_times", "terminal_trial_operations", "terminal_trial_rhs_evaluated"))
        assert arrays["terminal_trial_rhs_evaluated"].sum() <= calls
        if "failure_state" in arrays:
            assert np.array_equal(arrays["failure_state"], arrays["terminal_trial_states"][-1], equal_nan=True)
            assert record["failure"]["time"] == arrays["terminal_trial_times"][-1]
        if record["status"] == "rhs_limit":
            assert calls == record["rhs_limit"] and not arrays["terminal_trial_rhs_evaluated"][-1]


def check_sample_power(model, summary, arrays):
    f, gates, edges, incidence, _ = coefficients(model)
    assert f["gate_weight_amp"] is None and f["gate_weight_mem"] is None
    checked = 0
    for method, points in summary["samples"].items():
        for time, point in points.items():
            if point["power"] is None:
                continue
            stem = f"sample_{method}_{round(float(time)*100)}_"
            state, rhs, ga = arrays[stem+"state"], arrays[stem+"rhs"], arrays[stem+"amplitude_gradient"]
            theta, omega, p, amp, mem = np.split(state, 5)
            h = amp*f["gamma_amp"]
            mod = 1 + gates["s_mod"]*(np.exp(h[edges[:, 0]]+h[edges[:, 1]])-1)
            gt = incidence @ (gates["s_edge"]*f["k_edge"]*mod*(np.sin(incidence.T@theta)-np.sin(f["eta_edge"])))
            gp, gr = f["storage_p"]*p, f["lambda_mem"]*mem
            gradient = np.concatenate((gt, f["mass"]*omega, gp, ga, gr))
            chain = float(np.dot(gradient, rhs))
            loss = float(np.dot(f["damping"], omega**2) + np.dot(f["rate_p"], gp**2) +
                         np.dot(1/f["tau_amp"], ga**2) + np.dot(1/f["tau_mem_sync"], gr**2))
            equal(chain, point["power"]["chain"], stem+"chain")
            equal(-loss, point["power"]["expected"], stem+"expected")
            # Cancellation makes a fixed relative tolerance on the residual inappropriate.
            assert abs(abs(chain+loss)-point["power"]["absolute_residual"]) <= 1e-12*max(1., abs(chain), loss)
            checked += 1
    return checked


def check_failure_exponents(model, summary, arrays):
    f, gates, edges, incidence, beta = coefficients(model)
    active = beta > 0
    result, candidate = {}, False
    for method, terminal in summary["terminal"].items():
        if "failure_exponent_labels" not in terminal:
            continue
        stem = f"terminal_{method}_failure_state_"
        if stem+"state" not in arrays:
            assert not terminal["failure_exponent_labels"]
            continue
        state = arrays[stem+"state"]
        theta, _, _, amp, _ = np.split(state, 5)
        node = f["gamma_amp"]*amp
        forms = dict(endpoint_sum=node[edges[:, 0]]+node[edges[:, 1]], incidence_sum=abs(incidence).T@node)
        np.testing.assert_array_equal(beta, arrays[stem+"beta"])
        psi = np.cos(f["eta_edge"])-np.cos(incidence.T@theta)-np.sin(f["eta_edge"])*(incidence.T@theta-f["eta_edge"])
        np.testing.assert_allclose(psi, arrays[stem+"psi"], rtol=2e-12, atol=2e-12)
        labels, evaluated = set(), {}
        for form, values in forms.items():
            np.testing.assert_allclose(values, arrays[stem+form], rtol=2e-12, atol=2e-12)
            violations = dict(overflow=[], underflow=[])
            for edge, value in enumerate(values):
                if np.log(np.finfo(float).tiny) < value < np.log(np.finfo(float).max):
                    continue
                try:
                    with np.errstate(over="raise", under="raise", invalid="raise"):
                        np.exp(value)
                except FloatingPointError:
                    violations["underflow" if value < 0 else "overflow"].append(edge)
            groups = {}
            for name, mask in (("active", active), ("inactive", ~active)):
                groups[name] = None if not mask.any() else dict(min=float(values[mask].min()), max=float(values[mask].max()),
                    **{kind: [int(edge) for edge in indexes if mask[edge]] for kind, indexes in violations.items()})
                if groups[name] is not None:
                    for kind in violations:
                        if groups[name][kind]:
                            labels.add(f"active_exp_{kind}" if name == "active" else "inactive_zero_exp_out_of_range")
            equal(groups, terminal["failure_state"]["exponents"][form], method+"."+form)
            evaluated[form] = groups
        assert sorted(labels) == terminal["failure_exponent_labels"]
        candidate |= "inactive_zero_exp_out_of_range" in labels
        result[method] = dict(labels=sorted(labels), exponents=evaluated,
                             max_abs_amplitude=float(abs(amp).max()))
    assert candidate == summary["inactive_zero_skip_candidate"]
    return candidate, result


def audit(execution_sha):
    execution = json.loads(read_bytes(BASE/"phase1/execution.json", execution_sha))
    assert execution["status"] == "COMPLETED" and len(execution["tasks"]) == 46
    release = json.loads(read_bytes(BASE/"phase1_release.json", execution["release_sha256"]))
    assert release["released"] and release["phase"] == "phase1"
    for key in ("implementation_review", "qualification"):
        read_bytes(release[key]["path"], release[key]["sha256"])
    assert release["source_sha256"] == execution["source_sha256"]
    for path, expected in execution["source_sha256"].items():
        read_bytes(path, expected)
    read_bytes(Path("campaigns")/RUN/"protocol.md", execution["protocol_sha256"])
    read_bytes(BASE/"design_adoption.json", execution["design_adoption_sha256"])
    read_bytes(Path("reports")/RUN/"design_review.md", execution["design_review_sha256"])
    inventory = json.loads(read_bytes(BASE/"prepare.json", execution["prepared_sha256"]))
    for path, expected in inventory["parents"].items():
        read_bytes(path, expected)
    models = {mode: read_ref(ref) for mode, ref in inventory["models"].items()}
    receipts = [read_ref(ref) for ref in execution["tasks"].values()]
    by_case = {name: {} for name in TARGETS}
    for record in receipts:
        name, method = record["case_id"], record["method"]
        assert method not in by_case[name] and record["action_id"] in execution["tasks"]
        assert record["phase_action_id"] == release["action_id"]
        by_case[name][method] = record
    output, sample_calls = {}, 0
    for name, samples in TARGETS.items():
        source = inventory["cases"][name]
        assert source["target_samples"] == samples
        init = read_ref(source["initialization"])
        if source["state"] is not None:
            state = read_ref(source["state"], "npz")["state"]
            assert init["passed"] and init["p0_exact"]
        else:
            diagnostic = read_ref(source["diagnostics"], "gzip")
            state = np.asarray(diagnostic[source["stage"]]["best"]["state"], dtype="<f8")
            assert init["state"] is None and diagnostic["returned_point"] == "stage2"
        assert state.shape == (50,) and state.dtype == np.float64 and np.isfinite(state).all()
        assert digest(state.tobytes()) == source["state_sha256"]
        assert digest(state[20:30].tobytes()) == source["p0_sha256"]
        paths, fixed, records = {}, {}, by_case[name]
        expected_methods = set(REFS)
        if name in ("modulated_008", "modulated_124", "modulated_138", "modulated_204", "static_027"):
            expected_methods |= {"finer", "finest"}
        elif name == "modulated_268_stage2":
            expected_methods |= set(STEPS)
        elif name == "modulated_268_stage1":
            expected_methods |= {"main", "fine"}
        assert set(records) == expected_methods
        if source["arrays"]:
            old = read_ref(source["arrays"], "npz")
            future = read_ref(source["future"])
            assert old["initial_state"].tobytes() == state.tobytes()
            for method in ("main", "fine"):
                old_record = future["integrals"][method]
                old_path = old[method+"_path"]
                stride = round(.01/STEPS[method])
                assert rows(old_path) == old_record["accepted_steps"]//stride+1
                assert old_record["actual_rhs_calls"] == 12*old_record["accepted_steps"]+old_record["interrupted_step_rhs_calls"]
                fixed[method] = old_path[:samples+1]
        for method, record in records.items():
            arrays = read_ref(record["arrays"], "npz")
            check_job(record, arrays, state, samples, method)
            (paths if method in REFS else fixed)[method] = arrays["grid_states"]
        summary = read_ref(execution["summaries"][name])
        mass = np.asarray(models[source["mode"]]["functional"]["mass"])
        common, consistent = prefix(paths, mass)
        reference = dict(common_end=(common-1)/100, agreement_end=(consistent-1)/100,
                         completed=consistent == samples+1 and all(records[method]["completed"] for method in REFS),
                         agreement_pairs=pair_metrics(paths, mass, consistent))
        equal(reference, summary["reference"], name+".reference")
        fixed_result = precision(fixed, paths["radau_strict"], mass, consistent)
        equal(fixed_result, summary["fixed_precision"], name+".fixed_precision")
        growth_result = growth(paths, models[source["mode"]], consistent)
        equal(growth_result, summary["growth"], name+".growth")
        horizons = {str(index/100): pair_metrics(paths | fixed, mass, index+1)
                    for index in sorted({100, 200, 500, 800, 1500, samples})}
        equal(horizons, summary["horizons"], name+".horizons")
        saved_summary = read_ref(summary["arrays"], "npz")
        calls = summary["algebra_and_sample_counts"]["rhs"]
        assert calls == sum(map(len, summary["samples"].values()))
        sample_calls += calls
        samples_checked = check_sample_power(models[source["mode"]], summary, saved_summary)
        candidate, failure_evidence = check_failure_exponents(models[source["mode"]], summary, saved_summary)
        output[name] = dict(state_sha256=source["state_sha256"], jobs=len(records), reference=reference,
                            fixed_precision=fixed_result, growth=growth_result, power_samples_checked=samples_checked,
                            inactive_zero_skip_candidate=candidate, failure_evidence=failure_evidence,
                            statuses={method: record["status"] for method, record in records.items()})
    total = sum(record["actual_rhs_calls"] for record in receipts)
    assert total == execution["actual_integration_rhs_calls"] and total <= 4191840+383328
    assert sum(record["rhs_limit"] for record in receipts) == 4191840+383328
    assert sample_calls == execution["sample_rhs_calls"] <= 210
    return dict(status="verified", run_id=RUN, execution_sha256=execution_sha, jobs=46, cases=output,
                actual_integration_rhs_calls=total, sample_rhs_calls=execution["sample_rhs_calls"],
                source_sha256=digest(Path(__file__).read_bytes()), checked_files=CHECKED,
                new_rhs_calls=0, new_integrations=0, future_truth_reads=0)


def self_test():
    mass = np.ones(2)
    clean = np.zeros((21, 10))
    changed = clean.copy()
    changed[7:, 6] = 1e-3
    assert prefix(dict(a=clean, b=clean, c=changed), mass) == (21, 7)
    assert difference(clean, changed, mass)["observed"]["peak"] == 0
    changed[:] = clean
    changed[4, 2] = 9e-7
    assert prefix(dict(a=clean, b=changed, c=clean), mass) == (21, 4)
    grids = {name: clean.copy() for name in STEPS}
    grids["main"][14:] = np.nan
    result = precision(grids, clean, mass, 21)
    assert result["end_time"] == .13 and result["evidence"] and result["adjacent_rms_ratios"] == [None]*3
    model = dict(functional=dict(mass=[1, 1], k_edge=[1], gamma_amp=[1, 1]), gates=dict(s_edge=[1], s_mod=[1]))
    increasing = clean[:11].copy()
    increasing[:, 6:8] = np.linspace(0, 3, 11)[:, None]
    result = growth({name: increasing for name in REFS}, model, 11)
    assert result["window"]["start"] == 0 and result["window"]["end"] == .05
    assert growth({name: increasing for name in REFS}, model, 6)["evaluated"] is False
    print("Five independent artificial-array checks passed; no RHS or integration called.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execution-sha")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert args.execution_sha is None and not args.write
        self_test()
    else:
        assert args.execution_sha is not None
        result = audit(args.execution_sha)
        encoded = (json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n").encode()
        if args.write:
            path = ROOT/BASE/"independent_phase1_audit.json"
            with path.open("xb") as stream:
                stream.write(encoded)
        print(json.dumps({key: result[key] for key in ("status", "jobs", "actual_integration_rhs_calls", "sample_rhs_calls")}))
